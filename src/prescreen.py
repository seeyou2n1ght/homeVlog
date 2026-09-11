import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from src.ffmpeg import run_ffmpeg, get_duration, build_hw_decode_args
from src.scheduler import acquire_with_retry, VideoLease
from src.utils import parse_res

logger = logging.getLogger("homevlog")


def _calc_sample_timestamps(duration: float, segments: int, margin: float = 0.5) -> list[float]:
    if duration <= 1.0:
        return [0.0]
    ts = []
    step = duration / segments
    for i in range(segments + 1):
        t = min(i * step, duration - margin)
        ts.append(max(t, 0.0))
    return ts


def _extract_frame(filepath: str, timestamp: float, width: int, height: int, timeout: float = 30.0, gpu: str = "qsv") -> np.ndarray | None:
    """使用硬件解码在指定时间点抽取单帧 RGB 图像。"""
    args = build_hw_decode_args(
        input_path=str(filepath),
        width=width,
        height=height,
        start_time=timestamp,
        vframes=1,
        gpu=gpu,
    )
    from src.scheduler import get_qsv_semaphore, get_nv_semaphore
    hardware = get_qsv_semaphore() if gpu == "qsv" else get_nv_semaphore()
    expected = width * height * 3
    for attempt in range(2):
        if not acquire_with_retry(hardware):
            return None
        try:
            result = run_ffmpeg(args, timeout=timeout)
        finally:
            hardware.release()
        if result.returncode == 0 and len(result.stdout) >= expected:
            frame = np.frombuffer(result.stdout[:expected], dtype=np.uint8).reshape((height, width, 3))
            return frame
        if attempt == 0:
            import time
            time.sleep(0.1)

    if result.returncode != 0:
        logger.warning("FFmpeg extract_frame failed for %s at %.1f: %s", filepath, timestamp, result.stderr_text[-500:])
    elif len(result.stdout) < expected:
        logger.warning("FFmpeg extract_frame too small for %s at %.1f: got %d, expected %d", filepath, timestamp, len(result.stdout), expected)
    return None


def _calc_dynamic_threshold(base_threshold: float, mean_luma: float, is_prior_active: bool = False) -> float:
    """计算综合光照（暗光/强光）与时间邻域先验保护后的自适应判定阈值。"""
    if mean_luma < 50.0:
        # 暗光 / 红外夜视低动态范围场景: 提升灵敏度以捕捉微弱动态
        dyn = max(2.0, base_threshold * 0.40)
    elif mean_luma > 180.0:
        # 逆光 / 强光噪点密集场景: 适当抑制噪点上浮
        dyn = base_threshold * 1.25
    else:
        dyn = base_threshold

    if is_prior_active:
        # 时间邻域弹性保护：前段包含明确动态时，本段门槛下调 35% 防止因暂歇漏切
        dyn = max(1.8, dyn * 0.65)
    return round(float(dyn), 2)


def _calc_spatial_concentration(diff_map: np.ndarray) -> float:
    """计算 4x4 空间网格的能量集中度。
    
    真实前景移动（集中度高 >= 1.35）；全局光照/白平衡跳变（集中度低 < 1.2，整图弥漫）。
    """
    h, w = diff_map.shape[:2]
    gh, gw = max(1, h // 4), max(1, w // 4)
    grid_means = []
    for r in range(4):
        for c in range(4):
            cell = diff_map[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw]
            if cell.size > 0:
                grid_means.append(float(np.mean(cell)))
    if not grid_means:
        return 1.0
    mean_val = float(np.mean(diff_map))
    max_val = max(grid_means)
    if max_val == 0.0 or mean_val == 0.0:
        return 1.0
    return float(max_val / (mean_val + 1e-4))


def _prescreen_keyframes(
    filepath: str,
    duration: float,
    max_keyframes: int,
    threshold: float,
    gpu: str = "qsv",
    is_prior_active: bool = False,
    width: int = 320, height: int = 180, timeout: float = 120.0,
) -> dict:
    """基于 PyAV 仅解码 I-Frame (Keyframe) 进行毫秒级粗筛，带自适应动态阈值与空间集中度早停机制。"""
    import av
    import cv2
    diffs: list[float] = []
    sample_ts: list[float] = []
    has_audio = 0

    from src.utils import get_disk_semaphore
    io_sem = get_disk_semaphore()

    _t_sem = time.monotonic()
    _sem_ok = acquire_with_retry(io_sem, timeout=30.0, retries=6)
    sem_wait = round(time.monotonic() - _t_sem, 2)
    if not _sem_ok:
        logger.warning("prescreen keyframes: io semaphore acquire timeout for %s", filepath)
        return {"status": "FAILED", "error": "io semaphore acquire timeout", "has_audio": 0}

    effective_threshold = threshold
    try:
        with av.open(str(filepath), options={"buffer_size": "2097152"}, timeout=30.0) as container:
            if not container.streams.video:
                return {"status": "FAILED", "error": "No video stream", "has_audio": 0}
            has_audio = 1 if len(container.streams.audio) > 0 else 0
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"

            first_frame: np.ndarray | None = None
            prev_frame: np.ndarray | None = None
            concentrations: list[float] = []
            mean_lumas: list[float] = []
            k = 0
            # 关键帧稀疏步长采样：按 max_keyframes 均匀稀疏采样（默认目标 20 帧），杜绝遍历上千关键帧
            target_samples = max(10, max_keyframes) if max_keyframes > 0 else 20
            est_keyframes = max(1, int(duration / 2.5))
            kf_step = max(1, est_keyframes // target_samples)

            for frame in container.decode(stream):
                k += 1
                if k > 1 and (k - 1) % kf_step != 0:
                    continue
                y_raw = frame.to_ndarray(format="gray")
                curr_frame = cv2.resize(y_raw, (width, height), interpolation=cv2.INTER_AREA)
                if time.monotonic() - _t_sem > timeout:
                    return {"status": "SUSPICIOUS", "has_audio": has_audio, "error": "prescreen coverage timeout"}

                t_frame = float(frame.pts * stream.time_base) if (frame.pts is not None and stream.time_base) else float(k)
                sample_ts.append(t_frame)

                if first_frame is None:
                    first_frame = curr_frame
                    prev_frame = curr_frame
                    k += 1
                    continue

                # 动态环境光自适应与先验保护
                mean_luma = float(np.mean(curr_frame))
                mean_lumas.append(mean_luma)
                current_threshold = _calc_dynamic_threshold(threshold, mean_luma, is_prior_active=is_prior_active)
                effective_threshold = current_threshold

                diff_prev_map = cv2.absdiff(curr_frame, prev_frame)
                diff_first_map = cv2.absdiff(curr_frame, first_frame)

                diff_prev = float(np.mean(diff_prev_map))
                diff_first = float(np.mean(diff_first_map))
                d = max(diff_prev, diff_first)
                diffs.append(d)

                active_map = diff_prev_map if diff_prev >= diff_first else diff_first_map
                concentration = _calc_spatial_concentration(active_map)
                concentrations.append(concentration)

                # 空间集中度智能早停与双轨防漏检保护:
                # 1. 时间邻域先验保护下放宽集中度门限至 1.15
                # 2. 局部高能集中 (concentration >= conc_thresh)
                # 3. 或全图绝对差分极大 (d >= current_threshold * 1.5)
                # 4. 或任意 4x4 局部网格单点峰值突破安全门限 (max_cell_energy >= current_threshold * 1.8)
                conc_thresh = 1.15 if is_prior_active else 1.35
                h, w = active_map.shape[:2]
                gh, gw = max(1, h // 4), max(1, w // 4)
                max_cell_energy = max(
                    [float(np.mean(active_map[r * gh : (r + 1) * gh, c * gw : (c + 1) * gw]))
                     for r in range(4) for c in range(4)]
                ) if active_map.size > 0 else 0.0

                is_motion = False
                if d > current_threshold:
                    if mean_luma < 50.0:
                        # 暗光/红外防噪点专项逻辑：严格剔除弥漫型全图白噪点
                        # 必须具备局部能量聚集 (concentration >= 1.25 且局部块达到有效能量 6.5)，或局部出现显著动作 (max_cell_energy >= 12.0)
                        if (concentration >= 1.25 and max_cell_energy >= 6.5) or max_cell_energy >= 12.0:
                            is_motion = True
                        else:
                            logger.debug(
                                "Prescreen night diffuse noise suppressed for %s: d=%.2f, th=%.2f, conc=%.2f, max_cell=%.2f",
                                Path(filepath).name, d, current_threshold, concentration, max_cell_energy,
                            )
                    else:
                        if (
                            concentration >= conc_thresh
                            or d >= current_threshold * 1.5
                            or max_cell_energy >= current_threshold * 1.8
                        ):
                            is_motion = True
                        else:
                            logger.debug(
                                "Prescreen diffuse motion suppressed for %s: d=%.2f, th=%.2f, conc=%.2f, max_cell=%.2f",
                                Path(filepath).name, d, current_threshold, concentration, max_cell_energy,
                            )
                elif max_cell_energy >= max(14.0, current_threshold * 2.2):
                    # [双轨防漏检保护] 远距离/角落局部小目标剧烈运动：即使全图均值 d 被大背景稀释，单点能量爆发依然强制唤醒
                    is_motion = True

                if is_motion:
                    return {
                        "status": "SUSPICIOUS",
                        "has_audio": has_audio,
                        "result_json": json.dumps({
                            "mode": "keyframes",
                            "sample_ts": sample_ts,
                            "diffs": diffs,
                            "max_diff": max(diffs),
                            "threshold": current_threshold,
                            "mean_luma": mean_luma,
                            "concentration": round(concentration, 2),
                            "early_stop": True,
                            "checked_pairs": len(diffs),
                            "sem_wait": sem_wait,
                        }),
                    }

                prev_frame = curr_frame
                k += 1
                # A negative decision requires reaching EOF, never just the first N keyframes.
    except Exception as e:
        logger.debug("PyAV keyframes prescreen failed for %s: %s", filepath, e)
        return {"status": "FALLBACK", "error": str(e), "has_audio": has_audio}
    finally:
        io_sem.release()

    if not diffs:
        return {"status": "SUSPICIOUS", "has_audio": has_audio, "result_json": json.dumps({"mode": "keyframes", "reason": "insufficient_samples", "diffs": [], "sem_wait": sem_wait})}

    max_diff = max(diffs)
    # 所有检查帧均未满足局部动作条件（弥散光影已在循环中成功抑制），判定为静止
    return {
        "status": "STATIC",
        "has_audio": has_audio,
        "result_json": json.dumps({
            "mode": "keyframes",
            "sample_ts": sample_ts,
            "diffs": diffs,
            "max_diff": max_diff,
            "threshold": effective_threshold,
            "mean_luma": round(float(np.mean(mean_lumas)), 2) if mean_lumas else 100.0,
            "concentration": round(max(concentrations), 2) if concentrations else 1.0,
            "early_stop": False,
            "checked_pairs": len(diffs),
            "sem_wait": sem_wait,
        }),
    }


def prescreen_file(
    filepath: str,
    duration: float,
    config: dict,
    gpu: str = "qsv",
    is_prior_active: bool = False,
) -> dict:
    """按需逐帧提取 + 即时早停 diff: 只要发现一次动静即终止提取，极速抛弃。"""
    det_cfg = config.get("detection", {})
    mode = det_cfg.get("prescreen_mode", "legacy_seek")
    segments = det_cfg.get("prescreen_segments", 5)
    width, height = parse_res(
        det_cfg.get("prescreen_resolution", "320x180")
    )
    threshold = det_cfg.get("prescreen_diff_threshold", 12)
    timestamp_margin = det_cfg.get("timestamp_margin", 0.5)

    if duration <= 0:
        actual_dur = get_duration(filepath)
        if actual_dur is not None and actual_dur > 0:
            duration = actual_dur
    if mode in ("keyframes", "auto"):
        kf_res = _prescreen_keyframes(
            filepath=filepath,
            duration=duration,
            max_keyframes=segments,
            width=width, height=height, timeout=det_cfg.get("prescreen_extract_timeout", 120.0),
            threshold=threshold,
            gpu=gpu,
            is_prior_active=is_prior_active,
        )
        if kf_res.get("status") != "FALLBACK":
            return kf_res
    elif mode == "stream_fps":
        return _prescreen_stream_fps(
            filepath,
            duration,
            segments,
            width,
            height,
            threshold,
            gpu,
            timeout=det_cfg.get("prescreen_extract_timeout", 30.0),
            is_prior_active=is_prior_active,
        )
    ts_list = _calc_sample_timestamps(duration, segments, timestamp_margin)

    diffs: list[float] = []

    # 提取第 0 帧
    first_frame = _extract_frame(filepath, ts_list[0], width, height, gpu=gpu)
    if first_frame is None:
        return {"status": "FAILED", "error": f"failed to extract first frame at t={ts_list[0]:.1f}s"}
    
    prev_frame = first_frame
    effective_threshold = threshold

    for i in range(1, len(ts_list)):
        curr_frame = _extract_frame(filepath, ts_list[i], width, height, gpu=gpu)
        if curr_frame is None:
            return {"status": "FAILED", "error": f"failed to extract frame at t={ts_list[i]:.1f}s"}

        # 动态环境光自适应与先验保护
        mean_luma = float(np.mean(curr_frame))
        current_threshold = _calc_dynamic_threshold(threshold, mean_luma, is_prior_active=is_prior_active)
        effective_threshold = current_threshold

        # 双重比对防盲区：与上一帧比（抓取瞬间动作），与首帧比（抓取长时间停留或场景改变）
        diff_prev = float(np.mean(np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32))))
        diff_first = float(np.mean(np.abs(curr_frame.astype(np.float32) - first_frame.astype(np.float32))))
        d = max(diff_prev, diff_first)
        
        diffs.append(d)

        # 【核心优化】动态判定早停：只要大于阈值，证明有人移动，立刻中止文件处理！
        if d > current_threshold:
            return {
                "status": "SUSPICIOUS",
                "result_json": json.dumps({
                    "mode": "legacy_seek",
                    "sample_ts": ts_list[:i + 1],
                    "diffs": diffs,
                    "max_diff": max(diffs),
                    "threshold": current_threshold,
                    "mean_luma": mean_luma,
                    "early_stop": True,
                    "checked_pairs": len(diffs),
                    "ffmpeg_calls": i + 1,
                }),
            }

        prev_frame = curr_frame

    if not diffs:
        return {"status": "FAILED", "error": "no frame pairs to compare"}

    max_diff = max(diffs)
    status = "SUSPICIOUS" if max_diff > effective_threshold else "STATIC"

    return {
        "status": status,
        "result_json": json.dumps({
            "mode": "legacy_seek",
            "sample_ts": ts_list,
            "diffs": diffs,
            "max_diff": max_diff,
            "threshold": effective_threshold,
            "early_stop": False,
            "checked_pairs": len(diffs),
            "ffmpeg_calls": len(ts_list),
        }),
    }


def _build_stream_fps_args(
    filepath: str,
    sample_fps: float,
    max_frames: int,
    width: int,
    height: int,
    gpu: str,
) -> list[str]:
    if gpu == "qsv":
        hw_args = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
        vf = f"scale_qsv=w={width}:h={height},hwdownload,format=nv12,fps={sample_fps:.6f}"
    else:
        hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        vf = f"scale_cuda={width}:{height},hwdownload,format=nv12,fps={sample_fps:.6f}"

    return [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        *hw_args,
        "-i",
        str(filepath),
        "-vf",
        vf,
        "-frames:v",
        str(max_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]


def _prescreen_stream_fps(
    filepath: str,
    duration: float,
    segments: int,
    width: int,
    height: int,
    threshold: float,
    gpu: str,
    timeout: float,
    is_prior_active: bool = False,
) -> dict:
    """用单个低 FPS 解码流做预筛，避免每个采样点启动一次 seek 进程。"""
    frame_size = width * height * 3
    max_frames = max(2, segments + 1)
    sample_fps = max(1.0 / max(duration, 1.0), segments / max(duration, 1.0))
    cmd = _build_stream_fps_args(filepath, sample_fps, max_frames, width, height, gpu)

    if gpu == "qsv":
        from src.utils import get_qsv_semaphore
        io_sem = VideoLease(get_qsv_semaphore())
    else:
        from src.utils import get_nv_semaphore
        io_sem = VideoLease(get_nv_semaphore())

    stderr_lines: list[str] = []
    diffs: list[float] = []
    sample_ts: list[float] = []
    proc: subprocess.Popen | None = None
    completed_read = False
    effective_threshold = threshold

    # AGENTS.md 铁律：acquire 必须带 timeout 并重试，禁止无限阻塞
    if not acquire_with_retry(io_sem):
        logger.warning("prescreen stream_fps: io semaphore acquire timeout for %s", filepath)
        return {"status": "FAILED", "error": "io semaphore acquire timeout"}
    from src.renderer import FFmpegProcessRegistry
    process_key = f"prescreen:{filepath}:{time.monotonic()}"
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        FFmpegProcessRegistry.register(process_key, proc)

        def _read_stderr():
            if proc and proc.stderr:
                for line in proc.stderr:
                    stderr_lines.append(line.decode("utf-8", errors="replace"))

        stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
        stderr_thread.start()
        stream_timeout = max(timeout, min(duration * 0.5, 900.0), 60.0)

        def _kill_on_timeout():
            try:
                if proc and proc.poll() is None:
                    proc.kill()
            except OSError:
                pass

        watchdog = threading.Timer(stream_timeout, _kill_on_timeout)
        watchdog.daemon = True
        watchdog.start()

        try:
            if proc.stdout is None:
                return {"status": "FAILED", "error": "ffmpeg stdout 不可用"}
            
            first_raw = proc.stdout.read(frame_size)
            if len(first_raw) < frame_size:
                # Wait a bit for stderr to populate
                time.sleep(0.5)
                err = "".join(stderr_lines[-5:])
                return {"status": "FAILED", "error": f"stream_fps 首帧读取失败: {err}"}

            first_frame = np.frombuffer(first_raw, dtype=np.uint8).reshape((height, width, 3))
            prev_frame = first_frame
            sample_ts.append(0.0)

            for i in range(1, max_frames):
                raw = proc.stdout.read(frame_size)
                if len(raw) < frame_size:
                    break

                curr_frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                mean_luma = float(np.mean(curr_frame))
                current_threshold = _calc_dynamic_threshold(threshold, mean_luma, is_prior_active=is_prior_active)
                effective_threshold = current_threshold

                diff_prev = float(
                    np.mean(np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32)))
                )
                diff_first = float(
                    np.mean(np.abs(curr_frame.astype(np.float32) - first_frame.astype(np.float32)))
                )
                d = max(diff_prev, diff_first)
                diffs.append(d)
                sample_ts.append(min(i / sample_fps, duration))

                if d > current_threshold:
                    return {
                        "status": "SUSPICIOUS",
                        "result_json": json.dumps({
                            "mode": "stream_fps",
                            "sample_ts": sample_ts,
                            "diffs": diffs,
                            "max_diff": max(diffs),
                            "threshold": current_threshold,
                            "mean_luma": mean_luma,
                            "early_stop": True,
                            "checked_pairs": len(diffs),
                            "ffmpeg_calls": 1,
                            "sample_fps": sample_fps,
                        }),
                    }

                prev_frame = curr_frame
            completed_read = True
        finally:
            watchdog.cancel()
            if proc and proc.poll() is None:
                try:
                    if completed_read:
                        proc.wait(timeout=5)
                    else:
                        proc.terminate()
                        proc.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        proc.kill()
                        proc.wait(timeout=2)
                    except OSError:
                        pass
            if proc and proc.stdout:
                proc.stdout.close()
            if proc and proc.stderr:
                proc.stderr.close()
            stderr_thread.join(timeout=2)
    finally:
        try:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        finally:
            FFmpegProcessRegistry.deregister(process_key)
            io_sem.release()

    if proc and proc.returncode not in (0, None):
        err = "".join(stderr_lines[-5:])
        return {
            "status": "FAILED",
            "error": f"stream_fps ffmpeg 失败，returncode={proc.returncode}: {err}",
        }

    if not diffs:
        err = "".join(stderr_lines[-5:])
        return {"status": "FAILED", "error": f"stream_fps 没有可比较帧对: {err}"}

    max_diff = max(diffs)
    status = "SUSPICIOUS" if max_diff > effective_threshold else "STATIC"
    return {
        "status": status,
        "result_json": json.dumps({
            "mode": "stream_fps",
            "sample_ts": sample_ts,
            "diffs": diffs,
            "max_diff": max_diff,
            "threshold": effective_threshold,
            "early_stop": False,
            "checked_pairs": len(diffs),
            "ffmpeg_calls": 1,
            "sample_fps": sample_fps,
        }),
    }
