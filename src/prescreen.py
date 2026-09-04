import json
import logging
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from src.ffmpeg import run_ffmpeg, get_duration, build_hw_decode_args
from src.scheduler import acquire_with_retry
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
    result = run_ffmpeg(args, timeout=timeout)
    if result.returncode != 0:
        logger.error("FFmpeg extract_frame failed for %s at %.1f: %s", filepath, timestamp, result.stderr_text[-500:])
        return None
    raw = result.stdout
    expected = width * height * 3
    if len(raw) < expected:
        logger.error("FFmpeg extract_frame too small for %s at %.1f: got %d, expected %d", filepath, timestamp, len(raw), expected)
        return None
    frame = np.frombuffer(raw[:expected], dtype=np.uint8).reshape((height, width, 3))
    return frame


def _prescreen_keyframes(
    filepath: str,
    duration: float,
    max_keyframes: int,
    threshold: float,
    gpu: str = "qsv",
) -> dict:
    """基于 PyAV 仅解码 I-Frame (Keyframe) 进行毫秒级粗筛，带即时早停机制。"""
    import av
    import cv2
    diffs: list[float] = []
    sample_ts: list[float] = []
    has_audio = 0

    # 关键帧预筛为 CPU 软解（av.open 无 hwaccel），不消耗 QSV/NV 硬件槽位，
    # 仅属 NAS IO 负载，统一走 Disk IO 信号量——
    # 避免与分析阶段长占用 QSV 槽形成饥饿（生产实测 137 文件因此被误判 FAILED）
    from src.utils import get_disk_semaphore
    io_sem = get_disk_semaphore()

    # AGENTS.md 铁律：acquire 必须带 timeout 并重试，禁止无限阻塞
    # 预筛属排队型负载（非死锁风险），预算放宽至 30s×6，避免高峰拥塞误判 FAILED
    _t_sem = time.monotonic()
    _sem_ok = acquire_with_retry(io_sem, timeout=30.0, retries=6)
    sem_wait = round(time.monotonic() - _t_sem, 2)
    if not _sem_ok:
        logger.warning("prescreen keyframes: io semaphore acquire timeout for %s", filepath)
        return {"status": "FAILED", "error": "io semaphore acquire timeout", "has_audio": 0}
    try:
        with av.open(str(filepath), options={"buffer_size": "2097152"}) as container:
            if not container.streams.video:
                return {"status": "FAILED", "error": "No video stream", "has_audio": 0}
            has_audio = 1 if len(container.streams.audio) > 0 else 0
            stream = container.streams.video[0]
            stream.codec_context.skip_frame = "NONKEY"

            first_frame: np.ndarray | None = None
            prev_frame: np.ndarray | None = None
            k = 0
            for frame in container.decode(stream):
                # 提取 Y 平面并快速切片下采样
                y_raw = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape((frame.height, frame.width))
                curr_frame = y_raw[::8, ::8]

                t_frame = float(frame.pts * stream.time_base) if (frame.pts is not None and stream.time_base) else float(k)
                sample_ts.append(t_frame)

                if first_frame is None:
                    first_frame = curr_frame
                    prev_frame = curr_frame
                    k += 1
                    continue

                # 动态环境光自适应
                mean_luma = float(np.mean(curr_frame))
                current_threshold = threshold
                if mean_luma < 50.0:
                    current_threshold = max(1.5, threshold * 0.3)

                diff_prev = float(cv2.norm(curr_frame, prev_frame, cv2.NORM_L1) / curr_frame.size)
                diff_first = float(cv2.norm(curr_frame, first_frame, cv2.NORM_L1) / curr_frame.size)
                d = max(diff_prev, diff_first)
                diffs.append(d)

                # 即时早停：一旦发现动作，立即标记为 SUSPICIOUS 返回
                if d > current_threshold:
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
                            "early_stop": True,
                            "checked_pairs": len(diffs),
                            "sem_wait": sem_wait,
                        }),
                    }

                prev_frame = curr_frame
                k += 1
                if k >= max_keyframes:
                    break
    except Exception as e:
        logger.debug("PyAV keyframes prescreen failed for %s: %s", filepath, e)
        return {"status": "FALLBACK", "error": str(e), "has_audio": has_audio}
    finally:
        io_sem.release()

    if not diffs:
        return {"status": "STATIC", "has_audio": has_audio, "result_json": json.dumps({"mode": "keyframes", "diffs": [], "early_stop": False, "sem_wait": sem_wait})}

    max_diff = max(diffs)
    status = "SUSPICIOUS" if max_diff > threshold else "STATIC"
    return {
        "status": status,
        "has_audio": has_audio,
        "result_json": json.dumps({
            "mode": "keyframes",
            "sample_ts": sample_ts,
            "diffs": diffs,
            "max_diff": max_diff,
            "threshold": threshold,
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
    if mode in ("keyframes", "stream_fps", "auto"):
        kf_res = _prescreen_keyframes(
            filepath=filepath,
            duration=duration,
            max_keyframes=min(segments, 15),
            threshold=threshold,
            gpu=gpu,
        )
        if kf_res.get("status") != "FALLBACK":
            return kf_res
        if mode == "stream_fps":
            return _prescreen_stream_fps(
                filepath,
                duration,
                segments,
                width,
                height,
                threshold,
                gpu,
                timeout=det_cfg.get("prescreen_extract_timeout", 30.0),
            )
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
        )
    ts_list = _calc_sample_timestamps(duration, segments, timestamp_margin)

    diffs: list[float] = []

    # 提取第 0 帧
    first_frame = _extract_frame(filepath, ts_list[0], width, height, gpu=gpu)
    if first_frame is None:
        return {"status": "FAILED", "error": f"failed to extract first frame at t={ts_list[0]:.1f}s"}
    
    prev_frame = first_frame

    for i in range(1, len(ts_list)):
        curr_frame = _extract_frame(filepath, ts_list[i], width, height, gpu=gpu)
        if curr_frame is None:
            return {"status": "FAILED", "error": f"failed to extract frame at t={ts_list[i]:.1f}s"}

        # 动态环境光自适应：检测当前画面平均亮度（Luma）
        mean_luma = float(np.mean(curr_frame))
        # 极低对比度（红外夜视）下，动态下调阈值，防止微小动作漏报
        current_threshold = threshold
        if mean_luma < 50.0:
            current_threshold = max(1.5, threshold * 0.3)

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
    # 取最后一帧亮度作为参考
    final_threshold = threshold
    if prev_frame is not None and float(np.mean(prev_frame)) < 50.0:
        final_threshold = max(1.5, threshold * 0.3)
        
    status = "SUSPICIOUS" if max_diff > final_threshold else "STATIC"

    return {
        "status": status,
        "result_json": json.dumps({
            "mode": "legacy_seek",
            "sample_ts": ts_list,
            "diffs": diffs,
            "max_diff": max_diff,
            "threshold": final_threshold,
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
) -> dict:
    """用单个低 FPS 解码流做预筛，避免每个采样点启动一次 seek 进程。"""
    frame_size = width * height * 3
    max_frames = max(2, segments + 1)
    sample_fps = max(1.0 / max(duration, 1.0), segments / max(duration, 1.0))
    cmd = _build_stream_fps_args(filepath, sample_fps, max_frames, width, height, gpu)

    if gpu == "qsv":
        from src.utils import get_qsv_semaphore
        io_sem = get_qsv_semaphore()
    else:
        from src.utils import get_nv_semaphore
        io_sem = get_nv_semaphore()

    stderr_lines: list[str] = []
    diffs: list[float] = []
    sample_ts: list[float] = []
    proc: subprocess.Popen | None = None
    completed_read = False

    # AGENTS.md 铁律：acquire 必须带 timeout 并重试，禁止无限阻塞
    if not acquire_with_retry(io_sem):
        logger.warning("prescreen stream_fps: io semaphore acquire timeout for %s", filepath)
        return {"status": "FAILED", "error": "io semaphore acquire timeout"}
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

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
                current_threshold = threshold
                if mean_luma < 50.0:
                    current_threshold = max(1.5, threshold * 0.3)

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
    status = "SUSPICIOUS" if max_diff > threshold else "STATIC"
    return {
        "status": status,
        "result_json": json.dumps({
            "mode": "stream_fps",
            "sample_ts": sample_ts,
            "diffs": diffs,
            "max_diff": max_diff,
            "threshold": threshold,
            "early_stop": False,
            "checked_pairs": len(diffs),
            "ffmpeg_calls": 1,
            "sample_fps": sample_fps,
        }),
    }
