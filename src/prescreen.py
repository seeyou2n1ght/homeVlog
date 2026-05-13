import json
import logging
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.database import VlogDatabase
from src.ffmpeg import run_ffmpeg, get_duration, build_hw_decode_args
from src.utils import load_config, parse_res
from src.monitor import get_perf, PerfRecord

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

    io_sem.acquire()
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


def run_prescreen_for_cam(
    db: VlogDatabase,
    date: str,
    cam_index: int,
    parallel: int = 4,
) -> dict:
    """Run Pass1 prescreen using dual-GPU work-stealing queue."""
    from queue import Queue
    config = load_config()
    pending = db.get_prescreen_pending(date, cam_index)
    if not pending:
        logger.info("prescreen: no PENDING files for %s cam%d", date, cam_index)
        return {"done": 0, "static": 0, "suspicious": 0, "failed": 0}

    logger.info("prescreen: %d files for %s cam%d", len(pending), date, cam_index)
    parallel = config.get("detection", {}).get("prescreen_parallel", parallel)

    sorted_tasks = sorted(pending, key=lambda t: t.get("file_duration") or 0, reverse=True)
    task_queue: Queue[dict | None] = Queue()
    for task in sorted_tasks:
        task_queue.put(task)
        
    for _ in range(parallel):
        task_queue.put(None)

    stats = {"done": 0, "static": 0, "suspicious": 0, "failed": 0}
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = []
        for i in range(parallel):
            # Alternating GPUs for workers: qsv, cuda, qsv, cuda...
            gpu = "qsv" if i % 2 == 0 else "cuda"
            futures.append(pool.submit(_prescreen_worker, db, task_queue, config, gpu))
            
        for f in as_completed(futures):
            try:
                local = f.result()
                for k in stats:
                    stats[k] += local[k]
            except Exception:
                logger.exception("prescreen worker crashed")

    elapsed = time.monotonic() - t0
    logger.info(
        "prescreen %s cam%d done in %.1fs: static=%d suspicious=%d failed=%d (%.1f files/s)",
        date, cam_index, elapsed, stats["static"], stats["suspicious"], stats["failed"],
        len(pending) / elapsed if elapsed > 0 else 0,
    )
    return stats


def _prescreen_worker(db: VlogDatabase, task_queue, config: dict, gpu: str) -> dict:
    """Process PENDING files from queue with given GPU (Work-Stealing)."""
    perf = get_perf()
    local = {"done": 0, "static": 0, "suspicious": 0, "failed": 0}

    while True:
        task = task_queue.get()
        if task is None:
            break

        filepath = task["filepath"]
        duration = task["file_duration"] or 300.0

        t0 = time.monotonic()
        result = prescreen_file(filepath, duration, config, gpu)
        elapsed = time.monotonic() - t0

        extra = {"status": result["status"]}
        try:
            rj = json.loads(result.get("result_json", "{}"))
            extra["max_diff"] = rj.get("max_diff", 0)
            extra["n_segments"] = len(rj.get("sample_ts", []))
        except (json.JSONDecodeError, TypeError):
            pass

        perf.add(PerfRecord(
            stage="prescreen", file=Path(filepath).name, gpu=gpu,
            duration=round(elapsed, 3), extra=extra,
        ))

        local["done"] += 1
        if result["status"] == "FAILED":
            db.set_prescreen_result(filepath, "FAILED", result.get("result_json", ""))
            local["failed"] += 1
        else:
            db.set_prescreen_result(filepath, result["status"], result.get("result_json", ""))
            if result["status"] == "STATIC":
                local["static"] += 1
            else:
                local["suspicious"] += 1

    return local
