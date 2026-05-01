import json
import logging
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
    """Extract single RGB frame at given timestamp using HW decode."""
    from src.ffmpeg import build_hw_decode_args
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
        return None
    raw = result.stdout
    expected = width * height * 3
    if len(raw) < expected:
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
    segments = det_cfg.get("prescreen_segments", 5)
    width, height = parse_res(
        det_cfg.get("prescreen_resolution", "320x180")
    )
    threshold = det_cfg.get("prescreen_diff_threshold", 12)
    timestamp_margin = det_cfg.get("timestamp_margin", 0.5)

    actual_dur = get_duration(filepath)
    if actual_dur is not None and actual_dur > 0:
        duration = actual_dur
    ts_list = _calc_sample_timestamps(duration, segments, timestamp_margin)

    min_check = min(10, len(ts_list) - 1)
    early_threshold = threshold * 0.5
    diffs: list[float] = []

    # 提取第 0 帧
    prev_frame = _extract_frame(filepath, ts_list[0], width, height, gpu=gpu)
    if prev_frame is None:
        return {"status": "FAILED", "error": f"failed to extract first frame at t={ts_list[0]:.1f}s"}

    for i in range(1, len(ts_list)):
        curr_frame = _extract_frame(filepath, ts_list[i], width, height, gpu=gpu)
        if curr_frame is None:
            return {"status": "FAILED", "error": f"failed to extract frame at t={ts_list[i]:.1f}s"}

        diff = np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32))
        d = float(np.mean(diff))
        diffs.append(d)

        # 【核心优化】动态判定早停：只要大于 threshold，证明有人移动，立刻中止文件处理！
        if d > threshold:
            return {
                "status": "SUSPICIOUS",
                "result_json": json.dumps({
                    "sample_ts": ts_list[:i + 1],
                    "diffs": diffs,
                    "max_diff": max(diffs),
                    "threshold": threshold,
                    "early_stop": True,
                    "checked_pairs": len(diffs),
                }),
            }

        prev_frame = curr_frame

    if not diffs:
        return {"status": "FAILED", "error": "no frame pairs to compare"}

    max_diff = max(diffs)
    status = "SUSPICIOUS" if max_diff > threshold else "STATIC"

    return {
        "status": status,
        "result_json": json.dumps({
            "sample_ts": ts_list,
            "diffs": diffs,
            "max_diff": max_diff,
            "threshold": threshold,
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
