import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src.database import VlogDatabase
from src.ffmpeg import run_ffmpeg, get_duration, build_qsv_decode_args
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


def _extract_frame(filepath: str, timestamp: float, width: int, height: int, timeout: float = 30.0) -> np.ndarray | None:
    """Extract single RGB frame at given timestamp using QSV decode."""
    args = build_qsv_decode_args(
        input_path=str(filepath),
        width=width,
        height=height,
        start_time=timestamp,
        vframes=1,
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


def _extract_frames_bulk(filepath: str, timestamps: list[float], width: int, height: int) -> list[np.ndarray]:
    """Extract frames at given timestamps (one ffmpeg seek per timestamp)."""
    frames: list[np.ndarray] = []
    for t in timestamps:
        frame = _extract_frame(filepath, t, width, height)
        if frame is None:
            raise RuntimeError(f"failed to extract frame at t={t:.1f}s")
        frames.append(frame)
    return frames


def prescreen_file(
    filepath: str,
    duration: float,
    config: dict,
) -> dict:
    """Analyze one file: extract 6 frames at segment boundaries, diff, classify."""
    det_cfg = config.get("detection", {})
    segments = det_cfg.get("prescreen_segments", 5)
    width, height = parse_res(
        det_cfg.get("prescreen_resolution", "320x180")
    )
    threshold = det_cfg.get("prescreen_diff_threshold", 12)
    timestamp_margin = det_cfg.get("timestamp_margin", 0.5)
    extract_timeout = det_cfg.get("prescreen_extract_timeout", 30.0)

    actual_dur = get_duration(filepath)
    if actual_dur is not None and actual_dur > 0:
        duration = actual_dur
    ts_list = _calc_sample_timestamps(duration, segments, timestamp_margin)
    try:
        frames = _extract_frames_bulk(filepath, ts_list, width, height)
    except RuntimeError as e:
        return {"status": "FAILED", "error": str(e)}

    diffs = []
    for i in range(len(frames) - 1):
        diff = np.abs(frames[i + 1].astype(np.float32) - frames[i].astype(np.float32))
        diffs.append(float(np.mean(diff)))

    max_diff = max(diffs)
    if max_diff > threshold:
        status = "SUSPICIOUS"
    else:
        status = "STATIC"

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
    """Run Pass1 prescreen for all PENDING files of a given date/cam."""
    config = load_config()
    pending = db.get_prescreen_pending(date, cam_index)
    if not pending:
        logger.info("prescreen: no PENDING files for %s cam%d", date, cam_index)
        return {"done": 0, "static": 0, "suspicious": 0, "failed": 0}

    logger.info("prescreen: %d files for %s cam%d", len(pending), date, cam_index)
    parallel = config.get("detection", {}).get("prescreen_parallel", parallel)

    stats = {"done": 0, "static": 0, "suspicious": 0, "failed": 0}
    t0 = time.monotonic()

    if parallel <= 1:
        for task in pending:
            local = _process_one(db, task, config)
            for k in stats:
                stats[k] += local[k]
    else:
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            futures = {pool.submit(_process_one, db, task, config): task for task in pending}
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


def _process_one(db: VlogDatabase, task: dict, config: dict) -> dict:
    filepath = task["filepath"]
    duration = task["file_duration"] or 300.0
    perf = get_perf()

    t0 = time.monotonic()
    result = prescreen_file(filepath, duration, config)
    elapsed = time.monotonic() - t0

    # Parse prescreen result for perf record
    extra = {"status": result["status"]}
    try:
        rj = json.loads(result.get("result_json", "{}"))
        extra["max_diff"] = rj.get("max_diff", 0)
        extra["n_segments"] = len(rj.get("sample_ts", []))
    except (json.JSONDecodeError, TypeError):
        pass

    perf.add(PerfRecord(
        stage="prescreen",
        file=Path(filepath).name,
        gpu="qsv",
        duration=round(elapsed, 3),
        extra=extra,
    ))

    local = {"done": 1, "static": 0, "suspicious": 0, "failed": 0}

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
