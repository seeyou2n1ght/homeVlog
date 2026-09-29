"""Compare fixed analysis decoder paths without changing production state.

Run with ``uv run python scripts/compare_analysis_decoders.py SOURCE --report REPORT``.
Exit 2 means non-equivalent frames or motion labels, not a performance win.
This is an input equivalence gate, not a human recognition accuracy benchmark.
"""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
import sys
import time

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import load_config
from src.hardware.ffmpeg import FFmpegProcessRegistry
from src.stages.detector import MotionDetector


def compare_frames(left, right) -> dict:
    compared = identical = absolute_error = pixels = 0
    same_shape = True
    for a, b in itertools.zip_longest(left, right):
        if a is None or b is None:
            return {"equivalent": False, "reason": "frame_count", "frames_compared": compared}
        compared += 1
        if a.shape != b.shape:
            same_shape = False
            continue
        difference = np.abs(a.astype(np.int16) - b.astype(np.int16))
        identical += int(not np.any(difference))
        absolute_error += int(difference.sum())
        pixels += difference.size
    return {
        "equivalent": compared > 0 and same_shape and identical == compared,
        "frames_compared": compared,
        "identical_frames": identical,
        "same_shape": same_shape,
        "mean_abs_pixel_difference": absolute_error / pixels if pixels else None,
    }


def compare_source(source: Path, config: dict, reference: str, candidate: str) -> dict:
    identity = (source.stat().st_size, source.stat().st_mtime_ns)
    with av.open(str(source), timeout=30) as container:
        stream = container.streams.video[0]
        duration = float(stream.duration * stream.time_base) if stream.duration else container.duration / av.time_base
    arms, pools, labels = {}, {}, {}
    match = re.search(r"_(\d{8})(\d{2})(\d{2})(\d{2})_", source.name)
    start_offset = (int(match[2]) * 3600 + int(match[3]) * 60 + int(match[4])) if match else 0
    for arm, device in (("reference_1", reference), ("candidate_1", candidate),
                        ("candidate_2", candidate), ("reference_2", reference)):
        detector = MotionDetector(config, decode_gpu=device)
        detector._stream_motion = False
        fps = detector._resolve_effective_fps(duration)
        started = time.monotonic()
        pool, candidates, meta = detector._decode_file_pipe(str(source), duration, fps)
        del candidates
        expected_min_frames = max(1, int(duration * fps) - 2)
        if not meta.get("complete") or len(pool) < expected_min_frames:
            raise RuntimeError(f"Incomplete decode: {source.name} {arm}")
        elapsed = time.monotonic() - started
        digest = hashlib.sha256()
        for frame in pool:
            digest.update(frame.tobytes())
        # Motion comparison deliberately excludes audio/YOLO and human overrides.
        motion, _ = detector.analyze_frames(pool, start_offset=start_offset, file_duration=duration, fps=fps)
        labels[arm] = motion
        pools[arm] = pool
        arms[arm] = {"device": device, "frames": len(pool), "expected_min_frames": expected_min_frames,
                     "effective_fps": fps,
                     "decode_wall_s": elapsed, "gray_sha256": digest.hexdigest()}
        print(f"{source.name} {arm}: {len(pool)} frames, {elapsed:.3f}s", flush=True)
    if identity != (source.stat().st_size, source.stat().st_mtime_ns):
        raise RuntimeError(f"Source changed during comparison: {source.name}")
    pairs = {}
    for a, b in (("reference_1", "reference_2"), ("candidate_1", "candidate_2"),
                 ("reference_1", "candidate_1")):
        result = compare_frames(pools[a], pools[b])
        result["motion_labels_equal"] = labels[a] == labels[b]
        pairs[f"{a}/{b}"] = result
    return {"source": source.name, "source_identity": identity, "duration_s": duration,
            "motion_start_offset_s": start_offset,
            "arms": arms, "pairs": pairs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sources", nargs="+", type=Path)
    parser.add_argument("--reference", choices=("qsv", "cuda"), default="qsv")
    parser.add_argument("--candidate", choices=("qsv", "cuda"), default="cuda")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    if args.report.resolve() in {source.resolve() for source in args.sources}:
        parser.error("report must not overwrite a source video")
    config = load_config()
    report = {"config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "gray frames and motion labels; no audio, YOLO, or human holdout",
              "status": "incomplete", "samples": []}
    exit_code = 0
    try:
        for source in args.sources:
            result = compare_source(source, config, args.reference, args.candidate)
            report["samples"].append(result)
            if not all(p["equivalent"] and p["motion_labels_equal"] for p in result["pairs"].values()):
                exit_code = 2
        report["status"] = "equivalent" if exit_code == 0 else "different"
    finally:
        FFmpegProcessRegistry.kill_all()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.report.with_suffix(args.report.suffix + ".tmp")
        temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
        temporary.replace(args.report)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
