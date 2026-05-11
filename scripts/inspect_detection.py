"""Inspect motion detection accuracy on a single file.

Usage:
  uv run python scripts/inspect_detection.py <filepath> [--extract-frames] [--csv]
  uv run python scripts/inspect_detection.py <filepath> --extract-frames --open-dir

Outputs:
  - Per-frame labels to CSV (if --csv)
  - Transition boundary frames as PNG (if --extract-frames)
  - Summary: segment list, motion ratio, threshold metrics
"""

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.detector import MotionDetector
from src.segment import build_segments
from src.ffmpeg import get_duration
from src.utils import load_config


def extract_frame(filepath: str, timestamp: float, output: Path, width: int = 640, height: int = 360) -> bool:
    """Extract a single frame at timestamp as PNG for visual inspection."""
    output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(timestamp),
        "-i", str(filepath),
        "-vframes", "1",
        "-q:v", "2",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Inspect motion detection accuracy")
    parser.add_argument("filepath", type=str, help="Path to MP4 file to analyze")
    parser.add_argument("--extract-frames", action="store_true", help="Extract PNG frames at transition boundaries")
    parser.add_argument("--csv", action="store_true", help="Write per-frame labels to CSV")
    parser.add_argument("--open-dir", action="store_true", help="Open output directory in Explorer")
    parser.add_argument("--outdir", type=str, default=None, help="Output directory (default: ./inspect_out)")
    args = parser.parse_args()

    filepath = Path(args.filepath)
    if not filepath.exists():
        print(f"ERROR: file not found: {filepath}")
        sys.exit(1)

    outdir = Path(args.outdir) if args.outdir else Path("./inspect_out")
    outdir.mkdir(parents=True, exist_ok=True)

    config = load_config()

    file_dur = get_duration(str(filepath))
    print(f"Analyzing: {filepath.name}")
    print(f"Duration: {file_dur:.1f}s")
    print()

    detector = MotionDetector(config, decode_gpu="cuda")
    labels = detector.analyze(str(filepath), start_offset=0.0, file_duration=file_dur)

    if not labels:
        print("ERROR: no frames analyzed")
        sys.exit(1)

    # Per-frame stats
    motion_frames = sum(1 for l in labels if l["is_motion"])
    total_frames = len(labels)
    print(f"Frames analyzed: {total_frames}")
    print(f"Motion frames:  {motion_frames} ({100 * motion_frames / total_frames:.1f}%)")
    print(f"Static frames:  {total_frames - motion_frames} ({100 * (total_frames - motion_frames) / total_frames:.1f}%)")
    print()

    # Build segments to see the final output (Before YOLO)
    seg_cfg = config.get("segment", {})
    from src.segment import _filter_short
    
    segments_raw = build_segments(
        labels,
        str(filepath),
        min_motion_dur=seg_cfg.get("min_motion_duration", 2.0),
        min_static_dur=seg_cfg.get("min_static_duration", 30.0),
        file_offset=0.0,
        apply_smoothing=False, # 延迟到 yolo 后再平滑
    )
    
    segments_no_yolo = _filter_short(
        [s for s in segments_raw], # copy
        seg_cfg.get("min_motion_duration", 2.0),
        seg_cfg.get("min_static_duration", 30.0),
        seg_cfg.get("gap_tolerance", 0.5)
    )
    
    print(f"--- 1. Segments (WITHOUT YOLO, Pure Frame Diff) ---")
    print(f"Count: {len(segments_no_yolo)}")
    print(f"{'Start':>10s} {'End':>10s} {'Dur':>8s} {'State':>10s}")
    for s in segments_no_yolo:
        print(f"{s.start_time:>9.1f}s {s.end_time:>9.1f}s {s.end_time - s.start_time:>7.1f}s {s.state:>10s}")
    print()
    
    # Run YOLO Verifier
    from src.yolo_verifier import YoloVerifier
    yolo_verifier = YoloVerifier(config)
    segments_yolo = yolo_verifier.verify(str(filepath), segments_raw, gpu="cuda")
    
    # Apply filtering now
    segments_yolo_filtered = _filter_short(
        segments_yolo,
        seg_cfg.get("min_motion_duration", 2.0),
        seg_cfg.get("min_static_duration", 30.0),
        seg_cfg.get("gap_tolerance", 0.5)
    )

    print(f"--- 2. Segments (WITH YOLO Verification) ---")
    print(f"Count: {len(segments_yolo_filtered)}")
    print(f"{'Start':>10s} {'End':>10s} {'Dur':>8s} {'State':>10s}")
    for s in segments_yolo_filtered:
        print(f"{s.start_time:>9.1f}s {s.end_time:>9.1f}s {s.end_time - s.start_time:>7.1f}s {s.state:>10s}")
    print()

    # Detect transition points (state changes in RAW labels, before smoothing)
    transitions = []
    for i in range(1, len(labels)):
        if labels[i]["is_motion"] != labels[i - 1]["is_motion"]:
            direction = "MOTION_START" if labels[i]["is_motion"] else "MOTION_END"
            transitions.append({
                "index": i,
                "time": labels[i]["time"],
                "direction": direction,
            })

    print(f"Raw transitions (before smoothing): {len(transitions)}")
    for t in transitions:
        print(f"  {t['direction']:>14s} at frame #{t['index']:>5d}  t={t['time']:.1f}s")

    # CSV output
    if args.csv:
        csv_path = outdir / f"{filepath.stem}_labels.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["frame_index", "time_sec", "is_motion", "formatted_time"])
            for i, label in enumerate(labels):
                if label["is_motion"]:
                    t = label["time"]
                    mm = int(t // 60)
                    ss = t % 60
                    writer.writerow([i, f"{t:.2f}", 1, f"{mm}:{ss:05.2f}"])
        print(f"\nCSV written: {csv_path}")

    # Extract frames at transition points
    if args.extract_frames:
        frame_dir = outdir / "frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

        # Extract transition boundary frames (last static + first motion, and vice versa)
        for t in transitions[:40]:  # cap at 40 transitions
            t_sec = t["time"]
            frame_interval = 1.0 / detector.fps
            # Frame just before transition
            t_before = max(t_sec - frame_interval, 0)
            label_before = "static" if t["direction"] == "MOTION_START" else "motion"
            fn_before = frame_dir / f"T{t['index']:04d}_{t_before:.1f}s_{label_before}_before.png"
            if not fn_before.exists():
                extract_frame(str(filepath), t_before, fn_before)
            # Frame at transition
            label_after = "motion" if t["direction"] == "MOTION_START" else "static"
            fn_after = frame_dir / f"T{t['index']:04d}_{t_sec:.1f}s_{label_after}_after.png"
            if not fn_after.exists():
                extract_frame(str(filepath), t_sec, fn_after)

        # Extract random samples from long DYNAMIC and STATIC segments
        for i, s in enumerate(segments):
            dur = s.end_time - s.start_time
            if dur > 10:  # only for segments > 10s
                mid = (s.start_time + s.end_time) / 2
                tag = s.state.lower()
                fn = frame_dir / f"seg{i:02d}_{tag}_{mid:.1f}s.png"
                if not fn.exists():
                    extract_frame(str(filepath), mid, fn)

        print(f"Frames extracted to: {frame_dir} ({len(list(frame_dir.glob('*.png')))} PNGs)")

    print(f"\nOutput directory: {outdir}")

    if args.open_dir:
        os.startfile(str(outdir))


if __name__ == "__main__":
    main()
