"""Measure one isolated run using its own code/config and reviewed source intervals.

uv run python scripts/measure_vlog_quality.py --snapshot scratch/quality_20260320/baseline \
    --date 20260320 --annotations scratch/quality_20260320/samples/review.json

Run separately for before/after snapshots. Interval scores measure playback policy
on the reviewed sample only, not population-level recognition accuracy.
"""
import argparse
from collections import defaultdict
import inspect
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--annotations", type=Path)
    args = parser.parse_args()
    snapshot = args.snapshot.resolve()
    sys.path.insert(0, str(snapshot))
    from src.database import VlogDatabase
    from src.timeline import build_timeline_from_rows, compute_display_plans
    from src.utils import load_config

    cfg = load_config(snapshot / "config/settings.yaml")
    db = VlogDatabase(snapshot / "data/vlog.db")
    try:
        rows = db.get_all_file_tasks_for_date(args.date, args.cam)
        timeline = build_timeline_from_rows(rows, args.date, config=cfg)
    finally:
        db.close()
    seg, render, presence, micro = [cfg.get(k, {}) for k in ("segment", "render", "presence", "micro_motion")]
    options = {k: seg[k] for k in ("static_keyframe_interval", "keyframe_display_duration", "min_static_display_duration", "max_static_display_duration") if k in seg}
    options.update(speed_ramping=render.get("speed_ramping_enabled", True),
                   ramp_duration_s=render.get("ramp_duration_s", 1),
                   presence_speed_factor=presence.get("speed_factor", 4),
                   night_stationary_speed_factor=presence.get("night_speed_factor", 16),
                   micro_motion_cruise_speed=micro.get("cruise_speed", 16))
    if "micro_motion_anchor_s" in inspect.signature(compute_display_plans).parameters:
        options["micro_motion_anchor_s"] = micro.get("anchor_duration_s", 3)
    if "output_fps" in inspect.signature(compute_display_plans).parameters:
        options["output_fps"] = cfg.get("output", {}).get("fps", 20)
    plans = compute_display_plans(timeline, **options)
    source_seconds, display_seconds = defaultdict(float), defaultdict(float)
    by_file = defaultdict(list)
    for item, (duration, _) in zip(timeline, plans):
        source_seconds[item.state] += item.end_in_file - item.start_in_file
        display_seconds[item.state] += duration
        by_file[Path(item.filepath).name].append(item)
    result = {
        "snapshot": str(snapshot), "date": args.date, "files": len(rows),
        "source_seconds": sum(float(r.get("file_duration") or 0) for r in rows),
        "source_seconds_by_state": dict(source_seconds),
        "display_seconds_by_state": dict(display_seconds),
        "planned_display_seconds": sum(display_seconds.values()),
        "unresolved_files": [r["filepath"] for r in rows if r.get("prescreen_status") != "STATIC" and r.get("analysis_status") != "ANALYZED"],
    }
    result["media"] = []
    for path in (snapshot / "output").glob(f"DailyVlog_{args.date}_*.mp4"):
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                               capture_output=True, text=True, check=True, timeout=30)
        result["media"].append({"path": str(path), "bytes": path.stat().st_size, "probe": json.loads(probe.stdout)})
    perf_files = sorted((snapshot / "logs/perf").glob(f"perf_{args.date}_cam{args.cam}_*.json"))
    if perf_files:
        perf = json.loads(perf_files[-1].read_text(encoding="utf-8"))
        result["pipeline_seconds"] = perf.get("pipeline_duration")
        result["perf_file"] = str(perf_files[-1])
    if args.annotations:
        review = json.loads(args.annotations.read_text(encoding="utf-8"))
        result["review_scope"] = review["scope"]
        scores = []
        for sample in review["intervals"]:
            normal = covered = 0.0
            for item in by_file[sample["file"]]:
                overlap = max(0, min(sample["end"], item.end_in_file) - max(sample["start"], item.start_in_file))
                covered += overlap
                if item.state in ("DYNAMIC", "DYNAMIC_AUDIO"):
                    normal += overlap
            duration = sample["end"] - sample["start"]
            if covered > duration + 0.001:
                raise ValueError(f"Overlapping timeline would inflate sample {sample['id']}")
            scores.append(dict(sample, normal_seconds=normal, covered_seconds=covered,
                               correct_seconds=normal if sample["expected"] == "normal" else max(0, covered-normal),
                               duration=duration))
        result["review_scores"] = scores
        result["review_correct_seconds"] = sum(s["correct_seconds"] for s in scores)
        result["review_total_seconds"] = sum(s["duration"] for s in scores)
    target = snapshot / "quality_measurement.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
