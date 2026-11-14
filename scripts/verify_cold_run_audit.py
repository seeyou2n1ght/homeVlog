"""Standalone Cold-Run Acceptance Verifier for HomeVlog."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import av
from src.hardware.render_cache import _window_is_continuous

ap = argparse.ArgumentParser(description="Verify HomeVlog delivery clock drift and seam continuity")
ap.add_argument("date", nargs="?", default="20260320", help="Delivery date YYYYMMDD (default: 20260320)")
ap.add_argument("--mac", type=str, default=None, help="Optional camera MAC filter (e.g. B888805AA3CD)")
args = ap.parse_args()

output_dir = Path("output")
if args.mac:
    META_PATH = output_dir / f"DailyVlog_{args.date}_{args.mac}.meta.json"
    VIDEO_PATH = output_dir / f"DailyVlog_{args.date}_{args.mac}.mp4"
else:
    candidates = sorted(output_dir.glob(f"DailyVlog_{args.date}_*.meta.json"))
    if len(candidates) == 1:
        META_PATH = candidates[0]
        VIDEO_PATH = META_PATH.with_name(META_PATH.name[:-len(".meta.json")] + ".mp4")
    elif len(candidates) > 1:
        names = ", ".join(c.name for c in candidates)
        raise SystemExit(f"Multiple meta files matched {args.date}; pass --mac explicitly: {names}")
    else:
        META_PATH = output_dir / f"DailyVlog_{args.date}_B888805AA3CD.meta.json"
        VIDEO_PATH = output_dir / f"DailyVlog_{args.date}_B888805AA3CD.mp4"

assert META_PATH.exists(), f"Meta JSON missing: {META_PATH}"
assert VIDEO_PATH.exists(), f"Delivered MP4 missing: {VIDEO_PATH}"

meta = json.loads(META_PATH.read_text(encoding="utf-8"))
mapping = meta.get("delivery_mapping", {})

# 1. Clock Drift Verification (<= 1 tick)
assert mapping.get("status") == "AVAILABLE", "Delivery status is not AVAILABLE"
clock_val = mapping.get("clock_validation", {})
assert clock_val.get("max_error_ticks", 999) <= 1, f"Clock drift exceeded 1 tick: {clock_val}"
print(f"[PASS] Clock Drift Verified: <= 1 tick ({clock_val.get('packets')} packets verified, max_error_ticks={clock_val.get('max_error_ticks')})")

# 2. Source coverage; highlight counts are not independent event ground truth.
assert meta.get("skipped_sources") == [], f"Sources were skipped: {meta.get('skipped_sources')}"
assert meta.get("unavailable_source_ranges") == [], f"Unavailable ranges detected: {meta.get('unavailable_source_ranges')}"
highlights = meta.get("timeline_highlights", [])
print(f"Timeline highlights reported by the pipeline: {len(highlights)}")

# 3. Seam Continuity Check
segments = mapping.get("segments", [])
batch_max_frame = {}
for s in segments:
    b = s["batch_index"]
    batch_max_frame[b] = max(batch_max_frame.get(b, 0), s["output_end_frame"])

num_batches = len(batch_max_frame)
expected_seams = max(0, num_batches - 1)
fps = mapping.get("fps", 20.0)
seam_times = [batch_max_frame[i] / fps for i in range(expected_seams)]
assert len(seam_times) == expected_seams, f"Expected {expected_seams} seams, got {len(seam_times)}"

with av.open(str(VIDEO_PATH)) as container:
    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base)
    for idx, boundary in enumerate(seam_times):
        diag = {}
        ok = _window_is_continuous(container, stream, boundary, duration, diagnostics=diag)
        assert ok, f"Seam {idx} at {boundary:.2f}s failed continuity! Diagnostics: {diag}"

print(f"[PASS] All {expected_seams} Seams Verified Continuous with 0 Discontinuities!")
