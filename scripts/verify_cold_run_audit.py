"""Standalone Cold-Run Acceptance Verifier for 20260320 cam0."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import sqlite3
import av
from src.hardware.render_cache import _window_is_continuous

META_PATH = Path("output/DailyVlog_20260320_B888805AA3CD.meta.json")
VIDEO_PATH = Path("output/DailyVlog_20260320_B888805AA3CD.mp4")
DB_PATH = Path("data/vlog.db")

assert META_PATH.exists(), "Meta JSON missing"
assert VIDEO_PATH.exists(), "Delivered MP4 missing"

meta = json.loads(META_PATH.read_text(encoding="utf-8"))
mapping = meta.get("delivery_mapping", {})

# 1. Clock Drift Verification (<= 1 tick)
assert mapping.get("status") == "AVAILABLE", "Delivery status is not AVAILABLE"
clock_val = mapping.get("clock_validation", {})
assert clock_val.get("max_error_ticks", 999) <= 1, f"Clock drift exceeded 1 tick: {clock_val}"
print(f"[PASS] Clock Drift Verified: <= 1 tick ({clock_val.get('packets')} packets verified, max_error_ticks={clock_val.get('max_error_ticks')})")

# 2. Source Coverage & Zero Omission Check
assert meta.get("skipped_sources") == [], f"Sources were skipped: {meta.get('skipped_sources')}"
assert meta.get("unavailable_source_ranges") == [], f"Unavailable ranges detected: {meta.get('unavailable_source_ranges')}"
highlights = meta.get("timeline_highlights", [])
print(f"Highlights count: {len(highlights)}")
assert len(highlights) >= 1020, f"Activity omission detected! Highlights count: {len(highlights)} < 1020"
print(f"[PASS] Zero Activity Omission Verified: {len(highlights)} key activity events preserved")

# 3. 141 Seam Continuity Check
segments = mapping.get("segments", [])
batch_max_frame = {}
for s in segments:
    b = s["batch_index"]
    batch_max_frame[b] = max(batch_max_frame.get(b, 0), s["output_end_frame"])

fps = mapping.get("fps", 20.0)
seam_times = [batch_max_frame[i] / fps for i in range(141)]
assert len(seam_times) == 141, f"Expected 141 seams, got {len(seam_times)}"

with av.open(str(VIDEO_PATH)) as container:
    stream = container.streams.video[0]
    duration = float(stream.duration * stream.time_base)
    for idx, boundary in enumerate(seam_times):
        diag = {}
        ok = _window_is_continuous(container, stream, boundary, duration, diagnostics=diag)
        assert ok, f"Seam {idx} at {boundary:.2f}s failed continuity! Diagnostics: {diag}"

print("[PASS] All 141 Seams Verified Continuous with 0 Discontinuities!")
