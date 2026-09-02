"""
Performance Benchmark & Throughput Test Suite for HomeVlog.
Validates algorithm speed, memory footprint, and high-volume scalability.
"""

import time
import numpy as np
import pytest

from src.detector import _median_filter, _smooth_labels
from src.segment import build_segments, merge_cross_file, Segment
from src.timeline import TimelineSegment, build_concat_filter
from src.database import VlogDatabase


class TestAlgorithmThroughputBenchmark:
    """Benchmarks for core algorithm speed and efficiency."""

    def test_median_filter_10k_points_throughput(self):
        # 10,000 frame energy points (represents ~33 minutes of video at 5fps)
        data = np.random.uniform(0.0, 20.0, size=10000).tolist()
        t0 = time.monotonic()
        filtered = _median_filter(data, window=7)
        elapsed = time.monotonic() - t0

        assert len(filtered) == 10000
        # Benchmark requirement: 10k points processed in < 100ms
        assert elapsed < 0.10, f"Median filter took {elapsed:.4f}s for 10k points (expected < 0.10s)"

    def test_smooth_labels_10k_frames_throughput(self):
        raw = (np.random.rand(10000) > 0.8).tolist()
        t0 = time.monotonic()
        smoothed = _smooth_labels(raw, min_motion=3, min_static=5, noise_suppress=2)
        elapsed = time.monotonic() - t0

        assert len(smoothed) == 10000
        assert elapsed < 0.05, f"Label smoothing took {elapsed:.4f}s for 10k frames (expected < 0.05s)"

    def test_segment_building_and_merging_throughput(self):
        # 5,000 alternating frames
        labels = [
            {"time": float(i), "is_motion": bool(i % 10 < 3), "energy": float(i % 15)}
            for i in range(5000)
        ]
        t0 = time.monotonic()
        segs = build_segments(labels, source_file="bench.mp4", min_motion_dur=2.0, min_static_dur=5.0)
        merged = merge_cross_file(segs, gap_tolerance=0.5)
        elapsed = time.monotonic() - t0

        assert len(merged) > 0
        assert elapsed < 0.10, f"Segment building took {elapsed:.4f}s for 5k frames (expected < 0.10s)"

    def test_concat_filtergraph_generation_throughput(self):
        # 100 segments timeline
        segs = [
            TimelineSegment(
                filepath=f"/dummy/file_{i // 5}.mp4",
                input_index=i // 5,
                start_in_file=float((i % 5) * 20),
                end_in_file=float((i % 5 + 1) * 20),
                state="DYNAMIC" if i % 2 == 0 else "STATIC",
                duration=20.0,
            )
            for i in range(100)
        ]
        rows = [{"filepath": f"/dummy/file_{k}.mp4", "has_audio": 1} for k in range(20)]

        t0 = time.monotonic()
        fc = build_concat_filter(timeline=segs, rows=rows)
        elapsed = time.monotonic() - t0

        assert len(fc) > 1000
        assert elapsed < 0.05, f"Filtergraph generation took {elapsed:.4f}s for 100 segments (expected < 0.05s)"

    def test_database_batch_inserts_and_queries_throughput(self, isolated_db):
        db = isolated_db
        t0 = time.monotonic()
        for i in range(100):
            db.add_file_task(
                filepath=f"/dummy/task_{i:03d}.mp4",
                cam_index=0,
                date="20260901",
                file_start_time=f"2026090100{i % 60:02d}00",
                file_end_time=f"2026090100{(i+1) % 60:02d}00",
                file_duration=60.0,
            )
        tasks = db.get_all_file_tasks_for_date("20260901", 0)
        elapsed = time.monotonic() - t0

        assert len(tasks) == 100
        assert elapsed < 0.20, f"Database 100 tasks insert & fetch took {elapsed:.4f}s (expected < 0.20s)"
