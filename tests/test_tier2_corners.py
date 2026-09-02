"""
Tier 2: Boundary & Corner Case Test Suite for HomeVlog.
Covers:
- 2s natural gap handling and cross-file boundary preservation
- Zero-motion / completely static video streams
- Silent audio and missing audio stream fallbacks
- Max hardware concurrency contention and thread safety
- Empty timeline and resource safeguard threshold handling
- Micro/sub-second segment duration clamping
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.database import VlogDatabase
from src.prescreen import prescreen_file
from src.renderer import render_vlog
from src.segment import Segment, merge_cross_file, build_segments
from src.timeline import TimelineSegment, build_concat_filter, build_timeline
from src.utils import (
    get_nv_semaphore,
    get_qsv_semaphore,
    reset_semaphores,
    check_disk_space,
    ts_to_unix,
)
from tests.helpers import (
    parse_ffmpeg_filtergraph,
    verify_filtergraph_labels_closure,
)


class TestGapHandling:
    """Tests for natural gaps and timestamp discontinuities between surveillance files."""

    def test_2s_natural_gap_preserves_separate_segments(self):
        # File 1 ends at 100.0s, File 2 starts at 102.0s (2.0s gap > 0.5s tolerance)
        segs = [
            Segment(start_time=0.0, end_time=100.0, state="STATIC", source_file="00_20260901000000_20260901000140.mp4", file_start_offset=0.0),
            Segment(start_time=102.0, end_time=200.0, state="STATIC", source_file="00_20260901000142_20260901000320.mp4", file_start_offset=102.0),
        ]
        merged = merge_cross_file(segs, gap_tolerance=0.5)
        # Should remain 2 separate segments
        assert len(merged) == 2
        assert merged[0].end_time == 100.0
        assert merged[1].start_time == 102.0

    def test_gap_smaller_than_tolerance_merges(self):
        # Gap is 0.3s <= 0.5s tolerance -> merges into one segment
        segs = [
            Segment(start_time=0.0, end_time=100.0, state="STATIC", source_file="f1", file_start_offset=0.0),
            Segment(start_time=100.3, end_time=200.0, state="STATIC", source_file="f2", file_start_offset=100.3),
        ]
        merged = merge_cross_file(segs, gap_tolerance=0.5)
        assert len(merged) == 1
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 200.0


class TestZeroMotionVideos:
    """Tests for handling videos with zero movement / completely static background."""

    def test_zero_motion_timeline_filter_generation(self):
        segs = [
            TimelineSegment(
                filepath="/dummy/static_video.mp4",
                input_index=0,
                start_in_file=0.0,
                end_in_file=300.0,
                state="STATIC",
                duration=300.0,
            )
        ]
        rows = [{"filepath": "/dummy/static_video.mp4", "has_audio": 0}]

        filter_str = build_concat_filter(
            timeline=segs,
            rows=rows,
            static_keyframe_interval=30.0,
            keyframe_display_duration=0.5,
            min_static_display_duration=1.5,
        )

        parsed = parse_ffmpeg_filtergraph(filter_str)
        assert len(parsed["video_trims"]) == 1
        # 300s compressed by 60x = 5s target duration
        assert "setpts=(PTS-STARTPTS)/60.0" in parsed["video_trims"][0]
        assert len(parsed["audio_nulls"]) == 1
        assert "anullsrc=r=48000:cl=mono:d=5.000" in parsed["audio_nulls"][0]

        ok, msg = verify_filtergraph_labels_closure(filter_str)
        assert ok, msg


class TestSilentAndMissingAudio:
    """Tests for videos without audio stream or with silent audio."""

    def test_missing_audio_stream_fallback(self):
        segs = [
            TimelineSegment(
                filepath="/dummy/no_audio.mp4",
                input_index=0,
                start_in_file=10.0,
                end_in_file=20.0,
                state="DYNAMIC",
                duration=10.0,
            )
        ]
        # has_audio = 0
        rows = [{"filepath": "/dummy/no_audio.mp4", "has_audio": 0}]

        filter_str = build_concat_filter(timeline=segs, rows=rows)
        parsed = parse_ffmpeg_filtergraph(filter_str)

        # Dynamic segment without audio should use anullsrc for duration 10.0s
        assert len(parsed["audio_nulls"]) == 1
        assert "anullsrc=r=48000:cl=mono:d=10.000" in parsed["audio_nulls"][0]
        assert len(parsed["audio_trims"]) == 0

        ok, msg = verify_filtergraph_labels_closure(filter_str)
        assert ok, msg

    def test_mixed_audio_and_no_audio_files(self):
        segs = [
            TimelineSegment(filepath="f_with_audio.mp4", input_index=0, start_in_file=0.0, end_in_file=10.0, state="DYNAMIC", duration=10.0),
            TimelineSegment(filepath="f_no_audio.mp4", input_index=1, start_in_file=0.0, end_in_file=10.0, state="DYNAMIC", duration=10.0),
        ]
        rows = [
            {"filepath": "f_with_audio.mp4", "has_audio": 1},
            {"filepath": "f_no_audio.mp4", "has_audio": 0},
        ]

        filter_str = build_concat_filter(timeline=segs, rows=rows)
        parsed = parse_ffmpeg_filtergraph(filter_str)

        assert len(parsed["audio_trims"]) == 1
        assert len(parsed["audio_nulls"]) == 1
        assert parsed["concat_n"] == 2

        ok, msg = verify_filtergraph_labels_closure(filter_str)
        assert ok, msg


class TestMaxConcurrencyContention:
    """Tests for hardware semaphore concurrency bounds under high thread contention."""

    def test_nv_semaphore_concurrency_limit(self, mock_config):
        with patch("src.utils.load_config", return_value=mock_config):
            reset_semaphores()
            sem = get_nv_semaphore()
            limit = 3

            active_workers = 0
            max_observed = 0
            lock = threading.Lock()
            barrier = threading.Barrier(12)

            def worker():
                nonlocal active_workers, max_observed
                barrier.wait()
                with sem:
                    with lock:
                        active_workers += 1
                        if active_workers > max_observed:
                            max_observed = active_workers
                    time.sleep(0.02)
                    with lock:
                        active_workers -= 1

            threads = [threading.Thread(target=worker) for _ in range(12)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert max_observed <= limit

    def test_qsv_semaphore_concurrency_limit(self, mock_config):
        with patch("src.utils.load_config", return_value=mock_config):
            reset_semaphores()
            sem = get_qsv_semaphore()
            limit = 8

            active_workers = 0
            max_observed = 0
            lock = threading.Lock()
            barrier = threading.Barrier(20)

            def worker():
                nonlocal active_workers, max_observed
                barrier.wait()
                with sem:
                    with lock:
                        active_workers += 1
                        if active_workers > max_observed:
                            max_observed = active_workers
                    time.sleep(0.01)
                    with lock:
                        active_workers -= 1

            threads = [threading.Thread(target=worker) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert max_observed <= limit

    def test_database_wal_concurrent_reads_writes(self, isolated_db):
        db = isolated_db
        # Pre-populate tasks
        for i in range(20):
            db.add_file_task(
                filepath=f"/dummy/file_{i}.mp4",
                cam_index=0,
                date="20260901",
                file_start_time=f"2026090100{i:02d}00",
                file_end_time=f"2026090100{i+1:02d}00",
                file_duration=60.0,
            )

        barrier = threading.Barrier(10)
        errors = []

        def worker(w_id):
            barrier.wait()
            try:
                for i in range(10):
                    fp = f"/dummy/file_{w_id * 2 + (i % 2)}.mp4"
                    db.set_prescreen_result(fp, "SUSPICIOUS", "{}")
                    db.get_all_file_tasks_for_date("20260901", 0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestEmptyTimelineAndResourceGuards:
    """Tests for empty timeline handling and disk space safeguard thresholds."""

    def test_build_timeline_empty_db(self, isolated_db):
        timeline = build_timeline(isolated_db, date="20260901", cam_index=0)
        assert timeline == []

    def test_render_vlog_empty_timeline_returns_none(self, isolated_db):
        res = render_vlog(isolated_db, date="20260901", cam_index=0)
        assert res is None

    def test_check_disk_space_threshold(self, temp_test_dir):
        # Requiring 100,000 GB will fail on any normal disk
        assert not check_disk_space(temp_test_dir, min_gb=100000000)
        # Requiring 0 GB should pass
        assert check_disk_space(temp_test_dir, min_gb=0)


class TestMicroSegmentClamping:
    """Tests for handling sub-second and ultra-short segment durations."""

    def test_subsecond_segment_clamping_in_timeline(self, isolated_db):
        db = isolated_db
        # Add a file with a 0.05s segment
        db.add_file_task(
            filepath="/dummy/micro.mp4",
            cam_index=0,
            date="20260901",
            file_start_time="20260901000000",
            file_end_time="20260901000100",
            file_duration=60.0,
        )
        micro_segs = [
            Segment(start_time=10.0, end_time=10.05, state="DYNAMIC", source_file="/dummy/micro.mp4", file_start_offset=0.0)
        ]
        from src.segment import segments_to_json
        db.set_analysis_result("/dummy/micro.mp4", "ANALYZED", segments_to_json(micro_segs))

        timeline = build_timeline(db, date="20260901", cam_index=0)
        assert len(timeline) == 1
        # Duration should be clamped to at least min_segment_duration (0.1s)
        assert pytest.approx(timeline[0].duration, abs=1e-4) == 0.1 or timeline[0].duration >= 0.1
