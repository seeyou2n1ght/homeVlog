"""
Tier 3: Cross-Feature Interaction Test Suite for HomeVlog.
Covers:
- Multimodal DYNAMIC_AUDIO interactions, 1x speed preservation & YOLO demotion exemption
- Early termination timestamp closure across batch partitioning boundaries
- Dynamic work-stealing preemption transitions during streaming pipeline execution
- YOLOv11 tensor batching, energy-based inference bypass & error fallback resilience
"""

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.database import VlogDatabase
from src.pipeline import StreamingOrchestrator
from src.segment import Segment, build_segments, segments_to_json
from src.timeline import TimelineSegment, build_concat_filter, build_timeline
from src.utils import WorkStealingManager
from src.yolo_verifier import YoloVerifier
from tests.helpers import (
    parse_ffmpeg_filtergraph,
    verify_filtergraph_labels_closure,
)


class TestMultimodalDynamicAudioInteractions:
    """Tests for DYNAMIC_AUDIO multimodal state, audio preservation, and YOLO verification exemption."""

    def test_dynamic_audio_preserved_at_1x_speed(self):
        # A DYNAMIC segment with audio must be rendered at 1.0x speed with audio track
        segs = [
            TimelineSegment(
                filepath="/dummy/audio_event.mp4",
                input_index=0,
                start_in_file=10.0,
                end_in_file=25.0,
                state="DYNAMIC",
                duration=15.0,
            )
        ]
        rows = [{"filepath": "/dummy/audio_event.mp4", "has_audio": 1}]

        filter_str = build_concat_filter(segs, rows=rows)
        parsed = parse_ffmpeg_filtergraph(filter_str)

        assert len(parsed["video_trims"]) == 1
        # Preserved at 1.0x speed
        assert "setpts=PTS-STARTPTS" in parsed["video_trims"][0]
        # Audio trimmed from input stream, not replaced by nullsrc
        assert len(parsed["audio_trims"]) == 1
        assert "[0:a]atrim=start=10.000:end=25.000" in parsed["audio_trims"][0]
        assert len(parsed["audio_nulls"]) == 0

        ok, msg = verify_filtergraph_labels_closure(filter_str)
        assert ok, msg

    def test_yolo_verifier_exempts_dynamic_audio_or_high_energy(self, mock_config):
        mock_config["yolo"]["enabled"] = True
        mock_config["yolo"]["skip_energy_threshold"] = 12.0

        with patch("src.yolo_verifier.YoloVerifier.__init__", return_value=None):
            verifier = YoloVerifier(mock_config)
            verifier.enabled = True
            verifier.skip_energy_threshold = 12.0
            verifier.sample_fps = 0.5
            verifier.target_classes = {0, 1, 2}
            verifier.confidence = 0.25
            verifier.device = "cpu"
            verifier.model = MagicMock()

            # Segment 1: High energy (15.0 >= 12.0) -> should be skipped from demotion
            seg_high_energy = Segment(start_time=0.0, end_time=10.0, state="DYNAMIC", source_file="f1", file_start_offset=0.0, max_energy=15.0)
            # Segment 2: Low energy (5.0 < 12.0) -> would be verified by YOLO
            seg_low_energy = Segment(start_time=10.0, end_time=20.0, state="DYNAMIC", source_file="f1", file_start_offset=0.0, max_energy=5.0)

            frames_buffer = {i: np.zeros((234, 416, 3), dtype=np.uint8) for i in range(100)}

            # Mock _verify_segment_batch to simulate YOLO finding no objects
            with patch.object(verifier, "_verify_segment_batch", return_value=False):
                verified = verifier.verify("f1", [seg_high_energy, seg_low_energy], frames_buffer=frames_buffer)

                # High energy segment must stay DYNAMIC
                assert verified[0].state == "DYNAMIC"
                # Low energy segment with no detections demotes to STATIC
                assert verified[1].state == "STATIC"


class TestEarlyTerminationAndBatchPartitioning:
    """Tests for early termination timeline closure and physical file batch splitting."""

    def test_early_term_closure_and_timeline_duration(self, isolated_db):
        db = isolated_db
        # File duration is 300s, but early terminated after 20 frames (4s)
        db.add_file_task(
            filepath="/dummy/early_term.mp4",
            cam_index=0,
            date="20260901",
            file_start_time="20260901000000",
            file_end_time="20260901000500",
            file_duration=300.0,
        )

        # Labels generated with early termination point anchored at 300.0s
        segs = [
            Segment(start_time=0.0, end_time=300.0, state="STATIC", source_file="/dummy/early_term.mp4", file_start_offset=0.0, max_energy=0.2)
        ]
        db.set_analysis_result("/dummy/early_term.mp4", "ANALYZED", segments_to_json(segs))

        timeline = build_timeline(db, date="20260901", cam_index=0)
        assert len(timeline) == 1
        assert timeline[0].start_in_file == 0.0
        assert timeline[0].end_in_file == 300.0
        assert timeline[0].duration == 300.0

    def test_cross_file_contiguous_merge_and_alternating_states(self, isolated_db):
        db = isolated_db
        # 3 files: File 0 has STATIC, File 1 has DYNAMIC, File 2 has STATIC
        states = ["STATIC", "DYNAMIC", "STATIC"]
        for i, st in enumerate(states):
            db.add_file_task(
                filepath=f"/dummy/batch_file_{i}.mp4",
                cam_index=0,
                date="20260901",
                file_start_time=f"20260901000{i}00",
                file_end_time=f"20260901000{i+1}00",
                file_duration=100.0,
            )
            segs = [
                Segment(start_time=i * 100.0, end_time=(i + 1) * 100.0, state=st, source_file=f"/dummy/batch_file_{i}.mp4", file_start_offset=i * 100.0)
            ]
            db.set_analysis_result(f"/dummy/batch_file_{i}.mp4", "ANALYZED", segments_to_json(segs))

        timeline = build_timeline(db, date="20260901", cam_index=0)
        # Because states alternate, they remain 3 distinct timeline segments
        assert len(timeline) == 3
        assert timeline[0].state == "STATIC"
        assert timeline[1].state == "DYNAMIC"
        assert timeline[2].state == "STATIC"


class TestStreamingWorkStealingPreemption:
    """Tests for work-stealing preemption state transitions during pipeline streaming."""

    def test_work_stealing_dynamic_preemption_lifecycle(self, mock_config):
        mock_config["hardware"]["device"] = "cuda:0"
        mock_config["scheduler"]["watermark_high"] = 5
        mock_config["scheduler"]["watermark_low"] = 2
        mgr = WorkStealingManager(config=mock_config)

        # 1. Low watermark -> QSV
        assert mgr.get_analysis_device(queue_size=2) == "qsv"
        assert mgr.state == "NORMAL_DECOUPLED"

        # 2. Queue builds up to 6 (>= watermark_high 5) -> CUDA burst
        assert mgr.get_analysis_device(queue_size=6) == "cuda"
        assert mgr.state == "COOPERATIVE_BURST"

        # 3. Pass 2 Render starts -> preempts NVDEC and forces QSV
        mgr.register_render_start()
        assert mgr.get_analysis_device(queue_size=10) == "qsv"
        assert mgr.state == "RENDER_PREEMPTION_YIELD"

        # 4. Pass 2 Render finishes -> restores CUDA burst if queue is still high
        mgr.register_render_end()
        assert mgr.get_analysis_device(queue_size=8) == "cuda"
        assert mgr.state == "COOPERATIVE_BURST"

        # 5. Queue drains down to 1 (<= watermark_low 2) -> returns to Normal Decoupled
        assert mgr.get_analysis_device(queue_size=1) == "qsv"
        assert mgr.state == "NORMAL_DECOUPLED"


class TestYoloVerifierResilience:
    """Tests for YOLO verifier error handling and fallbacks."""

    def test_yolo_verifier_fallback_on_exception(self, mock_config):
        mock_config["yolo"]["enabled"] = True
        with patch("src.yolo_verifier.YoloVerifier.__init__", return_value=None):
            verifier = YoloVerifier(mock_config)
            verifier.enabled = True
            verifier.skip_energy_threshold = 12.0
            verifier.sample_fps = 0.5
            verifier.target_classes = {0, 1, 2}
            verifier.confidence = 0.25
            verifier.device = "cpu"
            verifier.model = MagicMock(side_effect=RuntimeError("GPU out of memory simulation"))

            seg = Segment(start_time=0.0, end_time=10.0, state="DYNAMIC", source_file="f1", file_start_offset=0.0, max_energy=5.0)
            frames_buffer = {0: np.zeros((234, 416, 3), dtype=np.uint8)}

            # On exception, verify should fallback to retaining the segment state without throwing
            verified = verifier.verify("f1", [seg], frames_buffer=frames_buffer)
            assert len(verified) == 1
            assert verified[0].state == "DYNAMIC"
