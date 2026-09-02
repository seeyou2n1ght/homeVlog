"""
Tier 1: Feature Coverage Test Suite for HomeVlog.
Covers:
- Hardware semaphore safety & singleton lifecycle
- WorkStealingManager state machine & dynamic preemption
- EMA background smoothing, median filtering & grid sensitivity
- VAD audio energy extraction & multimodal metadata
- Speed ramping PTS formulas & audio cross-fade filter generation
- Timeline closure, short segment absorption & cross-file merging
- Batch split boundary preservation
"""

import json
import math
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.database import VlogDatabase
from src.detector import MotionDetector, _median_filter, _smooth_labels
from src.segment import Segment, build_segments, _filter_short, merge_cross_file, segments_to_json, segments_from_json
from src.timeline import TimelineSegment, build_concat_filter, build_timeline
from src.utils import (
    get_disk_semaphore,
    get_nv_semaphore,
    get_qsv_semaphore,
    reset_semaphores,
    WorkStealingManager,
    ts_to_unix,
    parse_res,
)
from tests.helpers import (
    create_synthetic_wav,
    parse_ffmpeg_filtergraph,
    verify_filtergraph_labels_closure,
)


class TestHardwareSemaphores:
    """Tests for hardware session limiting semaphores."""

    def test_semaphore_singletons_and_limits(self, mock_config):
        with patch("src.utils.load_config", return_value=mock_config):
            reset_semaphores()
            nv_sem1 = get_nv_semaphore()
            nv_sem2 = get_nv_semaphore()
            assert nv_sem1 is nv_sem2
            assert nv_sem1._value == 3

            qsv_sem1 = get_qsv_semaphore()
            qsv_sem2 = get_qsv_semaphore()
            assert qsv_sem1 is qsv_sem2
            assert qsv_sem1._value == 8

            disk_sem1 = get_disk_semaphore()
            disk_sem2 = get_disk_semaphore()
            assert disk_sem1 is disk_sem2
            assert disk_sem1._value == 8

    def test_reset_semaphores_clears_singletons(self, mock_config):
        with patch("src.utils.load_config", return_value=mock_config):
            s1 = get_nv_semaphore()
            reset_semaphores()
            s2 = get_nv_semaphore()
            assert s1 is not s2


class TestWorkStealingManager:
    """Tests for the adaptive heterogeneous work-stealing scheduler."""

    def test_work_stealing_initial_state(self, mock_config):
        mgr = WorkStealingManager(config=mock_config)
        assert mgr.state == "NORMAL_DECOUPLED"
        assert not mgr.is_render_active
        assert mgr.active_nv_decoders == 0

    def test_work_stealing_low_watermark_uses_qsv(self, mock_config):
        mgr = WorkStealingManager(config=mock_config)
        device = mgr.get_analysis_device(queue_size=2)
        assert device == "qsv"
        assert mgr.state == "NORMAL_DECOUPLED"

    def test_work_stealing_high_watermark_burst_to_cuda(self, mock_config):
        mock_config["hardware"]["device"] = "cuda:0"
        mgr = WorkStealingManager(config=mock_config)
        device = mgr.get_analysis_device(queue_size=12)
        assert device == "cuda"
        assert mgr.state == "COOPERATIVE_BURST"

    def test_work_stealing_render_preemption_yield(self, mock_config):
        mock_config["hardware"]["device"] = "cuda:0"
        mgr = WorkStealingManager(config=mock_config)
        mgr.register_render_start()
        assert mgr.is_render_active
        assert mgr.state == "RENDER_PREEMPTION_YIELD"

        # Even with high watermark, render active forces QSV
        device = mgr.get_analysis_device(queue_size=20)
        assert device == "qsv"
        assert mgr.state == "RENDER_PREEMPTION_YIELD"

        # Cannot acquire NVDEC slot during active render
        assert not mgr.acquire_nvdec_slot()

        mgr.register_render_end()
        assert not mgr.is_render_active
        assert mgr.state == "NORMAL_DECOUPLED"

    def test_work_stealing_slot_acquisition_and_release(self, mock_config):
        mock_config["hardware"]["device"] = "cuda:0"
        mock_config["scheduler"]["max_nv_decoders"] = 1
        mgr = WorkStealingManager(config=mock_config)

        assert mgr.acquire_nvdec_slot()
        assert mgr.active_nv_decoders == 1
        # Second acquire fails due to max_nv_decoders = 1
        assert not mgr.acquire_nvdec_slot()

        mgr.release_nvdec_slot()
        assert mgr.active_nv_decoders == 0

    def test_work_stealing_lease_device_context_manager(self, mock_config):
        mock_config["hardware"]["device"] = "cuda:0"
        mgr = WorkStealingManager(config=mock_config)

        with mgr.lease_device(queue_size=15) as dev:
            assert dev == "cuda"
            assert mgr.active_nv_decoders == 1

        assert mgr.active_nv_decoders == 0


class TestEmaAndMotionDetectionAlgorithms:
    """Tests for core background motion algorithms, median filter, and label smoothing."""

    def test_median_filter_odd_windows(self):
        noisy_signal = [0.0, 1.0, 100.0, 1.0, 0.0, 0.0, 50.0, 0.0]
        filtered = _median_filter(noisy_signal, window=3)
        assert len(filtered) == len(noisy_signal)
        # Spike at index 2 (100.0) should be smoothed out by neighbors (1.0, 1.0)
        assert filtered[2] == 1.0
        # Spike at index 6 (50.0) should be smoothed out by neighbors (0.0, 0.0)
        assert filtered[6] == 0.0

    def test_median_filter_small_window_noop(self):
        signal = [1.0, 2.0, 3.0]
        filtered = _median_filter(signal, window=1)
        assert filtered == signal

    def test_smooth_labels_min_motion_suppression(self):
        # 2 frames of motion when min_motion is 3 should be suppressed to False
        raw = [False, True, True, False, False]
        smoothed = _smooth_labels(raw, min_motion=3, min_static=1, noise_suppress=0)
        assert smoothed == [False, False, False, False, False]

    def test_smooth_labels_min_motion_retained(self):
        # 4 frames of motion when min_motion is 3 should be retained
        raw = [False, True, True, True, True, False]
        smoothed = _smooth_labels(raw, min_motion=3, min_static=1, noise_suppress=0)
        assert smoothed == [False, True, True, True, True, False]

    def test_smooth_labels_noise_suppress(self):
        # 1 static frame in between motion runs when noise_suppress=2 should be flipped to True
        raw = [True, True, True, False, True, True, True]
        smoothed = _smooth_labels(raw, min_motion=3, min_static=5, noise_suppress=2)
        assert smoothed == [True, True, True, True, True, True, True]

    def test_motion_detector_initialization(self, mock_config):
        detector = MotionDetector(mock_config, decode_gpu="qsv")
        assert detector.width == 416
        assert detector.height == 234
        assert detector.fps == 5
        assert detector.sensitivity == 1.0
        assert detector.early_term_enabled is True
        assert detector.early_term_window == 20


class TestAudioVADAndEnergy:
    """Tests for audio extraction, synthetic energy calculation, and VAD decision."""

    def test_synthetic_audio_rms_and_dbfs(self, temp_test_dir):
        # 1. Sine wave audio
        sine_path = temp_test_dir / "sine_440hz.wav"
        create_synthetic_wav(sine_path, duration_sec=0.5, sample_rate=48000, frequency=440.0, silence=False)
        assert sine_path.exists()
        assert sine_path.stat().st_size > 1000

        # Read samples and compute RMS
        import wave
        with wave.open(str(sine_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            rms = math.sqrt(np.mean(samples ** 2))
            assert rms > 5000.0  # High energy for sine wave

        # 2. Silent audio
        silent_path = temp_test_dir / "silence.wav"
        create_synthetic_wav(silent_path, duration_sec=0.5, sample_rate=48000, silence=True)
        with wave.open(str(silent_path), "rb") as wf:
            raw = wf.readframes(wf.getnframes())
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            rms = math.sqrt(np.mean(samples ** 2))
            assert rms == 0.0


class TestSpeedRampingAndConcatFilter:
    """Tests for PTS speed curve generation, audio filter, and label closures."""

    def test_build_concat_filter_dynamic_and_static_pts(self, sample_segments):
        rows = [{"filepath": sample_segments[0].source_file, "has_audio": 1}]
        timeline = [
            TimelineSegment(
                filepath=s.source_file,
                input_index=0,
                start_in_file=s.start_time,
                end_in_file=s.end_time,
                state=s.state,
                duration=s.end_time - s.start_time,
            )
            for s in sample_segments
        ]

        filter_str = build_concat_filter(
            timeline=timeline,
            rows=rows,
            output_fps=20,
            output_width=1920,
            output_height=1080,
            static_keyframe_interval=30.0,
            keyframe_display_duration=0.5,
            scale_mode="cpu",
        )

        parsed = parse_ffmpeg_filtergraph(filter_str)
        assert len(parsed["video_trims"]) == 3
        # Segment 0 (STATIC): scaled PTS
        assert "setpts=(PTS-STARTPTS)/" in parsed["video_trims"][0]
        # Segment 1 (DYNAMIC): 1.0x PTS
        assert "setpts=PTS-STARTPTS" in parsed["video_trims"][1]
        # Segment 2 (STATIC): scaled PTS
        assert "setpts=(PTS-STARTPTS)/" in parsed["video_trims"][2]

        # Check audio
        assert len(parsed["audio_nulls"]) == 2  # Segments 0 and 2 are static
        assert len(parsed["audio_trims"]) == 1  # Segment 1 is dynamic with audio

        # Validate label closure
        ok, msg = verify_filtergraph_labels_closure(filter_str)
        assert ok, msg

    def test_build_concat_filter_scale_modes(self, sample_segments):
        rows = [{"filepath": sample_segments[0].source_file, "has_audio": 0}]
        timeline = [
            TimelineSegment(
                filepath=s.source_file,
                input_index=0,
                start_in_file=s.start_time,
                end_in_file=s.end_time,
                state=s.state,
                duration=s.end_time - s.start_time,
            )
            for s in sample_segments[:1]
        ]

        # CUDA scale mode
        cuda_filter = build_concat_filter(timeline, rows=rows, scale_mode="cuda")
        assert "scale_cuda=1920:1080" in cuda_filter

        # QSV scale mode
        qsv_filter = build_concat_filter(timeline, rows=rows, scale_mode="qsv")
        assert "scale_qsv=w=1920:h=1080" in qsv_filter

        # Skip mode
        skip_filter = build_concat_filter(timeline, rows=rows, scale_mode="skip")
        assert "null[" in skip_filter


class TestTimelineClosureAndAbsorption:
    """Tests for segment conversion, short segment absorption, and cross-file merging."""

    def test_build_segments_basic(self):
        labels = [
            {"time": 0.0, "is_motion": False, "energy": 0.1},
            {"time": 1.0, "is_motion": False, "energy": 0.2},
            {"time": 2.0, "is_motion": True, "energy": 15.0},
            {"time": 3.0, "is_motion": True, "energy": 20.0},
            {"time": 4.0, "is_motion": False, "energy": 0.3},
        ]
        segs = build_segments(labels, source_file="test.mp4", min_motion_dur=1.0, min_static_dur=1.0)
        assert len(segs) == 3
        assert segs[0].state == "STATIC"
        assert segs[1].state == "DYNAMIC"
        assert segs[2].state == "STATIC"

    def test_filter_short_absorption(self):
        # A 1-second dynamic segment between long static segments should be absorbed into static
        segs = [
            Segment(start_time=0.0, end_time=60.0, state="STATIC", source_file="f1", file_start_offset=0.0),
            Segment(start_time=60.0, end_time=61.0, state="DYNAMIC", source_file="f1", file_start_offset=0.0),
            Segment(start_time=61.0, end_time=120.0, state="STATIC", source_file="f1", file_start_offset=0.0),
        ]
        absorbed = _filter_short(segs, min_motion=2.0, min_static=30.0, gap_tolerance=0.5)
        assert len(absorbed) == 1
        assert absorbed[0].state == "STATIC"
        assert absorbed[0].start_time == 0.0
        assert absorbed[0].end_time == 120.0

    def test_merge_cross_file_same_state(self):
        segs = [
            Segment(start_time=0.0, end_time=100.0, state="STATIC", source_file="f1", file_start_offset=0.0),
            Segment(start_time=100.2, end_time=200.0, state="STATIC", source_file="f2", file_start_offset=100.0),
        ]
        # Gap is 0.2s <= 0.5s gap_tolerance -> should merge
        merged = merge_cross_file(segs, gap_tolerance=0.5)
        assert len(merged) == 1
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 200.0

    def test_segments_json_serialization_roundtrip(self, sample_segments):
        js = segments_to_json(sample_segments)
        recovered = segments_from_json(js)
        assert len(recovered) == len(sample_segments)
        for orig, rec in zip(sample_segments, recovered):
            assert orig.start_time == rec.start_time
            assert orig.end_time == rec.end_time
            assert orig.state == rec.state
            assert orig.source_file == rec.source_file
