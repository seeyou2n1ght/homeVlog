"""
Unit and Integration Tests for Milestone 3 (R3):
- PyAV In-Memory Audio Extraction & AudioEnergyVAD
- Mathematical Formulation of RMS, dBFS, and Adaptive 15th Percentile Noise Floor
- Multimodal Fusion Decision Matrix (DYNAMIC / DYNAMIC_AUDIO / STATIC)
- Audio Activity Early Termination Veto
- YOLO Verifier DYNAMIC_AUDIO Exemption
- Segment Building and Serialization Integration
"""

import math
from pathlib import Path
import numpy as np
import pytest

from src.detector import AudioEnergyVAD, MotionDetector
from src.segment import build_segments, Segment, segments_to_json, segments_from_json, _filter_short
from src.yolo_verifier import YoloVerifier
from src.utils import load_config


class TestAudioEnergyVAD:
    """Mathematical verification and event detection tests for AudioEnergyVAD."""

    def test_vad_initialization_parameters(self):
        vad = AudioEnergyVAD(
            sample_rate=16000,
            window_ms=50,
            noise_margin_db=12.0,
            min_dbfs=-42.0,
            min_speech_duration=0.3,
            enabled=True,
        )
        assert vad.sample_rate == 16000
        assert vad.window_ms == 50
        assert vad.window_samples == 800  # 16000 * 0.05 = 800
        assert vad.noise_margin_db == 12.0
        assert vad.min_dbfs == -42.0
        assert vad.min_speech_duration == 0.3
        assert vad.enabled is True

    def test_compute_rms_dbfs_exactness_sine_wave(self):
        """
        Verify RMS and dBFS mathematical accuracy on a known sine wave.
        A pure sine wave of peak amplitude 1.0 has theoretical RMS = 1 / sqrt(2) ~ 0.7071068.
        The logarithmic energy is 20 * log10(0.7071068) ~ -3.0103 dBFS.
        """
        vad = AudioEnergyVAD(sample_rate=16000, window_ms=50)
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        sine_wave = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        rms, dbfs = vad.compute_rms_dbfs(sine_wave)
        expected_rms = 1.0 / np.sqrt(2.0)
        expected_dbfs = 20.0 * np.log10(expected_rms + 1e-7)

        assert math.isclose(rms, expected_rms, abs_tol=1e-3)
        assert math.isclose(dbfs, expected_dbfs, abs_tol=1e-2)

    def test_compute_rms_dbfs_silence_and_scaling(self):
        vad = AudioEnergyVAD(sample_rate=16000, window_ms=50)
        silence = np.zeros(16000, dtype=np.float32)
        rms_silence, dbfs_silence = vad.compute_rms_dbfs(silence)

        assert rms_silence == 0.0
        assert dbfs_silence < -130.0  # 20*log10(1e-7) = -140 dBFS

        # Halving amplitude should drop dBFS by ~6.02 dB
        t = np.linspace(0, 1.0, 16000, endpoint=False)
        sine_full = np.sin(2 * np.pi * 1000 * t).astype(np.float32)
        sine_half = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)

        _, dbfs_full = vad.compute_rms_dbfs(sine_full)
        _, dbfs_half = vad.compute_rms_dbfs(sine_half)

        assert math.isclose(dbfs_full - dbfs_half, 6.0206, abs_tol=0.05)

    def test_adaptive_15th_percentile_noise_floor_tracking(self):
        """
        Construct audio with 85% steady room background noise and 15% loud activity.
        Verify that the adaptive 15th percentile accurately isolates the steady-state floor.
        """
        vad = AudioEnergyVAD(sample_rate=16000, window_ms=50)
        # 10 seconds total audio = 200 windows of 50ms (800 samples each)
        np.random.seed(42)
        # Background noise with RMS ~ 0.00316 (-50 dBFS)
        bg_noise = (np.random.randn(800 * 170) * 0.00316).astype(np.float32)
        # Loud speech with RMS ~ 0.0562 (-25 dBFS)
        speech = (np.random.randn(800 * 30) * 0.0562).astype(np.float32)

        full_audio = np.concatenate([bg_noise, speech])
        timestamps, dbfs_vec, noise_floor = vad.compute_dbfs_windows(full_audio)

        assert len(dbfs_vec) == 200
        # 15th percentile should reflect background noise level (~ -50 dBFS)
        assert -53.0 < noise_floor < -47.0

    def test_dual_threshold_activation_logic(self):
        """
        Test dual thresholding:
          1. Signal must exceed (N_audio + noise_margin_db)
          2. Signal must exceed min_dbfs (-42.0 dBFS)
        """
        vad = AudioEnergyVAD(
            sample_rate=16000,
            window_ms=50,
            noise_margin_db=12.0,
            min_dbfs=-42.0,
            min_speech_duration=0.2,
        )

        np.random.seed(123)
        # 6 seconds audio:
        # 0s - 2s: Quiet room floor (RMS ~ 0.001 -> ~ -60 dBFS)
        # 2s - 4s: Human speech (RMS ~ 0.07 -> ~ -23 dBFS, delta = 37dB > 12dB, -23 > -42)
        # 4s - 6s: Quiet room floor
        quiet1 = (np.random.randn(16000 * 2) * 0.001).astype(np.float32)
        t_speech = np.linspace(0, 2.0, 16000 * 2, endpoint=False)
        speech = (0.1 * np.sin(2 * np.pi * 300 * t_speech)).astype(np.float32)
        quiet2 = (np.random.randn(16000 * 2) * 0.001).astype(np.float32)

        audio = np.concatenate([quiet1, speech, quiet2])
        events, stats = vad.detect_events(audio, start_offset=0.0)

        assert len(events) == 1
        ev_start, ev_end, state = events[0]
        assert state == "ACTIVE"
        assert math.isclose(ev_start, 2.0, abs_tol=0.1)
        assert math.isclose(ev_end, 4.0, abs_tol=0.1)
        assert stats["events_count"] == 1
        assert stats["noise_floor_db"] < -50.0

    def test_silence_below_min_dbfs_suppression(self):
        """
        Verify that in an ultra-quiet room (-90 dBFS), a slight murmur at -50 dBFS
        is NOT triggered as active because it is below min_dbfs (-42 dBFS).
        """
        vad = AudioEnergyVAD(
            sample_rate=16000,
            window_ms=50,
            noise_margin_db=12.0,
            min_dbfs=-42.0,
            min_speech_duration=0.2,
        )

        # Ultra quiet: RMS ~ 1e-4 (-80 dBFS)
        np.random.seed(99)
        quiet = (np.random.randn(16000 * 3) * 1e-4).astype(np.float32)
        # Murmur: RMS ~ 0.002 (-54 dBFS)
        murmur = (np.random.randn(16000 * 2) * 0.002).astype(np.float32)

        audio = np.concatenate([quiet, murmur])
        events, stats = vad.detect_events(audio, start_offset=0.0)

        assert len(events) == 0
        assert stats["events_count"] == 0

    def test_steady_high_noise_suppression(self):
        """
        Verify that constant steady loud noise (e.g. AC unit at -30 dBFS)
        is recognized as the 15th percentile noise floor and does not trigger false positives.
        """
        vad = AudioEnergyVAD(
            sample_rate=16000,
            window_ms=50,
            noise_margin_db=12.0,
            min_dbfs=-42.0,
            min_speech_duration=0.2,
        )

        # Constant steady noise at RMS ~ 0.0316 (-30 dBFS)
        np.random.seed(55)
        steady_noise = (np.random.randn(16000 * 5) * 0.0316).astype(np.float32)

        events, stats = vad.detect_events(steady_noise, start_offset=0.0)
        assert len(events) == 0
        assert stats["noise_floor_db"] > -35.0

    def test_short_click_transient_filtering(self):
        """
        Verify that a single high-energy click of 30ms (< min_speech_duration 300ms)
        is filtered out and does not produce a false speech segment.
        """
        vad = AudioEnergyVAD(
            sample_rate=16000,
            window_ms=50,
            noise_margin_db=12.0,
            min_dbfs=-42.0,
            min_speech_duration=0.3,
        )

        np.random.seed(11)
        quiet = (np.random.randn(16000 * 3) * 0.001).astype(np.float32)
        # 30ms spike (480 samples)
        quiet[16000 : 16480] = 0.5

        events, _ = vad.detect_events(quiet, start_offset=0.0)
        assert len(events) == 0

    def test_is_active_at_helper(self):
        vad = AudioEnergyVAD()
        events = [(2.0, 5.0, "ACTIVE"), (10.0, 14.5, "ACTIVE")]

        assert vad.is_active_at(3.5, events) is True
        assert vad.is_active_at(1.95, events) is True  # within 0.1s tolerance
        assert vad.is_active_at(5.05, events) is True  # within 0.1s tolerance
        assert vad.is_active_at(7.0, events) is False
        assert vad.is_active_at(12.0, events) is True
        assert vad.is_active_at(20.0, events) is False


class TestMultimodalFusionMatrix:
    """Tests for Multimodal Fusion Decision Matrix in detector and segment builder."""

    def test_multimodal_fusion_truth_table(self):
        """
        Verify all 4 branches of the Multimodal Fusion Decision Matrix:
          1. Visual DYNAMIC + Audio ACTIVE  -> DYNAMIC
          2. Visual DYNAMIC + Audio SILENT  -> DYNAMIC
          3. Visual STATIC  + Audio ACTIVE  -> DYNAMIC_AUDIO (1.0x speed, original audio kept)
          4. Visual STATIC  + Audio SILENT  -> STATIC (speed ramping fast-forward)
        """
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "analysis_early_term_enabled": False,
                "ema_background_enabled": False,
                "audio_vad_enabled": True,
                "vad_noise_margin_db": 12.0,
                "vad_min_dbfs": -42.0,
                "vad_min_speech_duration": 0.2,
                "min_motion_frames": 1,
                "min_static_frames": 1,
                "noise_suppress_frames": 0,
                "median_filter_window": 1,
            }
        }
        detector = MotionDetector(config)

        # 40 frames total at 5 fps = 8.0 seconds
        # Visual:
        #   Frames 0-9  (0.0s-1.8s): Dynamic (moving square)
        #   Frames 10-19 (2.0s-3.8s): Dynamic (moving square)
        #   Frames 20-29 (4.0s-5.8s): Static (solid gray)
        #   Frames 30-39 (6.0s-7.8s): Static (solid gray)
        frames = []
        for i in range(40):
            frame = np.full((180, 320), 50, dtype=np.uint8)
            if i < 20:
                # Visual motion
                x = (i * 10) % 200
                frame[40:100, x : x + 60] = 200
            frames.append(frame)

        # Audio at 16kHz:
        #   0.0s - 2.0s: Active Speech (Visual Dynamic + Audio Active -> DYNAMIC)
        #   2.0s - 4.0s: Silent (Visual Dynamic + Audio Silent -> DYNAMIC)
        #   4.0s - 6.0s: Active Speech (Visual Static + Audio Active -> DYNAMIC_AUDIO)
        #   6.0s - 8.0s: Silent (Visual Static + Audio Silent -> STATIC)
        np.random.seed(7)
        audio_parts = []
        # 0s-2s: speech
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        audio_parts.append((0.1 * np.sin(2 * np.pi * 400 * t)).astype(np.float32))
        # 2s-4s: quiet
        audio_parts.append((np.random.randn(32000) * 0.001).astype(np.float32))
        # 4s-6s: speech
        audio_parts.append((0.1 * np.sin(2 * np.pi * 400 * t)).astype(np.float32))
        # 6s-8s: quiet
        audio_parts.append((np.random.randn(32000) * 0.001).astype(np.float32))

        audio_data = np.concatenate(audio_parts)

        results, meta = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=8.0,
            fps=5,
            audio_data=audio_data,
            audio_sample_rate=16000,
        )

        assert len(results) > 0
        assert meta["has_audio"] == 1
        assert len(meta["audio_events"]) >= 2

        # Verify states in each 2-second quarter:
        # Quarter 1 (0-2s): Visual Dynamic + Audio Active -> DYNAMIC
        q1_states = [r["state"] for r in results if 0.2 <= r["time"] <= 1.8]
        assert all(s == "DYNAMIC" for s in q1_states)

        # Quarter 2 (2-4s): Visual Dynamic + Audio Silent -> DYNAMIC
        q2_states = [r["state"] for r in results if 2.2 <= r["time"] <= 3.8]
        assert all(s == "DYNAMIC" for s in q2_states)

        # Quarter 3 (4-6s): Visual Static + Audio Active -> DYNAMIC_AUDIO
        q3_states = [r["state"] for r in results if 4.2 <= r["time"] <= 5.8]
        assert all(s == "DYNAMIC_AUDIO" for s in q3_states)

        # Quarter 4 (6-8s): Visual Static + Audio Silent -> STATIC
        q4_states = [r["state"] for r in results if 6.2 <= r["time"] <= 7.8]
        assert all(s == "STATIC" for s in q4_states)


class TestEarlyTerminationAudioVeto:
    """Verify that ongoing audio activity strictly vetoes early termination."""

    def test_audio_activity_vetoes_early_termination(self):
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "analysis_early_term_enabled": True,
                "analysis_early_term_window": 15,
                "analysis_early_term_threshold": 2.0,
                "ema_background_enabled": False,
                "early_term_cooldown_guard": False,  # Cooldown disabled to test audio veto directly
                "audio_vad_enabled": True,
                "vad_noise_margin_db": 12.0,
                "vad_min_dbfs": -42.0,
                "vad_min_speech_duration": 0.2,
                "min_motion_frames": 1,
                "min_static_frames": 1,
                "noise_suppress_frames": 0,
                "median_filter_window": 1,
            }
        }
        detector = MotionDetector(config)

        # 60 frames total at 5 fps = 12 seconds
        # Frame 0-4: brief motion, Frame 5-59 (55 frames): completely static visually
        frames = []
        for i in range(60):
            frame = np.full((180, 320), 50, dtype=np.uint8)
            if i < 5:
                frame[40:100, 40:100] = 200
            frames.append(frame)

        # Test Case 1: Audio is completely silent -> Early termination triggers around frame 20
        silent_audio = np.zeros(16000 * 12, dtype=np.float32)
        results_silent, meta_silent = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=12.0,
            fps=5,
            audio_data=silent_audio,
        )
        assert meta_silent["early_terminated"] is True
        assert len(results_silent) < 40

        # Test Case 2: Audio has active conversation spanning 0.0s - 10.0s -> Early termination is VETOED
        np.random.seed(42)
        t_active = np.linspace(0, 10.0, 16000 * 10, endpoint=False)
        speech_audio = (0.1 * np.sin(2 * np.pi * 300 * t_active)).astype(np.float32)
        quiet_tail = (np.random.randn(16000 * 2) * 0.001).astype(np.float32)
        active_audio = np.concatenate([speech_audio, quiet_tail])

        results_active, meta_active = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=12.0,
            fps=5,
            audio_data=active_audio,
        )

        # Decoded frames should NOT be cut short early during active audio
        # Frames between 1.0s and 10.0s must be present and labeled DYNAMIC_AUDIO
        assert len(results_active) >= 50
        active_audio_frames = [r for r in results_active if 2.0 <= r["time"] <= 9.5]
        assert len(active_audio_frames) > 0
        assert all(r["state"] == "DYNAMIC_AUDIO" for r in active_audio_frames)


class TestSegmentAndYoloIntegration:
    """Test segment construction, JSON serialization, and YOLO verifier exemption."""

    def test_segment_is_dynamic_property(self):
        seg_dyn = Segment(0.0, 5.0, "DYNAMIC", "test.mp4", 0.0)
        seg_audio = Segment(5.0, 10.0, "DYNAMIC_AUDIO", "test.mp4", 0.0)
        seg_static = Segment(10.0, 40.0, "STATIC", "test.mp4", 0.0)

        assert seg_dyn.is_dynamic is True
        assert seg_audio.is_dynamic is True
        assert seg_static.is_dynamic is False

    def test_build_segments_with_dynamic_audio(self):
        frame_labels = [
            {"time": 0.0, "is_motion": True, "state": "DYNAMIC", "energy": 5.0},
            {"time": 1.0, "is_motion": True, "state": "DYNAMIC", "energy": 6.0},
            {"time": 2.0, "is_motion": True, "state": "DYNAMIC_AUDIO", "energy": 0.5},
            {"time": 3.0, "is_motion": True, "state": "DYNAMIC_AUDIO", "energy": 0.4},
            {"time": 4.0, "is_motion": False, "state": "STATIC", "energy": 0.1},
            {"time": 5.0, "is_motion": False, "state": "STATIC", "energy": 0.1},
        ]

        segments = build_segments(
            frame_labels=frame_labels,
            source_file="test.mp4",
            min_motion_dur=0.5,
            min_static_dur=0.5,
            file_offset=0.0,
            gap_tolerance=0.5,
        )

        assert len(segments) == 3
        assert segments[0].state == "DYNAMIC"
        assert segments[0].start_time == 0.0
        assert segments[0].end_time == 1.0
        assert segments[0].max_energy == 6.0

        assert segments[1].state == "DYNAMIC_AUDIO"
        assert segments[1].start_time == 2.0
        assert segments[1].end_time == 3.0

        assert segments[2].state == "STATIC"
        assert segments[2].start_time == 4.0
        assert segments[2].end_time == 5.0

    def test_segment_json_roundtrip_preserves_state(self):
        original = [
            Segment(0.0, 5.0, "DYNAMIC", "a.mp4", 0.0, 8.5),
            Segment(5.0, 15.0, "DYNAMIC_AUDIO", "a.mp4", 0.0, 1.2),
            Segment(15.0, 60.0, "STATIC", "a.mp4", 0.0, 0.2),
        ]

        js = segments_to_json(original)
        reconstructed = segments_from_json(js)

        assert len(reconstructed) == 3
        for orig, recon in zip(original, reconstructed):
            assert orig.start_time == recon.start_time
            assert orig.end_time == recon.end_time
            assert orig.state == recon.state
            assert orig.source_file == recon.source_file
            assert orig.file_start_offset == recon.file_start_offset
            assert math.isclose(orig.max_energy, recon.max_energy, abs_tol=1e-5)

    def test_yolo_verifier_exempts_dynamic_audio(self):
        """
        Verify that YoloVerifier skips object verification and preserves DYNAMIC_AUDIO
        segments without demoting them to STATIC even when no objects are detected.
        """
        config = {
            "yolo": {
                "enabled": True,
                "model_path": "yolo11n.pt",
                "device": "cpu",
                "target_classes": [0],
                "confidence": 0.5,
                "sample_fps": 1.0,
                "skip_energy_threshold": 20.0,
            }
        }

        verifier = YoloVerifier(config)
        # Force verifier to be enabled for test
        verifier.enabled = True

        segments = [
            # DYNAMIC_AUDIO segment (should be strictly preserved without demotion)
            Segment(start_time=0.0, end_time=10.0, state="DYNAMIC_AUDIO", source_file="clip.mp4", file_start_offset=0.0, max_energy=1.0),
            # Pure visual DYNAMIC segment with low energy and empty frames (should be demoted to STATIC by YOLO)
            Segment(start_time=10.0, end_time=20.0, state="DYNAMIC", source_file="clip.mp4", file_start_offset=0.0, max_energy=5.0),
            # Already STATIC segment (untouched)
            Segment(start_time=20.0, end_time=50.0, state="STATIC", source_file="clip.mp4", file_start_offset=0.0, max_energy=0.1),
        ]

        # Empty black frames buffer (no objects in frames 10..20)
        frames_buffer = {
            f: np.zeros((360, 640, 3), dtype=np.uint8) for f in range(250)
        }

        verified = verifier.verify(
            filepath="clip.mp4",
            segments=segments,
            gpu="cuda",
            device="cpu",
            frames_buffer=frames_buffer,
            analysis_fps=5.0,
        )

        assert len(verified) == 3
        # DYNAMIC_AUDIO must NOT be demoted to STATIC
        assert verified[0].state == "DYNAMIC_AUDIO" or verified[0].is_dynamic is True
        assert verified[0].state != "STATIC"

        # DYNAMIC segment with no objects is demoted to STATIC
        assert verified[1].state == "STATIC"

        # STATIC remains STATIC
        assert verified[2].state == "STATIC"

    def test_filter_short_preserves_valid_dynamic_audio(self):
        """
        Verify that _filter_short properly respects min_motion for DYNAMIC_AUDIO
        and absorbs short static intervals.
        """
        segments = [
            Segment(0.0, 5.0, "DYNAMIC", "test.mp4", 0.0),
            Segment(5.0, 6.0, "STATIC", "test.mp4", 0.0),        # 1.0s static (< min_static 10.0s) -> absorbed
            Segment(6.0, 10.0, "DYNAMIC_AUDIO", "test.mp4", 0.0), # 4.0s audio (> min_motion 2.0s) -> retained
            Segment(10.0, 10.5, "DYNAMIC_AUDIO", "test.mp4", 0.0),# 0.5s audio (< min_motion 2.0s) -> absorbed
            Segment(10.5, 40.0, "STATIC", "test.mp4", 0.0),
        ]

        filtered = _filter_short(segments, min_motion=2.0, min_static=10.0, gap_tolerance=0.5)
        # Should have dynamic part, dynamic_audio part, and static part
        assert len(filtered) >= 2
        states = [s.state for s in filtered]
        assert "STATIC" in states
        assert any(s in ("DYNAMIC", "DYNAMIC_AUDIO") for s in states)

    def test_analyze_frames_fallback_without_audio(self):
        """
        Verify that analyze_frames works seamlessly with audio_data=None or disabled VAD.
        """
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "analysis_early_term_enabled": False,
                "audio_vad_enabled": False,
            }
        }
        detector = MotionDetector(config)

        frames = [np.full((180, 320), 50, dtype=np.uint8) for _ in range(10)]
        results, meta = detector.analyze_frames(frames=frames, start_offset=0.0, file_duration=2.0, fps=5)

        assert len(results) == 10
        assert meta["has_audio"] == 0
        assert all(r["state"] == "STATIC" for r in results)
        assert all(r["is_audio_active"] is False for r in results)

    def test_audio_vad_edge_cases(self):
        """Test empty arrays, 1-sample arrays, and extreme values in AudioEnergyVAD."""
        vad = AudioEnergyVAD()
        # Empty array
        ev, st = vad.detect_events(np.array([], dtype=np.float32))
        assert ev == []
        assert st["total_windows"] == 0

        # Sub-window short array (100 samples < 800)
        short_arr = (np.random.randn(100) * 0.01).astype(np.float32)
        ev_short, st_short = vad.detect_events(short_arr)
        assert isinstance(ev_short, list)

        # Full scale maximum audio (0 dBFS square wave)
        square = np.full(16000 * 2, 1.0, dtype=np.float32)
        rms_sq, dbfs_sq = vad.compute_rms_dbfs(square)
        assert math.isclose(rms_sq, 1.0, abs_tol=1e-3)
        assert math.isclose(dbfs_sq, 0.0, abs_tol=0.01)


class TestSettingsSchema:
    """Verify settings.yaml schema extension and parser compatibility."""

    def test_settings_schema_audio_vad_keys(self):
        cfg = load_config()
        assert "detection" in cfg
        det = cfg["detection"]
        assert "audio_vad_enabled" in det
        assert det["audio_vad_enabled"] is True
        assert "vad_noise_margin_db" in det
        assert float(det["vad_noise_margin_db"]) == 12.0
        assert "vad_min_dbfs" in det
        assert float(det["vad_min_dbfs"]) == -42.0
        assert "vad_window_ms" in det
        assert int(det["vad_window_ms"]) == 50

        assert "audio_vad" in cfg
        audio_cfg = cfg["audio_vad"]
        assert audio_cfg["enabled"] is True
        assert float(audio_cfg["energy_threshold_db"]) == 12.0
        assert float(audio_cfg["min_dbfs"]) == -42.0
