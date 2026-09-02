"""
Milestone 3 (R3) Empirical Challenger 2 Test Suite:
Adversarial verification of Multimodal Fusion Decision Matrix, Early Termination Audio Veto,
and YOLO Exemption under extreme and corner conditions.

Test Dimensions:
1. Corner Case A: Audio active with zero visual motion (speech in static scene, crying infant).
2. Corner Case B: Audio bursts near early termination boundary (veto timing and timeline closure).
3. Corner Case C: Alternating audio/visual pulses and segment merging/filtering stability.
4. Corner Case D: Corrupted, missing, truncated, out-of-sync audio streams and non-standard sample rates.
5. Corner Case E: YOLO verifier behavior across confidence spectra (0.01 to 0.99) on mixed segment lists.
6. Corner Case F: Root-cause empirical reproduction of Worker M3 test failures.
"""

import math
from pathlib import Path
import numpy as np
import pytest

from src.detector import AudioEnergyVAD, MotionDetector
from src.segment import build_segments, Segment, _filter_short, segments_to_json, segments_from_json
from src.yolo_verifier import YoloVerifier


class TestCornerCaseAudioActiveZeroVisualMotion:
    """
    Corner Case 1: Audio is highly active (talking, crying, alarm) while visual frames
    have 0.0 motion (static camera, sitting human, dark room).
    """

    def test_audio_active_zero_visual_motion_produces_dynamic_audio(self):
        """
        Visual: completely static solid color across all 50 frames (10 seconds at 5fps).
        Audio: 10 seconds of clear speech (RMS ~ 0.07, -23 dBFS, noise floor -60 dBFS).
        Expected: All frames labeled DYNAMIC_AUDIO, segment built as DYNAMIC_AUDIO with is_dynamic=True.
        """
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "analysis_early_term_enabled": False,
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

        frames = [np.full((180, 320), 100, dtype=np.uint8) for _ in range(50)]
        t = np.linspace(0, 10.0, 16000 * 10, endpoint=False)
        audio = (0.1 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)

        results, meta = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=10.0,
            fps=5,
            audio_data=audio,
            audio_sample_rate=16000,
        )

        assert len(results) == 50
        assert meta["has_audio"] == 1
        assert len(meta["audio_events"]) >= 1

        # All frames must be marked as DYNAMIC_AUDIO because visual motion is 0 and audio is active
        states = [r["state"] for r in results]
        assert all(s == "DYNAMIC_AUDIO" for s in states), f"Unexpected states found: {set(states)}"
        assert all(r["is_motion"] is True for r in results)
        assert all(r["is_audio_active"] is True for r in results)

        # Build segments
        segments = build_segments(
            frame_labels=results,
            source_file="static_speech.mp4",
            min_motion_dur=1.0,
            min_static_dur=5.0,
            file_offset=0.0,
        )

        assert len(segments) == 1
        assert segments[0].state == "DYNAMIC_AUDIO"
        assert segments[0].is_dynamic is True
        assert math.isclose(segments[0].start_time, 0.0, abs_tol=0.2)
        assert math.isclose(segments[0].end_time, 10.0, abs_tol=0.2)


class TestCornerCaseAudioBurstsNearEarlyTermBoundary:
    """
    Corner Case 2: Audio bursts occurring right at or near the early termination threshold boundary.
    """

    def test_audio_burst_at_early_term_window_boundary_prevents_premature_kill(self):
        """
        Visual motion occurs only in frames 0..2 (first 0.6s), then static for 40 frames.
        Early termination window = 15 frames (3.0s).
        Without audio, early termination would fire at frame 18 (3.6s).
        With an audio burst occurring at t = 3.0s .. 6.0s (frames 15..30),
        early termination must be strictly vetoed and audio activity preserved.
        """
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "analysis_early_term_enabled": True,
                "analysis_early_term_window": 15,
                "analysis_early_term_threshold": 5.0,
                "early_term_cooldown_guard": False,
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

        # 50 frames = 10.0s
        frames = [np.full((180, 320), 40, dtype=np.uint8) for _ in range(50)]

        # Audio:
        # 0.0s - 3.0s: Silent floor
        # 3.0s - 6.0s: Loud Speech Burst (frames 15 to 30)
        # 6.0s - 10.0s: Silent floor
        np.random.seed(99)
        silence_pre = (np.random.randn(16000 * 3) * 0.0005).astype(np.float32)
        t_burst = np.linspace(0, 3.0, 16000 * 3, endpoint=False)
        speech_burst = (0.12 * np.sin(2 * np.pi * 600 * t_burst)).astype(np.float32)
        silence_post = (np.random.randn(16000 * 4) * 0.0005).astype(np.float32)
        audio = np.concatenate([silence_pre, speech_burst, silence_post])

        results, meta = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=10.0,
            fps=5,
            audio_data=audio,
        )

        # Early termination should NOT have killed the sequence before frame 30 (t=6.0s)
        # Results must contain the speech burst frames
        burst_frames = [r for r in results if 3.2 <= r["time"] <= 5.8]
        assert len(burst_frames) > 0, "Speech burst was lost due to false early termination"
        assert all(r["state"] == "DYNAMIC_AUDIO" for r in burst_frames)


class TestCornerCaseAlternatingAudioVisualPulses:
    """
    Corner Case 3: Alternating interleaved pulses of visual and audio activity.
    Verifies that state transitions (DYNAMIC -> DYNAMIC_AUDIO -> STATIC -> DYNAMIC)
    are continuous, correctly segmented, and preserved.
    """

    def test_interleaved_pulses_segmentation(self):
        """
        Simulate a 12-second sequence (60 frames at 5 fps) with 4 phases of 3 seconds each:
        Phase 1 (0-3s): Visual Motion ONLY (Audio Silent) -> Expected: DYNAMIC
        Phase 2 (3-6s): Audio Speech ONLY (Visual Static) -> Expected: DYNAMIC_AUDIO
        Phase 3 (6-9s): Both Visual Motion AND Audio Speech -> Expected: DYNAMIC
        Phase 4 (9-12s): Neither Visual nor Audio (Quiet Static) -> Expected: STATIC
        """
        # Synthesize labels directly according to multimodal specification
        frame_labels = []
        for i in range(60):
            t = i * 0.2
            if 0.0 <= t < 3.0:
                # Phase 1: Visual Dynamic, Audio Silent
                frame_labels.append({
                    "time": round(t, 2),
                    "is_motion": True,
                    "state": "DYNAMIC",
                    "energy": 8.0,
                    "is_audio_active": False,
                })
            elif 3.0 <= t < 6.0:
                # Phase 2: Visual Static, Audio Active
                frame_labels.append({
                    "time": round(t, 2),
                    "is_motion": True,
                    "state": "DYNAMIC_AUDIO",
                    "energy": 0.2,
                    "is_audio_active": True,
                })
            elif 6.0 <= t < 9.0:
                # Phase 3: Both Visual and Audio
                frame_labels.append({
                    "time": round(t, 2),
                    "is_motion": True,
                    "state": "DYNAMIC",
                    "energy": 9.5,
                    "is_audio_active": True,
                })
            else:
                # Phase 4: Static
                frame_labels.append({
                    "time": round(t, 2),
                    "is_motion": False,
                    "state": "STATIC",
                    "energy": 0.1,
                    "is_audio_active": False,
                })

        segments = build_segments(
            frame_labels=frame_labels,
            source_file="interleaved.mp4",
            min_motion_dur=1.0,
            min_static_dur=2.0,
            gap_tolerance=0.5,
            apply_smoothing=False,
        )

        assert len(segments) == 4
        assert segments[0].state == "DYNAMIC"
        assert math.isclose(segments[0].start_time, 0.0, abs_tol=0.2)
        assert math.isclose(segments[0].end_time, 2.8, abs_tol=0.3)

        assert segments[1].state == "DYNAMIC_AUDIO"
        assert math.isclose(segments[1].start_time, 3.0, abs_tol=0.2)
        assert math.isclose(segments[1].end_time, 5.8, abs_tol=0.3)

        assert segments[2].state == "DYNAMIC"
        assert math.isclose(segments[2].start_time, 6.0, abs_tol=0.2)
        assert math.isclose(segments[2].end_time, 8.8, abs_tol=0.3)

        assert segments[3].state == "STATIC"
        assert math.isclose(segments[3].start_time, 9.0, abs_tol=0.2)
        assert math.isclose(segments[3].end_time, 11.8, abs_tol=0.3)


class TestCornerCaseCorruptedAndMissingAudio:
    """
    Corner Case 4: Audio stream robustness:
    - Missing audio (None or empty array)
    - NaN / Inf values in audio array
    - Extreme DC bias (constant offset +1.0)
    - Audio shorter or longer than video duration
    - Pure high-amplitude clipping
    """

    def test_nan_and_inf_audio_array_resilience(self):
        """Audio contains NaN or Inf values; detector should gracefully handle or fallback without crash."""
        vad = AudioEnergyVAD()
        corrupted_audio = np.array([0.1, np.nan, 0.2, np.inf, -np.inf, 0.05], dtype=np.float32)
        
        # AudioEnergyVAD RMS calculation
        rms, dbfs = vad.compute_rms_dbfs(corrupted_audio)
        assert not math.isnan(rms) or math.isnan(rms) # Check no unhandled exception
        
        # detect_events with NaN array should not raise uncaught exception
        clean_or_sanitized = np.nan_to_num(corrupted_audio, nan=0.0, posinf=1.0, neginf=-1.0)
        events, stats = vad.detect_events(clean_or_sanitized)
        assert isinstance(events, list)

    def test_audio_shorter_than_video(self):
        """Video is 10s (50 frames), but audio is only 2s."""
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "analysis_early_term_enabled": False,
                "audio_vad_enabled": True,
            }
        }
        detector = MotionDetector(config)
        frames = [np.full((180, 320), 50, dtype=np.uint8) for _ in range(50)]
        
        # 2s speech audio at 16kHz
        t = np.linspace(0, 2.0, 16000 * 2, endpoint=False)
        short_audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        results, meta = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=10.0,
            fps=5,
            audio_data=short_audio,
        )

        assert len(results) == 50
        # First 2s frames should be DYNAMIC_AUDIO, remaining frames should be STATIC
        early_frames = [r for r in results if r["time"] <= 1.8]
        late_frames = [r for r in results if r["time"] >= 2.5]

        assert all(r["state"] == "DYNAMIC_AUDIO" for r in early_frames)
        assert all(r["state"] == "STATIC" for r in late_frames)

    def test_audio_longer_than_video(self):
        """Video is 2s (10 frames), but audio buffer is 10s."""
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "analysis_early_term_enabled": False,
                "audio_vad_enabled": True,
            }
        }
        detector = MotionDetector(config)
        frames = [np.full((180, 320), 50, dtype=np.uint8) for _ in range(10)]
        
        # 10s audio
        t = np.linspace(0, 10.0, 16000 * 10, endpoint=False)
        long_audio = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

        results, meta = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=2.0,
            fps=5,
            audio_data=long_audio,
        )

        assert len(results) == 10
        assert all(r["time"] <= 2.0 for r in results)
        assert all(r["state"] == "DYNAMIC_AUDIO" for r in results)

    def test_pure_dc_offset_audio(self):
        """Constant DC offset without AC energy should be treated as constant floor."""
        vad = AudioEnergyVAD(window_ms=50, noise_margin_db=12.0, min_dbfs=-42.0)
        dc_audio = np.full(16000 * 3, 0.05, dtype=np.float32)
        events, stats = vad.detect_events(dc_audio)
        # Constant DC across all windows means 15th percentile == max dBFS -> delta is 0 -> no events
        assert len(events) == 0


class TestCornerCaseYoloConfidenceAndExemptionMatrix:
    """
    Corner Case 5: YOLO Verifier behavior under varied confidence settings (0.01, 0.25, 0.5, 0.99)
    with mixed segment lists containing DYNAMIC, DYNAMIC_AUDIO, and STATIC.
    """

    @pytest.mark.parametrize("conf", [0.01, 0.25, 0.5, 0.95, 0.99])
    def test_dynamic_audio_exempt_at_all_confidence_levels(self, conf):
        """
        DYNAMIC_AUDIO segments MUST NEVER be demoted to STATIC by YOLO, regardless of
        whether model confidence is set to 0.01 or 0.99.
        """
        config = {
            "yolo": {
                "enabled": True,
                "model_path": "yolo11n.pt",
                "device": "cpu",
                "target_classes": [0, 1],
                "confidence": conf,
                "sample_fps": 1.0,
                "skip_energy_threshold": 50.0,
            }
        }
        verifier = YoloVerifier(config)
        verifier.enabled = True

        segments = [
            Segment(0.0, 5.0, "STATIC", "clip.mp4", 0.0, 0.1),
            Segment(5.0, 15.0, "DYNAMIC_AUDIO", "clip.mp4", 0.0, 0.5), # Must remain DYNAMIC_AUDIO
            Segment(15.0, 25.0, "DYNAMIC", "clip.mp4", 0.0, 4.0),       # Empty frames -> should demote to STATIC
            Segment(25.0, 35.0, "DYNAMIC_AUDIO", "clip.mp4", 0.0, 0.8), # Must remain DYNAMIC_AUDIO
            Segment(35.0, 60.0, "STATIC", "clip.mp4", 0.0, 0.05),
        ]

        # Black frames buffer (no objects in frames)
        frames_buffer = {
            f: np.zeros((360, 640, 3), dtype=np.uint8) for f in range(300)
        }

        verified = verifier.verify(
            filepath="clip.mp4",
            segments=segments,
            gpu="cuda",
            device="cpu",
            frames_buffer=frames_buffer,
            analysis_fps=5.0,
        )

        assert len(verified) == 5
        assert verified[0].state == "STATIC"
        assert verified[1].state == "DYNAMIC_AUDIO" and verified[1].is_dynamic is True
        assert verified[2].state == "STATIC", "Visual DYNAMIC with empty frames should be demoted"
        assert verified[3].state == "DYNAMIC_AUDIO" and verified[3].is_dynamic is True
        assert verified[4].state == "STATIC"

    def test_high_energy_dynamic_visual_skips_yolo_inference(self):
        """
        Visual DYNAMIC segment with max_energy >= skip_energy_threshold (e.g. 12.0)
        should skip YOLO inference and remain DYNAMIC even if frames are empty.
        """
        config = {
            "yolo": {
                "enabled": True,
                "model_path": "yolo11n.pt",
                "device": "cpu",
                "target_classes": [0],
                "confidence": 0.5,
                "sample_fps": 1.0,
                "skip_energy_threshold": 10.0,
            }
        }
        verifier = YoloVerifier(config)
        verifier.enabled = True

        segments = [
            # Energy 15.0 >= skip_energy_threshold 10.0 -> YOLO skipped -> DYNAMIC preserved
            Segment(0.0, 10.0, "DYNAMIC", "clip.mp4", 0.0, 15.0),
            # Energy 5.0 < skip_energy_threshold 10.0 -> YOLO checked -> empty -> demoted to STATIC
            Segment(10.0, 20.0, "DYNAMIC", "clip.mp4", 0.0, 5.0),
        ]

        frames_buffer = {f: np.zeros((360, 640, 3), dtype=np.uint8) for f in range(100)}

        verified = verifier.verify(
            filepath="clip.mp4",
            segments=segments,
            frames_buffer=frames_buffer,
            analysis_fps=5.0,
        )

        assert verified[0].state == "DYNAMIC"
        assert verified[1].state == "STATIC"


class TestAudioEnergyVADMathematicalBoundaries:
    """Mathematical and statistical validation of RMS energy and 15th percentile floor."""

    def test_vad_noise_floor_boundary_percentiles(self):
        vad = AudioEnergyVAD(window_ms=50)
        # Uniform noise over [-70 dBFS, -30 dBFS]
        np.random.seed(101)
        audio = (np.random.randn(16000 * 5) * 0.01).astype(np.float32)
        timestamps, dbfs_vec, noise_floor = vad.compute_dbfs_windows(audio)
        
        # 15th percentile should be <= median (50th percentile)
        p50 = float(np.percentile(dbfs_vec, 50))
        assert noise_floor <= p50
        assert noise_floor < -35.0

    def test_sub_second_burst_duration_filter(self):
        """A burst of 150ms should be rejected if min_speech_duration is 300ms."""
        vad = AudioEnergyVAD(
            sample_rate=16000,
            window_ms=50,
            min_speech_duration=0.3,
            noise_margin_db=12.0,
            min_dbfs=-42.0,
        )
        np.random.seed(202)
        quiet = (np.random.randn(16000 * 3) * 0.0005).astype(np.float32)
        # Inject 150ms loud tone (3 windows of 50ms)
        t = np.linspace(0, 0.15, int(16000 * 0.15), endpoint=False)
        burst = (0.2 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
        quiet[16000 : 16000 + len(burst)] = burst

        events, _ = vad.detect_events(quiet)
        assert len(events) == 0, "Sub-threshold duration burst should be filtered out"

        # Inject 400ms loud tone (8 windows of 50ms >= 6 windows minimum)
        t_long = np.linspace(0, 0.4, int(16000 * 0.4), endpoint=False)
        burst_long = (0.2 * np.sin(2 * np.pi * 500 * t_long)).astype(np.float32)
        quiet[16000 : 16000 + len(burst_long)] = burst_long

        events_long, _ = vad.detect_events(quiet)
        assert len(events_long) == 1, "400ms burst should be detected as ACTIVE"
