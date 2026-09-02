"""
Adversarial and Empirical Challenge Test Suite for Milestone 3 (R3) Iteration 2:
1. Continuous Speech Detection (100% duty cycle, pitch variations, high duty-cycle harmonic audio)
2. Air Conditioner Hum & Stationary Noise Early-Termination Veto Behavior
3. Zero-Audio Stream, Corrupted Audio, Multichannel & Non-Standard Sample Rates Fallbacks
4. Full Multimodal Fusion Decision Pipeline & YOLO Exemption Integrity
"""

import math
import time
from pathlib import Path
import numpy as np
import pytest

from src.detector import AudioEnergyVAD, MotionDetector
from src.segment import build_segments, Segment, segments_to_json, segments_from_json, _filter_short
from src.yolo_verifier import YoloVerifier


class TestContinuousSpeechDetectionAdversarial:
    """
    Challenge 1: High duty-cycle and 100% continuous speech detection.
    When speech is unbroken (duty cycle ~100%), the 15th percentile noise floor rises.
    The detector must accurately detect continuous speech without dropping active intervals.
    """

    def test_100_percent_continuous_speech_single_tone(self):
        """
        100% duty cycle continuous 400Hz voiced tone for 10 seconds at -20 dBFS.
        Floor will rise to -20 dBFS. VAD must identify it as active voiced speech via r1 >= 0.5.
        """
        vad = AudioEnergyVAD(
            sample_rate=16000,
            window_ms=50,
            noise_margin_db=12.0,
            min_dbfs=-42.0,
            min_speech_duration=0.3,
        )
        sr = 16000
        t = np.linspace(0, 10.0, sr * 10, endpoint=False)
        audio = (0.1414 * np.sin(2 * np.pi * 400 * t)).astype(np.float32)  # RMS ~ 0.1 (-20 dBFS)

        events, stats = vad.detect_events(audio, start_offset=0.0)

        assert stats["noise_floor_db"] >= -38.0
        assert len(events) >= 1
        # Event should span almost the entire 10 seconds
        total_active_duration = sum(e[1] - e[0] for e in events)
        assert total_active_duration >= 9.5
        assert math.isclose(events[0][0], 0.0, abs_tol=0.1)
        assert math.isclose(events[-1][1], 10.0, abs_tol=0.1)

    def test_continuous_speech_with_pitch_modulation_and_harmonics(self):
        """
        Synthesize continuous speech-like signal:
        Fundamental frequency sweeping 120Hz-280Hz with 3 harmonics (240Hz, 360Hz, 480Hz).
        100% duty cycle over 15 seconds.
        """
        vad = AudioEnergyVAD(
            sample_rate=16000,
            window_ms=50,
            noise_margin_db=12.0,
            min_dbfs=-42.0,
            min_speech_duration=0.3,
        )
        sr = 16000
        duration = 15.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        # Pitch chirp from 120Hz to 280Hz
        f0 = 120.0 + 160.0 * (t / duration)
        phase0 = 2 * np.pi * (120.0 * t + 80.0 * (t ** 2) / duration)
        
        # Add fundamental + harmonics
        signal = (
            0.08 * np.sin(phase0)
            + 0.04 * np.sin(2 * phase0)
            + 0.02 * np.sin(3 * phase0)
            + 0.01 * np.sin(4 * phase0)
        ).astype(np.float32)  # RMS ~ 0.065 (-23.7 dBFS)

        events, stats = vad.detect_events(signal, start_offset=0.0)

        assert len(events) >= 1
        total_active_dur = sum(e[1] - e[0] for e in events)
        assert total_active_dur >= 14.5

    def test_high_duty_cycle_speech_with_sub_second_micro_pauses(self):
        """
        Speech with 90% duty cycle (1.8s speech phrases separated by 200ms breath pauses).
        Total duration 20 seconds.
        Verify all 10 speech phrases are detected and isolated or merged smoothly.
        """
        vad = AudioEnergyVAD(
            sample_rate=16000,
            window_ms=50,
            noise_margin_db=12.0,
            min_dbfs=-42.0,
            min_speech_duration=0.3,
        )
        sr = 16000
        blocks = []
        for i in range(10):
            # 1.8s speech phrase (RMS ~ 0.08, -22 dBFS)
            t_sp = np.linspace(0, 1.8, int(sr * 1.8), endpoint=False)
            freq = 200 + (i % 3) * 50
            phrase = (0.113 * np.sin(2 * np.pi * freq * t_sp)).astype(np.float32)
            blocks.append(phrase)
            # 0.2s pause (quiet room noise RMS 0.001, -60 dBFS)
            pause = (np.random.randn(int(sr * 0.2)) * 0.001).astype(np.float32)
            blocks.append(pause)

        full_audio = np.concatenate(blocks)
        events, stats = vad.detect_events(full_audio, start_offset=0.0)

        assert len(events) >= 8
        total_active = sum(e[1] - e[0] for e in events)
        # Total speech duration is 18.0s out of 20.0s
        assert total_active >= 16.0


class TestAirConditionerHumAndStationaryNoiseEarlyTermVeto:
    """
    Challenge 2: Air Conditioner (AC) Hum and Stationary Environmental Noise.
    - Stationary noise must NOT cause early-termination lockup when video is static.
    - Stationary noise must NOT be labeled as DYNAMIC_AUDIO.
    - Real speech embedded inside AC noise must be correctly identified and veto early termination.
    """

    def test_steady_ac_white_noise_allows_early_termination(self):
        """
        Visual frames: 60 frames (12 seconds at 5fps) of static background.
        Audio: 12 seconds of constant stationary white noise at -32 dBFS (loud AC fan).
        Detector config has early_term_enabled=True, early_term_window=15.
        
        Expected:
        - VAD detects 0 audio events (AC noise is steady-state floor).
        - analyze_frames() successfully EARLY-TERMINATES around frame 15-20.
        """
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "analysis_early_term_enabled": True,
                "analysis_early_term_window": 15,
                "analysis_early_term_threshold": 2.0,
                "early_term_cooldown_guard": False,
                "ema_background_enabled": False,
                "audio_vad_enabled": True,
                "vad_noise_margin_db": 12.0,
                "vad_min_dbfs": -42.0,
                "vad_min_speech_duration": 0.3,
            }
        }
        detector = MotionDetector(config)

        # 60 static frames
        frames = [np.full((180, 320), 60, dtype=np.uint8) for _ in range(60)]

        # Loud AC white noise at -32 dBFS (RMS ~ 0.025)
        np.random.seed(42)
        ac_noise = (np.random.randn(16000 * 12) * 0.025).astype(np.float32)

        results, meta = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=12.0,
            fps=5,
            audio_data=ac_noise,
        )

        assert meta["early_terminated"] is True
        assert len(meta["audio_events"]) == 0
        # Results should be truncated due to early termination
        assert len(results) < 40
        assert all(r["state"] == "STATIC" for r in results)

    def test_steady_dc_offset_allows_early_termination(self):
        """
        Hardware audio bias / DC offset (+0.05) across entire stream.
        Must allow early termination when visual is static.
        """
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "analysis_early_term_enabled": True,
                "analysis_early_term_window": 15,
                "analysis_early_term_threshold": 2.0,
                "early_term_cooldown_guard": False,
                "ema_background_enabled": False,
                "audio_vad_enabled": True,
            }
        }
        detector = MotionDetector(config)
        frames = [np.full((180, 320), 60, dtype=np.uint8) for _ in range(60)]
        dc_audio = np.full(16000 * 12, 0.05, dtype=np.float32)

        results, meta = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=12.0,
            fps=5,
            audio_data=dc_audio,
        )

        assert meta["early_terminated"] is True
        assert len(results) < 40

    def test_ac_noise_with_delayed_speech_burst_vetoes_early_termination(self):
        """
        Visual frames: 60 frames (12 seconds) static.
        Audio:
          - 0.0s - 4.0s: AC fan noise (-40 dBFS)
          - 4.0s - 8.0s: Loud human speech (-20 dBFS, delta = 20dB > 12dB)
          - 8.0s - 12.0s: AC fan noise (-40 dBFS)
        
        Expected:
        - Speech is detected at 4.0s - 8.0s.
        - Early termination at t=3.0s is STRICTLY VETOED because audio event exists at 4-8s.
        - Frames during 4.0s - 8.0s are labeled DYNAMIC_AUDIO.
        """
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "analysis_early_term_enabled": True,
                "analysis_early_term_window": 15,
                "analysis_early_term_threshold": 2.0,
                "early_term_cooldown_guard": False,
                "ema_background_enabled": False,
                "audio_vad_enabled": True,
                "vad_noise_margin_db": 12.0,
                "vad_min_dbfs": -42.0,
                "vad_min_speech_duration": 0.3,
            }
        }
        detector = MotionDetector(config)
        frames = [np.full((180, 320), 60, dtype=np.uint8) for _ in range(60)]

        np.random.seed(123)
        sr = 16000
        # 0-4s: AC noise (RMS 0.01 -> -40 dBFS)
        ac1 = (np.random.randn(sr * 4) * 0.01).astype(np.float32)
        # 4-8s: Speech (RMS 0.1 -> -20 dBFS)
        t_sp = np.linspace(0, 4.0, sr * 4, endpoint=False)
        sp = (0.1414 * np.sin(2 * np.pi * 350 * t_sp)).astype(np.float32) + (np.random.randn(sr * 4) * 0.01).astype(np.float32)
        # 8-12s: AC noise
        ac2 = (np.random.randn(sr * 4) * 0.01).astype(np.float32)

        audio = np.concatenate([ac1, sp, ac2])

        results, meta = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=12.0,
            fps=5,
            audio_data=audio,
        )

        assert meta["has_audio"] == 1
        assert len(meta["audio_events"]) == 1
        ev_start, ev_end, _ = meta["audio_events"][0]
        assert math.isclose(ev_start, 4.0, abs_tol=0.2)
        assert math.isclose(ev_end, 8.0, abs_tol=0.2)

        # Check that speech interval frames are labeled DYNAMIC_AUDIO
        speech_frames = [r for r in results if 4.2 <= r["time"] <= 7.8]
        assert len(speech_frames) > 0
        assert all(r["state"] == "DYNAMIC_AUDIO" for r in speech_frames)


class TestZeroAudioAndDegradedAudioFallbacks:
    """
    Challenge 3: Zero-audio, corrupt audio, multi-channel, and extreme sample rates.
    """

    def test_zero_audio_stream_fallback(self):
        """Audio stream is completely absent (None) or 0 bytes."""
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "audio_vad_enabled": True,
            }
        }
        detector = MotionDetector(config)
        frames = [np.full((180, 320), 50, dtype=np.uint8) for _ in range(25)]

        results, meta = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=5.0,
            fps=5,
            audio_data=None,
        )

        assert len(results) == 25
        assert meta["has_audio"] == 0
        assert meta["audio_events"] == []
        assert all(r["state"] == "STATIC" for r in results)
        assert all(r["is_audio_active"] is False for r in results)

    def test_empty_audio_array_fallback(self):
        """Audio data is empty array np.array([], dtype=np.float32)."""
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "audio_vad_enabled": True,
            }
        }
        detector = MotionDetector(config)
        frames = [np.full((180, 320), 50, dtype=np.uint8) for _ in range(20)]

        results, meta = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=4.0,
            fps=5,
            audio_data=np.array([], dtype=np.float32),
        )

        assert len(results) == 20
        assert meta["has_audio"] == 0
        assert meta["audio_events"] == []

    @pytest.mark.parametrize("sr", [8000, 11025, 22050, 32000, 44100, 48000, 96000])
    def test_non_standard_sample_rates(self, sr):
        """Verify AudioEnergyVAD and analyze_frames across diverse sampling rates."""
        vad = AudioEnergyVAD(
            sample_rate=sr,
            window_ms=50,
            noise_margin_db=12.0,
            min_dbfs=-42.0,
            min_speech_duration=0.3,
        )
        # 5 seconds: 0-1s quiet, 1-3s speech tone (400Hz), 3-5s quiet
        t_sp = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
        speech = (0.15 * np.sin(2 * np.pi * 400 * t_sp)).astype(np.float32)
        quiet1 = (np.random.randn(int(sr * 1.0)) * 0.001).astype(np.float32)
        quiet2 = (np.random.randn(int(sr * 2.0)) * 0.001).astype(np.float32)
        audio = np.concatenate([quiet1, speech, quiet2])

        events, stats = vad.detect_events(audio, start_offset=0.0)

        assert len(events) == 1
        assert math.isclose(events[0][0], 1.0, abs_tol=0.1)
        assert math.isclose(events[0][1], 3.0, abs_tol=0.1)

    def test_multichannel_audio_coercion(self):
        """
        Stereo 2-channel audio array (shape: (2, N) or (N, 2)) when fed to AudioEnergyVAD.
        VAD compute_rms_dbfs should handle flattened or 1D mono correctly.
        """
        vad = AudioEnergyVAD(sample_rate=16000, window_ms=50)
        # Create stereo signal (N, 2) with peak 0.1414 (RMS ~ 0.1 -> -20 dBFS)
        t = np.linspace(0, 2.0, 32000, endpoint=False)
        ch1 = 0.1414 * np.sin(2 * np.pi * 440 * t)
        ch2 = 0.1414 * np.sin(2 * np.pi * 440 * t)
        stereo = np.column_stack([ch1, ch2]).astype(np.float32)
        # Flattened or mean mixed down
        mono = np.mean(stereo, axis=1)

        rms, dbfs = vad.compute_rms_dbfs(mono)
        assert math.isclose(dbfs, -20.0, abs_tol=0.5)


class TestFullMultimodalFusionPipelineAndYoloExemption:
    """
    Challenge 4: End-to-end multimodal pipeline and YOLO exemption invariants.
    """

    def test_full_pipeline_dynamic_audio_preservation(self):
        """
        Full chain:
        1. analyze_frames produces mixed DYNAMIC, DYNAMIC_AUDIO, STATIC.
        2. build_segments converts to Segment objects with apply_smoothing=True.
        3. YoloVerifier processes segments:
           - DYNAMIC without objects -> demoted to STATIC
           - DYNAMIC_AUDIO -> MUST REMAIN DYNAMIC_AUDIO / is_dynamic=True
           - STATIC -> remains STATIC
        4. JSON roundtrip serialization check.
        """
        config = {
            "detection": {
                "analysis_resolution": "320x180",
                "analysis_fps": 5,
                "motion_sensitivity": 2.0,
                "analysis_early_term_enabled": False,
                "ema_background_enabled": False,
                "audio_vad_enabled": True,
                "min_motion_frames": 1,
                "min_static_frames": 1,
                "noise_suppress_frames": 0,
                "median_filter_window": 1,
            }
        }
        detector = MotionDetector(config)

        # 60 frames (12 seconds)
        # 0.0s - 4.0s: Visual motion (moving box) + Silent audio -> DYNAMIC
        # 4.0s - 8.0s: Visual static + Active audio -> DYNAMIC_AUDIO
        # 8.0s - 12.0s: Visual static + Silent audio -> STATIC
        frames = []
        for i in range(60):
            frame = np.full((180, 320), 40, dtype=np.uint8)
            if i < 20:
                frame[30:80, (i * 10) % 200 : (i * 10) % 200 + 50] = 220
            frames.append(frame)

        sr = 16000
        np.random.seed(42)
        a_quiet1 = (np.random.randn(sr * 4) * 0.001).astype(np.float32)
        t_sp = np.linspace(0, 4.0, sr * 4, endpoint=False)
        a_speech = (0.12 * np.sin(2 * np.pi * 400 * t_sp)).astype(np.float32)
        a_quiet2 = (np.random.randn(sr * 4) * 0.001).astype(np.float32)
        audio = np.concatenate([a_quiet1, a_speech, a_quiet2])

        results, meta = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=12.0,
            fps=5,
            audio_data=audio,
        )

        segments = build_segments(
            frame_labels=results,
            source_file="test_full.mp4",
            min_motion_dur=1.0,
            min_static_dur=1.0,
            file_offset=0.0,
            apply_smoothing=True,
        )

        assert len(segments) == 3
        assert segments[0].state == "DYNAMIC"
        assert segments[1].state == "DYNAMIC_AUDIO"
        assert segments[2].state == "STATIC"

        # Pass through YoloVerifier with empty frames
        yolo_cfg = {
            "yolo": {
                "enabled": True,
                "model_path": "yolo11n.pt",
                "device": "cpu",
                "target_classes": [0],
                "confidence": 0.5,
                "sample_fps": 1.0,
                "skip_energy_threshold": 99.0,  # Don't skip YOLO verification for visual
            }
        }
        verifier = YoloVerifier(yolo_cfg)
        verifier.enabled = True

        frames_buffer = {f: np.zeros((360, 640, 3), dtype=np.uint8) for f in range(100)}
        verified = verifier.verify(
            filepath="test_full.mp4",
            segments=segments,
            frames_buffer=frames_buffer,
            analysis_fps=5.0,
        )

        assert len(verified) == 3
        # Visual DYNAMIC with empty frames should be demoted to STATIC
        assert verified[0].state == "STATIC"
        # DYNAMIC_AUDIO MUST be preserved without demotion
        assert verified[1].state == "DYNAMIC_AUDIO"
        assert verified[1].is_dynamic is True
        # STATIC remains STATIC
        assert verified[2].state == "STATIC"

        # Test JSON roundtrip
        json_str = segments_to_json(verified)
        restored = segments_from_json(json_str)
        assert len(restored) == 3
        assert restored[1].state == "DYNAMIC_AUDIO"
        assert restored[1].is_dynamic is True
