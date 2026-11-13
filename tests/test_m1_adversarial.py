"""
Adversarial Stress and Empirical Challenge Test Suite for Milestone 1 (M1).
Features tested:
1. Fallback Safeguards Stress:
   - Dense active ratio (>0.60) fallback trigger & _fallback_full_decode execution
   - Excessive isolated windows (>8) fallback trigger & _fallback_full_decode execution
   - Long GOP / sparse keyframes (avg > 8.0s or max > 15.0s) fallback trigger & full decode
   - Insufficient keyframes (<2) fallback trigger
2. Audio-Only Activity in Visually Dead Scene:
   - Completely static video frames with an Audio VAD event in the middle:
     fine window awakened, labeled DYNAMIC_AUDIO, zero recall loss in build_segments, strict timeline closure
3. Hardware Fallback & Semaphore Safety:
   - Semaphore acquisition failure handling in coarse scan & window decode
   - Decode process exception mid-stream: FFmpeg process termination, deregistration, and exact semaphore release
   - PyAV container exception: disk semaphore release guarantee
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from src.algorithms.segment import build_segments
from src.hardware.ffmpeg import FFmpegProcessRegistry, run_ffmpeg
from src.hardware.scheduler import get_disk_semaphore, get_nvdec_semaphore
import src.stages.detector as det_module
from src.stages.detector import MotionDetector


@pytest.fixture
def synthetic_video_factory():
    """Factory fixture generating isolated synthetic MP4 files via FFmpeg lavfi."""
    created_files = []

    def _generate(
        duration_s: float = 10.0,
        fps: int = 20,
        gop: int = 20,
        visual_type: str = "testsrc",  # "testsrc" or "black"
        audio_event: tuple[float, float] | None = None,  # (start_s, end_s)
    ) -> Path:
        tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tf.close()
        out_path = Path(tf.name)
        created_files.append(out_path)

        # Video source
        if visual_type == "testsrc":
            v_src = f"testsrc2=size=320x180:rate={fps}"
        else:
            v_src = f"color=c=black:s=320x180:r={fps}:d={duration_s}"

        # Audio source
        if audio_event is not None:
            ev_s, ev_e = audio_event
            ev_dur = max(0.1, ev_e - ev_s)
            delay_ms = int(ev_s * 1000)
            cmd = [
                "-f", "lavfi", "-i", v_src,
                "-f", "lavfi", "-i", f"sine=frequency=1000:duration={ev_dur}",
                "-filter_complex", f"[1:a]adelay={delay_ms}|{delay_ms},apad=whole_dur={duration_s}[aout]",
                "-map", "0:v", "-map", "[aout]",
                "-t", str(duration_s),
                "-c:v", "libx264", "-g", str(gop), "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(out_path)
            ]
        else:
            cmd = [
                "-f", "lavfi", "-i", v_src,
                "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
                "-t", str(duration_s),
                "-c:v", "libx264", "-g", str(gop), "-pix_fmt", "yuv420p",
                "-c:a", "aac",
                str(out_path)
            ]

        res = run_ffmpeg(cmd, timeout=30)
        assert res.returncode == 0, f"FFmpeg synthetic media generation failed: {res.stderr.decode(errors='replace')}"
        assert out_path.exists() and out_path.stat().st_size > 0
        return out_path

    yield _generate

    for f in created_files:
        try:
            f.unlink(missing_ok=True)
        except OSError:
            pass


class TestFallbackSafeguardsStress:
    """Targeted stress testing of all Milestone 1 fallback safeguard rules."""

    def test_dense_active_ratio_fallback_end_to_end_no_crash(self, synthetic_video_factory):
        """Stress Test 1.1: Dense motion (>0.60 active ratio) triggers fallback and executes _fallback_full_decode without crashing."""
        # 10s video with continuous high motion (testsrc2) -> active ratio ~ 1.0 > 0.60
        video_path = synthetic_video_factory(duration_s=10.0, fps=20, gop=20, visual_type="testsrc")

        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "analysis_fps": 5.0,
                "motion_sensitivity": 1.0,
                "iframe_fallback_active_ratio": 0.60,
            },
            "audio_vad": {"enabled": False},
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)

        with patch.object(detector, "_fallback_full_decode", wraps=detector._fallback_full_decode) as spy_fallback:
            labels, yolo_buf = detector.analyze(str(video_path), file_duration=10.0, duration_verified=True, has_audio=0)

            # 1. Verify fallback was triggered
            spy_fallback.assert_called_once()
            fallback_reason = spy_fallback.call_args[1].get("fallback_reason")
            assert "FALLBACK_DENSE_MOTION" in str(fallback_reason)

            # 2. Verify _fallback_full_decode executed cleanly without crashing
            assert len(labels) > 0
            assert detector.decode_completion["complete"] is True

            # 3. Verify strict timeline closure on fallback output
            assert abs(labels[0]["time"] - 0.0) < 1e-3
            assert abs(labels[-1]["time"] - 10.0) < 1e-3
            # Strictly monotonic timestamps
            times = [lbl["time"] for lbl in labels]
            assert all(b > a for a, b in zip(times, times[1:]))

    def test_excessive_isolated_windows_fallback_end_to_end_no_crash(self, synthetic_video_factory):
        """Stress Test 1.2: Excessive isolated windows (>8) trigger fallback to avoid SMB seek bubble."""
        video_path = synthetic_video_factory(duration_s=10.0, fps=20, gop=20, visual_type="testsrc")

        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "analysis_fps": 5.0,
                "iframe_max_isolated_windows": 8,
                "iframe_window_padding": 1.0,
                "iframe_window_gap_merge": 2.0,
            },
            "audio_vad": {"enabled": False},
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)

        # 10 scattered active timestamps across a 300s timeline:
        active_pts = [12.0, 42.0, 72.0, 102.0, 132.0, 162.0, 192.0, 222.0, 252.0, 282.0]
        coarse_records = []
        for t in range(0, 300, 3):
            e = 15.0 if float(t) in active_pts else 0.1
            coarse_records.append((float(t), e, e))
        meta = {"complete": True, "keyframes": len(coarse_records), "avg_gop_s": 3.0, "max_gop_s": 3.0}

        synthetic_full_frames = [np.zeros((180, 320), dtype=np.uint8) for _ in range(50)]
        decode_meta = {"complete": True, "effective_fps": 5.0, "frames": 50, "decode_time": 0.1, "completion_reason": "NORMAL_COMPLETION"}

        with patch.object(detector, "_scan_keyframes_coarse", return_value=(coarse_records, meta)), \
             patch.object(detector, "_decode_file", return_value=(synthetic_full_frames, {}, np.zeros(0, dtype=np.float32), decode_meta)), \
             patch.object(detector, "_fallback_full_decode", wraps=detector._fallback_full_decode) as spy_fallback:

            labels, yolo_buf = detector.analyze(str(video_path), file_duration=300.0, duration_verified=True, has_audio=0)

            spy_fallback.assert_called_once()
            fallback_reason = spy_fallback.call_args[1].get("fallback_reason")
            assert "FALLBACK_ISOLATED_WINDOWS" in str(fallback_reason)
            assert len(labels) > 0
            assert detector.decode_completion["complete"] is True

    def test_sparse_gop_avg_triggers_fallback_end_to_end_no_crash(self, synthetic_video_factory):
        """Stress Test 1.3: Sparse GOP (avg > 8.0s) triggers fallback and runs without crashing."""
        # GOP=240 at 20fps -> GOP interval = 12.0s > 8.0s across 24s file
        video_path = synthetic_video_factory(duration_s=24.0, fps=20, gop=240, visual_type="testsrc")

        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "analysis_fps": 5.0,
                "motion_sensitivity": 1.0,
            },
            "audio_vad": {"enabled": False},
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)

        pts_list, dur, avg_gop, max_gop = detector._extract_keyframe_pts(str(video_path))
        assert avg_gop > 8.0, f"Expected avg_gop > 8.0, got {avg_gop}"

        with patch.object(detector, "_fallback_full_decode", wraps=detector._fallback_full_decode) as spy_fallback:
            labels, yolo_buf = detector.analyze(str(video_path), file_duration=24.0, duration_verified=True, has_audio=0)

            spy_fallback.assert_called_once()
            assert spy_fallback.call_args[1].get("fallback_reason") == "FALLBACK_SPARSE_GOP"
            assert len(labels) > 0
            assert abs(labels[0]["time"] - 0.0) < 1e-3
            assert abs(labels[-1]["time"] - 24.0) < 1e-3

    def test_irregular_max_gop_triggers_fallback(self):
        """Stress Test 1.4: Irregular GOP where avg <= 8.0s but max > 15.0s triggers fallback."""
        cfg = {"detection": {"iframe_prescreen_enabled": True}, "audio_vad": {"enabled": False}, "hardware": {"device": "cpu"}}
        detector = MotionDetector(cfg)

        # 3 keyframes: t=0, t=1, t=18 in an 18s file -> avg = 6.0s (<= 8s), but max = 17.0s (> 15s)
        coarse_records = [(0.0, 0.1, 0.1), (1.0, 0.1, 0.1), (18.0, 0.1, 0.1)]
        meta = {"complete": True, "keyframes": 3, "avg_gop_s": 6.0, "max_gop_s": 17.0}

        with patch.object(detector, "_extract_keyframe_pts", return_value=([0.0, 1.0, 18.0], 18.0, 6.0, 17.0)), \
             patch.object(detector, "_scan_keyframes_coarse", return_value=(coarse_records, meta)), \
             patch.object(detector, "_fallback_full_decode", return_value=([{"time": 0.0}, {"time": 18.0}], {})) as mock_fallback:

            labels, yolo_buf = detector.analyze("irregular_gop.mp4", file_duration=18.0, duration_verified=True, has_audio=0)
            mock_fallback.assert_called_once()
            assert mock_fallback.call_args[1].get("fallback_reason") == "FALLBACK_SPARSE_GOP"

    def test_insufficient_keyframes_triggers_fallback(self):
        """Stress Test 1.5: Media with < 2 keyframes triggers fallback."""
        cfg = {"detection": {"iframe_prescreen_enabled": True}, "audio_vad": {"enabled": False}, "hardware": {"device": "cpu"}}
        detector = MotionDetector(cfg)

        coarse_records = [(0.0, 0.1, 0.1)]
        meta = {"complete": True, "keyframes": 1, "avg_gop_s": 3.0, "max_gop_s": 3.0}

        with patch.object(detector, "_extract_keyframe_pts", return_value=([0.0], 10.0, 10.0, 10.0)), \
             patch.object(detector, "_scan_keyframes_coarse", return_value=(coarse_records, meta)), \
             patch.object(detector, "_fallback_full_decode", return_value=([{"time": 0.0}, {"time": 10.0}], {})) as mock_fallback:

            labels, yolo_buf = detector.analyze("one_keyframe.mp4", file_duration=10.0, duration_verified=True, has_audio=0)
            mock_fallback.assert_called_once()
            assert mock_fallback.call_args[1].get("fallback_reason") == "FALLBACK_SPARSE_GOP"


class TestAudioVadInVisuallyDeadScene:
    """Stress testing multimodal audio awakening in a completely static video scene."""

    def test_audio_event_in_visually_dead_scene_awakens_and_zero_recall_loss(self, synthetic_video_factory):
        """Stress Test 2.1: Completely static black video with Audio VAD event in the middle:
        - Awakens window (not bypassed by Zero-Decode fast path)
        - Labels active audio region as DYNAMIC_AUDIO
        - Leaves quiescent region as STATIC
        - Produces DYNAMIC_AUDIO segment in build_segments (ZERO RECALL LOSS)
        - Strictly closes timeline
        """
        # 20 seconds total: pure black video; audio tone from 8.0s to 12.0s
        video_path = synthetic_video_factory(
            duration_s=20.0,
            fps=20,
            gop=20,
            visual_type="black",
            audio_event=(8.0, 12.0),
        )

        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "analysis_fps": 5.0,
                "motion_sensitivity": 1.0,
                "iframe_window_padding": 3.0,
                "iframe_window_gap_merge": 6.0,
            },
            "audio_vad": {"enabled": True, "min_speech_duration": 0.5},
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)

        labels, yolo_buf = detector.analyze(str(video_path), file_duration=20.0, duration_verified=True)

        # 1. Verify fine decode was invoked (NORMAL_COMPLETION for downstream policy, TIER2_WINDOWED_COMPLETION in last_perf)
        assert detector.decode_completion["reason"] == "NORMAL_COMPLETION"
        assert detector.last_perf["completion_reason"] == "TIER2_WINDOWED_COMPLETION"

        # 2. Verify state labeling
        states = set(lbl["state"] for lbl in labels)
        assert "DYNAMIC_AUDIO" in states, "DYNAMIC_AUDIO state must be present in labels"
        assert "STATIC" in states, "STATIC state must be present in labels"
        assert "DYNAMIC" not in states, "Pure black video must have no visual DYNAMIC state"

        # 3. Verify audio event window coverage
        audio_labels = [lbl for lbl in labels if lbl["state"] == "DYNAMIC_AUDIO"]
        assert len(audio_labels) > 0
        min_audio_t = min(lbl["time"] for lbl in audio_labels)
        max_audio_t = max(lbl["time"] for lbl in audio_labels)
        assert min_audio_t <= 8.5
        assert max_audio_t >= 11.5

        # 4. Zero Recall Loss: Downstream segmentation contract
        segments = build_segments(labels, source_file="test_audio.mp4", min_motion_dur=0.5, min_static_dur=1.0)
        assert len(segments) >= 2
        audio_segs = [s for s in segments if s.state == "DYNAMIC_AUDIO"]
        assert len(audio_segs) == 1, "Exactly one DYNAMIC_AUDIO segment must be identified"
        seg = audio_segs[0]
        assert seg.start_time <= 8.5
        assert seg.end_time >= 12.0

        # 5. Strict timeline closure
        assert abs(labels[0]["time"] - 0.0) < 1e-3
        assert abs(labels[-1]["time"] - 20.0) < 1e-3


class TestHardwareFallbackAndSemaphoreSafety:
    """Stress testing hardware resource exhaustion, timeouts, and process crash cleanup."""

    def test_coarse_scan_semaphore_timeout_clean_fallback(self, synthetic_video_factory):
        """Stress Test 3.1: Hardware semaphore acquisition timeout in coarse scan triggers safe fallback."""
        video_path = synthetic_video_factory(duration_s=5.0, fps=20, gop=20, visual_type="testsrc")

        cfg = {"detection": {"iframe_prescreen_enabled": True, "analysis_fps": 5.0}, "hardware": {"device": "cpu"}}
        detector = MotionDetector(cfg)

        box = {"target_sem": None}
        orig_build = detector._build_coarse_scan_cmd
        def fake_build(filepath, decode_gpu):
            cmd, sem = orig_build(filepath, decode_gpu)
            box["target_sem"] = sem
            return cmd, sem

        orig_acquire = det_module.acquire_with_retry
        def mock_acquire(sem, timeout=30.0, retries=6):
            if box["target_sem"] is not None and sem is box["target_sem"]:
                return False
            return orig_acquire(sem, timeout, retries)

        with patch.object(detector, "_build_coarse_scan_cmd", side_effect=fake_build), \
             patch("src.stages.detector.acquire_with_retry", side_effect=mock_acquire), \
             patch.object(detector, "_fallback_full_decode", wraps=detector._fallback_full_decode) as spy_fallback:

            labels, yolo_buf = detector.analyze(str(video_path), file_duration=5.0, duration_verified=True, has_audio=0)
            spy_fallback.assert_called_once()
            assert len(labels) > 0
            assert detector.decode_completion["complete"] is True

    def test_coarse_scan_stdout_read_exception_process_cleanup_and_semaphore_release(self, synthetic_video_factory):
        """Stress Test 3.2: Mid-stream FFmpeg pipe exception cleanly terminates process and releases semaphore."""
        video_path = synthetic_video_factory(duration_s=5.0, fps=20, gop=20, visual_type="testsrc")

        cfg = {"detection": {"iframe_prescreen_enabled": True, "analysis_fps": 5.0}, "hardware": {"device": "cpu"}}
        detector = MotionDetector(cfg)

        disk = get_disk_semaphore()
        init_avail = disk.available

        orig_start = FFmpegProcessRegistry.start
        box = {"proc": None, "pipe_key": None}
        def fake_start(pipe_key, cmd, **kwargs):
            box["pipe_key"] = pipe_key
            p = orig_start(pipe_key, cmd, **kwargs)
            box["proc"] = p
            orig_read = p.stdout.read
            count = [0]
            def crashing_read(n):
                count[0] += 1
                if count[0] > 1:
                    raise OSError("Simulated hardware bus disconnect on pipe read")
                return orig_read(n)
            p.stdout.read = crashing_read
            return p

        with patch.object(FFmpegProcessRegistry, "start", side_effect=fake_start):
            detector.analyze(str(video_path), file_duration=5.0, duration_verified=True, has_audio=0)

        # 1. Semaphore must be strictly balanced
        assert disk.available == init_avail, f"Semaphore leak detected! Initial: {init_avail}, Final: {disk.available}"

        # 2. Process must be terminated
        if box["proc"]:
            assert FFmpegProcessRegistry.has_exited(box["proc"]) is True

    def test_window_decode_exception_process_cleanup_and_semaphore_release(self, synthetic_video_factory):
        """Stress Test 3.3: Mid-stream exception in Tier 2 window decode cleanly terminates process and releases lease."""
        video_path = synthetic_video_factory(duration_s=10.0, fps=20, gop=20, visual_type="testsrc")

        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "analysis_fps": 5.0,
                "iframe_window_padding": 1.0,
                "iframe_window_gap_merge": 2.0,
            },
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)
        disk = get_disk_semaphore()
        init_avail = disk.available

        orig_start = FFmpegProcessRegistry.start
        box = {"window_proc": None}
        def fake_start(pipe_key, cmd, **kwargs):
            p = orig_start(pipe_key, cmd, **kwargs)
            if "pipe_window" in pipe_key:
                box["window_proc"] = p
                orig_read = p.stdout.read
                count = [0]
                def crashing_read(n):
                    count[0] += 1
                    if count[0] > 1:
                        raise OSError("Simulated window decode pipe error")
                    return orig_read(n)
                p.stdout.read = crashing_read
            return p

        with patch.object(FFmpegProcessRegistry, "start", side_effect=fake_start), \
             patch.object(detector, "_fallback_full_decode", wraps=detector._fallback_full_decode) as spy_fallback:

            labels, yolo_buf = detector.analyze(str(video_path), file_duration=10.0, duration_verified=True, has_audio=0)
            spy_fallback.assert_called_once()
            assert len(labels) > 0

        # Verify semaphore release and process termination
        assert disk.available == init_avail, f"Semaphore leak! Initial: {init_avail}, Final: {disk.available}"
        if box["window_proc"]:
            assert FFmpegProcessRegistry.has_exited(box["window_proc"]) is True

    def test_extract_keyframe_pts_exception_releases_disk_semaphore(self):
        """Stress Test 3.4: Container parsing exception in _extract_keyframe_pts guarantees disk semaphore release."""
        cfg = {"detection": {"iframe_prescreen_enabled": True}, "hardware": {"device": "cpu"}}
        detector = MotionDetector(cfg)
        disk = get_disk_semaphore()
        init_avail = disk.available

        with patch("av.open", side_effect=RuntimeError("Corrupt media container header")):
            with pytest.raises(RuntimeError, match="Corrupt media container header"):
                detector._extract_keyframe_pts("corrupted_sample.mp4")

        assert disk.available == init_avail, f"Disk semaphore leak in _extract_keyframe_pts! Init={init_avail}, Final={disk.available}"


class TestZeroDecodeFastPathStress:
    """Empirical challenge 1: Zero-Decode Fast Path under micro-noise and single-cell burst."""

    def test_micro_noise_below_threshold_triggers_zero_decode_and_zero_clock_drift(self):
        """Micro-noise below threshold triggers zero decode, generates clean STATIC labels, and 0 tick clock drift."""
        tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tf.close()
        out_path = Path(tf.name)
        try:
            cmd = [
                "-f", "lavfi", "-i", "color=c=black:s=320x180:r=20:d=10",
                "-vf", "noise=alls=5:allf=t",
                "-t", "10",
                "-c:v", "libx264", "-g", "20", "-pix_fmt", "yuv420p",
                "-an",
                str(out_path)
            ]
            res = run_ffmpeg(cmd, timeout=30)
            assert res.returncode == 0

            cfg = {
                "detection": {
                    "iframe_prescreen_enabled": True,
                    "analysis_fps": 5.0,
                    "motion_sensitivity": 1.0,
                    "min_motion_threshold": 2.0,
                    "iframe_energy_margin": 0.80,
                    "iframe_burst_ratio": 1.8,
                },
                "audio_vad": {"enabled": False},
                "hardware": {"device": "cpu"},
            }
            detector = MotionDetector(cfg)
            start_offset = 1234.56
            file_duration = 10.0

            with patch.object(detector, "_decode_file_window") as mock_decode_win, \
                 patch.object(detector, "_fallback_full_decode") as mock_fallback:

                labels, yolo_buf = detector.analyze(
                    str(out_path),
                    start_offset=start_offset,
                    file_duration=file_duration,
                    duration_verified=True,
                    has_audio=0,
                )

                # 1. Zero decode was triggered
                mock_decode_win.assert_not_called()
                mock_fallback.assert_not_called()
                assert detector.decode_completion["reason"] == "ZERO_DECODE_FAST_PATH"
                assert detector.last_perf["completion_reason"] == "ZERO_DECODE_FAST_PATH"
                assert detector.decode_completion["complete"] is True

                # 2. Pure clean STATIC labels
                assert len(labels) > 0
                assert all(lbl["state"] == "STATIC" for lbl in labels)
                assert all(lbl["is_motion"] is False for lbl in labels)
                assert all(lbl["is_audio_active"] is False for lbl in labels)
                assert all(lbl["energy"] == 0.0 for lbl in labels)

                # 3. Exact start and closure
                times = [l["time"] for l in labels]
                assert abs(times[0] - start_offset) < 1e-4
                assert abs(times[-1] - (start_offset + file_duration)) < 1e-4
                assert all(b > a for a, b in zip(times, times[1:]))

                # 4. Zero clock drift downstream in build_segments
                segments = build_segments(labels, "test_micronoise.mp4", file_offset=start_offset)
                assert len(segments) == 1
                assert segments[0].state == "STATIC"
                assert abs(segments[0].start_time - start_offset) < 1e-4
                assert abs(segments[0].end_time - (start_offset + file_duration)) < 1e-4
                assert abs(segments[0].duration - file_duration) < 1e-4
        finally:
            out_path.unlink(missing_ok=True)

    def test_micro_noise_adversarial_boundary_just_below_threshold(self):
        """Micro-noise with energy and max_cell_energy infinitesimally below threshold triggers zero decode."""
        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "analysis_fps": 5.0,
                "min_motion_threshold": 2.0,
                "iframe_energy_margin": 0.80,
                "iframe_burst_ratio": 1.8,
            },
            "audio_vad": {"enabled": False},
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)
        start_offset = 5000.0
        file_duration = 30.0

        # eff_threshold = 2.0 * 0.80 = 1.60
        # burst_threshold = 2.0 * 1.8 = 3.60
        # Energy and cell energy are 0.999 * threshold
        coarse_records = []
        for t in range(0, 31, 3):
            coarse_records.append((float(t), 1.599, 3.599))
        meta = {"complete": True, "keyframes": len(coarse_records), "avg_gop_s": 3.0, "max_gop_s": 3.0}

        with patch.object(detector, "_extract_keyframe_pts", return_value=([float(t) for t in range(0, 31, 3)], 30.0, 3.0, 3.0)), \
             patch.object(detector, "_scan_keyframes_coarse", return_value=(coarse_records, meta)), \
             patch.object(detector, "_decode_file_window") as mock_win, \
             patch.object(detector, "_fallback_full_decode") as mock_fallback:

            labels, yolo_buf = detector.analyze(
                "boundary_micronoise.mp4",
                start_offset=start_offset,
                file_duration=file_duration,
                duration_verified=True,
                has_audio=0,
            )

            mock_win.assert_not_called()
            mock_fallback.assert_not_called()
            assert detector.decode_completion["reason"] == "ZERO_DECODE_FAST_PATH"
            assert all(lbl["state"] == "STATIC" for lbl in labels)
            times = [lbl["time"] for lbl in labels]
            assert abs(times[0] - start_offset) < 1e-4
            assert abs(times[-1] - (start_offset + file_duration)) < 1e-4
            assert all(b > a for a, b in zip(times, times[1:]))

    def test_single_8x8_cell_spatial_burst_triggers_fine_decode_when_mean_energy_low(self):
        """Single 8x8 cell spatial burst awakens Tier 2 fine decode even when frame mean energy is low."""
        tf = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tf.close()
        out_path = Path(tf.name)
        try:
            # 30s video with flash of a 16x16 white square at t=15.0 in 320x180 resolution
            vf = "drawbox=x=60:y=60:w=16:h=16:color=white:t=fill:enable='between(t,14.95,15.05)'"
            cmd = [
                "-f", "lavfi", "-i", "color=c=black:s=320x180:r=20:d=30",
                "-vf", vf,
                "-t", "30",
                "-c:v", "libx264", "-g", "20", "-pix_fmt", "yuv420p",
                "-an",
                str(out_path)
            ]
            res = run_ffmpeg(cmd, timeout=30)
            assert res.returncode == 0

            cfg = {
                "detection": {
                    "iframe_prescreen_enabled": True,
                    "analysis_fps": 5.0,
                    "motion_sensitivity": 1.0,
                    "min_motion_threshold": 2.0,
                    "iframe_energy_margin": 0.80,
                    "iframe_burst_ratio": 1.8,
                    "iframe_window_padding": 3.0,
                },
                "audio_vad": {"enabled": False},
                "hardware": {"device": "cpu"},
            }
            detector = MotionDetector(cfg)

            with patch.object(detector, "_fallback_full_decode", wraps=detector._fallback_full_decode) as spy_fallback:
                labels, yolo_buf = detector.analyze(
                    str(out_path),
                    start_offset=0.0,
                    file_duration=30.0,
                    duration_verified=True,
                    has_audio=0,
                )

                # 1. Tier 2 fine decode was awakened (NOT zero decode, NOT fallback)
                spy_fallback.assert_not_called()
                assert detector.decode_completion["reason"] == "NORMAL_COMPLETION"
                assert detector.last_perf["completion_reason"] == "TIER2_WINDOWED_COMPLETION"
                assert detector.last_perf["frames"] > 0

                # 2. Verify spatial cell energy burst was captured in fine window
                max_cell_e = max(l.get("max_cell_energy", 0.0) for l in labels)
                assert max_cell_e >= 50.0, f"Spatial cell burst should be preserved in fine window labels, got {max_cell_e}"

                # 3. Timeline closure and monotonicity
                assert abs(labels[0]["time"] - 0.0) < 1e-4
                assert abs(labels[-1]["time"] - 30.0) < 1e-4
                times = [l["time"] for l in labels]
                assert all(b > a for a, b in zip(times, times[1:]))
        finally:
            out_path.unlink(missing_ok=True)

    @pytest.mark.parametrize("burst_cell_energy,expected_active", [
        (3.59, False),  # just below burst_threshold (3.60)
        (3.61, True),   # just above burst_threshold (3.60)
        (10.0, True),   # clear burst
    ])
    def test_single_cell_burst_parameterized_oracle(self, burst_cell_energy, expected_active):
        """Oracle verification of single cell burst threshold boundary with near-zero mean energy."""
        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "iframe_energy_margin": 0.80,
                "iframe_burst_ratio": 1.8,
            },
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)
        records = [(15.0, 0.1, burst_cell_energy)]
        windows = detector._resolve_awakened_windows(
            keyframe_trace=records,
            audio_events=[],
            file_duration=30.0,
            threshold=2.0,
            gop_duration=3.0,
        )
        if expected_active:
            assert len(windows) == 1
            assert windows[0][0] <= 12.0
            assert windows[0][1] >= 18.0
        else:
            assert len(windows) == 0


class TestWindowPaddingAndBoundaryClamping:
    """Empirical challenge 2: Window padding, boundary clamping at t=0 and t=file_duration, and gap merge."""

    def test_event_at_t_zero_clamps_to_zero_without_negative_time(self):
        """Active event at t=0.0 clamps cleanly to 0.0 without negative times."""
        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "iframe_window_padding": 4.0,
                "iframe_window_gap_merge": 6.0,
            },
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)
        records = [(0.0, 10.0, 10.0)]
        windows = detector._resolve_awakened_windows(
            keyframe_trace=records,
            audio_events=[],
            file_duration=20.0,
            threshold=1.0,
            gop_duration=3.0,
        )
        assert len(windows) == 1
        assert windows[0][0] == 0.0
        assert windows[0][1] == 4.0

    def test_event_at_file_duration_clamps_without_overshoot(self):
        """Active event at t=file_duration clamps cleanly to file_duration without overshoot."""
        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "iframe_window_padding": 4.0,
                "iframe_window_gap_merge": 6.0,
            },
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)
        file_duration = 25.0
        records = [(file_duration, 10.0, 10.0)]
        windows = detector._resolve_awakened_windows(
            keyframe_trace=records,
            audio_events=[],
            file_duration=file_duration,
            threshold=1.0,
            gop_duration=3.0,
        )
        assert len(windows) == 1
        assert windows[0][0] == 21.0
        assert windows[0][1] == file_duration

    def test_simultaneous_boundary_events_at_both_ends(self):
        """Events simultaneously at t=0.0 and t=file_duration clamp to both ends."""
        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "iframe_window_padding": 3.0,
                "iframe_window_gap_merge": 6.0,
            },
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)
        file_duration = 30.0
        records = [(0.0, 10.0, 10.0), (file_duration, 10.0, 10.0)]
        windows = detector._resolve_awakened_windows(
            keyframe_trace=records,
            audio_events=[],
            file_duration=file_duration,
            threshold=1.0,
            gop_duration=3.0,
        )
        assert len(windows) == 2
        assert windows[0] == (0.0, 3.0)
        assert windows[1] == (27.0, file_duration)

    def test_gap_merge_boundary_exact_and_slightly_above(self):
        """Verify merging behavior when window gap is exactly at or slightly above iframe_window_gap_merge."""
        gap_merge = 6.0
        padding = 3.0
        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "iframe_window_padding": padding,
                "iframe_window_gap_merge": gap_merge,
            },
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)
        file_duration = 60.0

        # Window 1: active at t=5.0 -> [2.0, 8.0] (end = 8.0)
        # Case A: Window 2 active at t = 17.0 -> w2_start = 14.0. Gap = 14.0 - 8.0 = 6.0 == gap_merge
        records_exact = [(5.0, 10.0, 10.0), (17.0, 10.0, 10.0)]
        w_exact = detector._resolve_awakened_windows(
            keyframe_trace=records_exact,
            audio_events=[],
            file_duration=file_duration,
            threshold=1.0,
            gop_duration=3.0,
        )
        assert len(w_exact) == 1
        assert w_exact[0] == (2.0, 20.0)

        # Case B: Window 2 active at t = 17.001 -> w2_start = 14.001. Gap = 6.001 > 6.0
        records_above = [(5.0, 10.0, 10.0), (17.001, 10.0, 10.0)]
        w_above = detector._resolve_awakened_windows(
            keyframe_trace=records_above,
            audio_events=[],
            file_duration=file_duration,
            threshold=1.0,
            gop_duration=3.0,
        )
        assert len(w_above) == 2
        assert w_above[0] == (2.0, 8.0)
        assert w_above[1] == (14.001, 20.001)

    def test_out_of_bound_audio_events_clamped(self):
        """Audio events with negative start or end > file_duration clamp cleanly."""
        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "iframe_window_padding": 2.0,
                "iframe_window_gap_merge": 4.0,
            },
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)
        file_duration = 20.0
        audio_events = [(-2.0, 1.0, "SPEECH"), (19.0, 25.0, "SPEECH")]
        windows = detector._resolve_awakened_windows(
            keyframe_trace=[],
            audio_events=audio_events,
            file_duration=file_duration,
            threshold=1.0,
            gop_duration=2.0,
        )
        assert len(windows) == 2
        assert windows[0][0] == 0.0
        assert windows[0][1] == 3.0
        assert windows[1][0] == 17.0
        assert windows[1][1] == file_duration


class TestTimelineClosureAndMonotonicityStress:
    """Empirical challenge 3: Timeline closure, timestamp strict monotonicity, and zero clock drift."""

    def _make_dummy_trace(self, detector, start_t, end_t, fps=5.0, is_motion=True):
        from src.algorithms.motion_trace import MotionTrace
        dur = end_t - start_t
        n = max(1, int(round(dur * fps)))
        trace = MotionTrace(detector, fps=fps)
        e = 10.0 if is_motion else 0.1
        trace.energies = [e] * n
        trace.confidences = [0.9] * n
        trace.active_cells = [5] * n
        trace.max_cell_energies = [10.0] * n
        return trace

    @pytest.mark.parametrize("start_offset", [0.0, 500.0, 86399.5])
    @pytest.mark.parametrize("file_duration", [10.0, 33.333, 100.0, 3600.0])
    def test_stitched_windows_and_gaps_strictly_monotonic_and_exact_closure(self, start_offset, file_duration):
        """Verify timestamps across stitched windows and static gaps are strictly monotonic and close exactly."""
        cfg = {"detection": {"iframe_prescreen_enabled": True}, "hardware": {"device": "cpu"}}
        detector = MotionDetector(cfg)

        w1_s = max(0.0, file_duration * 0.2)
        w1_e = min(file_duration, w1_s + min(4.0, file_duration * 0.1))
        w2_s = max(w1_e + 2.0, file_duration * 0.7)
        w2_e = min(file_duration, w2_s + min(4.0, file_duration * 0.1))

        windows = [(w1_s, w1_e), (w2_s, w2_e)]
        window_traces = [
            (ws, we, self._make_dummy_trace(detector, ws, we, fps=5.0))
            for ws, we in windows
        ]

        labels = detector._stitch_windowed_labels(
            windows=windows,
            window_traces=window_traces,
            audio_events=[],
            start_offset=start_offset,
            file_duration=file_duration,
            effective_fps=5.0,
            baseline_threshold=1.0,
        )

        assert len(labels) >= 2
        times = [l["time"] for l in labels]

        assert abs(times[0] - start_offset) < 1e-4, f"Start mismatch: {times[0]} vs {start_offset}"
        assert abs(times[-1] - (start_offset + file_duration)) < 1e-4, f"Closure mismatch: {times[-1]} vs {start_offset + file_duration}"

        diffs = [b - a for a, b in zip(times, times[1:])]
        min_diff = min(diffs)
        assert min_diff > 0, f"Non-monotonic timestamp detected: min diff = {min_diff} at offset={start_offset}, dur={file_duration}"

    def test_audio_event_overlapping_window_boundary_strictly_monotonic(self):
        """Audio event straddling window boundary preserves strict monotonicity and timeline closure."""
        cfg = {"detection": {"iframe_prescreen_enabled": True}, "hardware": {"device": "cpu"}}
        detector = MotionDetector(cfg)
        start_offset = 100.0
        file_duration = 20.0

        windows = [(5.0, 10.0)]
        window_traces = [(5.0, 10.0, self._make_dummy_trace(detector, 5.0, 10.0, fps=5.0))]
        audio_events = [(4.0, 7.0, "SPEECH")]

        labels = detector._stitch_windowed_labels(
            windows=windows,
            window_traces=window_traces,
            audio_events=audio_events,
            start_offset=start_offset,
            file_duration=file_duration,
            effective_fps=5.0,
            baseline_threshold=1.0,
        )

        times = [l["time"] for l in labels]
        assert abs(times[0] - start_offset) < 1e-4
        assert abs(times[-1] - (start_offset + file_duration)) < 1e-4
        diffs = [b - a for a, b in zip(times, times[1:])]
        assert all(d > 0 for d in diffs), f"Non-monotonic timestamps: min diff = {min(diffs)}"

        audio_labels = [l for l in labels if l["is_audio_active"]]
        assert len(audio_labels) > 0
        assert any(l["state"] == "DYNAMIC_AUDIO" for l in audio_labels)

    def test_downstream_timeline_segments_continuity_and_zero_clock_drift(self):
        """Verify downstream build_segments and timeline integration has zero clock drift across stitched labels."""
        cfg = {"detection": {"iframe_prescreen_enabled": True}, "hardware": {"device": "cpu"}}
        detector = MotionDetector(cfg)
        start_offset = 3600.0
        file_duration = 50.0

        windows = [(10.0, 20.0), (35.0, 42.0)]
        window_traces = [
            (ws, we, self._make_dummy_trace(detector, ws, we, fps=5.0))
            for ws, we in windows
        ]

        labels = detector._stitch_windowed_labels(
            windows=windows,
            window_traces=window_traces,
            audio_events=[],
            start_offset=start_offset,
            file_duration=file_duration,
            effective_fps=5.0,
            baseline_threshold=1.0,
        )

        segments = build_segments(labels, "test_stitched.mp4", file_offset=start_offset)
        assert len(segments) >= 3

        assert abs(segments[0].start_time - start_offset) < 1e-4
        assert abs(segments[-1].end_time - (start_offset + file_duration)) < 1e-4

        for s1, s2 in zip(segments, segments[1:]):
            assert abs(s2.start_time - s1.end_time) < 1e-4, f"Seam discontinuity: {s1.end_time} vs {s2.start_time}"

        total_dur = sum(s.duration for s in segments)
        assert abs(total_dur - file_duration) < 1e-4, f"Clock drift: total_dur={total_dur} vs {file_duration}"

