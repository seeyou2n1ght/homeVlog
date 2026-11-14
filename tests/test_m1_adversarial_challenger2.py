"""Adversarial Challenge & Stress-Test Suite for Milestone 1 (M1) Iteration 2.

Author: Challenger 2 (EMPIRICAL CHALLENGER)
Focus Areas:
1. Device Lease Re-entry Stress:
   - Sequential files across all types (static, active, dense, sparse GOP, exception)
   - Zero generator re-entry errors and zero unreleased semaphore leaks
   - Concurrency stress across worker threads
   - NVDEC acquisition timeout degradation safety
2. Fallback Safeguards Stress:
   - Dense active ratio (>0.60 vs <=0.60 boundary)
   - Excessive isolated windows (>8 vs <=8 boundary)
   - Long GOP average (>8.0s vs <=8.0s boundary)
   - Long GOP max (>15.0s with >=3 keyframes)
   - Insufficient keyframes (<2)
3. Audio-Only Activity in Visually Dead Scene:
   - Completely static video with Audio VAD events
   - Fine window awakening, DYNAMIC_AUDIO labeling, zero recall loss in build_segments
   - Timeline monotonicity and boundary closure
4. PyAV Open Failure:
   - Missing input reports an incomplete decode and releases its I/O permit
"""

import os
import tempfile
import threading
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest

from src.algorithms.segment import build_segments
from src.hardware.ffmpeg import FFmpegProcessRegistry, run_ffmpeg
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

        if visual_type == "testsrc":
            v_src = f"testsrc2=size=320x180:rate={fps}"
        else:
            v_src = f"color=c=black:s=320x180:r={fps}:d={duration_s}"

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


class MockHardwareScheduler:
    """Mock hardware scheduler tracking exact semaphore acquisitions, releases, and active slots."""

    def __init__(self, max_slots: int = 2):
        self.max_slots = max_slots
        self.active_slots = 0
        self.acquire_count = 0
        self.release_count = 0
        self.lock = threading.Lock()
        self.timeout_mode = False

    def acquire_nvdec_slot(self, timeout: float = 0.0) -> bool:
        if self.timeout_mode:
            return False
        with self.lock:
            if self.active_slots < self.max_slots:
                self.active_slots += 1
                self.acquire_count += 1
                return True
            return False

    def release_nvdec_slot(self) -> None:
        with self.lock:
            if self.active_slots <= 0:
                raise RuntimeError("NVDEC slot underflow: release called without active acquisition!")
            self.active_slots -= 1
            self.release_count += 1

    def get_analysis_device(self, queue_size: int, prescreen_idle: bool = False, file_duration: float = 0.0) -> str:
        return "cuda"

    @contextmanager
    def lease_device(self, queue_size: int = 1, prescreen_idle: bool = False, file_duration: float = 0.0):
        device = self.get_analysis_device(queue_size, prescreen_idle, file_duration)
        acquired = False
        if device == "cuda":
            if self.acquire_nvdec_slot():
                acquired = True
            else:
                device = "qsv"
        try:
            yield device
        finally:
            if acquired:
                self.release_nvdec_slot()


# ============================================================================
# 1. Device Lease Re-entry Stress Tests
# ============================================================================

class TestDeviceLeaseReentryStress:
    """Stress-test detector.device_lease across complex multi-file lifecycles."""

    def test_device_lease_sequential_files_mixed_workload(self):
        """Stress: 15 sequential files of diverse types using a SINGLE MotionDetector instance.
        Verifies:
        - Zero generator re-entry errors ('_GeneratorContextManager' object has no attribute 'args')
        - Zero slot leaks (active_slots == 0 after every file)
        - Exact acquire == release balance
        """
        scheduler = MockHardwareScheduler(max_slots=2)
        detector = MotionDetector({"detection": {"iframe_prescreen_enabled": True}}, decode_gpu="cuda")
        detector.device_lease = lambda dur=30.0: scheduler.lease_device(file_duration=dur)

        # Workload definitions
        # Type A: Pure static (1 lease in Tier 1 -> zero decode)
        static_keyframes = [(float(t), 0.05, 0.05) for t in range(0, 30, 3)]
        static_meta = {"complete": True, "keyframes": 10, "avg_gop": 3.0, "max_gop": 3.0}

        # Type B: Active (1 lease in Tier 1 + 1 lease in Tier 2 window decode = 2 leases)
        active_keyframes = [(0.0, 0.05, 0.05), (3.0, 15.0, 20.0), (6.0, 0.05, 0.05), (9.0, 0.05, 0.05)]
        active_meta = {"complete": True, "keyframes": 4, "avg_gop": 3.0, "max_gop": 3.0}
        active_synth_frames = [np.full((180, 320), 200, dtype=np.uint8) for _ in range(30)]

        # Type C: Dense motion fallback (1 lease in Tier 1 + 1 lease in fallback full decode = 2 leases)
        dense_keyframes = [(float(t), 25.0, 30.0) for t in range(0, 30, 3)]
        dense_meta = {"complete": True, "keyframes": 10, "avg_gop": 3.0, "max_gop": 3.0}
        full_synth_frames = [np.zeros((180, 320), dtype=np.uint8) for _ in range(150)]
        decode_meta = {"complete": True, "effective_fps": 5.0, "frames": 150, "decode_time": 0.1, "completion_reason": "NORMAL_COMPLETION"}

        # Type D: Sparse GOP fallback (demux -> 1 lease in fallback full decode = 1 lease)
        sparse_meta = {"complete": True, "keyframes": 2, "avg_gop": 15.0, "max_gop": 15.0}

        file_types = ["static", "active", "dense", "sparse", "active"] * 3  # 15 files total

        for idx, f_type in enumerate(file_types):
            filename = f"file_{idx:02d}_{f_type}.mp4"

            if f_type == "static":
                with patch.object(detector, "_scan_keyframes_coarse", return_value=(static_keyframes, static_meta)), \
                     patch.object(detector, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
                     patch.object(detector.vad, "detect_events", return_value=([], {})):
                    labels, _ = detector.analyze(filename, file_duration=30.0, duration_verified=True, has_audio=0)
                assert detector.decode_completion["reason"] == "ZERO_DECODE_FAST_PATH"

            elif f_type == "active":
                with patch.object(detector, "_scan_keyframes_coarse", return_value=(active_keyframes, active_meta)), \
                     patch.object(detector, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
                     patch.object(detector.vad, "detect_events", return_value=([], {})), \
                     patch.object(detector, "_decode_file_window", return_value=(active_synth_frames, 30)):
                    labels, _ = detector.analyze(filename, file_duration=12.0, duration_verified=True, has_audio=0)
                assert detector.decode_completion["reason"] == "NORMAL_COMPLETION"
                assert detector.last_perf["completion_reason"] == "TIER2_WINDOWED_COMPLETION"

            elif f_type == "dense":
                with patch.object(detector, "_scan_keyframes_coarse", return_value=(dense_keyframes, dense_meta)), \
                     patch.object(detector, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
                     patch.object(detector.vad, "detect_events", return_value=([], {})), \
                     patch.object(detector, "_decode_file", return_value=(full_synth_frames, {}, np.zeros(0, dtype=np.float32), decode_meta)):
                    labels, _ = detector.analyze(filename, file_duration=30.0, duration_verified=True, has_audio=0)
                assert detector.decode_completion["reason"] in ("NORMAL_COMPLETION", "FALLBACK_DENSE_MOTION")

            elif f_type == "sparse":
                with patch.object(detector, "_extract_keyframe_pts", return_value=([0.0, 15.0], 30.0, 15.0, 15.0)), \
                     patch.object(detector, "_scan_keyframes_coarse", return_value=([(0.0, 0.1, 0.1), (15.0, 0.1, 0.1)], sparse_meta)), \
                     patch.object(detector, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
                     patch.object(detector.vad, "detect_events", return_value=([], {})), \
                     patch.object(detector, "_decode_file", return_value=(full_synth_frames, {}, np.zeros(0, dtype=np.float32), decode_meta)):
                    labels, _ = detector.analyze(filename, file_duration=30.0, duration_verified=True, has_audio=0)
                assert detector.decode_completion["reason"] in ("NORMAL_COMPLETION", "FALLBACK_SPARSE_GOP")

            # Invariant: slots MUST be completely released after EVERY file
            assert scheduler.active_slots == 0, f"Lease slot leak detected after {filename}! active_slots={scheduler.active_slots}"
            assert scheduler.acquire_count == scheduler.release_count, f"Acquire/Release mismatch after {filename}"

        assert scheduler.acquire_count > 0

    def test_device_lease_concurrent_threads_stress(self):
        """Stress: 4 concurrent worker threads repeatedly leasing device through detector instances."""
        scheduler = MockHardwareScheduler(max_slots=3)
        errors = []

        def worker_loop(thread_id: int):
            try:
                det = MotionDetector({"detection": {"iframe_prescreen_enabled": True}}, decode_gpu="cuda")
                det.device_lease = lambda dur=10.0: scheduler.lease_device(file_duration=dur)

                keyframes = [(0.0, 0.05, 0.05), (3.0, 10.0, 15.0), (6.0, 0.05, 0.05)]
                meta = {"complete": True, "keyframes": 3, "avg_gop": 3.0, "max_gop": 3.0}
                frames = [np.full((180, 320), 200, dtype=np.uint8) for _ in range(15)]

                for i in range(8):
                    with patch.object(det, "_scan_keyframes_coarse", return_value=(keyframes, meta)), \
                         patch.object(det, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
                         patch.object(det, "_decode_file_window", return_value=(frames, 15)):
                        det.analyze(f"thread_{thread_id}_file_{i}.mp4", file_duration=10.0, duration_verified=True, has_audio=0)
                    time.sleep(0.005)
            except Exception as e:
                errors.append(f"Thread {thread_id} crashed: {type(e).__name__}: {e}")

        threads = [threading.Thread(target=worker_loop, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        assert not errors, f"Concurrent thread errors: {errors}"
        assert scheduler.active_slots == 0, f"Slot leak in concurrent test! active_slots={scheduler.active_slots}"
        assert scheduler.acquire_count == scheduler.release_count

    def test_device_lease_nvdec_timeout_degrades_cleanly_without_leak(self):
        """Stress: When NVDEC acquisition times out, lease degrades to QSV without slot underflow."""
        scheduler = MockHardwareScheduler(max_slots=1)
        scheduler.timeout_mode = True  # Always fail NVDEC acquisition

        detector = MotionDetector({"detection": {"iframe_prescreen_enabled": True}}, decode_gpu="cuda")
        detector.device_lease = lambda dur=30.0: scheduler.lease_device(file_duration=dur)

        active_keyframes = [(0.0, 0.05, 0.05), (3.0, 15.0, 20.0), (6.0, 0.05, 0.05)]
        active_meta = {"complete": True, "keyframes": 3, "avg_gop": 3.0, "max_gop": 3.0}
        active_synth_frames = [np.full((180, 320), 200, dtype=np.uint8) for _ in range(15)]

        with patch.object(detector, "_scan_keyframes_coarse", return_value=(active_keyframes, active_meta)), \
             patch.object(detector, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
             patch.object(detector.vad, "detect_events", return_value=([], {})), \
             patch.object(detector, "_decode_file_window", return_value=(active_synth_frames, 15)):
            labels, _ = detector.analyze("timeout_test.mp4", file_duration=10.0, duration_verified=True, has_audio=0)

        # Since acquire returned False, acquired_slot was False -> no release should be called
        assert scheduler.active_slots == 0
        assert scheduler.acquire_count == 0
        assert scheduler.release_count == 0
        assert detector.decode_completion["complete"] is True


# ============================================================================
# 2. Fallback Safeguards Stress Tests
# ============================================================================

class TestFallbackSafeguardsStressChallenger2:
    """Stress-test exact boundaries of all Milestone 1 fallback safeguard rules."""

    def test_dense_motion_active_ratio_boundary(self):
        """Boundary Test: Active ratio > 0.60 triggers fallback; active ratio <= 0.60 proceeds to Tier 2."""
        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "analysis_fps": 5.0,
                "iframe_fallback_active_ratio": 0.60,
                "iframe_window_padding": 1.0,
                "iframe_window_gap_merge": 2.0,
            },
            "audio_vad": {"enabled": False},
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)

        file_dur = 100.0
        gop_meta = {"keyframes": 34, "avg_gop_s": 3.0, "max_gop_s": 3.0}

        # Case A: Total active duration = 59.0s (59% <= 60%) -> PROCEED_TIER2
        windows_sub = [(10.0, 69.0)]
        should_fb, reason = detector._evaluate_fallback_safeguards(windows_sub, file_dur, gop_meta)
        assert should_fb is False
        assert reason == "PROCEED_TIER2"

        # Case B: Total active duration = 61.0s (61% > 60%) -> FALLBACK_DENSE_MOTION_ACTIVE_RATIO
        windows_super = [(10.0, 71.0)]
        should_fb, reason = detector._evaluate_fallback_safeguards(windows_super, file_dur, gop_meta)
        assert should_fb is True
        assert "FALLBACK_DENSE_MOTION_ACTIVE_RATIO" in reason

    def test_excessive_isolated_windows_boundary(self):
        """Boundary Test: Windows > 8 triggers fallback; windows <= 8 proceeds to Tier 2."""
        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "analysis_fps": 5.0,
                "iframe_max_isolated_windows": 8,
                "iframe_fallback_active_ratio": 0.90,  # avoid triggering active ratio
            },
            "audio_vad": {"enabled": False},
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)
        file_dur = 500.0
        gop_meta = {"keyframes": 100, "avg_gop_s": 3.0, "max_gop_s": 3.0}

        # Exactly 8 windows (durations 2s each -> total 16s << 500s)
        windows_8 = [(float(i * 30), float(i * 30 + 2)) for i in range(8)]
        should_fb, reason = detector._evaluate_fallback_safeguards(windows_8, file_dur, gop_meta)
        assert should_fb is False
        assert reason == "PROCEED_TIER2"

        # Exactly 9 windows
        windows_9 = [(float(i * 30), float(i * 30 + 2)) for i in range(9)]
        should_fb, reason = detector._evaluate_fallback_safeguards(windows_9, file_dur, gop_meta)
        assert should_fb is True
        assert "FALLBACK_ISOLATED_WINDOWS_9_GT_8" in reason

    def test_sparse_gop_average_boundary(self):
        """Boundary Test: avg_gop > 8.0s triggers fallback; avg_gop <= 8.0s proceeds."""
        cfg = {"detection": {"iframe_prescreen_enabled": True}, "audio_vad": {"enabled": False}, "hardware": {"device": "cpu"}}
        detector = MotionDetector(cfg)
        windows = [(1.0, 4.0)]

        # avg_gop = 8.0s -> proceeds
        meta_ok = {"keyframes": 5, "avg_gop_s": 8.0, "max_gop_s": 8.0}
        should_fb, reason = detector._evaluate_fallback_safeguards(windows, 40.0, meta_ok)
        assert should_fb is False

        # avg_gop = 8.1s -> fallback
        meta_sparse = {"keyframes": 5, "avg_gop_s": 8.1, "max_gop_s": 8.1}
        should_fb, reason = detector._evaluate_fallback_safeguards(windows, 40.0, meta_sparse)
        assert should_fb is True
        assert "FALLBACK_SPARSE_GOP_8.1S" in reason

    def test_sparse_gop_max_boundary(self):
        """Boundary Test: max_gop > 15.0s with >= 3 keyframes triggers fallback."""
        cfg = {"detection": {"iframe_prescreen_enabled": True}, "audio_vad": {"enabled": False}, "hardware": {"device": "cpu"}}
        detector = MotionDetector(cfg)
        windows = [(1.0, 4.0)]

        # max_gop = 15.0s -> proceeds
        meta_ok = {"keyframes": 3, "avg_gop_s": 5.0, "max_gop_s": 15.0}
        should_fb, reason = detector._evaluate_fallback_safeguards(windows, 30.0, meta_ok)
        assert should_fb is False

        # max_gop = 15.1s -> fallback
        meta_sparse = {"keyframes": 3, "avg_gop_s": 5.0, "max_gop_s": 15.1}
        should_fb, reason = detector._evaluate_fallback_safeguards(windows, 30.0, meta_sparse)
        assert should_fb is True
        assert "FALLBACK_SPARSE_GOP_15.1S" in reason

    def test_insufficient_keyframes_triggers_fallback(self):
        """Boundary Test: media with < 2 keyframes triggers fallback."""
        cfg = {"detection": {"iframe_prescreen_enabled": True}, "audio_vad": {"enabled": False}, "hardware": {"device": "cpu"}}
        detector = MotionDetector(cfg)
        windows = [(1.0, 4.0)]

        meta_one = {"keyframes": 1, "avg_gop_s": 3.0, "max_gop_s": 3.0}
        should_fb, reason = detector._evaluate_fallback_safeguards(windows, 30.0, meta_one)
        assert should_fb is True
        assert reason == "INSUFFICIENT_KEYFRAMES"


# ============================================================================
# 3. Audio-Only Activity in Visually Dead Scene Tests
# ============================================================================

class TestAudioInVisuallyDeadSceneChallenger2:
    """Stress testing multimodal audio awakening in completely static scenes."""

    def test_audio_event_in_static_scene_awakens_fine_window_and_preserves_recall(self, synthetic_video_factory):
        """Adversarial Stress: Completely static black video with Audio VAD event in the middle:
        - Awakens fine decode window (zero-decode fast path bypassed)
        - Labels audio region as DYNAMIC_AUDIO
        - Labels quiescent region as STATIC
        - Produces DYNAMIC_AUDIO segment in build_segments (zero recall loss)
        - Strictly monotonic and closes timeline [0.0, file_duration]
        """
        file_dur = 20.0
        video_path = synthetic_video_factory(
            duration_s=file_dur,
            fps=20,
            gop=20,
            visual_type="black",
            audio_event=(7.0, 11.0),
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

        labels, yolo_buf = detector.analyze(str(video_path), file_duration=file_dur, duration_verified=True)

        # 1. Telemetry verification: fine window decode was invoked
        assert detector.decode_completion["complete"] is True
        assert detector.decode_completion["reason"] == "NORMAL_COMPLETION"
        assert detector.last_perf["completion_reason"] == "TIER2_WINDOWED_COMPLETION"

        # 2. State verification
        states = set(lbl["state"] for lbl in labels)
        assert "DYNAMIC_AUDIO" in states, "DYNAMIC_AUDIO state must be present"
        assert "STATIC" in states, "STATIC state must be present"
        assert "DYNAMIC" not in states, "Pure black video must have no visual DYNAMIC state"

        # 3. Timestamps coverage of audio event
        audio_labels = [lbl for lbl in labels if lbl["state"] == "DYNAMIC_AUDIO"]
        assert len(audio_labels) > 0
        min_audio_t = min(lbl["time"] for lbl in audio_labels)
        max_audio_t = max(lbl["time"] for lbl in audio_labels)
        assert min_audio_t <= 7.5
        assert max_audio_t >= 10.5

        # 4. Zero recall loss in segmentation contract
        segments = build_segments(labels, source_file="static_audio.mp4", min_motion_dur=0.5, min_static_dur=1.0)
        audio_segs = [s for s in segments if s.state == "DYNAMIC_AUDIO"]
        assert len(audio_segs) == 1, "Exactly one DYNAMIC_AUDIO segment must be produced"
        seg = audio_segs[0]
        assert seg.start_time <= 7.5
        assert seg.end_time >= 11.0

        # 5. Strict timeline closure & monotonicity
        assert abs(labels[0]["time"] - 0.0) < 1e-3
        assert abs(labels[-1]["time"] - file_dur) < 1e-3
        times = [lbl["time"] for lbl in labels]
        assert all(b > a for a, b in zip(times, times[1:])), "Timestamps must be strictly monotonic"

    def test_multiple_audio_events_separated_awaken_multiple_windows(self):
        """Stress: Multiple separated audio events in a static scene awaken independent fine windows."""
        cfg = {
            "detection": {
                "iframe_prescreen_enabled": True,
                "analysis_fps": 5.0,
                "iframe_window_padding": 2.0,
                "iframe_window_gap_merge": 4.0,
            },
            "audio_vad": {"enabled": True},
            "hardware": {"device": "cpu"},
        }
        detector = MotionDetector(cfg)

        # 100s file, pure static keyframes (energy 0.01)
        keyframe_trace = [(float(t), 0.01, 0.01) for t in range(0, 100, 3)]
        # Two audio events at [10, 12] and [60, 62], gap = 48s > 4.0s
        audio_events = [(10.0, 12.0, "SPEECH"), (60.0, 62.0, "SPEECH")]

        windows = detector._resolve_awakened_windows(
            keyframe_trace=keyframe_trace,
            audio_events=audio_events,
            file_duration=100.0,
            threshold=1.0,
            gop_duration=3.0,
        )

        assert len(windows) == 2, f"Expected exactly 2 awakened windows, got {len(windows)}: {windows}"
        w1, w2 = windows[0], windows[1]
        assert w1[0] <= 10.0 and w1[1] >= 12.0
        assert w2[0] <= 60.0 and w2[1] >= 62.0


# ============================================================================
# 4. PyAV Open Failure
# ============================================================================

def test_pyav_missing_file_reports_incomplete_and_releases_io(tmp_path, monkeypatch, caplog):
    missing_source = tmp_path / "missing.mp4"
    io_permit = threading.BoundedSemaphore(1)
    monkeypatch.setattr("src.hardware.scheduler.get_disk_semaphore", lambda: io_permit)
    detector = MotionDetector({}, decode_gpu="cpu")

    frames, yolo_buffer, _, metadata = detector._decode_file_pyav(str(missing_source), 10.0)

    assert len(frames) == 0 and not yolo_buffer
    assert metadata["frames"] == 0
    assert metadata["complete"] is False
    assert metadata["completion_error"].startswith("FileNotFoundError:")
    assert missing_source.name in metadata["completion_error"]
    assert f"PyAV decode error for {missing_source.name}" in caplog.text
    assert io_permit.acquire(blocking=False)
    io_permit.release()
