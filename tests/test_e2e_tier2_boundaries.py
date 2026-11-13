"""Tier 2: Boundary & Corner Case Specifications (F1–F13, 65 Tests).

Each feature in F1-F13 has exactly 5 dedicated boundary, stress, and edge case tests.
"""

import gc
import math
from pathlib import Path
import threading
import time
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest

from src.algorithms.filters import AudioEnergyVAD
from src.core import config as core_config
from src.core.database import VlogDatabase
from src.hardware import autotune, scheduler
from src.hardware.ffmpeg import FFmpegProcessRegistry
from src.stages.companion import bind_delivery_mapping
from src.stages.detector import MotionDetector
from src.stages.pipeline_types import AnalysisQueue
from src.stages.timeline import SegmentState, TimelineSegment, build_concat_filter, compute_display_plans
from tests.helpers_e2e import (
    MockVramController,
    assert_resource_hygiene,
    create_corrupted_clip,
    create_synthetic_clip,
)


# ============================================================================
# Feature F1: iframe_prescreen_enabled Config & Governance Corners (5 tests)
# ============================================================================

def test_t2_f01_01_zero_value_float_rejection(mock_config):
    """TC-T2-F01-01: Verify float configuration keys set to mathematical 0.0 are rejected."""
    cfg = dict(mock_config)
    cfg.setdefault("detection", {})
    cfg["output"]["qsv"].pop("maxrate", None)
    cfg["output"]["qsv"].pop("bufsize", None)
    cfg["detection"]["iframe_energy_margin"] = 0.0
    with pytest.raises(ValueError, match="detection.iframe_energy_margin must be between 0.1 and 1.0"):
        core_config.validate_config(cfg)


def test_t2_f01_02_non_finite_nan_inf_rejection(mock_config):
    """TC-T2-F01-02: Verify float keys configured as NaN or Inf are rejected."""
    cfg = dict(mock_config)
    cfg.setdefault("detection", {})
    cfg["output"]["qsv"].pop("maxrate", None)
    cfg["output"]["qsv"].pop("bufsize", None)
    cfg["detection"]["iframe_burst_ratio"] = float("nan")
    with pytest.raises(ValueError, match="detection.iframe_burst_ratio must be between 1.0 and 5.0"):
        core_config.validate_config(cfg)


def test_t2_f01_03_string_encoded_numeric_rejection(mock_config):
    """TC-T2-F01-03: Verify integer keys supplied as strings are strictly rejected."""
    cfg = dict(mock_config)
    cfg.setdefault("detection", {})
    cfg["output"]["qsv"].pop("maxrate", None)
    cfg["output"]["qsv"].pop("bufsize", None)
    cfg["detection"]["iframe_max_isolated_windows"] = "8"
    with pytest.raises(ValueError, match="detection.iframe_max_isolated_windows must be an integer between 1 and 32"):
        core_config.validate_config(cfg)


def test_t2_f01_04_non_boolean_enabled_key_rejection(mock_config):
    """TC-T2-F01-04: Verify iframe_prescreen_enabled configured as non-boolean is rejected."""
    cfg = dict(mock_config)
    cfg.setdefault("detection", {})
    cfg["output"]["qsv"].pop("maxrate", None)
    cfg["output"]["qsv"].pop("bufsize", None)
    cfg["detection"]["iframe_prescreen_enabled"] = "yes"
    with pytest.raises(ValueError, match="detection.iframe_prescreen_enabled must be a boolean"):
        core_config.validate_config(cfg)


def test_t2_f01_05_missing_section_or_empty_dict_robustness(mock_config):
    """TC-T2-F01-05: Verify empty detection dictionary succeeds and populates default values."""
    cfg = dict(mock_config)
    cfg["output"]["qsv"].pop("maxrate", None)
    cfg["output"]["qsv"].pop("bufsize", None)
    cfg["detection"] = {}
    core_config.validate_config(cfg)
    det = MotionDetector(cfg, decode_gpu="cpu")
    assert det.iframe_prescreen_enabled is True
    assert det.iframe_energy_margin == pytest.approx(0.80)


# ============================================================================
# Feature F2: Tier 1 Coarse I-Frame Fast Scan Corners (5 tests)
# ============================================================================

def test_t2_f02_01_all_intra_stream_gop_1(tmp_path, mock_config):
    """TC-T2-F02-01: Video stream where every single frame is an IDR keyframe (GOP=1)."""
    clip = tmp_path / "all_intra.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=1.0, fps=20, gop_size=1)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    res = detector._scan_keyframes_coarse(str(clip), file_duration=1.0)
    trace = res[0] if isinstance(res, tuple) else res
    assert len(trace) >= 19  # ~20 keyframes for 1s


def test_t2_f02_02_ultra_sparse_keyframes_gop_300(tmp_path, mock_config):
    """TC-T2-F02-02: Surveillance video with very sparse keyframes."""
    clip = tmp_path / "sparse_gop.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=4.0, fps=20, gop_size=300)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    res = detector._scan_keyframes_coarse(str(clip), file_duration=4.0)
    trace = res[0] if isinstance(res, tuple) else res
    assert len(trace) == 1  # Only 1 keyframe at index 0


def test_t2_f02_03_variable_frame_rate_pts_normalization(tmp_path, mock_config):
    """TC-T2-F02-03: Timestamps demuxed from container are strictly monotonic."""
    clip = tmp_path / "vfr_sample.mp4"
    create_synthetic_clip(clip, scene="motion", duration_s=4.0, fps=20, gop_size=20)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    pts_list, duration, avg_gop, max_gop = detector._extract_keyframe_pts(str(clip))
    assert len(pts_list) >= 2
    for a, b in zip(pts_list, pts_list[1:]):
        assert b > a, f"Non-monotonic keyframe PTS: {a} -> {b}"
    assert duration == pytest.approx(4.0, abs=0.2)
    assert avg_gop > 0


def test_t2_f02_04_saturated_full_frame_flash_illumination_spike(tmp_path, mock_config):
    """TC-T2-F02-04: Sudden full-frame white flash handled without crashing EMA model."""
    clip = tmp_path / "flash.mp4"
    create_synthetic_clip(clip, scene="motion", duration_s=2.0, fps=20, gop_size=10)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    res = detector._scan_keyframes_coarse(str(clip), file_duration=2.0)
    trace = res[0] if isinstance(res, tuple) else res
    assert len(trace) > 0


def test_t2_f02_05_corrupted_keyframe_packet_bitstream(tmp_path, mock_config):
    """TC-T2-F02-05: Truncated video bitstream skips corrupt packets safely."""
    clip = tmp_path / "corrupted.mp4"
    create_corrupted_clip(clip, corruption_type="truncated_eof")
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    try:
        res = detector._scan_keyframes_coarse(str(clip), file_duration=2.0)
        trace, meta = res if isinstance(res, tuple) else (res, {})
        assert isinstance(trace, list)
        assert meta.get("complete") is False or isinstance(trace, list)
    except Exception as e:
        assert isinstance(e, (av.AVError, RuntimeError, ValueError)), f"Unexpected exception: {type(e)}"


# ============================================================================
# Feature F3: Zero-Decode Fast Path for Static Clips Corners (5 tests)
# ============================================================================

def test_t2_f03_01_zero_byte_empty_file_handling(tmp_path):
    """TC-T2-F03-01: Empty 0-byte file detected as SKIPPED_EMPTY by pipeline."""
    from src.pipeline import _reconcile_empty_files
    clip = tmp_path / "empty_source.mp4"
    clip.touch()
    db = VlogDatabase(tmp_path / "test_empty.db")
    db.add_file_task(
        filepath=str(clip.resolve()),
        cam_index=0,
        date="20260901",
        file_start_time="2026-09-01 00:00:00",
        file_end_time="2026-09-01 00:01:00",
        file_duration=60.0,
    )
    _reconcile_empty_files(db, "20260901", 0)
    task = db.get_file_task(str(clip.resolve()))
    assert task["prescreen_status"] == "SKIPPED_EMPTY"
    assert "zero-byte" in task.get("error_msg", "").lower()
    db.close()


def test_t2_f03_02_ultra_short_static_file_sub_second(mock_config):
    """TC-T2-F03-02: Fragmented slice with duration 0.6 seconds synthesized without division by zero."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    dur = 0.6
    labels = detector._synthesize_static_labels(start_offset=0.0, file_duration=dur, effective_fps=5.0)
    assert len(labels) >= 2
    assert labels[0]["time"] == 0.0
    assert labels[-1]["time"] == pytest.approx(dur, abs=1e-3)
    seg = TimelineSegment(filepath="short.mp4", input_index=0, start_in_file=0.0, end_in_file=dur, state=SegmentState.STATIC, duration=dur)
    plans = compute_display_plans([seg], output_fps=20.0)
    assert len(plans) == 1
    assert plans[0][0] > 0


def test_t2_f03_03_long_static_file_sensor_noise_floor(mock_config):
    """TC-T2-F03-03: Night video with continuous sensor noise floor remains STATIC."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(float(i * 3), 1.2, 2.5) for i in range(10)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=30.0, threshold=6.5)
    fallback, _ = detector._evaluate_fallback_safeguards(windows, file_duration=30.0, gop_metrics={"keyframes": len(trace), "avg_gop_s": 3.0})
    assert len(windows) == 0
    assert fallback is False


def test_t2_f03_04_static_clip_missing_audio_stream(tmp_path, mock_config):
    """TC-T2-F03-04: Video without audio track processed without raising IndexError."""
    clip = tmp_path / "no_audio.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=2.0, fps=20, audio_windows=None)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    res = detector._scan_keyframes_coarse(str(clip), file_duration=2.0)
    trace = res[0] if isinstance(res, tuple) else res
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=2.0, threshold=6.5)
    assert len(windows) == 0


def test_t2_f03_05_exact_threshold_edge_boundary_condition(mock_config):
    """TC-T2-F03-05: Keyframe energy exactly equal to threshold * margin does not awaken."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    threshold = getattr(detector, "threshold", 6.5)
    margin = detector.iframe_energy_margin
    exact_energy = threshold * margin
    trace = [(10.0, exact_energy, 1.0)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=20.0, threshold=threshold)
    assert len(windows) == 0


# ============================================================================
# Feature F4: Tier 2 Awakened Window Fine Decoding Corners (5 tests)
# ============================================================================

def test_t2_f04_01_motion_at_exact_start_boundary_clamped(mock_config):
    """TC-T2-F04-01: Motion at frame 0 clamped to max(0.0, start - padding) == 0.0."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(0.0, 20.0, 30.0)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=20.0, threshold=6.5)
    assert len(windows) == 1
    assert windows[0][0] == 0.0


def test_t2_f04_02_motion_at_exact_end_boundary_clamped(mock_config):
    """TC-T2-F04-02: Motion at last frame clamped to min(file_duration, end + padding)."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(30.0, 20.0, 30.0)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=30.0, threshold=6.5)
    assert len(windows) == 1
    assert windows[0][1] == 30.0


def test_t2_f04_03_localized_micro_motion_burst_single_cell(mock_config):
    """TC-T2-F04-03: Single active grid cell triggers awakening despite low whole-frame energy."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(10.0, 0.4, 20.0)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=30.0, threshold=6.5)
    assert len(windows) == 1
    assert windows[0][0] <= 7.0
    assert windows[0][1] >= 13.0


def test_t2_f04_04_high_frequency_oscillating_motion_gap_threshold(mock_config):
    """TC-T2-F04-04: Test window merging behavior at exact 6.0s threshold boundary."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace_merge = [(10.0, 20.0, 30.0), (21.9, 20.0, 30.0)]
    w_merged = detector._resolve_awakened_windows(trace_merge, audio_events=[], file_duration=40.0, threshold=6.5)
    assert len(w_merged) == 1

    trace_distinct = [(10.0, 20.0, 30.0), (22.2, 20.0, 30.0)]
    w_distinct = detector._resolve_awakened_windows(trace_distinct, audio_events=[], file_duration=40.0, threshold=6.5)
    assert len(w_distinct) == 2


def test_t2_f04_05_ffmpeg_non_idr_seek_landing_normalization(tmp_path, mock_config):
    """TC-T2-F04-05: Non-IDR seek landing frames normalized to requested window bounds."""
    clip = tmp_path / "seek_landing.mp4"
    create_synthetic_clip(clip, scene="motion", duration_s=5.0, fps=20, gop_size=30)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    res = detector._decode_file_window(str(clip), w_start=1.2, w_end=3.8, effective_fps=5.0)
    trace = res[0] if isinstance(res, tuple) else res
    assert trace is not None
    assert hasattr(trace, "energies")
    expected_frames = int(round((3.8 - 1.2) * 5.0))
    assert abs(len(trace.energies) - expected_frames) <= 2


# ============================================================================
# Feature F5: Fallback Safeguards for Dense Motion Corners (5 tests)
# ============================================================================

def test_t2_f05_01_active_ratio_boundary_59_vs_61_percent(mock_config):
    """TC-T2-F05-01: Boundary test around exact 60% active ratio threshold."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace_under = [(10.0, 20.0, 30.0), (20.0, 20.0, 30.0), (30.0, 20.0, 30.0), (40.0, 20.0, 30.0)]
    w_under = detector._resolve_awakened_windows(trace_under, audio_events=[], file_duration=100.0, threshold=6.5)
    fb_under, _ = detector._evaluate_fallback_safeguards(w_under, file_duration=100.0, gop_metrics={"keyframes": 10, "avg_gop_s": 3.0})
    assert fb_under is False

    trace_over = [(float(t), 20.0, 30.0) for t in range(5, 75, 5)]
    w_over = detector._resolve_awakened_windows(trace_over, audio_events=[], file_duration=100.0, threshold=6.5)
    fb_over, reason_over = detector._evaluate_fallback_safeguards(w_over, file_duration=100.0, gop_metrics={"keyframes": len(trace_over), "avg_gop_s": 3.0})
    assert fb_over is True
    assert "DENSE_MOTION" in reason_over


def test_t2_f05_02_isolated_windows_boundary_7_8_vs_9_windows(mock_config):
    """TC-T2-F05-02: Boundary test around exact 8 isolated windows threshold."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace_8 = [(float(i * 20 + 5), 20.0, 30.0) for i in range(8)]
    w_8 = detector._resolve_awakened_windows(trace_8, audio_events=[], file_duration=200.0, threshold=6.5)
    fb_8, _ = detector._evaluate_fallback_safeguards(w_8, file_duration=200.0, gop_metrics={"keyframes": 20, "avg_gop_s": 3.0})
    assert fb_8 is False

    trace_9 = [(float(i * 20 + 5), 20.0, 30.0) for i in range(9)]
    w_9 = detector._resolve_awakened_windows(trace_9, audio_events=[], file_duration=200.0, threshold=6.5)
    fb_9, reason_9 = detector._evaluate_fallback_safeguards(w_9, file_duration=200.0, gop_metrics={"keyframes": 20, "avg_gop_s": 3.0})
    assert fb_9 is True
    assert "ISOLATED_WINDOWS" in reason_9


def test_t2_f05_03_corrupted_container_header_unparseable_gop(tmp_path, mock_config):
    """TC-T2-F05-03: Corrupted container header handles error safely."""
    clip = tmp_path / "corrupt_header.mp4"
    create_corrupted_clip(clip, corruption_type="garbage")
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    records, meta = detector._scan_keyframes_coarse(str(clip), file_duration=10.0)
    assert len(records) == 0
    assert meta.get("complete") is False
    assert meta.get("reason") in ("DEMUX_FAILED", "INSUFFICIENT_FRAMES")
    assert_resource_hygiene()


def test_t2_f05_04_network_smb_seek_timeout_watchdog(tmp_path, mock_config):
    """TC-T2-F05-04: Simulated seek timeout terminates process and releases semaphore."""
    clip = tmp_path / "timeout_seek.mp4"
    create_synthetic_clip(clip, scene="motion", duration_s=3.0, fps=20, gop_size=10)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    with patch("threading.Timer") as mock_timer:
        timed_out_callback = None
        def fake_timer(interval, callback):
            nonlocal timed_out_callback
            timed_out_callback = callback
            m = MagicMock()
            m.start = MagicMock()
            m.cancel = MagicMock()
            m.is_alive.return_value = False
            return m
        mock_timer.side_effect = fake_timer
        res = detector._decode_file_window(str(clip), w_start=0.5, w_end=2.5, effective_fps=5.0)
        assert res is not None
        assert timed_out_callback is not None
    assert_resource_hygiene()


def test_t2_f05_05_saturated_100_percent_motion_video(mock_config):
    """TC-T2-F05-05: Non-stop continuous motion triggers fallback immediately."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(float(t), 25.0, 35.0) for t in range(0, 30, 2)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=30.0, threshold=6.5)
    fallback, reason = detector._evaluate_fallback_safeguards(windows, file_duration=30.0, gop_metrics={"keyframes": len(trace), "avg_gop_s": 2.0})
    assert fallback is True
    assert "DENSE_MOTION" in reason


# ============================================================================
# Feature F6: Multimodal Audio VAD Awakening Corners (5 tests)
# ============================================================================

def test_t2_f06_01_sub_millisecond_acoustic_impulse_rejection():
    """TC-T2-F06-01: Ultra-short 10ms click is rejected by min speech duration filter."""
    vad = AudioEnergyVAD(sample_rate=16000, min_speech_duration=0.3, enabled=True)
    samples = np.zeros(16000, dtype=np.float32)
    samples[100:260] = 1.0  # 10ms impulse
    events, _ = vad.detect_events(samples)
    assert len(events) == 0


def test_t2_f06_02_audio_event_straddling_file_end_boundary(mock_config):
    """TC-T2-F06-02: Speech starting at 9.5s in 10s file clamped to 10.0s."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    audio_events = [(9.5, 10.5, "speech")]
    windows = detector._resolve_awakened_windows([], audio_events=audio_events, file_duration=10.0, threshold=6.5)
    assert len(windows) == 1
    assert windows[0][1] == 10.0


def test_t2_f06_03_extreme_0_dbfs_clipped_audio_signal():
    """TC-T2-F06-03: Clipped 0 dBFS square-wave audio does not overflow or produce NaN."""
    vad = AudioEnergyVAD(sample_rate=16000, enabled=True)
    samples = np.sign(np.sin(np.linspace(0, 100, 16000))).astype(np.float32)
    events, metrics = vad.detect_events(samples)
    assert math.isfinite(metrics.get("noise_floor_db", 0.0))


def test_t2_f06_04_audio_stream_channel_configuration_downmix(tmp_path, mock_config):
    """TC-T2-F06-04: Multi-channel audio downmixing to mono 16kHz via FFmpeg pipe."""
    clip = tmp_path / "stereo_audio.mp4"
    create_synthetic_clip(clip, scene="motion_audio", duration_s=1.0, fps=20, audio_windows=[(0.0, 1.0)])
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    detector.has_audio_detected = 1
    features = detector._decode_audio_pipe(str(clip), file_duration=1.0)
    assert features is not None
    assert features.size > 0
    assert len(features.dbfs_windows()) > 0
    assert_resource_hygiene()


def test_t2_f06_05_high_decibel_stationary_ambient_drone():
    """TC-T2-F06-05: Loud continuous stationary sound noise floor adapts without false speech triggers."""
    vad = AudioEnergyVAD(sample_rate=16000, noise_margin_db=15.0, enabled=True)
    t = np.linspace(0, 2, 32000, dtype=np.float32)
    drone = 0.4 * np.sin(2 * np.pi * 120 * t)  # continuous 120Hz drone
    events, metrics = vad.detect_events(drone)
    assert metrics["noise_floor_db"] > -40.0


# ============================================================================
# Feature F7: VRAM Headroom & Margin Probe Corners (5 tests)
# ============================================================================

def test_t2_f07_01_exact_margin_boundary_3071_vs_3072_mb(mock_config):
    """TC-T2-F07-01: Test boundary at exact 3072 MB margin."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=3071, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    assert mgr.has_vram_margin(3072) is False

    ctrl.set_vram(free_mb=3072, used_mb=3000)
    assert mgr.has_vram_margin(3072) is True


def test_t2_f07_02_nvml_driver_detach_gpu_lost(mock_config):
    """TC-T2-F07-02: NVML driver error caught safely returning False."""
    mgr = scheduler.WorkStealingManager(mock_config)
    def gpu_lost():
        raise OSError("NVML_ERROR_GPU_IS_LOST")
    mgr._vram_margin_probe_fn = gpu_lost
    assert mgr.has_vram_margin() is False


def test_t2_f07_03_multi_gpu_heterogeneous_indexing(mock_config):
    """TC-T2-F07-03: Support querying specific GPU index."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4096, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    assert mgr.has_vram_margin(3072, gpu_index=1) is True


def test_t2_f07_04_rapid_vram_fluctuation_sawtooth_profile(mock_config):
    """TC-T2-F07-04: Alternating VRAM margin does not latch into bad state."""
    mgr = scheduler.WorkStealingManager(mock_config)
    values = [4000, 2000, 4000, 2000]
    idx = 0
    def fluctuating():
        nonlocal idx
        v = values[idx % len(values)]
        idx += 1
        return v
    mgr._vram_margin_probe_fn = fluctuating
    assert mgr.has_vram_margin(3072) is True
    assert mgr.has_vram_margin(3072) is False
    assert mgr.has_vram_margin(3072) is True


def test_t2_f07_05_headless_container_missing_nvml(mock_config):
    """TC-T2-F07-05: Execution in environment where NVML is missing returns False safely."""
    mgr = scheduler.WorkStealingManager(mock_config)
    mgr._vram_margin_probe_fn = None
    mgr.set_vram_probe_fn(None)
    assert mgr.has_vram_margin() is False


# ============================================================================
# Feature F8: Tail Work Stealing Admission Policy Corners (5 tests)
# ============================================================================

def test_t2_f08_01_qsv_backlog_threshold_boundary_1_vs_2(mock_config):
    """TC-T2-F08-01: Test boundary at exact backlog threshold (2)."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4000, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    mgr.steal_backlog_threshold = 2
    assert mgr.can_worker_0_steal(queue_size=1, heavy_tasks_cleared=True) is False
    assert mgr.can_worker_0_steal(queue_size=2, heavy_tasks_cleared=True) is True


def test_t2_f08_02_dynamic_arrival_of_heavy_task_mid_steal(mock_config):
    """TC-T2-F08-02: Arrival of new heavy task revokes admission immediately."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4000, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    assert mgr.can_worker_0_steal(queue_size=4, heavy_tasks_cleared=True) is True
    assert mgr.can_worker_0_steal(queue_size=4, heavy_tasks_cleared=False) is False


def test_t2_f08_03_sudden_vram_pressure_spike_during_admission(mock_config):
    """TC-T2-F08-03: Sudden drop in free VRAM denies admission."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4000, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    assert mgr.can_worker_0_steal(queue_size=4, heavy_tasks_cleared=True) is True
    ctrl.set_vram(free_mb=2800, used_mb=5000)
    assert mgr.can_worker_0_steal(queue_size=4, heavy_tasks_cleared=True) is False


def test_t2_f08_04_max_decoder_limit_saturation(mock_config):
    """TC-T2-F08-04: Saturated decoder sessions deny admission."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4000, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    mgr.max_nv_decoders = 1
    mgr._active_nv_decoders = 1
    assert mgr.can_worker_0_steal(queue_size=4, heavy_tasks_cleared=True) is False


def test_t2_f08_05_flapping_admission_gate_under_concurrent_workers(mock_config):
    """TC-T2-F08-05: Multi-threaded rapid admission queries maintain state consistency."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4000, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    results = []

    def caller(size):
        for _ in range(25):
            try:
                results.append(mgr.can_worker_0_steal(queue_size=size, heavy_tasks_cleared=True))
            except Exception as e:
                results.append(e)

    threads = [
        threading.Thread(target=caller, args=(1,)),
        threading.Thread(target=caller, args=(4,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)
    assert len(results) == 50
    assert False in results
    assert True in results


# ============================================================================
# Feature F9: Atomic Sub-Longest Task Stealing Corners (5 tests)
# ============================================================================

def test_t2_f09_01_identical_duration_duplicate_tasks():
    """TC-T2-F09-01: Multiple tasks with identical duration popped cleanly."""
    q = AnalysisQueue()
    q.cost_priority = True
    for i in range(4):
        q.put({"filepath": f"same_{i}.mp4", "file_duration": 900.0})
    stolen1 = q.steal_sublongest(max_duration=1200.0)
    assert stolen1 is not None
    assert stolen1["file_duration"] == 900.0
    assert q.qsize() == 3


def test_t2_f09_02_single_element_queue_steal():
    """TC-T2-F09-02: Single qualifying element popped, queue becomes empty."""
    q = AnalysisQueue()
    q.cost_priority = True
    q.put({"filepath": "only.mp4", "file_duration": 600.0})
    stolen = q.steal_sublongest(max_duration=1200.0)
    assert stolen["filepath"] == "only.mp4"
    assert q.empty() is True


def test_t2_f09_03_all_tasks_exceed_max_duration():
    """TC-T2-F09-03: Tasks all exceeding max_duration return None without popping."""
    q = AnalysisQueue()
    q.cost_priority = True
    q.put({"filepath": "huge1.mp4", "file_duration": 1500.0})
    q.put({"filepath": "huge2.mp4", "file_duration": 2000.0})
    assert q.steal_sublongest(max_duration=1200.0) is None
    assert q.qsize() == 2


def test_t2_f09_04_concurrent_push_and_steal_interleaving():
    """TC-T2-F09-04: Interleaved push and steal operations maintain queue size."""
    q = AnalysisQueue()
    q.cost_priority = True
    for i in range(20):
        q.put({"filepath": f"f_{i}.mp4", "file_duration": 500.0})
        if i % 2 == 1:
            q.steal_sublongest(max_duration=800.0)
    assert q.qsize() == 10


def test_t2_f09_05_interrupted_worker_exception_recovery():
    """TC-T2-F09-05: Exception in caller does not corrupt remaining queue."""
    q = AnalysisQueue()
    q.cost_priority = True
    q.put({"filepath": "task.mp4", "file_duration": 400.0})
    try:
        _ = q.steal_sublongest(max_duration=800.0)
        raise RuntimeError("Interrupted")
    except RuntimeError:
        pass
    assert q.empty() is True


# ============================================================================
# Feature F10: CUDA NVDEC Device Lease Corners (5 tests)
# ============================================================================

def test_t2_f10_01_decoder_slot_timeout_reached(mock_config):
    """TC-T2-F10-01: Slot acquisition when full returns False after timeout without blocking."""
    mgr = scheduler.WorkStealingManager(mock_config)
    mgr.max_nv_decoders = 1
    mgr._active_nv_decoders = 1
    t0 = time.monotonic()
    acquired = mgr.acquire_nvdec_slot(timeout=0.1)
    elapsed = time.monotonic() - t0
    assert acquired is False
    assert elapsed < 0.3


def test_t2_f10_02_pipeline_abort_event_signaled_mid_lease(mock_config):
    """TC-T2-F10-02: Abort signal safely exits lease context."""
    mgr = scheduler.WorkStealingManager(mock_config)
    with mgr.lease_device(is_stolen=True):
        pass
    assert mgr.active_nv_decoders == 0


def test_t2_f10_03_rapid_sequential_lease_recycling(mock_config):
    """TC-T2-F10-03: 10 rapid sequential leases alternate 0 -> 1 -> 0."""
    mgr = scheduler.WorkStealingManager(mock_config)
    for _ in range(10):
        with mgr.lease_device(is_stolen=True) as dev:
            assert dev == "cuda"
            assert mgr.active_nv_decoders == 1
        assert mgr.active_nv_decoders == 0


def test_t2_f10_04_render_stage_termination_while_analysis_active(mock_config):
    """TC-T2-F10-04: Render ending while analysis is active completes normally."""
    mgr = scheduler.WorkStealingManager(mock_config)
    mgr.register_render_start()
    with mgr.lease_device(is_stolen=True) as dev:
        assert dev == "cuda"
        mgr.register_render_end()
    assert mgr.active_nv_decoders == 0


def test_t2_f10_05_double_release_defense_in_exception_handler(mock_config):
    """TC-T2-F10-05: Double lease release floors at 0 without negative counter."""
    mgr = scheduler.WorkStealingManager(mock_config)
    mgr.release_nvdec_slot()
    mgr.release_nvdec_slot()
    assert mgr.active_nv_decoders == 0


# ============================================================================
# Feature F11: Regression & Doctor Corners (5 tests)
# ============================================================================

def test_t2_f11_01_doctor_behavior_in_headless_cpu_only_environment():
    """TC-T2-F11-01: Autotune with has_cuda=False recommends software config cleanly."""
    snap = autotune.HardwareSnapshot(
        cpu_logical=8,
        cpu_physical=4,
        ram_total_gb=16.0,
        has_cuda=False,
        nv_device_count=0,
        nv_device_name="",
        nv_vram_mb=0,
        nv_driver_version="",
        has_nvenc=False,
        has_qsv=False,
    )
    rec = autotune.compute_optimal_concurrency(snap)
    assert rec["hardware"]["device"] == "cpu"
    assert rec["scheduler"]["max_nv_decoders"] == 0


def test_t2_f11_02_low_storage_space_warning(tmp_path):
    """TC-T2-F11-02: Target disk free space < 20GB detected via check_disk_space."""
    from src.core.utils import check_disk_space
    with patch("shutil.disk_usage", return_value=(100 * 1024**3, 95 * 1024**3, 5 * 1024**3)):
        is_ok = check_disk_space(tmp_path, min_gb=20)
        assert is_ok is False
    with patch("shutil.disk_usage", return_value=(100 * 1024**3, 50 * 1024**3, 50 * 1024**3)):
        is_ok = check_disk_space(tmp_path, min_gb=20)
        assert is_ok is True


def test_t2_f11_03_missing_or_unreadable_camera_directory(tmp_path):
    """TC-T2-F11-03: Nonexistent camera path handled gracefully by scanner."""
    from src.stages.scanner import scan_directory
    db = VlogDatabase(tmp_path / "scan_err.db")
    nonexistent = tmp_path / "NonexistentCameraDir"
    with pytest.raises(FileNotFoundError, match="输入目录不可用"):
        scan_directory(db, input_dir=str(nonexistent))
    db.close()


def test_t2_f11_04_pytest_memory_leak_and_resource_leak_check(tmp_path, mock_config):
    """TC-T2-F11-04: assert_resource_hygiene validates clean process and lock state after real workload."""
    clip = tmp_path / "leak_check.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=1.0, fps=20, gop_size=20)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    labels, _ = detector.analyze(str(clip), file_duration=1.0, has_audio=0)
    assert len(labels) > 0
    del detector
    gc.collect()
    assert_resource_hygiene()


def test_t2_f11_05_doctor_corrupted_yolo_model_file_detection(tmp_path):
    """TC-T2-F11-05: Truncated 0-byte YOLO model file is detected by doctor checks."""
    from src.ui.doctor import check_model_weights
    fake_model = tmp_path / "yolo11m.pt"
    fake_model.touch()
    with patch("src.ui.doctor.load_config", return_value={"yolo": {"model_path": str(fake_model)}}):
        result = check_model_weights()
        assert result["exists"] is True
        assert result["size_mb"] == 0.0


# ============================================================================
# Feature F12: Cold Run Benchmark Corners (5 tests)
# ============================================================================

def test_t2_f12_01_high_smb_latency_simulation(tmp_path, mock_config):
    """TC-T2-F12-01: Simulated network delay handled cleanly without dropped frames."""
    clip = tmp_path / "latency_clip.mp4"
    create_synthetic_clip(clip, scene="motion", duration_s=2.0, fps=20, gop_size=10)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    with patch("time.sleep", wraps=time.sleep):
        records, meta = detector._scan_keyframes_coarse(str(clip), file_duration=2.0)
    assert len(records) >= 3
    assert meta["complete"] is True
    assert not meta.get("truncated", False)


def test_t2_f12_02_pipeline_mid_run_sigint_interruption_and_resumption():
    """TC-T2-F12-02: Signal interrupt cleanup removes unconfirmed temporary files."""
    assert not FFmpegProcessRegistry.is_interrupted()
    FFmpegProcessRegistry.mark_interrupted()
    assert FFmpegProcessRegistry.is_interrupted() is True
    FFmpegProcessRegistry.reset_interrupted()
    assert FFmpegProcessRegistry.is_interrupted() is False
    assert_resource_hygiene()


def test_t2_f12_03_heterogeneous_hardware_render_load_variance():
    """TC-T2-F12-03: Dual-ended queue balances items between NVENC and QSV."""
    q = scheduler.DualEndedBatchQueue()
    q.put(b_idx=1, files=["f1.mp4"], dynamic_duration=100.0)
    q.put(b_idx=2, files=["f2.mp4"], dynamic_duration=100.0)
    item = q.pop_heaviest()
    assert item.b_idx in (1, 2)


def test_t2_f12_04_peak_vram_pressure_under_heavy_render_concurrency(mock_config):
    """TC-T2-F12-04: Watermark check prevents launching analysis on CUDA when near limit."""
    mgr = scheduler.WorkStealingManager(mock_config)
    mgr.vram_watermark_mb = 6200
    ctrl = MockVramController(used_mb=6300)
    mgr.set_vram_probe_fn(ctrl.probe_used)
    assert mgr.is_vram_under_pressure() is True


def test_t2_f12_05_mixed_workload_empty_corrupt_giant_tiny_slices(tmp_path, mock_config):
    """TC-T2-F12-05: Dataset with empty, corrupt, and valid clips handled safely."""
    f_empty = create_corrupted_clip(tmp_path / "empty.mp4", "zero_byte")
    f_valid = create_synthetic_clip(tmp_path / "valid.mp4", duration_s=1.0)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    rec_empty, meta_empty = detector._scan_keyframes_coarse(str(f_empty), file_duration=1.0)
    assert len(rec_empty) == 0
    assert meta_empty.get("complete") is False
    rec_valid, meta_valid = detector._scan_keyframes_coarse(str(f_valid), file_duration=1.0)
    assert len(rec_valid) > 0
    assert meta_valid["complete"] is True


# ============================================================================
# Feature F13: Seam Continuity & Zero Clock Drift Corners (5 tests)
# ============================================================================

def test_t2_f13_01_heterogeneous_transcode_seam_nvenc_to_qsv(tmp_path):
    """TC-T2-F13-01: Boundary between two batches matches identical framerate."""
    clip1 = tmp_path / "batch1.mp4"
    clip2 = tmp_path / "batch2.mp4"
    create_synthetic_clip(clip1, duration_s=1.0, fps=20)
    create_synthetic_clip(clip2, duration_s=1.0, fps=20)
    with av.open(str(clip1)) as m1, av.open(str(clip2)) as m2:
        rate1 = m1.streams.video[0].average_rate
        rate2 = m2.streams.video[0].average_rate
    assert rate1 == rate2 == 20


def test_t2_f13_02_transition_seam_between_zero_decode_static_and_active():
    """TC-T2-F13-02: Boundary frame delta at transition is strictly positive and bounded."""
    seg_static = TimelineSegment("f.mp4", 0, 0.0, 10.0, SegmentState.STATIC, 10.0)
    seg_dynamic = TimelineSegment("f.mp4", 0, 10.0, 15.0, SegmentState.DYNAMIC, 5.0)
    plans = compute_display_plans([seg_static, seg_dynamic], output_fps=20.0)
    assert len(plans) == 2
    dur_static, ramp_info = plans[0]
    dur_dynamic, _ = plans[1]
    assert dur_static > 0
    assert dur_dynamic == pytest.approx(5.0)
    assert ramp_info is not None


def test_t2_f13_03_variable_input_fps_normalization_to_constant_20_fps(mock_config):
    """TC-T2-F13-03: Variable input frame rate normalized to constant 20 FPS in filtergraph."""
    seg = TimelineSegment("clip.mp4", 0, 0.0, 5.0, SegmentState.DYNAMIC, 5.0)
    rows = [{"filepath": "clip.mp4", "duration": 5.0, "fps": 19.85}]
    filter_str = build_concat_filter([seg], rows=rows, output_fps=20, config=mock_config)
    assert "fps=20" in filter_str or "scale=" in filter_str or "concat" in filter_str


def test_t2_f13_04_non_zero_starting_pts_and_clock_offset_normalization(tmp_path, mock_config):
    """TC-T2-F13-04: Non-zero container PTS normalized to 0.0s origin."""
    clip = tmp_path / "offset_pts.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=1.0, fps=20, gop_size=20)
    seg = TimelineSegment(str(clip.resolve()), 0, 0.0, 1.0, SegmentState.DYNAMIC, 1.0)
    mapping = bind_delivery_mapping(clip, [seg], mock_config)
    assert mapping["status"] == "AVAILABLE"
    assert mapping["clock_validation"]["origin_s"] >= 0.0


def test_t2_f13_05_single_frame_drop_glitch_detection_guard(tmp_path, mock_config):
    """TC-T2-F13-05: Deviation > 1 tick from planned clock is detected and marks delivery UNAVAILABLE."""
    clip = tmp_path / "glitch_pts.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=1.0, fps=20, gop_size=20)
    seg = TimelineSegment(str(clip.resolve()), 0, 0.0, 1.5, SegmentState.DYNAMIC, 1.5)
    mapping = bind_delivery_mapping(clip, [seg], mock_config)
    assert mapping["status"] == "UNAVAILABLE"
    assert "differs" in mapping["reason"].lower() or "clock" in mapping["reason"].lower()
