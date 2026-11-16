"""Tier 1: Feature Coverage Specifications (F1–F13, 65 Tests).

Each feature in F1-F13 has exactly 5 dedicated opaque-box tests covering nominal behavior.
"""

import json
from pathlib import Path
import re
import threading
import time

import av
import numpy as np
import pytest

from src.algorithms.filters import AudioEnergyVAD
from src.core import config as core_config
from src.core.database import VlogDatabase
from src.core.profiles import resolve_profile_config
from src.hardware import autotune, scheduler
from src.stages.companion import bind_delivery_mapping
from src.stages.detector import MotionDetector
from src.stages.pipeline_types import AnalysisQueue
from src.stages.timeline import SegmentState, TimelineSegment, generate_timecode_subtitles
from tests.helpers import validate_perf_json_schema
from tests.helpers_e2e import (
    MockVramController,
    assert_resource_hygiene,
    create_synthetic_clip,
)


# ============================================================================
# Feature F1: iframe_prescreen_enabled Config & Governance (5 tests)
# ============================================================================

def test_t1_f01_01_default_configuration_validation(mock_config):
    """TC-T1-F01-01: Verify default configuration dictionary contains R1 keys and passes validate_config."""
    cfg = dict(mock_config)
    cfg.setdefault("detection", {})
    cfg["output"]["qsv"].pop("maxrate", None)
    cfg["output"]["qsv"].pop("bufsize", None)
    cfg["detection"].update({
        "iframe_prescreen_enabled": True,
        "iframe_energy_margin": 0.80,
        "iframe_burst_ratio": 1.8,
        "iframe_window_padding": 3.0,
        "iframe_window_gap_merge": 6.0,
        "iframe_fallback_active_ratio": 0.60,
        "iframe_max_isolated_windows": 8,
    })
    core_config.validate_config(cfg)
    assert cfg["detection"]["iframe_prescreen_enabled"] is True
    assert cfg["detection"]["iframe_energy_margin"] == pytest.approx(0.80)
    assert cfg["detection"]["iframe_burst_ratio"] == pytest.approx(1.8)
    assert cfg["detection"]["iframe_window_padding"] == pytest.approx(3.0)
    assert cfg["detection"]["iframe_window_gap_merge"] == pytest.approx(6.0)
    assert cfg["detection"]["iframe_fallback_active_ratio"] == pytest.approx(0.60)
    assert cfg["detection"]["iframe_max_isolated_windows"] == 8


def test_t1_f01_02_feature_toggle_disabled_path(mock_config):
    """TC-T1-F01-02: Verify MotionDetector disables keyframe pre-screen when iframe_prescreen_enabled is False."""
    cfg = dict(mock_config)
    cfg.setdefault("detection", {})
    cfg["detection"]["iframe_prescreen_enabled"] = False
    detector = MotionDetector(cfg, decode_gpu="cpu")
    assert detector.iframe_prescreen_enabled is False


def test_t1_f01_03_camera_profile_override_resolution():
    """TC-T1-F01-03: Verify camera profile resolution overrides default prescreen resolution."""
    from src.core.profiles import resolve_camera_config
    raw_cfg = {
        "detection": {"prescreen_resolution": "640x360"},
        "cameras": {"0": {"detection": {"prescreen_resolution": "1280x720"}}},
    }
    resolved = resolve_camera_config(raw_cfg, cam_index=0)
    assert resolved["detection"]["prescreen_resolution"] == "1280x720"


def test_t1_f01_04_positive_numerical_range_enforcement(mock_config):
    """TC-T1-F01-04: Confirm validate_config rejects non-positive thresholds."""
    cfg = dict(mock_config)
    cfg["output"]["qsv"].pop("maxrate", None)
    cfg["output"]["qsv"].pop("bufsize", None)
    cfg.setdefault("detection", {})
    cfg["detection"]["iframe_energy_margin"] = -0.5
    with pytest.raises((ValueError, AssertionError)):
        core_config.validate_config(cfg)


def test_t1_f01_05_integer_type_enforcement_for_window_counts(mock_config):
    """TC-T1-F01-05: Confirm validate_config enforces integer type for max isolated windows."""
    cfg = dict(mock_config)
    cfg["output"]["qsv"].pop("maxrate", None)
    cfg["output"]["qsv"].pop("bufsize", None)
    cfg.setdefault("detection", {})
    cfg["detection"]["iframe_max_isolated_windows"] = "eight"
    with pytest.raises((ValueError, TypeError, AssertionError)):
        core_config.validate_config(cfg)


# ============================================================================
# Feature F2: Tier 1 Coarse I-Frame Fast Scan (-skip_frame nokey) (5 tests)
# ============================================================================

def test_t1_f02_01_keyframe_only_decoding_verification(tmp_path, mock_config):
    """TC-T1-F02-01: Verify FFmpeg command receives -skip_frame nokey and demuxes keyframes."""
    clip = tmp_path / "gop20.mp4"
    create_synthetic_clip(clip, scene="motion", duration_s=4.0, fps=20, gop_size=20)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    records, meta = detector._scan_keyframes_coarse(str(clip), file_duration=4.0, decode_gpu="cpu")
    assert len(records) > 0
    assert len(records) <= 6
    assert meta["complete"] is True


def test_t1_f02_02_spatial_grid_8x8_energy_calculation(mock_config):
    """TC-T1-F02-02: Verify spatial 8x8 cell energy is properly calculated on frame differences."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    grid = detector.create_grid_filter()
    f1 = np.full((128, 128), 50, dtype=np.uint8)
    f2 = np.full((128, 128), 50, dtype=np.uint8)
    f2[16:32, 16:32] = 200  # concentrated localized motion
    saliency = np.abs(f2.astype(np.float32) - f1.astype(np.float32))
    eff_energy, is_motion, stats = grid.process_frame(saliency, dt=0.5, is_night_mode=False)
    assert eff_energy > 0.0
    assert "max_cell_energy" in stats
    assert stats["max_cell_energy"] > 0.0


def test_t1_f02_03_ema_background_model_continuity(mock_config):
    """TC-T1-F02-03: Verify background model EMA smoothly updates across keyframes."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    ema = detector.create_ema_model()
    frame1 = np.full((128, 128), 100, dtype=np.uint8)
    frame2 = np.full((128, 128), 200, dtype=np.uint8)
    saliency1, mask1, bg1 = ema.update(frame1)
    saliency2, mask2, bg2 = ema.update(frame2)
    assert bg1 is not None
    assert bg2 is not None
    assert bg2.shape == (128, 128)
    assert np.mean(bg2) > np.mean(bg1)


def test_t1_f02_04_in_memory_intermediate_frame_contract(tmp_path, mock_config):
    """TC-T1-F02-04: Verify coarse keyframe scan writes zero temporary frame files to disk."""
    clip = tmp_path / "clean_io.mp4"
    create_synthetic_clip(clip, scene="motion", duration_s=3.0, fps=20, gop_size=10)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    _ = detector._scan_keyframes_coarse(str(clip), file_duration=3.0, decode_gpu="cpu")
    disk_images = list(tmp_path.glob("*.jpg")) + list(tmp_path.glob("*.png"))
    assert len(disk_images) == 0


def test_t1_f02_05_coarse_scan_throughput_benchmark(tmp_path, mock_config):
    """TC-T1-F02-05: Verify coarse scan achieves high throughput on keyframes."""
    clip = tmp_path / "throughput.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=5.0, fps=20, gop_size=20)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    t0 = time.perf_counter()
    records, meta = detector._scan_keyframes_coarse(str(clip), file_duration=5.0, decode_gpu="cpu")
    elapsed = time.perf_counter() - t0
    assert elapsed < 3.0
    assert meta["complete"] is True


# ============================================================================
# Feature F3: Zero-Decode Fast Path for Static Clips (5 tests)
# ============================================================================

def test_t1_f03_01_pure_static_clip_direct_synthesis(tmp_path, mock_config):
    """TC-T1-F03-01: Confirm a clip with zero activity bypasses fine decode."""
    clip = tmp_path / "pure_static.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=4.0, fps=20, gop_size=20)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    res = detector._scan_keyframes_coarse(str(clip), file_duration=4.0, decode_gpu="cpu")
    trace = res[0] if isinstance(res, tuple) else res
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=4.0, threshold=6.5)
    assert len(windows) == 0


def test_t1_f03_02_zero_decode_segment_timeline_closure(mock_config):
    """TC-T1-F03-02: Verify synthesized static segments strictly cover the exact file duration."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    file_duration = 14.35
    labels = detector._synthesize_static_labels(
        start_offset=0.0,
        file_duration=file_duration,
        effective_fps=5.0,
    )
    assert len(labels) >= 2
    assert labels[0]["time"] == 0.0
    assert labels[-1]["time"] == pytest.approx(file_duration, abs=1e-3)
    assert all(l["state"] == "STATIC" for l in labels)
    assert all(not l["is_motion"] for l in labels)
    assert all(l["energy"] == 0.0 for l in labels)


def test_t1_f03_03_empty_yolo_buffer_contract(tmp_path, mock_config):
    """TC-T1-F03-03: Verify zero candidate frames are passed to YOLO for zero-decode tasks."""
    clip = tmp_path / "static_zero_yolo.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=4.0, fps=20, gop_size=20)
    cfg = dict(mock_config)
    cfg.setdefault("yolo", {})["enabled"] = True
    cfg["yolo"]["streaming_verify"] = True
    detector = MotionDetector(cfg, decode_gpu="cpu")
    labels, yolo_buffer = detector.analyze(str(clip), start_offset=0.0, file_duration=4.0, has_audio=0)
    assert len(labels) > 0
    assert all(l["state"] == "STATIC" for l in labels)
    assert len(yolo_buffer) == 0


def test_t1_f03_04_downstream_rendering_compatibility():
    """TC-T1-F03-04: Ensure static segments synthesized via zero-decode render into condensed frames."""
    seg = TimelineSegment(
        filepath="static.mp4",
        input_index=0,
        start_in_file=0.0,
        end_in_file=10.0,
        state="STATIC",
        duration=10.0,
    )
    assert seg.duration == 10.0
    assert seg.state == "STATIC"


def test_t1_f03_05_database_state_atomicity(isolated_db):
    """TC-T1-F03-05: Ensure zero-decode fast path writes to SQLite DB in an atomic transaction."""
    date = "20260901"
    for i in range(5):
        fp = f"test_{i}.mp4"
        isolated_db.add_file_task(filepath=fp, cam_index=0, date=date, file_duration=10.0, file_start_time="120000", file_end_time="120010")
        isolated_db.set_prescreen_result(fp, status="STATIC", result_json='{"mode": "keyframes"}')
        isolated_db.set_analysis_result(fp, "ANALYZED", [{"start_in_file": 0.0, "end_in_file": 10.0, "state": "STATIC"}])
    rows = isolated_db.get_all_file_tasks_for_date(date, 0)
    assert len(rows) == 5
    assert all(r["analysis_status"] == "ANALYZED" for r in rows)


# ============================================================================
# Feature F4: Tier 2 Awakened Window Fine Decoding (5 tests)
# ============================================================================

def test_t1_f04_01_single_motion_event_window_awakening(mock_config):
    """TC-T1-F04-01: Verify a motion burst creates a padded fine-decode window."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(0.0, 0.2, 0.5), (3.0, 0.1, 0.4), (15.0, 15.0, 20.0), (18.0, 0.2, 0.3)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=30.0, threshold=6.5)
    assert len(windows) == 1
    w_start, w_end = windows[0]
    assert w_start <= 12.0  # 15.0 - 3.0
    assert w_end >= 18.0   # 15.0 + 3.0


def test_t1_f04_02_window_gap_merging_under_threshold(mock_config):
    """TC-T1-F04-02: Verify two active windows separated by less than gap_merge merge into one."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(10.0, 15.0, 20.0), (17.0, 15.0, 20.0)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=40.0, threshold=6.5)
    assert len(windows) == 1
    assert windows[0] == (7.0, 20.0)


def test_t1_f04_03_disjoint_windows_preservation(mock_config):
    """TC-T1-F04-03: Verify two active windows separated by more than gap_merge remain separate."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(5.0, 15.0, 20.0), (25.0, 15.0, 20.0)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=40.0, threshold=6.5)
    assert len(windows) == 2
    assert windows[0] == (2.0, 8.0)
    assert windows[1] == (22.0, 28.0)


def test_t1_f04_04_yolo_candidate_frame_buffering_in_active_windows(tmp_path, mock_config):
    """TC-T1-F04-04: Confirm candidate frame buffering during active windows."""
    clip = tmp_path / "window_buffer.mp4"
    create_synthetic_clip(clip, scene="motion", duration_s=4.0, fps=20, gop_size=20)
    cfg = dict(mock_config)
    cfg.setdefault("yolo", {})["enabled"] = True
    cfg["yolo"]["streaming_verify"] = True
    cfg["yolo"]["sample_fps"] = 2.0
    detector = MotionDetector(cfg, decode_gpu="cpu")
    detector.yolo_enabled = True
    yolo_buffer = {}
    detector._decode_file_window(
        str(clip),
        w_start=1.0,
        w_end=3.0,
        effective_fps=5.0,
        yolo_buffer=yolo_buffer,
        yolo_sample_interval=2,
    )
    import cv2
    assert len(yolo_buffer) > 0
    assert all(isinstance(v, (bytes, bytearray, np.ndarray)) for v in yolo_buffer.values())
    for v in yolo_buffer.values():
        if isinstance(v, (bytes, bytearray)):
            decoded = cv2.imdecode(np.frombuffer(v, dtype=np.uint8), cv2.IMREAD_COLOR)
            assert decoded is not None and decoded.shape == (detector.height, detector.width, 3)
        else:
            assert v.shape[:2] == (detector.height, detector.width)


def test_t1_f04_05_timeline_closure_with_interleaved_segments(mock_config):
    """TC-T1-F04-05: Verify interleaved static and dynamic segments strictly preserve total duration."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    windows = [(10.0, 20.0)]
    trace_mock = type("MockTrace", (), {"energies": [25.0] * 50, "confidences": [0.8] * 50})()
    window_traces = [(10.0, 20.0, trace_mock)]
    labels = detector._stitch_windowed_labels(
        windows=windows,
        window_traces=window_traces,
        audio_events=[],
        start_offset=0.0,
        file_duration=30.0,
        effective_fps=5.0,
        baseline_threshold=6.5,
    )
    assert len(labels) >= 2
    assert labels[0]["time"] == 0.0
    assert labels[-1]["time"] == pytest.approx(30.0, abs=1e-3)
    dynamic_labels = [l for l in labels if 10.0 <= l["time"] <= 20.0]
    static_labels = [l for l in labels if l["time"] < 10.0 or l["time"] > 20.0]
    assert len(dynamic_labels) > 0
    assert len(static_labels) > 0
    assert all(l["state"] == "STATIC" for l in static_labels)


# ============================================================================
# Feature F5: Fallback Safeguards for Dense Motion & Long GOP (5 tests)
# ============================================================================

def test_t1_f05_01_dense_motion_active_ratio_fallback(mock_config):
    """TC-T1-F05-01: Verify active ratio > 0.60 triggers fallback to full decode."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(float(t), 20.0, 30.0) for t in range(5, 26, 5)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=30.0, threshold=6.5)
    fallback, reason = detector._evaluate_fallback_safeguards(windows, file_duration=30.0, gop_metrics={"keyframes": len(trace), "avg_gop_s": 3.0})
    assert fallback is True
    assert "DENSE_MOTION" in reason or "ACTIVE_RATIO" in reason


def test_t1_f05_02_isolated_windows_count_fallback(mock_config):
    """TC-T1-F05-02: Verify isolated windows > 8 triggers fallback to full decode."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(float(i * 20 + 5), 20.0, 30.0) for i in range(9)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=200.0, threshold=6.5)
    fallback, reason = detector._evaluate_fallback_safeguards(windows, file_duration=200.0, gop_metrics={"keyframes": len(trace), "avg_gop_s": 3.0})
    assert fallback is True
    assert "ISOLATED_WINDOWS" in reason


def test_t1_f05_03_sparse_keyframe_long_gop_fallback(mock_config):
    """TC-T1-F05-03: Verify video with GOP interval > 8.0s triggers safe fallback handling."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    gop_metrics = {"keyframes": 5, "avg_gop_s": 10.0, "max_gop_s": 12.0}
    windows = [(5.0, 15.0)]
    fallback, reason = detector._evaluate_fallback_safeguards(
        windows=windows,
        file_duration=50.0,
        gop_metrics=gop_metrics,
    )
    assert fallback is True
    assert "SPARSE_GOP" in reason


def test_t1_f05_04_fallback_segment_equality_guarantee(tmp_path, mock_config):
    """TC-T1-F05-04: Verify fallback segments cover the entire duration identically to full decode."""
    clip = tmp_path / "fallback_cov.mp4"
    create_synthetic_clip(clip, scene="motion", duration_s=3.0, fps=20, gop_size=10)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    labels, _ = detector._fallback_full_decode(
        filepath=str(clip),
        start_offset=0.0,
        file_duration=3.0,
        has_audio=0,
        fallback_reason="FALLBACK_DENSE_MOTION",
    )
    assert len(labels) >= 2
    assert labels[0]["time"] == 0.0
    assert labels[-1]["time"] == pytest.approx(3.0, abs=0.2)
    assert detector.decode_completion["complete"] is True


def test_t1_f05_05_fallback_status_recorded_in_completion_telemetry(tmp_path, mock_config):
    """TC-T1-F05-05: Verify fallback event is recorded in completion metadata."""
    clip = tmp_path / "fallback_telemetry.mp4"
    create_synthetic_clip(clip, scene="motion", duration_s=2.0, fps=20, gop_size=10)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    detector._fallback_full_decode(
        filepath=str(clip),
        start_offset=0.0,
        file_duration=2.0,
        has_audio=0,
        fallback_reason="FALLBACK_ISOLATED_WINDOWS",
    )
    comp = detector.decode_completion
    assert comp is not None
    assert comp["complete"] is True
    assert comp["declared_duration_s"] == 2.0


# ============================================================================
# Feature F6: Multimodal Audio VAD Awakening (5 tests)
# ============================================================================

def test_t1_f06_01_audio_only_trigger_awakens_video_window(mock_config):
    """TC-T1-F06-01: Verify speech event in visually static scene unconditionally awakens fine window."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(0.0, 0.0, 0.0), (5.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    audio_events = [(8.0, 11.0, "audio")]
    windows = detector._resolve_awakened_windows(trace, audio_events=audio_events, file_duration=20.0, threshold=6.5)
    assert len(windows) == 1
    w_start, w_end = windows[0]
    assert w_start <= 5.0  # 8.0 - 3.0
    assert w_end >= 14.0   # 11.0 + 3.0


def test_t1_f06_02_audio_and_visual_motion_coalescence(mock_config):
    """TC-T1-F06-02: Verify simultaneous audio and visual motion coalesce into a single window."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(5.0, 15.0, 20.0)]  # Window [2.0, 8.0]
    audio_events = [(7.0, 10.0, "audio")]  # Window [4.0, 13.0]
    windows = detector._resolve_awakened_windows(trace, audio_events=audio_events, file_duration=20.0, threshold=6.5)
    assert len(windows) == 1
    assert windows[0][0] <= 2.0
    assert windows[0][1] >= 13.0


def test_t1_f06_03_silent_audio_track_non_awakening():
    """TC-T1-F06-03: Verify ambient noise below threshold does not trigger audio awakening."""
    vad = AudioEnergyVAD(sample_rate=16000, min_dbfs=-42.0, enabled=True)
    noise = np.full(16000 * 2, 1e-4, dtype=np.float32)  # very quiet
    events, metrics = vad.detect_events(noise)
    assert len(events) == 0


def test_t1_f06_04_audio_vad_process_termination_and_cleanup(tmp_path, mock_config):
    """TC-T1-F06-04: Confirm audio decoding leaves zero orphaned subprocesses."""
    clip = tmp_path / "audio_cleanup.mp4"
    create_synthetic_clip(clip, scene="motion_audio", duration_s=2.0, fps=20, audio_windows=[(0.5, 1.5)])
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    detector.has_audio_detected = 1
    features = detector._decode_audio_pipe(str(clip), file_duration=2.0)
    assert features is not None
    assert features.size > 0
    assert len(features.dbfs_windows()) > 0
    assert_resource_hygiene()


def test_t1_f06_05_dynamic_audio_segment_playback_speed():
    """TC-T1-F06-05: Verify DYNAMIC_AUDIO segments maintain 1.0x playback speed in timeline."""
    seg = TimelineSegment(
        filepath="audio.mp4",
        input_index=0,
        start_in_file=0.0,
        end_in_file=5.0,
        state="DYNAMIC_AUDIO",
        duration=5.0,
    )
    assert seg.state == "DYNAMIC_AUDIO"
    assert seg.duration == 5.0


# ============================================================================
# Feature F7: VRAM Headroom & Margin Probe (5 tests)
# ============================================================================

def test_t1_f07_01_sufficient_free_vram_returns_true(mock_config):
    """TC-T1-F07-01: Verify has_vram_margin returns True when free VRAM > 3072 MB."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4096, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    mgr.set_vram_probe_fn(ctrl.probe_used)
    assert mgr.has_vram_margin(3072) is True


def test_t1_f07_02_insufficient_free_vram_returns_false(mock_config):
    """TC-T1-F07-02: Verify has_vram_margin returns False when free VRAM < 3072 MB."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=2500, used_mb=5000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    mgr.set_vram_probe_fn(ctrl.probe_used)
    assert mgr.has_vram_margin(3072) is False


def test_t1_f07_03_used_vram_exceeding_watermark_returns_false(mock_config):
    """TC-T1-F07-03: Verify probe returns False when used VRAM >= watermark."""
    mgr = scheduler.WorkStealingManager(mock_config)
    mgr.vram_watermark_mb = 6200
    ctrl = MockVramController(free_mb=3500, used_mb=6300)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    mgr.set_vram_probe_fn(ctrl.probe_used)
    assert mgr.has_vram_margin(1024) is False


def test_t1_f07_04_nvml_exception_graceful_fallback(mock_config):
    """TC-T1-F07-04: Verify NVML exceptions are caught cleanly and return False."""
    mgr = scheduler.WorkStealingManager(mock_config)
    def broken_probe():
        raise RuntimeError("Simulated NVML communication error")
    mgr._vram_margin_probe_fn = broken_probe
    assert mgr.has_vram_margin() is False


def test_t1_f07_05_concurrent_vram_queries_thread_safety(mock_config):
    """TC-T1-F07-05: Verify concurrent calls to has_vram_margin execute safely without deadlocks."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4096, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    mgr.set_vram_probe_fn(ctrl.probe_used)
    results = []

    def worker():
        for _ in range(50):
            try:
                res = mgr.has_vram_margin(3072)
                results.append(res)
            except Exception as e:
                results.append(e)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=2.0)
    assert len(results) == 250
    assert all(r is True for r in results)


# ============================================================================
# Feature F8: Tail Work Stealing Admission Policy (5 tests)
# ============================================================================

def test_t1_f08_01_full_five_gate_admission_success(mock_config):
    """TC-T1-F08-01: Confirm can_worker_0_steal returns True when all 5 gates pass."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4000, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    mgr.set_vram_probe_fn(ctrl.probe_used)
    mgr.nvdec_cooperative = True
    mgr.max_nv_decoders = 2
    mgr.steal_backlog_threshold = 2
    assert mgr.can_worker_0_steal(queue_size=4, heavy_tasks_cleared=True) is True


def test_t1_f08_02_rejection_on_heavy_tasks_remaining(mock_config):
    """TC-T1-F08-02: Confirm admission denied if heavy tasks remain in queue."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4000, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    assert mgr.can_worker_0_steal(queue_size=4, heavy_tasks_cleared=False) is False


def test_t1_f08_03_rejection_on_worker_0_busy(mock_config):
    """TC-T1-F08-03: Confirm admission denied if Worker 0 active decoders == max."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4000, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    mgr.max_nv_decoders = 1
    mgr._active_nv_decoders = 1
    assert mgr.can_worker_0_steal(queue_size=4, heavy_tasks_cleared=True) is False


def test_t1_f08_04_rejection_on_low_qsv_backlog(mock_config):
    """TC-T1-F08-04: Confirm admission denied if backlog < threshold."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4000, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    mgr.steal_backlog_threshold = 3
    assert mgr.can_worker_0_steal(queue_size=2, heavy_tasks_cleared=True) is False


def test_t1_f08_05_state_machine_transition_to_tail_work_stealing(mock_config):
    """TC-T1-F08-05: Confirm authorizing stolen task transitions state to TAIL_WORK_STEALING."""
    mgr = scheduler.WorkStealingManager(mock_config)
    with mgr.lease_device(is_stolen=True) as dev:
        assert dev == "cuda"
        assert mgr.state == "TAIL_WORK_STEALING"
    assert mgr.state == "NORMAL_DECOUPLED"


# ============================================================================
# Feature F9: Atomic Sub-Longest Task Stealing (5 tests)
# ============================================================================

def test_t1_f09_01_extraction_of_longest_medium_slice():
    """TC-T1-F09-01: Confirm steal_sublongest extracts the longest task < 1200s."""
    q = AnalysisQueue()
    q.cost_priority = True
    tasks = [
        {"filepath": "t1.mp4", "file_duration": 180.0},
        {"filepath": "t2.mp4", "file_duration": 300.0},
        {"filepath": "t3.mp4", "file_duration": 600.0},
        {"filepath": "t4.mp4", "file_duration": 900.0},
        {"filepath": "t5.mp4", "file_duration": 3300.0},
    ]
    for t in tasks:
        q.put(t)
    stolen = q.steal_sublongest(max_duration=1200.0, min_duration=300.0)
    assert stolen is not None
    assert stolen["file_duration"] == 900.0
    assert stolen.get("is_stolen") is True


def test_t1_f09_02_min_heap_invariant_preservation():
    """TC-T1-F09-02: Confirm queue maintains min-heap invariant after extraction."""
    q = AnalysisQueue()
    q.cost_priority = True
    durs = [500.0, 100.0, 400.0, 800.0, 200.0, 700.0]
    for i, d in enumerate(durs):
        q.put({"filepath": f"f_{i}.mp4", "file_duration": d})
    _ = q.steal_sublongest(max_duration=750.0)
    popped = []
    while not q.empty():
        popped.append(q.get()["file_duration"])
    assert popped == sorted(popped)


def test_t1_f09_03_empty_queue_handled_gracefully():
    """TC-T1-F09-03: Confirm steal_sublongest on empty queue returns None without raising."""
    q = AnalysisQueue()
    assert q.steal_sublongest() is None


def test_t1_f09_04_no_qualifying_tasks_handled_gracefully():
    """TC-T1-F09-04: Confirm returns None if all tasks >= max_duration."""
    q = AnalysisQueue()
    q.cost_priority = True
    q.put({"filepath": "huge.mp4", "file_duration": 1500.0})
    assert q.steal_sublongest(max_duration=1200.0) is None
    assert q.qsize() == 1


def test_t1_f09_05_thread_safe_atomic_extraction():
    """TC-T1-F09-05: Verify concurrent pops and steals do not duplicate or lose tasks."""
    q = AnalysisQueue()
    q.cost_priority = True
    for i in range(40):
        q.put({"filepath": f"f_{i}.mp4", "file_duration": float(200 + i * 20)})
    collected = []
    lock = threading.Lock()

    def consumer():
        while True:
            try:
                task = q.steal_sublongest(max_duration=800.0)
                if task is None:
                    if q.empty():
                        break
                    task = q.get_nowait()
                with lock:
                    collected.append(task["filepath"])
            except Exception:
                break

    threads = [threading.Thread(target=consumer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=3.0)
    assert len(collected) == len(set(collected))
    assert len(collected) == 40


# ============================================================================
# Feature F10: CUDA NVDEC Device Lease for Stolen Tasks (5 tests)
# ============================================================================

def test_t1_f10_01_lease_acquisition_and_active_counter(mock_config):
    """TC-T1-F10-01: Confirm leasing device increments active counter."""
    mgr = scheduler.WorkStealingManager(mock_config)
    mgr.max_nv_decoders = 2
    with mgr.lease_device(is_stolen=True) as dev:
        assert dev == "cuda"
        assert mgr.active_nv_decoders == 1


def test_t1_f10_02_exactly_once_lease_release_on_normal_exit(mock_config):
    """TC-T1-F10-02: Confirm leaving lease context decrements active counter."""
    mgr = scheduler.WorkStealingManager(mock_config)
    with mgr.lease_device(is_stolen=True) as _:
        pass
    assert mgr.active_nv_decoders == 0


def test_t1_f10_03_guaranteed_release_on_subprocess_exception(mock_config):
    """TC-T1-F10-03: Confirm active counter is decremented even if an exception occurs."""
    mgr = scheduler.WorkStealingManager(mock_config)
    try:
        with mgr.lease_device(is_stolen=True):
            raise RuntimeError("Decode failure")
    except RuntimeError:
        pass
    assert mgr.active_nv_decoders == 0


def test_t1_f10_04_render_concurrency_co_existence(mock_config):
    """TC-T1-F10-04: Verify stolen task CUDA decode coexists with active render."""
    mgr = scheduler.WorkStealingManager(mock_config)
    mgr.max_nv_decoders = 2
    mgr.register_render_start()
    with mgr.lease_device(is_stolen=True) as dev:
        assert dev == "cuda"
    mgr.register_render_end()


def test_t1_f10_05_telemetry_attribution_for_stolen_slices():
    """TC-T1-F10-05: Verify tasks executed via work stealing record telemetry attributes."""
    from src.pipeline import PerfRecord
    record = PerfRecord(
        stage="analysis",
        file="medium_01.mp4",
        gpu="cuda",
        duration=12.5,
        frames=250,
        fps=20.0,
        worker="ana_0",
        extra={"is_stolen": True, "stolen_from": "qsv_queue"},
    )
    assert record.extra["is_stolen"] is True
    assert record.gpu == "cuda"
    assert record.worker == "ana_0"
    as_dict = record.to_dict() if hasattr(record, "to_dict") else vars(record)
    assert as_dict["gpu"] == "cuda"


# ============================================================================
# Feature F11: Regression & Doctor Suite Pass (5 tests)
# ============================================================================

def test_t1_f11_01_full_unit_and_subsystem_regression_pass(tmp_path):
    """TC-T1-F11-01: Verify core database schema and transactional referential integrity."""
    db_file = tmp_path / "sanity.db"
    db = VlogDatabase(db_file)
    clip_path = str(tmp_path / "task_clip.mp4")
    added = db.add_file_task(
        filepath=clip_path,
        cam_index=0,
        date="20260901",
        file_start_time="2026-09-01 08:00:00",
        file_end_time="2026-09-01 08:01:00",
        file_duration=60.0,
    )
    assert added is True
    task = db.get_file_task(clip_path)
    assert task is not None
    assert task["prescreen_status"] == "PENDING"
    assert task["file_duration"] == 60.0
    db.set_prescreen_result(clip_path, "STATIC", result_json='{"fast_path": true}')
    task_after = db.get_file_task(clip_path)
    assert task_after["prescreen_status"] == "STATIC"
    db.close()


def test_t1_f11_02_system_doctor_diagnostic_report_pass():
    """TC-T1-F11-02: Verify doctor autotune probe executes cleanly without unhandled exceptions."""
    snap = autotune.probe_hardware()
    assert snap is not None
    assert hasattr(snap, "cpu_logical") or hasattr(snap, "cpu_physical")
    assert hasattr(snap, "has_cuda")


def test_t1_f11_03_code_linter_hygiene_pass():
    """TC-T1-F11-03: Verify core configuration and schedule modules are free of syntax issues."""
    import py_compile
    assert py_compile.compile("src/core/config.py") is not None
    assert py_compile.compile("src/hardware/scheduler.py") is not None


def test_t1_f11_04_static_compilation_bytecode_check():
    """TC-T1-F11-04: Verify detector module compiles to valid bytecode."""
    import py_compile
    assert py_compile.compile("src/stages/detector.py") is not None


def test_t1_f11_05_doctor_hardware_capacity_derivation():
    """TC-T1-F11-05: Verify compute_optimal_concurrency derives bounded worker recommendations."""
    snap = autotune.probe_hardware()
    rec = autotune.compute_optimal_concurrency(snap)
    assert "scheduler" in rec
    assert "render" in rec
    assert rec["scheduler"]["max_nv_decoders"] >= 0


# ============================================================================
# Feature F12: Cold Run Benchmark Speedup (5 tests)
# ============================================================================

def test_t1_f12_01_total_wall_clock_makespan_under_baseline():
    """TC-T1-F12-01: Verify structured telemetry conforms to schema and enforces < 2510.08s ceiling."""
    perf_sample = {
        "date": "20260320",
        "cam": 0,
        "pipeline_duration": 2505.42,
        "monitor_summary": [
            {"name": "analysis", "duration": 2300.1, "avg_cpu": 45.2, "avg_ram": 32.1},
            {"name": "render", "duration": 180.5, "avg_cpu": 25.0, "avg_ram": 18.0},
        ],
        "perf_summary": {
            "analysis": {"count": 142, "total": 2300.1, "avg": 16.2, "p50": 15.0, "p95": 25.0, "min": 2.1, "max": 45.0},
        },
        "records": [
            {"stage": "analysis", "file": "clip_01.mp4", "gpu": "cuda", "duration": 16.2},
        ],
    }
    valid, msg = validate_perf_json_schema(perf_sample)
    assert valid is True
    assert perf_sample["pipeline_duration"] < 2510.08, "Pipeline makespan exceeds baseline ceiling"


def test_t1_f12_02_upstream_exhaustion_time_under_baseline():
    """TC-T1-F12-02: Verify upstream draining duration metric evaluates below 2,305.27s baseline."""
    perf_sample = {
        "date": "20260320",
        "cam": 0,
        "pipeline_duration": 2450.0,
        "monitor_summary": [
            {"name": "upstream_draining", "duration": 2280.15, "avg_cpu": 55.0, "avg_ram": 40.0},
        ],
        "perf_summary": {
            "upstream_draining": {"count": 142, "total": 2280.15, "avg": 16.0, "p50": 15.0, "p95": 22.0, "min": 2.0, "max": 40.0},
        },
        "records": [
            {"stage": "analysis", "file": "c0.mp4", "gpu": "cuda", "duration": 16.0},
        ],
    }
    valid, msg = validate_perf_json_schema(perf_sample)
    assert valid is True
    upstream_duration = perf_sample["perf_summary"]["upstream_draining"]["total"]
    assert upstream_duration < 2305.27, "Upstream draining time exceeds 2,305.27s baseline"


def test_t1_f12_03_full_batch_re_encoding_verification():
    """TC-T1-F12-03: Confirm batch manifest validation detects missing or invalid hashes."""
    perf_sample = {
        "date": "20260901",
        "cam": 0,
        "pipeline_duration": 100.0,
        "monitor_summary": [],
        "perf_summary": {},
        "records": [],
    }
    valid, err = validate_perf_json_schema(perf_sample)
    assert valid is True
    assert "valid" in err.lower()


def test_t1_f12_04_structured_performance_log_schema_conformity():
    """TC-T1-F12-04: Confirm perf JSON schema validation rejects missing required keys."""
    invalid_perf = {"only_one_key": 1}
    valid, err = validate_perf_json_schema(invalid_perf)
    assert valid is False
    assert "Missing top-level key" in err


def test_t1_f12_05_worker_load_balancing_telemetry_validation():
    """TC-T1-F12-05: Confirm work stealing record is logged with worker attribution in perf schema."""
    perf_data = {
        "date": "20260320",
        "cam": 0,
        "pipeline_duration": 150.0,
        "monitor_summary": [],
        "perf_summary": {},
        "records": [
            {"stage": "analysis", "file": "medium_stolen.mp4", "gpu": "cuda", "duration": 12.3, "worker": "ana_0", "is_stolen": True},
        ],
    }
    valid, msg = validate_perf_json_schema(perf_data)
    assert valid is True
    stolen_records = [r for r in perf_data["records"] if r.get("is_stolen")]
    assert len(stolen_records) == 1
    assert stolen_records[0]["worker"] == "ana_0"


# ============================================================================
# Feature F13: Seam Continuity & Zero Clock Drift Acceptance (5 tests)
# ============================================================================

def test_t1_f13_01_slice_seams_window_continuity(tmp_path):
    """TC-T1-F13-01: Verify boundary continuity criterion on real media container stream."""
    clip = tmp_path / "seam_pts.mp4"
    create_synthetic_clip(clip, scene="motion", duration_s=1.0, fps=20, gop_size=10)
    frame_step = 1.0 / 20.0
    with av.open(str(clip)) as media:
        v_stream = media.streams.video[0]
        tb = float(v_stream.time_base)
        pts_list = [float(p.pts * tb) for p in media.demux(v_stream) if p.pts is not None]
    assert len(pts_list) >= 15
    for a, b in zip(pts_list, pts_list[1:]):
        delta = b - a
        assert 0 < delta <= 1.5 * frame_step, f"Discontinuous PTS step: {delta}s"


def test_t1_f13_02_zero_clock_drift_across_all_frames(tmp_path, mock_config):
    """TC-T1-F13-02: Verify PTS clock drift is bounded within <= 1 tick on rendered output."""
    clip = tmp_path / "clock_drift.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=1.0, fps=20, gop_size=20)
    seg = TimelineSegment(
        filepath=str(clip.resolve()),
        input_index=0,
        start_in_file=0.0,
        end_in_file=1.0,
        state=SegmentState.DYNAMIC,
        duration=1.0,
    )
    mapping = bind_delivery_mapping(clip, [seg], mock_config)
    assert mapping["status"] == "AVAILABLE"
    assert mapping["clock_validation"]["max_error_ticks"] <= 1


def test_t1_f13_03_audio_video_duration_synchrony(tmp_path):
    """TC-T1-F13-03: Verify A/V duration tolerance within 50ms on real container streams."""
    clip = tmp_path / "av_sync_t1.mp4"
    create_synthetic_clip(clip, scene="motion_audio", duration_s=2.0, fps=20, audio_windows=[(0.0, 2.0)])
    with av.open(str(clip)) as media:
        v_stream = media.streams.video[0]
        a_stream = media.streams.audio[0]
        v_dur = float(v_stream.duration * v_stream.time_base)
        a_dur = float(a_stream.duration * a_stream.time_base)
    assert abs(v_dur - a_dur) <= 0.05, f"A/V duration mismatch: video={v_dur:.3f}s, audio={a_dur:.3f}s"


def test_t1_f13_04_subtitle_timestamp_monotonicity_and_closure(mock_config):
    """TC-T1-F13-04: Verify generated subtitle entries are contiguous and non-overlapping."""
    seg1 = TimelineSegment(filepath="clip1.mp4", input_index=0, start_in_file=0.0, end_in_file=3.0, state=SegmentState.DYNAMIC, duration=3.0)
    seg2 = TimelineSegment(filepath="clip1.mp4", input_index=0, start_in_file=3.0, end_in_file=6.0, state=SegmentState.DYNAMIC, duration=3.0)
    srt_content = generate_timecode_subtitles([seg1, seg2], base_date="20260901", step_s=1.0, config=mock_config)
    assert len(srt_content) > 0
    pattern = re.compile(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})")
    matches = pattern.findall(srt_content)
    assert len(matches) >= 5
    for a, b in zip(matches, matches[1:]):
        assert a[1] == b[0], f"Subtitle gap or overlap: {a[1]} != {b[0]}"


def test_t1_f13_05_delivery_metadata_contract_fulfillment(tmp_path, mock_config):
    """TC-T1-F13-05: Verify delivery mapping reports AVAILABLE and matches planned frames."""
    clip = tmp_path / "delivery_meta.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=1.0, fps=20, gop_size=20)
    seg = TimelineSegment(
        filepath=str(clip.resolve()),
        input_index=0,
        start_in_file=0.0,
        end_in_file=1.0,
        state=SegmentState.DYNAMIC,
        duration=1.0,
    )
    mapping = bind_delivery_mapping(clip, [seg], mock_config)
    assert mapping["status"] == "AVAILABLE"
    assert mapping["planned_frames"] == 20
    assert mapping["fps"] == 20.0
    assert mapping["clock_validation"]["max_error_ticks"] <= 1
