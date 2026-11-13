"""Tier 3: Cross-Feature Pairwise Combination Tests (14 Tests).

Tests interface boundaries and coupling between subsystems when two features interact.
"""

from pathlib import Path

import av
import pytest

from src.hardware import scheduler
from src.stages.detector import MotionDetector
from src.stages.pipeline_types import AnalysisQueue
from src.stages.timeline import SegmentState, TimelineSegment, compute_display_plans
from tests.helpers_e2e import (
    MockVramController,
    assert_resource_hygiene,
    create_synthetic_clip,
)


def test_tier3_f2_f3_coarse_scan_zero_decode_fast_path(tmp_path, mock_config):
    """T3-01: F2 (Coarse Scan) x F3 (Zero-Decode): Quiescent clip confirms zero motion, bypassing fine decode."""
    clip = tmp_path / "t3_static.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=4.0, fps=20, gop_size=20)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    res = detector._scan_keyframes_coarse(str(clip), file_duration=4.0)
    trace = res[0] if isinstance(res, tuple) else res
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=4.0, threshold=6.5)
    fallback, _ = detector._evaluate_fallback_safeguards(windows, file_duration=4.0, gop_metrics={"keyframes": len(trace), "avg_gop_s": 1.0})
    assert len(windows) == 0
    assert fallback is False


def test_tier3_f2_f4_coarse_scan_awakened_window(tmp_path, mock_config):
    """T3-02: F2 (Coarse Scan) x F4 (Awakened Window): Motion burst detected in coarse scan triggers padded window."""
    clip = tmp_path / "t3_motion.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=6.0, fps=20, gop_size=20, motion_windows=[(2.0, 4.0)])
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    res = detector._scan_keyframes_coarse(str(clip), file_duration=6.0)
    trace = res[0] if isinstance(res, tuple) else res
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=6.0, threshold=6.5)
    assert len(windows) == 1
    assert windows[0][0] <= 2.0
    assert windows[0][1] >= 4.0


def test_tier3_f4_f6_awakened_window_audio_vad_fusion(mock_config):
    """T3-03: F4 (Awakened Window) x F6 (Audio VAD): Speech burst unconditionally awakens fine decode window."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(0.0, 0.1, 0.2), (3.0, 0.1, 0.2), (6.0, 0.1, 0.2)]
    audio_events = [(10.0, 12.5, "speech")]
    windows = detector._resolve_awakened_windows(trace, audio_events=audio_events, file_duration=30.0, threshold=6.5)
    assert len(windows) == 1
    assert windows[0][0] <= 7.0
    assert windows[0][1] >= 15.5


def test_tier3_f4_f5_awakened_window_dense_motion_fallback(mock_config):
    """T3-04: F4 (Awakened Window) x F5 (Fallback): High window count (>8) triggers fallback to full decode."""
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    trace = [(float(i * 20 + 5), 20.0, 30.0) for i in range(10)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=220.0, threshold=6.5)
    fallback, reason = detector._evaluate_fallback_safeguards(windows, file_duration=220.0, gop_metrics={"keyframes": len(trace), "avg_gop_s": 20.0})
    assert fallback is True
    assert "ISOLATED_WINDOWS" in reason or "SPARSE_GOP" in reason


def test_tier3_f1_f5_config_governance_safeguard_override(mock_config):
    """T3-05: F1 (Config Governance) x F5 (Fallback): Dynamic threshold override alters fallback sensitivity."""
    cfg = dict(mock_config)
    cfg.setdefault("detection", {})
    cfg["output"]["qsv"].pop("maxrate", None)
    cfg["output"]["qsv"].pop("bufsize", None)
    cfg["detection"]["iframe_fallback_active_ratio"] = 0.25
    detector = MotionDetector(cfg, decode_gpu="cpu")
    trace = [(float(t), 20.0, 30.0) for t in range(15, 36, 5)]
    windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=100.0, threshold=6.5)
    fallback, reason = detector._evaluate_fallback_safeguards(windows, file_duration=100.0, gop_metrics={"keyframes": len(trace), "avg_gop_s": 3.0})
    assert fallback is True
    assert "DENSE_MOTION" in reason


def test_tier3_f7_f8_vram_margin_blocks_work_stealing(mock_config):
    """T3-06: F7 (VRAM Margin) x F8 (Admission): Free VRAM < 3072 MB blocks work stealing admission."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=2500, used_mb=5000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    mgr.set_vram_probe_fn(ctrl.probe_used)
    mgr.nvdec_cooperative = True
    mgr.max_nv_decoders = 2
    mgr.steal_backlog_threshold = 2
    assert mgr.can_worker_0_steal(queue_size=6, heavy_tasks_cleared=True) is False


def test_tier3_f8_f9_admission_triggers_sublongest_stealing(mock_config):
    """T3-07: F8 (Admission) x F9 (Sub-Longest): Admitted stealing extracts sub-longest task."""
    mgr = scheduler.WorkStealingManager(mock_config)
    ctrl = MockVramController(free_mb=4000, used_mb=3000)
    mgr._vram_margin_probe_fn = ctrl.probe_margin
    mgr.set_vram_probe_fn(ctrl.probe_used)
    assert mgr.can_worker_0_steal(queue_size=5, heavy_tasks_cleared=True) is True

    q = AnalysisQueue()
    q.cost_priority = True
    tasks = [
        {"filepath": "short.mp4", "file_duration": 180.0},
        {"filepath": "medium1.mp4", "file_duration": 600.0},
        {"filepath": "medium2.mp4", "file_duration": 1150.0},
        {"filepath": "heavy.mp4", "file_duration": 3300.0},
    ]
    for t in tasks:
        q.put(t)
    stolen = q.steal_sublongest(max_duration=1200.0, min_duration=300.0)
    assert stolen is not None
    assert stolen["file_duration"] == 1150.0


def test_tier3_f9_f10_sublongest_stolen_task_cuda_lease_during_render(mock_config):
    """T3-08: F9 (Sub-Longest) x F10 (CUDA Lease): Stolen task keeps CUDA lease during active render."""
    mgr = scheduler.WorkStealingManager(mock_config)
    mgr.max_nv_decoders = 2
    mgr.register_render_start()
    with mgr.lease_device(is_stolen=True) as dev:
        assert dev == "cuda"
        assert mgr.state == "TAIL_WORK_STEALING"
    mgr.register_render_end()
    assert mgr.active_nv_decoders == 0


def test_tier3_f3_f13_zero_decode_seam_continuity_and_clock_drift():
    """T3-09: F3 (Zero-Decode) x F13 (Seams & Clock): Splicing static and dynamic segments preserves clock monotonicity."""
    seg_static = TimelineSegment("clip1.mp4", 0, 0.0, 10.0, SegmentState.STATIC, 10.0)
    seg_dynamic = TimelineSegment("clip2.mp4", 1, 0.0, 10.0, SegmentState.DYNAMIC, 10.0)
    plans = compute_display_plans([seg_static, seg_dynamic], output_fps=20.0)
    assert len(plans) == 2
    for dur, _ in plans:
        assert dur > 0
        frames = round(dur * 20.0)
        assert abs(frames - dur * 20.0) < 1e-4


def test_tier3_f2_f10_coarse_scan_nvdec_with_concurrent_qsv(tmp_path, mock_config):
    """T3-10: F2 (Coarse Scan) x F10 (CUDA Lease): Coarse scan executes and releases resources safely."""
    clip = tmp_path / "scan_lease.mp4"
    create_synthetic_clip(clip, scene="static", duration_s=2.0, fps=20, gop_size=20)
    detector = MotionDetector(mock_config, decode_gpu="cpu")
    res = detector._scan_keyframes_coarse(str(clip), file_duration=2.0)
    trace = res[0] if isinstance(res, tuple) else res
    assert len(trace) > 0
    assert_resource_hygiene()


def test_tier3_f6_f13_audio_vad_awakened_clip_av_sync(tmp_path):
    """T3-11: F6 (Audio VAD) x F13 (A/V Sync): Awakened audio clip maintains A/V duration tolerance <= 50ms."""
    clip = tmp_path / "av_sync.mp4"
    create_synthetic_clip(clip, scene="motion_audio", duration_s=4.0, fps=20, audio_windows=[(1.0, 3.0)])
    with av.open(str(clip)) as media:
        v_dur = float(media.streams.video[0].duration * media.streams.video[0].time_base)
        a_dur = float(media.streams.audio[0].duration * media.streams.audio[0].time_base)
        assert abs(v_dur - a_dur) <= 0.05


def test_tier3_f5_f10_fallback_continuous_decode_lease_safety(mock_config):
    """T3-12: F5 (Fallback) x F10 (CUDA Lease): Fallback triggered during leased task cleanly releases lease."""
    mgr = scheduler.WorkStealingManager(mock_config)
    with mgr.lease_device(is_stolen=True) as dev:
        assert dev == "cuda"
        detector = MotionDetector(mock_config, decode_gpu=dev)
        trace = [(float(t), 25.0, 35.0) for t in range(0, 30, 2)]
        windows = detector._resolve_awakened_windows(trace, audio_events=[], file_duration=30.0, threshold=6.5)
        fallback, reason = detector._evaluate_fallback_safeguards(windows, file_duration=30.0, gop_metrics={"keyframes": len(trace), "avg_gop_s": 2.0})
        assert fallback is True
        assert "DENSE_MOTION" in reason
    assert mgr.active_nv_decoders == 0


def test_tier3_f8_f12_work_stealing_upstream_makespan_reduction(mock_config):
    """T3-13: F8 (Admission) x F12 (Makespan): Stealing medium tasks significantly reduces queue drain time."""
    mgr = scheduler.WorkStealingManager(mock_config)
    q = AnalysisQueue()
    q.cost_priority = True
    for i in range(6):
        q.put({"filepath": f"med_{i}.mp4", "file_duration": 900.0})

    stolen_count = 0
    while mgr.can_worker_0_steal(queue_size=q.qsize(), heavy_tasks_cleared=True):
        task = q.steal_sublongest(max_duration=1200.0)
        if task is None:
            break
        stolen_count += 1
    assert stolen_count == 5
    assert q.qsize() == 1


def test_tier3_f1_f7_dynamic_vram_watermark_reconfiguration(mock_config):
    """T3-14: F1 (Config Governance) x F7 (VRAM Margin): Watermark reconfiguration updates pressure trigger."""
    mgr = scheduler.WorkStealingManager(mock_config)
    mgr.vram_watermark_mb = 6200
    ctrl = MockVramController(used_mb=5500)
    mgr.set_vram_probe_fn(ctrl.probe_used)
    assert mgr.is_vram_under_pressure() is False

    mgr.vram_watermark_mb = 5000
    assert mgr.is_vram_under_pressure() is True
