"""
Unit and Integration Tests for Milestone 2 (R2):
- Robust Temporal Sliding EMA Background Model
- Dynamic 8x8 Spatial Grid & Connected-Component Noise Suppression
- Spatial-Temporal Confidence Memory Cooldown Guard
- Strict Anti-Miss Early Termination Timeline Closure Invariant
"""

import math
from pathlib import Path
import numpy as np
import pytest

from src.detector import (
    EmaBackgroundModel,
    SpatialGridMotionFilter,
    MotionDetector,
    _median_filter,
    _smooth_labels,
)
from src.segment import build_segments, Segment
from src.utils import load_config


class TestEmaBackgroundModel:
    """Tests for selective EMA background estimation and dual-difference fusion."""

    def test_first_frame_initialization(self):
        model = EmaBackgroundModel(alpha_bg=0.05, alpha_fg=0.005, beta=0.6)
        frame0 = np.full((100, 100), 120, dtype=np.uint8)
        saliency, d_frame, d_bg = model.update(frame0)

        assert saliency.shape == (100, 100)
        assert np.all(saliency == 0.0)
        assert np.all(d_frame == 0.0)
        assert np.all(d_bg == 0.0)
        assert model.background is not None
        assert np.allclose(model.background, 120.0)

    def test_adaptive_learning_rate_fg_vs_bg_divergence(self):
        """
        Verify that foreground pixels (|diff| > fg_threshold) update slowly (alpha_fg = 0.005),
        while background pixels (|diff| <= fg_threshold) update faster (alpha_bg = 0.05).
        """
        model = EmaBackgroundModel(alpha_bg=0.05, alpha_fg=0.005, beta=0.6, fg_threshold=12.0)
        base = np.full((50, 50), 100, dtype=np.uint8)
        model.update(base)

        # Frame 1: Left half has small diff (+5 <= 12, static bg), right half has large diff (+60 > 12, fg)
        frame1 = base.copy()
        frame1[:, :25] = 105  # bg region
        frame1[:, 25:] = 160  # fg region

        model.update(frame1)

        # Expected bg update: 100 * 0.95 + 105 * 0.05 = 95 + 5.25 = 100.25
        # Expected fg update: 100 * 0.995 + 160 * 0.005 = 99.5 + 0.8 = 100.30
        bg_val = float(np.mean(model.background[:, :25]))
        fg_val = float(np.mean(model.background[:, 25:]))

        assert math.isclose(bg_val, 100.25, abs_tol=1e-3)
        assert math.isclose(fg_val, 100.30, abs_tol=1e-3)

        # After 10 more frames of holding frame1
        for _ in range(10):
            model.update(frame1)

        # Bg region should adapt much faster towards 105 than fg adapts towards 160
        bg_val_10 = float(np.mean(model.background[:, :25]))
        fg_val_10 = float(np.mean(model.background[:, 25:]))

        # Bg has moved ~43% towards target (1 - 0.95^11 = ~0.43)
        assert bg_val_10 > 102.0
        # Fg has moved only ~5.3% towards target (1 - 0.995^11 = ~0.053)
        assert fg_val_10 < 104.0

    def test_dual_difference_resting_human_detection(self):
        """
        Simulate an object/person entering and then sitting completely still.
        Standard frame diff D_frame drops to 0, but EMA D_bg and dual diff M_t remain high.
        """
        model = EmaBackgroundModel(alpha_bg=0.05, alpha_fg=0.005, beta=0.6, fg_threshold=12.0)
        empty_room = np.full((100, 100), 50, dtype=np.uint8)
        model.update(empty_room)

        # Person enters: central 40x40 area becomes 150
        person_frame = empty_room.copy()
        person_frame[30:70, 30:70] = 150

        # First movement frame
        saliency_entry, d_frame_entry, d_bg_entry = model.update(person_frame)
        assert np.max(d_frame_entry[30:70, 30:70]) == 100.0
        assert np.max(saliency_entry[30:70, 30:70]) == 100.0

        # Person sits still for 30 consecutive frames
        for _ in range(30):
            saliency, d_frame, d_bg = model.update(person_frame)

        # D_frame is now 0 because person is stationary
        assert np.all(d_frame == 0.0)

        # But D_bg is still high because alpha_fg=0.005 slowly adapts (100 * 0.995^31 ~ 85.6)
        person_bg_diff = float(np.mean(d_bg[30:70, 30:70]))
        assert person_bg_diff > 75.0

        # Saliency M_t = max(0, 0.6 * D_bg) is ~ 50.0
        person_saliency = float(np.mean(saliency[30:70, 30:70]))
        assert person_saliency > 45.0

    def test_gradual_ambient_lighting_change_absorption(self):
        """
        Gradual lighting change across the whole frame should be smoothly absorbed
        without producing large dual-difference motion saliency.
        """
        model = EmaBackgroundModel(alpha_bg=0.05, alpha_fg=0.005, beta=0.6, fg_threshold=12.0)
        frame = np.full((60, 60), 50, dtype=np.uint8)
        model.update(frame)

        max_saliencies = []
        for i in range(50):
            # Slow gradual ambient drift +0.1 per frame
            frame = np.full((60, 60), int(50 + i * 0.1), dtype=np.uint8)
            saliency, _, _ = model.update(frame)
            max_saliencies.append(float(np.max(saliency)))

        # Saliency should remain very small (< 2.0, far below fg_threshold=12.0)
        assert max(max_saliencies) < 2.0

    def test_reset_functionality(self):
        model = EmaBackgroundModel()
        model.update(np.full((20, 20), 100, dtype=np.uint8))
        assert model.background is not None
        assert model.prev_frame is not None

        model.reset()
        assert model.background is None
        assert model.prev_frame is None


class TestSpatialGridMotionFilter:
    """Tests for 8x8 spatial grid extraction, connected component noise filtering, and cooldown."""

    def test_grid_energy_extraction_geometry(self):
        grid_filter = SpatialGridMotionFilter(grid_rows=8, grid_cols=8)
        saliency = np.zeros((234, 416), dtype=np.float32)

        # Inject energy in grid cell (row 2, col 3)
        # Row 2 spans y in [round(2*234/8) : round(3*234/8)] = [58 : 87]
        # Col 3 spans x in [round(3*416/8) : round(4*416/8)] = [156 : 208]
        saliency[58:87, 156:208] = 10.0

        energies = grid_filter.extract_grid_energies(saliency)
        assert energies.shape == (8, 8)
        assert math.isclose(energies[2, 3], 10.0, abs_tol=0.1)
        # Other cells should be 0
        energies[2, 3] = 0.0
        assert np.all(energies == 0.0)

    def test_isolated_single_cell_ir_noise_suppression(self):
        """
        Isolated single-cell activation with moderate energy (IR night vision noise)
        should be filtered out by connected component filtering.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=1.5,
            sens_multiplier=1.5,
        )

        saliency = np.zeros((80, 80), dtype=np.float32)
        # Cell (1, 1) has energy 3.0 (above threshold 2.25, but below extreme spike 5.625)
        # Row 1 spans y in [10:20], Col 1 spans x in [10:20]
        saliency[10:20, 10:20] = 3.0

        eff_energy, is_motion, stats = grid_filter.process_frame(saliency, dt=0.2)

        # Single isolated cell should be suppressed
        assert stats["active_cells"] == 0
        assert is_motion is False
        assert not grid_filter.active_grid[1, 1]
        # Effective energy should be dampened
        assert eff_energy < 0.5

    def test_contiguous_cluster_boost_and_detection(self):
        """
        Contiguous 2-cell cluster (e.g. human body resting) should be preserved
        and boosted by cluster_boost.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=1.5,
            sens_multiplier=1.5,
            cluster_boost=1.2,
        )

        saliency = np.zeros((80, 80), dtype=np.float32)
        # Cells (2, 2) and (2, 3) active with energy 4.0
        saliency[20:30, 20:30] = 4.0
        saliency[20:30, 30:40] = 4.0

        eff_energy, is_motion, stats = grid_filter.process_frame(saliency, dt=0.2)

        assert stats["active_cells"] == 2
        assert is_motion is True
        assert grid_filter.active_grid[2, 2] is True or grid_filter.active_grid[2, 2] == 1
        assert grid_filter.active_grid[2, 3] is True or grid_filter.active_grid[2, 3] == 1
        # Effective energy boosted by 1.2
        assert eff_energy >= 4.0 * 1.2

    def test_large_magnitude_single_cell_spike_retention(self):
        """
        Isolated single cell with extreme magnitude energy (> 2.5 * thresh)
        should be retained as high-confidence motion.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=1.5,
            sens_multiplier=1.5,
        )

        saliency = np.zeros((80, 80), dtype=np.float32)
        # Threshold is ~2.25. Inject energy 15.0 > 2.5 * 2.25 = 5.625
        saliency[10:20, 10:20] = 15.0

        eff_energy, is_motion, stats = grid_filter.process_frame(saliency, dt=0.2)

        assert stats["active_cells"] == 1
        assert is_motion is True
        assert grid_filter.active_grid[1, 1] is True or grid_filter.active_grid[1, 1] == 1

    def test_adaptive_noise_floor_tracking_on_inactive_cells(self):
        """
        Noise floor should smoothly track ambient noise on inactive cells
        without being corrupted on active cells.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            cell_noise_alpha=0.1,
            base_noise_thresh=1.0,
        )

        # Baseline noise = 1.0. Feed quiet noise at 2.0 on cell (0, 0)
        saliency = np.zeros((80, 80), dtype=np.float32)
        saliency[0:10, 0:10] = 1.2  # below active threshold (1.0 * 1.5 = 1.5)

        for _ in range(5):
            grid_filter.process_frame(saliency, dt=0.2)

        # Noise floor on (0, 0) should adapt upwards towards 1.2
        assert grid_filter.noise_floor_grid[0, 0] > 1.05


class TestSpatialTemporalConfidenceCooldown:
    """Tests for spatial-temporal confidence decay grid and early termination guard."""

    def test_confidence_grid_exponential_decay(self):
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            cooldown_half_life=15.0,
        )

        saliency = np.zeros((80, 80), dtype=np.float32)
        saliency[20:40, 20:40] = 10.0  # active cells

        # Frame 0: Trigger motion
        grid_filter.process_frame(saliency, dt=0.2)
        assert np.max(grid_filter.confidence_grid) == 1.0

        # Static frames for 5 seconds
        saliency_static = np.zeros((80, 80), dtype=np.float32)
        grid_filter.process_frame(saliency_static, dt=5.0)

        # Expected confidence: exp(-5.0 / 15.0) = exp(-0.3333) ~ 0.7165
        conf_5s = float(np.max(grid_filter.confidence_grid))
        assert math.isclose(conf_5s, math.exp(-5.0 / 15.0), abs_tol=1e-3)

        # 10 more seconds (total 15s elapsed since motion)
        grid_filter.process_frame(saliency_static, dt=10.0)
        conf_15s = float(np.max(grid_filter.confidence_grid))
        assert math.isclose(conf_15s, math.exp(-15.0 / 15.0), abs_tol=1e-3)

        # 30 more seconds (total 45s elapsed since motion)
        grid_filter.process_frame(saliency_static, dt=30.0)
        conf_45s = float(np.max(grid_filter.confidence_grid))
        assert math.isclose(conf_45s, math.exp(-45.0 / 15.0), abs_tol=1e-3)
        assert conf_45s < 0.05

    def test_early_termination_blocked_by_confidence_cooldown(self):
        """
        When motion stops, early termination must be blocked during the cooldown period
        even if consecutive static frames >= term_window.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            cooldown_half_life=15.0,
        )

        # 1. Trigger motion in frame 1
        saliency_motion = np.zeros((80, 80), dtype=np.float32)
        saliency_motion[20:40, 20:40] = 8.0
        grid_filter.process_frame(saliency_motion, dt=0.2)

        # 2. 20 consecutive static frames at dt = 0.2s (total 4.0s elapsed)
        saliency_static = np.zeros((80, 80), dtype=np.float32)
        for _ in range(20):
            grid_filter.process_frame(saliency_static, dt=0.2)

        # Check early termination condition with window=20, threshold=2.0
        can_term = grid_filter.can_early_terminate(
            consecutive_static=20,
            term_window=20,
            current_energy=0.1,
            term_threshold=2.0,
        )

        # Blocked because max confidence is exp(-4.0/15.0) ~ 0.766 >= 0.05
        assert can_term is False

    def test_early_termination_allowed_after_full_cooldown_decay(self):
        """
        Once cooldown period has elapsed (confidence < 0.05) and consecutive static frames >= window,
        early termination must be permitted.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            cooldown_half_life=15.0,
        )

        # Motion
        saliency_motion = np.zeros((80, 80), dtype=np.float32)
        saliency_motion[20:40, 20:40] = 8.0
        grid_filter.process_frame(saliency_motion, dt=0.2)

        # Static frames for 50 seconds (250 frames at dt=0.2s)
        saliency_static = np.zeros((80, 80), dtype=np.float32)
        for _ in range(250):
            grid_filter.process_frame(saliency_static, dt=0.2)

        can_term = grid_filter.can_early_terminate(
            consecutive_static=250,
            term_window=20,
            current_energy=0.05,
            term_threshold=2.0,
        )

        # Allowed because max confidence < 0.05
        assert can_term is True

    def test_early_termination_immediate_when_no_prior_motion(self):
        """
        If video starts with no motion from the beginning, confidence is 0.0,
        allowing early termination as soon as term_window is reached.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            cooldown_half_life=15.0,
        )

        saliency_static = np.zeros((80, 80), dtype=np.float32)
        for _ in range(20):
            grid_filter.process_frame(saliency_static, dt=0.2)

        can_term = grid_filter.can_early_terminate(
            consecutive_static=20,
            term_window=20,
            current_energy=0.0,
            term_threshold=2.0,
        )

        assert can_term is True


class TestTimelineClosureInvariant:
    """Tests guaranteeing that early-terminated videos strictly close their timeline."""

    def test_early_terminated_analysis_timestamp_closure(self, mock_config):
        mock_config["detection"]["analysis_early_term_enabled"] = True
        mock_config["detection"]["analysis_early_term_window"] = 10
        mock_config["detection"]["analysis_early_term_threshold"] = 2.0
        mock_config["detection"]["early_term_cooldown_guard"] = True

        detector = MotionDetector(mock_config, decode_gpu="cuda")

        # 30 static frames
        frames = [np.full((234, 416), 50, dtype=np.uint8) for _ in range(30)]

        start_offset = 1200.0
        file_duration = 300.0

        results, _ = detector.analyze_frames(
            frames=frames,
            start_offset=start_offset,
            file_duration=file_duration,
            fps=5.0,
        )

        assert len(results) > 0
        # Final result timestamp MUST strictly equal start_offset + file_duration
        assert results[-1]["time"] == start_offset + file_duration

        # Downstream Segment building must produce a terminal segment ending at start_offset + file_duration
        segments = build_segments(
            frame_labels=results,
            source_file="test_video.mp4",
            min_motion_dur=1.0,
            min_static_dur=5.0,
            file_offset=start_offset,
        )

        assert len(segments) > 0
        assert segments[0].start_time == start_offset
        assert segments[-1].end_time == start_offset + file_duration
        assert segments[-1].state == "STATIC"

    def test_normal_completion_timeline_bounds(self, mock_config):
        mock_config["detection"]["analysis_early_term_enabled"] = False
        detector = MotionDetector(mock_config, decode_gpu="cuda")

        frames = [np.full((234, 416), 50, dtype=np.uint8) for _ in range(20)]
        start_offset = 500.0
        file_duration = 4.0

        results, _ = detector.analyze_frames(
            frames=frames,
            start_offset=start_offset,
            file_duration=file_duration,
            fps=5.0,
        )

        assert len(results) == 20
        # Timestamps should be monotonic and bounded by start_offset + file_duration
        for r in results:
            assert r["time"] >= start_offset
            assert r["time"] <= start_offset + file_duration


class TestMotionDetectorFullIntegration:
    """Integration tests for MotionDetector with full M2 configuration."""

    def test_detector_initialization_with_m2_config(self, mock_config):
        detector = MotionDetector(mock_config, decode_gpu="qsv")

        assert detector.ema_enabled is True
        assert detector.ema_alpha_bg == 0.05
        assert detector.ema_alpha_fg == 0.005
        assert detector.bg_diff_weight == 0.6
        assert detector.grid_rows == 8
        assert detector.grid_cols == 8
        assert detector.cooldown_half_life == 15.0
        assert detector.min_connected_cells == 2
        assert detector.early_term_cooldown_guard is True

    def test_settings_yaml_contains_all_m2_keys(self):
        cfg = load_config()
        det = cfg["detection"]

        assert "ema_background_enabled" in det
        assert det["ema_background_enabled"] is True
        assert "ema_alpha_bg" in det
        assert "ema_alpha_fg" in det
        assert "bg_diff_weight" in det
        assert "grid_rows" in det
        assert "grid_cols" in det
        assert "cooldown_half_life" in det
        assert "min_connected_cells" in det
        assert "early_term_cooldown_guard" in det
        assert det["early_term_cooldown_guard"] is True

    def test_synthetic_sitting_scenario_preserves_motion_and_guards_early_term(self, mock_config):
        """
        End-to-end test of sitting scenario:
        1. 5 static frames
        2. 5 movement frames (person enters)
        3. 15 stationary frames (person sits still)
        4. 30 static frames (empty room)
        """
        mock_config["detection"]["analysis_early_term_enabled"] = True
        mock_config["detection"]["analysis_early_term_window"] = 10
        mock_config["detection"]["early_term_cooldown_guard"] = True
        mock_config["detection"]["cooldown_half_life"] = 1.0  # 1.0s decay for test speed

        detector = MotionDetector(mock_config, decode_gpu="cuda")

        frames = []
        base = np.full((234, 416), 40, dtype=np.uint8)

        # 5 static frames
        for _ in range(5):
            frames.append(base.copy())

        # 5 movement frames (person enters and sits)
        for i in range(5):
            f = base.copy()
            f[80:160, 150:250] = int(140 + i * 5)
            frames.append(f)

        # 15 stationary frames (person sitting still at 160)
        sitting_frame = base.copy()
        sitting_frame[80:160, 150:250] = 160
        for _ in range(15):
            frames.append(sitting_frame.copy())

        # 80 empty frames (person left, ghost absorbs, cooldown decays, early term triggers)
        for _ in range(80):
            frames.append(base.copy())

        file_duration = 60.0  # 1 minute file duration
        results, _ = detector.analyze_frames(
            frames=frames,
            start_offset=0.0,
            file_duration=file_duration,
            fps=5.0,
        )

        assert len(results) > 0
        motion_flags = [r["is_motion"] for r in results]

        # Saliency and motion should be detected during movement and sitting
        assert any(motion_flags[5:20])

        # Early termination triggered after cooldown and timeline strictly closes at file_duration
        assert results[-1]["time"] == file_duration
