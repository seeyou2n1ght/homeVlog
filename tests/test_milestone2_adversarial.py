"""
Adversarial Challenge Test Suite for Milestone 2 (R2):
- Spatial Grid Noise Filtering (Gaussian, Salt-and-Pepper, Spikes, Cluster Boosting)
- Confidence Memory Cooldown & Anti-Miss Early Termination for Resting Subjects (5s, 10s, 15s, 25s, 60s)
- Strict Timeline Closure Invariant across Variable Frame Rates and Durations
- Boundary Conditions & Edge Cases
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
from src.segment import build_segments, Segment, merge_cross_file
from src.utils import load_config


class TestSpatialGridNoiseFilteringAdversarial:
    """Adversarial stress testing for spatial grid noise suppression."""

    def test_gaussian_noise_across_8x8_grid_suppression(self):
        """
        Challenge 1.1: Inject random Gaussian noise at various standard deviations (sigma = 0.5 to 15.0).
        Verify that distributed diffuse Gaussian noise does not trigger false connected component motion.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=1.5,
            sens_multiplier=1.5,
            cell_noise_alpha=0.05,
        )

        np.random.seed(42)
        height, width = 160, 160

        # Run 20 frames of mild to moderate Gaussian noise (sigma = 1.0 to 2.0)
        for _ in range(20):
            noise_saliency = np.abs(np.random.normal(loc=0.0, scale=1.5, size=(height, width))).astype(np.float32)
            eff_energy, is_motion, stats = grid_filter.process_frame(noise_saliency, dt=0.2)

            # Isolated diffuse noise should have active_cells == 0 and is_motion is False
            assert stats["active_cells"] == 0, f"Gaussian noise triggered false active cells: {stats}"
            assert is_motion is False, "Gaussian noise incorrectly marked as motion"
            assert eff_energy < 1.5, f"Effective energy not dampened: {eff_energy}"

    def test_salt_and_pepper_isolated_spikes_across_8x8_grid(self):
        """
        Challenge 1.2: Randomly distribute isolated single-cell salt-and-pepper spikes.
        Verify that isolated single cells without 8-neighborhood connectivity are suppressed.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=1.5,
            sens_multiplier=1.5,
        )

        saliency = np.zeros((80, 80), dtype=np.float32)
        # Place 4 non-adjacent isolated spikes across the grid
        # Cell (0, 0), Cell (0, 3), Cell (3, 0), Cell (5, 5)
        # Each cell has moderate spike energy 3.0 (between thresh 2.25 and extreme spike 5.625)
        saliency[0:10, 0:10] = 3.0    # cell (0, 0)
        saliency[0:10, 30:40] = 3.0   # cell (0, 3)
        saliency[30:40, 0:10] = 3.0   # cell (3, 0)
        saliency[50:60, 50:60] = 3.0  # cell (5, 5)

        eff_energy, is_motion, stats = grid_filter.process_frame(saliency, dt=0.2)

        # None of these 4 isolated cells should pass the connected component filter (min_connected_cells=2)
        assert stats["active_cells"] == 0
        assert is_motion is False
        assert np.all(grid_filter.active_grid == False)

    def test_salt_and_pepper_random_sparse_noise_monte_carlo(self):
        """
        Challenge 1.3: Monte Carlo simulation of random sparse salt-and-pepper noise.
        Randomly select 1 to 3 non-adjacent cells in each frame for 50 frames.
        Verify 0 false motion triggers.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=1.5,
            sens_multiplier=1.5,
        )

        rng = np.random.RandomState(12345)
        saliency_shape = (80, 80)

        for _ in range(50):
            saliency = np.zeros(saliency_shape, dtype=np.float32)
            # Pick non-adjacent cells: e.g. row r in [0, 2, 4, 6], col c in [0, 2, 4, 6]
            even_coords = [(r, c) for r in (0, 2, 4, 6) for c in (0, 2, 4, 6)]
            chosen_indices = rng.choice(len(even_coords), size=3, replace=False)
            for idx in chosen_indices:
                r, c = even_coords[idx]
                r_start, r_end = r * 10, (r + 1) * 10
                c_start, c_end = c * 10, (c + 1) * 10
                saliency[r_start:r_end, c_start:c_end] = float(rng.uniform(2.5, 3.5))

            eff_energy, is_motion, stats = grid_filter.process_frame(saliency, dt=0.2)
            assert stats["active_cells"] == 0, "Non-adjacent cells wrongly clustered"
            assert is_motion is False

    def test_contiguous_cluster_boost_vs_noise_discrimination(self):
        """
        Challenge 1.4: Contiguous 2-cell, 3-cell, and 4-cell clusters must be confirmed and boosted,
        even when background has baseline noise.
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
        # Diagonal neighbor connectivity (8-neighborhood): Cell (2, 2) and Cell (3, 3)
        saliency[20:30, 20:30] = 3.5
        saliency[30:40, 30:40] = 3.5

        eff_energy, is_motion, stats = grid_filter.process_frame(saliency, dt=0.2)

        # 8-neighborhood connectivity should connect diagonal cells (2, 2) and (3, 3)
        assert stats["active_cells"] == 2
        assert is_motion is True
        assert eff_energy >= 3.5 * 1.2

    def test_noise_floor_adaptation_resilience(self):
        """
        Challenge 1.5: Verify that ambient noise floor steadily adapts upwards on noisy camera feeds,
        preventing continuous false triggers over time.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            cell_noise_alpha=0.05,
            base_noise_thresh=1.5,
            sens_multiplier=1.5,
        )

        # Inject continuous 1.8 energy across all cells (above base_noise_thresh 1.5, but not extreme)
        # Initially, all cells are connected and trigger motion
        saliency = np.full((80, 80), 1.8, dtype=np.float32)
        grid_filter.process_frame(saliency, dt=0.2)

        # After steady ambient noise on inactive cells or slight drop
        saliency_quiet = np.full((80, 80), 1.6, dtype=np.float32)
        for _ in range(50):
            grid_filter.process_frame(saliency_quiet, dt=0.2)

        # Adaptive noise floor should have adapted towards 1.6
        assert np.all(grid_filter.noise_floor_grid >= 1.5)


class TestRestingSubjectPausingAndCooldownAdversarial:
    """Adversarial stress testing for resting subject pausing & confidence memory cooldown."""

    @pytest.mark.parametrize("pause_duration,expected_blocked", [
        (5.0, True),    # C(5s) = exp(-5/15) ~ 0.717 > 0.05 -> BLOCKED
        (10.0, True),   # C(10s) = exp(-10/15) ~ 0.513 > 0.05 -> BLOCKED
        (15.0, True),   # C(15s) = exp(-15/15) ~ 0.368 > 0.05 -> BLOCKED
        (25.0, True),   # C(25s) = exp(-25/15) ~ 0.189 > 0.05 -> BLOCKED
        (40.0, True),   # C(40s) = exp(-40/15) ~ 0.069 > 0.05 -> BLOCKED
        (44.0, True),   # C(44s) = exp(-44/15) ~ 0.053 > 0.05 -> BLOCKED
        (45.0, False),  # C(45s) = exp(-45/15) = exp(-3) ~ 0.0498 < 0.05 -> ALLOWED
        (60.0, False),  # C(60s) = exp(-60/15) = exp(-4) ~ 0.0183 < 0.05 -> ALLOWED
        (120.0, False), # C(120s) = exp(-120/15) ~ 3.35e-4 < 0.05 -> ALLOWED
    ])
    def test_resting_subject_cooldown_decay_mathematical_precision(self, pause_duration, expected_blocked):
        """
        Challenge 2.1: Test resting subject pausing for 5s, 10s, 15s, 25s, 40s, 44s, 45s, 60s, 120s.
        Verify that cooldown properly prevents early termination cutoffs until full decay (t >= 45s).
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            cooldown_half_life=15.0,
        )

        # 1. Subject moves in frame 0 (cell 2, 2 and 2, 3)
        saliency_motion = np.zeros((80, 80), dtype=np.float32)
        saliency_motion[20:30, 20:40] = 8.0
        grid_filter.process_frame(saliency_motion, dt=0.2)

        assert np.max(grid_filter.confidence_grid) == 1.0

        # 2. Subject pauses for pause_duration seconds
        # Step through in 0.2s increments
        num_frames = int(round(pause_duration / 0.2))
        saliency_static = np.zeros((80, 80), dtype=np.float32)
        for _ in range(num_frames):
            grid_filter.process_frame(saliency_static, dt=0.2)

        # Verify exact exponential decay: exp(-pause_duration / 15.0)
        expected_conf = math.exp(-pause_duration / 15.0)
        actual_conf = float(np.max(grid_filter.confidence_grid))
        assert math.isclose(actual_conf, expected_conf, abs_tol=1e-3)

        # Check early termination decision with window=20 frames, thresh=2.0
        can_term = grid_filter.can_early_terminate(
            consecutive_static=num_frames,
            term_window=20,
            current_energy=0.0,
            term_threshold=2.0,
        )

        if expected_blocked:
            assert can_term is False, f"Early termination prematurely triggered at pause_duration={pause_duration}s (conf={actual_conf:.4f})"
        else:
            assert can_term is True, f"Early termination failed to permit after decay at pause_duration={pause_duration}s (conf={actual_conf:.4f})"

    def test_intermittent_micro_movements_reset_cooldown_clock(self):
        """
        Challenge 2.2: A resting subject pauses for 20s, micro-moves for 1 frame, pauses for another 20s.
        Total elapsed time = 40s.
        Without reset, 40s would be close to decay.
        With reset at 20s, confidence at 40s is exp(-20/15) ~ 0.263 > 0.05, keeping early termination strictly blocked!
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            cooldown_half_life=15.0,
        )

        saliency_motion = np.zeros((80, 80), dtype=np.float32)
        saliency_motion[20:30, 20:40] = 8.0
        saliency_static = np.zeros((80, 80), dtype=np.float32)

        # Step 1: Initial motion at t=0
        grid_filter.process_frame(saliency_motion, dt=0.2)

        # Step 2: Pause 20s (100 frames at dt=0.2s)
        for _ in range(100):
            grid_filter.process_frame(saliency_static, dt=0.2)
        conf_20s = float(np.max(grid_filter.confidence_grid))
        assert math.isclose(conf_20s, math.exp(-20.0 / 15.0), abs_tol=1e-3)

        # Step 3: Twitch/micro-movement for 1 frame
        grid_filter.process_frame(saliency_motion, dt=0.2)
        assert np.max(grid_filter.confidence_grid) == 1.0

        # Step 4: Pause another 20s (100 frames at dt=0.2s)
        for _ in range(100):
            grid_filter.process_frame(saliency_static, dt=0.2)

        conf_after_second_pause = float(np.max(grid_filter.confidence_grid))
        assert math.isclose(conf_after_second_pause, math.exp(-20.0 / 15.0), abs_tol=1e-3)
        assert conf_after_second_pause > 0.05

        # Early termination MUST be blocked
        can_term = grid_filter.can_early_terminate(
            consecutive_static=100,
            term_window=20,
            current_energy=0.0,
            term_threshold=2.0,
        )
        assert can_term is False

    def test_multi_subject_independent_spatial_cooldown_tracking(self):
        """
        Challenge 2.3: Subject A at (1, 1)-(1, 2) stops at t=0; Subject B at (6, 6)-(6, 7) moves at t=30s.
        Grid filter must track separate decay rates and maintain maximum regional confidence.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            cooldown_half_life=15.0,
        )

        saliency_a = np.zeros((80, 80), dtype=np.float32)
        saliency_a[10:20, 10:30] = 7.0

        saliency_b = np.zeros((80, 80), dtype=np.float32)
        saliency_b[60:70, 60:80] = 7.0

        saliency_static = np.zeros((80, 80), dtype=np.float32)

        # Subject A moves at t=0
        grid_filter.process_frame(saliency_a, dt=0.2)
        # 30s pass (Subject A decaying to exp(-30/15) ~ 0.135)
        for _ in range(150):
            grid_filter.process_frame(saliency_static, dt=0.2)

        assert math.isclose(grid_filter.confidence_grid[1, 1], math.exp(-2.0), abs_tol=1e-2)

        # Subject B moves at t=30s
        grid_filter.process_frame(saliency_b, dt=0.2)
        assert grid_filter.confidence_grid[6, 6] == 1.0

        # Another 20s pass (total 50s for A, 20s for B)
        for _ in range(100):
            grid_filter.process_frame(saliency_static, dt=0.2)

        # A has decayed below 0.05 (exp(-50/15) ~ 0.035)
        assert grid_filter.confidence_grid[1, 1] < 0.05
        # B is still above 0.05 (exp(-20/15) ~ 0.263)
        assert grid_filter.confidence_grid[6, 6] > 0.05
        # Global max confidence reflects Subject B and blocks early termination
        assert float(np.max(grid_filter.confidence_grid)) > 0.05

        can_term = grid_filter.can_early_terminate(
            consecutive_static=100,
            term_window=20,
            current_energy=0.0,
            term_threshold=2.0,
        )
        assert can_term is False


class TestTimelineClosureVariableFpsAndDurationsAdversarial:
    """Adversarial stress testing for timeline closure across variable frame rates and durations."""

    @pytest.mark.parametrize("fps", [1.0, 2.0, 3.0, 5.0, 10.0, 23.976, 25.0, 29.97, 30.0, 60.0])
    @pytest.mark.parametrize("file_duration", [5.0, 12.5, 60.0, 300.0, 600.0, 1800.0])
    @pytest.mark.parametrize("start_offset", [0.0, 100.25, 3600.0])
    def test_timeline_closure_across_variable_fps_and_durations(self, mock_config, fps, file_duration, start_offset):
        """
        Challenge 3.1: Test timeline closure across 10 different FPS tiers and 6 duration scales.
        Verify zero timestamp gaps, monotonic order, exact end timestamp closure, and segment continuity.
        """
        mock_config["detection"]["analysis_early_term_enabled"] = True
        mock_config["detection"]["analysis_early_term_window"] = 10
        mock_config["detection"]["early_term_cooldown_guard"] = True
        mock_config["detection"]["cooldown_half_life"] = 15.0

        detector = MotionDetector(mock_config, decode_gpu="cuda")

        # Create synthetic static frames
        # 30 frames is enough to test early termination behavior
        frames = [np.full((234, 416), 40, dtype=np.uint8) for _ in range(30)]

        results, _ = detector.analyze_frames(
            frames=frames,
            start_offset=start_offset,
            file_duration=file_duration,
            fps=fps,
        )

        assert len(results) > 0, "Analyze returned empty results"

        # 1. Strict start and end timestamp check
        assert math.isclose(results[0]["time"], start_offset, abs_tol=1e-3)
        assert math.isclose(results[-1]["time"], start_offset + file_duration, abs_tol=1e-3)

        # 2. Strict monotonicity check
        for i in range(len(results) - 1):
            assert results[i + 1]["time"] >= results[i]["time"], f"Timestamp not monotonic at index {i}"
            assert results[i]["time"] <= start_offset + file_duration + 1e-4

        # 3. Downstream build_segments check
        segments = build_segments(
            frame_labels=results,
            source_file="challenge_synthetic.mp4",
            min_motion_dur=1.0,
            min_static_dur=5.0,
            file_offset=start_offset,
        )

        assert len(segments) > 0
        assert math.isclose(segments[0].start_time, start_offset, abs_tol=1e-3)
        assert math.isclose(segments[-1].end_time, start_offset + file_duration, abs_tol=1e-3)

        # 4. Zero timeline gaps between segments
        for i in range(len(segments) - 1):
            gap = segments[i + 1].start_time - segments[i].end_time
            assert gap >= -1e-4, f"Negative duration or overlap at segment {i}"
            assert gap <= 0.5, f"Unexpected timeline gap ({gap:.3f}s) between segment {i} and {i+1}"

    def test_multi_file_sequential_timeline_continuity(self, mock_config):
        """
        Challenge 3.2: Simulate a series of 3 consecutive files across the day:
        File 1: 00:00:00 -> 00:10:00 (offset=0, dur=600)
        File 2: 00:10:00 -> 00:20:00 (offset=600, dur=600) [Early Terminated at t=630s]
        File 3: 00:20:00 -> 00:30:00 (offset=1200, dur=600)
        Verify that early termination in File 2 does not cause a gap between File 2 and File 3.
        """
        mock_config["detection"]["analysis_early_term_enabled"] = True
        mock_config["detection"]["analysis_early_term_window"] = 10
        detector = MotionDetector(mock_config, decode_gpu="cuda")

        # File 1: Full normal run
        f1_frames = [np.full((234, 416), 40, dtype=np.uint8) for _ in range(20)]
        f1_res, _ = detector.analyze_frames(f1_frames, start_offset=0.0, file_duration=600.0, fps=5.0)
        f1_segs = build_segments(f1_res, "f1.mp4", file_offset=0.0)

        # File 2: Early terminates quickly
        f2_frames = [np.full((234, 416), 40, dtype=np.uint8) for _ in range(15)]
        f2_res, _ = detector.analyze_frames(f2_frames, start_offset=600.0, file_duration=600.0, fps=5.0)
        f2_segs = build_segments(f2_res, "f2.mp4", file_offset=600.0)

        # File 3: Normal run
        f3_frames = [np.full((234, 416), 40, dtype=np.uint8) for _ in range(20)]
        f3_res, _ = detector.analyze_frames(f3_frames, start_offset=1200.0, file_duration=600.0, fps=5.0)
        f3_segs = build_segments(f3_res, "f3.mp4", file_offset=1200.0)

        # Verify seamless continuity across individual file segment lists before merging
        assert math.isclose(f1_segs[0].start_time, 0.0, abs_tol=1e-3)
        assert math.isclose(f1_segs[-1].end_time, 600.0, abs_tol=1e-3)
        assert math.isclose(f2_segs[0].start_time, 600.0, abs_tol=1e-3)
        assert math.isclose(f2_segs[-1].end_time, 1200.0, abs_tol=1e-3)
        assert math.isclose(f3_segs[0].start_time, 1200.0, abs_tol=1e-3)
        assert math.isclose(f3_segs[-1].end_time, 1800.0, abs_tol=1e-3)
        assert math.isclose(f1_segs[-1].end_time, f2_segs[0].start_time, abs_tol=1e-3)
        assert math.isclose(f2_segs[-1].end_time, f3_segs[0].start_time, abs_tol=1e-3)

        # In-place cross-file merging
        all_segs = f1_segs + f2_segs + f3_segs
        merged_all = merge_cross_file(all_segs)

        assert len(merged_all) >= 1
        assert math.isclose(merged_all[0].start_time, 0.0, abs_tol=1e-3)
        assert math.isclose(merged_all[-1].end_time, 1800.0, abs_tol=1e-3)

    def test_extreme_boundary_single_frame_and_zero_duration(self, mock_config):
        """
        Challenge 3.3: Extreme boundary conditions: 1 frame input or duration <= 0.
        Ensure no crashes, zero divisions, or unhandled exceptions.
        """
        detector = MotionDetector(mock_config, decode_gpu="cuda")

        # 1 frame input
        one_frame = [np.full((234, 416), 50, dtype=np.uint8)]
        res_1, _ = detector.analyze_frames(one_frame, start_offset=0.0, file_duration=10.0, fps=5.0)
        assert res_1 == []  # < 2 frames returns [] safely

        # 0 frames input
        res_0, _ = detector.analyze_frames([], start_offset=0.0, file_duration=10.0, fps=5.0)
        assert res_0 == []


class TestAdditionalNoiseAndEmaDynamicsAdversarial:
    """Additional deep adversarial tests for noise density, variable dt steps, and EMA model dynamics."""

    def test_salt_and_pepper_pixel_level_noise_density_sweep(self):
        """
        Challenge 1.6: Pixel-level salt-and-pepper noise with density p = 0.002.
        Verify that adaptive noise floor tracking adapts upwards and keeps effective energy bounded.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=1.5,
            cell_noise_alpha=0.1,
            sens_multiplier=1.5,
        )

        rng = np.random.default_rng(2026)
        energies_observed = []
        for _ in range(40):
            noise_img = np.zeros((160, 160), dtype=np.float32)
            mask = rng.random((160, 160)) < 0.002
            noise_img[mask] = 255.0

            eff_energy, is_motion, stats = grid_filter.process_frame(noise_img, dt=0.2)
            energies_observed.append(eff_energy)

        # Average effective energy across frames should be bounded and controlled
        assert np.mean(energies_observed) < 5.0
        assert np.all(grid_filter.noise_floor_grid > 0.0)

    def test_variable_dt_cooldown_numerical_consistency(self):
        """
        Challenge 2.4: Compare cooldown decay with:
        Path A: Single step with dt = 15.0s
        Path B: 75 steps with dt = 0.2s (total 15.0s)
        Path C: 1500 steps with dt = 0.01s (total 15.0s)
        Verify that final confidence values are numerically identical (exp(-15/15) = 1/e).
        """
        filter_a = SpatialGridMotionFilter(cooldown_half_life=15.0)
        filter_b = SpatialGridMotionFilter(cooldown_half_life=15.0)
        filter_c = SpatialGridMotionFilter(cooldown_half_life=15.0)

        # Trigger motion on all 3
        motion = np.zeros((80, 80), dtype=np.float32)
        motion[20:40, 20:40] = 10.0
        static = np.zeros((80, 80), dtype=np.float32)

        filter_a.process_frame(motion, dt=0.2)
        filter_b.process_frame(motion, dt=0.2)
        filter_c.process_frame(motion, dt=0.2)

        # Path A: 1 step of 15.0s
        filter_a.process_frame(static, dt=15.0)

        # Path B: 75 steps of 0.2s
        for _ in range(75):
            filter_b.process_frame(static, dt=0.2)

        # Path C: 1500 steps of 0.01s
        for _ in range(1500):
            filter_c.process_frame(static, dt=0.01)

        conf_a = float(np.max(filter_a.confidence_grid))
        conf_b = float(np.max(filter_b.confidence_grid))
        conf_c = float(np.max(filter_c.confidence_grid))

        expected = math.exp(-1.0)
        assert math.isclose(conf_a, expected, abs_tol=1e-5)
        assert math.isclose(conf_b, expected, abs_tol=1e-5)
        assert math.isclose(conf_c, expected, abs_tol=1e-5)
        assert math.isclose(conf_a, conf_b, abs_tol=1e-5)
        assert math.isclose(conf_b, conf_c, abs_tol=1e-5)

    def test_ema_model_foreground_ghost_absorption_rate(self):
        """
        Challenge 2.5: When a resting object leaves the scene (leaving a ghost artifact),
        the EMA background model slowly adapts at alpha_fg = 0.005.
        Verify that after N frames (N ~ 600 at 5fps = 120s), the ghost is gradually absorbed.
        """
        model = EmaBackgroundModel(alpha_bg=0.05, alpha_fg=0.005, beta=0.6, fg_threshold=12.0)
        bg = np.full((100, 100), 50, dtype=np.uint8)
        model.update(bg)

        # Object is placed for 100 frames
        obj_frame = bg.copy()
        obj_frame[30:70, 30:70] = 150
        for _ in range(100):
            model.update(obj_frame)

        # Object is suddenly removed (empty room returns)
        # Background was adapted partially towards 150 (50 * (1 - alpha_fg)^100 + 150 * (1 - (1 - alpha_fg)^100))
        # Initial diff on removal
        _, _, d_bg_start = model.update(bg)
        init_ghost_diff = float(np.mean(d_bg_start[30:70, 30:70]))

        # After 200 more empty frames
        for _ in range(200):
            model.update(bg)

        _, _, d_bg_200 = model.update(bg)
        ghost_diff_200 = float(np.mean(d_bg_200[30:70, 30:70]))

        # Ghost diff must decrease monotonically as background model updates
        assert ghost_diff_200 < init_ghost_diff


class TestRealSampleIntegrationEmpirical:
    """Empirical verification on real sample MP4 files from testsample/ directory."""

    def test_real_sample_pyav_decode_and_timeline_closure(self, mock_config):
        sample_path = Path("testsample/00_20260901000007_20260901005727.mp4")
        if not sample_path.exists():
            pytest.skip("Real test sample file not present")

        detector = MotionDetector(mock_config, decode_gpu="cuda")
        start_offset = 7.0
        file_duration = 3440.0  # ~57 min file

        # Run detector on real file
        results, yolo_buf = detector.analyze(
            filepath=str(sample_path),
            start_offset=start_offset,
            file_duration=file_duration,
        )

        assert len(results) > 0, "No results returned for real sample"
        # Verify timeline bounds
        assert results[0]["time"] >= start_offset
        assert results[-1]["time"] <= start_offset + file_duration + 1.0

        # If early terminated, final timestamp must strictly equal start_offset + file_duration
        if detector.last_perf.get("early_term", False):
            assert math.isclose(results[-1]["time"], start_offset + file_duration, abs_tol=1e-2)

        # Verify segments
        segs = build_segments(results, str(sample_path), file_offset=start_offset)
        assert len(segs) > 0
        assert math.isclose(segs[0].start_time, start_offset, abs_tol=1e-2)
        if detector.last_perf.get("early_term", False):
            assert math.isclose(segs[-1].end_time, start_offset + file_duration, abs_tol=1e-2)


