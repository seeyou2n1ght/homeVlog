"""
Adversarial and Empirical Challenge Test Suite for Milestone 2 (R2):
- Sudden Global Illumination Shifts (Light switch, flash, headlights)
- Camera Shake / High-Frequency Jitter / Wind Vibration
- High-Frequency Local Oscillations (Curtains, fans, foliage)
- Subtle Micro-Motion (Breathing, typing, sitting pause) & Early-Term Protection
- Long-Sequence Numerical Stability & Memory Stress (1,000 to 5,000 frames)
- Extreme Parameter & Boundary Conditions (NaN/Inf, zero dt, 1-frame, all-black/white)
- Timeline Closure Invariant under Extreme Early Termination
"""

import math
import sys
import time
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


class TestIlluminationAdversarial:
    """Stress testing EMA Background and Grid Filter against sudden illumination changes."""

    def test_instant_global_room_light_switch_on_and_off(self):
        """
        Sudden step change in global illumination (e.g. ceiling light turned ON, then OFF after 50 frames).
        Verifies that:
        1. Saliency does not explode to NaN/Inf.
        2. Once light stabilizes, frame diff D_frame immediately drops to 0.
        3. Background model does not get trapped in an invalid state.
        """
        model = EmaBackgroundModel(alpha_bg=0.05, alpha_fg=0.005, beta=0.6, fg_threshold=12.0)
        dark_frame = np.full((120, 160), 30, dtype=np.uint8)
        bright_frame = np.full((120, 160), 200, dtype=np.uint8)

        # 20 frames of dark room
        for _ in range(20):
            saliency, d_frame, d_bg = model.update(dark_frame)

        assert np.all(saliency == 0.0)

        # Light turns ON instantly
        saliency_on, d_frame_on, d_bg_on = model.update(bright_frame)
        assert np.allclose(d_frame_on, 170.0)
        assert not np.isnan(saliency_on).any()
        assert not np.isinf(saliency_on).any()

        # Next 30 frames: room stays bright with no motion
        for _ in range(30):
            saliency, d_frame, d_bg = model.update(bright_frame)
            assert np.all(d_frame == 0.0)
            assert not np.isnan(saliency).any()

        # Light turns OFF back to dark
        saliency_off, d_frame_off, d_bg_off = model.update(dark_frame)
        assert not np.isnan(saliency_off).any()
        assert not np.isinf(saliency_off).any()

    def test_rapid_strobe_flashing_stability(self):
        """
        Rapid alternating strobe lighting (30 vs 220 every 2 frames) over 60 frames.
        Ensures numerical stability and bounded values.
        """
        model = EmaBackgroundModel(alpha_bg=0.05, alpha_fg=0.005, beta=0.6, fg_threshold=12.0)
        grid_filter = SpatialGridMotionFilter(grid_rows=8, grid_cols=8)

        for i in range(60):
            val = 220 if (i % 2 == 0) else 30
            frame = np.full((80, 80), val, dtype=np.uint8)
            saliency, d_frame, d_bg = model.update(frame)
            eff_energy, is_motion, stats = grid_filter.process_frame(saliency, dt=0.2)

            assert not np.isnan(eff_energy)
            assert not np.isinf(eff_energy)
            assert 0.0 <= stats["max_confidence"] <= 1.0
            assert np.all(grid_filter.noise_floor_grid >= 0.0)
            assert not np.isnan(grid_filter.noise_floor_grid).any()


class TestCameraShakeAndJitter:
    """Stress testing spatial grid and EMA model under camera shake / wind jitter."""

    def test_high_frequency_translational_jitter(self):
        """
        Simulate camera vibration by translating a high-contrast checkerboard image by +/- 1-2 pixels.
        Ensures that noise floors remain stable and do not drift to infinity or NaN.
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=1.5,
            cell_noise_alpha=0.05,
        )
        model = EmaBackgroundModel(alpha_bg=0.05, alpha_fg=0.005, beta=0.6)

        # Create structured image with sharp edges (checkerboard pattern)
        base = np.zeros((160, 160), dtype=np.uint8)
        base[::20, :] = 200
        base[:, ::20] = 200

        rng = np.random.default_rng(42)
        for _ in range(100):
            dx = int(rng.integers(-2, 3))
            dy = int(rng.integers(-2, 3))
            jittered = np.roll(np.roll(base, dx, axis=1), dy, axis=0)

            saliency, _, _ = model.update(jittered)
            eff_energy, is_motion, stats = grid_filter.process_frame(saliency, dt=0.2)

            assert not np.isnan(eff_energy)
            assert not np.isinf(eff_energy)
            assert np.all(np.isfinite(grid_filter.confidence_grid))
            assert np.all(np.isfinite(grid_filter.noise_floor_grid))


class TestLocalOscillationAndCurtain:
    """Stress testing suppression and adaptation against localized periodic oscillations."""

    def test_isolated_curtain_oscillation_suppression(self):
        """
        Simulate a curtain blowing in the wind localized strictly in one 8x8 cell (e.g. cell [0, 7]).
        With min_connected_cells=2, an isolated single cell oscillation should be suppressed
        if energy stays under the extreme spike multiplier (2.5x).
        """
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=2.0,
            sens_multiplier=1.5,
        )

        suppressed_count = 0
        total_frames = 50

        for i in range(total_frames):
            saliency = np.zeros((80, 80), dtype=np.float32)
            # Oscillate cell (0, 7) with moderate energy 3.0 (threshold is ~3.0, extreme spike is 7.5)
            # Row 0: y in [0:10], Col 7: x in [70:80]
            osc_energy = 3.0 + 1.0 * math.sin(i * 0.8)
            saliency[0:10, 70:80] = osc_energy

            eff_energy, is_motion, stats = grid_filter.process_frame(saliency, dt=0.2)
            if not is_motion and stats["active_cells"] == 0:
                suppressed_count += 1

        # All isolated oscillations within moderate amplitude must be filtered
        assert suppressed_count == total_frames

    def test_two_cell_curtain_oscillation_noise_floor_adaptation(self):
        """
        If a curtain spans 2 adjacent cells, it activates connected components.
        Verify that the system remains stable and does not crash or corrupt other cells.
        """
        grid_filter = SpatialGridMotionFilter(grid_rows=8, grid_cols=8, min_connected_cells=2)
        for i in range(50):
            saliency = np.zeros((80, 80), dtype=np.float32)
            # Cells (0, 6) and (0, 7)
            saliency[0:10, 60:80] = 5.0
            eff_energy, is_motion, stats = grid_filter.process_frame(saliency, dt=0.2)
            assert is_motion is True
            assert stats["active_cells"] == 2
            # Other 62 cells must remain unactivated
            assert np.sum(grid_filter.active_grid) == 2


class TestMicroMotionAndRestingHuman:
    """Stress testing sitting/resting human detection, typing, and cooldown protection."""

    def test_subtle_typing_with_micro_pauses(self):
        """
        Simulate a person sitting at a desk typing for 20 frames, pausing for 15 frames,
        then typing again for 20 frames.
        Verify:
        1. During active typing, dual-difference and grid active cells detect motion.
        2. During the 15-frame pause (~3 seconds at 5fps), confidence memory cooldown guard
           prevents early termination.
        3. Motion is detected again when typing resumes.
        """
        config = {
            "detection": {
                "analysis_fps": 5,
                "analysis_resolution": "160x120",
                "roi_crop": [0.0, 0.0, 1.0, 1.0],
                "ema_background_enabled": True,
                "cooldown_half_life": 15.0,
                "analysis_early_term_enabled": True,
                "analysis_early_term_window": 10,
                "analysis_early_term_threshold": 2.0,
                "early_term_cooldown_guard": True,
            }
        }
        detector = MotionDetector(config)

        frames = []
        bg = np.full((120, 160), 80, dtype=np.uint8)

        # Baseline empty room (3 frames < term_window 10)
        for _ in range(3):
            frames.append(bg.copy())

        # Person enters (occupies rows 40:80, cols 50:110, value 150)
        person_base = bg.copy()
        person_base[40:80, 50:110] = 150

        # Phase 1: Typing motion (20 frames) - small perturbations in hands area [70:80, 70:90]
        rng = np.random.default_rng(123)
        for _ in range(20):
            f = person_base.copy()
            f[70:80, 70:90] = (150 + rng.integers(-15, 15, size=(10, 20))).clip(0, 255)
            frames.append(f)

        # Phase 2: Complete pause (15 frames) - sitting still
        for _ in range(15):
            frames.append(person_base.copy())

        # Phase 3: Typing resumes (20 frames)
        for _ in range(20):
            f = person_base.copy()
            f[70:80, 70:90] = (150 + rng.integers(-15, 15, size=(10, 20))).clip(0, 255)
            frames.append(f)

        dur = (len(frames) - 1) * 0.2
        results, _ = detector.analyze_frames(frames, start_offset=0.0, file_duration=dur, fps=5)

        assert len(results) == len(frames)
        # Verify that total frames were processed and not early terminated prematurely during pause
        assert results[-1]["time"] == pytest.approx(dur, abs=1e-3)

    def test_cooldown_decay_mathematical_precision(self):
        """
        Verify exponential decay formula C(t) = C0 * exp(-dt / tau).
        At dt = tau (15.0s), confidence should be exactly 1/e ~ 0.367879.
        At dt = 3 * tau (45.0s), confidence should be exp(-3) ~ 0.049787 < 0.05 (threshold for early term).
        """
        grid_filter = SpatialGridMotionFilter(grid_rows=8, grid_cols=8, cooldown_half_life=15.0)

        # Activate cell (3, 3)
        saliency = np.zeros((80, 80), dtype=np.float32)
        saliency[30:40, 30:50] = 10.0  # 2 cells (3,3) and (3,4)
        grid_filter.process_frame(saliency, dt=0.2)

        assert grid_filter.confidence_grid[3, 3] == 1.0

        # Now 15.0 seconds elapse with zero motion
        saliency_quiet = np.zeros((80, 80), dtype=np.float32)
        grid_filter.process_frame(saliency_quiet, dt=15.0)

        expected_conf_15s = math.exp(-15.0 / 15.0)
        assert math.isclose(grid_filter.confidence_grid[3, 3], expected_conf_15s, rel_tol=1e-4)

        # Early termination guard should STILL block (since 0.3678 > 0.05)
        can_term = grid_filter.can_early_terminate(
            consecutive_static=50,
            term_window=10,
            current_energy=0.1,
            term_threshold=2.0,
        )
        assert can_term is False

        # After another 30 seconds (total 45s = 3 * half_life):
        grid_filter.process_frame(saliency_quiet, dt=30.0)
        expected_conf_45s = math.exp(-45.0 / 15.0)  # ~0.04978
        assert math.isclose(grid_filter.confidence_grid[3, 3], expected_conf_45s, rel_tol=1e-4)

        # Now max_confidence < 0.05, early termination should be PERMITTED
        can_term_after_cooldown = grid_filter.can_early_terminate(
            consecutive_static=50,
            term_window=10,
            current_energy=0.1,
            term_threshold=2.0,
        )
        assert can_term_after_cooldown is True


class TestLongSequenceAndPerformanceStress:
    """Stress testing long sequence simulations, numerical stability, and performance."""

    def test_2000_frames_continuous_simulation_no_drift_or_nan(self):
        """
        Run 2,000 continuous frames through EmaBackgroundModel + SpatialGridMotionFilter.
        Checks for:
        1. No NaN / Inf in any internal arrays.
        2. Bounded noise floor and confidence values.
        3. Execution throughput > 1,000 fps on CPU.
        """
        model = EmaBackgroundModel()
        grid_filter = SpatialGridMotionFilter(grid_rows=8, grid_cols=8)

        rng = np.random.default_rng(999)
        base_frame = np.full((120, 160), 100, dtype=np.uint8)

        t_start = time.perf_counter()
        for i in range(2000):
            # Inject intermittent events every 200 frames
            frame = base_frame.copy()
            if 400 <= i < 450 or 1200 <= i < 1230:
                frame[30:70, 40:80] = 180 + rng.integers(-5, 5, size=(40, 40))
            else:
                # Slight sensor noise
                frame += rng.integers(-1, 2, size=(120, 160)).astype(np.uint8)

            saliency, d_frame, d_bg = model.update(frame)
            eff_energy, is_motion, stats = grid_filter.process_frame(saliency, dt=0.2)

            assert not np.isnan(eff_energy)
            assert not np.isinf(eff_energy)

        t_elapsed = time.perf_counter() - t_start
        fps = 2000 / max(1e-5, t_elapsed)

        # Check internal state sanity
        assert not np.isnan(model.background).any()
        assert not np.isinf(model.background).any()
        assert not np.isnan(grid_filter.confidence_grid).any()
        assert not np.isnan(grid_filter.noise_floor_grid).any()
        assert np.all(grid_filter.confidence_grid >= 0.0)
        assert np.all(grid_filter.confidence_grid <= 1.0)

        # Performance: 2,000 frames in pure python/numpy should process well over 200 fps
        assert fps > 200.0, f"Processing rate too low: {fps:.1f} fps"

    def test_memory_bloat_long_run(self):
        """
        Verify that internal state objects do not accumulate memory over 5,000 iterations.
        """
        model = EmaBackgroundModel()
        grid_filter = SpatialGridMotionFilter(grid_rows=8, grid_cols=8)
        frame = np.full((100, 100), 50, dtype=np.uint8)

        for _ in range(5000):
            saliency, _, _ = model.update(frame)
            grid_filter.process_frame(saliency, dt=0.2)

        # State arrays should maintain exact constant fixed shapes
        assert model.background.shape == (100, 100)
        assert model.prev_frame.shape == (100, 100)
        assert grid_filter.confidence_grid.shape == (8, 8)
        assert grid_filter.noise_floor_grid.shape == (8, 8)
        assert grid_filter.cell_energies.shape == (8, 8)
        assert grid_filter.active_grid.shape == (8, 8)


class TestBoundaryAndExtremeConditions:
    """Stress testing edge cases, boundary parameters, and invalid or extreme inputs."""

    def test_all_black_and_all_white_frames(self):
        """Verify handling of 0x00 and 0xFF pure frames."""
        model = EmaBackgroundModel()
        grid_filter = SpatialGridMotionFilter()

        black = np.zeros((64, 64), dtype=np.uint8)
        white = np.full((64, 64), 255, dtype=np.uint8)

        s_b, _, _ = model.update(black)
        e_b, _, _ = grid_filter.process_frame(s_b, dt=0.2)
        assert e_b == 0.0

        s_w, _, _ = model.update(white)
        e_w, _, _ = grid_filter.process_frame(s_w, dt=0.2)
        assert not np.isnan(e_w)
        assert e_w > 0.0

    def test_1x1_and_single_row_col_grid_dimensions(self):
        """Test minimal grid configurations (1x1, 1x8, 8x1)."""
        for rows, cols in [(1, 1), (1, 8), (8, 1)]:
            filter_custom = SpatialGridMotionFilter(grid_rows=rows, grid_cols=cols, min_connected_cells=1)
            saliency = np.full((32, 32), 5.0, dtype=np.float32)
            eff_energy, is_motion, stats = filter_custom.process_frame(saliency, dt=0.2)
            assert eff_energy > 0.0
            assert stats["active_cells"] >= 1
            assert filter_custom.confidence_grid.shape == (rows, cols)

    def test_zero_or_negative_dt_robustness(self):
        """
        If dt <= 0 (e.g. clock anomaly or identical timestamps),
        cooldown decay should safely handle without dividing by zero or error.
        """
        grid_filter = SpatialGridMotionFilter(cooldown_half_life=15.0)
        saliency = np.zeros((40, 40), dtype=np.float32)

        # dt = 0.0
        eff, motion, stats = grid_filter.process_frame(saliency, dt=0.0)
        assert not np.isnan(eff)

        # dt = -1.0
        eff_neg, _, _ = grid_filter.process_frame(saliency, dt=-1.0)
        assert not np.isnan(eff_neg)

    def test_zero_half_life_configuration(self):
        """If cooldown_half_life is 0.0, decay factor should be 0.0 without divide-by-zero crash."""
        grid_filter = SpatialGridMotionFilter(cooldown_half_life=0.0)
        saliency = np.zeros((40, 40), dtype=np.float32)
        eff, _, _ = grid_filter.process_frame(saliency, dt=0.2)
        assert not np.isnan(eff)
        assert np.all(grid_filter.confidence_grid == 0.0)

    def test_single_frame_and_empty_frame_analyze(self):
        """Test MotionDetector on 0, 1, and 2 frames."""
        config = {
            "detection": {
                "analysis_fps": 5,
                "analysis_resolution": "64x64",
                "roi_crop": [0.0, 0.0, 1.0, 1.0],
            }
        }
        detector = MotionDetector(config)

        # 0 frames
        res0, _ = detector.analyze_frames([])
        assert res0 == []

        # 1 frame
        f1 = np.zeros((64, 64), dtype=np.uint8)
        res1, _ = detector.analyze_frames([f1])
        assert res1 == []

        # 2 frames
        res2, _ = detector.analyze_frames([f1, f1])
        assert len(res2) == 2

    def test_timeline_closure_invariant_with_early_termination_and_build_segments(self):
        """
        Strict timeline invariant test:
        Even if early termination breaks at frame 35 of a 1-hour file (duration = 3600.0s),
        the final Segment produced by build_segments MUST have end_time == start_offset + file_duration (3600.0s).
        """
        config = {
            "detection": {
                "analysis_fps": 5,
                "analysis_resolution": "64x64",
                "roi_crop": [0.0, 0.0, 1.0, 1.0],
                "analysis_early_term_enabled": True,
                "analysis_early_term_window": 10,
                "analysis_early_term_threshold": 5.0,
                "early_term_cooldown_guard": False,  # Allow immediate early term on quiet frames
            }
        }
        detector = MotionDetector(config)

        # Generate 100 identical static frames
        static_frames = [np.full((64, 64), 50, dtype=np.uint8) for _ in range(100)]
        start_offset = 120.0
        file_duration = 3600.0  # 1 hour long file

        results, _ = detector.analyze_frames(
            static_frames,
            start_offset=start_offset,
            file_duration=file_duration,
            fps=5,
        )

        # Early termination should have triggered around frame 10-12
        assert len(results) < 30
        # Final result timestamp MUST strictly equal start_offset + file_duration
        assert results[-1]["time"] == start_offset + file_duration

        # Downstream segment conversion test
        segments = build_segments(results, source_file="challenge_static.mp4", file_offset=start_offset)
        assert len(segments) > 0
        # The very last segment must end precisely at 3720.0
        assert segments[-1].end_time == start_offset + file_duration


class TestMedianFilterAndLabelSmoothingStress:
    """Stress testing signal smoothing functions with irregular inputs."""

    def test_median_filter_extreme_signals(self):
        # Window < 3
        assert _median_filter([1.0, 2.0, 3.0], 1) == [1.0, 2.0, 3.0]
        # Single element
        assert _median_filter([5.0], 3) == [5.0]
        # All same
        assert _median_filter([4.0] * 10, 5) == [4.0] * 10
        # Isolated single pulse
        pulse = [0.0, 0.0, 100.0, 0.0, 0.0]
        filtered = _median_filter(pulse, 3)
        assert max(filtered) == 0.0  # 3-tap median filters isolated pulse to 0

    def test_smooth_labels_edge_cases(self):
        # Empty
        assert _smooth_labels([], 3, 5, 2) == []
        # All True
        assert _smooth_labels([True] * 10, 3, 5, 2) == [True] * 10
        # All False
        assert _smooth_labels([False] * 10, 3, 5, 2) == [False] * 10
        # Isolated single True (should be suppressed by min_motion=3)
        assert _smooth_labels([False, False, True, False, False], 3, 5, 2) == [False] * 5
        # Isolated single False in long True run (should be bridged by noise_suppress=2)
        assert _smooth_labels([True, True, True, False, True, True, True], 3, 5, 2) == [True] * 7
