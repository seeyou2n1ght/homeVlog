"""核心时空滤波、滑动平均背景建模与音频信号检测算法模块.

包含：
1. EmaBackgroundModel: 双差分滑动背景前景分离模型；
2. SpatialGridMotionFilter: 8x8 空间网格滤波、8-邻域连通域降噪与置信度记忆冷却；
3. AudioEnergyVAD: 短时 RMS 能量分帧与自适应音频活动检测器；
4. _median_filter & _smooth_labels: 时域能量中值滤波与动作状态平滑。
"""

import numpy as np


class EmaBackgroundModel:
    """
    Lightweight Temporal Sliding Background Model with Selective EMA Update.

    Mathematical Model:
        B_t(x, y) = (1 - alpha(x, y)) * B_{t-1}(x, y) + alpha(x, y) * I_t(x, y)
        where:
          - alpha(x, y) = alpha_fg (~0.005) when |I_t(x, y) - B_{t-1}(x, y)| > fg_threshold (foreground motion)
          - alpha(x, y) = alpha_bg (~0.05) when pixel is static background

    Gated Dual-difference Motion Saliency:
        D_frame(x, y) = |I_t(x, y) - I_{t-1}(x, y)|
        D_bg(x, y)    = |I_t(x, y) - B_t(x, y)|
        M_t(x, y)     = D_frame(x, y) + beta * min(D_frame(x, y), D_bg(x, y))  (beta ~ 0.6)
    """

    def __init__(
        self,
        alpha_bg: float = 0.05,
        alpha_fg: float = 0.005,
        beta: float = 0.6,
        fg_threshold: float = 12.0,
    ):
        self.alpha_bg = float(alpha_bg)
        self.alpha_fg = float(alpha_fg)
        self.beta = float(beta)
        self.fg_threshold = float(fg_threshold)
        self.background: np.ndarray | None = None
        self.prev_frame: np.ndarray | None = None

    def reset(self) -> None:
        """Reset background model and previous frame cache."""
        self.background = None
        self.prev_frame = None

    def update(self, gray_roi: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Update EMA background model with incoming grayscale ROI frame.

        Args:
            gray_roi: 2D NumPy array representing the grayscale region of interest.

        Returns:
            tuple of (saliency_map, d_frame, d_bg) as float32 arrays with identical shape.
        """
        roi_float = np.squeeze(gray_roi).astype(np.float32)
        if self.background is None or self.prev_frame is None:
            self.background = roi_float.copy()
            self.prev_frame = roi_float.copy()
            zeros = np.zeros_like(roi_float)
            return zeros, zeros, zeros

        # 1. Instantaneous frame difference: D_frame = |I_t - I_{t-1}|
        d_frame = np.abs(roi_float - self.prev_frame)

        # 2. Prior background difference: D_bg_prior = |I_t - B_{t-1}|
        d_bg_prior = np.abs(roi_float - self.background)

        # 3. Selective EMA background update based on prior foreground segmentation
        alpha_map = np.where(d_bg_prior > self.fg_threshold, self.alpha_fg, self.alpha_bg)
        self.background = (1.0 - alpha_map) * self.background + alpha_map * roi_float

        # 4. Posterior background difference: D_bg = |I_t - B_t|
        d_bg = np.abs(roi_float - self.background)

        # 5. Gated dual-difference motion saliency: Frame difference (temporal motion)
        # is the primary driver. Background difference (presence) boosts moving foreground
        # without hallucinating false motion when an object or person is stationary.
        saliency_map = d_frame + self.beta * np.minimum(d_frame, d_bg)

        self.prev_frame = roi_float
        return saliency_map, d_frame, d_bg


class SpatialGridMotionFilter:
    """
    Dynamic 8x8 Spatial Grid Energy Extractor, Connected Component Noise Filter,
    and Spatial-Temporal Confidence Memory Cooldown Manager.

    Features:
    - Divides ROI into grid_rows x grid_cols (default 8x8 = 64 cells).
    - Tracks per-cell adaptive noise floor using slow EMA (alpha ~ 0.02).
    - 8-neighborhood connected component labeling: filters isolated single-cell spikes.
    - Cluster boost: amplifies genuine multi-cell clustered target movement.
    - Spatial-temporal confidence memory grid C_{r,c}(t) with exponential half-life decay.
    - Guards early termination: only permitted when consecutive static frames >= window,
      current energy < threshold, and all regional confidence has decayed (max C_{r,c} < 0.05).
    """

    def __init__(
        self,
        grid_rows: int = 8,
        grid_cols: int = 8,
        cooldown_half_life: float = 15.0,
        min_connected_cells: int = 2,
        cell_noise_alpha: float = 0.02,
        sens_multiplier: float = 1.5,
        base_noise_thresh: float = 1.5,
        cluster_boost: float = 1.2,
        ambient_drift_suppress: bool = True,
        ambient_drift_active_ratio: float = 0.35,
        ambient_drift_max_energy: float = 5.0,
    ):
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.cooldown_half_life = float(cooldown_half_life)
        self.min_connected_cells = int(min_connected_cells)
        self.cell_noise_alpha = float(cell_noise_alpha)
        self.sens_multiplier = float(sens_multiplier)
        self.base_noise_thresh = float(base_noise_thresh)
        self.cluster_boost = float(cluster_boost)
        self.ambient_drift_suppress = bool(ambient_drift_suppress)
        self.ambient_drift_active_ratio = float(ambient_drift_active_ratio)
        self.ambient_drift_max_energy = float(ambient_drift_max_energy)

        self.confidence_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        self.noise_floor_grid = np.full((self.grid_rows, self.grid_cols), self.base_noise_thresh, dtype=np.float32)
        self.cell_energies = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)
        self.active_grid = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)

    def reset(self) -> None:
        """Reset internal grid state and noise floors."""
        self.confidence_grid.fill(0.0)
        self.noise_floor_grid.fill(self.base_noise_thresh)
        self.cell_energies.fill(0.0)
        self.active_grid.fill(False)

    def extract_grid_energies(self, saliency_map: np.ndarray) -> np.ndarray:
        """Partition saliency map into grid_rows x grid_cols and compute per-cell mean energy."""
        import cv2
        h, w = saliency_map.shape[:2]
        row_edges = np.linspace(0, h, self.grid_rows + 1, dtype=int)
        col_edges = np.linspace(0, w, self.grid_cols + 1, dtype=int)
        # One native pass replaces 64 Python/NumPy reductions per frame.
        # Preserve linspace boundaries, including empty cells in tiny ROIs.
        integral = cv2.integral(saliency_map, sdepth=cv2.CV_64F)
        corners = integral[np.ix_(row_edges, col_edges)]
        sums = corners[1:, 1:] - corners[:-1, 1:] - corners[1:, :-1] + corners[:-1, :-1]
        areas = np.diff(row_edges)[:, None] * np.diff(col_edges)[None, :]
        means = np.divide(sums, areas, out=np.zeros_like(sums), where=areas > 0)
        return means.astype(np.float32)

    def find_connected_components(self, binary_grid: np.ndarray) -> list[set[tuple[int, int]]]:
        """Extract 8-connected components from a 2D boolean grid."""
        rows, cols = binary_grid.shape
        visited = np.zeros((rows, cols), dtype=bool)
        components: list[set[tuple[int, int]]] = []

        for r in range(rows):
            for c in range(cols):
                if binary_grid[r, c] and not visited[r, c]:
                    comp: set[tuple[int, int]] = set()
                    queue = [(r, c)]
                    visited[r, c] = True
                    while queue:
                        cr, cc = queue.pop(0)
                        comp.add((cr, cc))
                        for dr in (-1, 0, 1):
                            for dc in (-1, 0, 1):
                                if dr == 0 and dc == 0:
                                    continue
                                nr, nc = cr + dr, cc + dc
                                if 0 <= nr < rows and 0 <= nc < cols:
                                    if binary_grid[nr, nc] and not visited[nr, nc]:
                                        visited[nr, nc] = True
                                        queue.append((nr, nc))
                            components.append(comp)
        return components

    def process_frame(
        self,
        saliency_map: np.ndarray,
        dt: float,
        is_night_mode: bool = False,
    ) -> tuple[float, bool, dict]:
        """Process a single frame's motion saliency map with optional night mode sensitivity."""
        energies = self.extract_grid_energies(saliency_map)
        self.cell_energies = energies

        # 1. Raw cell activation against adaptive per-cell noise floor
        cell_thresholds = np.maximum(
            self.base_noise_thresh,
            self.noise_floor_grid * self.sens_multiplier,
        )
        raw_active = energies > cell_thresholds

        # 2. 8-Neighborhood connected component filtering & spatial noise suppression
        total_cells = self.grid_rows * self.grid_cols
        raw_active_ratio = float(np.sum(raw_active)) / max(1, total_cells)
        mean_floor = float(np.mean(self.noise_floor_grid))
        is_global_flash = bool(raw_active_ratio >= 0.85 and (global_mean := float(np.mean(saliency_map)) if saliency_map.size > 0 else 0.0) > max(3.0, mean_floor * 2.5))

        filtered_active = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)

        min_conn = 1 if is_night_mode else self.min_connected_cells
        isolated_multiplier = 1.6 if is_night_mode else 2.5

        if not is_global_flash:
            components = self.find_connected_components(raw_active)
            for comp in components:
                if len(comp) >= min_conn:
                    # Contiguous cluster: confirmed genuine target movement
                    for r, c in comp:
                        filtered_active[r, c] = True
                else:
                    # Isolated single-cell: filter unless it is an extreme high-magnitude spike
                    for r, c in comp:
                        if energies[r, c] > cell_thresholds[r, c] * isolated_multiplier:
                            filtered_active[r, c] = True

        # 白天大面积慢速光影干扰（Sunrise/Sunset Ambient Drift）检测与软抑制：
        # 当激活网格占比较大 (>= 35%)，但全局网格方差系数极低（光照均匀漂移无局部焦点），
        # 且全图最高单元能量低于显著运动阈值 (< 5.0) 时，判定为自然光照漫射偏转，避免触发数分钟虚假动态审核。
        is_ambient_drift = False
        num_raw_active = int(np.sum(filtered_active))
        if (
            self.ambient_drift_suppress
            and not is_night_mode
            and num_raw_active >= int(total_cells * self.ambient_drift_active_ratio)
        ):
            max_cell_e = float(np.max(energies))
            if max_cell_e < self.ambient_drift_max_energy:
                active_vals = energies[filtered_active]
                mean_act = float(np.mean(active_vals)) if active_vals.size > 0 else 0.0
                std_act = float(np.std(active_vals)) if active_vals.size > 0 else 0.0
                focal_ratio = max_cell_e / (mean_act + 1e-4)
                cov = std_act / (mean_act + 1e-4)
                if focal_ratio < 1.75 and cov < 0.35:
                    is_ambient_drift = True
                    filtered_active.fill(False)

        self.active_grid = filtered_active
        num_active_cells = int(np.sum(filtered_active))

        # 3. Adaptive noise floor tracking on inactive cells
        inactive_mask = ~filtered_active
        if np.any(inactive_mask):
            self.noise_floor_grid[inactive_mask] = (
                (1.0 - self.cell_noise_alpha) * self.noise_floor_grid[inactive_mask]
                + self.cell_noise_alpha * energies[inactive_mask]
            )
            # 底噪限幅保护：防止传感器高 ISO 热噪过度抬高底噪门限而淹没真实人体微动
            max_allowed_floor = float(self.base_noise_thresh * 2.2) if not is_night_mode else float(self.base_noise_thresh * 1.8)
            self.noise_floor_grid = np.minimum(self.noise_floor_grid, max_allowed_floor)

        # 4. Spatial-temporal confidence decay grid C_{r,c}(t)
        decay_factor = (
            float(np.exp(-dt / max(0.1, self.cooldown_half_life)))
            if self.cooldown_half_life > 0
            else 0.0
        )
        self.confidence_grid[filtered_active] = 1.0
        self.confidence_grid[~filtered_active] *= decay_factor

        # 5. Compute frame-level effective energy
        global_mean = float(np.mean(saliency_map)) if saliency_map.size > 0 else 0.0
        if num_active_cells > 0:
            active_energy_mean = float(np.mean(energies[filtered_active])) * self.cluster_boost
            effective_energy = max(global_mean, active_energy_mean)
            is_motion = True
        else:
            effective_energy = min(global_mean, float(np.mean(energies))) * 0.4
            is_motion = False

        stats = {
            "active_cells": num_active_cells,
            "max_confidence": float(np.max(self.confidence_grid)),
            "mean_noise_floor": float(np.mean(self.noise_floor_grid)),
            "max_cell_energy": float(np.max(energies)),
            "effective_energy": effective_energy,
            "is_global_flash": is_global_flash,
            "is_ambient_drift": is_ambient_drift,
        }
        return effective_energy, is_motion, stats


    def should_terminate_early(
        self,
        consecutive_static: int,
        current_energy: float,
        term_window: int = 50,
        term_threshold: float = 0.8,
    ) -> bool:
        """Anti-Miss Early Termination Guard."""
        if consecutive_static < term_window:
            return False
        if current_energy >= term_threshold:
            return False
        max_conf = float(np.max(self.confidence_grid))
        if max_conf >= 0.05:
            return False
        return True

    def can_early_terminate(
        self,
        consecutive_static: int,
        term_window: int = 50,
        current_energy: float = 0.0,
        term_threshold: float = 0.8,
    ) -> bool:
        """Alias for should_terminate_early with flexible keyword argument ordering."""
        return self.should_terminate_early(
            consecutive_static=consecutive_static,
            current_energy=current_energy,
            term_window=term_window,
            term_threshold=term_threshold,
        )


class AudioEnergyVAD:
    """In-Memory Short-Time RMS Energy Envelope Voice & Audio Activity Detector (VAD)."""

    def __init__(
        self,
        sample_rate: int = 16000,
        window_ms: int = 50,
        noise_margin_db: float = 12.0,
        min_dbfs: float = -42.0,
        min_speech_duration: float = 0.3,
        enabled: bool = True,
    ):
        self.sample_rate = int(sample_rate)
        self.window_ms = int(window_ms)
        self.window_samples = max(1, int(self.sample_rate * self.window_ms / 1000.0))
        self.noise_margin_db = float(noise_margin_db)
        self.min_dbfs = float(min_dbfs)
        self.min_speech_duration = float(min_speech_duration)
        self.enabled = bool(enabled)

    def compute_rms_dbfs(self, samples: np.ndarray) -> tuple[float, float]:
        """Compute RMS energy and dBFS for a 1D audio sample array."""
        if samples.size == 0:
            return 0.0, -140.0
        s = samples.astype(np.float32)
        rms = float(np.sqrt(np.mean(s ** 2)))
        dbfs = float(20.0 * np.log10(rms + 1e-7))
        return rms, dbfs

    def compute_dbfs_windows(
        self, audio: np.ndarray, start_offset: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Partition audio into ~50ms windows and calculate dBFS for each window."""
        if not self.enabled or audio.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), -140.0

        if hasattr(audio, "dbfs_windows"):
            values = audio.dbfs_windows()
            times = start_offset + np.arange(len(values), dtype=np.float64) * self.window_samples / self.sample_rate
            return times, values, float(np.percentile(values, 15)) if len(values) else -140.0
        n_samples = len(audio)
        w_size = self.window_samples
        n_windows = n_samples // w_size

        if n_windows == 0:
            rms, dbfs = self.compute_rms_dbfs(audio)
            return (
                np.array([start_offset], dtype=np.float32),
                np.array([dbfs], dtype=np.float32),
                dbfs,
            )

        truncated = np.asarray(audio[: n_windows * w_size], dtype=np.float32).reshape((n_windows, w_size))
        sum_sq = np.sum(truncated * truncated, axis=1)
        rms_vec = np.sqrt(sum_sq / float(w_size))
        dbfs_vec = 20.0 * np.log10(rms_vec + 1e-7)

        dt = w_size / float(self.sample_rate)
        timestamps = start_offset + np.arange(n_windows, dtype=np.float32) * dt
        noise_floor_db = float(np.percentile(dbfs_vec, 15))
        return timestamps, dbfs_vec, noise_floor_db

    def detect_events(
        self, audio: np.ndarray, start_offset: float = 0.0
    ) -> tuple[list[tuple[float, float, str]], dict]:
        """Detect active audio event intervals from raw mono float32 audio buffer."""
        if not self.enabled or audio.size == 0:
            return [], {
                "noise_floor_db": -140.0,
                "total_windows": 0,
                "active_windows": 0,
                "events_count": 0,
            }

        timestamps, dbfs_vec, noise_floor_db = self.compute_dbfs_windows(
            audio, start_offset=start_offset
        )
        if dbfs_vec.size == 0:
            return [], {
                "noise_floor_db": -140.0,
                "total_windows": 0,
                "active_windows": 0,
                "events_count": 0,
            }

        effective_min_dbfs = max(self.min_dbfs, -38.0) if noise_floor_db < -60.0 else self.min_dbfs
        is_relative_active = (dbfs_vec >= (noise_floor_db + self.noise_margin_db)) & (
            dbfs_vec >= effective_min_dbfs
        )

        n_windows = len(dbfs_vec)
        w_size = self.window_samples
        if hasattr(audio, "voiced_windows"):
            active_mask = is_relative_active.copy()
            if noise_floor_db >= -38.0:
                active_mask |= (dbfs_vec >= max(self.min_dbfs, -35.0)) & audio.voiced_windows()
        elif noise_floor_db >= -38.0 and n_windows > 0 and len(audio) >= n_windows * w_size:
            candidate_idx = np.where(~is_relative_active & (dbfs_vec >= max(self.min_dbfs, -35.0)))[0]
            if len(candidate_idx) > 0:
                truncated = audio[: n_windows * w_size].reshape((n_windows, w_size)).astype(np.float32)
                cand_blocks = truncated[candidate_idx]
                rms_sq = np.mean(cand_blocks ** 2, axis=1)
                cov1 = np.mean(cand_blocks[:, 1:] * cand_blocks[:, :-1], axis=1)
                r1_vec = cov1 / (rms_sq + 1e-9)
                std_vec = np.std(cand_blocks, axis=1)
                is_voiced = (r1_vec >= 0.5) & (std_vec > 1e-4)
                active_mask = is_relative_active.copy()
                active_mask[candidate_idx[is_voiced]] = True
            else:
                active_mask = is_relative_active
        else:
            active_mask = is_relative_active

        dt = self.window_samples / self.sample_rate
        min_active_windows = max(1, int(round(self.min_speech_duration / dt)))

        events: list[tuple[float, float, str]] = []
        n = len(active_mask)
        i = 0
        while i < n:
            if active_mask[i]:
                run_start = i
                while i < n and active_mask[i]:
                    i += 1
                run_end = i
                if (run_end - run_start) >= min_active_windows:
                    t_start = float(timestamps[run_start])
                    t_end = float(timestamps[run_end - 1] + dt)
                    events.append((t_start, t_end, "ACTIVE"))
            else:
                i += 1

        stats = {
            "noise_floor_db": round(noise_floor_db, 2),
            "total_windows": int(n),
            "active_windows": int(np.sum(active_mask)),
            "events_count": len(events),
        }
        return events, stats

    def is_active_at(
        self, t: float, events: list[tuple[float, float, str]], tolerance: float = 0.1
    ) -> bool:
        """Check whether timestamp t falls within any detected active audio event."""
        for t_start, t_end, _ in events:
            if (t_start - tolerance) <= t <= (t_end + tolerance):
                return True
        return False


def _median_filter(signal: list[float], window: int) -> list[float]:
    """时域中值滤波。"""
    if window < 3:
        return signal
    arr = np.array(signal, dtype=np.float32)
    pad_width = window // 2
    padded = np.pad(arr, pad_width, mode="edge")
    try:
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, window)
        return np.median(windows, axis=1).tolist()
    except ImportError:
        n = len(arr)
        result = np.empty_like(arr)
        for i in range(n):
            left = max(0, i - pad_width)
            right = min(n, i + pad_width + 1)
            result[i] = np.median(arr[left:right])
        return result.tolist()


def _smooth_labels(
    raw: list[bool],
    min_motion: int,
    min_static: int,
    noise_suppress: int,
) -> list[bool]:
    """时域离散二值标签平滑。"""
    if not raw:
        return raw
    smoothed = list(raw)
    n = len(smoothed)

    i = 0
    while i < n:
        if smoothed[i]:
            run_end = i
            while run_end < n and smoothed[run_end]:
                run_end += 1
            if run_end - i < min_motion:
                for j in range(i, run_end):
                    smoothed[j] = False
            i = run_end
        else:
            run_end = i
            while run_end < n and not smoothed[run_end]:
                run_end += 1
            if 0 < run_end - i < noise_suppress:
                for j in range(i, run_end):
                    smoothed[j] = True
            i = run_end

    i = 0
    while i < n:
        if not smoothed[i]:
            run_end = i
            while run_end < n and not smoothed[run_end]:
                run_end += 1
            if 0 < run_end - i < min_static:
                for j in range(i, run_end):
                    smoothed[j] = True
            i = run_end
        else:
            i += 1

    return smoothed
