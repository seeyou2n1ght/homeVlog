import logging
import subprocess
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np

from src.utils import parse_res, ts_to_unix
from src.monitor import get_perf, PerfRecord

logger = logging.getLogger("homevlog")


class EmaBackgroundModel:
    """
    Lightweight Temporal Sliding Background Model with Selective EMA Update.

    Mathematical Model:
        B_t(x, y) = (1 - alpha(x, y)) * B_{t-1}(x, y) + alpha(x, y) * I_t(x, y)
        where:
          - alpha(x, y) = alpha_fg (~0.005) when |I_t(x, y) - B_{t-1}(x, y)| > fg_threshold (foreground motion)
          - alpha(x, y) = alpha_bg (~0.05) when pixel is static background

    Dual-difference Motion Saliency:
        D_frame(x, y) = |I_t(x, y) - I_{t-1}(x, y)|
        D_bg(x, y)    = |I_t(x, y) - B_t(x, y)|
        M_t(x, y)     = max(D_frame(x, y), beta * D_bg(x, y))  (beta ~ 0.6)
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

        # 5. Dual-difference motion saliency fusion: M_t = max(D_frame, beta * D_bg)
        saliency_map = np.maximum(d_frame, self.beta * d_bg)

        self.prev_frame = roi_float
        return saliency_map, d_frame, d_bg


class SpatialGridMotionFilter:
    """
    Dynamic 8x8 Spatial Grid Energy Extractor, Connected Component Noise Filter,
    and Spatial-Temporal Confidence Memory Cooldown Manager.

    Features:
    - Divides ROI into grid_rows x grid_cols (default 8x8 = 64 cells).
    - Extracts per-cell energy E_{r,c}(t) and maintains adaptive per-cell noise floors.
    - Suppresses isolated single-cell spikes (IR night vision noise) via 8-connected components.
    - Boosts contiguous active grid clusters (subtle human movement on couch/desk).
    - Tracks spatial-temporal confidence decay grid C_{r,c}(t) in [0.0, 1.0] with exponential cooldown.
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
    ):
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.cooldown_half_life = float(cooldown_half_life)
        self.min_connected_cells = int(min_connected_cells)
        self.cell_noise_alpha = float(cell_noise_alpha)
        self.sens_multiplier = float(sens_multiplier)
        self.base_noise_thresh = float(base_noise_thresh)
        self.cluster_boost = float(cluster_boost)

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
        """
        Partition saliency map into grid_rows x grid_cols and compute per-cell mean energy.
        """
        h, w = saliency_map.shape[:2]
        row_edges = np.linspace(0, h, self.grid_rows + 1, dtype=int)
        col_edges = np.linspace(0, w, self.grid_cols + 1, dtype=int)
        energies = np.zeros((self.grid_rows, self.grid_cols), dtype=np.float32)

        for r in range(self.grid_rows):
            r_start, r_end = row_edges[r], row_edges[r + 1]
            for c in range(self.grid_cols):
                c_start, c_end = col_edges[c], col_edges[c + 1]
                cell = saliency_map[r_start:r_end, c_start:c_end]
                energies[r, c] = float(np.mean(cell)) if cell.size > 0 else 0.0

        return energies

    def find_connected_components(self, binary_grid: np.ndarray) -> list[set[tuple[int, int]]]:
        """
        Extract 8-connected components from a 2D boolean grid.
        """
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
    ) -> tuple[float, bool, dict]:
        """
        Process a single frame's motion saliency map.

        Args:
            saliency_map: 2D float32 motion saliency map.
            dt: Elapsed time since last frame in seconds.

        Returns:
            tuple of (effective_frame_energy, is_motion_detected, stats_dict)
        """
        energies = self.extract_grid_energies(saliency_map)
        self.cell_energies = energies

        # 1. Raw cell activation against adaptive per-cell noise floor
        cell_thresholds = np.maximum(
            self.base_noise_thresh,
            self.noise_floor_grid * self.sens_multiplier,
        )
        raw_active = energies > cell_thresholds

        # 2. 8-Neighborhood connected component filtering & spatial noise suppression
        components = self.find_connected_components(raw_active)
        filtered_active = np.zeros((self.grid_rows, self.grid_cols), dtype=bool)

        for comp in components:
            if len(comp) >= self.min_connected_cells:
                # Contiguous cluster: confirmed genuine target movement
                for r, c in comp:
                    filtered_active[r, c] = True
            else:
                # Isolated single-cell: filter unless it is an extreme high-magnitude spike
                for r, c in comp:
                    if energies[r, c] > cell_thresholds[r, c] * 2.5:
                        filtered_active[r, c] = True

        self.active_grid = filtered_active
        num_active_cells = int(np.sum(filtered_active))

        # 3. Adaptive noise floor tracking on inactive cells
        inactive_mask = ~filtered_active
        if np.any(inactive_mask):
            self.noise_floor_grid[inactive_mask] = (
                (1.0 - self.cell_noise_alpha) * self.noise_floor_grid[inactive_mask]
                + self.cell_noise_alpha * energies[inactive_mask]
            )

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
            # All noise suppressed or quiet background -> dampen raw diff noise
            effective_energy = global_mean * 0.3
            is_motion = False

        stats = {
            "active_cells": num_active_cells,
            "max_confidence": float(np.max(self.confidence_grid)),
            "mean_energy": float(np.mean(energies)),
            "effective_energy": effective_energy,
        }
        return effective_energy, is_motion, stats

    def can_early_terminate(
        self,
        consecutive_static: int,
        term_window: int,
        current_energy: float,
        term_threshold: float,
    ) -> bool:
        """
        Anti-Miss Early Termination Guard:
        Early termination break is only permitted when:
          (a) consecutive static frames >= term_window
          (b) current energy < term_threshold
          (c) all regional memory confidence decayed below 0.05
        """
        if consecutive_static < term_window:
            return False
        if current_energy >= term_threshold:
            return False
        max_conf = float(np.max(self.confidence_grid))
        if max_conf >= 0.05:
            return False
        return True


class AudioEnergyVAD:
    """
    In-Memory Short-Time RMS Energy Envelope Voice & Audio Activity Detector (VAD).

    Mathematical Formulation:
      1. Window Framing:
         Divides audio stream into ~50ms windows (N = sample_rate * window_ms / 1000).
      2. Short-Time RMS:
         RMS_k = sqrt( (1 / N) * sum_{n=0}^{N-1} s_k[n]^2 )
      3. Logarithmic Energy (dBFS):
         dBFS_k = 20 * log10(RMS_k + 1e-7)
      4. Adaptive Steady-State Background Noise Floor Tracking:
         N_audio = percentile(dBFS_all, 15)
      5. Audio Activity Activation:
         dBFS_k > (N_audio + noise_margin_db) and dBFS_k >= min_dbfs
      6. Event Interval Extraction:
         Returns [(t_start, t_end, "ACTIVE"), ...] for active speech/sound segments.
    """

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
        """
        Compute RMS energy and dBFS for a 1D audio sample array.
        """
        if samples.size == 0:
            return 0.0, -140.0
        s = samples.astype(np.float32)
        rms = float(np.sqrt(np.mean(s ** 2)))
        dbfs = float(20.0 * np.log10(rms + 1e-7))
        return rms, dbfs

    def compute_dbfs_windows(
        self, audio: np.ndarray, start_offset: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Partition audio into ~50ms windows and calculate dBFS for each window.

        Args:
            audio: 1D mono float32 audio NumPy array.
            start_offset: Base time offset in seconds.

        Returns:
            tuple of (timestamps, dbfs_array, noise_floor_db)
        """
        if not self.enabled or audio.size == 0:
            return np.array([], dtype=np.float32), np.array([], dtype=np.float32), -140.0

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

        # Truncate to exact multiple of window size for fast zero-copy vectorized computing
        truncated = np.asarray(audio[: n_windows * w_size], dtype=np.float32).reshape((n_windows, w_size))
        # Vectorized RMS: sqrt(sum(w^2, axis=1) / w_size) — avoid pow2 allocations
        sum_sq = np.sum(truncated * truncated, axis=1)
        rms_vec = np.sqrt(sum_sq / float(w_size))
        dbfs_vec = 20.0 * np.log10(rms_vec + 1e-7)

        # Time for each window center or start
        dt = w_size / float(self.sample_rate)
        timestamps = start_offset + np.arange(n_windows, dtype=np.float32) * dt

        # Adaptive 15th percentile noise floor
        noise_floor_db = float(np.percentile(dbfs_vec, 15))
        return timestamps, dbfs_vec, noise_floor_db

    def detect_events(
        self, audio: np.ndarray, start_offset: float = 0.0
    ) -> tuple[list[tuple[float, float, str]], dict]:
        """
        Detect active audio event intervals from raw mono float32 audio buffer.

        Args:
            audio: 1D mono float32 audio NumPy array.
            start_offset: Base time offset in seconds.

        Returns:
            tuple of (events, stats_dict)
            where events is [(t_start, t_end, "ACTIVE"), ...]
        """
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

        # 1. Relative threshold activation: delta above 15th percentile noise floor AND absolute min_dbfs
        is_relative_active = (dbfs_vec >= (noise_floor_db + self.noise_margin_db)) & (
            dbfs_vec >= self.min_dbfs
        )

        # 2. Continuous voiced/harmonic audio activation:
        # In 100% duty cycle continuous speech, 15th percentile noise floor rises to speech levels (>= -38.0 dBFS).
        # We distinguish genuine speech/tones from stationary noise (r1 ~ 0) and flat DC bias (std == 0)
        # via lag-1 autocorrelation (r1 >= 0.5) and AC sample variation (std > 1e-4).
        n_windows = len(dbfs_vec)
        w_size = self.window_samples
        if noise_floor_db >= -38.0 and n_windows > 0 and len(audio) >= n_windows * w_size:
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

        # Group consecutive active windows into intervals
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


class MotionDetector:
    def __init__(self, config: dict, decode_gpu: str = "cuda"):
        det = config.get("detection", {})
        self.config = config
        self.width, self.height = parse_res(det.get("analysis_resolution", "640x360"))
        self.fps = det.get("analysis_fps", 5)
        self.sensitivity = det.get("motion_sensitivity", 4.0)
        self.roi = det.get("roi_crop", [0.1, 0.12, 0.8, 0.85])
        self.min_motion_frames = det.get("min_motion_frames", 3)
        self.min_static_frames = det.get("min_static_frames", 5)
        self.noise_suppress = det.get("noise_suppress_frames", 2)
        self.grid_cols = det.get("grid_cols", 8)
        self.grid_rows = det.get("grid_rows", 8)
        self.median_window = det.get("median_filter_window", 7)
        self.pipe_buf_mult = det.get("pipe_buffer_multiplier", 4)
        self.decode_timeout = det.get("decode_timeout", 30)
        self.decode_gpu = decode_gpu

        # 核心性能开关
        self.adaptive_fps_enabled = det.get("analysis_fps_adaptive", True)
        self.fps_tiers = det.get("analysis_fps_tiers", {"short": 5, "medium": 3, "long": 2})
        self.fps_tier_thresholds = det.get(
            "analysis_fps_tier_thresholds", {"short_max": 120, "medium_max": 600}
        )

        # R2: 时域滑动背景模型 (EMA Background)
        self.ema_enabled = det.get("ema_background_enabled", True)
        self.ema_alpha_bg = det.get("ema_alpha_bg", 0.05)
        self.ema_alpha_fg = det.get("ema_alpha_fg", 0.005)
        self.bg_diff_weight = det.get("bg_diff_weight", 0.6)
        self.fg_threshold = det.get("fg_threshold", 12.0)

        # R2: 动态空间网格与抗噪
        self.cooldown_half_life = det.get("cooldown_half_life", 15.0)
        self.min_connected_cells = det.get("min_connected_cells", 2)
        self.cell_noise_alpha = det.get("cell_noise_alpha", 0.02)
        self.base_noise_thresh = det.get("base_noise_thresh", 1.5)
        self.cluster_boost = det.get("cluster_boost", 1.2)

        # R2: 早停与时空置信度衰减保护
        self.early_term_enabled = det.get("analysis_early_term_enabled", True)
        self.early_term_window = det.get("analysis_early_term_window", 30)
        self.early_term_threshold = det.get("analysis_early_term_threshold", 2.0)
        self.early_term_cooldown_guard = det.get("early_term_cooldown_guard", True)

        # R3: 音频 VAD 多模态事件唤醒 (Audio-Assisted Activity Detection)
        audio_cfg = config.get("audio_vad", {})
        self.audio_vad_enabled = det.get(
            "audio_vad_enabled", audio_cfg.get("enabled", True)
        )
        self.vad_noise_margin_db = float(
            det.get(
                "vad_noise_margin_db",
                audio_cfg.get(
                    "noise_margin_db", audio_cfg.get("energy_threshold_db", 12.0)
                ),
            )
        )
        self.vad_min_dbfs = float(
            det.get(
                "vad_min_dbfs",
                audio_cfg.get(
                    "min_dbfs", audio_cfg.get("min_absolute_dbfs", -42.0)
                ),
            )
        )
        self.vad_window_ms = int(
            det.get("vad_window_ms", audio_cfg.get("window_ms", 50))
        )
        self.vad_min_speech_duration = float(
            det.get(
                "vad_min_speech_duration",
                audio_cfg.get("speech_min_dur", 0.3),
            )
        )
        self.vad_sample_rate = int(audio_cfg.get("sample_rate", 16000))

        self.vad = AudioEnergyVAD(
            sample_rate=self.vad_sample_rate,
            window_ms=self.vad_window_ms,
            noise_margin_db=self.vad_noise_margin_db,
            min_dbfs=self.vad_min_dbfs,
            min_speech_duration=self.vad_min_speech_duration,
            enabled=self.audio_vad_enabled,
        )

        # YOLO 联动推理开关与采样参数
        yolo_cfg = config.get("yolo", {})
        self.yolo_enabled = yolo_cfg.get("enabled", False)
        self.yolo_sample_fps = float(yolo_cfg.get("sample_fps", 0.5))

        self.last_perf: dict = {}


    def create_ema_model(self) -> EmaBackgroundModel:
        """Helper to create configured EMA background model."""
        return EmaBackgroundModel(
            alpha_bg=self.ema_alpha_bg,
            alpha_fg=self.ema_alpha_fg,
            beta=self.bg_diff_weight,
            fg_threshold=self.fg_threshold,
        )

    def create_grid_filter(self) -> SpatialGridMotionFilter:
        """Helper to create configured spatial grid motion filter."""
        return SpatialGridMotionFilter(
            grid_rows=self.grid_rows,
            grid_cols=self.grid_cols,
            cooldown_half_life=self.cooldown_half_life,
            min_connected_cells=self.min_connected_cells,
            cell_noise_alpha=self.cell_noise_alpha,
            sens_multiplier=max(1.0, self.sensitivity * 0.5),
            base_noise_thresh=self.base_noise_thresh,
            cluster_boost=self.cluster_boost,
        )

    def analyze_frames(
        self,
        frames: list[np.ndarray],
        start_offset: float = 0.0,
        file_duration: float = 0.0,
        fps: float | None = None,
        audio_data: np.ndarray | None = None,
        audio_sample_rate: int = 16000,
    ) -> tuple[list[dict], dict]:
        """
        In-memory frame analysis pipeline for fast unit testing and synthetic verification.
        Supports multimodal fusion with optional in-memory audio buffer.
        """
        effective_fps = fps if fps is not None else self.fps
        dt = 1.0 / effective_fps if effective_fps > 0 else 0.2
        frame_interval = dt

        roi_x = int(self.width * self.roi[0])
        roi_y = int(self.height * self.roi[1])
        roi_w = int(self.width * self.roi[2])
        roi_h = int(self.height * self.roi[3])

        ema_model = self.create_ema_model()
        grid_filter = self.create_grid_filter()
        prev_gray: np.ndarray | None = None
        consecutive_static = 0
        early_terminated = False
        energies: list[float] = []

        # R3: Audio VAD event extraction if audio buffer provided
        audio_events: list[tuple[float, float, str]] = []
        vad_stats: dict = {}
        if self.audio_vad_enabled and audio_data is not None and len(audio_data) > 0:
            vad_detector = AudioEnergyVAD(
                sample_rate=audio_sample_rate,
                window_ms=self.vad_window_ms,
                noise_margin_db=self.vad_noise_margin_db,
                min_dbfs=self.vad_min_dbfs,
                min_speech_duration=self.vad_min_speech_duration,
                enabled=True,
            )
            audio_events, vad_stats = vad_detector.detect_events(
                audio_data, start_offset=start_offset
            )

        for i, frame in enumerate(frames):
            t_curr = start_offset + min(i * frame_interval, file_duration if file_duration > 0 else 999999.0)

            if len(frame.shape) == 3:
                gray = cv2.cvtColor(
                    cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST),
                    cv2.COLOR_RGB2GRAY if frame.shape[2] == 3 else cv2.COLOR_BGR2GRAY,
                )
            else:
                gray = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

            roi = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]

            if self.ema_enabled:
                saliency, _, _ = ema_model.update(roi)
            else:
                if prev_gray is not None:
                    saliency = cv2.absdiff(roi, prev_gray).astype(np.float32)
                else:
                    saliency = np.zeros_like(roi, dtype=np.float32)
                prev_gray = roi

            eff_energy, is_grid_motion, _ = grid_filter.process_frame(saliency, dt)
            energies.append(eff_energy)

            if is_grid_motion:
                consecutive_static = 0
            else:
                consecutive_static += 1

            # Early termination check with Audio VAD VETO
            if self.early_term_enabled and len(energies) >= self.early_term_window:
                is_audio_active_now = False
                if audio_events:
                    for ev_start, ev_end, _ in audio_events:
                        if ev_end >= (t_curr - 0.2):
                            is_audio_active_now = True
                            break

                if is_audio_active_now:
                    # Audio VAD Veto: Do not early terminate while speech/audio is ongoing or upcoming
                    can_term = False
                elif self.early_term_cooldown_guard:
                    can_term = grid_filter.can_early_terminate(
                        consecutive_static=consecutive_static,
                        term_window=self.early_term_window,
                        current_energy=eff_energy,
                        term_threshold=self.early_term_threshold,
                    )
                else:
                    can_term = max(energies[-self.early_term_window:]) < self.early_term_threshold

                if can_term:
                    early_terminated = True
                    break

        if early_terminated and file_duration > 0 and energies:
            energies.append(0.0)

        if len(energies) < 2:
            return [], {"audio_events": audio_events, "vad_stats": vad_stats}

        if self.median_window >= 3:
            energies = _median_filter(energies, self.median_window)

        energies_arr = np.array(energies, dtype=np.float32)
        p5 = float(np.percentile(energies_arr, 5))
        p20 = float(np.percentile(energies_arr, 20))
        noise_spread = (p20 - p5) * 2.5
        threshold = p5 + max(0.5, self.sensitivity * noise_spread)

        raw_labels = [bool(e > threshold) for e in energies]

        smoothed = _smooth_labels(
            raw_labels, self.min_motion_frames, self.min_static_frames, self.noise_suppress
        )

        results = []
        for i, is_visual_motion in enumerate(smoothed):
            if early_terminated and i == len(smoothed) - 1:
                time_val = start_offset + file_duration
            else:
                time_val = start_offset + min(i * frame_interval, file_duration if file_duration > 0 else (len(smoothed) * frame_interval))

            # Multimodal Audio VAD check at time_val
            is_audio_active = False
            if audio_events:
                for ev_start, ev_end, _ in audio_events:
                    if (ev_start - 0.1) <= time_val <= (ev_end + 0.1):
                        is_audio_active = True
                        break

            # Multimodal Fusion Decision Matrix
            if is_visual_motion:
                state = "DYNAMIC"
                is_motion = True
            elif is_audio_active:
                state = "DYNAMIC_AUDIO"
                is_motion = True
            else:
                state = "STATIC"
                is_motion = False

            results.append({
                "time": time_val,
                "is_motion": is_motion,
                "state": state,
                "energy": float(energies[i]) if i < len(energies) else 0.0,
                "is_audio_active": is_audio_active,
            })

        meta = {
            "has_audio": 1 if audio_events or (audio_data is not None and len(audio_data) > 0) else 0,
            "audio_events": audio_events,
            "vad_stats": vad_stats,
            "early_terminated": early_terminated,
        }
        return results, meta

    def analyze(
        self, filepath: str, start_offset: float = 0.0, file_duration: float = 0.0
    ) -> tuple[list[dict], dict]:
        yolo_frames_buffer: dict[int, np.ndarray] = {}

        if self.decode_gpu == "qsv":
            from src.utils import get_qsv_semaphore
            io_sem = get_qsv_semaphore()
            hw_name = "qsv"
        else:
            from src.utils import get_nv_semaphore
            io_sem = get_nv_semaphore()
            hw_name = "cuda"

        io_sem.acquire()

        energies: list[float] = []
        ema_model = self.create_ema_model()
        grid_filter = self.create_grid_filter()
        prev_gray: np.ndarray | None = None
        consecutive_static = 0

        roi_x = int(self.width * self.roi[0])
        roi_y = int(self.height * self.roi[1])
        roi_w = int(self.width * self.roi[2])
        roi_h = int(self.height * self.roi[3])

        yolo_sample_interval = max(
            1, int(self.fps / max(0.1, getattr(self, "yolo_sample_fps", 0.5)))
        )

        total_frames = 0
        video_frame_count = 0
        early_terminated = False
        effective_fps = self.fps
        t_decode_start = 0.0

        # R3: In-Memory Audio Extraction Buffer & Online VAD state
        audio_samples_list: list[np.ndarray] = []
        audio_resampler = None
        has_audio = 0
        recent_audio_active = False
        running_audio_noise_floor = -60.0
        audio_active_until_time = -1.0
        total_audio_samples_decoded = 0

        try:
            from av.audio.resampler import AudioResampler
            audio_resampler = AudioResampler(format="fltp", layout="mono", rate=self.vad_sample_rate)
        except Exception as e:
            logger.debug(f"PyAV AudioResampler init: {e}")

        try:
            hw = None
            try:
                from av.codec.hwaccel import HWAccel
                hw = HWAccel(hw_name)
            except Exception as e:
                logger.debug(f"PyAV HWAccel init failed for {hw_name}: {e}")

            kwargs = {}
            if hw:
                kwargs["hwaccel"] = hw

            with av.open(str(filepath), **kwargs) as container:
                video_stream = container.streams.video[0] if container.streams.video else None
                audio_stream = container.streams.audio[0] if container.streams.audio else None

                has_audio = 1 if audio_stream is not None else 0
                try:
                    self.has_audio_detected = has_audio
                except Exception:
                    pass

                if video_stream is None:
                    logger.warning("No video stream found in %s", Path(filepath).name)
                    io_sem.release()
                    return [], yolo_frames_buffer

                if not hw:
                    video_stream.thread_type = "AUTO"

                video_fps = float(video_stream.average_rate) if video_stream.average_rate else 30.0
                if video_fps <= 0:
                    video_fps = 30.0

                if self.adaptive_fps_enabled and file_duration > 0:
                    if file_duration <= self.fps_tier_thresholds["short_max"]:
                        effective_fps = self.fps_tiers["short"]
                    elif file_duration <= self.fps_tier_thresholds["medium_max"]:
                        effective_fps = self.fps_tiers["medium"]
                    else:
                        effective_fps = self.fps_tiers["long"]
                else:
                    effective_fps = self.fps

                frame_step = max(1, int(round(video_fps / effective_fps)))
                dt = 1.0 / effective_fps

                # 增加对长文件和网络路径的超时容忍
                watchdog_timeout = max(self.decode_timeout * 3, 60.0)
                if file_duration > 0:
                    watchdog_timeout = max(watchdog_timeout, (file_duration / effective_fps) * 4)

                t_decode_start = time.monotonic()

                # Single-pass concurrent stream decoding
                streams_to_decode = [video_stream]
                if self.audio_vad_enabled and audio_stream is not None:
                    streams_to_decode.append(audio_stream)

                for frame in container.decode(*streams_to_decode):
                    # Handle Audio Frame
                    if isinstance(frame, av.AudioFrame) or getattr(frame, "type", "") == "audio":
                        try:
                            if audio_resampler:
                                resampled_frames = audio_resampler.resample(frame)
                                if resampled_frames:
                                    for rf in resampled_frames:
                                        arr = rf.to_ndarray().flatten().astype(np.float32)
                                        audio_samples_list.append(arr)
                                        total_audio_samples_decoded += len(arr)
                            else:
                                raw = frame.to_ndarray()
                                if raw.ndim > 1:
                                    raw = np.mean(raw, axis=0)
                                raw_f32 = raw.flatten().astype(np.float32)
                                if getattr(frame, "rate", 0) != self.vad_sample_rate and getattr(frame, "rate", 0) > 0:
                                    target_len = int(len(raw_f32) * self.vad_sample_rate / frame.rate)
                                    if target_len > 0:
                                        raw_f32 = np.interp(
                                            np.linspace(0, len(raw_f32), target_len, endpoint=False),
                                            np.arange(len(raw_f32)),
                                            raw_f32,
                                        ).astype(np.float32)
                                audio_samples_list.append(raw_f32)
                                total_audio_samples_decoded += len(raw_f32)

                            # Online audio activity tracking on recent samples with adaptive background noise baseline
                            if len(audio_samples_list) > 0:
                                last_chunk = audio_samples_list[-1]
                                check_len = min(len(last_chunk), 1600)
                                if check_len >= 400:
                                    rms_val = float(np.sqrt(np.mean(last_chunk[-check_len:] ** 2)))
                                    dbfs_val = float(20.0 * np.log10(rms_val + 1e-7))

                                    if running_audio_noise_floor < -59.0:
                                        running_audio_noise_floor = min(dbfs_val, -38.0)
                                    else:
                                        if dbfs_val < running_audio_noise_floor:
                                            running_audio_noise_floor = running_audio_noise_floor * 0.9 + dbfs_val * 0.1
                                        else:
                                            running_audio_noise_floor = (
                                                running_audio_noise_floor * 0.995 + min(dbfs_val, -38.0) * 0.005
                                            )
                                        running_audio_noise_floor = min(running_audio_noise_floor, -38.0)

                                    is_chunk_speech = (
                                        (dbfs_val >= (running_audio_noise_floor + self.vad_noise_margin_db))
                                        and (dbfs_val >= self.vad_min_dbfs)
                                    ) or (dbfs_val >= -28.0)

                                    current_audio_time = total_audio_samples_decoded / self.vad_sample_rate
                                    if is_chunk_speech:
                                        audio_active_until_time = current_audio_time + 0.5
                                        recent_audio_active = True
                                    else:
                                        recent_audio_active = bool(current_audio_time < audio_active_until_time)
                        except Exception as e:
                            logger.debug("Audio decode chunk failed: %s", e)
                        continue

                    # Handle Video Frame
                    video_frame_count += 1
                    if (video_frame_count - 1) % frame_step != 0:
                        continue

                    total_frames += 1

                    if getattr(self, "yolo_enabled", False) and (total_frames - 1) % yolo_sample_interval == 0:
                        try:
                            rgb_raw = frame.reformat(
                                width=self.width, height=self.height, format="rgb24"
                            ).to_ndarray()
                        except Exception:
                            rgb_raw = cv2.resize(
                                frame.to_ndarray(format="rgb24"),
                                (self.width, self.height),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        # 内存切片压缩为 JPEG 字节，单帧从 292KB 降至约 15KB
                        bgr_tmp = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                        ok_enc, buf_jpg = cv2.imencode(".jpg", bgr_tmp, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok_enc:
                            yolo_frames_buffer[total_frames - 1] = buf_jpg
                        else:
                            yolo_frames_buffer[total_frames - 1] = rgb_raw


                    # Analysis 灰度图极速提取 (直接提取 YUV420p 的 Y 平面，零色彩空间转换开销)
                    try:
                        if frame.planes:
                            y_raw = np.frombuffer(frame.planes[0], dtype=np.uint8).reshape((frame.height, frame.width))
                            gray = cv2.resize(y_raw, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
                        else:
                            gray = frame.to_ndarray(format="gray")
                    except Exception:
                        try:
                            gray = cv2.resize(
                                frame.to_ndarray(format="gray"),
                                (self.width, self.height),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        except Exception:
                            gray = cv2.cvtColor(
                                cv2.resize(
                                    frame.to_ndarray(format="rgb24"),
                                    (self.width, self.height),
                                    interpolation=cv2.INTER_NEAREST,
                                ),
                                cv2.COLOR_RGB2GRAY,
                            )

                    roi = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
                    gray = None

                    if self.ema_enabled:
                        saliency, _, _ = ema_model.update(roi)
                    else:
                        if prev_gray is not None:
                            saliency = cv2.absdiff(roi, prev_gray).astype(np.float32)
                        else:
                            saliency = np.zeros_like(roi, dtype=np.float32)
                        prev_gray = roi

                    eff_energy, is_grid_motion, _ = grid_filter.process_frame(saliency, dt)
                    energies.append(eff_energy)

                    if is_grid_motion:
                        consecutive_static = 0
                    else:
                        consecutive_static += 1

                    # Early termination check with Audio VAD VETO
                    if self.early_term_enabled and len(energies) >= self.early_term_window:
                        current_video_time = (total_frames - 1) * dt
                        is_audio_veto = recent_audio_active or (current_video_time < audio_active_until_time)
                        if is_audio_veto:
                            can_term = False
                        elif self.early_term_cooldown_guard:
                            can_term = grid_filter.can_early_terminate(
                                consecutive_static=consecutive_static,
                                term_window=self.early_term_window,
                                current_energy=eff_energy,
                                term_threshold=self.early_term_threshold,
                            )
                        else:
                            can_term = max(energies[-self.early_term_window:]) < self.early_term_threshold

                        if can_term:
                            early_terminated = True
                            break

                    if time.monotonic() - t_decode_start > watchdog_timeout:
                        logger.warning(
                            "PyAV decode timeout for %s (dur=%.1f, elapsed=%.1f)",
                            Path(filepath).name,
                            file_duration,
                            time.monotonic() - t_decode_start,
                        )
                        break

                # Flush audio resampler buffer
                if audio_resampler:
                    try:
                        flushed = audio_resampler.resample(None)
                        if flushed:
                            for rf in flushed:
                                arr = rf.to_ndarray().flatten().astype(np.float32)
                                audio_samples_list.append(arr)
                    except Exception:
                        pass

        except Exception as e:
            logger.warning("PyAV decode error for %s: %s", Path(filepath).name, e)
        finally:
            io_sem.release()

        t_decode_end = time.monotonic()

        # Combine in-memory audio chunks
        if audio_samples_list:
            full_audio = np.concatenate(audio_samples_list, axis=0)
        else:
            full_audio = np.array([], dtype=np.float32)

        # Execute full-file Audio VAD event extraction
        audio_events: list[tuple[float, float, str]] = []
        vad_stats: dict = {}
        if self.audio_vad_enabled and full_audio.size > 0:
            audio_events, vad_stats = self.vad.detect_events(
                full_audio, start_offset=start_offset
            )

        if early_terminated and file_duration > 0 and energies:
            energies.append(0.0)

        if len(energies) < 2:
            logger.warning("too few frames from %s: %d", Path(filepath).name, len(energies))
            self.last_perf = {
                "decode_time": t_decode_end - t_decode_start,
                "analysis_time": 0,
                "frames": total_frames,
                "motion_ratio": 0,
                "early_term": early_terminated,
                "has_audio": has_audio,
                "audio_events": len(audio_events),
            }
            return [], yolo_frames_buffer

        t_analysis_start = time.monotonic()
        if self.median_window >= 3:
            energies = _median_filter(energies, self.median_window)

        energies_arr = np.array(energies, dtype=np.float32)
        p5 = float(np.percentile(energies_arr, 5))
        p20 = float(np.percentile(energies_arr, 20))
        noise_spread = (p20 - p5) * 2.5
        threshold = p5 + max(0.5, self.sensitivity * noise_spread)

        raw_labels: list[bool] = [bool(e > threshold) for e in energies]
        smoothed = _smooth_labels(
            raw_labels, self.min_motion_frames, self.min_static_frames, self.noise_suppress
        )
        t_analysis_end = time.monotonic()

        motion_count = sum(1 for s in smoothed if s)
        self.last_perf = {
            "decode_time": round(t_decode_end - t_decode_start, 3),
            "analysis_time": round(t_analysis_end - t_analysis_start, 3),
            "frames": total_frames,
            "motion_ratio": round(motion_count / len(smoothed), 3) if smoothed else 0,
            "early_term": early_terminated,
            "effective_fps": effective_fps,
            "has_audio": has_audio,
            "audio_events": len(audio_events),
            "vad_noise_floor_db": vad_stats.get("noise_floor_db", -140.0),
        }

        actual_fps = (
            total_frames / file_duration
            if file_duration > 0 and total_frames > 0
            else effective_fps
        )
        frame_interval = 1.0 / actual_fps

        results = []
        for i, is_visual_motion in enumerate(smoothed):
            if early_terminated and i == len(smoothed) - 1:
                time_val = start_offset + file_duration
            else:
                time_val = start_offset + min(i * frame_interval, file_duration if file_duration > 0 else (len(smoothed) * frame_interval))

            # Multimodal Fusion Decision Matrix
            is_audio_active = self.vad.is_active_at(time_val, audio_events) if audio_events else False
            if is_visual_motion:
                state = "DYNAMIC"
                is_motion = True
            elif is_audio_active:
                state = "DYNAMIC_AUDIO"
                is_motion = True
            else:
                state = "STATIC"
                is_motion = False

            results.append({
                "time": time_val,
                "is_motion": is_motion,
                "state": state,
                "energy": float(energies[i]) if i < len(energies) else 0.0,
                "is_audio_active": is_audio_active,
            })
        return results, yolo_frames_buffer


def _median_filter(signal: list[float], window: int) -> list[float]:
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
