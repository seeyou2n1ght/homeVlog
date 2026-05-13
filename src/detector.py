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


class MotionDetector:
    def __init__(self, config: dict, decode_gpu: str = "cuda"):
        det = config.get("detection", {})
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
        self.decode_timeout = det.get("decode_timeout", 10)
        self.decode_gpu = decode_gpu

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
        prev_gray: np.ndarray | None = None
        
        roi_x = int(self.width * self.roi[0])
        roi_y = int(self.height * self.roi[1])
        roi_w = int(self.width * self.roi[2])
        roi_h = int(self.height * self.roi[3])
        
        yolo_sample_interval = max(1, int(self.fps / max(0.1, getattr(self, 'yolo_sample_fps', 0.5))))

        t_decode_start = time.monotonic()
        total_frames = 0

        try:
            import av
            hw = None
            try:
                from av.codec.hwaccel import HWAccel
                hw = HWAccel(hw_name)
            except Exception as e:
                logger.debug(f"PyAV HWAccel init failed for {hw_name}: {e}")

            kwargs = {}
            if hw:
                kwargs['hwaccel'] = hw

            with av.open(str(filepath), **kwargs) as container:
                stream = container.streams.video[0]
                if not hw:
                    stream.thread_type = "AUTO"

                video_fps = float(stream.average_rate) if stream.average_rate else 30.0
                if video_fps <= 0:
                    video_fps = 30.0
                
                frame_step = max(1, int(round(video_fps / self.fps)))
                
                for i, frame in enumerate(container.decode(stream)):
                    if i % frame_step != 0:
                        continue
                        
                    total_frames += 1
                    
                    frame_rgb = frame.to_ndarray(format='rgb24')
                    
                    if frame_rgb.shape[1] != self.width or frame_rgb.shape[0] != self.height:
                        frame_rgb = cv2.resize(frame_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)

                    if getattr(self, 'yolo_enabled', False) and (total_frames - 1) % yolo_sample_interval == 0:
                        yolo_frames_buffer[total_frames - 1] = frame_rgb.copy()

                    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
                    roi = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w].astype(np.float32)
                    frame_rgb = None
                    gray = None

                    if prev_gray is not None:
                        diff = np.abs(roi - prev_gray)
                        energies.append(float(np.mean(diff)))

                    prev_gray = roi
                    
                    if time.monotonic() - t_decode_start > self.decode_timeout * 5:
                        logger.warning("PyAV decode timeout for %s", Path(filepath).name)
                        break

        except Exception as e:
            logger.warning("PyAV decode error for %s: %s", Path(filepath).name, e)
        finally:
            io_sem.release()

        t_decode_end = time.monotonic()

        if len(energies) < 2:
            logger.warning(
                "too few frames from %s: %d", Path(filepath).name, len(energies)
            )
            self.last_perf = {
                "decode_time": t_decode_end - t_decode_start,
                "analysis_time": 0,
                "frames": total_frames,
                "motion_ratio": 0,
            }
            return []

        t_analysis_start = time.monotonic()

        # Temporal median filter to suppress isolated GOP boundary spikes
        # (1-2 frame artifacts) while preserving sustained motion (3+ frames).
        if self.median_window >= 3:
            energies = _median_filter(energies, self.median_window)

        # Adaptive threshold: Robust noise floor estimation
        # IQR fails when motion frames > 50% (Q3 becomes a motion value).
        # Instead, we estimate the noise from the bottom 20% of frames.
        energies_arr = np.array(energies, dtype=np.float32)
        p5 = float(np.percentile(energies_arr, 5))
        p20 = float(np.percentile(energies_arr, 20))

        # Approximate standard IQR scale from the p5-p20 spread
        noise_spread = (p20 - p5) * 2.5

        # Absolute minimum offset prevents triggering on clean digital black
        min_offset = 0.5
        threshold = p5 + max(min_offset, self.sensitivity * noise_spread)

        raw_labels: list[bool] = [bool(e > threshold) for e in energies]
        # First frame is always False (no predecessor to compare)
        raw_labels.insert(0, False)

        logger.debug(
            "adaptive threshold for %s: p5=%.3f p20=%.3f spread=%.3f threshold=%.3f "
            "motion_frames=%d/%d (%.1f%%)",
            Path(filepath).name,
            p5,
            p20,
            noise_spread,
            threshold,
            sum(raw_labels),
            len(raw_labels),
            100 * sum(raw_labels) / len(raw_labels),
        )

        smoothed = _smooth_labels(
            raw_labels,
            self.min_motion_frames,
            self.min_static_frames,
            self.noise_suppress,
        )
        t_analysis_end = time.monotonic()

        motion_count = sum(1 for s in smoothed if s)
        self.last_perf = {
            "decode_time": round(t_decode_end - t_decode_start, 3),
            "analysis_time": round(t_analysis_end - t_analysis_start, 3),
            "frames": total_frames,
            "motion_ratio": round(motion_count / len(smoothed), 3) if smoothed else 0,
        }

        # 核心优化：依据文件真实时长和解出的总帧数做等比映射，消除变帧率/丢帧引起的时钟漂移
        actual_fps = (
            total_frames / file_duration
            if file_duration > 0 and total_frames > 0
            else self.fps
        )
        frame_interval = 1.0 / actual_fps if actual_fps > 0 else 1.0 / self.fps

        padded_energies = [0.0] + energies
        results = []
        for i, is_motion in enumerate(smoothed):
            # 防止最后的时间戳超过 file_duration
            time_val = start_offset + min(i * frame_interval, file_duration)
            results.append(
                {
                    "time": time_val,
                    "is_motion": is_motion,
                    "energy": float(padded_energies[i]) if i < len(padded_energies) else 0.0,
                }
            )
        return results, yolo_frames_buffer


def _median_filter(signal: list[float], window: int) -> list[float]:
    """1D median filter using NumPy vectorized operations (O(N) instead of Python loop)."""
    if window < 3:
        return signal
    arr = np.array(signal, dtype=np.float32)
    pad_width = window // 2
    # 使用 'edge' 填充保证边界平滑
    padded = np.pad(arr, pad_width, mode='edge')
    try:
        # NumPy 1.20+ sliding_window_view
        from numpy.lib.stride_tricks import sliding_window_view
        windows = sliding_window_view(padded, window)
        return np.median(windows, axis=1).tolist()
    except ImportError:
        # Fallback for very old numpy versions
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



