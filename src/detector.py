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
        self.decode_timeout = det.get("decode_timeout", 30)
        self.decode_gpu = decode_gpu

        # 核心性能开关
        self.adaptive_fps_enabled = det.get("analysis_fps_adaptive", True)
        self.fps_tiers = det.get("analysis_fps_tiers", {"short": 5, "medium": 3, "long": 2})
        self.fps_tier_thresholds = det.get(
            "analysis_fps_tier_thresholds", {"short_max": 120, "medium_max": 600}
        )
        self.early_term_enabled = det.get("analysis_early_term_enabled", True)
        self.early_term_window = det.get("analysis_early_term_window", 30)
        self.early_term_threshold = det.get("analysis_early_term_threshold", 2.0)

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

        total_frames = 0
        early_terminated = False
        effective_fps = self.fps
        t_decode_start = 0.0

        try:
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
                
                # Lazy Metadata: Detect audio and update DB
                has_audio = 1 if any(s.type == 'audio' for s in container.streams) else 0
                try:
                    from src.database import VlogDatabase
                    # We can't easily get the shared DB here without passing it, 
                    # but we can open a temporary one or rely on the orchestrator to do it.
                    # For now, let's just log it and we'll fix the DB update in the orchestrator.
                    self.has_audio_detected = has_audio
                except Exception:
                    pass

                if not hw:
                    stream.thread_type = "AUTO"

                video_fps = float(stream.average_rate) if stream.average_rate else 30.0
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

                # 增加对长文件和网络路径的超时容忍
                watchdog_timeout = max(self.decode_timeout * 3, 60.0)
                if file_duration > 0:
                    watchdog_timeout = max(watchdog_timeout, (file_duration / effective_fps) * 4)

                t_decode_start = time.monotonic()
                for i, frame in enumerate(container.decode(stream)):
                    if i % frame_step != 0:
                        continue

                    total_frames += 1

                    if getattr(self, 'yolo_enabled', False) and (total_frames - 1) % yolo_sample_interval == 0:
                        try:
                            yolo_frames_buffer[total_frames - 1] = frame.reformat(
                                width=self.width, height=self.height, format='rgb24'
                            ).to_ndarray()
                        except Exception:
                            yolo_frames_buffer[total_frames - 1] = cv2.resize(
                                frame.to_ndarray(format='rgb24'),
                                (self.width, self.height),
                                interpolation=cv2.INTER_LINEAR,
                            )

                    # Analysis 灰度图处理（GPU Reformat 优先）
                    try:
                        # 性能优化：直接提取 Y 通道（如果是 NV12/YUV）避免全彩色重格式化
                        # PyAV 的 'gray' 格式在 GPU 路径通常就是提取 Y 平面，速度最快
                        tmp_frame = frame.reformat(width=self.width, height=self.height, format='gray')
                        gray = tmp_frame.to_ndarray()
                    except Exception:
                        # 降级到 OpenCV，但使用 INTER_NEAREST 换取极致速度
                        gray = cv2.cvtColor(
                            cv2.resize(
                                frame.to_ndarray(format='rgb24'),
                                (self.width, self.height),
                                interpolation=cv2.INTER_NEAREST,
                            ),
                            cv2.COLOR_RGB2GRAY,
                        )

                    # 性能极限：在 uint8 空间直接计算，减少 float32 转换开销
                    roi = gray[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
                    gray = None

                    if prev_gray is not None:
                        # 使用 NumPy 向量化指令加速帧差计算
                        # .mean() 会隐式转 float，但在大量像素下 uint8 减法更快
                        diff_sum = np.sum(cv2.absdiff(roi, prev_gray))
                        energy = float(diff_sum / (roi_w * roi_h))
                        energies.append(energy)

                        if self.early_term_enabled and len(energies) >= self.early_term_window:
                            if max(energies[-self.early_term_window:]) < self.early_term_threshold:
                                early_terminated = True
                                break

                    prev_gray = roi

                    if time.monotonic() - t_decode_start > watchdog_timeout:
                        logger.warning("PyAV decode timeout for %s (dur=%.1f, elapsed=%.1f)", Path(filepath).name, file_duration, time.monotonic() - t_decode_start)
                        break

        except Exception as e:
            logger.warning("PyAV decode error for %s: %s", Path(filepath).name, e)
        finally:
            io_sem.release()

        t_decode_end = time.monotonic()

        if early_terminated and file_duration > 0 and energies:
            energies.append(0.0)

        if len(energies) < 2:
            logger.warning(
                "too few frames from %s: %d", Path(filepath).name, len(energies)
            )
            self.last_perf = {
                "decode_time": t_decode_end - t_decode_start,
                "analysis_time": 0,
                "frames": total_frames,
                "motion_ratio": 0,
                "early_term": early_terminated,
            }
            return []

        t_analysis_start = time.monotonic()
        if self.median_window >= 3:
            energies = _median_filter(energies, self.median_window)

        energies_arr = np.array(energies, dtype=np.float32)
        p5 = float(np.percentile(energies_arr, 5))
        p20 = float(np.percentile(energies_arr, 20))
        noise_spread = (p20 - p5) * 2.5
        threshold = p5 + max(0.5, self.sensitivity * noise_spread)

        raw_labels: list[bool] = [bool(e > threshold) for e in energies]
        raw_labels.insert(0, False)

        smoothed = _smooth_labels(raw_labels, self.min_motion_frames, self.min_static_frames, self.noise_suppress)
        t_analysis_end = time.monotonic()

        motion_count = sum(1 for s in smoothed if s)
        self.last_perf = {
            "decode_time": round(t_decode_end - t_decode_start, 3),
            "analysis_time": round(t_analysis_end - t_analysis_start, 3),
            "frames": total_frames,
            "motion_ratio": round(motion_count / len(smoothed), 3) if smoothed else 0,
            "early_term": early_terminated,
            "effective_fps": effective_fps,
        }

        actual_fps = total_frames / file_duration if file_duration > 0 and total_frames > 0 else effective_fps
        frame_interval = 1.0 / actual_fps

        padded_energies = [0.0] + energies
        results = []
        for i, is_motion in enumerate(smoothed):
            if early_terminated and i == len(smoothed) - 1:
                time_val = start_offset + file_duration
            else:
                time_val = start_offset + min(i * frame_interval, file_duration)
                
            results.append({
                "time": time_val,
                "is_motion": is_motion,
                "energy": float(padded_energies[i]) if i < len(padded_energies) else 0.0,
            })
        return results, yolo_frames_buffer


def _median_filter(signal: list[float], window: int) -> list[float]:
    if window < 3:
        return signal
    arr = np.array(signal, dtype=np.float32)
    pad_width = window // 2
    padded = np.pad(arr, pad_width, mode='edge')
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
