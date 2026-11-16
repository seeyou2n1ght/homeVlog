from dataclasses import dataclass
from contextlib import contextmanager, nullcontext
import logging
import math
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from typing import NamedTuple

import av
import cv2
import numpy as np

from src.core.utils import parse_res
from src.hardware.scheduler import acquire_with_retry, VideoLease
from src.hardware.ffmpeg import (
    run_ffmpeg, FFmpegProcessRegistry, MediaTerminationUnconfirmed,
    MediaStartTerminationUnconfirmed,
)

# 时空滤波 / 背景建模 / 音频 VAD 算法统一由 src.algorithms.filters 提供（单一实现，防漂移）。
# 此处 re-export 以提供时空滤波与背景建模核心算法接口。
from src.algorithms.filters import (
    AudioEnergyVAD,
    EmaBackgroundModel,
    SpatialGridMotionFilter,
    video_frame_to_gray,
    _median_filter,
    _smooth_labels,
)

def _encode_yuv420p_to_jpeg(raw: bytes, w: int, h: int, gray_fallback: np.ndarray) -> bytes | np.ndarray:
    yuv = np.frombuffer(raw, dtype=np.uint8).reshape((h * 3 // 2, w))
    bgr = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_I420)
    ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if ok:
        return encoded.tobytes()
    return gray_fallback.copy()

__all__ = [
    "AudioEnergyVAD",
    "EmaBackgroundModel",
    "SpatialGridMotionFilter",
    "MotionDetector",
    "KeyframeRecord",
    "_median_filter",
    "_smooth_labels",
    "DecodeStreamDecision",
    "evaluate_stream_completion",
]


class KeyframeRecord(NamedTuple):
    pts: float
    energy: float
    max_cell_energy: float


@dataclass(frozen=True)
class DecodeStreamDecision:
    complete: bool
    truncated: bool
    corrupted_tail: bool
    reason: str


def evaluate_stream_completion(
    total_frames: int,
    file_duration: float,
    effective_fps: float,
    is_aborted: bool = False,
    is_interrupted: bool = False,
    returncode: int | None = 0,
    has_tail_error: bool = False,
    timed_out: bool = False,
) -> DecodeStreamDecision:
    """统一评估管道流 (Pipe) 与内存解码 (PyAV) 的流完整性与截断判定策略。

    不可变纯函数，杜绝双轨解码器的判定门限不同步。
    """
    if is_aborted or is_interrupted:
        return DecodeStreamDecision(complete=False, truncated=False, corrupted_tail=False, reason="INTERRUPTED")

    if timed_out:
        return DecodeStreamDecision(complete=False, truncated=False, corrupted_tail=False, reason="TIMEOUT")

    if total_frames < 2:
        return DecodeStreamDecision(complete=False, truncated=False, corrupted_tail=False, reason="INSUFFICIENT_FRAMES")

    # Frame coverage cannot prove successful termination (device, I/O and
    # watchdog errors may arrive after most frames have already been decoded).
    is_complete = returncode == 0 or has_tail_error

    if not is_complete:
        return DecodeStreamDecision(complete=False, truncated=False, corrupted_tail=False, reason="PROCESS_ERROR")

    # Permit sample rounding, not a percentage of a long recording.
    is_truncated = total_frames + 2 < file_duration * effective_fps
    reason = "CORRUPTED_TAIL" if has_tail_error else ("TRUNCATED_EOF" if is_truncated else "NORMAL_COMPLETION")
    return DecodeStreamDecision(
        complete=True,
        truncated=is_truncated,
        corrupted_tail=has_tail_error,
        reason=reason,
    )


logger = logging.getLogger("homevlog")


def detect_audio_activity(filepath: str, duration: float, config: dict):
    """Run the bounded audio-only gate used for visually static files.

    This avoids a second video decode for files whose full-frame prescreen is
    already static while preserving the conservative rule that an audio event
    keeps the file on the analysis path.
    """
    detector = MotionDetector(config, decode_gpu="cpu")
    detector.has_audio_detected = 1
    features = detector._decode_audio_pipe(filepath, duration)
    if not features.size:
        return [], {}
    return detector.vad.detect_events(features)


class MotionDetector:
    def __init__(self, config: dict, decode_gpu: str = "cuda"):
        det = config.get("detection", {})
        self.config = config
        self.width, self.height = parse_res(det.get("analysis_resolution", "640x360"))
        self.fps = det.get("analysis_fps", 5)
        self.sensitivity = float(det.get("motion_sensitivity", 1.0))
        self.min_motion_threshold = float(det.get("min_motion_threshold", 2.5))
        self.roi = det.get("roi_crop", [0.1, 0.12, 0.8, 0.85])
        self.min_motion_frames = det.get("min_motion_frames", 3)
        self.min_static_frames = det.get("min_static_frames", 5)
        self.noise_suppress = det.get("noise_suppress_frames", 2)
        self.grid_cols = det.get("grid_cols", 8)
        self.grid_rows = det.get("grid_rows", 8)
        self.median_window = det.get("median_filter_window", 7)
        self.decode_timeout = det.get("decode_timeout", 30)
        self.decode_gpu = decode_gpu

        # R1: I 帧关键帧跳帧预筛选与局部精析
        self.iframe_prescreen_enabled = bool(det.get("iframe_prescreen_enabled", True))
        self.iframe_min_file_duration = float(det.get("iframe_min_file_duration", 0.0))
        self.iframe_energy_margin = float(det.get("iframe_energy_margin", 0.80))
        self.iframe_burst_ratio = float(det.get("iframe_burst_ratio", 1.80))
        self.iframe_window_padding = float(det.get("iframe_window_padding", 3.0))
        self.iframe_window_gap_merge = float(det.get("iframe_window_gap_merge", 6.0))
        self.iframe_fallback_active_ratio = float(det.get("iframe_fallback_active_ratio", 0.60))
        self.iframe_max_isolated_windows = int(det.get("iframe_max_isolated_windows", 8))

        # 核心性能开关
        self.adaptive_fps_enabled = det.get("analysis_fps_adaptive", True)
        self.fps_tiers = det.get(
            "analysis_fps_tiers", {"short": 5, "medium": 2, "long": 1, "ultra_long": 0.5}
        )
        self.fps_tier_thresholds = det.get(
            "analysis_fps_tier_thresholds", {"short_max": 120, "medium_max": 600, "long_max": 1800}
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
        self.ambient_drift_suppress = bool(det.get("ambient_drift_suppress", True))
        self.ambient_drift_active_ratio = float(det.get("ambient_drift_active_ratio", 0.35))
        self.ambient_drift_max_energy = float(det.get("ambient_drift_max_energy", 5.0))
        self.camera_motion_suppress = bool(det.get("camera_motion_suppress", True))
        self.camera_motion_active_ratio = float(det.get("camera_motion_active_ratio", 0.65))
        self.camera_motion_max_cov = float(det.get("camera_motion_max_cov", 0.60))
        self.camera_motion_max_focal_ratio = float(det.get("camera_motion_max_focal_ratio", 3.5))

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
        self.yolo_enabled = yolo_cfg.get("enabled", False) and yolo_cfg.get("streaming_verify", True)
        self.yolo_sample_fps = float(yolo_cfg.get("sample_fps", 0.5))
        self.yolo_max_frames = max(1, int(yolo_cfg.get("max_frames_per_file", 512)))

        self.buffer_limit = int(det.get("analysis_buffer_mb", 256) * 1024 * 1024)
        self.last_perf: dict = {}
        self._analysis_start_offset = 0.0


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
            sens_multiplier=1.0 / max(0.001, self.sensitivity),
            base_noise_thresh=self.base_noise_thresh,
            cluster_boost=self.cluster_boost,
            ambient_drift_suppress=self.ambient_drift_suppress,
            ambient_drift_active_ratio=self.ambient_drift_active_ratio,
            ambient_drift_max_energy=self.ambient_drift_max_energy,
            camera_motion_suppress=self.camera_motion_suppress,
            camera_motion_active_ratio=self.camera_motion_active_ratio,
            camera_motion_max_cov=self.camera_motion_max_cov,
            camera_motion_max_focal_ratio=self.camera_motion_max_focal_ratio,
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

        from src.algorithms.motion_trace import MotionTrace
        early_terminated = False

        # R3: Audio VAD event extraction if audio buffer provided
        audio_events: list[tuple[float, float, str]] = []
        vad_stats: dict = {}
        if self.audio_vad_enabled and audio_data is not None and audio_data.size > 0:
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

        trace = frames if isinstance(frames, MotionTrace) else MotionTrace(
            self, effective_fps, is_night_mode=self._is_night_time(start_offset)
        )
        if trace is not frames:
            for frame in frames:
                trace.append(frame)
        energies = list(trace.energies)
        confidences = list(trace.confidences)
        active_cells = list(getattr(trace, "active_cells", ()))

        if len(energies) < 2:
            return [], {"audio_events": audio_events, "vad_stats": vad_stats}

        # Legacy frame counts describe durations at the configured base FPS.
        median_window = max(1, int(round(self.median_window * effective_fps / self.fps)))
        if median_window % 2 == 0:
            median_window += 1
        if median_window >= 3:
            energies = _median_filter(energies, median_window)

        energies_arr = np.array(energies, dtype=np.float32)
        p5 = float(np.percentile(energies_arr, 5))
        p20 = float(np.percentile(energies_arr, 20))
        # Cap adaptive spread/floor so sustained human motion (>20% active) cannot push threshold above real motion.
        noise_spread = min((p20 - p5) * 2.5, max(1.5, self.min_motion_threshold * 0.8))
        base_noise = min(p5, max(self.min_motion_threshold * 2.5, 6.0))
        threshold = max(self.min_motion_threshold, base_noise + noise_spread / max(0.001, self.sensitivity))

        raw_labels = [
            bool(
                e > threshold
                or (i < len(active_cells) and active_cells[i] >= 2 and e >= self.min_motion_threshold * 1.2)
            )
            for i, e in enumerate(energies)
        ]

        smoothed = _smooth_labels(
            raw_labels, max(1, round(self.min_motion_frames * effective_fps / self.fps)),
            max(1, round(self.min_static_frames * effective_fps / self.fps)),
            max(1, round(self.noise_suppress * effective_fps / self.fps))
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
                "raw_energy": float(trace.energies[i]),
                "confidence": float(confidences[i]) if i < len(confidences) else 0.0,
                "is_audio_active": is_audio_active,
                "active_cells": int(trace.active_cells[i]) if hasattr(trace, "active_cells") and i < len(trace.active_cells) else 0,
                "max_cell_energy": float(trace.max_cell_energies[i]) if hasattr(trace, "max_cell_energies") and i < len(trace.max_cell_energies) else 0.0,
            })

        # Sparse analysis samples must not erase short audio events.
        if audio_events and results:
            for ev_start, ev_end, _ in audio_events:
                start = max(start_offset, float(ev_start))
                end = min(start_offset + file_duration, float(ev_end)) if file_duration > 0 else float(ev_end)
                if end <= start:
                    continue
                results.extend((
                    {"time": start, "is_motion": True, "state": "DYNAMIC_AUDIO",
                     "energy": 0.0, "raw_energy": 0.0, "confidence": 0.0, "is_audio_active": True},
                    {"time": end, "is_motion": False, "state": "STATIC",
                     "energy": 0.0, "raw_energy": 0.0, "confidence": 0.0, "is_audio_active": False},
                ))
            results.sort(key=lambda item: item["time"])
            deduped = []
            for label in results:
                if deduped and abs(deduped[-1]["time"] - label["time"]) < 1e-6:
                    if label.get("is_audio_active") or label.get("is_motion"):
                        deduped[-1] = label
                else:
                    deduped.append(label)
            results = deduped

        # 时间轴闭环 (AGENTS.md 铁律): 末帧时间戳严格等于 start_offset + file_duration，
        # 杜绝渲染出的 Vlog 出现时间轴空洞或跳秒
        if file_duration > 0 and results:
            t_close = start_offset + file_duration
            if results[-1]["time"] < t_close - 1e-6:
                results.append({
                    "time": t_close,
                    "is_motion": False,
                    "state": "STATIC",
                    "energy": 0.0,
                    "confidence": 0.0,
                    "is_audio_active": False,
                })

        meta = {
            "has_audio": 1 if audio_events or (audio_data is not None and audio_data.size > 0) else 0,
            "audio_events": audio_events,
            "vad_stats": vad_stats,
            "early_terminated": early_terminated,
        }
        return results, meta

    # ---- 解耦解码与分析：信号量只保护硬件解码阶段 ----

    def _resolve_effective_fps(self, file_duration: float) -> float:
        """按文件时长档位解析自适应分析帧率（超前于解码确定，供管道命令构建）。"""
        if self.adaptive_fps_enabled and file_duration > 0:
            if file_duration <= self.fps_tier_thresholds["short_max"]:
                return self.fps_tiers["short"]
            elif file_duration <= self.fps_tier_thresholds["medium_max"]:
                return self.fps_tiers["medium"]
            elif file_duration <= self.fps_tier_thresholds.get("long_max", 1800):
                return self.fps_tiers["long"]
            else:
                # Explicit ultra-long tier; duration alone does not establish static content.
                return self.fps_tiers.get("ultra_long", self.fps_tiers["long"])
        return self.fps

    def _is_night_time(self, timestamp: float) -> bool:
        presence = self.config.get("presence", {})
        if not presence.get("enabled", False):
            return False
        start, end = presence.get("night_hours", [23, 7])
        hour = (float(timestamp) % 86400.0) / 3600.0
        return (start <= hour < end) if start < end else (hour >= start or hour < end)

    def _decode_file(
        self, filepath: str, file_duration: float = 0.0
    ) -> tuple[list, dict, object, dict]:
        """
        Phase 1 (信号量保护): 硬件解码采样帧到内存缓冲区。
        优先走 ffmpeg 子进程管道（GPU 下采样后仅回传采样帧）；
        管道失败时回退 PyAV 全帧解码路径，保证 NAS 环境下的容错。
        返回 (grayscale_frames, yolo_buffer, audio_samples, metadata)。
        """
        effective_fps = self._resolve_effective_fps(file_duration)
        predecoded_audio = getattr(self, "_predecoded_audio", None)
        self._predecoded_audio = None
        audio_thread = None
        audio_holder = []
        if predecoded_audio is None and getattr(self, "audio_vad_enabled", False) and getattr(self, "has_audio_detected", 1) != 0:
            def _async_audio():
                try:
                    audio_holder.append(self._decode_audio_pipe(filepath, file_duration))
                except Exception as exc:
                    audio_holder.append(exc)
            audio_thread = threading.Thread(target=_async_audio, daemon=True)
            audio_thread.start()

        try:
            frames, yolo_buffer, meta = self._decode_file_pipe(
                filepath, file_duration, effective_fps
            )
            if meta.get("aborted"):
                return [], {}, np.array([], dtype=np.float32), meta
            if frames and meta.get("complete", False):
                full_audio = predecoded_audio
                if full_audio is None and audio_thread is not None:
                    audio_join_timeout = max(60.0, min(file_duration * 0.5, 300.0))
                    audio_thread.join(timeout=audio_join_timeout)
                    if audio_holder:
                        res = audio_holder[0]
                        if isinstance(res, MediaTerminationUnconfirmed):
                            raise res
                        elif isinstance(res, Exception):
                            logger.warning("Audio decode failed for %s: %s; degrading to silent video", Path(filepath).name, res)
                            from src.algorithms.audio_features import AudioFeatures
                            full_audio = AudioFeatures(self.vad_sample_rate, self.vad_window_ms, self.buffer_limit // 4)
                        else:
                            full_audio = res
                    elif audio_thread.is_alive():
                        logger.warning(
                            "Audio decode thread timed out after %.1fs for %s; degrading to silent video",
                            audio_join_timeout, Path(filepath).name,
                        )
                        from src.algorithms.audio_features import AudioFeatures
                        full_audio = AudioFeatures(self.vad_sample_rate, self.vad_window_ms, self.buffer_limit // 4)
                if full_audio is None:
                    try:
                        full_audio = self._decode_audio_pipe(filepath, file_duration)
                    except MediaTerminationUnconfirmed:
                        raise
                    except Exception as exc:
                        logger.warning("Audio decode failed for %s: %s; degrading to silent video", Path(filepath).name, exc)
                        from src.algorithms.audio_features import AudioFeatures
                        full_audio = AudioFeatures(self.vad_sample_rate, self.vad_window_ms, self.buffer_limit // 4)
                meta["has_audio"] = int(bool(full_audio.size) or bool(getattr(self, "has_audio_detected", 0)))
                try:
                    self.has_audio_detected = meta["has_audio"]
                except Exception:
                    pass
                return frames, yolo_buffer, full_audio, meta
            if meta.get("aborted"):
                return [], {}, np.array([], dtype=np.float32), meta
            frames, yolo_buffer = [], {}
            logger.warning(
                "pipe decode incomplete or yielded 0 frames for %s, falling back to PyAV",
                Path(filepath).name,
            )
        except (MemoryError, MediaTerminationUnconfirmed):
            # A second full decode has the same storage bound and cannot recover.
            raise
        except Exception as e:
            logger.warning(
                "pipe decode failed for %s: %s; falling back to PyAV",
                Path(filepath).name, e,
            )
        finally:
            if audio_thread is not None and audio_thread.is_alive():
                audio_thread.join(timeout=2.0)
        return self._decode_file_pyav(filepath, file_duration)

    def _decode_file_pipe(
        self, filepath: str, file_duration: float, effective_fps: float
    ) -> tuple[list, dict, dict]:
        """ffmpeg 子进程管道解码：fps 抽帧 + GPU 缩放后仅下载采样帧。

        对比 PyAV 逐帧全量解码（采样 660 帧需解码 9900 帧并全量 4K 回下载），
        本路径解码侧在 GPU 完成 fps 过滤与缩放，仅 ~660 个 416x234 小帧经管道回传，
        实测可将单文件解码耗时从 ~240s 降至 ~15-30s。
        """
        FFmpegProcessRegistry.reap_cleanups()
        from src.hardware.frame_pool import FramePool
        from src.algorithms.motion_trace import MotionTrace
        decoded_frames = (MotionTrace(self, effective_fps, is_night_mode=self._is_night_time(getattr(self, "_analysis_start_offset", 0.0))) if getattr(self, "_stream_motion", False)
                          else FramePool(self.buffer_limit // 2))
        yolo_buffer: dict[int, np.ndarray] = {}
        jpeg_bytes = 0
        w, h = self.width, self.height
        use_color_yolo = bool(getattr(self, "yolo_enabled", False)) and (w % 2 == 0) and (h % 2 == 0)
        y_plane_bytes = w * h
        frame_size = (y_plane_bytes * 3) // 2 if use_color_yolo else y_plane_bytes
        pix_fmt = "yuvj420p" if use_color_yolo else "gray"
        range_suffix = ":out_range=full" if use_color_yolo else ""
        scale_range = ",scale=out_range=full" if use_color_yolo else ""

        if self.decode_gpu == "cpu":
            from src.hardware.scheduler import get_disk_semaphore
            io_sem = get_disk_semaphore()
            cpu_threads = int(self.config.get("detection", {}).get("cpu_decode_threads", 4))
            hw_args = ["-threads", str(cpu_threads)] if cpu_threads > 0 else []
            vf = f"fps={effective_fps:.3f},scale={w}:{h}{range_suffix}"
        elif self.decode_gpu == "qsv":
            from src.hardware.scheduler import get_qsv_semaphore
            io_sem = VideoLease(get_qsv_semaphore())
            hw_args = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
            vf = f"fps={effective_fps:.3f},scale_qsv=w={w}:h={h},hwdownload,format=nv12{scale_range}"
        else:
            from src.hardware.scheduler import get_nvdec_semaphore
            io_sem = VideoLease(get_nvdec_semaphore())
            hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            vf = f"fps={effective_fps:.3f},scale_cuda={w}:{h},hwdownload,format=nv12{scale_range}"

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            *hw_args, "-i", str(filepath),
            "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
        ]

        # AGENTS.md 铁律：acquire 必须带 timeout 并重试，禁止无限阻塞
        _t_sem = time.monotonic()
        _sem_ok = acquire_with_retry(io_sem, timeout=30.0, retries=6)
        sem_wait = round(time.monotonic() - _t_sem, 2)
        if not _sem_ok:
            logger.warning(
                "decode semaphore acquire timeout for %s, aborting decode",
                Path(filepath).name,
            )
            return decoded_frames, yolo_buffer, {
                "has_audio": 0, "effective_fps": effective_fps,
                "decode_time": 0.0, "frames": 0, "sem_wait": sem_wait,
            }

        yolo_sample_interval = max(
            1, int(round(effective_fps / max(0.1, getattr(self, "yolo_sample_fps", 0.5)))),
            int(np.ceil(max(1.0, file_duration * effective_fps) /
                       max(1, getattr(self, "yolo_max_frames", 512))))
        )
        # 解码弹性看门狗超时：至少 120s，长视频按 file_duration * 1.5 估算，上限放宽至 1800s (30分钟)，杜绝长录像误杀
        watchdog_timeout = max(120.0, min(max(self.decode_timeout * 2, file_duration * 1.5), 1800.0))

        t_decode_start = time.monotonic()
        total_frames = 0
        proc: subprocess.Popen | None = None
        watchdog_timer: threading.Timer | None = None
        timed_out = threading.Event()
        from src.core.utils import TEMP_DIR
        pipe_key = f"pipe_decode_{Path(filepath).name}_{time.monotonic()}"
        safe_stem = Path(filepath).stem.replace(" ", "_")
        err_log = TEMP_DIR / f".pipe_decode_{safe_stem}_{os.getpid()}_{threading.get_ident()}.log"
        try:
            with open(err_log, "wb") as f_err:
                try:
                    proc = FFmpegProcessRegistry.start(pipe_key,cmd,stdout=subprocess.PIPE,stderr=f_err,bufsize=1024*1024)
                except MediaStartTerminationUnconfirmed as error:
                    proc, pipe_key = error.process, error.process_key
                    raise
                assert proc.stdout is not None

                def _kill_on_timeout():
                    try:
                        if proc and not FFmpegProcessRegistry.has_exited(proc):
                            timed_out.set()
                            logger.warning(
                                "pipe decode watchdog timeout (%.1fs) for %s, terminating process",
                                watchdog_timeout, Path(filepath).name,
                            )
                            FFmpegProcessRegistry.terminate(pipe_key)
                    except Exception as error:
                        logger.error("Decode watchdog stop failed for %s: %s", pipe_key, error)

                watchdog_timer = threading.Timer(watchdog_timeout, _kill_on_timeout)
                watchdog_timer.daemon = True
                watchdog_timer.start()

                while True:
                    raw = proc.stdout.read(frame_size)
                    if len(raw) < frame_size:
                        break
                    gray = np.frombuffer(raw[:y_plane_bytes] if use_color_yolo else raw, dtype=np.uint8).reshape((h, w))

                    if getattr(self, "yolo_enabled", False) and total_frames % yolo_sample_interval == 0:
                        item = _encode_yuv420p_to_jpeg(raw, w, h, gray) if use_color_yolo else gray.copy()
                        jpeg_bytes += len(item) if isinstance(item, bytes) else item.nbytes
                        if jpeg_bytes > self.buffer_limit // 4:
                            raise MemoryError("YOLO candidate budget exceeded")
                        yolo_buffer[total_frames] = item

                    decoded_frames.append(gray)
                    total_frames += 1
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                timed_out.set()
                FFmpegProcessRegistry.terminate(pipe_key)
                FFmpegProcessRegistry.wait_for_exit(proc, timeout=5)
            except BaseException:
                if proc:
                    try:
                        FFmpegProcessRegistry.terminate(pipe_key)
                    except Exception:
                        pass
                raise
        finally:
            if watchdog_timer:
                watchdog_timer.cancel()
            try:
                if proc is not None:
                    FFmpegProcessRegistry.terminate(pipe_key)
                    FFmpegProcessRegistry.wait_for_exit(proc, timeout=5)
                if watchdog_timer is not None:
                    watchdog_timer.join(timeout=2)
                    if watchdog_timer.is_alive():
                        raise RuntimeError("Decode watchdog termination unconfirmed")
            except BaseException as error:
                readers = [watchdog_timer] if watchdog_timer is not None else []
                FFmpegProcessRegistry.defer_cleanup(pipe_key, proc, io_sem, readers)
                logger.error("Decode cleanup unconfirmed; retained process, original lease and stderr: %s", pipe_key)
                raise MediaTerminationUnconfirmed(f"Decode cleanup unconfirmed: {pipe_key}") from error
            if proc is not None:
                FFmpegProcessRegistry.deregister(pipe_key, proc)
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except OSError:
                    pass
            is_aborted = False
            err_tail = ""
            if err_log.exists():
                try:
                    err_tail = err_log.read_bytes().decode('utf-8', errors='ignore')[-500:]
                    err_log.unlink(missing_ok=True)
                except Exception:
                    pass
            if proc:
                if "received signal 2" in err_tail or proc.returncode in (255, -2, 130) or FFmpegProcessRegistry.is_interrupted():
                    is_aborted = True
                elif not decoded_frames and proc.returncode != 0 and err_tail.strip():
                    logger.warning("pipe decode stderr for %s: %s", Path(filepath).name, err_tail.strip())
            io_sem.release()

            decision = evaluate_stream_completion(
                total_frames=total_frames,
                file_duration=file_duration,
                effective_fps=effective_fps,
                is_aborted=is_aborted,
                is_interrupted=FFmpegProcessRegistry.is_interrupted(),
                returncode=proc.returncode if proc else -1,
                has_tail_error=False,
                timed_out=timed_out.is_set(),
            )
            is_complete = decision.complete
            is_truncated = decision.truncated
            if is_truncated:
                logger.info(
                    "Pipe decode for %s reached early EOF cleanly: container declared %.1fs, physical stream has %d frames (%.1fs)",
                    Path(filepath).name, file_duration, total_frames, total_frames / effective_fps,
                )

        meta = {
            "has_audio": 0,
            "effective_fps": effective_fps,
            "decode_time": round(time.monotonic() - t_decode_start, 3),
            "frames": total_frames,
            "sem_wait": sem_wait,
            "aborted": is_aborted,
            "complete": is_complete,
            "truncated": is_truncated,
            "completion_reason": decision.reason,
            "completion_error": err_tail if not decision.complete else "",
        }
        return decoded_frames, yolo_buffer, meta

    def _decode_audio_pipe(self, filepath: str, file_duration: float):
        from src.algorithms.audio_features import AudioFeatures
        features = AudioFeatures(self.vad_sample_rate, self.vad_window_ms, self.buffer_limit // 4)
        if not self.audio_vad_enabled or getattr(self, "has_audio_detected", None) == 0:
            return features
        pending = bytearray()
        def consume(data):
            pending.extend(data)
            count = len(pending) // 4 * 4
            if count:
                features.append(np.frombuffer(bytes(pending[:count]), dtype=np.float32))
                del pending[:count]
        result = run_ffmpeg(
            ["-vn", "-i", str(filepath), "-ac", "1", "-ar", str(self.vad_sample_rate), "-f", "f32le", "-"],
            timeout=max(120, min(file_duration * .5, 1800)), stdout_consumer=consume,
        )
        if result.returncode != 0 or pending:
            raise RuntimeError("Audio decode incomplete: " + result.stderr_text[-300:])
        return features

    def _decode_file_pyav(
        self, filepath: str, file_duration: float = 0.0
    ) -> tuple[list, dict, object, dict]:
        """
        Phase 1 (信号量保护): 硬件解码全部帧到内存缓冲区。
        返回 (grayscale_frames, yolo_buffer, audio_samples, metadata)。
        """
        from src.hardware.frame_pool import FramePool
        decoded_frames = FramePool(self.buffer_limit // 2)
        yolo_buffer: dict[int, np.ndarray] = {}
        jpeg_bytes = 0

        if self.decode_gpu == "cpu":
            from src.hardware.scheduler import get_disk_semaphore
            io_sem = get_disk_semaphore()
            hw_name = None
        elif self.decode_gpu == "qsv":
            from src.hardware.scheduler import get_qsv_semaphore
            io_sem = VideoLease(get_qsv_semaphore())
            hw_name = "qsv"
        else:
            from src.hardware.scheduler import get_nvdec_semaphore
            io_sem = VideoLease(get_nvdec_semaphore())
            hw_name = "cuda"

        # AGENTS.md 铁律：acquire 必须带 timeout 并重试，禁止无限阻塞
        _t_sem = time.monotonic()
        _sem_ok = acquire_with_retry(io_sem, timeout=30.0, retries=6)
        sem_wait = round(time.monotonic() - _t_sem, 2)
        if not _sem_ok:
            logger.warning(
                "decode semaphore acquire timeout for %s, aborting decode",
                Path(filepath).name,
            )
            return decoded_frames, yolo_buffer, np.array([], dtype=np.float32), {
                "has_audio": 0,
                "effective_fps": self.fps,
                "decode_time": 0.0,
                "frames": 0,
                "sem_wait": sem_wait,
            }

        from src.algorithms.audio_features import AudioFeatures
        audio_features = AudioFeatures(self.vad_sample_rate, self.vad_window_ms, self.buffer_limit // 4)
        audio_resampler = None
        has_audio = 0
        effective_fps = self.fps
        total_frames = 0
        video_frame_count = 0
        decode_complete = False
        meta_corrupted_tail = False
        timed_out = False
        completion_error = ""

        try:
            from av.audio.resampler import AudioResampler
            audio_resampler = AudioResampler(format="fltp", layout="mono", rate=self.vad_sample_rate)
        except Exception as e:
            logger.debug(f"PyAV AudioResampler init: {e}")

        t_decode_start = time.monotonic()
        try:
            hw = None
            if hw_name is not None:
                try:
                    from av.codec.hwaccel import HWAccel
                    hw = HWAccel(hw_name)
                except Exception as e:
                    logger.debug(f"PyAV HWAccel init failed for {hw_name}: {e}")

            kwargs = {}
            if hw:
                kwargs["hwaccel"] = hw

            with av.open(str(filepath), timeout=30.0, **kwargs) as container:
                video_stream = container.streams.video[0] if container.streams.video else None
                audio_stream = container.streams.audio[0] if container.streams.audio else None

                has_audio = 1 if audio_stream is not None else 0
                try:
                    self.has_audio_detected = has_audio
                except Exception:
                    pass

                if video_stream is None:
                    logger.warning("No video stream found in %s", Path(filepath).name)
                    return decoded_frames, yolo_buffer, np.array([], dtype=np.float32), {
                        "has_audio": 0, "effective_fps": effective_fps,
                        "decode_time": time.monotonic() - t_decode_start, "frames": 0,
                    }

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
                    elif file_duration <= self.fps_tier_thresholds.get("long_max", 1800):
                        effective_fps = self.fps_tiers["long"]
                    else:
                        # Explicit ultra-long tier; duration alone does not establish static content.
                        effective_fps = self.fps_tiers.get("ultra_long", self.fps_tiers["long"])
                else:
                    effective_fps = self.fps

                frame_step = max(1, int(round(video_fps / effective_fps)))
                effective_fps = video_fps / frame_step
                if getattr(self, "_stream_motion", False):
                    from src.algorithms.motion_trace import MotionTrace
                    decoded_frames = MotionTrace(self, effective_fps, is_night_mode=self._is_night_time(getattr(self, "_analysis_start_offset", 0.0)))

                # YOLO 抽样间隔必须以实际解码 effective_fps 为基准，
                # 保证 yolo_buffer 帧键与分析阶段时间轴严格对齐
                yolo_sample_interval = max(
                    1, int(round(effective_fps / max(0.1, getattr(self, "yolo_sample_fps", 0.5)))),
                    int(np.ceil(max(1.0, file_duration * effective_fps) /
                               max(1, getattr(self, "yolo_max_frames", 512))))
                )

                watchdog_timeout = max(self.decode_timeout * 3, 60.0)
                if file_duration > 0:
                    watchdog_timeout = max(watchdog_timeout, (file_duration / effective_fps) * 4)

                streams_to_decode = [video_stream]
                if self.audio_vad_enabled and audio_stream is not None:
                    streams_to_decode.append(audio_stream)

                for frame in container.decode(*streams_to_decode):
                    if FFmpegProcessRegistry.is_interrupted():
                        break
                    if isinstance(frame, av.AudioFrame) or getattr(frame, "type", "") == "audio":
                        try:
                            if audio_resampler:
                                resampled_frames = audio_resampler.resample(frame)
                                if resampled_frames:
                                    for rf in resampled_frames:
                                        audio_features.append(rf.to_ndarray().flatten().astype(np.float32))
                            else:
                                raw = frame.to_ndarray()
                                if raw.ndim > 1:
                                    raw = np.mean(raw, axis=0)
                                audio_features.append(raw.flatten().astype(np.float32))
                        except MemoryError:
                            raise
                        except Exception as e:
                            logger.debug("Audio decode chunk failed: %s", e)
                        continue

                    video_frame_count += 1
                    if (video_frame_count - 1) % frame_step != 0:
                        continue

                    total_frames += 1

                    if getattr(self, "yolo_enabled", False) and (total_frames - 1) % yolo_sample_interval == 0:
                        try:
                            bgr_tmp = frame.reformat(
                                width=self.width, height=self.height, format="bgr24"
                            ).to_ndarray()
                        except Exception:
                            try:
                                bgr_tmp = cv2.resize(
                                    frame.to_ndarray(format="bgr24"),
                                    (self.width, self.height),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                            except Exception:
                                rgb_raw = cv2.resize(
                                    frame.to_ndarray(format="rgb24"),
                                    (self.width, self.height),
                                    interpolation=cv2.INTER_LINEAR,
                                )
                                bgr_tmp = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                        item = bgr_tmp
                        jpeg_bytes += item.nbytes
                        if jpeg_bytes > self.buffer_limit // 4:
                            raise MemoryError("YOLO candidate budget exceeded")
                        yolo_buffer[total_frames - 1] = item

                    gray = cv2.resize(
                        video_frame_to_gray(frame),
                        (self.width, self.height),
                        interpolation=cv2.INTER_NEAREST,
                    )

                    decoded_frames.append(gray)

                    if time.monotonic() - t_decode_start > watchdog_timeout:
                        timed_out = True
                        logger.warning(
                            "PyAV decode timeout for %s (dur=%.1f, elapsed=%.1f)",
                            Path(filepath).name, file_duration,
                            time.monotonic() - t_decode_start,
                        )
                        break

                else:
                    decode_complete = (
                        not FFmpegProcessRegistry.is_interrupted() and
                        total_frames >= 2
                    )

                if audio_resampler:
                    try:
                        flushed = audio_resampler.resample(None)
                        if flushed:
                            for rf in flushed:
                                audio_features.append(rf.to_ndarray().flatten().astype(np.float32))
                    except Exception:
                        pass

        except Exception as e:
            completion_error = f"{type(e).__name__}: {e}"
            logger.warning("PyAV decode error for %s: %s", Path(filepath).name, e)
            if (isinstance(e, (av.error.InvalidDataError, av.error.EOFError))
                    and not FFmpegProcessRegistry.is_interrupted() and total_frames >= 2):
                decode_complete = True
                meta_corrupted_tail = True
                logger.info(
                    "PyAV decode error for %s at frame %d: %s. Preserving %d decoded frames.",
                    Path(filepath).name, total_frames, e, total_frames,
                )
        finally:
            io_sem.release()

        t_decode_end = time.monotonic()

        full_audio = audio_features

        decision = evaluate_stream_completion(
            total_frames=total_frames,
            file_duration=file_duration,
            effective_fps=effective_fps,
            is_aborted=False,
            is_interrupted=FFmpegProcessRegistry.is_interrupted(),
            returncode=0 if decode_complete else -1,
            has_tail_error=meta_corrupted_tail,
            timed_out=timed_out,
        )
        decode_complete = decision.complete
        meta_truncated = decision.truncated
        meta = {
            "has_audio": has_audio,
            "effective_fps": effective_fps,
            "complete": decode_complete,
            "decode_time": round(t_decode_end - t_decode_start, 3),
            "frames": total_frames,
            "sem_wait": sem_wait,
            "truncated": meta_truncated,
            "corrupted_tail": meta_corrupted_tail,
            "completion_reason": decision.reason,
            "completion_error": completion_error if not decision.complete else "",
        }
        return decoded_frames, yolo_buffer, full_audio, meta

    # ---- R1: I-Frame 关键帧跳帧预筛选与局部精析 (Tier 1 & Tier 2) ----

    def _extract_keyframe_pts(self, filepath: str) -> tuple[list[float], float, float, float]:
        """Pre-extract exact keyframe PTS timestamps, container duration, avg GOP, and max GOP.
        
        Uses PyAV container.demux() without video decoding (0.2s - 1.5s over SMB).
        Returns (pts_list, duration, avg_gop, max_gop).
        """
        from src.hardware.scheduler import get_disk_semaphore
        disk = get_disk_semaphore()
        if not acquire_with_retry(disk, timeout=30.0, retries=6):
            raise TimeoutError(f"I/O semaphore timeout probing keyframes for {filepath}")
        try:
            with av.open(str(filepath), timeout=30.0) as container:
                if not container.streams.video:
                    raise ValueError(f"No video stream in {filepath}")
                stream = container.streams.video[0]
                tb = float(stream.time_base) if stream.time_base is not None else (1.0 / 30.0)
                pts_list: list[float] = []
                for p in container.demux(stream):
                    if not p.is_keyframe:
                        continue
                    raw_ts = p.pts if p.pts is not None else p.dts
                    if raw_ts is not None:
                        pts = max(0.0, float(raw_ts * tb))
                    elif pts_list:
                        pts = pts_list[-1] + 3.0
                    else:
                        pts = 0.0
                    pts_list.append(pts)

                duration = (
                    float(stream.duration * tb) if (stream.duration is not None and stream.duration > 0)
                    else (float(container.duration or 0) / av.time_base if container.duration else 0.0)
                )
                if not pts_list:
                    return [], duration, 0.0, 0.0

                if duration <= 0.0:
                    duration = pts_list[-1] + 3.0

                gops = [b - a for a, b in zip(pts_list, pts_list[1:]) if b >= a]
                avg_gop = max(0.1, duration / max(1, len(pts_list)))
                max_gop = max(gops) if gops else avg_gop
                return pts_list, duration, avg_gop, max_gop
        finally:
            disk.release()

    def _build_coarse_scan_cmd(
        self, filepath: str, decode_gpu: str, file_duration: float = 0.0
    ) -> tuple[list[str], object]:
        """Construct FFmpeg command and acquire device lease for Tier 1 scan.
        
        Note: QSV hardware decoder ignores -skip_frame nokey; therefore 'qsv' and 'cpu'
        fall back to CPU software decode (127x speed, 0 GPU VRAM).
        """
        w, h = self.width, self.height
        if decode_gpu == "cuda":
            from src.hardware.scheduler import get_nvdec_semaphore, VideoLease
            io_sem = VideoLease(get_nvdec_semaphore())
            hw_args = ["-hwaccel", "cuda"]
        else:
            from src.hardware.scheduler import get_disk_semaphore
            io_sem = get_disk_semaphore()
            hw_args = []
        vf = f"scale={w}:{h},showinfo"
        demux_skip = ["-discard", "nokey"] if file_duration >= 180.0 else []

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "info", "-y",
            *demux_skip, "-skip_frame", "nokey",
            *hw_args, "-i", str(filepath),
            "-vf", vf,
            "-fps_mode", "passthrough",
            "-f", "rawvideo", "-pix_fmt", "gray", "-",
        ]
        return cmd, io_sem

    def _scan_keyframes_coarse(
        self,
        filepath: str,
        file_duration: float = 0.0,
        keyframe_pts: list[float] | None = None,
        decode_gpu: str = "cuda",
    ) -> tuple[list[KeyframeRecord], dict]:
        """Tier 1: Read only keyframes, extract PTS, calculate motion energy, EMA background update,
        and 8x8 spatial grid cell max energy in-memory.
        """
        FFmpegProcessRegistry.reap_cleanups()
        avg_gop = 3.0
        max_gop = 3.0
        if keyframe_pts is None and "_extract_keyframe_pts" in self.__dict__:
            try:
                keyframe_pts, extracted_dur, avg_gop, max_gop = self._extract_keyframe_pts(filepath)
                if file_duration <= 0.0:
                    file_duration = extracted_dur
            except Exception as e:
                logger.warning("Keyframe extraction failed for %s: %s", Path(filepath).name, e)
                return [], {"complete": False, "reason": "DEMUX_FAILED"}
            if not keyframe_pts:
                return [], {"complete": False, "reason": "NO_KEYFRAMES"}
        elif keyframe_pts is not None and not keyframe_pts:
            return [], {"complete": False, "reason": "NO_KEYFRAMES"}

        try:
            cmd, io_sem = self._build_coarse_scan_cmd(filepath, decode_gpu, file_duration=file_duration)
        except TypeError:
            cmd, io_sem = self._build_coarse_scan_cmd(filepath, decode_gpu)

        if not acquire_with_retry(io_sem, timeout=30.0, retries=6):
            logger.warning("Semaphore acquire timeout for coarse scan of %s", Path(filepath).name)
            return [], {"complete": False, "reason": "SEMAPHORE_TIMEOUT"}

        w, h = self.width, self.height
        frame_size = w * h
        x, y = int(w * self.roi[0]), int(h * self.roi[1])
        rw, rh = int(w * self.roi[2]), int(h * self.roi[3])

        ema = self.create_ema_model()
        grid = self.create_grid_filter()

        pipe_key = f"pipe_coarse_{Path(filepath).name}_{time.monotonic()}"
        safe_stem = Path(filepath).stem.replace(" ", "_")
        from src.core.utils import TEMP_DIR
        err_log = TEMP_DIR / f".pipe_coarse_{safe_stem}_{os.getpid()}_{threading.get_ident()}.log"
        watchdog_timeout = max(30.0, min(file_duration * 0.5, 120.0))

        proc = None
        watchdog_timer = None
        timed_out = threading.Event()
        records: list[KeyframeRecord] = []
        raw_rois: list[np.ndarray] = []
        err_text = ""
        t_start = time.monotonic()

        try:
            with open(err_log, "wb") as f_err:
                try:
                    proc = FFmpegProcessRegistry.start(
                        pipe_key, cmd, stdout=subprocess.PIPE, stderr=f_err, bufsize=1024*1024
                    )
                except MediaStartTerminationUnconfirmed as error:
                    proc, pipe_key = error.process, error.process_key
                    raise
                assert proc.stdout is not None

                def _kill_on_timeout():
                    try:
                        if proc and not FFmpegProcessRegistry.has_exited(proc):
                            timed_out.set()
                            FFmpegProcessRegistry.terminate(pipe_key)
                    except Exception as error:
                        logger.error("Coarse scan watchdog stop failed for %s: %s", pipe_key, error)

                watchdog_timer = threading.Timer(watchdog_timeout, _kill_on_timeout)
                watchdog_timer.daemon = True
                watchdog_timer.start()

                while True:
                    raw = proc.stdout.read(frame_size)
                    if len(raw) < frame_size:
                        break
                    gray = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))
                    roi = gray[y:y+rh, x:x+rw]
                    raw_rois.append(roi.copy())

                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    timed_out.set()
                    FFmpegProcessRegistry.terminate(pipe_key)
                    FFmpegProcessRegistry.wait_for_exit(proc, timeout=5)
                except BaseException:
                    if proc:
                        try:
                            FFmpegProcessRegistry.terminate(pipe_key)
                        except Exception:
                            pass
                    raise
        finally:
            if watchdog_timer:
                watchdog_timer.cancel()
            try:
                if proc is not None:
                    FFmpegProcessRegistry.terminate(pipe_key)
                    FFmpegProcessRegistry.wait_for_exit(proc, timeout=5)
                if watchdog_timer is not None:
                    watchdog_timer.join(timeout=2)
                    if watchdog_timer.is_alive():
                        raise RuntimeError("Coarse scan watchdog termination unconfirmed")
            except BaseException as error:
                readers = [watchdog_timer] if watchdog_timer is not None else []
                FFmpegProcessRegistry.defer_cleanup(pipe_key, proc, io_sem, readers)
                logger.error("Coarse scan cleanup unconfirmed; retained process, original lease and stderr: %s", pipe_key)
                raise MediaTerminationUnconfirmed(f"Coarse scan cleanup unconfirmed: {pipe_key}") from error
            if proc is not None:
                FFmpegProcessRegistry.deregister(pipe_key, proc)
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except OSError:
                    pass
            try:
                if err_log.exists():
                    err_text = err_log.read_bytes().decode("utf-8", errors="ignore")
            except OSError:
                pass
            err_log.unlink(missing_ok=True)
            io_sem.release()

        if not raw_rois:
            reason = "DEMUX_FAILED" if (proc is None or (proc.returncode is not None and proc.returncode != 0)) else "NO_KEYFRAMES"
            return [], {"complete": False, "reason": reason}

        if keyframe_pts is None:
            pts_list = [float(m) for m in re.findall(r"pts_time:([0-9.]+)", err_text)]
            if pts_list:
                keyframe_pts = pts_list
                duration = file_duration if file_duration > 0.0 else (pts_list[-1] + 3.0)
                if file_duration <= 0.0:
                    file_duration = duration
                gops = [b - a for a, b in zip(pts_list, pts_list[1:]) if b >= a]
                avg_gop = max(0.1, duration / max(1, len(pts_list)))
                max_gop = max(gops) if gops else avg_gop
            else:
                try:
                    keyframe_pts, extracted_dur, avg_gop, max_gop = self._extract_keyframe_pts(filepath)
                    if file_duration <= 0.0:
                        file_duration = extracted_dur
                except Exception as e:
                    logger.warning("Keyframe extraction failed for %s: %s", Path(filepath).name, e)
                    return [], {"complete": False, "reason": "DEMUX_FAILED"}
                if not keyframe_pts:
                    return [], {"complete": False, "reason": "NO_KEYFRAMES"}

        is_long_keyframe_stream = len(raw_rois) >= 60 or file_duration >= 180.0
        if is_long_keyframe_stream and self.ema_enabled:
            ema.alpha_bg = min(0.5, 1.0 - (1.0 - self.ema_alpha_bg) ** 15)
            ema.alpha_fg = min(0.1, 1.0 - (1.0 - self.ema_alpha_fg) ** 15)

        prev_t = 0.0
        prev_roi = None
        for frame_idx, roi in enumerate(raw_rois):
            t = keyframe_pts[frame_idx] if frame_idx < len(keyframe_pts) else (prev_t + 3.0)
            dt = max(0.1, t - prev_t) if frame_idx > 0 else 3.0
            prev_t = t

            if self.ema_enabled:
                saliency, _, _ = ema.update(roi)
            else:
                saliency = cv2.absdiff(roi, prev_roi).astype(np.float32) if prev_roi is not None else np.zeros_like(roi, dtype=np.float32)
                prev_roi = roi.copy()

            night_flag = False if is_long_keyframe_stream else self._is_night_time(getattr(self, "_analysis_start_offset", 0.0) + t)
            eff_energy, is_grid_motion, stats = grid.process_frame(
                saliency, dt, is_night_mode=night_flag
            )
            max_cell_e = float(stats.get("max_cell_energy", 0.0))

            records.append(KeyframeRecord(
                pts=t,
                energy=eff_energy,
                max_cell_energy=max_cell_e,
            ))

        meta = {
            "complete": (proc.returncode == 0 if proc else False) and len(records) > 0,
            "keyframes": len(records),
            "frames": len(records),
            "scan_time": round(time.monotonic() - t_start, 3),
            "avg_gop_s": avg_gop,
            "avg_gop": avg_gop,
            "max_gop_s": max_gop,
            "max_gop": max_gop,
        }
        return records, meta

    def _synthesize_static_labels(
        self,
        start_offset: float,
        file_duration: float,
        effective_fps: float,
    ) -> list[dict]:
        """Synthesize uniform STATIC labels for pure static clips with zero fine decoding.
        Guarantees strict timeline closure and exact timestamp consistency.
        """
        if file_duration <= 0.0:
            return []

        dt = 1.0 / effective_fps if effective_fps > 0 else 0.2
        total_frames = max(2, int(round(file_duration * effective_fps)))

        labels: list[dict] = []
        for i in range(total_frames):
            t_rel = min(i * dt, file_duration)
            labels.append({
                "time": start_offset + t_rel,
                "is_motion": False,
                "state": "STATIC",
                "energy": 0.0,
                "raw_energy": 0.0,
                "confidence": 0.0,
                "is_audio_active": False,
                "active_cells": 0,
                "max_cell_energy": 0.0,
            })

        t_close = start_offset + file_duration
        if abs(labels[-1]["time"] - t_close) > 1e-6:
            if labels[-1]["time"] < t_close:
                labels.append({
                    "time": t_close,
                    "is_motion": False,
                    "state": "STATIC",
                    "energy": 0.0,
                    "raw_energy": 0.0,
                    "confidence": 0.0,
                    "is_audio_active": False,
                    "active_cells": 0,
                    "max_cell_energy": 0.0,
                })
            else:
                labels[-1]["time"] = t_close

        return labels

    def _resolve_awakened_windows(
        self,
        keyframe_trace: list[tuple[float, float, float]],
        audio_events: list[tuple[float, float, str]],
        file_duration: float,
        threshold: float,
        gop_duration: float = 3.0,
    ) -> list[tuple[float, float]]:
        """Convert active keyframes and Audio VAD events into padded, merged temporal windows.
        All timestamps are file-relative in seconds within [0.0, file_duration].
        """
        if file_duration <= 0.0:
            return []

        eff_threshold = threshold * self.iframe_energy_margin
        burst_threshold = threshold * self.iframe_burst_ratio
        padding = max(self.iframe_window_padding, gop_duration)
        gap_merge = self.iframe_window_gap_merge

        raw_windows: list[tuple[float, float]] = []

        # 1. Visual active keyframes expansion
        for item in keyframe_trace:
            pts, energy, max_cell_energy = item[0], item[1], item[2]
            is_active = (energy > eff_threshold) or (max_cell_energy >= burst_threshold)
            if is_active:
                w_start = max(0.0, pts - padding)
                w_end = min(file_duration, pts + padding)
                if w_end > w_start:
                    raw_windows.append((w_start, w_end))

        # 2. Multimodal Audio VAD events expansion (unconditional awakening)
        for ev in audio_events:
            ev_start, ev_end = ev[0], ev[1]
            ev_s = max(0.0, float(ev_start))
            ev_e = min(file_duration, float(ev_end))
            if ev_e > ev_s:
                w_start = max(0.0, ev_s - padding)
                w_end = min(file_duration, ev_e + padding)
                raw_windows.append((w_start, w_end))

        if not raw_windows:
            return []

        # 3. Chronological sorting
        raw_windows.sort(key=lambda w: w[0])

        # 4. Adjacent window merging (gap <= iframe_window_gap_merge)
        merged: list[list[float]] = []
        for w_start, w_end in raw_windows:
            if not merged:
                merged.append([w_start, w_end])
            else:
                prev_start, prev_end = merged[-1]
                if w_start <= prev_end + gap_merge:
                    merged[-1][1] = max(prev_end, w_end)
                else:
                    merged.append([w_start, w_end])

        # 5. Strict boundary enforcement [0.0, file_duration]
        final_windows: list[tuple[float, float]] = []
        for start, end in merged:
            s = max(0.0, round(start, 3))
            e = min(file_duration, round(end, 3))
            if e > s:
                final_windows.append((s, e))

        return final_windows

    def _evaluate_fallback_safeguards(
        self,
        windows: list[tuple[float, float]],
        file_duration: float,
        gop_metrics: dict,
        filepath: str = "",
    ) -> tuple[bool, str]:
        """Evaluate fallback safeguards for Tier 2 Awakened Window Fine Decoding.
        Returns (should_fallback: bool, reason: str).
        """
        keyframe_count = gop_metrics.get("keyframes", gop_metrics.get("keyframe_count", 0))
        avg_gop = gop_metrics.get("avg_gop_s", gop_metrics.get("avg_gop", 3.0))
        max_gop = gop_metrics.get("max_gop_s", gop_metrics.get("max_gop", 3.0))

        # 1. Sparse GOP Safeguard
        if keyframe_count < 2:
            return True, "INSUFFICIENT_KEYFRAMES"

        if avg_gop > 8.0:
            return True, f"FALLBACK_SPARSE_GOP_{avg_gop:.1f}S"

        if max_gop > 15.0 and keyframe_count >= 3:
            return True, f"FALLBACK_SPARSE_GOP_{max_gop:.1f}S"

        # 2. Max Isolated Windows Safeguard (scales proportionally for long clips > 600s)
        max_isolated = (
            max(self.iframe_max_isolated_windows, int(self.iframe_max_isolated_windows * math.ceil(file_duration / 600.0)))
            if file_duration > 600.0
            else self.iframe_max_isolated_windows
        )
        if len(windows) > max_isolated:
            return True, f"FALLBACK_ISOLATED_WINDOWS_{len(windows)}_GT_{max_isolated}"

        # 3. Active Duration Ratio Safeguard
        total_active_duration = sum(w_end - w_start for w_start, w_end in windows)
        active_ratio = total_active_duration / max(0.001, file_duration)
        if active_ratio > self.iframe_fallback_active_ratio:
            return True, "FALLBACK_DENSE_MOTION_ACTIVE_RATIO"

        return False, "PROCEED_TIER2"

    def _decode_file_window(
        self,
        filepath: str,
        w_start: float,
        w_end: float,
        effective_fps: float | None = None,
        yolo_buffer: dict[int, np.ndarray] | None = None,
        yolo_sample_interval: int | None = None,
        file_duration: float = 0.0,
    ) -> tuple[object, int]:
        """Tier 2: Decode a single active window [w_start, w_end] via FFmpeg seek (-ss -to).
        Stream frames directly into MotionTrace and yolo_buffer.
        Returns (trace, frames_decoded).
        """
        from src.algorithms.motion_trace import MotionTrace
        if effective_fps is None:
            effective_fps = self._resolve_effective_fps(file_duration)
        if yolo_buffer is None:
            yolo_buffer = {}
        if yolo_sample_interval is None:
            yolo_sample_interval = max(
                1, int(round(effective_fps / max(0.1, getattr(self, "yolo_sample_fps", 0.5)))),
            )

        w, h = self.width, self.height
        use_color_yolo = bool(getattr(self, "yolo_enabled", False)) and (w % 2 == 0) and (h % 2 == 0)
        y_plane_bytes = w * h
        frame_size = (y_plane_bytes * 3) // 2 if use_color_yolo else y_plane_bytes
        pix_fmt = "yuvj420p" if use_color_yolo else "gray"
        range_suffix = ":out_range=full" if use_color_yolo else ""
        scale_range = ",scale=out_range=full" if use_color_yolo else ""
        window_duration = max(0.001, w_end - w_start)

        trace = MotionTrace(
            self, effective_fps,
            is_night_mode=self._is_night_time(getattr(self, "_analysis_start_offset", 0.0) + w_start)
        )

        if self.decode_gpu == "cpu":
            from src.hardware.scheduler import get_disk_semaphore
            io_sem = get_disk_semaphore()
            cpu_threads = int(self.config.get("detection", {}).get("cpu_decode_threads", 4))
            hw_args = ["-threads", str(cpu_threads)] if cpu_threads > 0 else []
            vf = f"fps={effective_fps:.3f},scale={w}:{h}{range_suffix}"
        elif self.decode_gpu == "qsv":
            from src.hardware.scheduler import get_qsv_semaphore, VideoLease
            io_sem = VideoLease(get_qsv_semaphore())
            hw_args = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
            vf = f"fps={effective_fps:.3f},scale_qsv=w={w}:h={h},hwdownload,format=nv12{scale_range}"
        else:  # cuda
            from src.hardware.scheduler import get_nvdec_semaphore, VideoLease
            io_sem = VideoLease(get_nvdec_semaphore())
            hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            vf = f"fps={effective_fps:.3f},scale_cuda={w}:{h},hwdownload,format=nv12{scale_range}"

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            "-ss", f"{w_start:.3f}", "-to", f"{w_end:.3f}",
            *hw_args, "-i", str(filepath),
            "-vf", vf, "-f", "rawvideo", "-pix_fmt", pix_fmt, "-",
        ]

        _sem_ok = acquire_with_retry(io_sem, timeout=30.0, retries=6)
        if not _sem_ok:
            raise TimeoutError(f"Semaphore acquire timeout for window [{w_start}, {w_end}] of {Path(filepath).name}")

        watchdog_timeout = max(30.0, min(window_duration * 3.0, 300.0))
        proc = None
        watchdog_timer = None
        timed_out = threading.Event()
        pipe_key = f"pipe_window_{Path(filepath).name}_{w_start:.1f}_{time.monotonic()}"
        from src.core.utils import TEMP_DIR
        safe_stem = Path(filepath).stem.replace(" ", "_")
        err_log = TEMP_DIR / f".pipe_win_{safe_stem}_{int(w_start)}_{os.getpid()}_{threading.get_ident()}.log"

        frames_decoded = 0
        dt = 1.0 / effective_fps
        try:
            with open(err_log, "wb") as f_err:
                try:
                    proc = FFmpegProcessRegistry.start(pipe_key, cmd, stdout=subprocess.PIPE, stderr=f_err, bufsize=1024*1024)
                except MediaStartTerminationUnconfirmed as error:
                    proc, pipe_key = error.process, error.process_key
                    raise

                assert proc.stdout is not None

                def _kill_on_timeout():
                    try:
                        if proc and not FFmpegProcessRegistry.has_exited(proc):
                            timed_out.set()
                            FFmpegProcessRegistry.terminate(pipe_key)
                    except Exception as error:
                        logger.error("Window decode watchdog stop failed for %s: %s", pipe_key, error)

                watchdog_timer = threading.Timer(watchdog_timeout, _kill_on_timeout)
                watchdog_timer.daemon = True
                watchdog_timer.start()

                while True:
                    raw = proc.stdout.read(frame_size)
                    if len(raw) < frame_size:
                        break
                    gray = np.frombuffer(raw[:y_plane_bytes] if use_color_yolo else raw, dtype=np.uint8).reshape((h, w))

                    # Global frame index for yolo_buffer alignment with yolo_verifier
                    t_frame_rel = min(w_start + frames_decoded * dt, w_end)
                    global_frame_idx = int(round(t_frame_rel * effective_fps))

                    if getattr(self, "yolo_enabled", False) and global_frame_idx % yolo_sample_interval == 0:
                        yolo_buffer[global_frame_idx] = (
                            _encode_yuv420p_to_jpeg(raw, w, h, gray) if use_color_yolo else gray.copy()
                        )

                    trace.append(gray)
                    frames_decoded += 1

            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                timed_out.set()
                FFmpegProcessRegistry.terminate(pipe_key)
                FFmpegProcessRegistry.wait_for_exit(proc, timeout=5)
            except BaseException:
                if proc:
                    try:
                        FFmpegProcessRegistry.terminate(pipe_key)
                    except Exception:
                        pass
                raise
        finally:
            if watchdog_timer:
                watchdog_timer.cancel()
            try:
                if proc is not None:
                    FFmpegProcessRegistry.terminate(pipe_key)
                    FFmpegProcessRegistry.wait_for_exit(proc, timeout=5)
                if watchdog_timer is not None:
                    watchdog_timer.join(timeout=2)
                    if watchdog_timer.is_alive():
                        raise RuntimeError("Window decode watchdog termination unconfirmed")
            except BaseException as error:
                readers = [watchdog_timer] if watchdog_timer is not None else []
                FFmpegProcessRegistry.defer_cleanup(pipe_key, proc, io_sem, readers)
                logger.error("Window decode cleanup unconfirmed; retained process, original lease and stderr: %s", pipe_key)
                raise MediaTerminationUnconfirmed(f"Window decode cleanup unconfirmed: {pipe_key}") from error
            if proc is not None:
                FFmpegProcessRegistry.deregister(pipe_key, proc)
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except OSError:
                    pass
            err_log.unlink(missing_ok=True)
            io_sem.release()

        return trace, frames_decoded

    def _stitch_windowed_labels(
        self,
        windows: list[tuple[float, float]],
        window_traces: list[tuple[float, float, object]],
        audio_events: list[tuple[float, float, str]],
        start_offset: float,
        file_duration: float,
        effective_fps: float,
        baseline_threshold: float,
    ) -> list[dict]:
        """Stitch fine-decoded window labels and interpolated STATIC gap labels into a contiguous timeline.
        Guarantees strict timeline closure at start_offset + file_duration.
        """
        dt = 1.0 / effective_fps if effective_fps > 0 else 0.2
        all_labels: list[dict] = []

        def _make_static_label(t_rel: float) -> dict:
            t_abs = start_offset + min(round(t_rel, 3), file_duration)
            is_audio = False
            if audio_events:
                for ev in audio_events:
                    ev_s, ev_e = float(ev[0]), float(ev[1])
                    if (ev_s - 0.1) <= t_rel <= (ev_e + 0.1):
                        is_audio = True
                        break
            state = "DYNAMIC_AUDIO" if is_audio else "STATIC"
            return {
                "time": t_abs,
                "is_motion": is_audio,
                "state": state,
                "energy": 0.0,
                "raw_energy": 0.0,
                "confidence": 0.0,
                "is_audio_active": is_audio,
                "active_cells": 0,
                "max_cell_energy": 0.0,
            }

        current_t = 0.0
        for w_start, w_end, trace in window_traces:
            # 1. Fill preceding Gap with STATIC labels
            while current_t < w_start - 1e-4:
                all_labels.append(_make_static_label(current_t))
                current_t += dt

            # 2. Process fine frames inside window
            energies = list(trace.energies) if hasattr(trace, "energies") else []
            confidences = list(trace.confidences) if hasattr(trace, "confidences") else []
            active_cells = list(getattr(trace, "active_cells", ()))

            if len(energies) >= 2:
                median_win = max(1, int(round(self.median_window * effective_fps / self.fps)))
                if median_win % 2 == 0:
                    median_win += 1
                if median_win >= 3:
                    energies = _median_filter(energies, median_win)

                raw_labels = [
                    bool(
                        e > baseline_threshold
                        or (i < len(active_cells) and active_cells[i] >= 2 and e >= self.min_motion_threshold * 1.2)
                    )
                    for i, e in enumerate(energies)
                ]
                smoothed = _smooth_labels(
                    raw_labels,
                    max(1, round(self.min_motion_frames * effective_fps / self.fps)),
                    max(1, round(self.min_static_frames * effective_fps / self.fps)),
                    max(1, round(self.noise_suppress * effective_fps / self.fps)),
                )
            else:
                smoothed = [bool(e > baseline_threshold) for e in energies]

            for i, is_visual_motion in enumerate(smoothed):
                t_win_rel = min(round(w_start + i * dt, 3), w_end)
                t_abs = start_offset + t_win_rel

                is_audio = False
                if audio_events:
                    for ev in audio_events:
                        ev_s, ev_e = float(ev[0]), float(ev[1])
                        if (ev_s - 0.1) <= t_win_rel <= (ev_e + 0.1):
                            is_audio = True
                            break

                if is_visual_motion:
                    state = "DYNAMIC"
                    is_motion = True
                elif is_audio:
                    state = "DYNAMIC_AUDIO"
                    is_motion = True
                else:
                    state = "STATIC"
                    is_motion = False

                all_labels.append({
                    "time": t_abs,
                    "is_motion": is_motion,
                    "state": state,
                    "energy": float(energies[i]) if i < len(energies) else 0.0,
                    "raw_energy": float(trace.energies[i]) if hasattr(trace, "energies") and i < len(trace.energies) else 0.0,
                    "confidence": float(confidences[i]) if i < len(confidences) else 0.0,
                    "is_audio_active": is_audio,
                    "active_cells": int(trace.active_cells[i]) if hasattr(trace, "active_cells") and i < len(trace.active_cells) else 0,
                    "max_cell_energy": float(trace.max_cell_energies[i]) if hasattr(trace, "max_cell_energies") and i < len(trace.max_cell_energies) else 0.0,
                })

            current_t = w_end

        # 3. Fill trailing Gap up to file_duration
        while current_t < file_duration - 1e-4:
            all_labels.append(_make_static_label(current_t))
            current_t += dt

        # 4. Multimodal Audio VAD exact event preservation
        if audio_events and all_labels:
            for ev in audio_events:
                ev_s, ev_e = float(ev[0]), float(ev[1])
                start = start_offset + max(0.0, ev_s)
                end = start_offset + min(file_duration, ev_e) if file_duration > 0 else (start_offset + ev_e)
                if end <= start:
                    continue
                all_labels.extend((
                    {"time": start, "is_motion": True, "state": "DYNAMIC_AUDIO",
                     "energy": 0.0, "raw_energy": 0.0, "confidence": 0.0, "is_audio_active": True},
                    {"time": end, "is_motion": False, "state": "STATIC",
                     "energy": 0.0, "raw_energy": 0.0, "confidence": 0.0, "is_audio_active": False},
                ))

        # 5. Sort & Deduplicate
        all_labels.sort(key=lambda item: item["time"])
        deduped = []
        for label in all_labels:
            if deduped and abs(deduped[-1]["time"] - label["time"]) < 1e-4:
                if label.get("is_audio_active") or label.get("is_motion"):
                    deduped[-1] = label
            else:
                deduped.append(label)
        all_labels = deduped

        # 6. Strict timeline closure
        t_close = start_offset + file_duration
        if not all_labels:
            all_labels = [_make_static_label(0.0), _make_static_label(file_duration)]
        elif all_labels[-1]["time"] < t_close - 1e-6:
            all_labels.append(_make_static_label(file_duration))
        elif all_labels[-1]["time"] > t_close:
            all_labels[-1]["time"] = t_close

        return all_labels

    @contextmanager
    def _acquire_device_lease(self):
        """Acquire a fresh device lease context manager and safely manage self.decode_gpu lifecycle.
        
        Handles:
        1. Callable factory returning a contextmanager (e.g., pipeline.py lambda).
        2. Direct contextmanager instance (e.g., nullcontext).
        3. Absent device_lease attribute (defaults to nullcontext(self.decode_gpu)).
        4. Safe temporary assignment and restoration of self.decode_gpu.
        """
        lease_attr = getattr(self, "device_lease", None)
        if callable(lease_attr):
            cm = lease_attr()
        elif lease_attr is not None:
            cm = lease_attr
        else:
            cm = nullcontext(self.decode_gpu)

        orig_gpu = self.decode_gpu
        with cm as device:
            effective_device = device if device else orig_gpu
            self.decode_gpu = effective_device
            try:
                yield effective_device
            finally:
                self.decode_gpu = orig_gpu

    def _analyze_windowed(
        self,
        filepath: str,
        start_offset: float,
        file_duration: float,
        has_audio: int | None,
    ) -> tuple[list[dict], dict[int, np.ndarray]]:
        """Orchestrate Tier 1 Coarse Scan -> Zero-Decode / Tier 2 Window Fine Decode."""
        effective_fps = self._resolve_effective_fps(file_duration)
        t_start = time.monotonic()

        # 1. Concurrent Audio VAD Decode
        audio_thread = None
        audio_holder = []
        if getattr(self, "audio_vad_enabled", False) and getattr(self, "has_audio_detected", 1) != 0:
            def _async_audio():
                try:
                    audio_holder.append(self._decode_audio_pipe(filepath, file_duration))
                except Exception as exc:
                    audio_holder.append(exc)
            audio_thread = threading.Thread(target=_async_audio, daemon=True)
            audio_thread.start()

        # 2. Keyframe Demux PTS & Tier 1 Coarse Scan
        with self._acquire_device_lease() as device:
            records, coarse_meta = self._scan_keyframes_coarse(
                filepath, file_duration, keyframe_pts=None, decode_gpu=device
            )

        if not records or not coarse_meta.get("complete", True):
            raise RuntimeError(f"Coarse keyframe scan incomplete for {filepath}")

        # 3. Join Audio VAD
        full_audio = None
        if audio_thread is not None:
            audio_join_timeout = max(60.0, min(file_duration * 0.5, 300.0))
            audio_thread.join(timeout=audio_join_timeout)
            if audio_holder:
                res = audio_holder[0]
                if isinstance(res, MediaTerminationUnconfirmed):
                    raise res
                elif isinstance(res, Exception):
                    logger.warning("Audio decode failed for %s: %s; degrading to silent video", Path(filepath).name, res)
                    from src.algorithms.audio_features import AudioFeatures
                    full_audio = AudioFeatures(self.vad_sample_rate, self.vad_window_ms, self.buffer_limit // 4)
                else:
                    full_audio = res
            elif audio_thread.is_alive():
                logger.warning(
                    "Audio decode thread timed out after %.1fs for %s; degrading to silent video",
                    audio_join_timeout, Path(filepath).name,
                )
                from src.algorithms.audio_features import AudioFeatures
                full_audio = AudioFeatures(self.vad_sample_rate, self.vad_window_ms, self.buffer_limit // 4)
        if full_audio is None:
            full_audio = np.array([], dtype=np.float32)

        audio_events: list[tuple[float, float, str]] = []
        vad_stats: dict = {}
        if self.audio_vad_enabled and full_audio is not None and full_audio.size > 0:
            audio_events, vad_stats = self.vad.detect_events(full_audio, start_offset=0.0)

        # Baseline noise threshold from coarse scan
        energies_arr = np.array([r[1] for r in records], dtype=np.float32)
        if len(energies_arr) >= 2:
            p5 = float(np.percentile(energies_arr, 5))
            p20 = float(np.percentile(energies_arr, 20))
            noise_spread = min((p20 - p5) * 2.5, max(1.5, self.min_motion_threshold * 0.8))
            base_noise = min(p5, max(self.min_motion_threshold * 2.5, 6.0))
            baseline_threshold = max(self.min_motion_threshold, base_noise + noise_spread / max(0.001, self.sensitivity))
        else:
            baseline_threshold = self.min_motion_threshold

        avg_gop = coarse_meta.get("avg_gop_s", coarse_meta.get("avg_gop", 3.0))
        max_gop = coarse_meta.get("max_gop_s", coarse_meta.get("max_gop", avg_gop))

        # 4. Sparse GOP Safeguard: if GOP is sparse or irregular, coarse scan cannot guarantee zero recall loss.
        # Fall back unconditionally to full decode.
        gop_metrics = {
            "keyframes": len(records),
            "keyframe_count": len(records),
            "avg_gop_s": avg_gop,
            "avg_gop": avg_gop,
            "max_gop_s": max_gop,
            "max_gop": max_gop,
        }
        if len(records) < 2 or avg_gop > 8.0 or (max_gop > 15.0 and len(records) >= 3):
            fallback_reason = "FALLBACK_SPARSE_GOP"
            logger.info("Sparse GOP detected (avg=%.1fs, count=%d); falling back to full decode", avg_gop, len(records))
            self._predecoded_audio = full_audio
            try:
                return self._fallback_full_decode(
                    filepath, start_offset, file_duration, has_audio, fallback_reason=fallback_reason
                )
            finally:
                self._predecoded_audio = None

        # 5. Resolve Awakened Windows
        windows = self._resolve_awakened_windows(
            keyframe_trace=records,
            audio_events=audio_events,
            file_duration=file_duration,
            threshold=baseline_threshold,
            gop_duration=avg_gop,
        )

        # 6. Check Zero-Decode Fast Path
        if not windows:
            labels = self._synthesize_static_labels(start_offset, file_duration, effective_fps)
            yolo_buffer: dict[int, np.ndarray] = {}
            coarse_scan_time = coarse_meta.get("scan_time", round(time.monotonic() - t_start, 3))
            self.decode_completion = {
                "reason": "ZERO_DECODE_FAST_PATH",
                "complete": True,
                "declared_duration_s": file_duration,
                "available_duration_s": file_duration,
                "sample_period_s": 1.0 / max(0.001, effective_fps),
                "error": "",
            }
            self.last_perf = {
                "decode_time": coarse_scan_time,
                "analysis_time": round(time.monotonic() - t_start, 3),
                "frames": len(labels),
                "motion_ratio": 0.0,
                "early_term": False,
                "effective_fps": effective_fps,
                "has_audio": int(bool(full_audio.size) or bool(getattr(self, "has_audio_detected", 0))),
                "audio_events": len(audio_events),
                "vad_noise_floor_db": vad_stats.get("noise_floor_db", -140.0),
                "sem_wait": 0.0,
                "completion_reason": "ZERO_DECODE_FAST_PATH",
            }
            return labels, yolo_buffer

        # 7. Evaluate Window-based Fallback Safeguards
        should_fallback, fallback_reason = self._evaluate_fallback_safeguards(
            windows, file_duration, gop_metrics, filepath
        )
        if should_fallback:
            logger.info(
                "I-frame prescreen fallback triggered for %s (%s); switching to full decode",
                Path(filepath).name, fallback_reason,
            )
            completion_fallback_reason = "FALLBACK_DENSE_MOTION" if "ACTIVE_RATIO" in fallback_reason else fallback_reason
            self._predecoded_audio = full_audio
            try:
                return self._fallback_full_decode(
                    filepath, start_offset, file_duration, has_audio, fallback_reason=completion_fallback_reason
                )
            finally:
                self._predecoded_audio = None

        # 8. Tier 2 Awakened Window Fine Decoding
        if "_decode_file_window" not in self.__dict__ and len(windows) > 1:
            coalesce_gap = 60.0 if file_duration >= 1800.0 else (45.0 if file_duration > 1200.0 else 12.0)
            coalesced: list[tuple[float, float]] = [windows[0]]
            for w_s, w_e in windows[1:]:
                prev_s, prev_e = coalesced[-1]
                if w_s - prev_e <= coalesce_gap:
                    coalesced[-1] = (prev_s, max(prev_e, w_e))
                else:
                    coalesced.append((w_s, w_e))
            windows = coalesced
        from src.algorithms.motion_trace import MotionTrace
        yolo_buffer = {}
        est_window_frames = max(
            1.0,
            sum(max(0.0, w_e - w_s) for w_s, w_e in windows) * effective_fps,
        )
        yolo_sample_interval = max(
            1, int(round(effective_fps / max(0.1, getattr(self, "yolo_sample_fps", 0.5)))),
            int(np.ceil(est_window_frames /
                       max(1, getattr(self, "yolo_max_frames", 512))))
        )

        window_traces: list[tuple[float, float, object]] = []
        total_window_frames = 0
        with self._acquire_device_lease() as device:
            for w_start, w_end in windows:
                res = self._decode_file_window(
                    filepath, w_start, w_end,
                    effective_fps=effective_fps,
                    yolo_buffer=yolo_buffer,
                    yolo_sample_interval=yolo_sample_interval,
                    file_duration=file_duration,
                )
                if isinstance(res, tuple):
                    first = res[0]
                    if isinstance(first, MotionTrace):
                        win_trace = first
                        frames_count = getattr(first, "_frame_count", len(first.energies) if hasattr(first, "energies") else 0)
                    elif isinstance(first, (list, tuple)):
                        win_trace = MotionTrace(self, effective_fps, is_night_mode=self._is_night_time(getattr(self, "_analysis_start_offset", 0.0) + w_start))
                        for f in first:
                            win_trace.append(f)
                        frames_count = len(first)
                    else:
                        win_trace = first
                        frames_count = 0
                    if len(res) >= 2 and isinstance(res[1], dict):
                        yolo_buffer.update(res[1])
                    if len(res) >= 3 and isinstance(res[2], dict):
                        frames_count = res[2].get("frames", frames_count)
                    elif len(res) == 2 and isinstance(res[1], int):
                        frames_count = res[1]
                else:
                    win_trace = res
                    frames_count = 0
                total_window_frames += frames_count
                window_traces.append((w_start, w_end, win_trace))

        # 8. Contiguous Timeline Stitching
        labels = self._stitch_windowed_labels(
            windows=windows,
            window_traces=window_traces,
            audio_events=audio_events,
            start_offset=start_offset,
            file_duration=file_duration,
            effective_fps=effective_fps,
            baseline_threshold=baseline_threshold,
        )

        t_end = time.monotonic()
        motion_count = sum(1 for lbl in labels if lbl.get("is_motion"))
        self.decode_completion = {
            "reason": "NORMAL_COMPLETION",
            "complete": True,
            "declared_duration_s": file_duration,
            "available_duration_s": file_duration,
            "sample_period_s": 1.0 / max(0.001, effective_fps),
            "error": "",
        }
        self.last_perf = {
            "decode_time": round(t_end - t_start, 3),
            "analysis_time": round(t_end - t_start, 3),
            "frames": total_window_frames,
            "motion_ratio": round(motion_count / len(labels), 3) if labels else 0.0,
            "early_term": False,
            "effective_fps": effective_fps,
            "has_audio": int(bool(full_audio.size) or bool(getattr(self, "has_audio_detected", 0))),
            "audio_events": len(audio_events),
            "vad_noise_floor_db": vad_stats.get("noise_floor_db", -140.0),
            "sem_wait": 0.0,
            "completion_reason": "TIER2_WINDOWED_COMPLETION",
        }
        return labels, yolo_buffer

    def _fallback_full_decode(
        self,
        filepath: str,
        start_offset: float,
        file_duration: float,
        has_audio: int | None,
        fallback_reason: str = "NORMAL_COMPLETION",
    ) -> tuple[list[dict], dict[int, np.ndarray]]:
        """Fallback to standard sequential full decode."""
        with self._acquire_device_lease() as device:
            self._stream_motion = True
            decoded_frames, yolo_buffer, full_audio, decode_meta = self._decode_file(filepath, file_duration)
        has_audio = decode_meta.get("has_audio", has_audio)
        effective_fps = decode_meta.get("effective_fps", self.fps)
        t_decode_end = time.monotonic()
        completion_reason = decode_meta.get("completion_reason")
        if not completion_reason or completion_reason == "UNKNOWN":
            completion_reason = fallback_reason
        self.decode_completion = {
            "reason": completion_reason,
            "complete": bool(decode_meta.get("complete", True)),
            "declared_duration_s": file_duration,
            "available_duration_s": min(file_duration, len(decoded_frames) / max(.001, effective_fps)),
            "sample_period_s": 1 / max(.001, effective_fps),
            "error": decode_meta.get("completion_error", ""),
        }

        if decode_meta.get("aborted") or not decode_meta.get("complete", True):
            decoded_frames = []
        if not decoded_frames:
            self.last_perf = {
                "decode_time": decode_meta.get("decode_time", 0.0),
                "analysis_time": 0,
                "frames": decode_meta.get("frames", 0),
                "motion_ratio": 0,
                "early_term": False,
                "has_audio": has_audio,
                "audio_events": 0,
                "sem_wait": decode_meta.get("sem_wait", 0.0),
                "completion_reason": self.decode_completion["reason"],
            }
            return [], yolo_buffer

        actual_dur = round(len(decoded_frames) / max(0.001, effective_fps), 3)
        if actual_dur + 2 / max(.001, effective_fps) < file_duration or decode_meta.get("truncated"):
            file_duration = actual_dur
            self.file_duration_detected = actual_dur

        results, frames_meta = self.analyze_frames(
            decoded_frames, start_offset=start_offset, file_duration=file_duration,
            fps=effective_fps, audio_data=full_audio, audio_sample_rate=self.vad_sample_rate,
        )
        early_term_any = False
        audio_events = frames_meta.get("audio_events", [])
        vad_stats = frames_meta.get("vad_stats", {})

        t_analysis_end = time.monotonic()

        motion_count = sum(1 for r in results if r.get("is_motion"))
        self.last_perf = {
            "decode_time": decode_meta.get("decode_time", 0.0),
            "analysis_time": round(t_analysis_end - t_decode_end, 3),
            "frames": decode_meta.get("frames", len(decoded_frames)),
            "motion_ratio": round(motion_count / len(results), 3) if results else 0,
            "early_term": early_term_any,
            "effective_fps": effective_fps,
            "has_audio": has_audio,
            "audio_events": len(audio_events),
            "vad_noise_floor_db": vad_stats.get("noise_floor_db", -140.0),
            "sem_wait": decode_meta.get("sem_wait", 0.0),
            "completion_reason": self.decode_completion["reason"],
        }
        return results, yolo_buffer

    def analyze(
        self, filepath: str, start_offset: float = 0.0, file_duration: float = 0.0,
        has_audio: int | None = None, duration_verified: bool = False,
        declared_duration: float | None = None,
    ) -> tuple[list[dict], dict[int, np.ndarray]]:
        """
        全文件分析入口。
        Phase 1: 硬件解码到内存 (信号量保护，单槽位)
        Phase 2: 连续 CPU 运动分析（由文件 worker 数限制并发）

        返回 (labels, yolo_buffer)：
        - labels: [{'time', 'is_motion', 'state', 'energy', 'is_audio_active'}, ...]
        - yolo_buffer: {frame_index: np.ndarray} 供 YOLO 流式验证复用的零拷贝帧池
        """
        self._analysis_start_offset = start_offset
        if declared_duration is not None and duration_verified:
            file_duration = max(file_duration, declared_duration)
        # Filename spans are day-relative placement metadata, not media duration.
        # Analysis is the lazy metadata boundary; probe only if metadata was not already resolved.
        if file_duration <= 0 or has_audio is None or not duration_verified:
            from src.hardware.scheduler import get_disk_semaphore
            disk = get_disk_semaphore()
            if not acquire_with_retry(disk):
                raise TimeoutError("Metadata I/O admission timed out")
            try:
                try:
                    with av.open(str(filepath), timeout=30.0) as container:
                        stream = container.streams.video[0]
                        actual_duration = (float(stream.duration * stream.time_base) if stream.duration
                                           else float(container.duration or 0) / av.time_base)
                        if actual_duration <= 0:
                            raise ValueError("Missing video duration")
                        file_duration = actual_duration
                        self.file_duration_detected = actual_duration
                        self.has_audio_detected = int(bool(container.streams.audio))
                except (FileNotFoundError, OSError, Exception) as e:
                    if file_duration > 0 and duration_verified:
                        self.file_duration_detected = file_duration
                        self.has_audio_detected = 1
                    else:
                        raise
            finally:
                disk.release()
        else:
            self.file_duration_detected = file_duration
            self.has_audio_detected = has_audio

        if self.iframe_prescreen_enabled and file_duration >= getattr(self, "iframe_min_file_duration", 0.0):
            try:
                return self._analyze_windowed(
                    filepath=filepath,
                    start_offset=start_offset,
                    file_duration=file_duration,
                    has_audio=has_audio,
                )
            except (MemoryError, MediaTerminationUnconfirmed):
                raise
            except Exception as e:
                logger.warning(
                    "I-frame windowed analysis failed for %s: %s; falling back to full decode",
                    Path(filepath).name, e,
                )

        return self._fallback_full_decode(
            filepath=filepath,
            start_offset=start_offset,
            file_duration=file_duration,
            has_audio=has_audio,
        )

