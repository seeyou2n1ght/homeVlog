import logging
import os
import subprocess
import threading
import time
from pathlib import Path

import av
import cv2
import numpy as np

from src.utils import parse_res
from src.scheduler import acquire_with_retry, VideoLease
from src.ffmpeg import run_ffmpeg, FFmpegProcessRegistry

# 时空滤波 / 背景建模 / 音频 VAD 算法统一由 src.filters 提供（单一实现，防漂移）。
# 此处 re-export 以保持 `from src.detector import ...` 的历史导入路径兼容。
from src.filters import (
    AudioEnergyVAD,
    EmaBackgroundModel,
    SpatialGridMotionFilter,
    _median_filter,
    _smooth_labels,
)

__all__ = [
    "AudioEnergyVAD",
    "EmaBackgroundModel",
    "SpatialGridMotionFilter",
    "MotionDetector",
    "_median_filter",
    "_smooth_labels",
]

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
        self.sensitivity = det.get("motion_sensitivity", 4.0)
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
            ambient_drift_suppress=self.ambient_drift_suppress,
            ambient_drift_active_ratio=self.ambient_drift_active_ratio,
            ambient_drift_max_energy=self.ambient_drift_max_energy,
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

        from src.motion_trace import MotionTrace
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

        trace = frames if isinstance(frames, MotionTrace) else MotionTrace(self, effective_fps)
        if trace is not frames:
            for frame in frames:
                trace.append(frame)
        energies = list(trace.energies)
        confidences = list(trace.confidences)

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
        noise_spread = (p20 - p5) * 2.5
        threshold = max(self.min_motion_threshold, p5 + self.sensitivity * noise_spread)

        raw_labels = [bool(e > threshold) for e in energies]

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
                "confidence": float(confidences[i]) if i < len(confidences) else 0.0,
                "is_audio_active": is_audio_active,
            })

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
        try:
            frames, yolo_buffer, meta = self._decode_file_pipe(
                filepath, file_duration, effective_fps
            )
            if meta.get("aborted"):
                return [], {}, np.array([], dtype=np.float32), meta
            if frames and meta.get("complete", False):
                try:
                    full_audio = self._decode_audio_pipe(filepath, file_duration)
                except Exception as exc:
                    logger.warning("Audio decode failed for %s: %s", filepath, exc)
                    meta["complete"] = False
                    return [], {}, np.array([], np.float32), meta
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
        except MemoryError:
            # A second full decode has the same storage bound and cannot recover.
            raise
        except Exception as e:
            logger.warning(
                "pipe decode failed for %s: %s; falling back to PyAV",
                Path(filepath).name, e,
            )
        return self._decode_file_pyav(filepath, file_duration)

    def _decode_file_pipe(
        self, filepath: str, file_duration: float, effective_fps: float
    ) -> tuple[list, dict, dict]:
        """ffmpeg 子进程管道解码：fps 抽帧 + GPU 缩放后仅下载采样帧。

        对比 PyAV 逐帧全量解码（采样 660 帧需解码 9900 帧并全量 4K 回下载），
        本路径解码侧在 GPU 完成 fps 过滤与缩放，仅 ~660 个 416x234 小帧经管道回传，
        实测可将单文件解码耗时从 ~240s 降至 ~15-30s。
        """
        from src.frame_pool import FramePool
        from src.motion_trace import MotionTrace
        decoded_frames = (MotionTrace(self, effective_fps) if getattr(self, "_stream_motion", False)
                          else FramePool(self.buffer_limit // 2))
        yolo_buffer: dict[int, np.ndarray] = {}
        jpeg_bytes = 0
        w, h = self.width, self.height
        frame_size = w * h  # 单通道灰度直通，IPC 管道数据量降低 66.7%

        if self.decode_gpu == "qsv":
            from src.utils import get_qsv_semaphore
            io_sem = VideoLease(get_qsv_semaphore())
            hw_args = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
            vf = f"fps={effective_fps:.3f},scale_qsv=w={w}:h={h},hwdownload,format=nv12"
        else:
            from src.utils import get_nvdec_semaphore
            io_sem = VideoLease(get_nvdec_semaphore())
            hw_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            vf = f"fps={effective_fps:.3f},scale_cuda={w}:{h},hwdownload,format=nv12"

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
            *hw_args, "-i", str(filepath),
            "-vf", vf, "-f", "rawvideo", "-pix_fmt", "gray", "-",
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
        from src.utils import TEMP_DIR
        pipe_key = f"pipe_decode_{Path(filepath).name}_{time.monotonic()}"
        safe_stem = Path(filepath).stem.replace(" ", "_")
        err_log = TEMP_DIR / f".pipe_decode_{safe_stem}_{os.getpid()}_{threading.get_ident()}.log"
        try:
            with open(err_log, "wb") as f_err:
                proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=f_err)
                FFmpegProcessRegistry.register(pipe_key, proc)
                assert proc.stdout is not None

                def _kill_on_timeout():
                    try:
                        if proc and proc.poll() is None:
                            logger.warning(
                                "pipe decode watchdog timeout (%.1fs) for %s, terminating process",
                                watchdog_timeout, Path(filepath).name,
                            )
                            proc.kill()
                    except Exception:
                        pass

                watchdog_timer = threading.Timer(watchdog_timeout, _kill_on_timeout)
                watchdog_timer.daemon = True
                watchdog_timer.start()

                while True:
                    raw = proc.stdout.read(frame_size)
                    if len(raw) < frame_size:
                        break
                    gray = np.frombuffer(raw, dtype=np.uint8).reshape((h, w))

                    if getattr(self, "yolo_enabled", False) and total_frames % yolo_sample_interval == 0:
                        bgr_tmp = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                        ok_enc, buf_jpg = cv2.imencode(".jpg", bgr_tmp, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        item = buf_jpg if ok_enc else bgr_tmp
                        jpeg_bytes += item.nbytes
                        if jpeg_bytes > self.buffer_limit // 4:
                            raise MemoryError("YOLO candidate budget exceeded")
                        yolo_buffer[total_frames] = item

                    decoded_frames.append(gray)
                    total_frames += 1
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=5)
            except BaseException:
                if proc:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                raise
        finally:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
            if watchdog_timer:
                watchdog_timer.cancel()
            FFmpegProcessRegistry.deregister(pipe_key)
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

            min_expected_frames = max(1, int(file_duration * effective_fps * 0.85))
            is_complete = bool(
                not is_aborted and total_frames > 0 and (
                    (proc and proc.returncode == 0 and total_frames >= min_expected_frames) or
                    total_frames >= max(1, file_duration * effective_fps - 2)
                )
            )

        meta = {
            "has_audio": 0,
            "effective_fps": effective_fps,
            "decode_time": round(time.monotonic() - t_decode_start, 3),
            "frames": total_frames,
            "sem_wait": sem_wait,
            "aborted": is_aborted,
            "complete": is_complete,
        }
        return decoded_frames, yolo_buffer, meta

    def _decode_audio_pipe(self, filepath: str, file_duration: float):
        from src.audio_features import AudioFeatures
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
        from src.frame_pool import FramePool
        decoded_frames = FramePool(self.buffer_limit // 2)
        yolo_buffer: dict[int, np.ndarray] = {}
        jpeg_bytes = 0

        if self.decode_gpu == "qsv":
            from src.utils import get_qsv_semaphore
            io_sem = VideoLease(get_qsv_semaphore())
            hw_name = "qsv"
        else:
            from src.utils import get_nvdec_semaphore
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

        from src.audio_features import AudioFeatures
        audio_features = AudioFeatures(self.vad_sample_rate, self.vad_window_ms, self.buffer_limit // 4)
        audio_resampler = None
        has_audio = 0
        effective_fps = self.fps
        total_frames = 0
        video_frame_count = 0
        decode_complete = False

        try:
            from av.audio.resampler import AudioResampler
            audio_resampler = AudioResampler(format="fltp", layout="mono", rate=self.vad_sample_rate)
        except Exception as e:
            logger.debug(f"PyAV AudioResampler init: {e}")

        t_decode_start = time.monotonic()
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
                    from src.motion_trace import MotionTrace
                    decoded_frames = MotionTrace(self, effective_fps)

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

                from src.renderer import FFmpegProcessRegistry
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
                            rgb_raw = frame.reformat(
                                width=self.width, height=self.height, format="rgb24"
                            ).to_ndarray()
                        except Exception:
                            rgb_raw = cv2.resize(
                                frame.to_ndarray(format="rgb24"),
                                (self.width, self.height),
                                interpolation=cv2.INTER_LINEAR,
                            )
                        bgr_tmp = cv2.cvtColor(rgb_raw, cv2.COLOR_RGB2BGR)
                        ok_enc, buf_jpg = cv2.imencode(".jpg", bgr_tmp, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        item = buf_jpg if ok_enc else bgr_tmp
                        jpeg_bytes += item.nbytes
                        if jpeg_bytes > self.buffer_limit // 4:
                            raise MemoryError("YOLO candidate budget exceeded")
                        yolo_buffer[total_frames - 1] = item

                    try:
                        if frame.planes:
                            y_raw = frame.to_ndarray(format="gray")
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

                    decoded_frames.append(gray)

                    if time.monotonic() - t_decode_start > watchdog_timeout:
                        logger.warning(
                            "PyAV decode timeout for %s (dur=%.1f, elapsed=%.1f)",
                            Path(filepath).name, file_duration,
                            time.monotonic() - t_decode_start,
                        )
                        break

                else:
                    decode_complete = (
                        not FFmpegProcessRegistry.is_interrupted() and
                        total_frames >= max(1, int(file_duration * effective_fps * 0.85))
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
            logger.warning("PyAV decode error for %s: %s", Path(filepath).name, e)
        finally:
            io_sem.release()

        t_decode_end = time.monotonic()

        full_audio = audio_features

        meta = {
            "has_audio": has_audio,
            "effective_fps": effective_fps,
            "complete": decode_complete,
            "decode_time": round(t_decode_end - t_decode_start, 3),
            "frames": total_frames,
            "sem_wait": sem_wait,
        }
        return decoded_frames, yolo_buffer, full_audio, meta

    def analyze(
        self, filepath: str, start_offset: float = 0.0, file_duration: float = 0.0, has_audio: int | None = None
    ) -> tuple[list[dict], dict[int, np.ndarray]]:
        """
        全文件分析入口。
        Phase 1: 硬件解码到内存 (信号量保护，单槽位)
        Phase 2: 连续 CPU 运动分析（由文件 worker 数限制并发）

        返回 (labels, yolo_buffer)：
        - labels: [{'time', 'is_motion', 'state', 'energy', 'is_audio_active'}, ...]
        - yolo_buffer: {frame_index: np.ndarray} 供 YOLO 流式验证复用的零拷贝帧池
        """
        if file_duration > 0 and has_audio is not None:
            self.file_duration_detected = file_duration
            self.has_audio_detected = int(has_audio)
        else:
            # Lazy container metadata belongs to Analysis, never the directory scanner.
            from src.scheduler import get_disk_semaphore
            disk = get_disk_semaphore()
            if not acquire_with_retry(disk):
                raise TimeoutError("Metadata I/O admission timed out")
            try:
                with av.open(str(filepath), timeout=30.0) as container:
                    if file_duration <= 0:
                        stream = container.streams.video[0]
                        duration = (float(stream.duration * stream.time_base) if stream.duration
                                    else float(container.duration or 0) / av.time_base)
                        if duration <= 0:
                            raise ValueError("Missing video duration")
                        file_duration = duration
                    self.file_duration_detected = file_duration
                    self.has_audio_detected = int(bool(container.streams.audio))
            finally:
                disk.release()
        # Phase 1: 解码
        from contextlib import nullcontext
        lease = self.device_lease() if hasattr(self, "device_lease") else nullcontext(self.decode_gpu)
        with lease as device:
            self.decode_gpu = device
            self._stream_motion = True
            decoded_frames, yolo_buffer, full_audio, decode_meta = self._decode_file(filepath, file_duration)
        has_audio = decode_meta["has_audio"]
        effective_fps = decode_meta["effective_fps"]
        t_decode_end = time.monotonic()

        if decode_meta.get("aborted") or not decode_meta.get("complete", True):
            decoded_frames = []
        if not decoded_frames:
            self.last_perf = {
                "decode_time": decode_meta["decode_time"],
                "analysis_time": 0,
                "frames": decode_meta["frames"],
                "motion_ratio": 0,
                "early_term": False,
                "has_audio": has_audio,
                "audio_events": 0,
                "sem_wait": decode_meta.get("sem_wait", 0.0),
            }
            return [], yolo_buffer

        # One continuous background model per file; the outer worker pool bounds CPU concurrency.
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
            "decode_time": decode_meta["decode_time"],
            "analysis_time": round(t_analysis_end - t_decode_end, 3),
            "frames": decode_meta["frames"],
            "motion_ratio": round(motion_count / len(results), 3) if results else 0,
            "early_term": early_term_any,
            "effective_fps": effective_fps,
            "has_audio": has_audio,
            "audio_events": len(audio_events),
            "vad_noise_floor_db": vad_stats.get("noise_floor_db", -140.0),
            "sem_wait": decode_meta.get("sem_wait", 0.0),
        }
        return results, yolo_buffer

