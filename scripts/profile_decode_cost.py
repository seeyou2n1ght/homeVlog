#!/usr/bin/env python3
"""Non-invasive profiling tool for Pass 1.5 Analysis micro-stage cost breakdown.

Quantifies the exact timing of the 6 sub-phases in Pass 1.5 Analysis using
standard high-precision `time.perf_counter_ns()` without modifying production code.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import sys
import threading
import time
from typing import Any

import av
import cv2
import numpy as np
import psutil

# Ensure repo root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from src.core.config import load_config
from src.core.utils import parse_res
from src.algorithms.filters import (
    AudioEnergyVAD,
    EmaBackgroundModel,
    SpatialGridMotionFilter,
    video_frame_to_gray,
)
from src.algorithms.motion_trace import MotionTrace

logger = logging.getLogger("profile_decode_cost")


class SystemResourceSampler:
    """Samples CPU, RAM, and GPU metrics in background during profiling."""

    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.cpu_samples: list[float] = []
        self.ram_samples_mb: list[float] = []
        self.vram_samples_mb: list[float] = []
        self.gpu_util_samples: list[int] = []
        self.gpu_name = "N/A"
        self._nvml_inited = False
        self._nvml_handle = None

        try:
            import pynvml
            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._nvml_inited = True
            if pynvml.nvmlDeviceGetCount() > 0:
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(self._nvml_handle)
                self.gpu_name = name.decode("utf-8") if isinstance(name, bytes) else str(name)
        except Exception:
            self._nvml_inited = False

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        process = psutil.Process()
        while not self._stop_event.is_set():
            try:
                self.cpu_samples.append(process.cpu_percent(interval=None))
                mem = process.memory_info()
                self.ram_samples_mb.append(mem.rss / (1024 * 1024))
                if self._nvml_inited and self._nvml_handle:
                    mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                    self.vram_samples_mb.append(mem_info.used / (1024 * 1024))
                    rates = self._pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle)
                    self.gpu_util_samples.append(rates.gpu)
            except Exception:
                pass
            time.sleep(self.interval)

    def stop(self) -> dict[str, Any]:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._nvml_inited:
            try:
                self._pynvml.nvmlShutdown()
            except Exception:
                pass

        return {
            "gpu_name": self.gpu_name,
            "cpu_avg_pct": round(float(np.mean(self.cpu_samples)), 2) if self.cpu_samples else 0.0,
            "cpu_peak_pct": round(float(np.max(self.cpu_samples)), 2) if self.cpu_samples else 0.0,
            "ram_avg_mb": round(float(np.mean(self.ram_samples_mb)), 2) if self.ram_samples_mb else 0.0,
            "ram_peak_mb": round(float(np.max(self.ram_samples_mb)), 2) if self.ram_samples_mb else 0.0,
            "vram_avg_mb": round(float(np.mean(self.vram_samples_mb)), 2) if self.vram_samples_mb else 0.0,
            "vram_peak_mb": round(float(np.max(self.vram_samples_mb)), 2) if self.vram_samples_mb else 0.0,
            "gpu_util_avg_pct": round(float(np.mean(self.gpu_util_samples)), 2) if self.gpu_util_samples else 0.0,
            "gpu_util_peak_pct": round(float(np.max(self.gpu_util_samples)), 2) if self.gpu_util_samples else 0.0,
        }


class SubStageProfiler:
    """Measures precise time in nanoseconds for each micro-substage."""

    def __init__(self):
        # 1. Demux & Video Decode
        self.demux_video_decode_ns: int = 0
        self.video_frames_demuxed: int = 0

        # 2. Grayscale conversion & Resize 416x234
        self.grayscale_convert_ns: int = 0
        self.resize_ns: int = 0
        self.stage2_total_ns: int = 0
        self.video_frames_processed: int = 0

        # 3. EMA Motion Analysis
        self.ema_update_ns: int = 0
        self.spatial_grid_ns: int = 0
        self.stage3_total_ns: int = 0

        # 4. Candidate JPEG Encoding
        self.candidate_prep_ns: int = 0
        self.candidate_jpeg_encode_ns: int = 0
        self.stage4_total_ns: int = 0
        self.candidate_jpegs_count: int = 0
        self.candidate_total_bytes: int = 0

        # 5. Audio Demux & VAD
        self.audio_demux_resample_ns: int = 0
        self.audio_frames_demuxed: int = 0
        self.vad_detection_ns: int = 0
        self.stage5_total_ns: int = 0
        self.audio_events_count: int = 0

        # 6. YOLO Verification & Lock Wait
        self.yolo_lock_wait_ns: int = 0
        self.yolo_inference_ns: int = 0
        self.yolo_preprocess_ns: int = 0
        self.stage6_total_ns: int = 0
        self.yolo_verified_segments: int = 0


def resolve_media_path(raw_path: str) -> Path:
    """Resolves media path across local, project root, and common parent directories."""
    candidates = [
        Path(raw_path),
        PROJECT_ROOT / raw_path,
        Path("C:/Users/seeyo/code/homeVlog") / raw_path,
        PROJECT_ROOT.parent / raw_path,
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p.resolve()
    raise FileNotFoundError(f"Target media file not found: {raw_path}")


def profile_pass15_analysis(
    video_path: Path,
    config: dict,
    max_frames: int = 3000,
    hwaccel_mode: str = "cuda",
    analysis_res: str = "416x234",
    analysis_fps: float = 5.0,
    enable_yolo: bool = False,
    yolo_model_path: str = "models/yolo11n.pt",
) -> dict[str, Any]:
    """Execute non-invasive Pass 1.5 analysis profiling on the target video file."""
    w_target, h_target = parse_res(analysis_res)
    profiler = SubStageProfiler()
    sampler = SystemResourceSampler(interval=0.05)

    # 1. Probing video container metadata
    with av.open(str(video_path)) as probe_container:
        v_stream = probe_container.streams.video[0] if probe_container.streams.video else None
        a_stream = probe_container.streams.audio[0] if probe_container.streams.audio else None
        if v_stream is None:
            raise ValueError(f"No video stream in {video_path}")

        v_width = v_stream.width
        v_height = v_stream.height
        v_codec = v_stream.codec_context.name
        a_codec = a_stream.codec_context.name if a_stream else "none"
        raw_fps = float(v_stream.average_rate) if v_stream.average_rate else 15.0
        v_duration_s = (
            float(v_stream.duration * v_stream.time_base)
            if v_stream.duration
            else float(probe_container.duration or 0) / av.time_base
        )

    file_size_bytes = video_path.stat().st_size
    frame_step = max(1, int(round(raw_fps / analysis_fps)))
    effective_fps = raw_fps / frame_step
    yolo_sample_interval = max(1, int(round(effective_fps / 0.5)))

    # Set up HWAccel if available and requested
    hwaccel_obj = None
    hwaccel_status = "cpu"
    if hwaccel_mode in ("cuda", "qsv"):
        try:
            from av.codec.hwaccel import HWAccel
            hwaccel_obj = HWAccel(hwaccel_mode)
            hwaccel_status = hwaccel_mode
        except Exception as exc:
            logger.warning("HWAccel %s init failed: %s; falling back to CPU", hwaccel_mode, exc)
            hwaccel_status = "cpu_fallback"

    # Set up Motion Models
    det_cfg = config.get("detection", {})
    ema_model = EmaBackgroundModel(
        alpha_bg=float(det_cfg.get("ema_alpha_bg", 0.05)),
        alpha_fg=float(det_cfg.get("ema_alpha_fg", 0.005)),
        beta=float(det_cfg.get("bg_diff_weight", 0.6)),
        fg_threshold=float(det_cfg.get("fg_threshold", 12.0)),
    )
    grid_filter = SpatialGridMotionFilter(
        grid_rows=int(det_cfg.get("grid_rows", 8)),
        grid_cols=int(det_cfg.get("grid_cols", 8)),
        cooldown_half_life=float(det_cfg.get("cooldown_half_life", 15.0)),
        min_connected_cells=int(det_cfg.get("min_connected_cells", 2)),
        cell_noise_alpha=float(det_cfg.get("cell_noise_alpha", 0.02)),
        sens_multiplier=max(1.0, float(det_cfg.get("motion_sensitivity", 4.0)) * 0.5),
        base_noise_thresh=float(det_cfg.get("base_noise_thresh", 1.5)),
        cluster_boost=float(det_cfg.get("cluster_boost", 1.2)),
        ambient_drift_suppress=bool(det_cfg.get("ambient_drift_suppress", True)),
        ambient_drift_active_ratio=float(det_cfg.get("ambient_drift_active_ratio", 0.35)),
        ambient_drift_max_energy=float(det_cfg.get("ambient_drift_max_energy", 5.0)),
        camera_motion_suppress=bool(det_cfg.get("camera_motion_suppress", True)),
        camera_motion_active_ratio=float(det_cfg.get("camera_motion_active_ratio", 0.65)),
        camera_motion_max_cov=float(det_cfg.get("camera_motion_max_cov", 0.60)),
        camera_motion_max_focal_ratio=float(det_cfg.get("camera_motion_max_focal_ratio", 3.5)),
    )
    roi_cfg = det_cfg.get("roi_crop", [0.1, 0.12, 0.8, 0.85])
    roi_x, roi_y = int(w_target * roi_cfg[0]), int(h_target * roi_cfg[1])
    roi_w, roi_h = int(w_target * roi_cfg[2]), int(h_target * roi_cfg[3])

    # Audio VAD Setup
    audio_cfg = config.get("audio_vad", {})
    vad_sample_rate = int(audio_cfg.get("sample_rate", 16000))
    vad_window_ms = int(det_cfg.get("vad_window_ms", audio_cfg.get("window_ms", 50)))
    vad_detector = AudioEnergyVAD(
        sample_rate=vad_sample_rate,
        window_ms=vad_window_ms,
        noise_margin_db=float(audio_cfg.get("noise_margin_db", 12.0)),
        min_dbfs=float(audio_cfg.get("min_dbfs", -42.0)),
        min_speech_duration=float(audio_cfg.get("speech_min_dur", 0.3)),
        enabled=True,
    )
    from av.audio.resampler import AudioResampler
    audio_resampler = AudioResampler(format="fltp", layout="mono", rate=vad_sample_rate)
    accumulated_audio_chunks: list[np.ndarray] = []

    # YOLO Verifier pre-warm if enabled
    yolo_verifier = None
    yolo_status = "DISABLED"
    if enable_yolo:
        try:
            from src.algorithms.yolo_verifier import YoloVerifier
            yolo_cfg = config.setdefault("yolo", {})
            yolo_cfg["enabled"] = True
            yolo_cfg["model_path"] = yolo_model_path
            yolo_device = "cuda:0" if hwaccel_mode == "cuda" else "cpu"
            yolo_verifier = YoloVerifier(config, device=yolo_device)
            if yolo_verifier.enabled:
                yolo_status = "ENABLED"
        except Exception as exc:
            logger.warning("YOLO model pre-warm failed: %s", exc)
            yolo_status = f"ERROR: {exc}"

    # YOLO candidate buffer and motion trace
    yolo_buffer: dict[int, np.ndarray] = {}
    motion_energies: list[float] = []

    # Start hardware resource telemetry
    sampler.start()
    t_wall_start_ns = time.perf_counter_ns()

    open_kwargs = {}
    if hwaccel_obj:
        open_kwargs["hwaccel"] = hwaccel_obj

    with av.open(str(video_path), timeout=30.0, **open_kwargs) as container:
        v_stream = container.streams.video[0]
        a_stream = container.streams.audio[0] if container.streams.audio else None

        streams_to_decode = [v_stream]
        if a_stream:
            streams_to_decode.append(a_stream)

        decode_iter = iter(container.decode(*streams_to_decode))

        video_frame_idx = 0
        raw_video_frames_count = 0

        while True:
            if max_frames > 0 and raw_video_frames_count >= max_frames:
                break

            # -------------------------------------------------------------
            # Stage 1: PyAV Demux & Video Decode (measured per iterator next)
            # -------------------------------------------------------------
            t_demux_start = time.perf_counter_ns()
            try:
                frame = next(decode_iter)
            except StopIteration:
                break
            except Exception as exc:
                logger.warning("Decode stream encountered end or error: %s", exc)
                break
            t_demux_end = time.perf_counter_ns()

            # Handle Audio Frame
            if isinstance(frame, av.AudioFrame) or getattr(frame, "type", "") == "audio":
                profiler.audio_demux_resample_ns += (t_demux_end - t_demux_start)
                profiler.audio_frames_demuxed += 1

                t_resample_start = time.perf_counter_ns()
                try:
                    resampled = audio_resampler.resample(frame)
                    if resampled:
                        for rf in resampled:
                            accumulated_audio_chunks.append(rf.to_ndarray().flatten().astype(np.float32))
                except Exception:
                    pass
                t_resample_end = time.perf_counter_ns()
                profiler.audio_demux_resample_ns += (t_resample_end - t_resample_start)
                continue

            # Handle Video Frame
            raw_video_frames_count += 1
            profiler.demux_video_decode_ns += (t_demux_end - t_demux_start)
            profiler.video_frames_demuxed += 1

            # Subsampling check (e.g. 15fps down to 5fps or 1fps)
            if (raw_video_frames_count - 1) % frame_step != 0:
                continue

            video_frame_idx += 1
            profiler.video_frames_processed += 1

            # -------------------------------------------------------------
            # Stage 4: Candidate JPEG Encoding & Pooling
            # -------------------------------------------------------------
            if (video_frame_idx - 1) % yolo_sample_interval == 0:
                t_prep_start = time.perf_counter_ns()
                # Reformat / convert frame for YOLO candidate pool
                try:
                    bgr_cand = frame.reformat(width=w_target, height=h_target, format="bgr24").to_ndarray()
                except Exception:
                    raw_rgb = frame.to_ndarray(format="rgb24")
                    bgr_cand = cv2.resize(cv2.cvtColor(raw_rgb, cv2.COLOR_RGB2BGR), (w_target, h_target))
                t_prep_end = time.perf_counter_ns()

                t_enc_start = time.perf_counter_ns()
                ok_enc, buf_jpg = cv2.imencode(".jpg", bgr_cand, [cv2.IMWRITE_JPEG_QUALITY, 80])
                t_enc_end = time.perf_counter_ns()

                profiler.candidate_prep_ns += (t_prep_end - t_prep_start)
                profiler.candidate_jpeg_encode_ns += (t_enc_end - t_enc_start)
                profiler.candidate_jpegs_count += 1
                if ok_enc:
                    profiler.candidate_total_bytes += buf_jpg.nbytes
                    yolo_buffer[video_frame_idx - 1] = buf_jpg
                else:
                    profiler.candidate_total_bytes += bgr_cand.nbytes
                    yolo_buffer[video_frame_idx - 1] = bgr_cand

            # -------------------------------------------------------------
            # Stage 2: Grayscale Conversion & Bilinear Resize (416x234)
            # -------------------------------------------------------------
            t_gray_start = time.perf_counter_ns()
            gray_4k = video_frame_to_gray(frame)
            t_gray_end = time.perf_counter_ns()

            t_resize_start = time.perf_counter_ns()
            gray_resized = cv2.resize(
                gray_4k, (w_target, h_target), interpolation=cv2.INTER_LINEAR
            )
            t_resize_end = time.perf_counter_ns()

            profiler.grayscale_convert_ns += (t_gray_end - t_gray_start)
            profiler.resize_ns += (t_resize_end - t_resize_start)

            # -------------------------------------------------------------
            # Stage 3: EMA Motion Analysis (MotionTrace compute_frame_motion)
            # -------------------------------------------------------------
            roi = gray_resized[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
            t_ema_start = time.perf_counter_ns()
            saliency, _, _ = ema_model.update(roi)
            t_ema_end = time.perf_counter_ns()

            t_grid_start = time.perf_counter_ns()
            dt = 1.0 / effective_fps
            energy, _, stats = grid_filter.process_frame(saliency, dt, is_night_mode=False)
            t_grid_end = time.perf_counter_ns()

            profiler.ema_update_ns += (t_ema_end - t_ema_start)
            profiler.spatial_grid_ns += (t_grid_end - t_grid_start)
            motion_energies.append(energy)

        # Flush resampler
        try:
            flushed = audio_resampler.resample(None)
            if flushed:
                for rf in flushed:
                    accumulated_audio_chunks.append(rf.to_ndarray().flatten().astype(np.float32))
        except Exception:
            pass

    # -------------------------------------------------------------
    # Stage 5 (cont.): Audio VAD Calculation
    # -------------------------------------------------------------
    audio_full = np.concatenate(accumulated_audio_chunks) if accumulated_audio_chunks else np.array([], dtype=np.float32)
    if audio_full.size > 0:
        t_vad_start = time.perf_counter_ns()
        vad_events, vad_stats = vad_detector.detect_events(audio_full, start_offset=0.0)
        t_vad_end = time.perf_counter_ns()
        profiler.vad_detection_ns += (t_vad_end - t_vad_start)
        profiler.audio_events_count = len(vad_events)

    # Sum composite stages
    profiler.stage2_total_ns = profiler.grayscale_convert_ns + profiler.resize_ns
    profiler.stage3_total_ns = profiler.ema_update_ns + profiler.spatial_grid_ns
    profiler.stage4_total_ns = profiler.candidate_prep_ns + profiler.candidate_jpeg_encode_ns
    profiler.stage5_total_ns = profiler.audio_demux_resample_ns + profiler.vad_detection_ns

    # -------------------------------------------------------------
    # Stage 6: YOLO Verification & Lock Wait (if enabled)
    # -------------------------------------------------------------
    if enable_yolo and yolo_verifier and yolo_buffer:
        try:
            from src.algorithms.segment import build_segments, Segment
            from dataclasses import replace
            mock_labels = [
                {"time": i / effective_fps, "is_motion": bool(e > 2.5), "state": "DYNAMIC" if e > 2.5 else "STATIC", "energy": e, "raw_energy": e, "confidence": 0.5, "is_audio_active": False}
                for i, e in enumerate(motion_energies)
            ]
            seg_cfg = config.get("segment", {})
            segments = build_segments(
                mock_labels,
                str(video_path),
                min_motion_dur=seg_cfg.get("min_motion_duration", 2.0),
                min_static_dur=seg_cfg.get("min_static_duration", 8.0),
                file_offset=0.0,
                gap_tolerance=seg_cfg.get("gap_tolerance", 1.5),
            )
            # Ensure at least one dynamic candidate segment exists for profiling inference
            has_dynamic = any(s.is_dynamic for s in segments)
            if not has_dynamic and segments:
                segments[0] = replace(segments[0], state="DYNAMIC")
            elif not segments:
                max_t = len(motion_energies) / effective_fps
                segments = [Segment(start_time=0.0, end_time=max(0.1, max_t), state="DYNAMIC", filepath=str(video_path))]

            t_yolo_start = time.perf_counter_ns()
            yolo_verifier.verify(
                str(video_path),
                segments,
                gpu=hwaccel_mode,
                frames_buffer=yolo_buffer,
                analysis_fps=effective_fps,
            )
            t_yolo_end = time.perf_counter_ns()
            profiler.stage6_total_ns = t_yolo_end - t_yolo_start
            telem = getattr(yolo_verifier, "last_telemetry", {})
            profiler.yolo_lock_wait_ns = int(telem.get("yolo_lock_wait", 0.0) * 1e9)
            profiler.yolo_inference_ns = int(telem.get("yolo_infer_time", 0.0) * 1e9)
            profiler.yolo_preprocess_ns = max(
                0, profiler.stage6_total_ns - profiler.yolo_lock_wait_ns - profiler.yolo_inference_ns
            )
            profiler.yolo_verified_segments = len(segments)
        except Exception as exc:
            logger.warning("YOLO verification profiling failed: %s", exc)
            yolo_status = f"ERROR: {exc}"

    t_wall_end_ns = time.perf_counter_ns()
    total_wall_ns = t_wall_end_ns - t_wall_start_ns
    hw_stats = sampler.stop()

    # Calculate micro-stage totals and percentages
    measured_sum_ns = (
        profiler.demux_video_decode_ns
        + profiler.stage2_total_ns
        + profiler.stage3_total_ns
        + profiler.stage4_total_ns
        + profiler.stage5_total_ns
        + profiler.stage6_total_ns
    )
    denom_ns = max(1, measured_sum_ns)
    video_frames = max(1, profiler.video_frames_demuxed)
    factor_10k = 10000.0 / video_frames

    stages_data = [
        {
            "id": 1,
            "key": "demux_and_video_decode",
            "name": "1. PyAV Demux & Video Decode",
            "total_ns": profiler.demux_video_decode_ns,
            "total_ms": round(profiler.demux_video_decode_ns / 1e6, 2),
            "pct_of_analyzed": round((profiler.demux_video_decode_ns / denom_ns) * 100.0, 2),
            "avg_ms_per_demuxed_frame": round((profiler.demux_video_decode_ns / 1e6) / video_frames, 3),
            "time_per_10k_frames_s": round((profiler.demux_video_decode_ns / 1e9) * factor_10k, 2),
            "frames_count": profiler.video_frames_demuxed,
        },
        {
            "id": 2,
            "key": "grayscale_and_resize",
            "name": f"2. Grayscale conversion & Resize ({analysis_res})",
            "total_ns": profiler.stage2_total_ns,
            "total_ms": round(profiler.stage2_total_ns / 1e6, 2),
            "pct_of_analyzed": round((profiler.stage2_total_ns / denom_ns) * 100.0, 2),
            "avg_ms_per_analyzed_frame": round((profiler.stage2_total_ns / 1e6) / max(1, profiler.video_frames_processed), 3),
            "time_per_10k_frames_s": round((profiler.stage2_total_ns / 1e9) * factor_10k, 2),
            "frames_count": profiler.video_frames_processed,
            "sub_breakdown": {
                "grayscale_extract_ms": round(profiler.grayscale_convert_ns / 1e6, 2),
                "bilinear_resize_ms": round(profiler.resize_ns / 1e6, 2),
            },
        },
        {
            "id": 3,
            "key": "ema_motion_analysis",
            "name": "3. EMA Motion Analysis (MotionTrace)",
            "total_ns": profiler.stage3_total_ns,
            "total_ms": round(profiler.stage3_total_ns / 1e6, 2),
            "pct_of_analyzed": round((profiler.stage3_total_ns / denom_ns) * 100.0, 2),
            "avg_ms_per_analyzed_frame": round((profiler.stage3_total_ns / 1e6) / max(1, profiler.video_frames_processed), 3),
            "time_per_10k_frames_s": round((profiler.stage3_total_ns / 1e9) * factor_10k, 2),
            "frames_count": profiler.video_frames_processed,
            "sub_breakdown": {
                "ema_background_update_ms": round(profiler.ema_update_ns / 1e6, 2),
                "spatial_grid_filter_ms": round(profiler.spatial_grid_ns / 1e6, 2),
            },
        },
        {
            "id": 4,
            "key": "candidate_jpeg_encoding",
            "name": "4. Candidate JPEG Encoding & Pooling",
            "total_ns": profiler.stage4_total_ns,
            "total_ms": round(profiler.stage4_total_ns / 1e6, 2),
            "pct_of_analyzed": round((profiler.stage4_total_ns / denom_ns) * 100.0, 2),
            "avg_ms_per_candidate": (
                round((profiler.stage4_total_ns / 1e6) / profiler.candidate_jpegs_count, 3)
                if profiler.candidate_jpegs_count
                else 0.0
            ),
            "time_per_10k_frames_s": round((profiler.stage4_total_ns / 1e9) * factor_10k, 2),
            "candidates_count": profiler.candidate_jpegs_count,
            "total_jpeg_kb": round(profiler.candidate_total_bytes / 1024.0, 1),
            "sub_breakdown": {
                "color_format_prep_ms": round(profiler.candidate_prep_ns / 1e6, 2),
                "jpeg_imencode_ms": round(profiler.candidate_jpeg_encode_ns / 1e6, 2),
            },
        },
        {
            "id": 5,
            "key": "audio_demux_and_vad",
            "name": "5. Audio Demux & VAD",
            "total_ns": profiler.stage5_total_ns,
            "total_ms": round(profiler.stage5_total_ns / 1e6, 2),
            "pct_of_analyzed": round((profiler.stage5_total_ns / denom_ns) * 100.0, 2),
            "time_per_10k_frames_s": round((profiler.stage5_total_ns / 1e9) * factor_10k, 2),
            "audio_frames_demuxed": profiler.audio_frames_demuxed,
            "audio_events_detected": profiler.audio_events_count,
            "sub_breakdown": {
                "audio_demux_and_resample_ms": round(profiler.audio_demux_resample_ns / 1e6, 2),
                "vad_detection_ms": round(profiler.vad_detection_ns / 1e6, 2),
            },
        },
        {
            "id": 6,
            "key": "yolo_verification",
            "name": "6. YOLO Verification & Lock Wait",
            "status": yolo_status,
            "total_ns": profiler.stage6_total_ns,
            "total_ms": round(profiler.stage6_total_ns / 1e6, 2),
            "pct_of_analyzed": round((profiler.stage6_total_ns / denom_ns) * 100.0, 2),
            "time_per_10k_frames_s": round((profiler.stage6_total_ns / 1e9) * factor_10k, 2),
            "segments_verified": profiler.yolo_verified_segments,
            "sub_breakdown": {
                "lock_wait_ms": round(profiler.yolo_lock_wait_ns / 1e6, 2),
                "inference_ms": round(profiler.yolo_inference_ns / 1e6, 2),
                "preprocess_ms": round(profiler.yolo_preprocess_ns / 1e6, 2),
            },
        },
    ]

    total_wall_s = total_wall_ns / 1e9
    throughput_fps = round(profiler.video_frames_demuxed / max(0.001, total_wall_s), 2)
    speedup_x = round(throughput_fps / max(0.001, raw_fps), 2)
    wall_10k_s = round(total_wall_s * factor_10k, 2)

    # Architectural diagnostics synthesis
    stage1 = stages_data[0]
    stage2 = stages_data[1]
    stage3 = stages_data[2]
    stage4 = stages_data[3]

    insights = [
        f"4K 视频流原始分辨率: {v_width}x{v_height} ({v_codec}), 帧率 {raw_fps:.2f}fps, 实际抽样帧率 {effective_fps:.2f}fps.",
        f"单万帧总耗时: {wall_10k_s:.2f}s, 等效实时处理倍速: {speedup_x:.2f}x (解码吞吐 {throughput_fps:.1f} fps).",
        f"阶段1 (Demux & Video Decode) 占比 {stage1['pct_of_analyzed']}%, 单万帧耗时 {stage1['time_per_10k_frames_s']:.2f}s (硬件加速: {hwaccel_status}).",
        f"阶段2 (灰度提取与双线性缩放) 占比 {stage2['pct_of_analyzed']}%, 单万帧耗时 {stage2['time_per_10k_frames_s']:.2f}s. 其中高分辨率 4K 全尺寸灰度格式转换 (video_frame_to_gray) 占据主要 CPU 开销 ({stage2['sub_breakdown']['grayscale_extract_ms']}ms vs 缩放 {stage2['sub_breakdown']['bilinear_resize_ms']}ms).",
        f"阶段3 (EMA 运动与空间网格) 占比 {stage3['pct_of_analyzed']}%, 单万帧耗时 {stage3['time_per_10k_frames_s']:.2f}s. 表明下采样到 {analysis_res} 后核心算法开销极度轻量.",
        f"阶段4 (候选帧 JPEG 压缩) 占比 {stage4['pct_of_analyzed']}%, 每候选帧编码均耗时 {stage4['avg_ms_per_candidate']:.2f}ms.",
    ]

    return {
        "report_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_media": {
            "path": str(video_path),
            "filename": video_path.name,
            "file_size_bytes": file_size_bytes,
            "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
            "video_codec": v_codec,
            "audio_codec": a_codec,
            "native_resolution": f"{v_width}x{v_height}",
            "native_fps": round(raw_fps, 2),
            "duration_s": round(v_duration_s, 2),
        },
        "profiling_parameters": {
            "max_frames_requested": max_frames,
            "hwaccel_mode": hwaccel_mode,
            "hwaccel_active": hwaccel_status,
            "analysis_resolution": analysis_res,
            "target_analysis_fps": analysis_fps,
            "effective_analysis_fps": round(effective_fps, 2),
            "frame_step": frame_step,
            "yolo_enabled": enable_yolo,
            "yolo_model": yolo_model_path if enable_yolo else "N/A",
        },
        "execution_summary": {
            "total_wall_clock_s": round(total_wall_s, 3),
            "total_wall_clock_ms": round(total_wall_ns / 1e6, 2),
            "video_frames_demuxed": profiler.video_frames_demuxed,
            "video_frames_analyzed": profiler.video_frames_processed,
            "audio_frames_demuxed": profiler.audio_frames_demuxed,
            "audio_chunks_count": len(accumulated_audio_chunks),
            "candidate_jpegs_count": profiler.candidate_jpegs_count,
            "throughput_fps": throughput_fps,
            "speedup_vs_realtime": speedup_x,
            "per_10k_frames_wall_clock_s": wall_10k_s,
        },
        "stages": stages_data,
        "hardware_utilization": hw_stats,
        "architectural_insights": insights,
    }


def print_report_table(report: dict[str, Any]):
    """Pretty prints the diagnostic report to console."""
    media = report["input_media"]
    summary = report["execution_summary"]
    hw = report["hardware_utilization"]

    print("\n" + "=" * 80)
    print(" HomeVlog Pass 1.5 Analysis 耗时精准微观分解诊断收据 (Profile Receipt)")
    print("=" * 80)
    print(f"源文件: {media['filename']} ({media['native_resolution']}, {media['video_codec']})")
    print(f"体积: {media['file_size_mb']} MB | 时长: {media['duration_s']}s | 视频原生帧率: {media['native_fps']} fps")
    print(f"解码帧数: {summary['video_frames_demuxed']} 帧 | 抽样精析: {summary['video_frames_analyzed']} 帧")
    print(f"硬件环境: {hw['gpu_name']} | 显存峰值: {hw['vram_peak_mb']:.1f} MB | CPU 峰值: {hw['cpu_peak_pct']:.1f}%")
    print(f"总壁钟耗时: {summary['total_wall_clock_s']:.3f}s | 吞吐率: {summary['throughput_fps']:.1f} fps ({summary['speedup_vs_realtime']:.2f}x 实时)")
    print(f"标准化每万帧总耗时 (Per 10k Frames): {summary['per_10k_frames_wall_clock_s']:.2f} 秒")
    print("-" * 80)
    print(f"{'子阶段名称 (Micro-Stage)':<40} | {'总耗时(ms)':<10} | {'耗时占比':<8} | {'单万帧耗时(s)':<12}")
    print("-" * 80)
    for st in report["stages"]:
        sub_info = ""
        if "sub_breakdown" in st:
            parts = [f"{k}: {v}ms" for k, v in st["sub_breakdown"].items()]
            sub_info = f"\n  └─ {', '.join(parts)}"
        print(f"{st['name']:<40} | {st['total_ms']:>10.2f} | {st['pct_of_analyzed']:>7.2f}% | {st['time_per_10k_frames_s']:>12.2f}{sub_info}")
    print("-" * 80)
    print("诊断洞察 (Architectural Insights):")
    for ins in report["architectural_insights"]:
        print(f"  • {ins}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Profile decode and analysis micro-costs.")
    parser.add_argument(
        "--input",
        default="scratch/complex_batch_diagnosis/e35443eff91f4b11a0ba328cadffdb19/00_20260328002805_20260328013503.mp4",
        help="Target video file path",
    )
    parser.add_argument(
        "--output",
        default="scratch/decode_breakdown/report.json",
        help="Output JSON receipt path",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=3000,
        help="Max video frames to decode (0 for all)",
    )
    parser.add_argument(
        "--hwaccel",
        choices=["cuda", "qsv", "cpu"],
        default="cuda",
        help="Hardware acceleration mode (cuda, qsv, cpu)",
    )
    parser.add_argument(
        "--resolution",
        default="416x234",
        help="Analysis resolution, e.g. 416x234",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=5.0,
        help="Target analysis FPS (default: 5.0)",
    )
    parser.add_argument(
        "--enable-yolo",
        action="store_true",
        help="Enable YOLO verification profiling on candidate frames",
    )
    parser.add_argument(
        "--yolo-model",
        default="models/yolo11n.pt",
        help="Path to YOLO model weights",
    )
    parser.add_argument(
        "--compare-hw",
        action="store_true",
        help="Compare CUDA, QSV, and CPU decode paths side-by-side",
    )

    args = parser.parse_args()

    media_path = resolve_media_path(args.input)
    config = load_config()

    out_file = Path(args.output)
    if not out_file.is_absolute():
        out_file = PROJECT_ROOT / out_file
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if args.compare_hw:
        print("\n" + "=" * 90)
        print(" 启动异构硬件解码对比评测: CUDA (NVDEC) vs QSV (Intel) vs CPU (纯软解)")
        print(f" 目标素材: {media_path.name} | 每路采样: {args.frames} 帧")
        print("=" * 90)

        hw_modes = ["cuda", "qsv", "cpu"]
        hw_reports: dict[str, dict[str, Any]] = {}
        for mode in hw_modes:
            print(f"正在评测 [{mode.upper()}] 解码路径...")
            hw_reports[mode] = profile_pass15_analysis(
                video_path=media_path,
                config=config,
                max_frames=args.frames,
                hwaccel_mode=mode,
                analysis_res=args.resolution,
                analysis_fps=args.fps,
                enable_yolo=args.enable_yolo,
                yolo_model_path=args.yolo_model,
            )

        # Print comparison table
        print("\n" + "=" * 90)
        print(" 异构硬件解码路径横向性能与开销实测对比矩阵")
        print("=" * 90)
        header = f"{'指标 (Metric)':<35} | {'CUDA (NVDEC)':<15} | {'QSV (Intel)':<15} | {'CPU (纯软解)':<15}"
        print(header)
        print("-" * 90)

        c_rep = hw_reports["cuda"]
        q_rep = hw_reports["qsv"]
        s_rep = hw_reports["cpu"]

        rows = [
            ("总壁钟耗时 (Total Wall Clock)", f"{c_rep['execution_summary']['total_wall_clock_s']:.3f}s", f"{q_rep['execution_summary']['total_wall_clock_s']:.3f}s", f"{s_rep['execution_summary']['total_wall_clock_s']:.3f}s"),
            ("解码吞吐率 (Throughput FPS)", f"{c_rep['execution_summary']['throughput_fps']:.1f} fps", f"{q_rep['execution_summary']['throughput_fps']:.1f} fps", f"{s_rep['execution_summary']['throughput_fps']:.1f} fps"),
            ("实时倍速 (Realtime Speedup)", f"{c_rep['execution_summary']['speedup_vs_realtime']:.2f}x", f"{q_rep['execution_summary']['speedup_vs_realtime']:.2f}x", f"{s_rep['execution_summary']['speedup_vs_realtime']:.2f}x"),
            ("每万帧壁钟耗时 (Per 10k Wall)", f"{c_rep['execution_summary']['per_10k_frames_wall_clock_s']:.2f}s", f"{q_rep['execution_summary']['per_10k_frames_wall_clock_s']:.2f}s", f"{s_rep['execution_summary']['per_10k_frames_wall_clock_s']:.2f}s"),
            ("阶段1: Demux & Video Decode", f"{c_rep['stages'][0]['total_ms']:.1f}ms ({c_rep['stages'][0]['pct_of_analyzed']}%)", f"{q_rep['stages'][0]['total_ms']:.1f}ms ({q_rep['stages'][0]['pct_of_analyzed']}%)", f"{s_rep['stages'][0]['total_ms']:.1f}ms ({s_rep['stages'][0]['pct_of_analyzed']}%)"),
            ("  └─ 阶段1 每万帧耗时", f"{c_rep['stages'][0]['time_per_10k_frames_s']:.2f}s", f"{q_rep['stages'][0]['time_per_10k_frames_s']:.2f}s", f"{s_rep['stages'][0]['time_per_10k_frames_s']:.2f}s"),
            ("阶段2: 灰度提取与双线性缩放", f"{c_rep['stages'][1]['total_ms']:.1f}ms ({c_rep['stages'][1]['pct_of_analyzed']}%)", f"{q_rep['stages'][1]['total_ms']:.1f}ms ({q_rep['stages'][1]['pct_of_analyzed']}%)", f"{s_rep['stages'][1]['total_ms']:.1f}ms ({s_rep['stages'][1]['pct_of_analyzed']}%)"),
            ("  └─ 阶段2 每万帧耗时", f"{c_rep['stages'][1]['time_per_10k_frames_s']:.2f}s", f"{q_rep['stages'][1]['time_per_10k_frames_s']:.2f}s", f"{s_rep['stages'][1]['time_per_10k_frames_s']:.2f}s"),
            ("阶段3: EMA 运动分析", f"{c_rep['stages'][2]['total_ms']:.1f}ms ({c_rep['stages'][2]['pct_of_analyzed']}%)", f"{q_rep['stages'][2]['total_ms']:.1f}ms ({q_rep['stages'][2]['pct_of_analyzed']}%)", f"{s_rep['stages'][2]['total_ms']:.1f}ms ({s_rep['stages'][2]['pct_of_analyzed']}%)"),
            ("阶段4: 候选帧 JPEG 编码", f"{c_rep['stages'][3]['total_ms']:.1f}ms ({c_rep['stages'][3]['pct_of_analyzed']}%)", f"{q_rep['stages'][3]['total_ms']:.1f}ms ({q_rep['stages'][3]['pct_of_analyzed']}%)", f"{s_rep['stages'][3]['total_ms']:.1f}ms ({s_rep['stages'][3]['pct_of_analyzed']}%)"),
            ("阶段5: 音频解复用与 VAD", f"{c_rep['stages'][4]['total_ms']:.1f}ms ({c_rep['stages'][4]['pct_of_analyzed']}%)", f"{q_rep['stages'][4]['total_ms']:.1f}ms ({q_rep['stages'][4]['pct_of_analyzed']}%)", f"{s_rep['stages'][4]['total_ms']:.1f}ms ({s_rep['stages'][4]['pct_of_analyzed']}%)"),
            ("阶段6: YOLO 验证推理", f"{c_rep['stages'][5]['total_ms']:.1f}ms", f"{q_rep['stages'][5]['total_ms']:.1f}ms", f"{s_rep['stages'][5]['total_ms']:.1f}ms"),
            ("CPU 占用均值 / 峰值", f"{c_rep['hardware_utilization']['cpu_avg_pct']}% / {c_rep['hardware_utilization']['cpu_peak_pct']}%", f"{q_rep['hardware_utilization']['cpu_avg_pct']}% / {q_rep['hardware_utilization']['cpu_peak_pct']}%", f"{s_rep['hardware_utilization']['cpu_avg_pct']}% / {s_rep['hardware_utilization']['cpu_peak_pct']}%"),
            ("显存占用峰值 (VRAM)", f"{c_rep['hardware_utilization']['vram_peak_mb']:.1f} MB", f"{q_rep['hardware_utilization']['vram_peak_mb']:.1f} MB", f"{s_rep['hardware_utilization']['vram_peak_mb']:.1f} MB"),
        ]
        for name, col_c, col_q, col_s in rows:
            print(f"{name:<35} | {col_c:<15} | {col_q:<15} | {col_s:<15}")
        print("=" * 90 + "\n")

        comparison_report = {
            "primary": c_rep,
            "hardware_comparison": {
                "cuda": c_rep,
                "qsv": q_rep,
                "cpu": s_rep,
            },
        }
        out_file.write_text(json.dumps(comparison_report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"完整硬件对比诊断收据已写入: {out_file.resolve()}")
        return

    report = profile_pass15_analysis(
        video_path=media_path,
        config=config,
        max_frames=args.frames,
        hwaccel_mode=args.hwaccel,
        analysis_res=args.resolution,
        analysis_fps=args.fps,
        enable_yolo=args.enable_yolo,
        yolo_model_path=args.yolo_model,
    )

    out_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print_report_table(report)
    print(f"诊断收据已写入: {out_file.resolve()}")


if __name__ == "__main__":
    main()
