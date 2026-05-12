import logging
import subprocess
import threading
import time
from pathlib import Path

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
    ) -> list[dict]:
        frame_size = self.width * self.height
        if self.decode_gpu == "qsv":
            args = [
                "-hwaccel",
                "qsv",
                "-hwaccel_output_format",
                "qsv",
                "-i",
                str(filepath),
                "-vf",
                f"scale_qsv=w={self.width}:h={self.height},hwdownload,format=nv12,extractplanes=y",
                "-r",
                str(self.fps),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ]
        else:
            args = [
                "-hwaccel",
                "cuda",
                "-hwaccel_output_format",
                "cuda",
                "-i",
                str(filepath),
                "-vf",
                f"scale_cuda={self.width}:{self.height},hwdownload,format=nv12,extractplanes=y",
                "-r",
                str(self.fps),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "gray",
                "-",
            ]

        if self.decode_gpu == "qsv":
            from src.utils import get_qsv_semaphore

            io_sem = get_qsv_semaphore()
        else:
            from src.utils import get_nv_semaphore

            io_sem = get_nv_semaphore()
        io_sem.acquire()
        try:
            proc = subprocess.Popen(
                ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=frame_size * self.pipe_buf_mult,
            )

            # Watchdog: 动态超时 — 基于文件时长或固定值取大
            # 长文件（如 65min）需要更多解码时间，不能用固定值误杀
            fixed_timeout = self.decode_timeout * 3
            dynamic_timeout = (
                (file_duration / max(self.fps, 1)) * 2 if file_duration > 0 else 0
            )
            watchdog_timeout = max(fixed_timeout, dynamic_timeout, 60.0)

            def _kill_on_timeout():
                try:
                    proc.kill()
                except OSError:
                    pass

            watchdog = threading.Timer(watchdog_timeout, _kill_on_timeout)
            watchdog.daemon = True
            watchdog.start()

            stderr_lines: list[str] = []

            def _read_stderr():
                for line in proc.stderr:
                    stderr_lines.append(line.decode("utf-8", errors="replace"))

            stderr_thread = threading.Thread(target=_read_stderr, daemon=True)
            stderr_thread.start()

            prev_gray: np.ndarray | None = None
            energies: list[float] = []

            roi_x = int(self.width * self.roi[0])
            roi_y = int(self.height * self.roi[1])
            roi_w = int(self.width * self.roi[2])
            roi_h = int(self.height * self.roi[3])

            t_decode_start = time.monotonic()
            total_frames = 0

            try:
                while True:
                    raw = proc.stdout.read(frame_size)
                    if len(raw) < frame_size:
                        break
                    total_frames += 1

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.height, self.width)
                    )
                    roi = frame[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w].astype(
                        np.float32
                    )
                    frame = None  # free

                    if prev_gray is not None:
                        diff = np.abs(roi - prev_gray)
                        energies.append(float(np.mean(diff)))
                    # First frame has no predecessor — excluded from energies

                    prev_gray = roi
            except Exception:
                logger.warning("ffmpeg read interrupted for %s", Path(filepath).name)
                proc.kill()
            finally:
                watchdog.cancel()

            t_decode_end = time.monotonic()

            try:
                proc.wait(timeout=self.decode_timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            finally:
                if proc.stdout:
                    proc.stdout.close()
                if proc.stderr:
                    proc.stderr.close()
            stderr_thread.join(timeout=2)
        finally:
            io_sem.release()

        if proc.returncode != 0 and proc.returncode != -9:
            err = "".join(stderr_lines[-5:]) if stderr_lines else ""
            logger.warning("ffmpeg warning for %s: %s", Path(filepath).name, err)

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
        return results


def _median_filter(signal: list[float], window: int) -> list[float]:
    """1D median filter — suppresses isolated spikes, preserves sustained runs."""
    arr = np.array(signal, dtype=np.float32)
    half = window // 2
    n = len(arr)
    result = np.empty_like(arr)
    for i in range(n):
        left = max(0, i - half)
        right = min(n, i + half + 1)
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


def _analysis_worker(
    db, task_queue, date: str, config: dict, decode_gpu: str
) -> tuple[dict, str]:
    """Process SUSPICIOUS files from queue with given decode GPU (Work-Stealing)."""
    from src.segment import build_segments, segments_to_json
    from src.yolo_verifier import YoloVerifier

    detector = MotionDetector(config, decode_gpu=decode_gpu)
    yolo_verifier = YoloVerifier(config)
    perf = get_perf()
    stats = {"done": 0, "analyzed": 0, "failed": 0}

    while True:
        task = task_queue.get()
        if task is None:
            break

        filepath = task["filepath"]
        t0 = time.monotonic()
        try:
            file_start_ts = ts_to_unix(task["file_start_time"])
            file_start_offset = file_start_ts - ts_to_unix(date + "000000")
            file_start_offset = max(file_start_offset, 0.0)

            labels = detector.analyze(
                filepath,
                start_offset=file_start_offset,
                file_duration=task.get("file_duration") or 300.0,
            )
            elapsed = time.monotonic() - t0
            lp = detector.last_perf if hasattr(detector, "last_perf") else {}

            if not labels:
                db.set_analysis_result(filepath, "FAILED", "")
                stats["failed"] += 1
                stats["done"] += 1
                perf.add(
                    PerfRecord(
                        stage="analysis",
                        file=Path(filepath).name,
                        gpu=decode_gpu,
                        duration=round(elapsed, 3),
                        frames=lp.get("frames", 0),
                        extra={"status": "FAILED", **lp},
                    )
                )
                continue

            segments = build_segments(
                labels,
                filepath,
                min_motion_dur=config.get("segment", {}).get(
                    "min_motion_duration", 2.0
                ),
                min_static_dur=config.get("segment", {}).get(
                    "min_static_duration", 30.0
                ),
                file_offset=file_start_offset,
                gap_tolerance=config.get("segment", {}).get("gap_tolerance", 0.5),
                apply_smoothing=False,  # 核心优化：延迟到全局 timeline 阶段平滑，防止边界截断
            )

            # YOLO 二次验证 (过滤掉只包含光影/风吹树叶的误报片段)
            segments = yolo_verifier.verify(filepath, segments, gpu=decode_gpu)

            js = segments_to_json(segments)
            db.set_analysis_result(filepath, "ANALYZED", js)
            stats["analyzed"] += 1

            perf.add(
                PerfRecord(
                    stage="analysis",
                    file=Path(filepath).name,
                    gpu=decode_gpu,
                    duration=round(elapsed, 3),
                    frames=lp.get("frames", 0),
                    fps=round(lp.get("frames", 0) / elapsed, 1) if elapsed > 0 else 0,
                    extra={"status": "ANALYZED", "n_segments": len(segments), **lp},
                )
            )

        except Exception:
            elapsed = time.monotonic() - t0
            logger.exception("analysis failed for %s [%s]", filepath, decode_gpu)
            db.set_analysis_result(filepath, "FAILED", "")
            stats["failed"] += 1
            perf.add(
                PerfRecord(
                    stage="analysis",
                    file=Path(filepath).name,
                    gpu=decode_gpu,
                    duration=round(elapsed, 3),
                    extra={"status": "ERROR"},
                )
            )

        stats["done"] += 1

    return stats, decode_gpu


def run_analysis_for_cam(
    db,
    date: str,
    cam_index: int,
    config: dict,
) -> dict:
    """Run Pass1.5 analysis using dual-GPU work-stealing queue."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from queue import Queue

    suspicious = db.get_suspicious_files(date, cam_index)
    if not suspicious:
        logger.info(
            "analysis: no SUSPICIOUS+PENDING files for %s cam%d", date, cam_index
        )
        return {"done": 0, "analyzed": 0, "failed": 0}

    logger.info("analysis: %d files for %s cam%d", len(suspicious), date, cam_index)

    # Sort by file_duration desc so longest files are processed first
    sorted_tasks = sorted(
        suspicious, key=lambda t: t.get("file_duration") or 0, reverse=True
    )

    task_queue: Queue[dict | None] = Queue()
    for task in sorted_tasks:
        task_queue.put(task)

    # Sentinel values to terminate 2 workers
    task_queue.put(None)
    task_queue.put(None)

    stats = {"done": 0, "analyzed": 0, "failed": 0}
    max_workers = config.get("detection", {}).get("analysis_max_workers", 2)
    t_start = time.monotonic()
    worker_durations: dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        f_nv = pool.submit(_analysis_worker, db, task_queue, date, config, "cuda")
        f_qsv = pool.submit(_analysis_worker, db, task_queue, date, config, "qsv")

        for f in as_completed([f_nv, f_qsv]):
            t_done = time.monotonic()
            worker_stats, gpu = f.result()
            worker_durations[gpu] = round(t_done - t_start, 1)
            for k in stats:
                stats[k] += worker_stats[k]

    total_elapsed = time.monotonic() - t_start
    logger.info(
        "analysis %s cam%d done in %.1fs: analyzed=%d failed=%d (cuda=%.1fs qsv=%.1fs)",
        date,
        cam_index,
        total_elapsed,
        stats["analyzed"],
        stats["failed"],
        worker_durations.get("cuda", 0),
        worker_durations.get("qsv", 0),
    )
    return stats
