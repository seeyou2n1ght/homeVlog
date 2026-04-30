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

    def analyze(self, filepath: str, start_offset: float = 0.0) -> list[dict]:
        frame_size = self.width * self.height
        if self.decode_gpu == "qsv":
            args = [
                "-hwaccel", "qsv",
                "-hwaccel_output_format", "qsv",
                "-i", str(filepath),
                "-vf", f"scale_qsv=w={self.width}:h={self.height},hwdownload,format=nv12,extractplanes=y",
                "-r", str(self.fps),
                "-f", "rawvideo",
                "-pix_fmt", "gray",
                "-",
            ]
        else:
            args = [
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
                "-i", str(filepath),
                "-vf", f"scale_cuda={self.width}:{self.height},hwdownload,format=nv12,extractplanes=y",
                "-r", str(self.fps),
                "-f", "rawvideo",
                "-pix_fmt", "gray",
                "-",
            ]

        proc = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"] + args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=frame_size * self.pipe_buf_mult,
        )

        # Watchdog: kill process if total time exceeds timeout
        def _kill_on_timeout():
            try:
                proc.kill()
            except OSError:
                pass
        watchdog = threading.Timer(self.decode_timeout * 3, _kill_on_timeout)
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

                frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width))
                roi = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].astype(np.float32)
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
        stderr_thread.join(timeout=2)

        if proc.returncode != 0 and proc.returncode != -9:
            err = "".join(stderr_lines[-5:]) if stderr_lines else ""
            logger.warning("ffmpeg warning for %s: %s", Path(filepath).name, err)

        if len(energies) < 2:
            logger.warning("too few frames from %s: %d", Path(filepath).name, len(energies))
            self.last_perf = {"decode_time": t_decode_end - t_decode_start, "analysis_time": 0, "frames": total_frames, "motion_ratio": 0}
            return []

        t_analysis_start = time.monotonic()

        # Temporal median filter to suppress isolated GOP boundary spikes
        # (1-2 frame artifacts) while preserving sustained motion (3+ frames).
        if self.median_window >= 3:
            energies = _median_filter(energies, self.median_window)

        # Adaptive threshold: IQR-based outlier detection
        energies_arr = np.array(energies, dtype=np.float32)
        q1 = float(np.percentile(energies_arr, 25))
        q3 = float(np.percentile(energies_arr, 75))
        iqr = q3 - q1
        threshold = q3 + self.sensitivity * iqr

        raw_labels: list[bool] = [bool(e > threshold) for e in energies]
        # First frame is always False (no predecessor to compare)
        raw_labels.insert(0, False)

        logger.debug(
            "adaptive threshold for %s: Q1=%.3f Q3=%.3f IQR=%.3f threshold=%.3f "
            "motion_frames=%d/%d (%.1f%%)",
            Path(filepath).name, q1, q3, iqr, threshold,
            sum(raw_labels), len(raw_labels), 100 * sum(raw_labels) / len(raw_labels),
        )

        smoothed = _smooth_labels(raw_labels, self.min_motion_frames, self.min_static_frames, self.noise_suppress)
        t_analysis_end = time.monotonic()

        motion_count = sum(1 for s in smoothed if s)
        self.last_perf = {
            "decode_time": round(t_decode_end - t_decode_start, 3),
            "analysis_time": round(t_analysis_end - t_analysis_start, 3),
            "frames": total_frames,
            "motion_ratio": round(motion_count / len(smoothed), 3) if smoothed else 0,
        }

        frame_interval = 1.0 / self.fps
        results = []
        for i, is_motion in enumerate(smoothed):
            results.append({
                "time": start_offset + i * frame_interval,
                "is_motion": is_motion,
            })
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


def _analysis_worker(db, tasks: list[dict], date: str, config: dict, decode_gpu: str) -> tuple[dict, str]:
    """Process a sub-list of SUSPICIOUS files with given decode GPU. Returns (stats, decode_gpu)."""
    from src.segment import build_segments, segments_to_json

    detector = MotionDetector(config, decode_gpu=decode_gpu)
    perf = get_perf()
    stats = {"done": 0, "analyzed": 0, "failed": 0}

    for task in tasks:
        filepath = task["filepath"]
        t0 = time.monotonic()
        try:
            file_start_ts = ts_to_unix(task["file_start_time"])
            file_start_offset = file_start_ts - ts_to_unix(date + "000000")
            file_start_offset = max(file_start_offset, 0.0)

            labels = detector.analyze(filepath, start_offset=file_start_offset)
            elapsed = time.monotonic() - t0
            lp = getattr(detector, "last_perf", {})

            if not labels:
                db.set_analysis_result(filepath, "FAILED", "")
                stats["failed"] += 1
                stats["done"] += 1
                perf.add(PerfRecord(
                    stage="analysis", file=Path(filepath).name, gpu=decode_gpu,
                    duration=round(elapsed, 3), frames=lp.get("frames", 0),
                    extra={"status": "FAILED", **lp},
                ))
                continue

            segments = build_segments(
                labels, filepath,
                min_motion_dur=config.get("segment", {}).get("min_motion_duration", 2.0),
                min_static_dur=config.get("segment", {}).get("min_static_duration", 30.0),
                file_offset=file_start_offset,
                gap_tolerance=config.get("segment", {}).get("gap_tolerance", 0.5),
            )
            js = segments_to_json(segments)
            db.set_analysis_result(filepath, "ANALYZED", js)
            stats["analyzed"] += 1

            perf.add(PerfRecord(
                stage="analysis", file=Path(filepath).name, gpu=decode_gpu,
                duration=round(elapsed, 3), frames=lp.get("frames", 0),
                fps=round(lp.get("frames", 0) / elapsed, 1) if elapsed > 0 else 0,
                extra={"status": "ANALYZED", "n_segments": len(segments), **lp},
            ))

        except Exception:
            elapsed = time.monotonic() - t0
            logger.exception("analysis failed for %s [%s]", filepath, decode_gpu)
            db.set_analysis_result(filepath, "FAILED", "")
            stats["failed"] += 1
            perf.add(PerfRecord(
                stage="analysis", file=Path(filepath).name, gpu=decode_gpu,
                duration=round(elapsed, 3), extra={"status": "ERROR"},
            ))

        stats["done"] += 1

    return stats, decode_gpu


def run_analysis_for_cam(
    db,
    date: str,
    cam_index: int,
    config: dict,
) -> dict:
    """Run Pass1.5 analysis for SUSPICIOUS files of a date/cam with dual-GPU parallel decode."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    suspicious = db.get_suspicious_files(date, cam_index)
    if not suspicious:
        logger.info("analysis: no SUSPICIOUS+PENDING files for %s cam%d", date, cam_index)
        return {"done": 0, "analyzed": 0, "failed": 0}

    logger.info("analysis: %d files for %s cam%d", len(suspicious), date, cam_index)

    # Sort by file_duration desc, interleave for balanced GPU load.
    # NVDEC is slower per-file in practice (~19.5s vs QSV ~13.7s),
    # so QSV gets more files to equalize wall time.
    sorted_tasks = sorted(suspicious, key=lambda t: t.get("file_duration") or 0, reverse=True)
    n_total = len(sorted_tasks)
    n_nvdec = round(n_total * 0.43)  # ~43% to NVDEC (slower GPU)
    list_nvdec = sorted_tasks[:n_nvdec]
    list_qsv = sorted_tasks[n_nvdec:]

    logger.info("analysis split: NVDEC=%d QSV=%d", len(list_nvdec), len(list_qsv))

    stats = {"done": 0, "analyzed": 0, "failed": 0}
    max_workers = config.get("detection", {}).get("analysis_max_workers", 2)
    t_start = time.monotonic()
    worker_durations: dict[str, float] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        f_nv = pool.submit(_analysis_worker, db, list_nvdec, date, config, "cuda")
        f_qsv = pool.submit(_analysis_worker, db, list_qsv, date, config, "qsv")

        for f in as_completed([f_nv, f_qsv]):
            t_done = time.monotonic()
            worker_stats, gpu = f.result()
            worker_durations[gpu] = round(t_done - t_start, 1)
            for k in stats:
                stats[k] += worker_stats[k]
            logger.info("analysis worker [%s] done in %.1fs: analyzed=%d failed=%d",
                       gpu, t_done - t_start, worker_stats["analyzed"], worker_stats["failed"])

    total_elapsed = time.monotonic() - t_start
    logger.info(
        "analysis %s cam%d done in %.1fs: analyzed=%d failed=%d (cuda=%.1fs qsv=%.1fs)",
        date, cam_index, total_elapsed, stats["analyzed"], stats["failed"],
        worker_durations.get("cuda", 0), worker_durations.get("qsv", 0),
    )
    return stats
