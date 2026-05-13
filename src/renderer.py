import logging
import subprocess
import threading
import time
from pathlib import Path

from src.database import VlogDatabase
from src.timeline import TimelineSegment, build_timeline, build_concat_filter
from src.ffmpeg import run_ffmpeg
from src.utils import load_config, OUTPUT_DIR, TEMP_DIR, parse_res
from src.monitor import get_perf, PerfRecord

logger = logging.getLogger("homevlog")


class FFmpegProcessRegistry:
    """Track live ffmpeg subprocesses so callers can kill them on timeout."""
    _lock = threading.Lock()
    _procs: dict[str, subprocess.Popen] = {}

    @classmethod
    def register(cls, key: str, proc: subprocess.Popen):
        with cls._lock:
            cls._procs[key] = proc

    @classmethod
    def deregister(cls, key: str):
        with cls._lock:
            cls._procs.pop(key, None)

    @classmethod
    def kill_all(cls):
        with cls._lock:
            for key, proc in cls._procs.items():
                try:
                    proc.kill()
                except OSError:
                    pass
            cls._procs.clear()


def render_date_cam(
    db: VlogDatabase,
    date: str,
    cam_index: int,
    encoder: str = "nv",
) -> str | None:
    """Render a single date-cam vlog."""
    config = load_config()
    out_cfg = config.get("output", {})
    audio_cfg = out_cfg.get("audio", {})
    pass2_cfg = config.get("pass2", {})
    hw_decode = pass2_cfg.get("hw_decode", True)

    timeline = build_timeline(db, date, cam_index)
    if not timeline:
        logger.warning("empty timeline for %s cam%d", date, cam_index)
        return None

    output_name = out_cfg.get("naming", "DailyVlog_{date}_cam{index}.mp4")
    output_name = output_name.replace("{date}", date).replace("{index}", str(cam_index))
    output_path = OUTPUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = out_cfg.get("fps", 20)
    width, height = parse_res(out_cfg.get("resolution", "1920x1080"))
    seg_cfg = config.get("segment", {})

    if hw_decode:
        return _batch_render(
            timeline, output_path, fps, width, height,
            seg_cfg, out_cfg, audio_cfg, date, cam_index, encoder,
        )

    # CPU decode path (single render)
    files, filter_complex = _prepare_render(
        timeline, fps, width, height, seg_cfg,
    )

    logger.info("render %s cam%d start: %d files, %d segments",
                date, cam_index, len(files), len(timeline))
    return _run_ffmpeg_render(
        files, filter_complex, output_path, encoder,
        fps, out_cfg, audio_cfg, date, cam_index,
    )


def _prepare_render(timeline, fps, width, height, seg_cfg):
    """Build file list (reindexed) and filter_complex for CPU decode path."""
    files = list(dict.fromkeys(t.filepath for t in timeline))
    _reindex_timeline(timeline, files)

    scale_mode = _detect_scale_mode(files[0], width, height)

    filter_complex = build_concat_filter(
        timeline,
        output_fps=fps,
        output_width=width,
        output_height=height,
        static_keyframe_interval=seg_cfg.get("static_keyframe_interval", 30.0),
        keyframe_display_duration=seg_cfg.get("keyframe_display_duration", 0.5),
        min_static_display_duration=seg_cfg.get("min_static_display_duration", 1.5),
        gap_tolerance=seg_cfg.get("gap_tolerance", 0.5),
        scale_mode=scale_mode,
    )
    return files, filter_complex


def _reindex_timeline(timeline, files):
    """Rebuild input_index for a timeline subset based on its file list."""
    mapping = {f: i for i, f in enumerate(files)}
    for t in timeline:
        t.input_index = mapping[t.filepath]


def _detect_scale_mode(first_file: str, out_w: int, out_h: int) -> str:
    """Check first input resolution; skip scale if already matches output."""
    from src.ffmpeg import run_ffprobe
    info = run_ffprobe(first_file)
    if info:
        for s in info.get("streams", []):
            if s.get("codec_type") == "video":
                if s.get("width") == out_w and s.get("height") == out_h:
                    return "skip"
    return "cpu"


# ---------------------------------------------------------------------------
# Batch render with HW decode (NVDEC / QSV hwaccel)
# ---------------------------------------------------------------------------

_DEFAULT_BATCH_MAX_FILES = 8  # 降低 Filter Complex 复杂度，提升显存周转率


def _batch_render(timeline, output_path, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index, encoder="nv"):
    """Split timeline into batches by file count, render each with HW decode, concat.

    When encoder="nv" and there are multiple batches, uses work-stealing mode:
    NV/QSV workers independently pull batches from a shared queue.
    """
    # batch_max_files: 可通过配置调整以适应不同 VRAM 大小
    batch_max_files = out_cfg.get("batch_max_files", _DEFAULT_BATCH_MAX_FILES)
    # Group segments into batches of ≤ batch_max_files unique files
    batches: list[list[TimelineSegment]] = []
    cur_batch: list[TimelineSegment] = []
    cur_files: set[str] = set()
    for seg in timeline:
        if seg.filepath not in cur_files and len(cur_files) >= batch_max_files and cur_batch:
            batches.append(cur_batch)
            cur_batch = []
            cur_files = set()
        cur_batch.append(seg)
        cur_files.add(seg.filepath)
    if cur_batch:
        batches.append(cur_batch)

    n_batches = len(batches)
    total_files = len(set(t.filepath for t in timeline))
    logger.info("batch-render %s cam%d: %d files, %d segs -> %d batches (max %d files/batch)",
                date, cam_index, total_files, len(timeline), n_batches, batch_max_files)

    # Work-stealing: NV/QSV 双 worker 独立消费 batch 队列
    if encoder == "nv" and n_batches >= 2:
        batch_paths = _render_batches_parallel(
            batches, output_path, fps, width, height,
            seg_cfg, out_cfg, audio_cfg, date, cam_index,
        )
    else:
        batch_paths = _render_batches_sequential(
            batches, output_path, fps, width, height,
            seg_cfg, out_cfg, audio_cfg, date, cam_index, encoder,
        )

    if batch_paths is None:
        return None

    # Concat all batches
    if len(batch_paths) == 1:
        batch_paths[0].rename(output_path)
    else:
        ok = concat_output_files(batch_paths, output_path)
        for p in batch_paths:
            try:
                p.unlink(missing_ok=True)
            except OSError as e:
                logger.warning("cleanup batch file failed: %s (%s)", p.name, e)
        if not ok:
            logger.error("batch-render concat failed for %s cam%d", date, cam_index)
            return None

    if not output_path.exists():
        logger.error("batch-render %s cam%d: output file missing after concat", date, cam_index)
        return None
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info("batch-render %s cam%d done: %s (%.1f MB)", date, cam_index, output_path.name, file_size_mb)
    return str(output_path)


def build_batch_render(batch_segs, bi, enc_for_batch, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index):
    """Prepare a single batch: deepcopy, reindex, build filter graph, render."""
    from copy import deepcopy
    batch_copy = deepcopy(batch_segs)
    files = list(dict.fromkeys(s.filepath for s in batch_copy))
    _reindex_timeline(batch_copy, files)

    # Scale mode depends on decode path:
    #   "nv"  (NVDEC) → cuda_passthrough (scale_cuda, no hwupload)
    #   "qsv" (QSV hwaccel) → qsv (scale_qsv)
    #   fallback → auto-detect based on input resolution
    if enc_for_batch == "nv":
        scale_mode = "cuda_passthrough"
    elif enc_for_batch == "qsv":
        scale_mode = "qsv"
    else:
        scale_mode = _detect_scale_mode(files[0], width, height)

    fc = build_concat_filter(
        batch_copy,
        output_fps=fps, output_width=width, output_height=height,
        static_keyframe_interval=seg_cfg.get("static_keyframe_interval", 30.0),
        keyframe_display_duration=seg_cfg.get("keyframe_display_duration", 0.5),
        min_static_display_duration=seg_cfg.get("min_static_display_duration", 1.5),
        gap_tolerance=seg_cfg.get("gap_tolerance", 0.5),
        scale_mode=scale_mode,
    )

    batch_path = TEMP_DIR / f"_batch{bi}_{date}_cam{cam_index}.mp4"

    result = _run_batch_render(
        files, fc, batch_path, enc_for_batch, fps, out_cfg, audio_cfg,
        date, cam_index, batch_idx=bi,
    )
    return result


def _render_batches_parallel(batches, output_path, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index):
    """Work-stealing: NV/QSV 双 worker 从共享队列抢 batch，先完成先取。

    使用哨兵值终止 worker。NV 天然更快，自然消费更多 batch。
    """
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue

    n_batches = len(batches)
    total_dur = sum(s.duration for b in batches for s in b)
    n_files = len(set(s.filepath for b in batches for s in b))
    render_timeout = max(7200, n_files * 120 + total_dur * 4)

    # 构建 batch 队列（带序号，保证输出顺序）
    batch_queue: Queue[tuple[int, list] | None] = Queue()
    for i, b in enumerate(batches):
        batch_queue.put((i, b))
    # 放入 2 个哨兵终止 2 个 worker
    batch_queue.put(None)
    batch_queue.put(None)

    # 按 batch 序号收集结果
    results: dict[int, str | None] = {}
    results_lock = threading.Lock()
    worker_elapsed: dict[str, float] = {"nv": 0.0, "qsv": 0.0}
    worker_counts: dict[str, int] = {"nv": 0, "qsv": 0}

    def _worker(enc: str) -> None:
        """单个 GPU worker: 循环从队列取 batch 并渲染。"""
        while True:
            item = batch_queue.get()
            if item is None:
                break
            bi, batch_segs = item
            t0 = time.monotonic()
            try:
                result = build_batch_render(
                    batch_segs, bi, enc,
                    fps, width, height, seg_cfg, out_cfg, audio_cfg,
                    date, cam_index,
                )
                elapsed = time.monotonic() - t0
                with results_lock:
                    results[bi] = result
                    worker_counts[enc] += 1
                    worker_elapsed[enc] += elapsed
                if result is None:
                    logger.error("work-stealing batch %d [%s] failed (%.0fs)", bi, enc, elapsed)
            except Exception as e:
                elapsed = time.monotonic() - t0
                logger.error("work-stealing batch %d [%s] error (%.0fs): %s", bi, enc, elapsed, e)
                with results_lock:
                    results[bi] = None
                    worker_counts[enc] += 1
                    worker_elapsed[enc] += elapsed

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_nv = pool.submit(_worker, "nv")
        f_qsv = pool.submit(_worker, "qsv")

        try:
            f_nv.result(timeout=render_timeout)
            f_qsv.result(timeout=render_timeout)
        except TimeoutError:
            logger.error(
                "work-stealing render timeout for %s cam%d (%.0fs limit)",
                date, cam_index, render_timeout,
            )
            FFmpegProcessRegistry.kill_all()
            time.sleep(1)
            return None
        except Exception as e:
            logger.error(
                "work-stealing render error for %s cam%d: %s",
                date, cam_index, e,
            )
            FFmpegProcessRegistry.kill_all()
            time.sleep(1)
            return None

    logger.info(
        "work-stealing done: nv=%d batches (%.0fs), qsv=%d batches (%.0fs)",
        worker_counts["nv"], worker_elapsed["nv"],
        worker_counts["qsv"], worker_elapsed["qsv"],
    )

    # 按序号组装结果，保证 concat 顺序
    batch_paths: list[Path] = []
    for i in range(n_batches):
        r = results.get(i)
        if r is None:
            logger.error("work-stealing: batch %d missing or failed", i)
            for p in batch_paths:
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            return None
        batch_paths.append(Path(r))

    return batch_paths


def _render_batches_sequential(batches, output_path, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index, encoder="nv"):
    """Render batches one at a time (fallback for single-batch or non-nv encoder)."""
    batch_paths: list[Path] = []
    for bi, batch_segs in enumerate(batches):
        result = build_batch_render(
            batch_segs, bi, encoder,
            fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index,
        )
        if result is None:
            logger.error("batch %d/%d failed, aborting render", bi + 1, len(batches))
            for p in batch_paths:
                p.unlink(missing_ok=True)
            return None
        batch_paths.append(Path(result))
    return batch_paths


def _run_batch_render(
    input_files, filter_complex, output_path, encoder,
    fps, out_cfg, audio_cfg, date, cam_index, batch_idx=0,
) -> str | None:
    """Execute ffmpeg render with HW decode (NVDEC or QSV hwaccel) per input."""
    hwaccel_args = []
    for fp in input_files:
        hwaccel_args += ["-fflags", "+genpts"]
        if encoder == "nv":
            hwaccel_args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif encoder == "qsv":
            hwaccel_args += ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
        hwaccel_args += ["-i", str(fp)]

    enc_args = _build_enc_args(encoder, out_cfg)
    audio_codec = audio_cfg.get("codec", "aac")
    audio_bitrate = audio_cfg.get("bitrate", "96k")
    audio_channels = audio_cfg.get("channels", 1)

    cmd = ["ffmpeg", "-hide_banner", "-y"]
    if encoder == "nv":
        cmd += ["-init_hw_device", "cuda=gpu:0"]
    elif encoder == "qsv":
        cmd += ["-init_hw_device", "qsv=qsv"]
    cmd += hwaccel_args

    fc_script = TEMP_DIR / f"_fc_batch{batch_idx}_{date}_cam{cam_index}.txt"
    fc_script.parent.mkdir(parents=True, exist_ok=True)
    fc_script.write_text(filter_complex, encoding="utf-8")
    if encoder == "nv":
        cmd += ["-filter_hw_device", "gpu"]
    cmd += ["-/filter_complex", str(fc_script), "-map", "[v]", "-map", "[a]", "-r", str(fps)]
    cmd += enc_args
    cmd += ["-c:a", audio_codec, "-b:a", audio_bitrate, "-ac", str(audio_channels)]
    cmd += ["-tag:v", "hvc1", "-movflags", "+faststart"]
    cmd += [str(output_path)]

    logger.info("batch-render cam%d batch%d (%s): %d files", cam_index, batch_idx, encoder, len(input_files))

    if encoder == "qsv":
        from src.utils import get_qsv_semaphore
        io_sem = get_qsv_semaphore()
    else:
        from src.utils import get_nv_semaphore
        io_sem = get_nv_semaphore()
    io_sem.acquire()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        FFmpegProcessRegistry.register(str(output_path), proc)
        t0 = time.monotonic()
        # timeout = max(7200, 所有文件总时长 * 4), 防止 ffmpeg 挂死时主线程永久阻塞
        render_timeout = max(7200, len(input_files) * 600)
        try:
            _, stderr = proc.communicate(timeout=render_timeout)
        except subprocess.TimeoutExpired:
            logger.error("batch-render cam%d batch%d communicate timeout (%ds)",
                         cam_index, batch_idx, render_timeout)
            proc.kill()
            proc.wait(timeout=10)
            return None
        finally:
            FFmpegProcessRegistry.deregister(str(output_path))
    finally:
        io_sem.release()
        
    elapsed = time.monotonic() - t0

    fc_script.unlink(missing_ok=True)
    perf = get_perf()

    if proc.returncode == 0:
        file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
        logger.info("batch-render cam%d batch%d done in %.1fs (%.1f MB)",
                    cam_index, batch_idx, elapsed, file_size_mb)
        perf.add(PerfRecord(
            stage="render", file=Path(output_path).name, gpu=encoder,
            duration=round(elapsed, 2),
            extra={"batch": batch_idx, "output_mb": round(file_size_mb, 1), "n_inputs": len(input_files)},
        ))
        return str(output_path)
    else:
        err_full = stderr.decode("utf-8", errors="replace")
        # 保留头尾信息以便定位根因（开头通常是核心错误，尾部是最后状态）
        if len(err_full) > 1000:
            err_display = err_full[:500] + "\n... [truncated] ...\n" + err_full[-500:]
        else:
            err_display = err_full
        logger.error("batch-render cam%d batch%d failed after %.1fs:\n%s",
                     cam_index, batch_idx, elapsed, err_display)
        perf.add(PerfRecord(
            stage="render", file=str(output_path), gpu=encoder,
            duration=round(elapsed, 2), extra={"batch": batch_idx, "status": "FAILED"},
        ))
        return None


def concat_output_files(files: list[Path], output: Path, timeout: float = 300) -> bool:
    """Losslessly concatenate multiple MP4 files via concat demuxer."""
    concat_list = output.with_name(f".concat_{output.stem}.txt")

    def _concat_path(path: Path) -> str:
        normalized = str(path).replace(chr(92), "/")
        return normalized.replace("'", r"\'")

    lines = [f"file '{_concat_path(f)}'" for f in files]
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", "-movflags", "+faststart", str(output)],
        timeout=timeout,
    )
    concat_list.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error("concat %d files failed (%.1fs): %s", len(files), result.duration, result.stderr_text[-300:])
        return False

    return True


def _run_ffmpeg_render(
    input_files, filter_complex, output_path, encoder,
    fps, out_cfg, audio_cfg, date, cam_index,
) -> str | None:
    """Execute ffmpeg render for CPU decode path."""
    input_args: list[str] = []
    for fp in input_files:
        input_args += ["-fflags", "+genpts", "-i", str(fp)]

    enc_args = _build_enc_args(encoder, out_cfg)
    audio_codec = audio_cfg.get("codec", "aac")
    audio_bitrate = audio_cfg.get("bitrate", "96k")
    audio_channels = audio_cfg.get("channels", 1)

    cmd = ["ffmpeg", "-hide_banner", "-y"]
    if encoder == "nv":
        cmd += ["-init_hw_device", "cuda=gpu:0"]
    elif encoder == "qsv":
        cmd += ["-init_hw_device", "qsv=qsv"]
    cmd += input_args

    fc_script = TEMP_DIR / f"_fc_{date}_cam{cam_index}.txt"
    fc_script.parent.mkdir(parents=True, exist_ok=True)
    fc_script.write_text(filter_complex, encoding="utf-8")
    if encoder == "nv":
        cmd += ["-filter_hw_device", "gpu"]
    cmd += ["-/filter_complex", str(fc_script), "-map", "[v]", "-map", "[a]", "-r", str(fps)]

    cmd += enc_args
    cmd += ["-c:a", audio_codec, "-b:a", audio_bitrate, "-ac", str(audio_channels)]
    cmd += ["-tag:v", "hvc1", "-movflags", "+faststart"]
    cmd += [str(output_path)]

    if encoder == "qsv":
        from src.utils import get_qsv_semaphore
        io_sem = get_qsv_semaphore()
    else:
        from src.utils import get_nv_semaphore
        io_sem = get_nv_semaphore()
    io_sem.acquire()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        FFmpegProcessRegistry.register(str(output_path), proc)
        t0 = time.monotonic()
        render_timeout = max(7200, len(input_files) * 600)
        try:
            _, stderr = proc.communicate(timeout=render_timeout)
        except subprocess.TimeoutExpired:
            logger.error("render %s cam%d communicate timeout (%ds)",
                         date, cam_index, render_timeout)
            proc.kill()
            proc.wait(timeout=10)
            fc_script.unlink(missing_ok=True)
            return None
        finally:
            FFmpegProcessRegistry.deregister(str(output_path))
    finally:
        io_sem.release()
        
    elapsed = time.monotonic() - t0

    perf = get_perf()

    if proc.returncode == 0:
        fc_script.unlink(missing_ok=True)
        file_size_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0
        logger.info("render %s cam%d done in %.1fs: %s (%.1f MB)",
                    date, cam_index, elapsed, output_path.name, file_size_mb)
        perf.add(PerfRecord(
            stage="render", file=Path(output_path).name, gpu=encoder,
            duration=round(elapsed, 2),
            extra={"output_mb": round(file_size_mb, 1), "n_inputs": len(input_files)},
        ))
        return str(output_path)
    else:
        err_full = stderr.decode("utf-8", errors="replace")
        if len(err_full) > 1000:
            err_display = err_full[:500] + "\n... [truncated] ...\n" + err_full[-500:]
        else:
            err_display = err_full
        logger.error("render %s cam%d failed after %.1fs:\n%s",
                     date, cam_index, elapsed, err_display)
        perf.add(PerfRecord(
            stage="render", file=str(output_path), gpu=encoder,
            duration=round(elapsed, 2), extra={"status": "FAILED"},
        ))
        return None


def _build_enc_args(encoder, out_cfg):
    if encoder == "qsv":
        qsv = out_cfg.get("qsv", {})
        return [
            "-c:v", qsv.get("codec", "hevc_qsv"),
            "-preset", qsv.get("preset", "fast"),
            "-global_quality", str(qsv.get("global_quality", 28)),
            "-maxrate", qsv.get("maxrate", "4M"),
            "-bufsize", qsv.get("bufsize", "8M"),
            "-g", "120",
            "-pix_fmt", "nv12",
        ]
    else:
        nv = out_cfg.get("nv", {})
        return [
            "-c:v", nv.get("codec", "hevc_nvenc"),
            "-preset", nv.get("preset", "p1"),
            "-cq", str(nv.get("cq", 28)),
            "-maxrate", nv.get("maxrate", "4M"),
            "-bufsize", nv.get("bufsize", "8M"),
            "-g", "120",
            "-pix_fmt", "yuv420p",
        ]
