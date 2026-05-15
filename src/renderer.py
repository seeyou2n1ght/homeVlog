import logging
import subprocess
import time
import threading
from pathlib import Path

from src.utils import parse_res, OUTPUT_DIR, TEMP_DIR
from src.ffmpeg import run_ffmpeg
from src.timeline import TimelineSegment, build_concat_filter, build_timeline
from src.monitor import get_perf, PerfRecord

logger = logging.getLogger("homevlog")


class FFmpegProcessRegistry:
    _processes = {}
    _lock = threading.Lock()

    @classmethod
    def register(cls, key, proc):
        with cls._lock:
            cls._processes[key] = proc

    @classmethod
    def deregister(cls, key):
        with cls._lock:
            cls._processes.pop(key, None)

    @classmethod
    def kill_all(cls):
        with cls._lock:
            for k, p in cls._processes.items():
                try:
                    p.kill()
                except Exception:
                    pass
            cls._processes.clear()


def render_vlog(db, date: str, cam_index: int, encoder: str = "nv") -> str | None:
    """Entry point for rendering a full day vlog for one camera."""
    from src.utils import load_config
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

    # 获取所有 rows 用于 metadata 查找
    all_rows = db.get_all_file_tasks_for_date(date, cam_index)

    if hw_decode:
        return _batch_render(
            timeline, output_path, fps, width, height,
            seg_cfg, out_cfg, audio_cfg, date, cam_index, encoder, all_rows
        )

    # CPU decode path (single render)
    files, filter_complex = _prepare_render(
        timeline, fps, width, height, seg_cfg, all_rows
    )

    logger.info("render %s cam%d start: %d files, %d segments",
                date, cam_index, len(files), len(timeline))
    return _run_ffmpeg_render(
        files, filter_complex, output_path, encoder,
        fps, out_cfg, audio_cfg, date, cam_index,
    )


def _prepare_render(timeline, fps, width, height, seg_cfg, rows):
    """Build file list (reindexed) and filter_complex for CPU decode path."""
    files = list(dict.fromkeys(t.filepath for t in timeline))
    _reindex_timeline(timeline, files)

    scale_mode = _detect_scale_mode(files[0], width, height)

    filter_complex = build_concat_filter(
        timeline,
        rows=rows,
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


_DEFAULT_BATCH_MAX_FILES = 8


def _batch_render(timeline, output_path, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index, encoder="nv", rows=None):
    batch_max_files = out_cfg.get("batch_max_files", _DEFAULT_BATCH_MAX_FILES)
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

    if encoder == "nv" and n_batches >= 2:
        batch_paths = _render_batches_parallel(
            batches, output_path, fps, width, height,
            seg_cfg, out_cfg, audio_cfg, date, cam_index, rows
        )
    else:
        batch_paths = _render_batches_sequential(
            batches, output_path, fps, width, height,
            seg_cfg, out_cfg, audio_cfg, date, cam_index, encoder, rows
        )

    if batch_paths is None:
        return None

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


def build_batch_render(batch_segs, bi, enc_for_batch, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index, rows):
    from copy import deepcopy
    batch_copy = deepcopy(batch_segs)
    files = list(dict.fromkeys(s.filepath for s in batch_copy))
    _reindex_timeline(batch_copy, files)

    if enc_for_batch == "nv":
        scale_mode = "cuda_passthrough"
    elif enc_for_batch == "qsv":
        scale_mode = "qsv"
    else:
        scale_mode = _detect_scale_mode(files[0], width, height)

    fc = build_concat_filter(
        batch_copy,
        rows=rows,
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


def _render_batches_parallel(batches, output_path, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index, rows):
    from concurrent.futures import ThreadPoolExecutor
    from queue import Queue

    n_batches = len(batches)
    total_dur = sum(s.duration for b in batches for s in b)
    n_files = len(set(s.filepath for b in batches for s in b))
    render_timeout = max(7200, n_files * 120 + total_dur * 4)

    batch_queue: Queue[tuple[int, list] | None] = Queue()
    for i, b in enumerate(batches):
        batch_queue.put((i, b))
    batch_queue.put(None)
    batch_queue.put(None)

    results: dict[int, str | None] = {}
    results_lock = threading.Lock()
    worker_elapsed: dict[str, float] = {"nv": 0.0, "qsv": 0.0}
    worker_counts: dict[str, int] = {"nv": 0, "qsv": 0}

    def _worker(enc: str) -> None:
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
                    date, cam_index, rows
                )
                elapsed = time.monotonic() - t0
                with results_lock:
                    results[bi] = result
                    worker_counts[enc] += 1
                    worker_elapsed[enc] += elapsed
            except Exception as e:
                elapsed = time.monotonic() - t0
                logger.error("work-stealing batch %d [%s] error (%.0fs): %s", bi, enc, elapsed, e)
                with results_lock:
                    results[bi] = None

    with ThreadPoolExecutor(max_workers=2) as pool:
        f_nv = pool.submit(_worker, "nv")
        f_qsv = pool.submit(_worker, "qsv")
        try:
            f_nv.result(timeout=render_timeout)
            f_qsv.result(timeout=render_timeout)
        except Exception as e:
            logger.error("work-stealing render error: %s", e)
            return None

    batch_paths: list[Path] = []
    for i in range(n_batches):
        r = results.get(i)
        if r is None:
            return None
        batch_paths.append(Path(r))
    return batch_paths


def _render_batches_sequential(batches, output_path, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index, encoder="nv", rows=None):
    batch_paths: list[Path] = []
    for bi, batch_segs in enumerate(batches):
        result = build_batch_render(
            batch_segs, bi, encoder,
            fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index, rows
        )
        if result is None:
            return None
        batch_paths.append(Path(result))
    return batch_paths


def _run_batch_render(input_files, filter_complex, output_path, encoder, fps, out_cfg, audio_cfg, date, cam_index, batch_idx=0) -> str | None:
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
            proc.kill()
            return None
        finally:
            FFmpegProcessRegistry.deregister(str(output_path))
    finally:
        io_sem.release()
        
    elapsed = time.monotonic() - t0
    fc_script.unlink(missing_ok=True)

    if proc.returncode == 0:
        return str(output_path)
    else:
        logger.error("batch-render cam%d batch%d failed:\n%s", cam_index, batch_idx, stderr.decode("utf-8", errors="replace")[-1000:])
        return None


def concat_output_files(files: list[Path], output: Path, timeout: float = 300) -> bool:
    concat_list = output.with_name(f".concat_{output.stem}.txt")
    def _concat_path(path: Path) -> str:
        return str(path).replace(chr(92), "/").replace("'", r"\'")
    lines = [f"file '{_concat_path(f)}'" for f in files]
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", "-movflags", "+faststart", str(output)], timeout=timeout)
    concat_list.unlink(missing_ok=True)
    return result.returncode == 0


def _run_ffmpeg_render(input_files, filter_complex, output_path, encoder, fps, out_cfg, audio_cfg, date, cam_index) -> str | None:
    input_args = []
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
        t0 = time.monotonic()
        try:
            _, stderr = proc.communicate(timeout=7200)
        except Exception:
            proc.kill()
            return None
    finally:
        io_sem.release()
    if proc.returncode == 0:
        fc_script.unlink(missing_ok=True)
        return str(output_path)
    return None


def _build_enc_args(encoder, out_cfg):
    if encoder == "qsv":
        qsv = out_cfg.get("qsv", {})
        return ["-c:v", qsv.get("codec", "hevc_qsv"), "-preset", qsv.get("preset", "fast"), "-global_quality", str(qsv.get("global_quality", 28)), "-maxrate", qsv.get("maxrate", "4M"), "-bufsize", qsv.get("bufsize", "8M"), "-g", "120", "-pix_fmt", "nv12"]
    else:
        nv = out_cfg.get("nv", {})
        return ["-c:v", nv.get("codec", "hevc_nvenc"), "-preset", nv.get("preset", "p1"), "-cq", str(nv.get("cq", 28)), "-maxrate", nv.get("maxrate", "4M"), "-bufsize", nv.get("bufsize", "8M"), "-g", "120", "-pix_fmt", "yuv420p"]
