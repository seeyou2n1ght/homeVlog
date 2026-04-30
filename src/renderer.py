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

_BATCH_MAX_FILES = 15  # NVDEC+NVENC limit: >15 inputs can OOM 8GB VRAM


def _batch_render(timeline, output_path, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index, encoder="nv"):
    """Split timeline into batches by file count, render each with HW decode, concat.

    When encoder="nv" and there are multiple batches, pairs consecutive batches
    to run NVDEC+NVENC and QSV decode+encode in parallel, utilizing both GPUs.
    """
    # Group segments into batches of ≤ _BATCH_MAX_FILES unique files
    batches: list[list[TimelineSegment]] = []
    cur_batch: list[TimelineSegment] = []
    cur_files: set[str] = set()
    for seg in timeline:
        if seg.filepath not in cur_files and len(cur_files) >= _BATCH_MAX_FILES and cur_batch:
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
                date, cam_index, total_files, len(timeline), n_batches, _BATCH_MAX_FILES)

    # Parallel NVENC+QSV: pair consecutive batches to run on different GPUs
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
        ok = _concat_files(batch_paths, output_path)
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


def _build_batch(batch_segs, bi, enc_for_batch, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index):
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
        gap_tolerance=seg_cfg.get("gap_tolerance", 0.5),
        scale_mode=scale_mode,
    )

    batch_dur = sum(s.duration for s in batch_segs)
    batch_path = TEMP_DIR / f"_batch{bi}_{date}_cam{cam_index}.mp4"
    logger.info("batch %d [%s]: %d files, %.0fs video",
                bi, enc_for_batch, len(files), batch_dur)

    result = _run_batch_render(
        files, fc, batch_path, enc_for_batch, fps, out_cfg, audio_cfg,
        date, cam_index, batch_idx=bi,
    )
    return result


def _render_batches_parallel(batches, output_path, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index):
    """Pair consecutive batches: NVDEC+NVENC + QSV decode+encode in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    batch_paths: list[Path] = []
    n_batches = len(batches)
    n_pairs = n_batches // 2

    total_dur = sum(s.duration for b in batches for s in b)
    n_files = len(set(s.filepath for b in batches for s in b))
    render_timeout = max(7200, n_files * 120 + total_dur * 4)

    logger.info("parallel-render %s cam%d: %d batches -> %d pairs + %d tail, timeout=%.0fs",
                date, cam_index, n_batches, n_pairs, n_batches % 2, render_timeout)

    for pair_i in range(n_pairs):
        bi_nv = pair_i * 2
        bi_qsv = pair_i * 2 + 1

        with ThreadPoolExecutor(max_workers=2) as pool:
            f_nv = pool.submit(
                _build_batch, batches[bi_nv], bi_nv, "nv",
                fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index,
            )
            f_qsv = pool.submit(
                _build_batch, batches[bi_qsv], bi_qsv, "qsv",
                fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index,
            )

            try:
                results = {}
                for f in as_completed([f_nv, f_qsv], timeout=render_timeout):
                    results[f] = f.result()
            except TimeoutError:
                logger.error("parallel-render pair %d timeout for %s cam%d (%.0fs limit)",
                             pair_i, date, cam_index, render_timeout)
                FFmpegProcessRegistry.kill_all()
                time.sleep(1)
                for p in batch_paths:
                    p.unlink(missing_ok=True)
                return None
            except Exception as e:
                logger.error("parallel-render pair %d error for %s cam%d: %s",
                             pair_i, date, cam_index, e)
                FFmpegProcessRegistry.kill_all()
                time.sleep(1)
                for p in batch_paths:
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass
                return None

        r_nv = results.get(f_nv)
        r_qsv = results.get(f_qsv)
        if r_nv and r_qsv:
            batch_paths.append(Path(r_nv))
            batch_paths.append(Path(r_qsv))
        else:
            logger.error("parallel-render pair %d failed: nv=%s qsv=%s",
                         pair_i, "ok" if r_nv else "FAIL", "ok" if r_qsv else "FAIL")
            for p in batch_paths:
                p.unlink(missing_ok=True)
            return None

    # Tail batch (odd count): sequential on default encoder
    if n_batches % 2 == 1:
        bi = n_batches - 1
        result = _build_batch(
            batches[bi], bi, "nv",
            fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index,
        )
        if result is None:
            logger.error("parallel-render tail batch %d failed", bi)
            for p in batch_paths:
                p.unlink(missing_ok=True)
            return None
        batch_paths.append(Path(result))

    return batch_paths


def _render_batches_sequential(batches, output_path, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index, encoder="nv"):
    """Render batches one at a time (fallback for single-batch or non-nv encoder)."""
    batch_paths: list[Path] = []
    for bi, batch_segs in enumerate(batches):
        result = _build_batch(
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
    cmd += ["-filter_complex_script", str(fc_script), "-map", "[v]", "-map", "[a]", "-r", str(fps)]
    cmd += enc_args
    cmd += ["-c:a", audio_codec, "-b:a", audio_bitrate, "-ac", str(audio_channels)]
    cmd += [str(output_path)]

    logger.info("batch-render cam%d batch%d (%s): %d files", cam_index, batch_idx, encoder, len(input_files))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    FFmpegProcessRegistry.register(str(output_path), proc)
    t0 = time.monotonic()
    try:
        _, stderr = proc.communicate()
    finally:
        FFmpegProcessRegistry.deregister(str(output_path))
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
        err = stderr.decode("utf-8", errors="replace")[-2000:]
        logger.error("batch-render cam%d batch%d failed after %.1fs: %s",
                     cam_index, batch_idx, elapsed, err[-500:])
        perf.add(PerfRecord(
            stage="render", file=str(output_path), gpu=encoder,
            duration=round(elapsed, 2), extra={"batch": batch_idx, "status": "FAILED"},
        ))
        return None


def _concat_files(files: list[Path], output: Path, timeout: float = 300) -> bool:
    """Losslessly concatenate multiple MP4 files via concat demuxer."""
    concat_list = output.with_name(f".concat_{output.stem}.txt")
    lines = [f"file {str(f).replace(chr(92), '/')}" for f in files]
    concat_list.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = run_ffmpeg(
        ["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy", str(output)],
        timeout=timeout,
    )
    concat_list.unlink(missing_ok=True)

    if result.returncode != 0:
        logger.error("concat %d files failed (%.1fs): %s", len(files), result.duration, result.stderr_text[-300:])
        return False
    logger.info("concat %d files done in %.1fs", len(files), result.duration)
    return True


def _run_ffmpeg_render(
    input_files, filter_complex, output_path, encoder,
    fps, out_cfg, audio_cfg, date, cam_index,
) -> str | None:
    """Execute ffmpeg render for CPU decode path."""
    input_args: list[str] = []
    for fp in input_files:
        input_args += ["-i", str(fp)]

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
    cmd += ["-filter_complex_script", str(fc_script), "-map", "[v]", "-map", "[a]", "-r", str(fps)]

    cmd += enc_args
    cmd += ["-c:a", audio_codec, "-b:a", audio_bitrate, "-ac", str(audio_channels)]
    cmd += [str(output_path)]

    logger.info("render %s cam%d (%s): %d files", date, cam_index, encoder, len(input_files))

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    FFmpegProcessRegistry.register(str(output_path), proc)
    t0 = time.monotonic()
    try:
        _, stderr = proc.communicate()
    finally:
        FFmpegProcessRegistry.deregister(str(output_path))
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
        err_tail = err_full[-3000:] if len(err_full) > 3000 else err_full
        logger.error("render %s cam%d failed after %.1fs: %s",
                     date, cam_index, elapsed, err_tail[-500:])
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
            "-pix_fmt", "nv12",
        ]
    else:
        nv = out_cfg.get("nv", {})
        return [
            "-c:v", nv.get("codec", "hevc_nvenc"),
            "-preset", nv.get("preset", "p1"),
            "-cq", str(nv.get("cq", 28)),
            "-maxrate", nv.get("maxrate", "4M"),
            "-pix_fmt", "yuv420p",
        ]
