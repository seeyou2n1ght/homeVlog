import logging
import re
import subprocess
import time
import threading
from pathlib import Path

from src.utils import TEMP_DIR
from src.scheduler import acquire_with_retry
from src.ffmpeg import run_ffmpeg
from src.timeline import build_concat_filter
from src.monitor import get_perf, PerfRecord

logger = logging.getLogger("homevlog")

_FFMPEG_PROGRESS_RE = re.compile(
    r"frame=\s*(\d+)\s+fps=\s*([\d.]+).*?speed=\s*([\d.]+)x"
)


def _parse_ffmpeg_progress(err_log: Path) -> dict:
    """从 ffmpeg stderr 日志提取最后一帧进度（编码帧数/fps/倍速）。"""
    try:
        tail = err_log.read_bytes()[-4000:].decode("utf-8", errors="replace")
    except OSError:
        return {}
    matches = _FFMPEG_PROGRESS_RE.findall(tail.replace("\r", "\n"))
    if not matches:
        return {}
    frames, fps, speed = matches[-1]
    return {
        "enc_frames": int(frames),
        "enc_fps": float(fps),
        "encode_speed_x": float(speed),
    }


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


def _reindex_timeline(timeline, files):
    """Rebuild input_index for a timeline subset based on its file list."""
    mapping = {f: i for i, f in enumerate(files)}
    for t in timeline:
        t.input_index = mapping[t.filepath]


def build_batch_render(batch_segs, bi, enc_for_batch, fps, width, height, seg_cfg, out_cfg, audio_cfg, date, cam_index, rows):
    from copy import deepcopy
    from src.utils import load_config
    cfg = {}
    try:
        cfg = load_config()
    except Exception:
        pass
    render_cfg = cfg.get("render", {})

    batch_copy = deepcopy(batch_segs)
    files = list(dict.fromkeys(s.filepath for s in batch_copy))
    _reindex_timeline(batch_copy, files)

    if enc_for_batch == "nv":
        scale_mode = "cuda_passthrough"
    elif enc_for_batch == "qsv":
        scale_mode = "qsv"
    else:
        scale_mode = "cpu"

    fc = build_concat_filter(
        batch_copy,
        rows=rows,
        output_fps=fps, output_width=width, output_height=height,
        static_keyframe_interval=seg_cfg.get("static_keyframe_interval", 30.0),
        keyframe_display_duration=seg_cfg.get("keyframe_display_duration", 0.5),
        min_static_display_duration=seg_cfg.get("min_static_display_duration", 1.5),
        gap_tolerance=seg_cfg.get("gap_tolerance", 0.5),
        scale_mode=scale_mode,
        speed_ramping=render_cfg.get("speed_ramping_enabled", seg_cfg.get("speed_ramping_enabled", True)),
        ramp_duration_s=render_cfg.get("ramp_duration_s", seg_cfg.get("ramp_duration_s", 1.0)),
        audio_fade_duration_s=render_cfg.get("audio_fade_duration_s", seg_cfg.get("audio_fade_duration_s", 0.15)),
        timecode_osd=render_cfg.get("timecode_osd_enabled", False),
        base_date=date,
    )

    batch_path = TEMP_DIR / f"_batch{bi}_{date}_cam{cam_index}.mp4"

    # 识别批次内纯静态文件（没有任何动态段），在解复用阶段跳过所有非关键帧解码
    files_with_dynamic = {s.filepath for s in batch_copy if s.state in ("DYNAMIC", "DYNAMIC_AUDIO")}
    pure_static_files = {f for f in files if f not in files_with_dynamic}

    result = _run_batch_render(
        files, fc, batch_path, enc_for_batch, fps, out_cfg, audio_cfg,
        date, cam_index, batch_idx=bi, pure_static_files=pure_static_files,
    )
    return result


def _run_batch_render(input_files, filter_complex, output_path, encoder, fps, out_cfg, audio_cfg, date, cam_index, batch_idx=0, pure_static_files=None) -> str | None:
    output_path = Path(output_path)

    # 断点续渲：已存在的完整批次产物直接复用
    try:
        if output_path.exists() and output_path.stat().st_size >= 512 * 1024:
            logger.info(
                "batch-render cam%d batch%d reusing existing output %s",
                cam_index, batch_idx, output_path.name,
            )
            return str(output_path)
    except OSError as e:
        logger.warning("batch-render reuse check failed for %s: %s", output_path.name, e)

    from src.utils import load_config
    cfg = {}
    try:
        cfg = load_config()
    except Exception:
        pass
    render_cfg = cfg.get("render", {})

    hwaccel_args = []
    for fp in input_files:
        hwaccel_args += ["-fflags", "+genpts"]
        # 对纯静态长文件跳过所有非关键帧解码，直接消除 batch_0 835s NVDEC 解码长尾
        if pure_static_files and str(fp) in pure_static_files:
            hwaccel_args += ["-skip_frame", "nokey"]
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
    tmp_output_path = output_path.with_name(f"{output_path.stem}.tmp.mp4")
    tmp_output_path.unlink(missing_ok=True)
    cmd += ["-c:a", audio_codec, "-b:a", audio_bitrate, "-ac", str(audio_channels)]
    cmd += ["-tag:v", "hvc1", "-movflags", "+faststart"]
    cmd += [str(tmp_output_path)]

    if encoder == "qsv":
        from src.utils import get_qsv_semaphore
        io_sem = get_qsv_semaphore()
    else:
        from src.utils import get_nv_semaphore
        io_sem = get_nv_semaphore()
    err_log = TEMP_DIR / f"_stderr_batch{batch_idx}_{date}_cam{cam_index}.log"

    # AGENTS.md 铁律：acquire 必须带 timeout 并重试，禁止无限阻塞
    # 渲染是产出关键路径，预算放宽至 30s×10：等待解码租约释放远优于批次 abort
    if not acquire_with_retry(io_sem, timeout=30.0, retries=10):
        logger.warning(
            "batch-render cam%d batch%d: io semaphore acquire timeout, aborting",
            cam_index, batch_idx,
        )
        fc_script.unlink(missing_ok=True)
        return None

    proc: subprocess.Popen | None = None
    t0 = time.monotonic()
    try:
        with open(err_log, "wb") as f_err:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=f_err)
            FFmpegProcessRegistry.register(str(tmp_output_path), proc)
            if encoder == "qsv":
                render_timeout = render_cfg.get("qsv_timeout_s", 360)
            else:
                render_timeout = max(7200, len(input_files) * 600)
            try:
                proc.wait(timeout=render_timeout)
            except subprocess.TimeoutExpired:
                logger.warning("Render timeout expired on %s for %s, killing process", encoder, tmp_output_path)
                proc.kill()
                try:
                    proc.wait(timeout=10)
                except (OSError, subprocess.TimeoutExpired):
                    pass
                tmp_output_path.unlink(missing_ok=True)
                return None
            except BaseException:
                # 包含 KeyboardInterrupt 等中断异常，立即杀灭当前子进程并清理未完成的临时文件
                if proc:
                    try:
                        proc.kill()
                        proc.wait(timeout=5)
                    except Exception:
                        pass
                tmp_output_path.unlink(missing_ok=True)
                raise
            finally:
                FFmpegProcessRegistry.deregister(str(tmp_output_path))
    except KeyboardInterrupt:
        tmp_output_path.unlink(missing_ok=True)
        raise
    except Exception:
        logger.exception("batch-render cam%d batch%d spawn/IO error", cam_index, batch_idx)
        tmp_output_path.unlink(missing_ok=True)
        return None
    finally:
        # 信号量释放单次原则 + 超时/失败路径统一清理临时脚本
        io_sem.release()
        fc_script.unlink(missing_ok=True)

    elapsed = time.monotonic() - t0

    if proc is not None and proc.returncode == 0 and tmp_output_path.exists():
        try:
            if tmp_output_path.stat().st_size >= 512 * 1024:
                # 原子重命名为正式批次成片，保证断点续传绝不复用半成品
                tmp_output_path.replace(output_path)
            else:
                logger.warning("batch-render cam%d batch%d output too small (%d bytes), discarding",
                               cam_index, batch_idx, tmp_output_path.stat().st_size)
                tmp_output_path.unlink(missing_ok=True)
                return None
        except OSError as e:
            logger.warning("batch-render atomic rename failed: %s", e)
            return None

        # 持久化编码吞吐指标
        enc_stats = _parse_ffmpeg_progress(err_log)
        size_mb = 0.0
        try:
            size_mb = round(output_path.stat().st_size / (1024 * 1024), 1)
        except OSError:
            pass
        get_perf().add(
            PerfRecord(
                stage="render_enc",
                file=output_path.name,
                gpu=encoder,
                duration=round(elapsed, 3),
                extra={**enc_stats, "size_mb": size_mb},
            )
        )
        err_log.unlink(missing_ok=True)
        return str(output_path)

    err_tail = ""
    if err_log.exists():
        try:
            err_tail = err_log.read_bytes()[-1000:].decode("utf-8", errors="replace")
        except Exception:
            pass
        err_log.unlink(missing_ok=True)
    # 删除残缺产物，防止断点续渲复用损坏批次（如 OOM -12 部分写出的 mp4）
    try:
        output_path.unlink(missing_ok=True)
    except OSError:
        pass
    logger.error(
        "batch-render cam%d batch%d failed after %.1fs:\n%s",
        cam_index, batch_idx, elapsed, err_tail,
    )
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


def _build_enc_args(encoder, out_cfg):
    if encoder == "qsv":
        qsv = out_cfg.get("qsv", {})
        return ["-c:v", qsv.get("codec", "hevc_qsv"), "-preset", qsv.get("preset", "fast"), "-global_quality", str(qsv.get("global_quality", 28)), "-maxrate", qsv.get("maxrate", "4M"), "-bufsize", qsv.get("bufsize", "8M"), "-g", "120", "-pix_fmt", "nv12"]
    else:
        nv = out_cfg.get("nv", {})
        return ["-c:v", nv.get("codec", "hevc_nvenc"), "-preset", nv.get("preset", "p1"), "-cq", str(nv.get("cq", 28)), "-maxrate", nv.get("maxrate", "4M"), "-bufsize", nv.get("bufsize", "8M"), "-g", "120", "-pix_fmt", "yuv420p"]
