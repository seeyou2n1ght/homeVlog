import logging
import re
import subprocess
import time
import threading
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.utils import TEMP_DIR
from src.scheduler import acquire_with_retry, VideoLease
from src.ffmpeg import run_ffmpeg
from src.stages.timeline import build_concat_filter, compute_display_plans
from src.monitor import get_perf, PerfRecord

logger = logging.getLogger("homevlog")

ACTIVE_STATES = ("DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE", "MICRO_MOTION")

_FFMPEG_PROGRESS_RE = re.compile(

    r"frame=\s*(\d+)\s+fps=\s*([\d.]+).*?speed=\s*([\d.]+)x"
)


def _parse_ffmpeg_progress(err_log: Path) -> dict:
    """从 ffmpeg stderr 日志提取最后一帧进度（编码帧数/fps/倍速）。"""
    try:
        with err_log.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 4000))
            tail = stream.read().decode("utf-8", errors="replace")
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


def _startup_watchdog_timeout(
    input_count: int,
    steady_timeout: float,
    configured_timeout: float,
    render_timeout: float,
) -> float:
    """Bounded startup grace for serial NAS input/container discovery."""
    per_input_budget = max(60.0, max(1, input_count) * 90.0)
    return max(
        steady_timeout,
        min(render_timeout, configured_timeout, per_input_budget),
    )


# Re-export from src.ffmpeg for backwards compatibility across tests and modules
from src.ffmpeg import FFmpegProcessRegistry


def _clean_ffmpeg_error(err_tail: str) -> str:
    """提取 FFmpeg stderr 中的关键错误日志，剔除逐帧编码进度（frame=...）噪声。"""
    if not err_tail:
        return ""
    # 按换行符与回车符统一拆分逐帧覆盖文本
    lines = err_tail.replace("\r", "\n").split("\n")
    meaningful = []
    for line in lines:
        line_s = line.strip()
        if not line_s:
            continue
        # 过滤典型逐帧状态与比特率进度
        if line_s.startswith("frame=") or ("fps=" in line_s and "speed=" in line_s and "size=" in line_s):
            continue
        meaningful.append(line_s)
    return "\n".join(meaningful[-40:])


def _reindex_timeline(timeline, files):
    """Rebuild input_index for a timeline subset based on its file list."""
    mapping = {f: i for i, f in enumerate(files)}
    for t in timeline:
        t.input_index = mapping[t.filepath]


def _normalize_file_timeline(timeline):
    """Clip overlapping DB intervals before constructing the virtual EDL."""
    normalized = []
    cursor = 0.0
    for seg in sorted(timeline, key=lambda item: (item.start_in_file, item.end_in_file)):
        start = max(cursor, float(seg.start_in_file))
        end = float(seg.end_in_file)
        if end <= start + 1e-4:
            continue
        seg.start_in_file = start
        seg.end_in_file = end
        seg.duration = end - start
        if (
            normalized
            and normalized[-1].filepath == seg.filepath
            and normalized[-1].state == seg.state
            and abs(normalized[-1].end_in_file - start) <= 1e-3
        ):
            normalized[-1].end_in_file = end
            normalized[-1].duration = normalized[-1].end_in_file - normalized[-1].start_in_file
        else:
            normalized.append(seg)
        cursor = end
    return normalized


@dataclass(frozen=True)
class VirtualConcatPlan:
    """One active decoder context with seekable source ranges.

    Dynamic/audio ranges retain every source frame.  Static ranges retain a
    bounded set of representative frames while their concat ``duration``
    directives preserve the original source-time axis for the shared display
    plan and subtitle inverse mapping.
    """

    text: str
    entries: int
    static_entries: int
    mapped_timeline: tuple = ()
    source_timeline: tuple = ()
    decoded_duration: float = 0.0
    source_duration: float = 0.0


def _ffconcat_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/").replace("'", r"'\''")


def build_virtual_concat_plan(
    timeline,
    static_sample_window_s: float = 0.25,
    source_override: str | Path | None = None,
    dynamic_coalesce_gap_s: float = 120.0,
    dynamic_window_max_s: float = 600.0,
) -> VirtualConcatPlan:
    """Build a concat-demuxer EDL for a single source file.

    The EDL is intentionally text-only.  No candidate frames are written to
    disk.  Declared durations make the virtual input PTS match the original
    per-file PTS even though static source windows contain only a few frames.
    """
    files = {str(seg.filepath) for seg in timeline}
    if len(files) != 1:
        raise ValueError("virtual concat rendering requires exactly one source file")
    source = str(source_override) if source_override is not None else next(iter(files))
    escaped = _ffconcat_path(source)
    sample_window = max(0.08, float(static_sample_window_s))
    lines = ["ffconcat version 1.0"]
    entries = 0
    static_entries = 0

    def add_entry(start: float, end: float, duration: float) -> None:
        nonlocal entries
        start = max(0.0, start)
        end = max(start + 0.04, end)
        lines.extend([
            f"file '{escaped}'",
            f"inpoint {start:.6f}",
            f"outpoint {end:.6f}",
            f"duration {max(0.04, duration):.6f}",
        ])
        entries += 1

    # Reopening a remote camera file for every small motion island is slower
    # than decoding a short static gap.  Merge nearby motion ranges into a
    # bounded full-frame window; uncovered static spans need one visual sample
    # because their final display duration is at most two seconds.
    dynamic_ranges = [
        [float(seg.start_in_file), float(seg.end_in_file)]
        for seg in timeline
        if seg.state in ACTIVE_STATES
    ]

    windows: list[list[float]] = []
    gap_limit = max(0.0, float(dynamic_coalesce_gap_s))
    window_limit = max(1.0, float(dynamic_window_max_s))
    for start, end in dynamic_ranges:
        if (
            windows
            and start - windows[-1][1] <= gap_limit
            and end - windows[-1][0] <= window_limit
        ):
            windows[-1][1] = end
        else:
            windows.append([start, end])

    coverage_start = float(timeline[0].start_in_file)
    coverage_end = float(timeline[-1].end_in_file)
    cursor = coverage_start

    def add_static(start: float, end: float) -> None:
        nonlocal static_entries
        duration = end - start
        if duration <= 0.001:
            return
        # Seek around the midpoint so the decoder has a keyframe on either
        # side of the requested interval.  The declared duration retains the
        # whole static wall-clock interval; only this small source window is
        # actually decoded.  Starting exactly at a segment boundary can land
        # after the first packet's DTS and make concatdec_select discard it.
        midpoint = start + duration * 0.5
        # Camera GOPs can exceed four seconds.  A ten-second representative
        # window guarantees an in-range decoded frame after hardware seek;
        # shorter windows intermittently produced empty static branches.
        half_window = min(duration * 0.5, max(5.0, sample_window * 0.5))
        source_start = max(start, midpoint - half_window)
        source_end = min(end, midpoint + half_window)
        add_entry(source_start, max(source_start + 0.04, source_end), duration)
        static_entries += 1

    for start, end in windows:
        add_static(cursor, start)
        add_entry(start, end, end - start)
        cursor = end
    add_static(cursor, coverage_end)

    return VirtualConcatPlan("\n".join(lines) + "\n", entries, static_entries)


def build_compact_virtual_concat_plan(
    timeline,
    static_sample_window_s: float = 10.0,
    source_override: str | Path | None = None,
    dynamic_coalesce_gap_s: float = 120.0,
    dynamic_window_max_s: float = 600.0,
) -> VirtualConcatPlan:
    """Build a compact mixed-file EDL and its virtual-to-source timeline map.

    Unlike ``build_virtual_concat_plan``, static gaps do not create artificial
    timestamp jumps.  The demuxer timeline contains only decoded media; the
    paired source timeline remains the authority for display duration, audio
    trims and wall-clock mapping.
    """
    from copy import deepcopy
    from src.timeline import TimelineSegment

    files = {str(seg.filepath) for seg in timeline}
    if len(files) != 1:
        raise ValueError("compact virtual rendering requires exactly one source file")
    source = str(source_override) if source_override is not None else next(iter(files))
    escaped = _ffconcat_path(source)
    lines = ["ffconcat version 1.0"]
    mapped: list[TimelineSegment] = []
    source_segments: list[TimelineSegment] = []
    entries = 0
    static_entries = 0
    virtual_cursor = 0.0

    dynamic_ranges = [
        [float(seg.start_in_file), float(seg.end_in_file)]
        for seg in timeline
        if seg.state in ACTIVE_STATES
    ]

    windows: list[list[float]] = []
    gap_limit = max(0.0, float(dynamic_coalesce_gap_s))
    window_limit = max(1.0, float(dynamic_window_max_s))
    for start, end in dynamic_ranges:
        if windows and start - windows[-1][1] <= gap_limit and end - windows[-1][0] <= window_limit:
            windows[-1][1] = end
        else:
            windows.append([start, end])

    def add_descriptor(start: float, end: float) -> None:
        nonlocal entries
        duration = max(0.04, end - start)
        lines.extend([
            f"file '{escaped}'",
            f"inpoint {start:.6f}",
            f"outpoint {end:.6f}",
            f"duration {duration:.6f}",
        ])
        entries += 1

    # Some camera MP4s keep the first repeated-file entry on the source PTS
    # axis when its inpoint is far from zero.  A tiny zero-origin anchor makes
    # all following concat offsets deterministic; no output segment consumes
    # the anchor itself.
    anchor_duration = 10.0
    add_descriptor(0.0, anchor_duration)
    virtual_cursor = anchor_duration

    def intersecting(start: float, end: float):
        for seg in timeline:
            a = max(start, float(seg.start_in_file))
            b = min(end, float(seg.end_in_file))
            if b > a + 1e-4:
                yield seg, a, b

    def add_full_window(start: float, end: float) -> None:
        nonlocal virtual_cursor
        if end <= start + 1e-4:
            return
        add_descriptor(start, end)
        for seg, a, b in intersecting(start, end):
            src = deepcopy(seg)
            src.start_in_file, src.end_in_file, src.duration = a, b, b - a
            source_segments.append(src)
            mapped.append(TimelineSegment(
                filepath=seg.filepath,
                input_index=seg.input_index,
                start_in_file=virtual_cursor + (a - start),
                end_in_file=virtual_cursor + (b - start),
                state=seg.state,
                duration=b - a,
            ))
        virtual_cursor += end - start

    def add_static_gap(start: float, end: float) -> None:
        nonlocal virtual_cursor, static_entries
        if end <= start + 1e-4:
            return
        covered = list(intersecting(start, end))
        if not covered:
            return
        midpoint = start + (end - start) * 0.5
        half = min((end - start) * 0.5, max(5.0, float(static_sample_window_s) * 0.5))
        sample_start = max(start, midpoint - half)
        sample_end = min(end, midpoint + half)
        sample_duration = max(0.04, sample_end - sample_start)
        add_descriptor(sample_start, sample_end)
        static_entries += 1
        # Normalization merges adjacent equal states, so an uncovered gap is
        # normally one static segment.  Split the sample proportionally if a
        # historical row still contains multiple adjacent static fragments.
        total_source = sum(b - a for _, a, b in covered)
        local_cursor = virtual_cursor
        for seg, a, b in covered:
            share = sample_duration * ((b - a) / total_source)
            src = deepcopy(seg)
            src.start_in_file, src.end_in_file, src.duration = a, b, b - a
            source_segments.append(src)
            mapped.append(TimelineSegment(
                filepath=seg.filepath,
                input_index=seg.input_index,
                start_in_file=local_cursor,
                end_in_file=local_cursor + share,
                state=seg.state,
                duration=share,
            ))
            local_cursor += share
        virtual_cursor += sample_duration

    coverage_start = float(timeline[0].start_in_file)
    coverage_end = float(timeline[-1].end_in_file)
    cursor = coverage_start
    for start, end in windows:
        add_static_gap(cursor, start)
        add_full_window(start, end)
        cursor = end
    add_static_gap(cursor, coverage_end)

    return VirtualConcatPlan(
        "\n".join(lines) + "\n", entries, static_entries,
        tuple(mapped), tuple(source_segments),
        virtual_cursor,
        max(0.0, coverage_end - coverage_start),
    )


_staging_locks: dict[str, threading.Lock] = {}
_staging_registry_lock = threading.Lock()


def _get_stage_lock(digest: str) -> threading.Lock:
    with _staging_registry_lock:
        if digest not in _staging_locks:
            _staging_locks[digest] = threading.Lock()
        return _staging_locks[digest]


def _stage_source_for_render(source: str | Path) -> Path | None:
    """Sequentially stage one NAS clip for low-latency local EDL seeks."""
    source_path = Path(source)
    try:
        source_stat = source_path.stat()
    except OSError as exc:
        logger.warning("render staging cannot stat %s: %s", source_path, exc)
        return None

    stage_dir = TEMP_DIR / "staging"
    try:
        stage_dir.mkdir(parents=True, exist_ok=True)
        free_bytes = shutil.disk_usage(stage_dir).free
    except OSError as exc:
        logger.warning("render staging directory unavailable: %s", exc)
        return None
    if free_bytes < source_stat.st_size + 2 * 1024**3:
        logger.warning(
            "render staging skipped for %s: local free space below 2GB reserve",
            source_path.name,
        )
        return None

    identity = f"{source_path.resolve()}:{source_stat.st_size}:{source_stat.st_mtime_ns}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    staged = stage_dir / f"{digest}{source_path.suffix}"
    partial = staged.with_suffix(staged.suffix + ".part")

    with _get_stage_lock(digest):
        if staged.exists() and staged.stat().st_size == source_stat.st_size:
            return staged

        from src.utils import get_disk_semaphore
        disk_sem = get_disk_semaphore()
        acquired = acquire_with_retry(disk_sem, timeout=30.0, retries=6)
        if not acquired:
            logger.warning("render staging I/O admission failed for %s", source_path.name)
            return None
        t0 = time.monotonic()
        try:
            partial.unlink(missing_ok=True)
            shutil.copyfile(source_path, partial)
            if partial.stat().st_size != source_stat.st_size:
                raise OSError("staged file size differs from source")
            partial.replace(staged)
            dur = round(time.monotonic() - t0, 3)
            size_mb = round(source_stat.st_size / 1024**2, 1)
            get_perf().add(PerfRecord(
                stage="render_stage", file=source_path.name, gpu="cpu",
                duration=dur,
                extra={"size_mb": size_mb, "speed_mb_s": round(size_mb / max(0.001, dur), 1)},
                start_time=round(t0, 3),
                end_time=round(time.monotonic(), 3),
            ))
            return staged
        except OSError as exc:
            logger.warning("render staging failed for %s; using source directly: %s", source_path.name, exc)
            partial.unlink(missing_ok=True)
            return None
        finally:
            disk_sem.release()


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
    # A few historical DB rows contain a tiny leading interval overlapping the
    # following full static interval.  Rendering those twice shifts the EDL
    # PTS and causes duration validation failures; clipping is lossless for the
    # covered source axis and keeps the final frame closed.
    batch_copy = _normalize_file_timeline(batch_copy)
    files = list(dict.fromkeys(s.filepath for s in batch_copy))
    _reindex_timeline(batch_copy, files)

    if enc_for_batch == "nv":
        scale_mode = "cuda_passthrough"
    elif enc_for_batch == "qsv":
        scale_mode = "qsv"
    else:
        scale_mode = "cpu"

    has_dynamic = any(s.state in ACTIVE_STATES for s in batch_copy)
    all_dynamic = bool(batch_copy) and all(s.state in ACTIVE_STATES for s in batch_copy)

    # Mixed EDL stays behind its own rollout switch.  The continuous sparse
    # filter remains the conservative fallback when a platform cannot preserve
    # concat-demuxer timestamps accurately.
    virtual_enabled = (
        bool(render_cfg.get("virtual_concat_enabled", True))
        and len(files) == 1
        and not all_dynamic
        and (not has_dynamic or bool(render_cfg.get("virtual_concat_mixed_enabled", False)))
    )
    virtual_mixed = virtual_enabled and has_dynamic
    prebuilt_plan = None
    if virtual_mixed:
        prebuilt_plan = build_compact_virtual_concat_plan(
            batch_copy,
            static_sample_window_s=render_cfg.get("static_sample_window_s", 10.0),
            dynamic_coalesce_gap_s=render_cfg.get("dynamic_coalesce_gap_s", 20.0),
            dynamic_window_max_s=render_cfg.get("dynamic_window_max_s", 600.0),
        )
        # 排除仅用于在 EDL 首部保持绝对时间轴对齐的 10.0s anchor 开销，真实评估有效解码覆盖率
        anchor_overhead = 10.0
        effective_decoded = max(0.0, prebuilt_plan.decoded_duration - anchor_overhead)
        decode_ratio = effective_decoded / max(0.001, prebuilt_plan.source_duration)
        max_ratio = float(render_cfg.get("virtual_concat_max_decode_ratio", 0.80))
        if decode_ratio > max_ratio:
            logger.info(
                "virtual concat batch %d skipped: decode coverage %.1f%% exceeds %.1f%%",
                bi, decode_ratio * 100.0, max_ratio * 100.0,
            )
            virtual_enabled = False
            virtual_mixed = False
            prebuilt_plan = None
    if virtual_mixed:
        if enc_for_batch == "nv":
            scale_mode = "cuda_passthrough"
        elif enc_for_batch == "qsv":
            scale_mode = "qsv"
        else:
            scale_mode = "cpu"
    descriptor_path = None
    staged_source = None
    input_args = None
    plan = None
    prepare_input = None
    if virtual_enabled:
        plan_builder = build_compact_virtual_concat_plan if has_dynamic else build_virtual_concat_plan
        plan = prebuilt_plan or plan_builder(
            batch_copy,
            static_sample_window_s=render_cfg.get("static_sample_window_s", 0.25),
            dynamic_coalesce_gap_s=render_cfg.get("dynamic_coalesce_gap_s", 20.0),
            dynamic_window_max_s=render_cfg.get("dynamic_window_max_s", 600.0),
        )
        descriptor_path = TEMP_DIR / f"_edl_batch{bi}_{date}_cam{cam_index}.ffconcat"
        descriptor_path.parent.mkdir(parents=True, exist_ok=True)

        def _prepare_virtual_input():
            nonlocal staged_source
            if render_cfg.get("local_staging_enabled", True):
                staged_source = _stage_source_for_render(files[0])
            execution_plan = plan_builder(
                batch_copy,
                static_sample_window_s=render_cfg.get("static_sample_window_s", 0.25),
                source_override=staged_source,
                dynamic_coalesce_gap_s=render_cfg.get("dynamic_coalesce_gap_s", 20.0),
                dynamic_window_max_s=render_cfg.get("dynamic_window_max_s", 600.0),
            )
            descriptor_path.write_text(execution_plan.text, encoding="utf-8")
            return True

        prepare_input = _prepare_virtual_input
        input_args = [
            "-fflags", "+genpts",
            "-segment_time_metadata", "1",
            "-f", "concat", "-safe", "0",
        ]
        if enc_for_batch == "nv":
            input_args += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif enc_for_batch == "qsv":
            input_args += ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
        input_args += ["-i", str(descriptor_path)]
        if has_dynamic:
            # Keep audio on the original continuous timeline. Repeated concat
            # inpoints can drop AAC priming packets and shorten short motion
            # islands even when video packet metadata is exact.
            input_args += ["-i", files[0]]

    filter_timeline = list(plan.mapped_timeline) if plan and plan.mapped_timeline else batch_copy
    source_timeline = list(plan.source_timeline) if plan and plan.source_timeline else batch_copy

    fc = build_concat_filter(
        filter_timeline,
        rows=rows,
        output_fps=fps, output_width=width, output_height=height,
        static_keyframe_interval=seg_cfg.get("static_keyframe_interval", 30.0),
        keyframe_display_duration=seg_cfg.get("keyframe_display_duration", 0.5),
        min_static_display_duration=seg_cfg.get("min_static_display_duration", 1.5),
        max_static_display_duration=seg_cfg.get("max_static_display_duration", 2.0),
        gap_tolerance=seg_cfg.get("gap_tolerance", 0.5),
        scale_mode=scale_mode,
        speed_ramping=render_cfg.get("speed_ramping_enabled", seg_cfg.get("speed_ramping_enabled", True)),
        ramp_duration_s=render_cfg.get("ramp_duration_s", seg_cfg.get("ramp_duration_s", 1.0)),
        audio_fade_duration_s=render_cfg.get("audio_fade_duration_s", seg_cfg.get("audio_fade_duration_s", 0.15)),
        timecode_osd=render_cfg.get("timecode_osd_enabled", False),
        base_date=date,
        preselected_static=virtual_enabled,
        concat_demuxer=virtual_enabled,
        simple_dynamic=(not virtual_enabled and len(batch_copy) == 1 and
                        batch_copy[0].state in ("DYNAMIC", "DYNAMIC_AUDIO")),
        sparse_mixed=(not virtual_enabled and bool(render_cfg.get("mixed_sparse_enabled", True))),
        static_sample_window_s=render_cfg.get("static_sample_window_s", 0.25),
        audio_input_offset=1 if virtual_enabled and has_dynamic else 0,
        source_timeline=source_timeline,
    )

    batch_path = TEMP_DIR / f"_batch{bi}_{date}_cam{cam_index}.mp4"

    # 识别批次内纯静态文件（没有任何动态段），在解复用阶段跳过所有非关键帧解码
    files_with_dynamic = {s.filepath for s in batch_copy if s.state in ACTIVE_STATES}
    pure_static_files = {f for f in files if f not in files_with_dynamic}
    presence_cfg = cfg.get("presence", {})
    micro_cfg = cfg.get("micro_motion", {})

    expected_duration = sum(d for d, _ in compute_display_plans(
        batch_copy,
        static_keyframe_interval=seg_cfg.get("static_keyframe_interval", 30.0),
        keyframe_display_duration=seg_cfg.get("keyframe_display_duration", 0.5),
        min_static_display_duration=seg_cfg.get("min_static_display_duration", 1.5),
        max_static_display_duration=seg_cfg.get("max_static_display_duration", 2.0),
        speed_ramping=render_cfg.get("speed_ramping_enabled", True),
        ramp_duration_s=render_cfg.get("ramp_duration_s", 1.0),
        presence_speed_factor=float(presence_cfg.get("speed_factor", 4.0)),
        micro_motion_anchor_s=float(micro_cfg.get("anchor_duration_s", 3.0)),
        micro_motion_cruise_speed=float(micro_cfg.get("cruise_speed", 16.0)),
    ))
    # Match decoder skipping to the same long-static predicate used by the filter graph.
    interval = seg_cfg.get("static_keyframe_interval", 30.0)
    pure_static_files = {f for f in pure_static_files if all(
        s.end_in_file - s.start_in_file >= 2 * interval for s in batch_copy if s.filepath == f)}
    if virtual_enabled:
        pure_static_files = set()
        logger.info(
            "virtual concat batch %d: %d EDL ranges (%d static samples), one decoder context",
            bi, plan.entries, plan.static_entries,
        )
    try:
        result = _run_batch_render(
            files, fc, batch_path, enc_for_batch, fps, out_cfg, audio_cfg,
            date, cam_index, batch_idx=bi, pure_static_files=pure_static_files,
            expected_duration=expected_duration, input_args=input_args,
            fingerprint_salt=plan.text if plan else "", prepare_input=prepare_input,
        )
        if result is not None or not virtual_mixed or FFmpegProcessRegistry.is_interrupted():
            return result

        logger.warning(
            "virtual concat batch %d failed validation; retrying continuous sparse path",
            bi,
        )
        fallback_scale = "cuda_passthrough" if enc_for_batch == "nv" else ("qsv" if enc_for_batch == "qsv" else "cpu")
        fallback_fc = build_concat_filter(
            batch_copy,
            rows=rows,
            output_fps=fps, output_width=width, output_height=height,
            static_keyframe_interval=seg_cfg.get("static_keyframe_interval", 30.0),
            keyframe_display_duration=seg_cfg.get("keyframe_display_duration", 0.5),
            min_static_display_duration=seg_cfg.get("min_static_display_duration", 1.5),
            max_static_display_duration=seg_cfg.get("max_static_display_duration", 2.0),
            gap_tolerance=seg_cfg.get("gap_tolerance", 0.5),
            scale_mode=fallback_scale,
            speed_ramping=render_cfg.get("speed_ramping_enabled", True),
            ramp_duration_s=render_cfg.get("ramp_duration_s", 1.0),
            audio_fade_duration_s=render_cfg.get("audio_fade_duration_s", 0.15),
            timecode_osd=render_cfg.get("timecode_osd_enabled", False),
            base_date=date,
            sparse_mixed=bool(render_cfg.get("mixed_sparse_enabled", True)),
            static_sample_window_s=render_cfg.get("static_sample_window_s", 10.0),
        )
        return _run_batch_render(
            files, fallback_fc, batch_path, enc_for_batch, fps, out_cfg, audio_cfg,
            date, cam_index, batch_idx=bi, pure_static_files=set(),
            expected_duration=expected_duration,
            fingerprint_salt="virtual-fallback-v1",
        )
    finally:
        if descriptor_path is not None:
            descriptor_path.unlink(missing_ok=True)
        if staged_source is not None:
            staged_source.unlink(missing_ok=True)


def _run_batch_render(input_files, filter_complex, output_path, encoder, fps, out_cfg, audio_cfg, date, cam_index, batch_idx=0, pure_static_files=None, expected_duration=None, input_args=None, fingerprint_salt="", prepare_input=None) -> str | None:
    output_path = Path(output_path)
    if FFmpegProcessRegistry.is_interrupted():
        logger.info("batch-render cam%d batch%d terminated by signal (Ctrl+C)", cam_index, batch_idx)
        return None

    from src.utils import load_config
    cfg = {}
    try:
        cfg = load_config()
    except Exception:
        pass
    render_cfg = cfg.get("render", {})
    from src.render_cache import render_fingerprint, reusable, valid_video, save_manifest
    fingerprint = render_fingerprint(
        input_files, filter_complex + fingerprint_salt,
        encoder, fps, out_cfg, audio_cfg, cfg,
    )
    if reusable(output_path, fingerprint, expected_duration):
        logger.info("Reusing verified batch %s", output_path.name)
        return str(output_path)

    if prepare_input is not None:
        try:
            if not prepare_input():
                logger.warning("batch-render cam%d batch%d input preparation failed", cam_index, batch_idx)
                return None
        except Exception:
            logger.exception("batch-render cam%d batch%d input preparation failed", cam_index, batch_idx)
            return None


    hwaccel_args = []
    if input_args is not None:
        hwaccel_args = list(input_args)
    else:
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

    cmd = ["ffmpeg", "-hide_banner", "-y", "-nostdin"]
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
        io_sem = VideoLease(get_qsv_semaphore(), 1 if input_args is not None else len(input_files))
    else:
        from src.utils import get_nvenc_semaphore
        io_sem = VideoLease(get_nvenc_semaphore(), 1 if input_args is not None else len(input_files))
    err_log = TEMP_DIR / f"_stderr_batch{batch_idx}_{date}_cam{cam_index}.log"

    # AGENTS.md 铁律：acquire 必须带 timeout 并重试，禁止无限阻塞
    # 渲染是产出关键路径，预算放宽至 30s×10：等待解码租约释放远优于批次 abort
    lease_wait_t0 = time.monotonic()
    if not acquire_with_retry(io_sem, timeout=30.0, retries=10):
        logger.warning(
            "batch-render cam%d batch%d: io semaphore acquire timeout, aborting",
            cam_index, batch_idx,
        )
        fc_script.unlink(missing_ok=True)
        return None
    lease_wait = round(time.monotonic() - lease_wait_t0, 3)


    proc: subprocess.Popen | None = None
    t0 = time.monotonic()
    try:
        with open(err_log, "wb") as f_err:
            proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=f_err)
            FFmpegProcessRegistry.register(str(tmp_output_path), proc)
            if encoder == "qsv":
                base_timeout = max(render_cfg.get("qsv_timeout_s", 600), len(input_files) * 300)
                dur_timeout = int((expected_duration or 0) * 2.5) + 60
                render_timeout = max(base_timeout, dur_timeout)
            else:
                render_timeout = max(3600, len(input_files) * 600)
            inactivity_timeout = float(render_cfg.get("inactivity_timeout_s", 120.0))
            configured_startup_timeout = float(
                render_cfg.get("startup_inactivity_timeout_s", 600.0)
            )
            # Opening several remote 4K MP4 inputs is serial inside FFmpeg and
            # can legitimately take minutes while analysis is also reading the
            # NAS.  Do not apply the steady-state encoder timeout before the
            # first frame/output byte exists.
            startup_timeout = _startup_watchdog_timeout(
                len(input_files), inactivity_timeout,
                configured_startup_timeout, render_timeout,
            )
            last_active_time = time.monotonic()
            last_frame_count = -1
            last_output_size = 0
            last_stderr_size = 0
            encoding_started = False

            try:
                while True:
                    try:
                        ret = proc.wait(timeout=2.0)
                        break
                    except subprocess.TimeoutExpired:
                        pass

                    now = time.monotonic()
                    if now - t0 > render_timeout:
                        raise subprocess.TimeoutExpired(cmd, render_timeout)

                    # 动态心跳与停滞检测 (Inactivity Watchdog)
                    has_progress = False
                    try:
                        cur_output_size = tmp_output_path.stat().st_size
                        if cur_output_size > last_output_size:
                            has_progress = True
                            last_output_size = cur_output_size
                            if cur_output_size > 0:
                                encoding_started = True
                    except OSError:
                        pass

                    progress = _parse_ffmpeg_progress(err_log)
                    cur_frames = progress.get("enc_frames", -1)
                    if cur_frames > last_frame_count:
                        has_progress = True
                        last_frame_count = cur_frames
                        if cur_frames >= 0:
                            encoding_started = True

                    # Before encoding starts, FFmpeg emits input/container
                    # discovery to stderr.  Growth proves that startup is still
                    # advancing even though no encoded frame exists yet.
                    if not encoding_started:
                        try:
                            cur_stderr_size = err_log.stat().st_size
                            if cur_stderr_size > last_stderr_size:
                                has_progress = True
                                last_stderr_size = cur_stderr_size
                        except OSError:
                            pass

                    if has_progress:
                        last_active_time = now
                    active_timeout = inactivity_timeout if encoding_started else startup_timeout
                    if not has_progress and now - last_active_time > active_timeout:
                        logger.warning(
                            "batch-render cam%d batch%d stalled on %s during %s "
                            "(no progress for %.1fs, frame=%d), terminating process",
                            cam_index, batch_idx, encoder,
                            "encoding" if encoding_started else "startup",
                            now - last_active_time, last_frame_count,
                        )
                        proc.kill()
                        try:
                            proc.wait(timeout=5)
                        except Exception:
                            pass
                        tmp_output_path.unlink(missing_ok=True)
                        return None
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
            if valid_video(tmp_output_path, expected_duration):
                # 原子重命名为正式批次成片，保证断点续传绝不复用半成品
                tmp_output_path.replace(output_path)
                save_manifest(output_path, fingerprint)
            else:
                logger.warning("batch-render cam%d batch%d output failed container/duration validation (%d bytes), discarding",
                               cam_index, batch_idx, tmp_output_path.stat().st_size)
                tmp_output_path.unlink(missing_ok=True)
                err_log.unlink(missing_ok=True)
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
                extra={**enc_stats, "size_mb": size_mb, "sem_wait": lease_wait},
                start_time=round(t0, 3),
                end_time=round(time.monotonic(), 3),
            )
        )
        err_log.unlink(missing_ok=True)
        return str(output_path)

    err_tail = ""
    if err_log.exists():
        try:
            err_tail = err_log.read_bytes()[-8192:].decode("utf-8", errors="replace")
        except Exception:
            pass
        err_log.unlink(missing_ok=True)
    # 删除残缺产物，防止断点续渲复用损坏批次（如 OOM -12 部分写出的 mp4）
    try:
        tmp_output_path.unlink(missing_ok=True)
    except OSError:
        pass

    clean_err = _clean_ffmpeg_error(err_tail)
    # 判断是否为用户主动中断 / Ctrl+C 信号引起的正常退出
    is_interrupted = (
        FFmpegProcessRegistry.is_interrupted()
        or "received signal 2" in err_tail
        or (proc is not None and proc.returncode in (255, -2, 130, -9, 15, 1, 3221225786, -1073741510) and not clean_err)
    )
    if is_interrupted:
        logger.info(
            "batch-render cam%d batch%d terminated by signal (Ctrl+C)",
            cam_index, batch_idx,
        )
    else:
        logger.error(
            "batch-render cam%d batch%d failed after %.1fs (code=%s):\n%s",
            cam_index, batch_idx, elapsed,
            proc.returncode if proc is not None else "unknown",
            clean_err or "Process terminated abnormally without explicit error log.",
        )
    return None


def concat_output_files(files: list[Path], output: Path, timeout: float = 300, faststart: bool = False) -> bool:
    from src.render_cache import valid_video
    if not files:
        return False
    concat_list = output.with_name(f".concat_{output.stem}.txt")
    temporary = output.with_name(output.stem + ".tmp.mp4")
    def _concat_path(path):
        return str(Path(path).resolve()).replace(chr(92), "/").replace("'", r"'\''")
    try:
        concat_list.write_text("\n".join(f"file '{_concat_path(f)}'" for f in files)+"\n", encoding="utf-8")
        cmd = ["-f", "concat", "-safe", "0", "-i", str(concat_list), "-c", "copy"]
        if faststart:
            cmd.extend(["-movflags", "+faststart"])
        cmd.append(str(temporary))
        result = run_ffmpeg(cmd, timeout=timeout)
        if result.returncode != 0 or not valid_video(temporary):
            return False
        temporary.replace(output)
        return True
    finally:
        concat_list.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)

def _build_enc_args(encoder, out_cfg):
    if encoder == "qsv":
        qsv = out_cfg.get("qsv", {})
        return [
            "-c:v", qsv.get("codec", "hevc_qsv"),
            "-preset", qsv.get("preset", "fast"),
            "-global_quality", str(qsv.get("global_quality", 28)),
            "-maxrate", qsv.get("maxrate", "4M"),
            "-bufsize", qsv.get("bufsize", "8M"),
            "-g", "60",
            "-forced_idr", "1",
            "-pix_fmt", qsv.get("pix_fmt", "nv12"),
            "-bsf:v", "dump_extra",
        ]
    else:
        nv = out_cfg.get("nv", {})
        args = [
            "-c:v", nv.get("codec", "hevc_nvenc"),
            "-preset", nv.get("preset", "p3"),
            "-cq", str(nv.get("cq", 28)),
            "-maxrate", nv.get("maxrate", "4M"),
            "-bufsize", nv.get("bufsize", "8M"),
            "-g", "60",
            "-forced-idr", "1",
            "-pix_fmt", nv.get("pix_fmt", "nv12"),
            "-bsf:v", "dump_extra",
        ]
        if nv.get("tune"):
            args.extend(["-tune", str(nv["tune"])])
        return args
