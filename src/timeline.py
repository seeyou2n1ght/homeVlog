import logging
from dataclasses import dataclass

from src.database import VlogDatabase
from src.segment import Segment, segments_from_json, merge_cross_file
from src.utils import load_config, ts_to_unix

logger = logging.getLogger("homevlog")


@dataclass
class TimelineSegment:
    filepath: str
    input_index: int
    start_in_file: float
    end_in_file: float
    state: str
    duration: float


def build_timeline(db: VlogDatabase, date: str, cam_index: int) -> list[TimelineSegment]:
    config = load_config()
    rows = db.get_all_file_tasks_for_date(date, cam_index)

    day_start = ts_to_unix(date + "000000")

    all_segments: list[Segment] = []
    for row in rows:
        file_start_unix = ts_to_unix(row["file_start_time"])
        file_end_unix = ts_to_unix(row["file_end_time"])
        file_offset = file_start_unix - day_start
        file_end_offset = file_end_unix - day_start

        if row["prescreen_status"] == "STATIC":
            all_segments.append(Segment(
                start_time=file_offset, end_time=file_end_offset,
                state="STATIC",
                source_file=row["filepath"],
                file_start_offset=file_offset,
            ))
        elif row["analysis_segments"]:
            segs = segments_from_json(row["analysis_segments"])
            for s in segs:
                s.source_file = row["filepath"]
            all_segments.extend(segs)

    all_segments.sort(key=lambda s: s.start_time)
    seg_cfg = config.get("segment", {})
    gap_tolerance = seg_cfg.get("gap_tolerance", 0.5)
    min_motion_dur = seg_cfg.get("min_motion_duration", 2.0)
    min_static_dur = seg_cfg.get("min_static_duration", 30.0)
    
    # 全局跨文件平滑与合并，解决边界截断问题
    merged = merge_cross_file(all_segments, gap_tolerance)
    from src.segment import _filter_short
    filtered = _filter_short(merged, min_motion_dur, min_static_dur, gap_tolerance)
    merged = filtered

    unique_files = list(dict.fromkeys(s.source_file for s in merged))
    file_to_idx = {f: i for i, f in enumerate(unique_files)}

    min_seg_dur = seg_cfg.get("min_segment_duration", 0.1)
    timeline: list[TimelineSegment] = []
    for s in merged:
        idx = file_to_idx[s.source_file]
        file_start = s.file_start_offset
        start_in_file = max(s.start_time - file_start, 0.0)
        end_in_file = max(s.end_time - file_start, start_in_file + min_seg_dur)
        timeline.append(TimelineSegment(
            filepath=s.source_file,
            input_index=idx,
            start_in_file=start_in_file,
            end_in_file=end_in_file,
            state=s.state,
            duration=end_in_file - start_in_file,
        ))

    logger.info(
        "timeline for %s cam%d: %d files, %d segments",
        date, cam_index, len(unique_files), len(timeline),
    )
    return timeline


def build_concat_filter(
    timeline: list[TimelineSegment],
    output_fps: int = 20,
    output_width: int = 1920,
    output_height: int = 1080,
    static_keyframe_interval: float = 30.0,
    keyframe_display_duration: float = 0.5,
    min_static_display_duration: float = 1.5,
    audio_sample_rate: int = 48000,
    gap_tolerance: float = 0.5,
    scale_mode: str = "cpu",
) -> str:
    """
    Build a complex FFmpeg filtergraph string for the entire timeline.

    Dynamic segments play at original speed with audio.
    Static segments: fast-forward via setpts (GPU path) or keyframe slideshow (CPU path).

    GPU scale is done ONCE per input file (not per segment) to avoid CUDA OOM
    with large segment counts. Per-file scaled streams are split+trimmed per segment.
    """
    kf_interval = max(static_keyframe_interval, 1.0)
    display_dur = max(keyframe_display_duration, 0.1)
    global_speed_factor = kf_interval / display_dur

    if scale_mode == "cuda":
        # CPU frames → upload to GPU → scale → download
        scale_filter = f"hwupload_cuda,scale_cuda={output_width}:{output_height},hwdownload,format=nv12"
    elif scale_mode == "cuda_passthrough":
        # NVDEC frames (already CUDA) → scale on GPU → download (no hwupload needed)
        scale_filter = f"scale_cuda={output_width}:{output_height},hwdownload,format=nv12"
    elif scale_mode == "qsv":
        scale_filter = f"scale_qsv=w={output_width}:h={output_height},hwdownload,format=nv12"
    elif scale_mode == "skip":
        scale_filter = None
    else:
        scale_filter = f"scale={output_width}:{output_height}"

    use_keyframe_slideshow = (scale_mode == "cpu")
    # --- Performance Optimization: Hybrid Keyframe Mode ---
    # In GPU mode, using setpts (fast-forward) still decodes many frames.
    # Hybrid mode uses the CPU-style "slideshow" approach (sparse fps) even on GPU.
    from src.utils import load_config
    try:
        tmp_cfg = load_config()
        if tmp_cfg.get("render", {}).get("static_mode") == "hybrid_keyframe":
            use_keyframe_slideshow = True
    except Exception:
        pass

    # --- Step 1: Count segments per input file ---
    segs_per_file: dict[int, int] = {}
    input_files: dict[int, str] = {}
    for seg in timeline:
        segs_per_file[seg.input_index] = segs_per_file.get(seg.input_index, 0) + 1
        input_files[seg.input_index] = seg.filepath

    input_has_audio: dict[int, bool] = {}
    try:
        from src.ffmpeg import run_ffprobe
        for idx, filepath in input_files.items():
            info = run_ffprobe(filepath) or {}
            input_has_audio[idx] = any(
                s.get("codec_type") == "audio" for s in info.get("streams", [])
            )
    except Exception:
        logger.exception("ffprobe audio detection failed; fallback to silent audio")
        input_has_audio = {idx: False for idx in input_files}

    # --- Step 2: Per-file scale (once per input) ---
    scale_parts: list[str] = []
    # Track which split output label to use for each file
    file_split_labels: dict[int, list[str]] = {}
    # Running counter for split output labels per file
    file_split_counter: dict[int, int] = {}

    for idx in sorted(segs_per_file):
        n_segs = segs_per_file[idx]

        if scale_filter is not None:
            scale_parts.append(f"[{idx}:v]{scale_filter}[scaled_{idx}]")
            base_label = f"scaled_{idx}"
        elif n_segs > 1:
            # skip mode + multiple segs: need split, so create a pass-through label
            scale_parts.append(f"[{idx}:v]null[skip_{idx}]")
            base_label = f"skip_{idx}"
        else:
            # skip mode + single seg: null pass-through to produce a label for trim
            scale_parts.append(f"[{idx}:v]null[skip_{idx}]")
            base_label = f"skip_{idx}"

        if n_segs > 1:
            out_labels = [f"[s{idx}_{k}]" for k in range(n_segs)]
            scale_parts.append(f"[{base_label}]split={n_segs}{''.join(out_labels)}")
            file_split_labels[idx] = [lbl.strip("[]") for lbl in out_labels]
        else:
            file_split_labels[idx] = [base_label]

        file_split_counter[idx] = 0

    # --- Step 3: Per-segment trim from scaled/split stream ---
    parts_v: list[str] = []
    parts_a: list[str] = []
    seg_count = 0

    for seg in timeline:
        idx = seg.input_index
        s = seg.start_in_file
        e = seg.end_in_file

        # Pick the correct source label for this segment
        src_k = file_split_counter[idx]
        src_label = file_split_labels[idx][src_k]
        file_split_counter[idx] = src_k + 1

        if seg.state == "DYNAMIC":
            dur = e - s
            parts_v.append(
                f"[{src_label}]trim=start={s:.3f}:end={e:.3f},setpts=PTS-STARTPTS[v{seg_count}]"
            )
            if input_has_audio.get(idx, False):
                parts_a.append(
                    f"[{idx}:a]atrim=start={s:.3f}:end={e:.3f},asetpts=PTS-STARTPTS,"
                    f"aformat=sample_rates={audio_sample_rate}[a{seg_count}]"
                )
            else:
                parts_a.append(
                    f"anullsrc=r={audio_sample_rate}:cl=mono:d={dur:.3f}[a{seg_count}]"
                )
        else:
            dur = e - s
            # Dynamic speed factor calculation to prevent flashes
            target_display_dur = max(dur / global_speed_factor, min_static_display_duration)
            target_display_dur = min(target_display_dur, dur)  # Cannot be slower than 1x
            seg_speed_factor = dur / target_display_dur if target_display_dur > 0 else 1.0

            if use_keyframe_slideshow:
                seg_kf_interval = seg_speed_factor * display_dur
                parts_v.append(
                    f"[{src_label}]trim=start={s:.3f}:end={e:.3f},"
                    f"setpts=(PTS-STARTPTS)/{seg_speed_factor:.1f},"
                    f"fps=fps={output_fps}[v{seg_count}]"
                )
            else:
                parts_v.append(
                    f"[{src_label}]trim=start={s:.3f}:end={e:.3f},"
                    f"setpts=(PTS-STARTPTS)/{seg_speed_factor:.1f}[v{seg_count}]"
                )
            parts_a.append(
                f"anullsrc=r={audio_sample_rate}:cl=mono:d={target_display_dur:.3f}[a{seg_count}]"
            )

        seg_count += 1

    labels = "".join(f"[v{i}][a{i}]" for i in range(seg_count))
    all_parts = scale_parts + parts_v + parts_a
    concat = f"{';'.join(all_parts)};{labels}concat=n={seg_count}:v=1:a=1[v_tmp][a_tmp];[v_tmp]fps={output_fps}[v];[a_tmp]aresample={audio_sample_rate}[a]"

    return concat
