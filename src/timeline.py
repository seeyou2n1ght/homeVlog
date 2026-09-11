import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.database import VlogDatabase
from src.segment import Segment, segments_from_json, merge_cross_file, split_segments_at_file_boundaries
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


@dataclass
class SpeedRampInfo:
    has_ramp_in: bool
    has_ramp_out: bool
    ramp_in_src_dur: float
    ramp_out_src_dur: float
    cruise_src_dur: float
    v_fast: float
    target_display_dur: float
    effective_speed: float
    pts_expr: str


def calculate_speed_ramping_curve(
    dur: float,
    v_fast: float,
    has_ramp_in: bool,
    has_ramp_out: bool,
    ramp_duration_s: float = 1.0,
    target_display_dur: float | None = None,
) -> SpeedRampInfo:
    """
    Compute non-linear PTS speed ramping and easing curve parameters with exact display duration conservation.
    
    Mathematical Model:
    - target_disp_dur: Desired duration in the resulting Vlog (default: dur / v_fast).
    - In the display timeline, ramp-in and ramp-out durations are budgeted as d_in and d_out:
      d_ramp = min(ramp_duration_s * 0.5, target_disp_dur * 0.25).
    - In FFmpeg setpts, PTS_out = d(s) where s = T - STARTT.
      Accelerating (ramp-in): d(s) = s * (1.0 - k_in * s) with k_in = (1 - 1/v_c) / (2 * s_in).
      At boundary s = s_in, d(s_in) = d_in = s_in * (v_c + 1) / (2 * v_c).
      Hence s_in = d_in * 2 * v_c / (v_c + 1).
    - Source conservation: dur = s_in + s_cruise + s_out = (d_in + d_out) * 2 * v_c / (v_c + 1) + d_c * v_c.
      This forms an exact quadratic equation for cruising speed v_c:
      d_c * v_c^2 + (d_c + 2 * d_ramp - dur) * v_c - dur = 0.
    - Solving for v_c ensures total output duration is 100.00% invariant, eliminating segment dilation.
    """
    v_fast = max(1.0, v_fast)
    if target_display_dur is None:
        target_display_dur = dur / v_fast if v_fast > 0 else dur
    target_disp_dur = max(0.05, min(target_display_dur, dur))

    if not has_ramp_in and not has_ramp_out:
        eff_speed = dur / target_disp_dur if target_disp_dur > 0 else 1.0
        return SpeedRampInfo(
            has_ramp_in=False,
            has_ramp_out=False,
            ramp_in_src_dur=0.0,
            ramp_out_src_dur=0.0,
            cruise_src_dur=dur,
            v_fast=eff_speed,
            target_display_dur=target_disp_dur,
            effective_speed=eff_speed,
            pts_expr=f"(PTS-STARTPTS)/{eff_speed:.4f}",
        )

    # 缓入/缓出在成片展示时间轴上的预算
    max_ramp_disp = min(ramp_duration_s * 0.5, target_disp_dur * 0.25)
    d_in = max_ramp_disp if has_ramp_in else 0.0
    d_out = max_ramp_disp if has_ramp_out else 0.0
    d_ramp = d_in + d_out
    d_c = max(0.0, target_disp_dur - d_ramp)

    if d_c <= 1e-6 or d_ramp <= 1e-6:
        v_c = max(1.0, dur / target_disp_dur)
    else:
        A = d_c
        B = d_c + 2.0 * d_ramp - dur
        C = -dur
        disc = B * B - 4.0 * A * C
        if disc < 0:
            v_c = max(1.0, dur / target_disp_dur)
        else:
            v_c = max(1.0, (-B + math.sqrt(disc)) / (2.0 * A))

    s_in = d_in * 2.0 * v_c / (v_c + 1.0) if d_in > 0 else 0.0
    s_out = d_out * 2.0 * v_c / (v_c + 1.0) if d_out > 0 else 0.0
    s_cruise = max(0.0, dur - s_in - s_out)

    inv_vc = 1.0 / v_c if v_c > 0 else 1.0
    k_in = (1.0 - inv_vc) / (2.0 * s_in) if s_in > 0 else 0.0
    k_out = (1.0 - inv_vc) / (2.0 * s_out) if s_out > 0 else 0.0
    s_mid = s_in + s_cruise
    p_in = d_in
    p_mid = d_in + (s_cruise / v_c if v_c > 0 else 0.0)
    actual_disp_dur = p_mid + d_out
    eff_speed = dur / actual_disp_dur if actual_disp_dur > 0 else 1.0

    if s_in > 0 and s_out > 0:
        pts_expr = (
            f"if(lt(T-STARTT,{s_in:.3f}),(T-STARTT)*(1.0-{k_in:.6f}*(T-STARTT)),"
            f"if(lt(T-STARTT,{s_mid:.3f}),{p_in:.4f}+(T-STARTT-{s_in:.3f})*{inv_vc:.6f},"
            f"{p_mid:.4f}+(T-STARTT-{s_mid:.3f})*({inv_vc:.6f}+{k_out:.6f}*(T-STARTT-{s_mid:.3f}))))/TB"
        )
    elif s_in > 0:
        pts_expr = (
            f"if(lt(T-STARTT,{s_in:.3f}),(T-STARTT)*(1.0-{k_in:.6f}*(T-STARTT)),"
            f"{p_in:.4f}+(T-STARTT-{s_in:.3f})*{inv_vc:.6f})/TB"
        )
    elif s_out > 0:
        pts_expr = (
            f"if(lt(T-STARTT,{s_mid:.3f}),(T-STARTT)*{inv_vc:.6f},"
            f"{p_mid:.4f}+(T-STARTT-{s_mid:.3f})*({inv_vc:.6f}+{k_out:.6f}*(T-STARTT-{s_mid:.3f})))/TB"
        )
    else:
        pts_expr = f"(PTS-STARTPTS)/{v_c:.4f}"

    return SpeedRampInfo(
        has_ramp_in=has_ramp_in,
        has_ramp_out=has_ramp_out,
        ramp_in_src_dur=s_in,
        ramp_out_src_dur=s_out,
        cruise_src_dur=s_cruise,
        v_fast=v_c,
        target_display_dur=actual_disp_dur,
        effective_speed=eff_speed,
        pts_expr=pts_expr,
    )


def compute_display_plans(
    timeline: list[TimelineSegment],
    static_keyframe_interval: float = 30.0,
    keyframe_display_duration: float = 0.5,
    min_static_display_duration: float = 1.5,
    max_static_display_duration: float | None = None,
    speed_ramping: bool = True,
    ramp_duration_s: float = 1.0,
) -> list[tuple[float, SpeedRampInfo | None]]:
    """计算每个 TimelineSegment 的成片展示时长，与 build_concat_filter 严格同源。

    返回 [(display_dur, ramp_info_or_None), ...]：
    - DYNAMIC/DYNAMIC_AUDIO: 展示时长 == 源时长，ramp_info 为 None；
    - STATIC: 按全局抽帧倍率压缩，受 min/max 双向边界约束；若相邻动态段且启用变速则返回 ramp_info。
    """
    kf_interval = max(static_keyframe_interval, 1.0)
    display_dur = max(keyframe_display_duration, 0.1)
    global_speed_factor = kf_interval / display_dur

    plans: list[tuple[float, SpeedRampInfo | None]] = []
    n = len(timeline)
    for i, seg in enumerate(timeline):
        dur = seg.end_in_file - seg.start_in_file
        if seg.state in ("DYNAMIC", "DYNAMIC_AUDIO"):
            plans.append((dur, None))
            continue
        target = max(dur / global_speed_factor, min_static_display_duration)
        target = min(target, dur)
        if max_static_display_duration is not None and max_static_display_duration > 0:
            target = min(target, max_static_display_duration)

        v_fast = dur / target if target > 0 else 1.0
        has_in = i > 0 and timeline[i - 1].state in ("DYNAMIC", "DYNAMIC_AUDIO")
        has_out = i < n - 1 and timeline[i + 1].state in ("DYNAMIC", "DYNAMIC_AUDIO")
        if speed_ramping and (has_in or has_out):
            info = calculate_speed_ramping_curve(
                dur=dur, v_fast=v_fast,
                has_ramp_in=has_in, has_ramp_out=has_out,
                ramp_duration_s=ramp_duration_s,
                target_display_dur=target,
            )
            plans.append((info.target_display_dur, info))
        else:
            plans.append((target, None))
    return plans


def src_offset_at_display(
    d: float,
    src_dur: float,
    disp_dur: float,
    ramp_info: SpeedRampInfo | None = None,
) -> float:
    """将成片展示时间偏移 d 逆映射回源片段时间偏移（支持变速 ramping 曲线）。"""
    if disp_dur <= 0:
        return 0.0
    if ramp_info is None:
        return d * (src_dur / disp_dur)

    v_fast = max(1.0, ramp_info.v_fast)
    inv_v = 1.0 / v_fast
    s_in = ramp_info.ramp_in_src_dur
    s_out = ramp_info.ramp_out_src_dur
    s_mid = s_in + ramp_info.cruise_src_dur
    k_in = (1.0 - inv_v) / (2.0 * s_in) if s_in > 0 else 0.0
    k_out = (1.0 - inv_v) / (2.0 * s_out) if s_out > 0 else 0.0
    p_in = s_in * (1.0 + inv_v) / 2.0 if s_in > 0 else 0.0
    p_mid = p_in + (ramp_info.cruise_src_dur / v_fast if ramp_info.cruise_src_dur > 0 else 0.0)

    if s_in > 0 and d < p_in and k_in > 0:
        # d = s - k_in * s^2  →  求逆（小根）
        disc = max(0.0, 1.0 - 4.0 * k_in * d)
        return (1.0 - math.sqrt(disc)) / (2.0 * k_in)
    if d < p_mid:
        return s_in + (d - p_in) * v_fast
    # ramp-out: d - p_mid = u * inv_v + k_out * u^2
    dp = d - p_mid
    if k_out > 0:
        u = (-inv_v + math.sqrt(inv_v * inv_v + 4.0 * k_out * dp)) / (2.0 * k_out)
    else:
        u = dp * v_fast
    return s_mid + u


def build_timecode_drawtext_filter(
    start_unix: float,
    speed_factor: float = 1.0,
    font_size: int = 24,
    font_color: str = "white",
    font_file: str | None = None,
    x: str = "w-tw-30",
    y: str = "30",
    box: bool = True,
    box_color: str = "black@0.5",
    box_border_w: int = 5,
) -> str:
    """
    Build an FFmpeg drawtext filter string to burn surveillance wall-clock timecode into video.
    """
    base_ts = int(start_unix)
    box_flag = 1 if box else 0
    font_opt = ""
    if font_file:
        escaped_font = str(font_file).replace("\\", "/").replace(":", "\\:")
        font_opt = f"fontfile='{escaped_font}':"

    return (
        f"drawtext={font_opt}text='%{{pts\\:localtime\\:{base_ts}\\:%Y-%m-%d %H\\\\\\:%M\\\\\\:%S}}':"
        f"x={x}:y={y}:fontsize={font_size}:fontcolor={font_color}:"
        f"box={box_flag}:boxcolor={box_color}:boxborderw={box_border_w}"
    )


def _format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hrs = total_ms // 3600000
    total_ms %= 3600000
    mins = total_ms // 60000
    total_ms %= 60000
    secs = total_ms // 1000
    ms = total_ms % 1000
    return f"{hrs:02d}:{mins:02d}:{secs:02d},{ms:03d}"


def _format_ass_timestamp(seconds: float) -> str:
    total_cs = int(round(seconds * 100))
    hrs = total_cs // 360000
    total_cs %= 360000
    mins = total_cs // 6000
    total_cs %= 6000
    secs = total_cs // 100
    cs = total_cs % 100
    return f"{hrs:d}:{mins:02d}:{secs:02d}.{cs:02d}"


def generate_timecode_subtitles(
    timeline: list[TimelineSegment],
    rows: list[dict] | None = None,
    base_date: str = "20260901",
    step_s: float = 1.0,
    format_type: str = "srt",
) -> str:
    """
    Generate an SRT/ASS subtitle string mapping presentation timestamps
    to real-world surveillance wall-clock timestamps (YYYY-MM-DD HH:MM:SS).
    """
    if not timeline:
        return ""

    import re
    rows_dict = {r["filepath"]: r for r in (rows or [])}
    day_start_unix = ts_to_unix(base_date + "000000")

    # 展示时长计划与渲染滤镜图严格同源，含 speed ramping 非线性映射
    cfg = load_config()
    seg_cfg = cfg.get("segment", {})
    render_cfg = cfg.get("render", {})
    plans = compute_display_plans(
        timeline,
        static_keyframe_interval=seg_cfg.get("static_keyframe_interval", 30.0),
        keyframe_display_duration=seg_cfg.get("keyframe_display_duration", 0.5),
        min_static_display_duration=seg_cfg.get("min_static_display_duration", 1.5),
        max_static_display_duration=seg_cfg.get("max_static_display_duration", 2.0),
        speed_ramping=render_cfg.get("speed_ramping_enabled", True),
        ramp_duration_s=float(render_cfg.get("ramp_duration_s", 1.0)),
    )

    is_ass = format_type.lower() in ("ass", "ssa")
    entries = []
    entry_idx = 1
    cur_out_time = 0.0

    if is_ass:
        entries.extend([
            "[Script Info]",
            "ScriptType: v4.00+",
            "PlayResX: 1920",
            "PlayResY: 1080",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
            "Style: Default,Arial,32,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,9,30,30,30,1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
        ])

    for seg, (seg_disp_dur, ramp_info) in zip(timeline, plans):
        row = rows_dict.get(seg.filepath)
        if row and row.get("file_start_time"):
            file_start_unix = ts_to_unix(row["file_start_time"])
        else:
            m = re.search(r"\d{2}_(\d{14})_\d{14}", seg.filepath)
            if m:
                file_start_unix = ts_to_unix(m.group(1))
            else:
                file_start_unix = day_start_unix

        seg_wall_start = file_start_unix + seg.start_in_file
        seg_src_dur = seg.end_in_file - seg.start_in_file

        n_steps = max(1, int(math.ceil(seg_disp_dur / step_s)))
        for k in range(n_steps):
            d0 = k * step_s
            t0 = cur_out_time + d0
            t1 = min(cur_out_time + (k + 1) * step_s, cur_out_time + seg_disp_dur)
            if t1 <= t0:
                continue

            src_offset = src_offset_at_display(d0, seg_src_dur, seg_disp_dur, ramp_info)
            wall_unix = seg_wall_start + src_offset
            wall_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(wall_unix))

            if is_ass:
                t0_str = _format_ass_timestamp(t0)
                t1_str = _format_ass_timestamp(t1)
                entries.append(f"Dialogue: 0,{t0_str},{t1_str},Default,,0,0,0,,{wall_str}")
            else:
                t0_str = _format_srt_timestamp(t0)
                t1_str = _format_srt_timestamp(t1)
                entries.append(f"{entry_idx}\n{t0_str} --> {t1_str}\n{wall_str}\n")
                entry_idx += 1

        cur_out_time += seg_disp_dur

    return "\n".join(entries)


def save_timecode_subtitles(
    timeline: list[TimelineSegment] | str,
    output_path: Path | str,
    rows: list[dict] | None = None,
    base_date: str = "20260901",
    step_s: float = 1.0,
    format_type: str = "srt",
) -> Path:
    """Save generated real-world timecode subtitles to a file."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(timeline, str):
        content = timeline
    else:
        content = generate_timecode_subtitles(
            timeline=timeline, rows=rows, base_date=base_date, step_s=step_s, format_type=format_type
        )
    out_p.write_text(content, encoding="utf-8")
    return out_p


def partition_timeline_by_batches(
    timeline: list[TimelineSegment] | list[Segment],
    batch_max_files: int = 8,
) -> list[list[Any]]:
    """
    Partition timeline segments into sequential batches, ensuring no batch
    exceeds batch_max_files distinct source files.
    """
    if not timeline:
        return []

    batches: list[list[Any]] = []
    cur_batch: list[Any] = []
    cur_files: set[str] = set()

    for seg in timeline:
        fp = seg.filepath if hasattr(seg, "filepath") else seg.source_file
        if fp not in cur_files and len(cur_files) >= batch_max_files and cur_batch:
            batches.append(cur_batch)
            cur_batch = []
            cur_files = set()
        cur_batch.append(seg)
        cur_files.add(fp)

    if cur_batch:
        batches.append(cur_batch)

    for batch in batches:
        batch_files = list(dict.fromkeys(
            s.filepath if hasattr(s, "filepath") else getattr(s, "source_file", "")
            for s in batch
        ))
        f_to_idx = {f: i for i, f in enumerate(batch_files)}
        for s in batch:
            fp = s.filepath if hasattr(s, "filepath") else getattr(s, "source_file", "")
            if hasattr(s, "input_index"):
                s.input_index = f_to_idx.get(fp, 0)

    return batches


def build_timeline_from_rows(
    rows: list[dict],
    date: str,
    target_files: set[str] | list[str] | None = None,
    config: dict | None = None,
) -> list[TimelineSegment]:
    """
    基于给定的数据库任务行集合构建时间轴片段，支持按 target_files 精准局部过滤。
    """
    if config is None:
        config = load_config()

    day_start = ts_to_unix(date + "000000")

    all_segments: list[Segment] = []
    files_meta: list[dict] = []
    for row in rows:
        file_start_unix = ts_to_unix(row["file_start_time"])
        file_end_unix = ts_to_unix(row["file_end_time"])
        day_offset = max(file_start_unix - day_start, 0.0)
        file_dur = (
            float(row["file_duration"])
            if (row.get("file_duration") is not None and float(row["file_duration"]) > 0)
            else max(file_end_unix - file_start_unix, 0.0)
        )
        file_offset = day_offset

        if row.get("prescreen_status") == "STATIC":
            file_end_offset = file_offset + file_dur
            files_meta.append({
                "filepath": row["filepath"],
                "file_start_offset": file_offset,
                "file_end_offset": file_end_offset,
                "duration": file_dur,
            })
            all_segments.append(Segment(
                start_time=file_offset, end_time=file_end_offset,
                state="STATIC",
                source_file=row["filepath"],
                file_start_offset=file_offset,
            ))
        elif row.get("segments"):
            segs = []
            for s_rec in row["segments"]:
                # 人工审核绝对优先 (AGENTS.md 铁律): human_label 严格高于 algo_status (FP 强制重置为静态，FN 强制重置为动态)
                manual = str(s_rec.get("manual_label") or s_rec.get("human_label") or "").strip().upper()
                if manual in ("CONFIRMED_MOTION", "MISSED_MOTION", "VERIFIED_MOTION", "FN", "TP", "DYNAMIC"):
                    eff_state = "DYNAMIC"
                elif manual in ("FALSE_ALARM", "CONFIRMED_STATIC", "FP", "TN", "STATIC"):
                    eff_state = "STATIC"
                else:
                    eff_state = s_rec.get("state", "STATIC")

                segs.append(Segment(
                    start_time=float(s_rec["start_time"]),
                    end_time=float(s_rec["end_time"]),
                    state=eff_state,
                    source_file=row["filepath"],
                    file_start_offset=float(s_rec.get("file_start_offset", file_offset)),
                    max_energy=float(s_rec.get("max_energy", 0.0) or 0.0),
                    avg_confidence=float(s_rec.get("avg_confidence", 0.0) or 0.0),
                ))

            if not segs:
                file_end_offset = file_offset + file_dur
                files_meta.append({
                    "filepath": row["filepath"],
                    "file_start_offset": file_offset,
                    "file_end_offset": file_end_offset,
                    "duration": file_dur,
                })
                all_segments.append(Segment(
                    start_time=file_offset, end_time=file_end_offset,
                    state="STATIC",
                    source_file=row["filepath"],
                    file_start_offset=file_offset,
                ))
            else:
                if segs[0].file_start_offset is not None and segs[0].file_start_offset >= 0:
                    file_offset = segs[0].file_start_offset
                if file_dur <= 0 and segs:
                    file_dur = max(s.end_time for s in segs) - file_offset
                file_end_offset = file_offset + file_dur
                files_meta.append({
                    "filepath": row["filepath"],
                    "file_start_offset": file_offset,
                    "file_end_offset": file_end_offset,
                    "duration": file_dur,
                })
                for s in segs:
                    s.source_file = row["filepath"]
                all_segments.extend(segs)
        elif row.get("analysis_segments"):
            segs = segments_from_json(row["analysis_segments"])
            if not segs:
                file_end_offset = file_offset + file_dur
                files_meta.append({
                    "filepath": row["filepath"],
                    "file_start_offset": file_offset,
                    "file_end_offset": file_end_offset,
                    "duration": file_dur,
                })
                all_segments.append(Segment(
                    start_time=file_offset, end_time=file_end_offset,
                    state="STATIC",
                    source_file=row["filepath"],
                    file_start_offset=file_offset,
                ))
            else:
                if segs[0].file_start_offset is not None and segs[0].file_start_offset >= 0:
                    file_offset = segs[0].file_start_offset
                if file_dur <= 0 and segs:
                    file_dur = max(s.end_time for s in segs) - file_offset
                file_end_offset = file_offset + file_dur
                files_meta.append({
                    "filepath": row["filepath"],
                    "file_start_offset": file_offset,
                    "file_end_offset": file_end_offset,
                    "duration": file_dur,
                })
                for s in segs:
                    s.source_file = row["filepath"]
                all_segments.extend(segs)
        else:
            file_end_offset = file_offset + file_dur
            files_meta.append({
                "filepath": row["filepath"],
                "file_start_offset": file_offset,
                "file_end_offset": file_end_offset,
                "duration": file_dur,
            })

    all_segments.sort(key=lambda s: s.start_time)
    seg_cfg = config.get("segment", {})
    gap_tolerance = seg_cfg.get("gap_tolerance", 1.5)
    min_motion_dur = seg_cfg.get("min_motion_duration", 2.0)
    min_static_dur = seg_cfg.get("min_static_duration", 8.0)

    # 全局跨文件平滑与合并，解决边界截断问题
    merged = merge_cross_file(all_segments, gap_tolerance)
    from src.segment import _filter_short
    filtered = _filter_short(merged, min_motion_dur, min_static_dur, gap_tolerance)

    # 长静止段宏观折叠（Macro-collapsing）：夜间/长时间无人静止段下采样，避免生成无意义长视频
    reviews = [r for row in rows for r in row.get("human_reviews", [])]
    macro_collapse_enabled = seg_cfg.get("macro_collapse_static", True) and not reviews
    if macro_collapse_enabled:
        collapsed_filtered = []
        for s in filtered:
            if s.state == "STATIC":
                overlapping = [
                    fm for fm in files_meta 
                    if fm["file_end_offset"] > s.start_time and fm["file_start_offset"] < s.end_time
                ]
                if len(overlapping) > 3:
                    step = max(2, len(overlapping) // 4)
                    selected_indices = set([0, len(overlapping) - 1] + list(range(0, len(overlapping), step)))
                    for idx, fm in enumerate(overlapping):
                        if idx in selected_indices:
                            st = max(s.start_time, fm["file_start_offset"])
                            et = min(s.end_time, fm["file_end_offset"])
                            if et > st:
                                collapsed_filtered.append(Segment(
                                    start_time=st, end_time=et, state="STATIC",
                                    source_file=fm["filepath"], file_start_offset=fm["file_start_offset"]
                                ))
                    continue
            collapsed_filtered.append(s)
        filtered = collapsed_filtered

    # 严格按物理文件边界切分，防止跨文件批次渲染时超出物理文件时长
    split_segs = split_segments_at_file_boundaries(filtered, files_meta)
    from src.feedback import overlay_reviews
    split_segs = overlay_reviews(split_segs, reviews)

    # 若指定 target_files，则快速局部过滤出目标文件的切片
    target_set = set(target_files) if target_files is not None else None
    if target_set is not None:
        split_segs = [s for s in split_segs if s.source_file in target_set]

    unique_files = list(dict.fromkeys(s.source_file for s in split_segs))
    file_to_idx = {f: i for i, f in enumerate(unique_files)}

    min_seg_dur = seg_cfg.get("min_segment_duration", 0.1)
    timeline: list[TimelineSegment] = []
    for s in split_segs:
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

    return timeline


def build_timeline(db: VlogDatabase, date: str, cam_index: int) -> list[TimelineSegment]:
    config = load_config()
    rows = db.get_all_file_tasks_for_date(date, cam_index)
    try:
        segments_list = db.get_all_segments_for_date(date, cam_index)
        segs_by_file_id: dict[int, list[dict]] = {}
        for s in segments_list:
            segs_by_file_id.setdefault(s["file_id"], []).append(s)
        for r in rows:
            if r["id"] in segs_by_file_id:
                r["segments"] = segs_by_file_id[r["id"]]
    except Exception as e:
        logger.debug(f"Could not load relational segments: {e}")

    timeline = build_timeline_from_rows(rows, date, config=config)
    logger.debug(
        "timeline for %s cam%d: %d segments",
        date, cam_index, len(timeline),
    )
    return timeline



def build_concat_filter(
    timeline: list[TimelineSegment],
    rows: list[dict],
    output_fps: int = 20,
    output_width: int = 1920,
    output_height: int = 1080,
    static_keyframe_interval: float = 30.0,
    keyframe_display_duration: float = 0.5,
    min_static_display_duration: float = 1.5,
    max_static_display_duration: float | None = None,
    audio_sample_rate: int = 48000,
    gap_tolerance: float = 0.5,
    scale_mode: str = "cpu",
    speed_ramping: bool | None = None,
    ramp_duration_s: float | None = None,
    audio_fade_duration_s: float | None = None,
    timecode_osd: bool | None = None,
    timecode_font_size: int = 24,
    timecode_font_color: str = "white",
    base_date: str | None = None,
    preselected_static: bool = False,
    concat_demuxer: bool = False,
    simple_dynamic: bool = False,
    sparse_mixed: bool = False,
    static_sample_window_s: float = 0.25,
    audio_input_offset: int = 0,
    source_timeline: list[TimelineSegment] | None = None,
) -> str:
    """
    Build a complex FFmpeg filtergraph string for the entire timeline.
    Supports:
    - Non-linear speed ramping across static/dynamic transition zones.
    - Audio linear cross-fading (afade in/out) on dynamic boundaries.
    - Real-world surveillance wall-clock timecode OSD overlay (drawtext).
    """
    kf_interval = max(static_keyframe_interval, 1.0)

    tmp_cfg = {}
    try:
        tmp_cfg = load_config()
    except Exception:
        pass

    render_cfg = tmp_cfg.get("render", {})
    seg_cfg = tmp_cfg.get("segment", {})

    if speed_ramping is None:
        speed_ramping = False
    if ramp_duration_s is None:
        ramp_duration_s = float(render_cfg.get("ramp_duration_s", seg_cfg.get("ramp_duration_s", 1.0)))
    if audio_fade_duration_s is None:
        audio_fade_duration_s = float(render_cfg.get("audio_fade_duration_s", seg_cfg.get("audio_fade_duration_s", 0.15)))
    if timecode_osd is None:
        timecode_osd = False

    if scale_mode == "cuda":
        scale_core = f"hwupload_cuda,scale_cuda={output_width}:{output_height},hwdownload,format=nv12"
    elif scale_mode == "cuda_passthrough":
        scale_core = f"scale_cuda={output_width}:{output_height},hwdownload,format=nv12"
    elif scale_mode == "qsv":
        scale_core = f"scale_qsv=w={output_width}:h={output_height},hwdownload,format=nv12"
    elif scale_mode == "skip":
        scale_core = "null"
    else:
        scale_core = f"scale={output_width}:{output_height}"
    # A virtual concat input already contains only the source ranges that must
    # be rendered.  Keep its sparse static timestamps intact until the
    # per-segment speed transform; an early fps filter would expand the gaps
    # back to full-frame video and erase the decode saving.
    scale_filter = scale_core if preselected_static else f"{scale_core},fps={output_fps}"

    if simple_dynamic and len(timeline) == 1 and timeline[0].state in ("DYNAMIC", "DYNAMIC_AUDIO"):
        seg = timeline[0]
        dur = max(0.04, seg.end_in_file - seg.start_in_file)
        row = next((r for r in rows if r.get("filepath") == seg.filepath), None)
        has_audio = bool(row and row.get("has_audio"))
        video = (
            f"[0:v]trim=start={seg.start_in_file:.3f}:end={seg.end_in_file:.3f},"
            f"setpts=PTS-STARTPTS,{scale_core},fps={output_fps}[v]"
        )
        if has_audio:
            audio = (
                f"[0:a]atrim=start={seg.start_in_file:.3f}:end={seg.end_in_file:.3f},"
                f"asetpts=PTS-STARTPTS,aformat=sample_rates={audio_sample_rate}[a]"
            )
        else:
            audio = f"anullsrc=r={audio_sample_rate}:cl=mono:d={dur:.3f}[a]"
        return f"{video};{audio}"


    use_keyframe_slideshow = (scale_mode == "cpu")
    if render_cfg.get("static_mode") == "hybrid_keyframe":
        use_keyframe_slideshow = True

    # --- Step 1: Count segments per input file ---
    segs_per_file: dict[int, int] = {}
    input_files: dict[int, str] = {}
    for seg in timeline:
        segs_per_file[seg.input_index] = segs_per_file.get(seg.input_index, 0) + 1
        input_files[seg.input_index] = seg.filepath

    input_has_audio: dict[int, bool] = {}
    for idx, filepath in input_files.items():
        row = next((r for r in rows if r["filepath"] == filepath), None)
        if row and row.get("has_audio") is not None:
            input_has_audio[idx] = bool(row["has_audio"])
        else:
            input_has_audio[idx] = False

    # --- Step 2: Per-file scale (once per input) ---
    # 静态段关键帧抽取快路径 (hybrid_keyframe 的真实含义):
    # 当某输入文件在本批次内全部为 STATIC 段且各段时长 >= 2*kf_interval 时，
    # 在解码侧以 select 按 kf_interval 抽帧，置于 scale/hwdownload 之前——
    # 仅被选中的帧进入缩放与显存回下载，消除静态段全帧解码的渲染瓶颈。
    # 混合文件按区间分支；短静态段维持全帧链，保证 trim 区间必有帧可用。
    segs_by_file: dict[int, list[TimelineSegment]] = {}
    for seg in timeline:
        segs_by_file.setdefault(seg.input_index, []).append(seg)

    scale_parts: list[str] = []
    file_split_labels: dict[int, list[str]] = {}
    file_split_counter: dict[int, int] = {}
    n_keyframe_fast = 0

    for idx in sorted(segs_per_file):
        n_segs = segs_per_file[idx]
        file_segs = segs_by_file.get(idx, [])
        all_static = all(s.state not in ("DYNAMIC", "DYNAMIC_AUDIO") for s in file_segs)
        min_src_dur = min((s.end_in_file - s.start_in_file) for s in file_segs) if file_segs else 0.0
        use_kf_fastpath = (
            not preselected_static
            and all_static
            and min_src_dur >= 2.0 * kf_interval
        )
        has_dynamic = any(s.state in ("DYNAMIC", "DYNAMIC_AUDIO") for s in file_segs)
        has_static = any(s.state not in ("DYNAMIC", "DYNAMIC_AUDIO") for s in file_segs)
        use_sparse_mixed = sparse_mixed and has_dynamic and has_static

        input_v = f"[{idx}:v]"

        if use_sparse_mixed:
            sample_half = max(0.04, float(static_sample_window_s) * 0.5)
            raw_intervals: list[tuple[float, float]] = []
            for file_seg in file_segs:
                if file_seg.state in ("DYNAMIC", "DYNAMIC_AUDIO"):
                    s = max(0.0, file_seg.start_in_file - 0.05)
                    e = file_seg.end_in_file + 0.05
                else:
                    midpoint = (file_seg.start_in_file + file_seg.end_in_file) * 0.5
                    s = max(file_seg.start_in_file, midpoint - sample_half)
                    e = min(file_seg.end_in_file, midpoint + sample_half)
                if e > s:
                    raw_intervals.append((s, e))
            raw_intervals.sort(key=lambda x: x[0])

            merged_intervals: list[tuple[float, float]] = []
            for s, e in raw_intervals:
                if merged_intervals and s <= merged_intervals[-1][1] + 1.0:
                    merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], e))
                else:
                    merged_intervals.append((s, e))

            # 当碎片区间过多时(>30)，跳帧开销超越线性解码且易导致 FFmpeg 表达式解析爆栈，安全降级为常规全帧解码
            if len(merged_intervals) > 30:
                logger.info(
                    "sparse_mixed: %d intervals on file %d exceeds safety limit, fallback to continuous decode",
                    len(merged_intervals), idx,
                )
                if scale_filter is not None:
                    scale_parts.append(f"{input_v}{scale_filter}[scaled_{idx}]")
                else:
                    scale_parts.append(f"{input_v}null[skip_{idx}]")
            else:
                keep_terms = [f"between(t\\,{s:.3f}\\,{e:.3f})" for s, e in merged_intervals]
                def _balance_add(terms: list[str]) -> str:
                    if len(terms) == 1:
                        return terms[0]
                    mid = len(terms) // 2
                    return f"({_balance_add(terms[:mid])}+{_balance_add(terms[mid:])})"
                select_expr = _balance_add(keep_terms)
                scale_parts.append(f"{input_v}select='{select_expr}',{scale_core}[scaled_{idx}]")
        elif use_kf_fastpath:
            # 快路径剥离文件链尾部 fps：稀疏关键帧直接进入段级 trim/setpts，
            # 段级 fps 过滤器负责将每帧铺陈为 keyframe_display_duration 时长
            sel = f"select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,{kf_interval:.1f})'"
            scale_parts.append(f"{input_v}{sel},{scale_core}[scaled_{idx}]")
            n_keyframe_fast += 1
        elif scale_filter is not None:
            scale_parts.append(f"{input_v}{scale_filter}[scaled_{idx}]")
        else:
            scale_parts.append(f"{input_v}null[skip_{idx}]")
        base_label = f"scaled_{idx}" if (scale_filter is not None or use_kf_fastpath) else f"skip_{idx}"

        if n_segs > 1:
            out_labels = [f"[s{idx}_{k}]" for k in range(n_segs)]
            scale_parts.append(f"[{base_label}]split={n_segs}{''.join(out_labels)}")
            file_split_labels[idx] = [lbl.strip("[]") for lbl in out_labels]
        else:
            file_split_labels[idx] = [base_label]

        file_split_counter[idx] = 0

    if n_keyframe_fast:
        logger.info(
            "keyframe fast-path: %d/%d input files use select-based static extraction (kf=%.0fs)",
            n_keyframe_fast, len(segs_per_file), kf_interval,
        )

    # --- Step 3: Per-segment trim from scaled/split stream ---
    # 展示时长计划与字幕墙钟映射共用同一来源，杜绝两套时长计算漂移
    source_timeline = source_timeline or timeline
    if len(source_timeline) != len(timeline):
        raise ValueError("source_timeline must align one-to-one with render timeline")
    display_plans = compute_display_plans(
        source_timeline,
        static_keyframe_interval=static_keyframe_interval,
        keyframe_display_duration=keyframe_display_duration,
        min_static_display_duration=min_static_display_duration,
        max_static_display_duration=max_static_display_duration,
        speed_ramping=bool(speed_ramping),
        ramp_duration_s=ramp_duration_s,
    )

    parts_v: list[str] = []
    parts_a: list[str] = []
    seg_count = 0

    rows_dict = {r["filepath"]: r for r in (rows or [])}

    for i, seg in enumerate(timeline):
        idx = seg.input_index
        audio_idx = idx + audio_input_offset
        s = seg.start_in_file
        e = seg.end_in_file
        source_seg = source_timeline[i]
        source_s = source_seg.start_in_file
        source_e = source_seg.end_in_file
        dur = source_e - source_s

        is_dynamic = seg.state in ("DYNAMIC", "DYNAMIC_AUDIO")

        src_k = file_split_counter[idx]
        src_label = file_split_labels[idx][src_k]
        file_split_counter[idx] = src_k + 1

        # Timecode OSD filter if enabled
        osd_filter_str = ""
        if timecode_osd:
            row = rows_dict.get(seg.filepath)
            file_start_unix = 0.0
            if row and row.get("file_start_time"):
                file_start_unix = ts_to_unix(row["file_start_time"])
            elif base_date:
                file_start_unix = ts_to_unix(base_date + "000000")
            seg_start_unix = file_start_unix + source_s
            default_font = "C:/Windows/Fonts/arial.ttf" if (os.name == "nt" and Path("C:/Windows/Fonts/arial.ttf").exists()) else None
            osd_filter = build_timecode_drawtext_filter(
                start_unix=seg_start_unix,
                font_size=timecode_font_size,
                font_color=timecode_font_color,
                font_file=default_font,
            )
            osd_filter_str = f",{osd_filter}"

        if is_dynamic:
            # Video: 1.0x PTS
            parts_v.append(f"[{src_label}]trim=start={s:.3f}:end={e:.3f}{osd_filter_str},setpts=PTS-STARTPTS[v{seg_count}]")

            # Audio: Linear cross-fade on cut boundaries
            if input_has_audio.get(idx, False):
                fade_d = min(audio_fade_duration_s, dur / 2.0) if audio_fade_duration_s > 0 else 0.0
                if fade_d > 0:
                    fade_out_st = max(0.0, dur - fade_d)
                    afade_str = f",afade=t=in:ss=0:d={fade_d:.3f},afade=t=out:st={fade_out_st:.3f}:d={fade_d:.3f}"
                else:
                    afade_str = ""
                parts_a.append(
                    f"[{audio_idx}:a]atrim=start={source_s:.3f}:end={source_e:.3f},asetpts=PTS-STARTPTS{afade_str},"
                    f"aformat=sample_rates={audio_sample_rate}[a{seg_count}]"
                )
            else:
                parts_a.append(f"anullsrc=r={audio_sample_rate}:cl=mono:d={dur:.3f}[a{seg_count}]")
        else:
            # Static segment speed scaling and speed ramping (与字幕映射同源)
            actual_display_dur, ramp_info = display_plans[i]
            if preselected_static:
                # Sparse representatives no longer span the source duration;
                # normalize their compact PTS and let tpad own the exact
                # display duration from the source timeline plan.
                pts_filter = "PTS-STARTPTS"
            elif ramp_info is not None:
                pts_filter = ramp_info.pts_expr.replace(",", "\\,")
            else:
                v_fast = dur / actual_display_dur if actual_display_dur > 0 else 1.0
                pts_filter = f"(PTS-STARTPTS)/{v_fast:.4f}"

            if preselected_static or use_sparse_mixed:
                # A sparse EDL may contain only one picture for a static
                # interval.  Clone its final frame to the exact display-plan
                # duration before the segment enters concat; otherwise FFmpeg
                # has no following timestamp from which to infer duration.
                fps_filter = (
                    f",tpad=stop_mode=clone:stop_duration={actual_display_dur:.3f}"
                    f",fps=fps={output_fps},trim=duration={actual_display_dur:.3f}"
                    ",setpts=PTS-STARTPTS"
                )
            else:
                fps_filter = f",fps=fps={output_fps}" if use_keyframe_slideshow else ""
            parts_v.append(
                f"[{src_label}]trim=start={s:.3f}:end={e:.3f}{osd_filter_str},"
                f"setpts={pts_filter}"
                f"{fps_filter}[v{seg_count}]"
            )
            parts_a.append(f"anullsrc=r={audio_sample_rate}:cl=mono:d={actual_display_dur:.3f}[a{seg_count}]")

        seg_count += 1

    labels = "".join(f"[v{i}][a{i}]" for i in range(seg_count))
    all_parts = scale_parts + parts_v + parts_a
    concat = (
        f"{';'.join(all_parts)};{labels}concat=n={seg_count}:v=1:a=1[v_tmp][a_tmp];"
        f"[v_tmp]fps={output_fps}[v];"
        f"[a_tmp]aresample={audio_sample_rate}:async=1000:first_pts=0,asetpts=N/SR/TB[a]"
    )

    return concat

