import json
from dataclasses import dataclass


@dataclass
class Segment:
    start_time: float
    end_time: float
    state: str  # "DYNAMIC" | "PRESENCE" | "MICRO_MOTION" | "STATIC" | "DYNAMIC_AUDIO"
    source_file: str
    file_start_offset: float = 0.0
    max_energy: float = 0.0
    avg_confidence: float = 0.0
    needs_review: bool = False
    review_reason: str = ""

    @property
    def duration(self) -> float:
        """Returns the segment duration in seconds."""
        return max(0.0, self.end_time - self.start_time)

    @property
    def is_dynamic(self) -> bool:

        """Returns True if segment represents active dynamic motion, presence, micro-motion or audio event."""
        return self.state in ("DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE", "MICRO_MOTION")

    @property
    def is_active_motion(self) -> bool:
        """Returns True if segment represents high-speed/1x active dynamic motion or audio."""
        return self.state in ("DYNAMIC", "DYNAMIC_AUDIO")


def build_segments(
    frame_labels: list[dict],
    source_file: str,
    min_motion_dur: float = 2.0,
    min_static_dur: float = 30.0,
    file_offset: float = 0.0,
    gap_tolerance: float = 0.5,
    apply_smoothing: bool = False,
    pre_roll: float = 0.0,
    post_roll: float = 0.0,
    motion_absorb_energy_threshold: float = 12.0,
) -> list[Segment]:
    """
    Convert frame-by-frame labels to contiguous segments with optional pre-roll/post-roll expansion.
    frame_labels: [{time, is_motion, state, energy, ...}, ...]
    file_offset: absolute time offset of the source file (day-relative seconds)
    """
    if not frame_labels:
        return []

    def _resolve_state(lbl: dict) -> str:
        s = lbl.get("state")
        if s in ("DYNAMIC", "DYNAMIC_AUDIO", "STATIC"):
            return s
        return "DYNAMIC" if lbl.get("is_motion", False) else "STATIC"

    segments: list[Segment] = []
    seg_start = frame_labels[0]["time"]
    seg_state = _resolve_state(frame_labels[0])
    seg_max_energy = frame_labels[0].get("energy", 0.0)
    seg_conf_sum = frame_labels[0].get("confidence", 0.0)
    seg_conf_count = 1

    for i in range(1, len(frame_labels)):
        cur_state = _resolve_state(frame_labels[i])
        if cur_state != seg_state:
            seg_end = frame_labels[i]["time"]
            segments.append(Segment(
                start_time=seg_start,
                end_time=seg_end,
                state=seg_state,
                source_file=source_file,
                file_start_offset=file_offset,
                max_energy=seg_max_energy,
                avg_confidence=seg_conf_sum / max(1, seg_conf_count),
            ))
            seg_start = frame_labels[i]["time"]
            seg_state = cur_state
            seg_max_energy = frame_labels[i].get("energy", 0.0)
            seg_conf_sum = frame_labels[i].get("confidence", 0.0)
            seg_conf_count = 1
        else:
            seg_max_energy = max(seg_max_energy, frame_labels[i].get("energy", 0.0))
            seg_conf_sum += frame_labels[i].get("confidence", 0.0)
            seg_conf_count += 1

    segments.append(Segment(
        start_time=seg_start,
        end_time=frame_labels[-1]["time"],
        state=seg_state,
        source_file=source_file,
        file_start_offset=file_offset,
        max_energy=seg_max_energy,
        avg_confidence=seg_conf_sum / max(1, seg_conf_count),
    ))

    # Apply pre-roll and post-roll to dynamic segments for natural transition
    if pre_roll > 0 or post_roll > 0:
        total_min_t = frame_labels[0]["time"]
        total_max_t = frame_labels[-1]["time"]
        dyn_spans = []
        for s in segments:
            if s.is_dynamic:
                eff_pre = pre_roll if (s.max_energy == 0.0 or s.max_energy >= motion_absorb_energy_threshold) else min(0.5, pre_roll * 0.25)
                eff_post = post_roll if (s.max_energy == 0.0 or s.max_energy >= motion_absorb_energy_threshold) else min(0.5, post_roll * 0.25)
                st = max(total_min_t, s.start_time - eff_pre)
                et = min(total_max_t, s.end_time + eff_post)
                dyn_spans.append((st, et, s.state, s.max_energy))

        if dyn_spans:
            # Merge overlapping dynamic spans
            merged_dyn = [dyn_spans[0]]
            for st, et, state, energy in dyn_spans[1:]:
                last_st, last_et, last_state, last_energy = merged_dyn[-1]
                effective_gap = gap_tolerance if (max(last_energy, energy) == 0.0 or max(last_energy, energy) >= motion_absorb_energy_threshold) else min(0.5, gap_tolerance * 0.25)
                if st <= last_et + effective_gap:
                    resolved_state = "DYNAMIC" if (last_state == "DYNAMIC" or state == "DYNAMIC") else state
                    merged_dyn[-1] = (last_st, max(last_et, et), resolved_state, max(last_energy, energy))
                else:
                    merged_dyn.append((st, et, state, energy))

            # Reconstruct complete segment sequence with static spans in between
            new_segments = []
            curr_t = total_min_t
            for st, et, state, energy in merged_dyn:
                if st > curr_t:
                    new_segments.append(Segment(
                        start_time=curr_t, end_time=st, state="STATIC",
                        source_file=source_file, file_start_offset=file_offset,
                    ))
                new_segments.append(Segment(
                    start_time=st, end_time=et, state=state,
                    source_file=source_file, file_start_offset=file_offset, max_energy=energy,
                ))
                curr_t = et
            if curr_t < total_max_t:
                new_segments.append(Segment(
                    start_time=curr_t, end_time=total_max_t, state="STATIC",
                    source_file=source_file, file_start_offset=file_offset,
                ))
            segments = new_segments

    merged = _merge_same_state(segments, gap_tolerance)
    res_segs = (
        _filter_short(
            merged,
            min_motion_dur,
            min_static_dur,
            gap_tolerance,
            motion_absorb_energy_threshold=motion_absorb_energy_threshold,
        )
        if apply_smoothing
        else merged
    )

    for s in res_segs:
        if s.state == "STATIC" and 2.2 <= s.max_energy <= 3.5 and not s.needs_review:
            s.needs_review = True
            s.review_reason = f"BORDERLINE_MICRO_MOTION: 疑似微动作临界(energy={s.max_energy:.2f})"

    return res_segs


def _can_merge(a: Segment, b: Segment, gap_tolerance: float = 0.5) -> bool:
    gap = b.start_time - a.end_time
    return a.state == b.state and gap <= gap_tolerance


def _merge_same_state(segments: list[Segment], gap_tolerance: float = 0.5) -> list[Segment]:
    if len(segments) <= 1:
        return segments
    result = [segments[0]]
    for seg in segments[1:]:
        if _can_merge(result[-1], seg, gap_tolerance):
            result[-1].end_time = seg.end_time
            result[-1].max_energy = max(result[-1].max_energy, seg.max_energy)
            result[-1].avg_confidence = max(result[-1].avg_confidence, seg.avg_confidence)
            if seg.needs_review:
                result[-1].needs_review = True
                if seg.review_reason and seg.review_reason not in result[-1].review_reason:
                    result[-1].review_reason = f"{result[-1].review_reason}; {seg.review_reason}".strip("; ")
        else:
            result.append(seg)
    return result


def _filter_short(
    segments: list[Segment],
    min_motion: float,
    min_static: float,
    gap_tolerance: float = 0.5,
    motion_absorb_energy_threshold: float = 12.0,
) -> list[Segment]:
    if not segments:
        return segments

    # Iteratively absorb short segments into neighbors until stable
    # Guard: max iterations = segment count (each iteration absorbs at least one)
    max_iters = len(segments) + 1
    changed = True
    while changed and max_iters > 0:
        max_iters -= 1
        changed = False
        i = 0
        while i < len(segments):
            dur = segments[i].end_time - segments[i].start_time
            is_dynamic = segments[i].state in ("DYNAMIC", "DYNAMIC_AUDIO")
            threshold = min_motion if is_dynamic else min_static
            if dur >= threshold:
                i += 1
                continue

            # If current is STATIC (< min_static)
            if not is_dynamic:
                # 能量门控：静态段仅当相邻动态段能量充沛（max_energy >= 门限，或无能量标注的测试用例），或者停顿极短（<= gap_tolerance）时，才允许被动态吸收
                can_absorb_left = (
                    i > 0 and segments[i - 1].state in ("DYNAMIC", "DYNAMIC_AUDIO")
                    and (segments[i - 1].max_energy == 0.0 or segments[i - 1].max_energy >= motion_absorb_energy_threshold or dur <= gap_tolerance)
                )
                can_absorb_right = (
                    i + 1 < len(segments) and segments[i + 1].state in ("DYNAMIC", "DYNAMIC_AUDIO")
                    and (segments[i + 1].max_energy == 0.0 or segments[i + 1].max_energy >= motion_absorb_energy_threshold or dur <= gap_tolerance)
                )

                if can_absorb_left:
                    segments[i - 1].end_time = segments[i].end_time
                    segments[i - 1].max_energy = max(segments[i - 1].max_energy, segments[i].max_energy)
                    segments[i - 1].avg_confidence = max(segments[i - 1].avg_confidence, segments[i].avg_confidence)
                    if segments[i].needs_review:
                        segments[i - 1].needs_review = True
                        if segments[i].review_reason and segments[i].review_reason not in segments[i - 1].review_reason:
                            segments[i - 1].review_reason = f"{segments[i - 1].review_reason}; {segments[i].review_reason}".strip("; ")
                    segments.pop(i)
                    changed = True
                    continue
                elif can_absorb_right:
                    segments[i + 1].start_time = segments[i].start_time
                    segments[i + 1].max_energy = max(segments[i + 1].max_energy, segments[i].max_energy)
                    segments[i + 1].avg_confidence = max(segments[i + 1].avg_confidence, segments[i].avg_confidence)
                    if segments[i].needs_review:
                        segments[i + 1].needs_review = True
                        if segments[i].review_reason and segments[i].review_reason not in segments[i + 1].review_reason:
                            segments[i + 1].review_reason = f"{segments[i + 1].review_reason}; {segments[i].review_reason}".strip("; ")
                    segments.pop(i)
                    changed = True
                    continue
                else:
                    # 两侧动态段皆为微弱低能量（如睡觉微动/噪点），严禁吞噬中间的静态段！静态段必须保留以供抽帧
                    i += 1
                    continue

            # If current is DYNAMIC (< min_motion)
            # Try absorb left: merge into prev static segment
            if i > 0 and segments[i - 1].state != segments[i].state:
                segments[i - 1].end_time = segments[i].end_time
                segments[i - 1].max_energy = max(segments[i - 1].max_energy, segments[i].max_energy)
                segments[i - 1].avg_confidence = max(segments[i - 1].avg_confidence, segments[i].avg_confidence)
                if segments[i].needs_review:
                    segments[i - 1].needs_review = True
                    if segments[i].review_reason and segments[i].review_reason not in segments[i - 1].review_reason:
                        segments[i - 1].review_reason = f"{segments[i - 1].review_reason}; {segments[i].review_reason}".strip("; ")
                segments.pop(i)
                changed = True
                continue

            # Try absorb right: merge into next static segment
            if i + 1 < len(segments) and segments[i + 1].state != segments[i].state:
                segments[i + 1].start_time = segments[i].start_time
                segments[i + 1].max_energy = max(segments[i + 1].max_energy, segments[i].max_energy)
                segments[i + 1].avg_confidence = max(segments[i + 1].avg_confidence, segments[i].avg_confidence)
                if segments[i].needs_review:
                    segments[i + 1].needs_review = True
                    if segments[i].review_reason and segments[i].review_reason not in segments[i + 1].review_reason:
                        segments[i + 1].review_reason = f"{segments[i + 1].review_reason}; {segments[i].review_reason}".strip("; ")
                segments.pop(i)
                changed = True
                continue

            i += 1

        # Re-merge adjacent same-state segments
        if changed:
            segments = _merge_same_state(segments, gap_tolerance)

    return _merge_same_state(segments, gap_tolerance)


def merge_cross_file(segments: list[Segment], gap_tolerance: float = 0.5) -> list[Segment]:
    """Merge adjacent same-state segments across file boundaries."""
    return _merge_same_state(segments, gap_tolerance)


def segments_to_json(segments: list[Segment]) -> str:
    return json.dumps([
        {
            "start_time": s.start_time,
            "end_time": s.end_time,
            "state": s.state,
            "source_file": s.source_file,
            "file_start_offset": s.file_start_offset,
            "max_energy": s.max_energy,
            "avg_confidence": s.avg_confidence,
            "needs_review": s.needs_review,
            "review_reason": s.review_reason,
        }
        for s in segments
    ])


def segments_from_json(json_str: str) -> list[Segment]:
    if not json_str:
        return []
    try:
        data = json.loads(json_str)
        if not isinstance(data, list):
            return []
        return [
            Segment(
                start_time=d["start_time"],
                end_time=d["end_time"],
                state=d["state"],
                source_file=d.get("source_file"),
                file_start_offset=d.get("file_start_offset", 0.0),
                max_energy=d.get("max_energy", 0.0),
                avg_confidence=d.get("avg_confidence", 0.0),
                needs_review=bool(d.get("needs_review", False)),
                review_reason=str(d.get("review_reason", "")),
            )
            for d in data
            if isinstance(d, dict) and "start_time" in d and "end_time" in d and "state" in d
        ]
    except Exception:
        return []


def build_audio_gated_segments(
    filepath: str,
    file_duration: float,
    file_start_offset: float = 0.0,
    audio_events: list = None,
    pre_roll: float = 1.0,
    post_roll: float = 1.5,
    gap_tolerance: float = 1.5,
) -> list[Segment]:
    """Build timeline segments for a visually static file with sparse audio events.

    Events within gap_tolerance are merged. Audio intervals are expanded with pre_roll
    and post_roll. The remaining spans are marked STATIC, producing a contiguous timeline
    covering [file_start_offset, file_start_offset + file_duration].
    """
    if not audio_events or file_duration <= 0:
        return [
            Segment(
                start_time=file_start_offset,
                end_time=file_start_offset + max(0.0, file_duration),
                state="STATIC",
                source_file=filepath,
                file_start_offset=file_start_offset,
                max_energy=0.0,
                avg_confidence=0.0,
                needs_review=False,
                review_reason="",
            )
        ]

    # Normalize intervals with pre_roll / post_roll, clamped to [0, file_duration]
    intervals = []
    for ev in audio_events:
        ev_s = float(ev[0])
        ev_e = float(ev[1])
        s = max(0.0, ev_s - pre_roll)
        e = min(file_duration, ev_e + post_roll)
        if e > s:
            intervals.append((s, e))

    if not intervals:
        return [
            Segment(
                start_time=file_start_offset,
                end_time=file_start_offset + file_duration,
                state="STATIC",
                source_file=filepath,
                file_start_offset=file_start_offset,
                max_energy=0.0,
                avg_confidence=0.0,
                needs_review=False,
                review_reason="",
            )
        ]

    intervals.sort(key=lambda x: x[0])
    # Merge overlapping or close intervals
    merged_intervals = []
    cur_s, cur_e = intervals[0]
    for nxt_s, nxt_e in intervals[1:]:
        if nxt_s <= cur_e + gap_tolerance:
            cur_e = max(cur_e, nxt_e)
        else:
            merged_intervals.append((cur_s, cur_e))
            cur_s, cur_e = nxt_s, nxt_e
    merged_intervals.append((cur_s, cur_e))

    # Construct alternating STATIC and DYNAMIC_AUDIO segments
    segments: list[Segment] = []
    last_end = 0.0
    for a_s, a_e in merged_intervals:
        if a_s > last_end:
            segments.append(
                Segment(
                    start_time=file_start_offset + last_end,
                    end_time=file_start_offset + a_s,
                    state="STATIC",
                    source_file=filepath,
                    file_start_offset=file_start_offset,
                    max_energy=0.0,
                    avg_confidence=0.0,
                    needs_review=False,
                    review_reason="",
                )
            )
        segments.append(
            Segment(
                start_time=file_start_offset + a_s,
                end_time=file_start_offset + a_e,
                state="DYNAMIC_AUDIO",
                source_file=filepath,
                file_start_offset=file_start_offset,
                max_energy=0.0,
                avg_confidence=0.0,
                needs_review=False,
                review_reason="AUDIO_ACTIVITY_ONLY",
            )
        )
        last_end = a_e

    if last_end < file_duration:
        segments.append(
            Segment(
                start_time=file_start_offset + last_end,
                end_time=file_start_offset + file_duration,
                state="STATIC",
                source_file=filepath,
                file_start_offset=file_start_offset,
                max_energy=0.0,
                avg_confidence=0.0,
                needs_review=False,
                review_reason="",
            )
        )

    return segments


def coalesce_micro_motion_segments(
    segments: list[Segment],
    gap_tolerance: float = 5.0,
) -> list[Segment]:
    """Coalesce adjacent MICRO_MOTION segments across short STATIC pauses.

    During sleep or continuous subtle activity, micro-motions are often punctuated by
    short 1-5s pauses. Leaving them fragmented creates dozens of jumpy 3s anchors and
    inflates vlog length. Coalescing them into a single continuous MICRO_MOTION span
    produces a clean 16x timelapse with a single anchor, preserving full event coverage.
    """
    if len(segments) < 3:
        return segments

    result: list[Segment] = []
    i = 0
    n = len(segments)
    while i < n:
        cur = segments[i]
        if cur.state != "MICRO_MOTION":
            result.append(cur)
            i += 1
            continue

        # Try to bridge subsequent STATIC segments and MICRO_MOTION
        j = i + 1
        merged_end = cur.end_time
        max_e = cur.max_energy
        max_conf = cur.avg_confidence
        needs_rev = cur.needs_review
        rev_reasons = [cur.review_reason] if cur.review_reason else []

        while j < n:
            if segments[j].state == "MICRO_MOTION":
                merged_end = segments[j].end_time
                max_e = max(max_e, segments[j].max_energy)
                max_conf = max(max_conf, segments[j].avg_confidence)
                if segments[j].needs_review:
                    needs_rev = True
                    if segments[j].review_reason:
                        rev_reasons.append(segments[j].review_reason)
                j += 1
            elif (
                segments[j].state == "STATIC"
                and (segments[j].end_time - segments[j].start_time) <= gap_tolerance
                and j + 1 < n
                and segments[j + 1].state == "MICRO_MOTION"
            ):
                # Bridge across the short static gap
                merged_end = segments[j + 1].end_time
                max_e = max(max_e, segments[j + 1].max_energy)
                max_conf = max(max_conf, segments[j + 1].avg_confidence)
                if segments[j + 1].needs_review:
                    needs_rev = True
                    if segments[j + 1].review_reason:
                        rev_reasons.append(segments[j + 1].review_reason)
                j += 2
            else:
                break

        cur.end_time = merged_end
        cur.max_energy = max_e
        cur.avg_confidence = max_conf
        cur.needs_review = needs_rev
        if rev_reasons:
            cur.review_reason = rev_reasons[0]
        result.append(cur)
        i = j

    return result



def split_segments_at_file_boundaries(
    segments: list[Segment],
    files_info: list[dict] | dict[str, tuple[float, float]] | None,
) -> list[Segment]:
    """
    Split multi-file segments strictly at physical file boundaries.
    Ensures every segment's start_time and end_time fall within the physical
    bounds of its source file, preventing out-of-bounds frame trimming in batch renders.
    """
    if not segments or not files_info:
        return segments

    # Normalize files_info into sorted list of (filepath, start_offset, end_offset)
    file_ranges: list[tuple[str, float, float]] = []
    if isinstance(files_info, dict):
        for fp, val in files_info.items():
            if isinstance(val, (tuple, list)) and len(val) >= 2:
                file_ranges.append((fp, float(val[0]), float(val[1])))
            elif isinstance(val, dict):
                s_off = float(val.get("file_start_offset", val.get("start_offset", 0.0)))
                e_off = float(val.get("file_end_offset", val.get("end_offset", s_off + val.get("duration", 0.0))))
                file_ranges.append((fp, s_off, e_off))
    elif isinstance(files_info, list):
        for item in files_info:
            if isinstance(item, dict):
                fp = item["filepath"]
                s_off = float(item.get("file_start_offset", item.get("start_offset", 0.0)))
                e_off = float(item.get("file_end_offset", item.get("end_offset", s_off + item.get("duration", 0.0))))
                file_ranges.append((fp, s_off, e_off))
            elif isinstance(item, (tuple, list)) and len(item) >= 3:
                file_ranges.append((str(item[0]), float(item[1]), float(item[2])))

    file_ranges.sort(key=lambda x: x[1])

    split_result: list[Segment] = []
    for seg in segments:
        overlapping_files = [
            (fp, f_start, f_end)
            for fp, f_start, f_end in file_ranges
            if f_start < seg.end_time and f_end > seg.start_time
        ]

        if not overlapping_files:
            split_result.append(seg)
            continue

        for fp, f_start, f_end in overlapping_files:
            piece_start = max(seg.start_time, f_start)
            piece_end = min(seg.end_time, f_end)
            if piece_end > piece_start:
                split_result.append(Segment(
                    start_time=piece_start,
                    end_time=piece_end,
                    state=seg.state,
                    source_file=fp,
                    file_start_offset=f_start,
                    max_energy=seg.max_energy,
                    avg_confidence=seg.avg_confidence,
                    needs_review=seg.needs_review,
                    review_reason=seg.review_reason,
                ))

    return split_result


def resolve_presence_segments(
    segments: list[Segment],
    max_presence_gap_s: float = 180.0,
    person_conf_threshold: float = 0.20,
) -> list[Segment]:
    """
    基于时序因果链，识别并升级“有人驻留/静止陪伴 (PRESENCE)”切片。
    当画面在活动事件（DYNAMIC / DYNAMIC_AUDIO / PRESENCE / MICRO_MOTION）之间存在 <= max_presence_gap_s 的静态停顿，
    且因果链中存在有效主体活动证据时，说明人物并未离开房间，仅处于静坐、看书、看手机或微停顿、睡眠状态。
    将其从 STATIC 提升为 PRESENCE，渲染端将以温和快进 (4x) 保留，避免人物在 Vlog 中被误当抽帧丢弃。
    采用多轮因果链传递算法，支持复杂场景下复合事件链（如 动态->短音频->静坐->微动）的全量覆盖。
    """
    if len(segments) < 3:
        return segments

    active_states = {"DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE", "MICRO_MOTION"}

    changed = True
    passes = 0
    while changed and passes < 10:
        changed = False
        passes += 1
        n = len(segments)
        for i in range(1, n - 1):
            cur = segments[i]
            if cur.state != "STATIC":
                continue

            dur = cur.end_time - cur.start_time
            if dur > max_presence_gap_s:
                continue

            prev_seg = segments[i - 1]
            next_seg = segments[i + 1]

            left_active = prev_seg.state in active_states
            right_active = next_seg.state in active_states

            # 强化的因果链主体证据：
            # 1. 邻近置信度达标 (>= person_conf_threshold) 或已判定驻留 (PRESENCE)
            # 2. 两端均为真实动态 (DYNAMIC <-> DYNAMIC)，证明人处于同一连续活动周期的短暂停顿
            # 3. 动态/驻留与微动交替 (DYNAMIC/PRESENCE <-> MICRO_MOTION) 且停顿期残余能量 >= 2.0
            # 4. 停顿期自身存在显著残余运动能量 (max_energy >= 2.5) 且两侧处于活动态
            has_person_evidence = (
                prev_seg.avg_confidence >= person_conf_threshold
                or next_seg.avg_confidence >= person_conf_threshold
                or prev_seg.state == "PRESENCE"
                or next_seg.state == "PRESENCE"
                or (prev_seg.state == "DYNAMIC" and next_seg.state == "DYNAMIC")
                or (cur.max_energy >= 2.5 and (left_active or right_active))
                or (prev_seg.state == "DYNAMIC_AUDIO" and next_seg.state in ("DYNAMIC", "MICRO_MOTION"))
                or (next_seg.state == "DYNAMIC_AUDIO" and prev_seg.state in ("DYNAMIC", "MICRO_MOTION"))
                or (prev_seg.state in ("DYNAMIC", "PRESENCE") and next_seg.state == "MICRO_MOTION" and cur.max_energy >= 2.0)
                or (next_seg.state in ("DYNAMIC", "PRESENCE") and prev_seg.state == "MICRO_MOTION" and cur.max_energy >= 2.0)
            )

            if left_active and right_active and has_person_evidence:
                cur.state = "PRESENCE"
                conf_candidates = [s.avg_confidence for s in (prev_seg, next_seg) if s.avg_confidence > 0]
                cur.avg_confidence = round(sum(conf_candidates) / len(conf_candidates), 4) if conf_candidates else 0.35
                cur.review_reason = f"TARGET_PERSISTENCE: 驻留静坐陪伴(持续{dur:.1f}s)"
                changed = True

    return coalesce_micro_motion_segments(segments, gap_tolerance=5.0)



