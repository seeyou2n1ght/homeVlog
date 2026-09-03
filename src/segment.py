import json
from dataclasses import dataclass


@dataclass
class Segment:
    start_time: float
    end_time: float
    state: str  # "DYNAMIC" | "STATIC" | "DYNAMIC_AUDIO"
    source_file: str
    file_start_offset: float
    max_energy: float = 0.0
    avg_confidence: float = 0.0

    @property
    def is_dynamic(self) -> bool:
        """Returns True if segment represents active dynamic motion or audio event."""
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
            seg_end = frame_labels[i - 1]["time"]
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
        total_max_t = frame_labels[-1]["time"]
        dyn_spans = []
        for s in segments:
            if s.is_dynamic:
                st = max(0.0, s.start_time - pre_roll)
                et = min(total_max_t, s.end_time + post_roll)
                dyn_spans.append((st, et, s.state, s.max_energy))

        if dyn_spans:
            # Merge overlapping dynamic spans
            merged_dyn = [dyn_spans[0]]
            for st, et, state, energy in dyn_spans[1:]:
                last_st, last_et, last_state, last_energy = merged_dyn[-1]
                if st <= last_et + gap_tolerance:
                    resolved_state = "DYNAMIC" if (last_state == "DYNAMIC" or state == "DYNAMIC") else state
                    merged_dyn[-1] = (last_st, max(last_et, et), resolved_state, max(last_energy, energy))
                else:
                    merged_dyn.append((st, et, state, energy))

            # Reconstruct complete segment sequence with static spans in between
            new_segments = []
            curr_t = 0.0
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
    if apply_smoothing:
        return _filter_short(merged, min_motion_dur, min_static_dur, gap_tolerance)
    return merged


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
        else:
            result.append(seg)
    return result


def _filter_short(
    segments: list[Segment],
    min_motion: float,
    min_static: float,
    gap_tolerance: float = 0.5,
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

            # Try absorb left: merge into prev segment
            if i > 0 and segments[i - 1].state != segments[i].state:
                segments[i - 1].end_time = segments[i].end_time
                segments[i - 1].max_energy = max(segments[i - 1].max_energy, segments[i].max_energy)
                segments.pop(i)
                changed = True
                continue

            # Try absorb right
            if i + 1 < len(segments) and segments[i + 1].state != segments[i].state:
                segments[i + 1].start_time = segments[i].start_time
                segments[i + 1].max_energy = max(segments[i + 1].max_energy, segments[i].max_energy)
                segments.pop(i)
                changed = True
                continue

            i += 1

        # Re-merge adjacent same-state segments
        if changed:
            merged: list[Segment] = []
            for seg in segments:
                if not merged or not _can_merge(merged[-1], seg, gap_tolerance):
                    merged.append(seg)
                else:
                    merged[-1].end_time = seg.end_time
                    merged[-1].max_energy = max(merged[-1].max_energy, seg.max_energy)
            segments = merged

    # Merge adjacent same-state segments
    filtered: list[Segment] = []
    for seg in segments:
        if not filtered or not _can_merge(filtered[-1], seg, gap_tolerance):
            filtered.append(seg)
        else:
            filtered[-1].end_time = seg.end_time
            filtered[-1].max_energy = max(filtered[-1].max_energy, seg.max_energy)

    return filtered


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
                file_start_offset=d.get("file_start_offset"),
                max_energy=d.get("max_energy", 0.0),
                avg_confidence=d.get("avg_confidence", 0.0),
            )
            for d in data
            if isinstance(d, dict) and "start_time" in d and "end_time" in d and "state" in d
        ]
    except Exception:
        return []


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
                ))

    return split_result

