import json
import logging
from dataclasses import dataclass

logger = logging.getLogger("homevlog")


@dataclass
class Segment:
    start_time: float
    end_time: float
    state: str  # "DYNAMIC" | "STATIC"
    source_file: str
    file_start_offset: float


def build_segments(
    frame_labels: list[dict],
    source_file: str,
    min_motion_dur: float = 2.0,
    min_static_dur: float = 30.0,
    file_offset: float = 0.0,
    gap_tolerance: float = 0.5,
    apply_smoothing: bool = False,
) -> list[Segment]:
    """
    Convert frame-by-frame labels to contiguous segments.
    frame_labels: [{time, is_motion}, ...]
    file_offset: absolute time offset of the source file (day-relative seconds)
    """
    if not frame_labels:
        return []

    segments: list[Segment] = []
    seg_start = frame_labels[0]["time"]
    seg_state = "DYNAMIC" if frame_labels[0]["is_motion"] else "STATIC"

    for i in range(1, len(frame_labels)):
        cur_state = "DYNAMIC" if frame_labels[i]["is_motion"] else "STATIC"
        if cur_state != seg_state:
            seg_end = frame_labels[i - 1]["time"]
            segments.append(Segment(
                start_time=seg_start,
                end_time=seg_end,
                state=seg_state,
                source_file=source_file,
                file_start_offset=file_offset,
            ))
            seg_start = frame_labels[i]["time"]
            seg_state = cur_state

    segments.append(Segment(
        start_time=seg_start,
        end_time=frame_labels[-1]["time"],
        state=seg_state,
        source_file=source_file,
        file_start_offset=file_offset,
    ))

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
            threshold = min_motion if segments[i].state == "DYNAMIC" else min_static
            if dur >= threshold:
                i += 1
                continue

            # Try absorb left: merge into prev segment
            if i > 0 and segments[i - 1].state != segments[i].state:
                segments[i - 1].end_time = segments[i].end_time
                segments.pop(i)
                changed = True
                continue

            # Try absorb right
            if i + 1 < len(segments) and segments[i + 1].state != segments[i].state:
                segments[i + 1].start_time = segments[i].start_time
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
            segments = merged

    # Merge adjacent same-state segments
    filtered: list[Segment] = []
    for seg in segments:
        if not filtered or not _can_merge(filtered[-1], seg, gap_tolerance):
            filtered.append(seg)
        else:
            filtered[-1].end_time = seg.end_time

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
        }
        for s in segments
    ])


def segments_from_json(json_str: str) -> list[Segment]:
    data = json.loads(json_str)
    return [
        Segment(
            start_time=d["start_time"],
            end_time=d["end_time"],
            state=d["state"],
            source_file=d["source_file"],
            file_start_offset=d["file_start_offset"],
        )
        for d in data
    ]
