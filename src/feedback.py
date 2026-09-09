"""Stable human labels and exact source-time overlays independent of algorithm segments."""
from copy import copy

LABELS = {"TP": "CONFIRMED_MOTION", "FP": "FALSE_ALARM", "FN": "MISSED_MOTION",
          "TN": "CONFIRMED_STATIC", "VERIFIED_MOTION": "CONFIRMED_MOTION"}
NEGATIVE_LABELS = {"FALSE_ALARM", "CONFIRMED_STATIC"}
POSITIVE_LABELS = {"CONFIRMED_MOTION", "MISSED_MOTION"}


def normalize_label(label):
    label = str(label or "").strip().upper()
    return LABELS.get(label, label)


def overlay_reviews(segments, reviews):
    """Split at review boundaries; never extend a human label outside its reviewed interval."""
    for review in reviews:
        label = normalize_label(review["manual_label"])
        if label not in NEGATIVE_LABELS | POSITIVE_LABELS:
            continue
        updated = []
        for seg in segments:
            start = max(seg.start_time, review["start_time"])
            end = min(seg.end_time, review["end_time"])
            if seg.source_file != review["filepath"] or end <= start:
                updated.append(seg)
                continue
            for a, b, state in ((seg.start_time, start, seg.state),
                                (start, end, "STATIC" if label in NEGATIVE_LABELS else "DYNAMIC"),
                                (end, seg.end_time, seg.state)):
                if b > a:
                    part = copy(seg)
                    part.start_time, part.end_time, part.state = a, b, state
                    updated.append(part)
        segments = updated
    return segments
