"""Freeze an explicitly selected human-review cohort without changing production."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def prepare(db_path, output, reviewed_after, reviewed_before=None):
    from scripts.audit_tool.service import unpack_notes, unpack_review_basis
    from src.stages.feedback import normalize_label, POSITIVE_LABELS, NEGATIVE_LABELS
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    source = sqlite3.connect(Path(db_path).resolve().as_uri()+"?mode=ro", uri=True)
    source.row_factory = sqlite3.Row
    source.execute("BEGIN")
    rows = [dict(r) for r in source.execute(
        "SELECT s.*,h.id AS review_id FROM human_reviews h JOIN segments s ON "
        "s.filepath=h.filepath AND s.start_time=h.start_time AND s.end_time=h.end_time "
        "AND s.manual_label=h.manual_label WHERE h.reviewed_at>=? AND (? IS NULL OR h.reviewed_at<?) "
        "ORDER BY h.id", (reviewed_after, reviewed_before, reviewed_before))]
    count = source.execute("SELECT COUNT(*) FROM human_reviews WHERE reviewed_at>=? AND (? IS NULL OR reviewed_at<?)",
                           (reviewed_after, reviewed_before, reviewed_before)).fetchone()[0]
    if len(rows) != count or not rows:
        raise ValueError("Reviews must match current intervals exactly; empty/duplicate/missing mirror rejected")
    with sqlite3.connect(output / "production_snapshot.sqlite") as target:
        source.backup(target)
    source.close()
    samples = []
    for row in rows:
        label = normalize_label(row["manual_label"])
        if label not in POSITIVE_LABELS | NEGATIVE_LABELS:
            raise ValueError(f"Unknown review label: {label}")
        scenario, notes = unpack_notes(row.get("review_notes"))
        basis = unpack_review_basis(row.get("review_notes"))
        qualified = basis == "video_audio_interval" or (basis == "video_interval" and label in POSITIVE_LABELS)
        path = Path(row["filepath"])
        stat = path.stat()
        samples.append({"segment_id": row["id"], "review_id": row["review_id"], "filepath": str(path),
            "date": row["date"], "cam_index": row["cam_index"], "start_time": row["start_time"],
            "end_time": row["end_time"], "gold_label": label, "scenario": scenario or "UNKNOWN",
            "reference_active": (label in POSITIVE_LABELS) if qualified else None,
            "review_status": "independent_human" if qualified else "pending_confirmation", "review_basis": basis,
            "review_evidence": {"basis": "user_confirmed_workbench_review", "review_id": row["review_id"],
                                "review_basis": basis, "full_interval_declared": basis in {"video_interval", "video_audio_interval"},
                                "reviewed_at": row["reviewed_at"], "snapshot": "production_snapshot.sqlite"},
            "notes": notes, "original_state": row["state"], "failure_type": "UNSPECIFIED",
            "expected_speed": None, "source_stat": {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}})
    # Date blocks prevent adjacent clips or different cameras of one day leaking.
    dates = sorted({s["date"] for s in samples}, key=lambda d: hashlib.sha256(("review-split-v1:"+d).encode()).hexdigest())
    holdout = set(dates[:max(1, round(len(dates)*.2))]) if len(dates) > 1 else set()
    missed_dates = [date for date in dates if any(s["date"] == date and s["gold_label"] == "MISSED_MOTION" for s in samples)]
    if len(missed_dates) > 1:
        holdout.add(missed_dates[0])
        holdout.discard(missed_dates[1])
    for sample in samples:
        sample["split"] = "holdout" if sample["date"] in holdout else "tuning"
        sample["retention_policy"] = "visible_activity_may_be_accelerated"
    for name, selected in (("all", samples), ("tuning", [s for s in samples if s["split"] == "tuning"]),
                           ("holdout", [s for s in samples if s["split"] == "holdout"])):
        payload = {"metadata": {"scope": "selected_workbench_activity_reviews_not_person_boxes",
                                "split_policy": "locked_date_blocks_v1", "independent_population_estimate": False},
                   "samples": selected}
        (output / f"{name}.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    (output / "raw_reviews.json").write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding="utf-8")
    shutil.copy2(PROJECT / "config/settings.yaml", output / "settings.yaml")
    report = {"total": len(samples), "source_count": len({s["filepath"] for s in samples}),
              "date_count": len(dates), "labels": dict(Counter(s["gold_label"] for s in samples)),
              "splits": {name: dict(Counter(s["gold_label"] for s in samples if s["split"] == name)) for name in ("tuning", "holdout")},
              "unknown_scenarios": sum(s["scenario"] == "UNKNOWN" for s in samples),
              "qualified_references": sum(s["review_status"] == "independent_human" for s in samples),
              "pending_confirmation": sum(s["review_status"] != "independent_human" for s in samples),
              "review_bases": dict(Counter(s["review_basis"] for s in samples)), "production_writes": False,
              "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()}}
    (output / "preparation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=PROJECT / "data/vlog.db")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewed-after", required=True)
    parser.add_argument("--reviewed-before")
    args = parser.parse_args()
    print(json.dumps(prepare(args.db,args.output,args.reviewed_after,args.reviewed_before),ensure_ascii=True,indent=2))
