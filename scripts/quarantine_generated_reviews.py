"""Quarantine a certified legacy heuristic dataset without discarding evidence.

Read-only by default. Application requires the exact dataset SHA-256 and an
unverified_heuristic sidecar, matches every label/interval/note/timestamp, backs
up the database, and archives original rows before removing manual overrides.
"""
import argparse
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import uuid


def plan_quarantine(connection, samples):
    connection.row_factory = sqlite3.Row
    plan = []
    for sample in samples:
        segment = connection.execute("SELECT * FROM segments WHERE id=?", (sample["segment_id"],)).fetchone()
        if segment is None:
            raise ValueError(f"Missing segment {sample['segment_id']}")
        row = dict(segment)
        try:
            notes = json.loads(row.get("review_notes") or "{}")
        except ValueError as error:
            raise ValueError(f"Review changed or provenance mismatch: {sample['segment_id']}") from error
        if (row["filepath"] != sample["filepath"] or row["manual_label"] != sample["raw_label"]
                or round(row["start_time"], 2) != round(sample["start_time"], 2)
                or round(row["end_time"], 2) != round(sample["end_time"], 2)
                or row["reviewed_at"] != sample["reviewed_at"]
                or notes != {"notes": sample["notes"], "scenario": sample["scenario"]}):
            raise ValueError(f"Review changed or provenance mismatch: {sample['segment_id']}")
        review = connection.execute("SELECT * FROM human_reviews WHERE filepath=? AND start_time=? AND end_time=?",
                                    (row["filepath"], row["start_time"], row["end_time"])).fetchone()
        if (review is None or review["manual_label"] != row["manual_label"]
                or review["reviewed_at"] != row["reviewed_at"] or review["review_notes"] != row["review_notes"]):
            raise ValueError(f"Review mirror changed: {sample['segment_id']}")
        plan.append({"segment": row, "human_review": dict(review)})
    if len({item["segment"]["id"] for item in plan}) != len(plan):
        raise ValueError("Duplicate segment identities")
    return plan


def quarantine_generated_reviews(db_path, dataset_path, expected_sha256, report_dir, *, apply=False):
    db_path, dataset_path, report_dir = map(Path, (db_path, dataset_path, report_dir))
    payload = dataset_path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Dataset checksum mismatch")
    metadata = json.loads(dataset_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
    if metadata.get("review_status") != "unverified_heuristic":
        raise ValueError("Explicit unverified heuristic provenance is required")
    samples = json.loads(payload)
    run_dir = report_dir / uuid.uuid4().hex
    run_dir.mkdir(parents=True, exist_ok=False)
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as source:
        plan = plan_quarantine(source, samples)
        if apply:
            with closing(sqlite3.connect(run_dir / "before.sqlite")) as backup:
                source.backup(backup)
    (run_dir / "original_rows.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    affected = sorted({(item["segment"]["date"], item["segment"]["cam_index"]) for item in plan})
    if apply:
        with closing(sqlite3.connect(str(db_path), timeout=15.)) as target, target:
            target.execute("BEGIN IMMEDIATE")
            current = plan_quarantine(target, samples)
            if current != plan:
                raise ValueError("Database changed since backup; no labels removed")
            target.execute("""CREATE TABLE IF NOT EXISTS quarantined_generated_reviews (
                dataset_sha256 TEXT NOT NULL, segment_id INTEGER NOT NULL,
                original_rows_json TEXT NOT NULL, reason TEXT NOT NULL,
                quarantined_at TEXT DEFAULT (datetime('now','localtime')),
                PRIMARY KEY(dataset_sha256,segment_id))""")
            for item in plan:
                row, review = item["segment"], item["human_review"]
                target.execute("INSERT INTO quarantined_generated_reviews(dataset_sha256,segment_id,original_rows_json,reason) VALUES (?,?,?,?)",
                               (expected_sha256, row["id"], json.dumps(item, ensure_ascii=False), "UNVERIFIED_HEURISTIC_LABEL"))
                target.execute("""UPDATE segments SET manual_label=NULL,review_notes=NULL,reviewed_at=NULL,
                                  needs_review=1,review_reason='UNVERIFIED_HEURISTIC_LABEL' WHERE id=?""", (row["id"],))
                target.execute("DELETE FROM human_reviews WHERE id=?", (review["id"],))
            for date, camera in affected:
                target.execute("UPDATE render_tasks SET status='PENDING',error_msg='Generated labels quarantined; timeline requires rebuild' WHERE date=? AND cam_index=?",
                               (date, camera))
    receipt = {"applied": apply, "count": len(plan), "dataset_sha256": expected_sha256,
               "affected_scopes": affected, "backup": str(run_dir / "before.sqlite") if apply else None,
               "original_rows": str(run_dir / "original_rows.json"), "production_media_changed": False}
    (run_dir / "receipt.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--report-dir", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    print(json.dumps(quarantine_generated_reviews(args.db, args.dataset, args.sha256, args.report_dir, apply=args.apply)))
