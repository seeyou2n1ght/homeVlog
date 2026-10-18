"""Export recorded reviews and select unreviewed candidates without writing labels."""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tool.service import AuditService
from src.core.database import VlogDatabase


def suggest_candidates(db: VlogDatabase) -> list[dict]:
    """Heuristics select review candidates; they cannot create human truth."""
    groups = [
        ("ADULT_MOTION", "confident_object", "state='DYNAMIC' AND avg_confidence>=0.45", 30),
        ("LIGHT_SHADOW", "high_energy_without_object", "state='DYNAMIC' AND avg_confidence=0 AND max_energy>=15", 25),
        ("INFANT_MOTION", "weak_or_micro_motion", "((state='STATIC' AND max_energy BETWEEN 2 AND 4) OR state='MICRO_MOTION')", 25),
        ("EMPTY_ROOM", "low_energy", "state='STATIC' AND max_energy<0.8", 20),
    ]
    candidates = []
    with db._lock:
        for scenario, reason, condition, limit in groups:
            rows = db.conn.execute(
                "SELECT id AS segment_id,filepath,start_time,end_time,duration,"
                "state AS algo_state,max_energy,avg_confidence FROM segments "
                f"WHERE manual_label IS NULL AND {condition} ORDER BY id LIMIT ?", (limit,),
            ).fetchall()
            candidates.extend({**dict(row), "suggested_scenario": scenario,
                               "selection_reason": reason, "review_required": True} for row in rows)
    return candidates


def build_gold_baseline(db_path: Path | None = None, force_init: bool = False,
                        output_path: Path | None = None, candidates_path: Path | None = None,
                        archive_version: bool = True) -> dict:
    """Export recorded reviews; force_init refreshes candidates, never labels.

    Exported legacy reviews still require independent provenance verification
    before they can be called holdout truth.
    """
    out_path = output_path or PROJECT_ROOT / "tests/benchmark_data/gold_ground_truth.json"
    candidate_file = candidates_path or out_path.with_name(out_path.stem + ".candidates.json")
    db = VlogDatabase(db_path)
    try:
        with db._lock:
            reviewed_count = db.conn.execute(
                "SELECT COUNT(*) FROM segments WHERE manual_label IS NOT NULL"
            ).fetchone()[0]
        candidates = []
        if reviewed_count < 20 or force_init:
            candidates = suggest_candidates(db)
            candidate_file.parent.mkdir(parents=True, exist_ok=True)
            candidate_file.write_text(json.dumps({"label_source": "heuristic_candidates",
                "review_required": True, "samples": candidates}, ensure_ascii=False, indent=2), encoding="utf-8")
        if reviewed_count:
            AuditService(db=db).export_gold_benchmark(output_path=out_path, archive_version=archive_version)
            data = json.loads(out_path.read_text(encoding="utf-8"))
        else:
            # Never replace an existing dataset with empty or fabricated truth.
            data = []
        distribution = {}
        for item in data:
            label = item.get("gold_label", "UNKNOWN")
            distribution[label] = distribution.get(label, 0) + 1
        return {"output_path": str(out_path) if reviewed_count else None, "total_samples": len(data),
                "distribution": distribution, "reviewed_count": reviewed_count,
                "candidates_path": str(candidate_file) if reviewed_count < 20 or force_init else None,
                "candidate_count": len(candidates)}
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export recorded reviews and suggest candidates without changing human labels")
    parser.add_argument("--db", type=Path, help="Database to export")
    parser.add_argument("--output", type=Path, help="Review dataset output")
    parser.add_argument("--force-init", action="store_true", help="Refresh unreviewed candidates; never overwrite labels")
    args = parser.parse_args()
    print(json.dumps(build_gold_baseline(args.db, args.force_init, args.output), ensure_ascii=False, indent=2))
