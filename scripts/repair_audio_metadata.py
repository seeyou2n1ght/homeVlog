"""Inspect legacy audio-metadata suspects without changing media or database state."""
import argparse
import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("repair_audio")


def repair_metadata(db_path: str = "data/vlog.db", date: str | None = None, clean_batches: bool = False):
    if clean_batches:
        raise ValueError("--clean-batches is disabled: batch removal requires verified delivery and workspace ownership")

    query = "SELECT COUNT(*) FROM file_tasks WHERE prescreen_status = 'SUSPICIOUS' AND has_audio = 0"
    params = []
    if date:
        query += " AND date = ?"
        params.append(date)

    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        count = conn.execute(query, params).fetchone()[0]
    finally:
        conn.close()
    logger.info("Found %d records with prescreen_status='SUSPICIOUS' and has_audio=0", count)

    logger.info("Inspection only: SUSPICIOUS does not establish an audio stream; no rows or batches changed")
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect audio metadata suspects (read-only)")
    parser.add_argument("--db", default="data/vlog.db", help="Path to SQLite database")
    parser.add_argument("--date", default=None, help="Target date YYYYMMDD (optional, default all)")
    parser.add_argument("--clean-batches", action="store_true", help="Disabled legacy option; never removes batches")
    args = parser.parse_args()
    if args.clean_batches:
        parser.error("--clean-batches is disabled; this tool only inspects metadata")

    repair_metadata(db_path=args.db, date=args.date, clean_batches=args.clean_batches)
