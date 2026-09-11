"""Repair corrupted has_audio metadata in file_tasks table and purge bad batch caches."""
import argparse
import glob
import logging
import sqlite3
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("repair_audio")


def repair_metadata(db_path: str = "data/vlog.db", date: str | None = None, clean_batches: bool = False):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    query = "SELECT COUNT(*) FROM file_tasks WHERE prescreen_status = 'SUSPICIOUS' AND has_audio = 0"
    params = []
    if date:
        query += " AND date = ?"
        params.append(date)

    cursor.execute(query, params)
    count = cursor.fetchone()[0]
    logger.info("Found %d records with prescreen_status='SUSPICIOUS' and has_audio=0", count)

    if count > 0:
        update_sql = "UPDATE file_tasks SET has_audio = 1, updated_at = datetime('now', 'localtime') WHERE prescreen_status = 'SUSPICIOUS' AND has_audio = 0"
        if date:
            update_sql += " AND date = ?"
        cursor.execute(update_sql, params)
        conn.commit()
        logger.info("Successfully updated %d records to has_audio = 1", cursor.rowcount)
    else:
        logger.info("No records need repair.")

    if clean_batches:
        temp_dir = Path("temp")
        pattern = f"_batch*_{date}_*.mp4*" if date else "_batch*.mp4*"
        files = list(temp_dir.glob(pattern))
        logger.info("Cleaning %d cached batch files matching %s", len(files), pattern)
        for f in files:
            try:
                f.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Failed to delete %s: %s", f.name, exc)

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Repair corrupted audio metadata in vlog.db")
    parser.add_argument("--db", default="data/vlog.db", help="Path to SQLite database")
    parser.add_argument("--date", default=None, help="Target date YYYYMMDD (optional, default all)")
    parser.add_argument("--clean-batches", action="store_true", help="Clean affected temp batch mp4 and manifest files")
    args = parser.parse_args()

    repair_metadata(db_path=args.db, date=args.date, clean_batches=args.clean_batches)
