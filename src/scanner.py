import logging
import os
import re
import time
from dataclasses import dataclass

from src.database import VlogDatabase
from src.utils import load_config, ts_to_unix

logger = logging.getLogger("homevlog")

FILENAME_RE = re.compile(
    r"^(\d{2})_(\d{14})_(\d{14})\.mp4$"
)


def parse_filename(filename: str) -> dict | None:
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    cam = int(m.group(1))
    start_str = m.group(2)
    end_str = m.group(3)
    start_ts = ts_to_unix(start_str)
    end_ts = ts_to_unix(end_str)
    date = start_str[:8]
    return {
        "cam_index": cam,
        "date": date,
        "file_start_time": start_str,
        "file_end_time": end_str,
        "start_ts": start_ts,
        "end_ts": end_ts,
    }




@dataclass
class ScanResult:
    added: int
    skipped: int
    frozen_pending: int


def scan_directory(
    db: VlogDatabase,
    input_dir: str | None = None,
) -> ScanResult:
    config = load_config()
    if input_dir is None:
        input_dir = config["paths"]["input_dir"]
    freeze_minutes = config.get("recovery", {}).get("scanner_freeze_minutes", 10)
    stabilize_wait = config.get("recovery", {}).get("file_stabilize_wait", 1.2)

    logger.info("scanning: %s (freeze=%dmin)", input_dir, freeze_minutes)

    if not os.path.isdir(input_dir):
        logger.error("input_dir not found: %s", input_dir)
        return ScanResult(added=0, skipped=0, frozen_pending=0)

    added = 0
    skipped = 0
    frozen = 0

    for entry in os.scandir(input_dir):
        if not entry.is_file() or not entry.name.endswith(".mp4"):
            continue

        info = parse_filename(entry.name)
        if info is None:
            logger.debug("skip unrecognized filename: %s", entry.name)
            continue

        stat = entry.stat()
        age_min = (time.time() - stat.st_mtime) / 60.0

        if age_min < freeze_minutes:
            frozen += 1
            continue

        # Stabilize check: only for files near the freeze boundary.
        # Files well past the freeze window are definitely stable.
        if freeze_minutes > 0 and age_min < freeze_minutes + 5:
            time.sleep(stabilize_wait)
            try:
                stat2 = os.stat(entry.path)
                if stat2.st_size != stat.st_size:
                    frozen += 1
                    continue
            except OSError:
                frozen += 1
                continue

        ok = db.add_file_task(
            filepath=entry.path,
            cam_index=info["cam_index"],
            date=info["date"],
            file_start_time=info["file_start_time"],
            file_end_time=info["file_end_time"],
            file_duration=info["end_ts"] - info["start_ts"],
        )
        if ok:
            added += 1
        else:
            skipped += 1

    logger.info("scan done: added=%d skipped=%d frozen=%d", added, skipped, frozen)
    return ScanResult(added=added, skipped=skipped, frozen_pending=frozen)


def get_date_cam_groups(db: VlogDatabase) -> list[tuple[str, int]]:
    rows = db.conn.execute(
        """SELECT DISTINCT date, cam_index FROM file_tasks
           ORDER BY date, cam_index"""
    ).fetchall()
    return [(r["date"], r["cam_index"]) for r in rows]
