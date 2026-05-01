"""HomeVlog — 家庭监控素材智能浓缩系统

Usage:
  uv run python main.py            # full pipeline (scan → prescreen → analyze → render)
  uv run python main.py --scan     # scan only
  uv run python main.py --no-render  # scan + prescreen + analyze, skip render
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import setup_logging
from src.pipeline import run_pipeline
from src.database import VlogDatabase
from src.scanner import scan_directory, get_date_cam_groups

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="HomeVlog — 监控素材智能浓缩")
    parser.add_argument("--scan", action="store_true", help="scan only, do not process")
    parser.add_argument("--no-render", action="store_true", help="skip Pass2 rendering")
    parser.add_argument("--date", type=str, help="process specific date (YYYYMMDD)")
    parser.add_argument("--cam", type=int, help="process specific camera index")
    args = parser.parse_args()

    if args.scan:
        db = VlogDatabase()
        try:
            scan_directory(db)
            groups = get_date_cam_groups(db)
            print(f"Found {len(groups)} date-cam groups:")
            for date, cam in groups:
                print(f"  {date} cam{cam}")
        finally:
            db.close()
        return

    if args.date:
        db = VlogDatabase()
        try:
            from src.pipeline import process_date_cam
            from src.monitor import get_monitor
            monitor = get_monitor()
            monitor.start()
            cam = args.cam if args.cam is not None else 0
            ok = process_date_cam(db, args.date, cam, skip_render=args.no_render)
            monitor.shutdown()
            print(f"{'OK' if ok else 'FAILED'}")
        finally:
            db.close()
        return

    result = run_pipeline(skip_render=args.no_render)
    print(f"\nDone: {result}")


if __name__ == "__main__":
    main()
