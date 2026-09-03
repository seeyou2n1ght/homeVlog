"""HomeVlog — 家庭监控素材智能浓缩系统

Usage:
  uv run python main.py            # full pipeline (scan → prescreen → analyze → render)
  uv run python main.py --scan     # scan only
  uv run python main.py --no-render  # scan + prescreen + analyze, skip render
"""

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import setup_logging, load_config, reset_semaphores, get_logger
from src.pipeline import run_pipeline
from src.database import VlogDatabase
from src.scanner import scan_directory, get_date_cam_groups
from src.ui import print_scan_results, console

logger = setup_logging()


def main():
    parser = argparse.ArgumentParser(description="HomeVlog — 监控素材智能浓缩系统")
    parser.add_argument("--input-dir", type=str, default=None, help="override input directory containing video clips")
    parser.add_argument("--config", type=str, default=None, help="path to custom settings.yaml")
    parser.add_argument("--scan", action="store_true", help="scan only, do not process")
    parser.add_argument("--no-render", action="store_true", help="skip Pass2 rendering")
    parser.add_argument("--date", type=str, help="process specific date (YYYYMMDD)")
    parser.add_argument("--cam", type=int, help="process specific camera index")
    parser.add_argument("--no-tui", action="store_true", help="disable interactive live dashboard and use plain output")
    parser.add_argument("--debug", action="store_true", help="enable verbose debug logging")
    args = parser.parse_args()

    import logging
    if args.debug:
        setup_logging(level_override=logging.DEBUG)

    if args.config:
        load_config(args.config, reload=True)
        reset_semaphores()

    config = load_config()
    from src.utils import get_input_dirs
    input_dirs = [args.input_dir] if args.input_dir else get_input_dirs(config)
    dashboard_enabled = not args.no_tui

    if args.scan:
        db = VlogDatabase()
        try:
            scan_directory(db, input_dir=input_dirs)
            groups = get_date_cam_groups(db)
            if groups:
                from src.scanner import resolve_camera_identity
                cam_map = {}
                for d, cam in groups:
                    tasks = db.get_all_file_tasks_for_date(d, cam)
                    s_dir = str(Path(tasks[0]["filepath"]).parent) if tasks else (input_dirs[0] if input_dirs else "")
                    disp, _ = resolve_camera_identity(s_dir, cam_index=cam, config=config)
                    cam_map[cam] = disp
                print_scan_results(groups, camera_display_names=cam_map)
            else:
                console.print("[yellow]未发现符合命名格式的监控切片文件。[/yellow]")
        finally:
            db.close()
        return

    if args.date:
        db = VlogDatabase()
        try:
            from src.pipeline import process_date_cam
            from src.monitor import get_monitor
            scan_directory(db, input_dir=input_dirs)
            monitor = get_monitor()
            monitor.start()
            cam = args.cam if args.cam is not None else 0
            ok = process_date_cam(db, args.date, cam, skip_render=args.no_render, dashboard_enabled=dashboard_enabled)
            monitor.shutdown()

            tasks = db.get_all_file_tasks_for_date(args.date, cam)
            sample_dir = str(Path(tasks[0]["filepath"]).parent) if tasks else (input_dirs[0] if input_dirs else "")
            from src.scanner import resolve_camera_identity
            cam_display, _ = resolve_camera_identity(sample_dir, cam_index=cam, config=config)
            if ok:
                console.print(f"[bold green]✔ 指定日期机位处理完成: {args.date} ({cam_display}) (SUCCESS)[/bold green]")
            else:
                console.print(f"[bold red]✖ 指定日期机位处理失败: {args.date} ({cam_display}) (FAILED)[/bold red]")

        finally:
            db.close()
        return

    result = run_pipeline(skip_render=args.no_render, input_dir=input_dirs, dashboard_enabled=dashboard_enabled)

    console.print(f"[bold cyan]流水线总览:[/] 处理组合总数={result['total']}, 成功={result['ok']}, 失败={result['failed']}")


if __name__ == "__main__":
    main()

