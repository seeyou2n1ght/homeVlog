import logging
import sys
import time

from src.database import VlogDatabase
from src.scanner import scan_directory, get_date_cam_groups
from src.prescreen import run_prescreen_for_cam
from src.detector import run_analysis_for_cam
from src.renderer import render_date_cam
from src.scheduler import choose_encoder
from src.utils import load_config, OUTPUT_DIR, DB_PATH
from src.monitor import get_monitor, get_perf

logger = logging.getLogger("homevlog")


def process_date_cam(
    db: VlogDatabase,
    date: str,
    cam_index: int,
    skip_render: bool = False,
) -> bool:
    """Run full pipeline for one date-camera: prescreen -> analysis -> render."""
    config = load_config()
    monitor = get_monitor()
    perf = get_perf()
    t_pipeline_start = time.monotonic()

    if db.is_render_completed(date, cam_index):
        logger.info("skip %s cam%d: already completed", date, cam_index)
        return True

    logger.info("=== pipeline %s cam%d start ===", date, cam_index)

    # Pass 1: Prescreen
    with monitor.stage(f"prescreen_{date}_cam{cam_index}"):
        prescreen_result = run_prescreen_for_cam(db, date, cam_index)
    if not db.is_prescreen_complete(date, cam_index):
        logger.error("prescreen incomplete for %s cam%d", date, cam_index)
        return False

    # Pass 1.5: Analysis
    with monitor.stage(f"analysis_{date}_cam{cam_index}"):
        analysis_result = run_analysis_for_cam(db, date, cam_index, config)
    if not db.is_analysis_complete(date, cam_index):
        logger.warning("analysis incomplete for %s cam%d (some files failed)", date, cam_index)

    # Pass 2: Render
    if skip_render:
        logger.info("render skipped for %s cam%d", date, cam_index)
        _dump_perf(perf, monitor, date, cam_index, time.monotonic() - t_pipeline_start)
        return True

    encoder = choose_encoder(cam_index)
    db.upsert_render_task(date, cam_index, "RENDERING")
    output_path = None
    try:
        with monitor.stage(f"render_{date}_cam{cam_index}"):
            output_path = render_date_cam(db, date, cam_index, encoder=encoder)
    except Exception:
        logger.exception("render failed for %s cam%d", date, cam_index)

    if output_path is not None:
        db.set_render_status(date, cam_index, "COMPLETED", output_file=output_path)
        completed_marker = OUTPUT_DIR / f".completed_{date}_cam{cam_index}"
        completed_marker.touch(exist_ok=True)
        logger.info("=== pipeline %s cam%d COMPLETED ===", date, cam_index)
    else:
        db.set_render_status(date, cam_index, "FAILED")
        logger.error("=== pipeline %s cam%d FAILED ===", date, cam_index)

    _dump_perf(perf, monitor, date, cam_index, time.monotonic() - t_pipeline_start)
    return output_path is not None


def _dump_perf(perf, monitor, date: str, cam_index: int, pipeline_duration: float):
    """Persist perf data for one date-cam group."""
    try:
        from pathlib import Path
        perf_path = DB_PATH.parent / f"perf_{date}_cam{cam_index}.json"
        perf.dump(perf_path, metadata={
            "date": date,
            "cam": cam_index,
            "pipeline_duration": round(pipeline_duration, 2),
            "monitor_summary": monitor.stages_data(),
            "perf_summary": perf.summary_by_stage(),
        })
        perf.reset()
    except Exception:
        logger.warning("failed to dump perf data for %s cam%d", date, cam_index)


def _recover_interrupted(db: VlogDatabase):
    """Roll back RENDERING tasks that were interrupted by a crash."""
    rows = db.conn.execute(
        "SELECT date, cam_index FROM render_tasks WHERE status='RENDERING'"
    ).fetchall()
    for r in rows:
        logger.warning("recovering interrupted render: %s cam%d -> PENDING", r["date"], r["cam_index"])
        db.set_render_status(r["date"], r["cam_index"], "PENDING")


def run_pipeline(skip_render: bool = False) -> dict:
    """Main entry: scan input, run pipeline for all pending date-cam groups."""
    config = load_config()
    db = VlogDatabase()
    monitor = get_monitor()
    monitor.start()

    try:
        _recover_interrupted(db)
        scan_directory(db)
        groups = get_date_cam_groups(db)
        if not groups:
            logger.info("no date-cam groups found")
            return {"total": 0, "ok": 0, "failed": 0, "skipped": 0}

        total = 0
        ok = 0
        failed = 0
        skipped = 0

        for date, cam_index in groups:
            if db.is_render_completed(date, cam_index):
                logger.info("skip %s cam%d: already completed", date, cam_index)
                skipped += 1
                continue

            # 空间熔断
            from src.utils import check_disk_space, OUTPUT_DIR, cleanup_resources
            if not check_disk_space(OUTPUT_DIR, min_gb=20):
                logger.error("Pipeline paused due to low disk space.")
                break

            total += 1
            
            # 全域异常装甲与故障隔离
            try:
                ok_flag = process_date_cam(db, date, cam_index, skip_render=skip_render)
                if ok_flag:
                    ok += 1
                else:
                    failed += 1
            except Exception:
                logger.exception("CRITICAL ERROR: process_date_cam crashed for %s cam%d. Skipping to next day.", date, cam_index)
                failed += 1

            # 强制深层 GC 和僵尸进程清洗（为下一天的战役打扫战场）
            cleanup_resources()

        logger.info("all done: total=%d ok=%d failed=%d skipped=%d", total, ok, failed, skipped)
        logger.info("\n%s", monitor.summary())
        return {"total": total, "ok": ok, "failed": failed, "skipped": skipped}

    finally:
        monitor.shutdown()
        db.close()
