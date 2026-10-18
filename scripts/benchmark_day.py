"""Measure one frozen day through the production pipeline in isolated paths.

The workload contains explicit historical task/segment/review rows and source
stats. No scan, production database modification, or historical asset reuse.
Run hardware measurements sequentially. A completed run is not an A/B gain.
"""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import shutil
import sys
import time
import uuid


def seed_workload(db, workload, mode):
    """Seed only an empty experiment DB; retain source groups and human reviews."""
    if db.conn.execute("SELECT count(*) FROM file_tasks").fetchone()[0]:
        raise ValueError("Day benchmark requires a new empty database")
    tables = {"camera_registry": workload["camera_registry"], "file_tasks": workload["tasks"],
              "human_reviews": workload["human_reviews"]}
    if mode == "render_only":
        tables["segments"] = workload["segments"]
    with db.conn:
        for table, rows in tables.items():
            allowed = {row["name"] for row in db.conn.execute(f"PRAGMA table_info({table})")}
            for original in rows:
                row = dict(original)
                if table == "file_tasks" and mode == "full":
                    start, end = row["file_start_time"], row["file_end_time"]
                    # Use scanner placement from the actual filename, not the
                    # historical lazy-probe-adjusted file_end_time.
                    import re
                    match = re.search(r"_(\d{14})_(\d{14})", Path(row["filepath"]).name)
                    if not match:
                        raise ValueError("Source lacks scanner placement")
                    start, end = match.groups()
                    row.update(file_start_time=start, file_end_time=end,
                               file_duration=(datetime.strptime(end, "%Y%m%d%H%M%S") - datetime.strptime(start, "%Y%m%d%H%M%S")).total_seconds(),
                               prescreen_status="PENDING", prescreen_result=None,
                               analysis_status="PENDING", analysis_segments=None,
                               retry_count=0, error_msg=None, has_audio=0,
                               duration_verified=0, processing_fingerprint=None, decode_completion=None)
                if not set(row).issubset(allowed):
                    raise ValueError(f"Unknown workload columns for {table}")
                columns = list(row)
                db.conn.execute(f"INSERT INTO {table} ({','.join(columns)}) VALUES ({','.join('?' for _ in columns)})",
                                [row[column] for column in columns])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, type=Path)
    parser.add_argument("--code-root", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("render_only", "full"))
    parser.add_argument("--store-dir", required=True, type=Path)
    args = parser.parse_args()
    code_root = args.code_root.resolve()
    sys.path.insert(0, str(code_root))
    workload = json.loads(args.workload.read_text(encoding="utf-8"))
    if workload.get("schema") != 1 or not workload.get("tasks"):
        raise ValueError("Missing frozen day workload")
    date, camera = workload["date"], workload["cam_index"]
    tasks = workload["tasks"]
    paths = {task["filepath"] for task in tasks}
    if len(paths) != len(tasks) or any((task["date"], task["cam_index"]) != (date, camera) for task in tasks):
        raise ValueError("Workload must contain unique sources from exactly one date/camera")
    from src.core.replay_journal import run_lease, ReplayJournal, atomic_json, digest
    from src.core.replay_evidence import production_code_sha256, content_sha256, runtime_identity, model_path
    if production_code_sha256(code_root) != workload["code_sha256"] or runtime_identity() != workload["runtime"]:
        raise ValueError("Frozen code/runtime changed")
    config_path = code_root / "config/settings.yaml"
    if content_sha256(config_path) != workload["config_file_sha256"]:
        raise ValueError("Frozen config changed")
    run = args.store_dir.resolve() / uuid.uuid4().hex
    run.mkdir(parents=True, exist_ok=False)
    shutil.copy2(__file__, run / "harness.py")
    logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(run / "run.log", encoding="utf-8"), logging.StreamHandler()])
    logging.info("Isolated day benchmark: %s (%s)", run, args.mode)
    import src.core.config as core
    replacements = {"PROJECT_ROOT": Path(workload["source_project_root"]),
                    "OUTPUT_DIR": run / "output", "TEMP_DIR": run / "temp",
                    "TEMP_STAGING_DIR": run / "temp/staging", "TEMP_BATCHES_DIR": run / "temp/batches",
                    "TEMP_SCRIPTS_DIR": run / "temp/scripts", "LOGS_DIR": run / "logs", "DB_PATH": run / "data/vlog.db"}
    for module_name, module in list(sys.modules.items()):
        if module_name == "src" or module_name.startswith("src."):
            for name, path in replacements.items():
                if hasattr(module, name):
                    setattr(module, name, path)
    config = core.load_config(config_path)
    from src.core.profiles import resolve_camera_config
    effective = resolve_camera_config(config, cam_index=camera, filepath=tasks[0]["filepath"])
    if effective != workload["historical_effective_config"]:
        raise ValueError("Effective day config differs from captured analysis")
    model = model_path(config)
    if content_sha256(model) != workload["model_sha256"]:
        raise ValueError("Frozen model changed")
    from src.stages import archiver
    archiver.DEFAULT_ARCHIVE_DIR = run / "data/feedback_archive"
    archiver.DEFAULT_IMAGES_DIR = archiver.DEFAULT_ARCHIVE_DIR / "images"
    archiver.DEFAULT_MANIFEST = archiver.DEFAULT_ARCHIVE_DIR / "manifest.jsonl"
    archiver.CACHE_DIR = run / "temp/audit_cache"
    from src.core.database import VlogDatabase
    from src.pipeline import process_date_cam
    from src.hardware.monitor import get_monitor, get_perf
    from src.core.utils import cleanup_resources, cleanup_staging_files
    from src.hardware.render_cache import processing_fingerprint
    historical_stats = {item["filepath"]: item for item in workload["historical_source_stats"]}
    with run_lease(run):
        journal = ReplayJournal(run, {"workload_sha256": content_sha256(args.workload),
                                     "harness_sha256": content_sha256(__file__), "mode": args.mode,
                                     "code_sha256": workload["code_sha256"], "code_root": str(code_root),
                                     "runtime": workload["runtime"]})
        atomic_json(run / "workload.json", workload)
        sources = []
        freeze_started = time.monotonic()
        for index, task in enumerate(tasks, 1):
            logging.info("Freeze day source %s/%s: %s", index, len(tasks), Path(task["filepath"]).name)
            source = journal.freeze_source(task["filepath"], content_sha256)
            expected = historical_stats.get(task["filepath"])
            if not expected or (source["size"], source["mtime_ns"]) != (expected["size"], expected["mtime_ns"]):
                raise ValueError("Historical source stat changed; fixed analysis cannot be reused")
            sources.append(source)
        db = VlogDatabase(core.DB_PATH)
        monitor = get_monitor()
        status = {"status": "RUNNING", "mode": args.mode, "source_freeze_s": time.monotonic() - freeze_started,
                  "source_count": len(tasks), "run": str(run)}
        atomic_json(run / "measurement.json", status)
        try:
            seed_workload(db, workload, args.mode)
            if args.mode == "render_only" and any(task["processing_fingerprint"] != processing_fingerprint(task["filepath"], effective) for task in tasks):
                raise ValueError("Captured analysis fingerprint changed")
            before = digest([db.get_all_file_tasks_for_date(date, camera), db.get_all_segments_for_date(date, camera)])
            get_perf().reset()
            monitor.start()
            started = time.monotonic()
            success = process_date_cam(db, date, camera, force_render=True, dashboard_enabled=False, config=config)
            status.update(pipeline_call_wall_s=time.monotonic() - started, pipeline_success=bool(success))
            if not success:
                raise RuntimeError("Day pipeline failed; diagnostics and valid batches retained")
            after = digest([db.get_all_file_tasks_for_date(date, camera), db.get_all_segments_for_date(date, camera)])
            if args.mode == "render_only" and before != after:
                raise ValueError("Render-only measurement changed the captured analysis/reviews")
            for source in sources:
                stat = Path(source["filepath"]).stat()
                if (stat.st_size, stat.st_mtime_ns) != (source["size"], source["mtime_ns"]):
                    raise ValueError("Source changed during day measurement")
            if (production_code_sha256(code_root) != workload["code_sha256"]
                    or content_sha256(config_path) != workload["config_file_sha256"]
                    or content_sha256(args.workload) != journal.value["identity"]["workload_sha256"]
                    or content_sha256(__file__) != journal.value["identity"]["harness_sha256"]
                    or content_sha256(model) != workload["model_sha256"]
                    or runtime_identity() != workload["runtime"]):
                raise ValueError("Day measurement identity changed")
            status.update(status="PIPELINE_COMPLETED_PENDING_MEDIA_ACCEPTANCE", analysis_unchanged=before == after,
                          analysis_before_sha256=before, analysis_after_sha256=after,
                          source_sha256_captured=len(sources), quality_gate="UNAVAILABLE")
        except BaseException as error:
            status.update(status="FAILED_OR_INTERRUPTED", error_type=type(error).__name__, error=str(error))
            raise
        finally:
            atomic_json(run / "measurement.json", status)
            monitor.shutdown()
            cleanup_resources()
            cleanup_staging_files()
            db.close()
    print(json.dumps(status, ensure_ascii=False))


if __name__ == "__main__":
    main()
