"""Run the production analysis pipeline in isolated paths and retain measurements.

Run explicitly with uv; the workbench only reads completed receipts, never
starts independent concurrent hardware jobs. No production DB or media writes.
"""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import re
import sys
import time
import uuid

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--params", type=Path, help="JSON object of dotted candidate parameters")
    parser.add_argument("--store-dir", type=Path, default=Path("temp/benchmark_replays"))
    parser.add_argument("--resume", type=Path, help="Resume a matching checkpoint directory")
    parser.add_argument("--max-ranges", type=int, help="Checkpoint and stop after this many newly processed date/camera ranges")
    args = parser.parse_args()
    if args.max_ranges is not None and args.max_ranges < 1:
        parser.error("--max-ranges must be positive")
    project = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project))
    from src.core.replay_journal import run_lease
    store = args.store_dir.resolve()
    if not args.resume and len(list(store.glob("*/receipt.json"))) >= 256:
        raise ValueError("Replay store reached 256 receipts; explicitly archive selected old jobs before continuing")
    run = args.resume.resolve() if args.resume else store / uuid.uuid4().hex
    if args.resume:
        if not run.is_dir() or not (run / "checkpoint.json").is_file():
            raise ValueError("Resume requires a formal checkpoint directory")
    else:
        run.mkdir(parents=True, exist_ok=False)
    with run_lease(run):
        attempt = run / "attempts" / uuid.uuid4().hex
        attempt.mkdir(parents=True)
        logging.basicConfig(level=logging.INFO, handlers=[logging.FileHandler(attempt / "run.log", encoding="utf-8"), logging.StreamHandler()])
        logging.info("Isolated replay: %s; attempt: %s", run, attempt)
        execute(args, run, project)


def execute(args, run, project):
    dataset_path, config_path = args.dataset.resolve(), args.config.resolve()
    from src.core.replay_journal import ReplayJournal, atomic_json
    # Core eagerly imports database/utils; update their snapshots before stages.
    import src.core.config as core
    paths = {"OUTPUT_DIR": run / "output", "TEMP_DIR": run / "temp",
             "TEMP_STAGING_DIR": run / "temp/staging", "TEMP_BATCHES_DIR": run / "temp/batches",
             "TEMP_SCRIPTS_DIR": run / "temp/scripts", "LOGS_DIR": run / "logs", "DB_PATH": run / "data/vlog.db"}
    for name, value in paths.items():
        for module_name, module in list(sys.modules.items()):
            if module_name == "src.core" or module_name.startswith("src.core."):
                if hasattr(module, name):
                    setattr(module, name, value)
    from src.core.benchmark import load_benchmark_dataset, assess_replay_quality
    from src.core.tuner import _set_nested_param
    from src.core.replay_evidence import content_sha256, production_code_sha256, model_path, replay_receipt, sample_key, receipt_sha256, runtime_identity, project_predictions
    from src.core.profiles import resolve_profile_config
    from src.core.database import VlogDatabase

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if args.params:
        params = json.loads(args.params.read_text(encoding="utf-8"))
        if not isinstance(params, dict):
            raise ValueError("Candidate params must be a JSON object")
        for name, value in params.items():
            _set_nested_param(raw, name, value)
    raw.setdefault("yolo", {})["model_path"] = str(model_path(raw).resolve())
    settings = run / "settings.yaml"
    if args.resume:
        if yaml.safe_load(settings.read_text(encoding="utf-8")) != raw:
            raise ValueError("Requested replay settings differ from frozen settings")
    else:
        settings.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False), encoding="utf-8")
    config = core.load_config(settings)
    samples, metadata = load_benchmark_dataset(dataset_path)
    if not samples:
        raise ValueError("Dataset has no reference intervals")
    code_hash = production_code_sha256(project)
    runtime = runtime_identity()
    model = model_path(config)
    model_hash = content_sha256(model) if model.exists() else None
    journal = ReplayJournal(run, {"raw_config": raw, "samples": samples, "metadata": metadata,
                                  "code_sha256": code_hash, "runtime": runtime,
                                  "model_sha256": model_hash}, resume=bool(args.resume))
    if not args.resume:
        atomic_json(run / "references.json", {"metadata": metadata, "samples": samples})
    sources, inputs = [], {}
    for sample in samples:
        path = Path(sample["filepath"])
        match = re.search(r"_(\d{14})_(\d{14})", path.name)
        if not match:
            raise ValueError(f"Source lacks scanner timestamps: {path.name}")
        start, end = match.groups()
        if sample["date"] != start[:8]:
            raise ValueError("Reference date does not match source placement")
        key = str(path)
        if key in inputs and inputs[key] != (sample["cam_index"], sample["date"], start, end):
            raise ValueError("Source has conflicting camera/date assignments")
        inputs[key] = (sample["cam_index"], sample["date"], start, end)
    for index, path in enumerate(inputs, 1):
        logging.info("Freezing source identity %s/%s: %s", index, len(inputs), Path(path).name)
        sources.append(journal.freeze_source(path, content_sha256))
    # The archive owns paths derived from PROJECT_ROOT rather than core.TEMP_DIR.
    # Isolate it explicitly while retaining the real project/model identity.
    from src.stages import archiver
    archiver.DEFAULT_ARCHIVE_DIR = run / "data/feedback_archive"
    archiver.DEFAULT_IMAGES_DIR = archiver.DEFAULT_ARCHIVE_DIR / "images"
    archiver.DEFAULT_MANIFEST = archiver.DEFAULT_ARCHIVE_DIR / "manifest.jsonl"
    archiver.CACHE_DIR = run / "temp/audit_cache"
    from src.pipeline import process_date_cam
    from src.hardware.monitor import get_monitor, get_perf
    from src.core.utils import cleanup_resources, cleanup_staging_files
    db = VlogDatabase(core.DB_PATH)
    monitor = get_monitor()
    outcomes = []
    try:
        for path, (camera, date, start, end) in inputs.items():
            duration = (datetime.strptime(end, "%Y%m%d%H%M%S") - datetime.strptime(start, "%Y%m%d%H%M%S")).total_seconds()
            if duration <= 0:
                raise ValueError("Source placement must have positive duration")
            db.add_file_task(path, camera, date, start, end, duration)
        def range_result(date, camera):
            paths = {path for path, value in inputs.items() if (value[1], value[0]) == (date, camera)}
            # Filter by frozen source membership; DB camera indexes can be remapped.
            result = {"tasks": [dict(row) for row in db.conn.execute("SELECT * FROM file_tasks ORDER BY filepath") if row["filepath"] in paths],
                      "segments": [dict(row) for row in db.conn.execute("SELECT * FROM segments ORDER BY id") if row["filepath"] in paths]}
            references = [sample for sample in samples if sample["filepath"] in paths]
            predictions = project_predictions(references, result["tasks"], result["segments"])
            covered = len(result["tasks"]) == len(paths) and all(type(value) is bool for value in predictions.values())
            return result, covered
        monitor.start()
        processed = 0
        for date, camera in sorted({(value[1], value[0]) for value in inputs.values()}):
            key = json.dumps([date, camera])
            prior, covered = range_result(date, camera)
            if journal.range_complete(key, prior, covered):
                logging.info("Verified completed range; reuse %s cam%s", date, camera)
                outcomes.append({"date": date, "cam_index": camera, "success": True, "reused": True})
                continue
            if args.max_ranges is not None and processed >= args.max_ranges:
                print(json.dumps({"run": str(run), "status": "INCOMPLETE", "resume_required": True}))
                return
            get_perf().reset()
            success = False
            started = time.monotonic()
            try:
                success = process_date_cam(db, date, camera, skip_render=True, force_render=True,
                                           dashboard_enabled=False, config=resolve_profile_config(config))
            finally:
                current, covered = range_result(date, camera)
                journal.record_range(key, current, success, covered)
                journal.value["ranges"][key]["wall_s"] = time.monotonic() - started
                journal.save()
                cleanup_resources()
            processed += 1
            outcomes.append({"date": date, "cam_index": camera, "success": bool(success and covered), "reused": False})
            atomic_json(run / "result.json", {"outcomes": outcomes,
                                              "tasks": [dict(row) for row in db.conn.execute("SELECT * FROM file_tasks")],
                                              "segments": [dict(row) for row in db.conn.execute("SELECT * FROM segments")]})
        result = {"outcomes": outcomes,
                  "tasks": [dict(row) for row in db.conn.execute("SELECT * FROM file_tasks")],
                  "segments": [dict(row) for row in db.conn.execute("SELECT * FROM segments")]}
        atomic_json(run / "result.json", result)
        if not all(item["success"] for item in outcomes):
            raise RuntimeError("Production replay failed; diagnostics retained, no receipt published")
        for source in sources:
            stat = Path(source["filepath"]).stat()
            if (stat.st_size, stat.st_mtime_ns) != (source["size"], source["mtime_ns"]):
                raise ValueError("Source changed during replay")
        if production_code_sha256(project) != code_hash or (content_sha256(model) if model.exists() else None) != model_hash:
            raise ValueError("Code/model changed during replay")
        if runtime_identity() != runtime:
            raise ValueError("Runtime changed during replay")
        receipt = replay_receipt(config, samples, sources, result["tasks"], result["segments"],
                                 code_sha256=code_hash, model_sha256=model_hash)
        verified = [sample for sample in samples if sample.get("review_status") in ("independent_visual", "independent_human")
                    and type(sample.get("reference_active")) is bool and sample.get("review_evidence")]
        covered = [sample for sample in verified if type(receipt["predictions"][sample_key(sample)]) is bool]
        receipt["quality_gate"] = assess_replay_quality(covered, [receipt["predictions"][sample_key(sample)] for sample in covered])
        receipt["uncovered_verified_intervals"] = len(verified) - len(covered)
        if receipt["uncovered_verified_intervals"]:
            receipt["quality_gate"]["verdict"] = "UNAVAILABLE"
        receipt["payload_sha256"] = receipt_sha256(receipt)
        if (run / "receipt.json").exists():
            existing = json.loads((run / "receipt.json").read_text(encoding="utf-8"))
            if existing.get("payload_sha256") != receipt_sha256(existing):
                raise ValueError("Existing replay receipt checksum mismatch")
            stable = lambda value: {key: item for key, item in value.items() if key not in ("created_at", "payload_sha256")}
            if stable(existing) != stable(receipt):
                raise ValueError("Existing replay receipt differs from verified results")
            logging.info("Existing complete receipt retained after checkpoint verification")
        else:
            atomic_json(run / "receipt.json", receipt)
        print(json.dumps({"receipt": str(run / "receipt.json"), "quality_gate": receipt["quality_gate"]["verdict"]}))
    finally:
        monitor.shutdown()
        cleanup_resources()
        cleanup_staging_files()
        db.close()


if __name__ == "__main__":
    main()
