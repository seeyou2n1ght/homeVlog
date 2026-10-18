"""HomeVlog 轻量重渲染与即时秒级浓缩引擎 (Re-render Engine)

用于人工审核打标后，直接基于修正后的 segments 表重新压制成片。
具备前置素材物理存在性与 NAS 网络存储可访问性校验、断点保护与进度追踪。
"""

import time
import logging
import json
import shutil
import threading
import hashlib
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.utils import OUTPUT_DIR, load_config
from src.database import VlogDatabase
from src.core.database import is_skipped_source
from src.timeline import partition_timeline_by_batches
from src.renderer import RenderInterrupted, build_batch_render, concat_output_files
from src.stages.companion import promote_vlog_assets, cleanup_delivery_candidate

logger = logging.getLogger("homevlog.rerender")


def capture_review_inputs(db, date, cam_index):
    """Capture all rows in one read transaction, preserving caller transactions."""
    with db._lock:
        connection = db.conn
        owns_transaction = not connection.in_transaction
        if owns_transaction:
            connection.execute("BEGIN")
        try:
            rows = db.get_all_file_tasks_for_date(date, cam_index)
        finally:
            if owns_transaction:
                connection.rollback()
    if not rows:
        raise ValueError("No file tasks found in database for date")
    from src.core.profiles import resolve_camera_config
    config = resolve_camera_config(load_config(), cam_index=cam_index, filepath=rows[0]["filepath"])
    snapshot = hashlib.sha256(json.dumps([rows, config], sort_keys=True, default=str).encode()).hexdigest()
    return rows, config, snapshot


def build_review_batches(rows, date, config):
    from src.stages.timeline import build_timeline_from_rows
    healthy_rows = [row for row in rows if not is_skipped_source(row)]
    timeline = build_timeline_from_rows(healthy_rows, date, config=config, resolve_presence=False)
    if not timeline:
        raise ValueError("Timeline is empty after filtering")
    batches = partition_timeline_by_batches(timeline, batch_max_files=config.get("render", {}).get("batch_max_files", 1))
    return healthy_rows, timeline, batches


def render_review_batch(batch, index, date, cam_index, rows, config, **options):
    from src.utils import parse_res
    output = config.get("output", {})
    width, height = parse_res(output.get("resolution", "1920x1080"))
    return build_batch_render(
        batch_segs=batch, bi=index, enc_for_batch=config.get("render", {}).get("encoder", "nv"),
        fps=output.get("fps", 20), width=width, height=height,
        seg_cfg=config.get("segment", {}), out_cfg=output, audio_cfg=output.get("audio", {}),
        date=date, cam_index=cam_index, rows=rows, config=config, **options,
    )


def inspect_review_render(db, date, cam_index):
    rows, config, snapshot = capture_review_inputs(db, date, cam_index)
    can_proceed, _, missing = check_source_files_accessibility(rows)
    if not can_proceed:
        return {"status": "UNAVAILABLE", "missing_files": missing}
    healthy, _, batches = build_review_batches(rows, date, config)
    reports = []
    for index, batch in enumerate(batches):
        before = len(reports)
        render_review_batch(batch, index, date, cam_index, healthy, config,
                            inspect_only=True, cache_observer=reports.append)
        if len(reports) == before:
            raise RuntimeError("Render inspection interrupted before cache decision")
        report = reports[-1]
        files = set(report["source_files"])
        report["review_intervals"] = [
            {key: review.get(key) for key in ("start_time", "end_time", "manual_label", "reviewed_at")}
            for row in healthy if row["filepath"] in files for review in row.get("human_reviews", [])
        ]
    return {"status": "AVAILABLE", "snapshot_sha256": snapshot, "batches": reports,
            "total_batches": len(batches), "reusable_batches": sum(report["cache_hit"] for report in reports),
            "scope": "primary path only; runtime revalidates cache and may use fallback"}


def check_source_files_accessibility(file_tasks: list[dict]) -> tuple[bool, list[str], list[str]]:
    """前置校验素材文件的物理存在性与可访问性。
    
    返回:
        (can_proceed, valid_files, missing_files)
    """
    valid_files = []
    missing_files = []

    for t in file_tasks:
        if is_skipped_source(t):
            continue
        fp_str = t.get("filepath")
        if not fp_str:
            continue
        p = Path(fp_str)
        try:
            if p.is_file() and p.stat().st_size > 0:
                valid_files.append(fp_str)
            else:
                missing_files.append(fp_str)
        except Exception:
            missing_files.append(fp_str)

    if missing_files:
        return False, valid_files, missing_files

    return True, valid_files, missing_files


class ReRenderManager:
    """管理异步重渲染任务的生命周期与进度状态。"""
    
    _instance = None
    _lock = threading.RLock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_manager()
            return cls._instance

    def _init_manager(self):
        self.tasks: dict[str, dict[str, Any]] = {}
        self.active_threads: dict[str, threading.Thread] = {}
        self.cancel_flags: dict[str, threading.Event] = {}

    def get_task_key(self, date: str, cam_index: int) -> str:
        return f"{date}_cam{cam_index}"

    def get_status(self, date: str, cam_index: int) -> dict[str, Any]:
        key = self.get_task_key(date, cam_index)
        with self._lock:
            task = self.tasks.get(key)
            if not task:
                return {"status": "IDLE", "date": date, "cam_index": cam_index}
            return deepcopy(task)

    def cancel_task(self, date: str, cam_index: int) -> bool:
        key = self.get_task_key(date, cam_index)
        with self._lock:
            event = self.cancel_flags.get(key)
            task = self.tasks.get(key)
            if event and not event.is_set() and task and task.get("status") in ("CHECKING", "RENDERING", "CONCATING"):
                event.set()
                task["status"] = "CANCELLING"
                return True
        return False

    def start_rerender(self, db: VlogDatabase, date: str, cam_index: int, output_version: str = "reviewed", expected_snapshot: str | None = None) -> dict[str, Any]:
        key = self.get_task_key(date, cam_index)
        
        # 检查是否已有同任务在执行
        with self._lock:
            existing = self.tasks.get(key)
            active = self.active_threads.get(key)
            if (active and active.is_alive()) or (existing and existing.get("status") in ("CHECKING", "RENDERING", "CONCATING", "CANCELLING")):
                return {"error": "Task already running", "status": (existing or {}).get("status", "CHECKING")}

            cancel_event = threading.Event()
            self.cancel_flags[key] = cancel_event

            task_state = {
                "status": "CHECKING",
                "date": date,
                "cam_index": cam_index,
                "progress": 0,
                "current_batch": 0,
                "total_batches": 0,
                "missing_files": [],
                "output_file": "",
                "elapsed_s": 0.0,
                "error": "",
                "start_time": time.time(),
                "batch_results": [],
            }
            self.tasks[key] = task_state

            # Register before releasing the lock so simultaneous starts cannot race.
            t = threading.Thread(
                target=self._run_render_worker,
                args=(db, date, cam_index, output_version, cancel_event, key, expected_snapshot),
                daemon=True,
            )
            self.active_threads[key] = t
            t.start()

        return {"status": "CHECKING", "message": "Render task initiated"}

    def _run_render_worker(
        self, db, date, cam_index, output_version, cancel_event, task_key, expected_snapshot=None,
    ):
        from src.core.process_lock import day_task_lease
        from src.stages.scanner import resolve_output_filename
        from src.stages import renderer
        try:
            rows,config,snapshot=capture_review_inputs(db,date,cam_index)
            base=resolve_output_filename(config.get('output',{}).get('naming','DailyVlog_{date}_{mac}.mp4'),
                                         date,cam_index,sample_filepath=rows[0]['filepath'],config=config)
            name=f'{Path(base).stem}_{output_version}{Path(base).suffix}' if output_version else base
            with day_task_lease(db.db_path,renderer.TEMP_DIR,OUTPUT_DIR/name,date,cam_index):
                from src.core.delivery_recovery import recover_output
                recover_output(OUTPUT_DIR/name)
                self._run_render_worker_owned(db,date,cam_index,output_version,cancel_event,task_key,
                                              expected_snapshot if expected_snapshot is not None else snapshot)
        except Exception as error:
            logger.exception('Review day task ownership failed: %s',error)
            with self._lock:
                if task_key in self.tasks:
                    self.tasks[task_key].update(status='CANCELLED' if cancel_event.is_set() else 'FAILED',error=str(error))

    def _run_render_worker_owned(
        self,
        db: VlogDatabase,
        date: str,
        cam_index: int,
        output_version: str,
        cancel_event: threading.Event,
        task_key: str,
        expected_snapshot: str | None = None,
    ):
        t0 = time.time()
        logger.info("Re-render started for %s cam%d (version=%s)", date, cam_index, output_version)

        def update_state(**kwargs):
            with self._lock:
                if task_key in self.tasks:
                    if cancel_event.is_set() and kwargs.get("status") in ("CHECKING", "RENDERING", "CONCATING"):
                        kwargs["status"] = "CANCELLING"
                    self.tasks[task_key].update(kwargs)
                    self.tasks[task_key]["elapsed_s"] = round(time.time() - t0, 1)

        candidate = None
        try:
            # 1. 查询当天任务行
            file_tasks, config, snapshot = capture_review_inputs(db, date, cam_index)
            update_state(snapshot_sha256=snapshot)
            if expected_snapshot is not None and snapshot != expected_snapshot:
                update_state(status="FAILED", error="Review or configuration changed after preview; inspect again")
                return

            # 2. 前置物理存在性与可访问性探测
            update_state(status="CHECKING", progress=5)
            can_proceed, valid_files, missing_files = check_source_files_accessibility(file_tasks)
            update_state(missing_files=missing_files)

            if not can_proceed:
                err_msg = f"素材文件不完整：{len(missing_files)} 个不可达，请检查 NAS 网络共享或物理存储连接"
                logger.error("Re-render blocked: %s", err_msg)
                update_state(status="FAILED", error=err_msg)
                return

            if cancel_event.is_set():
                update_state(status="CANCELLED")
                return

            # 3. 基于修正后的 segments 表构建时间轴
            update_state(status="RENDERING", progress=10)
            healthy_rows, timeline, batches = build_review_batches(file_tasks, date, config)

            # 4. 批次切分
            out_cfg = config.get("output", {})
            audio_cfg = out_cfg.get("audio", {})

            total_batches = len(batches)
            update_state(total_batches=total_batches, current_batch=0, progress=15)

            # 5. 渲染输出路径
            out_dir = OUTPUT_DIR
            out_dir.mkdir(parents=True, exist_ok=True)
            naming = out_cfg.get("naming", "DailyVlog_{date}_{mac}.mp4")
            sample_path = healthy_rows[0]["filepath"] if healthy_rows else (file_tasks[0]["filepath"] if file_tasks else None)
            from src.scanner import resolve_output_filename
            base_name = resolve_output_filename(naming, date, cam_index, sample_filepath=sample_path, config=config)
            if output_version:
                stem = Path(base_name).stem
                suffix = Path(base_name).suffix
                final_out_name = f"{stem}_{output_version}{suffix}"
            else:
                final_out_name = base_name

            final_output_path = out_dir / final_out_name
            candidate = final_output_path.with_name(final_output_path.stem + ".pending.mp4")

            # 6. 执行批次渲染
            batch_outputs = []
            fps = out_cfg.get("fps", 20)
            batch_results = []

            for bi, b_segs in enumerate(batches):
                if cancel_event.is_set():
                    update_state(status="CANCELLED")
                    return

                update_state(
                    current_batch=bi + 1,
                    progress=15 + int((bi / total_batches) * 70)
                )

                # 调用底层批次渲染
                result = {"batch_index": bi, "status": "RUNNING", "attempts": []}
                batch_results.append(result)

                def on_cache_decision(report):
                    result["attempts"].append(report)
                    update_state(batch_results=deepcopy(batch_results))

                res_batch = render_review_batch(b_segs, bi, date, cam_index, healthy_rows, config,
                                               cache_observer=on_cache_decision)
                result["status"] = "COMPLETED" if res_batch else "FAILED"
                result["cache_hit"] = result["attempts"][-1]["cache_hit"] if result["attempts"] else None
                update_state(batch_results=deepcopy(batch_results))

                if not res_batch:
                    err = f"Batch {bi+1}/{total_batches} render failed"
                    logger.error(err)
                    update_state(status="FAILED", error=err)
                    return

                batch_outputs.append(Path(res_batch))

            if cancel_event.is_set():
                update_state(status="CANCELLED")
                return

            # 7. 合并最终成片
            update_state(status="CONCATING", progress=90)
            cleanup_batches = config.get("render", {}).get("cleanup_batches_on_success", False)
            if len(batch_outputs) == 1:
                shutil.copy2(batch_outputs[0], candidate)
                ok = True
            else:
                ok = concat_output_files(batch_outputs, candidate)

            if not ok or not candidate.exists():
                update_state(status="FAILED", error="Final concat failed")
                return

            # 8. 同步更新标准化成片伴随资产包 (.srt + .meta.json)
            try:
                from src.pipeline import _save_vlog_companion_assets
                from src.scanner import resolve_camera_identity
                cam_disp = None
                try:
                    s_dir = str(Path(sample_path).parent) if sample_path else ""
                    cam_disp, _ = resolve_camera_identity(s_dir, cam_index=cam_index, config=config)
                except Exception:
                    pass
                tot_files = len(healthy_rows)
                tot_dur = sum(float(r.get("file_duration") or 0.0) for r in healthy_rows)
                assets_ok = _save_vlog_companion_assets(
                    output_path=candidate,
                    date=date,
                    cam_index=cam_index,
                    cam_display=cam_disp,
                    total_files=tot_files,
                    total_input_dur=tot_dur,
                    elapsed_wall=time.time() - t0,
                    db=db,
                    config=config,
                    rows=healthy_rows,
                )
                if not assets_ok:
                    update_state(status="FAILED", error="Companion asset generation failed")
                    return
            except Exception as exc:
                logger.warning("Failed to refresh companion assets after rerender: %s", exc)
                raise

            from src.hardware.render_cache import render_fingerprint, save_manifest
            fingerprint = render_fingerprint(
                valid_files, json.dumps([vars(s) for s in timeline], sort_keys=True),
                "reviewed", fps, out_cfg, audio_cfg, config,
            )
            # Commit and terminal status share the cancellation lock.
            with self._lock:
                with promote_vlog_assets(candidate, final_output_path, config, keep_previous=True,fingerprint=fingerprint,
                                         context={'role':'review','date':date,'cam_index':cam_index}):
                    save_manifest(final_output_path, fingerprint)
                    if cancel_event.is_set():
                        raise InterruptedError("Review render cancelled before commit")

                # 9. 成功终态
                file_size_mb = round(final_output_path.stat().st_size / (1024 * 1024), 2)
                logger.info("Re-render finished: %s (%.1f MB) in %.1fs", final_output_path.name, file_size_mb, time.time() - t0)
                update_state(
                    status="COMPLETED",
                    progress=100,
                    output_file=str(final_output_path),
                    file_size_mb=file_size_mb
                )
            if cleanup_batches:
                for batch in batch_outputs:
                    try:
                        batch.unlink(missing_ok=True)
                        batch.with_name(batch.name + ".json").unlink(missing_ok=True)
                    except OSError:
                        logger.warning("Review delivery committed; could not remove batch %s", batch)

        except RenderInterrupted as e:
            logger.warning("Re-render interrupted; preserving completed batches: %s", e)
            update_state(status="CANCELLED", error=str(e))
        except Exception as e:
            logger.exception("Re-render worker encountered unhandled exception: %s", e)
            update_state(status="CANCELLED" if cancel_event.is_set() else "FAILED", error=str(e))
        finally:
            if candidate is not None:
                cleanup_delivery_candidate(candidate)
