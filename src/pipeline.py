import json
import hashlib
import logging
import subprocess
import time
import threading
import queue
import stat
import shutil
from contextvars import copy_context
from src.core.run_observation import RunObservation, current_run, current_batch
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.ui import (
    PipelineDashboard,
    print_batch_startup_banner,
    print_error_summary,
    print_startup_banner,
    print_summary_card,
)
from src.core.database import VlogDatabase
from src.stages.scanner import scan_directory, get_date_cam_groups, ScanResult
from src.stages.prescreen import prescreen_file
from src.stages.detector import MotionDetector, detect_audio_activity
from src.stages.renderer import RenderInterrupted, build_batch_render, concat_output_files
from src.hardware.ffmpeg import MediaTerminationUnconfirmed
from src.stages.companion import (
    dump_perf as _dump_perf,
    build_headline as _build_headline,
    save_vlog_companion_assets as _save_vlog_companion_assets,
    promote_vlog_assets,
    cleanup_delivery_candidate,
    build_batch_summary_item,
)
from src.core.config import (
    load_config,
    OUTPUT_DIR,
    uses_software_pipeline,
)
from src.hardware.scheduler import (
    WorkStealingManager,
    RenderBatchItem,
    DualEndedBatchQueue,
)
from src.hardware.ffmpeg import FFmpegProcessRegistry
from src.hardware.lifecycle import cleanup_resources
from src.core.utils import (
    InputStorageUnavailable,
    check_disk_space,
    check_input_storage,
    get_input_dirs,
    is_directory_accessible,
    register_dashboard,
    unregister_dashboard,
)
from dataclasses import dataclass, field
from contextlib import nullcontext
from src.hardware.monitor import get_monitor, get_perf, PerfRecord

logger = logging.getLogger("homevlog")



from src.stages.pipeline_types import (
    PipelineTask,
    AnalysisQueue,
    RenderMessageQueue,
)


def _compute_qsv_queue_weight(
    segments,
    file_dur_s: float = 0.0,
    render_cfg: dict | None = None,
    *,
    single_file_batch: bool = True,
) -> tuple[bool, float]:
    """Compute queue sort / QSV admission weight reflecting real decode cost.

    Accounts for full-decode PRESENCE/MICRO_MOTION (and non-sparse NIGHT_STATIONARY)
    spans >= 120s so QSV steals fast virtual-concat / sparse-night batches while heavy
    daytime full-decode batches remain on NVDEC/NVENC.
    """
    cfg = render_cfg or {}
    night_sparse_enabled = bool(cfg.get("night_sparse_enabled", True))
    min_night_dur = max(4.0, float(cfg.get("night_sparse_min_duration_s", 16.0)))

    active_source_duration = 0.0
    normal_speed_duration = 0.0
    presence_micro_dur = 0.0
    night_total_dur = 0.0
    night_sparse_dur = 0.0
    night_sparse_count = 0
    active_count = 0

    for seg in segments:
        if hasattr(seg, "end_time") and hasattr(seg, "start_time"):
            dur = max(0.0, float(seg.end_time) - float(seg.start_time))
        elif getattr(seg, "duration", None) is not None:
            dur = max(0.0, float(seg.duration))
        else:
            dur = max(0.0, float(getattr(seg, "end_in_file", 0.0)) - float(getattr(seg, "start_in_file", 0.0)))

        state = getattr(seg, "state", "")
        is_1x = bool(getattr(seg, "is_active_motion", False)) or state in ("DYNAMIC", "DYNAMIC_AUDIO")
        is_active = bool(getattr(seg, "is_dynamic", False)) or (bool(state) and state != "STATIC") or is_1x
        if not is_active:
            continue

        active_count += 1
        active_source_duration += dur
        if is_1x:
            normal_speed_duration += dur
        elif state == "NIGHT_STATIONARY":
            night_total_dur += dur
            if dur >= min_night_dur:
                night_sparse_dur += dur
                night_sparse_count += 1
        else:
            presence_micro_dur += dur

    has_dynamic = active_source_duration > 0.0
    effective_dyn_dur = (
        normal_speed_duration
        if normal_speed_duration > 0.0
        else (min(60.0, active_source_duration) if has_dynamic else 0.0)
    )

    night_sparse_active = (
        single_file_batch
        and night_sparse_enabled
        and night_sparse_count > 0
        and (
            night_sparse_count == active_count
            or (
                night_sparse_dur >= max(min_night_dur, 30.0)
                and night_sparse_dur / max(0.001, active_source_duration) >= 0.10
            )
        )
    )
    non_sparse_night_dur = (night_total_dur - night_sparse_dur) if night_sparse_active else night_total_dur
    full_decode_non_dyn = presence_micro_dur + non_sparse_night_dur

    if full_decode_non_dyn >= 120.0:
        effective_dyn_dur += 0.35 * full_decode_non_dyn
    elif float(file_dur_s) >= 1800.0 and active_source_duration >= 1200.0:
        effective_dyn_dur += 0.25 * max(0.0, active_source_duration - normal_speed_duration)

    return has_dynamic, effective_dyn_dur


class StreamingOrchestrator:
    """终极流式管线编排器：实现预筛、分析、渲染的全重叠并发执行与自适应硬件调度。"""

    def __init__(
        self,
        db: VlogDatabase,
        date: str,
        cam_index: int,
        config: dict,
        render_enabled: bool = True,
        dashboard_enabled: bool = True,
        cam_name: str | None = None,
    ):
        # Admission precedes scheduler/UI construction and cancellation reset.
        FFmpegProcessRegistry.reset_interrupted()
        self.db = db
        self.date = date
        self.cam_index = cam_index
        self.cam_name = cam_name
        self.config = config
        self.render_enabled = render_enabled

        # 异构硬件自适应调度器
        self.work_stealing = WorkStealingManager(config=self.config)



        # 队列定义
        self.prescreen_queue = queue.Queue()
        self.analysis_queue = AnalysisQueue()
        pipe_cfg = config.get("pipeline", {})
        render_cfg = config.get("render", {})
        if pipe_cfg.get("cost_priority", True) and render_cfg.get("batch_max_files", 1) == 1:
            self.analysis_queue.cost_priority = True
        self.render_batch_queue = RenderMessageQueue()

        # 配置提取
        pipe_cfg = config.get("pipeline", {})
        render_cfg = config.get("render", {})
        
        io_limit = int(config.get("hardware", {}).get("max_io_concurrency", 8))
        det_cfg = config.get("detection", {})
        analysis_configured = "analysis_max_workers" in det_cfg
        prescreen_configured = "prescreen_parallel" in det_cfg
        analysis_workers = int(det_cfg.get("analysis_max_workers", 4))
        render_workers = max(1, int(render_cfg.get("max_concurrency", 2)))
        requested_prescreen = int(det_cfg.get("prescreen_parallel", 8))
        self.analysis_workers = max(1, analysis_workers)
        self.render_workers = render_workers
        self.prescreen_workers = max(1, min(requested_prescreen,
                                            max(1, io_limit - self.analysis_workers)))
        reserved_analysis = self.analysis_workers if analysis_configured else 0
        reserved_prescreen = self.prescreen_workers if prescreen_configured else 0
        # A render batch charges one I/O slot per input for its whole FFmpeg
        # lifetime. Keep enough slots for analysis metadata/audio/decode work;
        # otherwise a batch of exactly max_io_concurrency starves every
        # analysis worker and turns overlap into a deadlock.
        render_io_budget = max(1, (io_limit - reserved_analysis - reserved_prescreen) // render_workers)
        requested_batch = int(render_cfg.get("batch_max_files", 4))
        self.batch_max_files = max(1, min(requested_batch, render_io_budget))
        self.analysis_queue.cost_priority = self.batch_max_files == 1
        if self.batch_max_files < requested_batch:
            logger.warning(
                "batch_max_files reduced from %d to %d to reserve %d I/O slots for analysis",
                requested_batch, self.batch_max_files, io_limit - self.batch_max_files,
            )
        self.render_delay = pipe_cfg.get("render_start_delay", 5)

        # 状态控制
        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
        # Only successful completion of both upstream queues opens tail capacity.
        # stop_event alone also represents failures and cancellation.
        self.upstream_drained = threading.Event()
        self._render_interruption_reason: str | None = None
        self._media_cleanup_failure: str | None = None
        self._storage_unavailable_failure: str | None = None
        self.render_finished_event = threading.Event()
        self.batch_paths = []
        self.batch_lock = threading.Lock()
        self.error_lock = threading.Lock()
        self.errors: list[str] = []
        self.render_worker_stats: dict[str, dict] = {}

        # 时间邻域先验保护状态追踪
        self._prev_task_map: dict[str, str] = {}
        self._prescreen_results: dict[str, str] = {}
        self._prescreen_results_lock = threading.Lock()
        self._prescreen_results_condition = threading.Condition(self._prescreen_results_lock)

        # 冷启动预热保护：避免启动瞬间队列无积压导致 SJF 短作业优先失效
        self._analysis_warmup_event = threading.Event()

        # 异步前瞻预分期与编队动态竞价追踪
        self._prefetch_queue = queue.Queue()
        self._prefetched_files = set()
        self._prefetched_lock = threading.Lock()
        self._active_nv_renders = 0
        self._active_qsv_renders = 0

        # Rich 仪表盘 (tqdm 已下线，统一由 PipelineDashboard 呈现)
        self.observation = current_run.get()
        self.dashboard_enabled = dashboard_enabled
        self.dashboard: PipelineDashboard | None = None
        self._prescreen_t0 = 0.0
        self._analysis_t0 = 0.0

    def _guard_worker(self, worker, *args):
        token = current_run.set(self.observation)
        try:
            worker(*args)
        except BaseException as exc:
            logger.exception("Pipeline worker terminated unexpectedly")
            self._add_error(f"render batch pipeline worker failed: {exc}")
            self.abort_event.set()
            self.stop_event.set()
            self.render_finished_event.set()
            from src.hardware.ffmpeg import FFmpegProcessRegistry
            FFmpegProcessRegistry.kill_all()

        finally:
            current_run.reset(token)

    def _add_error(self, message: str):
        with self.error_lock:
            self.errors.append(message)
        if self.dashboard is not None:
            self.dashboard.add_alert(message)

    def _stop_for_unconfirmed_media(self, context: str, error: MediaTerminationUnconfirmed):
        """Stop producers before a quarantined reader can overlap a retry."""
        from src.hardware.ffmpeg import FFmpegProcessRegistry
        self._media_cleanup_failure = f"{context} cleanup unconfirmed: {error}"
        self.abort_event.set()
        self.stop_event.set()
        self._add_error(self._media_cleanup_failure)
        FFmpegProcessRegistry.mark_interrupted()
        FFmpegProcessRegistry.kill_all()
        logger.error("Stopping pipeline with unresolved media reader: %s", self._media_cleanup_failure)

    def _stop_for_storage_unavailable(self, context: str, error: BaseException):
        """Abort workers when source storage goes offline so pending files stay PENDING."""
        from src.hardware.ffmpeg import FFmpegProcessRegistry
        with self.error_lock:
            if self._storage_unavailable_failure is not None:
                return
            self._storage_unavailable_failure = f"{context}: {error}"
        self.abort_event.set()
        self.stop_event.set()
        self.render_finished_event.set()
        self._add_error(f"input storage unavailable: {error}")
        FFmpegProcessRegistry.kill_all()
        logger.error("Stopping pipeline due to unavailable input storage (%s): %s", context, error,
                     extra={"reason": "INPUT_STORAGE_UNAVAILABLE", "date": self.date, "cam": self.cam_index,
                            "disposition": "stop_batch"})

    @staticmethod
    def _is_source_dir_offline(filepaths: list[str] | tuple[str, ...]) -> bool:
        """Return True if any non-trivial source parent directory cannot be enumerated."""
        for fp in filepaths:
            parent_dir = Path(fp).parent
            if str(parent_dir) not in ("", ".") and not is_directory_accessible(parent_dir):
                return True
        return False

    def _sync_queue_levels(self) -> None:
        """同步三阶段积压水位与调度器状态到仪表盘。"""
        if self.dashboard is None:
            return
        self.dashboard.set_queue_status(
            prescreen_q=self.prescreen_queue.qsize(),
            analysis_q=self.analysis_queue.qsize(),
            render_q=self.render_batch_queue.qsize(),
        )
        self.dashboard.set_scheduler_state(
            self.work_stealing.state,
            self.work_stealing.active_nv_decoders,
            self.work_stealing.max_nv_decoders,
        )

    def _queue_analysis_task(self, task) -> None:
        task["_analysis_queued_at"] = time.monotonic()
        self.analysis_queue.put(task)

    def _needs_prefetch(self, filepath):
        with self._prefetched_lock:
            return filepath in self._prefetched_files

    def _lookahead_prefetch_worker(self):
        """后台异步 I/O Worker：超前拉取即将压制的源文件至本地 SSD，消除 GPU 串行 I/O 等待气泡。"""
        from src.algorithms.segment import ACTIVE_STATES, SegmentState
        from src.stages.renderer import _stage_source_for_render
        from src.stages.timeline import build_timeline_from_rows

        stream_config = dict(self.config)
        stream_config["presence"] = {**self.config.get("presence", {}), "enabled": False}
        while not self.abort_event.is_set() and (not self.stop_event.is_set() or not self._prefetch_queue.empty()):
            try:
                fp = self._prefetch_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if fp is None:
                self._prefetch_queue.task_done()
                break
            try:
                try:
                    rows = self.db.get_all_file_tasks_for_date(self.date, self.cam_index, filepaths=[fp])
                    timeline = build_timeline_from_rows(
                        rows, self.date, target_files=[fp], config=stream_config, resolve_presence=False,
                    ) if rows else []
                except Exception:
                    timeline = []
                pure_dynamic = (SegmentState.DYNAMIC, SegmentState.DYNAMIC_AUDIO, "DYNAMIC", "DYNAMIC_AUDIO")
                if timeline and all(segment.state in pure_dynamic for segment in timeline):
                    continue
                _stage_source_for_render(fp, needed=lambda: self._needs_prefetch(fp))
            except InputStorageUnavailable as exc:
                self._stop_for_storage_unavailable("render prefetch", exc)
                return
            except Exception:
                logger.debug("Render prefetch failed", exc_info=True)
            finally:
                self._prefetch_queue.task_done()

    def _await_prior_prescreen(self, filepath: str) -> bool:
        """Resolve chronological prior state without holding media/hardware leases."""
        prev_fp = self._prev_task_map.get(filepath)
        if prev_fp is None:
            return False
        with self._prescreen_results_condition:
            while prev_fp not in self._prescreen_results:
                if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                    raise InterruptedError("cancelled waiting for predecessor prescreen")
                self._prescreen_results_condition.wait(timeout=0.1)
            return self._prescreen_results[prev_fp] == "SUSPICIOUS"

    def _prescreen_worker(self, gpu: str = "qsv", worker_id: str = ""):
        """预筛 Worker：将扫描到的文件进行快速筛选。"""
        worker_name = worker_id or f"pre_{gpu}"
        while not self.abort_event.is_set() and (not self.stop_event.is_set() or not self.prescreen_queue.empty()):
            try:
                task = self.prescreen_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            filepath = task["filepath"]
            duration = float(task.get("file_duration") or 0.0)
            t0 = time.monotonic()

            try:
                prev_fp = self._prev_task_map.get(filepath)
                with self._prescreen_results_lock:
                    prior_known = prev_fp is None or prev_fp in self._prescreen_results
                    is_prior_active = self._prescreen_results.get(prev_fp) == "SUSPICIOUS"
                res = prescreen_file(
                    filepath, duration, self.config, gpu=gpu,
                    is_prior_active=is_prior_active,
                    prior_active_resolver=(None if prior_known else
                                           lambda: self._await_prior_prescreen(filepath)),
                )
                if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                    continue
                if (res["status"] == "STATIC" and res.get("has_audio") and
                        self.config.get("audio_vad", {}).get("prescreen_audio_gate", False)):
                    # A visual negative does not need the full video analysis
                    # path just to discover whether its audio is silent.
                    # Decode audio features only; failures stay conservative.
                    try:
                        visual_result = json.loads(res.get("result_json") or "{}")
                        audio_events, audio_stats = detect_audio_activity(
                            filepath, duration, self.config
                        )
                        if audio_events:
                            res["status"] = "SUSPICIOUS"
                            res["result_json"] = json.dumps({
                                "mode": "keyframes_audio_gate",
                                "visual_result": visual_result,
                                "audio_events": audio_events,
                                "vad_stats": audio_stats,
                            })
                        else:
                            res["result_json"] = json.dumps({
                                "mode": "keyframes_audio_gate",
                                "visual_result": visual_result,
                                "audio_events": [],
                                "vad_stats": audio_stats,
                            })
                    except MediaTerminationUnconfirmed:
                        raise
                    except Exception as exc:
                        if self.abort_event.is_set():
                            return
                        logger.warning(
                            "audio gate failed for %s; keeping conservative analysis path: %s",
                            Path(filepath).name, exc,
                        )
                        res["status"] = "SUSPICIOUS"
                result_json = res.get("result_json", "")
                prescreen_has_audio = res.get("has_audio")
                if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                    continue
                self.db.set_prescreen_result(filepath, res["status"], result_json,
                                             has_audio=prescreen_has_audio, error_msg=res.get("error"))
                if res.get("media_duration", 0) > 0:
                    self.db.set_file_metadata(filepath, int(prescreen_has_audio or 0), res["media_duration"])
                    task["file_duration"] = res["media_duration"]
                    task["duration_verified"] = 1
                if prescreen_has_audio is not None:
                    task["has_audio"] = int(prescreen_has_audio)
                if res["status"] in ("STATIC", "SUSPICIOUS"):
                    from src.core.identity import processing_fingerprint
                    self.db.set_processing_fingerprint(filepath, processing_fingerprint(filepath, self.config))
                with self._prescreen_results_condition:
                    self._prescreen_results[filepath] = res["status"]
                    self._prescreen_results_condition.notify_all()

                extra = {"status": res["status"]}
                try:
                    extra.update(json.loads(result_json or "{}"))
                except (json.JSONDecodeError, TypeError):
                    if res.get("error"):
                        extra["error"] = res["error"]
                t_done = time.monotonic()
                get_perf().add(
                    PerfRecord(
                        stage="prescreen",
                        file=Path(filepath).name,
                        gpu=gpu,
                        duration=round(t_done - t0, 3),
                        extra=extra,
                        start_time=round(t0, 3),
                        end_time=round(t_done, 3),
                        worker=worker_name,
                    )
                )

                if res["status"] == "SUSPICIOUS":
                    res_json = res.get("result_json") or ""
                    task["prescreen_result"] = res_json
                    if "keyframes_audio_gate" in res_json:
                        task["is_audio_gated_static"] = True
                        try:
                            gate_data = json.loads(res_json)
                            task["audio_events"] = gate_data.get("audio_events", [])
                        except Exception:
                            task["audio_events"] = []
                    self._queue_analysis_task(task)
                    if not self._analysis_warmup_event.is_set():
                        if self.analysis_queue.qsize() >= 10 or self.prescreen_queue.empty():
                            self._analysis_warmup_event.set()
                    if self.dashboard is not None:
                        self.dashboard.update_analysis(
                            completed=self.dashboard.analysis_done,
                            total=self.dashboard.analysis_total + 1,
                        )
                elif res["status"] == "FAILED":
                    self._add_error(f"prescreen returned FAILED: {Path(filepath).name}")
                    self.render_batch_queue.put(
                        {"filepath": filepath, "status": "FAILED"}
                    )
                else:
                    self.render_batch_queue.put(
                        {"filepath": filepath, "status": "STATIC"}
                    )

                if self.dashboard is not None:
                    el_s = max(0.1, time.monotonic() - self._prescreen_t0)
                    n_done = self.dashboard.prescreen_done + 1
                    avg_s = el_s / max(1, n_done)
                    self.dashboard.update_prescreen(
                        completed=n_done,
                        latest_file=Path(filepath).name,
                        speed_str=f"均速 {avg_s:.1f}s/个",
                    )
                    self._sync_queue_levels()

            except MediaTerminationUnconfirmed as exc:
                self._stop_for_unconfirmed_media(f"prescreen {Path(filepath).name}", exc)
                try:
                    self.db.set_prescreen_result(filepath, "FAILED", "", error_msg=str(exc))
                except Exception:
                    logger.exception("Cannot persist FAILED after unconfirmed prescreen stop: %s", filepath)
                    self._add_error(f"prescreen FAILED persistence failed: {Path(filepath).name}")
                break
            except InputStorageUnavailable as exc:
                self._stop_for_storage_unavailable(f"prescreen {Path(filepath).name}", exc)
                break
            except Exception as exc:
                if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                    continue
                logger.exception("Streaming: prescreen failed for %s", filepath)
                self.db.set_prescreen_result(filepath, "FAILED", "", error_msg=str(exc))
                self._add_error(f"prescreen failed: {Path(filepath).name}")
                t_err = time.monotonic()
                get_perf().add(
                    PerfRecord(
                        stage="prescreen",
                        file=Path(filepath).name,
                        gpu=gpu,
                        duration=round(t_err - t0, 3),
                        extra={"status": "ERROR"},
                        start_time=round(t0, 3),
                        end_time=round(t_err, 3),
                        worker=worker_name,
                    )
                )
                # 必须回传 FAILED 消息推进渲染保序滑窗，否则窗口停摆导致整条流水线死锁
                self.render_batch_queue.put(
                    {"filepath": filepath, "status": "FAILED"}
                )
            finally:
                with self._prescreen_results_condition:
                    # Exceptions/cancellation must also unblock successors.
                    self._prescreen_results.setdefault(filepath, "FAILED")
                    self._prescreen_results_condition.notify_all()
                self.prescreen_queue.task_done()
        if not self._analysis_warmup_event.is_set():
            self._analysis_warmup_event.set()

    def _execute_analysis_task(
        self,
        task: dict,
        filepath: str,
        file_start_offset: float,
        detector: MotionDetector,
        yolo_verifier,
        gpu: str,
        perf,
        t0: float,
        worker_id: str = "",
        defer_post: bool = False,
    ):
        from src.algorithms.segment import build_segments

        if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
            return

        analysis_queue_wait = max(0.0, t0 - float(task.get("_analysis_queued_at") or t0))

        if self.dashboard is not None:
            dur = float(task.get("file_duration") or 0.0)
            dur_label = f"({dur/60:.1f}m)" if dur > 60 else (f"({dur:.0f}s)" if dur > 0 else "")
            self.dashboard.update_analysis(
                completed=self.dashboard.analysis_done,
                latest_file=Path(filepath).name,
                speed_str=f"正在分析 {dur_label}".strip(),
            )

        # Check for visually static files with audio events (Fast Path)
        is_audio_gated = bool(task.get("is_audio_gated_static", False))
        audio_events = task.get("audio_events")
        if not is_audio_gated or not audio_events:
            prescreen_raw = task.get("prescreen_result")
            if not prescreen_raw:
                row = self.db.get_file_task(filepath)
                prescreen_raw = row.get("prescreen_result") if row else None
            if prescreen_raw and "keyframes_audio_gate" in prescreen_raw:
                try:
                    data = json.loads(prescreen_raw)
                    audio_events = data.get("audio_events", [])
                    if audio_events:
                        is_audio_gated = True
                except Exception:
                    pass

        if is_audio_gated and audio_events:
            total_audio_dur = sum(max(0.0, float(e[1]) - float(e[0])) for e in audio_events)
            if len(audio_events) > 12 or total_audio_dur > 45.0:
                logger.info(
                    "Bypassing audio_gate_fast_path for %s: %d events (%.1fs) require full visual+YOLO verification",
                    Path(filepath).name, len(audio_events), total_audio_dur,
                )
                is_audio_gated = False

        if is_audio_gated and audio_events:
            from src.algorithms.segment import build_audio_gated_segments
            seg_cfg = self.config.get("segment", {})
            file_dur = float(task.get("file_duration") or 0.0)
            if file_dur <= 0.0:
                from src.stages.prescreen import get_duration
                actual_dur = get_duration(filepath)
                if actual_dur and actual_dur > 0:
                    file_dur = actual_dur
            segments = build_audio_gated_segments(
                filepath=filepath,
                file_duration=file_dur,
                file_start_offset=file_start_offset,
                audio_events=audio_events,
                pre_roll=min(float(seg_cfg.get("pre_roll", 1.0)), 0.4),
                post_roll=min(float(seg_cfg.get("post_roll", 1.5)), 0.6),
                gap_tolerance=min(float(seg_cfg.get("gap_tolerance", 1.5)), 1.0),
            )
            self.db.set_analysis_result(filepath, "ANALYZED", segments)
            from src.core.identity import processing_fingerprint
            self.db.set_processing_fingerprint(filepath, processing_fingerprint(filepath, self.config))
            t_ana_done = time.monotonic()
            perf.add(
                PerfRecord(
                    stage="analysis",
                    file=Path(filepath).name,
                    gpu=gpu,
                    duration=round(t_ana_done - t0, 3),
                    extra={
                        "status": "ANALYZED",
                        "mode": "audio_gate_fast_path",
                        "audio_events": len(audio_events),
                        "decode_time_s": 0.0,
                        "analysis_time_s": 0.0,
                        "yolo_time_s": 0.0,
                        "analysis_queue_wait_s": round(
                            max(0.0, t0 - float(task.get("_analysis_queued_at") or t0)), 3
                        ),
                    },
                    start_time=round(t0, 3),
                    end_time=round(t_ana_done, 3),
                    worker=worker_id or "ana",
                )
            )
            dynamic_duration = sum(
                (s.end_time - s.start_time)
                for s in segments
                if s.state == "DYNAMIC_AUDIO"
            )
            self.render_batch_queue.put(
                {
                    "filepath": filepath,
                    "status": "ANALYZED",
                    "is_heavy": False,
                    "dynamic_duration": dynamic_duration,
                }
            )
            if self.dashboard is not None:
                self.dashboard.update_analysis(
                    completed=self.dashboard.analysis_done + 1,
                    latest_file=Path(filepath).name,
                )
                self._sync_queue_levels()
            return

        previous_extent = json.loads(task["decode_completion"]) if task.get("decode_completion") else {}
        labels, yolo_buffer = detector.analyze(
            filepath,
            start_offset=file_start_offset,
            file_duration=float(task.get("file_duration") or 0.0),
            has_audio=task.get("has_audio"),
            duration_verified=bool(task.get("duration_verified")),
            declared_duration=previous_extent.get("declared_duration_s"),
        )
        if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
            return
        
        gpu = getattr(detector, "decode_gpu", gpu)
        if hasattr(detector, 'has_audio_detected'):
            final_has_audio = max(int(task.get("has_audio") or 0), int(detector.has_audio_detected or 0))
            self.db.set_file_metadata(filepath, final_has_audio, getattr(detector, "file_duration_detected", None),
                                      decode_completion=getattr(detector, "decode_completion", None))

        detector_lp = dict(getattr(detector, "last_perf", {}) or {})
        detector_fps = getattr(detector, "fps", 5.0)
        effective_gpu = gpu

        if labels:
            def _finalize_post():
                try:
                    if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                        return
                    seg_cfg = self.config.get("segment", {})
                    motion_absorb_threshold = float(seg_cfg.get("motion_absorb_energy_threshold", 12.0))
                    segments = build_segments(
                        labels,
                        filepath,
                        min_motion_dur=seg_cfg.get("min_motion_duration", 2.0),
                        min_static_dur=seg_cfg.get("min_static_duration", 8.0),
                        file_offset=file_start_offset,
                        gap_tolerance=seg_cfg.get("gap_tolerance", 1.5),
                        apply_smoothing=seg_cfg.get("apply_smoothing", False),
                        pre_roll=seg_cfg.get("pre_roll", 1.0),
                        post_roll=seg_cfg.get("post_roll", 1.5),
                        motion_absorb_energy_threshold=motion_absorb_threshold,
                    )

                    yolo_before = len(segments)
                    is_yolo_active = False
                    yolo_telemetry = {}
                    if yolo_verifier:
                        is_yolo_active = True
                        yolo_device = "cpu" if effective_gpu == "cpu" else self.config.get("hardware", {}).get("device", "cpu")
                        # yolo_buffer 帧键以解码实际 effective_fps 为基准，禁止使用固定的 detector.fps
                        effective_fps = detector_lp.get("effective_fps") or detector_fps
                        segments = yolo_verifier.verify(
                            filepath, segments, gpu=effective_gpu, device=yolo_device, frames_buffer=yolo_buffer, analysis_fps=effective_fps,
                            labels=labels,
                        )
                        yolo_telemetry = dict(getattr(yolo_verifier, "last_telemetry", {}) or {})
                        from src.algorithms.segment import _merge_same_state, _filter_short
                        segments = _merge_same_state(segments, gap_tolerance=seg_cfg.get("gap_tolerance", 1.5))
                        if seg_cfg.get("apply_smoothing", True):
                            segments = _filter_short(
                                segments,
                                min_motion=seg_cfg.get("min_motion_duration", 2.0),
                                min_static=seg_cfg.get("min_static_duration", 8.0),
                                gap_tolerance=seg_cfg.get("gap_tolerance", 1.5),
                                motion_absorb_energy_threshold=motion_absorb_threshold,
                            )
                    if (yolo_verifier is None or not getattr(yolo_verifier, "enabled", True)) and self.config.get("yolo", {}).get("enabled", False):
                        for seg in segments:
                            if seg.is_dynamic:
                                seg.needs_review = True
                                seg.review_reason = "YOLO_UNAVAILABLE"
                    from src.algorithms.segment import refine_activity_segments
                    segments = refine_activity_segments(segments, labels, self.config)
                    yolo_after = len(segments)

                    self.db.set_analysis_result(filepath, "ANALYZED", segments)
                    from src.core.identity import processing_fingerprint
                    self.db.set_processing_fingerprint(filepath, processing_fingerprint(filepath, self.config))

                    t_ana_done = time.monotonic()
                    dur_total = round(t_ana_done - t0, 3)
                    lp = detector_lp
                    decode_time = lp.get("decode_time", 0.0)
                    analysis_time = lp.get("analysis_time", 0.0)
                    yolo_time = yolo_telemetry.get("yolo_duration", 0.0)
                    yolo_lock_wait = yolo_telemetry.get("yolo_lock_wait", 0.0)
                    yolo_infer = yolo_telemetry.get("yolo_infer_time", 0.0)
                    frames_count = lp.get("frames", 0)
                    overhead = max(0.0, round(dur_total - decode_time - analysis_time - yolo_time, 3))

                    perf.add(
                        PerfRecord(
                            stage="analysis",
                            file=Path(filepath).name,
                            gpu=effective_gpu,
                            duration=dur_total,
                            frames=frames_count,
                            fps=round(frames_count / max(0.001, dur_total), 1),
                            extra={
                                "status": "ANALYZED",
                                "is_stolen": bool(task.get("is_stolen", False)),
                                "segments_before_yolo": yolo_before,
                                "segments_after_yolo": yolo_after,
                                "yolo_streaming_verify": is_yolo_active,
                                "decode_time_s": decode_time,
                                "analysis_time_s": analysis_time,
                                "yolo_time_s": yolo_time,
                                "yolo_lock_wait_s": yolo_lock_wait,
                                "yolo_infer_s": yolo_infer,
                                "overhead_s": overhead,
                                "analysis_queue_wait_s": round(analysis_queue_wait, 3),
                                "decode_fps": round(frames_count / max(0.001, decode_time), 1) if decode_time > 0 else 0.0,
                                "motion_fps": round(frames_count / max(0.001, analysis_time), 1) if analysis_time > 0 else 0.0,
                                **lp,
                                **yolo_telemetry,
                            },
                            start_time=round(t0, 3),
                            end_time=round(t_ana_done, 3),
                            worker=worker_id or "ana",
                        )
                    )
                    task_file_dur = float(task.get("file_duration") or 0.0)
                    has_dynamic, effective_dyn_dur = _compute_qsv_queue_weight(
                        segments,
                        file_dur_s=task_file_dur,
                        render_cfg=self.config.get("render", {}),
                        single_file_batch=True,
                    )
                    self.render_batch_queue.put(
                        {
                            "filepath": filepath,
                            "status": "ANALYZED",
                            "is_heavy": has_dynamic,
                            "dynamic_duration": effective_dyn_dur,
                        }
                    )
                    if self.dashboard is not None:
                        el_s = max(0.1, time.monotonic() - self._analysis_t0)
                        n_done = self.dashboard.analysis_done + 1
                        task_el = max(0.1, time.monotonic() - t0)
                        frames_done = lp.get("frames", 0)
                        fps_val = frames_done / task_el if frames_done else 0.0
                        hw_label = {"qsv": "核显", "cpu": "CPU"}.get(effective_gpu, "独显")
                        speed_str = f"{fps_val:.1f} 帧/秒（{hw_label}解码）" if fps_val > 0 else f"均速 {el_s / n_done:.1f}s/个"
                        self.dashboard.update_analysis(
                            completed=n_done,
                            latest_file=Path(filepath).name,
                            speed_str=speed_str,
                        )
                        self._sync_queue_levels()
                finally:
                    if isinstance(yolo_buffer, dict):
                        yolo_buffer.clear()

            if defer_post:
                return _finalize_post
            _finalize_post()
            return None
        else:
            decode_frames = detector_lp.get("frames", 0)
            err_msg = (
                f"corrupted media or unreadable stream ({decode_frames} frames decoded)"
                if decode_frames < 2
                else "analysis produced no labels"
            )
            completion = getattr(detector, "decode_completion", {})
            if not completion.get("complete", True):
                err_msg = f"decode {completion.get('reason', 'FAILED')}: {completion.get('error', '')}"
            skip_corrupt = bool(self.config.get("recovery", {}).get("skip_corrupt", False))
            if skip_corrupt and decode_frames < 2:
                logger.warning(
                    "Source %s has unreadable stream (%d frames); safely degraded/skipped by skip_corrupt policy",
                    Path(filepath).name, decode_frames,
                )
                self.db.set_analysis_result(filepath, "SKIPPED_CORRUPT", "", error_msg=err_msg)
                queue_status = "SKIPPED_CORRUPT"
            else:
                self.db.set_analysis_result(filepath, "FAILED", "", error_msg=err_msg)
                self._add_error(f"{err_msg}: {Path(filepath).name}")
                queue_status = "FAILED"
            t_ana_fail = time.monotonic()
            perf.add(
                PerfRecord(
                    stage="analysis",
                    file=Path(filepath).name,
                    gpu=gpu,
                    duration=round(t_ana_fail - t0, 3),
                    extra={
                        "status": "CORRUPT_SKIPPED" if queue_status == "SKIPPED_CORRUPT" else "NO_LABELS",
                        "analysis_queue_wait_s": round(analysis_queue_wait, 3),
                    },
                    start_time=round(t0, 3),
                    end_time=round(t_ana_fail, 3),
                    worker=worker_id or "ana",
                )
            )
            self.render_batch_queue.put(
                {"filepath": filepath, "status": queue_status}
            )

            if self.dashboard is not None:
                el_s = max(0.1, time.monotonic() - self._analysis_t0)
                n_done = self.dashboard.analysis_done + 1
                task_el = max(0.1, time.monotonic() - t0)
                frames_done = detector_lp.get("frames", 0)
                fps_val = frames_done / task_el if frames_done else 0.0
                hw_label = {"qsv": "核显", "cpu": "CPU"}.get(gpu, "独显")
                speed_str = f"{fps_val:.1f} 帧/秒（{hw_label}解码）" if fps_val > 0 else f"均速 {el_s / n_done:.1f}s/个"
                self.dashboard.update_analysis(
                    completed=n_done,
                    latest_file=Path(filepath).name,
                    speed_str=speed_str,
                )
                self._sync_queue_levels()
            return None

    def _analysis_worker(
        self,
        fixed_gpu: str | None = None,
        worker_id: str = "",
        cpu_max_duration_s: float | None = None,
    ):
        """分析 Worker：对 SUSPICIOUS 文件进行运动检测与目标验证 (支持异构自适应工作窃取)。"""
        default_gpu = "cpu" if fixed_gpu == "cpu" else "qsv"
        default_detector = MotionDetector(self.config, decode_gpu=default_gpu)
        detector_cuda: MotionDetector | None = None
        
        # [性能极限] 在线程生命周期内复用验证器实例，避免重复初始化开销
        yolo_verifier = None
        yolo_cfg = self.config.get("yolo", {})
        if yolo_cfg.get("enabled", False) and yolo_cfg.get("streaming_verify", True):
            from src.algorithms.yolo_verifier import YoloVerifier
            try:
                if fixed_gpu == "cpu" and uses_software_pipeline(self.config):
                    yolo_verifier = YoloVerifier(self.config, device="cpu")
                else:
                    yolo_verifier = YoloVerifier(self.config)
            except Exception as exc:
                # Keep consuming the queue; absence of semantics must not strand unfinished tasks.
                logger.warning("YOLO initialization failed; retaining visual/audio motion: %s", exc)
                self._add_error("YOLO unavailable; motion retained and marked for review")
            
        perf = get_perf()

        # 冷启动预热保护：等待预筛先形成候选积压，确保 SJF 短作业优先正常排序出清
        if not self._analysis_warmup_event.is_set():
            self._analysis_warmup_event.wait(timeout=2.0)

        is_cuda_heavy_worker = fixed_gpu != "cpu" and worker_id == "ana_0"
        heavy_dur = float(self.config.get("scheduler", {}).get("heavy_file_duration_s", 600.0))
        steal_max_dur = float(self.config.get("scheduler", {}).get("steal_sublongest_max_duration", heavy_dur))
        steal_min_dur = float(self.config.get("scheduler", {}).get("steal_sublongest_min_duration", 300.0))
        post_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"yolo_{worker_id or 'ana'}")
        pending_post_future = None
        try:
            while not self.abort_event.is_set() and (not self.stop_event.is_set() or not self.analysis_queue.empty()):
                task = None
                is_stolen = False
                try:
                    if cpu_max_duration_s is not None:
                        if hasattr(self.analysis_queue, "get_short"):
                            task = self.analysis_queue.get_short(
                                timeout=0.5, max_duration=cpu_max_duration_s
                            )
                        else:
                            task = self.analysis_queue.get(timeout=0.5)
                    elif is_cuda_heavy_worker:
                        # 1. 优先消费夜间巨型长任务 (>= heavy_dur, LPT)
                        if hasattr(self.analysis_queue, "has_heavy_tasks") and self.analysis_queue.has_heavy_tasks(heavy_dur):
                            task = self.analysis_queue.get_heavy(timeout=0.5, min_duration=heavy_dur)
                        elif hasattr(self.analysis_queue, "steal_sublongest"):
                            # 2. 夜间重任务已出清：检查四门准入门限并动态窃取次长切片
                            q_size = self.analysis_queue.qsize()
                            if hasattr(self.work_stealing, "can_worker_0_steal") and self.work_stealing.can_worker_0_steal(queue_size=q_size, heavy_tasks_cleared=True):
                                task = self.analysis_queue.steal_sublongest(max_duration=steal_max_dur, min_duration=steal_min_dur)
                                if task is not None:
                                    is_stolen = True
                            if task is None:
                                task = self.analysis_queue.get(timeout=0.5)
                        else:
                            task = self.analysis_queue.get(timeout=0.5)
                    else:
                        task = self.analysis_queue.get(timeout=0.5)
                except queue.Empty:
                    if cpu_max_duration_s is not None and self.prescreen_queue.unfinished_tasks == 0:
                        has_short = (
                            self.analysis_queue.has_short_tasks(cpu_max_duration_s)
                            if hasattr(self.analysis_queue, "has_short_tasks")
                            else not self.analysis_queue.empty()
                        )
                        if not has_short:
                            break
                    continue

                filepath = task["filepath"]
                file_dur = float(task.get("file_duration") or 0.0)
                if fixed_gpu != "cpu" and not is_cuda_heavy_worker and file_dur >= heavy_dur and getattr(self.analysis_queue, "cost_priority", False):
                    # 杜绝 QSV Worker 误领 55 分钟夜间长文件导致慢速爬行瘫痪核显编队；
                    # 若预筛未完、队列中仍有 <heavy_dur 轻切片、或 NVDEC Worker 0 空闲则退回重任务；
                    # 当所有轻切片与预筛已清空且 Worker 0 正在忙于巨型文件时，允许 QSV 协助处理 <1800s 的中重型文件以消除尾部串行阻塞
                    has_short = (
                        self.analysis_queue.has_short_tasks(heavy_dur)
                        if hasattr(self.analysis_queue, "has_short_tasks")
                        else True
                    )
                    should_defer_heavy = (
                        not self.prescreen_queue.empty()
                        or getattr(self.prescreen_queue, "unfinished_tasks", 0) > 0
                        or self.work_stealing.active_nv_decoders == 0
                        or (
                            self.analysis_queue.qsize() > 0
                            and (file_dur >= 1800.0 or has_short)
                        )
                    )
                    if should_defer_heavy:
                        self.analysis_queue.put(task)
                        self.analysis_queue.task_done()
                        time.sleep(0.2)
                        continue
                t0 = time.monotonic()
                analysis_queue_wait = max(
                    0.0, t0 - float(task.get("_analysis_queued_at") or t0)
                )

                task_done_deferred = False
                try:
                    from src.core.identity import ts_to_unix

                    file_start_ts = ts_to_unix(task["file_start_time"])
                    file_start_offset = max(
                        file_start_ts - ts_to_unix(self.date + "000000"), 0.0
                    )

                    allow_defer = "_execute_analysis_task" not in self.__dict__
                    extra_kw = {"defer_post": True} if allow_defer else {}

                    if fixed_gpu is not None:
                        gpu = fixed_gpu
                        if gpu == "cuda":
                            if detector_cuda is None:
                                detector_cuda = MotionDetector(self.config, decode_gpu="cuda")
                            detector = detector_cuda
                        else:
                            detector = default_detector
                        post_fn = self._execute_analysis_task(
                            task, filepath, file_start_offset, detector, yolo_verifier, gpu, perf, t0, worker_id=worker_id, **extra_kw
                        )
                    elif is_cuda_heavy_worker:
                        if detector_cuda is None:
                            detector_cuda = MotionDetector(self.config, decode_gpu="cuda")
                        detector = detector_cuda
                        file_dur = float(task.get("file_duration") or 0.0)
                        task_is_stolen = bool(task.get("is_stolen", False)) or is_stolen
                        detector.device_lease = lambda dur=file_dur, stolen=task_is_stolen: self.work_stealing.lease_device(
                            self.analysis_queue.qsize(),
                            prescreen_idle=False,
                            file_duration=dur,
                            is_stolen=stolen,
                        )
                        post_fn = self._execute_analysis_task(
                            task, filepath, file_start_offset, detector, yolo_verifier, "cuda", perf, t0, worker_id=worker_id, **extra_kw
                        )
                    else:
                        detector = default_detector
                        file_dur = float(task.get("file_duration") or 0.0)
                        if file_dur <= 0.0:
                            row = self.db.get_file_task(filepath)
                            if row and row.get("file_duration"):
                                file_dur = float(row["file_duration"])
                        detector.device_lease = lambda dur=file_dur: self.work_stealing.lease_device(
                            self.analysis_queue.qsize(),
                            prescreen_idle=(self.prescreen_queue.empty() and self.analysis_queue.qsize() <= self.work_stealing.watermark_low),
                            file_duration=dur,
                        )
                        post_fn = self._execute_analysis_task(
                            task, filepath, file_start_offset, detector, yolo_verifier, "adaptive", perf, t0, worker_id=worker_id, **extra_kw
                        )

                    if callable(post_fn):
                        if pending_post_future is not None:
                            pending_post_future.result()
                            pending_post_future = None

                        def _run_post(fn=post_fn, fp=filepath, t_start=t0, q_wait=analysis_queue_wait, g_used=gpu if 'gpu' in locals() else "adaptive"):
                            try:
                                fn()
                            except Exception as exc:
                                if not (self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted()):
                                    if self._is_source_dir_offline([fp]):
                                        self._stop_for_storage_unavailable(f"analysis {Path(fp).name}", exc)
                                        return
                                    logger.exception("Streaming: async post-analysis failed for %s", fp)
                                    self.db.set_analysis_result(fp, "FAILED", "", error_msg=str(exc))
                                    self._add_error(f"analysis failed: {Path(fp).name}")
                                    t_ana_err = time.monotonic()
                                    perf.add(
                                        PerfRecord(
                                            stage="analysis",
                                            file=Path(fp).name,
                                            gpu=g_used,
                                            duration=round(t_ana_err - t_start, 3),
                                            extra={
                                                "status": "ERROR",
                                                "analysis_queue_wait_s": round(q_wait, 3),
                                            },
                                            start_time=round(t_start, 3),
                                            end_time=round(t_ana_err, 3),
                                            worker=worker_id or "ana",
                                        )
                                    )
                                    self.render_batch_queue.put({"filepath": fp, "status": "FAILED"})
                            finally:
                                self.analysis_queue.task_done()

                        pending_post_future = post_executor.submit(copy_context().run, _run_post)
                        task_done_deferred = True

                except MediaTerminationUnconfirmed as exc:
                    self._stop_for_unconfirmed_media(f"analysis {Path(filepath).name}", exc)
                    try:
                        self.db.set_analysis_result(filepath, "FAILED", "", error_msg=str(exc))
                    except Exception:
                        logger.exception("Cannot persist FAILED after unconfirmed analysis stop: %s", filepath)
                        self._add_error(f"analysis FAILED persistence failed: {Path(filepath).name}")
                    break
                except InputStorageUnavailable as exc:
                    self._stop_for_storage_unavailable(f"analysis {Path(filepath).name}", exc)
                    break
                except Exception as exc:
                    if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                        continue
                    if self._is_source_dir_offline([filepath]):
                        self._stop_for_storage_unavailable(f"analysis {Path(filepath).name}", exc)
                        break
                    logger.exception("Streaming: analysis failed for %s", filepath)
                    self.db.set_analysis_result(filepath, "FAILED", "", error_msg=str(exc))
                    self._add_error(f"analysis failed: {Path(filepath).name}")
                    t_ana_err = time.monotonic()
                    perf.add(
                        PerfRecord(
                            stage="analysis",
                            file=Path(filepath).name,
                            gpu=gpu if 'gpu' in locals() else "adaptive",
                            duration=round(t_ana_err - t0, 3),
                            extra={
                                "status": "ERROR",
                                "analysis_queue_wait_s": round(analysis_queue_wait, 3),
                            },
                            start_time=round(t0, 3),
                            end_time=round(t_ana_err, 3),
                            worker=worker_id or "ana",
                        )
                    )
                    self.render_batch_queue.put({"filepath": filepath, "status": "FAILED"})
                finally:
                    if not task_done_deferred:
                        self.analysis_queue.task_done()
        finally:
            if pending_post_future is not None:
                try:
                    pending_post_future.result()
                except Exception:
                    pass
            post_executor.shutdown(wait=True)

    def _render_manager(self):
        """渲染管理器：In-Order Sliding Window 保序消费完成消息，构建时间严格单调的批次并启动渲染 Worker。"""
        out_cfg = self.config.get("output", {})
        fps = out_cfg.get("fps", 20)
        from src.core.utils import parse_res

        width, height = parse_res(out_cfg.get("resolution", "1920x1080"))
        seg_cfg = self.config.get("segment", {})
        audio_cfg = out_cfg.get("audio", {})
        render_cfg = self.config.get("render", {})
        render_hw_policy = self.config.get("pipeline", {}).get("render_gpu_policy", "heterogeneous")
        max_qsv_dynamic_s = float(render_cfg.get("max_qsv_dynamic_duration_s", 420.0))
        nv_budget = int(self.config.get("hardware", {}).get("max_nv_concurrency", 2))
        nv_tail_budget = int(render_cfg.get("nv_tail_concurrency", nv_budget))

        # ---- In-Order Sliding Window：按物理录制时间严格保序的批次调度 ----
        # 乱序到达的完成消息（STATIC/ANALYZED/FAILED）经滑动窗口重排：
        # 仅当队首（最早录制）文件就绪时窗口才向前推进，确保跨批次时间轴 100% 单调。
        all_rows = self.db.get_all_file_tasks_for_date(self.date, self.cam_index)
        ordered_files: list[str] = [
            r["filepath"]
            for r in sorted(all_rows, key=lambda r: r.get("file_start_time") or "")
        ]
        ordered_set = set(ordered_files)
        order_index = {filepath: idx for idx, filepath in enumerate(ordered_files)}
        ready_heavy: dict[str, bool] = {}
        dynamic_duration_by_file: dict[str, float] = {}
        active_states = {"DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE", "NIGHT_STATIONARY", "MICRO_MOTION"}

        def _is_seg_dynamic(s) -> bool:
            if isinstance(s, dict):
                return s.get("state") in active_states
            return getattr(s, "state", "") in active_states or bool(getattr(s, "is_dynamic", False))

        for r in all_rows:
            fp = r["filepath"]
            segs = r.get("segments")
            if segs:
                try:
                    dyn_dur = sum(
                        float(s.get("end_time", 0.0) - s.get("start_time", 0.0))
                        if isinstance(s, dict)
                        else float(s.end_time - s.start_time)
                        for s in segs
                        if _is_seg_dynamic(s)
                    )
                    dynamic_duration_by_file[fp] = dyn_dur
                    ready_heavy[fp] = (dyn_dur > 0)
                except Exception:
                    dynamic_duration_by_file[fp] = 0.0
            elif hasattr(self.db, "get_dynamic_duration_for_file") and r.get("prescreen_status") != "STATIC":
                dyn_dur = float(self.db.get_dynamic_duration_for_file(fp))
                dynamic_duration_by_file[fp] = dyn_dur
                ready_heavy[fp] = (dyn_dur > 0)
            else:
                dynamic_duration_by_file[fp] = 0.0
                if r.get("prescreen_status") == "STATIC":
                    ready_heavy[fp] = False

        immediate_file_batches = self.batch_max_files == 1
        dispatched_files: set[str] = set()
        ready_status: dict[str, str] = {}
        render_ready_at_by_file: dict[str, float] = {}
        batch_ready_at: dict[int, float] = {}
        head = 0
        pending_files: list[str] = []
        batch_idx = 0
        dispatched_batch_ids: list[int] = []
        if not immediate_file_batches:
            time.sleep(max(0, self.render_delay))
        heavy_queue = DualEndedBatchQueue()
        light_queue = queue.Queue()
        nv_retry_queue = queue.Queue()
        batch_retries: dict[int, int] = {}
        all_dispatched_event = threading.Event()
        render_start_t: list[float] = []
        # 已进入终态（失败已上报 / 空时间轴跳过）的批次集合，用于收尾对账去重
        terminal_batch_ids: set[int] = set()
        qsv_workers_alive = 0

        def _is_heavy_batch(files_to_batch: list[str]) -> bool:
            """检查批次中是否包含需复杂处理的 DYNAMIC 动作片段。"""
            if self.abort_event.is_set() or getattr(self.db, "is_closed", False):
                return False
            if files_to_batch and all(fp in ready_heavy for fp in files_to_batch):
                return any(ready_heavy[fp] for fp in files_to_batch)
            try:
                active_states = {"DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE", "NIGHT_STATIONARY", "MICRO_MOTION"}
                for fp in files_to_batch:
                    if fp in ready_heavy:
                        if ready_heavy[fp]:
                            return True
                        continue
                    r = self.db.get_file_task_summary(fp)
                    if not r:
                        continue
                    if r.get("prescreen_status") == "SUSPICIOUS":
                        active = 0.0
                        if hasattr(self.db, "get_dynamic_duration_for_file"):
                            active = float(self.db.get_dynamic_duration_for_file(fp))
                        elif r.get("segments"):
                            active = sum(
                                float(s.get("end_time", 0.0) - s.get("start_time", 0.0))
                                if isinstance(s, dict)
                                else float(s.end_time - s.start_time)
                                for s in r["segments"]
                                if _is_seg_dynamic(s)
                            )
                        dynamic_duration_by_file[fp] = active
                        ready_heavy[fp] = (active > 0)
                        if active > 0:
                            return True
                    ready_heavy[fp] = False
            except Exception:
                pass
            return False

        nv_batch_durations: list[float] = []
        nv_duration_lock = threading.Lock()

        def _render_worker(gpu: str, worker_id: str = "", tail_cpu_decode: bool = False):
            nonlocal qsv_workers_alive
            handoff_to_nv = False
            worker_name = worker_id or gpu
            t_worker_start = time.monotonic()
            busy_time = 0.0
            batches_count = 0
            dynamic_sec_total = 0.0
            qsv_light_count = 0
            qsv_stolen_count = 0
            qsv_steal_rejected = 0
            qsv_upstream_yields = 0

            try:
                while not self.abort_event.is_set() and not FFmpegProcessRegistry.is_interrupted():
                    item = None
                    is_stolen = False
                    is_light = False
                    qsv_admission_limit = max_qsv_dynamic_s
                    if gpu == "nv":
                        # NVENC (3060Ti) 专职优先消费 heavy 动态任务 (LPT 最长优先)，空闲时协助消费 light 任务
                        try:
                            item = nv_retry_queue.get_nowait()
                        except queue.Empty:
                            item = heavy_queue.pop_heaviest(timeout=0.5)
                        if item is None:
                            try:
                                item = light_queue.get(timeout=0.5)
                                if item is not None:
                                    is_light = True
                            except queue.Empty:
                                with self.batch_lock:
                                    active_qsv = self._active_qsv_renders
                                    qsv_alive = qsv_workers_alive
                                if (all_dispatched_event.is_set() and heavy_queue.empty()
                                        and light_queue.empty() and nv_retry_queue.empty()
                                        and active_qsv == 0 and qsv_alive == 0):
                                    break
                                continue
                    else:
                        # QSV Worker (Intel UHD 770):
                        # 1. 专职优先消费 light_queue (纯静态抽帧秒级批次)
                        try:
                            item = light_queue.get_nowait()
                            if item is not None:
                                is_light = True
                        except queue.Empty:
                            # 2. 当 light_queue 为空时，在异构模式下逆向窃取消费中轻度动态批次 (SPT 最短优先)
                            try:
                                if render_hw_policy == "heterogeneous":
                                    with self.batch_lock:
                                        active_nv = self._active_nv_renders
                                    q_len = heavy_queue.qsize()
                                    # 收尾硬屏障 (Tail Guard):
                                    # 当所有素材派发完毕且队列已空，或剩余批次不足以填满 NVENC 编队时，QSV 立即停止偷取并退出，
                                    # 避免 QSV 在尾盘偷取最后任务，拖累端到端完工时间 (Makespan)
                                    if all_dispatched_event.is_set() and (heavy_queue.empty() or q_len <= active_nv):
                                        break

                                    # 实时 Makespan 竞价模型 (Cost-Based ETA Bidding):
                                    # 双路 NVENC 平均耗时 ~42s/批次，预估 NVENC 编队清空当前所有排队批次所需的剩余完工时间 (T_nv_eta)：
                                    with nv_duration_lock:
                                        if nv_batch_durations:
                                            import statistics
                                            avg_nv_time = statistics.median(nv_batch_durations[-8:])
                                            avg_nv_time = min(90.0, max(20.0, float(avg_nv_time)))
                                        else:
                                            avg_nv_time = 42.0
                                    t_nv_eta = max(30.0, ((q_len + active_nv) / 2.0) * avg_nv_time)
                                    # 上游仍有分析或预筛待处理时，QSV 仅消化纯静态轻量批次，把核显全额让渡给上游分析编队；
                                    # 仅在上游清零后才允许窃取动态批次（或进一步交接给 nv_tail_0）。
                                    upstream_busy = (
                                        (not self.upstream_drained.is_set())
                                        and (
                                            (not self.analysis_queue.empty())
                                            or getattr(self.prescreen_queue, "unfinished_tasks", 0) > 0
                                        )
                                    )
                                    if all_dispatched_event.is_set():
                                        # QSV 实测平均压制倍速 ~2x，允许窃取估算耗时可落在 t_nv_eta 内的中轻型尾批，
                                        # 同时避免卡在 "QSV 拒收但低于 180s 导致 nv_tail_0 也不接力" 的调度死区。
                                        effective_max_dyn = min(max_qsv_dynamic_s, 180.0, t_nv_eta * 1.4)
                                    else:
                                        effective_max_dyn = max_qsv_dynamic_s
                                    allowed_dyn_s = effective_max_dyn if t_nv_eta >= 35.0 else min(effective_max_dyn, max(60.0, t_nv_eta * 1.5))
                                    if upstream_busy:
                                        item = None
                                    else:
                                        item = heavy_queue.steal_lightest(
                                            max_dynamic_s=allowed_dyn_s,
                                        )
                                    if item is None:
                                        if all_dispatched_event.is_set() and (heavy_queue.empty() or q_len <= active_nv):
                                            break
                                        lightest = heavy_queue.lightest_dynamic_duration()
                                        # No more sources can arrive. If even the lightest batch
                                        # exceeds QSV's effective tail admission, QSV cannot take any
                                        # remaining work. Reuse this thread on NV within its existing
                                        # configured lease budget; keep every remaining encoder NV.
                                        if (all_dispatched_event.is_set()
                                                and self.upstream_drained.is_set()
                                                and render_gpus.count("nv") == 2
                                                and nv_tail_budget >= 3
                                                and nv_budget >= 2
                                                and lightest is not None
                                                and lightest > effective_max_dyn
                                                and self.work_stealing.has_vram_headroom()):
                                            handoff_to_nv = True
                                            break
                                        if not heavy_queue.empty():
                                            if upstream_busy:
                                                qsv_upstream_yields += 1
                                            else:
                                                qsv_steal_rejected += 1
                                        time.sleep(0.5)
                                        continue
                                    else:
                                        is_stolen = True
                                        qsv_admission_limit = allowed_dyn_s
                                else:
                                    item = heavy_queue.pop_heaviest(timeout=0.5)
                            except Exception:
                                pass

                    if item is None:
                        with self.batch_lock:
                            active_qsv = self._active_qsv_renders
                            qsv_alive = qsv_workers_alive
                        if (all_dispatched_event.is_set() and heavy_queue.empty()
                                and light_queue.empty() and nv_retry_queue.empty()
                                and active_qsv == 0 and (gpu != "nv" or qsv_alive == 0)):
                            break
                        continue

                    if is_light:
                        qsv_light_count += 1
                    elif is_stolen:
                        qsv_stolen_count += 1

                    if isinstance(item, RenderBatchItem):
                        b_idx, files_to_batch = item.b_idx, item.files
                    else:
                        b_idx, files_to_batch = item
                    if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                        break
                    if not render_start_t:
                        render_start_t.append(time.monotonic())
                    t_r0 = time.monotonic()
                    render_queue_wait = max(0.0, t_r0 - batch_ready_at.get(b_idx, t_r0))
                    batch_token = current_batch.set((b_idx, worker_name))
                    if self.observation:
                        self.observation.batch(b_idx, "running", worker=worker_name)
                    logger.info("render batch %d started on %s (%d files, worker=%s)", b_idx, gpu, len(files_to_batch), worker_name,
                                extra={"event": "batch_started", "queue_wait_s": round(render_queue_wait, 3)})
                    if gpu == "nv":
                        with self.batch_lock:
                            self._active_nv_renders += 1
                        self.work_stealing.register_render_start()
                    elif gpu == "qsv":
                        with self.batch_lock:
                            self._active_qsv_renders += 1
                    try:
                        all_rows = []
                        try:
                            if self.abort_event.is_set() or getattr(self.db, "is_closed", False):
                                break
                            all_rows = self.db.get_all_file_tasks_for_date(
                                self.date, self.cam_index, filepaths=files_to_batch,
                            )
                        except Exception as e:
                            if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                                break
                            logger.warning("DB query failed for batch %d: %s", b_idx, e)
                        from src.stages.timeline import build_timeline_from_rows
                        # Streaming batches must use a stable per-file timeline;
                        # cross-file presence depends on analysis results that may
                        # arrive after this batch.
                        stream_config = dict(self.config)
                        stream_config["presence"] = dict(self.config.get("presence", {}))
                        stream_config["presence"]["enabled"] = False
                        batch_segs = build_timeline_from_rows(
                            all_rows, self.date, target_files=files_to_batch, config=stream_config,
                            resolve_presence=False,
                        )

                        if not batch_segs:
                            from src.stages.timeline import TimelineSegment
                            rows_by_path = {r["filepath"]: r for r in all_rows}

                            # 辅助函数：精准判定行记录是否包含活跃动态切片
                            active_states = {"DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE", "NIGHT_STATIONARY", "MICRO_MOTION"}
                            def _row_has_dynamic(r: dict, fp: str) -> bool:
                                segs = r.get("segments")
                                if segs is not None:
                                    return any(
                                        (s.get("state") in active_states) if isinstance(s, dict)
                                        else (getattr(s, "state", "") in active_states or getattr(s, "is_dynamic", False))
                                        for s in segs
                                    )
                                if hasattr(self.db, "has_dynamic_segments_for_file"):
                                    return self.db.has_dynamic_segments_for_file(fp)
                                return False

                            # 检查批次文件是否已被宏观折叠（Macro-collapse）或分析确认无需画面输出
                            is_all_collapsed_or_skipped = True
                            for filepath in files_to_batch:
                                row = rows_by_path.get(filepath)
                                if not row:
                                    is_all_collapsed_or_skipped = False
                                    break
                                ana_status = row.get("analysis_status")
                                pre_status = row.get("prescreen_status")
                                if pre_status == "STATIC":
                                    continue
                                if ana_status in ("ANALYZED", "COMPLETED"):
                                    if not _row_has_dynamic(row, filepath):
                                        continue
                                is_all_collapsed_or_skipped = False
                                break

                            if is_all_collapsed_or_skipped:
                                logger.info(
                                    "render batch %d skipped: static footage collapsed by timeline (%d file(s))",
                                    b_idx, len(files_to_batch),
                                )
                                with self.batch_lock:
                                    terminal_batch_ids.add(b_idx)
                                if self.observation:
                                    self.observation.batch(b_idx, "skipped")
                                if self.dashboard is not None:
                                    self.dashboard.render_batch_finished(len(files_to_batch))
                                continue

                            # 若确实存在未提交分段的异常竞争，采用保守兜底，但绝对禁止盲目将静态段提升为 DYNAMIC
                            for fallback_idx, filepath in enumerate(files_to_batch):
                                row = rows_by_path.get(filepath)
                                if not row:
                                    continue
                                duration = float(row.get("file_duration") or 0.0)
                                if duration <= 0:
                                    continue
                                has_dynamic = _row_has_dynamic(row, filepath)
                                state = "DYNAMIC" if has_dynamic else "STATIC"
                                batch_segs.append(TimelineSegment(
                                    filepath=filepath, input_index=fallback_idx,
                                    start_in_file=0.0, end_in_file=duration,
                                    state=state, duration=duration,
                                ))
                            if batch_segs:
                                logger.warning(
                                    "render batch %d had no committed segments; conservative fallback for %d file(s)",
                                    b_idx, len(batch_segs),
                                )
                            else:
                                logger.warning("render batch %d has no usable timeline or metadata", b_idx)
                                with self.batch_lock:
                                    terminal_batch_ids.add(b_idx)
                                if self.observation:
                                    self.observation.batch(b_idx, "skipped")
                                if self.dashboard is not None:
                                    self.dashboard.render_batch_finished(len(files_to_batch))
                                continue

                        # 异构长尾保护：QSV 仅消费中轻量动态任务，若真实切片包含超过门限的 DYNAMIC 动作或超长 4K 活动流则交还 NVENC
                        if gpu == "qsv" and render_hw_policy == "heterogeneous":
                            src_dur = sum(float(r.get("file_duration") or 0.0) for r in all_rows)
                            _, dynamic_dur = _compute_qsv_queue_weight(
                                batch_segs,
                                file_dur_s=src_dur,
                                render_cfg=render_cfg,
                                single_file_batch=(len(files_to_batch) <= 1),
                            )
                            if is_stolen and dynamic_dur > qsv_admission_limit:
                                logger.info(
                                    "render batch %d on qsv contains %.1fs dynamic motion (exceeds %.1fs), offloading to nv heavy queue",
                                    b_idx, dynamic_dur, qsv_admission_limit,
                                )
                                if self.observation:
                                    self.observation.batch(b_idx, "queued")
                                heavy_queue.put(b_idx, files_to_batch, dynamic_dur)
                                continue

                        # Once a consumer owns the file, queued prefetch must not
                        # recreate its cache after the renderer has released it.
                        with self._prefetched_lock:
                            self._prefetched_files.difference_update(files_to_batch)
                        render_kwargs = {"config": self.config}
                        if tail_cpu_decode:
                            render_kwargs["cpu_decode"] = True
                        res_path = build_batch_render(
                            batch_segs, b_idx, gpu, fps, width, height,
                            seg_cfg, out_cfg, audio_cfg, self.date, self.cam_index,
                            all_rows, **render_kwargs,
                        )
                        r_dur = round(time.monotonic() - t_r0, 3)
                        busy_time += r_dur
                        batches_count += 1
                        batch_dyn = sum(s.duration for s in batch_segs if s.state in ("DYNAMIC", "DYNAMIC_AUDIO"))
                        dynamic_sec_total += batch_dyn
                        backend_label = "cpu_nvenc" if tail_cpu_decode and gpu == "nv" else gpu
                        logger.info("render batch %d finished on %s in %.3fs (output=%s)", b_idx, backend_label, r_dur, bool(res_path),
                                    extra={"event": "batch_attempt_finished", "duration_s": r_dur,
                                           "status": "ready" if res_path else "no_output"})
                        if res_path:
                            if gpu == "nv" and not is_light and r_dur >= 5.0:
                                with nv_duration_lock:
                                    nv_batch_durations.append(r_dur)
                            with self.batch_lock:
                                self.batch_paths.append((b_idx, Path(res_path)))
                            get_perf().add(
                                PerfRecord(
                                    stage="render",
                                    file=f"batch_{b_idx}.mp4",
                                    gpu=gpu,
                                    duration=r_dur,
                                    extra={
                                        "batch_files": len(files_to_batch),
                                        "segments": len(batch_segs),
                                        "dynamic_duration_s": round(batch_dyn, 2),
                                        "normal_speed_duration_s": round(batch_dyn, 2),
                                        "active_source_duration_s": round(sum(s.duration for s in batch_segs if s.state != "STATIC"), 2),
                                        "queue_active_source_duration_s": round(sum(dynamic_duration_by_file.get(fp, 0.0) for fp in files_to_batch), 2),
                                        "source_duration_s": round(sum(float(r.get("file_duration") or 0) for r in all_rows), 2),
                                        "is_stolen": is_stolen,
                                        "is_light": is_light,
                                        "cpu_decode": bool(tail_cpu_decode and gpu == "nv"),
                                        "render_queue_wait_s": round(render_queue_wait, 3),
                                    },
                                    start_time=round(t_r0, 3),
                                    end_time=round(time.monotonic(), 3),
                                    worker=worker_name,
                                )
                            )
                            if self.observation:
                                self.observation.batch(b_idx, "ready")
                            if self.dashboard is not None:
                                self.dashboard.render_batch_finished(len(files_to_batch))
                                # 精准渲染阶段自身计时与剩余预估
                                t_start = render_start_t[0] if render_start_t else t_r0
                                el_sec = max(0.1, time.monotonic() - t_start)
                                n_done = self.dashboard.render_done + 1
                                avg_s = el_sec / max(1, n_done)
                                enc_name = "NVENC" if gpu == "nv" else gpu.upper()
                                self.dashboard.update_render(
                                    completed=n_done,
                                    latest_batch=f"批次 {b_idx}（{enc_name} 编码）",
                                    speed_str=f"上批 {r_dur:.1f}s │ 均速 {avg_s:.1f}s/批",
                                )
                                self._sync_queue_levels()
                        else:
                            if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                                break
                            if self._is_source_dir_offline(files_to_batch):
                                self._stop_for_storage_unavailable(f"render batch {b_idx}", InputStorageUnavailable("source directory offline"))
                                break
                            # Return QSV failures to an NV-only queue so retry
                            # scheduling and telemetry remain owned by NV.
                            if gpu == "qsv" and render_hw_policy == "heterogeneous":
                                logger.warning(
                                    "render batch %d returned no output on qsv; queued for nv retry",
                                    b_idx,
                                )
                                if self.observation:
                                    self.observation.batch(b_idx, "retry")
                                nv_retry_queue.put((b_idx, files_to_batch))
                                continue

                            can_retry = False
                            with self.batch_lock:
                                if gpu == "nv" and batch_retries.get(b_idx, 0) < 1:
                                    batch_retries[b_idx] = batch_retries.get(b_idx, 0) + 1
                                    can_retry = True
                            if can_retry:
                                logger.warning(
                                    "render batch %d returned no output on %s; queued for retry (attempt 1/1)",
                                    b_idx, gpu,
                                )
                                if self.observation:
                                    self.observation.batch(b_idx, "retry")
                                nv_retry_queue.put((b_idx, files_to_batch))
                                continue

                            self._add_error(f"render batch {b_idx} returned no output")
                            with self.batch_lock:
                                terminal_batch_ids.add(b_idx)
                            if self.observation:
                                self.observation.batch(b_idx, "failed")
                            if self.dashboard is not None:
                                self.dashboard.render_batch_finished(len(files_to_batch))
                    except MediaTerminationUnconfirmed as exc:
                        self._stop_for_unconfirmed_media(f"render batch {b_idx}", exc)
                        logger.error("Render cleanup unconfirmed; stopping without retries and preserving completed batches: %s", exc)
                        break
                    except InputStorageUnavailable as exc:
                        self._stop_for_storage_unavailable(f"render batch {b_idx}", exc)
                        break
                    except RenderInterrupted as exc:
                        with self.batch_lock:
                            self._render_interruption_reason = str(exc)
                        self.abort_event.set()
                        self.stop_event.set()
                        FFmpegProcessRegistry.mark_interrupted()
                        FFmpegProcessRegistry.kill_all()
                        logger.warning("Rendering interrupted; preserving completed batches: %s", exc)
                        break
                    except Exception as exc:
                        if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                            break
                        if self._is_source_dir_offline(files_to_batch):
                            self._stop_for_storage_unavailable(f"render batch {b_idx}", exc)
                            break
                        logger.exception("Streaming: render batch %d failed on %s", b_idx, gpu)
                        if gpu == "qsv" and render_hw_policy == "heterogeneous":
                            logger.warning(
                                "render batch %d crashed on qsv; queued for nv retry",
                                b_idx,
                            )
                            if self.observation:
                                self.observation.batch(b_idx, "retry")
                            nv_retry_queue.put((b_idx, files_to_batch))
                            continue

                        can_retry = False
                        with self.batch_lock:
                            if gpu == "nv" and batch_retries.get(b_idx, 0) < 1:
                                batch_retries[b_idx] = batch_retries.get(b_idx, 0) + 1
                                can_retry = True
                        if can_retry:
                            logger.warning(
                                "render batch %d crashed on %s; queued for retry (attempt 1/1)",
                                b_idx, gpu,
                            )
                            if self.observation:
                                self.observation.batch(b_idx, "retry")
                            nv_retry_queue.put((b_idx, files_to_batch))
                            continue

                        self._add_error(f"render batch {b_idx} failed on {gpu}")
                        with self.batch_lock:
                            terminal_batch_ids.add(b_idx)
                        if self.observation:
                            self.observation.batch(b_idx, "failed")
                        if self.dashboard is not None:
                            self.dashboard.render_batch_finished(len(files_to_batch))
                    finally:
                        current_batch.reset(batch_token)
                        if gpu == "nv":
                            with self.batch_lock:
                                self._active_nv_renders = max(0, self._active_nv_renders - 1)
                            self.work_stealing.register_render_end()
                        elif gpu == "qsv":
                            with self.batch_lock:
                                self._active_qsv_renders = max(0, self._active_qsv_renders - 1)
            finally:
                if gpu == "qsv":
                    with self.batch_lock:
                        qsv_workers_alive = max(0, qsv_workers_alive - 1)
                t_worker_total = max(0.001, time.monotonic() - t_worker_start)
                idle_time = max(0.0, t_worker_total - busy_time)
                with self.batch_lock:
                    self.render_worker_stats[worker_name] = {
                        "gpu": gpu,
                        "worker_id": worker_name,
                        "total_time_s": round(t_worker_total, 2),
                        "busy_time_s": round(busy_time, 2),
                        "idle_time_s": round(idle_time, 2),
                        "utilization_pct": round(busy_time / t_worker_total * 100, 1),
                        "batches_rendered": batches_count,
                        "dynamic_sec_rendered": round(dynamic_sec_total, 1),
                        "qsv_light_batches": qsv_light_count,
                        "qsv_stolen_batches": qsv_stolen_count,
                        "qsv_steal_rejected": qsv_steal_rejected,
                        "qsv_upstream_yields": qsv_upstream_yields,
                        "handoff_to_nv": handoff_to_nv,
                    }
            if handoff_to_nv and not self.abort_event.is_set():
                from src.hardware.scheduler import nv_tail_capacity
                # The shared NV budget is unchanged until all recognition work
                # has completed. Revocation must not free an unconfirmed reader.
                capacity = nv_tail_capacity() if nv_budget == 2 else nullcontext(True)
                with capacity as admitted:
                    if admitted and not self.abort_event.is_set():
                        tail_cpu_decode = (
                            render_hw_policy == "heterogeneous"
                            and bool(render_cfg.get("nv_tail_use_cpu_decode", True))
                        )
                        logger.info(
                            "QSV worker %s finished its eligible queue; reusing thread as nv_tail_0 (upstream drained, cpu_decode=%s)",
                            worker_name, tail_cpu_decode,
                        )
                        _render_worker("nv", "nv_tail_0", tail_cpu_decode=tail_cpu_decode)
                    else:
                        with self.batch_lock:
                            self.render_worker_stats[worker_name]["handoff_to_nv"] = False

        # 启动渲染编队：异构模式下采用 2 NVENC + 1 QSV 三路并行
        render_threads = []
        if uses_software_pipeline(self.config):
            render_gpus = ["cpu"] * self.render_workers
        elif render_hw_policy == "nv_only":
            render_gpus = ["nv"] * self.render_workers
        elif render_hw_policy == "qsv_only":
            render_gpus = ["qsv"] * self.render_workers
        else:
            nv_count = min(nv_budget, max(1, self.render_workers - 1))
            logger.info(
                "Streaming: render_gpu_policy='heterogeneous' active with %d workers: deploying "
                "%d NVENC + 1 QSV with in-band parameter set injection (-bsf:v dump_extra).",
                self.render_workers,
                nv_count,
            )
            render_gpus = ["nv"] if self.render_workers == 1 else ["nv"] * nv_count + ["qsv"]
        qsv_workers_alive = render_gpus.count("qsv")
        worker_counts: dict[str, int] = {}
        for gpu in render_gpus:
            idx = worker_counts.get(gpu, 0)
            worker_counts[gpu] = idx + 1
            w_id = f"{gpu}_{idx}"
            t = threading.Thread(target=self._guard_worker, args=(_render_worker, gpu, w_id), daemon=True)
            t.start()
            render_threads.append(t)

        prefetch_thread = None
        local_staging_enabled = bool(render_cfg.get("local_staging_enabled", render_cfg.get("local_staging", True)))
        if local_staging_enabled:
            prefetch_thread = threading.Thread(
                target=self._guard_worker, args=(self._lookahead_prefetch_worker,), daemon=True
            )
            prefetch_thread.start()

        def _estimate_dynamic_duration(files: list[str]) -> float:
            """Active source seconds (including presence/night), not 1x display time."""
            return sum(dynamic_duration_by_file.get(fp, 0.0) for fp in files)

        def _enqueue_batch(b_idx: int, files: list[str]):
            batch_ready_at[b_idx] = max(
                (render_ready_at_by_file.get(fp, time.monotonic()) for fp in files),
                default=time.monotonic(),
            )
            if local_staging_enabled and files:
                for fp in files:
                    with self._prefetched_lock:
                        if fp not in self._prefetched_files:
                            self._prefetched_files.add(fp)
                            self._prefetch_queue.put(fp)

            if self.observation:
                self.observation.batch(b_idx, "queued")
            logger.info("Batch %d queued", b_idx, extra={"event": "batch_queued", "batch": b_idx})
            dyn_dur = _estimate_dynamic_duration(files)
            if dyn_dur > 0 or _is_heavy_batch(files):
                if dyn_dur <= 0:
                    dyn_dur = 60.0
                heavy_queue.put(b_idx, files, dyn_dur)
            else:
                light_queue.put((b_idx, files))
            dispatched_batch_ids.append(b_idx)
            if self.dashboard is not None:
                self.dashboard.render_batch_dispatched(len(files))
                self.dashboard.update_render(
                    completed=self.dashboard.render_done,
                    total=self.dashboard.render_total + 1,
                )
            self._sync_queue_levels()

        while not self.abort_event.is_set() and (not self.stop_event.is_set() or not self.render_batch_queue.empty()):
            try:
                msg = self.render_batch_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            filepath = msg.get("filepath")
            if filepath is not None:
                render_ready_at_by_file[filepath] = float(
                    msg.get("_render_ready_at") or time.monotonic()
                )
                if "dynamic_duration" in msg:
                    dynamic_duration_by_file[filepath] = float(msg["dynamic_duration"])
                if "is_heavy" in msg:
                    ready_heavy[filepath] = bool(msg["is_heavy"])
                elif filepath in dynamic_duration_by_file:
                    ready_heavy[filepath] = (dynamic_duration_by_file[filepath] > 0)
                status = msg.get("status", "FAILED")
                if immediate_file_batches and filepath in order_index:
                    # Each output now owns exactly one physical file.  Its
                    # immutable chronological ordinal is the batch id, so a
                    # later-ready file can render immediately while final
                    # concat remains deterministic after sorting by id.
                    if status not in ("FAILED", "SKIPPED_EMPTY", "SKIPPED_EXCLUDED", "SKIPPED_CORRUPT") and filepath not in dispatched_files:
                        _enqueue_batch(order_index[filepath], [filepath])
                        dispatched_files.add(filepath)
                elif filepath in ordered_set:
                    ready_status[filepath] = status
                elif status not in ("FAILED", "SKIPPED_EMPTY", "SKIPPED_EXCLUDED", "SKIPPED_CORRUPT"):
                    pending_files.append(filepath)
                while head < len(ordered_files) and ordered_files[head] in ready_status:
                    fp_head = ordered_files[head]
                    if ready_status[fp_head] not in ("FAILED", "SKIPPED_EMPTY", "SKIPPED_EXCLUDED", "SKIPPED_CORRUPT"):
                        pending_files.append(fp_head)
                    head += 1
                while len(pending_files) >= self.batch_max_files:
                    _enqueue_batch(batch_idx, list(pending_files[:self.batch_max_files]))
                    pending_files = pending_files[self.batch_max_files:]
                    batch_idx += 1
            self.render_batch_queue.task_done()

        try:
            if pending_files:
                _enqueue_batch(batch_idx, list(pending_files))
                pending_files = []

            all_dispatched_event.set()
            heavy_queue.notify_all()

            if prefetch_thread is not None:
                self._prefetch_queue.put(None)
                prefetch_thread.join(timeout=5.0)

            for _ in render_threads:
                light_queue.put(None)
            for t in render_threads:
                t.join()

            # 渲染对账：分发批次必须全部到达终态（产出或已上报失败），
            # 任何无记录的缺失即为静默丢批，立即上报
            with self.batch_lock:
                produced_ids = {bi for bi, _ in self.batch_paths}
                accounted = produced_ids | terminal_batch_ids
            missing = [bi for bi in dispatched_batch_ids if bi not in accounted]
            if self.abort_event.is_set():
                logger.info("Rendering cancelled: %d completed batches retained for resume", len(produced_ids))
            elif missing:
                self._add_error(f"render batches silently dropped: {missing}")
                logger.error("render reconciliation failed: dispatched=%s produced=%s missing=%s",
                             dispatched_batch_ids, sorted(produced_ids), missing)
        finally:
            self.render_finished_event.set()


    def run(self):
        raw_tasks = self.db.get_all_file_tasks_for_date(self.date, self.cam_index)
        all_tasks = [PipelineTask.from_dict(t) for t in raw_tasks]
        for i, t in enumerate(all_tasks):
            if i > 0:
                self._prev_task_map[t["filepath"]] = all_tasks[i - 1]["filepath"]
            if t.get("prescreen_status") and t["prescreen_status"] != "PENDING":
                self._prescreen_results[t["filepath"]] = t["prescreen_status"]

        pending_prescreen = [t for t in all_tasks if t["prescreen_status"] == "PENDING"]
        pending_analysis = [t for t in all_tasks if t["prescreen_status"] == "SUSPICIOUS" and t["analysis_status"] == "PENDING"]

        # Rich 实时仪表盘启动 (支持断点续跑的进度种子回填)
        self.dashboard = PipelineDashboard(
            date=self.date,
            cam_index=self.cam_index,
            total_prescreen=max(1, len(all_tasks)),
            render_enabled=self.render_enabled,
            enabled=self.dashboard_enabled,
            cam_name=self.cam_name,
        )
        if self.observation is not None:
            self.observation.dashboard = self.dashboard
        self.dashboard.start()
        if self.dashboard.enabled:
            register_dashboard(self.dashboard)
        self._prescreen_t0 = time.monotonic()
        self._analysis_t0 = time.monotonic()

        prescreen_done_pre = len(all_tasks) - len(pending_prescreen)
        analysis_done_pre = len(
            [t for t in all_tasks if t["prescreen_status"] == "SUSPICIOUS" and t["analysis_status"] != "PENDING"]
        )
        self.dashboard.update_prescreen(completed=prescreen_done_pre, total=len(all_tasks))
        self.dashboard.analysis_cached = sum(
            t["prescreen_status"] == "SUSPICIOUS" and t["analysis_status"] in ("ANALYZED", "COMPLETED")
            for t in all_tasks
        )
        self.dashboard.update_analysis(completed=analysis_done_pre, total=len(pending_analysis) + analysis_done_pre)
        self._sync_queue_levels()

        for task in all_tasks:
            if task["prescreen_status"] == "PENDING":
                self.prescreen_queue.put(task)
            elif task["prescreen_status"] == "STATIC":
                self.render_batch_queue.put({"filepath": task["filepath"], "status": "STATIC"})
            elif task["prescreen_status"] == "SUSPICIOUS":
                if task["analysis_status"] == "PENDING":
                    self._queue_analysis_task(task)
                else:
                    self.render_batch_queue.put({"filepath": task["filepath"], "status": task["analysis_status"]})
            elif task["prescreen_status"] != "PENDING":
                status = "SKIPPED_EMPTY" if task["prescreen_status"] == "SKIPPED_EMPTY" else "FAILED"
                self.render_batch_queue.put({"filepath": task["filepath"], "status": status})

        # 若无需执行预筛（例如断点续跑），或预热队列已积压足量任务，即刻解除预热门控
        if not pending_prescreen or len(pending_analysis) >= 6:
            self._analysis_warmup_event.set()

        threads = []
        requested_prescreen_parallel = int(self.config.get("detection", {}).get("prescreen_parallel", 8))
        # Keep analysis admission possible while prescreen workers are reading
        # from a slow NAS. The same I/O budget is shared by all stages.
        prescreen_parallel = self.prescreen_workers
        if prescreen_parallel < requested_prescreen_parallel:
            logger.warning(
                "prescreen_parallel reduced from %d to %d to reserve I/O slots for analysis",
                requested_prescreen_parallel, prescreen_parallel,
            )
        software_pipeline = uses_software_pipeline(self.config)
        prescreen_gpu_policy = self.config.get("pipeline", {}).get("prescreen_gpu_policy", "qsv_only")
        for i in range(prescreen_parallel):
            gpu = "qsv"
            if software_pipeline:
                gpu = "cpu"
            elif prescreen_gpu_policy == "alternating":
                gpu = "qsv" if i % 2 == 0 else "cuda"
            elif prescreen_gpu_policy == "cuda_only":
                gpu = "cuda"
            w_id = f"pre_{i}"
            t = threading.Thread(target=self._guard_worker, args=(self._prescreen_worker, gpu, w_id), daemon=True)
            t.start()
            threads.append(t)

        analysis_max_workers = self.analysis_workers
        analysis_device = "cpu" if software_pipeline else None
        for i in range(analysis_max_workers):
            w_id = f"ana_{i}"
            t = threading.Thread(target=self._guard_worker, args=(self._analysis_worker, analysis_device, w_id), daemon=True)
            t.start()
            threads.append(t)

        det_cfg = self.config.get("detection", {})
        cpu_analysis_workers = int(det_cfg.get("cpu_analysis_workers", 0))
        cpu_max_dur_s = float(det_cfg.get("cpu_analysis_max_duration_s", 600.0))
        if not software_pipeline and cpu_analysis_workers > 0 and analysis_max_workers > 0:
            for j in range(cpu_analysis_workers):
                w_id = f"ana_cpu_{j}"
                t = threading.Thread(
                    target=self._guard_worker,
                    args=(self._analysis_worker, "cpu", w_id, cpu_max_dur_s),
                    daemon=True,
                )
                t.start()
                threads.append(t)

        if self.render_enabled:
            t_rm = threading.Thread(target=self._guard_worker, args=(self._render_manager,), daemon=True)
            t_rm.start()
            threads.append(t_rm)

        def _wait_queue_interruptible(q: queue.Queue):
            while not self.abort_event.is_set():
                with q.all_tasks_done:
                    if q.unfinished_tasks == 0:
                        break
                    q.all_tasks_done.wait(timeout=0.2)

        try:
            # These are subdivisions of the outer streaming clock, not extra work.
            monitor = get_monitor()
            with monitor.stage(f"upstream_{self.date}_cam{self.cam_index}", include_in_total=False):
                _wait_queue_interruptible(self.prescreen_queue)
                self._analysis_warmup_event.set()
                _wait_queue_interruptible(self.analysis_queue)
            if not self.abort_event.is_set():
                self.upstream_drained.set()
            if self.render_enabled:
                if self.observation and not self.abort_event.is_set():
                    self.observation.set_phase("渲染收尾中")
                with monitor.stage(f"render_tail_{self.date}_cam{self.cam_index}", include_in_total=False):
                    _wait_queue_interruptible(self.render_batch_queue)
                    self.stop_event.set()
                    # Completion requires worker exit and batch reconciliation.
                    while not self.abort_event.is_set():
                        if self.render_finished_event.wait(timeout=0.2):
                            break
            else:
                self.stop_event.set()
        except (KeyboardInterrupt, SystemExit):
            logger.info("StreamingOrchestrator interrupted. Terminating subprocesses...", extra={"event": "cancellation_requested"})
            self.abort_event.set()
            self.stop_event.set()
            self.render_finished_event.set()
            from src.hardware.ffmpeg import FFmpegProcessRegistry
            FFmpegProcessRegistry.mark_interrupted()
            FFmpegProcessRegistry.kill_all()
            if self.dashboard is not None and self.observation is None:
                self.dashboard.stop()
                self.dashboard = None
                unregister_dashboard()
            # 给工作线程 1 秒平稳收尾时间，避免孤儿线程继续向已关闭的 DB 发起请求
            for t in threads:
                t.join(timeout=1.0)
            raise

        self.stop_event.set()
        for t in threads:
            t.join(timeout=5.0)

        if self.dashboard is not None and self.observation is None:
            self.dashboard.stop()
            self.dashboard = None
            unregister_dashboard()

        if self._render_interruption_reason is not None:
            raise KeyboardInterrupt(self._render_interruption_reason)
        if self._media_cleanup_failure is not None:
            raise MediaTerminationUnconfirmed(self._media_cleanup_failure)
        if self._storage_unavailable_failure is not None:
            raise InputStorageUnavailable(self._storage_unavailable_failure)

        with self.batch_lock:
            self.batch_paths.sort(key=lambda x: x[0])
            final_paths = [p for _, p in self.batch_paths]

        with self.error_lock:
            errors = list(self.errors)
        if errors:
            logger.warning("Streaming: some tasks failed during run: %s", "; ".join(errors[:5]))

        return final_paths




def _reconcile_empty_files(db: VlogDatabase, date: str, cam_index: int) -> None:
    """Skip only confirmed zero-byte sources; recheck exclusions on every retry."""
    for row in db.get_all_file_tasks_for_date(date, cam_index):
        previous = row["prescreen_status"]
        if previous not in ("PENDING", "FAILED", "SKIPPED_EMPTY"):
            continue
        filepath = row["filepath"]
        try:
            info = Path(filepath).stat()
            empty = stat.S_ISREG(info.st_mode) and info.st_size == 0
        except OSError as exc:
            # An unavailable NAS path must never be mistaken for an empty file.
            if previous == "SKIPPED_EMPTY":
                db.set_prescreen_result(filepath, "FAILED", error_msg=str(exc))
            continue
        if empty:
            db.set_prescreen_result(filepath, "SKIPPED_EMPTY", error_msg="zero-byte source file")
            logger.warning("跳过空文件（0 字节）: %s；缺失时段 %s 至 %s，将记录在成片元数据中",
                           filepath, row["file_start_time"], row["file_end_time"])
        elif previous == "SKIPPED_EMPTY":
            db.set_prescreen_result(filepath, "PENDING")


def process_date_cam(
    db: VlogDatabase, date: str, cam_index: int,
    skip_render: bool = False, dashboard_enabled: bool = True,
    force_render: bool = False, config: dict | None = None,
) -> bool:
    from src.stages.delivery_recovery import pending_transactions,recover_output,reject_legacy_recovery
    context={'role':'pipeline','date':date,'cam_index':cam_index}
    if not force_render and db.is_render_completed(date,cam_index) and not pending_transactions(OUTPUT_DIR,context=context):
        with db._lock:
            completed = db.conn.execute(
                'SELECT output_file FROM render_tasks WHERE date=? AND cam_index=?',
                (date, cam_index),
            ).fetchone()
        reject_legacy_recovery(OUTPUT_DIR, completed['output_file'] or None)
        return _process_date_cam_owned(db,date,cam_index,skip_render,dashboard_enabled,force_render,config)
    FFmpegProcessRegistry.require_cleanups()
    from src.core.process_lock import day_task_lease
    from src.core.profiles import resolve_camera_config
    from src.stages.scanner import resolve_output_filename
    from src.stages import renderer
    settings=load_config() if config is None else config
    rows=db.get_all_file_tasks_for_date(date,cam_index)
    sample=rows[0]['filepath'] if rows else None
    effective=resolve_camera_config(settings,cam_index=cam_index,filepath=sample) if settings.get('cameras') else settings
    filename=resolve_output_filename(effective.get('output',{}).get('naming','DailyVlog_{date}_{mac}.mp4'),
                                     date,cam_index,sample_filepath=sample,config=effective)
    with day_task_lease(db.db_path,renderer.TEMP_DIR,OUTPUT_DIR/filename,date,cam_index):
        for backup in pending_transactions(OUTPUT_DIR,context=context):
            journal=json.loads((backup/'transaction.json').read_text(encoding='utf-8'))
            if Path(journal['output']).resolve()!=(OUTPUT_DIR/filename).resolve():
                raise RuntimeError(f'Interrupted delivery destination differs; retained for inspection: {backup}')
        recover_output(OUTPUT_DIR/filename,mark_pending=lambda:db.set_render_status(date,cam_index,'PENDING',
                       error_msg='interrupted asset package recovered'))
        return _process_date_cam_owned(db,date,cam_index,skip_render,dashboard_enabled,force_render,settings)


def _process_date_cam_owned(
    db: VlogDatabase, date: str, cam_index: int,
    skip_render: bool = False, dashboard_enabled: bool = True,
    force_render: bool = False, config: dict | None = None,
) -> bool:
    # Completion records survive moving delivered assets to offline storage.
    # Check before fingerprint invalidation so configuration changes cannot
    # silently reopen an already delivered day.
    if not force_render and db.is_render_completed(date, cam_index):
        pending = db.get_pending_file_count_for_date(date, cam_index)
        logger.info("已完成，跳过 %s cam%d（产物可已移至备份介质）；重做请使用 --force-render",
                    date, cam_index)
        if pending:
            logger.warning("%s cam%d 已交付，但有 %d 个待处理素材（可能为新增素材）；"
                           "本次不自动重做，请使用 --force-render 纳入成片", date, cam_index, pending)
        return True
    observation = RunObservation(date, cam_index)
    token = current_run.set(observation)
    status = "failed"
    try:
        logger.info("Starting %s cam%d run_id=%s", date, cam_index, observation.identity["run_id"],
                    extra={"event": "run_started"})
        success = _execute_date_cam(db, date, cam_index, skip_render, dashboard_enabled, config, observation)
        if success:
            status = "analysis_completed" if skip_render else "completed"
        return success
    except (KeyboardInterrupt, SystemExit):
        status = "cancelled"
        raise
    finally:
        try:
            if observation.committed:
                status = "completed"
            totals = observation.totals()
            retained = totals["encoded"] + totals["reused"] + totals["ready"] if status in ("failed", "cancelled") else 0
            labels = {"completed": "已交付", "cancelled": "已取消", "failed": "失败", "analysis_completed": "分析完成"}
            observation.phase = labels[status]
            logger.info("%s cam%d %s · 编码 %d · 复用 %d · 保留 %d 批 · run_id=%s",
                        date, cam_index, labels[status], totals["encoded"], totals["reused"], retained,
                        observation.identity["run_id"], extra={"event": "run_finished", "status": status,
                        "reason": observation.reason or status.upper(), "counts": totals, "retained": retained})
            if observation.dashboard is not None:
                observation.dashboard.stop()
                unregister_dashboard()
        finally:
            current_run.reset(token)


def _execute_date_cam(db, date, cam_index, skip_render, dashboard_enabled, config, observation):
    if config is None:
        config = load_config()
    all_tasks = db.get_all_file_tasks_for_date(date, cam_index)
    sample_filepath = all_tasks[0]["filepath"] if all_tasks else None
    if config.get("cameras"):
        from src.core.profiles import resolve_camera_config
        config = resolve_camera_config(config, cam_index=cam_index, filepath=sample_filepath)
    monitor = get_monitor()
    t_start = time.monotonic()
    # 失败任务自愈：将历史 FAILED 预筛/分析重置为 PENDING（retry_count 上限防护），
    # 使此前因信号量饥饿等原因丢失的文件在本次运行中补齐
    db.invalidate_stale_results(date, cam_index, config)
    _reconcile_empty_files(db, date, cam_index)
    db.reset_failed_tasks(date, cam_index)
    all_tasks = db.get_all_file_tasks_for_date(date, cam_index)
    total_files = len(all_tasks)
    total_input_dur = sum(float(t["file_duration"]) if t.get("file_duration") is not None else 0.0
                          for t in all_tasks)
    analysis_snapshot = [
        {
            "filepath": row.get("filepath"),
            "file_duration": row.get("file_duration"),
            "duration_verified": row.get("duration_verified"),
            "prescreen_status": row.get("prescreen_status"),
            "analysis_status": row.get("analysis_status"),
            "segments": [
                (float(s["start_time"]), float(s["end_time"]), str(s["state"]))
                if isinstance(s, dict)
                else (float(s.start_time), float(s.end_time), str(s.state))
                for s in row.get("segments", [])
            ],
        }
        for row in all_tasks
    ]
    run_context = {
        "run_id": observation.identity["run_id"],
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "analysis_snapshot_sha256": hashlib.sha256(
            json.dumps(analysis_snapshot, sort_keys=True, default=str).encode()
        ).hexdigest(),
    }
    try:
        run_context["code_revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip()
        run_context["code_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        run_context["code_revision"] = "unknown"

    # Rich 启动 Banner (含机位别名与高度自定义成片命名解析)
    out_cfg = config.get("output", {})
    naming_template = out_cfg.get("naming", "DailyVlog_{date}_{mac}.mp4")
    sample_filepath = all_tasks[0]["filepath"] if all_tasks else None
    from src.stages.scanner import resolve_output_filename, resolve_camera_ui_name
    output_name = resolve_output_filename(naming_template, date, cam_index, sample_filepath=sample_filepath, config=config)

    cam_display = None
    try:
        sample_dir = str(Path(sample_filepath).parent) if sample_filepath else ""
        cam_display = resolve_camera_ui_name(sample_dir, cam_index=cam_index, config=config)
    except Exception:
        cam_display = f"Cam {cam_index}"

    print_startup_banner(
        date=date,
        cam_index=cam_index,
        total_files=total_files,
        total_duration_s=total_input_dur,
        output_path=str(OUTPUT_DIR / output_name),
        cam_name=cam_display,
        device_info="CPU（预筛、精析与渲染）" if uses_software_pipeline(config) else None,
    )
    logger.info("=== STREAMING pipeline %s cam%d start: %d files, %.1fs ===", date, cam_index, total_files, total_input_dur)
    output_path = OUTPUT_DIR / output_name

    from src.hardware.render_cache import render_fingerprint, save_manifest
    def final_fingerprint():
        rows = db.get_all_file_tasks_for_date(date, cam_index)
        decisions = [
            {
                "filepath": r.get("filepath"),
                "file_duration": r.get("file_duration"),
                "prescreen_status": r.get("prescreen_status"),
                "analysis_status": r.get("analysis_status"),
                "segments": [
                    (float(s["start_time"]), float(s["end_time"]), str(s["state"]))
                    if isinstance(s, dict)
                    else (float(s.start_time), float(s.end_time), str(s.state))
                    for s in r.get("segments", [])
                ],
            }
            for r in rows
        ]
        return render_fingerprint([r["filepath"] for r in rows], json.dumps(decisions, sort_keys=True),
                                  "final", out_cfg.get("fps", 20), out_cfg, out_cfg.get("audio", {}), config)
    orchestrator = StreamingOrchestrator(db, date, cam_index, config, render_enabled=not skip_render,
                                         dashboard_enabled=dashboard_enabled, cam_name=cam_display)
    def fail(message, reason="PIPELINE_FAILED"):
        observation.reason = reason
        logger.error("Pipeline %s cam%d failed: %s", date, cam_index, message,
                     extra={"reason": reason, "date": date, "cam": cam_index, "disposition": "failed"})
        if not skip_render:
            try:
                db.set_render_status(date, cam_index, "FAILED", error_msg=message)
            except Exception:
                logger.exception("Could not persist failed delivery status")
        _dump_perf(get_perf(), monitor, date, cam_index, time.monotonic() - t_start,
                   worker_stats=getattr(orchestrator, "render_worker_stats", {}),
                   run_context={**run_context, "status": "failed", "error": message, "reason": reason})
        return False

    def record_media_failure(context, error):
        try:
            fail(f"{context} media termination unconfirmed: {error}")
        except Exception:
            logger.exception("Could not record fatal media failure; batch must still stop")

    try:
        with monitor.stage(f"pipeline_{date}_cam{cam_index}"):
            batch_paths = orchestrator.run()
    except (KeyboardInterrupt, SystemExit):
        _dump_perf(get_perf(), monitor, date, cam_index, time.monotonic() - t_start,
                   worker_stats=orchestrator.render_worker_stats,
                   run_context={**run_context, "status": "cancelled"})
        raise
    except MediaTerminationUnconfirmed as exc:
        record_media_failure("streaming", exc)
        # The failure remains fatal for this batch even if later reaping can
        # confirm termination. Only an explicit new execution may resume it.
        raise
    except InputStorageUnavailable as exc:
        fail(f"input storage unavailable: {exc}", "INPUT_STORAGE_UNAVAILABLE")
        raise
    except Exception as exc:
        logger.exception("streaming pipeline failed for %s cam%d", date, cam_index)
        return fail(f"streaming pipeline failed: {exc}")

    with orchestrator.error_lock:
        run_errors = list(orchestrator.errors)
    task_rows = db.get_all_file_tasks_for_date(date, cam_index)
    if all_tasks and not task_rows:
        return fail("source task reconciliation unavailable")
    # Lazy probing/decoding can replace nominal durations during this run.
    # Delivery metrics and the completion card must use the reconciled values,
    # including a known zero, rather than the pre-analysis startup estimate.
    total_input_dur = sum(float(row["file_duration"]) if row.get("file_duration") is not None else 0.0
                          for row in task_rows)
    total_files = len(task_rows)
    incomplete = [
        row for row in task_rows
        if row.get("prescreen_status") not in ("STATIC", "SUSPICIOUS", "SKIPPED_EMPTY", "SKIPPED_EXCLUDED")
        or (row.get("prescreen_status") == "SUSPICIOUS"
            and row.get("analysis_status") not in ("ANALYZED", "COMPLETED", "SKIPPED_EXCLUDED", "SKIPPED_CORRUPT"))
    ]
    if skip_render:
        if run_errors or incomplete:
            return fail("; ".join(run_errors[:3]) or f"{len(incomplete)} source tasks incomplete")
        _dump_perf(get_perf(), monitor, date, cam_index, time.monotonic() - t_start,
                   worker_stats=getattr(orchestrator, "render_worker_stats", {}),
                   run_context={**run_context, "status": "analysis_completed"})
        return True
    if not batch_paths:
        return fail("; ".join(run_errors[:3]) or "no render batches produced")

    # 运行期间的告警与异常汇总 (含管线内部错误)
    if run_errors:
        print_error_summary(run_errors)

    # 渲染批次级失败（含对账缺失）严禁合成半成品或删除已成批次
    render_batch_errors = [
        e for e in run_errors
        if e.startswith("render batch") or e.startswith("render batches silently dropped")
    ]
    if render_batch_errors:
        logger.error(
            "render %s cam%d has %d batch failures, aborting concat to preserve valid batches for resume: %s",
            date, cam_index, len(render_batch_errors), "; ".join(render_batch_errors[:3]),
        )
        return fail("; ".join(render_batch_errors[:3]), "RENDER_BATCH_FAILED")

    # A render batch can succeed while an upstream file task failed. Never
    # publish a day that silently omits failed or unfinished source material.
    if incomplete:
        logger.error(
            "render %s cam%d blocked: %d source tasks incomplete (first=%s)",
            date, cam_index, len(incomplete), incomplete[0].get("filepath"),
        )
        first = incomplete[0]
        return fail(
            f"{len(incomplete)} source tasks incomplete; first={first['filepath']}: "
            f"{first.get('error_msg') or first.get('prescreen_status') or first.get('analysis_status')}"
        )

    candidate = output_path.with_name(output_path.stem + ".pending.mp4")
    concat_t0 = time.monotonic()
    concat_timings = {}
    ok = False
    try:
        try:
            observation.set_phase("最终拼接中")
            with monitor.stage(f"final_concat_{date}_cam{cam_index}"):
                db.set_render_status(date, cam_index, "RENDERING")
                if len(batch_paths) == 1:
                    shutil.copy2(batch_paths[0], candidate)
                    ok = True
                else:
                    ok = concat_output_files(batch_paths, candidate, timings=concat_timings)
        except (KeyboardInterrupt, SystemExit):
            concat_timings["status"] = "cancelled"
            raise
        finally:
            t_concat_done = time.monotonic()
            get_perf().add(PerfRecord(
                stage="final_concat", file=output_path.name, gpu="cpu",
                duration=round(t_concat_done - concat_t0, 3),
                extra={"batch_count": len(batch_paths),
                       "status": "completed" if ok else "failed", **concat_timings},
                start_time=round(concat_t0, 3), end_time=round(t_concat_done, 3),
                worker="concat",
            ))
    except (KeyboardInterrupt, SystemExit):
        cleanup_delivery_candidate(candidate)
        _dump_perf(get_perf(), monitor, date, cam_index, time.monotonic() - t_start,
                   worker_stats=getattr(orchestrator, "render_worker_stats", {}),
                   run_context={**run_context, "status": "cancelled"})
        raise
    except MediaTerminationUnconfirmed as exc:
        record_media_failure("finalize render output", exc)
        raise  # A pending reader/writer may still own the candidate.
    except Exception as exc:
        logger.exception("finalize render output failed")
        cleanup_delivery_candidate(candidate)
        return fail(f"finalize render output failed: {exc}")

    if not ok:
        stage = concat_timings.get("failure_stage", "unknown")
        logger.error("concat_output_files failed for %s cam%d at %s", date, cam_index, stage)
        cleanup_delivery_candidate(candidate)
        return fail(f"final batch concat failed: {stage}")

    # 生成标准化成片伴随资产包：.srt 现实世界时间码字幕 + .meta.json 结构化清单
    media_metrics = {}
    retain_candidate = False
    try:
        observation.set_phase("时钟校验与元数据生成中")
        assets_ok = _save_vlog_companion_assets(
            output_path=candidate,
            date=date,
            cam_index=cam_index,
            cam_display=cam_display,
            total_files=total_files,
            total_input_dur=total_input_dur,
            elapsed_wall=time.monotonic() - t_start,
            db=db,
            config=config,
            media_metrics=media_metrics,
        )
        if not assets_ok:
            return fail("required companion assets failed", "DELIVERY_COMPANION_FAILED")
        observation.set_phase("交付提交中")
        fingerprint = final_fingerprint()
        with promote_vlog_assets(candidate, output_path, config,fingerprint=fingerprint,
                                 context={'role':'pipeline','date':date,'cam_index':cam_index}):
            save_manifest(output_path, fingerprint)
            db.set_render_status(date, cam_index, "COMPLETED", output_file=str(output_path))
            if not db.is_render_completed(date, cam_index):
                raise RuntimeError("completion record was not committed")
        observation.committed = True
    except (KeyboardInterrupt, SystemExit):
        try:
            db.set_render_status(date, cam_index, "PENDING", error_msg="delivery commit cancelled")
        except Exception:
            logger.exception("Could not persist cancelled delivery status")
        _dump_perf(get_perf(), monitor, date, cam_index, time.monotonic() - t_start,
                   worker_stats=getattr(orchestrator, "render_worker_stats", {}),
                   run_context={**run_context, "status": "cancelled"})
        raise
    except MediaTerminationUnconfirmed as exc:
        retain_candidate = True
        record_media_failure("delivery commit", exc)
        raise
    except Exception as exc:
        logger.exception("Delivery commit failed")
        return fail(f"delivery commit failed: {exc}")
    finally:
        if not retain_candidate:
            cleanup_delivery_candidate(candidate)
    observation.set_phase("已交付")
    # End-to-end means the final container, manifest, optional subtitles and DB
    # completion marker are all durable.  Previous reports stopped this clock
    # before concat and under-reported the user-observed runtime by ~20 seconds.
    elapsed_wall = time.monotonic() - t_start
    print_summary_card(
        date=date,
        cam_index=cam_index,
        total_files=total_files,
        total_input_dur=total_input_dur,
        output_path=output_path,
        elapsed_wall=elapsed_wall,
        cam_name=cam_display,
    )
    logger.info("Pipeline %s cam%d finished in %.1fs", date, cam_index, elapsed_wall)

    _dump_perf(get_perf(), monitor, date, cam_index, elapsed_wall,
               headline=_build_headline(output_path, total_input_dur, elapsed_wall,
                                        output_duration_s=media_metrics.get("duration_s")),
               worker_stats=getattr(orchestrator, "render_worker_stats", {}), run_context={**run_context, "status": "completed"})

    # 自动回收本地预暂存文件，释放磁盘存储空间
    from src.core.utils import cleanup_staging_files
    cleanup_staging_files()
    failure_root = output_path.parent / ".concat_failures"
    stale_failure_dir = failure_root / candidate.stem
    if stale_failure_dir.exists():
        shutil.rmtree(stale_failure_dir, ignore_errors=True)
        try:
            failure_root.rmdir()
        except OSError:
            pass
    if config.get("render", {}).get("cleanup_batches_on_success", False):
        for batch in batch_paths:
            try:
                batch.unlink(missing_ok=True)
                batch.with_name(batch.name + ".json").unlink(missing_ok=True)
            except OSError:
                logger.warning("Delivery committed; could not remove batch %s", batch)
    return True


def run_pipeline(skip_render: bool = False, input_dir: list[str] | None = None, dashboard_enabled: bool = True,
                 force_render: bool = False, config: dict | None = None) -> dict:
    FFmpegProcessRegistry.require_cleanups()
    from src.core.utils import cleanup_temp_artifacts
    logger.info("准备临时工作目录和任务数据库…", extra={"startup_progress": True})
    cleanup_temp_artifacts()
    db = VlogDatabase()
    logger.info("任务数据库已就绪，正在初始化硬件监控…", extra={"startup_progress": True})
    monitor = get_monitor()
    monitor.start()
    try:
        scan_result = scan_directory(db, input_dir=input_dir)
        verified_input_dirs = (
            ([str(input_dir)] if isinstance(input_dir, str)
             else [str(p) for p in input_dir] if input_dir is not None
             else get_input_dirs(config or load_config()))
            if isinstance(scan_result, ScanResult) else []
        )
        groups = get_date_cam_groups(db)
        if not groups:
            return {"total": 0, "ok": 0, "failed": 0, "skipped": 0, "remaining": 0, "stopped_reason": None}

        # Rich 批量任务启动 Banner
        dates = sorted(set(d for d, _ in groups))
        from src.stages.scanner import resolve_camera_ui_name
        cam_names = set()
        for d, c in groups:
            t_list = db.get_all_file_tasks_for_date(d, c)
            s_dir = str(Path(t_list[0]["filepath"]).parent) if t_list else ""
            cam_names.add(resolve_camera_ui_name(s_dir, cam_index=c, config=config or load_config()))
        cam_labels = sorted(cam_names)
        print_batch_startup_banner(
            total_groups=len(groups),
            date_range=(dates[0], dates[-1]),
            cameras=cam_labels,
            output_dir=str(OUTPUT_DIR),
            device_info="CPU（预筛、精析与渲染）" if uses_software_pipeline(config or load_config()) else None,
        )

        ok, failed, skipped = 0, 0, 0
        stopped_reason = None
        batch_summary_list = []
        for date, cam_index in groups:
            already_completed = not force_render and db.is_render_completed(date, cam_index)
            if not already_completed and not check_disk_space(OUTPUT_DIR):
                stopped_reason = "LOW_DISK_SPACE"
                logger.warning("Batch stopped: LOW_DISK_SPACE; %d groups remain", len(groups) - ok - failed - skipped)
                break
            if not already_completed and verified_input_dirs:
                bad_dir = check_input_storage(verified_input_dirs)
                if bad_dir is not None:
                    stopped_reason = "INPUT_STORAGE_UNAVAILABLE"
                    logger.error(
                        "Batch stopped: INPUT_STORAGE_UNAVAILABLE (%s); %d groups remain",
                        bad_dir, len(groups) - ok - failed - skipped,
                    )
                    break
            t0 = time.monotonic()
            success = False
            try:
                if process_date_cam(db, date, cam_index, skip_render=skip_render,
                                    dashboard_enabled=dashboard_enabled, force_render=force_render, config=config):
                    if already_completed:
                        skipped += 1
                    else:
                        ok += 1
                    success = True
                else:
                    failed += 1
            except KeyboardInterrupt:
                FFmpegProcessRegistry.mark_interrupted()
                FFmpegProcessRegistry.kill_all()
                logger.warning("run_pipeline interrupted. Halting batch pipeline.")
                raise
            except MediaTerminationUnconfirmed:
                logger.exception("Batch stopped: MEDIA_TERMINATION_UNCONFIRMED")
                failed += 1
                stopped_reason = "MEDIA_TERMINATION_UNCONFIRMED"
            except InputStorageUnavailable:
                logger.exception("Batch stopped: INPUT_STORAGE_UNAVAILABLE")
                failed += 1
                stopped_reason = "INPUT_STORAGE_UNAVAILABLE"
            except Exception:
                logger.exception("pipeline crash")
                failed += 1
            finally:
                wall_s = time.monotonic() - t0
                status = "SKIPPED" if success and already_completed else ("SUCCESS" if success else "FAILED")
                batch_summary_list.append(build_batch_summary_item(
                    db, date, cam_index, OUTPUT_DIR, load_config(), wall_s, status, skip_render=skip_render,
                ))
            # A day may return False after recording a fatal reader failure.
            # Inspect before cleanup so successful later termination cannot
            # silently turn that failure into permission to continue this batch.
            try:
                FFmpegProcessRegistry.require_cleanups()
            except MediaTerminationUnconfirmed:
                logger.exception("Batch stopped: MEDIA_TERMINATION_UNCONFIRMED")
                stopped_reason = "MEDIA_TERMINATION_UNCONFIRMED"
            cleanup_resources()
            if stopped_reason:
                break

        if len(groups) > 1:
            from src.ui import print_batch_summary_table
            print_batch_summary_table(batch_summary_list)

        return {"total": len(groups), "ok": ok, "failed": failed, "skipped": skipped,
                "remaining": len(groups) - ok - failed - skipped, "stopped_reason": stopped_reason}
    finally:
        monitor.shutdown()
        db.close()


__all__ = [
    "StreamingOrchestrator",
    "WorkStealingManager",
    "process_date_cam",
    "run_pipeline",
]
