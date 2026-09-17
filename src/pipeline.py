import json
import logging
import time
import threading
import queue
from pathlib import Path

from src.ui import (
    PipelineDashboard,
    print_batch_startup_banner,
    print_error_summary,
    print_startup_banner,
    print_summary_card,
)
from src.database import VlogDatabase
from src.scanner import scan_directory, get_date_cam_groups
from src.prescreen import prescreen_file
from src.detector import MotionDetector, detect_audio_activity
from src.renderer import build_batch_render, concat_output_files
from src.config import (
    load_config,
    OUTPUT_DIR,
    LOGS_DIR,
)
from src.scheduler import (
    WorkStealingManager,
    RenderBatchItem,
    DualEndedBatchQueue,
)
from src.ffmpeg import FFmpegProcessRegistry
from src.utils import (
    cleanup_resources,
    check_disk_space,
    register_dashboard,
    unregister_dashboard,
)
from dataclasses import dataclass, field
from src.monitor import get_monitor, get_perf, PerfRecord

logger = logging.getLogger("homevlog")


@dataclass
class PipelineTask:
    """Strongly-typed task contract flowing through prescreen, analysis, and render."""
    filepath: str
    cam_index: int = 0
    date: str = ""
    file_start_time: str = ""
    file_end_time: str = ""
    file_duration: float = 0.0
    has_audio: int = 0
    prescreen_status: str = "PENDING"
    prescreen_result: str = ""
    analysis_status: str = "PENDING"
    analysis_segments: str = ""
    id: int | None = None
    is_audio_gated_static: bool = False
    audio_events: list = field(default_factory=list)

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, data) -> "PipelineTask":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return data
        return cls(
            filepath=str(data.get("filepath", "")),
            cam_index=int(data.get("cam_index", 0)),
            date=str(data.get("date", "")),
            file_start_time=str(data.get("file_start_time", "")),
            file_end_time=str(data.get("file_end_time", "")),
            file_duration=float(data.get("file_duration") or 0.0),
            has_audio=int(data.get("has_audio") or 0),
            prescreen_status=str(data.get("prescreen_status", "PENDING")),
            prescreen_result=str(data.get("prescreen_result") or ""),
            analysis_status=str(data.get("analysis_status", "PENDING")),
            analysis_segments=str(data.get("analysis_segments", "")),
            id=data.get("id"),
            is_audio_gated_static=bool(data.get("is_audio_gated_static", False)),
            audio_events=list(data.get("audio_events") or []),
        )



class AnalysisQueue(queue.Queue):
    """Priority queue for detailed analysis.

    Single-file render batches no longer depend on chronological analysis
    completion. Short clips are admitted first (Shortest Job First / SJF)
    to immediately produce render batches and eliminate early-stage NVENC starvation.
    Multi-file batches retain chronological order for compatibility.
    """
    def _init(self, maxsize):
        self.queue = []
        self.sequence = 0
        self.cost_priority = False

    def _put(self, task):
        import heapq
        if self.cost_priority:
            # SJF: 短作业优先（时长升序），让 300s 标准切片以秒级出清迅速喂饱渲染编队
            priority = (float(task.get("file_duration") or 0.0), task.get("file_start_time") or "")
        else:
            priority = (0.0, task.get("file_start_time") or "")
        heapq.heappush(self.queue, (priority, self.sequence, task))
        self.sequence += 1

    def _get(self):
        import heapq
        return heapq.heappop(self.queue)[2]


class RenderMessageQueue(queue.Queue):
    """Render readiness queue that stamps producer time for queue-wait metrics."""

    def put(self, item, *args, **kwargs):
        if isinstance(item, dict):
            item.setdefault("_render_ready_at", time.monotonic())
        return super().put(item, *args, **kwargs)


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
    ):
        self.db = db
        self.date = date
        self.cam_index = cam_index
        self.config = config
        self.render_enabled = render_enabled

        # 异构硬件自适应调度器
        self.work_stealing = WorkStealingManager(config=self.config)



        # 队列定义
        self.prescreen_queue = queue.Queue()
        self.analysis_queue = AnalysisQueue()
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
        self.io_limit = io_limit
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
        from src.renderer import FFmpegProcessRegistry
        FFmpegProcessRegistry.reset_interrupted()
        self.stop_event = threading.Event()
        self.abort_event = threading.Event()
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

        # 冷启动预热保护：避免启动瞬间队列无积压导致 SJF 短作业优先失效
        self._analysis_warmup_event = threading.Event()

        # 异步前瞻预分期与编队动态竞价追踪
        self._prefetch_queue = queue.Queue()
        self._prefetched_files = set()
        self._prefetched_lock = threading.Lock()
        self._active_nv_renders = 0

        # Rich 仪表盘 (tqdm 已下线，统一由 PipelineDashboard 呈现)
        self.dashboard_enabled = dashboard_enabled
        self.dashboard: PipelineDashboard | None = None
        self._prescreen_t0 = 0.0
        self._analysis_t0 = 0.0

    def _guard_worker(self, worker, *args):
        try:
            worker(*args)
        except BaseException as exc:
            logger.exception("Pipeline worker terminated unexpectedly")
            self._add_error(f"render batch pipeline worker failed: {exc}")
            self.abort_event.set()
            self.stop_event.set()
            self.render_finished_event.set()
            from src.renderer import FFmpegProcessRegistry
            FFmpegProcessRegistry.kill_all()

    def _add_error(self, message: str):
        with self.error_lock:
            self.errors.append(message)
        if self.dashboard is not None:
            self.dashboard.add_alert(message)

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
        from src.renderer import _stage_source_for_render
        while not self.abort_event.is_set() and (not self.stop_event.is_set() or not self._prefetch_queue.empty()):
            try:
                fp = self._prefetch_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if fp is None:
                self._prefetch_queue.task_done()
                break
            try:
                _stage_source_for_render(fp, needed=lambda: self._needs_prefetch(fp))
            except Exception:
                pass
            finally:
                self._prefetch_queue.task_done()

    def _prescreen_worker(self, gpu: str = "qsv", worker_id: str = ""):
        """预筛 Worker：将扫描到的文件进行快速筛选。"""
        worker_name = worker_id or f"pre_{gpu}"
        while not self.abort_event.is_set() and (not self.stop_event.is_set() or not self.prescreen_queue.empty()):
            try:
                task = self.prescreen_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            filepath = task["filepath"]
            duration = task.get("file_duration") or 300.0
            t0 = time.monotonic()

            # 时间邻域弹性保护：前置相邻素材若存在活动，本段门槛弹性下调以防漏切
            prev_fp = self._prev_task_map.get(filepath)
            is_prior_active = False
            if prev_fp:
                with self._prescreen_results_lock:
                    is_prior_active = (self._prescreen_results.get(prev_fp) == "SUSPICIOUS")

            try:
                res = prescreen_file(filepath, duration, self.config, gpu=gpu, is_prior_active=is_prior_active)
                if (res["status"] == "STATIC" and res.get("has_audio") and
                        self.config.get("audio_vad", {}).get("prescreen_audio_gate", False)):
                    # A visual negative does not need the full video analysis
                    # path just to discover whether its audio is silent.
                    # Decode audio features only; failures stay conservative.
                    try:
                        audio_events, audio_stats = detect_audio_activity(
                            filepath, duration, self.config
                        )
                        if audio_events:
                            res["status"] = "SUSPICIOUS"
                            res["result_json"] = json.dumps({
                                "mode": "keyframes_audio_gate",
                                "audio_events": audio_events,
                                "vad_stats": audio_stats,
                            })
                        else:
                            res["result_json"] = json.dumps({
                                "mode": "keyframes_audio_gate",
                                "audio_events": [],
                                "vad_stats": audio_stats,
                            })
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
                self.db.set_prescreen_result(filepath, res["status"], result_json, has_audio=prescreen_has_audio)
                if prescreen_has_audio is not None:
                    task["has_audio"] = int(prescreen_has_audio)
                if res["status"] in ("STATIC", "SUSPICIOUS"):
                    from src.render_cache import processing_fingerprint
                    self.db.set_processing_fingerprint(filepath, processing_fingerprint(filepath, self.config))
                with self._prescreen_results_lock:
                    self._prescreen_results[filepath] = res["status"]

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
                        if self.analysis_queue.qsize() >= 2 or self.prescreen_queue.empty():
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

            except Exception:
                logger.exception("Streaming: prescreen failed for %s", filepath)
                self.db.set_prescreen_result(filepath, "FAILED", "")
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
    ):
        from src.segment import build_segments, segments_to_json

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
            from src.algorithms.segment import build_audio_gated_segments
            seg_cfg = self.config.get("segment", {})
            file_dur = float(task.get("file_duration") or 300.0)
            segments = build_audio_gated_segments(
                filepath=filepath,
                file_duration=file_dur,
                file_start_offset=file_start_offset,
                audio_events=audio_events,
                pre_roll=seg_cfg.get("pre_roll", 1.0),
                post_roll=seg_cfg.get("post_roll", 1.5),
                gap_tolerance=seg_cfg.get("gap_tolerance", 1.5),
            )
            js = segments_to_json(segments)
            self.db.set_analysis_result(filepath, "ANALYZED", js)
            from src.render_cache import processing_fingerprint
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

        labels, yolo_buffer = detector.analyze(
            filepath,
            start_offset=file_start_offset,
            file_duration=float(task.get("file_duration") or 300.0),
            has_audio=task.get("has_audio"),
        )
        
        gpu = getattr(detector, "decode_gpu", gpu)
        if hasattr(detector, 'has_audio_detected'):
            final_has_audio = max(int(task.get("has_audio") or 0), int(detector.has_audio_detected or 0))
            self.db.set_file_metadata(filepath, final_has_audio, getattr(detector, "file_duration_detected", None))

        if labels:
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
                yolo_device = self.config.get("hardware", {}).get("device", "cpu")
                # yolo_buffer 帧键以解码实际 effective_fps 为基准，禁止使用固定的 detector.fps
                effective_fps = getattr(detector, "last_perf", {}).get("effective_fps") or detector.fps
                segments = yolo_verifier.verify(
                    filepath, segments, gpu=gpu, device=yolo_device, frames_buffer=yolo_buffer, analysis_fps=effective_fps
                )
                yolo_telemetry = getattr(yolo_verifier, "last_telemetry", {})
                from src.segment import _merge_same_state, _filter_short
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
            from src.segment import refine_activity_segments
            segments = refine_activity_segments(segments, labels, self.config)
            yolo_after = len(segments)

            from src.archiver import FrameArchiver
            rate = detector.last_perf.get("effective_fps", detector.fps)
            for seg in segments:
                if not seg.needs_review or not yolo_buffer:
                    continue
                midpoint = (seg.start_time + seg.end_time) / 2 - seg.file_start_offset
                key = min(yolo_buffer, key=lambda k: abs(k / rate - midpoint))
                payload = yolo_buffer[key]
                if getattr(payload, "ndim", 0) == 1:
                    FrameArchiver.remember(filepath, key / rate, payload)
            js = segments_to_json(segments)
            self.db.set_analysis_result(filepath, "ANALYZED", js)
            from src.render_cache import processing_fingerprint
            self.db.set_processing_fingerprint(filepath, processing_fingerprint(filepath, self.config))

            t_ana_done = time.monotonic()
            dur_total = round(t_ana_done - t0, 3)
            lp = detector.last_perf if hasattr(detector, "last_perf") else {}
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
                    gpu=gpu,
                    duration=dur_total,
                    frames=frames_count,
                    fps=round(frames_count / max(0.001, dur_total), 1),
                    extra={
                        "status": "ANALYZED",
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
            dynamic_duration = sum(
                (s.end_time - s.start_time)
                for s in segments
                if getattr(s, "is_dynamic", False) or getattr(s, "state", "") in ("DYNAMIC", "DYNAMIC_AUDIO")
            )
            has_dynamic = dynamic_duration > 0
            self.render_batch_queue.put(
                {
                    "filepath": filepath,
                    "status": "ANALYZED",
                    "is_heavy": has_dynamic,
                    "dynamic_duration": dynamic_duration,
                }
            )
        else:
            self.db.set_analysis_result(filepath, "FAILED", "")
            self._add_error(f"analysis produced no labels: {Path(filepath).name}")
            t_ana_fail = time.monotonic()
            perf.add(
                PerfRecord(
                    stage="analysis",
                    file=Path(filepath).name,
                    gpu=gpu,
                    duration=round(t_ana_fail - t0, 3),
                    extra={
                        "status": "NO_LABELS",
                        "analysis_queue_wait_s": round(analysis_queue_wait, 3),
                    },
                    start_time=round(t0, 3),
                    end_time=round(t_ana_fail, 3),
                    worker=worker_id or "ana",
                )
            )
            self.render_batch_queue.put(
                {"filepath": filepath, "status": "FAILED"}
            )

        if self.dashboard is not None:
            el_s = max(0.1, time.monotonic() - self._analysis_t0)
            n_done = self.dashboard.analysis_done + 1
            task_el = max(0.1, time.monotonic() - t0)
            frames_done = getattr(detector, "last_perf", {}).get("frames", 0)
            fps_val = frames_done / task_el if frames_done else 0.0
            hw_label = "核显" if gpu == "qsv" else "独显"
            speed_str = f"{fps_val:.1f} 帧/秒（{hw_label}解码）" if fps_val > 0 else f"均速 {el_s / n_done:.1f}s/个"
            self.dashboard.update_analysis(
                completed=n_done,
                latest_file=Path(filepath).name,
                speed_str=speed_str,
            )
            self._sync_queue_levels()

    def _analysis_worker(self, fixed_gpu: str | None = None, worker_id: str = ""):
        """分析 Worker：对 SUSPICIOUS 文件进行运动检测与目标验证 (支持异构自适应工作窃取)。"""
        detectors: dict[str, MotionDetector] = {
            "qsv": MotionDetector(self.config, decode_gpu="qsv"),
        }
        detector_cuda: MotionDetector | None = None
        
        # [性能极限] 在线程生命周期内复用验证器实例，避免重复初始化开销
        yolo_verifier = None
        yolo_cfg = self.config.get("yolo", {})
        if yolo_cfg.get("enabled", False) and yolo_cfg.get("streaming_verify", True):
            from src.yolo_verifier import YoloVerifier
            try:
                yolo_verifier = YoloVerifier(self.config)
            except Exception as exc:
                # Keep consuming the queue; absence of semantics must not strand unfinished tasks.
                logger.warning("YOLO initialization failed; retaining visual/audio motion: %s", exc)
                self._add_error("YOLO unavailable; motion retained and marked for review")
            
        perf = get_perf()

        # 冷启动预热保护：等待预筛先形成候选积压，确保 SJF 短作业优先正常排序出清
        if not self._analysis_warmup_event.is_set():
            self._analysis_warmup_event.wait(timeout=2.0)

        while not self.abort_event.is_set() and (not self.stop_event.is_set() or not self.analysis_queue.empty()):
            try:
                task = self.analysis_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            filepath = task["filepath"]
            t0 = time.monotonic()
            analysis_queue_wait = max(
                0.0, t0 - float(task.get("_analysis_queued_at") or t0)
            )

            try:
                from src.utils import ts_to_unix

                file_start_ts = ts_to_unix(task["file_start_time"])
                file_start_offset = max(
                    file_start_ts - ts_to_unix(self.date + "000000"), 0.0
                )

                if fixed_gpu is not None:
                    gpu = fixed_gpu
                    if gpu == "cuda":
                        if detector_cuda is None:
                            detector_cuda = MotionDetector(self.config, decode_gpu="cuda")
                        detector = detector_cuda
                    else:
                        detector = detectors["qsv"]
                    self._execute_analysis_task(
                        task, filepath, file_start_offset, detector, yolo_verifier, gpu, perf, t0, worker_id=worker_id
                    )
                else:
                    q_size = self.analysis_queue.qsize()
                    prescreen_idle = self.prescreen_queue.empty() and q_size <= self.work_stealing.watermark_low
                    detector = detectors["qsv"]
                    detector.device_lease = lambda: self.work_stealing.lease_device(q_size, prescreen_idle=prescreen_idle)
                    self._execute_analysis_task(
                        task, filepath, file_start_offset, detector, yolo_verifier, "adaptive", perf, t0, worker_id=worker_id
                    )

            except Exception:
                logger.exception("Streaming: analysis failed for %s", filepath)
                self.db.set_analysis_result(filepath, "FAILED", "")
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
                self.analysis_queue.task_done()

    def _render_manager(self):
        """渲染管理器：In-Order Sliding Window 保序消费完成消息，构建时间严格单调的批次并启动渲染 Worker。"""
        out_cfg = self.config.get("output", {})
        fps = out_cfg.get("fps", 20)
        from src.utils import parse_res

        width, height = parse_res(out_cfg.get("resolution", "1920x1080"))
        seg_cfg = self.config.get("segment", {})
        audio_cfg = out_cfg.get("audio", {})
        render_cfg = self.config.get("render", {})
        render_hw_policy = self.config.get("pipeline", {}).get("render_gpu_policy", "heterogeneous")
        max_qsv_dynamic_s = float(render_cfg.get("max_qsv_dynamic_duration_s", 420.0))

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
        for r in all_rows:
            fp = r["filepath"]
            raw_segs = r.get("analysis_segments")
            if raw_segs and ("DYNAMIC" in raw_segs or "DYNAMIC_AUDIO" in raw_segs):
                try:
                    from src.timeline import segments_from_json
                    segs = segments_from_json(raw_segs)
                    dyn_dur = sum((s.end_time - s.start_time) for s in segs if getattr(s, "is_dynamic", False))
                    dynamic_duration_by_file[fp] = dyn_dur
                    ready_heavy[fp] = (dyn_dur > 0)
                except Exception:
                    dynamic_duration_by_file[fp] = 0.0
            else:
                dynamic_duration_by_file[fp] = 0.0
                if r.get("prescreen_status") == "STATIC":
                    ready_heavy[fp] = False

        render_cost_by_file = {
            r["filepath"]: float(r.get("file_duration") or 0.0)
            for r in all_rows
        }
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
        all_dispatched_event = threading.Event()
        render_start_t: list[float] = []
        # 已进入终态（失败已上报 / 空时间轴跳过）的批次集合，用于收尾对账去重
        terminal_batch_ids: set[int] = set()
        dispatch_sequence = 0
        dispatch_lock = threading.Lock()

        def _is_heavy_batch(files_to_batch: list[str]) -> bool:
            """检查批次中是否包含需复杂处理的 DYNAMIC 动作片段。"""
            if self.abort_event.is_set() or getattr(self.db, "is_closed", False):
                return False
            if files_to_batch and all(fp in ready_heavy for fp in files_to_batch):
                return any(ready_heavy[fp] for fp in files_to_batch)
            try:
                for fp in files_to_batch:
                    if fp in ready_heavy:
                        if ready_heavy[fp]:
                            return True
                        continue
                    r = self.db.get_file_task_summary(fp)
                    if not r:
                        continue
                    if r.get("prescreen_status") == "SUSPICIOUS":
                        raw_segs = r.get("analysis_segments")
                        if raw_segs and ("DYNAMIC" in raw_segs or "DYNAMIC_AUDIO" in raw_segs):
                            ready_heavy[fp] = True
                            return True
                    ready_heavy[fp] = False
            except Exception:
                pass
            return False

        nv_batch_durations: list[float] = []
        nv_duration_lock = threading.Lock()

        def _render_worker(gpu: str, worker_id: str = ""):
            worker_name = worker_id or gpu
            nonlocal dispatch_sequence
            t_worker_start = time.monotonic()
            busy_time = 0.0
            batches_count = 0
            dynamic_sec_total = 0.0
            qsv_light_count = 0
            qsv_stolen_count = 0
            qsv_steal_rejected = 0

            try:
                while not self.abort_event.is_set() and not FFmpegProcessRegistry.is_interrupted():
                    item = None
                    is_stolen = False
                    is_light = False
                    if gpu == "nv":
                        # NVENC (3060Ti) 专职优先消费 heavy 动态任务 (LPT 最长优先)，空闲时协助消费 light 任务
                        item = heavy_queue.pop_heaviest(timeout=0.5)
                        if item is None:
                            try:
                                item = light_queue.get(timeout=0.5)
                                if item is not None:
                                    is_light = True
                            except queue.Empty:
                                if all_dispatched_event.is_set() and heavy_queue.empty() and light_queue.empty():
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
                                    # 当所有素材派发完毕：若队列已空，或剩余批次 <= 1 且 NVENC 正在处理任务，QSV 立即退出
                                    if all_dispatched_event.is_set() and (heavy_queue.empty() or (q_len <= 1 and active_nv >= 1)):
                                        break

                                    # 实时 Makespan 竞价模型 (Cost-Based ETA Bidding):
                                    # 双路 NVENC 平均耗时 ~42s/批次，预估 NVENC 编队清空当前所有排队批次所需的剩余完工时间 (T_nv_eta)：
                                    # 当 NVENC 积压明显 (t_nv_eta >= 35s) 时，QSV 编码能力 (100+ fps, ~5x 实时) 完全能并发出清 300~360s 标准切片，
                                    # 平滑放宽窃取门限至 max_qsv_dynamic_s，彻底根除 QSV 500+ 次被拒空转与长尾失衡。
                                    with nv_duration_lock:
                                        avg_nv_time = (sum(nv_batch_durations[-8:]) / len(nv_batch_durations[-8:])) if nv_batch_durations else 42.0
                                    t_nv_eta = max(30.0, ((q_len + active_nv) / 2.0) * avg_nv_time)
                                    allowed_dyn_s = max_qsv_dynamic_s if t_nv_eta >= 35.0 else min(max_qsv_dynamic_s, max(60.0, t_nv_eta * 2.5))
                                    item = heavy_queue.steal_lightest(max_dynamic_s=allowed_dyn_s)
                                    if item is None:
                                        if all_dispatched_event.is_set() and heavy_queue.empty() and light_queue.empty():
                                            break
                                        if not heavy_queue.empty():
                                            qsv_steal_rejected += 1
                                        time.sleep(0.5)
                                        continue
                                    else:
                                        is_stolen = True
                                else:
                                    item = heavy_queue.pop_heaviest(timeout=0.5)
                            except Exception:
                                pass

                    if item is None:
                        if all_dispatched_event.is_set() and heavy_queue.empty() and light_queue.empty():
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
                    logger.info("render batch %d started on %s (%d files, worker=%s)", b_idx, gpu, len(files_to_batch), worker_name)
                    if gpu == "nv":
                        with self.batch_lock:
                            self._active_nv_renders += 1
                        self.work_stealing.register_render_start()
                    try:
                        all_rows = []
                        try:
                            if self.abort_event.is_set() or getattr(self.db, "is_closed", False):
                                break
                            all_rows = self.db.get_all_file_tasks_for_date(self.date, self.cam_index)
                        except Exception as e:
                            if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                                break
                            logger.warning("DB query failed for batch %d: %s", b_idx, e)
                        from src.timeline import build_timeline_from_rows
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
                            from src.timeline import TimelineSegment
                            rows_by_path = {r["filepath"]: r for r in all_rows}

                            # 检查批次文件是否已被宏观折叠（Macro-collapse）或分析确认无需画面输出
                            is_all_collapsed_or_skipped = True
                            for filepath in files_to_batch:
                                row = rows_by_path.get(filepath)
                                if not row:
                                    is_all_collapsed_or_skipped = False
                                    break
                                ana_status = row.get("analysis_status")
                                pre_status = row.get("prescreen_status")
                                if pre_status == "STATIC" or ana_status in ("ANALYZED", "COMPLETED"):
                                    raw_segs = row.get("analysis_segments") or ""
                                    if "DYNAMIC" not in raw_segs and "DYNAMIC_AUDIO" not in raw_segs:
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
                                raw_segs = row.get("analysis_segments") or ""
                                has_dynamic = ("DYNAMIC" in raw_segs or "DYNAMIC_AUDIO" in raw_segs)
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
                                if self.dashboard is not None:
                                    self.dashboard.render_batch_finished(len(files_to_batch))
                                continue

                        # 异构长尾保护：QSV 仅消费中轻量动态任务，若真实切片包含超过门限的 DYNAMIC 动作则交还 NVENC
                        if gpu == "qsv" and render_hw_policy == "heterogeneous":
                            dynamic_dur = sum(s.duration for s in batch_segs if s.state in ("DYNAMIC", "DYNAMIC_AUDIO"))
                            if dynamic_dur > max_qsv_dynamic_s:
                                logger.info(
                                    "render batch %d on qsv contains %.1fs dynamic motion (exceeds %.1fs), offloading to nv heavy queue",
                                    b_idx, dynamic_dur, max_qsv_dynamic_s,
                                )
                                heavy_queue.put(b_idx, files_to_batch, dynamic_dur)
                                continue

                        # Once a consumer owns the file, queued prefetch must not
                        # recreate its cache after the renderer has released it.
                        with self._prefetched_lock:
                            self._prefetched_files.difference_update(files_to_batch)
                        res_path = build_batch_render(
                            batch_segs, b_idx, gpu, fps, width, height,
                            seg_cfg, out_cfg, audio_cfg, self.date, self.cam_index,
                            all_rows
                        )
                        r_dur = round(time.monotonic() - t_r0, 3)
                        busy_time += r_dur
                        batches_count += 1
                        batch_dyn = sum(s.duration for s in batch_segs if s.state in ("DYNAMIC", "DYNAMIC_AUDIO"))
                        dynamic_sec_total += batch_dyn
                        logger.info("render batch %d finished on %s in %.3fs (output=%s)", b_idx, gpu, r_dur, bool(res_path))
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
                                        "is_stolen": is_stolen,
                                        "is_light": is_light,
                                        "render_queue_wait_s": round(render_queue_wait, 3),
                                    },
                                    start_time=round(t_r0, 3),
                                    end_time=round(time.monotonic(), 3),
                                    worker=worker_name,
                                )
                            )
                            if self.dashboard is not None:
                                self.dashboard.render_batch_finished(len(files_to_batch))
                                # 精准渲染阶段自身计时与剩余预估
                                t_start = render_start_t[0] if render_start_t else t_r0
                                el_sec = max(0.1, time.monotonic() - t_start)
                                n_done = self.dashboard.render_done + 1
                                n_total = max(n_done, self.dashboard.render_total)
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
                            self._add_error(f"render batch {b_idx} returned no output")
                            with self.batch_lock:
                                terminal_batch_ids.add(b_idx)
                            if self.dashboard is not None:
                                self.dashboard.render_batch_finished(len(files_to_batch))
                    except Exception:
                        if self.abort_event.is_set() or FFmpegProcessRegistry.is_interrupted():
                            break
                        logger.exception("Streaming: render batch %d failed on %s", b_idx, gpu)
                        self._add_error(f"render batch {b_idx} failed on {gpu}")
                        with self.batch_lock:
                            terminal_batch_ids.add(b_idx)
                        if self.dashboard is not None:
                            self.dashboard.render_batch_finished(len(files_to_batch))
                    finally:
                        if gpu == "nv":
                            with self.batch_lock:
                                self._active_nv_renders = max(0, self._active_nv_renders - 1)
                            self.work_stealing.register_render_end()
            finally:
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
                    }

        # 启动渲染编队：异构模式下采用 2 NVENC + 1 QSV 三路并行
        render_threads = []
        if render_hw_policy == "nv_only":
            render_gpus = ["nv"] * self.render_workers
        elif render_hw_policy == "qsv_only":
            render_gpus = ["qsv"] * self.render_workers
        else:
            logger.info(
                "Streaming: render_gpu_policy='heterogeneous' active with %d workers: deploying "
                "2 NVENC + 1 QSV with in-band parameter set injection (-bsf:v dump_extra).",
                self.render_workers,
            )
            render_gpus = ["nv"] if self.render_workers == 1 else ["nv"] * min(2, self.render_workers - 1) + ["qsv"]
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
            """估算批次的真实动态时长，用于 LPT/SPT 双端队列精准定价。"""
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
                    if status != "FAILED" and filepath not in dispatched_files:
                        _enqueue_batch(order_index[filepath], [filepath])
                        dispatched_files.add(filepath)
                elif filepath in ordered_set:
                    ready_status[filepath] = status
                elif status != "FAILED":
                    pending_files.append(filepath)
                while head < len(ordered_files) and ordered_files[head] in ready_status:
                    fp_head = ordered_files[head]
                    if ready_status[fp_head] != "FAILED":
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
            if missing:
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
        )
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
                self.render_batch_queue.put({"filepath": task["filepath"], "status": "FAILED"})

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
        prescreen_gpu_policy = self.config.get("pipeline", {}).get("prescreen_gpu_policy", "qsv_only")
        for i in range(prescreen_parallel):
            gpu = "qsv"
            if prescreen_gpu_policy == "alternating":
                gpu = "qsv" if i % 2 == 0 else "cuda"
            elif prescreen_gpu_policy == "cuda_only":
                gpu = "cuda"
            w_id = f"pre_{i}"
            t = threading.Thread(target=self._guard_worker, args=(self._prescreen_worker, gpu, w_id), daemon=True)
            t.start()
            threads.append(t)

        analysis_max_workers = self.analysis_workers
        for i in range(analysis_max_workers):
            w_id = f"ana_{i}"
            t = threading.Thread(target=self._guard_worker, args=(self._analysis_worker, None, w_id), daemon=True)
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
            _wait_queue_interruptible(self.prescreen_queue)
            self._analysis_warmup_event.set()
            _wait_queue_interruptible(self.analysis_queue)
            if self.render_enabled:
                _wait_queue_interruptible(self.render_batch_queue)
            # 通知上游与调度管理器：所有输入队列消费完毕，准备收尾
            self.stop_event.set()
            if self.render_enabled:
                # 等待所有分发的渲染批次真正执行完成并完成对账
                while not self.abort_event.is_set():
                    if self.render_finished_event.wait(timeout=0.2):
                        break
        except (KeyboardInterrupt, SystemExit):
            logger.warning("StreamingOrchestrator interrupted by user (Ctrl+C). Terminating subprocesses...")
            self.abort_event.set()
            self.stop_event.set()
            self.render_finished_event.set()
            from src.renderer import FFmpegProcessRegistry
            FFmpegProcessRegistry.mark_interrupted()
            FFmpegProcessRegistry.kill_all()
            if self.dashboard is not None:
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

        if self.dashboard is not None:
            self.dashboard.stop()
            self.dashboard = None
        unregister_dashboard()

        with self.batch_lock:
            self.batch_paths.sort(key=lambda x: x[0])
            final_paths = [p for _, p in self.batch_paths]

        with self.error_lock:
            errors = list(self.errors)
        if errors:
            logger.warning("Streaming: some tasks failed during run: %s", "; ".join(errors[:5]))

        return final_paths


def process_date_cam(
    db: VlogDatabase, date: str, cam_index: int,
    skip_render: bool = False, dashboard_enabled: bool = True,
    force_render: bool = False,
) -> bool:
    config = load_config()
    monitor = get_monitor()
    t_start = time.monotonic()
    # 失败任务自愈：将历史 FAILED 预筛/分析重置为 PENDING（retry_count 上限防护），
    # 使此前因信号量饥饿等原因丢失的文件在本次运行中补齐
    db.invalidate_stale_results(date, cam_index, config)
    db.reset_failed_tasks(date, cam_index)
    all_tasks = db.get_all_file_tasks_for_date(date, cam_index)
    total_files = len(all_tasks)
    total_input_dur = sum(float(t.get("file_duration") or 300.0) for t in all_tasks)

    # Rich 启动 Banner (含机位别名与高度自定义成片命名解析)
    out_cfg = config.get("output", {})
    naming_template = out_cfg.get("naming", "DailyVlog_{date}_{mac}.mp4")
    sample_filepath = all_tasks[0]["filepath"] if all_tasks else None
    from src.scanner import resolve_output_filename, resolve_camera_identity
    output_name = resolve_output_filename(naming_template, date, cam_index, sample_filepath=sample_filepath, config=config)

    cam_display = None
    try:
        sample_dir = str(Path(sample_filepath).parent) if sample_filepath else ""
        cam_display, _ = resolve_camera_identity(sample_dir, cam_index=cam_index, config=config)
    except Exception:
        cam_display = None

    print_startup_banner(
        date=date,
        cam_index=cam_index,
        total_files=total_files,
        total_duration_s=total_input_dur,
        output_path=str(OUTPUT_DIR / output_name),
        cam_name=cam_display,
    )
    logger.info("=== STREAMING pipeline %s cam%d start: %d files, %.1fs ===", date, cam_index, total_files, total_input_dur)
    output_path = OUTPUT_DIR / output_name

    from src.render_cache import render_fingerprint, reusable, save_manifest
    def final_fingerprint():
        rows = db.get_all_file_tasks_for_date(date, cam_index)
        decisions = [{k: r.get(k) for k in ("filepath", "file_duration", "prescreen_status", "analysis_segments", "human_reviews")} for r in rows]
        return render_fingerprint([r["filepath"] for r in rows], json.dumps(decisions, sort_keys=True),
                                  "final", out_cfg.get("fps", 20), out_cfg, out_cfg.get("audio", {}), config)
    if (not force_render and not skip_render and db.is_render_completed(date, cam_index) and
            db.get_pending_file_count_for_date(date, cam_index) == 0 and
            reusable(output_path, final_fingerprint())):
        logger.info("Verified completed output: %s", output_path)
        return True
    orchestrator = StreamingOrchestrator(db, date, cam_index, config, render_enabled=not skip_render, dashboard_enabled=dashboard_enabled)
    try:
        with monitor.stage(f"pipeline_{date}_cam{cam_index}"):
            batch_paths = orchestrator.run()
    except Exception:
        logger.exception("streaming pipeline failed for %s cam%d", date, cam_index)
        db.set_render_status(date, cam_index, "FAILED")
        return False

    if skip_render:
        return True
    if not batch_paths:
        db.set_render_status(date, cam_index, "FAILED")
        return False

    # 运行期间的告警与异常汇总 (含管线内部错误)
    with orchestrator.error_lock:
        run_errors = list(orchestrator.errors)
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
        db.set_render_status(date, cam_index, "FAILED")
        elapsed_wall = time.monotonic() - t_start
        _dump_perf(get_perf(), monitor, date, cam_index, elapsed_wall, worker_stats=getattr(orchestrator, "render_worker_stats", {}))
        return False

    # A render batch can succeed while an upstream file task failed. Never
    # publish a day that silently omits failed or unfinished source material.
    task_rows = db.get_all_file_tasks_for_date(date, cam_index)
    incomplete = [
        row for row in task_rows
        if row.get("prescreen_status") in ("PENDING", "FAILED")
        or (row.get("prescreen_status") == "SUSPICIOUS"
            and row.get("analysis_status") != "ANALYZED")
    ]
    if incomplete:
        logger.error(
            "render %s cam%d blocked: %d source tasks incomplete (first=%s)",
            date, cam_index, len(incomplete), incomplete[0].get("filepath"),
        )
        db.set_render_status(date, cam_index, "FAILED")
        elapsed_wall = time.monotonic() - t_start
        _dump_perf(get_perf(), monitor, date, cam_index, elapsed_wall, worker_stats=getattr(orchestrator, "render_worker_stats", {}))
        return False

    db.upsert_render_task(date, cam_index, "RENDERING")
    concat_t0 = time.monotonic()
    try:
        if len(batch_paths) == 1:
            batch_paths[0].replace(output_path)
            ok = True
        else:
            ok = concat_output_files(batch_paths, output_path)
            if ok:
                for p in batch_paths:
                    p.unlink(missing_ok=True)
                    p.with_name(p.name + ".json").unlink(missing_ok=True)
            else:
                logger.error("concat_output_files failed for %s cam%d", date, cam_index)
                db.set_render_status(date, cam_index, "FAILED")
                elapsed_wall = time.monotonic() - t_start
                _dump_perf(get_perf(), monitor, date, cam_index, elapsed_wall, worker_stats=getattr(orchestrator, "render_worker_stats", {}))
                return False
    except Exception:
        logger.exception("finalize render output failed")
        db.set_render_status(date, cam_index, "FAILED")
        return False

    t_concat_done = time.monotonic()
    get_perf().add(PerfRecord(
        stage="final_concat", file=output_path.name, gpu="cpu",
        duration=round(t_concat_done - concat_t0, 3),
        extra={"batch_count": len(batch_paths)},
        start_time=round(concat_t0, 3),
        end_time=round(t_concat_done, 3),
        worker="concat",
    ))

    save_manifest(output_path, final_fingerprint())

    # 生成标准化成片伴随资产包：.srt 现实世界时间码字幕 + .meta.json 结构化清单
    _save_vlog_companion_assets(
        output_path=output_path,
        date=date,
        cam_index=cam_index,
        cam_display=cam_display,
        total_files=total_files,
        total_input_dur=total_input_dur,
        elapsed_wall=time.monotonic() - t_start,
        db=db,
        config=config,
    )

    db.set_render_status(date, cam_index, "COMPLETED", output_file=str(output_path))
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
               headline=_build_headline(output_path, total_input_dur, elapsed_wall),
               worker_stats=getattr(orchestrator, "render_worker_stats", {}))

    # 自动回收本地预暂存文件，释放磁盘存储空间
    from src.utils import cleanup_staging_files, cleanup_temp_artifacts
    cleanup_staging_files()
    if config.get("render", {}).get("cleanup_batches_on_success", False):
        cleanup_temp_artifacts(clean_batches=True)
    return True


def _dump_perf(perf, monitor, date: str, cam_index: int, pipeline_duration: float, headline: dict | None = None, worker_stats: dict | None = None):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        perf_dir = LOGS_DIR / "perf"
        perf_dir.mkdir(parents=True, exist_ok=True)
        perf_path = perf_dir / f"perf_{date}_cam{cam_index}_{timestamp}.json"
        yolo_sum = perf.yolo_summary() if hasattr(perf, "yolo_summary") else {}
        wait_sum = perf.wait_summary() if hasattr(perf, "wait_summary") else {}
        metadata = {
            "date": date,
            "cam": cam_index,
            "pipeline_duration": round(pipeline_duration, 2),
            "monitor_summary": monitor.stages_data(),
            "perf_summary": perf.summary_by_stage(),
        }
        if worker_stats:
            metadata["worker_stats"] = worker_stats
        if yolo_sum:
            metadata["yolo_summary"] = yolo_sum
        if wait_sum:
            metadata["wait_summary"] = wait_sum
        if headline:
            metadata["headline"] = headline
        perf.dump(perf_path, metadata=metadata)
        perf.reset()
    except Exception:
        pass


def _build_headline(output_path: Path, total_input_dur: float, elapsed_wall: float) -> dict:
    """汇总单日头条指标：处理倍速、浓缩率、产出体积（性能评估的顶层仪表盘）。"""
    headline: dict = {
        "input_dur_s": round(total_input_dur, 1),
        "wall_s": round(elapsed_wall, 1),
        "speedup_x": round(total_input_dur / max(elapsed_wall, 0.1), 2),
    }
    try:
        from src.utils import size_metrics
        headline.update(size_metrics(output_path.stat().st_size))
    except OSError:
        pass
    try:
        from src.ffmpeg import get_duration
        out_dur = get_duration(str(output_path))
        if out_dur and out_dur > 0:
            headline["output_dur_s"] = round(out_dur, 1)
            headline["condensation_x"] = round(total_input_dur / out_dur, 1)
    except Exception:
        pass
    return headline


def _save_vlog_companion_assets(
    output_path: Path,
    date: str,
    cam_index: int,
    cam_display: str,
    total_files: int,
    total_input_dur: float,
    elapsed_wall: float,
    db: VlogDatabase,
    config: dict,
) -> None:
    """生成同名标准交付资产包：.srt 现实世界时间码字幕 + .meta.json 自描述结构化清单。"""
    try:
        from src.timeline import build_timeline_from_rows, save_timecode_subtitles, compute_display_plans
        rows = db.get_all_file_tasks_for_date(date, cam_index)
        full_timeline = build_timeline_from_rows(rows, date, config=config, resolve_presence=False)

        # 1. 生成伴随 .srt 字幕（默认开启，可在配置中显式关闭）
        srt_path = output_path.with_suffix(".srt")
        if config.get("render", {}).get("generate_subtitles", True):
            try:
                save_timecode_subtitles(
                    full_timeline,
                    srt_path,
                    rows=rows,
                    base_date=date,
                )
            except Exception as e:
                logger.warning("save_timecode_subtitles failed: %s", e)

        # 2. 生成伴随 .meta.json 结构化清单
        meta_path = output_path.with_suffix(".meta.json")
        vlog_size = output_path.stat().st_size if output_path.exists() else 0
        from src.ffmpeg import get_duration
        vlog_dur = get_duration(str(output_path)) or 0.0

        dyn_dur = sum(s.duration for s in full_timeline if getattr(s, "state", "") in ("DYNAMIC", "DYNAMIC_AUDIO"))
        sta_dur = sum(s.duration for s in full_timeline if getattr(s, "state", "") == "STATIC")

        # 提取动态高光片段（用于下游相册/Web 秒级定位）
        highlights = []
        cur_vlog_pos = 0.0
        seg_cfg = config.get("segment", {})
        render_cfg = config.get("render", {})
        presence_cfg = config.get("presence", {})
        micro_cfg = config.get("micro_motion", {})
        plans = compute_display_plans(
            full_timeline,
            static_keyframe_interval=seg_cfg.get("static_keyframe_interval", 30.0),
            keyframe_display_duration=seg_cfg.get("keyframe_display_duration", 0.5),
            min_static_display_duration=seg_cfg.get("min_static_display_duration", 1.5),
            max_static_display_duration=seg_cfg.get("max_static_display_duration", 2.0),
            speed_ramping=render_cfg.get("speed_ramping_enabled", True),
            ramp_duration_s=float(render_cfg.get("ramp_duration_s", 1.0)),
            presence_speed_factor=float(presence_cfg.get("speed_factor", 4.0)),
            night_stationary_speed_factor=float(presence_cfg.get("night_speed_factor", 16.0)),
            micro_motion_cruise_speed=float(micro_cfg.get("cruise_speed", 16.0)),
        )
        for seg, (disp_dur, _) in zip(full_timeline, plans):
            start_vlog = cur_vlog_pos
            end_vlog = cur_vlog_pos + disp_dur
            cur_vlog_pos = end_vlog
            if getattr(seg, "state", "") in ("DYNAMIC", "DYNAMIC_AUDIO"):
                highlights.append({
                    "vlog_start_s": round(start_vlog, 2),
                    "vlog_end_s": round(end_vlog, 2),
                    "vlog_duration_s": round(disp_dur, 2),
                    "state": seg.state,
                    "source_file": Path(getattr(seg, "filepath", getattr(seg, "source_file", ""))).name,
                    "max_energy": round(float(getattr(seg, "max_energy", 0.0) or 0.0), 1),
                    "avg_confidence": round(float(getattr(seg, "avg_confidence", 0.0) or 0.0), 2),
                })

        from src.scanner import camera_key
        first_fp = rows[0]["filepath"] if rows else ""
        cam_id = rows[0].get("camera_id") if rows and rows[0].get("camera_id") else (camera_key(first_fp, cam_index) if first_fp else f"cam_{cam_index}")

        from src.utils import size_metrics
        manifest_data = {
            "version": "1.0",
            "date": date,
            "camera": {
                "id": cam_id,
                "name": cam_display,
                "cam_index": cam_index,
            },
            "metrics": {
                "raw_duration_s": round(total_input_dur, 2),
                "vlog_duration_s": round(vlog_dur, 2),
                "condensation_ratio": round(total_input_dur / max(vlog_dur, 0.1), 2),
                "dynamic_duration_s": round(dyn_dur, 2),
                "static_duration_s": round(sta_dur, 2),
                "total_source_files": total_files,
                **size_metrics(vlog_size),
            },
            "timeline_highlights": highlights,
            "performance": {
                "wall_clock_s": round(elapsed_wall, 2),
                "speedup_x": round(total_input_dur / max(elapsed_wall, 0.1), 2),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        }
        meta_path.write_text(json.dumps(manifest_data, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Saved vlog companion assets: %s, %s", srt_path.name, meta_path.name)
    except Exception as e:
        logger.warning("Failed to save vlog companion assets for %s: %s", output_path.name, e)


def run_pipeline(skip_render: bool = False, input_dir: list[str] | None = None, dashboard_enabled: bool = True) -> dict:
    from src.utils import cleanup_temp_artifacts
    cleanup_temp_artifacts()
    db = VlogDatabase()
    monitor = get_monitor()
    monitor.start()
    try:
        scan_directory(db, input_dir=input_dir)
        groups = get_date_cam_groups(db)
        if not groups:
            return {"total": 0, "ok": 0, "failed": 0, "skipped": 0}

        # Rich 批量任务启动 Banner
        dates = sorted(d for d, _ in groups)
        cam_labels = sorted({f"Cam {c}" for _, c in groups})
        print_batch_startup_banner(
            total_groups=len(groups),
            date_range=(dates[0], dates[-1]),
            cameras=cam_labels,
            output_dir=str(OUTPUT_DIR),
        )

        ok, failed = 0, 0
        batch_summary_list = []
        for date, cam_index in groups:
            if not check_disk_space(OUTPUT_DIR, min_gb=20):
                break
            t0 = time.monotonic()
            success = False
            try:
                if process_date_cam(db, date, cam_index, skip_render=skip_render, dashboard_enabled=dashboard_enabled):
                    ok += 1
                    success = True
                else:
                    failed += 1
            except KeyboardInterrupt:
                from src.renderer import FFmpegProcessRegistry
                FFmpegProcessRegistry.mark_interrupted()
                FFmpegProcessRegistry.kill_all()
                logger.warning("run_pipeline interrupted by user (Ctrl+C). Halting batch pipeline.")
                raise
            except Exception:
                logger.exception("pipeline crash")
                failed += 1
            finally:
                wall_s = time.monotonic() - t0
                tasks = db.get_all_file_tasks_for_date(date, cam_index)
                sample_dir = str(Path(tasks[0]["filepath"]).parent) if tasks else ""
                from src.scanner import resolve_camera_identity, resolve_output_filename
                disp, _ = resolve_camera_identity(sample_dir, cam_index=cam_index, config=load_config())
                out_name = resolve_output_filename(
                    load_config().get("output", {}).get("naming", "DailyVlog_{date}_{mac}.mp4"),
                    date, cam_index, tasks[0]["filepath"] if tasks else None, config=load_config()
                )
                out_p = Path(OUTPUT_DIR) / out_name
                meta_p = out_p.with_suffix(".meta.json")
                item = {
                    "date": date,
                    "cam_index": cam_index,
                    "cam_name": disp,
                    "total_files": len(tasks),
                    "input_duration_s": sum(t.get("file_duration", 0.0) or 0.0 for t in tasks),
                    "vlog_duration_s": 0.0,
                    "output_size_mb": round(out_p.stat().st_size / (1024 * 1024), 2) if out_p.exists() else 0.0,
                    "wall_clock_s": round(wall_s, 2),
                    "status": "SUCCESS" if success else "FAILED",
                }
                if meta_p.exists():
                    try:
                        m_d = json.loads(meta_p.read_text(encoding="utf-8"))
                        m_m = m_d.get("metrics", {})
                        item["vlog_duration_s"] = m_m.get("vlog_duration_s", 0.0)
                        item["condensation_ratio"] = m_m.get("condensation_ratio", 0.0)
                        item["speedup_x"] = m_d.get("performance", {}).get("speedup_x", 0.0)
                    except Exception:
                        pass
                batch_summary_list.append(item)
            cleanup_resources()

        if len(groups) > 1:
            from src.ui import print_batch_summary_table
            print_batch_summary_table(batch_summary_list)

        return {"total": len(groups), "ok": ok, "failed": failed}
    finally:
        monitor.shutdown()
        db.close()


__all__ = [
    "StreamingOrchestrator",
    "WorkStealingManager",
    "process_date_cam",
    "run_pipeline",
]
