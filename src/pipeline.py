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
from src.detector import MotionDetector
from src.renderer import build_batch_render, concat_output_files
from src.utils import (
    load_config,
    OUTPUT_DIR,
    LOGS_DIR,
    cleanup_resources,
    check_disk_space,
    register_dashboard,
    unregister_dashboard,
    WorkStealingManager,
)
from src.monitor import get_monitor, get_perf, PerfRecord
from src.timeline import build_timeline

logger = logging.getLogger("homevlog")


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
        self.work_stealing.enable_cold_start_burst()


        # 队列定义
        self.prescreen_queue = queue.Queue()
        self.analysis_queue = queue.Queue()
        self.render_batch_queue = queue.Queue()

        # 配置提取
        pipe_cfg = config.get("pipeline", {})
        render_cfg = config.get("render", {})
        
        self.batch_max_files = render_cfg.get("batch_max_files", 4)
        self.render_delay = pipe_cfg.get("render_start_delay", 5)

        # 状态控制
        self.stop_event = threading.Event()
        self.batch_paths = []
        self.batch_lock = threading.Lock()
        self.error_lock = threading.Lock()
        self.errors: list[str] = []

        # Rich 仪表盘 (tqdm 已下线，统一由 PipelineDashboard 呈现)
        self.dashboard_enabled = dashboard_enabled
        self.dashboard: PipelineDashboard | None = None
        self._prescreen_t0 = 0.0
        self._analysis_t0 = 0.0

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

    def _prescreen_worker(self, gpu: str = "qsv"):
        """预筛 Worker：将扫描到的文件进行快速筛选。"""
        while not self.stop_event.is_set() or not self.prescreen_queue.empty():
            try:
                task = self.prescreen_queue.get(timeout=1)
            except queue.Empty:
                continue

            filepath = task["filepath"]
            duration = task.get("file_duration") or 300.0
            t0 = time.monotonic()

            try:
                res = prescreen_file(filepath, duration, self.config, gpu=gpu)
                result_json = res.get("result_json", "")
                self.db.set_prescreen_result(filepath, res["status"], result_json)

                extra = {"status": res["status"]}
                try:
                    extra.update(json.loads(result_json or "{}"))
                except (json.JSONDecodeError, TypeError):
                    if res.get("error"):
                        extra["error"] = res["error"]
                get_perf().add(
                    PerfRecord(
                        stage="prescreen",
                        file=Path(filepath).name,
                        gpu=gpu,
                        duration=round(time.monotonic() - t0, 3),
                        extra=extra,
                    )
                )

                if res["status"] == "SUSPICIOUS":
                    self.analysis_queue.put(task)
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
                get_perf().add(
                    PerfRecord(
                        stage="prescreen",
                        file=Path(filepath).name,
                        gpu=gpu,
                        duration=round(time.monotonic() - t0, 3),
                        extra={"status": "ERROR"},
                    )
                )
            finally:
                self.prescreen_queue.task_done()

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
    ):
        from src.segment import build_segments, segments_to_json

        labels, yolo_buffer = detector.analyze(
            filepath,
            start_offset=file_start_offset,
            file_duration=task.get("file_duration") or 300.0,
        )
        
        if hasattr(detector, 'has_audio_detected'):
            self.db.set_file_metadata(filepath, detector.has_audio_detected)

        if labels:
            seg_cfg = self.config.get("segment", {})
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
            )
            
            yolo_before = len(segments)
            is_yolo_active = False
            if yolo_verifier:
                is_yolo_active = True
                yolo_device = self.config.get("hardware", {}).get("device", "cpu")
                segments = yolo_verifier.verify(
                    filepath, segments, gpu=gpu, device=yolo_device, frames_buffer=yolo_buffer, analysis_fps=detector.fps
                )
            yolo_after = len(segments)

            js = segments_to_json(segments)
            self.db.set_analysis_result(filepath, "ANALYZED", js)

            lp = detector.last_perf if hasattr(detector, "last_perf") else {}
            perf.add(
                PerfRecord(
                    stage="analysis",
                    file=Path(filepath).name,
                    gpu=gpu,
                    duration=round(time.monotonic() - t0, 3),
                    frames=lp.get("frames", 0),
                    extra={
                        "status": "ANALYZED",
                        "segments_before_yolo": yolo_before,
                        "segments_after_yolo": yolo_after,
                        "yolo_streaming_verify": is_yolo_active,
                        **lp,
                    },
                )
            )
            self.render_batch_queue.put(
                {"filepath": filepath, "status": "ANALYZED"}
            )
        else:
            self.db.set_analysis_result(filepath, "FAILED", "")
            self._add_error(f"analysis produced no labels: {Path(filepath).name}")
            self.render_batch_queue.put(
                {"filepath": filepath, "status": "FAILED"}
            )

        if self.dashboard is not None:
            el_s = max(0.1, time.monotonic() - self._analysis_t0)
            n_done = self.dashboard.analysis_done + 1
            task_el = max(0.1, time.monotonic() - t0)
            frames_done = getattr(detector, "last_perf", {}).get("frames", 0)
            fps_val = frames_done / task_el if frames_done else 0.0
            speed_str = f"{fps_val:.1f} fps" if fps_val > 0 else f"均速 {el_s / n_done:.1f}s/个"
            self.dashboard.update_analysis(
                completed=n_done,
                latest_file=Path(filepath).name,
                speed_str=f"{speed_str} ({gpu.upper()})",
            )
            self._sync_queue_levels()

    def _analysis_worker(self, fixed_gpu: str | None = None):
        """分析 Worker：对 SUSPICIOUS 文件进行运动检测与目标验证 (支持异构自适应工作窃取)。"""
        detectors: dict[str, MotionDetector] = {
            "qsv": MotionDetector(self.config, decode_gpu="qsv"),
        }
        detector_cuda: MotionDetector | None = None
        
        # [性能极限] 在线程生命周期内复用验证器实例，避免重复初始化开销
        yolo_verifier = None
        yolo_cfg = self.config.get("yolo", {})
        if yolo_cfg.get("enabled", False) and yolo_cfg.get("streaming_verify", False):
            from src.yolo_verifier import YoloVerifier
            yolo_verifier = YoloVerifier(self.config)
            
        perf = get_perf()

        while not self.stop_event.is_set() or not self.analysis_queue.empty():
            try:
                task = self.analysis_queue.get(timeout=1)
            except queue.Empty:
                continue

            filepath = task["filepath"]
            t0 = time.monotonic()

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
                        task, filepath, file_start_offset, detector, yolo_verifier, gpu, perf, t0
                    )
                else:
                    q_size = self.analysis_queue.qsize()
                    with self.work_stealing.lease_device(q_size) as gpu:
                        if gpu == "cuda":
                            if detector_cuda is None:
                                detector_cuda = MotionDetector(self.config, decode_gpu="cuda")
                            detector = detector_cuda
                        else:
                            detector = detectors["qsv"]
                        self._execute_analysis_task(
                            task, filepath, file_start_offset, detector, yolo_verifier, gpu, perf, t0
                        )

            except Exception:
                logger.exception("Streaming: analysis failed for %s", filepath)
                self.db.set_analysis_result(filepath, "FAILED", "")
                self._add_error(f"analysis failed: {Path(filepath).name}")
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

        # ---- In-Order Sliding Window：按物理录制时间严格保序的批次调度 ----
        # 乱序到达的完成消息（STATIC/ANALYZED/FAILED）经滑动窗口重排：
        # 仅当队首（最早录制）文件就绪时窗口才向前推进，确保跨批次时间轴 100% 单调。
        all_rows = self.db.get_all_file_tasks_for_date(self.date, self.cam_index)
        ordered_files: list[str] = [
            r["filepath"]
            for r in sorted(all_rows, key=lambda r: r.get("file_start_time") or "")
        ]
        ordered_set = set(ordered_files)
        ready_status: dict[str, str] = {}
        head = 0

        pending_files: list[str] = []
        batch_idx = 0
        time.sleep(max(0, self.render_delay))
        heavy_queue = queue.Queue()
        light_queue = queue.Queue()
        all_dispatched_event = threading.Event()
        render_start_t: list[float] = []

        def _is_heavy_batch(files_to_batch: list[str]) -> bool:
            """检查批次中是否包含需复杂处理的 DYNAMIC 动作片段。"""
            try:
                rows = self.db.get_all_file_tasks_for_date(self.date, self.cam_index)
                rows_map = {r["filepath"]: r for r in rows}
                for fp in files_to_batch:
                    r = rows_map.get(fp)
                    if not r:
                        continue
                    if r.get("prescreen_status") == "SUSPICIOUS":
                        raw_segs = r.get("analysis_segments")
                        if raw_segs and ("DYNAMIC" in raw_segs or "DYNAMIC_AUDIO" in raw_segs):
                            return True
            except Exception:
                pass
            return False

        def _render_worker(gpu: str):
            while True:
                item = None
                if gpu == "nv":
                    # NVENC 优先取 heavy 任务，其次协助消费 light 任务
                    try:
                        item = heavy_queue.get_nowait()
                    except queue.Empty:
                        try:
                            item = light_queue.get(timeout=0.5)
                        except queue.Empty:
                            if all_dispatched_event.is_set() and heavy_queue.empty() and light_queue.empty():
                                break
                            continue
                elif gpu == "qsv":
                    # QSV 严禁处理 heavy 任务，仅协助处理 light 任务
                    # 收尾防拖尾保护：当分发完毕且剩余轻任务 <= 1 时，QSV 主动退出让位 NVENC
                    if all_dispatched_event.is_set() and light_queue.qsize() <= 1:
                        break
                    try:
                        item = light_queue.get(timeout=0.5)
                    except queue.Empty:
                        if all_dispatched_event.is_set() and light_queue.empty():
                            break
                        continue

                if item is None:
                    break

                b_idx, files_to_batch = item
                if not render_start_t:
                    render_start_t.append(time.monotonic())
                t_r0 = time.monotonic()
                if gpu == "nv":
                    self.work_stealing.register_render_start()
                try:
                    all_rows = self.db.get_all_file_tasks_for_date(self.date, self.cam_index)
                    from src.timeline import build_timeline_from_rows
                    batch_segs = build_timeline_from_rows(
                        all_rows, self.date, target_files=files_to_batch, config=self.config
                    )

                    if not batch_segs:
                        logger.warning(f"render batch {b_idx} has no timeline segments, skipping")
                        continue

                    res_path = build_batch_render(
                        batch_segs, b_idx, gpu, fps, width, height,
                        seg_cfg, out_cfg, audio_cfg, self.date, self.cam_index,
                        all_rows
                    )
                    r_dur = round(time.monotonic() - t_r0, 3)
                    if res_path:
                        with self.batch_lock:
                            self.batch_paths.append((b_idx, Path(res_path)))
                        get_perf().add(
                            PerfRecord(
                                stage="render",
                                file=f"batch_{b_idx}.mp4",
                                gpu=gpu,
                                duration=r_dur,
                                extra={"batch_files": len(files_to_batch), "segments": len(batch_segs)},
                            )
                        )
                        if self.dashboard is not None:
                            # 精准渲染阶段自身计时与剩余预估
                            t_start = render_start_t[0] if render_start_t else t_r0
                            el_sec = max(0.1, time.monotonic() - t_start)
                            n_done = self.dashboard.render_done + 1
                            n_total = max(n_done, self.dashboard.render_total)
                            avg_s = el_sec / max(1, n_done)
                            self.dashboard.update_render(
                                completed=n_done,
                                latest_batch=f"Batch {b_idx} on {gpu.upper()}",
                                speed_str=f"均速 {avg_s:.1f}s/批",
                            )
                    else:
                        if gpu == "qsv":
                            # QSV 编码失败或异常，自动 failover 回退到 NVENC 重试
                            logger.warning("Streaming: QSV batch %d failed, falling back to NVENC", b_idx)
                            heavy_queue.put((b_idx, files_to_batch))
                        else:
                            self._add_error(f"render batch {b_idx} returned no output")
                except Exception:
                    logger.exception("Streaming: render batch %d failed on %s", b_idx, gpu)
                    if gpu == "qsv":
                        heavy_queue.put((b_idx, files_to_batch))
                    else:
                        self._add_error(f"render batch {b_idx} failed on {gpu}")
                finally:
                    if gpu == "nv":
                        self.work_stealing.register_render_end()

        # 启动 2 个 NVENC 主力 Worker (压榨 3060Ti 双编引擎，UHD 770 专职 100% 解码)
        render_threads = []
        render_gpus = ["nv", "nv"]
        for gpu in render_gpus:
            t = threading.Thread(target=_render_worker, args=(gpu,), daemon=True)
            t.start()
            render_threads.append(t)


        def _enqueue_batch(b_idx: int, files: list[str]):
            if _is_heavy_batch(files):
                heavy_queue.put((b_idx, files))
            else:
                light_queue.put((b_idx, files))
            if self.dashboard is not None:
                self.dashboard.update_render(
                    completed=self.dashboard.render_done,
                    total=self.dashboard.render_total + 1,
                )

        while not self.stop_event.is_set() or not self.render_batch_queue.empty():
            try:
                msg = self.render_batch_queue.get(timeout=2)
            except queue.Empty:
                continue

            filepath = msg.get("filepath")
            if filepath is not None:
                status = msg.get("status", "FAILED")
                if filepath in ordered_set:
                    # 记录就绪状态（FAILED 同样计入，用于推进窗口指针）
                    ready_status[filepath] = status
                elif status != "FAILED":
                    # 不在有序队列中的文件（DB 与消息不一致的兜底）直接追加，避免永久滞留
                    pending_files.append(filepath)

                # 推进保序滑动窗口：仅当队首文件已就绪时向前推进
                while head < len(ordered_files) and ordered_files[head] in ready_status:
                    fp_head = ordered_files[head]
                    # FAILED 文件跳过入批，但指针必须推进
                    if ready_status[fp_head] != "FAILED":
                        pending_files.append(fp_head)
                    head += 1

                # 窗口推进可能一次放入多个文件，循环按 batch_max_files 切分
                while len(pending_files) >= self.batch_max_files:
                    _enqueue_batch(batch_idx, list(pending_files[: self.batch_max_files]))
                    pending_files = pending_files[self.batch_max_files :]
                    batch_idx += 1
            self.render_batch_queue.task_done()

        # 消费循环结束：flush 窗口中剩余的就绪文件（含 head 未达队尾的兜底场景）
        if pending_files:
            _enqueue_batch(batch_idx, list(pending_files))
            pending_files = []

        all_dispatched_event.set()

        for _ in render_threads:
            heavy_queue.put(None)
            light_queue.put(None)
        for t in render_threads:
            t.join()


    def run(self):
        all_tasks = self.db.get_all_file_tasks_for_date(self.date, self.cam_index)
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

        for task in all_tasks:
            if task["prescreen_status"] == "PENDING":
                self.prescreen_queue.put(task)
            elif task["prescreen_status"] == "STATIC":
                self.render_batch_queue.put({"filepath": task["filepath"], "status": "STATIC"})
            elif task["prescreen_status"] == "SUSPICIOUS":
                if task["analysis_status"] == "PENDING":
                    self.analysis_queue.put(task)
                else:
                    self.render_batch_queue.put({"filepath": task["filepath"], "status": task["analysis_status"]})
            else:
                self.render_batch_queue.put({"filepath": task["filepath"], "status": "FAILED"})

        threads = []
        prescreen_parallel = self.config.get("detection", {}).get("prescreen_parallel", 8)
        prescreen_gpu_policy = self.config.get("pipeline", {}).get("prescreen_gpu_policy", "qsv_only")
        for i in range(prescreen_parallel):
            gpu = "qsv"
            if prescreen_gpu_policy == "alternating":
                gpu = "qsv" if i % 2 == 0 else "cuda"
            elif prescreen_gpu_policy == "cuda_only":
                gpu = "cuda"
            t = threading.Thread(target=self._prescreen_worker, args=(gpu,), daemon=True)
            t.start()
            threads.append(t)

        analysis_max_workers = self.config.get("detection", {}).get("analysis_max_workers", 8)
        for _ in range(analysis_max_workers):
            t = threading.Thread(target=self._analysis_worker, daemon=True)
            t.start()
            threads.append(t)

        if self.render_enabled:
            t_rm = threading.Thread(target=self._render_manager, daemon=True)
            t_rm.start()
            threads.append(t_rm)

        self.prescreen_queue.join()
        self.analysis_queue.join()
        if self.render_enabled:
            self.render_batch_queue.join()

        self.stop_event.set()
        for t in threads:
            t.join(timeout=3600)

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


def process_date_cam(db: VlogDatabase, date: str, cam_index: int, skip_render: bool = False, dashboard_enabled: bool = True) -> bool:
    config = load_config()
    monitor = get_monitor()
    t_start = time.monotonic()
    if db.is_render_completed(date, cam_index):
        pending_count = db.get_pending_file_count_for_date(date, cam_index)
        if pending_count == 0:
            logger.info("skip %s cam%d: already completed", date, cam_index)
            return True
        db.upsert_render_task(date, cam_index, "PENDING")

    all_tasks = db.get_all_file_tasks_for_date(date, cam_index)
    total_files = len(all_tasks)
    total_input_dur = sum(float(t.get("file_duration") or 300.0) for t in all_tasks)

    # Rich 启动 Banner (含机位别名解析)
    out_cfg = config.get("output", {})
    output_name = out_cfg.get("naming", "DailyVlog_{date}_cam{index}.mp4").replace("{date}", date).replace("{index}", str(cam_index))
    cam_display = None
    try:
        from src.scanner import resolve_camera_identity
        sample_dir = str(Path(all_tasks[0]["filepath"]).parent) if all_tasks else ""
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

    db.upsert_render_task(date, cam_index, "RENDERING")
    try:
        if len(batch_paths) == 1:
            batch_paths[0].rename(output_path)
            ok = True
        else:
            ok = concat_output_files(batch_paths, output_path)
            for p in batch_paths:
                p.unlink(missing_ok=True)
    except Exception:
        logger.exception("finalize render output failed")
        db.set_render_status(date, cam_index, "FAILED")
        return False

    elapsed_wall = time.monotonic() - t_start
    # 运行期间的告警与异常汇总 (含管线内部错误)
    with orchestrator.error_lock:
        run_errors = list(orchestrator.errors)
    if run_errors:
        print_error_summary(run_errors)

    if ok:
        db.set_render_status(date, cam_index, "COMPLETED", output_file=str(output_path))
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
    else:
        db.set_render_status(date, cam_index, "FAILED")

    _dump_perf(get_perf(), monitor, date, cam_index, elapsed_wall)
    return ok


def _dump_perf(perf, monitor, date: str, cam_index: int, pipeline_duration: float):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        perf_path = LOGS_DIR / f"perf_{date}_cam{cam_index}_{timestamp}.json"
        perf.dump(perf_path, metadata={"date": date, "cam": cam_index, "pipeline_duration": round(pipeline_duration, 2), "monitor_summary": monitor.stages_data(), "perf_summary": perf.summary_by_stage()})
        perf.reset()
    except Exception:
        pass


def run_pipeline(skip_render: bool = False, input_dir: str | None = None, dashboard_enabled: bool = True) -> dict:
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
        for date, cam_index in groups:
            if not check_disk_space(OUTPUT_DIR, min_gb=20):
                break
            try:
                if process_date_cam(db, date, cam_index, skip_render=skip_render, dashboard_enabled=dashboard_enabled):
                    ok += 1
                else:
                    failed += 1
            except Exception:
                logger.exception("pipeline crash")
                failed += 1
            cleanup_resources()
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

