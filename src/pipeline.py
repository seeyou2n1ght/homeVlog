import logging
import time
import threading
import queue
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from src.database import VlogDatabase
from src.scanner import scan_directory, get_date_cam_groups
from src.prescreen import prescreen_file
from src.detector import MotionDetector
from src.renderer import build_batch_render, concat_output_files, _reindex_timeline
from src.scheduler import choose_encoder
from src.utils import load_config, OUTPUT_DIR, DB_PATH, LOGS_DIR, REPORTS_DIR, cleanup_resources, check_disk_space
from src.monitor import get_monitor, get_perf, PerfRecord
from src.timeline import build_timeline, TimelineSegment

logger = logging.getLogger("homevlog")

class StreamingOrchestrator:
    """终极流式管线编排器：实现预筛、分析、渲染的全重叠并发执行。"""
    
    def __init__(self, db: VlogDatabase, date: str, cam_index: int, config: dict):
        self.db = db
        self.date = date
        self.cam_index = cam_index
        self.config = config
        
        # 队列定义
        self.prescreen_queue = queue.Queue()
        self.analysis_queue = queue.Queue()
        self.render_batch_queue = queue.Queue()
        
        # 配置提取
        pipe_cfg = config.get("pipeline", {})
        det_cfg = config.get("detection", {})
        out_cfg = config.get("output", {})
        
        self.batch_max_files = out_cfg.get("batch_max_files", 8)
        self.render_delay = pipe_cfg.get("render_start_delay", 60)
        
        # 状态控制
        self.stop_event = threading.Event()
        self.batch_paths = []
        self.batch_lock = threading.Lock()

    def _prescreen_worker(self):
        """预筛 Worker：将扫描到的文件进行快速筛选。"""
        while not self.stop_event.is_set() or not self.prescreen_queue.empty():
            try:
                task = self.prescreen_queue.get(timeout=1)
            except queue.Empty:
                continue
                
            filepath = task["filepath"]
            duration = task.get("file_duration") or 300.0
            
            try:
                res = prescreen_file(filepath, duration, self.config)
                self.db.set_prescreen_result(filepath, res["status"], "")
                
                if res["status"] == "SUSPICIOUS":
                    self.analysis_queue.put(task)
                else:
                    # STATIC 文件也需要通知 Batcher，因为它可能包含在 Timeline 中
                    self.render_batch_queue.put({"filepath": filepath, "status": "STATIC"})
            except Exception:
                logger.exception("Streaming: prescreen failed for %s", filepath)
                self.db.set_prescreen_result(filepath, "FAILED", "")
            finally:
                self.prescreen_queue.task_done()

    def _analysis_worker(self, gpu: str):
        """分析 Worker：对 SUSPICIOUS 文件进行运动检测。"""
        detector = MotionDetector(self.config, decode_gpu=gpu)
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
                from src.segment import build_segments, segments_to_json
                
                file_start_ts = ts_to_unix(task["file_start_time"])
                file_start_offset = max(file_start_ts - ts_to_unix(self.date + "000000"), 0.0)
                
                labels = detector.analyze(filepath, start_offset=file_start_offset, 
                                        file_duration=task.get("file_duration") or 300.0)
                
                if labels:
                    segments = build_segments(
                        labels, filepath,
                        min_motion_dur=self.config.get("segment", {}).get("min_motion_duration", 1.0),
                        min_static_dur=self.config.get("segment", {}).get("min_static_duration", 30.0),
                        file_offset=file_start_offset,
                        gap_tolerance=self.config.get("segment", {}).get("gap_tolerance", 0.5),
                    )
                    js = segments_to_json(segments)
                    self.db.set_analysis_result(filepath, "ANALYZED", js)
                    
                    lp = detector.last_perf if hasattr(detector, "last_perf") else {}
                    perf.add(PerfRecord(
                        stage="analysis", file=Path(filepath).name, gpu=gpu,
                        duration=round(time.monotonic() - t0, 3), frames=lp.get("frames", 0),
                        extra={"status": "ANALYZED", **lp},
                    ))
                    # 通知 Batcher
                    self.render_batch_queue.put({"filepath": filepath, "status": "ANALYZED"})
                else:
                    self.db.set_analysis_result(filepath, "FAILED", "")
                    self.render_batch_queue.put({"filepath": filepath, "status": "FAILED"})
            except Exception:
                logger.exception("Streaming: analysis failed for %s [%s]", filepath, gpu)
                self.db.set_analysis_result(filepath, "FAILED", "")
                self.render_batch_queue.put({"filepath": filepath, "status": "FAILED"})
            finally:
                self.analysis_queue.task_done()

    def _render_manager(self):
        """渲染管理器：监听分析完成的文件，构建批次并启动渲染 Worker。"""
        out_cfg = self.config.get("output", {})
        fps = out_cfg.get("fps", 20)
        from src.utils import parse_res
        width, height = parse_res(out_cfg.get("resolution", "1920x1080"))
        seg_cfg = self.config.get("segment", {})
        audio_cfg = out_cfg.get("audio", {})
        
        pending_files = []
        batch_idx = 0
        
        # 等待一定时间，让流水线充盈
        time.sleep(5) 
        
        batch_queue = queue.Queue()
        
        def _render_worker(gpu: str):
            while True:
                item = batch_queue.get()
                if item is None:
                    break
                b_idx, files_to_batch = item
                try:
                    full_timeline = build_timeline(self.db, self.date, self.cam_index)
                    batch_segs = [s for s in full_timeline if s.filepath in files_to_batch]
                    
                    if not batch_segs:
                        continue
                    
                    res_path = build_batch_render(
                        batch_segs, b_idx, gpu, fps, width, height,
                        seg_cfg, out_cfg, audio_cfg, self.date, self.cam_index
                    )
                    if res_path:
                        with self.batch_lock:
                            self.batch_paths.append((b_idx, Path(res_path)))
                except Exception:
                    logger.exception("Streaming: render batch %d failed on %s", b_idx, gpu)
                finally:
                    batch_queue.task_done()

        render_threads = []
        for gpu in ["nv", "qsv"]:
            t = threading.Thread(target=_render_worker, args=(gpu,), daemon=True)
            t.start()
            render_threads.append(t)

        while not self.stop_event.is_set() or not self.render_batch_queue.empty():
            try:
                msg = self.render_batch_queue.get(timeout=2)
                pending_files.append(msg["filepath"])
                
                if len(pending_files) >= self.batch_max_files:
                    batch_queue.put((batch_idx, list(pending_files)))
                    pending_files = []
                    batch_idx += 1
                self.render_batch_queue.task_done()
            except queue.Empty:
                continue
        
        if pending_files:
            batch_queue.put((batch_idx, list(pending_files)))
            
        batch_queue.put(None)
        batch_queue.put(None)
            
        for t in render_threads:
            t.join()

    def run(self):
        t_start = time.monotonic()
        
        # 完美断点续传：将当天的所有任务按现有状态分发到正确的队列中
        all_tasks = self.db.get_all_file_tasks_for_date(self.date, self.cam_index)
        for task in all_tasks:
            pre_status = task["prescreen_status"]
            ana_status = task["analysis_status"]
            
            if pre_status == "PENDING":
                self.prescreen_queue.put(task)
            elif pre_status == "STATIC":
                self.render_batch_queue.put({"filepath": task["filepath"], "status": "STATIC"})
            elif pre_status == "SUSPICIOUS":
                if ana_status == "PENDING":
                    self.analysis_queue.put(task)
                else:
                    self.render_batch_queue.put({"filepath": task["filepath"], "status": ana_status})
            else:
                self.render_batch_queue.put({"filepath": task["filepath"], "status": pre_status})
        
        threads = []
        
        # 启动预筛 Worker
        prescreen_parallel = self.config.get("detection", {}).get("prescreen_parallel", 4)
        for _ in range(prescreen_parallel):
            t = threading.Thread(target=self._prescreen_worker, daemon=True)
            t.start()
            threads.append(t)
            
        # 启动分析 Worker
        analysis_max_workers = self.config.get("detection", {}).get("analysis_max_workers", 2)
        qsv_threshold = self.config.get("detection", {}).get("qsv_fallback_threshold", 50)
        
        if len(tasks) > qsv_threshold and analysis_max_workers >= 2:
            t_nv = threading.Thread(target=self._analysis_worker, args=("cuda",), daemon=True)
            t_nv.start()
            threads.append(t_nv)
            t_qsv = threading.Thread(target=self._analysis_worker, args=("qsv",), daemon=True)
            t_qsv.start()
            threads.append(t_qsv)
            for _ in range(analysis_max_workers - 2):
                t = threading.Thread(target=self._analysis_worker, args=("cuda",), daemon=True)
                t.start()
                threads.append(t)
        else:
            for _ in range(analysis_max_workers):
                t = threading.Thread(target=self._analysis_worker, args=("cuda",), daemon=True)
                t.start()
                threads.append(t)
        
        t_rm = threading.Thread(target=self._render_manager, daemon=True)
        t_rm.start()
        threads.append(t_rm)
        
        # 阻塞等待所有任务入队并处理
        self.prescreen_queue.join()
        logger.info("Streaming: Prescreen queue empty.")
        self.analysis_queue.join()
        logger.info("Streaming: Analysis queue empty.")
        self.render_batch_queue.join()
        logger.info("Streaming: Render batch queue empty.")
        
        # 通知各线程停止
        self.stop_event.set()
        for t in threads:
            t.join(timeout=3600) # 渲染可能很久
            
        with self.batch_lock:
            self.batch_paths.sort(key=lambda x: x[0])
            final_paths = [p for _, p in self.batch_paths]
            
        return final_paths

def _save_summary_report(db: VlogDatabase, date: str, cam_index: int):
    from src.timeline import build_timeline
    from src.utils import ts_to_unix
    timeline = build_timeline(db, date, cam_index)
    if not timeline:
        return
        
    total_duration = sum(seg.duration for seg in timeline)
    static_duration = sum(seg.duration for seg in timeline if seg.state == "STATIC")
    dynamic_duration = sum(seg.duration for seg in timeline if seg.state == "DYNAMIC")
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    report_file = REPORTS_DIR / f"summary_{date}_cam{cam_index}_{timestamp}.txt"
    
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(f"{'='*50}\n")
        f.write(f"Summary Report for {date} Camera {cam_index}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total Source Duration: {total_duration:.2f}s\n")
        f.write(f"Static Duration:       {static_duration:.2f}s\n")
        f.write(f"Dynamic Duration:      {dynamic_duration:.2f}s\n")
        f.write(f"\nSegment Details:\n")
        
        rows = db.get_all_file_tasks_for_date(date, cam_index)
        day_start = ts_to_unix(date + "000000")
        
        from src.segment import Segment, segments_from_json, merge_cross_file
        from src.utils import load_config
        config = load_config()
        
        all_segments: list[Segment] = []
        for row in rows:
            file_start_unix = ts_to_unix(row["file_start_time"])
            file_end_unix = ts_to_unix(row["file_end_time"])
            file_offset = file_start_unix - day_start
            file_end_offset = file_end_unix - day_start

            if row["prescreen_status"] == "STATIC":
                all_segments.append(Segment(
                    start_time=file_offset, end_time=file_end_offset,
                    state="STATIC", source_file=row["filepath"], file_start_offset=file_offset
                ))
            elif row["analysis_segments"]:
                segs = segments_from_json(row["analysis_segments"])
                for s in segs:
                    s.source_file = row["filepath"]
                all_segments.extend(segs)

        all_segments.sort(key=lambda s: s.start_time)
        seg_cfg = config.get("segment", {})
        gap_tolerance = seg_cfg.get("gap_tolerance", 0.5)
        merged = merge_cross_file(all_segments, gap_tolerance)
        
        def format_time(seconds: float) -> str:
            h = int(seconds // 3600)
            m = int((seconds % 3600) // 60)
            s = seconds % 60
            return f"{h:02d}:{m:02d}:{s:05.2f}"
            
        for s in merged:
            f.write(f"[{s.state:^7}] {format_time(s.start_time)} -> {format_time(s.end_time)} (dur: {s.end_time - s.start_time:.2f}s)\n")
        f.write(f"{'='*50}\n")

def process_date_cam(db: VlogDatabase, date: str, cam_index: int, skip_render: bool = False) -> bool:
    config = load_config()
    monitor = get_monitor()
    t_start = time.monotonic()
    
    if db.is_render_completed(date, cam_index):
        # 检查是否有新文件加入 (PENDING 状态的预筛或分析任务)
        pending_count = db.get_pending_file_count_for_date(date, cam_index)
        if pending_count == 0:
            logger.info("skip %s cam%d: already completed and no new files", date, cam_index)
            _save_summary_report(db, date, cam_index)
            return True
        else:
            logger.info("resuming %s cam%d: found %d new files despite previous completion", date, cam_index, pending_count)
            # 重置渲染状态，允许重新流式处理和渲染
            db.upsert_render_task(date, cam_index, "PENDING")

    logger.info("=== STREAMING pipeline %s cam%d start ===", date, cam_index)
    
    out_cfg = config.get("output", {})
    output_name = out_cfg.get("naming", "DailyVlog_{date}_cam{index}.mp4")
    output_name = output_name.replace("{date}", date).replace("{index}", str(cam_index))
    output_path = OUTPUT_DIR / output_name

    orchestrator = StreamingOrchestrator(db, date, cam_index, config)
    with monitor.stage(f"pipeline_{date}_cam{cam_index}"):
        batch_paths = orchestrator.run()
    
    if skip_render:
        _save_summary_report(db, date, cam_index)
        return True

    if not batch_paths:
        db.set_render_status(date, cam_index, "FAILED")
        return False

    db.upsert_render_task(date, cam_index, "RENDERING")
    if len(batch_paths) == 1:
        batch_paths[0].rename(output_path)
        ok = True
    else:
        ok = concat_output_files(batch_paths, output_path)
        for p in batch_paths: p.unlink(missing_ok=True)

    if ok:
        db.set_render_status(date, cam_index, "COMPLETED", output_file=str(output_path))
    else:
        db.set_render_status(date, cam_index, "FAILED")
        
    _dump_perf(get_perf(), monitor, date, cam_index, time.monotonic() - t_start)
    _save_summary_report(db, date, cam_index)
    return ok

def _dump_perf(perf, monitor, date: str, cam_index: int, pipeline_duration: float):
    try:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        perf_path = LOGS_DIR / f"perf_{date}_cam{cam_index}_{timestamp}.json"
        perf.dump(perf_path, metadata={
            "date": date, "cam": cam_index,
            "pipeline_duration": round(pipeline_duration, 2),
            "monitor_summary": monitor.stages_data(),
            "perf_summary": perf.summary_by_stage(),
        })
        perf.reset()
    except Exception: pass

def run_pipeline(skip_render: bool = False) -> dict:
    db = VlogDatabase()
    monitor = get_monitor()
    monitor.start()
    try:
        scan_directory(db)
        groups = get_date_cam_groups(db)
        if not groups: return {"total": 0, "ok": 0, "failed": 0, "skipped": 0}
        ok, failed = 0, 0
        for date, cam_index in groups:
            if not check_disk_space(OUTPUT_DIR, min_gb=20): break
            try:
                if process_date_cam(db, date, cam_index, skip_render=skip_render): ok += 1
                else: failed += 1
            except Exception:
                logger.exception("pipeline crash")
                failed += 1
            cleanup_resources()
        return {"total": len(groups), "ok": ok, "failed": failed}
    finally:
        monitor.shutdown()
        db.close()
