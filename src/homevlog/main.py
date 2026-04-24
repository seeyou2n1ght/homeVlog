"""
HomeVlog 主入口 (V2.3 三级异步流水线 Dashboard 版)

数据流:
  [Decoder Thread] ─帧队列→ [GPU Worker Thread] ─检测队列→ [Logic Thread]
  增加富状态跟踪、阶段耗时埋点和 SQLite 指标落地。
"""
import os
import queue
import time
import threading
import atexit
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional

from tqdm import tqdm

from .config import load_config
from .database.sqlite_manager import DatabaseManager
from .scanner import FileScanner
from .hal import get_detector
from .pipeline.decoder import VideoDecoder
from .pipeline.tracker import SimpleTracker
from .pipeline.aggregator import Aggregator
from .utils.ffmpeg_tools import FFmpegToolkit
from .utils.time_utils import parse_video_filename
from .utils.motion_analyzer import BlockMotionDetector


def process_pending_files(
    pending: List[str],
    config,
    db: DatabaseManager,
    detector,
    total_files: int,
) -> Dict[str, List[str]]:
    central_frame_queue: queue.Queue = queue.Queue(maxsize=128)
    result_queue: queue.Queue = queue.Queue(maxsize=128)
    cutter_queue: queue.Queue = queue.Queue(maxsize=32)
    
    stop_event = threading.Event()
    active_decoders: List[VideoDecoder] = []
    decoders_lock = threading.Lock()
    
    # 共享状态记录
    file_ids: Dict[str, int] = {}
    file_indices: Dict[str, int] = {}
    start_times: Dict[str, float] = {}
    phase_wait_ms: Dict[str, int] = defaultdict(int)
    phase_infer_ms: Dict[str, int] = defaultdict(int)
    queue_depths: Dict[str, List[int]] = defaultdict(list)
    
    def cleanup_resources():
        stop_event.set()
        with decoders_lock:
            for d in active_decoders:
                try: d.close()
                except: pass
            active_decoders.clear()
        for q in [central_frame_queue, result_queue, cutter_queue]:
            while not q.empty():
                try: q.get_nowait()
                except: break
                
    atexit.register(cleanup_resources)
    
    day_clips: Dict[str, List[str]] = defaultdict(list)
    day_clips_lock = threading.Lock()
    
    batch_size = config.hardware.batch_size
    hwaccel = "none" if config.hardware.use_mock else config.hardware.ffmpeg_hwaccel

    # ── Thread 1: Decoder Worker ──
    def decoder_worker() -> None:
        analyzers: Dict[str, BlockMotionDetector] = {}
        for idx, fp in enumerate(pending, 1):
            if stop_event.is_set(): break
            fname = os.path.basename(fp)
            file_indices[fp] = idx
            source_size = os.path.getsize(fp)
            print(f"[{idx}/{total_files}] ⏳ 正在处理: {fname} (大小: {source_size/1024/1024:.1f}MB)...")
            
            file_id = db.mark_file_started(fname, source_size)
            file_ids[fp] = file_id
            start_times[fp] = time.time()
            
            decoder = VideoDecoder(hwaccel=hwaccel, io_timeout=config.cutting.io_timeout_sec)
            with decoders_lock:
                active_decoders.append(decoder)
                
            if config.detection.enable_pixel_motion:
                analyzers[fp] = BlockMotionDetector(
                    block_size=config.detection.pixel_motion_grid,
                    history_size=config.detection.pixel_motion_history_size,
                    threshold_factor=config.detection.pixel_motion_threshold_factor
                )
                
            try:
                # 兼容旧代码里可能有的 _scan_pts
                pts_list = None
                if hasattr(decoder, '_scan_pts'):
                    pts_list = decoder._scan_pts(fp)
                    decoder.start(
                        fp,
                        pts_list=pts_list,
                        fps=config.detection.fps,
                        infer_resolution=config.detection.infer_resolution,
                    )
                else:
                    decoder.start(
                        fp,
                        fps=config.detection.fps,
                        infer_resolution=config.detection.infer_resolution,
                    )
                frame_count = 0
                while not stop_event.is_set():
                    data = decoder.get_frame()
                    if data is None:
                        break
                    pts, frame = data
                    
                    pm_flag = None
                    if config.detection.enable_pixel_motion and fp in analyzers:
                        pm_flag = analyzers[fp].analyze(frame)
                        
                    central_frame_queue.put((fp, pts, frame, pm_flag))
                    frame_count += 1
            finally:
                decoder.close()
                with decoders_lock:
                    if decoder in active_decoders:
                        active_decoders.remove(decoder)
                if fp in analyzers:
                    del analyzers[fp]
            central_frame_queue.put((fp, None, None, None))
            
        central_frame_queue.put((None, None, None, None))

    # ── Thread 2: GPU Worker ──
    def gpu_worker() -> None:
        processed_frames = 0
        try:
            batch_frames: List = []
            batch_meta: List = []  
            current_fp = None
            
            while True:
                wait_start = time.time()
                fp, pts, frame, pm_flag = central_frame_queue.get()
                wait_end = time.time()
                
                if fp is not None:
                    phase_wait_ms[fp] += int((wait_end - wait_start) * 1000)
                    queue_depths[fp].append(central_frame_queue.qsize())
                    current_fp = fp
                
                if fp is None:  
                    if batch_frames:
                        infer_start = time.time()
                        results = detector.infer_batch(batch_frames)
                        infer_time = int((time.time() - infer_start) * 1000)
                        if current_fp:
                            phase_infer_ms[current_fp] += infer_time
                        for m, d in zip(batch_meta, results):
                            result_queue.put((m[0], m[1], d, m[2]))
                    break
                    
                if pts is None:  
                    if batch_frames:
                        infer_start = time.time()
                        results = detector.infer_batch(batch_frames)
                        infer_time = int((time.time() - infer_start) * 1000)
                        phase_infer_ms[fp] += infer_time
                        
                        for m, d in zip(batch_meta, results):
                            result_queue.put((m[0], m[1], d, m[2]))
                        processed_frames += len(batch_frames)
                        batch_frames = []
                        batch_meta = []
                    result_queue.put((fp, None, None, None))
                    continue
                    
                batch_frames.append(frame)
                batch_meta.append((fp, pts, pm_flag))
                
                if len(batch_frames) >= batch_size:
                    infer_start = time.time()
                    results = detector.infer_batch(batch_frames)
                    infer_time = int((time.time() - infer_start) * 1000)
                    phase_infer_ms[fp] += infer_time
                    
                    for m, d in zip(batch_meta, results):
                        result_queue.put((m[0], m[1], d, m[2]))
                    processed_frames += len(batch_frames)
                    batch_frames = []
                    batch_meta = []
        except Exception as e:
            print(f"  [GPU Worker] ❌ Error: {e}")
        finally:
            result_queue.put((None, None, None, None))

    # ── Thread 3: Logic Worker ──
    def logic_worker() -> None:
        trackers = {}
        aggregators = {}
        frame_counts = defaultdict(int)
        start_times = {}
        
        try:
            while True:
                fp, pts, detections, pixel_motion = result_queue.get()
                if fp is None:  
                    break
                    
                if fp not in trackers:
                    trackers[fp] = SimpleTracker(
                        iou_threshold=config.detection.iou_threshold,
                        debounce_frames=config.detection.debounce_frames,
                    )
                    aggregators[fp] = Aggregator(
                        static_interval_ms=config.detection.static_interval_ms,
                        padding_ms=config.cutting.keyframe_padding_ms,
                    )
                    start_times[fp] = time.time()
                    
                if pts is None:  
                    cut_segments, events = aggregators[fp].finalize()
                    cutter_queue.put((fp, cut_segments, events, start_times[fp], frame_counts[fp]))
                    
                    del trackers[fp]
                    del aggregators[fp]
                    del frame_counts[fp]
                    del start_times[fp]
                    continue
                    
                # 富状态更新
                state = trackers[fp].update(detections, pixel_motion)
                aggregators[fp].add_frame_state(pts, state)
                frame_counts[fp] += 1
                
        except Exception as e:
            print(f"  [Logic Worker] ❌ Error: {e}")
        finally:
            cutter_queue.put(None)

    # ── Thread 4: Cutter Worker ──
    def cutter_worker() -> None:
        while True:
            job = cutter_queue.get()
            if job is None:
                break
            fp, cut_segments, events, start_time, frame_count = job
            clips = []
            fname = os.path.basename(fp)
            file_id = file_ids.get(fp, -1)
            
            try:
                if stop_event.is_set(): continue
                
                io_cut_start = time.time()
                if cut_segments:
                    temp_dir = os.path.join(config.paths.local_output, "temp_clips")
                    clips = FFmpegToolkit.cut_segments(
                        fp, cut_segments, temp_dir,
                        parallel_jobs=config.cutting.parallel_jobs
                    )
                io_cut_ms = int((time.time() - io_cut_start) * 1000)
                
                info = parse_video_filename(fp)
                date_key = info.date_str if info else "unknown"
                with day_clips_lock:
                    day_clips[date_key].extend(clips)
                    
                elapsed = max(time.time() - start_time, 0.001)
                
                output_size = sum(os.path.getsize(c) for c in clips if os.path.exists(c))
                
                depths = queue_depths.get(fp, [0])
                avg_depth = sum(depths) / len(depths) if depths else 0.0
                
                # 记录 V2.3 指标
                if file_id != -1:
                    db.log_analytics_events(file_id, events)
                    db.log_performance(
                        file_id=file_id,
                        total_time_ms=int(elapsed * 1000),
                        phase_wait_ms=phase_wait_ms.get(fp, 0),
                        phase_infer_ms=phase_infer_ms.get(fp, 0),
                        phase_io_cut_ms=io_cut_ms,
                        decode_fps=frame_count / elapsed,
                        infer_fps=frame_count / elapsed,
                        avg_queue_depth=avg_depth
                    )
                    db.mark_file_completed(file_id, output_size)
                    
                idx = file_indices.get(fp, "?")
                print(f"[{idx}/{total_files}] ✅ 完成: {fname} | 耗时: {elapsed:.1f}s | 平均速度: {frame_count/elapsed:.1f} FPS | 产出片段: {len(clips)} 段")
            except Exception as e:
                idx = file_indices.get(fp, "?")
                print(f"[{idx}/{total_files}] ❌ 失败: {fname} | 错误: {e}")
                db.mark_file_failed(fname)

    # ── 启动所有流水线线程 ──
    t_decoder = threading.Thread(target=decoder_worker, name="decoder-worker", daemon=True)
    t_gpu = threading.Thread(target=gpu_worker, name="gpu-worker", daemon=True)
    t_logic = threading.Thread(target=logic_worker, name="logic-worker", daemon=True)
    t_cutter = threading.Thread(target=cutter_worker, name="cutter-worker", daemon=True)
    
    t_decoder.start()
    t_gpu.start()
    t_logic.start()
    t_cutter.start()
    
    try:
        while t_cutter.is_alive():
            t_cutter.join(timeout=1.0)
    except KeyboardInterrupt:
        print("\n[Main] Interrupted by user. Shutting down pipeline...")
        cleanup_resources()
    finally:
        cleanup_resources()
        atexit.unregister(cleanup_resources)
        
    return dict(day_clips)


def main() -> None:
    config = load_config()
    Path(config.paths.local_output).mkdir(parents=True, exist_ok=True)

    db = DatabaseManager(config.paths.sqlite_db)
    scanner = FileScanner(config.scanner, config.paths.nas_input)

    detector = get_detector(
        use_mock=config.hardware.use_mock,
        gpu_id=config.hardware.gpu_id
    )
    detector.load_model(config.detection.model_path)

    print("=== HomeVlog Pipeline V2.4 Heterogeneous Edition ===")
    print(f"[Main] Input : {config.paths.nas_input}")
    print(f"[Main] Output: {config.paths.local_output}")

    try:
        print("\n[Scanner] Discovering files...")
        scanner.scan_once()
        time.sleep(config.scanner.write_detect_interval_sec)
        ready = scanner.scan_once()

        pending = [f for f in ready if not db.is_file_processed(os.path.basename(f))]
        if not pending:
            print("[Main] No new files. Exiting.")
            return
        print(f"[Main] {len(pending)} file(s) to process.\n")

        # 这里就不记 processed_days 了，保留精简
        day_clips = process_pending_files(sorted(pending), config, db, detector, len(pending))

        print(f"\n{'='*50}")
        print("[Main] Merging daily vlogs...")
        for date_key in sorted(day_clips.keys()):
            clips = day_clips[date_key]
            if not clips:
                continue
            output_name = f"vlog_{date_key}.mp4"
            final_path = str(Path(config.paths.local_output) / output_name)
            print(f"  [{date_key}] Merging {len(clips)} clip(s) → {output_name}")
            FFmpegToolkit.merge_videos(clips, final_path)
            FFmpegToolkit.cleanup(clips)
            print(f"  [{date_key}] ✅ Saved: {final_path}")

    except KeyboardInterrupt:
        print("\n[Main] Shutdown by user.")
    finally:
        detector.release()
        print("\n=== HomeVlog Offline ===")

if __name__ == "__main__":
    main()
