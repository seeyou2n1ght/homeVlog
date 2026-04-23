"""
HomeVlog 主入口 (V2.3 三级异步流水线版)

数据流:
  [Decoder Thread] ─帧队列→ [GPU Worker Thread] ─检测队列→ [Logic Thread]
  三个硬件单元（NVDEC/QSV, CUDA, CPU）同时工作，零空闲。
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


def process_pending_files(
    pending: List[str],
    config,
    db: DatabaseManager,
    detector,
) -> Dict[str, List[str]]:
    """
    四级异步全局流水线处理所有视频：
      [DecoderWorker] → central_frame_queue → [GPUWorker] → result_queue → [LogicWorker] → cutter_queue → [CutterWorker]
    返回按天分组的 clip 路径列表。
    """
    central_frame_queue: queue.Queue = queue.Queue(maxsize=128)
    result_queue: queue.Queue = queue.Queue(maxsize=128)
    cutter_queue: queue.Queue = queue.Queue(maxsize=32)
    
    # 停止信号与资源管理器
    stop_event = threading.Event()
    active_decoders: List[VideoDecoder] = []
    decoders_lock = threading.Lock()
    
    def cleanup_resources():
        """强制清理所有可能运行的子进程"""
        stop_event.set()
        with decoders_lock:
            for d in active_decoders:
                try: d.close()
                except: pass
            active_decoders.clear()
        # 尽量排空队列
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
        """连续遍历文件，喂给 central_frame_queue"""
        for fp in pending:
            if stop_event.is_set(): break
            fname = os.path.basename(fp)
            print(f"\n[Worker] Processing: {fname}")
            db.mark_file_started(fp)
            
            decoder = VideoDecoder(hwaccel=hwaccel, io_timeout=config.cutting.io_timeout_sec)
            with decoders_lock:
                active_decoders.append(decoder)
                
            try:
                pts_list = decoder._scan_pts(fp)
                decoder.start(
                    fp,
                    pts_list=pts_list,
                    fps=config.detection.fps,
                    infer_resolution=config.detection.infer_resolution,
                )
                frame_count = 0
                while not stop_event.is_set():
                    data = decoder.get_frame()
                    if data is None:
                        break
                    pts, frame = data
                    central_frame_queue.put((fp, pts, frame))
                    frame_count += 1
            finally:
                decoder.close()
                with decoders_lock:
                    if decoder in active_decoders:
                        active_decoders.remove(decoder)
            # 发送单个文件结束符
            central_frame_queue.put((fp, None, None))
            
        # 发送全局结束符
        central_frame_queue.put((None, None, None))

    # ── Thread 2: GPU Worker ──
    def gpu_worker() -> None:
        """从 central 取帧 → 攒 batch → 推理 → result_queue"""
        processed_frames = 0
        try:
            batch_frames: List = []
            batch_meta: List = []  # (fp, pts)
            
            while True:
                fp, pts, frame = central_frame_queue.get()
                
                if fp is None:  # 全局结束
                    if batch_frames:
                        results = detector.infer_batch(batch_frames)
                        for m, d in zip(batch_meta, results):
                            result_queue.put((m[0], m[1], d))
                    break
                    
                if pts is None:  # 单文件结束
                    if batch_frames:
                        results = detector.infer_batch(batch_frames)
                        for m, d in zip(batch_meta, results):
                            result_queue.put((m[0], m[1], d))
                        processed_frames += len(batch_frames)
                        batch_frames = []
                        batch_meta = []
                    print(f"  [GPU] End of file: {os.path.basename(fp)}. Total processed in session: {processed_frames}")
                    result_queue.put((fp, None, None))
                    continue
                    
                batch_frames.append(frame)
                batch_meta.append((fp, pts))
                
                if len(batch_frames) >= batch_size:
                    results = detector.infer_batch(batch_frames)
                    for m, d in zip(batch_meta, results):
                        result_queue.put((m[0], m[1], d))
                    processed_frames += len(batch_frames)
                    # 每 500 帧打印一次全局进度
                    if processed_frames % 504 == 0:
                        print(f"  [GPU] Processed {processed_frames} frames (Global)")
                    batch_frames = []
                    batch_meta = []
        except Exception as e:
            print(f"  [GPU Worker] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            result_queue.put((None, None, None))

    # ── Thread 3: Logic Worker ──
    def logic_worker() -> None:
        """从 result_queue 取检测结果，分别路由到对应的 Tracker 和 Aggregator"""
        trackers = {}
        aggregators = {}
        frame_counts = defaultdict(int)
        start_times = {}
        
        try:
            while True:
                fp, pts, detections = result_queue.get()
                if fp is None:  # 全局结束
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
                    
                if pts is None:  # 单文件结束
                    cut_segments = aggregators[fp].finalize()
                    cutter_queue.put((fp, cut_segments, start_times[fp], frame_counts[fp]))
                    
                    del trackers[fp]
                    del aggregators[fp]
                    del frame_counts[fp]
                    del start_times[fp]
                    continue
                    
                is_active = trackers[fp].update(detections)
                aggregators[fp].add_frame_state(pts, is_active)
                frame_counts[fp] += 1
                
        except Exception as e:
            print(f"  [Logic Worker] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cutter_queue.put(None)

    # ── Thread 4: Cutter Worker ──
    def cutter_worker() -> None:
        """异步切割文件，避免阻塞 Logic Worker 处理下一文件"""
        while True:
            job = cutter_queue.get()
            if job is None:
                break
            fp, cut_segments, start_time, frame_count = job
            clips = []
            fname = os.path.basename(fp)
            
            try:
                if stop_event.is_set(): continue
                print(f"  [Aggregator] {fname}: {len(cut_segments)} segments to keep.")
                if cut_segments:
                    temp_dir = os.path.join(config.paths.local_output, "temp_clips")
                    clips = FFmpegToolkit.cut_segments(
                        fp, cut_segments, temp_dir,
                        parallel_jobs=config.cutting.parallel_jobs,
                        stop_event=stop_event
                    )
                
                info = parse_video_filename(fp)
                date_key = info.date_str if info else "unknown"
                with day_clips_lock:
                    day_clips[date_key].extend(clips)
                    
                elapsed = max(time.time() - start_time, 0.001)
                db.log_performance(
                    file_name=fname,
                    decode_fps=frame_count / elapsed,
                    infer_fps=frame_count / elapsed,
                    total_time_ms=int(elapsed * 1000),
                )
                db.mark_file_completed(fp)
                print(f"  [Worker] ✅ {fname}: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} fps). {len(clips)} clips.")
            except Exception as e:
                print(f"  [Cutter Worker] ❌ Error processing {fname}: {e}")
                db.mark_file_failed(fp, str(e))

    # ── 启动所有流水线线程 ──
    t_decoder = threading.Thread(target=decoder_worker, name="decoder-worker", daemon=True)
    t_gpu = threading.Thread(target=gpu_worker, name="gpu-worker", daemon=True)
    t_logic = threading.Thread(target=logic_worker, name="logic-worker", daemon=True)
    t_cutter = threading.Thread(target=cutter_worker, name="cutter-worker", daemon=True)
    
    t_decoder.start()
    t_gpu.start()
    t_logic.start()
    t_cutter.start()
    
    # 主线程等待最终的 cutter_worker 完成
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
        gpu_id=config.hardware.gpu_id,
        conf_threshold=config.detection.confidence_threshold,
        target_classes=config.detection.classes,
    )
    detector.load_model(config.detection.model_path)

    print("=== HomeVlog Pipeline V2.4 (Global Pipeline Edition) ===")
    print(f"[Main] Input : {config.paths.nas_input}")
    print(f"[Main] Output: {config.paths.local_output}")

    try:
        # ── 扫描文件 ──
        print("\n[Scanner] Discovering files...")
        scanner.scan_once()
        time.sleep(config.scanner.write_detect_interval_sec)
        ready = scanner.scan_once()

        pending = [f for f in ready if not db.is_file_processed(f)]
        if not pending:
            print("[Main] No new files. Exiting.")
            return
        print(f"[Main] {len(pending)} file(s) to process.\n")

        # ── 按日期分组标为处理中 ──
        day_buckets: Dict[str, List[str]] = defaultdict(list)
        for fp in sorted(pending):
            info = parse_video_filename(fp)
            day_buckets[info.date_str if info else "unknown"].append(fp)

        for d, files in sorted(day_buckets.items()):
            print(f"  {d}: {len(files)} file(s)")
            db.mark_day_processing(d, len(files))

        # ── 执行全局流水线 ──
        total_start = time.time()
        print(f"\n{'='*50}")
        print(f"[Pipeline] Starting Global 4-Stage Pipeline")
        print(f"{'='*50}")
        day_clips = process_pending_files(sorted(pending), config, db, detector)

        # ── 每日合成 ──
        print(f"\n{'='*50}")
        print("[Main] Merging daily vlogs...")
        for date_key in sorted(day_clips.keys()):
            clips = day_clips[date_key]
            if not clips:
                print(f"  [{date_key}] No clips. Skipping.")
                continue
            output_name = f"vlog_{date_key}.mp4"
            final_path = str(Path(config.paths.local_output) / output_name)
            print(f"  [{date_key}] Merging {len(clips)} clip(s) → {output_name}")
            FFmpegToolkit.merge_videos(clips, final_path)
            FFmpegToolkit.cleanup(clips)
            db.mark_day_completed(date_key, final_path)
            print(f"  [{date_key}] ✅ Saved: {final_path}")

        total_elapsed = time.time() - total_start
        print(f"\n[Main] Total: {total_elapsed:.1f}s")

    except KeyboardInterrupt:
        print("\n[Main] Shutdown by user.")
    finally:
        detector.release()
        print("\n=== HomeVlog Offline ===")

if __name__ == "__main__":
    main()
