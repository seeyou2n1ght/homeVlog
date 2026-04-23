import os
import sys
import time
from pathlib import Path
from tqdm import tqdm

from .config import load_config
from .database.sqlite_manager import DatabaseManager
from .scanner import FileScanner
from .hal import get_detector
from .pipeline.decoder import VideoDecoder
from .pipeline.tracker import SimpleTracker
from .pipeline.aggregator import Aggregator
from .utils.ffmpeg_tools import FFmpegToolkit

def process_single_video(file_path, config, db, detector):
    """
    处理单个视频文件的全生命周期 (V2.2 Batch Edition)
    """
    print(f"\n[Worker] Processing: {os.path.basename(file_path)}")
    db.mark_file_started(file_path)
    
    hwaccel = "none" if config.hardware.use_mock else config.hardware.ffmpeg_hwaccel
    decoder = VideoDecoder(hwaccel=hwaccel, io_timeout=config.cutting.io_timeout_sec)
    tracker = SimpleTracker(
        iou_threshold=config.detection.iou_threshold, 
        debounce_frames=config.detection.debounce_frames
    )
    aggregator = Aggregator(
        static_interval_ms=config.detection.static_interval_ms,
        padding_ms=config.cutting.keyframe_padding_ms
    )
    
    # V2.2: 注入降维参数
    decoder.start(
        file_path, 
        fps=config.detection.fps, 
        infer_resolution=config.detection.infer_resolution
    )
    
    start_time = time.time()
    frame_count = 0
    batch_size = config.hardware.batch_size
    batch_frames = []
    batch_pts = []
    
    def flush_batch():
        """执行 Batch 推理并分发状态"""
        if not batch_frames:
            return
        results_batch = detector.infer_batch(batch_frames)
        for pts, results in zip(batch_pts, results_batch):
            is_active = tracker.update(results)
            aggregator.add_frame_state(pts, is_active)
        batch_frames.clear()
        batch_pts.clear()
    
    try:
        pbar = tqdm(unit=" frames", desc="Analyzing")
        while True:
            data = decoder.get_frame()
            if data is None:
                flush_batch() # 别忘了清空最后的尾巴
                break
                
            pts, frame = data
            batch_frames.append(frame)
            batch_pts.append(pts)
            
            # 攒够 Batch Size，集中推理
            if len(batch_frames) >= batch_size:
                flush_batch()
            
            frame_count += 1
            pbar.update(1)
        pbar.close()
            
        cut_segments = aggregator.finalize()
        print(f"[Aggregator] Identified {len(cut_segments)} segments to keep.")
        
        if cut_segments:
            temp_dir = os.path.join(config.paths.local_output, "temp_clips")
            # V2.2: 并发物理切割
            clips = FFmpegToolkit.cut_segments(
                file_path, cut_segments, temp_dir, 
                parallel_jobs=config.cutting.parallel_jobs
            )
            
            output_name = f"vlog_{os.path.basename(file_path)}"
            final_path = os.path.join(config.paths.local_output, output_name)
            FFmpegToolkit.merge_videos(clips, final_path)
            
            FFmpegToolkit.cleanup(clips)
            print(f"[Cutter] Saved condensed vlog to: {final_path}")
        
        total_time_sec = time.time() - start_time
        safe_time_sec = total_time_sec if total_time_sec > 0.001 else 0.001
        
        db.log_performance(
            os.path.basename(file_path), 
            decode_fps=frame_count / safe_time_sec, 
            infer_fps=frame_count / safe_time_sec, 
            total_time_ms=int(total_time_sec * 1000)
        )
        db.mark_file_completed(file_path)
        
    except Exception as e:
        print(f"[Worker] Critical Error processing {file_path}: {e}")
    finally:
        decoder.close()

def main():
    config = load_config()
    db = DatabaseManager(config.paths.sqlite_db)
    scanner = FileScanner(config.scanner, config.paths.nas_input)
    
    detector = get_detector(use_mock=config.hardware.use_mock, gpu_id=config.hardware.gpu_id)
    detector.load_model(config.detection.model_path)
    
    print("=== HomeVlog Pipeline Started (V2.2 Performance Edition) ===")
    try:
        for new_file in scanner.wait_for_files(db):
            process_single_video(new_file, config, db, detector)
    except KeyboardInterrupt:
        print("\n[Main] Shutdown requested by user.")
    finally:
        detector.release()
        print("=== HomeVlog Offline ===")

if __name__ == "__main__":
    main()
