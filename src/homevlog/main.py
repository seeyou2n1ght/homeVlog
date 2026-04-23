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
    处理单个视频文件的全生命周期
    """
    print(f"\n[Worker] Processing: {os.path.basename(file_path)}")
    db.mark_file_started(file_path)
    
    # 1. 初始化管道组件
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
    
    # 2. 启动解码
    decoder.start(file_path)
    
    start_time = time.time()
    frame_count = 0
    
    try:
        # 3. 处理帧流 (带进度条)
        pbar = tqdm(unit=" frames", desc="Analyzing")
        while True:
            data = decoder.get_frame()
            if data is None:
                break
                
            pts, frame = data
            
            # 推理
            results = detector.infer(frame)
            
            # 追踪与状态聚合
            is_active = tracker.update(results)
            aggregator.add_frame_state(pts, is_active)
            
            frame_count += 1
            pbar.update(1)
        pbar.close()
            
        # 4. 生成切割任务
        cut_segments = aggregator.finalize()
        print(f"[Aggregator] Identified {len(cut_segments)} segments to keep.")
        
        if cut_segments:
            # 5. 执行物理切割
            temp_dir = os.path.join(config.paths.local_output, "temp_clips")
            clips = FFmpegToolkit.cut_segments(file_path, cut_segments, temp_dir)
            
            # 6. 合并为最终浓缩视频
            output_name = f"vlog_{os.path.basename(file_path)}"
            final_path = os.path.join(config.paths.local_output, output_name)
            FFmpegToolkit.merge_videos(clips, final_path)
            
            # 清理
            FFmpegToolkit.cleanup(clips)
            print(f"[Cutter] Saved condensed vlog to: {final_path}")
        
        # 7. 记录性能指标并更新状态
        total_time_sec = time.time() - start_time
        safe_time_sec = total_time_sec if total_time_sec > 0.001 else 0.001
        
        db.log_performance(
            os.path.basename(file_path), 
            decode_fps=frame_count / safe_time_sec, 
            infer_fps=frame_count / safe_time_sec, # Mock 环境下两者一致
            total_time_ms=int(total_time_sec * 1000)
        )
        db.mark_file_completed(file_path)
        
    except Exception as e:
        print(f"[Worker] Critical Error processing {file_path}: {e}")
        # 这里可以扩展异常处理逻辑
    finally:
        decoder.close()

def main():
    # 1. 环境准备
    config = load_config()
    db = DatabaseManager(config.paths.sqlite_db)
    scanner = FileScanner(config.scanner, config.paths.nas_input)
    
    # 2. 初始化硬件 (Mock or TRT)
    detector = get_detector(use_mock=config.hardware.use_mock, gpu_id=config.hardware.gpu_id)
    detector.load_model(config.detection.model_path)
    
    print("=== HomeVlog Pipeline Started (V2.1) ===")
    
    try:
        # 3. 进入主监听循环
        for new_file in scanner.wait_for_files(db):
            process_single_video(new_file, config, db, detector)
    except KeyboardInterrupt:
        print("\n[Main] Shutdown requested by user.")
    finally:
        detector.release()
        print("=== HomeVlog Offline ===")

if __name__ == "__main__":
    main()
