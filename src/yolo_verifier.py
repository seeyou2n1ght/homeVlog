import logging
import subprocess
import time
import threading
import numpy as np

logger = logging.getLogger("homevlog")

class YoloVerifier:
    _model_lock = threading.Lock()
    _shared_model = None

    def __init__(self, config: dict, device: str | None = None):
        yolo_cfg = config.get("yolo", {})
        self.enabled = yolo_cfg.get("enabled", False)
        if not self.enabled:
            return
            
        try:
            from ultralytics import YOLO
            import torch
        except ImportError:
            logger.error("ultralytics or torch not installed. YOLO verifier disabled.")
            self.enabled = False
            return
            
        model_path = yolo_cfg.get("model_path", "yolo11n.pt")
        self.target_classes = set(yolo_cfg.get("target_classes", [0, 1, 2, 3, 15, 16]))
        self.confidence = yolo_cfg.get("confidence", 0.25)
        self.sample_fps = yolo_cfg.get("sample_fps", 0.5)
        self.skip_energy_threshold = yolo_cfg.get("skip_energy_threshold", 12.0)
        
        if device is None:
            device = yolo_cfg.get("device", config.get("hardware", {}).get("device", "cpu"))
            
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU for YOLO.")
            device = "cpu"
            
        self.device = device
        
        with YoloVerifier._model_lock:
            if YoloVerifier._shared_model is None:
                logger.info(f"Loading shared YOLO model {model_path} on {device}...")
                model = YOLO(model_path, task="detect")
                try:
                    model.to(device)
                    model.fuse()
                except Exception as e:
                    logger.debug(f"YOLO model fuse failed: {e}")
                YoloVerifier._shared_model = model
        self.model = YoloVerifier._shared_model
        
    def verify(self, filepath: str, segments: list, gpu: str = "qsv", device: str | None = None, frames_buffer: dict = None, analysis_fps: float = 5.0) -> list:
        if not self.enabled or not frames_buffer:
            return segments
        
        # 统一设备检查
        if device is None:
            device = self.device
        
        verified_segments = []
        for seg in segments:
            if seg.state != "DYNAMIC":
                verified_segments.append(seg)
                continue
                
            local_start = max(0.0, seg.start_time - seg.file_start_offset)
            local_end = max(0.0, seg.end_time - seg.file_start_offset)
            duration = local_end - local_start
            
            if duration <= 0:
                verified_segments.append(seg)
                continue
                
            # [性能极限] 极大动作跳过
            if seg.max_energy >= self.skip_energy_threshold:
                verified_segments.append(seg)
                continue
                
            t0 = time.monotonic()
            # [性能极限] 使用批量推理逻辑
            is_really_dynamic = self._verify_segment_batch(filepath, local_start, duration, frames_buffer, analysis_fps)
            elapsed = time.monotonic() - t0
            
            if is_really_dynamic:
                logger.debug(f"YOLO DYNAMIC (Batch) for {Path(filepath).name} at {local_start:.1f}s (took {elapsed:.3f}s)")
                verified_segments.append(seg)
            else:
                logger.debug(f"YOLO STATIC (Batch) for {Path(filepath).name} at {local_start:.1f}s (took {elapsed:.3f}s)")
                seg.state = "STATIC"
                verified_segments.append(seg)
                
        return verified_segments

    def _verify_segment_batch(self, filepath: str, start_time: float, duration: float, frames_buffer: dict, analysis_fps: float) -> bool:
        """
        极限优化：将片段内的待检帧作为 Batch 一次性送入 GPU 推理
        """
        start_frame_idx = int(start_time * analysis_fps)
        end_frame_idx = int((start_time + duration) * analysis_fps)
        
        # 采样待检帧
        sample_step = max(1, int(analysis_fps / self.sample_fps))
        frames_to_check = []
        for f_idx in range(start_frame_idx, end_frame_idx + 1, sample_step):
            if f_idx in frames_buffer:
                frames_to_check.append(frames_buffer[f_idx])
                
        if not frames_to_check:
            return True # Fallback

        try:
            # [性能极限] 一次性处理 List of NP Arrays，触发 Ultralytics 内部的 Batch 推理
            results = self.model(frames_to_check, verbose=False, stream=True)
            
            for r in results:
                if r.boxes is not None and len(r.boxes.cls) > 0:
                    classes = r.boxes.cls.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    
                    for cls, conf in zip(classes, confs):
                        if int(cls) in self.target_classes and conf >= self.confidence:
                            return True
        except Exception as e:
            logger.error(f"YOLO Batch inference failed for {filepath}: {e}")
            return True # Fallback
            
        return False
from pathlib import Path
