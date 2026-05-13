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
        self.sample_fps = yolo_cfg.get("sample_fps", 1.0)
        self.skip_energy_threshold = yolo_cfg.get("skip_energy_threshold", 15.0)
        
        if device is None:
            device = yolo_cfg.get("device", config.get("hardware", {}).get("device", "cpu"))
            
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU for YOLO.")
            device = "cpu"
            
        self.device = device
        
        with YoloVerifier._model_lock:
            if YoloVerifier._shared_model is None:
                logger.info(f"Loading shared YOLO model {model_path} on {device}...")
                YoloVerifier._shared_model = YOLO(model_path, task="detect")
        self.model = YoloVerifier._shared_model
        
    def verify(self, filepath: str, segments: list, gpu: str = "qsv", device: str | None = None, frames_buffer: dict = None, analysis_fps: float = 5.0) -> list:
        """
        验证动态片段。若片段无目标，将其状态翻转为 STATIC。
        使用内存驻留的 frames_buffer 进行推理，避免重复启动 ffmpeg。
        """
        if not self.enabled or not frames_buffer:
            return segments
        
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
                
            if seg.max_energy >= self.skip_energy_threshold:
                logger.debug(f"YOLO Skipped (Massive Motion, energy {seg.max_energy:.1f} >= {self.skip_energy_threshold}) for {filepath} at {local_start:.1f}s")
                verified_segments.append(seg)
                continue
                
            t0 = time.monotonic()
            is_really_dynamic = self._verify_segment(filepath, local_start, duration, frames_buffer, analysis_fps)
            elapsed = time.monotonic() - t0
            
            if is_really_dynamic:
                logger.debug(f"YOLO Confirmed DYNAMIC for {filepath} at {local_start:.1f}s, dur {duration:.1f}s (took {elapsed:.2f}s)")
                verified_segments.append(seg)
            else:
                logger.debug(f"YOLO Flipped to STATIC for {filepath} at {local_start:.1f}s, dur {duration:.1f}s (took {elapsed:.2f}s)")
                seg.state = "STATIC"
                verified_segments.append(seg)
                
        return verified_segments

    def _verify_segment(self, filepath: str, start_time: float, duration: float, frames_buffer: dict, analysis_fps: float) -> bool:
        start_frame_idx = int(start_time * analysis_fps)
        end_frame_idx = int((start_time + duration) * analysis_fps)
        
        frames_to_check = []
        for f_idx in range(start_frame_idx, end_frame_idx + 1):
            if f_idx in frames_buffer:
                frames_to_check.append(frames_buffer[f_idx])
                
        if not frames_to_check:
            logger.debug(f"YOLO: No frames in buffer for {filepath} at {start_time:.1f}s")
            return True # Fallback to original DYNAMIC state on missing frames

        try:
            for frame in frames_to_check:
                # frame 已经是 np.ndarray (H, W, 3) 的 RGB 图像
                # OpenVINO 后端原生支持多线程推理
                results = self.model(frame, verbose=False, device=self.device)
                
                for r in results:
                    if r.boxes is None or len(r.boxes.cls) == 0:
                        continue
                        
                    classes = r.boxes.cls.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    
                    for cls, conf in zip(classes, confs):
                        if int(cls) in self.target_classes and conf >= self.confidence:
                            return True
        except Exception as e:
            logger.error(f"YOLO memory inference failed for {filepath}: {e}")
            return True # Fallback to original DYNAMIC state on error
            
        return False