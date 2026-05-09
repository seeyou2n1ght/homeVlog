import logging
import subprocess
import time
import numpy as np

logger = logging.getLogger("homevlog")

class YoloVerifier:
    def __init__(self, config: dict):
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
        
        device = yolo_cfg.get("device", "cuda:0")
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU for YOLO.")
            device = "cpu"
            
        logger.info(f"Loading YOLO model {model_path} on {device}...")
        self.model = YOLO(model_path)
        self.device = device
        
    def verify(self, filepath: str, segments: list, gpu: str = "qsv") -> list:
        """
        验证动态片段。若片段无目标，将其状态翻转为 STATIC。
        """
        if not self.enabled:
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
                
            t0 = time.monotonic()
            is_really_dynamic = self._verify_segment(filepath, local_start, duration, gpu)
            elapsed = time.monotonic() - t0
            
            if is_really_dynamic:
                logger.debug(f"YOLO Confirmed DYNAMIC for {filepath} at {local_start:.1f}s, dur {duration:.1f}s (took {elapsed:.2f}s)")
                verified_segments.append(seg)
            else:
                logger.debug(f"YOLO Flipped to STATIC for {filepath} at {local_start:.1f}s, dur {duration:.1f}s (took {elapsed:.2f}s)")
                seg.state = "STATIC"
                verified_segments.append(seg)
                
        return verified_segments

    def _verify_segment(self, filepath: str, start_time: float, duration: float, gpu: str) -> bool:
        width, height = 640, 360
        frame_size = width * height * 3
        
        args = [
            "ffmpeg", "-hide_banner", "-loglevel", "error", "-y"
        ]
        
        # Optionally add HW decode if we want, but seeking with SW decode is often fine and robust.
        # HW decode for seeking might have issues on GOP boundaries depending on exact parameters.
        if gpu == "cuda":
            args.extend(["-hwaccel", "cuda"])
        elif gpu == "qsv":
            args.extend(["-hwaccel", "qsv"])

        args.extend([
            "-ss", str(start_time),
            "-t", str(duration),
            "-i", str(filepath),
            "-vf", f"fps={self.sample_fps},scale={width}:{height}",
            "-f", "image2pipe",
            "-pix_fmt", "rgb24",
            "-vcodec", "rawvideo",
            "-"
        ])
        
        try:
            from src.utils import get_io_semaphore
            io_sem = get_io_semaphore()
            io_sem.acquire()
            try:
                proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                import threading
                def _kill_on_timeout():
                    try:
                        proc.kill()
                    except OSError:
                        pass
                
                # 动态超时: 每抽1帧给1秒余量 + 固定15秒
                watchdog = threading.Timer(duration * self.sample_fps + 15.0, _kill_on_timeout)
                watchdog.daemon = True
                watchdog.start()
                
                try:
                    while True:
                        raw = proc.stdout.read(frame_size)
                        if len(raw) < frame_size:
                            break
                        
                        frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3))
                        
                        # YOLO inference
                        results = self.model(frame, verbose=False, device=self.device)
                        
                        for r in results:
                            if r.boxes is None or len(r.boxes.cls) == 0:
                                continue
                                
                            classes = r.boxes.cls.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            
                            for cls, conf in zip(classes, confs):
                                if int(cls) in self.target_classes and conf >= self.confidence:
                                    return True
                finally:
                    watchdog.cancel()
                    try:
                        proc.kill()
                        proc.wait()
                    except OSError:
                        pass
            finally:
                io_sem.release()
                
        except Exception as e:
            logger.error(f"YOLO verification failed for {filepath}: {e}")
            return True # Fallback to original DYNAMIC state on error
            
        return False