import logging
import time
import threading
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("homevlog")

class YoloVerifier:
    _model_lock = threading.Lock()
    _shared_model = None
    _shared_key = None
    # 推理串行锁：共享模型被多个分析线程并发调用，
    # ultralytics predictor 非线程安全，CUDA 上下文必须串行进入
    _inference_lock = threading.Lock()

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
            
        raw_path = yolo_cfg.get("model_path", "models/yolo11n.pt")
        from src.utils import PROJECT_ROOT
        p = Path(raw_path)
        if not p.is_absolute():
            # 依次探测: 相对当前工作目录 -> 相对 PROJECT_ROOT -> 相对 PROJECT_ROOT/models
            if p.exists():
                model_path = str(p)
            elif (PROJECT_ROOT / p).exists():
                model_path = str(PROJECT_ROOT / p)
            elif (PROJECT_ROOT / "models" / p.name).exists():
                model_path = str(PROJECT_ROOT / "models" / p.name)
            else:
                model_path = str(PROJECT_ROOT / p)
        else:
            model_path = str(p)

        self.target_classes = set(yolo_cfg.get("target_classes", [0, 1, 2, 3, 15, 16]))

        self.confidence = yolo_cfg.get("confidence", 0.25)
        self.sample_fps = yolo_cfg.get("sample_fps", 0.5)
        self.batch_size = max(1, int(yolo_cfg.get("batch_size", 4)))
        
        if device is None:
            device = yolo_cfg.get("device", config.get("hardware", {}).get("device", "cpu"))
            
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU for YOLO.")
            device = "cpu"
            
        self.device = device

        # 抑制 ultralytics 模型加载/fuse 的 stdout 摘要噪音，保持终端仪表盘版面整洁
        logging.getLogger("ultralytics").setLevel(logging.WARNING)

        weight = Path(model_path).resolve()
        stat = weight.stat()
        model_key = (str(weight), stat.st_size, stat.st_mtime_ns, device)
        with YoloVerifier._model_lock:
            if YoloVerifier._shared_model is None or YoloVerifier._shared_key != model_key:
                logger.info(f"Loading shared YOLO model {model_path} on {device}...")
                model = YOLO(model_path, task="detect")
                try:
                    model.to(device)
                    model.fuse()
                except Exception as e:
                    logger.debug(f"YOLO model fuse failed: {e}")
                YoloVerifier._shared_model = model
                YoloVerifier._shared_key = model_key
        self.model = YoloVerifier._shared_model
        
    def verify(self, filepath: str, segments: list, gpu: str = "qsv", device: str | None = None,
               frames_buffer: dict | None = None, analysis_fps: float = 5.0) -> list:
        """Bounded inference over shared JPEGs. Missing evidence never means a negative result."""
        if not self.enabled:
            return segments
        import cv2
        import torch
        from bisect import bisect_left
        from src.segment import _merge_same_state

        frames_buffer = frames_buffer or {}
        keys = sorted(frames_buffer)
        jobs = []
        counts, confirmed, confidences = {}, {}, {}
        for i, seg in enumerate(segments):
            if not seg.is_dynamic:
                continue
            start = max(0, seg.start_time - seg.file_start_offset) * analysis_fps
            end = max(0, seg.end_time - seg.file_start_offset) * analysis_fps
            matched = keys[bisect_left(keys, start):bisect_left(keys, end)]
            if len(matched) > 8:
                matched = [matched[j] for j in np.linspace(0, len(matched)-1, 8, dtype=int)]
            jobs.extend((i, k) for k in matched)
            counts[i], confirmed[i], confidences[i] = 0, False, 0.0

        batch_size = max(1, int(getattr(self, "batch_size", 4)))
        try:
            for offset in range(0, len(jobs), batch_size):
                frames, owners, thresholds = [], [], []
                for i, k in jobs[offset:offset+batch_size]:
                    raw = frames_buffer[k]
                    try:
                        if isinstance(raw, (bytes, bytearray)) or getattr(raw, "ndim", 0) == 1:
                            img = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                        else:
                            img = raw
                        if img is None:
                            continue
                        luma = float(np.mean(img))
                        threshold = min(self.confidence, 0.15 if luma < 50 else 0.20 if luma < 100 else self.confidence)
                        if segments[i].max_energy >= 3.5:
                            threshold = min(threshold, 0.20)
                        frames.append(img)
                        owners.append(i)
                        thresholds.append(threshold)
                    except (ValueError, cv2.error):
                        logger.warning("Invalid cached YOLO frame: %s index %s", filepath, k)
                if not frames:
                    continue
                with self._inference_lock, torch.inference_mode():
                    results = list(self.model(frames, conf=min(thresholds), verbose=False, stream=True))
                    if len(results) != len(frames):
                        raise RuntimeError("YOLO returned an incomplete batch")
                    for i, threshold, result in zip(owners, thresholds, results):
                        counts[i] += 1
                        if result.boxes is None:
                            continue
                        for cls, conf in zip(result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                            if int(cls) in self.target_classes:
                                confidences[i] = max(confidences[i], float(conf))
                                confirmed[i] |= float(conf) >= threshold
        except Exception as exc:
            logger.warning("YOLO verification failed for %s: %s; retaining motion", filepath, exc)
            for i in counts:
                segments[i].needs_review = True
                segments[i].review_reason = "YOLO_FAILED"
            return segments

        for i, count in counts.items():
            seg = segments[i]
            seg.avg_confidence = round(confidences[i], 4)
            if count == 0 or (not confirmed[i] and count < 2):
                seg.needs_review = True
                seg.review_reason = "INSUFFICIENT_YOLO_EVIDENCE"
            elif confirmed[i]:
                if confidences[i] < 0.28:
                    seg.needs_review = True
                    seg.review_reason = "BORDERLINE_CONFIDENCE"
            elif seg.state == "DYNAMIC_AUDIO":
                seg.needs_review = True
                seg.review_reason = "MULTIMODAL_AUDIO_NO_TARGET"
            else:
                seg.state = "STATIC"
                seg.needs_review = True
                seg.review_reason = "YOLO_NEGATIVE_REQUIRES_AUDIT"
        return _merge_same_state(segments, gap_tolerance=1.5)
