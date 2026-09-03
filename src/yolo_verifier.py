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
        self.skip_energy_threshold = yolo_cfg.get("skip_energy_threshold", 12.0)
        
        if device is None:
            device = yolo_cfg.get("device", config.get("hardware", {}).get("device", "cpu"))
            
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA not available. Falling back to CPU for YOLO.")
            device = "cpu"
            
        self.device = device

        # 抑制 ultralytics 模型加载/fuse 的 stdout 摘要噪音，保持终端仪表盘版面整洁
        logging.getLogger("ultralytics").setLevel(logging.WARNING)

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

        # 1. 筛选需要 YOLO 验证的动态片段，收集全局待检帧
        segs_to_verify: list[tuple[int, Any, float, float]] = [] # (index, seg, local_start, duration)
        for idx, seg in enumerate(segments):
            if seg.state == "DYNAMIC_AUDIO":
                continue
            if seg.state != "DYNAMIC":
                continue
            local_start = max(0.0, seg.start_time - seg.file_start_offset)
            local_end = max(0.0, seg.end_time - seg.file_start_offset)
            duration = local_end - local_start
            if duration <= 0 or seg.max_energy >= self.skip_energy_threshold:
                continue
            segs_to_verify.append((idx, seg, local_start, duration))

        if not segs_to_verify:
            return segments

        # 2. 全局聚合所有候选片段的采样帧，建立帧到片段的映射关系
        sample_step = max(1, int(analysis_fps / self.sample_fps))
        all_frames: list[np.ndarray] = []
        frame_to_seg_idx: list[int] = []

        import cv2
        for seg_idx, seg, start_t, dur in segs_to_verify:
            start_frame_idx = int(start_t * analysis_fps)
            end_frame_idx = int((start_t + dur) * analysis_fps)
            for f_idx in range(start_frame_idx, end_frame_idx + 1, sample_step):
                if f_idx in frames_buffer:
                    buf_item = frames_buffer[f_idx]
                    # 支持 JPEG 压缩切片与原始 ndarray 自动兼容解码
                    if isinstance(buf_item, (bytes, bytearray, np.ndarray)) and getattr(buf_item, "ndim", 0) == 1:
                        img = cv2.imdecode(np.frombuffer(buf_item, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            all_frames.append(img)
                            frame_to_seg_idx.append(seg_idx)

                    elif isinstance(buf_item, np.ndarray):
                        all_frames.append(buf_item)
                        frame_to_seg_idx.append(seg_idx)

        # 若无提取到有效帧，保持原状
        if not all_frames:
            return segments

        # 3. 在 torch.inference_mode() 保护下执行全局单次 Batch 前向推理
        seg_has_target: dict[int, bool] = {s_idx: False for s_idx, _, _, _ in segs_to_verify}
        # 暗光/红外微光场景自适应检测：若样本帧平均灰度 < 50，调低 person 判定门槛至 0.15，防婴儿被误杀
        is_night_scene = bool(np.mean([np.mean(f) for f in all_frames[:min(5, len(all_frames))]]) < 50.0)
        target_conf = 0.15 if is_night_scene else self.confidence

        t0 = time.monotonic()
        try:
            import torch
            with torch.inference_mode():
                results = self.model(all_frames, verbose=False, stream=True)
                for frame_i, r in enumerate(results):
                    if r.boxes is not None and len(r.boxes.cls) > 0:
                        classes = r.boxes.cls.cpu().numpy()
                        confs = r.boxes.conf.cpu().numpy()
                        s_idx = frame_to_seg_idx[frame_i]
                        for cls, conf in zip(classes, confs):
                            if int(cls) in self.target_classes and conf >= target_conf:
                                seg_has_target[s_idx] = True
                                break

            elapsed = time.monotonic() - t0
            logger.debug(
                f"YOLO Global Batch inference for {Path(filepath).name}: "
                f"{len(all_frames)} frames across {len(segs_to_verify)} segments in {elapsed:.3f}s"
            )
        except Exception as e:
            logger.error(f"YOLO Global Batch inference failed for {filepath}: {e}")
            return segments

        # 4. 根据推理结果精准调整各片段状态
        verified_segments = []
        target_lookup = set(s_idx for s_idx, has_t in seg_has_target.items() if has_t)
        for idx, seg in enumerate(segments):
            if any(idx == s_idx for s_idx, _, _, _ in segs_to_verify):
                if idx in target_lookup:
                    verified_segments.append(seg)
                else:
                    seg.state = "STATIC"
                    verified_segments.append(seg)
            else:
                verified_segments.append(seg)

        return verified_segments

