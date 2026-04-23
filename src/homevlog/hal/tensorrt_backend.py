"""
TensorRT 推理后端 — 高性能手写预处理版
绕过 ultralytics predictor 层，直接调用 AutoBackend forward，
预处理改为 12 帧批量向量化操作（np.stack → torch → GPU 一次传输）。
"""
import numpy as np
import torch
from typing import List
from .base import BaseDetector, DetectionResult

_COCO_NAME_TO_ID: dict[str, int] = {
    "person": 0, "baby": 0, "cat": 15, "dog": 16,
}


class TensorRTDetector(BaseDetector):

    def __init__(
        self,
        gpu_id: int = 0,
        conf_threshold: float = 0.5,
        target_classes: List[str] | None = None,
    ):
        self.gpu_id = gpu_id
        self.conf_threshold = conf_threshold
        self.autobackend = None
        self.engine_batch_size: int = 1

        classes = target_classes or ["person", "cat", "dog"]
        self.class_ids: List[int] = sorted(
            {_COCO_NAME_TO_ID[c] for c in classes if c in _COCO_NAME_TO_ID}
        )
        self.class_ids_set = set(self.class_ids)
        # ultralytics NMS 需要的 class 名称映射
        self.names: dict[int, str] = {}

    def load_model(self, model_path: str) -> None:
        import re
        from ultralytics.nn.autobackend import AutoBackend

        device = torch.device(f"cuda:{self.gpu_id}")
        print(f"[TRTHAL] Loading engine: {model_path}")

        # 直接构造 AutoBackend，跳过 YOLO predictor 的惰性初始化
        self.autobackend = AutoBackend(
            model=model_path,
            device=device,
            fp16=False,  # 引擎输入层绑定通常是 FP32，TRT 内部会自动转 FP16
        )
        self.names = self.autobackend.names  # {0:'person', 15:'cat', ...}

        # 从 AutoBackend 中直接安全提取引擎的 batch size，避免 dummy tensor 崩溃
        if hasattr(self.autobackend, "backend") and hasattr(self.autobackend.backend, "bindings"):
            img_binding = self.autobackend.backend.bindings.get("images")
            if img_binding:
                self.engine_batch_size = img_binding.shape[0]

        print(
            f"[TRTHAL] Ready. batch={self.engine_batch_size}, "
            f"classes={self.class_ids}, GPU:{self.gpu_id}"
        )

    def _preprocess(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        手写向量化预处理：12帧 BGR numpy → 1个 FP16 CUDA tensor
        全程批量操作，无 for 循环。
        """
        batch_np = np.stack(frames)
        tensor = torch.from_numpy(batch_np).to(f"cuda:{self.gpu_id}", dtype=torch.float32)
        # NHWC -> NCHW
        tensor = tensor.permute(0, 3, 1, 2)
        # BGR to RGB on GPU
        tensor = tensor[:, [2, 1, 0], :, :].contiguous()
        tensor.div_(255.0)
        return tensor

    def _postprocess(
        self, preds: torch.Tensor, real_count: int
    ) -> List[List[DetectionResult]]:
        """[Phase 3] 解析 TRT 输出 + 手写 torchvision NMS + 类别过滤，完全脱离 ultralytics"""
        import torchvision
        
        try:
            # preds: (batch, 84, 8400) -> (batch, 8400, 84)
            preds = preds.transpose(1, 2)
            
            results: List[List[DetectionResult]] = []
            for i in range(real_count):
                x = preds[i]
                
                box = x[:, :4]  # xc, yc, w, h
                scores = x[:, 4:]  # class scores
                
                max_scores, max_classes = torch.max(scores, dim=1)
                
                # 1. 过滤置信度 & 目标类别
                mask = max_scores > self.conf_threshold
                class_mask = torch.zeros_like(mask)
                for cid in self.class_ids_set:
                    class_mask |= (max_classes == cid)
                mask &= class_mask
                
                box = box[mask]
                max_scores = max_scores[mask]
                max_classes = max_classes[mask]
                
                if not box.shape[0]:
                    results.append([])
                    continue
                    
                # 2. xywh -> xyxy
                x1 = box[:, 0] - box[:, 2] / 2
                y1 = box[:, 1] - box[:, 3] / 2
                x2 = box[:, 0] + box[:, 2] / 2
                y2 = box[:, 1] + box[:, 3] / 2
                boxes = torch.stack((x1, y1, x2, y2), dim=1)
                
                # 3. Class-aware NMS (各类别独立 NMS)
                max_coordinate = boxes.max()
                offsets = max_classes.float() * (max_coordinate + 1)
                boxes_for_nms = boxes + offsets.unsqueeze(1)
                
                keep = torchvision.ops.nms(boxes_for_nms, max_scores, 0.45)
                if keep.shape[0] > 100:  # max_det=100
                    keep = keep[:100]
                    
                keep_boxes = boxes[keep]
                keep_scores = max_scores[keep]
                keep_classes = max_classes[keep]
                
                frame_dets = []
                for j in range(keep.shape[0]):
                    x1_f, y1_f, x2_f, y2_f = keep_boxes[j].tolist()
                    cls_id = int(keep_classes[j])
                    frame_dets.append(DetectionResult(
                        class_id=cls_id,
                        label=self.names.get(cls_id, str(cls_id)),
                        confidence=float(keep_scores[j]),
                        bbox=[x1_f, y1_f, x2_f, y2_f],
                    ))
                results.append(frame_dets)
                
            return results
        except Exception as e:
            return [[] for _ in range(real_count)]

    def infer_batch(self, frames: List[np.ndarray]) -> List[List[DetectionResult]]:
        """批量推理：自动按 engine batch size 分组，不足时填充"""
        bs = self.engine_batch_size
        all_results: List[List[DetectionResult]] = []

        for start in range(0, len(frames), bs):
            chunk = frames[start: start + bs]
            real_count = len(chunk)
            # 不足 batch 时用最后一帧填充
            if real_count < bs:
                chunk = chunk + [chunk[-1]] * (bs - real_count)

            tensor = self._preprocess(chunk)
            preds = self.autobackend(tensor)
            dets = self._postprocess(preds, real_count)
            all_results.extend(dets)

        return all_results

    def release(self) -> None:
        self.autobackend = None
        torch.cuda.empty_cache()
        print("[TRTHAL] Released.")
