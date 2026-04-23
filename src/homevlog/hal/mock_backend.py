import time
import numpy as np
from typing import List
from .base import BaseDetector, DetectionResult

class MockDetector(BaseDetector):
    """
    开发期使用的伪后端。
    内置状态机模拟目标的：移动 -> 静止 -> 消失 周期。
    支持 Batch 推理接口。
    """
    def __init__(self):
        self.frame_count = 0
        self.cycle_length = 300  # 每 300 帧一个周期
        self.classes = ["person", "baby", "cat", "dog"]

    def load_model(self, model_path: str):
        print(f"[MockHAL] Stub: Loading model from {model_path} (Simulation mode)")

    def _infer_single(self) -> List[DetectionResult]:
        self.frame_count += 1
        phase = self.frame_count % self.cycle_length
        results = []
        
        # 1. 前 100 帧：模拟一个正在横穿画面的人 (移动状态)
        if 0 <= phase < 100:
            x_offset = phase * 10
            results.append(DetectionResult(
                class_id=0, label="person", confidence=0.92,
                bbox=[100 + x_offset, 200, 200 + x_offset, 500]
            ))
        # 2. 100~200 帧：模拟目标停留在原地 (静止状态)
        elif 100 <= phase < 200:
            results.append(DetectionResult(
                class_id=0, label="person", confidence=0.95,
                bbox=[1100, 200, 1200, 500]
            ))
        
        return results

    def infer_batch(self, frames: List[np.ndarray]) -> List[List[DetectionResult]]:
        """Batch 模拟推理"""
        return [self._infer_single() for _ in frames]

    def release(self):
        print("[MockHAL] Stub: Releasing virtual resources")
