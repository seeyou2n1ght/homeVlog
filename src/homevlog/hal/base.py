from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class DetectionResult:
    """标准化的检测结果格式"""
    def __init__(self, class_id: int, label: str, confidence: float, bbox: List[float]):
        self.class_id = class_id
        self.label = label
        self.confidence = confidence
        self.bbox = bbox  # [x1, y1, x2, y2] 格式

class BaseDetector(ABC):
    """检测器基类"""
    @abstractmethod
    def load_model(self, model_path: str):
        """加载模型引擎"""
        pass

    @abstractmethod
    def infer(self, frame: np.ndarray) -> List[DetectionResult]:
        """单帧推理"""
        pass

    @abstractmethod
    def release(self):
        """释放显存资源"""
        pass
