from typing import List
from .base import BaseDetector
from .mock_backend import MockDetector
from .tensorrt_backend import TensorRTDetector


def get_detector(
    use_mock: bool = False,
    gpu_id: int = 0,
    conf_threshold: float = 0.5,
    target_classes: List[str] | None = None,
) -> BaseDetector:
    """HAL 工厂方法：根据配置返回对应的检测后端"""
    if use_mock:
        print("[HAL] Initializing MockBackend for local development.")
        return MockDetector()
    else:
        print(f"[HAL] Initializing TensorRTBackend on GPU:{gpu_id}")
        return TensorRTDetector(
            gpu_id=gpu_id,
            conf_threshold=conf_threshold,
            target_classes=target_classes,
        )
