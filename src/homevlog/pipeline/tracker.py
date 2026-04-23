from typing import List, Optional
from ..hal.base import DetectionResult

class TrackedObject:
    def __init__(self, detection: DetectionResult):
        self.last_detection = detection
        self.hit_count = 1
        self.miss_count = 0

class SimpleTracker:
    """
    轻量级追踪与去抖动器
    职责：过滤瞬时误报，平滑运动/静止状态切换。
    """
    def __init__(self, iou_threshold: float = 0.5, debounce_frames: int = 5):
        self.iou_threshold = iou_threshold
        self.debounce_frames = debounce_frames
        
        # 当前活跃的追踪目标 (简化版只追踪最重要的一个或整体状态)
        self.is_active = False # 当前是否处于“有意义运动”状态
        self.active_counter = 0 # 状态持续计数器

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, detections: List[DetectionResult]) -> bool:
        """
        更新追踪状态。
        返回：经过平滑后的“是否有意义运动”判定结果。
        """
        has_detection = len(detections) > 0
        
        if has_detection:
            if not self.is_active:
                self.active_counter += 1
                if self.active_counter >= self.debounce_frames:
                    self.is_active = True
                    self.active_counter = 0 # 重置用于下一次切换
            else:
                self.active_counter = 0 # 维持 active 状态，重置消亡计数
        else:
            if self.is_active:
                self.active_counter += 1
                if self.active_counter >= self.debounce_frames:
                    self.is_active = False
                    self.active_counter = 0
            else:
                self.active_counter = 0
                
        return self.is_active
