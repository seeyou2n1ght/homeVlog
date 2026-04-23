from typing import List, Optional
from ..hal.base import DetectionResult

class SimpleTracker:
    """
    轻量级追踪与去抖动器
    职责：过滤瞬时误报，平滑运动/静止状态切换。
    核心逻辑：基于连续帧 BBox 的 IoU 计算。
    """
    def __init__(self, iou_threshold: float = 0.80, debounce_frames: int = 5):
        self.iou_threshold = iou_threshold
        self.debounce_frames = debounce_frames
        
        self.last_detections: List[DetectionResult] = []
        
        # 当前平滑后的状态 (True: 运动期, False: 静止期或无目标)
        self.is_motion = False 
        self.state_counter = 0 # 状态持续计数器

    def _calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        
        if interArea <= 0:
            return 0.0
            
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, detections: List[DetectionResult]) -> bool:
        """
        更新追踪状态。
        返回：经过平滑后的“是否处于运动期”判定结果。
        """
        raw_motion = False
        
        # 1. 判定当前帧的原始状态
        if len(detections) == 0:
            # 无目标视为静止期（在 Aggregator 中会被抽稀，如果有 Padding 会保留最后一点）
            raw_motion = False
        elif len(detections) != len(self.last_detections):
            # 目标数量发生变化，必然是发生了运动或进出
            raw_motion = True
        else:
            # 目标数量相同，检查是否发生了位移 (IoU < 阈值)
            # 简化的贪心匹配：只要有任意一个当前目标无法在上一帧找到高 IoU 的同类目标，即视为运动
            for curr_det in detections:
                max_iou = 0.0
                for last_det in self.last_detections:
                    if curr_det.class_id == last_det.class_id:
                        iou = self._calculate_iou(curr_det.bbox, last_det.bbox)
                        if iou > max_iou:
                            max_iou = iou
                            
                if max_iou < self.iou_threshold:
                    raw_motion = True
                    break
                    
        # 更新记录
        self.last_detections = detections
        
        # 2. 去抖动 (Debounce)
        if raw_motion != self.is_motion:
            self.state_counter += 1
            if self.state_counter >= self.debounce_frames:
                self.is_motion = raw_motion
                self.state_counter = 0 # 重置计数器，状态正式切换
        else:
            self.state_counter = 0 # 维持原状态，清空动荡计数
            
        return self.is_motion
