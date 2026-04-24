from typing import List, Optional, Set
from dataclasses import dataclass, field
import math
from ..hal.base import DetectionResult

@dataclass
class FrameState:
    is_motion: bool
    classes: Set[str] = field(default_factory=set)
    avg_distance: Optional[float] = None

class SimpleTracker:
    """
    轻量级追踪与去抖动器 (V2.4 异构版)
    职责：过滤瞬时误报，平滑运动/静止状态切换，并提取陪伴距离等业务特征。
    """
    def __init__(self, iou_threshold: float = 0.80, debounce_frames: int = 5, frame_width: float = 640.0, frame_height: float = 640.0):
        self.iou_threshold = iou_threshold
        self.debounce_frames = debounce_frames
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        self.last_detections: List[DetectionResult] = []
        
        # 当前平滑后的状态 (True: 运动期, False: 静止期或无目标)
        self.is_motion = False 
        self.state_counter = 0

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

    def _get_center(self, bbox: List[float]) -> tuple[float, float]:
        return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)

    def _calculate_distance(self, detections: List[DetectionResult]) -> Optional[float]:
        persons = [d for d in detections if d.label == 'person']
        babies = [d for d in detections if d.label == 'baby']
        
        if not persons or not babies:
            return None
            
        # 简单处理：取最近的人和宝宝的距离
        min_dist = float('inf')
        for p in persons:
            pc = self._get_center(p.bbox)
            for b in babies:
                bc = self._get_center(b.bbox)
                # 归一化欧氏距离 (基于画面对角线或宽度)
                # 这里简单除以画面宽度进行归一化
                dist = math.sqrt((pc[0] - bc[0])**2 + (pc[1] - bc[1])**2) / self.frame_width
                if dist < min_dist:
                    min_dist = dist
                    
        return min_dist

    def update(self, detections: List[DetectionResult]) -> FrameState:
        """
        更新追踪状态。
        返回：富状态对象 FrameState
        """
        raw_motion = False
        
        if len(detections) == 0:
            raw_motion = False
        elif len(detections) != len(self.last_detections):
            raw_motion = True
        else:
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
                    
        self.last_detections = detections
        
        if raw_motion != self.is_motion:
            self.state_counter += 1
            if self.state_counter >= self.debounce_frames:
                self.is_motion = raw_motion
                self.state_counter = 0
        else:
            self.state_counter = 0
            
        # 提取业务特征
        classes = {d.label for d in detections}
        avg_distance = self._calculate_distance(detections)
            
        return FrameState(
            is_motion=self.is_motion,
            classes=classes,
            avg_distance=avg_distance
        )
