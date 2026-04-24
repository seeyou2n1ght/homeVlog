from typing import List, Tuple, Optional, Set
from dataclasses import dataclass, field
from .tracker import FrameState

@dataclass
class TimeSegment:
    start_pts: float
    end_pts: float
    is_motion: bool
    classes: Set[str] = field(default_factory=set)
    distances: List[float] = field(default_factory=list)

class Aggregator:
    """
    时间序列聚合器 (V2.4 异构版)
    职责：逻辑分段、脉冲抽稀、区间合并，并聚合分类和距离特征。
    """
    def __init__(self, static_interval_ms: int = 60000, padding_ms: int = 2500):
        self.static_interval_sec = static_interval_ms / 1000.0
        self.padding_sec = padding_ms / 1000.0
        self.segments: List[TimeSegment] = []
        self._current_segment: Optional[TimeSegment] = None
        self._last_pulse_pts = -99999.0

    def add_frame_state(self, pts: float, state: FrameState):
        """添加一帧的富状态判定"""
        # 1. 初始化第一个片段
        if self._current_segment is None:
            self._current_segment = TimeSegment(
                start_pts=pts, end_pts=pts, is_motion=state.is_motion,
                classes=set(state.classes), distances=[state.avg_distance] if state.avg_distance is not None else []
            )
            return

        # 2. 如果状态发生切换，封存旧片段，开启新片段
        if state.is_motion != self._current_segment.is_motion:
            self._current_segment.end_pts = pts
            self.segments.append(self._current_segment)
            self._current_segment = TimeSegment(
                start_pts=pts, end_pts=pts, is_motion=state.is_motion,
                classes=set(state.classes), distances=[state.avg_distance] if state.avg_distance is not None else []
            )
        else:
            # 持续状态，累加特征
            self._current_segment.end_pts = pts
            self._current_segment.classes.update(state.classes)
            if state.avg_distance is not None:
                self._current_segment.distances.append(state.avg_distance)

    def finalize(self) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float, str, float]]]:
        """
        生成最终的切割时间轴任务，以及业务事件统计。
        返回: 
        - raw_cuts: List[(start, end)] 用于物理切割
        - events: List[(start, end, classes_str, avg_distance)] 用于写入 analytics_events
        """
        if self._current_segment:
            self.segments.append(self._current_segment)
            self._current_segment = None

        raw_cuts = []
        events = []
        
        for seg in self.segments:
            classes_str = ",".join(sorted(list(seg.classes))) if seg.classes else ""
            avg_dist = sum(seg.distances) / len(seg.distances) if seg.distances else -1.0
            
            if seg.is_motion:
                raw_cuts.append((seg.start_pts, seg.end_pts))
                if classes_str:  # 只记录有目标的事件
                    events.append((seg.start_pts, seg.end_pts, classes_str, avg_dist))
            else:
                curr_pts = seg.start_pts
                while curr_pts <= seg.end_pts:
                    if curr_pts - self._last_pulse_pts >= self.static_interval_sec:
                        pulse_end = min(curr_pts + 1.0, seg.end_pts)
                        raw_cuts.append((curr_pts, pulse_end))
                        if classes_str:
                            events.append((curr_pts, pulse_end, classes_str, avg_dist))
                        self._last_pulse_pts = curr_pts
                        curr_pts += self.static_interval_sec
                    else:
                        curr_pts = self._last_pulse_pts + self.static_interval_sec
                    if self.static_interval_sec <= 0: break
        
        merged_cuts = self._merge_intervals(raw_cuts)
        return merged_cuts, events

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        if not intervals:
            return []
            
        padded = []
        for start, end in intervals:
            padded.append((max(0.0, start - self.padding_sec), end + self.padding_sec))
            
        padded.sort(key=lambda x: x[0])
        
        merged = [list(padded[0])]
        for current in padded[1:]:
            last_merged = merged[-1]
            if current[0] <= last_merged[1]:
                last_merged[1] = max(last_merged[1], current[1])
            else:
                merged.append(list(current))
                
        return [(m[0], m[1]) for m in merged]
