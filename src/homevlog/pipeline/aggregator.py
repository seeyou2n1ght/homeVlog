from typing import List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TimeSegment:
    start_pts: float
    end_pts: float
    is_motion: bool

class Aggregator:
    """
    时间序列聚合器
    职责：逻辑分段、脉冲抽稀、区间合并。
    """
    def __init__(self, static_interval_ms: int = 60000, padding_ms: int = 2500):
        self.static_interval_sec = static_interval_ms / 1000.0
        self.padding_sec = padding_ms / 1000.0
        self.segments: List[TimeSegment] = []
        self._current_segment: Optional[TimeSegment] = None
        self._last_pulse_pts = -99999.0

    def add_frame_state(self, pts: float, is_active: bool):
        """添加一帧的状态判定"""
        # 1. 初始化第一个片段
        if self._current_segment is None:
            self._current_segment = TimeSegment(pts, pts, is_active)
            return

        # 2. 如果状态发生切换，封存旧片段，开启新片段
        if is_active != self._current_segment.is_motion:
            self._current_segment.end_pts = pts
            self.segments.append(self._current_segment)
            self._current_segment = TimeSegment(pts, pts, is_active)
        else:
            # 持续状态
            self._current_segment.end_pts = pts

    def finalize(self) -> List[Tuple[float, float]]:
        """
        生成最终的切割时间轴任务。
        包含脉冲抽稀和区间合并逻辑。
        """
        if self._current_segment:
            self.segments.append(self._current_segment)
            self._current_segment = None

        raw_cuts = []
        for seg in self.segments:
            if seg.is_motion:
                # 运动状态：全段保留
                raw_cuts.append((seg.start_pts, seg.end_pts))
            else:
                # 静止状态：应用脉冲抽稀 (每 static_interval_sec 保留一帧心跳)
                curr_pts = seg.start_pts
                while curr_pts <= seg.end_pts:
                    if curr_pts - self._last_pulse_pts >= self.static_interval_sec:
                        # 保留一个极短的片段 (如 1s)
                        raw_cuts.append((curr_pts, min(curr_pts + 1.0, seg.end_pts)))
                        self._last_pulse_pts = curr_pts
                        curr_pts += self.static_interval_sec  # 优化：直接跳到下一个脉冲点
                    else:
                        curr_pts = self._last_pulse_pts + self.static_interval_sec

        # 3. 增加 Padding 并合并重叠区间
        return self._merge_intervals(raw_cuts)

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """核心：区间合并算法 (处理盲点 7.2)"""
        if not intervals:
            return []
            
        # 先应用 Padding
        padded = []
        for start, end in intervals:
            # 这里的 Padding 要防止 start 小于 0
            padded.append((max(0.0, start - self.padding_sec), end + self.padding_sec))
            
        # 按开始时间排序
        padded.sort(key=lambda x: x[0])
        
        merged = [list(padded[0])]
        for current in padded[1:]:
            last_merged = merged[-1]
            if current[0] <= last_merged[1]:
                # 有重叠，合并
                last_merged[1] = max(last_merged[1], current[1])
            else:
                merged.append(list(current))
                
        return [(m[0], m[1]) for m in merged]
