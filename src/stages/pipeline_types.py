"""Pipeline message queue definitions and strongly-typed task structures."""
from dataclasses import dataclass, field
import queue
import time
from typing import Any


@dataclass
class PipelineTask:
    """Strongly-typed task contract flowing through prescreen, analysis, and render."""
    filepath: str
    cam_index: int = 0
    date: str = ""
    file_start_time: str = ""
    file_end_time: str = ""
    file_duration: float = 0.0
    duration_verified: int = 0
    has_audio: int = 0
    prescreen_status: str = "PENDING"
    prescreen_result: str = ""
    analysis_status: str = "PENDING"
    segments: list = field(default_factory=list)
    id: int | None = None
    is_audio_gated_static: bool = False
    audio_events: list = field(default_factory=list)
    is_stolen: bool = False

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_dict(cls, data) -> "PipelineTask":
        if isinstance(data, cls):
            return data
        if not isinstance(data, dict):
            return data
        return cls(
            filepath=str(data.get("filepath", "")),
            cam_index=int(data.get("cam_index", 0)),
            date=str(data.get("date", "")),
            file_start_time=str(data.get("file_start_time", "")),
            file_end_time=str(data.get("file_end_time", "")),
            file_duration=float(data.get("file_duration") or 0.0),
            duration_verified=int(data.get("duration_verified") or 0),
            has_audio=int(data.get("has_audio") or 0),
            prescreen_status=str(data.get("prescreen_status", "PENDING")),
            prescreen_result=str(data.get("prescreen_result") or ""),
            analysis_status=str(data.get("analysis_status", "PENDING")),
            segments=list(data.get("segments") or []),
            id=data.get("id"),
            is_audio_gated_static=bool(data.get("is_audio_gated_static", False)),
            audio_events=list(data.get("audio_events") or []),
            is_stolen=bool(data.get("is_stolen", False)),
        )


class AnalysisQueue(queue.Queue):
    """Dual-ended priority queue for detailed analysis.

    Supports CUDA analysis worker consuming heaviest tasks first (LPT),
    while QSV analysis workers consume lightest tasks first (SJF / SPT).
    """
    def _init(self, maxsize):
        self.queue = []
        self.sequence = 0
        self.cost_priority = False

    def _put(self, task):
        import heapq
        dur = float(task.get("file_duration") or 0.0)
        if self.cost_priority:
            # SJF: 短作业优先（时长升序），让 300s 标准切片以秒级出清迅速喂饱渲染编队
            priority = (dur, task.get("file_start_time") or "")
        else:
            priority = (0.0, task.get("file_start_time") or "")
        heapq.heappush(self.queue, (priority, self.sequence, task))
        self.sequence += 1

    def _get(self):
        import heapq
        return heapq.heappop(self.queue)[2]

    def _get_heavy(self, min_duration=600.0):
        import heapq
        if not self.queue:
            raise queue.Empty
        heavy_idx = None
        max_dur = -1.0
        for i, (_prio, _seq, task) in enumerate(self.queue):
            dur = float(task.get("file_duration") or 0.0)
            if dur >= min_duration and dur > max_dur:
                max_dur = dur
                heavy_idx = i
        if heavy_idx is not None:
            _, _, task = self.queue.pop(heavy_idx)
            heapq.heapify(self.queue)
            return task
        # 当队列中已无重文件时，Worker 0 协助出清最短轻文件
        return heapq.heappop(self.queue)[2]

    def get_heavy(self, block=True, timeout=None, min_duration=600.0):
        with self.not_empty:
            if not block:
                if not self._qsize():
                    raise queue.Empty
                return self._get_heavy(min_duration)
            elif timeout is None:
                while not self._qsize():
                    self.not_empty.wait()
                return self._get_heavy(min_duration)
            elif timeout < 0:
                raise ValueError("'timeout' must be a non-negative number")
            else:
                endtime = time.monotonic() + timeout
                while not self._qsize():
                    remaining = endtime - time.monotonic()
                    if remaining <= 0.0:
                        raise queue.Empty
                    self.not_empty.wait(remaining)
                return self._get_heavy(min_duration)

    def has_heavy_tasks(self, min_duration: float = 1200.0) -> bool:
        """Check whether any task with duration >= min_duration remains queued.

        Thread-safe non-destructive inspection under self.mutex.
        """
        with self.mutex:
            return any(
                float(task.get("file_duration") or 0.0) >= min_duration
                for _prio, _seq, task in self.queue
            )

    def steal_sublongest(
        self,
        max_duration: float = 1200.0,
        min_duration: float = 300.0,
    ) -> Any | None:
        """Dynamically steal the sub-longest ('次长切片') task from the analysis queue.

        Thread-safe under self.mutex.
        Scans queued tasks and finds the candidate whose file_duration satisfies:
            min_duration <= dur < max_duration
        with the maximum duration among candidates.

        Tie-breaking rule:
            1. Higher duration (primary)
            2. Earlier file_start_time (secondary, chronological order)
            3. Lower sequence number (tertiary, FIFO insertion order)

        If a qualifying candidate is found:
            - Removes the element from self.queue via pop(index)
            - Immediately restores the min-heap invariant with heapq.heapify(self.queue)
            - Wakes any producer waiting on queue space via self.not_full.notify()
            - Stamps task with is_stolen = True
            - Returns the stolen task (CRITICAL: does NOT call task_done()!)

        If the queue is empty or no task meets the criteria, returns None
        without corrupting the heap or raising exceptions.
        """
        import heapq
        with self.mutex:
            if not self.queue:
                return None

            best_idx = None
            best_dur = -1.0
            best_start_time = None
            best_seq = None

            for i, (_prio, seq, task) in enumerate(self.queue):
                dur = float(task.get("file_duration") or 0.0)
                if min_duration <= dur < max_duration:
                    start_time = str(task.get("file_start_time") or "")
                    is_better = False
                    if best_idx is None:
                        is_better = True
                    elif dur > best_dur:
                        is_better = True
                    elif dur == best_dur:
                        if start_time < best_start_time:
                            is_better = True
                        elif start_time == best_start_time and seq < best_seq:
                            is_better = True

                    if is_better:
                        best_idx = i
                        best_dur = dur
                        best_start_time = start_time
                        best_seq = seq

            if best_idx is None:
                return None

            _, _, task = self.queue.pop(best_idx)
            heapq.heapify(self.queue)
            self.not_full.notify()

            try:
                task["is_stolen"] = True
            except (TypeError, KeyError):
                pass
            try:
                setattr(task, "is_stolen", True)
            except Exception:
                pass

            return task


class RenderMessageQueue(queue.Queue):
    """Render readiness queue that stamps producer time for queue-wait metrics."""

    def put(self, item, *args, **kwargs):
        if isinstance(item, dict):
            item.setdefault("_render_ready_at", time.monotonic())
        return super().put(item, *args, **kwargs)
