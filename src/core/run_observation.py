"""In-process display and log context for one date/camera execution."""
from contextvars import ContextVar
import logging
from threading import Lock
import time
import uuid

current_run = ContextVar("homevlog_run", default=None)
current_batch = ContextVar("homevlog_batch", default=None)
logger = logging.getLogger("homevlog")


class RunObservation:
    def __init__(self, date, cam):
        self.identity = {"run_id": uuid.uuid4().hex, "date": date, "cam": cam}
        self.phase = "流水处理"
        self.dashboard = None
        self.reason = None
        self.committed = False
        self._lock = Lock()
        self._batches = {}

    def batch(self, index, state, **fields):
        with self._lock:
            item = self._batches.setdefault(index, {})
            if state == "ready" and item.get("state") in ("reused", "encoded"):
                return
            if state == "running":
                item.clear()
                item["started"] = time.monotonic()
            item.update(state=state, **fields)

    def note(self, index, message, *, new_attempt=False):
        with self._lock:
            if index in self._batches:
                self._batches[index]["note"] = message
                if new_attempt:
                    self._batches[index].pop("progress_at", None)

    def progress(self, index, frames):
        with self._lock:
            item = self._batches.get(index)
            if item and item["state"] == "running":
                item.update(frames=frames, progress_at=time.monotonic())

    def snapshot(self):
        with self._lock:
            counts = {}
            active = []
            for index, item in self._batches.items():
                state = item["state"]
                counts[state] = counts.get(state, 0) + 1
                if state == "running":
                    active.append((index, item.copy()))
        return counts, sorted(active, key=lambda pair: pair[1]["started"])

    def totals(self):
        counts, _ = self.snapshot()
        return {key: counts.get(key, 0) for key in ("encoded", "reused", "skipped", "failed", "ready")}

    def set_phase(self, phase):
        self.phase = phase
        logger.info("%s", phase, extra={"event": "phase_changed", "phase": phase})
        if self.dashboard is not None:
            self.dashboard.update_view(force=True)


def log_context():
    run = current_run.get()
    fields = dict(run.identity) if run else {}
    batch = current_batch.get()
    if batch is not None:
        fields.update(batch=batch[0], worker=batch[1])
    return fields
