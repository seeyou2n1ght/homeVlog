import json
import logging
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

import psutil

logger = logging.getLogger("homevlog")


@dataclass(eq=False)
class StageStats:
    name: str
    start_ts: float
    end_ts: float = 0.0
    cpu_pct: list[float] = field(default_factory=list)
    # Per-GPU metrics: list of lists, one per GPU
    gpu_load: list[list[int]] = field(default_factory=list)
    gpu_mem: list[list[int]] = field(default_factory=list)
    gpu_enc_load: list[list[int]] = field(default_factory=list)
    gpu_dec_load: list[list[int]] = field(default_factory=list)
    ram_pct: list[float] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts

    @property
    def avg_cpu(self) -> float:
        return sum(self.cpu_pct) / len(self.cpu_pct) if self.cpu_pct else 0.0

    @property
    def avg_ram(self) -> float:
        return sum(self.ram_pct) / len(self.ram_pct) if self.ram_pct else 0.0

    def gpu_summary(self, gpu_idx: int = 0) -> dict:
        """Summary for a specific GPU."""
        loads = self.gpu_load[gpu_idx] if gpu_idx < len(self.gpu_load) else []
        mems = self.gpu_mem[gpu_idx] if gpu_idx < len(self.gpu_mem) else []
        enc = self.gpu_enc_load[gpu_idx] if gpu_idx < len(self.gpu_enc_load) else []
        dec = self.gpu_dec_load[gpu_idx] if gpu_idx < len(self.gpu_dec_load) else []
        return {
            "avg_load": sum(loads) / len(loads) if loads else 0.0,
            "peak_mem_mb": max(mems) if mems else 0,
            "avg_enc": sum(enc) / len(enc) if enc else 0.0,
            "avg_dec": sum(dec) / len(dec) if dec else 0.0,
        }


class Monitor:
    def __init__(self, interval: float | None = None):
        if interval is None:
            from src.utils import load_config
            interval = load_config().get("logging", {}).get("monitor_interval", 2.0)
        self.interval = interval
        self._stages: list[StageStats] = []
        self._active_stages = set()
        self._active_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu_available = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu_handles = []
            count = pynvml.nvmlDeviceGetCount()
            gpu_names = []
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                self._gpu_handles.append(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode()
                gpu_names.append(name)
            self._gpu_available = len(self._gpu_handles) > 0
            if self._gpu_available:
                logger.info("Hardware: CPU=%d cores, RAM=%.1fGB, GPU=[%s]", 
                            psutil.cpu_count(logical=True), 
                            psutil.virtual_memory().total / (1024**3),
                            ", ".join(gpu_names))
        except Exception:
            self._nvml = None
            self._gpu_handles = []
        self._num_gpus = len(self._gpu_handles)

    def _sample(self):
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory().percent
        per_gpu_load: list[int] = []
        per_gpu_mem: list[int] = []
        per_gpu_enc: list[int] = []
        per_gpu_dec: list[int] = []
        for h in self._gpu_handles:
            try:
                util = self._nvml.nvmlDeviceGetUtilizationRates(h)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(h)
                per_gpu_load.append(util.gpu)
                per_gpu_mem.append(int(mem.used / 1024 / 1024))
            except Exception:
                per_gpu_load.append(0)
                per_gpu_mem.append(0)
            try:
                enc_util, _ = self._nvml.nvmlDeviceGetEncoderUtilization(h)
                per_gpu_enc.append(enc_util)
            except Exception:
                per_gpu_enc.append(0)
            try:
                dec_util, _ = self._nvml.nvmlDeviceGetDecoderUtilization(h)
                per_gpu_dec.append(dec_util)
            except Exception:
                per_gpu_dec.append(0)
        return cpu, ram, per_gpu_load, per_gpu_mem, per_gpu_enc, per_gpu_dec

    def _ensure_gpu_lists(self, s: StageStats):
        """Ensure per-GPU lists have correct length."""
        while len(s.gpu_load) < self._num_gpus:
            s.gpu_load.append([])
        while len(s.gpu_mem) < self._num_gpus:
            s.gpu_mem.append([])
        while len(s.gpu_enc_load) < self._num_gpus:
            s.gpu_enc_load.append([])
        while len(s.gpu_dec_load) < self._num_gpus:
            s.gpu_dec_load.append([])

    def _poll_loop(self):
        while not self._stop_event.is_set():
            cpu, ram, loads, mems, encs, decs = self._sample()
            with self._active_lock:
                for s in self._active_stages:
                    s.cpu_pct.append(cpu)
                    s.ram_pct.append(ram)
                    self._ensure_gpu_lists(s)
                    for i in range(self._num_gpus):
                        s.gpu_load[i].append(loads[i])
                        s.gpu_mem[i].append(mems[i])
                        s.gpu_enc_load[i].append(encs[i])
                        s.gpu_dec_load[i].append(decs[i])
            self._stop_event.wait(self.interval)

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self._thread = None

    def shutdown(self):
        """Stop polling and release NVML resources."""
        self.stop()
        try:
            if self._nvml:
                self._nvml.nvmlShutdown()
        except Exception:
            pass

    @contextmanager
    def stage(self, name: str):
        s = StageStats(name=name, start_ts=time.monotonic())
        with self._active_lock:
            self._active_stages.add(s)
            self._stages.append(s)
        logger.info("stage [%s] start", name)
        try:
            yield s
        finally:
            s.end_ts = time.monotonic()
            with self._active_lock:
                self._active_stages.discard(s)
            gpu0 = s.gpu_summary(0)
            logger.info(
                "stage [%s] done in %.1fs, cpu=%.1f%% ram=%.1f%% gpu0: load=%d%% enc=%d%% dec=%d%% mem=%dMB",
                name, s.duration, s.avg_cpu, s.avg_ram,
                int(gpu0["avg_load"]), int(gpu0["avg_enc"]), int(gpu0["avg_dec"]), gpu0["peak_mem_mb"],
            )

    def summary(self) -> str:
        with self._active_lock:
            stages = list(self._stages)
            
        lines = ["--- Monitor Summary ---"]
        total = 0.0
        for s in stages:
            total += s.duration
            parts = [
                f"  [{s.name}] {s.duration:.1f}s  cpu={s.avg_cpu:.1f}%  ram={s.avg_ram:.1f}%"
            ]
            for i in range(self._num_gpus):
                g = s.gpu_summary(i)
                parts.append(
                    f"  gpu{i}: load={g['avg_load']:.0f}% enc={g['avg_enc']:.0f}% "
                    f"dec={g['avg_dec']:.0f}% mem_peak={g['peak_mem_mb']}MB"
                )
            lines.append(" | ".join(parts))
        lines.append(f"  TOTAL: {total:.1f}s")
        return "\n".join(lines)

    def stages_data(self) -> list[dict]:
        """Return structured stage data for JSON serialization."""
        with self._active_lock:
            stages = list(self._stages)
            
        result = []
        for s in stages:
            entry = {
                "name": s.name,
                "duration": round(s.duration, 2),
                "avg_cpu": round(s.avg_cpu, 1),
                "avg_ram": round(s.avg_ram, 1),
            }
            for i in range(self._num_gpus):
                g = s.gpu_summary(i)
                entry[f"gpu{i}"] = {k: round(v, 1) for k, v in g.items()}
            result.append(entry)
        return result



# ---------------------------------------------------------------------------
# PerfCollector — per-operation structured metrics
# ---------------------------------------------------------------------------

@dataclass
class PerfRecord:
    stage: str          # "prescreen" | "analysis" | "render"
    file: str           # short filename
    gpu: str            # "cuda" | "qsv" | "nv" | "qsv_enc" | "cpu"
    duration: float     # seconds
    frames: int = 0
    fps: float = 0.0
    extra: dict = field(default_factory=dict)


class PerfCollector:
    """Lightweight per-operation metrics collector. Thread-safe."""

    def __init__(self):
        self._records: list[dict] = []
        self._lock = threading.Lock()

    def add(self, record: PerfRecord):
        with self._lock:
            self._records.append(asdict(record))

    def dump(self, path: Path, metadata: dict | None = None):
        """Write all records + optional metadata to JSON."""
        data = metadata or {}
        with self._lock:
            data["records"] = list(self._records)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("perf data written: %s (%d records)", path, len(data["records"]))

    def summary_by_stage(self) -> dict[str, dict]:
        """Group records by stage, compute stats."""
        by_stage: dict[str, list[float]] = {}
        with self._lock:
            for r in self._records:
                stage = r.get("stage", "unknown")
                by_stage.setdefault(stage, []).append(r.get("duration", 0))
        result = {}
        for stage, durs in by_stage.items():
            durs_sorted = sorted(durs)
            n = len(durs_sorted)
            result[stage] = {
                "count": n,
                "total": round(sum(durs_sorted), 2),
                "avg": round(sum(durs_sorted) / n, 2) if n else 0,
                "p50": round(durs_sorted[n // 2], 2) if n else 0,
                "p95": round(durs_sorted[int(n * 0.95)], 2) if n else 0,
                "min": round(durs_sorted[0], 2) if n else 0,
                "max": round(durs_sorted[-1], 2) if n else 0,
            }
        return result

    def reset(self):
        with self._lock:
            self._records.clear()


_monitor: Monitor | None = None
_perf: PerfCollector | None = None


def get_monitor() -> Monitor:
    global _monitor
    if _monitor is None:
        _monitor = Monitor()
    return _monitor


def get_perf() -> PerfCollector:
    global _perf
    if _perf is None:
        _perf = PerfCollector()
    return _perf
