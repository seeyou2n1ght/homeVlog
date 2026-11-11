import json
import re
import subprocess
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import logging
import psutil

logger = logging.getLogger("homevlog")
@dataclass(eq=False)
class StageStats:
    name: str
    start_ts: float
    end_ts: float = 0.0
    include_in_total: bool = True
    # O(1) streaming scalar accumulators
    count: int = 0
    sum_cpu: float = 0.0
    peak_cpu: float = 0.0
    sum_ram: float = 0.0
    peak_ram: float = 0.0
    gpu_stats: list[dict[str, Any]] = field(default_factory=list)
    # Backward-compatibility fallback fields
    cpu_pct: list[float] = field(default_factory=list)
    ram_pct: list[float] = field(default_factory=list)
    gpu_load: list[list[int]] = field(default_factory=list)
    gpu_mem: list[list[int]] = field(default_factory=list)
    gpu_enc_load: list[list[int]] = field(default_factory=list)
    gpu_dec_load: list[list[int]] = field(default_factory=list)
    # Intel iGPU engine utilization: {"3D": [...], "Video Decode": [...], ...}
    igpu_engine: dict[str, list[float]] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.end_ts - self.start_ts

    @property
    def avg_cpu(self) -> float:
        if self.count > 0:
            return round(self.sum_cpu / self.count, 1)
        if self.cpu_pct:
            return round(sum(self.cpu_pct) / len(self.cpu_pct), 1)
        return 0.0

    @property
    def avg_ram(self) -> float:
        if self.count > 0:
            return round(self.sum_ram / self.count, 1)
        if self.ram_pct:
            return round(sum(self.ram_pct) / len(self.ram_pct), 1)
        return 0.0

    def record_sample(
        self,
        cpu: float = 0.0,
        ram: float = 0.0,
        loads: list[int] | None = None,
        mems: list[int] | None = None,
        encs: list[int] | None = None,
        decs: list[int] | None = None,
        igpu: dict[str, float] | None = None,
    ) -> None:
        """Stream a single hardware sample in O(1) time and memory."""
        self.count += 1
        self.sum_cpu += cpu
        if cpu > self.peak_cpu:
            self.peak_cpu = cpu
        self.sum_ram += ram
        if ram > self.peak_ram:
            self.peak_ram = ram
        if loads:
            while len(self.gpu_stats) < len(loads):
                self.gpu_stats.append({
                    "count": 0, "sum_load": 0.0, "peak_mem_mb": 0, "sum_enc": 0.0, "sum_dec": 0.0
                })
            for i, load in enumerate(loads):
                g = self.gpu_stats[i]
                g["count"] += 1
                g["sum_load"] += load
                if mems and i < len(mems) and mems[i] > g["peak_mem_mb"]:
                    g["peak_mem_mb"] = mems[i]
                if encs and i < len(encs):
                    g["sum_enc"] += encs[i]
                if decs and i < len(decs):
                    g["sum_dec"] += decs[i]
        if igpu:
            for eng, val in igpu.items():
                self.igpu_engine.setdefault(eng, []).append(val)

    def gpu_summary(self, gpu_idx: int = 0) -> dict:
        """Summary for a specific GPU with zero-division protection."""
        if gpu_idx < len(self.gpu_stats):
            g = self.gpu_stats[gpu_idx]
            cnt = g.get("count", 0)
            if cnt > 0:
                return {
                    "avg_load": round(g["sum_load"] / cnt, 1),
                    "peak_mem_mb": int(g["peak_mem_mb"]),
                    "avg_enc": round(g["sum_enc"] / cnt, 1),
                    "avg_dec": round(g["sum_dec"] / cnt, 1),
                }
        if gpu_idx < len(self.gpu_load) and self.gpu_load[gpu_idx]:
            loads = self.gpu_load[gpu_idx]
            mems = self.gpu_mem[gpu_idx] if gpu_idx < len(self.gpu_mem) else []
            enc = self.gpu_enc_load[gpu_idx] if gpu_idx < len(self.gpu_enc_load) else []
            dec = self.gpu_dec_load[gpu_idx] if gpu_idx < len(self.gpu_dec_load) else []
            return {
                "avg_load": round(sum(loads) / len(loads), 1) if loads else 0.0,
                "peak_mem_mb": int(max(mems)) if mems else 0,
                "avg_enc": round(sum(enc) / len(enc), 1) if enc else 0.0,
                "avg_dec": round(sum(dec) / len(dec), 1) if dec else 0.0,
            }
        return {
            "avg_load": 0.0,
            "peak_mem_mb": 0,
            "avg_enc": 0.0,
            "avg_dec": 0.0,
        }

    def igpu_summary(self) -> dict:
        """Summary for Intel iGPU engines with zero-division protection."""
        result = {}
        for engine, samples in self.igpu_engine.items():
            if samples:
                result[engine] = round(sum(samples) / len(samples), 1)
            else:
                result[engine] = 0.0
        return result


class Monitor:
    def __init__(self, interval: float | None = None):
        if interval is None:
            from src.core.config import load_config
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
                logger.info("Hardware: CPU=%s cores, RAM=%.1fGB, GPU=[%s]", 
                            psutil.cpu_count(logical=True), 
                            psutil.virtual_memory().total / (1024**3),
                            ", ".join(gpu_names))
        except Exception:
            self._nvml = None
            self._gpu_handles = []
        self._num_gpus = len(self._gpu_handles)

        # Intel iGPU 采样: 消除同步 PowerShell / typeperf 探测开销 (<2ms 极速初始化)
        self._igpu_counters: dict[str, str] = {}  # {"Video Decode": counter_path, ...}
        self._igpu_available = False

    def _sample(self):
        cpu = psutil.cpu_percent(interval=None)
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

    def _sample_igpu(self) -> dict[str, float]:
        """Intel iGPU 采样已下线，直接返回空字典以根除高频 typeperf.exe 子进程派生。"""
        return {}

    def _ensure_gpu_lists(self, s: StageStats):
        """Deprecated compatibility no-op; GPU stats are now dynamically sized in O(1)."""
        pass

    def _poll_loop(self):
        while not self._stop_event.is_set():
            cpu, ram, loads, mems, encs, decs = self._sample()
            with self._active_lock:
                for s in self._active_stages:
                    s.record_sample(cpu=cpu, ram=ram, loads=loads, mems=mems, encs=encs, decs=decs)
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
    def stage(self, name: str, *, include_in_total: bool = True):
        s = StageStats(name=name, start_ts=time.monotonic(), include_in_total=include_in_total)
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
            igpu = s.igpu_summary()
            igpu_str = ""
            if igpu:
                parts = [f"{k}={v:.0f}%" for k, v in igpu.items()]
                igpu_str = " | iGPU: " + " ".join(parts)
            logger.info(
                "stage [%s] done in %.1fs, cpu=%.1f%% ram=%.1f%% gpu0: load=%d%% enc=%d%% dec=%d%% mem=%dMB%s",
                name, s.duration, s.avg_cpu, s.avg_ram,
                int(gpu0["avg_load"]), int(gpu0["avg_enc"]), int(gpu0["avg_dec"]), gpu0["peak_mem_mb"],
                igpu_str,
            )

    def summary(self) -> str:
        with self._active_lock:
            stages = list(self._stages)
            
        lines = ["--- Monitor Summary ---"]
        total = 0.0
        for s in stages:
            if s.include_in_total:
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
                "include_in_total": s.include_in_total,
                "avg_cpu": round(s.avg_cpu, 1),
                "avg_ram": round(s.avg_ram, 1),
            }
            for i in range(self._num_gpus):
                g = s.gpu_summary(i)
                entry[f"gpu{i}"] = {k: round(v, 1) for k, v in g.items()}
            igpu = s.igpu_summary()
            if igpu:
                entry["igpu"] = igpu
            result.append(entry)
        return result



# ---------------------------------------------------------------------------
# PerfCollector — per-operation structured metrics
# ---------------------------------------------------------------------------

@dataclass
class PerfRecord:
    stage: str          # prescreen | analysis | render_stage | render_enc | render | final_concat
    file: str           # short filename
    gpu: str            # cuda | qsv | nv | qsv_enc | cpu
    duration: float     # seconds
    frames: int = 0
    fps: float = 0.0
    extra: dict = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0
    worker: str = ""


class PerfCollector:
    """Lightweight per-operation metrics collector. Thread-safe."""

    def __init__(self):
        self._records: list[dict] = []
        self._lock = threading.Lock()

    def add(self, record: PerfRecord):
        with self._lock:
            self._records.append(asdict(record))

    def dump(self, path: Path, metadata: dict | None = None):
        """Write all records + optional metadata to JSON via atomic file replacement."""
        data = metadata or {}
        with self._lock:
            data["records"] = list(self._records)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            tmp_path.replace(path)
        finally:
            tmp_path.unlink(missing_ok=True)
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

    def yolo_summary(self) -> dict:
        """Compute aggregate YOLO verification statistics across analysis records."""
        with self._lock:
            records = list(self._records)
        yolo_recs = [r for r in records if r.get("stage") == "analysis" and "yolo_frames" in r.get("extra", {})]
        if not yolo_recs:
            return {}
        total_duration = sum(r["extra"].get("yolo_duration", 0.0) for r in yolo_recs)
        total_lock_wait = sum(r["extra"].get("yolo_lock_wait", 0.0) for r in yolo_recs)
        total_infer = sum(r["extra"].get("yolo_infer_time", 0.0) for r in yolo_recs)
        total_frames = sum(r["extra"].get("yolo_frames", 0) for r in yolo_recs)
        candidate_segs = sum(r["extra"].get("yolo_candidate_segments", 0) for r in yolo_recs)
        confirmed_segs = sum(r["extra"].get("yolo_confirmed_segments", 0) for r in yolo_recs)
        rejected_segs = sum(r["extra"].get("yolo_rejected_segments", 0) for r in yolo_recs)
        return {
            "files_evaluated": len(yolo_recs),
            "total_duration_s": round(total_duration, 2),
            "total_lock_wait_s": round(total_lock_wait, 2),
            "total_infer_s": round(total_infer, 2),
            "total_frames": total_frames,
            "avg_infer_ms_per_frame": round((total_infer * 1000 / total_frames), 2) if total_frames > 0 else 0.0,
            "candidate_segments": candidate_segs,
            "confirmed_segments": confirmed_segs,
            "rejected_segments": rejected_segs,
            "suppression_rate_pct": round(rejected_segs / max(1, candidate_segs) * 100, 1),
        }

    def wait_summary(self) -> dict:
        """Aggregate queue and hardware wait fields without mixing them into work time."""
        with self._lock:
            records = list(self._records)
        fields = ("analysis_queue_wait_s", "render_queue_wait_s", "sem_wait")
        result = {}
        for field in fields:
            values = [float(r.get("extra", {}).get(field, 0.0) or 0.0) for r in records]
            if values:
                result[field] = {
                    "count": sum(v > 0 for v in values),
                    "total": round(sum(values), 3),
                    "max": round(max(values), 3),
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


# ---------------------------------------------------------------------------
# Intel iGPU Performance Counter Discovery (Windows only) - Deprecated
# ---------------------------------------------------------------------------


def _discover_intel_gpu_counters() -> dict[str, str]:
    """Deprecated: Intel iGPU counter discovery removed to eliminate startup delay."""
    return {}
