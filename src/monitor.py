import json
import re
import subprocess
import time
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from pathlib import Path

import logging
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
    # Intel iGPU engine utilization: {"3D": [...], "Video Decode": [...], ...}
    igpu_engine: dict[str, list[float]] = field(default_factory=dict)

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

    def igpu_summary(self) -> dict:
        """Summary for Intel iGPU engines."""
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
                logger.info("Hardware: CPU=%s cores, RAM=%.1fGB, GPU=[%s]", 
                            psutil.cpu_count(logical=True), 
                            psutil.virtual_memory().total / (1024**3),
                            ", ".join(gpu_names))
        except Exception:
            self._nvml = None
            self._gpu_handles = []
        self._num_gpus = len(self._gpu_handles)

        # Intel iGPU 采样: 通过 Windows Performance Counter 获取引擎利用率
        self._igpu_counters: dict[str, str] = {}  # {"Video Decode": counter_path, ...}
        self._igpu_available = False
        try:
            self._igpu_counters = _discover_intel_gpu_counters()
            if self._igpu_counters:
                self._igpu_available = True
                engines = ", ".join(self._igpu_counters.keys())
                logger.info("Intel iGPU monitor: discovered engines [%s]", engines)
        except Exception as e:
            logger.debug("Intel iGPU counter discovery failed: %s", e)

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

    def _sample_igpu(self) -> dict[str, float]:
        """采样 Intel iGPU 各引擎利用率 (%)，通过 Windows typeperf 单次读取。"""
        if not self._igpu_available:
            return {}
        try:
            counters = list(self._igpu_counters.values())
            cmd = ["typeperf"] + counters + ["-sc", "1", "-y"]
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
            )
            if result.returncode != 0:
                return {}
            # typeperf 输出格式: 第 2 行为列头，第 3 行为数据
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
            data_line = None
            for line in lines:
                if line.startswith('"') and not line.startswith('\"\\\\'):
                    # 数据行以时间戳开头 (如 "09/02/2026 14:55:00.000")
                    data_line = line
                    break
            if not data_line:
                return {}
            parts = [p.strip().strip('"') for p in data_line.split('","')]
            # parts[0] 是时间戳，parts[1:] 是各 counter 的值
            values = parts[1:]
            engine_names = list(self._igpu_counters.keys())
            result_dict: dict[str, float] = {}
            for i, name in enumerate(engine_names):
                if i < len(values):
                    try:
                        result_dict[name] = max(0.0, float(values[i]))
                    except (ValueError, TypeError):
                        result_dict[name] = 0.0
            return result_dict
        except Exception:
            return {}

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
            igpu_data = self._sample_igpu()
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
                    for engine, val in igpu_data.items():
                        s.igpu_engine.setdefault(engine, []).append(val)
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


# ---------------------------------------------------------------------------
# Intel iGPU Performance Counter Discovery (Windows only)
# ---------------------------------------------------------------------------

# Windows GPU Engine counter 格式:
# \GPU Engine(pid_0_luid_0x00000000_0xXXXX_phys_0_eng_N_engtype_XXX)\Utilization Percentage
# 其中 engtype 为 3D / VideoDecode / VideoEncode / Copy 等
_ENGTYPE_PATTERN = re.compile(r"engtype_(\w+)")
_LUID_PATTERN = re.compile(r"luid_0x[0-9a-fA-F]+_0x([0-9a-fA-F]+)")


def _discover_intel_gpu_counters() -> dict[str, str]:
    """通过 typeperf 枚举 Windows GPU Engine counter，识别 Intel iGPU 的引擎。

    返回 {"Video Decode": "\\\\GPU Engine(...)\\\\Utilization Percentage", ...}
    """
    import sys
    if sys.platform != "win32":
        return {}

    # 第一步: 获取 GPU 适配器描述，找到 Intel 设备的 LUID
    intel_luids: set[str] = set()
    try:
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_VideoController | Select-Object Name, PNPDeviceID | ConvertTo-Json -Compress"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
        )
        if result.returncode == 0 and result.stdout.strip():
            import json as _json
            adapters = _json.loads(result.stdout)
            if isinstance(adapters, dict):
                adapters = [adapters]
            for a in adapters:
                name = (a.get("Name") or "").lower()
                if "intel" in name:
                    # 记录找到了 Intel 适配器
                    intel_luids.add("intel_found")
    except Exception:
        pass

    if not intel_luids:
        return {}

    # 第二步: 枚举 GPU Engine counter，用 pid_0 (系统总计) 采集各引擎
    try:
        result = subprocess.run(
            ["typeperf", "-qx", "GPU Engine"],
            capture_output=True, text=True, timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000),
        )
        if result.returncode != 0:
            return {}
    except Exception:
        return {}

    # 解析 counter 行，按 LUID 分组找出非 NVIDIA 的 GPU（即 Intel iGPU）
    # 策略: 收集所有 pid_0 的 counter，按 LUID 分组，排除 NVIDIA LUID
    luid_counters: dict[str, dict[str, str]] = {}  # {luid: {engtype: counter_path}}
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "Utilization Percentage" not in line:
            continue
        if "pid_0_" not in line:
            continue

        luid_match = _LUID_PATTERN.search(line)
        eng_match = _ENGTYPE_PATTERN.search(line)
        if not luid_match or not eng_match:
            continue

        luid = luid_match.group(1).lower()
        engtype = eng_match.group(1)
        luid_counters.setdefault(luid, {})[engtype] = line

    if not luid_counters:
        return {}

    # 第三步: 如果有多个 LUID，尝试识别哪个是 Intel（非 NVIDIA）
    # NVIDIA GPU 通常先被 NVML 发现，我们取不被 NVML 管理的那个 LUID
    # 简单策略: 如果只有 1 个 LUID，它就是唯一的 GPU（可能不对）
    # 如果有 2 个 LUID，排除已通过 NVML 发现的那个
    # 检查每个 LUID 是否包含 "3D" engtype（Intel iGPU 一般有 3D 引擎）
    target_luid = None
    if len(luid_counters) == 1:
        target_luid = list(luid_counters.keys())[0]
    else:
        # 多 GPU 时，选包含 VideoDecode 引擎且引擎数量最多的非首 LUID
        # （NVIDIA 通常是 phys_0，Intel 是 phys_1，但 LUID 排序不可靠）
        # 更可靠的方式: 用 LUID 数量最多引擎类型的那个（Intel 通常有更多引擎类型）
        candidates = sorted(luid_counters.items(), key=lambda x: len(x[1]), reverse=True)
        for luid, engines in candidates:
            if "VideoDecode" in engines or "VideoEncode" in engines:
                target_luid = luid
                break
        if target_luid is None:
            target_luid = candidates[0][0]

    engines = luid_counters[target_luid]
    # 标准化引擎名称
    name_map = {
        "3D": "3D",
        "VideoDecode": "Video Decode",
        "VideoEncode": "Video Encode",
        "Copy": "Copy",
        "VideoProcessing": "Video Processing",
    }
    result_counters: dict[str, str] = {}
    for engtype, counter_path in engines.items():
        display_name = name_map.get(engtype, engtype)
        result_counters[display_name] = counter_path

    return result_counters
