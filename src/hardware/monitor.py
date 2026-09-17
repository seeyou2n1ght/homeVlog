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
        """采样 Intel iGPU 各引擎利用率 (%)，通过 Windows typeperf 单次读取。

        counter 路径为通配符形式（新版 Windows 无 pid_0 聚合实例），
        输出列与进程实例一一对应。同一物理引擎会因不同进程产生多个
        实例，Windows 已为每个实例给出引擎忙碌百分比；这里取最大值，
        避免把共享引擎的进程实例相加后生成超过 100% 的伪指标。
        """
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
            lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]

            # 列头行: 含 "GPU Engine" 实例路径；数据行: 以日期时间戳开头
            header_line = None
            data_line = None
            for line in lines:
                if "GPU Engine" in line:
                    header_line = line
                elif line.startswith('"') and not line.startswith('"\\\\'):
                    data_line = line
            if not header_line or not data_line:
                return {}

            headers = [p.strip().strip('"') for p in header_line.split('","')]
            values = [p.strip().strip('"') for p in data_line.split('","')]
            # headers[0]/values[0] 为 (PDH-CSV 版本/时间戳)，数据列从 1 开始
            engtype_pat = re.compile(r"engtype_(\w+)\)")
            result_dict: dict[str, float] = {}
            for i, col in enumerate(headers[1:], start=1):
                m = engtype_pat.search(col)
                if not m or i >= len(values):
                    continue
                eng = m.group(1)
                # 由显示名映射表反查：engtype → display name
                disp = _IGPU_ENG_DISPLAY.get(eng, eng)
                try:
                    val = max(0.0, float(values[i]))
                except (ValueError, TypeError):
                    continue
                result_dict[disp] = max(result_dict.get(disp, 0.0), min(100.0, val))
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
# Intel iGPU Performance Counter Discovery (Windows only)
# ---------------------------------------------------------------------------

# Windows GPU Engine counter 格式:
# \GPU Engine(pid_0_luid_0x00000000_0xXXXX_phys_0_eng_N_engtype_XXX)\Utilization Percentage
# 其中 engtype 为 3D / VideoDecode / VideoEncode / Copy 等
_ENGTYPE_PATTERN = re.compile(r"engtype_(\w+)")
_LUID_PATTERN = re.compile(r"luid_0x[0-9a-fA-F]+_0x([0-9a-fA-F]+)")

# iGPU 引擎类型 → 显示名映射（发现与采样共用）
_IGPU_ENG_DISPLAY = {
    "3D": "3D",
    "VideoDecode": "Video Decode",
    "VideoEncode": "Video Encode",
    "Compute": "Compute",
}


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

    # 第二步: 枚举 GPU Engine counter。
    # 注意: 新版 Windows 已不提供 pid_0 系统级聚合实例（本机实测仅有按进程实例），
    # 因此按 (luid, engtype) 全量分组，采样时用通配符路径动态聚合。
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

    # 按 LUID 分组收集引擎类型全集（不限 pid）
    luid_engines: dict[str, set[str]] = {}
    luid_full: dict[str, str] = {}  # short_luid -> 完整 luid_..._phys_N 段
    full_pat = re.compile(r"(luid_0x[0-9a-fA-F]+_0x[0-9a-fA-F]+_phys_\d+)")
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or "Utilization Percentage" not in line:
            continue
        luid_match = _LUID_PATTERN.search(line)
        eng_match = _ENGTYPE_PATTERN.search(line)
        full_match = full_pat.search(line)
        if not luid_match or not eng_match or not full_match:
            continue
        luid = luid_match.group(1).lower()
        luid_engines.setdefault(luid, set()).add(eng_match.group(1))
        luid_full[luid] = full_match.group(1)

    if not luid_engines:
        return {}

    # 第三步: 识别 Intel LUID——排除带 NVIDIA 特征引擎 (OFA 光流/VR) 的 LUID，
    # 在剩余候选中优先选含 VideoDecode 且引擎类型最多者
    nvidia_markers = ("OFA", "VR")
    candidates = [
        (luid, engs) for luid, engs in luid_engines.items()
        if not any(any(e.startswith(m) for m in nvidia_markers) for e in engs)
    ] or list(luid_engines.items())
    candidates.sort(key=lambda x: ("VideoDecode" in x[1], len(x[1])), reverse=True)
    target_luid = candidates[0][0]

    # 生成通配符 counter 路径（采样时按进程实例动态求和）
    full_seg = luid_full[target_luid]
    name_map = {
        "3D": "3D",
        "VideoDecode": "Video Decode",
        "VideoEncode": "Video Encode",
        "Compute": "Compute",
    }
    result_counters: dict[str, str] = {}
    for engtype, display_name in name_map.items():
        if engtype in luid_engines[target_luid]:
            result_counters[display_name] = (
                f"\\GPU Engine(pid_*_{full_seg}_eng_*_engtype_{engtype})"
                "\\Utilization Percentage"
            )

    return result_counters
