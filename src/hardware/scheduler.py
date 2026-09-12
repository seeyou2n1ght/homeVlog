"""异构硬件自适应调度器与硬件并发信号量控制模块.

负责管理：
1. Intel UHD 770 (QSV) 与 NVIDIA RTX 3060Ti (NVDEC/NVENC/Tensor Core) 的算力解耦与动态协同；
2. 硬件并发信号量隔离 (NV, QSV, Disk I/O) 与槽位生命周期管理。
"""

from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
import logging
import threading
import time

from src.config import load_config
from src.ffmpeg import FFmpegProcessRegistry

logger = logging.getLogger("homevlog")

_disk_semaphore: threading.Semaphore | None = None
_nv_semaphore: threading.Semaphore | None = None
_qsv_semaphore: threading.Semaphore | None = None
_io_lock = threading.Lock()
_nvml_initialized: bool = False
_nvml_lock = threading.Lock()


def _interrupted():
    return FFmpegProcessRegistry.is_interrupted()


def _acquire_native(sem, timeout):
    deadline = time.monotonic() + timeout
    while not _interrupted():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        if sem.acquire(timeout=min(.5, remaining)):
            return True
    return False


class FileBudget:
    """Atomic weighted admission avoids deadlock between multi-input render jobs."""
    resource_name = "input I/O"
    def __init__(self, limit):
        self.limit = max(1, int(limit))
        self.available = self.limit
        self.condition = threading.Condition()
        from collections import deque
        self.waiters = deque()

    def acquire(self, timeout=30.0, weight=1):
        if weight > self.limit:
            raise ValueError("Batch exceeds max_io_concurrency; reduce batch_max_files")
        with self.condition:
            ticket = object()
            self.waiters.append(ticket)
            try:
                deadline = time.monotonic() + timeout
                while self.waiters[0] is not ticket or self.available < weight:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or _interrupted():
                        return False
                    self.condition.wait(timeout=min(.5, remaining))
                self.available -= weight
                return True
            finally:
                self.waiters.remove(ticket)
                self.condition.notify_all()

    def release(self, weight=1):
        with self.condition:
            if self.available + weight > self.limit:
                raise ValueError("I/O budget released twice")
            self.available += weight
            self.condition.notify_all()


class VideoLease:
    """Acquire hardware then shared file I/O in the same order on every video path."""
    def __init__(self, hardware, inputs=1):
        self.hardware = hardware
        self.disk = None
        self.inputs = max(1, inputs)

    def acquire(self, timeout=30.0):
        deadline = time.monotonic() + timeout
        self.resource_name = getattr(self.hardware, "resource_name", "video hardware")
        if not _acquire_native(self.hardware, timeout):
            return False
        self.disk = get_disk_semaphore()
        self.resource_name = f"input I/O ({self.inputs} files)"
        try:
            if self.disk.acquire(timeout=max(0, deadline-time.monotonic()), weight=self.inputs):
                return True
        except BaseException:
            self.hardware.release()
            raise
        self.hardware.release()
        return False

    def release(self):
        try:
            self.disk.release(weight=self.inputs)
        finally:
            self.hardware.release()


def get_nv_vram_used_mb(gpu_index: int = 0) -> int | None:
    """获取指定 NVIDIA GPU 当前物理已用显存 (MB)。

    返回 None 表示系统无 NVIDIA GPU、驱动未加载或 NVML 探测不可用。
    """
    global _nvml_initialized
    try:
        import pynvml
        with _nvml_lock:
            if not _nvml_initialized:
                pynvml.nvmlInit()
                _nvml_initialized = True
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return int(mem.used / 1024 / 1024)
    except Exception:
        return None


def acquire_with_retry(
    sem: threading.Semaphore,
    timeout: float = 30.0,
    retries: int = 3,
) -> bool:
    """带超时与重试的信号量获取（AGENTS.md 铁律：禁止无限阻塞的 acquire）。

    防止 prescreen 与 analysis 争用 QSV/NV 信号量时发生死锁：
    单次 acquire 限时 `timeout` 秒，最多尝试 `retries` 次。

    Args:
        sem: 目标信号量。
        timeout: 单次 acquire 超时秒数（推荐 30s）。
        retries: 最大尝试次数（含首次，推荐 3 次）。

    Returns:
        True: 成功持有信号量，调用方必须在 finally 块中恰好 release 一次。
        False: 重试耗尽仍未获取，调用方应走各自的失败返回路径。
    """
    max_attempts = max(1, int(retries))
    for attempt in range(1, max_attempts + 1):
        if _interrupted():
            return False
        acquired = (sem.acquire(timeout=timeout) if isinstance(sem, (FileBudget, VideoLease))
                    else _acquire_native(sem, timeout))
        if acquired:
            return True
        if _interrupted():
            return False
        logger.warning(
            "semaphore acquire timeout: %s after %.1fs (attempt %d/%d)",
            getattr(sem, "resource_name", type(sem).__name__), timeout, attempt, max_attempts,
        )
    logger.error("semaphore acquire failed after %d attempts", max_attempts)
    return False


_nvenc_semaphore = None
_nvdec_semaphore = None

def reset_semaphores() -> None:
    """重置缓存的信号量，支持热重载配置或单元测试隔离。"""
    global _disk_semaphore, _nv_semaphore, _nvenc_semaphore, _nvdec_semaphore, _qsv_semaphore
    with _io_lock:
        _disk_semaphore = None
        _nv_semaphore = None
        _nvenc_semaphore = None
        _nvdec_semaphore = None
        _qsv_semaphore = None


def get_disk_semaphore() -> threading.Semaphore:
    """获取磁盘 I/O 并发信号量。"""
    global _disk_semaphore
    if _disk_semaphore is None:
        with _io_lock:
            if _disk_semaphore is None:
                config = load_config()
                limit = config.get("hardware", {}).get("max_io_concurrency", 8)
                _disk_semaphore = FileBudget(limit)
    return _disk_semaphore


def get_nvenc_semaphore() -> threading.Semaphore:
    """获取专供 NVENC 渲染 Worker 使用的并发信号量 (默认上限 2)。"""
    global _nvenc_semaphore
    if _nvenc_semaphore is None:
        with _io_lock:
            if _nvenc_semaphore is None:
                config = load_config()
                limit = config.get("hardware", {}).get("max_nv_concurrency", 2)
                _nvenc_semaphore = threading.Semaphore(limit)
                _nvenc_semaphore.resource_name = "NVENC hardware"
    return _nvenc_semaphore


def get_nvdec_semaphore() -> threading.Semaphore:
    """获取专供 NVDEC 分析 Worker 使用的并发信号量 (默认上限 1，与 NVENC 物理解耦)。"""
    global _nvdec_semaphore
    if _nvdec_semaphore is None:
        with _io_lock:
            if _nvdec_semaphore is None:
                config = load_config()
                sched_cfg = config.get("scheduler", {})
                hw_cfg = config.get("hardware", {})
                limit = sched_cfg.get("max_nv_decoders", hw_cfg.get("max_nv_decoders", 1))
                _nvdec_semaphore = threading.Semaphore(limit)
                _nvdec_semaphore.resource_name = "NVDEC hardware"
    return _nvdec_semaphore


def get_nv_semaphore() -> threading.Semaphore:
    """向后兼容别名，返回 NVENC 渲染信号量。"""
    return get_nvenc_semaphore()


def get_qsv_semaphore() -> threading.Semaphore:
    """获取 Intel QSV 硬件解码并发信号量 (默认上限 8，针对 12600K 双 VDBox 优化)。"""
    global _qsv_semaphore
    if _qsv_semaphore is None:
        with _io_lock:
            if _qsv_semaphore is None:
                config = load_config()
                limit = config.get("hardware", {}).get("max_qsv_concurrency", 8)
                _qsv_semaphore = threading.Semaphore(limit)
                _qsv_semaphore.resource_name = "QSV hardware"
    return _qsv_semaphore


class WorkStealingManager:
    """
    异构硬件自适应工作窃取调度器 (Heterogeneous Adaptive Work-Stealing Scheduler)
    
    协调 Intel UHD 770 (QSV) 与 NVIDIA RTX 3060Ti (NVDEC/NVENC/Tensor Core) 的算力解耦与动态协同：
    - NORMAL_DECOUPLED: Prescreen & Analysis 默认 100% 走 QSV，3060Ti 专职 YOLO 与 NVENC 渲染。
    - COOPERATIVE_BURST: 当 analysis_queue 积压超过高水位 (watermark_high) 且 3060Ti 未渲染时，
      动态调度 NVDEC ("cuda") 协同解码抽干队列。
    - RENDER_PREEMPTION_YIELD: 当 Pass 2 NVENC 批次渲染启动时，强制让步 NVDEC，
      所有新增分析任务降级回 QSV，杜绝 NVENC 会话超限与显存/PCIe带宽争用。
    """

    def __init__(self, config: dict | None = None):
        if config is None:
            config = load_config()
        self.config = config

        sched_cfg = config.get("scheduler", {})
        hw_cfg = config.get("hardware", {})

        self.watermark_high: int = sched_cfg.get("watermark_high", 10)
        self.watermark_low: int = sched_cfg.get("watermark_low", 3)
        self.nvdec_cooperative: bool = sched_cfg.get("nvdec_cooperative", True)
        self.max_nv_decoders: int = sched_cfg.get(
            "max_nv_decoders",
            hw_cfg.get("max_nv_decoders", 1),
        )
        self.device: str = hw_cfg.get("device", "cuda:0")
        self.cold_start_burst: bool = sched_cfg.get("cold_start_burst", False)
        self.cooperative_during_render: bool = sched_cfg.get("cooperative_during_render", False)
        self.vram_watermark_mb: int = sched_cfg.get(
            "vram_watermark_mb",
            hw_cfg.get("vram_watermark_mb", 6200),
        )
        self._vram_probe_fn = get_nv_vram_used_mb
        self._last_vram_warn_time = 0.0

        self._lock = threading.Lock()
        self._render_active_count = 0
        self._active_nv_decoders = 0
        self._state = "NORMAL_DECOUPLED"

    def set_vram_probe_fn(self, fn) -> None:
        """设置显存探测函数（支持单元测试注入 Mock）。"""
        with self._lock:
            self._vram_probe_fn = fn

    def is_vram_under_pressure(self) -> bool:
        """检测当前显卡物理显存是否超过安全水位线。"""
        if self._vram_probe_fn is None:
            return False
        try:
            used_mb = self._vram_probe_fn()
            if used_mb is not None and used_mb >= self.vram_watermark_mb:
                now = time.monotonic()
                if now - getattr(self, "_last_vram_warn_time", 0.0) >= 60.0:
                    self._last_vram_warn_time = now
                    logger.warning(
                        "WorkStealing: VRAM usage (%d MB) exceeded watermark (%d MB), yielding NVDEC decode to QSV to prevent OOM (dGPU YOLO & NVENC remain active)",
                        used_mb,
                        self.vram_watermark_mb,
                    )
                return True
        except Exception as e:
            logger.debug("WorkStealing: VRAM probe error: %s", e)
        return False

    def enable_cold_start_burst(self) -> None:
        """激活冷启动破冰模式：在渲染任务就绪前优先调用 NVDEC 协同解码冲刷队列。"""
        with self._lock:
            self.cold_start_burst = True

    def disable_cold_start_burst(self) -> None:
        """停用冷启动破冰模式，回归常规水位线管控。"""
        with self._lock:
            self.cold_start_burst = False

    def register_render_start(self) -> None:
        """Pass 2 NVENC 渲染批次开始信号。"""
        with self._lock:
            self._render_active_count += 1
            self.cold_start_burst = False
            if not self.cooperative_during_render:
                self._state = "RENDER_PREEMPTION_YIELD"

    def register_render_end(self) -> None:
        """Pass 2 NVENC 渲染批次结束信号：恢复 NVDEC 工作窃取能力。"""
        with self._lock:
            self._render_active_count = max(0, self._render_active_count - 1)
            if self._render_active_count == 0:
                self._state = "NORMAL_DECOUPLED"

    @property
    def is_render_active(self) -> bool:
        with self._lock:
            return self._render_active_count > 0

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def active_nv_decoders(self) -> int:
        with self._lock:
            return self._active_nv_decoders

    def get_analysis_device(self, queue_size: int, is_render_active: bool | None = None, prescreen_idle: bool = False) -> str:
        """
        根据队列水位与渲染状态决策当前分析任务的解码硬件设备。
        
        返回值: "qsv" | "cuda"
        """
        with self._lock:
            render_active = (
                is_render_active if is_render_active is not None else (self._render_active_count > 0)
            )

            if prescreen_idle:
                self._state = "NORMAL_DECOUPLED"
                return "qsv"

            if not self.nvdec_cooperative or "cuda" not in self.device.lower():
                self._state = "NORMAL_DECOUPLED"
                return "qsv"

            # 显存物理安全防御：已用显存超警戒线时强制让步 QSV
            if self.is_vram_under_pressure():
                self._state = "VRAM_PRESSURE_YIELD"
                return "qsv"

            # 渲染进行中分支：开启协同且显存安全且队列积压时，允许借调 NVDEC 槽位
            if render_active:
                if (
                    self.cooperative_during_render
                    and queue_size >= self.watermark_high
                    and self._active_nv_decoders < self.max_nv_decoders
                ):
                    self._state = "COOPERATIVE_RENDER_COEXIST"
                    return "cuda"
                self._state = "RENDER_PREEMPTION_YIELD"
                return "qsv"

            # 冷启动破冰协同条件：显式启用了 cold_start_burst 且队列有一定积压 (> watermark_low) 且无渲染运行
            is_cold_burst = (
                self.cold_start_burst
                and self._render_active_count == 0
                and queue_size > self.watermark_low
            )
            is_queue_backlog = (queue_size >= self.watermark_high)

            if is_cold_burst or is_queue_backlog:
                if self._active_nv_decoders < self.max_nv_decoders:
                    self._state = "COOPERATIVE_BURST"
                    return "cuda"
                else:
                    return "qsv"
            elif queue_size <= self.watermark_low:
                self._state = "NORMAL_DECOUPLED"
                return "qsv"
            else:
                if self._state in ("COOPERATIVE_BURST", "COOPERATIVE_RENDER_COEXIST") and self._active_nv_decoders < self.max_nv_decoders:
                    return "cuda"
                return "qsv"

    def acquire_nvdec_slot(self) -> bool:
        """尝试占用一个 NVDEC 解码协同槽位。"""
        with self._lock:
            if self._render_active_count > 0 and not self.cooperative_during_render:
                return False
            if self.is_vram_under_pressure():
                self._state = "VRAM_PRESSURE_YIELD"
                return False
            if self._active_nv_decoders < self.max_nv_decoders:
                self._active_nv_decoders += 1
                return True
            return False

    def release_nvdec_slot(self) -> None:
        """释放 NVDEC 解码协同槽位。"""
        with self._lock:
            self._active_nv_decoders = max(0, self._active_nv_decoders - 1)

    @contextmanager
    def lease_device(self, queue_size: int, prescreen_idle: bool = False):
        """上下文管理器：自动决策解码设备并在使用 CUDA 时安全管理 NVDEC 槽位生命周期。"""
        device = self.get_analysis_device(queue_size, prescreen_idle=prescreen_idle)
        acquired_slot = False
        if device == "cuda":
            if self.acquire_nvdec_slot():
                acquired_slot = True
            else:
                device = "qsv"
        try:
            yield device
        finally:
            if acquired_slot:
                self.release_nvdec_slot()


@dataclass
class RenderBatchItem:
    """渲染批次载荷元数据。"""
    b_idx: int
    files: list[str]
    dynamic_duration: float
    sequence: int = 0


class DualEndedBatchQueue:
    """线程安全的双端批次调度队列：
    支持 NVENC 从队首按 LPT (最长优先) 消费最重任务，
    支持 QSV 从队尾按 SPT (最短优先) 在安全门限内逆向窃取轻中度任务。
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._items: deque[RenderBatchItem] = deque()
        self._seq = 0

    def put(self, b_idx: int, files: list[str], dynamic_duration: float) -> None:
        """插入批次，保持按 dynamic_duration 降序排列 (动态时长相同则按 sequence 升序)。"""
        with self._lock:
            self._seq += 1
            item = RenderBatchItem(
                b_idx=b_idx,
                files=list(files),
                dynamic_duration=float(dynamic_duration),
                sequence=self._seq,
            )
            inserted = False
            for idx, existing in enumerate(self._items):
                if item.dynamic_duration > existing.dynamic_duration:
                    self._items.insert(idx, item)
                    inserted = True
                    break
                elif item.dynamic_duration == existing.dynamic_duration:
                    if item.sequence < existing.sequence:
                        self._items.insert(idx, item)
                        inserted = True
                        break
            if not inserted:
                self._items.append(item)
            self._not_empty.notify()

    def pop_heaviest(self, timeout: float = 0.5) -> RenderBatchItem | None:
        """NVENC 消费入口：从队首取出动态时长最大的批次 (LPT)。"""
        with self._not_empty:
            deadline = time.monotonic() + timeout
            while not self._items:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._not_empty.wait(timeout=min(0.5, remaining))
            return self._items.popleft()

    def steal_lightest(self, max_dynamic_s: float) -> RenderBatchItem | None:
        """QSV 窃取入口：若队尾最轻批次的动态时长 <= max_dynamic_s，则逆向窃取 (SPT)。"""
        with self._lock:
            if not self._items:
                return None
            tail = self._items[-1]
            if tail.dynamic_duration <= max_dynamic_s:
                return self._items.pop()
            return None

    def notify_all(self) -> None:
        """唤醒所有等待中的 worker。"""
        with self._not_empty:
            self._not_empty.notify_all()

    def qsize(self) -> int:
        with self._lock:
            return len(self._items)

    def empty(self) -> bool:
        with self._lock:
            return len(self._items) == 0

