"""异构硬件自适应调度器与硬件并发信号量控制模块.

负责管理：
1. Intel UHD 770 (QSV) 与 NVIDIA RTX 3060Ti (NVDEC/NVENC/Tensor Core) 的算力解耦与动态协同；
2. 硬件并发信号量隔离 (NV, QSV, Disk I/O) 与槽位生命周期管理。
"""

import threading
from contextlib import contextmanager


_disk_semaphore: threading.Semaphore | None = None
_nv_semaphore: threading.Semaphore | None = None
_qsv_semaphore: threading.Semaphore | None = None
_io_lock = threading.Lock()


def reset_semaphores() -> None:
    """重置缓存的信号量，支持热重载配置或单元测试隔离。"""
    global _disk_semaphore, _nv_semaphore, _qsv_semaphore
    with _io_lock:
        _disk_semaphore = None
        _nv_semaphore = None
        _qsv_semaphore = None


def get_disk_semaphore() -> threading.Semaphore:
    """获取磁盘 I/O 并发信号量。"""
    global _disk_semaphore
    if _disk_semaphore is None:
        with _io_lock:
            if _disk_semaphore is None:
                from src.utils import load_config
                config = load_config()
                limit = config.get("hardware", {}).get("max_io_concurrency", 8)
                _disk_semaphore = threading.Semaphore(limit)
    return _disk_semaphore


def get_nv_semaphore() -> threading.Semaphore:
    """获取 NVIDIA 硬件编解码并发信号量 (默认上限 3，针对 3060Ti 优化)。"""
    global _nv_semaphore
    if _nv_semaphore is None:
        with _io_lock:
            if _nv_semaphore is None:
                from src.utils import load_config
                config = load_config()
                limit = config.get("hardware", {}).get("max_nv_concurrency", 3)
                _nv_semaphore = threading.Semaphore(limit)
    return _nv_semaphore


def get_qsv_semaphore() -> threading.Semaphore:
    """获取 Intel QSV 硬件解码并发信号量 (默认上限 8，针对 12600K 双 VDBox 优化)。"""
    global _qsv_semaphore
    if _qsv_semaphore is None:
        with _io_lock:
            if _qsv_semaphore is None:
                from src.utils import load_config
                config = load_config()
                limit = config.get("hardware", {}).get("max_qsv_concurrency", 8)
                _qsv_semaphore = threading.Semaphore(limit)
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
            from src.utils import load_config
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

        self._lock = threading.Lock()
        self._render_active_count = 0
        self._active_nv_decoders = 0
        self._state = "NORMAL_DECOUPLED"

    def enable_cold_start_burst(self) -> None:
        """激活冷启动破冰模式：在渲染任务就绪前优先调用 NVDEC 协同解码冲刷队列。"""
        with self._lock:
            self.cold_start_burst = True

    def disable_cold_start_burst(self) -> None:
        """停用冷启动破冰模式，回归常规水位线管控。"""
        with self._lock:
            self.cold_start_burst = False

    def register_render_start(self) -> None:
        """Pass 2 NVENC 渲染批次开始信号：阻断 NVDEC 工作窃取，优先保证 NVENC 编码会话与带宽。"""
        with self._lock:
            self._render_active_count += 1
            self.cold_start_burst = False
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

    def get_analysis_device(self, queue_size: int, is_render_active: bool | None = None) -> str:
        """
        根据队列水位与渲染状态决策当前分析任务的解码硬件设备。
        
        返回值: "qsv" | "cuda"
        """
        with self._lock:
            render_active = (
                is_render_active if is_render_active is not None else (self._render_active_count > 0)
            )

            if render_active:
                self._state = "RENDER_PREEMPTION_YIELD"
                return "qsv"

            if not self.nvdec_cooperative or "cuda" not in self.device.lower():
                self._state = "NORMAL_DECOUPLED"
                return "qsv"

            # 冷启动破冰协同条件：显式启用了 cold_start_burst 且队列非空且无渲染运行
            is_cold_burst = (self.cold_start_burst and self._render_active_count == 0 and queue_size >= 1)
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
                if self._state == "COOPERATIVE_BURST" and self._active_nv_decoders < self.max_nv_decoders:
                    return "cuda"
                return "qsv"

    def acquire_nvdec_slot(self) -> bool:
        """尝试占用一个 NVDEC 解码协同槽位。"""
        with self._lock:
            if self._render_active_count > 0:
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
    def lease_device(self, queue_size: int):
        """上下文管理器：自动决策解码设备并在使用 CUDA 时安全管理 NVDEC 槽位生命周期。"""
        device = self.get_analysis_device(queue_size)
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
