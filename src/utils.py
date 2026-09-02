import logging
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / "temp"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_PATH = PROJECT_ROOT / "data" / "vlog.db"

SETTINGS: dict = {}
_config_lock = threading.Lock()

_disk_semaphore: threading.Semaphore | None = None
_nv_semaphore: threading.Semaphore | None = None
_qsv_semaphore: threading.Semaphore | None = None
_io_lock = threading.Lock()

def reset_semaphores() -> None:
    """Reset cached semaphores to allow reloading configuration or clean testing."""
    global _disk_semaphore, _nv_semaphore, _qsv_semaphore
    with _io_lock:
        _disk_semaphore = None
        _nv_semaphore = None
        _qsv_semaphore = None

def get_disk_semaphore() -> threading.Semaphore:
    global _disk_semaphore
    if _disk_semaphore is None:
        with _io_lock:
            if _disk_semaphore is None:
                config = load_config()
                limit = config.get("hardware", {}).get("max_io_concurrency", 8)
                _disk_semaphore = threading.Semaphore(limit)
    return _disk_semaphore

def get_nv_semaphore() -> threading.Semaphore:
    global _nv_semaphore
    if _nv_semaphore is None:
        with _io_lock:
            if _nv_semaphore is None:
                config = load_config()
                # 默认限制并发的 NVENC/NVDEC 会话数为 3 (针对 3060Ti 优化)
                limit = config.get("hardware", {}).get("max_nv_concurrency", 3)
                _nv_semaphore = threading.Semaphore(limit)
    return _nv_semaphore

def get_qsv_semaphore() -> threading.Semaphore:
    global _qsv_semaphore
    if _qsv_semaphore is None:
        with _io_lock:
            if _qsv_semaphore is None:
                config = load_config()
                # 默认 QSV 并发限制 (针对 12600K 双 VDBox 优化)
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
            config = load_config()
        self.config = config

        sched_cfg = config.get("scheduler", {})
        pipe_cfg = config.get("pipeline", {})
        det_cfg = config.get("detection", {})
        hw_cfg = config.get("hardware", {})

        self.watermark_high: int = sched_cfg.get(
            "watermark_high",
            pipe_cfg.get("watermark_high", det_cfg.get("qsv_fallback_threshold", 10)),
        )
        self.watermark_low: int = sched_cfg.get(
            "watermark_low",
            pipe_cfg.get("watermark_low", 3),
        )
        self.nvdec_cooperative: bool = sched_cfg.get(
            "nvdec_cooperative",
            pipe_cfg.get("nvdec_cooperative", True),
        )
        self.max_nv_decoders: int = sched_cfg.get(
            "max_nv_decoders",
            hw_cfg.get("max_nv_decoders", 1),
        )
        self.device: str = hw_cfg.get("device", "cuda:0")

        self._lock = threading.Lock()
        self._render_active_count = 0
        self._active_nv_decoders = 0
        self._state = "NORMAL_DECOUPLED"

    def register_render_start(self) -> None:
        """Pass 2 NVENC 渲染批次开始信号：阻断 NVDEC 工作窃取，优先保证 NVENC 编码会话与带宽。"""
        with self._lock:
            self._render_active_count += 1
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

            if queue_size >= self.watermark_high:
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

def load_config(config_path: str | Path | None = None, reload: bool = False) -> dict:
    global SETTINGS, CONFIG_PATH
    with _config_lock:
        if config_path is not None:
            CONFIG_PATH = Path(config_path).resolve()
            reload = True
        if SETTINGS and not reload:
            return SETTINGS
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"config not found: {CONFIG_PATH}")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            SETTINGS = yaml.safe_load(f) or {}

        for d in [OUTPUT_DIR, TEMP_DIR, LOGS_DIR, DB_PATH.parent]:
            d.mkdir(parents=True, exist_ok=True)

        return SETTINGS


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_logging() -> logging.Logger:
    config = load_config()
    level = getattr(logging, config.get("logging", {}).get("level", "INFO").upper(), logging.INFO)

    logger = logging.getLogger("homevlog")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    fmt = logging.Formatter(
        "[%(asctime)s.%(msecs)03d] [%(levelname)-5s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from logging.handlers import RotatingFileHandler
    log_cfg = config.get("logging", {})
    # 生成带时间戳的日志文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"homevlog_{timestamp}.log"
    
    fh = RotatingFileHandler(
        log_file,
        maxBytes=log_cfg.get("rotation_max_bytes", 10485760),
        backupCount=log_cfg.get("rotation_backup_count", 5),
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # 控制台日志：仅在 WARNING/ERROR 或未启用进度条时输出，避免打乱 tqdm 终端 UI
    sh = TqdmLoggingHandler()
    sh.setLevel(max(level, logging.WARNING))
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def ts_to_unix(ts_str: str) -> float:
    """Parse YYYYMMDDHHMMSS timestamp to Unix epoch seconds."""
    t = time.strptime(ts_str, "%Y%m%d%H%M%S")
    return time.mktime(t)


def parse_res(spec: str) -> tuple[int, int]:
    """Parse 'WxH' resolution string to (width, height)."""
    parts = spec.split("x")
    return int(parts[0]), int(parts[1])


def cleanup_resources():
    """Deep GC and kill tracked ffmpeg processes only."""
    import gc
    
    # 1. Force Python GC
    gc.collect()
    
    # 2. Kill registered ffmpeg (DO NOT use psutil to kill all system ffmpegs!)
    try:
        from src.renderer import FFmpegProcessRegistry
        killed = FFmpegProcessRegistry.kill_all()
        if killed:
            logging.getLogger("homevlog").warning("cleanup: killed registered ffmpeg processes")
    except Exception as e:
        logging.getLogger("homevlog").warning("cleanup error: %s", e)


def check_disk_space(path: Path, min_gb: int | None = None) -> bool:
    """Check if free disk space is above minimum threshold."""
    import shutil
    if min_gb is None:
        config = load_config()
        min_gb = config.get("recovery", {}).get("min_disk_space_gb", 20)
    try:
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024 ** 3)
        if free_gb < min_gb:
            logging.getLogger("homevlog").error("Disk space critically low on %s: %.1f GB free (< %d GB)", path, free_gb, min_gb)
            return False
        return True
    except Exception as e:
        logging.getLogger("homevlog").warning("Failed to check disk space: %s", e)
        return True
