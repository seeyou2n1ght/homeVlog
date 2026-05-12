import logging
import sys
import threading
import time
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

def get_disk_semaphore() -> threading.Semaphore:
    global _disk_semaphore
    if _disk_semaphore is None:
        with _io_lock:
            if _disk_semaphore is None:
                config = load_config()
                limit = config.get("pipeline", {}).get("max_io_concurrency", 16)
                _disk_semaphore = threading.Semaphore(limit)
    return _disk_semaphore

def get_nv_semaphore() -> threading.Semaphore:
    global _nv_semaphore
    if _nv_semaphore is None:
        with _io_lock:
            if _nv_semaphore is None:
                config = load_config()
                # 默认限制并发的 NVENC 会话数为 2 (消费级显卡默认限制，除非破解)
                limit = config.get("pipeline", {}).get("max_nv_concurrency", 2)
                _nv_semaphore = threading.Semaphore(limit)
    return _nv_semaphore

def get_qsv_semaphore() -> threading.Semaphore:
    global _qsv_semaphore
    if _qsv_semaphore is None:
        with _io_lock:
            if _qsv_semaphore is None:
                config = load_config()
                limit = config.get("pipeline", {}).get("max_qsv_concurrency", 4)
                _qsv_semaphore = threading.Semaphore(limit)
    return _qsv_semaphore

def load_config() -> dict:
    global SETTINGS
    if SETTINGS:
        return SETTINGS
    with _config_lock:
        if SETTINGS:
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
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
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

    # 使用 tqdm 兼容的处理器代替标准 StreamHandler
    sh = TqdmLoggingHandler()
    sh.setLevel(level)
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
