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
    fh = RotatingFileHandler(
        LOGS_DIR / "homevlog.log",
        maxBytes=log_cfg.get("rotation_max_bytes", 10485760),
        backupCount=log_cfg.get("rotation_backup_count", 5),
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
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
    """Deep GC and kill orphaned ffmpeg processes."""
    import gc
    import psutil
    
    # 1. Force Python GC
    gc.collect()
    
    # 2. Kill orphan ffmpeg (Run ONLY between batch days)
    killed = 0
    for proc in psutil.process_iter(['pid', 'name']):
        try:
            name = proc.info.get('name', '')
            if name and 'ffmpeg' in name.lower():
                proc.kill()
                killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
            
    if killed > 0:
        logging.getLogger("homevlog").warning("cleanup: killed %d zombie ffmpeg processes", killed)


def check_disk_space(path: Path, min_gb: int = 20) -> bool:
    """Check if free disk space is above minimum threshold."""
    import shutil
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
