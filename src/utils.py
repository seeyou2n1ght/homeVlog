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

from src.scheduler import (
    reset_semaphores,
    get_disk_semaphore,
    get_nv_semaphore,
    get_qsv_semaphore,
    WorkStealingManager,
)


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


_active_dashboard = None
_dashboard_lock = threading.Lock()


def register_dashboard(dashboard) -> None:
    """注册活跃的终端仪表盘实例，用于实时告警联动。"""
    global _active_dashboard
    with _dashboard_lock:
        _active_dashboard = dashboard


def unregister_dashboard() -> None:
    """注销终端仪表盘实例。"""
    global _active_dashboard
    with _dashboard_lock:
        _active_dashboard = None


class ContextualFormatter(logging.Formatter):
    """支持自动回退的子系统上下文日志格式化器。"""

    def format(self, record: logging.LogRecord) -> str:
        if not hasattr(record, "subsystem"):
            # 若未注入 subsystem，从 logger name 提取或默认为 core
            name_parts = record.name.split(".")
            record.subsystem = name_parts[-1] if len(name_parts) > 1 else "core"
        return super().format(record)


class JsonLinesFormatter(logging.Formatter):
    """结构化 JSONL 日志格式化器。"""

    def format(self, record: logging.LogRecord) -> str:
        import json
        log_obj = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)) + f".{int(record.msecs):03d}",
            "level": record.levelname,
            "subsystem": getattr(record, "subsystem", "core"),
            "logger": record.name,
            "message": record.getMessage(),
            "caller": f"{record.filename}:{record.lineno}",
        }
        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_obj, ensure_ascii=False)


class RichConsoleBridgeHandler(logging.Handler):
    """
    智能富文本控制台日志桥接器：
    - 当 Live 仪表盘激活时：将 WARNING/ERROR 汇流到仪表盘告警面板，避免破坏终端布局；
    - 当无 Live 仪表盘时：通过 Rich 格式化输出彩色控制台日志。
    """

    def __init__(self, level=logging.WARNING):
        super().__init__(level=level)
        from rich.console import Console
        self._console = Console(stderr=True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            with _dashboard_lock:
                dash = _active_dashboard

            if dash is not None:
                # 仪表盘处于激活态：仅将告警与异常送入跑马灯
                if record.levelno >= logging.WARNING:
                    brief = f"[{record.levelname}] {record.getMessage()}"
                    dash.add_alert(brief)
            else:
                # 仪表盘未激活：直接着色输出到终端
                color = "white"
                if record.levelno >= logging.ERROR:
                    color = "bold red"
                elif record.levelno >= logging.WARNING:
                    color = "yellow"
                elif record.levelno == logging.INFO:
                    color = "cyan"

                sub = getattr(record, "subsystem", "core")
                self._console.print(f"[{color}][{record.levelname:<5}][/{color}] [dim]\\[{sub}][/dim] {record.getMessage()}")
        except Exception:
            self.handleError(record)


class SubsystemAdapter(logging.LoggerAdapter):
    """自动绑定子系统名称的 Logger 适配器。"""

    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        if "subsystem" not in extra:
            extra["subsystem"] = self.extra.get("subsystem", "core")
        kwargs["extra"] = extra
        return msg, kwargs


def get_logger(subsystem: str = "core") -> logging.LoggerAdapter:
    """获取绑定了指定子系统上下文的 Logger 适配器。"""
    base_logger = logging.getLogger("homevlog")
    if not base_logger.handlers:
        setup_logging()
    return SubsystemAdapter(base_logger, {"subsystem": subsystem})


def setup_logging(level_override: int | None = None) -> logging.Logger:
    """
    初始化全新的多目标分流日志体系：
    1. 主运行日志：logs/homevlog_{timestamp}.log (默认 INFO)
    2. 独立错误排错日志：logs/error_{timestamp}.log (仅 WARNING/ERROR/CRITICAL)
    3. 结构化 JSONL 事件流：logs/events_{timestamp}.jsonl
    4. 智能富文本终端桥接器 (RichConsoleBridgeHandler)
    """
    config = load_config()
    log_cfg = config.get("logging", {})

    level = level_override
    if level is None:
        level = getattr(logging, log_cfg.get("level", "INFO").upper(), logging.INFO)

    logger = logging.getLogger("homevlog")
    logger.setLevel(logging.DEBUG)  # 允许底层捕获全级别，具体级别由 Handler 自行把控

    if logger.handlers:
        return logger

    for d in [LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = LOGS_DIR / f"homevlog_{timestamp}.log"
    err_file = LOGS_DIR / f"error_{timestamp}.log"
    jsonl_file = LOGS_DIR / f"events_{timestamp}.jsonl"

    from logging.handlers import RotatingFileHandler
    max_bytes = log_cfg.get("rotation_max_bytes", 10485760)
    backup_count = log_cfg.get("rotation_backup_count", 5)

    human_fmt = ContextualFormatter(
        "[%(asctime)s.%(msecs)03d] [%(levelname)-5s] [%(subsystem)-10s] %(message)s (%(filename)s:%(lineno)d)",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 1. 主日志 (INFO 及以上)
    fh_main = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh_main.setLevel(level)
    fh_main.setFormatter(human_fmt)
    logger.addHandler(fh_main)

    # 2. 独立错误排错日志 (WARNING 及以上)
    fh_err = RotatingFileHandler(err_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh_err.setLevel(logging.WARNING)
    fh_err.setFormatter(human_fmt)
    logger.addHandler(fh_err)

    # 3. 结构化 JSONL 日志流 (记录 INFO 及以上事件)
    fh_jsonl = RotatingFileHandler(jsonl_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
    fh_jsonl.setLevel(level)
    fh_jsonl.setFormatter(JsonLinesFormatter())
    logger.addHandler(fh_jsonl)

    # 4. 控制台桥接 Handler (WARNING/ERROR 或交互式通知)
    sh = RichConsoleBridgeHandler(level=max(level, logging.WARNING))
    sh.setFormatter(human_fmt)
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


def cleanup_temp_artifacts(clean_batches: bool = False) -> int:
    """清理 temp/ 下上次运行遗留的中间临时产物。

    默认 clean_batches=False：保留已渲染完成的完整 _batch*.mp4，仅清理未完成的 *.tmp.mp4、
    滤镜脚本 _fc_*.txt、日志 _stderr_*.log 与拼接列表 .concat_*.txt，保障断点续传复用。
    当 clean_batches=True（如用户传入 --clean-temp 时）：全量清理所有批次成片。
    返回清理的文件数。
    """
    removed = 0
    if not TEMP_DIR.exists():
        return 0
    patterns = ["*.tmp.mp4", "_batch*.tmp.mp4", "_fc_*.txt", "_stderr_*.log", ".concat_*.txt"]
    if clean_batches:
        patterns.extend(["_batch*.mp4"])
    for pattern in patterns:
        for f in TEMP_DIR.glob(pattern):
            try:
                f.unlink()
                removed += 1
            except OSError:
                pass
    if removed:
        logging.getLogger("homevlog").info("temp cleanup: removed %d stale artifacts (clean_batches=%s)", removed, clean_batches)
    return removed


def cleanup_resources(db=None):
    """Deep GC, clear CUDA cache, truncate SQLite WAL, and kill tracked ffmpeg processes."""
    import gc
    gc.collect()
    
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    if db is not None:
        try:
            db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass

    try:
        from src.renderer import FFmpegProcessRegistry
        FFmpegProcessRegistry.kill_all()
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


def get_input_dirs(config: dict | None = None) -> list[str]:
    """
    解析配置中的监控素材输入路径，支持多态格式：
    1. paths.input_dirs: ["path1", "path2", ...]
    2. paths.input_dirs: "single_path"
    3. paths.input_dir: "legacy_single_path"
    返回规整后的有效路径字符串列表。
    """
    if config is None:
        config = load_config()
    paths_cfg = config.get("paths", {})
    raw_dirs = paths_cfg.get("input_dirs")
    if raw_dirs is None:
        raw_dirs = paths_cfg.get("input_dir", "")

    if isinstance(raw_dirs, str):
        return [raw_dirs.strip()] if raw_dirs.strip() else []
    elif isinstance(raw_dirs, list):
        return [str(p).strip() for p in raw_dirs if str(p).strip()]
    return []

