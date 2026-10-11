import logging
import os
import re
import time
import threading
from dataclasses import dataclass
from pathlib import Path

from src.core.database import VlogDatabase
from src.core.config import load_config
from src.core.identity import (
    FILENAME_RE,
    CAMERA_DIR_RE,
    camera_key,
    parse_camera_dir,
    parse_filename,
    ts_to_unix,
)

logger = logging.getLogger("homevlog")


def resolve_camera_identity(dir_path: str, cam_index: int = 0, config: dict | None = None) -> tuple[str, str]:
    """
    自适应解析机位身份 (实现三级平滑降级):
    1. 提取 MAC 地址与别名 (L1): 若配置了别名，显示为 "baby_room (B888805AA3CD)"，标识符为 "baby_room"；
    2. 原生 MAC 地址 (L2): 若未配置别名但识别到 MAC，显示为 "B888805AA3CD (Cam 0)"，标识符为 "B888805AA3CD"；
    3. 目录名或索引 (L3): 若完全无 MAC，取文件夹名 "FrontDoor (Cam 0)"，标识符为 "FrontDoor"；
       若文件夹无名称，最终兜底为 "Cam 0" 与 "cam0"。
    返回: (display_name, identifier)
    """
    if config is None:
        config = load_config()
    cam_info = parse_camera_dir(dir_path)
    mac = cam_info.get("mac")
    aliases = config.get("cameras", {}) or {}

    if mac:
        if mac in aliases:
            raw_alias = aliases[mac]
            if isinstance(raw_alias, dict):
                alias = raw_alias.get("alias") or raw_alias.get("name") or raw_alias.get("profile") or mac
            else:
                alias = str(raw_alias)
            return f"{alias} ({mac})", alias
        return f"{mac} (Cam {cam_index})", mac

    p = Path(dir_path) if dir_path else None
    folder_name = p.name if p and p.name else ""
    if folder_name and folder_name.lower() not in ["", ".", "..", "video", "footage", "input", "output", "temp"]:
        return f"{folder_name} (Cam {cam_index})", folder_name

    return f"Cam {cam_index}", f"cam{cam_index}"


def resolve_camera_ui_name(dir_path: str | Path | None = None, cam_index: int = 0, config: dict | None = None) -> str:
    """Human-facing label: semantic mapping, camera ID, then MAC address."""
    if config is None:
        config = load_config()
    dir_str = str(dir_path) if dir_path is not None else ""
    info = parse_camera_dir(dir_str)
    mac = info.get("mac")
    if not mac and dir_str:
        match = re.search(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{12}(?![A-Fa-f0-9])", Path(dir_str).name)
        mac = match.group().upper() if match else None
    alias_raw = (config.get("cameras", {}) or {}).get(mac) if mac else None
    if alias_raw:
        if isinstance(alias_raw, dict):
            return str(alias_raw.get("alias") or alias_raw.get("name") or alias_raw.get("profile") or mac)
        return str(alias_raw)
    if info.get("cam_id") is not None:
        return f"摄像头 {info['cam_id']}"
    if mac:
        return mac
    return resolve_camera_identity(dir_str, cam_index, config)[1]


def resolve_output_filename(
    template: str,
    date: str,
    cam_index: int,
    sample_filepath: str | None = None,
    config: dict | None = None,
) -> str:
    """自适应格式化成片文件名，支持 {date}, {mac}, {camera}, {index} 占位符。

    占位符定义:
    - {date}: 日期字符串 (如 '20260402')
    - {mac}: 摄像机物理 MAC 地址 (如 'B888805AA3CD'，若无则自适应降级为 'cam{cam_index}')
    - {camera}: 语义机位别名 (如 'baby_room'，未设置别名时降级为 MAC，再无则降级为 'cam{cam_index}')
    - {index}: 摄像机逻辑编号数字 (如 0)
    """
    if config is None:
        config = load_config()
    sample_dir = str(Path(sample_filepath).parent) if sample_filepath else ""
    cam_info = parse_camera_dir(sample_dir)
    mac = cam_info.get("mac") or f"cam{cam_index}"
    _, identifier = resolve_camera_identity(sample_dir, cam_index=cam_index, config=config)
    camera = identifier

    name = template.replace("{date}", str(date))
    name = name.replace("{mac}", str(mac))
    name = name.replace("{camera}", str(camera))
    name = name.replace("{index}", str(cam_index))
    return name


class _ScanProgress:
    """Report blocking scan operations without moving NAS/DB work to a thread."""

    def __init__(self, interval_s=10.0):
        self.interval_s = interval_s
        self.started = time.monotonic()
        self.state = ("准备扫描", 0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, name="scan-progress", daemon=True)

    def update(self, operation, count):
        self.state = (operation, count)

    def _heartbeat(self):
        while not self._stop.wait(self.interval_s):
            operation, count = self.state
            logger.info("扫描仍在进行: 已处理=%d，耗时=%.1fs，当前操作: %s",
                        count, time.monotonic() - self.started, operation,
                        extra={"startup_progress": True})

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *_exc):
        self._stop.set()
        self._thread.join()


@dataclass
class ScanResult:
    added: int
    skipped: int
    frozen_pending: int


def scan_directory(
    db: VlogDatabase,
    input_dir: str | list[str] | None = None,
) -> ScanResult:
    config = load_config()
    if input_dir is None:
        from src.core.utils import get_input_dirs
        target_dirs = get_input_dirs(config)
    elif isinstance(input_dir, str):
        target_dirs = [input_dir]
    elif isinstance(input_dir, (list, tuple)):
        target_dirs = [str(p) for p in input_dir]
    else:
        target_dirs = []

    freeze_minutes = config.get("recovery", {}).get("scanner_freeze_minutes", 10)
    stabilize_wait = config.get("recovery", {}).get("file_stabilize_wait", 1.2)
    skip_today = config.get("recovery", {}).get("skip_today", False)
    current_date = time.strftime("%Y%m%d")

    total_added = 0
    total_skipped = 0
    total_frozen = 0

    with _ScanProgress() as progress:
        try:
            for target_dir in target_dirs:
                progress.update(f"检查目录 {target_dir}", total_added + total_skipped + total_frozen)
                logger.info("scanning: %s (freeze=%dmin, skip_today=%s)", target_dir, freeze_minutes, skip_today,
                            extra={"startup_progress": True})
                if not os.path.isdir(target_dir):
                    raise FileNotFoundError(f"输入目录不可用，停止本轮处理: {target_dir}")

                progress.update(f"枚举目录 {target_dir}", total_added + total_skipped + total_frozen)
                with os.scandir(target_dir) as entries:
                    for entry in entries:
                        try:
                            progress.update(f"检查文件 {entry.path}", total_added + total_skipped + total_frozen)
                            if not entry.is_file() or not entry.name.endswith(".mp4"):
                                continue

                            info = parse_filename(entry.name)
                            if info is None:
                                logger.debug("skip unrecognized filename: %s", entry.name)
                                continue

                            if skip_today and info["date"] == current_date:
                                logger.debug("skip today's file: %s", entry.name)
                                total_skipped += 1
                                continue

                            # 如果启用了 skip_today 或 freeze_minutes <= 0，则不再执行基于 age 的 "忽略最新素材" (freeze) 逻辑
                            if not skip_today and freeze_minutes > 0:
                                progress.update(f"读取文件属性 {entry.path}", total_added + total_skipped + total_frozen)
                                stat = entry.stat()
                                age_min = (time.time() - stat.st_mtime) / 60.0
                                if age_min < freeze_minutes:
                                    total_frozen += 1
                                    continue

                                # Stabilize check: only for files near the freeze boundary.
                                if freeze_minutes > 0 and age_min < freeze_minutes + 5:
                                    time.sleep(stabilize_wait)
                                    try:
                                        stat2 = os.stat(entry.path)
                                        if stat2.st_size != stat.st_size:
                                            total_frozen += 1
                                            continue
                                    except OSError:
                                        total_frozen += 1
                                        continue

                            duration = info["end_ts"] - info["start_ts"]
                            progress.update(f"登记文件 {entry.path}", total_added + total_skipped + total_frozen)
                            ok = db.add_file_task(
                                filepath=entry.path,
                                cam_index=info["cam_index"],
                                date=info["date"],
                                file_start_time=info["file_start_time"],
                                file_end_time=info["file_end_time"],
                                file_duration=duration,
                            )
                            if ok:
                                total_added += 1
                            else:
                                total_skipped += 1
                        except OSError:
                            logger.exception("无法扫描文件: %s", entry.path)
                            raise
                        finally:
                            progress.update(f"枚举目录 {target_dir}", total_added + total_skipped + total_frozen)
        except OSError:
            logger.exception("扫描失败，停止本轮处理；已入库记录保留，下次可继续。当前操作: %s", progress.state[0])
            raise
        logger.info("扫描完成: 新增=%d 跳过=%d 冻结=%d，耗时=%.1fs",
                    total_added, total_skipped, total_frozen, time.monotonic() - progress.started,
                    extra={"startup_progress": True})
    return ScanResult(added=total_added, skipped=total_skipped, frozen_pending=total_frozen)



def get_date_cam_groups(
    db: VlogDatabase, dates: list[str] | None = None, cam_index: int | None = None,
) -> list[tuple[str, int]]:
    rows = db.conn.execute(
        """SELECT DISTINCT date, cam_index FROM file_tasks
           ORDER BY date, cam_index"""
    ).fetchall()
    selected_dates = set(dates) if dates else None
    return [(r["date"], r["cam_index"]) for r in rows
            if (selected_dates is None or r["date"] in selected_dates)
            and (cam_index is None or r["cam_index"] == cam_index)]
