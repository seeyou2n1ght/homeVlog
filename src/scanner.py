import logging
import os
import re
import time
from dataclasses import dataclass

from src.database import VlogDatabase
from src.utils import load_config, ts_to_unix

from pathlib import Path

logger = logging.getLogger("homevlog")

FILENAME_RE = re.compile(
    r"^(\d{2})_(\d{14})_(\d{14})\.mp4$"
)

CAMERA_DIR_RE = re.compile(
    r"XiaomiCamera_(?P<cam_id>\d+)_(?P<mac>[A-Fa-f0-9]{12})", re.IGNORECASE
)


def parse_camera_dir(dir_path: str) -> dict:
    """
    解析小米摄像头在 NAS 上自动创建的目录命名：XiaomiCamera_CameraID_CameraMAC
    例如：'XiaomiCamera_01_B888805AA3CD'
    返回: {'cam_id': '01', 'mac': 'B888805AA3CD'}，未匹配时返回空字典 {}。
    """
    if not dir_path:
        return {}
    p = Path(dir_path)
    for part in [p.name, p.parent.name]:
        m = CAMERA_DIR_RE.search(part)
        if m:
            return {"cam_id": m.group("cam_id"), "mac": m.group("mac").upper()}
    return {}


def parse_filename(filename: str) -> dict | None:
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    cam = int(m.group(1))
    start_str = m.group(2)
    end_str = m.group(3)
    start_ts = ts_to_unix(start_str)
    end_ts = ts_to_unix(end_str)
    date = start_str[:8]
    return {
        "cam_index": cam,
        "date": date,
        "file_start_time": start_str,
        "file_end_time": end_str,
        "start_ts": start_ts,
        "end_ts": end_ts,
    }





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
            alias = aliases[mac]
            return f"{alias} ({mac})", alias
        return f"{mac} (Cam {cam_index})", mac

    p = Path(dir_path) if dir_path else None
    folder_name = p.name if p and p.name else ""
    if folder_name and folder_name.lower() not in ["", ".", "..", "video", "footage", "input", "output", "temp"]:
        return f"{folder_name} (Cam {cam_index})", folder_name

    return f"Cam {cam_index}", f"cam{cam_index}"


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
        from src.utils import get_input_dirs
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

    for target_dir in target_dirs:
        logger.info("scanning: %s (freeze=%dmin, skip_today=%s)", target_dir, freeze_minutes, skip_today)
        if not os.path.isdir(target_dir):
            logger.error("input_dir not found: %s", target_dir)
            continue

        for entry in os.scandir(target_dir):
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

            stat = entry.stat()
            age_min = (time.time() - stat.st_mtime) / 60.0

            # 如果启用了 skip_today，则不再执行基于 age 的 "忽略最新素材" (freeze) 逻辑
            if not skip_today:
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
    logger.info("scan done: total_added=%d total_skipped=%d total_frozen=%d", total_added, total_skipped, total_frozen)
    return ScanResult(added=total_added, skipped=total_skipped, frozen_pending=total_frozen)



def get_date_cam_groups(db: VlogDatabase) -> list[tuple[str, int]]:
    rows = db.conn.execute(
        """SELECT DISTINCT date, cam_index FROM file_tasks
           ORDER BY date, cam_index"""
    ).fetchall()
    return [(r["date"], r["cam_index"]) for r in rows]
