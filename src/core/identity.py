"""Camera and media identity extraction utilities."""
from functools import lru_cache
import hashlib
import json
import logging
import os
from pathlib import Path, PureWindowsPath
import re
import time

logger = logging.getLogger("homevlog")

FILENAME_RE = re.compile(
    r"^(\d{2})_(\d{14})_(\d{14})\.mp4$"
)

CAMERA_DIR_RE = re.compile(
    r"XiaomiCamera_(?P<cam_id>\d+)_(?P<mac>[A-Fa-f0-9]{12})", re.IGNORECASE
)


def ts_to_unix(ts_str: str) -> float:
    """Parse YYYYMMDDHHMMSS timestamp to Unix epoch seconds."""
    t = time.strptime(ts_str, "%Y%m%d%H%M%S")
    return time.mktime(t)


def camera_key(filepath: str, channel: int = 0) -> str:
    """Directory MAC is physical identity; retain channel for multi-channel exports."""
    parent = PureWindowsPath(str(filepath)).parent
    match = CAMERA_DIR_RE.search(str(parent))
    identity = match.group("mac").upper() if match else str(parent).replace("\\", "/").casefold()
    return f"{identity}:channel{channel}"


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


def source_identity(filepath: str | Path) -> list:
    """Extract canonical path, file size, and modification timestamp as a stable identity triple."""
    path = Path(filepath)
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0:
        raise OSError("Visual source is not a nonempty file")
    return [os.path.normcase(str(path.resolve())), stat.st_size, stat.st_mtime_ns]


@lru_cache(maxsize=2048)
def _resolve_canonical_path(p_str: str) -> Path:
    return Path(p_str).resolve()


def file_identity(path: str | Path) -> list:
    """Extract canonical path, file size, and modification timestamp without throwing on missing file."""
    p = _resolve_canonical_path(str(path))
    try:
        stat = p.stat()
        return [str(p), stat.st_size, stat.st_mtime_ns]
    except OSError:
        return [str(p), None, None]


def processing_fingerprint(filepath: str | Path, config: dict) -> str:
    """Deterministic hashing of media file identity and analysis configuration parameters."""
    from src.core.config import PROJECT_ROOT

    selected = {
        key: config.get(key, {})
        for key in ("detection", "yolo", "audio_vad", "segment", "presence", "micro_motion")
    }
    model = Path((selected.get("yolo") or {}).get("model_path", "models/yolo11m.pt"))
    if not model.is_absolute():
        model = PROJECT_ROOT / model
    payload = [
        2,
        [file_identity(filepath)],
        "analysis-v5-ordered-prescreen-prior",
        "analysis",
        0,
        {},
        {},
        selected,
        file_identity(model),
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
