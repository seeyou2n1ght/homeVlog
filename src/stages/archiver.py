"""src/archiver.py

人工审核校正帧画面归档与备份模块。
将用户复核标记过的困难样本（误报、漏报、动态确认）关键帧物理固化存储，
摆脱对远程 NAS 视频源物理生命周期的依赖，为后续模型自适应微调与算法优化提供持续积累的数据资产。
"""

import json
import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

from src.database import VlogDatabase
from src.utils import PROJECT_ROOT

logger = logging.getLogger("homevlog.archiver")

DEFAULT_ARCHIVE_DIR = PROJECT_ROOT / "data" / "feedback_archive"
DEFAULT_IMAGES_DIR = DEFAULT_ARCHIVE_DIR / "images"
DEFAULT_MANIFEST = DEFAULT_ARCHIVE_DIR / "manifest.jsonl"
CACHE_DIR = PROJECT_ROOT / "temp" / "audit_cache"

_manifest_lock = threading.Lock()


def get_archive_dirs(custom_archive_dir: Path | None = None) -> tuple[Path, Path, Path]:
    """获取或初始化归档目录与清单路径。"""
    archive_dir = custom_archive_dir or DEFAULT_ARCHIVE_DIR
    images_dir = archive_dir / "images"
    manifest_path = archive_dir / "manifest.jsonl"

    images_dir.mkdir(parents=True, exist_ok=True)
    return archive_dir, images_dir, manifest_path


class FrameArchiver:
    """Fault-isolated feedback archiving with a bounded recent JPEG fallback."""
    from collections import OrderedDict
    _cache = OrderedDict()
    _lock = threading.Lock()
    _bytes = 0

    @classmethod
    def remember(cls, filepath, timestamp, jpeg):
        payload = bytes(jpeg)
        if len(payload) > 1024 * 1024:
            return
        with cls._lock:
            key = (str(filepath), float(timestamp))
            cls._bytes -= len(cls._cache.pop(key, b""))
            cls._cache[key] = payload
            cls._bytes += len(payload)
            while cls._bytes > 8 * 1024 * 1024 or len(cls._cache) > 64:
                cls._bytes -= len(cls._cache.popitem(last=False)[1])

    @classmethod
    def cached(cls, filepath, timestamp):
        with cls._lock:
            candidates = [(abs(t-timestamp), payload) for (fp,t),payload in cls._cache.items()
                          if fp == str(filepath) and abs(t-timestamp) <= 2.0]
            return min(candidates, key=lambda item: item[0])[1] if candidates else None

    @staticmethod
    def archive(db, segment_id, archive_dir=None):
        try:
            return _extract_and_archive_frame(db, segment_id, archive_dir)
        except Exception as exc:
            logger.warning("Feedback archive unavailable for segment %s: %s", segment_id, exc)
            return None


def extract_and_archive_frame(db, segment_id, archive_dir=None):
    return FrameArchiver.archive(db, segment_id, archive_dir)


def _extract_and_archive_frame(
    db: VlogDatabase,
    segment_id: int,
    archive_dir: Path | None = None,
) -> Path | None:
    """提取单个复核切片的代表性关键帧并持久化至归档库中。"""
    segment = db.get_segment_by_id(segment_id)
    if not segment:
        logger.warning("归档失败: 未找到 segment_id=%d", segment_id)
        return None

    manual_label = segment.get("manual_label")
    if not manual_label or manual_label == "UNREVIEWED":
        return None

    _, images_dir, manifest_path = get_archive_dirs(archive_dir)

    start_t = float(segment.get("start_time", 0.0))
    end_t = float(segment.get("end_time", 0.0))
    file_offset = float(segment.get("file_start_offset", 0.0) or 0.0)
    t_mid = round((start_t + end_t) / 2.0, 2)
    local_mid = max(0.0, t_mid - file_offset) if (t_mid >= file_offset and file_offset > 0) else t_mid

    date_str = str(segment.get("date", "unknown"))
    cam_idx = int(segment.get("cam_index", 0))
    safe_label = manual_label.lower()

    # 构造归档文件名: 日期_机位_切片ID_时间戳_标签.jpg
    filename = f"{date_str}_cam{cam_idx}_seg{segment_id}_{t_mid:.2f}_{safe_label}.jpg"
    target_img = images_dir / filename

    # 1. 尝试从原始视频高清抽取
    source_fp = Path(segment.get("filepath", ""))
    extracted = False

    try:
        source_available = source_fp.exists() and source_fp.stat().st_size > 0
    except OSError as exc:
        logger.warning("Source unavailable; trying cached frame: %s", exc)
        source_available = False
    if source_available:
        cmd = [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{local_mid:.3f}",
            "-i", str(source_fp),
            "-vframes", "1",
            "-q:v", "2",
            str(target_img),
        ]
        try:
            from src.ffmpeg import run_ffmpeg
            res = run_ffmpeg(cmd[1:], capture_output=True, timeout=12)
            if res.returncode == 0 and target_img.exists() and target_img.stat().st_size > 0:
                extracted = True
        except Exception as e:
            logger.debug("FFmpeg 直接抽取切片帧失败: %s", e)

    # 2. 次级兜底：若原始 NAS 视频离线，寻找缓存池已抽取的首帧或切片帧
    if not extracted and CACHE_DIR.exists():
        stem = source_fp.stem
        # 寻找匹配的缓存文件
        candidates = list(CACHE_DIR.glob(f"{stem}_*_{t_mid:.2f}_*.jpg"))
        if not candidates:
            candidates = list(CACHE_DIR.glob(f"{stem}_*.jpg"))

        for c in candidates:
            if c.exists() and c.stat().st_size > 0 and not c.name.startswith("_file_missing"):
                try:
                    shutil.copy2(str(c), str(target_img))
                    extracted = True
                    break
                except Exception as e:
                    logger.debug("从缓存复制归档帧失败: %s", e)

    if not extracted:
        cached = FrameArchiver.cached(str(source_fp), local_mid)
        if cached:
            logger.warning("Source unavailable; archiving memory frame for segment %s", segment_id)
            temporary = target_img.with_suffix(".jpg.tmp")
            temporary.write_bytes(cached)
            temporary.replace(target_img)
            extracted = True
    if not extracted or not target_img.exists():
        logger.warning(
            "未能成功抽取或归档帧: segment_id=%d, fp=%s, t=%.2f",
            segment_id, source_fp.name, t_mid
        )
        return None

    # 3. 记录元数据清单与回填数据库
    try:
        rel_path = str(target_img.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        rel_path = str(target_img).replace("\\", "/")
    meta_record = {
        "segment_id": segment_id,
        "filepath": str(source_fp),
        "filename": source_fp.name,
        "date": date_str,
        "cam_index": cam_idx,
        "start_time": start_t,
        "end_time": end_t,
        "duration": float(segment.get("duration", 0.0)),
        "predicted_state": segment.get("state"),
        "manual_label": manual_label,
        "review_notes": segment.get("review_notes") or "",
        "frame_timestamp": t_mid,
        "archived_image_path": rel_path,
        "archived_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
    }

    import os
    payload = (json.dumps(meta_record, ensure_ascii=False) + "\n").encode("utf-8")
    with _manifest_lock:
        fd = os.open(manifest_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY | getattr(os, "O_BINARY", 0), 0o600)
        try:
            if os.write(fd, payload) != len(payload):
                raise OSError("Incomplete manifest append")
            os.fsync(fd)
        finally:
            os.close(fd)

    db.set_segment_archived_path(segment_id, rel_path)
    logger.info("已归档切片帧: %s (label=%s)", filename, manual_label)
    return target_img


def batch_archive_all_reviewed(
    db: VlogDatabase,
    archive_dir: Path | None = None,
    force: bool = False,
) -> dict[str, int]:
    """批量归档数据库中所有已人工审核且尚未归档的切片画面。"""
    with db._lock:
        if force:
            rows = db.conn.execute(
                "SELECT id FROM segments WHERE manual_label IS NOT NULL AND manual_label != 'UNREVIEWED'"
            ).fetchall()
        else:
            rows = db.conn.execute(
                """SELECT id FROM segments 
                   WHERE manual_label IS NOT NULL 
                     AND manual_label != 'UNREVIEWED'
                     AND (archived_frame_path IS NULL OR archived_frame_path = '')"""
            ).fetchall()

    seg_ids = [r["id"] for r in rows]
    total = len(seg_ids)
    archived = 0
    failed = 0

    for sid in seg_ids:
        res = extract_and_archive_frame(db, sid, archive_dir=archive_dir)
        if res is not None:
            archived += 1
        else:
            failed += 1

    return {"total": total, "archived": archived, "failed": failed}


def get_archive_stats(archive_dir: Path | None = None) -> dict[str, Any]:
    """统计归档库容量、图片总数与标签分布。"""
    arch_dir, images_dir, manifest_path = get_archive_dirs(archive_dir)

    total_images = 0
    total_bytes = 0
    if images_dir.exists():
        for f in images_dir.glob("*.jpg"):
            if f.is_file():
                total_images += 1
                total_bytes += f.stat().st_size

    label_counts: dict[str, int] = {}
    if manifest_path.exists():
        with _manifest_lock:
            try:
                with open(manifest_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                            lbl = item.get("manual_label", "UNKNOWN")
                            label_counts[lbl] = label_counts.get(lbl, 0) + 1
                        except Exception:
                            continue
            except Exception as e:
                logger.debug("读取 manifest 失败: %s", e)

    try:
        arch_rel = str(arch_dir.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        arch_rel = str(arch_dir).replace("\\", "/")

    return {
        "archive_dir": arch_rel,
        "total_images": total_images,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "label_counts": label_counts,
    }


def main():
    import argparse
    import sys

    # Windows 控制台编码防护
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="HomeVlog 人工复核画面归档管理工具")
    parser.add_argument("--stats", action="store_true", help="显示归档库统计信息")
    parser.add_argument("--batch", action="store_true", help="批量全量归档所有已复核但未归档的切片画面")
    parser.add_argument("--force", action="store_true", help="强制重新抽取并覆盖已存在的归档帧")

    args = parser.parse_args()

    db = VlogDatabase()
    try:
        if args.batch:
            print("[*] 开始执行全量打标切片帧归档...")
            res = batch_archive_all_reviewed(db, force=args.force)
            print(f"[+] 归档完成: 总计候选 {res['total']}, 成功归档 {res['archived']}, 失败/跳过 {res['failed']}")

        stats = get_archive_stats()
        print("\n" + "=" * 60)
        print(" [Archive] HomeVlog 真实反馈帧归档库状态")
        print("=" * 60)
        print(f" 物理目录: {stats['archive_dir']}")
        print(f" 归档图片: {stats['total_images']} 张")
        print(f" 占用空间: {stats['total_size_mb']} MB")
        print(f" 标签分布: {stats['label_counts'] if stats['label_counts'] else '暂无数据'}")
        print("=" * 60 + "\n")
    finally:
        db.close()


if __name__ == "__main__":
    main()
