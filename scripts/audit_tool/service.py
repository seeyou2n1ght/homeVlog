"""HomeVlog 独立审核工作台业务逻辑层 (Audit Service)

提供数据库检索、FFmpeg 截帧与 WebP 动图生成、YOLO 增强检测与打标导出功能。
"""

import io
import os
import csv
import json
import time
import logging
import threading
import subprocess
from pathlib import Path
from typing import Any

from src.utils import PROJECT_ROOT, load_config
from src.database import VlogDatabase

logger = logging.getLogger("homevlog.audit")
CACHE_DIR = PROJECT_ROOT / "temp" / "audit_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


class AuditService:
    def __init__(self, db: VlogDatabase | None = None):
        self.db = db or VlogDatabase()
        self.config = load_config()
        self._shared_yolo = None

    def get_overview(self) -> dict[str, Any]:
        """获取全局质量与审核统计大屏数据。"""
        with self.db._lock:
            total_files = self.db.conn.execute("SELECT COUNT(*) FROM file_tasks").fetchone()[0]
            analyzed_files = self.db.conn.execute(
                "SELECT COUNT(*) FROM file_tasks WHERE analysis_status='ANALYZED'"
            ).fetchone()[0]
            
            total_segments = self.db.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
            reviewed_segments = self.db.conn.execute(
                "SELECT COUNT(*) FROM segments WHERE manual_label IS NOT NULL"
            ).fetchone()[0]

            label_counts = dict(self.db.conn.execute(
                "SELECT manual_label, COUNT(*) FROM segments WHERE manual_label IS NOT NULL GROUP BY manual_label"
            ).fetchall())

            # 混淆矩阵统计
            # TP: DYNAMIC 被确认为有效动态
            tp = label_counts.get("CONFIRMED_MOTION", 0)
            # FP: 原判定为 DYNAMIC，但人工核验为 FALSE_ALARM (误判光影)
            fp = label_counts.get("FALSE_ALARM", 0)
            # FN: 原判定为 STATIC，但人工核验为 MISSED_MOTION (漏判有人)
            fn = label_counts.get("MISSED_MOTION", 0)
            # TN: 原判定为 STATIC，且确认为 CONFIRMED_STATIC
            tn = label_counts.get("CONFIRMED_STATIC", 0)

            precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 100.0
            recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 100.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 100.0

            return {
                "total_files": total_files,
                "analyzed_files": analyzed_files,
                "total_segments": total_segments,
                "reviewed_segments": reviewed_segments,
                "labels": {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                },
                "metrics": {
                    "precision": round(precision, 1),
                    "recall": round(recall, 1),
                    "f1_score": round(f1, 1),
                }
            }

    def get_file_tree(self) -> list[dict[str, Any]]:
        """获取按日期与摄像头聚合的文件树与审核进度。"""
        with self.db._lock:
            query = """
                SELECT f.id, f.filepath, f.cam_index, f.date, f.file_duration,
                       f.prescreen_status, f.analysis_status,
                       COUNT(s.id) as seg_count,
                       SUM(CASE WHEN s.manual_label IS NOT NULL THEN 1 ELSE 0 END) as reviewed_count
                FROM file_tasks f
                LEFT JOIN segments s ON f.id = s.file_id
                GROUP BY f.id
                ORDER BY f.date DESC, f.cam_index ASC, f.file_start_time ASC
            """
            rows = self.db.conn.execute(query).fetchall()

        tree_map: dict[str, dict[int, list[dict]]] = {}
        for r in rows:
            dt = r["date"]
            cam = r["cam_index"]
            item = {
                "id": r["id"],
                "filepath": r["filepath"],
                "filename": Path(r["filepath"]).name,
                "duration": round(float(r["file_duration"] or 0), 1),
                "prescreen_status": r["prescreen_status"],
                "analysis_status": r["analysis_status"],
                "seg_count": r["seg_count"],
                "reviewed_count": r["reviewed_count"],
            }
            tree_map.setdefault(dt, {}).setdefault(cam, []).append(item)

        result = []
        for dt, cams in tree_map.items():
            cam_list = []
            for cam_idx, files in cams.items():
                cam_list.append({
                    "cam_index": cam_idx,
                    "files": files,
                })
            result.append({
                "date": dt,
                "cameras": cam_list,
            })
        return result

    def get_anomalies(
        self,
        category: str = "all",
        date: str | None = None,
        cam_index: int | None = None,
        limit: int = 60,
    ) -> list[dict[str, Any]]:
        """获取疑难/争议切片优先队列 (Active Learning)，支持分类过滤。"""
        return self.db.get_anomaly_segments(category=category, date=date, cam_index=cam_index, limit=limit)

    def _resolve_local_timestamp(self, filepath: str, timestamp: float) -> tuple[float, float]:
        """将绝对秒数或相对秒数换算为安全的视频文件内相对秒数，并返回 (local_t, file_duration)。"""
        file_duration = 300.0
        file_offset = 0.0
        with self.db._lock:
            try:
                row = self.db.conn.execute(
                    "SELECT file_duration, file_start_time FROM file_tasks WHERE filepath=?",
                    (filepath,)
                ).fetchone()
                if row:
                    file_duration = float(row["file_duration"] or 300.0)
                    st_str = row["file_start_time"]
                    if st_str and len(st_str) >= 14:
                        hh = int(st_str[8:10])
                        mm = int(st_str[10:12])
                        ss = int(st_str[12:14])
                        file_offset = float(hh * 3600 + mm * 60 + ss)
            except Exception:
                pass

        if file_offset == 0.0:
            stem = Path(filepath).stem
            parts = stem.split("_")
            for p in parts:
                if len(p) == 14 and p.isdigit():
                    hh = int(p[8:10])
                    mm = int(p[10:12])
                    ss = int(p[12:14])
                    file_offset = float(hh * 3600 + mm * 60 + ss)
                    break

        local_t = float(timestamp)
        if file_offset > 0 and local_t >= file_offset:
            local_t = local_t - file_offset

        if file_duration > 0:
            local_t = max(0.0, min(local_t, max(0.0, file_duration - 0.1)))

        return local_t, file_duration

    def get_file_segments(self, file_id: int | None = None, filepath: str | None = None) -> dict[str, Any]:
        """获取单个文件的所有分段与时间轴信息。"""
        with self.db._lock:
            if file_id is not None:
                file_row = self.db.conn.execute("SELECT * FROM file_tasks WHERE id=?", (file_id,)).fetchone()
            elif filepath is not None:
                file_row = self.db.conn.execute("SELECT * FROM file_tasks WHERE filepath=?", (filepath,)).fetchone()
            else:
                return {"file": None, "segments": []}

            if not file_row:
                return {"file": None, "segments": []}

            fid = file_row["id"]
            segs = self.db.conn.execute(
                "SELECT * FROM segments WHERE file_id=? ORDER BY start_time ASC",
                (fid,)
            ).fetchall()

            return {
                "file": dict(file_row),
                "segments": [dict(s) for s in segs]
            }

    def extract_frame(self, filepath: str, timestamp: float, width: int = 640) -> Path | None:
        """从视频抽取指定时间点的画面并保存为 JPEG 缓存。支持文件缺失时生成优雅占位图。"""
        fp = Path(filepath)
        placeholder = CACHE_DIR / "_file_missing_placeholder.jpg"
        if not fp.exists() or fp.stat().st_size == 0:
            if not placeholder.exists():
                import numpy as np
                import cv2
                img = np.zeros((360, 640, 3), dtype=np.uint8)
                img[:] = (20, 25, 38)
                cv2.putText(img, "FILE NOT FOUND OR INACCESSIBLE", (80, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 116, 139), 2)
                cv2.putText(img, "(Check NAS Network Share Connection)", (120, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (71, 85, 105), 1)
                cv2.imwrite(str(placeholder), img)
            return placeholder

        local_t, file_duration = self._resolve_local_timestamp(filepath, timestamp)

        # 构造缓存键: 文件名_相对时间戳_分辨率
        cache_name = f"{fp.stem}_{local_t:.2f}_{width}.jpg"
        out_file = CACHE_DIR / cache_name
        if out_file.exists() and out_file.stat().st_size > 0:
            return out_file

        cmd = [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{local_t:.3f}",
            "-i", str(fp),
            "-vframes", "1",
            "-vf", f"scale={width}:-1",
            "-q:v", "3",
            str(out_file)
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=10)
            if res.returncode == 0 and out_file.exists() and out_file.stat().st_size > 0:
                return out_file
        except Exception as e:
            logger.error("FFmpeg frame extract failed for %s @ %.2f (local %.2f): %s", filepath, timestamp, local_t, e)

        if not placeholder.exists():
            try:
                import numpy as np
                import cv2
                img = np.zeros((360, 640, 3), dtype=np.uint8)
                img[:] = (20, 25, 38)
                cv2.putText(img, "FRAME EXTRACTION FAILED", (120, 170),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (239, 68, 68), 2)
                cv2.putText(img, f"File: {fp.name}", (120, 210),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (148, 163, 184), 1)
                cv2.imwrite(str(placeholder), img)
            except Exception:
                pass
        return placeholder if placeholder.exists() else None

    def generate_preview_clip(self, filepath: str, start_time: float, end_time: float, max_dur: float = 6.0) -> Path | None:
        """截取指定片段生成短动图 (WebP)，供浏览器直接预览。"""
        fp = Path(filepath)
        if not fp.exists():
            return None

        local_start, file_duration = self._resolve_local_timestamp(filepath, start_time)
        local_end, _ = self._resolve_local_timestamp(filepath, end_time)
        if local_end <= local_start:
            local_end = min(file_duration, local_start + 4.0)

        actual_dur = min(max(local_end - local_start, 1.0), max_dur)
        cache_name = f"{fp.stem}_{local_start:.1f}_{actual_dur:.1f}.webp"
        out_file = CACHE_DIR / cache_name
        if out_file.exists() and out_file.stat().st_size > 0:
            return out_file

        cmd = [
            "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
            "-ss", f"{local_start:.3f}",
            "-t", f"{actual_dur:.3f}",
            "-i", str(fp),
            "-vf", "fps=10,scale=480:-1:flags=lanczos",
            "-vcodec", "libwebp",
            "-lossless", "0",
            "-compression_level", "3",
            "-q:v", "60",
            "-loop", "0",
            str(out_file)
        ]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=15)
            if res.returncode == 0 and out_file.exists():
                return out_file
        except Exception as e:
            logger.error("FFmpeg clip generate failed for %s @ %.1f-%.1f (local %.1f): %s", filepath, start_time, end_time, local_start, e)
        return None

    def detect_and_draw_yolo(self, filepath: str, timestamp: float) -> dict[str, Any]:
        """对指定帧运行高精 YOLO 推理，并在图片上绘制预测框与置信度。"""
        import cv2
        local_t, _ = self._resolve_local_timestamp(filepath, timestamp)
        frame_path = self.extract_frame(filepath, local_t, width=800)
        if not frame_path or not frame_path.exists():
            return {"error": "Failed to extract frame"}

        # 加载共享 YOLO 模型
        if self._shared_yolo is None:
            from ultralytics import YOLO
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            # 优先尝试 models/yolo11s.pt, 其次 models/yolo11n.pt
            m_path = PROJECT_ROOT / "models" / "yolo11s.pt"
            if not m_path.exists():
                m_path = PROJECT_ROOT / "models" / "yolo11n.pt"
            
            try:
                self._shared_yolo = YOLO(str(m_path))
                self._shared_yolo.to(device)
            except Exception as e:
                return {"error": f"Failed to load YOLO model: {e}"}

        img = cv2.imread(str(frame_path))
        if img is None:
            return {"error": "Failed to read image"}

        results = self._shared_yolo(img, verbose=False)
        r = results[0]
        detected_objects = []

        if r.boxes is not None and len(r.boxes) > 0:
            names = r.names
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                xyxy = [int(v) for v in box.xyxy[0].tolist()]
                cls_name = names.get(cls_id, str(cls_id))

                detected_objects.append({
                    "class": cls_name,
                    "confidence": round(conf, 3),
                    "box": xyxy
                })

                # 绘制半透明边框与文字标签
                color = (0, 230, 115) if cls_name in ("person", "dog", "cat") else (255, 165, 0)
                cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), color, 2)
                label_txt = f"{cls_name} {conf:.2f}"
                cv2.putText(img, label_txt, (xyxy[0], max(xyxy[1] - 8, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # 保存带框图片
        drawn_name = f"annotated_{frame_path.name}"
        drawn_path = CACHE_DIR / drawn_name
        cv2.imwrite(str(drawn_path), img)

        return {
            "image_url": f"/api/frame_cached?name={drawn_name}",
            "objects": detected_objects,
            "count": len(detected_objects)
        }

    def submit_review(self, segment_id: int, manual_label: str, notes: str = "") -> bool:
        """提交人工打标修正并落盘，同时异步触发代表帧画面物理归档。"""
        ok = self.db.update_segment_review(segment_id, manual_label, notes)
        if ok and manual_label in ("CONFIRMED_MOTION", "VERIFIED_MOTION", "FALSE_ALARM", "MISSED_MOTION", "CONFIRMED_STATIC"):
            # 异步非阻塞执行代表帧物理归档
            def _async_archive():
                try:
                    from src.archiver import extract_and_archive_frame
                    extract_and_archive_frame(self.db, segment_id)
                except Exception as e:
                    logger.debug("Async frame archive error for seg %d: %s", segment_id, e)

            t = threading.Thread(target=_async_archive, daemon=True)
            t.start()
        return ok

    def get_archive_stats(self) -> dict[str, Any]:
        """获取反馈帧归档库容量与分布统计。"""
        from src.archiver import get_archive_stats
        return get_archive_stats()

    def batch_archive_all(self, force: bool = False) -> dict[str, int]:
        """全量补齐历史打标切片的代表帧归档。"""
        from src.archiver import batch_archive_all_reviewed
        return batch_archive_all_reviewed(self.db, force=force)

    def export_report(self, fmt: str = "csv") -> tuple[str, str]:
        """导出全量人工复核评估报表。"""
        with self.db._lock:
            rows = self.db.conn.execute("""
                SELECT s.id, s.date, s.cam_index, f.filepath,
                       s.start_time, s.end_time, s.duration,
                       s.state as predicted_state,
                       s.max_energy, s.avg_confidence,
                       s.manual_label, s.review_notes, s.reviewed_at
                FROM segments s
                JOIN file_tasks f ON s.file_id = f.id
                ORDER BY s.date, s.cam_index, s.start_time
            """).fetchall()

        data = [dict(r) for r in rows]
        if fmt == "json":
            return json.dumps(data, indent=2, ensure_ascii=False), "application/json"

        # 默认 CSV
        output = io.StringIO()
        if data:
            writer = csv.DictWriter(output, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
        return output.getvalue(), "text/csv; charset=utf-8"
