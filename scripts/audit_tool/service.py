"""HomeVlog 独立审核工作台业务逻辑层 (Audit Service)

提供数据库检索、FFmpeg 截帧与 WebP 动图生成、YOLO 增强检测与打标导出功能。
"""

import io
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
from src.feedback import normalize_label

logger = logging.getLogger("homevlog.audit")
CACHE_DIR = PROJECT_ROOT / "temp" / "audit_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR = PROJECT_ROOT / "data" / "feedback_archive"


def pack_notes(notes: str = "", scenario: str = "") -> str:
    """结构化存储场景归因与用户备注。"""
    notes_clean = (notes or "").strip()
    scenario_clean = (scenario or "").strip()
    if not scenario_clean:
        return notes_clean
    return json.dumps({"scenario": scenario_clean, "notes": notes_clean}, ensure_ascii=False)


def unpack_notes(notes_str: str | None) -> tuple[str, str]:
    """解析结构化备注，返回 (scenario, user_notes)。"""
    if not notes_str:
        return "", ""
    s = str(notes_str).strip()
    if s.startswith("{") and s.endswith("}"):
        try:
            data = json.loads(s)
            if isinstance(data, dict):
                return str(data.get("scenario", "")).strip(), str(data.get("notes", "")).strip()
        except Exception:
            pass
    if s.startswith("[") and "]" in s:
        sc, _, rest = s[1:].partition("]")
        return sc.strip(), rest.strip()
    return "", s


class AuditService:
    def __init__(self, db: VlogDatabase | None = None):
        self.db = db or VlogDatabase()
        self.config = load_config()
        self._shared_yolo = None

    def is_registered_filepath(self, filepath: str) -> bool:
        if not filepath:
            return False
        with self.db._lock:
            row = self.db.conn.execute(
                "SELECT 1 FROM file_tasks WHERE filepath=?", (str(Path(filepath)),)
            ).fetchone()
        return row is not None

    def get_overview(self) -> dict[str, Any]:
        """获取全局质量与审核统计大屏数据，严格依据预测状态与人工真值交叉构建混淆矩阵。"""
        with self.db._lock:
            total_files = self.db.conn.execute("SELECT COUNT(*) FROM file_tasks").fetchone()[0]
            analyzed_files = self.db.conn.execute(
                "SELECT COUNT(*) FROM file_tasks WHERE analysis_status='ANALYZED'"
            ).fetchone()[0]

            total_segments = self.db.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
            reviewed_segments = self.db.conn.execute(
                "SELECT COUNT(*) FROM segments WHERE manual_label IS NOT NULL"
            ).fetchone()[0]

            rows = self.db.conn.execute(
                "SELECT state, manual_label, COUNT(*) as cnt FROM segments WHERE manual_label IS NOT NULL GROUP BY state, manual_label"
            ).fetchall()

            tp = 0
            fp = 0
            fn = 0
            tn = 0
            for r in rows:
                st = r["state"]
                lbl = normalize_label(r["manual_label"])
                cnt = int(r["cnt"])
                if st in ("DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE"):
                    if lbl in ("CONFIRMED_MOTION", "VERIFIED_MOTION", "TP"):
                        tp += cnt
                    elif lbl in ("FALSE_ALARM", "FP", "CONFIRMED_STATIC"):
                        fp += cnt
                elif st in ("STATIC", "NIGHT_STATIONARY"):
                    if lbl in ("MISSED_MOTION", "FN", "CONFIRMED_MOTION"):
                        fn += cnt
                    elif lbl in ("CONFIRMED_STATIC", "TN"):
                        tn += cnt

            precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else None
            recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else None
            f1 = (2 * tp / (2 * tp + fp + fn)) * 100 if (2 * tp + fp + fn) else None

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
                    "precision": round(precision, 1) if precision is not None else None,
                    "recall": round(recall, 1) if recall is not None else None,
                    "f1_score": round(f1, 1) if f1 is not None else None,
                },
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
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        """获取疑难/争议切片优先队列 (Active Learning)，支持分类过滤并附带人眼可读绝对时间戳与场景归因。"""
        rows = self.db.get_anomaly_segments(
            category=category, date=date, cam_index=cam_index, limit=limit, offset=offset
        )
        for item in rows:
            sc, user_notes = unpack_notes(item.get("review_notes"))
            item["scenario"] = sc
            item["user_notes"] = user_notes
            st = float(item.get("start_time", 0.0))
            et = float(item.get("end_time", 0.0))
            dt_str = item.get("date", "")
            base_unix = 0.0
            if dt_str and len(dt_str) == 8:
                try:
                    from src.utils import ts_to_unix
                    base_unix = ts_to_unix(dt_str + "000000")
                except Exception:
                    pass
            if base_unix > 0:
                item["wall_start_str"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(base_unix + st))
                item["wall_end_str"] = time.strftime("%H:%M:%S", time.localtime(base_unix + et))
            else:
                item["wall_start_str"] = f"+{st:.1f}s"
                item["wall_end_str"] = f"+{et:.1f}s"
            item["in_file_start"] = round(max(0.0, st - float(item.get("file_start_offset", 0.0))), 2)
            item["in_file_end"] = round(max(0.0, et - float(item.get("file_start_offset", 0.0))), 2)
        return rows

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

            seg_list = []
            for s in segs:
                sd = dict(s)
                sc, user_notes = unpack_notes(sd.get("review_notes"))
                sd["scenario"] = sc
                sd["user_notes"] = user_notes
                seg_list.append(sd)

            return {
                "file": dict(file_row),
                "segments": seg_list
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
        ]
        if width > 0:
            cmd.extend(["-vf", f"scale={width}:-1"])
        cmd.extend(["-q:v", "2", str(out_file)])
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

        # 加载共享 YOLO 模型，严格遵循生产配置 settings.yaml
        if self._shared_yolo is None:
            from ultralytics import YOLO
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            model_rel = self.config.get("yolo", {}).get("model_path", "models/yolo11m.pt")
            m_path = PROJECT_ROOT / model_rel
            if not m_path.exists():
                for fb in ["models/yolo11s.pt", "models/yolo11n.pt"]:
                    fb_path = PROJECT_ROOT / fb
                    if fb_path.exists():
                        m_path = fb_path
                        break

            try:
                self._shared_yolo = YOLO(str(m_path))
                self._shared_yolo.to(device)
            except Exception as e:
                return {"error": f"Failed to load YOLO model from {m_path}: {e}"}

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

    def submit_review(
        self,
        segment_id: int,
        manual_label: str,
        notes: str = "",
        scenario: str = "",
    ) -> bool:
        """提交人工打标修正并落盘，同时异步触发代表帧画面物理归档。"""
        packed = pack_notes(notes=notes, scenario=scenario)
        ok = self.db.update_segment_review(segment_id, manual_label, packed)
        if ok and manual_label in (
            "CONFIRMED_MOTION", "VERIFIED_MOTION", "FALSE_ALARM", "MISSED_MOTION", "CONFIRMED_STATIC",
            "TP", "FP", "FN", "TN"
        ):
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

    def clear_review(self, segment_id: int) -> bool:
        """撤销人工审核标注。"""
        if hasattr(self.db, "clear_segment_review"):
            return self.db.clear_segment_review(segment_id)
        with self.db._lock:
            try:
                self.db.conn.execute(
                    "UPDATE segments SET manual_label=NULL, review_notes=NULL, reviewed_at=NULL WHERE id=?",
                    (segment_id,),
                )
                self.db.conn.commit()
                return True
            except Exception as e:
                logger.error("Failed to clear review for segment %s: %s", segment_id, e)
                return False

    def save_bounding_box(
        self,
        boxes: list[dict],
        image_name: str | None = None,
        segment_id: int | None = None,
    ) -> dict[str, Any]:
        """保存人工标注的目标边界框 (BBox) 至 .txt 标注文件 (YOLO 格式)。
        
        boxes 格式示例:
            [{"class_id": 0, "x_center": 0.45, "y_center": 0.52, "width": 0.12, "height": 0.28}, ...]
        """
        valid_lines = []
        for b in boxes:
            cls_id = int(b.get("class_id", b.get("class", 0)))
            # COCO: 0=person, 15=cat, 16=dog
            if cls_id not in (0, 15, 16):
                cls_id = 0
            xc = float(b.get("x_center", b.get("xc", 0.0)))
            yc = float(b.get("y_center", b.get("yc", 0.0)))
            w = float(b.get("width", b.get("w", 0.0)))
            h = float(b.get("height", b.get("h", 0.0)))

            # 约束归一化范围 [0, 1]
            xc = max(0.0, min(1.0, xc))
            yc = max(0.0, min(1.0, yc))
            w = max(0.001, min(1.0, w))
            h = max(0.001, min(1.0, h))

            valid_lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        txt_content = "\n".join(valid_lines) + ("\n" if valid_lines else "")
        saved_paths = []

        # 1. 若提供了 segment_id，写入归档目录中的对应伴随 txt
        if segment_id is not None:
            seg = self.db.get_segment_by_id(segment_id)
            if seg:
                arch_path = seg.get("archived_frame_path")
                if not arch_path:
                    try:
                        from src.archiver import extract_and_archive_frame
                        arch_path = extract_and_archive_frame(self.db, segment_id)
                    except Exception as e:
                        logger.debug("Failed to extract frame during save_bbox: %s", e)

                if arch_path:
                    p = Path(arch_path)
                    if not p.is_absolute():
                        p = PROJECT_ROOT / p
                    txt_p = p.with_suffix(".txt")
                    txt_p.parent.mkdir(parents=True, exist_ok=True)
                    txt_p.write_text(txt_content, encoding="utf-8")
                    saved_paths.append(str(txt_p))

        # 2. 若提供了 image_name，同时在 cache 或 archive 中保存伴随 txt
        if image_name:
            clean_name = Path(image_name).name
            # 在临时缓存目录保存
            cache_txt = CACHE_DIR / f"{Path(clean_name).stem}.txt"
            cache_txt.write_text(txt_content, encoding="utf-8")
            saved_paths.append(str(cache_txt))

            # 在归档 images 目录查找若存在同名图片，也写入 txt
            arch_img = ARCHIVE_DIR / "images" / clean_name
            if arch_img.exists():
                arch_txt = arch_img.with_suffix(".txt")
                arch_txt.write_text(txt_content, encoding="utf-8")
                saved_paths.append(str(arch_txt))

        return {
            "success": True,
            "box_count": len(valid_lines),
            "saved_paths": saved_paths,
        }

    def export_yolo_dataset(
        self,
        val_ratio: float = 0.2,
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """将已复核并归档的样本一键导出为标准 YOLO 数据集结构。"""
        from scripts.export_dataset import (
            build_yolo_dataset,
            load_archive_items,
            load_db_fallback_items,
        )

        out_path = Path(output_dir) if output_dir else (PROJECT_ROOT / "data" / "yolo_dataset")
        archive_dir = ARCHIVE_DIR

        # 确保历史打标项尽可能归档
        self.batch_archive_all()

        items = load_archive_items(archive_dir)
        if not items:
            db_path = PROJECT_ROOT / "data" / "vlog.db"
            items = load_db_fallback_items(db_path)

        if not items:
            return {
                "success": False,
                "message": "未找到任何已复核的样本数据。请先在工作台审核打标切片。",
                "stats": {},
            }

        m_path = PROJECT_ROOT / self.config.get("yolo", {}).get("model_path", "models/yolo11m.pt")
        if not m_path.exists():
            for fallback in ["models/yolo11s.pt", "models/yolo11n.pt"]:
                fb = PROJECT_ROOT / fallback
                if fb.exists():
                    m_path = fb
                    break

        stats = build_yolo_dataset(
            items=items,
            output_dir=out_path,
            val_ratio=val_ratio,
            yolo_model_path=m_path if m_path.exists() else None,
        )

        return {
            "success": True,
            "output_dir": str(out_path),
            "yaml_path": str(out_path / "data.yaml"),
            "train_cmd": f"uv run python scripts/train_yolo.py --data {out_path / 'data.yaml'}",
            "stats": stats,
        }

    def get_tuning_insights(self) -> dict[str, Any]:
        """根据已审核样本的场景归因与能量/置信度分布，生成 settings.yaml 量化调优建议。"""
        with self.db._lock:
            rows = self.db.conn.execute("""
                SELECT state, manual_label, max_energy, avg_confidence, duration, review_notes
                FROM segments
                WHERE manual_label IS NOT NULL
            """).fetchall()

        scenario_counts: dict[str, int] = {}
        fp_energies: list[float] = []
        fn_energies: list[float] = []
        tp_count = 0
        fp_count = 0
        fn_count = 0
        tn_count = 0
        short_jitter_count = 0

        for r in rows:
            st = r["state"]
            lbl = normalize_label(r["manual_label"])
            e = float(r["max_energy"] or 0.0)
            dur = float(r["duration"] or 0.0)
            sc, _ = unpack_notes(r["review_notes"])
            if sc:
                scenario_counts[sc] = scenario_counts.get(sc, 0) + 1

            if st in ("DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE"):
                if lbl in ("CONFIRMED_MOTION", "VERIFIED_MOTION", "TP"):
                    tp_count += 1
                elif lbl in ("FALSE_ALARM", "FP", "CONFIRMED_STATIC"):
                    fp_count += 1
                    fp_energies.append(e)
                    if dur < 3.0:
                        short_jitter_count += 1
            elif st in ("STATIC", "NIGHT_STATIONARY"):
                if lbl in ("MISSED_MOTION", "FN", "CONFIRMED_MOTION"):
                    fn_count += 1
                    fn_energies.append(e)
                elif lbl in ("CONFIRMED_STATIC", "TN"):
                    tn_count += 1

        recommendations = []
        det_cfg = self.config.get("detection", {})
        seg_cfg = self.config.get("segment", {})
        yolo_cfg = self.config.get("yolo", {})
        presence_cfg = self.config.get("presence", {})

        # 1. 光影/车灯误报调优
        light_shadow_fp = scenario_counts.get("LIGHT_SHADOW", 0) + scenario_counts.get("HEADLIGHT", 0)
        curr_thresh = float(det_cfg.get("min_motion_threshold", 2.5))
        if light_shadow_fp >= 3 or (fp_count >= 5 and fp_energies):
            sorted_fp = sorted(fp_energies)
            p75 = sorted_fp[int(len(sorted_fp) * 0.75)] if sorted_fp else curr_thresh
            suggested_thresh = round(min(max(p75 * 1.1, curr_thresh + 0.5), 5.5), 1)
            recommendations.append({
                "category": "误报抑制 (光影刚性)",
                "param": "detection.min_motion_threshold",
                "current_value": curr_thresh,
                "suggested_value": suggested_thresh,
                "rationale": f"已复核 {fp_count} 处误报 (含 {light_shadow_fp} 处光影/车灯)，75分位能量为 {p75:.1f}。调高底噪门限可有效过滤刚性光影。"
            })
            recommendations.append({
                "category": "误报抑制 (慢速漫射)",
                "param": "detection.ambient_drift_suppress",
                "current_value": det_cfg.get("ambient_drift_suppress", True),
                "suggested_value": True,
                "rationale": "确保大面积慢速光影漫射软抑制处于开启状态。"
            })

        # 2. 窗帘风动误报调优
        curtain_fp = scenario_counts.get("CURTAIN", 0)
        if curtain_fp >= 2:
            recommendations.append({
                "category": "误报抑制 (窗帘摆动)",
                "param": "detection.roi_crop",
                "current_value": str(det_cfg.get("roi_crop", [0.1, 0.12, 0.8, 0.85])),
                "suggested_value": "在机位配置中设置遮挡区域或裁剪 ROI",
                "rationale": f"检测到 {curtain_fp} 处窗帘误报。物理边缘摆动建议通过 ROI 裁剪或机位屏蔽区彻底排除。"
            })

        # 3. 婴儿/弱光动作漏报调优
        infant_fn = scenario_counts.get("INFANT_MOTION", 0) + scenario_counts.get("DARK_ROOM", 0)
        curr_yolo_conf = float(yolo_cfg.get("confidence", 0.3))
        if infant_fn >= 2 or fn_count >= 3:
            suggested_conf = max(0.18, round(curr_yolo_conf - 0.05, 2))
            recommendations.append({
                "category": "漏报召回 (微弱动静/微光)",
                "param": "yolo.confidence",
                "current_value": curr_yolo_conf,
                "suggested_value": suggested_conf,
                "rationale": f"检测到 {fn_count} 处动作漏报 (含 {infant_fn} 处婴儿微动/暗光)。建议适当调低检测置信度下限以提升召回率。"
            })
            recommendations.append({
                "category": "驻留保护 (人物置信度)",
                "param": "presence.person_conf_threshold",
                "current_value": presence_cfg.get("person_conf_threshold", 0.20),
                "suggested_value": 0.15,
                "rationale": "调低驻留状态下限，防止暗光下人物坐卧时过早退出 4x 驻留陪伴流。"
            })

        # 4. 短碎片毛刺调优
        curr_min_dur = float(seg_cfg.get("min_motion_duration", 2.0))
        if short_jitter_count >= 3:
            recommendations.append({
                "category": "毛刺过滤 (极短碎片)",
                "param": "segment.min_motion_duration",
                "current_value": curr_min_dur,
                "suggested_value": round(curr_min_dur + 0.5, 1),
                "rationale": f"发现 {short_jitter_count} 处 <3.0s 的短突发误报切片。提高最小运动时长门槛可平滑吸收瞬间干扰。"
            })

        return {
            "total_reviewed": len(rows),
            "distribution": {
                "tp": tp_count,
                "fp": fp_count,
                "fn": fn_count,
                "tn": tn_count,
            },
            "scenario_counts": scenario_counts,
            "energy_stats": {
                "fp_count": len(fp_energies),
                "fp_avg_energy": round(sum(fp_energies) / len(fp_energies), 2) if fp_energies else 0.0,
                "fn_count": len(fn_energies),
                "fn_avg_energy": round(sum(fn_energies) / len(fn_energies), 2) if fn_energies else 0.0,
            },
            "recommendations": recommendations,
        }

    def get_archive_stats(self) -> dict[str, Any]:
        """获取反馈帧归档库容量与分布统计。"""
        from src.archiver import get_archive_stats
        return get_archive_stats()

    def batch_archive_all(self, force: bool = False) -> dict[str, int]:
        """全量补齐历史打标切片的代表帧归档。"""
        from src.archiver import batch_archive_all_reviewed
        return batch_archive_all_reviewed(self.db, force=force)

    def export_report(self, fmt: str = "csv", only_reviewed: bool = False) -> tuple[str, str]:
        """导出全量或已复核的人工评估报表。"""
        where_clause = "WHERE s.manual_label IS NOT NULL" if only_reviewed else ""
        with self.db._lock:
            rows = self.db.conn.execute(f"""
                SELECT s.id, s.date, s.cam_index, f.filepath,
                       s.start_time, s.end_time, s.duration,
                       s.state as predicted_state,
                       s.max_energy, s.avg_confidence,
                       s.manual_label, s.review_notes, s.reviewed_at
                FROM segments s
                JOIN file_tasks f ON s.file_id = f.id
                {where_clause}
                ORDER BY s.date, s.cam_index, s.start_time
            """).fetchall()

        data = []
        for r in rows:
            d = dict(r)
            sc, user_notes = unpack_notes(d.get("review_notes"))
            d["scenario"] = sc
            d["user_notes"] = user_notes
            data.append(d)

        if fmt == "json":
            return json.dumps(data, indent=2, ensure_ascii=False), "application/json"

        # 默认 CSV
        output = io.StringIO()
        if data:
            writer = csv.DictWriter(output, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)
        return output.getvalue(), "text/csv; charset=utf-8"
