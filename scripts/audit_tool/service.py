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
import hashlib
import os
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from src.utils import PROJECT_ROOT, load_config
from src.database import VlogDatabase
from src.feedback import normalize_label
from src.hardware.ffmpeg import FFmpegProcessRegistry, run_ffmpeg
from src.core.visual_index import VisualReaderStopError, inspect_source_identity_owned
from src.core.benchmark import (
    DEFAULT_BENCHMARK_FILE,
    VERSIONS_DIR,
    load_benchmark_dataset,
    merge_benchmark_samples,
    save_benchmark_dataset,
)
from src.core.tuner import (
    TuningSimulator,
    apply_config_safe,
    generate_config_diff,
    rollback_config,
)

logger = logging.getLogger("homevlog.audit")
CACHE_DIR = PROJECT_ROOT / "temp" / "audit_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR = PROJECT_ROOT / "data" / "feedback_archive"


REVIEW_BASES = {"unknown", "representative_frame", "video_interval", "video_audio_interval", "uncertain"}


def unpack_review_basis(notes_str: str | None) -> str:
    try:
        data = json.loads(notes_str or "null")
        basis = data.get("review_basis") if isinstance(data, dict) else None
        return basis if isinstance(basis, str) and basis in REVIEW_BASES else "unknown"
    except (TypeError, ValueError):
        return "unknown"


def pack_notes(notes: str = "", scenario: str = "", review_basis: str | None = None) -> str:
    """结构化存储场景归因与用户备注。"""
    notes_clean = (notes or "").strip()
    scenario_clean = (scenario or "").strip()
    if review_basis is not None and (not isinstance(review_basis, str) or review_basis not in REVIEW_BASES):
        raise ValueError("Invalid review_basis")
    if not scenario_clean and review_basis is None:
        return notes_clean
    data = {"scenario": scenario_clean, "notes": notes_clean}
    if review_basis is not None:
        data["review_basis"] = review_basis
    return json.dumps(data, ensure_ascii=False)


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
    CONTEXT_KEYS = (
        "detection.profile", "detection.motion_sensitivity", "detection.min_motion_threshold",
        "detection.fg_threshold", "detection.ambient_drift_suppress", "detection.camera_motion_suppress",
        "segment.min_motion_duration", "yolo.confidence", "presence.speed_factor",
        "presence.night_speed_factor", "micro_motion.day_energy_threshold",
        "micro_motion.night_energy_threshold", "audio_vad.noise_margin_db", "audio_vad.min_dbfs",
    )

    def get_review_context(self, segment_id: int) -> dict[str, Any]:
        from src.core import config as core
        from src.core.profiles import resolve_camera_config, _explicit_config
        from src.core.tuner import _get_nested_param
        from src.stages.timeline import build_timeline_from_rows
        from src.core.replay_evidence import interval_display_details
        from src.core.utils import ts_to_unix
        with self.db._lock:
            connection = self.db.conn
            owns_transaction = not connection.in_transaction
            if owns_transaction:
                connection.execute("BEGIN")
            try:
                segment = self.db.get_segment_by_id(segment_id)
                if not segment:
                    raise ValueError("Unknown segment")
                tasks = self.db.get_all_file_tasks_for_date(segment["date"], segment["cam_index"],
                                                           filepaths=[segment["filepath"]])
            finally:
                if owns_transaction:
                    connection.rollback()
        if not tasks:
            raise ValueError("Source metadata unavailable")
        with core._config_lock:
            self.config = load_config(reload=True)
            config = self.config
            effective = resolve_camera_config(config, filepath=segment["filepath"], cam_index=segment["cam_index"])
            config_hash = generate_config_diff({})["config_sha256"]
        explicit = _explicit_config(config)
        parameters = [{"param": key, "global": _get_nested_param(config, key),
                       "camera": _get_nested_param(effective, key),
                       "explicit_global": _get_nested_param(explicit, key) is not None}
                      for key in self.CONTEXT_KEYS]
        row = tasks[0]
        unreviewed = deepcopy(row)
        unreviewed["human_reviews"] = []
        for item in unreviewed.get("segments", []):
            item["manual_label"] = item["human_label"] = None
        offset = ts_to_unix(row["file_start_time"]) - ts_to_unix(segment["date"] + "000000")
        comparison = {}
        for name, input_row in (("analysis", unreviewed), ("reviewed", row)):
            timeline = build_timeline_from_rows([input_row], segment["date"], config=effective, resolve_presence=False)
            comparison[name] = interval_display_details(effective, timeline, segment["start_time"], segment["end_time"], offset)
        scenario, notes = unpack_notes(segment.get("review_notes"))
        return {"segment": {key: segment.get(key) for key in ("id", "filepath", "date", "cam_index", "start_time", "end_time", "state", "manual_label")},
                "scenario": scenario, "user_notes": notes,
                "review_basis": unpack_review_basis(segment.get("review_notes")), "parameters": parameters,
                "config_sha256": config_hash, "comparison": comparison,
                "comparison_basis": "saved_analysis_vs_human_overlay_shared_plan",
                "notice": "展示贡献按共享计划比例分摊；不是已渲染帧数或识别收益。动态噪声门限仍随输入变化。"}
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

    def exclude_file(self, filepath: str, reason: str = "MANUAL_EXCLUDED") -> bool:
        """从工作台手动排除坏片/无需处理的原片"""
        return self.db.exclude_file_task(filepath, reason=reason)

    def get_overview(self) -> dict[str, Any]:
        """获取审核库存；准确度只能来自独立参考上的生产重放。"""
        with self.db._lock:
            total_files = self.db.conn.execute("SELECT COUNT(*) FROM file_tasks").fetchone()[0]
            analyzed_files = self.db.conn.execute(
                "SELECT COUNT(*) FROM file_tasks WHERE analysis_status='ANALYZED'"
            ).fetchone()[0]

            total_segments = self.db.conn.execute("SELECT COUNT(*) FROM segments").fetchone()[0]
            reviewed_segments = self.db.conn.execute(
                "SELECT COUNT(*) FROM segments WHERE manual_label IS NOT NULL"
            ).fetchone()[0]

            inventory = self.get_tuning_insights()
            tp, fp, fn, tn = (inventory["distribution"][key] for key in ("tp", "fp", "fn", "tn"))
            # Label inventory can be changed by human overlays and is not a
            # current independent prediction measurement.
            precision = recall = f1 = None

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
                "labels_basis": inventory["distribution_basis"],
                "unknown_labels": inventory["unknown_labels"],
                "metrics_basis": "requires_independent_production_replay",
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
        return self._decorate_review_rows(rows)

    @staticmethod
    def _decorate_review_rows(rows):
        for item in rows:
            sc, user_notes = unpack_notes(item.get("review_notes"))
            item["scenario"] = sc
            item["user_notes"] = user_notes
            item["review_basis"] = unpack_review_basis(item.get("review_notes"))
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

    def get_review_queue(self, category="gold_candidates", *, date=None, cam_index=None,
                         state=None, scenario=None, search="", hide_reviewed=False,
                         limit=50, offset=0, seed=0, reference_id=None):
        """Read a live inventory page or reproducible random audit without changing labels."""
        if category not in ("gold_candidates", "all", "reviewed", "random", "visual"):
            raise ValueError("Unknown review queue category")
        if not 1 <= limit <= 200 or offset < 0 or not 0 <= seed <= 2147483647:
            raise ValueError("Invalid queue page or random seed")
        visual_reference = None
        if category == "visual":
            from src.core.visual_index import source_identity, signature_distance
            with self.db._lock:
                reference = self.db.conn.execute("""SELECT vf.* FROM visual_features vf JOIN segments s ON s.id=vf.segment_id
                    AND s.filepath=vf.filepath AND s.start_time=vf.start_time AND s.end_time=vf.end_time
                    AND s.file_start_offset=vf.file_start_offset WHERE s.id=?""",(reference_id,)).fetchone()
            if reference:
                try:
                    visual_reference = dict(reference)
                    if json.loads(reference["source_identity"]) != source_identity(reference["filepath"]):
                        visual_reference = None
                    else:
                        signature_distance(reference["signature"],reference["signature"])
                except (OSError,ValueError,KeyError,TypeError):
                    visual_reference = None
        clauses, params = ["1=1"], []
        for column, value in (("date", date), ("cam_index", cam_index), ("state", state), ("audit_scene", scenario)):
            if value is not None and value != "":
                clauses.append(f"{column}=?")
                params.append(value)
        if search:
            if len(search) > 200:
                raise ValueError("Search is too long")
            escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            clauses.append("(filepath LIKE ? ESCAPE '\\' OR state LIKE ? ESCAPE '\\' OR CAST(id AS TEXT) LIKE ? ESCAPE '\\' OR review_reason LIKE ? ESCAPE '\\' OR review_notes LIKE ? ESCAPE '\\')")
            params.extend([f"%{escaped}%"] * 5)
        prefix = """WITH annotated AS (
            SELECT *, CASE WHEN json_valid(review_notes)
                THEN COALESCE(NULLIF(TRIM(CAST(json_extract(review_notes,'$.scenario') AS TEXT)),''),'UNKNOWN')
                ELSE 'UNKNOWN' END AS audit_scene FROM segments),
            scoped AS (SELECT * FROM annotated WHERE """ + " AND ".join(clauses) + "), "
        grouping = "filepath,date,cam_index,state,audit_scene,CAST(start_time/300 AS INTEGER)"
        representatives = prefix + f"""ranked AS (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY {grouping}
                ORDER BY needs_review DESC,max_energy DESC,id ASC) AS representative_rank,
                COUNT(*) OVER (PARTITION BY {grouping}) AS represented_count
            FROM scoped WHERE manual_label IS NULL),
            reps AS (SELECT * FROM ranked WHERE representative_rank=1),
            diverse AS (SELECT *,ROW_NUMBER() OVER (
                PARTITION BY date,cam_index,state,audit_scene ORDER BY needs_review DESC,max_energy DESC,id) AS diversity_rank FROM reps) """
        with self.db._lock:
            connection = self.db.conn
            owns_transaction = not connection.in_transaction
            if owns_transaction:
                connection.execute("BEGIN")
            try:
                stock = connection.execute(prefix + "totals AS (SELECT COUNT(*) AS total,COUNT(manual_label) AS reviewed FROM scoped) SELECT * FROM totals", params).fetchone()
                count_selected = connection.execute(prefix + f"groups AS (SELECT 1 FROM scoped WHERE manual_label IS NULL GROUP BY {grouping}) SELECT COUNT(*) FROM groups", params).fetchone()[0]
                counts = {"all": stock["total"], "reviewed": stock["reviewed"],
                          "unreviewed": stock["total"]-stock["reviewed"], "gold_candidates": count_selected}
                indexed = None
                captured_indexed = None
                if category == "visual":
                    from src.core.visual_index import signature_distance
                    def distance(signature):
                        try:
                            return signature_distance(visual_reference["signature"],signature) if visual_reference else None
                        except (ValueError,KeyError,TypeError):
                            return None
                    def valid_signature(signature):
                        try:
                            signature_distance(signature,signature)
                            return 1
                        except (ValueError,KeyError,TypeError):
                            return 0
                    connection.create_function("audit_visual_valid",1,valid_signature)
                    connection.create_function("audit_visual_distance",1,distance)
                    visual_query = prefix+"""features AS (
                        SELECT s.*,v.frame_time AS visual_frame_time,v.origin AS visual_origin,
                            v.source_identity AS visual_source_identity,audit_visual_distance(v.signature) AS visual_distance
                        FROM scoped s JOIN visual_features v ON s.id=v.segment_id AND s.filepath=v.filepath
                            AND s.start_time=v.start_time AND s.end_time=v.end_time AND s.file_start_offset=v.file_start_offset
                        WHERE v.frame_time >= s.start_time-s.file_start_offset AND v.frame_time < s.end_time-s.file_start_offset
                            AND audit_visual_valid(v.signature)=1),
                        result AS (SELECT * FROM features WHERE visual_distance IS NOT NULL"""
                    if hide_reviewed:
                        visual_query += " AND manual_label IS NULL"
                    visual_query += ") "
                    total = connection.execute(visual_query+"SELECT COUNT(*) FROM result",params).fetchone()[0]
                    indexed = total
                    captured_indexed = connection.execute(visual_query+"SELECT COUNT(*) FROM features",params).fetchone()[0]
                    query = visual_query+"SELECT * FROM result ORDER BY visual_distance,id LIMIT ? OFFSET ?"
                    query_params = params+[limit,offset]
                elif category == "gold_candidates":
                    total = count_selected
                    query = representatives + "SELECT * FROM diverse ORDER BY diversity_rank,date DESC,cam_index,state,audit_scene,id LIMIT ? OFFSET ?"
                    query_params = params + [limit, offset]
                else:
                    filter_sql = " WHERE manual_label IS NOT NULL" if category == "reviewed" else " WHERE 1=1"
                    if hide_reviewed:
                        filter_sql += " AND manual_label IS NULL"
                    total = connection.execute(prefix + "result AS (SELECT * FROM scoped" + filter_sql + ") SELECT COUNT(*) FROM result", params).fetchone()[0]
                    if category == "random":
                        connection.create_function("audit_random_rank", 2, lambda row_id, value: hashlib.blake2b(f"{value}:{row_id}".encode(), digest_size=8).hexdigest(), deterministic=True)
                        order, extra = "audit_random_rank(id,?),id", [seed]
                    else:
                        order, extra = "date DESC,cam_index,filepath,start_time,id", []
                    query = prefix + "result AS (SELECT * FROM scoped" + filter_sql + f") SELECT * FROM result ORDER BY {order} LIMIT ? OFFSET ?"
                    query_params = params + extra + [limit, offset]
                rows = [dict(row) for row in connection.execute(query, query_params).fetchall()]
            finally:
                if owns_transaction:
                    connection.rollback()
        if category == "visual":
            from src.core.visual_index import source_identity
            identities = {}
            valid_rows = []
            for row in rows:
                filepath = row["filepath"]
                if filepath not in identities:
                    try:
                        identities[filepath] = source_identity(filepath)
                    except OSError:
                        identities[filepath] = None
                try:
                    expected = json.loads(row["visual_source_identity"])
                    if identities[filepath] is not None and isinstance(expected,list) and len(expected) == 3 and identities[filepath] == expected:
                        valid_rows.append(row)
                except (ValueError,TypeError):
                    logger.warning("Invalid visual source identity for segment %s",row["id"])
            rows = valid_rows
        for row in rows:
            row["reason_desc"] = row.get("review_reason") or (f"已复核 ({row['manual_label']})" if row.get("manual_label") else "未审片段")
            row["selection_basis"] = "source_time_state_scenario" if category == "gold_candidates" else category
            # Representative groups are not proven visually equivalent and must
            # never propagate a label to unseen members.
            row.pop("representative_rank", None)
            row.pop("diversity_rank", None)
        ids = [row["id"] for row in rows]
        return {"items": self._decorate_review_rows(rows), "counts": counts, "total_items": total,
                "status": "UNAVAILABLE" if category == "visual" and not visual_reference else "AVAILABLE",
                "reference_id": reference_id, "indexed_items": indexed, "captured_index_items": captured_indexed,
                "visual_notice": "按代表画面的外观近似程度排序，距离越小越相似；画面相似不说明事件相同，仅标当前复核项。未索引条目保留在全部库存。统计来自已采集画面，本页源已核对。" if category == "visual" else None,
                "offset": offset, "limit": limit, "seed": seed, "category": category,
                "scope": {"date": date, "cam_index": cam_index, "state": state, "scenario": scenario, "search": search, "hide_reviewed": hide_reviewed},
                "page_identity": hashlib.sha256(json.dumps([category, params, hide_reviewed, seed, ids]).encode()).hexdigest(),
                "notice": "精选按同源5分钟/状态/场景分组并跨日期机位轮询；不是视觉相似认证。随机抽查仅对未变化库存同种子复现，不自动认证其余片段。"}

    def get_delivery_comparison(self, segment_id):
        """Read artifact-bound mappings; never reconstruct historical plans from today's DB."""
        from src.core.config import OUTPUT_DIR
        import math
        segment = self.db.get_segment_by_id(segment_id)
        if not segment:
            raise ValueError("Unknown segment")
        root = OUTPUT_DIR.resolve()
        candidates = [(meta.with_suffix("").with_suffix(".mp4"), meta, "current") for meta in root.glob("*.meta.json")]
        problems = []
        for marker in root.glob(".delivery_backup_*/review_history.json"):
            try:
                history = json.loads(marker.read_text(encoding="utf-8"))
                output = Path(history["output"])
                assets = history["assets"]
                candidates.append((marker.parent / assets[output.name], marker.parent / assets[output.with_suffix(".meta.json").name], "previous"))
            except (OSError, ValueError, KeyError) as exc:
                problems.append(f"History index unavailable: {marker.name}: {exc}")
        versions = []
        for media, metadata, role in candidates:
            try:
                media, metadata = media.resolve(), metadata.resolve()
                if not media.is_relative_to(root) or not metadata.is_relative_to(root):
                    continue
                meta = json.loads(metadata.read_text(encoding="utf-8"))
                if meta.get("date") != segment["date"] or meta.get("camera", {}).get("cam_index") != segment["cam_index"]:
                    continue
                mapping = meta.get("delivery_mapping", {})
                if mapping.get("version") != 1 or mapping.get("status") != "AVAILABLE":
                    problems.append(f"{role}: {metadata.name} lacks a verified delivery mapping")
                    continue
                stat = media.stat()
                identity = mapping["media_identity"]
                if stat.st_size != identity["size"] or stat.st_mtime_ns != identity["mtime_ns"]:
                    raise ValueError("Media identity changed; stale mapping refused")
                fps = float(mapping["fps"])
                if not math.isfinite(fps) or fps <= 0 or fps != identity["fps"] or mapping["planned_frames"] != identity["frames"]:
                    raise ValueError("Invalid frame clock")
                start = float(segment["start_time"]) - float(segment.get("file_start_offset") or 0)
                end = float(segment["end_time"]) - float(segment.get("file_start_offset") or 0)
                intervals = []
                for item in mapping["segments"]:
                    if Path(item["source_file"]).resolve() != Path(segment["filepath"]).resolve():
                        continue
                    if item["source_end_s"] <= start or item["source_start_s"] >= end:
                        continue
                    first, last = int(item["output_start_frame"]), int(item["output_end_frame"])
                    if not 0 <= first < last <= identity["frames"]:
                        raise ValueError("Invalid mapped frame bounds")
                    intervals.append({"start": first/fps, "end": last/fps, "start_frame": first,
                                      "end_frame": last, "state": item["state"]})
                if not intervals:
                    continue
                version_id = hashlib.sha256(json.dumps([str(media), identity, mapping], sort_keys=True).encode()).hexdigest()
                versions.append({"version_id": version_id, "role": role, "filepath": str(media),
                    "media_identity": identity, "generated_at": meta.get("performance", {}).get("generated_at"),
                    "intervals": intervals})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                problems.append(f"{role}: {metadata.name}: {exc}")
        versions.sort(key=lambda item: (item["role"] != "current", item.get("generated_at") or ""))
        return {"segment_id": segment_id, "status": "AVAILABLE" if versions else "UNAVAILABLE",
                "versions": versions, "problems": problems,
                "notice": "播放审核区间覆盖的完整展示分段，包含上下文；静态缩略不证明动作保留。旧交付缺少映射时不推算。成功审核重渲会保留旧版本并占用磁盘。"}

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

    @staticmethod
    def _preview_identity(fp: Path) -> tuple:
        return inspect_source_identity_owned(fp, CACHE_DIR / 'source_identity', timeout_s=3)

    def _cached_preview(self, fp: Path, parameters: tuple, suffix: str, args: list[str], timeout: int) -> Path | None:
        """Publish only successful extraction from an unchanged source identity."""
        temporary = None
        try:
            identity = self._preview_identity(fp)
            key = hashlib.sha256(json.dumps((identity, parameters)).encode()).hexdigest()
            output = CACHE_DIR / f"preview_v2_{key}{suffix}"
            if output.exists() and output.stat().st_size > 0:
                return output
            temporary = CACHE_DIR / f"preview_pending_{uuid.uuid4().hex}{suffix}"
            result = run_ffmpeg(args[1:] + [str(temporary)], timeout=timeout, io_timeout=min(3.,timeout))
            if (result.returncode == 0 and not result.timed_out and not FFmpegProcessRegistry.is_interrupted()
                    and temporary.exists() and temporary.stat().st_size > 0
                    and self._preview_identity(fp) == identity):
                temporary.replace(output)
                return output
            logger.error("Preview extraction rejected for %s: exit=%s stderr=%r", fp, result.returncode, result.stderr[-2000:])
        except VisualReaderStopError as exc:
            logger.error("Preview identity reader unsafe; new source queries remain blocked until kernel exit is verified: %s", exc)
        except (OSError, subprocess.SubprocessError, RuntimeError) as exc:
            logger.error("Preview extraction failed for %s: %s", fp, exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return None

    def extract_frame(self, filepath: str, timestamp: float, width: int = 640) -> Path | None:
        """Extract one frame; failures are explicit and never cached as source frames."""
        local_t, _ = self._resolve_local_timestamp(filepath, timestamp)
        seek = f"{local_t:.3f}"
        args = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-ss", seek, "-i", filepath, "-vframes", "1"]
        if width > 0:
            args.extend(["-vf", f"scale={width}:-1"])
        args.extend(["-q:v", "2"])
        return self._cached_preview(Path(filepath), ("frame", seek, width), ".jpg", args, 10)

    def generate_preview_clip(self, filepath: str, start_time: float, end_time: float, max_dur: float = 6.0) -> Path | None:
        """Preview at most max_dur seconds, without extending short source intervals."""
        local_start, _ = self._resolve_local_timestamp(filepath, start_time)
        local_end, _ = self._resolve_local_timestamp(filepath, end_time)
        duration = min(local_end - local_start, max_dur)
        if duration <= 0 or round(duration, 3) <= 0:
            return None
        seek, length = f"{local_start:.3f}", f"{duration:.3f}"
        args = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                "-ss", seek, "-t", length, "-i", filepath,
                "-vf", "fps=10,scale=480:-1:flags=lanczos", "-vcodec", "libwebp",
                "-lossless", "0", "-compression_level", "3", "-q:v", "60", "-loop", "0"]
        return self._cached_preview(Path(filepath), ("clip", seek, length), ".webp", args, 15)

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
        preserve_attribution: bool = False,
        review_basis: str | None = None,
    ) -> bool:
        """提交人工打标修正并落盘，同时异步触发代表帧画面物理归档。"""
        packed = None if preserve_attribution and review_basis is None else pack_notes(notes=notes, scenario=scenario, review_basis=review_basis)
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

    def submit_batch_review(
        self,
        segment_ids: list[int],
        manual_label: str,
        notes: str = "",
        scenario: str = "",
        preserve_attribution: bool = False,
    ) -> int:
        """批量提交一组聚类切片的人工打标。"""
        if not segment_ids:
            return 0
        packed = None if preserve_attribution else pack_notes(notes=notes, scenario=scenario)
        count = self.db.batch_update_segment_review(segment_ids, manual_label, packed)
        if count > 0:
            def _async_archive():
                try:
                    from src.archiver import extract_and_archive_frame
                    extract_and_archive_frame(self.db, segment_ids[0])
                except Exception as e:
                    logger.debug("Async frame archive error for batch %s: %s", segment_ids, e)

            t = threading.Thread(target=_async_archive, daemon=True)
            t.start()
        return count

    def export_gold_benchmark(
        self,
        output_path: str | Path | None = None,
        version: str = "1.0",
        incremental_merge: bool = True,
        archive_version: bool = True,
    ) -> dict[str, Any]:
        """导出人工复核切片为标准的基准评测数据集，支持增量合并与历史快照版本化。"""
        out_file = Path(output_path) if output_path else DEFAULT_BENCHMARK_FILE

        with self.db._lock:
            rows = self.db.conn.execute(
                """SELECT id, filepath, cam_index, date, start_time, end_time, duration,
                          state, max_energy, avg_confidence, manual_label, review_notes, reviewed_at
                   FROM segments
                   WHERE manual_label IS NOT NULL
                   ORDER BY date DESC, filepath ASC, start_time ASC"""
            ).fetchall()

        benchmark_items = []
        for r in rows:
            sc, user_notes = unpack_notes(r["review_notes"])
            benchmark_items.append({
                "segment_id": r["id"],
                "filepath": r["filepath"],
                "filename": Path(r["filepath"]).name,
                "cam_index": r["cam_index"],
                "date": r["date"],
                "start_time": round(float(r["start_time"]), 2),
                "end_time": round(float(r["end_time"]), 2),
                "duration": round(float(r["duration"] or (r["end_time"] - r["start_time"])), 2),
                "algo_state": r["state"],
                "max_energy": round(float(r["max_energy"] or 0.0), 2),
                "avg_confidence": round(float(r["avg_confidence"] or 0.0), 3),
                "gold_label": normalize_label(r["manual_label"]),
                "raw_label": r["manual_label"],
                "scenario": sc,
                "notes": user_notes,
                "review_basis": unpack_review_basis(r["review_notes"]),
                "reviewed_at": r["reviewed_at"],
            })

        from src.core.benchmark import _dataset_lock
        with _dataset_lock:
            if incremental_merge and out_file.exists():
                # Invalid or incomplete prior data must never be overwritten.
                existing_items, _ = load_benchmark_dataset(out_file)
                benchmark_items = merge_benchmark_samples(existing_items, benchmark_items)
            saved_path = save_benchmark_dataset(
                benchmark_items, output_path=out_file, version=version,
                archive_version_copy=archive_version,
            )

        return {
            "success": True,
            "count": len(benchmark_items),
            "output_path": str(saved_path),
            "version": version,
        }

    def get_benchmark_versions(self) -> list[dict[str, Any]]:
        """列出已归档的基准评测数据集版本历史。"""
        if not VERSIONS_DIR.exists():
            return []
        versions = []
        for meta_p in sorted(VERSIONS_DIR.glob("*.meta.json"), reverse=True):
            try:
                with open(meta_p, "r", encoding="utf-8") as f:
                    m = json.load(f)
                v_data = meta_p.with_name(meta_p.name[:-10] + ".json")
                versions.append({
                    "version": m.get("version", "unknown"),
                    "file": str(v_data),
                    "created_at": m.get("created_at"),
                    "total_samples": m.get("total_samples", 0),
                    "distribution": m.get("distribution", {}),
                })
            except Exception:
                continue
        return versions

    def simulate_tuning(self, candidate_params: dict[str, Any], expected_sha256: str | None = None) -> dict[str, Any]:
        """运行 What-If 离线参数灵敏度仿真。"""
        from src.core import config as core
        with core._config_lock:
            self.config = load_config(reload=True)
            preview_hash = generate_config_diff(candidate_params)["config_sha256"]
            if expected_sha256 is not None and expected_sha256 != preview_hash:
                from src.core.tuner import ConfigurationConflict
                raise ConfigurationConflict("Configuration changed since context view; reopen the parameter panel")
            from src.core.replay_evidence import ProductionReplayStore
            store = ProductionReplayStore()
            references = store.references()
            simulator = TuningSimulator(benchmark_samples=references, config=self.config, replay=store)
            result = simulator.simulate(candidate_params)
            result["config_sha256"] = preview_hash
            return result

    def get_tuning_diff(self, proposed_params: dict[str, Any]) -> dict[str, Any]:
        """获取候选参数与当前 settings.yaml 的 Diff 比较。"""
        return generate_config_diff(proposed_params)

    def apply_tuning(self, proposed_params: dict[str, Any], expected_sha256: str | None = None) -> dict[str, Any]:
        """安全应用调优参数并刷新当前服务的配置缓存。"""
        res = apply_config_safe(proposed_params, create_backup=True, expected_sha256=expected_sha256)
        self.config = load_config(reload=True)
        return res

    def rollback_tuning(self, backup_path: str) -> dict[str, Any]:
        """回滚指定备份的配置并刷新缓存。"""
        res = rollback_config(backup_path)
        self.config = load_config(reload=True)
        return res

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

    def _target_frame(self, segment_id):
        """Resolve only the registered archive with matching v2 captured provenance."""
        import math
        import cv2
        import numpy as np
        from src.core.replay_journal import digest
        segment = self.db.get_segment_by_id(segment_id)
        if not segment or not segment.get("archived_frame_path"):
            raise ValueError("当前片段没有归档帧；请先复核事件并生成原片帧")
        image = Path(segment["archived_frame_path"])
        image = image if image.is_absolute() else PROJECT_ROOT / image
        image = image.resolve()
        if not image.is_relative_to((ARCHIVE_DIR / "images").resolve()):
            raise ValueError("归档帧不属于当前反馈库")
        canonical = lambda value: os.path.normcase(str(Path(value).resolve()))
        provenance = None
        manifest = ARCHIVE_DIR / "manifest.jsonl"
        if manifest.exists():
            for line in manifest.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        continue
                    candidate = Path(record.get("archived_image_path", ""))
                    candidate = candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
                    if record.get("segment_id") == segment_id and candidate.resolve() == image:
                        provenance = record
                except (ValueError, TypeError):
                    continue
        payload = image.read_bytes()
        identity = provenance.get("source_identity") if provenance else None
        stamp = float(provenance.get("frame_timestamp", float("nan"))) if provenance else float("nan")
        if (not provenance or provenance.get("provenance_version") != 2
                or provenance.get("frame_origin") != "ffmpeg_seek"
                or provenance.get("image_sha256") != hashlib.sha256(payload).hexdigest()
                or canonical(provenance.get("filepath", "")) != canonical(segment["filepath"])
                or not isinstance(identity, list) or len(identity) != 3
                or canonical(identity[0]) != canonical(segment["filepath"])
                or type(identity[1]) is not int or identity[1] <= 0
                or type(identity[2]) is not int or identity[2] <= 0
                or not math.isfinite(stamp) or not segment["start_time"] <= stamp < segment["end_time"]):
            raise ValueError("归档来源或时刻不能核验；请重新生成原片帧，不沿用旧图目标标签")
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("归档帧无法解码")
        snapshot = digest({"segment_id": segment_id, "source": segment["filepath"],
                           "start": segment["start_time"], "end": segment["end_time"],
                           "provenance": provenance})
        return image, payload, provenance, snapshot, decoded.shape

    def get_target_review(self, segment_id):
        from scripts.export_dataset import DatasetItem, verified_target_annotation, validate_target_boxes
        image, payload, provenance, snapshot, shape = self._target_frame(segment_id)
        path = image.with_suffix(".targets.json")
        previous = path.read_bytes() if path.exists() else b""
        review = json.loads(previous) if previous else None
        if review is not None and not isinstance(review, dict):
            raise ValueError("目标记录格式错误，不能当作已复核目标")
        if review:
            if review.get("image_sha256") != provenance["image_sha256"]:
                review = None  # A new archived image cannot inherit old target labels.
            elif review.get("all_targets_reviewed") is True:
                item = DatasetItem(image, "UNKNOWN", segment_id, provenance["filepath"], provenance["frame_timestamp"])
                verified_target_annotation(item)
            else:
                validate_target_boxes(review.get("boxes", []), shape[1], shape[0])
        return {"status": "AVAILABLE", "segment_id": segment_id, "snapshot": snapshot,
                "review_sha256": hashlib.sha256(previous).hexdigest() if previous else None,
                "image_sha256": hashlib.sha256(payload).hexdigest(), "width": shape[1], "height": shape[0],
                "frame_timestamp": provenance["frame_timestamp"], "source_video": provenance["filepath"],
                "timestamp_kind": provenance.get("timestamp_kind"), "review": review,
                "image_url": f"/api/target_image?segment_id={segment_id}&snapshot={snapshot}"}

    def save_target_review(self, segment_id, *, snapshot, review_sha256, boxes, decision, all_targets_reviewed):
        from scripts.export_dataset import validate_target_boxes
        from src.core.replay_journal import run_lease, atomic_json
        if decision not in {"objects", "no_targets", "uncertain", "draft"}:
            raise ValueError("请选择目标、无目标、不能确定或草稿")
        lock_dir = ARCHIVE_DIR / ".target_locks" / str(segment_id)
        lock_dir.mkdir(parents=True, exist_ok=True)
        with run_lease(lock_dir):
            context = self.get_target_review(segment_id)
            if context["snapshot"] != snapshot or context["review_sha256"] != review_sha256:
                raise ValueError("图片、来源或目标记录已变化；重新打开后再核对")
            validate_target_boxes(boxes, context["width"], context["height"])
            complete = decision in {"objects", "no_targets"}
            if (complete and all_targets_reviewed is not True
                    or decision == "objects" and not boxes or decision == "no_targets" and boxes):
                raise ValueError("完整复核须确认全部可见目标；无目标不能同时保留框")
            image, _, provenance, current_snapshot, _ = self._target_frame(segment_id)
            if current_snapshot != snapshot:
                raise ValueError("归档帧已变化")
            path = image.with_suffix(".targets.json")
            review = {"schema": 1, "label_source": "human_target_review", "decision": decision,
                      "all_targets_reviewed": complete, "no_targets": not boxes,
                      "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"), "boxes": boxes,
                      "source_video": provenance["filepath"], "source_identity": provenance["source_identity"],
                      "frame_timestamp": provenance["frame_timestamp"], "timestamp_kind": provenance.get("timestamp_kind"),
                      "image_sha256": context["image_sha256"], "previous_review_sha256": review_sha256}
            if path.exists():
                history = ARCHIVE_DIR / "target_history" / hashlib.sha256(str(image).encode()).hexdigest()
                history.mkdir(parents=True, exist_ok=True)
                history_path = history / f"{review_sha256}.json"
                if not history_path.exists():
                    atomic_json(history_path, json.loads(path.read_text(encoding="utf-8")))
            atomic_json(path, review)
        return {"success": True, "training_eligible": complete, "context": self.get_target_review(segment_id)}

    def _validate_agent_proposal(self, segment_id, review, payload):
        """Validate a suggestion against registered segment and captured image, offline."""
        import cv2
        import numpy as np
        from scripts.export_dataset import validate_target_boxes
        segment = self.db.get_segment_by_id(segment_id)
        if not segment or not isinstance(review, dict):
            raise ValueError("智能体建议没有对应的登记片段")
        provenance = review.get("source_provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("智能体建议来源格式非法")
        canonical = lambda value: os.path.normcase(os.path.abspath(os.path.normpath(str(value))))
        identity = provenance.get("source_identity")
        decision = review.get("decision")
        if (review.get("schema") != 1 or review.get("label_source") != "agent_visual_review"
                or review.get("segment_id") != segment_id or provenance.get("segment_id") != segment_id
                or provenance.get("provenance_version") != 2 or provenance.get("frame_origin") != "ffmpeg_seek"
                or canonical(provenance.get("filepath", "")) != canonical(segment["filepath"])
                or not isinstance(identity, list) or len(identity) != 3
                or canonical(identity[0]) != canonical(segment["filepath"])
                or type(identity[1]) is not int or identity[1] <= 0
                or type(identity[2]) is not int or identity[2] <= 0
                or provenance.get("start_time") != segment["start_time"]
                or provenance.get("end_time") != segment["end_time"]
                or not segment["start_time"] <= float(provenance.get("frame_timestamp", -1)) < segment["end_time"]
                or decision not in {"objects", "no_targets", "uncertain"}
                or type(review.get("all_visible_targets_reviewed")) is not bool
                or review["all_visible_targets_reviewed"] != (decision != "uncertain")
                or review.get("human_truth_certified") is not False
                or review.get("independent_test_eligible") is not False
                or review.get("image_sha256") != hashlib.sha256(payload).hexdigest()
                or provenance.get("image_sha256") != review["image_sha256"]):
            raise ValueError("智能体建议的来源、图片或复核身份不匹配")
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("智能体建议图片无法解码")
        boxes = review.get("boxes")
        validate_target_boxes(boxes, decoded.shape[1], decoded.shape[0])
        if decision == "objects" and not boxes or decision == "no_targets" and boxes:
            raise ValueError("智能体建议目标状态与框不一致")
        return decoded.shape

    def import_agent_target_proposal(self, segment_id, review, image_payload):
        """Publish an immutable suggestion; never write human targets or business rows."""
        from src.core.replay_journal import atomic_json, digest, run_lease
        self._validate_agent_proposal(segment_id, review, image_payload)
        root = ARCHIVE_DIR / "agent_proposals" / str(int(segment_id))
        with run_lease(ARCHIVE_DIR / ".target_locks" / str(int(segment_id))):
            self._validate_agent_proposal(segment_id, review, image_payload)
            revision = uuid.uuid4().hex
            folder = root / revision
            folder.mkdir(parents=True)
            with (folder / "frame.jpg").open("xb") as stream:
                stream.write(image_payload)
                stream.flush()
                os.fsync(stream.fileno())
            atomic_json(folder / "review.json", review)
            atomic_json(root / "current.json", {"revision": revision, "review_digest": digest(review)})
        return self.get_agent_target_proposal(segment_id)

    def _agent_proposal_frame(self, segment_id):
        from src.core.replay_journal import digest
        root = ARCHIVE_DIR / "agent_proposals" / str(int(segment_id))
        pointer_path = root / "current.json"
        if not pointer_path.is_file():
            raise ValueError("该片段尚无智能体复核建议")
        pointer_bytes = pointer_path.read_bytes()
        pointer = json.loads(pointer_bytes)
        revision = pointer.get("revision", "")
        if not isinstance(revision, str) or len(revision) != 32 or any(c not in "0123456789abcdef" for c in revision):
            raise ValueError("智能体建议版本非法")
        folder = root / revision
        review = json.loads((folder / "review.json").read_text(encoding="utf-8"))
        payload = (folder / "frame.jpg").read_bytes()
        shape = self._validate_agent_proposal(segment_id, review, payload)
        if digest(review) != pointer.get("review_digest") or pointer_path.read_bytes() != pointer_bytes:
            raise ValueError("智能体建议已变化，请重新打开")
        return review, payload, digest(pointer), shape

    def get_agent_target_proposal(self, segment_id):
        review, _, snapshot, shape = self._agent_proposal_frame(segment_id)
        try:
            human = self.get_target_review(segment_id)
            human_priority = bool(human.get("review", {}).get("all_targets_reviewed")) if human.get("review") else False
            human_status = "AVAILABLE"
        except (ValueError, OSError, TypeError, KeyError) as error:
            human_priority, human_status = False, str(error)
        return {"status": "AVAILABLE", "segment_id": segment_id, "snapshot": snapshot,
                "review": review, "width": shape[1], "height": shape[0],
                "human_review_has_priority": human_priority, "human_review_status": human_status,
                "training_eligible": False, "independent_test_eligible": False,
                "image_url": f"/api/agent_target_image?segment_id={segment_id}&snapshot={snapshot}"}

    def prepare_target_frame(self, segment_id):
        from src.archiver import extract_and_archive_frame
        if not extract_and_archive_frame(self.db, segment_id, archive_dir=ARCHIVE_DIR):
            raise ValueError("无法生成原片帧；先确认事件已复核且原片可读取")
        return self.get_target_review(segment_id)

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

        out_path = Path(output_dir) if output_dir else (PROJECT_ROOT / "data" / "yolo_dataset" / "exports" / uuid.uuid4().hex)
        archive_dir = ARCHIVE_DIR

        items = load_archive_items(archive_dir)
        if not items:
            items = load_db_fallback_items(self.db.db_path)

        if not items:
            return {
                "success": False,
                "message": "未找到任何已复核的样本数据。请先在工作台审核打标切片。",
                "stats": {},
            }

        stats = build_yolo_dataset(
            items=items,
            output_dir=out_path,
            val_ratio=val_ratio,
            yolo_model_path=None,
        )

        usable = bool(stats.get("usable"))
        return {
            "success": usable,
            "message": "仅导出已绑定图像且完整人工复核的目标标注。" if usable else
                       "尚无可用人工目标训练/调优/留出集，需要至少三个有效来源组。事件标签不证明无人，旧txt和自动预测均未作为真值导出。",
            "output_dir": str(out_path),
            "yaml_path": str(out_path / "data.yaml"),
            "train_cmd": f'uv run python scripts/train_yolo.py --data "{out_path / "data.yaml"}"' if usable else None,
            "stats": stats,
        }

    def get_tuning_insights(self) -> dict[str, Any]:
        """Describe recorded reviews; candidates require measured replay."""
        from collections import Counter
        from src.core.profiles import resolve_camera_config
        from src.scanner import parse_camera_dir
        with self.db._lock:
            rows = [dict(row) for row in self.db.conn.execute(
                "SELECT state,manual_label,max_energy,duration,review_notes,filepath,"
                "cam_index,start_time,id FROM segments WHERE manual_label IS NOT NULL")]
        labels, scenarios, cross, groups = Counter(), Counter(), Counter(), {}
        energies = {"fp": [], "fn": []}
        categories = {"CONFIRMED_MOTION": "tp", "FALSE_ALARM": "fp",
                      "MISSED_MOTION": "fn", "CONFIRMED_STATIC": "tn"}
        disagreements = []
        for row in rows:
            label = normalize_label(row["manual_label"])
            category = categories.get(label, "unknown")
            labels[category] += 1
            scenario, _ = unpack_notes(row.get("review_notes"))
            scenarios[scenario or "UNKNOWN"] += 1
            cross[(label, row["state"])] += 1
            if category in energies:
                energies[category].append(float(row.get("max_energy") or 0))
            cfg = resolve_camera_config(self.config, filepath=row["filepath"], cam_index=row["cam_index"])
            identity = parse_camera_dir(str(Path(row["filepath"]).parent)).get("mac")
            identity = identity or str(Path(row["filepath"]).parent) or f"cam{row['cam_index']}"
            hour = (float(row["start_time"]) / 3600) % 24
            lower, upper = cfg.get("presence", {}).get("night_hours", [23, 7])
            night = (hour >= lower or hour < upper) if lower > upper else lower <= hour < upper
            key = (identity, "night" if night else "day")
            group = groups.setdefault(key, Counter())
            group[category] += 1
            if label == "MISSED_MOTION" and row["state"] not in ("STATIC",):
                disagreements.append({"segment_id": row["id"], "state": row["state"],
                                      "reason": "Check event boundary, speed or detection semantics"})
        candidates = []
        if labels["fp"]:
            candidates.append({"category": "误报候选调查", "param": "detection.min_motion_threshold",
                               "status": "REQUIRES_REPLAY", "suggested_value": None,
                               "rationale": "先分机位确认视觉、声音或目标误识别原因；能量分布不能证明门限收益"})
        if labels["fn"]:
            candidates.append({"category": "漏检候选调查", "param": "yolo.confidence",
                               "status": "REQUIRES_REPLAY", "suggested_value": None,
                               "rationale": "先区分未检出、事件边界和播放速度；不直接降低目标门限"})
        return {"total_reviewed": len(rows),
                "distribution": {key: labels[key] for key in ("tp", "fp", "fn", "tn")},
                "distribution_basis": "explicit_review_labels_not_confusion_matrix",
                "unknown_labels": labels["unknown"], "scenario_counts": dict(scenarios),
                "prediction_cross_tab": [{"label": label, "state": state, "count": count}
                                         for (label, state), count in sorted(cross.items())],
                "groups": [{"camera": camera, "period": period, "counts": dict(counts),
                            "total": sum(counts.values())} for (camera, period), counts in sorted(groups.items())],
                "requires_attribution": disagreements,
                "energy_stats": {key: value for category, values in energies.items()
                                 for key, value in ((category + "_count", len(values)),
                                    (category + "_avg_energy", round(sum(values)/len(values), 2) if values else 0))},
                "recommendations": candidates, "evidence_basis": "review_inventory",
                "auto_apply_allowed": False}

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
