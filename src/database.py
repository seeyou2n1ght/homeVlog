import sqlite3
import threading
import logging
from pathlib import Path

from src.utils import DB_PATH

logger = logging.getLogger("homevlog")

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS file_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL UNIQUE,
    cam_index INTEGER NOT NULL,
    date TEXT NOT NULL,
    file_start_time TEXT NOT NULL,
    file_end_time TEXT NOT NULL,
    file_duration REAL,

    prescreen_status TEXT DEFAULT 'PENDING',
    prescreen_result TEXT,

    analysis_status TEXT DEFAULT 'PENDING',
    analysis_segments TEXT,

    retry_count INTEGER DEFAULT 0,
    error_msg TEXT,
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
    has_audio INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES file_tasks(id) ON DELETE CASCADE,
    filepath TEXT NOT NULL,
    cam_index INTEGER NOT NULL,
    date TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    duration REAL NOT NULL,
    state TEXT NOT NULL,
    max_energy REAL DEFAULT 0.0,
    avg_confidence REAL DEFAULT 0.0,
    file_start_offset REAL DEFAULT 0.0,

    -- 人工审核打标与反向纠偏
    manual_label TEXT,
    review_notes TEXT,
    reviewed_at TEXT,
    archived_frame_path TEXT,
    created_at TEXT DEFAULT (datetime('now', 'localtime')),
    UNIQUE(file_id, start_time, end_time)
);

CREATE TABLE IF NOT EXISTS render_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    cam_index INTEGER NOT NULL,
    status TEXT DEFAULT 'PENDING',
    output_file TEXT,
    retry_count INTEGER DEFAULT 0,
    error_msg TEXT,
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
    UNIQUE(date, cam_index)
);

CREATE INDEX IF NOT EXISTS idx_file_tasks_date ON file_tasks(date, cam_index);
CREATE INDEX IF NOT EXISTS idx_file_tasks_prescreen ON file_tasks(prescreen_status);
CREATE INDEX IF NOT EXISTS idx_file_tasks_analysis ON file_tasks(analysis_status);
CREATE INDEX IF NOT EXISTS idx_segments_file ON segments(file_id);
CREATE INDEX IF NOT EXISTS idx_segments_date ON segments(date, cam_index);
CREATE INDEX IF NOT EXISTS idx_segments_state ON segments(state);
CREATE INDEX IF NOT EXISTS idx_segments_manual ON segments(manual_label);
CREATE INDEX IF NOT EXISTS idx_render_tasks_status ON render_tasks(status);
"""


class VlogDatabase:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA_SQL)
        self._migrate()

    def _migrate(self):
        with self._lock:
            cursor = self.conn.execute("PRAGMA table_info(file_tasks)")
            columns = [row["name"] for row in cursor.fetchall()]
            if "has_audio" not in columns:
                self.conn.execute("ALTER TABLE file_tasks ADD COLUMN has_audio INTEGER DEFAULT 0")

            cursor_seg = self.conn.execute("PRAGMA table_info(segments)")
            seg_columns = [row["name"] for row in cursor_seg.fetchall()]
            if "archived_frame_path" not in seg_columns:
                self.conn.execute("ALTER TABLE segments ADD COLUMN archived_frame_path TEXT")
            self.conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn

    def set_file_metadata(self, filepath: str, has_audio: int):
        with self._lock:
            try:
                self.conn.execute(
                    "UPDATE file_tasks SET has_audio=? WHERE filepath=?",
                    (has_audio, str(filepath))
                )
                self.conn.commit()
            except Exception as e:
                logger.error("DB error in set_file_metadata: %s", e)
                self.conn.rollback()

    def add_file_task(
        self,
        filepath: str,
        cam_index: int,
        date: str,
        file_start_time: str,
        file_end_time: str,
        file_duration: float = 0.0,
    ) -> bool:
        with self._lock:
            try:
                self.conn.execute(
                    """INSERT OR IGNORE INTO file_tasks
                       (filepath, cam_index, date, file_start_time, file_end_time, file_duration)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (str(filepath), cam_index, date, file_start_time, file_end_time, file_duration),
                )
                self.conn.commit()
                return True
            except Exception as e:
                logger.error("DB error in add_file_task for %s: %s", filepath, e)
                self.conn.rollback()
                return False

    def set_prescreen_result(self, filepath: str, status: str, result_json: str = "", has_audio: int | None = None):
        with self._lock:
            try:
                if has_audio is not None:
                    self.conn.execute(
                        """UPDATE file_tasks
                           SET prescreen_status=?, prescreen_result=?, has_audio=?, updated_at=datetime('now', 'localtime')
                           WHERE filepath=?""",
                        (status, result_json, has_audio, str(filepath)),
                    )
                else:
                    self.conn.execute(
                        """UPDATE file_tasks
                           SET prescreen_status=?, prescreen_result=?, updated_at=datetime('now', 'localtime')
                           WHERE filepath=?""",
                        (status, result_json, str(filepath)),
                    )
                self.conn.commit()
            except Exception as e:
                logger.error("DB error in set_prescreen_result for %s: %s", filepath, e)
                self.conn.rollback()

    def set_analysis_result(self, filepath: str, status: str, segments: list | str = ""):
        with self._lock:
            try:
                import json
                segments_list = []
                segments_json = ""
                if isinstance(segments, str):
                    segments_json = segments
                    if segments.strip():
                        try:
                            segments_list = json.loads(segments)
                        except Exception:
                            segments_list = []
                elif isinstance(segments, list):
                    segments_list = segments
                    try:
                        dict_list = [s.to_dict() if hasattr(s, "to_dict") else dict(s) for s in segments]
                        segments_json = json.dumps(dict_list)
                    except Exception:
                        segments_json = ""

                # 1. 更新 file_tasks 状态与兼容用 JSON
                self.conn.execute(
                    """UPDATE file_tasks
                       SET analysis_status=?, analysis_segments=?, updated_at=datetime('now', 'localtime')
                       WHERE filepath=?""",
                    (status, segments_json, str(filepath)),
                )

                # 2. 查询 file_task 元数据以供 segments 外键关联
                row = self.conn.execute(
                    "SELECT id, cam_index, date FROM file_tasks WHERE filepath=?",
                    (str(filepath),)
                ).fetchone()

                if row and segments_list:
                    file_id = row["id"]
                    cam_index = row["cam_index"]
                    date_val = row["date"]

                    # 幂等清理该文件旧分段
                    self.conn.execute("DELETE FROM segments WHERE file_id=?", (file_id,))

                    records = []
                    for s in segments_list:
                        if hasattr(s, "start_time"):
                            st = float(s.start_time)
                            et = float(s.end_time)
                            state = str(s.state)
                            energy = float(getattr(s, "max_energy", 0.0) or 0.0)
                            conf = float(getattr(s, "avg_confidence", 0.0) or 0.0)
                            offset = float(getattr(s, "file_start_offset", 0.0) or 0.0)
                        elif isinstance(s, dict):
                            st = float(s.get("start_time", s.get("start", 0.0)))
                            et = float(s.get("end_time", s.get("end", 0.0)))
                            state = str(s.get("state", s.get("label", "STATIC")))
                            energy = float(s.get("max_energy", 0.0) or 0.0)
                            conf = float(s.get("avg_confidence", 0.0) or 0.0)
                            offset = float(s.get("file_start_offset", 0.0) or 0.0)
                        else:
                            continue

                        dur = max(et - st, 0.0)
                        records.append((
                            file_id, str(filepath), cam_index, date_val,
                            st, et, dur, state, energy, conf, offset
                        ))

                    if records:
                        self.conn.executemany(
                            """INSERT OR REPLACE INTO segments
                               (file_id, filepath, cam_index, date, start_time, end_time, duration,
                                state, max_energy, avg_confidence, file_start_offset)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            records,
                        )

                self.conn.commit()
            except Exception as e:
                logger.error("DB error in set_analysis_result for %s: %s", filepath, e)
                self.conn.rollback()

    def get_segments_for_file(self, filepath: str) -> list[dict]:
        with self._lock:
            try:
                rows = self.conn.execute(
                    """SELECT * FROM segments WHERE filepath=? ORDER BY start_time""",
                    (str(filepath),)
                ).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                logger.error("DB error in get_segments_for_file for %s: %s", filepath, e)
                return []

    def get_all_segments_for_date(self, date: str, cam_index: int) -> list[dict]:
        with self._lock:
            try:
                rows = self.conn.execute(
                    """SELECT * FROM segments WHERE date=? AND cam_index=? ORDER BY file_id, start_time""",
                    (date, cam_index)
                ).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                logger.error("DB error in get_all_segments_for_date: %s", e)
                return []

    def update_segment_review(self, segment_id: int, manual_label: str, notes: str = "", archived_frame_path: str | None = None) -> bool:
        with self._lock:
            try:
                if archived_frame_path:
                    self.conn.execute(
                        """UPDATE segments
                           SET manual_label=?, review_notes=?, reviewed_at=datetime('now', 'localtime'), archived_frame_path=?
                           WHERE id=?""",
                        (manual_label, notes, archived_frame_path, segment_id),
                    )
                else:
                    self.conn.execute(
                        """UPDATE segments
                           SET manual_label=?, review_notes=?, reviewed_at=datetime('now', 'localtime')
                           WHERE id=?""",
                        (manual_label, notes, segment_id),
                    )
                self.conn.commit()
                return True
            except Exception as e:
                logger.error("DB error in update_segment_review: %s", e)
                self.conn.rollback()
                return False

    def set_segment_archived_path(self, segment_id: int, archived_frame_path: str) -> bool:
        with self._lock:
            try:
                self.conn.execute(
                    "UPDATE segments SET archived_frame_path=? WHERE id=?",
                    (archived_frame_path, segment_id),
                )
                self.conn.commit()
                return True
            except Exception as e:
                logger.error("DB error in set_segment_archived_path: %s", e)
                self.conn.rollback()
                return False

    def get_segment_by_id(self, segment_id: int) -> dict | None:
        with self._lock:
            try:
                row = self.conn.execute(
                    "SELECT * FROM segments WHERE id=?",
                    (segment_id,),
                ).fetchone()
                return dict(row) if row else None
            except Exception as e:
                logger.error("DB error in get_segment_by_id: %s", e)
                return None

    def get_anomaly_segments(
        self,
        category: str = "all",
        date: str | None = None,
        cam_index: int | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """主动召回高争议与潜在误判/漏判的切片，支持按错检类型多维过滤 (Active Learning 模式)。

        Categories:
            - 'all': 召回所有争议疑难切片
            - 'fp_suspect': 疑似误报 (算法判定 DYNAMIC，但 YOLO 无目标且能量偏低，疑似光影/微尘)
            - 'fn_suspect': 疑似漏报 (算法判定 STATIC，但能量接近临界阈值，疑似微动漏检)
            - 'jitter': 极短毛刺 (时长 < 3.0s 的动态突变碎片)
            - 'reviewed': 已完成人工复核的切片
        """
        with self._lock:
            try:
                where_clauses = []
                params = []

                if category == "fp_suspect":
                    where_clauses.append("(state = 'DYNAMIC' AND avg_confidence = 0.0 AND max_energy < 8.0)")
                elif category == "fn_suspect":
                    where_clauses.append("(state = 'STATIC' AND max_energy >= 1.5 AND max_energy <= 3.0)")
                elif category == "jitter":
                    where_clauses.append("(duration < 3.0 AND state = 'DYNAMIC')")
                elif category == "reviewed":
                    where_clauses.append("manual_label IS NOT NULL")
                else:  # 'all' 或默认
                    where_clauses.append(
                        "((state = 'DYNAMIC' AND avg_confidence = 0.0 AND max_energy < 8.0) "
                        "OR (state = 'STATIC' AND max_energy >= 1.5 AND max_energy <= 3.0) "
                        "OR (duration < 3.0 AND state = 'DYNAMIC'))"
                    )

                if date:
                    where_clauses.append("date = ?")
                    params.append(date)
                if cam_index is not None:
                    where_clauses.append("cam_index = ?")
                    params.append(cam_index)

                where_str = " AND ".join(where_clauses)
                query = f"""
                    SELECT * FROM segments
                    WHERE {where_str}
                    ORDER BY (manual_label IS NULL) DESC, max_energy DESC
                    LIMIT ?
                """
                params.append(limit)

                rows = self.conn.execute(query, params).fetchall()
                results = []
                for r in rows:
                    item = dict(r)
                    # 语义化标记错检原因
                    if item.get("manual_label"):
                        item["anomaly_type"] = "reviewed"
                        item["reason_desc"] = f"已复核 ({item['manual_label']})"
                    elif item.get("state") == "DYNAMIC" and float(item.get("avg_confidence", 0.0)) == 0.0 and float(item.get("max_energy", 0.0)) < 8.0:
                        item["anomaly_type"] = "fp_suspect"
                        item["reason_desc"] = "疑似光影刚性 (无目标置信度)"
                    elif item.get("state") == "STATIC" and float(item.get("max_energy", 0.0)) >= 1.5 and float(item.get("max_energy", 0.0)) <= 3.0:
                        item["anomaly_type"] = "fn_suspect"
                        item["reason_desc"] = "疑似微动作漏判 (能量临界)"
                    elif float(item.get("duration", 0.0)) < 3.0 and item.get("state") == "DYNAMIC":
                        item["anomaly_type"] = "jitter"
                        item["reason_desc"] = f"极短突发碎片 ({float(item['duration']):.1f}s)"
                    else:
                        item["anomaly_type"] = "other"
                        item["reason_desc"] = "待复核样本"

                    results.append(item)

                return results
            except Exception as e:
                logger.error("DB error in get_anomaly_segments: %s", e)
                return []

    def reset_failed_tasks(self, date: str, cam_index: int, max_retries: int = 3) -> dict:
        """将 FAILED 预筛/分析任务重置为 PENDING，使重跑时自愈补齐丢失内容。

        retry_count 上限防护：连续损坏的文件最多重试 max_retries 次后保持 FAILED，
        避免坏文件造成无限重跑。返回各类重置数量。
        """
        with self._lock:
            try:
                cur = self.conn.execute(
                    """UPDATE file_tasks
                       SET prescreen_status='PENDING', retry_count=retry_count+1,
                           updated_at=datetime('now', 'localtime')
                       WHERE date=? AND cam_index=? AND prescreen_status='FAILED'
                         AND retry_count < ?""",
                    (date, cam_index, max_retries),
                )
                n_pre = cur.rowcount
                cur = self.conn.execute(
                    """UPDATE file_tasks
                       SET analysis_status='PENDING', retry_count=retry_count+1,
                           updated_at=datetime('now', 'localtime')
                       WHERE date=? AND cam_index=? AND analysis_status='FAILED'
                         AND retry_count < ?""",
                    (date, cam_index, max_retries),
                )
                n_ana = cur.rowcount
                self.conn.commit()
            except Exception as e:
                logger.error("DB error in reset_failed_tasks: %s", e)
                self.conn.rollback()
                return {"prescreen": 0, "analysis": 0}
        if n_pre or n_ana:
            logger.info(
                "reset failed tasks for %s cam%d: prescreen=%d analysis=%d",
                date, cam_index, n_pre, n_ana,
            )
        return {"prescreen": n_pre, "analysis": n_ana}

    def get_all_file_tasks_for_date(self, date: str, cam_index: int) -> list[dict]:
        with self._lock:
            try:
                rows = self.conn.execute(
                    """SELECT * FROM file_tasks
                       WHERE date=? AND cam_index=?
                       ORDER BY file_start_time""",
                    (date, cam_index),
                ).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                logger.error("DB error in get_all_file_tasks_for_date: %s", e)
                return []

    def upsert_render_task(self, date: str, cam_index: int, status: str = "PENDING"):
        with self._lock:
            try:
                self.conn.execute(
                    """INSERT INTO render_tasks (date, cam_index, status, updated_at)
                       VALUES (?, ?, ?, datetime('now', 'localtime'))
                       ON CONFLICT(date, cam_index) DO UPDATE SET
                         status=excluded.status, updated_at=excluded.updated_at""",
                    (date, cam_index, status),
                )
                self.conn.commit()
            except Exception as e:
                logger.error("DB error in upsert_render_task: %s", e)
                self.conn.rollback()

    def set_render_status(self, date: str, cam_index: int, status: str, output_file: str = ""):
        with self._lock:
            try:
                self.conn.execute(
                    """INSERT INTO render_tasks
                       (date, cam_index, status, output_file, updated_at)
                       VALUES (?, ?, ?, ?, datetime('now', 'localtime'))
                       ON CONFLICT(date, cam_index) DO UPDATE SET
                         status=excluded.status,
                         output_file=excluded.output_file,
                         updated_at=excluded.updated_at""",
                    (date, cam_index, status, output_file),
                )
                self.conn.commit()
            except Exception as e:
                logger.error("DB error in set_render_status: %s", e)
                self.conn.rollback()

    def is_render_completed(self, date: str, cam_index: int) -> bool:
        with self._lock:
            try:
                row = self.conn.execute(
                    "SELECT status FROM render_tasks WHERE date=? AND cam_index=?",
                    (date, cam_index),
                ).fetchone()
                return row is not None and row["status"] == "COMPLETED"
            except Exception as e:
                logger.error("DB error in is_render_completed: %s", e)
                return False

    def get_pending_file_count_for_date(self, date: str, cam_index: int) -> int:
        with self._lock:
            try:
                row = self.conn.execute(
                    """SELECT COUNT(*) as cnt FROM file_tasks
                       WHERE date=? AND cam_index=? AND
                       (prescreen_status='PENDING' OR (prescreen_status='SUSPICIOUS' AND analysis_status='PENDING'))""",
                    (date, cam_index)
                ).fetchone()
                return row["cnt"] if row else 0
            except Exception as e:
                logger.error("DB error in get_pending_file_count_for_date: %s", e)
                return 0



    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
