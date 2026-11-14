import math
import json
import sqlite3
import threading
import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path

import time

from src.core.config import DB_PATH
from src.core.identity import camera_key, ts_to_unix

logger = logging.getLogger("homevlog")


def is_skipped_source(row: dict) -> bool:
    """Terminal source exclusions are recorded, but never rendered/probed again."""
    return (row.get("prescreen_status") in ("SKIPPED_EMPTY", "SKIPPED_EXCLUDED")
            or row.get("analysis_status") in ("SKIPPED_EXCLUDED", "SKIPPED_CORRUPT"))

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS file_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL UNIQUE,
    cam_index INTEGER NOT NULL,
    camera_id TEXT,
    date TEXT NOT NULL,
    file_start_time TEXT NOT NULL,
    file_end_time TEXT NOT NULL,
    file_duration REAL,

    prescreen_status TEXT DEFAULT 'PENDING',
    prescreen_result TEXT,

    analysis_status TEXT DEFAULT 'PENDING',

    retry_count INTEGER DEFAULT 0,
    error_msg TEXT,
    updated_at TEXT DEFAULT (datetime('now', 'localtime')),
    has_audio INTEGER DEFAULT 0,
    duration_verified INTEGER DEFAULT 0,
    processing_fingerprint TEXT,
    decode_completion TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id INTEGER NOT NULL REFERENCES file_tasks(id) ON DELETE CASCADE,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    state TEXT NOT NULL,
    max_energy REAL DEFAULT 0.0,
    avg_confidence REAL DEFAULT 0.0,
    file_start_offset REAL DEFAULT 0.0,
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

CREATE TABLE IF NOT EXISTS camera_registry (
    camera_id TEXT PRIMARY KEY, cam_index INTEGER NOT NULL UNIQUE
);

-- Historical facts only. The production pipeline neither reads nor writes this table.
CREATE TABLE IF NOT EXISTS human_reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filepath TEXT NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    manual_label TEXT NOT NULL,
    review_notes TEXT,
    reviewed_at TEXT,
    archived_frame_path TEXT,
    UNIQUE(filepath, start_time, end_time)
);

CREATE INDEX IF NOT EXISTS idx_file_tasks_date ON file_tasks(date, cam_index);
CREATE INDEX IF NOT EXISTS idx_file_tasks_prescreen ON file_tasks(prescreen_status);
CREATE INDEX IF NOT EXISTS idx_file_tasks_analysis ON file_tasks(analysis_status);
CREATE INDEX IF NOT EXISTS idx_segments_file ON segments(file_id);
CREATE INDEX IF NOT EXISTS idx_segments_file_time ON segments(file_id, start_time);
CREATE INDEX IF NOT EXISTS idx_segments_state ON segments(state);
CREATE INDEX IF NOT EXISTS idx_render_tasks_status ON render_tasks(status);
"""


class VlogDatabase:
    def __init__(self, db_path: Path | None = None):
        self.db_path = Path(db_path) if db_path is not None else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.db_path), timeout=15.0, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA busy_timeout = 10000;")
            task_columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(file_tasks)")}
            segment_columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(segments)")}
            if "analysis_segments" in task_columns or "filepath" in segment_columns:
                raise RuntimeError(
                    f"Legacy HomeVlog database at {self.db_path}. "
                    "Back up and move the existing database, then restart to initialize a new database."
                )
            self._conn.execute("PRAGMA foreign_keys = ON;")
            self._conn.executescript(SCHEMA_SQL)
        except Exception:
            self._conn.close()
            raise

    def _camera_index(self, identity, preferred):
        row = self.conn.execute("SELECT cam_index FROM camera_registry WHERE camera_id=?", (identity,)).fetchone()
        if row:
            return row[0]
        used = {r[0] for r in self.conn.execute("SELECT cam_index FROM camera_registry")}
        index = preferred if preferred not in used else max(used, default=-1) + 1
        self.conn.execute("INSERT INTO camera_registry VALUES (?,?)", (identity,index))
        return index

    def set_processing_fingerprint(self, filepath, fingerprint):
        with self._lock:
            self.conn.execute("UPDATE file_tasks SET processing_fingerprint=? WHERE filepath=?", (fingerprint,filepath))
            self.conn.commit()

    def invalidate_stale_results(self, date, cam_index, config):
        from src.core.identity import processing_fingerprint
        rows = self.get_all_file_tasks_for_date(date, cam_index)
        stale = [r["filepath"] for r in rows if r["prescreen_status"] in ("STATIC", "SUSPICIOUS")
                 and r.get("processing_fingerprint") != processing_fingerprint(r["filepath"], config)]
        with self._lock:
            self.conn.executemany(
                """UPDATE file_tasks
                   SET prescreen_status='PENDING', analysis_status='PENDING', retry_count=0
                   WHERE filepath=?""",
                [(p,) for p in stale],
            )
            if stale:
                self.conn.executemany(
                    """DELETE FROM segments
                       WHERE file_id IN (SELECT id FROM file_tasks WHERE filepath=?)""",
                    [(p,) for p in stale],
                )
                self.conn.execute(
                    "UPDATE render_tasks SET status='PENDING' WHERE date=? AND cam_index=?",
                    (date, cam_index),
                )
            self.conn.commit()
        return len(stale)

    @property
    def is_closed(self) -> bool:
        return self._conn is None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise sqlite3.ProgrammingError("Cannot operate on a closed database.")
        return self._conn

    def set_file_metadata(self, filepath: str, has_audio: int, duration: float | None = None,
                          *, decode_completion: dict | None = None):
        with self._lock:
            try:
                new_end_time = None
                if duration is not None and duration > 0:
                    row = self.conn.execute(
                        "SELECT file_start_time FROM file_tasks WHERE filepath=?",
                        (str(filepath),)
                    ).fetchone()
                    if row and row["file_start_time"]:
                        try:
                            start_unix = ts_to_unix(row["file_start_time"])
                            end_unix = start_unix + float(duration)
                            new_end_time = time.strftime("%Y%m%d%H%M%S", time.localtime(end_unix))
                        except Exception as e:
                            logger.debug("Failed to calculate new file_end_time for %s: %s", filepath, e)

                self.conn.execute(
                    """UPDATE file_tasks
                       SET has_audio=?,
                           file_duration=COALESCE(?, file_duration),
                           file_end_time=COALESCE(?, file_end_time),
                           decode_completion=COALESCE(?, decode_completion),
                           duration_verified=CASE WHEN ? IS NULL THEN duration_verified ELSE 1 END
                       WHERE filepath=?""",
                    (has_audio, duration, new_end_time,
                     json.dumps(decode_completion) if decode_completion is not None else None,
                     duration, str(filepath))
                )
                self.conn.commit()
            except Exception as e:
                logger.error("DB error in set_file_metadata: %s", e)
                self.conn.rollback()
                raise

    def exclude_file_task(self, filepath: str, reason: str = "MANUAL_EXCLUDED") -> bool:
        """显式受控排除异常或损坏的原片，更新其为终态 SKIPPED_EXCLUDED，避免阻断全日交付。"""
        from src.core.process_lock import database_day_lease
        with self._lock:
            row = self.conn.execute(
                "SELECT date, cam_index FROM file_tasks WHERE filepath=?", (str(filepath),)
            ).fetchone()
        if row is None:
            return False
        with database_day_lease(self.db_path, row["date"], row["cam_index"]), self._lock:
            try:
                cur = self.conn.execute(
                    """UPDATE file_tasks
                       SET prescreen_status='SKIPPED_EXCLUDED',
                           analysis_status='SKIPPED_EXCLUDED',
                           error_msg=?,
                           updated_at=datetime('now', 'localtime')
                       WHERE filepath=?""",
                    (reason, str(filepath)),
                )
                self.conn.commit()
                return cur.rowcount > 0
            except Exception as e:
                logger.error("DB error in exclude_file_task for %s: %s", filepath, e)
                self.conn.rollback()
                raise

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
                identity = camera_key(filepath, cam_index)
                # Serialize identity allocation across independent SQLite connections.
                if not self.conn.in_transaction:
                    self.conn.execute("BEGIN IMMEDIATE")
                cam_index = self._camera_index(identity, cam_index)
                cursor = self.conn.execute(
                    """INSERT OR IGNORE INTO file_tasks
                       (filepath, cam_index, date, file_start_time, file_end_time, file_duration, camera_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (str(filepath), cam_index, date, file_start_time, file_end_time, file_duration, identity),
                )
                self.conn.commit()
                return cursor.rowcount > 0
            except Exception as e:
                logger.error("DB error in add_file_task for %s: %s", filepath, e)
                self.conn.rollback()
                return False

    def set_prescreen_result(self, filepath: str, status: str, result_json: str = "", has_audio: int | None = None,
                             error_msg: str | None = None):
        with self._lock:
            try:
                if has_audio is not None:
                    self.conn.execute(
                        """UPDATE file_tasks
                           SET prescreen_status=?, prescreen_result=?, has_audio=?, error_msg=?, updated_at=datetime('now', 'localtime')
                           WHERE filepath=? AND prescreen_status!='SKIPPED_EXCLUDED' AND analysis_status!='SKIPPED_EXCLUDED'""",
                        (status, result_json, has_audio, error_msg, str(filepath)),
                    )
                else:
                    self.conn.execute(
                        """UPDATE file_tasks
                           SET prescreen_status=?, prescreen_result=?, error_msg=?, updated_at=datetime('now', 'localtime')
                           WHERE filepath=? AND prescreen_status!='SKIPPED_EXCLUDED' AND analysis_status!='SKIPPED_EXCLUDED'""",
                        (status, result_json, error_msg, str(filepath)),
                    )
                self.conn.commit()
            except Exception as e:
                logger.error("DB error in set_prescreen_result for %s: %s", filepath, e)
                self.conn.rollback()
                raise

    def set_analysis_result(self, filepath: str, status: str | list = "ANALYZED", segments: list | str = "",
                            error_msg: str | None = None):
        with self._lock:
            try:
                import json
                # 兼容 set_analysis_result(filepath, segments) 的双参数调用形式
                if isinstance(status, (list, tuple)) or (isinstance(status, str) and status.strip().startswith("[") and not segments):
                    segments = status
                    status = "ANALYZED"

                segments_list = []
                if isinstance(segments, str):
                    if segments.strip():
                        try:
                            segments_list = json.loads(segments)
                        except json.JSONDecodeError:
                            if status == "ANALYZED":
                                raise ValueError("ANALYZED segments must be valid JSON")
                elif isinstance(segments, list):
                    segments_list = segments

                # 1. 更新 file_tasks 状态 (已剔除 analysis_segments)
                updated = self.conn.execute(
                    """UPDATE file_tasks
                       SET analysis_status=?, error_msg=?, updated_at=datetime('now', 'localtime')
                       WHERE filepath=? AND prescreen_status!='SKIPPED_EXCLUDED' AND analysis_status!='SKIPPED_EXCLUDED'""",
                    (status, error_msg, str(filepath)),
                )

                if updated.rowcount == 0:
                    self.conn.commit()
                    return

                # 2. 查询 file_task ID 以供 segments 外键关联
                row = self.conn.execute(
                    "SELECT id FROM file_tasks WHERE filepath=?",
                    (str(filepath),)
                ).fetchone()

                if row and status == "ANALYZED":
                    file_id = row["id"]

                    # ANALYZED replaces prior algorithm segments, including an explicitly empty result.
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

                        records.append((file_id, st, et, state, energy, conf, offset))

                    if records:
                        self.conn.executemany(
                            """INSERT OR REPLACE INTO segments
                               (file_id, start_time, end_time, state, max_energy, avg_confidence, file_start_offset)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            records,
                        )

                self.conn.commit()
            except Exception as e:
                logger.error("DB error in set_analysis_result for %s: %s", filepath, e)
                self.conn.rollback()
                raise

    def get_segments_for_file(self, filepath: str) -> list[dict]:
        with self._lock:
            try:
                rows = self.conn.execute(
                    """SELECT s.id, s.file_id, s.start_time, s.end_time, s.state,
                              s.max_energy, s.avg_confidence, s.file_start_offset, s.created_at,
                              (s.end_time - s.start_time) AS duration,
                              ft.filepath, ft.cam_index, ft.date
                       FROM segments s
                       JOIN file_tasks ft ON s.file_id = ft.id
                       WHERE ft.filepath = ?
                       ORDER BY s.start_time""",
                    (str(filepath),),
                ).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                logger.error("DB error in get_segments_for_file for %s: %s", filepath, e)
                return []

    def get_all_segments_for_date(self, date: str, cam_index: int) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT s.id, s.file_id, s.start_time, s.end_time, s.state,
                          s.max_energy, s.avg_confidence, s.file_start_offset, s.created_at,
                          (s.end_time - s.start_time) AS duration,
                          ft.filepath, ft.cam_index, ft.date
                   FROM segments s
                   JOIN file_tasks ft ON s.file_id = ft.id
                   WHERE ft.date = ? AND ft.cam_index = ?
                   ORDER BY s.file_id, s.start_time""",
                (date, cam_index)
            ).fetchall()
            return [dict(r) for r in rows]

    def sync_timeline_segments(self, date: str, cam_index: int, timeline: list) -> None:
        """Synchronize the resolved global timeline states back to the segments table.

        Ensures that segments upgraded via causal chains (e.g. STATIC -> PRESENCE)
        are accurately reflected in the database for auditing and reporting.
        """
        with self._lock:
            try:
                file_map_rows = self.conn.execute(
                    "SELECT id, filepath FROM file_tasks WHERE date=? AND cam_index=?",
                    (date, cam_index),
                ).fetchall()
                file_id_by_path = {r["filepath"]: r["id"] for r in file_map_rows}

                updates = []
                for seg in timeline:
                    st = getattr(seg, "start_in_file", 0.0)
                    et = getattr(seg, "end_in_file", 0.0)
                    state = getattr(seg, "state", "")
                    fp = getattr(seg, "filepath", "")
                    file_id = file_id_by_path.get(fp)
                    if file_id is None:
                        continue

                    if state in ("PRESENCE", "NIGHT_STATIONARY", "DYNAMIC", "DYNAMIC_AUDIO", "MICRO_MOTION"):
                        updates.append((state, file_id, st, et))

                if updates:
                    self.conn.executemany(
                        """UPDATE segments
                           SET state = ?
                           WHERE file_id = ?
                             AND (start_time - file_start_offset) >= ? - 0.15
                             AND (end_time - file_start_offset) <= ? + 0.15""",
                        updates,
                    )
                self.conn.commit()
            except Exception as e:
                logger.warning("Failed to sync timeline segments to DB: %s", e)
                self.conn.rollback()

    def has_dynamic_segments_for_file(self, filepath_or_id: str | int) -> bool:
        """快速检查指定文件在 SQLite 中是否存在活跃动态切片 (LIMIT 1 极速短路)."""
        with self._lock:
            if self._conn is None:
                return False
            try:
                if isinstance(filepath_or_id, int):
                    row = self.conn.execute(
                        """SELECT 1 FROM segments
                           WHERE file_id = ?
                             AND state IN ('DYNAMIC', 'DYNAMIC_AUDIO', 'PRESENCE', 'NIGHT_STATIONARY', 'MICRO_MOTION')
                           LIMIT 1""",
                        (filepath_or_id,),
                    ).fetchone()
                else:
                    row = self.conn.execute(
                        """SELECT 1 FROM segments s
                           JOIN file_tasks ft ON s.file_id = ft.id
                           WHERE ft.filepath = ?
                             AND s.state IN ('DYNAMIC', 'DYNAMIC_AUDIO', 'PRESENCE', 'NIGHT_STATIONARY', 'MICRO_MOTION')
                           LIMIT 1""",
                        (str(filepath_or_id),),
                    ).fetchone()
                return row is not None
            except Exception as e:
                logger.error("DB error in has_dynamic_segments_for_file: %s", e)
                return False

    def get_dynamic_duration_for_file(self, filepath_or_id: str | int) -> float:
        """聚合查询指定文件的有效动态时长 (秒)."""
        with self._lock:
            if self._conn is None:
                return 0.0
            try:
                if isinstance(filepath_or_id, int):
                    row = self.conn.execute(
                        """SELECT COALESCE(SUM(end_time - start_time), 0.0) FROM segments
                           WHERE file_id = ?
                             AND state IN ('DYNAMIC', 'DYNAMIC_AUDIO', 'PRESENCE', 'NIGHT_STATIONARY', 'MICRO_MOTION')""",
                        (filepath_or_id,),
                    ).fetchone()
                else:
                    row = self.conn.execute(
                        """SELECT COALESCE(SUM(s.end_time - s.start_time), 0.0) FROM segments s
                           JOIN file_tasks ft ON s.file_id = ft.id
                           WHERE ft.filepath = ?
                             AND s.state IN ('DYNAMIC', 'DYNAMIC_AUDIO', 'PRESENCE', 'NIGHT_STATIONARY', 'MICRO_MOTION')""",
                        (str(filepath_or_id),),
                    ).fetchone()
                return float(row[0]) if row and row[0] is not None else 0.0
            except Exception as e:
                logger.error("DB error in get_dynamic_duration_for_file: %s", e)
                return 0.0

    def reset_failed_tasks(self, date: str, cam_index: int, max_retries: int = 3) -> dict:
        """将 FAILED 预筛/分析任务重置为 PENDING，使重跑时自愈补齐丢失内容。

        retry_count 上限防护：连续损坏的文件最多重试 max_retries 次后保持 FAILED，
        避免坏文件造成无限重跑。返回各类重置数量。
        """
        with self._lock:
            if self._conn is None:
                return {"prescreen": 0, "analysis": 0}
            try:
                cur = self.conn.execute(
                    """UPDATE file_tasks
                       SET prescreen_status='PENDING', retry_count=retry_count+1, error_msg=NULL,
                           updated_at=datetime('now', 'localtime')
                       WHERE date=? AND cam_index=? AND prescreen_status='FAILED'
                         AND retry_count < ?""",
                    (date, cam_index, max_retries),
                )
                n_pre = cur.rowcount
                cur = self.conn.execute(
                    """UPDATE file_tasks
                       SET analysis_status='PENDING', retry_count=retry_count+1, error_msg=NULL,
                           updated_at=datetime('now', 'localtime')
                       WHERE date=? AND cam_index=? AND analysis_status='FAILED'
                         AND retry_count < ?""",
                    (date, cam_index, max_retries),
                )
                n_ana = cur.rowcount
                self.conn.commit()
            except Exception as e:
                if self._conn is None:
                    return {"prescreen": 0, "analysis": 0}
                logger.error("DB error in reset_failed_tasks: %s", e)
                try:
                    self.conn.rollback()
                except Exception:
                    pass
                return {"prescreen": 0, "analysis": 0}
        if n_pre or n_ana:
            logger.info(
                "reset failed tasks for %s cam%d: prescreen=%d analysis=%d",
                date, cam_index, n_pre, n_ana,
            )
        return {"prescreen": n_pre, "analysis": n_ana}

    def get_file_task_summary(self, filepath: str) -> dict | None:
        with self._lock:
            if self._conn is None:
                return None
            try:
                row = self.conn.execute(
                    "SELECT id, filepath, prescreen_status, analysis_status FROM file_tasks WHERE filepath=?",
                    (str(filepath),)
                ).fetchone()
                return dict(row) if row else None
            except Exception as e:
                logger.error("DB error in get_file_task_summary: %s", e)
                return None

    def get_file_task(self, filepath: str) -> dict | None:
        with self._lock:
            if self._conn is None:
                return None
            try:
                row = self.conn.execute(
                    "SELECT * FROM file_tasks WHERE filepath=?",
                    (str(filepath),)
                ).fetchone()
                return dict(row) if row else None
            except Exception as e:
                logger.error("DB error in get_file_task for %s: %s", filepath, e)
                return None

    def get_all_file_tasks_for_date(
        self, date: str, cam_index: int, filepaths: list[str] | None = None,
    ) -> list[dict]:
        with self._lock:
            if self._conn is None:
                return []
            try:
                params: list = [date, cam_index]
                path_filter = ""
                if filepaths is not None:
                    if not filepaths:
                        return []
                    path_filter = f" AND filepath IN ({','.join('?' * len(filepaths))})"
                    params.extend(str(path) for path in filepaths)
                rows = self.conn.execute(
                    f"""SELECT * FROM file_tasks
                        WHERE date=? AND cam_index=?{path_filter}
                        ORDER BY file_start_time""",
                    params,
                ).fetchall()
                result = [dict(r) for r in rows]
                if not result:
                    return []

                # 一次性批量预加载全部 segments，避免 140 次循环串行查询引发 N+1 性能雪崩
                from collections import defaultdict
                segs_by_file = defaultdict(list)
                file_ids = [r["id"] for r in result]
                placeholders = ",".join("?" * len(file_ids))
                seg_rows = self.conn.execute(
                    f"""SELECT s.id, s.file_id, s.start_time, s.end_time, s.state,
                               s.max_energy, s.avg_confidence, s.file_start_offset, s.created_at,
                               (s.end_time - s.start_time) AS duration
                        FROM segments s
                        WHERE s.file_id IN ({placeholders})
                        ORDER BY s.file_id, s.start_time""",
                    file_ids,
                ).fetchall()
                for sr in seg_rows:
                    segs_by_file[sr["file_id"]].append(dict(sr))

                for row in result:
                    row["segments"] = segs_by_file.get(row["id"], [])
                return result
            except Exception as e:
                if self._conn is None:
                    return []
                logger.error("DB error in get_all_file_tasks_for_date: %s", e)
                return []

    def upsert_render_task(self, date: str, cam_index: int, status: str = "PENDING"):
        with self._lock:
            if self._conn is None:
                return
            try:
                self.conn.execute(
                    """INSERT INTO render_tasks (date, cam_index, status, updated_at)
                       VALUES (?, ?, ?, datetime('now', 'localtime'))
                       ON CONFLICT(date, cam_index) DO UPDATE SET
                         status=excluded.status, error_msg=NULL, updated_at=excluded.updated_at""",
                    (date, cam_index, status),
                )
                self.conn.commit()
            except Exception as e:
                if self._conn is None:
                    return
                logger.error("DB error in upsert_render_task: %s", e)
                try:
                    self.conn.rollback()
                except Exception:
                    pass

    def set_render_status(self, date: str, cam_index: int, status: str, output_file: str = "",
                          error_msg: str | None = None):
        with self._lock:
            if self._conn is None:
                return
            try:
                self.conn.execute(
                    """INSERT INTO render_tasks
                       (date, cam_index, status, output_file, error_msg, updated_at)
                       VALUES (?, ?, ?, ?, ?, datetime('now', 'localtime'))
                       ON CONFLICT(date, cam_index) DO UPDATE SET
                         status=excluded.status,
                         output_file=excluded.output_file,
                         error_msg=excluded.error_msg,
                         updated_at=excluded.updated_at""",
                    (date, cam_index, status, output_file, error_msg),
                )
                self.conn.commit()
            except Exception as e:
                if self._conn is None:
                    return
                logger.error("DB error in set_render_status: %s", e)
                try:
                    self.conn.rollback()
                except Exception:
                    pass

                raise

    def is_render_completed(self, date: str, cam_index: int) -> bool:
        with self._lock:
            if self._conn is None:
                return False
            try:
                row = self.conn.execute(
                    "SELECT status FROM render_tasks WHERE date=? AND cam_index=?",
                    (date, cam_index),
                ).fetchone()
                return row is not None and row["status"] == "COMPLETED"
            except Exception as e:
                if self._conn is None:
                    return False
                logger.error("DB error in is_render_completed: %s", e)
                return False

    def get_pending_file_count_for_date(self, date: str, cam_index: int) -> int:
        with self._lock:
            if self._conn is None:
                return 0
            try:
                row = self.conn.execute(
                    """SELECT COUNT(*) as cnt FROM file_tasks
                       WHERE date=? AND cam_index=? AND
                       (prescreen_status='PENDING' OR (prescreen_status='SUSPICIOUS' AND analysis_status='PENDING'))""",
                    (date, cam_index)
                ).fetchone()
                return row["cnt"] if row else 0
            except Exception as e:
                if self._conn is None:
                    return 0
                logger.error("DB error in get_pending_file_count_for_date: %s", e)
                return 0



    def close(self):
        with self._lock:
            if self._conn:
                self._conn.close()
                self._conn = None
