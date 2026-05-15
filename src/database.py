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
    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
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

    def set_prescreen_result(self, filepath: str, status: str, result_json: str = ""):
        with self._lock:
            try:
                self.conn.execute(
                    """UPDATE file_tasks
                       SET prescreen_status=?, prescreen_result=?, updated_at=datetime('now')
                       WHERE filepath=?""",
                    (status, result_json, str(filepath)),
                )
                self.conn.commit()
            except Exception as e:
                logger.error("DB error in set_prescreen_result for %s: %s", filepath, e)
                self.conn.rollback()

    def get_prescreen_pending(self, date: str, cam_index: int) -> list[dict]:
        with self._lock:
            try:
                rows = self.conn.execute(
                    """SELECT * FROM file_tasks
                       WHERE date=? AND cam_index=? AND prescreen_status='PENDING'
                       ORDER BY file_start_time""",
                    (date, cam_index),
                ).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                logger.error("DB error in get_prescreen_pending: %s", e)
                return []

    def set_analysis_result(self, filepath: str, status: str, segments_json: str = ""):
        with self._lock:
            try:
                self.conn.execute(
                    """UPDATE file_tasks
                       SET analysis_status=?, analysis_segments=?, updated_at=datetime('now')
                       WHERE filepath=?""",
                    (status, segments_json, str(filepath)),
                )
                self.conn.commit()
            except Exception as e:
                logger.error("DB error in set_analysis_result for %s: %s", filepath, e)
                self.conn.rollback()

    def get_suspicious_files(self, date: str, cam_index: int) -> list[dict]:
        with self._lock:
            try:
                rows = self.conn.execute(
                    """SELECT * FROM file_tasks
                       WHERE date=? AND cam_index=?
                         AND prescreen_status='SUSPICIOUS'
                         AND analysis_status='PENDING'
                       ORDER BY file_start_time""",
                    (date, cam_index),
                ).fetchall()
                return [dict(r) for r in rows]
            except Exception as e:
                logger.error("DB error in get_suspicious_files: %s", e)
                return []

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

    def is_prescreen_complete(self, date: str, cam_index: int) -> bool:
        with self._lock:
            try:
                row = self.conn.execute(
                    """SELECT COUNT(*) as cnt FROM file_tasks
                       WHERE date=? AND cam_index=? AND prescreen_status='PENDING'""",
                    (date, cam_index),
                ).fetchone()
                return row is not None and row["cnt"] == 0
            except Exception as e:
                logger.error("DB error in is_prescreen_complete: %s", e)
                return False

    def is_analysis_complete(self, date: str, cam_index: int) -> bool:
        with self._lock:
            try:
                pending = self.conn.execute(
                    """SELECT COUNT(*) as cnt FROM file_tasks
                       WHERE date=? AND cam_index=?
                         AND prescreen_status IN ('SUSPICIOUS')
                         AND analysis_status='PENDING'""",
                    (date, cam_index),
                ).fetchone()
                return pending is not None and pending["cnt"] == 0
            except Exception as e:
                logger.error("DB error in is_analysis_complete: %s", e)
                return False

    def upsert_render_task(self, date: str, cam_index: int, status: str = "PENDING"):
        with self._lock:
            try:
                self.conn.execute(
                    """INSERT INTO render_tasks (date, cam_index, status, updated_at)
                       VALUES (?, ?, ?, datetime('now'))
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
                       VALUES (?, ?, ?, ?, datetime('now'))
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
