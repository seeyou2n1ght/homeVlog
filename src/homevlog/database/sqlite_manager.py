import sqlite3
from pathlib import Path
from datetime import datetime


class DatabaseManager:
    """
    SQLite 数据库管理器
    职责：状态幂等、业务事件统计、系统性能监控、日期维度聚合。
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        """初始化所有表结构"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            # 1. 业务流水表（断点恢复 / 幂等）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT UNIQUE,
                    file_hash TEXT,
                    status TEXT,       -- pending / processing / completed / failed
                    last_processed_at DATETIME,
                    error_msg TEXT
                )
            """)

            # 2. 日期维度聚合表（每日 Vlog 状态）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_days (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    status TEXT DEFAULT 'pending',
                    file_count INTEGER DEFAULT 0,
                    output_path TEXT,
                    updated_at DATETIME
                )
            """)

            # 3. 用户事件分析表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT,
                    class_name TEXT,
                    motion_duration_ms INTEGER,
                    static_duration_ms INTEGER,
                    file_name TEXT
                )
            """)

            # 4. 系统性能监控表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT,
                    decode_fps REAL,
                    infer_fps REAL,
                    total_time_ms INTEGER,
                    timestamp DATETIME
                )
            """)
            conn.commit()
        print(f"[DB] Initialized at {self.db_path} (WAL mode)")

    # ------------------------------------------------------------------ #
    #  文件维度方法                                                          #
    # ------------------------------------------------------------------ #

    def mark_file_started(self, source_path: str) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO processed_files (source_path, status, last_processed_at)
                VALUES (?, 'processing', ?)
                ON CONFLICT(source_path) DO UPDATE
                    SET status='processing', last_processed_at=?
            """, (source_path, datetime.now(), datetime.now()))
            conn.commit()

    def mark_file_completed(self, source_path: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE processed_files SET status='completed', last_processed_at=? WHERE source_path=?",
                (datetime.now(), source_path),
            )
            conn.commit()

    def mark_file_failed(self, source_path: str, error_msg: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE processed_files SET status='failed', error_msg=?, last_processed_at=? WHERE source_path=?",
                (error_msg, datetime.now(), source_path),
            )
            conn.commit()

    def is_file_processed(self, source_path: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT 1 FROM processed_files WHERE source_path=? AND status='completed'",
                (source_path,),
            )
            return cur.fetchone() is not None

    # ------------------------------------------------------------------ #
    #  日期维度方法                                                          #
    # ------------------------------------------------------------------ #

    def mark_day_processing(self, date_str: str, file_count: int) -> None:
        """记录某天开始处理，登记涉及的文件数"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO processed_days (date, status, file_count, updated_at)
                VALUES (?, 'pending', ?, ?)
                ON CONFLICT(date) DO UPDATE
                    SET file_count=?, updated_at=?
            """, (date_str, file_count, datetime.now(), file_count, datetime.now()))
            conn.commit()

    def mark_day_completed(self, date_str: str, output_path: str) -> None:
        """记录某天 Vlog 合成完成"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO processed_days (date, status, output_path, updated_at)
                VALUES (?, 'completed', ?, ?)
                ON CONFLICT(date) DO UPDATE
                    SET status='completed', output_path=?, updated_at=?
            """, (date_str, output_path, datetime.now(), output_path, datetime.now()))
            conn.commit()

    def is_day_completed(self, date_str: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT 1 FROM processed_days WHERE date=? AND status='completed'",
                (date_str,),
            )
            return cur.fetchone() is not None

    # ------------------------------------------------------------------ #
    #  性能监控方法                                                          #
    # ------------------------------------------------------------------ #

    def log_performance(
        self, file_name: str, decode_fps: float, infer_fps: float, total_time_ms: int
    ) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO performance_metrics
                    (file_name, decode_fps, infer_fps, total_time_ms, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (file_name, decode_fps, infer_fps, total_time_ms, datetime.now()))
            conn.commit()
