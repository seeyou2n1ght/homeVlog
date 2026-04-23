import sqlite3
import os
from pathlib import Path
from datetime import datetime

class DatabaseManager:
    """
    SQLite 数据库管理器
    职责：状态幂等、业务事件统计、系统性能监控。
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        # 强制开启 WAL 模式以提高并发性能
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        """初始化表结构"""
        # 确保目录存在
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        with self._get_connection() as conn:
            # 1. 业务流水表 (断点恢复/幂等)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_path TEXT UNIQUE,
                    file_hash TEXT,
                    status TEXT, -- 'pending', 'processing', 'completed', 'failed'
                    last_processed_at DATETIME,
                    error_msg TEXT
                )
            """)
            
            # 2. 用户事件分析表 (萌娃/宠物行为统计)
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
            
            # 3. 系统性能监控表
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
            print(f"[DB] Database initialized at {self.db_path} (WAL mode enabled)")

    def mark_file_started(self, source_path: str):
        """标记文件开始处理"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO processed_files (source_path, status, last_processed_at)
                VALUES (?, 'processing', ?)
                ON CONFLICT(source_path) DO UPDATE SET status='processing', last_processed_at=?
            """, (source_path, datetime.now(), datetime.now()))
            conn.commit()

    def mark_file_completed(self, source_path: str):
        """标记文件处理成功"""
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE processed_files SET status='completed', last_processed_at=? WHERE source_path=?",
                (datetime.now(), source_path)
            )
            conn.commit()

    def is_file_processed(self, source_path: str) -> bool:
        """检查文件是否已处理完成"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT status FROM processed_files WHERE source_path=? AND status='completed'",
                (source_path,)
            )
            return cursor.fetchone() is not None

    def log_performance(self, file_name: str, decode_fps: float, infer_fps: float, total_time_ms: int):
        """记录性能指标"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO performance_metrics (file_name, decode_fps, infer_fps, total_time_ms, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (file_name, decode_fps, infer_fps, total_time_ms, datetime.now()))
            conn.commit()
