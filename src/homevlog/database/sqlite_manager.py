import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple


class DatabaseManager:
    """
    SQLite 数据库管理器 (V2.4 Dashboard Edition)
    职责：状态幂等、高维业务事件统计、精准系统性能监控。
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        # 启用外键约束
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _init_db(self) -> None:
        """初始化所有表结构 (破坏性升级)"""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            # 破坏性升级检查：如果旧表存在旧字段，直接清空旧库
            cur = conn.execute("PRAGMA table_info(processed_files)")
            columns = [row[1] for row in cur.fetchall()]
            if 'source_path' in columns:
                print("[DB] Warning: Detected V2.2 old schema. Dropping all tables for V2.4 destructive update...")
                conn.execute("DROP TABLE IF EXISTS analytics_events")
                conn.execute("DROP TABLE IF EXISTS performance_metrics")
                conn.execute("DROP TABLE IF EXISTS processed_days")
                conn.execute("DROP TABLE IF EXISTS processed_files")
                conn.commit()

            # 1. 业务流水与生命周期表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_name TEXT UNIQUE,
                    source_size_bytes INTEGER,
                    output_size_bytes INTEGER,
                    status TEXT,       -- processing / completed / failed
                    created_at DATETIME
                )
            """)

            # 2. 系统性能探针表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER,
                    total_time_ms INTEGER,
                    phase_wait_ms INTEGER,
                    phase_infer_ms INTEGER,
                    phase_io_cut_ms INTEGER,
                    decode_fps REAL,
                    infer_fps REAL,
                    avg_queue_depth REAL,
                    FOREIGN KEY(file_id) REFERENCES processed_files(id) ON DELETE CASCADE
                )
            """)

            # 3. 业务事件明细表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS analytics_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_id INTEGER,
                    start_pts REAL,
                    end_pts REAL,
                    duration_ms INTEGER,
                    classes_detected TEXT,
                    avg_distance REAL,
                    FOREIGN KEY(file_id) REFERENCES processed_files(id) ON DELETE CASCADE
                )
            """)
            conn.commit()
        print(f"[DB] Initialized at {self.db_path} (WAL mode, V2.4 Schema)")

    # ------------------------------------------------------------------ #
    #  文件维度与状态流转                                                    #
    # ------------------------------------------------------------------ #

    def mark_file_started(self, file_name: str, source_size_bytes: int) -> int:
        """标记文件开始处理，并返回记录 ID"""
        with self._get_connection() as conn:
            cur = conn.execute("""
                INSERT INTO processed_files (file_name, source_size_bytes, status, created_at)
                VALUES (?, ?, 'processing', ?)
                ON CONFLICT(file_name) DO UPDATE
                    SET status='processing', source_size_bytes=?
            """, (file_name, source_size_bytes, datetime.now().isoformat(), source_size_bytes))
            conn.commit()
            
            # 获取刚刚插入或更新的 file_id
            cur = conn.execute("SELECT id FROM processed_files WHERE file_name=?", (file_name,))
            row = cur.fetchone()
            return row[0] if row else -1

    def mark_file_completed(self, file_id: int, output_size_bytes: Optional[int] = None) -> None:
        """标记文件处理完成，可选择记录输出文件大小以计算空间 ROI"""
        with self._get_connection() as conn:
            if output_size_bytes is not None:
                conn.execute(
                    "UPDATE processed_files SET status='completed', output_size_bytes=? WHERE id=?",
                    (output_size_bytes, file_id),
                )
            else:
                conn.execute(
                    "UPDATE processed_files SET status='completed' WHERE id=?",
                    (file_id,),
                )
            conn.commit()

    def mark_file_failed(self, file_name: str) -> None:
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE processed_files SET status='failed' WHERE file_name=?",
                (file_name,),
            )
            conn.commit()

    def is_file_processed(self, file_name: str) -> bool:
        with self._get_connection() as conn:
            cur = conn.execute(
                "SELECT 1 FROM processed_files WHERE file_name=? AND status='completed'",
                (file_name,),
            )
            return cur.fetchone() is not None

    # ------------------------------------------------------------------ #
    #  V2.4 性能与业务指标写入                                               #
    # ------------------------------------------------------------------ #

    def log_performance(
        self, file_id: int, total_time_ms: int, phase_wait_ms: int, phase_infer_ms: int,
        phase_io_cut_ms: int, decode_fps: float, infer_fps: float, avg_queue_depth: float
    ) -> None:
        """记录系统性能探针"""
        with self._get_connection() as conn:
            conn.execute("""
                INSERT INTO performance_metrics
                    (file_id, total_time_ms, phase_wait_ms, phase_infer_ms, phase_io_cut_ms, decode_fps, infer_fps, avg_queue_depth)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (file_id, total_time_ms, phase_wait_ms, phase_infer_ms, phase_io_cut_ms, decode_fps, infer_fps, avg_queue_depth))
            conn.commit()

    def log_analytics_events(self, file_id: int, events: List[Tuple[float, float, str, float]]) -> None:
        """批量记录业务事件明细"""
        if not events:
            return
            
        # events format: [(start_pts, end_pts, classes_detected_str, avg_distance), ...]
        insert_data = []
        for start_pts, end_pts, classes_str, avg_distance in events:
            duration_ms = int((end_pts - start_pts) * 1000)
            insert_data.append((file_id, start_pts, end_pts, duration_ms, classes_str, avg_distance))
            
        with self._get_connection() as conn:
            conn.executemany("""
                INSERT INTO analytics_events
                    (file_id, start_pts, end_pts, duration_ms, classes_detected, avg_distance)
                VALUES (?, ?, ?, ?, ?, ?)
            """, insert_data)
            conn.commit()
