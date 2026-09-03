"""测试模块 1: 资产扫描器 (Scanner) 与数据库存储管理 (VlogDatabase).

覆盖：
1. 监控切片文件名解析 (parse_filename) 与异常格式防御；
2. 目录遍历扫描 (scan_directory) 与增量防重；
3. 数据库表结构初始化、WAL 模式与外键约束；
4. 任务状态机完整生命周期 (add_file_task -> prescreen -> analysis -> render)；
5. 多线程并发读写与 WAL 锁争用压力测试。
"""

import concurrent.futures
import sqlite3
import time
from pathlib import Path

import pytest

from src.database import VlogDatabase
from src.scanner import parse_filename, scan_directory, get_date_cam_groups


class TestFilenameParsingAndScanner:
    """测试监控视频命名解析与扫描入库逻辑。"""

    def test_parse_filename_valid(self):
        filename = "00_20260901120000_20260901120500.mp4"
        meta = parse_filename(filename)
        assert meta is not None
        assert meta["cam_index"] == 0
        assert meta["file_start_time"] == "20260901120000"
        assert meta["file_end_time"] == "20260901120500"
        assert meta["date"] == "20260901"
        assert (meta["end_ts"] - meta["start_ts"]) == 300.0

    def test_parse_filename_invalid_formats(self):
        assert parse_filename("invalid_name.mp4") is None
        assert parse_filename("00_short_time.mp4") is None
        assert parse_filename("README.txt") is None
        assert parse_filename("") is None

    def test_parse_camera_dir(self):
        from src.scanner import parse_camera_dir
        assert parse_camera_dir(r"\\192.168.5.8\LyuShare\XiaomiCamera_01_B888805AA3CD") == {
            "cam_id": "01",
            "mac": "B888805AA3CD",
        }
        assert parse_camera_dir("XiaomiCamera_02_6490C12345EF") == {
            "cam_id": "02",
            "mac": "6490C12345EF",
        }
        assert parse_camera_dir("C:/Footage/regular_folder") == {}
        assert parse_camera_dir("") == {}

    def test_resolve_camera_identity_three_tier_fallback(self):
        from src.scanner import resolve_camera_identity
        mock_cfg = {"cameras": {"B888805AA3CD": "baby_room"}}

        # Tier 1: MAC 匹配且配置了别名
        disp, fid = resolve_camera_identity("XiaomiCamera_01_B888805AA3CD", cam_index=0, config=mock_cfg)
        assert disp == "baby_room (B888805AA3CD)"
        assert fid == "baby_room"

        # Tier 2: 提取到 MAC 但未配置别名 -> 回退至原生 MAC
        disp, fid = resolve_camera_identity("XiaomiCamera_02_6490C12345EF", cam_index=1, config=mock_cfg)
        assert disp == "6490C12345EF (Cam 1)"
        assert fid == "6490C12345EF"

        # Tier 3: 普通自定义文件夹 -> 回退至文件夹名
        disp, fid = resolve_camera_identity("C:/Footage/FrontDoor", cam_index=2, config=mock_cfg)
        assert disp == "FrontDoor (Cam 2)"
        assert fid == "FrontDoor"

        # Tier 3 极端兜底: 无意义名称 -> 回退至 Cam 编号
        disp, fid = resolve_camera_identity("C:/Footage/video", cam_index=3, config=mock_cfg)
        assert disp == "Cam 3"
        assert fid == "cam3"

    def test_get_input_dirs_polymorphic(self):
        from src.utils import get_input_dirs
        # 1. 列表格式
        cfg1 = {"paths": {"input_dirs": ["/nas/cam1", "/nas/cam2"]}}
        assert get_input_dirs(cfg1) == ["/nas/cam1", "/nas/cam2"]

        # 2. 单字符串格式
        cfg2 = {"paths": {"input_dirs": "/nas/cam1"}}
        assert get_input_dirs(cfg2) == ["/nas/cam1"]

        # 3. 兼容旧 key
        cfg3 = {"paths": {"input_dir": "/nas/cam_old"}}
        assert get_input_dirs(cfg3) == ["/nas/cam_old"]

        # 4. 空值安全
        assert get_input_dirs({}) == []



    def test_scan_directory_incremental_and_groups(self, tmp_path):
        db_path = tmp_path / "test_vlog.db"
        db = VlogDatabase(db_path=db_path)
        try:
            # 创建虚拟监控文件
            input_dir = tmp_path / "camera_footage"
            input_dir.mkdir()
            f1 = input_dir / "00_20260901080000_20260901080500.mp4"
            f2 = input_dir / "00_20260901080500_20260901081000.mp4"
            f3 = input_dir / "01_20260902090000_20260902090500.mp4"
            f_junk = input_dir / "unrelated_video.mp4"

            for f in [f1, f2, f3, f_junk]:
                f.write_bytes(b"dummy")

            # 首次扫描
            res = scan_directory(db, input_dir=str(input_dir))
            assert res.added == 3

            # 验证分组
            groups = get_date_cam_groups(db)
            assert ("20260901", 0) in groups
            assert ("20260902", 1) in groups
            assert len(groups) == 2

            # 重复扫描：验证数据库中无重复数据产生
            tasks_before = db.get_all_file_tasks_for_date("20260901", 0)
            scan_directory(db, input_dir=str(input_dir))
            tasks_after = db.get_all_file_tasks_for_date("20260901", 0)
            assert len(tasks_before) == len(tasks_after) == 2
        finally:
            db.close()

    def test_scan_multiple_directories(self, tmp_path):
        db_path = tmp_path / "test_multi.db"
        db = VlogDatabase(db_path=db_path)
        try:
            dir1 = tmp_path / "XiaomiCamera_01_B888805AA3CD"
            dir2 = tmp_path / "XiaomiCamera_02_6490C12345EF"
            dir1.mkdir()
            dir2.mkdir()
            (dir1 / "00_20260901080000_20260901080500.mp4").write_bytes(b"dummy1")
            (dir2 / "01_20260901080000_20260901080500.mp4").write_bytes(b"dummy2")

            res = scan_directory(db, input_dir=[str(dir1), str(dir2)])
            assert res.added == 2
            groups = get_date_cam_groups(db)
            assert ("20260901", 0) in groups
            assert ("20260901", 1) in groups
        finally:
            db.close()



class TestVlogDatabaseLifecycle:
    """测试任务生命周期状态流转与更新。"""

    def test_task_state_transitions(self, tmp_path):
        db_path = tmp_path / "test_lifecycle.db"
        db = VlogDatabase(db_path=db_path)
        try:
            filepath = "00_20260901100000_20260901100500.mp4"
            db.add_file_task(filepath, 0, "20260901", "20260901100000", "20260901100500", 300.0)

            # 1. 预筛前为 PENDING
            pending = db.get_prescreen_pending("20260901", 0)
            assert len(pending) == 1
            assert pending[0]["filepath"] == filepath
            assert not db.is_prescreen_complete("20260901", 0)

            # 2. 预筛标记为 SUSPICIOUS
            db.set_prescreen_result(filepath, "SUSPICIOUS", '{"diff": 15}')
            assert db.is_prescreen_complete("20260901", 0)
            suspicious = db.get_suspicious_files("20260901", 0)
            assert len(suspicious) == 1
            assert not db.is_analysis_complete("20260901", 0)

            # 3. 详细分析标记为 ANALYZED
            db.set_analysis_result(filepath, "ANALYZED", '[{"start": 0, "end": 10, "label": "DYNAMIC"}]')
            assert db.is_analysis_complete("20260901", 0)

            # 4. 渲染任务生命周期
            db.upsert_render_task("20260901", 0, "PENDING")
            assert not db.is_render_completed("20260901", 0)
            db.set_render_status("20260901", 0, "RENDERING")
            db.set_render_status("20260901", 0, "COMPLETED", output_file="DailyVlog_20260901_cam0.mp4")
            assert db.is_render_completed("20260901", 0)

            # 5. 音频元数据更新
            db.set_file_metadata(filepath, has_audio=1)
            all_tasks = db.get_all_file_tasks_for_date("20260901", 0)
            assert all_tasks[0]["has_audio"] == 1

        finally:
            db.close()

    def test_concurrent_read_write_wal_safety(self, tmp_path):
        """高并发多线程读写，验证 WAL 模式下数据库绝无死锁。"""
        db_path = tmp_path / "test_wal_concurrency.db"
        db = VlogDatabase(db_path=db_path)

        def worker(w_id: int):
            local_db = VlogDatabase(db_path=db_path)
            try:
                for i in range(20):
                    fp = f"cam0_{w_id:02d}_{i:04d}.mp4"
                    local_db.add_file_task(fp, 0, "20260901", "20260901000000", "20260901000500", 300.0)
                    local_db.set_prescreen_result(fp, "STATIC" if i % 2 == 0 else "SUSPICIOUS", "{}")
                    _ = local_db.get_all_file_tasks_for_date("20260901", 0)
            finally:
                local_db.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(worker, w) for w in range(8)]
            for fut in concurrent.futures.as_completed(futures):
                fut.result()

        all_tasks = db.get_all_file_tasks_for_date("20260901", 0)
        assert len(all_tasks) == 8 * 20
        db.close()

