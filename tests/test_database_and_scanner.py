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

    def test_camera_ui_name_priority(self):
        from src.scanner import resolve_camera_ui_name

        path = "XiaomiCamera_01_B888805AA3CD"
        assert resolve_camera_ui_name(path, 0, {"cameras": {"B888805AA3CD": "育儿房"}}) == "育儿房"
        assert resolve_camera_ui_name(path, 0, {"cameras": {}}) == "摄像头 01"
        assert resolve_camera_ui_name("Camera_B888805AA3CD", 0, {"cameras": {}}) == "B888805AA3CD"
        assert resolve_camera_ui_name(None, 0) == "cam0"
        assert resolve_camera_ui_name(None, 2) == "cam2"
        assert resolve_camera_ui_name("", 1) == "cam1"
        assert resolve_camera_ui_name(Path("Camera_B888805AA3CD"), 0, {"cameras": {}}) == "B888805AA3CD"

    def test_resolve_output_filename_custom_templates(self):
        from src.scanner import resolve_output_filename
        mock_cfg = {"cameras": {"B888805AA3CD": "baby_room"}}
        sample_path = r"\\192.168.5.8\LyuShare\XiaomiCamera_01_B888805AA3CD\00_20260402000000_20260402000500.mp4"

        # 默认模式：{date}_{mac}
        name1 = resolve_output_filename("DailyVlog_{date}_{mac}.mp4", "20260402", 0, sample_path, mock_cfg)
        assert name1 == "DailyVlog_20260402_B888805AA3CD.mp4"

        # 语义机位模式：{date}_{camera}
        name2 = resolve_output_filename("DailyVlog_{date}_{camera}.mp4", "20260402", 0, sample_path, mock_cfg)
        assert name2 == "DailyVlog_20260402_baby_room.mp4"

        # 传统编号模式：{date}_cam{index}
        name3 = resolve_output_filename("DailyVlog_{date}_cam{index}.mp4", "20260402", 0, sample_path, mock_cfg)
        assert name3 == "DailyVlog_20260402_cam0.mp4"

        # 自由组合模式：{date}_{camera}_{mac}_#cam{index}
        name4 = resolve_output_filename("{date}_{camera}_{mac}_#cam{index}.mp4", "20260402", 0, sample_path, mock_cfg)
        assert name4 == "20260402_baby_room_B888805AA3CD_#cam0.mp4"

        # 无 MAC 兜底降级
        fallback_path = "C:/Footage/regular_folder/00_20260402000000_20260402000500.mp4"
        name5 = resolve_output_filename("DailyVlog_{date}_{mac}.mp4", "20260402", 1, fallback_path, mock_cfg)
        assert name5 == "DailyVlog_20260402_cam1.mp4"

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



    def test_scan_directory_incremental_and_groups(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.stages.scanner.load_config", lambda: {
            "recovery": {"skip_today": False, "scanner_freeze_minutes": 0}
        })
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

    def test_scan_multiple_directories(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.stages.scanner.load_config", lambda: {
            "recovery": {"skip_today": False, "scanner_freeze_minutes": 0}
        })
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



def test_scan_failure_preserves_registered_files_and_can_resume(tmp_path, monkeypatch, caplog):
    import src.stages.scanner as scanner
    monkeypatch.setattr(scanner, "load_config", lambda: {"recovery": {"scanner_freeze_minutes": 0}})
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    name = "00_20260901080000_20260901080500.mp4"
    (first / name).write_bytes(b"dummy")
    db = VlogDatabase(tmp_path / "scan.db")
    try:
        with pytest.raises(FileNotFoundError):
            scanner.scan_directory(db, [str(first), str(second)])
        assert db.conn.execute("SELECT count(*) FROM file_tasks").fetchone()[0] == 1
        assert "扫描失败" in caplog.text
        second.mkdir()
        (second / name).write_bytes(b"dummy")
        result = scanner.scan_directory(db, [str(first), str(second)])
        assert result.added == 1 and result.skipped == 1
        assert not any(t.name == "scan-progress" for t in __import__("threading").enumerate())
    finally:
        db.close()


@pytest.mark.parametrize("freeze_minutes", [0, 10])
def test_scan_only_reads_attributes_when_freeze_check_requires_them(tmp_path, monkeypatch, caplog, freeze_minutes):
    from contextlib import nullcontext
    from unittest.mock import Mock
    import src.stages.scanner as scanner
    entry = Mock()
    entry.name = "00_20260901080000_20260901080500.mp4"
    entry.path = str(tmp_path / entry.name)
    entry.is_file.return_value = True
    entry.stat.side_effect = PermissionError("NAS attributes unavailable")
    monkeypatch.setattr(scanner, "load_config", lambda: {"recovery": {"scanner_freeze_minutes": freeze_minutes}})
    monkeypatch.setattr(scanner.os, "scandir", lambda _path: nullcontext(iter([entry])))
    db = VlogDatabase(tmp_path / "scan.db")
    try:
        if freeze_minutes:
            with pytest.raises(PermissionError):
                scanner.scan_directory(db, str(tmp_path))
            assert entry.path in caplog.text
            assert db.conn.execute("SELECT count(*) FROM file_tasks").fetchone()[0] == 0
        else:
            assert scanner.scan_directory(db, str(tmp_path)).added == 1
            entry.stat.assert_not_called()
    finally:
        db.close()


def test_blocking_nas_enumeration_reports_heartbeat_and_stops_reporter(tmp_path, monkeypatch, caplog):
    import threading
    import logging
    import src.stages.scanner as scanner
    caplog.set_level(logging.INFO, logger="homevlog")
    entered, reported, release = threading.Event(), threading.Event(), threading.Event()
    original_scandir = scanner.os.scandir
    original_progress = scanner._ScanProgress
    monkeypatch.setattr(scanner, "_ScanProgress", lambda: original_progress(interval_s=0.01))
    monkeypatch.setattr(scanner, "load_config", lambda: {"recovery": {"scanner_freeze_minutes": 0}})
    def blocked(path):
        entered.set()
        assert release.wait(2)
        return original_scandir(path)
    monkeypatch.setattr(scanner.os, "scandir", blocked)
    records = []
    class Capture(logging.Handler):
        def emit(self, record):
            if "扫描仍在进行" in record.getMessage():
                records.append(record)
                reported.set()
    capture = Capture()
    scanner.logger.addHandler(capture)
    db = VlogDatabase(tmp_path / "scan.db")
    errors = []
    def run():
        try:
            scanner.scan_directory(db, str(tmp_path))
        except BaseException as exc:
            errors.append(exc)
    worker = threading.Thread(target=run)
    try:
        worker.start()
        assert entered.wait(1) and reported.wait(1)
        assert "枚举目录" in records[0].getMessage()
        assert records[0].startup_progress
    finally:
        release.set()
        worker.join(3)
        scanner.logger.removeHandler(capture)
        db.close()
    assert not errors and not worker.is_alive()
    assert not any(t.name == "scan-progress" for t in threading.enumerate())


class TestVlogDatabaseLifecycle:
    """测试任务生命周期状态流转与更新。"""

    def test_task_state_transitions(self, tmp_path):
        db_path = tmp_path / "test_lifecycle.db"
        db = VlogDatabase(db_path=db_path)
        try:
            filepath = "00_20260901100000_20260901100500.mp4"
            db.add_file_task(filepath, 0, "20260901", "20260901100000", "20260901100500", 300.0)

            # 1. 初始为 PENDING
            tasks = db.get_all_file_tasks_for_date("20260901", 0)
            assert len(tasks) == 1
            assert tasks[0]["filepath"] == filepath
            assert tasks[0]["prescreen_status"] == "PENDING"
            assert db.get_pending_file_count_for_date("20260901", 0) == 1

            # 2. 预筛标记为 SUSPICIOUS
            db.set_prescreen_result(filepath, "SUSPICIOUS", '{"diff": 15}')
            tasks = db.get_all_file_tasks_for_date("20260901", 0)
            assert tasks[0]["prescreen_status"] == "SUSPICIOUS"
            assert db.get_pending_file_count_for_date("20260901", 0) == 1

            # 3. 详细分析标记为 ANALYZED
            db.set_analysis_result(filepath, "ANALYZED", '[{"start": 0, "end": 10, "label": "DYNAMIC"}]')
            tasks = db.get_all_file_tasks_for_date("20260901", 0)
            assert tasks[0]["analysis_status"] == "ANALYZED"
            assert db.get_pending_file_count_for_date("20260901", 0) == 0

            # 4. 渲染任务生命周期
            db.upsert_render_task("20260901", 0, "PENDING")
            assert not db.is_render_completed("20260901", 0)
            db.set_render_status("20260901", 0, "RENDERING")
            db.set_render_status("20260901", 0, "COMPLETED", output_file="DailyVlog_20260901_cam0.mp4")
            assert db.is_render_completed("20260901", 0)

            # 5. 音频元数据更新与截断时空同步
            db.set_file_metadata(filepath, has_audio=1)
            all_tasks = db.get_all_file_tasks_for_date("20260901", 0)
            assert all_tasks[0]["has_audio"] == 1
            assert all_tasks[0]["duration_verified"] == 0
            # 模拟截断流校准: 300s -> 58.0s，file_end_time 必须从 100500 精准同步为 100058
            db.set_file_metadata(filepath, has_audio=1, duration=58.0)
            all_tasks = db.get_all_file_tasks_for_date("20260901", 0)
            assert all_tasks[0]["file_duration"] == 58.0
            assert all_tasks[0]["file_end_time"] == "20260901100058"
            assert all_tasks[0]["duration_verified"] == 1

            # 6. 异常坏片受控排除
            assert db.exclude_file_task(filepath, reason="CORRUPTED_PHYSICAL_MEDIA")
            all_tasks = db.get_all_file_tasks_for_date("20260901", 0)
            assert all_tasks[0]["prescreen_status"] == "SKIPPED_EXCLUDED"
            assert all_tasks[0]["analysis_status"] == "SKIPPED_EXCLUDED"
            assert "CORRUPTED" in all_tasks[0]["error_msg"]

        finally:
            db.close()

    def test_reset_failed_tasks_self_healing(self, tmp_path):
        """FAILED 预筛/分析任务自动重置为 PENDING，retry_count 上限后不再重置。"""
        db_path = tmp_path / "test_reset.db"
        db = VlogDatabase(db_path=db_path)
        try:
            db.add_file_task("f1.mp4", 0, "20260901", "20260901100000", "20260901100500", 300.0)
            db.add_file_task("f2.mp4", 0, "20260901", "20260901100500", "20260901101000", 300.0)
            db.set_prescreen_result("f1.mp4", "FAILED", "")
            db.set_prescreen_result("f2.mp4", "SUSPICIOUS", "")
            db.set_analysis_result("f2.mp4", "FAILED", "")

            res = db.reset_failed_tasks("20260901", 0, max_retries=2)
            assert res == {"prescreen": 1, "analysis": 1}
            assert db.get_pending_file_count_for_date("20260901", 0) == 2

            # 再次失败并重置：retry_count 达到上限后不再重置
            db.set_prescreen_result("f1.mp4", "FAILED", "")
            db.reset_failed_tasks("20260901", 0, max_retries=2)
            db.set_prescreen_result("f1.mp4", "FAILED", "")
            res = db.reset_failed_tasks("20260901", 0, max_retries=2)
            assert res["prescreen"] == 0
            tasks = db.get_all_file_tasks_for_date("20260901", 0)
            assert tasks[0]["prescreen_status"] == "FAILED"
        finally:
            db.close()

    def test_failure_reasons_are_persisted_and_cleared(self, tmp_path):
        db = VlogDatabase(tmp_path / "reasons.db")
        try:
            db.add_file_task("clip.mp4", 0, "20260901", "20260901000000", "20260901000100", 60)
            db.set_prescreen_result("clip.mp4", "FAILED", error_msg="invalid media")
            assert db.get_file_task("clip.mp4")["error_msg"] == "invalid media"
            db.reset_failed_tasks("20260901", 0)
            assert db.get_file_task("clip.mp4")["error_msg"] is None
            db.set_render_status("20260901", 0, "FAILED", error_msg="batch validation failed")
            assert db.conn.execute("SELECT error_msg FROM render_tasks").fetchone()[0] == "batch validation failed"
            db.upsert_render_task("20260901", 0, "RENDERING")
            assert db.conn.execute("SELECT error_msg FROM render_tasks").fetchone()[0] is None
            db.set_render_status("20260901", 0, "COMPLETED", output_file="out.mp4")
            assert db.conn.execute("SELECT error_msg FROM render_tasks").fetchone()[0] is None
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


class TestSegmentsTableLifecycle:
    """测试 segments 关系表规范化读写、人工审核打标与时间轴反向纠偏生效"""

    def test_segments_table_lifecycle_and_review(self, tmp_path):
        from src.timeline import build_timeline
        from src.segment import Segment

        db_path = tmp_path / "test_segments.db"
        db = VlogDatabase(db_path=db_path)
        try:
            fp = "test_cam0_20260901100000_20260901100500.mp4"
            db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)

            # 模拟 Analysis 产出 3 个切片
            sample_segments = [
                Segment(start_time=0.0, end_time=10.0, state="STATIC", source_file=fp, file_start_offset=0.0, max_energy=0.5, avg_confidence=0.0),
                Segment(start_time=10.0, end_time=40.0, state="DYNAMIC", source_file=fp, file_start_offset=0.0, max_energy=5.5, avg_confidence=0.0),
                Segment(start_time=40.0, end_time=300.0, state="STATIC", source_file=fp, file_start_offset=0.0, max_energy=1.8, avg_confidence=0.0),
            ]
            db.set_analysis_result(fp, "ANALYZED", sample_segments)

            # 1. 验证 segments 表展开入库
            segs = db.get_segments_for_file(fp)
            assert len(segs) == 3
            assert segs[0]["state"] == "STATIC"
            assert segs[1]["state"] == "DYNAMIC"
            assert segs[1]["max_energy"] == 5.5

            # 2. 验证主动疑难排查召回 (Active Learning)
            anomalies = db.get_anomaly_segments(date="20260901", cam_index=0)
            assert len(anomalies) >= 2
            anomaly_states = [a["state"] for a in anomalies]
            assert "DYNAMIC" in anomaly_states
            assert "STATIC" in anomaly_states

            # 3. 模拟人工审核打标: 将 segs[1] 标记为 FALSE_ALARM (误判，实际为静止)
            dynamic_seg_id = segs[1]["id"]
            assert db.update_segment_review(dynamic_seg_id, "FALSE_ALARM", notes="窗帘被风吹动")

            updated_segs = db.get_segments_for_file(fp)
            dyn_updated = next(s for s in updated_segs if s["id"] == dynamic_seg_id)
            assert dyn_updated["manual_label"] == "FALSE_ALARM"
            assert dyn_updated["review_notes"] == "窗帘被风吹动"
            assert dyn_updated["reviewed_at"] is not None

            # 4. 验证反向纠偏: build_timeline 自动将打标后的 FALSE_ALARM 降级为 STATIC
            timeline = build_timeline(db, "20260901", 0)
            dynamic_timeline_segs = [t for t in timeline if t.state == "DYNAMIC"]
            assert len(dynamic_timeline_segs) == 0
        finally:
            db.close()

    def test_empty_analysis_replaces_old_segments_and_bad_json_fails(self, tmp_path):
        from src.segment import Segment

        db = VlogDatabase(tmp_path / "replace.sqlite")
        fp = "clip.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901000000", "20260901000100", 60)
        db.set_analysis_result(fp, [Segment(0, 60, "DYNAMIC", fp)])
        assert len(db.get_segments_for_file(fp)) == 1
        db.set_analysis_result(fp, "ANALYZED", [])
        assert db.get_segments_for_file(fp) == []
        with pytest.raises(ValueError):
            db.set_analysis_result(fp, "ANALYZED", "{broken")
        db.close()

    def test_file_task_query_can_be_scoped_to_render_batch(self, tmp_path):
        db = VlogDatabase(tmp_path / "scoped.sqlite")
        for index in range(3):
            fp = f"clip_{index}.mp4"
            db.add_file_task(fp, 0, "20260901", f"20260901000{index}00", f"20260901000{index}59", 59)
        rows = db.get_all_file_tasks_for_date("20260901", 0, filepaths=["clip_1.mp4"])
        assert [row["filepath"] for row in rows] == ["clip_1.mp4"]
        db.close()

    def test_gold_candidates_clustering_and_batch_review(self, tmp_path):
        from src.segment import Segment

        db = VlogDatabase(tmp_path / "gold.sqlite")
        fp = "sample_video.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901120000", "20260901123000", 1800)

        # 构造模拟切片:
        # 1. 连续 5 个树影晃动切片 (同文件、同类型 high_energy_no_target、相邻时间 < 300s)
        # 2. 1 个置信度摇摆切片 (borderline_confidence)
        # 3. 1 个微动临界切片 (fn_suspect)
        segs = [
            Segment(10.0, 15.0, "DYNAMIC", fp, max_energy=18.0, avg_confidence=0.0),
            Segment(20.0, 25.0, "DYNAMIC", fp, max_energy=22.0, avg_confidence=0.0),
            Segment(30.0, 35.0, "DYNAMIC", fp, max_energy=15.0, avg_confidence=0.0),
            Segment(40.0, 45.0, "DYNAMIC", fp, max_energy=12.0, avg_confidence=0.0),
            Segment(50.0, 55.0, "DYNAMIC", fp, max_energy=19.0, avg_confidence=0.0),
            Segment(600.0, 605.0, "DYNAMIC", fp, max_energy=5.0, avg_confidence=0.28),
            Segment(800.0, 810.0, "STATIC", fp, max_energy=2.5, avg_confidence=0.0),
        ]
        db.set_analysis_result(fp, segs)

        # 召回主动学习黄金候选集
        candidates = db.get_anomaly_segments(category="gold_candidates", date="20260901", cam_index=0, limit=10)
        assert len(candidates) >= 3

        # 验证聚类聚合生效: 5 个连续树影应被聚合成 1 个代表切片
        high_energy_reps = [c for c in candidates if c["anomaly_type"] == "high_energy_no_target"]
        assert len(high_energy_reps) == 1
        rep = high_energy_reps[0]
        assert rep["cluster_count"] == 5
        assert len(rep["cluster_ids"]) == 5
        assert rep["max_energy"] == 22.0  # 代表切片应当选能量/得分最高的一个

        # 验证多样性
        types = {c["anomaly_type"] for c in candidates}
        assert "borderline_confidence" in types
        assert "fn_suspect" in types

        # 验证批量更新
        updated_count = db.batch_update_segment_review(rep["cluster_ids"], "FALSE_ALARM", notes="树影晃动聚类一键纠偏")
        assert updated_count == 5

        # 再次获取候选集，已被标记的 5 个切片不应再出现
        next_candidates = db.get_anomaly_segments(category="gold_candidates", date="20260901", cam_index=0)
        assert not any(c["anomaly_type"] == "high_energy_no_target" for c in next_candidates)

        db.close()
