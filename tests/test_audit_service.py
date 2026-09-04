"""测试独立审核工作台 AuditService 与 API 逻辑"""

import pytest
from src.database import VlogDatabase
from src.segment import Segment
from scripts.audit_tool.service import AuditService


def test_audit_service_overview_and_export(tmp_path):
    db_path = tmp_path / "test_audit.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_cam0_20260901_001.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)

        sample_segs = [
            Segment(start_time=0.0, end_time=10.0, state="STATIC", source_file=fp, file_start_offset=0.0),
            Segment(start_time=10.0, end_time=30.0, state="DYNAMIC", source_file=fp, file_start_offset=0.0, max_energy=4.0),
            Segment(start_time=30.0, end_time=60.0, state="STATIC", source_file=fp, file_start_offset=0.0, max_energy=1.9),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        # 1. 初始 overview 指标
        ov = service.get_overview()
        assert ov["total_files"] == 1
        assert ov["total_segments"] == 3
        assert ov["reviewed_segments"] == 0

        # 2. 疑难排查队列
        anomalies = service.get_anomalies(limit=10)
        assert len(anomalies) >= 1

        # 3. 获取文件分段
        segs_data = service.get_file_segments(filepath=fp)
        assert len(segs_data["segments"]) == 3
        dyn_id = segs_data["segments"][1]["id"]

        # 4. 人工打标: 误报 (FALSE_ALARM)
        assert service.submit_review(dyn_id, "FALSE_ALARM", notes="树影晃动")

        # 5. 再次验证 overview 准确率指标
        ov_updated = service.get_overview()
        assert ov_updated["reviewed_segments"] == 1
        assert ov_updated["labels"]["fp"] == 1

        # 6. 测试报表导出
        csv_str, ctype_csv = service.export_report(fmt="csv")
        assert "FALSE_ALARM" in csv_str
        assert "text/csv" in ctype_csv

        json_str, ctype_json = service.export_report(fmt="json")
        assert "FALSE_ALARM" in json_str
        assert "application/json" in ctype_json

    finally:
        db.close()


def test_audit_service_category_filtering(tmp_path):
    db_path = tmp_path / "test_cat.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "cam0_20260901180000.mp4"
        file_start = 18 * 3600.0  # 64800.0
        db.add_file_task(fp, 0, "20260901", "20260901180000", "20260901180500", 300.0)

        sample_segs = [
            # FP 候选: DYNAMIC 但 avg_confidence == 0 (无主体)
            Segment(start_time=file_start + 10.0, end_time=file_start + 25.0, state="DYNAMIC", source_file=fp, file_start_offset=file_start, avg_confidence=0.0, max_energy=0.8),
            # FN 候选: STATIC 但 max_energy >= 1.5
            Segment(start_time=file_start + 30.0, end_time=file_start + 60.0, state="STATIC", source_file=fp, file_start_offset=file_start, max_energy=2.5),
            # 短碎片抖动: duration < 3.0
            Segment(start_time=file_start + 70.0, end_time=file_start + 72.0, state="DYNAMIC", source_file=fp, file_start_offset=file_start, avg_confidence=0.8, max_energy=3.0),
            # 正常切片
            Segment(start_time=file_start + 80.0, end_time=file_start + 110.0, state="DYNAMIC", source_file=fp, file_start_offset=file_start, avg_confidence=0.9, max_energy=5.0),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        # 标记最后一个切片为已复核 CONFIRMED_MOTION
        file_segs = service.get_file_segments(filepath=fp)["segments"]
        last_seg_id = file_segs[-1]["id"]
        service.submit_review(last_seg_id, "CONFIRMED_MOTION")

        # 1. 疑似误报 (FP)
        fps = service.get_anomalies(category="fp_suspect")
        assert len(fps) == 1
        assert fps[0]["anomaly_type"] == "fp_suspect"
        assert "无目标置信度" in fps[0]["reason_desc"]

        # 2. 疑似漏报 (FN)
        fns = service.get_anomalies(category="fn_suspect")
        assert len(fns) == 1
        assert fns[0]["anomaly_type"] == "fn_suspect"
        assert "微动作漏判" in fns[0]["reason_desc"]

        # 3. 碎片跳变
        jitters = service.get_anomalies(category="jitter")
        assert len(jitters) == 1
        assert jitters[0]["duration"] < 3.0

        # 4. 已复核
        revs = service.get_anomalies(category="reviewed")
        assert len(revs) == 1
        assert revs[0]["manual_label"] == "CONFIRMED_MOTION"

        # 5. 全部待排查疑难 (不包含正常的已复核切片)
        all_list = service.get_anomalies(category="all")
        assert len(all_list) == 3
    finally:
        db.close()


def test_resolve_local_timestamp(tmp_path):
    db_path = tmp_path / "test_resolve.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_cam0_resolve.mp4"
        file_start = 65114.0
        db.add_file_task(fp, 0, "20260320", "20260320180514", "20260320181029", 315.0)

        # 场景 A: 传入日内绝对秒数 (65373.5s) -> 应当换算为 259.5s
        t_local, dur = service._resolve_local_timestamp(fp, 65373.5)
        assert abs(t_local - 259.5) < 1e-3
        assert abs(dur - 315.0) < 1e-3

        # 场景 B: 传入相对秒数 (15.0s) -> 应当直接保留为 15.0s
        t_local_rel, _ = service._resolve_local_timestamp(fp, 15.0)
        assert abs(t_local_rel - 15.0) < 1e-3

        # 场景 C: 寻道越界 (350.0s > 315.0s) -> 截断到 314.9s
        t_clamped, _ = service._resolve_local_timestamp(fp, 350.0)
        assert abs(t_clamped - 314.9) < 1e-3

        # 场景 D: 负数截断
        t_neg, _ = service._resolve_local_timestamp(fp, -10.0)
        assert t_neg == 0.0
    finally:
        db.close()


def test_audit_http_export():
    import threading
    import urllib.request
    from http.server import HTTPServer
    from scripts.audit_tool.app import AuditHandler

    server = HTTPServer(("127.0.0.1", 0), AuditHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        # 1. 测试 CSV 导出 (含 UTF-8 BOM 与命名)
        csv_url = f"http://127.0.0.1:{port}/api/export?format=csv"
        req = urllib.request.Request(csv_url)
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            ctype = resp.headers.get("Content-Type", "")
            cdisp = resp.headers.get("Content-Disposition", "")
            assert "text/csv" in ctype
            assert "homevlog_audit_" in cdisp
            assert ".csv" in cdisp
            raw_bytes = resp.read()
            assert raw_bytes.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM 校验

        # 2. 测试 JSON 导出
        json_url = f"http://127.0.0.1:{port}/api/export?format=json"
        req = urllib.request.Request(json_url)
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            ctype = resp.headers.get("Content-Type", "")
            cdisp = resp.headers.get("Content-Disposition", "")
            assert "application/json" in ctype
            assert "homevlog_audit_" in cdisp
            assert ".json" in cdisp
            raw_bytes = resp.read()
            import json
            parsed = json.loads(raw_bytes.decode("utf-8"))
            assert isinstance(parsed, list)
    finally:
        server.shutdown()
        server.server_close()


