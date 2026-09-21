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
    import http.cookiejar
    import threading
    import urllib.error
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

        # 写接口同时要求同源 Host/Origin 和启动时 token cookie。
        unauth = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/review", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(unauth)
        assert exc.value.code == 403

        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        opener.open(f"http://127.0.0.1:{port}/").read()
        authenticated = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/review", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            opener.open(authenticated)
        assert exc.value.code == 400

        foreign = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/overview",
            headers={"Origin": "https://example.invalid"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(foreign)
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()


class TestReRenderWorkflow:
    """测试即时重浓缩引擎 (ReRenderManager) 与物理素材校验"""

    def test_check_source_files_accessibility(self, tmp_path):
        from scripts.audit_tool.rerender import check_source_files_accessibility

        # 创建 2 个真实文件和 1 个假文件
        f1 = tmp_path / "video1.mp4"
        f1.write_bytes(b"dummy video data 1")
        f2 = tmp_path / "video2.mp4"
        f2.write_bytes(b"dummy video data 2")
        f3 = tmp_path / "video3_missing.mp4"

        # 1. 混合测试
        tasks = [
            {"filepath": str(f1)},
            {"filepath": str(f2)},
            {"filepath": str(f3)},
        ]
        can_proceed, valid, missing = check_source_files_accessibility(tasks)
        assert can_proceed is False
        assert len(valid) == 2
        assert len(missing) == 1
        assert str(f3) in missing

        # 2. 全缺失硬阻断测试 (NAS 未挂载场景)
        tasks_all_missing = [{"filepath": str(tmp_path / f"non_existent_{i}.mp4")} for i in range(3)]
        can_proceed_all_missing, _, missing_all = check_source_files_accessibility(tasks_all_missing)
        assert can_proceed_all_missing is False
        assert len(missing_all) == 3

    def test_rerender_manager_lifecycle(self, tmp_path):
        from scripts.audit_tool.rerender import ReRenderManager
        from src.database import VlogDatabase

        mgr = ReRenderManager()
        db_path = tmp_path / "test_rerender.db"
        db = VlogDatabase(db_path=db_path)

        try:
            # 空任务状态
            st = mgr.get_status("20260901", 0)
            assert st["status"] == "IDLE"

            # 测试任务启动对无文件任务的拦截
            res = mgr.start_rerender(db, "20260901", 0)
            assert res["status"] == "CHECKING"

            # 稍等片刻，worker 会因为在数据库中找不到 file_tasks 而转为 FAILED
            import time
            for _ in range(20):
                st = mgr.get_status("20260901", 0)
                if st["status"] in ("FAILED", "COMPLETED"):
                    break
                time.sleep(0.05)

            assert st["status"] == "FAILED"
            assert "No file tasks found" in st["error"]

        finally:
            db.close()

    def test_cancel_is_scoped_to_the_rerender_task(self):
        import threading

        from scripts.audit_tool.rerender import ReRenderManager
        from src.hardware.ffmpeg import FFmpegProcessRegistry

        mgr = ReRenderManager()
        mgr._init_manager()
        key = mgr.get_task_key("20260901", 0)
        mgr.cancel_flags[key] = threading.Event()
        mgr.tasks[key] = {"status": "RENDERING"}
        FFmpegProcessRegistry.reset_interrupted()
        assert mgr.cancel_task("20260901", 0)
        assert mgr.tasks[key]["status"] == "CANCELLED"
        assert not FFmpegProcessRegistry.is_interrupted()

    def test_rerender_output_filename_resolution(self):
        from src.scanner import resolve_output_filename
        mock_cfg = {"cameras": {"B888805AA3CD": "living_room"}}
        sample_path = r"\\192.168.5.8\LyuShare\XiaomiCamera_01_B888805AA3CD\00_20260402000000_20260402000500.mp4"

        base = resolve_output_filename("DailyVlog_{date}_{mac}.mp4", "20260402", 0, sample_path, mock_cfg)
        assert base == "DailyVlog_20260402_B888805AA3CD.mp4"

        # 校验带版本标识的重浓缩派生命名
        from pathlib import Path
        output_version = "v2"
        stem = Path(base).stem
        suffix = Path(base).suffix
        final_name = f"{stem}_{output_version}{suffix}"
        assert final_name == "DailyVlog_20260402_B888805AA3CD_v2.mp4"


def test_audit_scenario_and_clear_review(tmp_path):
    """测试场景归因标签、备注打包解析与打标撤销"""
    db_path = tmp_path / "test_scenario.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_scenario_cam0.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)
        sample_segs = [
            Segment(start_time=0.0, end_time=15.0, state="DYNAMIC", source_file=fp, max_energy=5.0),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        file_segs = service.get_file_segments(filepath=fp)["segments"]
        seg_id = file_segs[0]["id"]

        # 1. 提交带场景归因与用户备注的审核
        ok = service.submit_review(seg_id, "FALSE_ALARM", notes="车灯反射白墙", scenario="HEADLIGHT")
        assert ok is True

        # 2. 检验 get_file_segments 与 get_anomalies 是否能正确解包 scenario 和 user_notes
        segs_after = service.get_file_segments(filepath=fp)["segments"]
        assert segs_after[0]["manual_label"] == "FALSE_ALARM"
        assert segs_after[0]["scenario"] == "HEADLIGHT"
        assert segs_after[0]["user_notes"] == "车灯反射白墙"

        # 3. 撤销打标
        assert service.clear_review(seg_id) is True

        segs_cleared = service.get_file_segments(filepath=fp)["segments"]
        assert segs_cleared[0]["manual_label"] is None
        assert segs_cleared[0]["scenario"] == ""
        assert segs_cleared[0]["user_notes"] == ""
    finally:
        db.close()


def test_confusion_matrix_cross_tabulation(tmp_path):
    """验证混淆矩阵基于 (predicted_state, manual_label) 交叉计算"""
    db_path = tmp_path / "test_matrix.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_matrix_cam0.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)
        sample_segs = [
            # 1. DYNAMIC -> CONFIRMED_MOTION (TP)
            Segment(start_time=0.0, end_time=10.0, state="DYNAMIC", source_file=fp, max_energy=10.0),
            # 2. DYNAMIC -> FALSE_ALARM (FP)
            Segment(start_time=10.0, end_time=20.0, state="DYNAMIC", source_file=fp, max_energy=3.0),
            # 3. STATIC -> MISSED_MOTION (FN)
            Segment(start_time=20.0, end_time=30.0, state="STATIC", source_file=fp, max_energy=2.0),
            # 4. STATIC -> CONFIRMED_STATIC (TN)
            Segment(start_time=30.0, end_time=40.0, state="STATIC", source_file=fp, max_energy=0.5),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        segs = service.get_file_segments(filepath=fp)["segments"]
        service.submit_review(segs[0]["id"], "CONFIRMED_MOTION")
        service.submit_review(segs[1]["id"], "FALSE_ALARM", scenario="LIGHT_SHADOW")
        service.submit_review(segs[2]["id"], "MISSED_MOTION", scenario="INFANT_MOTION")
        service.submit_review(segs[3]["id"], "CONFIRMED_STATIC")

        ov = service.get_overview()
        labels = ov["labels"]
        assert labels["tp"] == 1
        assert labels["fp"] == 1
        assert labels["fn"] == 1
        assert labels["tn"] == 1

        metrics = ov["metrics"]
        assert metrics["precision"] == 50.0  # 1 / (1 + 1)
        assert metrics["recall"] == 50.0     # 1 / (1 + 1)
        assert metrics["f1_score"] == 50.0
    finally:
        db.close()


def test_save_bounding_box(tmp_path, monkeypatch):
    """测试人工绘制 BBox 并保存为 YOLO 规范的 txt 文件"""
    from pathlib import Path
    db_path = tmp_path / "test_bbox.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    # 隔离归档目录到临时目录
    fake_archive = tmp_path / "feedback_archive"
    fake_archive.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("scripts.audit_tool.service.ARCHIVE_DIR", fake_archive)

    try:
        boxes = [
            {"class_id": 0, "x_center": 0.5, "y_center": 0.6, "width": 0.2, "height": 0.4},
            {"class_id": 15, "x_center": 0.8, "y_center": 0.9, "width": 0.1, "height": 0.1},
        ]
        res = service.save_bounding_box(boxes=boxes, image_name="test_peak.jpg")
        assert res["success"] is True
        assert res["box_count"] == 2
        assert len(res["saved_paths"]) >= 1

        # 检查生成的文件内容
        saved_txt = Path(res["saved_paths"][0])
        assert saved_txt.exists()
        lines = saved_txt.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert lines[0] == "0 0.500000 0.600000 0.200000 0.400000"
        assert lines[1] == "15 0.800000 0.900000 0.100000 0.100000"
    finally:
        db.close()


def test_tuning_insights(tmp_path):
    """测试根据人工审核记录量化生成 settings.yaml 调参建议"""
    db_path = tmp_path / "test_tuning.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_tuning_cam0.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)
        sample_segs = [
            Segment(start_time=0.0, end_time=10.0, state="DYNAMIC", source_file=fp, max_energy=3.2),
            Segment(start_time=10.0, end_time=20.0, state="DYNAMIC", source_file=fp, max_energy=3.8),
            Segment(start_time=20.0, end_time=30.0, state="DYNAMIC", source_file=fp, max_energy=4.0),
            Segment(start_time=30.0, end_time=40.0, state="STATIC", source_file=fp, max_energy=2.0),
            Segment(start_time=40.0, end_time=50.0, state="STATIC", source_file=fp, max_energy=1.8),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        segs = service.get_file_segments(filepath=fp)["segments"]
        # 3 处光影误报
        service.submit_review(segs[0]["id"], "FALSE_ALARM", scenario="LIGHT_SHADOW")
        service.submit_review(segs[1]["id"], "FALSE_ALARM", scenario="LIGHT_SHADOW")
        service.submit_review(segs[2]["id"], "FALSE_ALARM", scenario="LIGHT_SHADOW")
        # 2 处婴儿微动漏报
        service.submit_review(segs[3]["id"], "MISSED_MOTION", scenario="INFANT_MOTION")
        service.submit_review(segs[4]["id"], "MISSED_MOTION", scenario="INFANT_MOTION")

        insights = service.get_tuning_insights()
        assert insights["total_reviewed"] == 5
        assert insights["scenario_counts"]["LIGHT_SHADOW"] == 3
        assert insights["scenario_counts"]["INFANT_MOTION"] == 2

        # 检查是否给出了 min_motion_threshold 与 yolo.confidence 的调参建议
        recs = insights["recommendations"]
        params = [r["param"] for r in recs]
        assert "detection.min_motion_threshold" in params
        assert "yolo.confidence" in params
    finally:
        db.close()


def test_anomaly_offset_pagination(tmp_path):
    """测试 get_anomalies 的 offset 分页功能"""
    db_path = tmp_path / "test_page.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_page_cam0.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)
        sample_segs = [
            Segment(start_time=float(i * 10), end_time=float(i * 10 + 2), state="DYNAMIC", source_file=fp)
            for i in range(10)
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        page1 = service.get_anomalies(category="jitter", limit=3, offset=0)
        assert len(page1) == 3

        page2 = service.get_anomalies(category="jitter", limit=3, offset=3)
        assert len(page2) == 3

        # 两页的 id 应无交集
        ids1 = {x["id"] for x in page1}
        ids2 = {x["id"] for x in page2}
        assert len(ids1.intersection(ids2)) == 0
    finally:
        db.close()


def test_audit_http_new_endpoints(tmp_path, monkeypatch):
    """测试新增加的 HTTP 端点: /api/tuning_insights, /api/clear_review, /api/save_bbox, /api/export_yolo_dataset"""
    import http.cookiejar
    import json
    import threading
    import urllib.request
    from http.server import HTTPServer
    from scripts.audit_tool.app import AuditHandler

    fake_archive = tmp_path / "feedback_archive"
    fake_archive.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("scripts.audit_tool.service.ARCHIVE_DIR", fake_archive)

    server = HTTPServer(("127.0.0.1", 0), AuditHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        opener.open(f"http://127.0.0.1:{port}/").read()

        # GET /api/tuning_insights
        with opener.open(f"http://127.0.0.1:{port}/api/tuning_insights") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "total_reviewed" in data
            assert "recommendations" in data

        # POST /api/save_bbox
        bbox_payload = json.dumps({
            "boxes": [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.2}],
            "image_name": "test_http_bbox.jpg"
        }).encode("utf-8")
        req_bbox = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/save_bbox",
            data=bbox_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with opener.open(req_bbox) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert data["box_count"] == 1

        # POST /api/clear_review
        clear_payload = json.dumps({"segment_id": 99999}).encode("utf-8")
        req_clear = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/clear_review",
            data=clear_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with opener.open(req_clear) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "success" in data

        # POST /api/export_yolo_dataset
        exp_payload = json.dumps({"val_ratio": 0.2, "output_dir": str(tmp_path / "yolo_out")}).encode("utf-8")
        req_exp = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/export_yolo_dataset",
            data=exp_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with opener.open(req_exp) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "success" in data

    finally:
        server.shutdown()
        server.server_close()
