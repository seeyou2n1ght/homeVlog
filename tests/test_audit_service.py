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
