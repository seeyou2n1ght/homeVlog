"""测试 segments 关系表规范化读写、人工审核打标与时间轴反向纠偏生效"""

import pytest
from src.database import VlogDatabase
from src.timeline import build_timeline
from src.segment import Segment


def test_segments_table_lifecycle_and_review(tmp_path):
    db_path = tmp_path / "test_segments.db"
    db = VlogDatabase(db_path=db_path)
    try:
        fp = "test_cam0_20260901100000_20260901100500.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)

        # 模拟 Analysis 产出 3 个切片
        sample_segments = [
            Segment(start_time=0.0, end_time=10.0, state="STATIC", source_file=fp, file_start_offset=0.0, max_energy=0.5, avg_confidence=0.0),
            Segment(start_time=10.0, end_time=40.0, state="DYNAMIC", source_file=fp, file_start_offset=0.0, max_energy=5.5, avg_confidence=0.0), # 疑似光影假动态
            Segment(start_time=40.0, end_time=300.0, state="STATIC", source_file=fp, file_start_offset=0.0, max_energy=1.8, avg_confidence=0.0), # 临界微动
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
