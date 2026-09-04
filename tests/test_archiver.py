"""tests/test_archiver.py

测试人工审核校正帧画面物理归档、manifest元数据索引记录、
数据库路径回填以及批量补齐归档功能。
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.database import VlogDatabase
from src.segment import Segment
from src.archiver import (
    extract_and_archive_frame,
    batch_archive_all_reviewed,
    get_archive_stats,
)
from scripts.audit_tool.service import AuditService


def test_extract_and_archive_frame_lifecycle(tmp_path):
    db_path = tmp_path / "test_arch.db"
    archive_dir = tmp_path / "archive"
    db = VlogDatabase(db_path=db_path)

    try:
        fp = str(tmp_path / "test_video_20260901.mp4")
        # 创建空视频文件模拟物理存在
        Path(fp).write_bytes(b"dummy video data")

        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)
        sample_segments = [
            Segment(start_time=10.0, end_time=20.0, state="DYNAMIC", source_file=fp, file_start_offset=0.0, max_energy=5.0),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segments)
        segs = db.get_segments_for_file(fp)
        seg_id = segs[0]["id"]

        # 标记为 FALSE_ALARM (光影误报)
        db.update_segment_review(seg_id, "FALSE_ALARM", notes="树影晃动")

        # Mock FFmpeg 抽取帧生成真实的图片
        def fake_ffmpeg_run(cmd, capture_output, timeout):
            out_img_path = Path(cmd[-1])
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            out_img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 50)  # JPEG header
            mock_res = MagicMock()
            mock_res.returncode = 0
            return mock_res

        with patch("subprocess.run", side_effect=fake_ffmpeg_run):
            saved_path = extract_and_archive_frame(db, seg_id, archive_dir=archive_dir)
            assert saved_path is not None
            assert saved_path.exists()
            assert "seg" in saved_path.name
            assert "false_alarm" in saved_path.name

        # 验证 manifest.jsonl 写入
        manifest_file = archive_dir / "manifest.jsonl"
        assert manifest_file.exists()
        lines = [line.strip() for line in manifest_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["segment_id"] == seg_id
        assert record["manual_label"] == "FALSE_ALARM"
        assert record["review_notes"] == "树影晃动"
        assert record["frame_timestamp"] == 15.0  # (10 + 20) / 2

        # 验证数据库回填
        updated_seg = db.get_segment_by_id(seg_id)
        assert updated_seg is not None
        assert updated_seg["archived_frame_path"] is not None

        # 验证统计
        stats = get_archive_stats(archive_dir=archive_dir)
        assert stats["total_images"] == 1
        assert stats["label_counts"].get("FALSE_ALARM") == 1
    finally:
        db.close()


def test_batch_archive_all_reviewed(tmp_path):
    db_path = tmp_path / "test_batch_arch.db"
    archive_dir = tmp_path / "archive"
    db = VlogDatabase(db_path=db_path)

    try:
        fp = str(tmp_path / "test_video_20260902.mp4")
        Path(fp).write_bytes(b"dummy")

        db.add_file_task(fp, 0, "20260902", "20260902120000", "20260902120500", 300.0)
        sample_segments = [
            Segment(start_time=0.0, end_time=10.0, state="DYNAMIC", source_file=fp, file_start_offset=0.0, max_energy=4.0),
            Segment(start_time=10.0, end_time=25.0, state="STATIC", source_file=fp, file_start_offset=0.0, max_energy=1.0),
            Segment(start_time=25.0, end_time=40.0, state="DYNAMIC", source_file=fp, file_start_offset=0.0, max_energy=6.0),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segments)
        segs = db.get_segments_for_file(fp)

        # 标记两个切片，留一个未审核
        db.update_segment_review(segs[0]["id"], "FALSE_ALARM")
        db.update_segment_review(segs[1]["id"], "MISSED_MOTION", notes="婴儿踢被子漏检")

        def fake_ffmpeg_run(cmd, capture_output, timeout):
            out_img_path = Path(cmd[-1])
            out_img_path.parent.mkdir(parents=True, exist_ok=True)
            out_img_path.write_bytes(b"\xff\xd8\xff" + b"\x01" * 30)
            mock_res = MagicMock()
            mock_res.returncode = 0
            return mock_res

        with patch("subprocess.run", side_effect=fake_ffmpeg_run):
            summary = batch_archive_all_reviewed(db, archive_dir=archive_dir)
            assert summary["total"] == 2
            assert summary["archived"] == 2
            assert summary["failed"] == 0

        stats = get_archive_stats(archive_dir=archive_dir)
        assert stats["total_images"] == 2
        assert stats["label_counts"]["FALSE_ALARM"] == 1
        assert stats["label_counts"]["MISSED_MOTION"] == 1
    finally:
        db.close()


def test_audit_service_submit_review_archive_hook(tmp_path):
    db_path = tmp_path / "test_svc_arch.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = str(tmp_path / "test_video_20260903.mp4")
        Path(fp).write_bytes(b"dummy")

        db.add_file_task(fp, 0, "20260903", "20260903140000", "20260903140500", 300.0)
        sample_segments = [
            Segment(start_time=5.0, end_time=15.0, state="DYNAMIC", source_file=fp, file_start_offset=0.0, max_energy=4.0),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segments)
        segs = db.get_segments_for_file(fp)
        seg_id = segs[0]["id"]

        with patch("src.archiver.extract_and_archive_frame") as mock_extract:
            ok = service.submit_review(seg_id, "CONFIRMED_MOTION", notes="正常走动")
            assert ok is True
            # 等待后台线程执行
            import time
            time.sleep(0.1)
            assert mock_extract.called
    finally:
        db.close()
