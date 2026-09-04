"""测试即时重浓缩引擎 (ReRenderManager) 与物理素材校验"""

from pathlib import Path
import pytest
from src.database import VlogDatabase
from scripts.audit_tool.rerender import ReRenderManager, check_source_files_accessibility


def test_check_source_files_accessibility(tmp_path):
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
    assert can_proceed is True
    assert len(valid) == 2
    assert len(missing) == 1
    assert str(f3) in missing

    # 2. 全缺失硬阻断测试 (NAS 未挂载场景)
    tasks_all_missing = [{"filepath": str(tmp_path / f"non_existent_{i}.mp4")} for i in range(3)]
    can_proceed_all_missing, _, missing_all = check_source_files_accessibility(tasks_all_missing)
    assert can_proceed_all_missing is False
    assert len(missing_all) == 3


def test_rerender_manager_lifecycle(tmp_path):
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
