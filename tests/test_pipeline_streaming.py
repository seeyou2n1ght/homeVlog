"""测试模块 6: 流式管线编排 (StreamingOrchestrator) 与异常容错机制.

覆盖：
1. 流式并发编排器 (StreamingOrchestrator) 队列生命周期与收尾终止；
2. 异常切片（损坏文件、无有效标签）的隔离与容错，防止全管线死锁或崩溃；
3. 渲染管理器 (RenderManager) 批次聚合与 failover 重试；
4. 调度器与并发 Worker 协同无死锁。
"""

import queue
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.database import VlogDatabase
from src.pipeline import StreamingOrchestrator, process_date_cam


class TestStreamingOrchestratorLifecycle:
    """测试编排器队列管理与多线程终止条件。"""

    def test_orchestrator_initial_state(self, tmp_path):
        db_path = tmp_path / "test_pipe.db"
        db = VlogDatabase(db_path=db_path)
        cfg = {
            "pipeline": {"render_start_delay": 0},
            "render": {"batch_max_files": 4},
            "scheduler": {"watermark_high": 10, "watermark_low": 3},
        }
        try:
            orch = StreamingOrchestrator(
                db=db,
                date="20260901",
                cam_index=0,
                config=cfg,
                render_enabled=False,
                dashboard_enabled=False,
            )
            assert orch.prescreen_queue.empty()
            assert orch.analysis_queue.empty()
            assert orch.render_batch_queue.empty()
            assert len(orch.errors) == 0
        finally:
            db.close()

    def test_corrupt_file_fault_tolerance(self, tmp_path):
        """当某些切片在粗筛或分析中抛出异常时，管线优雅捕获并记录错误，不阻断正常运行。"""
        db_path = tmp_path / "test_corrupt.db"
        db = VlogDatabase(db_path=db_path)
        cfg = {
            "pipeline": {"render_start_delay": 0, "prescreen_gpu_policy": "qsv_only"},
            "render": {"batch_max_files": 4},
            "detection": {"prescreen_parallel": 1, "analysis_max_workers": 1},
        }
        try:
            # 插入一个模拟任务
            db.add_file_task("corrupt_clip.mp4", 0, "20260901", "20260901000000", "20260901000500", 300.0)

            orch = StreamingOrchestrator(
                db=db,
                date="20260901",
                cam_index=0,
                config=cfg,
                render_enabled=False,
                dashboard_enabled=False,
            )

            # 模拟 prescreen 抛出异常
            with patch("src.pipeline.prescreen_file", side_effect=RuntimeError("Corrupted bitstream")):
                paths = orch.run()
                assert len(paths) == 0

            # 验证错误被记录
            assert len(orch.errors) >= 1
            assert "prescreen failed" in orch.errors[0]

            # 数据库状态应被更新为 FAILED
            tasks = db.get_all_file_tasks_for_date("20260901", 0)
            assert tasks[0]["prescreen_status"] == "FAILED"
        finally:
            db.close()


class TestProcessDateCamPipeline:
    """测试 process_date_cam 全流程控制与跳过逻辑。"""

    def test_skip_already_completed(self, tmp_path):
        db_path = tmp_path / "test_skip.db"
        db = VlogDatabase(db_path=db_path)
        try:
            db.set_render_status("20260901", 0, "COMPLETED", output_file="out.mp4")
            # 模拟已完成且无待处理切片

            with patch("src.pipeline.load_config", return_value={}):
                with patch("src.pipeline.get_monitor"):
                    ok = process_date_cam(db, "20260901", 0, skip_render=True, dashboard_enabled=False)
                    assert ok
        finally:
            db.close()

    def test_in_order_sliding_window_dispatch(self, tmp_path):
        """验证即使文件分析以乱序完成，渲染管理器仍按物理录制时间严格保序打包批次。"""
        db_path = tmp_path / "test_in_order.db"
        db = VlogDatabase(db_path=db_path)

        # 4 个起止时间互不相同的任务，物理时间顺序 f0 < f1 < f2 < f3
        task_defs = [
            ("cam0_20260901000000_20260901003000.mp4", "20260901000000", "20260901003000"),
            ("cam0_20260901003000_20260901010000.mp4", "20260901003000", "20260901010000"),
            ("cam0_20260901010000_20260901013000.mp4", "20260901010000", "20260901013000"),
            ("cam0_20260901013000_20260901020000.mp4", "20260901013000", "20260901020000"),
        ]
        f0, f1, f2, f3 = (t[0] for t in task_defs)
        file_start = {fname: start for fname, start, _ in task_defs}
        for fname, start_ts, end_ts in task_defs:
            db.add_file_task(fname, 0, "20260901", start_ts, end_ts, 1800.0)
            db.set_prescreen_result(fname, "STATIC")

        cfg = {
            "pipeline": {"render_start_delay": 0},
            "render": {"batch_max_files": 2},
        }
        orch = StreamingOrchestrator(
            db=db, date="20260901", cam_index=0, config=cfg, render_enabled=True, dashboard_enabled=False
        )

        # 乱序完成消息推入 render_batch_queue (f2 -> f0 -> f3 -> f1)
        for fname in (f2, f0, f3, f1):
            orch.render_batch_queue.put({"filepath": fname, "status": "STATIC"})
        orch.stop_event.set()

        # 用 spy 捕获每个批次实际派发的目标文件集合（来自 build_timeline_from_rows 的 target_files）
        from src.timeline import build_timeline_from_rows as _real_build
        dispatched_targets: list[list[str]] = []

        def _spy_build(rows, date, target_files=None, config=None):
            files = list(target_files or [])
            dispatched_targets.append(files)
            return _real_build(rows, date, target_files=files, config=config)

        try:
            with patch("src.timeline.build_timeline_from_rows", side_effect=_spy_build):
                with patch(
                    "src.pipeline.build_batch_render",
                    side_effect=lambda segs, bi, *args, **kwargs: f"mock_batch_{bi}.mp4",
                ):
                    # 运行 _render_manager 单一组件
                    t = threading.Thread(target=orch._render_manager)
                    t.start()
                    t.join(timeout=10.0)
            assert not t.is_alive(), "_render_manager did not terminate in time"

            # 恰好 2 个批次被派发
            assert len(dispatched_targets) == 2, f"unexpected dispatch: {dispatched_targets}"

            # batch 0 的目标文件集合必须是时序最早的 {f0, f1}，batch 1 为 {f2, f3}
            assert set(dispatched_targets[0]) == {f0, f1}
            assert set(dispatched_targets[1]) == {f2, f3}

            # 每个批次内部文件时序严格单调递增（无重复、无乱序）
            for batch_files in dispatched_targets:
                starts = [file_start[fp] for fp in batch_files]
                assert starts == sorted(starts), f"batch not in time order: {batch_files}"
                assert len(set(batch_files)) == len(batch_files), f"duplicate file in batch: {batch_files}"

            # 两个批次序号严格升序且成功产出
            orch.batch_paths.sort(key=lambda x: x[0])
            assert [b[0] for b in orch.batch_paths] == [0, 1]
            assert all("mock_batch_" in p.name for _, p in orch.batch_paths)
        finally:
            db.close()

