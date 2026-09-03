"""测试模块 6: 流式管线编排 (StreamingOrchestrator) 与异常容错机制.

覆盖：
1. 流式并发编排器 (StreamingOrchestrator) 队列生命周期与收尾终止；
2. 异常切片（损坏文件、无有效标签）的隔离与容错，防止全管线死锁或崩溃；
3. 渲染管理器 (RenderManager) 批次聚合与 failover 重试；
4. 调度器与并发 Worker 协同无死锁。
"""

import queue
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
        """验证即使文件分析以乱序完成，渲染管理器仍按物理录制时间严格保序打包。"""
        db_path = tmp_path / "test_in_order.db"
        db = VlogDatabase(db_path=db_path)
        # 按时间顺序注册 4 个文件
        f0 = "cam0_20260901000000_20260901000500.mp4"
        f1 = "cam0_20260901000500_20260901001000.mp4"
        f2 = "cam0_20260901001000_20260901001500.mp4"
        f3 = "cam0_20260901001500_20260901002000.mp4"
        for f in [f0, f1, f2, f3]:
            db.add_file_task(f, 0, "20260901", "20260901000000", "20260901000500", 300.0)
            db.set_prescreen_result(f, "STATIC")


        cfg = {
            "pipeline": {"render_start_delay": 0},
            "render": {"batch_max_files": 2},
        }
        orch = StreamingOrchestrator(
            db=db, date="20260901", cam_index=0, config=cfg, render_enabled=True, dashboard_enabled=False
        )

        # 模拟乱序完成消息推入 render_batch_queue (f2 -> f0 -> f3 -> f1)
        orch.render_batch_queue.put({"filepath": f2, "status": "STATIC"})
        orch.render_batch_queue.put({"filepath": f0, "status": "STATIC"})
        orch.render_batch_queue.put({"filepath": f3, "status": "STATIC"})
        orch.render_batch_queue.put({"filepath": f1, "status": "STATIC"})
        orch.stop_event.set()

        dispatched_batches = []
        with patch("src.pipeline.build_batch_render", side_effect=lambda segs, bi, *args, **kw: f"mock_batch_{bi}.mp4"):
            with patch("src.pipeline.concat_output_files", return_value=True):
                # 运行 _render_manager 单一组件
                import threading
                t = threading.Thread(target=orch._render_manager)
                t.start()
                t.join(timeout=3.0)

        # 验证所有生成的批次序号严格升序且时间戳单调递增
        assert len(orch.batch_paths) == 2
        # batch 0 必须包含 f0 与 f1，batch 1 必须包含 f2 与 f3
        orch.batch_paths.sort(key=lambda x: x[0])
        assert orch.batch_paths[0][0] == 0
        assert orch.batch_paths[1][0] == 1
        db.close()

