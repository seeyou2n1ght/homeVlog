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
from src.pipeline import AnalysisQueue


@pytest.mark.parametrize("with_labels", [False, True])
def test_analysis_result_survives_queue_metrics(tmp_path, with_labels):
    from types import SimpleNamespace
    from src.monitor import PerfCollector
    db = VlogDatabase(tmp_path / "analysis.db")
    try:
        db.add_file_task("clip.mp4", 0, "20260901", "20260901000000", "20260901000010", 10)
        orch = StreamingOrchestrator(db, "20260901", 0, {}, render_enabled=False, dashboard_enabled=False)
        labels = [{"time": float(t), "state": "DYNAMIC", "energy": 8, "raw_energy": 8} for t in range(11)] if with_labels else []
        detector = SimpleNamespace(analyze=lambda *a, **k: (labels, {}), decode_gpu="cpu", last_perf={}, fps=1)
        perf = PerfCollector()
        now = time.monotonic()
        orch._execute_analysis_task({"file_duration": 10, "_analysis_queued_at": now - 3}, "clip.mp4", 0,
                                    detector, None, "cpu", perf, now)
        msg = orch.render_batch_queue.get_nowait()
        assert msg["status"] == ("ANALYZED" if with_labels else "FAILED")
        assert perf._records[-1]["extra"]["analysis_queue_wait_s"] == pytest.approx(3)
    finally:
        db.close()


def test_analysis_queue_prioritizes_short_files_when_enabled():
    q = AnalysisQueue()
    q.cost_priority = True
    q.put({"filepath": "long.mp4", "file_duration": 3600.0, "file_start_time": "20260901010000"})
    q.put({"filepath": "short.mp4", "file_duration": 300.0, "file_start_time": "20260901000000"})
    assert q.get()["filepath"] == "short.mp4"
    q.task_done()
    assert q.get()["filepath"] == "long.mp4"
    q.task_done()


def test_companions_use_normalized_timeline_without_database_writes(tmp_path):
    import json
    from src.segment import Segment
    from src.pipeline import _save_vlog_companion_assets
    db = VlogDatabase(tmp_path / "companion.db")
    try:
        db.add_file_task("clip.mp4", 0, "20260901", "20260901000000", "20260901000020", 20)
        db.set_prescreen_result("clip.mp4", "SUSPICIOUS")
        db.set_analysis_result("clip.mp4", [Segment(0, .5, "STATIC", "clip.mp4"),
                                            Segment(0, 10, "STATIC", "clip.mp4"),
                                            Segment(10, 20, "DYNAMIC", "clip.mp4")])
        before = db.get_all_file_tasks_for_date("20260901", 0)
        out = tmp_path / "day.mp4"
        out.write_bytes(b"placeholder")
        with patch.object(db, "sync_timeline_segments") as sync, patch("src.ffmpeg.get_duration", return_value=11.5):
            _save_vlog_companion_assets(out, "20260901", 0, "camera", 1, 20, 1, db,
                                        {"render": {"generate_subtitles": False, "speed_ramping_enabled": False}})
            sync.assert_not_called()
        meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["timeline_highlights"][0]["vlog_start_s"] == 1.5
        assert db.get_all_file_tasks_for_date("20260901", 0) == before
    finally:
        db.close()


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

    def test_subtitle_wiring_after_render(self, tmp_path):
        """集成回归: 成片成功后必须生成 .srt 字幕（此前 rows 变量名错误导致全量静默失败）。"""
        db_path = tmp_path / "test_srt.db"
        db = VlogDatabase(db_path=db_path)
        fname = "00_20260901100000_20260901100500.mp4"
        db.add_file_task(fname, 0, "20260901", "20260901100000", "20260901100500", 300.0)
        db.set_prescreen_result(fname, "STATIC")

        fake_batch = tmp_path / "_batch0.mp4"
        fake_batch.write_bytes(b"0" * (600 * 1024))
        out_dir = tmp_path / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        mock_orch = MagicMock()
        mock_orch.run.return_value = [fake_batch]
        mock_orch.errors = []

        cfg_patch = {
            "render": {"generate_subtitles": True, "batch_max_files": 4},
            "output": {"naming": "DailyVlog_{date}_cam{index}.mp4", "fps": 20},
            "segment": {},
        }
        from src.render_cache import processing_fingerprint
        db.set_processing_fingerprint(fname, processing_fingerprint(fname, cfg_patch))
        try:
            with patch("src.pipeline.StreamingOrchestrator", return_value=mock_orch), \
                 patch("src.pipeline.load_config", return_value=cfg_patch), \
                 patch("src.pipeline.get_monitor"), \
                 patch("src.pipeline.OUTPUT_DIR", out_dir), \
                 patch("src.pipeline._dump_perf"), \
                 patch("src.pipeline.print_startup_banner"), \
                 patch("src.pipeline.print_summary_card"):
                ok = process_date_cam(db, "20260901", 0, skip_render=False, dashboard_enabled=False)

            assert ok is True
            assert (out_dir / "DailyVlog_20260901_cam0.mp4").exists()
            srt = out_dir / "DailyVlog_20260901_cam0.srt"
            assert srt.exists(), "SRT subtitle was not generated"
            content = srt.read_text(encoding="utf-8")
            assert "2026-09-01 10:00:00" in content
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

        def _spy_build(rows, date, target_files=None, config=None, **kwargs):
            files = list(target_files or [])
            if target_files is not None:
                dispatched_targets.append(files)
            return _real_build(rows, date, target_files=target_files, config=config, **kwargs)

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
            assert not orch._prefetched_files
        finally:
            db.close()

    def test_single_file_batches_render_ready_files_without_head_blocking(self, tmp_path):
        db = VlogDatabase(db_path=tmp_path / "immediate.db")
        files = [f"f{i}.mp4" for i in range(3)]
        for i, filepath in enumerate(files):
            db.add_file_task(
                filepath, 0, "20260901", f"202609010{i}0000",
                f"202609010{i}1000", 600.0,
            )
            db.set_prescreen_result(filepath, "STATIC")
        cfg = {
            "pipeline": {"render_start_delay": 0, "render_gpu_policy": "nv_only"},
            "render": {"batch_max_files": 1, "max_concurrency": 1},
        }
        orch = StreamingOrchestrator(db, "20260901", 0, cfg, dashboard_enabled=False)
        # f0 is deliberately absent: f2 must not wait for the chronological head.
        orch.render_batch_queue.put({"filepath": files[2], "status": "STATIC"})
        orch.render_batch_queue.put({"filepath": files[1], "status": "STATIC"})
        orch.stop_event.set()
        calls = []

        def render(_segs, batch_id, _gpu, *_args, **_kwargs):
            calls.append(batch_id)
            return str(tmp_path / f"batch{batch_id}.mp4")

        try:
            with patch("src.pipeline.build_batch_render", side_effect=render):
                orch._render_manager()
            assert calls == [2, 1]
            assert sorted(batch_id for batch_id, _ in orch.batch_paths) == [1, 2]
            assert not orch.errors
        finally:
            db.close()

    def test_collapsed_static_batch_reconciles_cleanly(self, tmp_path):
        """验证被时间轴宏观折叠跳过的静态素材批次能作为终端状态干净对账，不触发虚假 DYNAMIC 膨胀兜底。"""
        db_path = tmp_path / "test_collapsed.db"
        db = VlogDatabase(db_path=db_path)

        fname = "cam0_20260901010000_20260901011000.mp4"
        db.add_file_task(fname, 0, "20260901", "20260901010000", "20260901011000", 600.0)
        db.set_prescreen_result(fname, "SUSPICIOUS")
        db.set_analysis_result(fname, "ANALYZED", "[]")  # 分析确认纯静态

        cfg = {
            "pipeline": {"render_start_delay": 0, "prescreen_gpu_policy": "qsv_only"},
            "render": {"batch_max_files": 1},
            "detection": {"prescreen_parallel": 1, "analysis_max_workers": 1},
        }
        orch = StreamingOrchestrator(
            db=db, date="20260901", cam_index=0, config=cfg, render_enabled=True, dashboard_enabled=False
        )

        try:
            # 模拟 build_timeline_from_rows 因宏观折叠返回空列表
            with patch("src.timeline.build_timeline_from_rows", return_value=[]):
                orch.render_batch_queue.put({"filepath": fname, "status": "ANALYZED"})
                t = threading.Thread(target=orch._render_manager)
                t.start()
                orch.stop_event.set()
                t.join(timeout=10.0)
                assert not t.is_alive()

            # 验证：批次应作为终端状态对账闭环，没有报告错误或丢批
            assert len(orch.errors) == 0
            assert len(orch.batch_paths) == 0  # 折叠批次无需物理渲染
        finally:
            db.close()


    def test_light_batches_not_stranded_by_sentinels(self, tmp_path):
        """回归: 纯静态轻批次在哨兵投递后滞留 light 队列时不得被 stranded。

        场景: 8 个 STATIC 文件 → 4 个轻批次；2 个 NV worker 忙于前 2 批时
        收尾哨兵已投 heavy 队列。旧逻辑 worker 拿到哨兵即退出，batch 2/3
        被静默丢弃；修复后必须全部产出且无静默丢批告警。
        """
        db_path = tmp_path / "test_strand.db"
        db = VlogDatabase(db_path=db_path)

        files = []
        for i in range(8):
            start = f"20260901{i:02d}0000"   # 14 位: YYYYMMDDHHMMSS，按小时递增
            end = f"20260901{i:02d}0500"
            fname = f"cam0_{start}_{end}.mp4"
            db.add_file_task(fname, 0, "20260901", start, end, 300.0)
            db.set_prescreen_result(fname, "STATIC")
            files.append(fname)

        cfg = {
            "pipeline": {"render_start_delay": 0},
            "render": {"batch_max_files": 2},
            "scheduler": {"watermark_high": 10, "watermark_low": 3},
        }
        orch = StreamingOrchestrator(
            db=db, date="20260901", cam_index=0, config=cfg, render_enabled=True, dashboard_enabled=False
        )

        # 全部就绪消息预置（轻批次路径），随后立即停止分发循环
        for fname in files:
            orch.render_batch_queue.put({"filepath": fname, "status": "STATIC"})
        orch.stop_event.set()

        def _slow_render(segs, bi, *args, **kwargs):
            time.sleep(0.2)  # 保证哨兵在 worker 完成首批前已投递
            return f"mock_batch_{bi}.mp4"

        try:
            with patch("src.pipeline.build_batch_render", side_effect=_slow_render):
                t = threading.Thread(target=orch._render_manager)
                t.start()
                t.join(timeout=30.0)
            assert not t.is_alive(), "_render_manager did not terminate (possible deadlock)"

            orch.batch_paths.sort(key=lambda x: x[0])
            produced = [b[0] for b in orch.batch_paths]
            assert produced == [0, 1, 2, 3], f"light batches stranded: produced={produced}"
            assert not any("silently dropped" in e for e in orch.errors), orch.errors
        finally:
            db.close()

    def test_orchestrator_waits_for_render_finished_event(self, tmp_path):
        """验证 StreamingOrchestrator.run() 必须等待 render_finished_event，不因 render_batch_queue 瞬时清空提前退出。"""
        db_path = tmp_path / "test_wait_render.db"
        db = VlogDatabase(db_path=db_path)

        # 准备 4 个预筛为 STATIC 的任务
        for i in range(4):
            start = f"20260901{i:02d}0000"
            end = f"20260901{i:02d}0500"
            fname = f"cam0_{start}_{end}.mp4"
            db.add_file_task(fname, 0, "20260901", start, end, 300.0)
            db.set_prescreen_result(fname, "STATIC")

        cfg = {
            "pipeline": {"render_start_delay": 0, "prescreen_gpu_policy": "qsv_only"},
            "render": {"batch_max_files": 2},
            "detection": {"prescreen_parallel": 1, "analysis_max_workers": 1},
        }
        orch = StreamingOrchestrator(
            db=db, date="20260901", cam_index=0, config=cfg, render_enabled=True, dashboard_enabled=False
        )

        render_done_flag = [False]

        def _delayed_render(segs, bi, *args, **kwargs):
            time.sleep(0.3)
            render_done_flag[0] = True
            return f"mock_batch_{bi}.mp4"

        try:
            with patch("src.pipeline.build_batch_render", side_effect=_delayed_render):
                paths = orch.run()
                # 只有当 batch 真正被渲染完且设置 render_finished_event 后，run 才能返回
                assert render_done_flag[0] is True
                assert len(paths) == 2
                assert len(orch.batch_paths) == 2
        finally:
            db.close()


def test_prescreen_audio_gate_disabled_by_default(tmp_path):
    db = VlogDatabase(db_path=tmp_path / "audio_gate.db")
    filepath = "cam0_audio_gate_test.mp4"
    db.add_file_task(filepath, 0, "20260901", "20260901000000", "20260901001000", 600.0)

    cfg = {
        "pipeline": {"render_start_delay": 0},
        "audio_vad": {"prescreen_audio_gate": False},
        "detection": {"prescreen_parallel": 1, "analysis_max_workers": 1},
    }
    orch = StreamingOrchestrator(db=db, date="20260901", cam_index=0, config=cfg, render_enabled=False, dashboard_enabled=False)
    orch.prescreen_queue.put({"filepath": filepath, "file_duration": 600.0})
    orch.stop_event.set()

    with patch("src.pipeline.prescreen_file", return_value={"status": "STATIC", "has_audio": True, "result_json": "{}"}), \
         patch("src.pipeline.detect_audio_activity") as mock_vad:
            orch._prescreen_worker("cpu")
            mock_vad.assert_not_called()
            row = db.conn.execute("SELECT prescreen_status FROM file_tasks WHERE filepath = ?", (filepath,)).fetchone()
            assert row[0] == "STATIC"
    db.close()


def test_prescreen_audio_gate_enabled_wakes_up_suspicious(tmp_path):
    """验证当启用 prescreen_audio_gate 时，画面静态但有声音事件的文件被成功唤醒为 SUSPICIOUS。"""
    db = VlogDatabase(db_path=tmp_path / "audio_gate_enabled.db")
    filepath = "cam0_crying_baby_dark.mp4"
    db.add_file_task(filepath, 0, "20260901", "20260901000000", "20260901001000", 600.0)

    cfg = {
        "pipeline": {"render_start_delay": 0},
        "audio_vad": {"prescreen_audio_gate": True},
        "detection": {"prescreen_parallel": 1, "analysis_max_workers": 1},
    }
    orch = StreamingOrchestrator(db=db, date="20260901", cam_index=0, config=cfg, render_enabled=False, dashboard_enabled=False)
    orch.prescreen_queue.put({"filepath": filepath, "file_duration": 600.0})
    orch.stop_event.set()

    with patch("src.pipeline.prescreen_file", return_value={"status": "STATIC", "has_audio": True, "result_json": "{}"}), \
         patch("src.pipeline.detect_audio_activity", return_value=([(10.0, 15.0, -28.0)], {"active_ratio": 0.05})) as mock_vad:
            orch._prescreen_worker("cpu")
            mock_vad.assert_called_once()
            row = db.conn.execute("SELECT prescreen_status FROM file_tasks WHERE filepath = ?", (filepath,)).fetchone()
            assert row[0] == "SUSPICIOUS"
    db.close()
