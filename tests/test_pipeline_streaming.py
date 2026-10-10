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
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

import pytest

from src.database import VlogDatabase
from src.pipeline import StreamingOrchestrator, process_date_cam
from src.pipeline import AnalysisQueue


@pytest.mark.parametrize("outcome", ["failed", "exception", "cancelled"])
def test_final_concat_failure_and_cancellation_are_measured(tmp_path, outcome):
    from src.monitor import PerfCollector
    from src.render_cache import processing_fingerprint
    config = {"output": {"naming": "DailyVlog_{date}_cam{index}.mp4"}}
    db = VlogDatabase(tmp_path / "finalize.db")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    batches = [tmp_path / f"batch{i}.mp4" for i in range(2)]
    for batch in batches:
        batch.write_bytes(b"completed batch")
    db.add_file_task(str(source), 0, "20260901", "20260901000000", "20260901000100", 60)
    db.set_prescreen_result(str(source), "STATIC")
    db.set_processing_fingerprint(str(source), processing_fingerprint(source, config))
    orch = MagicMock()
    orch.run.return_value = batches
    orch.errors = []
    orch.render_worker_stats = {}
    perf = PerfCollector()
    def concat(*args, **kwargs):
        kwargs["timings"].update(ffmpeg_s=2, validation_s=3, failure_stage="checkpoint_open")
        if outcome == "exception":
            raise OSError("failed to open output")
        if outcome == "cancelled":
            raise KeyboardInterrupt
        return False
    try:
        with patch("src.pipeline.load_config", return_value=config), \
             patch("src.pipeline.StreamingOrchestrator", return_value=orch), \
             patch("src.pipeline.OUTPUT_DIR", tmp_path), \
             patch("src.pipeline.get_monitor"), \
             patch("src.pipeline.print_startup_banner"), \
             patch("src.pipeline.get_perf", return_value=perf), \
             patch("src.pipeline._dump_perf") as dump, \
             patch("src.pipeline.concat_output_files", side_effect=concat):
            if outcome == "cancelled":
                with pytest.raises(KeyboardInterrupt):
                    process_date_cam(db, "20260901", 0, dashboard_enabled=False)
            else:
                assert not process_date_cam(db, "20260901", 0, dashboard_enabled=False)
            dump.assert_called_once()
        import json
        perf.dump(tmp_path / "perf.json")
        records = [r for r in json.loads((tmp_path / "perf.json").read_text())["records"] if r["stage"] == "final_concat"]
        assert len(records) == 1
        assert records[0]["extra"]["status"] == ("cancelled" if outcome == "cancelled" else "failed")
        assert records[0]["extra"]["failure_stage"] == "checkpoint_open"
        assert records[0]["extra"]["ffmpeg_s"] == 2
        assert records[0]["extra"]["validation_s"] == 3
        assert all(batch.read_bytes() == b"completed batch" for batch in batches)
        assert not db.is_render_completed("20260901", 0)
    finally:
        db.close()


def test_prefetch_skips_timeline_without_static_segments(tmp_path):
    db = VlogDatabase(tmp_path / "prefetch.db")
    try:
        for name in ("active.mp4", "static.mp4"):
            db.add_file_task(name, 0, "20260901", "20260901000000", "20260901000010", 10)
        orch = StreamingOrchestrator(db, "20260901", 0, {}, dashboard_enabled=False)
        orch._prefetched_files.update(("active.mp4", "static.mp4"))
        for name in ("active.mp4", "static.mp4", None):
            orch._prefetch_queue.put(name)
        orch.stop_event.set()

        def timeline(_rows, _date, target_files, **_kwargs):
            state = "DYNAMIC" if target_files == ["active.mp4"] else "STATIC"
            return [SimpleNamespace(state=state)]

        with patch("src.stages.timeline.build_timeline_from_rows", side_effect=timeline):
            with patch("src.stages.renderer._stage_source_for_render") as stage:
                orch._lookahead_prefetch_worker()
        stage.assert_called_once()
        assert stage.call_args.args == ("static.mp4",)
    finally:
        db.close()


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


def test_cancelled_analysis_keeps_task_pending(tmp_path):
    from types import SimpleNamespace
    from src.monitor import PerfCollector

    db = VlogDatabase(tmp_path / "cancelled_analysis.db")
    try:
        db.add_file_task("clip.mp4", 0, "20260901", "20260901000000", "20260901000010", 10)
        orch = StreamingOrchestrator(db, "20260901", 0, {}, render_enabled=False, dashboard_enabled=False)

        def cancelled_decode(*args, **kwargs):
            orch.abort_event.set()
            return [], {}

        detector = SimpleNamespace(analyze=cancelled_decode, decode_gpu="cpu", last_perf={}, fps=1)
        orch._execute_analysis_task({"file_duration": 10}, "clip.mp4", 0,
                                    detector, None, "cpu", PerfCollector(), time.monotonic())
        assert db.get_file_task("clip.mp4")["analysis_status"] == "PENDING"
        assert db.get_file_task("clip.mp4")["retry_count"] == 0
        assert orch.render_batch_queue.empty()
    finally:
        db.close()


def test_cancelled_prescreen_keeps_task_pending(tmp_path):
    db = VlogDatabase(tmp_path / "cancelled_prescreen.db")
    try:
        db.add_file_task("clip.mp4", 0, "20260901", "20260901000000", "20260901000010", 10)
        orch = StreamingOrchestrator(db, "20260901", 0, {}, render_enabled=False, dashboard_enabled=False)
        orch.prescreen_queue.put({"filepath": "clip.mp4", "file_duration": 10})

        def cancelled_decode(*args, **kwargs):
            orch.abort_event.set()
            return {"status": "FAILED", "error": "interrupted"}

        with patch("src.pipeline.prescreen_file", side_effect=cancelled_decode):
            worker = threading.Thread(target=orch._prescreen_worker)
            worker.start()
            worker.join(timeout=2)
        assert not worker.is_alive()
        assert db.get_file_task("clip.mp4")["prescreen_status"] == "PENDING"
        assert orch.render_batch_queue.empty()
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
                                            Segment(10, 20, "DYNAMIC", "clip.mp4",
                                                    max_energy=23.4, avg_confidence=0.76)])
        before = db.get_all_file_tasks_for_date("20260901", 0)
        out = tmp_path / "day.mp4"
        out.write_bytes(b"placeholder")
        media_metrics = {}
        with patch.object(db, "sync_timeline_segments") as sync, patch("src.stages.companion.get_duration", return_value=12.2501) as probe:
            assert _save_vlog_companion_assets(
                out, "20260901", 0, "camera", 1, 20, 1, db,
                {"render": {"generate_subtitles": False, "speed_ramping_enabled": False}},
                media_metrics=media_metrics,
            )
            sync.assert_not_called()
            from src.stages.companion import build_headline
            headline = build_headline(out, 20, 1, output_duration_s=media_metrics["duration_s"])
            assert headline["output_dur_s"] == 12.3
            probe.assert_called_once()
        meta = json.loads(out.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["metrics"]["vlog_duration_s"] == 12.25
        assert meta["timeline_highlights"][0]["vlog_start_s"] == 1.5
        assert meta["timeline_highlights"][0]["max_energy"] == 23.4
        assert meta["timeline_highlights"][0]["avg_confidence"] == 0.76
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

    def test_empty_file_exclusion_is_verified_and_reversible(self, tmp_path, caplog):
        from src.pipeline import _reconcile_empty_files
        from src.stages.timeline import build_timeline_from_rows
        db = VlogDatabase(tmp_path / "empty.db")
        empty, broken, missing = [tmp_path / name for name in ("empty.mp4", "broken.mp4", "missing.mp4")]
        empty.touch()
        broken.write_bytes(b"not a video")
        try:
            for path in (empty, broken, missing):
                db.add_file_task(str(path), 0, "20260901", "20260901000000", "20260901000100", 60)
                db.set_prescreen_result(str(path), "FAILED")
            _reconcile_empty_files(db, "20260901", 0)
            rows = {Path(row["filepath"]).name: row for row in db.get_all_file_tasks_for_date("20260901", 0)}
            assert rows["empty.mp4"]["prescreen_status"] == "SKIPPED_EMPTY"
            assert rows["broken.mp4"]["prescreen_status"] == "FAILED"
            assert rows["missing.mp4"]["prescreen_status"] == "FAILED"
            assert "跳过空文件" in caplog.text and "20260901000100" in caplog.text
            assert build_timeline_from_rows([rows["empty.mp4"]], "20260901", config={}) == []
            empty.write_bytes(b"restored material")
            _reconcile_empty_files(db, "20260901", 0)
            assert db.get_all_file_tasks_for_date("20260901", 0)[0]["prescreen_status"] == "PENDING"
            empty.write_bytes(b"")
            _reconcile_empty_files(db, "20260901", 0)
            empty.unlink()
            _reconcile_empty_files(db, "20260901", 0)
            assert db.get_all_file_tasks_for_date("20260901", 0)[0]["prescreen_status"] == "FAILED"
        finally:
            db.close()

    def test_keyboard_interrupt_preserves_cancelled_telemetry(self, tmp_path):
        db = VlogDatabase(tmp_path / "cancel.db")
        try:
            with patch("src.pipeline.StreamingOrchestrator.run", side_effect=KeyboardInterrupt), \
                 patch("src.pipeline._dump_perf") as dump:
                with pytest.raises(KeyboardInterrupt):
                    process_date_cam(db, "20260320", 0, dashboard_enabled=False)
            dump.assert_called_once()
            assert dump.call_args.kwargs["run_context"]["status"] == "cancelled"
            assert not db.is_render_completed("20260320", 0)
        finally:
            db.close()

    @pytest.mark.parametrize("cancelled", [False, True])
    def test_render_no_output_distinguishes_cancellation(self, tmp_path, caplog, cancelled):
        db = VlogDatabase(tmp_path / "cancel_render.db")
        db.add_file_task("clip.mp4", 0, "20260320", "20260320000000", "20260320000100", 60)
        db.set_prescreen_result("clip.mp4", "STATIC")
        cfg = {"pipeline": {"render_start_delay": 0, "render_gpu_policy": "nv_only"},
               "render": {"batch_max_files": 1, "max_concurrency": 1, "local_staging_enabled": False}}
        orch = StreamingOrchestrator(db, "20260320", 0, cfg, dashboard_enabled=False)
        orch.render_batch_queue.put({"filepath": "clip.mp4", "status": "STATIC"})
        orch.stop_event.set()

        def render(*args, **kwargs):
            if cancelled:
                orch.abort_event.set()
            return None

        try:
            with patch("src.pipeline.build_batch_render", side_effect=render):
                orch._render_manager()
            assert not orch.batch_paths
            assert bool(orch.errors) is (not cancelled)
            assert "render reconciliation failed" not in caplog.text
        finally:
            db.close()

    @pytest.mark.parametrize("skip_render", [False, True])
    @pytest.mark.parametrize("new_material", [False, True])
    def test_skip_already_completed(self, tmp_path, caplog, skip_render, new_material):
        db_path = tmp_path / "test_skip.db"
        db = VlogDatabase(db_path=db_path)
        output = tmp_path / "out.mp4"
        output.write_bytes(b"delivered")
        output.replace(tmp_path / "backup.mp4")
        db.add_file_task("old.mp4", 0, "20260901", "20260901000000", "20260901000100", 60)
        db.set_prescreen_result("old.mp4", "STATIC")
        db.set_processing_fingerprint("old.mp4", "old-config-fingerprint")
        db.set_render_status("20260901", 0, "COMPLETED", output_file=str(output))
        db.close()
        db = VlogDatabase(db_path=db_path)  # Persist across process restarts.
        try:
            if new_material:
                db.add_file_task("new.mp4", 0, "20260901", "20260901000100", "20260901000200", 60)
            before = db.get_all_file_tasks_for_date("20260901", 0)
            with patch("src.pipeline.load_config", return_value={"detection": {"changed": True}}), \
                 patch.object(db, "invalidate_stale_results", wraps=db.invalidate_stale_results) as invalidate, \
                 patch("src.pipeline.StreamingOrchestrator") as orchestrator:
                assert process_date_cam(db, "20260901", 0, skip_render=skip_render, dashboard_enabled=False)
                invalidate.assert_not_called()
                orchestrator.assert_not_called()
            assert db.is_render_completed("20260901", 0)
            assert db.get_all_file_tasks_for_date("20260901", 0) == before
            assert not output.exists()
            assert ("待处理素材" in caplog.text) is new_material
        finally:
            db.close()

    @pytest.mark.parametrize("status,force", [("COMPLETED", True), ("FAILED", False), ("PENDING", False)])
    def test_explicit_rebuild_or_unfinished_day_runs(self, tmp_path, status, force):
        db = VlogDatabase(tmp_path / "rebuild.db")
        db.set_render_status("20260901", 0, status)
        try:
            with patch("src.pipeline.StreamingOrchestrator") as orchestrator, \
                 patch("src.pipeline.load_config", return_value={}), \
                 patch("src.pipeline.get_monitor"), \
                 patch("src.pipeline.print_startup_banner"):
                orchestrator.return_value.run.return_value = []
                orchestrator.return_value.errors = []
                assert not process_date_cam(db, "20260901", 0, force_render=force, dashboard_enabled=False)
                orchestrator.return_value.run.assert_called_once()
            assert not db.is_render_completed("20260901", 0)
        finally:
            db.close()

    @pytest.mark.parametrize("empty_source", [False, True])
    def test_subtitle_wiring_after_render(self, tmp_path, empty_source, caplog):
        """集成回归: 成片成功后必须生成 .srt 字幕（此前 rows 变量名错误导致全量静默失败）。"""
        db_path = tmp_path / "test_srt.db"
        db = VlogDatabase(db_path=db_path)
        fname = str(tmp_path / "00_20260901100000_20260901100500.mp4")
        db.add_file_task(fname, 0, "20260901", "20260901100000", "20260901100500", 300.0)
        db.set_prescreen_result(fname, "STATIC")
        if empty_source:
            empty = tmp_path / "empty.mp4"
            empty.touch()
            db.add_file_task(str(empty), 0, "20260901", "20260901100500", "20260901100600", 60)
            db.set_prescreen_result(str(empty), "FAILED")

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
                 patch("src.core.utils.cleanup_staging_files"), \
                 patch("src.stages.companion.get_duration", return_value=300.0) as probe, \
                 patch("src.pipeline.print_startup_banner"), \
                 patch("src.pipeline.print_summary_card"):
                ok = process_date_cam(db, "20260901", 0, skip_render=False, dashboard_enabled=False)
                probe.assert_called_once()

            assert ok is True
            assert (out_dir / "DailyVlog_20260901_cam0.mp4").exists()
            srt = out_dir / "DailyVlog_20260901_cam0.srt"
            assert srt.exists(), "SRT subtitle was not generated"
            content = srt.read_text(encoding="utf-8")
            assert "2026-09-01 10:00:00" in content
            assert db.is_render_completed("20260901", 0)
            if empty_source:
                import json
                metadata = json.loads((out_dir / "DailyVlog_20260901_cam0.meta.json").read_text(encoding="utf-8"))
                assert metadata["skipped_sources"] == [{
                    "source_file": "empty.mp4", "reason": "empty_file", "size_bytes": 0,
                    "source_start": "20260901100500", "source_end": "20260901100600",
                    "nominal_duration_s": 60.0,
                }]
                assert "跳过空文件" in caplog.text
                assert "2026-09-01 10:05:00" not in content
        finally:
            db.close()

    @pytest.mark.parametrize("batch_max_files", [1, 2])
    def test_empty_source_advances_render_queue_without_rendering(self, tmp_path, batch_max_files):
        db = VlogDatabase(tmp_path / "empty_queue.db")
        files = [str(tmp_path / f"clip{i}.mp4") for i in range(3)]
        try:
            for i, filepath in enumerate(files):
                db.add_file_task(filepath, 0, "20260901", f"20260901000{i}00", f"20260901000{i+1}00", 60)
                db.set_prescreen_result(filepath, "SKIPPED_EMPTY" if i == 1 else "STATIC")
            orch = StreamingOrchestrator(db, "20260901", 0, {
                "pipeline": {"render_start_delay": 0, "render_gpu_policy": "nv_only"},
                "render": {"batch_max_files": batch_max_files, "local_staging_enabled": False},
            }, dashboard_enabled=False)
            for i in (2, 0, 1):
                orch.render_batch_queue.put({"filepath": files[i], "status": "SKIPPED_EMPTY" if i == 1 else "STATIC"})
            orch.stop_event.set()
            rendered = set()
            def render(segments, batch, *args, **kwargs):
                rendered.update(segment.filepath for segment in segments)
                return str(tmp_path / f"batch{batch}.mp4")
            thread = threading.Thread(target=orch._render_manager, daemon=True)
            with patch("src.pipeline.build_batch_render", side_effect=render):
                thread.start()
                try:
                    thread.join(timeout=10)
                    assert not thread.is_alive()
                    assert not orch.errors
                    assert rendered == {files[0], files[2]}
                finally:
                    orch.abort_event.set()
                    thread.join(timeout=5)
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

    def test_qsv_failure_is_retried_by_a_live_nv_worker(self, tmp_path):
        from src.hardware.scheduler import DualEndedBatchQueue

        db = VlogDatabase(tmp_path / "qsv_retry.db")
        filepath = "static.mp4"
        db.add_file_task(filepath, 0, "20260901", "20260901000000", "20260901000100", 60)
        db.set_prescreen_result(filepath, "STATIC")
        cfg = {
            "pipeline": {"render_start_delay": 0, "render_gpu_policy": "heterogeneous"},
            "render": {"batch_max_files": 1, "max_concurrency": 2, "local_staging_enabled": False},
        }
        orch = StreamingOrchestrator(db, "20260901", 0, cfg, dashboard_enabled=False)
        orch.render_batch_queue.put({"filepath": filepath, "status": "STATIC"})
        orch.stop_event.set()
        calls = []
        qsv_attempted = threading.Event()

        def wait_for_qsv(*_args, **_kwargs):
            # NV may legitimately steal the initial static job. Hold its empty
            # heavy-queue poll until QSV owns this test's failure injection.
            assert qsv_attempted.wait(5), "QSV did not receive the test batch"
            return None

        def render(_segments, _batch, gpu, *_args, **_kwargs):
            calls.append(gpu)
            if gpu == "qsv":
                qsv_attempted.set()
            return None if gpu == "qsv" else str(tmp_path / "nv_retry.mp4")

        try:
            with patch("src.pipeline.build_batch_render", side_effect=render), \
                    patch.object(DualEndedBatchQueue, "pop_heaviest", side_effect=wait_for_qsv):
                orch._render_manager()
            assert calls == ["qsv", "nv"]
            assert [batch for batch, _ in orch.batch_paths] == [0]
            assert not orch.errors
        finally:
            db.close()

    @pytest.mark.parametrize("failure", ["no_output", "exception"])
    def test_qsv_only_failure_finishes_with_error(self, tmp_path, failure):
        db = VlogDatabase(tmp_path / "qsv_only_failure.db")
        filepath = "static.mp4"
        db.add_file_task(filepath, 0, "20260901", "20260901000000", "20260901000100", 60)
        db.set_prescreen_result(filepath, "STATIC")
        cfg = {
            "pipeline": {"render_start_delay": 0, "render_gpu_policy": "qsv_only"},
            "render": {"batch_max_files": 1, "max_concurrency": 1, "local_staging_enabled": False},
        }
        orch = StreamingOrchestrator(db, "20260901", 0, cfg, dashboard_enabled=False)
        orch.render_batch_queue.put({"filepath": filepath, "status": "STATIC"})
        orch.stop_event.set()
        thread = threading.Thread(target=orch._render_manager, daemon=True)
        expected_error = (
            "render batch 0 returned no output" if failure == "no_output"
            else "render batch 0 failed on qsv"
        )

        try:
            effect = RuntimeError("controlled QSV failure") if failure == "exception" else None
            with patch("src.pipeline.build_batch_render", side_effect=effect, return_value=None) as render:
                thread.start()
                try:
                    thread.join(timeout=5.0)
                    assert not thread.is_alive(), "QSV-only failure stranded a retry without an NV worker"
                    assert orch.render_finished_event.is_set()
                    assert orch.errors == [expected_error]
                    assert not orch.batch_paths
                    render.assert_called_once()
                    assert render.call_args.args[2] == "qsv"
                finally:
                    orch.abort_event.set()
                    thread.join(timeout=5.0)
        finally:
            db.close()

    @pytest.mark.parametrize("failure", ["no_output", "exception"])
    def test_nv_transient_failure_is_retried_by_live_nv_worker(self, tmp_path, failure):
        db = VlogDatabase(tmp_path / "nv_retry.db")
        filepath = "heavy.mp4"
        db.add_file_task(filepath, 0, "20260901", "20260901000000", "20260901000100", 60)
        db.set_prescreen_result(filepath, "STATIC")
        cfg = {
            "pipeline": {"render_start_delay": 0, "render_gpu_policy": "nv_only"},
            "render": {"batch_max_files": 1, "max_concurrency": 1, "local_staging_enabled": False},
        }
        orch = StreamingOrchestrator(db, "20260901", 0, cfg, dashboard_enabled=False)
        orch.render_batch_queue.put({"filepath": filepath, "status": "STATIC"})
        orch.stop_event.set()
        calls = []

        def render(_segments, _batch, gpu, *_args, **_kwargs):
            calls.append(gpu)
            if len(calls) == 1:
                if failure == "exception":
                    raise RuntimeError("controlled NV failure")
                return None  # First attempt fails transiently
            return str(tmp_path / "nv_success.mp4")

        try:
            with patch("src.pipeline.build_batch_render", side_effect=render):
                orch._render_manager()
            assert calls == ["nv", "nv"]
            assert [batch for batch, _ in orch.batch_paths] == [0]
            assert not orch.errors
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


@pytest.mark.parametrize("events", [[], [(10.0, 15.0, -28.0)]])
def test_prescreen_audio_gate_enabled_wakes_up_suspicious(tmp_path, events):
    """验证当启用 prescreen_audio_gate 时，画面静态但有声音事件的文件被成功唤醒为 SUSPICIOUS。"""
    import json
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

    with patch("src.pipeline.prescreen_file", return_value={"status": "STATIC", "has_audio": True, "result_json": '{"visual_score": 0.01}'}), \
         patch("src.pipeline.detect_audio_activity", return_value=(events, {"active_ratio": 0.05})) as mock_vad:
            orch._prescreen_worker("cpu")
            mock_vad.assert_called_once()
            row = db.get_all_file_tasks_for_date("20260901", 0)[0]
            assert row["prescreen_status"] == ("SUSPICIOUS" if events else "STATIC")
            evidence = json.loads(row["prescreen_result"])
            assert evidence["visual_result"] == {"visual_score": 0.01}
            assert evidence["audio_events"] == [list(e) for e in events]
    db.close()


def test_batch_disk_stop_preserves_delivery_and_reports_remaining(tmp_path, monkeypatch):
    import src.pipeline as pipeline
    db = VlogDatabase(tmp_path / "stop.db")
    for date in ("20260901", "20260902"):
        db.add_file_task(date + ".mp4", 0, date, date + "000000", date + "000100", 60)
    db.set_render_status("20260901", 0, "COMPLETED", output_file="offline.mp4")
    monkeypatch.setattr(pipeline, "VlogDatabase", lambda: db)
    monkeypatch.setattr(pipeline, "scan_directory", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "get_monitor", MagicMock())
    monkeypatch.setattr(pipeline, "OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(pipeline, "cleanup_resources", lambda: None)
    monkeypatch.setattr("src.core.utils.cleanup_temp_artifacts", lambda: None)
    guard = MagicMock(return_value=False)
    monkeypatch.setattr(pipeline, "check_disk_space", guard)
    result = pipeline.run_pipeline(dashboard_enabled=False)
    assert result == {"total": 2, "ok": 0, "failed": 0, "skipped": 1,
                      "remaining": 1, "stopped_reason": "LOW_DISK_SPACE"}
    guard.assert_called_once_with(tmp_path)  # configured threshold, no hard-coded override


@pytest.mark.parametrize("state", ["PRESENCE", "NIGHT_STATIONARY", "MICRO_MOTION"])
def test_cached_active_states_keep_fresh_queue_weight(tmp_path, state):
    from src.algorithms.segment import Segment, segments_to_json
    from src.hardware.scheduler import DualEndedBatchQueue
    db = VlogDatabase(tmp_path / "weights.db")
    fp = "night.mp4"
    db.add_file_task(fp, 0, "20260901", "20260901000000", "20260901000100", 60)
    db.set_prescreen_result(fp, "SUSPICIOUS")
    db.set_analysis_result(fp, "ANALYZED", segments_to_json([Segment(0, 60, state, fp)]))
    config = {"pipeline": {"render_start_delay": 0, "render_gpu_policy": "nv_only"},
              "render": {"batch_max_files": 1, "max_concurrency": 1, "local_staging_enabled": False}}
    orch = StreamingOrchestrator(db, "20260901", 0, config, dashboard_enabled=False)
    orch.render_batch_queue.put({"filepath": fp, "status": "ANALYZED"})
    orch.stop_event.set()
    weights = []
    original = DualEndedBatchQueue.put
    def observe(self, batch, files, weight):
        weights.append(weight)
        return original(self, batch, files, weight)
    try:
        with patch.object(DualEndedBatchQueue, "put", observe), patch("src.pipeline.build_batch_render", return_value=str(tmp_path / "batch.mp4")):
            orch._render_manager()
        assert weights == [60]
        assert len(orch.batch_paths) == 1 and not orch.errors
    finally:
        db.close()


@pytest.mark.parametrize("failure", ["probe", "commit"])
def test_companion_failure_does_not_publish_reusable_media_metrics(tmp_path, failure):
    from src.stages.companion import save_vlog_companion_assets
    out = tmp_path / "day.mp4"
    out.write_bytes(b"placeholder")
    media_metrics = {}
    with patch("src.stages.companion.get_duration", return_value=0 if failure == "probe" else 12.2501), \
         patch("src.stages.companion.build_timeline_from_rows", return_value=[]), \
         patch("pathlib.Path.replace", side_effect=OSError("commit failed")):
        assert not save_vlog_companion_assets(
            out, "20260901", 0, "camera", 1, 20, 1, MagicMock(),
            {"render": {"generate_subtitles": False}}, rows=[], media_metrics=media_metrics,
        )
    assert media_metrics == {}


def test_headline_standalone_probe_and_failure_behavior(tmp_path):
    from src.stages.companion import build_headline
    out = tmp_path / "day.mp4"
    out.write_bytes(b"placeholder")
    with patch("src.stages.companion.get_duration", return_value=12.2501) as probe:
        assert build_headline(out, 20, 1)["output_dur_s"] == 12.3
        probe.assert_called_once()
    with patch("src.stages.companion.get_duration", side_effect=OSError("probe failed")):
        assert "output_dur_s" not in build_headline(out, 20, 1)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("cleanup", [False, True])
@pytest.mark.parametrize("failure", ["assets_false", "assets_exception", "promotion", "manifest", "database", "cancelled", "after_database_cancelled", "none"])
def test_delivery_commits_before_cleanup_and_preserves_previous_assets(delivery_workspace, monkeypatch, count, cleanup, failure):
    w = delivery_workspace
    pipeline = w.pipeline
    w.orch.run.return_value = w.batches[:count]
    w.config["render"]["cleanup_batches_on_success"] = cleanup
    unrelated = w.output.parent / "unrelated_batch.mp4"
    unrelated.write_bytes(b"preserve unrelated")
    if failure.startswith("assets"):
        replacement = MagicMock(return_value=False) if failure == "assets_false" else MagicMock(side_effect=OSError("assets exception"))
        monkeypatch.setattr(pipeline, "_save_vlog_companion_assets", replacement)
    elif failure in ("manifest", "cancelled"):
        def manifest_failure(path, fingerprint):
            from src.hardware.render_cache import _manifest_path
            _manifest_path(path).write_bytes(b"partially written manifest")
            raise KeyboardInterrupt if failure == "cancelled" else OSError("manifest failure")
        monkeypatch.setattr("src.hardware.render_cache.save_manifest", manifest_failure)
    elif failure == "database":
        w.db.conn.execute("CREATE TRIGGER reject_completed BEFORE INSERT ON render_tasks WHEN NEW.status='COMPLETED' BEGIN SELECT RAISE(ABORT, 'completion rejected'); END")
        w.db.conn.commit()
    elif failure == "promotion":
        replace = Path.replace
        def reject_metadata(source, destination):
            if source.name.endswith(".pending.meta.json"):
                raise OSError("partial promotion failure")
            return replace(source, destination)
        monkeypatch.setattr(Path, "replace", reject_metadata)
    elif failure == "after_database_cancelled":
        completed = w.db.is_render_completed
        def interrupt_completed_check(*args):
            if completed(*args):
                raise KeyboardInterrupt
            return False
        monkeypatch.setattr(w.db, "is_render_completed", interrupt_completed_check)
    if failure.endswith("cancelled"):
        with pytest.raises(KeyboardInterrupt):
            pipeline.process_date_cam(w.db, "20260901", 0, dashboard_enabled=False)
    else:
        assert pipeline.process_date_cam(w.db, "20260901", 0, dashboard_enabled=False) is (failure == "none")
    if failure != "none":
        assert {path: path.read_bytes() for path in w.original} == w.original
        assert not w.db.is_render_completed("20260901", 0)
    else:
        assert w.db.is_render_completed("20260901", 0)
        assert w.output.read_bytes() == b"new media"
        assert not w.output.with_suffix(".srt").exists()  # disabled subtitles remove the previous SRT only on commit
        from src.hardware.render_cache import _manifest_path, file_identity
        import json
        assert json.loads(_manifest_path(w.output).read_text())["output"] == file_identity(w.output)
    for index, batch in enumerate(w.batches):
        retained = failure != "none" or not cleanup or index >= count
        assert batch.exists() is retained
        assert batch.with_name(batch.name + ".json").exists() is retained
    assert unrelated.read_bytes() == b"preserve unrelated"
    assert not list(w.output.parent.glob("*.pending.*"))
    assert not list(w.output.parent.glob(".delivery_backup_*"))
    pipeline._dump_perf.assert_called_once()


def test_delivery_rollback_failure_retains_previous_files_and_recovery_map(delivery_workspace, monkeypatch):
    from src.stages.companion import promote_vlog_assets
    import json
    w = delivery_workspace
    candidate = w.output.with_name(w.output.stem + ".pending.mp4")
    candidate.write_bytes(b"new media")
    candidate.with_suffix(".meta.json").write_text("{}")
    replace = Path.replace
    def reject_restore(source, destination):
        if source.name == "0" and source.parent.name.startswith(".delivery_backup_"):
            raise OSError("restore unavailable")
        return replace(source, destination)
    monkeypatch.setattr(Path, "replace", reject_restore)
    with pytest.raises(OSError, match="restore unavailable"):
        with promote_vlog_assets(candidate, w.output, w.config):
            raise OSError("commit failed")
    backups = list(w.output.parent.glob(".delivery_backup_*"))
    assert len(backups) == 1
    recovery = json.loads((backups[0] / "recovery.json").read_text())
    for index, path in enumerate(w.original):
        assert recovery[str(index)] == str(path.resolve())
        assert (backups[0] / str(index)).read_bytes() == w.original[path]


@pytest.mark.parametrize("status", ["STATIC", "FAILED", "PENDING", "SUSPICIOUS"])
def test_analysis_only_reports_source_outcome_and_dumps_perf(delivery_workspace, status):
    w = delivery_workspace
    def analyze():
        w.db.set_prescreen_result(str(w.source), status)
        return []
    w.orch.run.side_effect = analyze
    assert w.pipeline.process_date_cam(w.db, "20260901", 0, skip_render=True, dashboard_enabled=False) is (status == "STATIC")
    assert not w.db.is_render_completed("20260901", 0)
    assert {path: path.read_bytes() for path in w.original} == w.original
    w.pipeline._dump_perf.assert_called_once()
    assert w.pipeline._dump_perf.call_args.kwargs["run_context"]["status"] == ("analysis_completed" if status == "STATIC" else "failed")


def test_delivery_rejects_unavailable_source_reconciliation(delivery_workspace, monkeypatch):
    w = delivery_workspace
    def lose_source_query():
        monkeypatch.setattr(w.db, "get_all_file_tasks_for_date", lambda *a, **k: [])
        return w.batches
    w.orch.run.side_effect = lose_source_query
    assert not w.pipeline.process_date_cam(w.db, "20260901", 0, dashboard_enabled=False)
    assert {path: path.read_bytes() for path in w.original} == w.original
    assert all(batch.exists() for batch in w.batches)
    assert w.pipeline._dump_perf.call_args.kwargs["run_context"]["error"] == "source task reconciliation unavailable"


@pytest.mark.parametrize("status", ["SKIPPED_EXCLUDED", "SKIPPED_CORRUPT"])
def test_delivery_accepts_controlled_skipped_sources(delivery_workspace, monkeypatch, status):
    w = delivery_workspace
    corrupt_fp = str(w.batches[0])
    if status == "SKIPPED_EXCLUDED":
        w.db.exclude_file_task(corrupt_fp, reason="manual_test_exclusion")
    else:
        w.db.set_analysis_result(corrupt_fp, "SKIPPED_CORRUPT", "", error_msg="unreadable_stream")
    for b in w.batches[1:]:
        w.db.set_prescreen_result(str(b), "SUSPICIOUS")
        w.db.set_analysis_result(str(b), "ANALYZED", "[]")
    w.orch.run.return_value = w.batches[1:]
    ok = w.pipeline.process_date_cam(w.db, "20260901", 0, dashboard_enabled=False)
    assert ok


@pytest.mark.parametrize("count", [1, 2])
def test_real_software_delivery_commits_media_subtitles_and_manifest(tmp_path, monkeypatch, count):
    import av
    import json
    from src import pipeline
    from src.segment import Segment
    from src.hardware.ffmpeg import run_ffmpeg
    from src.hardware.render_cache import processing_fingerprint, file_identity, _manifest_path, valid_video
    cfg = {"render": {"generate_subtitles": True},
           "output": {"fps": 20, "naming": "DailyVlog_{date}_cam{index}.mp4", "audio": {"codec": "aac"}}}
    db = VlogDatabase(tmp_path / "media.db")
    batches = []
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    try:
        for i in range(count):
            batch = tmp_path / f"batch{i}.mp4"
            encoded = run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=128x128:rate=20",
                                  "-f", "lavfi", "-i", f"sine=frequency={440 + i * 100}:sample_rate=48000",
                                  "-t", "1", "-c:v", "libx264", "-c:a", "aac", str(batch)], timeout=30)
            assert encoded.returncode == 0
            batches.append(batch)
            db.add_file_task(str(batch), 0, "20260901", f"2026090100000{i}", f"2026090100000{i + 1}", 1)
            db.set_prescreen_result(str(batch), "SUSPICIOUS")
            db.set_analysis_result(str(batch), [Segment(i, i + 1, "DYNAMIC_AUDIO", str(batch), file_start_offset=i)])
            db.set_processing_fingerprint(str(batch), processing_fingerprint(batch, cfg))
        orch = MagicMock()
        orch.errors = []
        orch.run.return_value = batches
        monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
        monkeypatch.setattr("src.stages.timeline.load_config", lambda: cfg)
        monkeypatch.setattr(pipeline, "StreamingOrchestrator", MagicMock(return_value=orch))
        monkeypatch.setattr(pipeline, "OUTPUT_DIR", out_dir)
        for name in ("get_monitor", "print_startup_banner", "print_summary_card", "_dump_perf"):
            monkeypatch.setattr(pipeline, name, MagicMock())
        monkeypatch.setattr("src.core.utils.cleanup_staging_files", lambda: None)
        assert pipeline.process_date_cam(db, "20260901", 0, dashboard_enabled=False)
        output = out_dir / "DailyVlog_20260901_cam0.mp4"
        assert valid_video(output, count, checkpoints=[1] if count == 2 else (), require_audio=True)
        with av.open(str(output)) as media:
            assert len(list(media.decode(video=0))) == count * 20
        with av.open(str(output)) as media:
            for stream in (media.streams.video[0], media.streams.audio[0]):
                assert float(stream.duration * stream.time_base) == pytest.approx(count, abs=.001)
        metadata = json.loads(output.with_suffix(".meta.json").read_text())
        # ffprobe format duration includes AAC container padding; stream clocks remain exact.
        assert metadata["metrics"]["vlog_duration_s"] == pytest.approx(count, abs=1 / 20)
        assert metadata["timeline_highlights"][-1]["vlog_end_s"] == count
        assert f"00:00:0{count},000" in output.with_suffix(".srt").read_text()
        assert json.loads(_manifest_path(output).read_text())["output"] == file_identity(output)
        assert db.is_render_completed("20260901", 0)
        assert all(batch.exists() for batch in batches)
    finally:
        db.close()
