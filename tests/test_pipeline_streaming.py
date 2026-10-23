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
from unittest.mock import patch, MagicMock, call

import pytest

from src.database import VlogDatabase
from src.pipeline import StreamingOrchestrator, process_date_cam
from src.pipeline import AnalysisQueue


@pytest.mark.parametrize('stage',['prescreen','analysis','audio_gate'])
@pytest.mark.parametrize('render_enabled',[False,True])
@pytest.mark.parametrize('persist_failure',[False,True])
def test_unconfirmed_source_reader_stops_day_before_next_source(delivery_workspace, monkeypatch, stage, render_enabled, persist_failure):
    import sqlite3
    from src.hardware.ffmpeg import FFmpegProcessRegistry,MediaTerminationUnconfirmed
    from src.hardware.render_cache import processing_fingerprint
    w=delivery_workspace
    w.config.update(pipeline={'render_start_delay':0,'prescreen_gpu_policy':'qsv_only','analysis_gpu_policy':'qsv_only'},
        detection={'prescreen_parallel':1,'analysis_max_workers':1},yolo={'enabled':False},
        audio_vad={'prescreen_audio_gate':stage=='audio_gate'})
    second=w.source.with_name('second.mp4');second.write_bytes(b'controlled second source')
    w.db.add_file_task(str(second),0,'20260901','20260901000100','20260901000200',60)
    for source in (w.source,second):
        w.db.set_prescreen_result(str(source),'SUSPICIOUS' if stage=='analysis' else 'PENDING')
        w.db.set_processing_fingerprint(str(source),processing_fingerprint(source,w.config))
    objects=[];calls=[]
    def factory(*args,**kwargs):
        instance=StreamingOrchestrator(*args,**kwargs);objects.append(instance);return instance
    def fail(*args,**kwargs):
        calls.append(args[0]['filepath'] if stage=='analysis' else args[0])
        raise MediaTerminationUnconfirmed('controlled source reader still alive')
    monkeypatch.setattr(w.pipeline,'StreamingOrchestrator',factory)
    if stage=='prescreen':
        monkeypatch.setattr(w.pipeline,'prescreen_file',fail)
    elif stage=='audio_gate':
        monkeypatch.setattr(w.pipeline,'prescreen_file',lambda *a,**k:dict(status='STATIC',has_audio=1))
        monkeypatch.setattr(w.pipeline,'detect_audio_activity',fail)
    else:
        monkeypatch.setattr(StreamingOrchestrator,'_execute_analysis_task',lambda self,*a,**k:fail(*a,**k))
    if persist_failure:
        setter='set_analysis_result' if stage=='analysis' else 'set_prescreen_result'
        original=getattr(w.db,setter)
        def cannot_persist(filepath,status,*args,**kwargs):
            if status=='FAILED':
                raise sqlite3.OperationalError('controlled FAILED persistence failure')
            return original(filepath,status,*args,**kwargs)
        monkeypatch.setattr(w.db,setter,cannot_persist)
    concat=MagicMock(side_effect=AssertionError('Source with unknown stop reached delivery'))
    monkeypatch.setattr(w.pipeline,'concat_output_files',concat)
    with patch.object(FFmpegProcessRegistry,'mark_interrupted') as mark,patch.object(FFmpegProcessRegistry,'kill_all') as stop:
        with pytest.raises(MediaTerminationUnconfirmed):
            process_date_cam(w.db,'20260901',0,dashboard_enabled=False,skip_render=not render_enabled)
        mark.assert_called_once()
        stop.assert_called()
    assert calls==[str(w.source)]
    assert objects[0].abort_event.is_set() and objects[0].stop_event.is_set() and objects[0]._media_cleanup_failure
    concat.assert_not_called()
    if render_enabled:
        assert w.db.conn.execute('SELECT status FROM render_tasks').fetchone()[0]=='FAILED'
    status_key='analysis_status' if stage=='analysis' else 'prescreen_status'
    if persist_failure:
        assert any('FAILED persistence failed' in error for error in objects[0].errors)
        assert w.db.get_file_task(str(w.source))[status_key]!='FAILED'
    else:
        assert w.db.get_file_task(str(w.source))[status_key]=='FAILED'
    assert w.db.get_file_task(str(second))[status_key]=='PENDING'
    assert {p:p.read_bytes() for p in w.original}==w.original


@pytest.mark.parametrize('policy', ['nv_only', 'qsv_only', 'heterogeneous'])
def test_unconfirmed_render_stop_aborts_without_hardware_retry(tmp_path, monkeypatch, policy):
    from src.stages.renderer import RenderTerminationUnconfirmed
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    from src.hardware.scheduler import DualEndedBatchQueue
    db=VlogDatabase(tmp_path/'unconfirmed.sqlite')
    db.add_file_task('source.mp4',0,'20260901','20260901000000','20260901000100',60)
    db.set_prescreen_result('source.mp4','STATIC')
    config={'pipeline':{'render_start_delay':0,'render_gpu_policy':policy},
            'render':{'batch_max_files':1,'max_concurrency':1,'local_staging_enabled':False}}
    orch=StreamingOrchestrator(db,'20260901',0,config,dashboard_enabled=False)
    completed=tmp_path/'completed.mp4';completed.write_bytes(b'previous completed batch')
    orch.batch_paths=[(-1,str(completed))]
    orch.render_batch_queue.put({'filepath':'source.mp4','status':'STATIC'})
    orch.stop_event.set()
    calls=[]
    def render(*args,**kwargs):
        calls.append(args[2])
        raise RenderTerminationUnconfirmed('controlled unresolved render reader')
    # Force the initial light batch to QSV under heterogeneous policy, then
    # reject any attempted failover. This is an orchestrator contract test.
    if policy=='heterogeneous':
        monkeypatch.setattr(DualEndedBatchQueue,'pop_heaviest',lambda *args,**kwargs: None)
    monkeypatch.setattr('src.pipeline.build_batch_render',render)
    with patch.object(FFmpegProcessRegistry,'mark_interrupted') as mark, patch.object(FFmpegProcessRegistry,'kill_all') as stop:
        worker=threading.Thread(target=orch._render_manager,daemon=True)
        try:
            worker.start();worker.join(5)
            assert not worker.is_alive()
            assert len(calls)==1
            assert orch.abort_event.is_set() and orch.render_finished_event.is_set()
            assert len(orch.errors)==1 and orch.errors[0].startswith('render batch 0 cleanup unconfirmed:')
            assert orch._media_cleanup_failure
            assert 'controlled unresolved render reader' in orch.errors[0]
            assert orch._render_interruption_reason is None  # failure, not external cancellation
            assert orch.batch_paths==[(-1,str(completed))]
            assert completed.read_bytes()==b'previous completed batch'
            mark.assert_called_once();stop.assert_called_once()
            assert db.conn.execute('SELECT status FROM render_tasks').fetchone() is None
        finally:
            orch.abort_event.set();worker.join(5);db.close()


def test_unconfirmed_stop_prevents_delivery_commit(delivery_workspace, monkeypatch):
    w=delivery_workspace
    w.orch.errors=['render batch 0 termination unconfirmed; retained resources']
    concat=MagicMock(side_effect=AssertionError('Unconfirmed reader reached delivery'))
    monkeypatch.setattr(w.pipeline,'concat_output_files',concat)
    assert not process_date_cam(w.db,'20260901',0,dashboard_enabled=False)
    assert w.db.conn.execute('SELECT status FROM render_tasks').fetchone()[0]=='FAILED'
    concat.assert_not_called()
    assert {p:p.read_bytes() for p in w.original}==w.original
    assert all(p.read_bytes()==b'new media' for p in w.batches)


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
             patch("src.pipeline.get_monitor") as monitor, \
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
            assert monitor.return_value.stage.call_args_list == [
                call("pipeline_20260901_cam0"), call("final_concat_20260901_cam0")]
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


@pytest.mark.parametrize("audio_fast_path", [False, True])
def test_derived_visual_index_published_only_after_current_source_decode(tmp_path, audio_fast_path):
    import numpy as np
    from src.core.visual_index import source_identity, frame_signature
    from src.monitor import PerfCollector
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"source")
    filepath = str(source)
    db = VlogDatabase(tmp_path / "visual.db")
    try:
        db.add_file_task(filepath, 0, "20260901", "20260901000000", "20260901000010", 10)
        orch = StreamingOrchestrator(db, "20260901", 0, {}, render_enabled=False, dashboard_enabled=False)
        labels = [{"time": float(t), "state": "DYNAMIC", "energy": 8, "raw_energy": 8} for t in range(11)]
        detector = SimpleNamespace(analyze=lambda *a, **k: (labels, {}), decode_gpu="cpu", last_perf={}, fps=1,
                                   visual_source=source_identity(filepath),
                                   visual_samples=[(5, frame_signature(np.zeros((16,16),np.uint8)))])
        task = {"file_duration": 10}
        if audio_fast_path:
            task.update(is_audio_gated_static=True, audio_events=[(2, 8)])
        orch._execute_analysis_task(task, filepath, 0, detector, None, "cpu", PerfCollector(), time.monotonic())
        rows = db.conn.execute("SELECT * FROM visual_features").fetchall()
        if audio_fast_path:
            assert rows == []  # Reused detector must not leak its previous source samples.
        else:
            assert len(rows) == 1 and rows[0]["origin"] == "single_pass_analysis"
            assert rows[0]["filepath"] == filepath
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
        with patch.object(db, "sync_timeline_segments") as sync, patch("src.stages.companion.get_duration", return_value=12.2501) as probe, \
             patch("src.stages.companion.bind_delivery_mapping", return_value={"status": "AVAILABLE"}):
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
                 patch("src.stages.companion.bind_delivery_mapping", return_value={"status": "AVAILABLE"}), \
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


@pytest.mark.parametrize('prior_status', ['SUSPICIOUS', 'STATIC', 'FAILED', 'exception'])
@pytest.mark.parametrize('prior_slow', [False, True])
def test_prescreen_prior_classification_independent_of_completion_order(tmp_path, prior_status, prior_slow):
    db = VlogDatabase(tmp_path / 'ordered_prior.db')
    for name, start in [('prior.mp4', '000000'), ('current.mp4', '000100')]:
        db.add_file_task(name, 0, '20260901', '20260901' + start, '20260901000200', 60)
    orch = StreamingOrchestrator(db, '20260901', 0,
        {'detection': {'prescreen_parallel': 2}, 'audio_vad': {'prescreen_audio_gate': False}},
        render_enabled=False, dashboard_enabled=False)
    orch._prev_task_map['current.mp4'] = 'prior.mp4'
    orch.stop_event.set()
    current_sampled, release_prior = threading.Event(), threading.Event()
    observed = []

    def sample(filepath, *_args, prior_active_resolver, is_prior_active, **_kwargs):
        if filepath == 'prior.mp4':
            if prior_slow:
                assert release_prior.wait(5)
            if prior_status == 'exception':
                raise OSError('controlled predecessor read failure')
            return {'status': prior_status, 'result_json': '{}'}
        current_sampled.set()
        active = prior_active_resolver() if prior_active_resolver else is_prior_active
        observed.append(active)
        return {'status': 'SUSPICIOUS' if active else 'STATIC', 'result_json': '{}'}

    threads = []
    try:
        with patch('src.pipeline.prescreen_file', side_effect=sample), \
             patch('src.hardware.render_cache.processing_fingerprint', return_value='controlled'), \
             patch('src.pipeline.FFmpegProcessRegistry.is_interrupted', return_value=False):
            orch.prescreen_queue.put({'filepath': 'prior.mp4', 'file_duration': 60})
            if prior_slow:
                orch.prescreen_queue.put({'filepath': 'current.mp4', 'file_duration': 60})
            first = threading.Thread(target=orch._prescreen_worker, args=('cpu',))
            threads.append(first)
            first.start()
            if not prior_slow:
                first.join(5)
                assert not first.is_alive()
                orch.prescreen_queue.put({'filepath': 'current.mp4', 'file_duration': 60})
            second = threading.Thread(target=orch._prescreen_worker, args=('cpu',))
            threads.append(second)
            second.start()
            assert current_sampled.wait(5)
            if prior_slow:
                assert not observed  # parallel sample completed before predecessor publication
            release_prior.set()
            for thread in threads:
                thread.join(5)
                assert not thread.is_alive()
        assert observed == [prior_status == 'SUSPICIOUS']
        assert db.get_file_task('current.mp4')['prescreen_status'] == (
            'SUSPICIOUS' if prior_status == 'SUSPICIOUS' else 'STATIC')
        assert orch.prescreen_queue.unfinished_tasks == 0
    finally:
        release_prior.set()
        orch.abort_event.set()
        for thread in threads:
            thread.join(5)
        db.close()


def test_prescreen_prior_resume_and_cancellation(tmp_path):
    db = VlogDatabase(tmp_path / 'prior_resume.db')
    try:
        orch = StreamingOrchestrator(db, '20260901', 0, {}, render_enabled=False, dashboard_enabled=False)
        orch._prev_task_map['current.mp4'] = 'prior.mp4'
        orch._prescreen_results['prior.mp4'] = 'SUSPICIOUS'
        assert orch._await_prior_prescreen('current.mp4')
        assert not orch._await_prior_prescreen('first.mp4')
        del orch._prescreen_results['prior.mp4']
        orch.abort_event.set()
        with pytest.raises(InterruptedError, match='cancelled'):
            orch._await_prior_prescreen('current.mp4')
    finally:
        db.close()


def test_ordered_prescreen_cache_revision_invalidates_only_requested_day(tmp_path):
    from src.hardware.render_cache import processing_fingerprint, render_fingerprint
    db = VlogDatabase(tmp_path / 'prior_cache.db')
    config = {'detection': {'prescreen_diff_threshold': 8.5}}
    selected = {key: config.get(key, {}) for key in
                ('detection', 'yolo', 'audio_vad', 'segment', 'presence', 'micro_motion')}
    try:
        for date in ('20260901', '20260902'):
            source = str(tmp_path / (date + '.mp4'))
            db.add_file_task(source, 0, date, date + '000000', date + '000100', 60)
            db.set_prescreen_result(source, 'STATIC')
            old = render_fingerprint([source], 'analysis-v4-window-evidence', 'analysis', 0, {}, {}, selected)
            assert old != processing_fingerprint(source, config)
            db.set_processing_fingerprint(source, old)
        peer_before = db.get_all_file_tasks_for_date('20260902', 0)
        assert db.invalidate_stale_results('20260901', 0, config) == 1
        assert db.get_all_file_tasks_for_date('20260901', 0)[0]['prescreen_status'] == 'PENDING'
        assert db.get_all_file_tasks_for_date('20260902', 0) == peer_before
    finally:
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


@pytest.mark.parametrize('subtitles', [False, True])
def test_unavailable_delivery_mapping_preserves_companions_and_metrics(tmp_path, subtitles):
    from src.stages.companion import save_vlog_companion_assets
    out = tmp_path / 'day.mp4'
    out.write_bytes(b'controlled invalid clock')
    original = {out.with_suffix('.meta.json'): b'old metadata', out.with_suffix('.srt'): b'old subtitles'}
    for path, content in original.items():
        path.write_bytes(content)
    metrics = {}
    with patch('src.stages.companion.build_timeline_from_rows', return_value=[]), \
         patch('src.stages.companion.bind_delivery_mapping', return_value={'status': 'UNAVAILABLE', 'reason': 'PTS jump'}), \
         patch('src.stages.companion.save_timecode_subtitles') as save_subtitles:
        assert not save_vlog_companion_assets(out, '20260901', 0, 'camera', 1, 20, 1,
                                              MagicMock(), {'render': {'generate_subtitles': subtitles}}, rows=[],
                                              media_metrics=metrics)
    assert metrics == {}
    save_subtitles.assert_not_called()
    assert {path: path.read_bytes() for path in original} == original
    assert not list(tmp_path.glob('*.tmp'))


def test_invalid_mapping_blocks_pipeline_completion_and_retains_batches(delivery_workspace, monkeypatch):
    w = delivery_workspace
    monkeypatch.setattr('src.stages.companion.bind_delivery_mapping',
                        lambda *args: {'status': 'UNAVAILABLE', 'reason': 'PTS jump'})
    assert not w.pipeline.process_date_cam(w.db, '20260901', 0, dashboard_enabled=False)
    assert not w.db.is_render_completed('20260901', 0)
    assert {path: path.read_bytes() for path in w.original} == w.original
    assert all(batch.exists() for batch in w.batches)
    assert not list(w.output.parent.glob('*.pending.*'))


@pytest.mark.parametrize("failure", ["probe", "commit"])
def test_companion_failure_does_not_publish_reusable_media_metrics(tmp_path, failure):
    from src.stages.companion import save_vlog_companion_assets
    out = tmp_path / "day.mp4"
    out.write_bytes(b"placeholder")
    media_metrics = {}
    with patch("src.stages.companion.get_duration", return_value=0 if failure == "probe" else 12.2501), \
         patch("src.stages.companion.build_timeline_from_rows", return_value=[]), \
         patch("src.stages.companion.bind_delivery_mapping", return_value={"status": "AVAILABLE"}), \
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


@pytest.mark.parametrize("calibrated_duration", [20.0, 0.0])
def test_delivery_metrics_use_duration_calibrated_during_analysis(delivery_workspace, calibrated_duration, monkeypatch):
    w = delivery_workspace
    measured={}
    save_assets=w.pipeline._save_vlog_companion_assets
    def capture_assets(**kwargs):
        measured['elapsed']=kwargs['elapsed_wall']
        return save_assets(**kwargs)
    monkeypatch.setattr(w.pipeline,'_save_vlog_companion_assets',capture_assets)
    def finish_analysis():
        w.db.set_file_metadata(str(w.source), has_audio=0, duration=calibrated_duration)
        return w.batches
    w.orch.run.side_effect = finish_analysis
    assert w.pipeline.process_date_cam(w.db, "20260901", 0, dashboard_enabled=False)
    import json
    meta = json.loads(w.output.with_suffix(".meta.json").read_text(encoding="utf-8"))
    assert meta["metrics"]["raw_duration_s"] == calibrated_duration
    assert meta['performance']['wall_clock_s']==round(measured['elapsed'],2)
    assert meta["performance"]["speedup_x"] == round(
        calibrated_duration / max(measured['elapsed'], 0.1), 2)


@pytest.mark.parametrize('shared',['database','batches','delivery'])
def test_real_process_day_ownership_and_force_kill_release(tmp_path,external_day_owner,shared):
    from src.core.process_lock import day_task_lease
    database=tmp_path/'a'/'state.sqlite';temporary=tmp_path/'temp_a';output=tmp_path/'out_a'/'day.mp4'
    owner=external_day_owner(database,temporary,output)
    other_database=database if shared=='database' else tmp_path/'b'/'state.sqlite'
    other_temporary=temporary if shared=='batches' else tmp_path/'temp_b'
    other_output=output if shared=='delivery' else tmp_path/'out_b'/'day.mp4'
    date='20260902' if shared in ('delivery','batches') else '20260901'
    with pytest.raises(RuntimeError,match='already owned'):
        with day_task_lease(other_database,other_temporary,other_output,date,0):
            pytest.fail('second process must not enter shared work')
    if shared=='delivery':
        # A later-resource conflict must release resources already acquired.
        with day_task_lease(other_database,other_temporary,tmp_path/'out_c'/'day.mp4',date,0):pass
    owner.kill();owner.communicate(timeout=5)
    with day_task_lease(other_database,other_temporary,other_output,date,0):pass
    assert list(tmp_path.rglob('*.lock'))  # Stale names do not block recovery.


def test_main_day_conflict_precedes_persistent_mutation(delivery_workspace,external_day_owner,monkeypatch):
    from src.stages import renderer
    w=delivery_workspace
    external_day_owner(w.db.db_path,renderer.TEMP_DIR,w.output)
    monkeypatch.setattr(w.db,'invalidate_stale_results',lambda *args:pytest.fail('must acquire before invalidation'))
    monkeypatch.setattr(w.db,'reset_failed_tasks',lambda *args:pytest.fail('must acquire before retry state mutation'))
    with pytest.raises(RuntimeError,match='already owned'):
        w.pipeline.process_date_cam(w.db,'20260901',0,dashboard_enabled=False)
    w.orch.run.assert_not_called()
    for path,content in w.original.items():assert path.read_bytes()==content


def test_cleanup_cannot_delete_an_owned_workspace(delivery_workspace,external_day_owner):
    from src.core import utils
    w=delivery_workspace
    external_day_owner(w.db.db_path,utils.TEMP_DIR,w.output)
    batch=utils.TEMP_DIR/'_batch_active.mp4';batch.write_bytes(b'preserve live batch')
    with pytest.raises(RuntimeError,match='already owned'):
        utils.cleanup_temp_artifacts(clean_batches=True)
    assert batch.read_bytes()==b'preserve live batch'


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


@pytest.mark.parametrize('phase', ['prepared', 'saved', 'promoted', 'committed'])
def test_metadata_only_delivery_transaction_preserves_other_assets(delivery_workspace, phase):
    from src.core.delivery_recovery import prepare_transaction, commit_transaction, recover_transaction, recover_output
    from src.hardware.render_cache import _manifest_path
    w = delivery_workspace
    metadata = w.output.with_suffix('.meta.json')
    candidate = w.output.parent / 'mapping-candidate.json'
    candidate.write_bytes(b'{"delivery_mapping":{"status":"AVAILABLE"}}')
    replacement = candidate.read_bytes()
    targets = [w.output, metadata, w.output.with_suffix('.srt'), _manifest_path(w.output)]
    backup = w.output.parent / '.delivery_backup_metadata_only'
    backup.mkdir()
    journal = prepare_transaction(backup, w.output, [(candidate, metadata)], targets,
                                  context={'role': 'pipeline', 'date': '20260901', 'cam_index': 0},
                                  keep_previous=False)
    if phase != 'prepared':
        metadata.replace(backup / '1')
    if phase in ('promoted', 'committed'):
        candidate.replace(metadata)
    if phase == 'committed':
        commit_transaction(backup, journal)
    assert recover_transaction(backup) == ('COMMITTED' if phase == 'committed' else 'RECOVERED')
    assert metadata.read_bytes() == (replacement if phase == 'committed' else w.original[metadata])
    for target in targets:
        if target != metadata:
            assert target.read_bytes() == w.original[target]
    assert not (backup / '0').exists()
    assert not (backup / '2').exists()
    assert not (backup / '3').exists()
    if phase == 'promoted':
        assert (backup / 'interrupted' / '1').read_bytes() == replacement
    assert recover_output(w.output) == 0
    assert not (backup / 'review_history.json').exists()


@pytest.mark.parametrize('phase',[1,2,3,4,5,6,'database','committed'])
def test_actual_process_asset_interrupt_recovers_or_retains_committed(delivery_workspace,interrupted_delivery_owner,phase):
    from src.core.delivery_recovery import recover_transaction,pending_transactions
    w=delivery_workspace
    candidate=w.output.with_name(w.output.stem+'.pending.mp4')
    candidate.write_bytes(b'candidate video');candidate.with_suffix('.meta.json').write_text('{}')
    interrupted_delivery_owner(candidate,w.output,w.db.db_path,phase)
    backup=next(w.output.parent.glob('.delivery_backup_*'))
    if phase=='committed':
        assert not pending_transactions(w.output.parent,output=w.output)
        assert recover_transaction(backup)=='COMMITTED'
        assert w.output.read_bytes()==b'candidate video'
        assert w.db.is_render_completed('20260901',0)
        return
    if phase=='database':assert w.db.is_render_completed('20260901',0)
    mark=lambda:w.db.set_render_status('20260901',0,'PENDING')
    assert recover_transaction(backup,mark_pending=mark)=='RECOVERED'
    assert not w.db.is_render_completed('20260901',0)
    for path,content in w.original.items():assert path.read_bytes()==content
    assert recover_transaction(backup,mark_pending=mark)=='RECOVERED'
    if phase in (5,6,'database'):assert (backup/'interrupted'/'0').read_bytes()==b'candidate video'


def test_recovery_preflight_preserves_external_changes(delivery_workspace,interrupted_delivery_owner):
    from src.core.delivery_recovery import recover_transaction,DeliveryRecoveryError
    w=delivery_workspace
    candidate=w.output.with_name(w.output.stem+'.pending.mp4')
    candidate.write_bytes(b'candidate video');candidate.with_suffix('.meta.json').write_text('{}')
    interrupted_delivery_owner(candidate,w.output,w.db.db_path,6)
    w.output.with_suffix('.meta.json').write_text('external edit')
    before={path:path.read_bytes() if path.exists() else None for path in w.original}
    with pytest.raises(DeliveryRecoveryError,match='Unrecognized current asset'):
        recover_transaction(next(w.output.parent.glob('.delivery_backup_*')))
    assert before=={path:path.read_bytes() if path.exists() else None for path in w.original}


def test_asset_recovery_resumes_after_partial_restore_and_db_failure(delivery_workspace,interrupted_delivery_owner,monkeypatch):
    from src.core import delivery_recovery
    w=delivery_workspace
    candidate=w.output.with_name(w.output.stem+'.pending.mp4')
    candidate.write_bytes(b'candidate video');candidate.with_suffix('.meta.json').write_text('{}')
    interrupted_delivery_owner(candidate,w.output,w.db.db_path,'database')
    backup=next(w.output.parent.glob('.delivery_backup_*'))
    copy=delivery_recovery._copy_durable
    def fail_after_video(source,target,**kwargs):
        copy(source,target,**kwargs)
        if target==w.output:raise OSError('recovery interrupted')
    with monkeypatch.context() as patcher:
        patcher.setattr(delivery_recovery,'_copy_durable',fail_after_video)
        with pytest.raises(OSError,match='recovery interrupted'):
            delivery_recovery.recover_transaction(backup)
    def db_failure():raise OSError('status unavailable')
    with pytest.raises(OSError,match='status unavailable'):
        delivery_recovery.recover_transaction(backup,mark_pending=db_failure)
    assert delivery_recovery.pending_transactions(w.output.parent,output=w.output)==[backup]
    assert delivery_recovery.recover_transaction(backup,mark_pending=lambda:w.db.set_render_status('20260901',0,'PENDING'))=='RECOVERED'
    for path,content in w.original.items():assert path.read_bytes()==content


def test_completed_record_cannot_bypass_interrupted_package(delivery_workspace,interrupted_delivery_owner,monkeypatch):
    w=delivery_workspace
    candidate=w.output.with_name(w.output.stem+'.pending.mp4')
    candidate.write_bytes(b'candidate video');candidate.with_suffix('.meta.json').write_text('{}')
    interrupted_delivery_owner(candidate,w.output,w.db.db_path,'database')
    def inspect_then_stop(*args,**kwargs):
        assert not w.db.is_render_completed('20260901',0)
        for path,content in w.original.items():assert path.read_bytes()==content
        return False
    monkeypatch.setattr(w.pipeline,'_process_date_cam_owned',inspect_then_stop)
    assert w.pipeline.process_date_cam(w.db,'20260901',0) is False


def test_real_av_package_restores_after_process_death(delivery_workspace,interrupted_delivery_owner):
    from src.hardware.ffmpeg import run_ffmpeg
    from src.hardware.render_cache import save_manifest
    from src.core.delivery_recovery import recover_transaction
    from src.core.replay_evidence import content_sha256
    w=delivery_workspace
    candidate=w.output.with_name(w.output.stem+'.pending.mp4')
    for target,seconds in ((w.output,1),(candidate,2)):
        result=run_ffmpeg(['-f','lavfi','-i','testsrc2=size=128x128:rate=20','-f','lavfi','-i',
                          'sine=frequency=440:sample_rate=48000','-t',str(seconds),'-c:v','libx264',
                          '-preset','ultrafast','-c:a','aac',str(target)],timeout=20)
        assert result.returncode==0
    candidate.with_suffix('.meta.json').write_text('{}')
    save_manifest(w.output,'previous')
    previous={path:content_sha256(path) for path in w.original}
    interrupted_delivery_owner(candidate,w.output,w.db.db_path,'database')
    assert recover_transaction(next(w.output.parent.glob('.delivery_backup_*')),
                               mark_pending=lambda:w.db.set_render_status('20260901',0,'PENDING'))=='RECOVERED'
    assert previous=={path:content_sha256(path) for path in w.original}
    result=run_ffmpeg(['-v','error','-xerror','-i',str(w.output),'-f','null','-'],timeout=20)
    assert result.returncode==0 and not result.stderr
    import av
    with av.open(str(w.output)) as video:
        assert video.streams.audio and sum(1 for frame in video.decode(video=0))==20


def test_committed_review_history_finalizes_after_process_death(delivery_workspace,interrupted_delivery_owner):
    from src.core.delivery_recovery import recover_output
    import json
    w=delivery_workspace
    candidate=w.output.with_name(w.output.stem+'.pending.mp4')
    candidate.write_bytes(b'candidate video');candidate.with_suffix('.meta.json').write_text('{}')
    interrupted_delivery_owner(candidate,w.output,w.db.db_path,'committed',keep_previous=True)
    backup=next(w.output.parent.glob('.delivery_backup_*'))
    assert not (backup/'review_history.json').exists()
    assert recover_output(w.output)==1
    history=json.loads((backup/'review_history.json').read_text())
    for path,content in w.original.items():
        assert (backup/history['assets'][path.name]).read_bytes()==content
    assert w.output.read_bytes()==b'candidate video' and w.db.is_render_completed('20260901',0)
    assert recover_output(w.output)==0


def test_changed_previous_asset_blocks_recovery_before_writes(delivery_workspace,interrupted_delivery_owner):
    from src.core.delivery_recovery import recover_transaction,DeliveryRecoveryError
    w=delivery_workspace
    candidate=w.output.with_name(w.output.stem+'.pending.mp4')
    candidate.write_bytes(b'candidate video');candidate.with_suffix('.meta.json').write_text('{}')
    interrupted_delivery_owner(candidate,w.output,w.db.db_path,'database')
    backup=next(w.output.parent.glob('.delivery_backup_*'))
    (backup/'0').write_bytes(b'changed backup')
    before={path:path.read_bytes() if path.exists() else None for path in w.original}
    with pytest.raises(DeliveryRecoveryError,match='Previous asset changed'):
        recover_transaction(backup)
    assert before=={path:path.read_bytes() if path.exists() else None for path in w.original}


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



def test_review_history_is_published_only_after_success(delivery_workspace):
    from src.stages.companion import promote_vlog_assets
    w = delivery_workspace
    candidate = w.output.with_name(w.output.stem + ".pending.mp4")
    candidate.write_bytes(b"new media")
    candidate.with_suffix(".meta.json").write_text("{}")
    with pytest.raises(RuntimeError, match="commit rejected"):
        with promote_vlog_assets(candidate,w.output,w.config,keep_previous=True):
            raise RuntimeError("commit rejected")
    assert not list(w.output.parent.glob(".delivery_backup_*/review_history.json"))
    assert all(path.read_bytes() == content for path,content in w.original.items())


@pytest.fixture
def pending_pipeline_reader(monkeypatch, request):
    """An actual old Windows owner holds one of two local I/O slots."""
    import json
    import os
    import subprocess
    import sys
    from src.core.visual_index import reader_process_state
    from src.hardware import scheduler
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    if os.name != 'nt':
        pytest.skip('Windows native deferred pipeline reader')
    import _winapi
    for name in ('_processes', '_jobs', '_deferred', '_starting'):
        monkeypatch.setattr(registry, name, {})
    monkeypatch.setattr(registry, '_interrupted', False)
    releases, observations = [], []
    class Budget(scheduler.FileBudget):
        def release(self, weight=1):
            super().release(weight)
            releases.append(weight)
    disk = Budget(2)
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: disk)
    assert disk.acquire(timeout=1)
    key = 'old-date-unconfirmed-reader'
    child = registry.start(key, [sys.executable, '-c', 'import time;time.sleep(30)'],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    registry.defer_cleanup(key, child, disk)
    registry.mark_interrupted()
    def observe(stage):
        observations.append(dict(stage=stage, **reader_process_state(child.pid, handle=child._handle),
                                 handle=int(child._handle), disk_available=disk.available,
                                 interrupted=registry.is_interrupted(), releases=list(releases)))
    def stop():
        registry.terminate(key)
        assert _winapi.WaitForSingleObject(child._handle, 5000) == _winapi.WAIT_OBJECT_0
        registry.wait_for_exit(child, timeout=5)
    observe('pending_native_owner')
    try:
        yield SimpleNamespace(registry=registry, child=child, key=key, disk=disk, releases=releases,
                              stop=stop, observe=observe)
    finally:
        if _winapi.WaitForSingleObject(child._handle, 0) != _winapi.WAIT_OBJECT_0:
            stop()
        assert _winapi.WaitForSingleObject(child._handle, 5000) == _winapi.WAIT_OBJECT_0
        registry.reap_cleanups()
        observe('finally_native_ended')
        assert disk.available == 2 and releases == [1]
        assert not registry._processes and not registry._jobs and not registry._deferred
        registry.reset_interrupted()
        request.node.user_properties.append(('native_pipeline_admission', json.dumps(observations, sort_keys=True)))


@pytest.mark.parametrize('skip_render', [False, True])
def test_pending_pipeline_api_rejects_before_other_date_business_changes(
        delivery_workspace, pending_pipeline_reader, monkeypatch, skip_render):
    from src.hardware.ffmpeg import MediaTerminationUnconfirmed
    w, pending = delivery_workspace, pending_pipeline_reader
    w.db.add_file_task('unread-next-date', 0, '20260902', '20260902000000', '20260902000100', 60)
    w.db.set_prescreen_result('unread-next-date', 'FAILED', error_msg='previous failure')
    before = tuple(w.db.conn.iterdump())
    owned = MagicMock(side_effect=AssertionError('Pending reader reached day mutation boundary'))
    monkeypatch.setattr(w.pipeline, '_process_date_cam_owned', owned)
    with pytest.raises(MediaTerminationUnconfirmed, match='still pending'):
        w.pipeline.process_date_cam(w.db, '20260902', 0, config=w.config,
                                    skip_render=skip_render, dashboard_enabled=False)
    assert tuple(w.db.conn.iterdump()) == before
    assert {path: path.read_bytes() for path in w.original} == w.original
    assert pending.disk.available == 1 and not pending.releases and pending.registry.is_interrupted()
    owned.assert_not_called()
    w.pipeline.StreamingOrchestrator.assert_not_called()
    pending.observe('api_rejected_unchanged_business')


def test_pending_pipeline_constructor_rejects_before_scheduler_then_recovers_once(
        pending_pipeline_reader, tmp_path, monkeypatch):
    from src import pipeline
    from src.hardware.ffmpeg import MediaTerminationUnconfirmed
    pending = pending_pipeline_reader
    db = VlogDatabase(tmp_path/'constructor-admission.sqlite')
    manager = MagicMock(wraps=pipeline.WorkStealingManager)
    monkeypatch.setattr(pipeline, 'WorkStealingManager', manager)
    config = {'hardware': {'device': 'cpu', 'max_io_concurrency': 2}, 'render': {'batch_max_files': 1}}
    before = tuple(db.conn.iterdump())
    try:
        with pytest.raises(MediaTerminationUnconfirmed, match='still pending'):
            StreamingOrchestrator(db, '20260902', 0, config, render_enabled=False, dashboard_enabled=False)
        manager.assert_not_called()
        assert tuple(db.conn.iterdump()) == before and pending.registry.is_interrupted()
        pending.stop()
        assert pending.disk.available == 1 and not pending.releases
        instance = StreamingOrchestrator(db, '20260902', 0, config,
                                         render_enabled=False, dashboard_enabled=False)
        assert instance.date == '20260902' and manager.call_count == 1
        assert pending.disk.available == 2 and pending.releases == [1]
        assert not pending.registry.is_interrupted()
        assert pending.registry.require_cleanups() == {'recovered': [], 'pending': []}
        assert pending.releases == [1] and tuple(db.conn.iterdump()) == before
        pending.observe('constructor_admitted_after_confirmed_recovery')
    finally:
        db.close()


@pytest.mark.parametrize('force_render', [False, True])
def test_pending_pipeline_preserves_readonly_completed_skip_but_blocks_force(
        delivery_workspace, pending_pipeline_reader, force_render):
    from src.hardware.ffmpeg import MediaTerminationUnconfirmed
    w, pending = delivery_workspace, pending_pipeline_reader
    w.db.set_render_status('20260901', 0, 'COMPLETED', output_file='offline-delivery.mp4')
    before = tuple(w.db.conn.iterdump())
    if force_render:
        with pytest.raises(MediaTerminationUnconfirmed, match='still pending'):
            w.pipeline.process_date_cam(w.db, '20260901', 0, config=w.config,
                                        force_render=True, dashboard_enabled=False)
    else:
        assert w.pipeline.process_date_cam(w.db, '20260901', 0, config=w.config,
                                           dashboard_enabled=False)
    assert tuple(w.db.conn.iterdump()) == before
    assert {path: path.read_bytes() for path in w.original} == w.original
    w.pipeline.StreamingOrchestrator.assert_not_called()
    assert pending.registry.is_interrupted() and pending.disk.available == 1 and not pending.releases


def test_pending_pipeline_start_claim_blocks_constructor_before_scheduler(monkeypatch):
    from src import pipeline
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry, MediaTerminationUnconfirmed
    for name in ('_processes', '_jobs', '_deferred', '_starting'):
        monkeypatch.setattr(registry, name, {})
    manager = MagicMock(side_effect=AssertionError('Scheduler constructed during pending startup'))
    monkeypatch.setattr(pipeline, 'WorkStealingManager', manager)
    registry._starting['unreturned-child'] = {'proc': None}
    try:
        with pytest.raises(MediaTerminationUnconfirmed, match='still pending'):
            StreamingOrchestrator(MagicMock(), '20260902', 0, {}, dashboard_enabled=False)
        manager.assert_not_called()
        assert registry.is_interrupted()
    finally:
        registry._starting.clear()
        registry.reset_interrupted()


@pytest.mark.parametrize('outcome', ['typed_fatal', 'false_pending', 'ordinary_false'])
def test_cross_date_pipeline_stops_only_for_unconfirmed_media_and_reports_remaining(tmp_path, monkeypatch, outcome):
    from src import pipeline
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry, MediaTerminationUnconfirmed
    for name in ('_processes', '_jobs', '_deferred', '_starting'):
        monkeypatch.setattr(registry, name, {})
    monkeypatch.setattr(registry, '_interrupted', False)
    db = VlogDatabase(tmp_path/'cross-date.sqlite')
    for date in ('20260901', '20260902'):
        db.add_file_task(date+'-unread', 0, date, date+'000000', date+'000100', 60)
    before, calls = tuple(db.conn.iterdump()), []
    asset = tmp_path/'retained-delivery.txt'
    asset.write_bytes(b'retained prior delivery')
    original_close = db.close
    monkeypatch.setattr(db, 'close', lambda: None)
    monkeypatch.setattr(pipeline, 'VlogDatabase', lambda: db)
    monkeypatch.setattr(pipeline, 'scan_directory', MagicMock())
    monkeypatch.setattr(pipeline, 'get_monitor', MagicMock())
    monkeypatch.setattr(pipeline, 'OUTPUT_DIR', tmp_path)
    monkeypatch.setattr(pipeline, 'check_disk_space', lambda *a: True)
    monkeypatch.setattr(pipeline, 'cleanup_resources', lambda: None)
    monkeypatch.setattr('src.core.utils.cleanup_temp_artifacts', lambda: None)
    monkeypatch.setattr(pipeline, 'print_batch_startup_banner', MagicMock())
    monkeypatch.setattr('src.ui.print_batch_summary_table', MagicMock())
    monkeypatch.setattr(pipeline, 'build_batch_summary_item', lambda *a, **k: {'status': a[6]})
    def process(database, date, camera, **kwargs):
        calls.append(date)
        if len(calls) == 1:
            if outcome == 'typed_fatal':
                raise MediaTerminationUnconfirmed('controlled old reader remains')
            if outcome == 'false_pending':
                registry.defer_cleanup('controlled-pending', SimpleNamespace(poll=lambda: None), MagicMock())
                registry.mark_interrupted()
            return False
        return True
    monkeypatch.setattr(pipeline, 'process_date_cam', process)
    try:
        result = pipeline.run_pipeline(dashboard_enabled=False, config={'hardware': {'device': 'cpu'}})
        stopped = outcome != 'ordinary_false'
        assert calls == (['20260901'] if stopped else ['20260901', '20260902'])
        assert result == {'total': 2, 'ok': 0 if stopped else 1, 'failed': 1, 'skipped': 0,
                          'remaining': 1 if stopped else 0,
                          'stopped_reason': 'MEDIA_TERMINATION_UNCONFIRMED' if stopped else None}
        assert tuple(db.conn.iterdump()) == before
        assert asset.read_bytes() == b'retained prior delivery'
    finally:
        registry._deferred.clear()
        registry.reset_interrupted()
        original_close()


def test_pending_pipeline_run_entry_rejects_before_cleanup_scan_or_database(monkeypatch):
    from src import pipeline
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry, MediaTerminationUnconfirmed
    for name in ('_processes', '_jobs', '_deferred', '_starting'):
        monkeypatch.setattr(registry, name, {})
    registry.defer_cleanup('controlled-pending', SimpleNamespace(poll=lambda: None), MagicMock())
    database, scan, cleanup = MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(pipeline, 'VlogDatabase', database)
    monkeypatch.setattr(pipeline, 'scan_directory', scan)
    monkeypatch.setattr('src.core.utils.cleanup_temp_artifacts', cleanup)
    try:
        with pytest.raises(MediaTerminationUnconfirmed, match='still pending'):
            pipeline.run_pipeline(dashboard_enabled=False)
        database.assert_not_called()
        scan.assert_not_called()
        cleanup.assert_not_called()
        assert registry.is_interrupted()
    finally:
        registry._deferred.clear()
        registry.reset_interrupted()


@pytest.mark.parametrize('stage', ['streaming', 'concat', 'companion'])
def test_public_media_failure_stays_fatal_across_dates_after_cleanup_is_empty(
        delivery_workspace, monkeypatch, stage):
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry, MediaTerminationUnconfirmed
    w = delivery_workspace
    second = w.source.with_name('next-date-unread.mp4')
    second.write_bytes(b'untouched following date')
    w.db.add_file_task(str(second), 0, '20260902', '20260902000000', '20260902000100', 60)
    second_before = w.db.get_file_task(str(second))
    assert not registry._deferred and not registry._starting
    failure = MediaTerminationUnconfirmed('old reader has ended but this day failed fatally')
    if stage == 'streaming':
        w.orch.run.side_effect = failure
    elif stage == 'concat':
        monkeypatch.setattr(w.pipeline, 'concat_output_files', MagicMock(side_effect=failure))
    else:
        monkeypatch.setattr('src.stages.companion.get_duration', MagicMock(side_effect=failure))
    original_close = w.db.close
    monkeypatch.setattr(w.db, 'close', lambda: None)
    monkeypatch.setattr(w.pipeline, 'VlogDatabase', lambda: w.db)
    monkeypatch.setattr(w.pipeline, 'scan_directory', MagicMock())
    monkeypatch.setattr(w.pipeline, 'check_disk_space', lambda *a: True)
    monkeypatch.setattr(w.pipeline, 'cleanup_resources', lambda: None)
    monkeypatch.setattr('src.core.utils.cleanup_temp_artifacts', lambda: None)
    monkeypatch.setattr(w.pipeline, 'print_batch_startup_banner', MagicMock())
    monkeypatch.setattr('src.ui.print_batch_summary_table', MagicMock())
    try:
        # The real public process_date_cam and its owned implementation run.
        # Only the stage that failed is replaced; no public-API throw shortcut.
        result = w.pipeline.run_pipeline(dashboard_enabled=False, config=w.config)
        assert result == {'total': 2, 'ok': 0, 'failed': 1, 'skipped': 0, 'remaining': 1,
                          'stopped_reason': 'MEDIA_TERMINATION_UNCONFIRMED'}
        w.orch.run.assert_called_once()
        assert w.pipeline.StreamingOrchestrator.call_count == 1
        assert w.pipeline.StreamingOrchestrator.call_args.args[1] == '20260901'
        assert not registry._deferred and not registry._starting
        assert w.db.get_file_task(str(second)) == second_before
        assert not w.db.is_render_completed('20260901', 0)
        assert w.db.conn.execute("SELECT status FROM render_tasks WHERE date='20260901'").fetchone()[0] == 'FAILED'
        assert {path: path.read_bytes() for path in w.original} == w.original
        assert all(path.read_bytes() == b'new media' for path in w.batches)
    finally:
        original_close()


def test_companion_media_probe_failure_is_typed_instead_of_false(delivery_workspace, monkeypatch, tmp_path):
    from src.hardware.ffmpeg import MediaTerminationUnconfirmed
    from src.stages.companion import save_vlog_companion_assets
    candidate = tmp_path/'probe-pending.mp4'
    candidate.write_bytes(b'controlled pending candidate')
    metadata = candidate.with_suffix('.meta.json')
    metadata.write_bytes(b'previous candidate metadata')
    metrics = {}
    monkeypatch.setattr('src.stages.companion.get_duration',
                        MagicMock(side_effect=MediaTerminationUnconfirmed('probe cleanup unconfirmed')))
    with pytest.raises(MediaTerminationUnconfirmed, match='probe cleanup unconfirmed'):
        save_vlog_companion_assets(candidate, '20260901', 0, 'camera', 0, 0, 1,
                                   delivery_workspace.db, {'render': {'generate_subtitles': False}},
                                   rows=[], media_metrics=metrics)
    assert metrics == {} and metadata.read_bytes() == b'previous candidate metadata'


@pytest.mark.parametrize("kind", ["matching", "unrelated", "history", "corrupt", "missing_output"])
def test_completed_delivery_checks_legacy_packages_without_mutation(delivery_workspace, monkeypatch, kind):
    import json
    from src.core.delivery_recovery import DeliveryRecoveryError
    w = delivery_workspace
    w.db.set_render_status("20260901", 0, "COMPLETED", output_file="" if kind == "missing_output" else str(w.output))
    backup = w.output.parent / ".delivery_backup_legacy"
    backup.mkdir()
    mapping = {"0": str(w.output if kind != "unrelated" else w.output.parent / "other.mp4")}
    (backup / "recovery.json").write_text("{" if kind == "corrupt" else json.dumps(mapping))
    if kind == "history":
        (backup / "review_history.json").write_text("{}")
    before_db = tuple(w.db.conn.iterdump())
    before_files = {p: p.read_bytes() for p in w.output.parent.rglob("*") if p.is_file()}
    monkeypatch.setattr(w.db, "invalidate_stale_results", lambda *a: pytest.fail("invalidation before recovery"))
    # Config naming changes must not hide a package for the recorded old output.
    config = {"output": {"naming": "renamed_{date}.mp4"}}
    if kind in ("matching", "corrupt", "missing_output"):
        with pytest.raises(DeliveryRecoveryError, match="[Ll]egacy"):
            w.pipeline.process_date_cam(w.db, "20260901", 0, config=config)
    else:
        assert w.pipeline.process_date_cam(w.db, "20260901", 0, config=config)
    assert tuple(w.db.conn.iterdump()) == before_db
    assert {p: p.read_bytes() for p in w.output.parent.rglob("*") if p.is_file()} == before_files
    w.pipeline.StreamingOrchestrator.assert_not_called()


@pytest.mark.parametrize('failure', ['hash', 'rename'])
def test_delivery_preparation_failure_does_not_create_recovery_blocker(delivery_workspace, monkeypatch, failure):
    from src.core import delivery_recovery
    from src.stages.companion import promote_vlog_assets
    w = delivery_workspace
    candidate = w.output.with_name('candidate.mp4')
    candidate.write_bytes(b'new media')
    candidate.with_suffix('.meta.json').write_text('{}')
    def fail(*args, **kwargs):
        raise OSError('controlled preparation failure')
    with monkeypatch.context() as patch:
        if failure == 'hash':
            patch.setattr(delivery_recovery, '_identity', fail)
        else:
            patch.setattr(Path, 'rename', fail)
        with pytest.raises(OSError, match='controlled preparation failure'):
            with promote_vlog_assets(candidate, w.output, w.config):
                pytest.fail('must not yield')
    for path, content in w.original.items():
        assert path.read_bytes() == content
    assert candidate.read_bytes() == b'new media'
    assert not list(w.output.parent.glob('.delivery_backup_*'))
    assert not list(w.output.parent.glob('.delivery_preparing_*'))
    assert delivery_recovery.recover_output(w.output) == 0
    with promote_vlog_assets(candidate, w.output, w.config):
        pass
    assert w.output.read_bytes() == b'new media'


def test_actual_process_exit_during_delivery_preparation_preserves_output(delivery_workspace):
    import subprocess
    import sys
    from src.core.delivery_recovery import recover_output
    from src.stages.companion import promote_vlog_assets
    w = delivery_workspace
    candidate = w.output.with_name('preparation-crash.mp4')
    candidate.write_bytes(b'new media')
    candidate.with_suffix('.meta.json').write_text('{}')
    code = '''
import os, sys
from pathlib import Path
from src.core import delivery_recovery
from src.stages.companion import promote_vlog_assets
delivery_recovery._identity = lambda path: os._exit(37)
with promote_vlog_assets(Path(sys.argv[1]), Path(sys.argv[2]), {'render': {'generate_subtitles': False}}):
    raise AssertionError('unexpected promotion')
'''
    result = subprocess.run([sys.executable, '-c', code, str(candidate), str(w.output)], capture_output=True, timeout=30)
    assert result.returncode == 37, result.stderr
    assert len(list(w.output.parent.glob('.delivery_preparing_*'))) == 1
    assert not list(w.output.parent.glob('.delivery_backup_*'))
    for path, content in w.original.items():
        assert path.read_bytes() == content
    assert recover_output(w.output) == 0
    with promote_vlog_assets(candidate, w.output, w.config):
        pass
    assert w.output.read_bytes() == b'new media'


def test_heterogeneous_tail_guard_leaves_last_heavy_batch_for_active_nv(tmp_path, monkeypatch):
    """Exercise the real worker branch after dispatch, with NV busy and one queued batch."""
    import queue
    import threading
    import src.pipeline as pipeline
    from src.hardware.scheduler import DualEndedBatchQueue

    db = VlogDatabase(tmp_path / "tail_guard.db")
    cfg = {
        "pipeline": {"render_start_delay": 0, "render_gpu_policy": "heterogeneous"},
        "render": {"batch_max_files": 1, "max_concurrency": 2, "local_staging_enabled": False},
    }
    for index in range(2):
        source = f"tail{index}.mp4"
        db.add_file_task(source, 0, "20260901", f"20260901000{index}00", f"20260901000{index+1}00", 60)
        db.set_prescreen_result(source, "STATIC")
    orch = StreamingOrchestrator(db, "20260901", 0, cfg, dashboard_enabled=False)
    for index, duration in enumerate((100, 10)):
        orch.render_batch_queue.put({"filepath": f"tail{index}.mp4", "status": "ANALYZED",
            "dynamic_duration": duration, "is_heavy": True})
    orch.stop_event.set()
    dispatched, nv_started, qsv_done = (threading.Event() for _ in range(3))
    original_queue = queue.Queue
    calls, steal_calls, observations = [], [], []

    class HeavyQueue(DualEndedBatchQueue):
        def notify_all(self):
            dispatched.set()
            return super().notify_all()

        def pop_heaviest(self, *args, **kwargs):
            assert dispatched.wait(5)
            return super().pop_heaviest(*args, **kwargs)

        def qsize(self):
            size = super().qsize()
            observations.append((size, orch._active_nv_renders))
            return size

        def steal_lightest(self, *args, **kwargs):
            steal_calls.append(kwargs)
            return super().steal_lightest(*args, **kwargs)

    class LightQueue(original_queue):
        def get_nowait(self):
            # Hold QSV before it snapshots active_nv; NV remains busy until QSV exits.
            assert nv_started.wait(5)
            raise queue.Empty

    queue_count = 0

    def make_queue(*args, **kwargs):
        nonlocal queue_count
        queue_count += 1
        return (LightQueue if queue_count == 1 else original_queue)(*args, **kwargs)

    original_guard = orch._guard_worker

    def guard(worker, *args):
        try:
            return original_guard(worker, *args)
        finally:
            if args and args[0] == "qsv":
                qsv_done.set()

    def render(_segments, batch, gpu, *_args, **_kwargs):
        calls.append((gpu, batch))
        if gpu == "nv" and batch == 0:
            nv_started.set()
            assert qsv_done.wait(5), "QSV did not terminate at the tail guard"
        return str(tmp_path / f"batch{batch}.mp4")

    monkeypatch.setattr(pipeline, "DualEndedBatchQueue", HeavyQueue)
    monkeypatch.setattr(pipeline.queue, "Queue", make_queue)
    monkeypatch.setattr(pipeline, "build_batch_render", render)
    monkeypatch.setattr(orch, "_guard_worker", guard)
    thread = threading.Thread(target=orch._render_manager, daemon=True)
    try:
        thread.start()
        thread.join(10)
        assert not thread.is_alive()
        assert (1, 1) in observations
        assert not steal_calls
        assert calls == [("nv", 0), ("nv", 1)]
        assert sorted(batch for batch, _path in orch.batch_paths) == [0, 1]
        assert not orch.errors
    finally:
        orch.abort_event.set()
        nv_started.set()
        qsv_done.set()
        dispatched.set()
        thread.join(5)
        db.close()


@pytest.mark.parametrize("mode", ["handoff", "budget", "eligible", "undispatched", "last", "cancel", "failure", "pressure", "unknown", "recovery"])
def test_qsv_tail_thread_reuses_nv_budget_without_changing_encoder(tmp_path, monkeypatch, mode):
    import src.pipeline as pipeline
    from src.hardware.scheduler import DualEndedBatchQueue
    db = VlogDatabase(tmp_path / "tail_handoff.db")
    cfg = {
        "pipeline": {"render_start_delay": 0, "render_gpu_policy": "heterogeneous"},
        "hardware": {"max_nv_concurrency": 2 if mode == "budget" else 3},
        "render": {"batch_max_files": 1, "max_concurrency": 3, "local_staging_enabled": False},
    }
    count = 3 if mode == "last" else 6
    for index in range(count):
        source = f"tail{index}.mp4"
        db.add_file_task(source, 0, "20260901", f"20260901000{index}00", f"20260901000{index+1}00", 60)
        db.set_prescreen_result(source, "STATIC")
    orch = StreamingOrchestrator(db, "20260901", 0, cfg, dashboard_enabled=False)
    probes = []

    def vram_probe():
        probes.append(True)
        if mode in ("pressure", "unknown"):
            release.set()
            return None if mode == "unknown" else orch.work_stealing.vram_watermark_mb
        if mode == "recovery" and len(probes) == 1:
            assert "nv_tail_0" not in orch.render_worker_stats
            return orch.work_stealing.vram_watermark_mb
        return 0

    orch.work_stealing.set_vram_probe_fn(vram_probe)
    for index in range(count):
        orch.render_batch_queue.put({"filepath": f"tail{index}.mp4", "status": "ANALYZED",
            "dynamic_duration": 100 if mode == "eligible" else 500 if mode == "undispatched" else 300,
            "is_heavy": True})
    if mode != "undispatched":
        orch.stop_event.set()
    dispatched, nv_pair, release, early_checked = (threading.Event() for _ in range(4))
    original_queue = queue.Queue
    lock = threading.Lock()
    calls, running, peak, failed = [], 0, 0, []

    class HeavyQueue(DualEndedBatchQueue):
        def notify_all(self):
            dispatched.set()
            return super().notify_all()

        def pop_heaviest(self, *args, **kwargs):
            if mode != "undispatched":
                assert dispatched.wait(5)
            return super().pop_heaviest(*args, **kwargs)

        def lightest_dynamic_duration(self):
            if mode == "budget":
                release.set()
            if mode == "undispatched" and not dispatched.is_set():
                early_checked.set()
            return super().lightest_dynamic_duration()

    class LightQueue(original_queue):
        def get_nowait(self):
            assert nv_pair.wait(5)
            raise queue.Empty

    queue_count = 0

    def make_queue(*args, **kwargs):
        nonlocal queue_count
        queue_count += 1
        return (LightQueue if queue_count == 1 else original_queue)(*args, **kwargs)

    def render(_segments, batch, gpu, *_args, **_kwargs):
        nonlocal running, peak
        with lock:
            calls.append((gpu, batch))
            if gpu == "nv":
                running += 1
                peak = max(peak, running)
                if running >= 2:
                    nv_pair.set()
                if running == 3:
                    if mode == "failure" and not failed:
                        failed.append(batch)
                    if mode == "cancel":
                        orch.abort_event.set()
                    release.set()
            else:
                release.set()
        try:
            assert release.wait(5), "QSV neither handed off nor made its eligibility decision"
            if batch in failed:
                return None
            return str(tmp_path / f"batch{batch}.mp4")
        finally:
            if gpu == "nv":
                with lock:
                    running -= 1

    monkeypatch.setattr(pipeline, "DualEndedBatchQueue", HeavyQueue)
    monkeypatch.setattr(pipeline.queue, "Queue", make_queue)
    monkeypatch.setattr(pipeline, "build_batch_render", render)
    original_guard = orch._guard_worker

    def guard(worker, *args):
        try:
            return original_guard(worker, *args)
        finally:
            if args and args[0] == "qsv":
                release.set()

    monkeypatch.setattr(orch, "_guard_worker", guard)
    thread = threading.Thread(target=orch._render_manager, daemon=True)
    try:
        thread.start()
        if mode == "undispatched":
            assert early_checked.wait(5)
            assert peak == 2 and "nv_tail_0" not in orch.render_worker_stats
            orch.stop_event.set()
        thread.join(10)
        assert not thread.is_alive()
        assert orch._active_nv_renders == orch._active_qsv_renders == 0
        handoff = mode in ("handoff", "undispatched", "cancel", "failure", "recovery")
        assert orch.render_worker_stats["qsv_0"]["handoff_to_nv"] is handoff
        assert ("nv_tail_0" in orch.render_worker_stats) is handoff
        assert peak == (3 if handoff else 2)
        if mode == "eligible":
            assert any(gpu == "qsv" for gpu, _ in calls)
        else:
            assert all(gpu == "nv" for gpu, _ in calls)
        if mode == "failure":
            assert len(failed) == 1
            assert sum(batch == failed[0] for _, batch in calls) == 2
            assert any(f"render batch {failed[0]} returned no output" in error for error in orch.errors)
        elif mode != "cancel":
            assert sorted(batch for batch, _ in orch.batch_paths) == list(range(count))
            assert not orch.errors
    finally:
        orch.abort_event.set()
        orch.stop_event.set()
        release.set()
        nv_pair.set()
        dispatched.set()
        thread.join(5)
        db.close()
