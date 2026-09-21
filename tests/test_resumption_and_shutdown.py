import subprocess
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.database import VlogDatabase
from src.pipeline import StreamingOrchestrator
from src.renderer import build_batch_render, FFmpegProcessRegistry
from src.timeline import TimelineSegment
from src.utils import cleanup_temp_artifacts, TEMP_DIR


def test_cleanup_continues_after_cuda_failure_and_checkpoints_database():
    from src.utils import cleanup_resources
    db = MagicMock()
    with patch("torch.cuda.is_available", return_value=True), \
         patch("torch.cuda.empty_cache", side_effect=RuntimeError("device unavailable")), \
         patch("src.hardware.ffmpeg.FFmpegProcessRegistry.kill_all") as kill:
        cleanup_resources(db)
    db.conn.execute.assert_called_once_with("PRAGMA wal_checkpoint(TRUNCATE)")
    kill.assert_called_once()


def test_cleanup_temp_artifacts_preserves_valid_batches(tmp_path):
    with patch("src.utils.TEMP_DIR", tmp_path):
        b1 = tmp_path / '_batch0_test.mp4'
        b1.write_bytes(b'x' * 1024)
        tmp_b = tmp_path / '_batch1_test.tmp.mp4'
        tmp_b.write_bytes(b'temp data')
        fc = tmp_path / '_fc_batch0_test.txt'
        fc.write_text('filter complex')

        cleaned = cleanup_temp_artifacts(clean_batches=False)
        assert b1.exists()
        assert not tmp_b.exists()
        assert not fc.exists()

        cleaned_all = cleanup_temp_artifacts(clean_batches=True)
        assert not b1.exists()


def test_pure_static_files_marked_for_skip_frame(tmp_path):
    segs = [
        TimelineSegment(
            filepath='static_file1.mp4',
            input_index=0,
            start_in_file=0.0,
            end_in_file=10.0,
            state='STATIC',
            duration=10.0,
        ),
        TimelineSegment(
            filepath='static_file2.mp4',
            input_index=1,
            start_in_file=0.0,
            end_in_file=15.0,
            state='STATIC',
            duration=15.0,
        ),
    ]

    with patch('src.renderer._run_batch_render') as mock_run:
        mock_run.return_value = 'mock_batch0.mp4'
        build_batch_render(
            segs,
            bi=0,
            enc_for_batch='nv',
            fps=20,
            width=1920,
            height=1080,
            seg_cfg={},
            out_cfg={},
            audio_cfg={},
            date='20260901',
            cam_index=0,
            rows=[],
        )

        assert mock_run.called
        call_kwargs = mock_run.call_args[1]
        pure_static = call_kwargs.get('pure_static_files')
        assert pure_static == set()  # Short static intervals must retain inter-frames.


def test_streaming_orchestrator_immediate_abort_on_event(tmp_path):
    db_path = tmp_path / 'test_abort.db'
    db = VlogDatabase(db_path=db_path)
    try:
        orch = StreamingOrchestrator(
            db=db,
            date='20260901',
            cam_index=0,
            config={},
            render_enabled=False,
            dashboard_enabled=False,
        )

        orch.abort_event.set()
        orch.stop_event.set()

        t0 = time.monotonic()
        paths = orch.run()
        elapsed = time.monotonic() - t0
        assert elapsed < 1.0
        assert paths == []
    finally:
        db.close()


def test_render_batch_error_preserves_batches_and_aborts_concat(tmp_path):
    from src.pipeline import process_date_cam
    db_path = tmp_path / 'test_fail_preserve.db'
    db = VlogDatabase(db_path=db_path)
    b0 = tmp_path / '_batch0_20260901_cam0.mp4'
    b0.write_bytes(b'valid batch content')

    try:
        db.add_file_task('f1.mp4', 0, '20260901', '20260901000000', '20260901000500', 300.0)
        with patch('src.pipeline.StreamingOrchestrator') as mock_orch_cls:
            mock_orch = MagicMock()
            mock_orch_cls.return_value = mock_orch
            mock_orch.run.return_value = [b0]
            mock_orch.error_lock = threading.Lock()
            mock_orch.errors = ['render batch 1 failed on nv']
            mock_orch.render_worker_stats = {}

            with patch('src.pipeline._dump_perf'), patch('src.pipeline.concat_output_files') as mock_concat:
                ok = process_date_cam(db, '20260901', 0, dashboard_enabled=False)
                assert not ok
                # 严禁执行 concat，严禁删除已成功产出的 batch 0
                assert not mock_concat.called
                assert b0.exists()
                # 状态必须打标为 FAILED
                assert not db.is_render_completed('20260901', 0)
    finally:
        db.close()


def test_closed_database_safe_returns_without_error(tmp_path, caplog):
    db_path = tmp_path / 'test_closed_db.db'
    db = VlogDatabase(db_path=db_path)
    db.close()
    assert db.is_closed

    # 验证数据库关闭后，调用各方法不会触发 AttributeError 或输出 ERROR 日志
    with caplog.at_level("ERROR"):
        assert db.get_all_file_tasks_for_date("20260901", 0) == []
        assert db.get_pending_file_count_for_date("20260901", 0) == 0
        assert not db.is_render_completed("20260901", 0)
        assert db.reset_failed_tasks("20260901", 0) == {"prescreen": 0, "analysis": 0}
        # 写入方法应安全无操作返回
        db.upsert_render_task("20260901", 0, "PENDING")
        db.set_render_status("20260901", 0, "COMPLETED")

    assert "DB error" not in caplog.text
    assert "NoneType" not in caplog.text


def test_ffmpeg_process_registry_interruption_and_error_clean():
    from src.renderer import _clean_ffmpeg_error

    FFmpegProcessRegistry.reset_interrupted()
    assert not FFmpegProcessRegistry.is_interrupted()

    FFmpegProcessRegistry.mark_interrupted()
    assert FFmpegProcessRegistry.is_interrupted()

    FFmpegProcessRegistry.reset_interrupted()
    assert not FFmpegProcessRegistry.is_interrupted()

    # 验证逐帧进度行被彻底过滤，仅保留真实报错
    progress_noise = (
        "frame=  100 fps=6.0 q=36.0 size=   1024KiB time=00:00:05.00 bitrate=1600.0kbits/s speed=0.30x\r"
        "frame=  200 fps=6.0 q=36.0 size=   2048KiB time=00:00:10.00 bitrate=1600.0kbits/s speed=0.30x\r"
        "frame=  300 fps=6.0 q=36.0 size=   3072KiB time=00:00:15.00 bitrate=1600.0kbits/s speed=0.30x\n"
        "[hevc_nvenc @ 000001] OpenEncodeSessionEx failed: out of memory (10)\n"
        "Error while opening encoder for output stream #0:0"
    )
    cleaned = _clean_ffmpeg_error(progress_noise)
    assert "frame=" not in cleaned
    assert "bitrate=" not in cleaned
    assert "OpenEncodeSessionEx failed: out of memory" in cleaned
    assert "Error while opening encoder" in cleaned

    # 验证若只有纯进度输出（无任何报错行），过滤后为空
    pure_progress = "frame= 2435 fps=6.0 q=36.0 size= 44032KiB time=00:02:01.75 bitrate=2962.7kbits/s speed=0.299x"
    assert _clean_ffmpeg_error(pure_progress) == ""


def test_batch_render_suppresses_progress_error_on_interruption(tmp_path, caplog):
    FFmpegProcessRegistry.mark_interrupted()
    try:
        segs = [
            TimelineSegment(
                filepath=str(tmp_path / "f1.mp4"),
                input_index=0,
                start_in_file=0.0,
                end_in_file=5.0,
                state="STATIC",
                duration=5.0,
            )
        ]
        # 创建一个假的 err_log 且内容全是进度信息
        mock_proc = MagicMock()
        mock_proc.returncode = 1  # Windows 下 kill 产生的典型退出码

        with patch("subprocess.Popen") as mock_popen, patch("src.renderer._build_enc_args", return_value=[]):
            mock_popen.return_value = mock_proc
            with caplog.at_level("INFO"):
                res = build_batch_render(
                    segs,
                    bi=1,
                    enc_for_batch="nv",
                    fps=20,
                    width=1920,
                    height=1080,
                    seg_cfg={},
                    out_cfg={},
                    audio_cfg={},
                    date="20260901",
                    cam_index=0,
                    rows=[],
                )
                assert res is None
        # 验证输出为 INFO 中断提示，绝无 ERROR 倒灌逐帧进度
        assert any("terminated by signal (Ctrl+C)" in record.message for record in caplog.records if record.levelname == "INFO")
        assert not any("batch-render cam0 batch1 failed" in record.message for record in caplog.records if record.levelname == "ERROR")
    finally:
        FFmpegProcessRegistry.reset_interrupted()

