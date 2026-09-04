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


def test_cleanup_temp_artifacts_preserves_valid_batches(tmp_path):
    b1 = TEMP_DIR / '_batch0_test.mp4'
    b1.write_bytes(b'x' * 1024)
    tmp_b = TEMP_DIR / '_batch1_test.tmp.mp4'
    tmp_b.write_bytes(b'temp data')
    fc = TEMP_DIR / '_fc_batch0_test.txt'
    fc.write_text('filter complex')

    try:
        cleaned = cleanup_temp_artifacts(clean_batches=False)
        assert b1.exists()
        assert not tmp_b.exists()
        assert not fc.exists()

        cleaned_all = cleanup_temp_artifacts(clean_batches=True)
        assert not b1.exists()
    finally:
        b1.unlink(missing_ok=True)
        tmp_b.unlink(missing_ok=True)
        fc.unlink(missing_ok=True)


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
        assert pure_static == {'static_file1.mp4', 'static_file2.mp4'}


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
