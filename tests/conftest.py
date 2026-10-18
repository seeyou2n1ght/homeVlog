"""
Pytest Fixtures and Global Configurations for HomeVlog Test Suite.
"""

import pytest
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

from src.database import VlogDatabase
from src.segment import Segment
from src.utils import reset_semaphores


@pytest.fixture(autouse=True)
def isolate_runtime_outputs(tmp_path, monkeypatch):
    """Tests must not mix mock telemetry or assets into production directories."""
    import sys
    workspace = tmp_path / "runtime"
    replacements = {
        "OUTPUT_DIR": workspace / "output", "TEMP_DIR": workspace / "temp",
        "TEMP_STAGING_DIR": workspace / "temp/staging", "TEMP_BATCHES_DIR": workspace / "temp/batches",
        "TEMP_SCRIPTS_DIR": workspace / "temp/scripts", "LOGS_DIR": workspace / "logs",
        "DB_PATH": workspace / "data/vlog.db", "CACHE_DIR": workspace / "temp/audit_cache",
        "ARCHIVE_DIR": workspace / "data/feedback_archive", "DEFAULT_ARCHIVE_DIR": workspace / "data/feedback_archive",
        "DEFAULT_IMAGES_DIR": workspace / "data/feedback_archive/images",
        "DEFAULT_MANIFEST": workspace / "data/feedback_archive/manifest.jsonl",
    }
    for name, value in replacements.items():
        (value.parent if name in ("DB_PATH", "DEFAULT_MANIFEST") else value).mkdir(parents=True, exist_ok=True)
    # Compatibility facades and eager package imports retain path snapshots.
    # Patch every already-loaded consumer; later imports inherit core paths.
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith(("src.", "scripts.audit_tool.")):
            for name, value in replacements.items():
                if hasattr(module, name):
                    monkeypatch.setattr(module, name, value)
    import logging
    logger = logging.getLogger("homevlog")
    previous = list(logger.handlers)
    for handler in previous:
        logger.removeHandler(handler)
    yield
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    for handler in previous:
        logger.addHandler(handler)


@pytest.fixture(autouse=True)
def clean_semaphores():
    """Reset hardware semaphores and registry interruption before and after each test for test isolation."""
    from src.renderer import FFmpegProcessRegistry
    reset_semaphores()
    FFmpegProcessRegistry.reset_interrupted()
    yield
    reset_semaphores()
    FFmpegProcessRegistry.reset_interrupted()


@pytest.fixture(autouse=True)
def reset_global_config_state():
    """Reset global config path and settings singleton before and after each test for test isolation."""
    import src.core.config as core
    import src.utils as legacy
    core.CONFIG_PATH = core.DEFAULT_CONFIG_PATH
    core.SETTINGS = {}
    legacy.CONFIG_PATH = core.DEFAULT_CONFIG_PATH
    legacy.SETTINGS = {}
    yield
    core.CONFIG_PATH = core.DEFAULT_CONFIG_PATH
    core.SETTINGS = {}
    legacy.CONFIG_PATH = core.DEFAULT_CONFIG_PATH
    legacy.SETTINGS = {}


@pytest.fixture
def temp_test_dir(tmp_path):
    """Provide an isolated workspace directory with subfolders for output, temp, logs, data."""
    workspace = tmp_path / "workspace"
    for sub in ["output", "temp", "logs", "data", "test_input"]:
        (workspace / sub).mkdir(parents=True, exist_ok=True)
    return workspace


@pytest.fixture
def mock_config(temp_test_dir):
    """Provide a decoupled and isolated configuration dictionary."""
    return {
        "paths": {
            "input_dir": str(temp_test_dir / "test_input"),
            "output_dir": str(temp_test_dir / "output"),
            "temp_dir": str(temp_test_dir / "temp"),
        },
        "hardware": {
            "device": "cpu",
            "max_nv_concurrency": 3,
            "max_qsv_concurrency": 8,
            "max_io_concurrency": 8,
            "max_nv_decoders": 1,
        },
        "scheduler": {
            "watermark_high": 10,
            "watermark_low": 3,
            "nvdec_cooperative": True,
            "max_nv_decoders": 1,
        },
        "recovery": {
            "skip_today": False,
            "scanner_freeze_minutes": 0,
            "min_disk_space_gb": 1,
            "file_stabilize_wait": 0.01,
        },
        "pipeline": {
            "streaming_mode": True,
            "render_start_delay": 0,
            "prescreen_gpu_policy": "alternating",
        },
        "detection": {
            "prescreen_mode": "stream_fps",
            "prescreen_parallel": 2,
            "prescreen_segments": 10,
            "prescreen_resolution": "320x180",
            "prescreen_diff_threshold": 6,
            "prescreen_extract_timeout": 10.0,
            "analysis_fps": 5,
            "analysis_resolution": "416x234",
            "analysis_max_workers": 2,
            "qsv_fallback_threshold": 10,
            "analysis_fps_adaptive": True,
            "analysis_fps_tiers": {"short": 5, "medium": 3, "long": 2},
            "analysis_fps_tier_thresholds": {"short_max": 120, "medium_max": 600},
            "analysis_early_term_enabled": True,
            "analysis_early_term_window": 20,
            "analysis_early_term_threshold": 2.5,
            "motion_sensitivity": 1.0,
            "roi_crop": [0.1, 0.12, 0.8, 0.85],
            "min_motion_frames": 3,
            "min_static_frames": 5,
            "noise_suppress_frames": 2,
            "median_filter_window": 7,
            "grid_cols": 8,
            "grid_rows": 8,
        },
        "segment": {
            "min_motion_duration": 2.0,
            "min_static_duration": 30.0,
            "gap_tolerance": 0.5,
            "static_keyframe_interval": 30.0,
            "keyframe_display_duration": 0.5,
            "min_static_display_duration": 1.5,
        },
        "yolo": {
            "enabled": False,
            "streaming_verify": False,
            "skip_energy_threshold": 12.0,
            "model_path": "yolo11n.pt",
            "device": "cpu",
            "target_classes": [0, 1, 2, 3, 15, 16],
            "confidence": 0.3,
            "sample_fps": 0.5,
        },
        "render": {
            "static_mode": "hybrid_keyframe",
            "hw_decode": True,
            "batch_max_files": 4,
            "concat_timeout": 120,
        },
        "output": {
            "per_camera_vlog": True,
            "naming": "DailyVlog_{date}_cam{index}.mp4",
            "resolution": "1920x1080",
            "fps": 20,
            "nv": {
                "codec": "hevc_nvenc",
                "preset": "p1",
                "cq": 30,
                "maxrate": "3M",
                "bufsize": "8M",
            },
            "qsv": {
                "codec": "hevc_qsv",
                "preset": "veryfast",
                "global_quality": 30,
                "maxrate": "3M",
                "bufsize": "8M",
            },
            "audio": {
                "codec": "aac",
                "bitrate": "96k",
                "channels": 1,
            },
        },
        "logging": {
            "level": "INFO",
            "rotation_max_bytes": 10485760,
            "rotation_backup_count": 5,
            "monitor_interval": 1.0,
        },
    }


@pytest.fixture
def delivery_workspace(tmp_path, monkeypatch):
    """Isolate delivery orchestration while keeping real SQLite/assets/rollback."""
    from types import SimpleNamespace
    from src import pipeline
    from src.hardware.render_cache import processing_fingerprint, _manifest_path
    config = {"output": {"naming": "DailyVlog_{date}_cam{index}.mp4"},
              "render": {"generate_subtitles": False}}
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    source = tmp_path / "source.mp4"
    source.write_bytes(b"controlled source")
    db = VlogDatabase(tmp_path / "delivery.db")
    db.add_file_task(str(source), 0, "20260901", "20260901000000", "20260901000100", 60)
    db.set_prescreen_result(str(source), "STATIC")
    db.set_processing_fingerprint(str(source), processing_fingerprint(source, config))
    batches = [tmp_path / f"batch{i}.mp4" for i in range(2)]
    for batch in batches:
        batch.write_bytes(b"new media")
        batch.with_name(batch.name + ".json").write_bytes(b"batch manifest")
    output = out_dir / "DailyVlog_20260901_cam0.mp4"
    protected = (output, output.with_suffix(".meta.json"), output.with_suffix(".srt"), _manifest_path(output))
    original = {}
    for index, path in enumerate(protected):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"old asset {index}".encode())
        original[path] = path.read_bytes()
    orch = MagicMock()
    orch.errors = []
    orch.run.return_value = batches
    for name, replacement in {
        "load_config": lambda: config, "OUTPUT_DIR": out_dir,
        "StreamingOrchestrator": MagicMock(return_value=orch), "get_monitor": MagicMock(),
        "print_startup_banner": MagicMock(), "print_summary_card": MagicMock(),
        "_dump_perf": MagicMock(), "_build_headline": MagicMock(return_value={}),
    }.items():
        monkeypatch.setattr(pipeline, name, replacement)
    def concat(paths, destination, **kwargs):
        destination.write_bytes(b"new media")
        return True
    monkeypatch.setattr(pipeline, "concat_output_files", concat)
    monkeypatch.setattr("src.stages.companion.get_duration", lambda _: 12.2501)
    # Synthetic bytes exercise publication, not the real media clock validator.
    monkeypatch.setattr("src.stages.companion.bind_delivery_mapping",
                        lambda *args: {"status": "AVAILABLE"})
    monkeypatch.setattr("src.core.utils.cleanup_staging_files", lambda: None)
    try:
        yield SimpleNamespace(db=db, config=config, output=output, batches=batches,
                              original=original, source=source, orch=orch, pipeline=pipeline)
    finally:
        db.close()


@pytest.fixture
def isolated_db(temp_test_dir):
    """Create an isolated SQLite database backed by a temporary file."""
    db_file = temp_test_dir / "data" / "vlog_test.db"
    db = VlogDatabase(db_path=db_file)
    yield db
    db.close()


@pytest.fixture
def sample_segments():
    """Provide a sample list of Segments representing mixed static and dynamic activities."""
    return [
        Segment(
            start_time=10.0,
            end_time=40.0,
            state="STATIC",
            source_file="/dummy/00_20260901000007_20260901005727.mp4",
            file_start_offset=0.0,
            max_energy=0.5,
        ),
        Segment(
            start_time=40.0,
            end_time=55.0,
            state="DYNAMIC",
            source_file="/dummy/00_20260901000007_20260901005727.mp4",
            file_start_offset=0.0,
            max_energy=18.2,
        ),
        Segment(
            start_time=55.0,
            end_time=120.0,
            state="STATIC",
            source_file="/dummy/00_20260901000007_20260901005727.mp4",
            file_start_offset=0.0,
            max_energy=0.8,
        ),
    ]


@pytest.fixture
def blocked_source_reader(tmp_path):
    """Owned native pipe reads model metadata/decode stalls without changing NAS."""
    import os
    import subprocess
    import sys
    worker = tmp_path / 'blocked_reader.py'
    worker.write_text('''import json, os, sys
from pathlib import Path
from src.core.visual_index import source_identity
filepath=json.loads(sys.stdin.buffer.readline())
if sys.argv[1]=='sampling':
    print(json.dumps(['identity',source_identity(filepath)]),flush=True)
    json.loads(sys.stdin.buffer.readline())
Path(sys.argv[2]).write_text(str(os.getpid()))
read_fd,write_fd=os.pipe()
os.read(read_fd,1)
''',encoding='utf-8')
    owned=[]
    def start(mode='metadata', ready=None):
        environment=os.environ.copy()
        environment['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
        executable=sys.executable
        if os.name=='nt' and sys.prefix!=sys.base_prefix:
            executable=sys._base_executable;environment['__PYVENV_LAUNCHER__']=sys.executable
        process=subprocess.Popen([executable,str(worker),mode,str(ready or tmp_path/'reader.ready')],
                                 env=environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        owned.append(process)
        return process
    start.worker=worker
    start.ready=tmp_path/'reader.ready'
    start.owned=owned
    yield start
    for process in owned:
        if process.poll() is None:process.kill()
        process.wait(timeout=5)
        for stream in (process.stdin,process.stdout,process.stderr):
            if stream:stream.close()


@pytest.fixture
def short_local_video(tmp_path):
    """Real local 20fps media for PTS and independently launched reader checks."""
    import av
    import numpy as np
    path=tmp_path/'short.mp4'
    with av.open(str(path),'w') as output:
        stream=output.add_stream('mpeg4',rate=20)
        stream.width=stream.height=64;stream.pix_fmt='yuv420p'
        for value in range(120):
            frame=av.VideoFrame.from_ndarray(np.full((64,64),value,np.uint8),format='gray')
            for packet in stream.encode(frame):output.mux(packet)
        for packet in stream.encode():output.mux(packet)
    return path


@pytest.fixture
def external_day_owner():
    """A real second process holds the same production lease primitive."""
    import os
    import subprocess
    import sys
    owned=[]
    def start(database,temporary,output,date='20260901',camera=0):
        script="from src.core.process_lock import day_task_lease; import os,sys;\nwith day_task_lease(*sys.argv[1:5],int(sys.argv[5])): print(os.getpid(),flush=True); sys.stdin.read()"
        environment=os.environ.copy();executable=sys.executable
        if os.name=='nt' and sys.prefix!=sys.base_prefix:
            executable=sys._base_executable;environment['__PYVENV_LAUNCHER__']=sys.executable
        process=subprocess.Popen([executable,'-c',script,str(database),str(temporary),str(output),date,str(camera)],
                                 cwd=str(Path(__file__).resolve().parents[1]),env=environment,
                                 stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        owned.append(process)
        line=process.stdout.readline().strip()
        assert line,str(process.communicate(timeout=5))
        assert int(line)==process.pid
        return process
    yield start
    for process in owned:
        if process.poll() is None:process.kill()
        process.communicate(timeout=5)


@pytest.fixture
def interrupted_delivery_owner(tmp_path):
    """An actual process stops at selected replacement/DB/commit boundaries."""
    import os
    import subprocess
    import sys
    script=tmp_path/'interrupt_assets.py'
    script.write_text('''import sys
from pathlib import Path
from src.stages.companion import promote_vlog_assets
from src.core import delivery_recovery
from src.hardware.render_cache import save_manifest
from src.database import VlogDatabase
candidate,output=map(Path,sys.argv[1:3]);phase=sys.argv[3]
replace=Path.replace;counter=0
def pause():
    print('READY',flush=True);sys.stdin.read()
def instrument(source,target):
    global counter
    result=replace(source,target)
    if not source.name.endswith('.tmp'):
        counter+=1
        if phase==str(counter):pause()
    return result
Path.replace=instrument
atomic=delivery_recovery.atomic_json
def commit(path,value):
    atomic(path,value)
    if phase=='committed' and value['state']=='COMMITTED':pause()
delivery_recovery.atomic_json=commit
with promote_vlog_assets(candidate,output,{'render':{'generate_subtitles':False}},fingerprint='candidate',keep_previous=sys.argv[5]=='1',
                         context={'role':'pipeline','date':'20260901','cam_index':0}):
    save_manifest(output,'candidate')
    db=VlogDatabase(db_path=sys.argv[4]);db.set_render_status('20260901',0,'COMPLETED',output_file=str(output));db.close()
    if phase=='database':pause()
''',encoding='utf-8')
    owned=[]
    def start(candidate,output,database,phase,keep_previous=False):
        environment=os.environ.copy();environment['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
        executable=sys.executable
        if os.name=='nt' and sys.prefix!=sys.base_prefix:
            executable=sys._base_executable;environment['__PYVENV_LAUNCHER__']=sys.executable
        process=subprocess.Popen([executable,str(script),str(candidate),str(output),str(phase),str(database),'1' if keep_previous else '0'],
                                 env=environment,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        owned.append(process)
        assert process.stdout.readline().strip()=='READY',process.communicate(timeout=5)
        process.kill();process.communicate(timeout=5)
        return process
    yield start
    for process in owned:
        if process.poll() is None:process.kill()
        process.communicate(timeout=5)
