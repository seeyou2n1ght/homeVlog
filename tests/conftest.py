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
def clean_semaphores():
    """Reset hardware semaphores before and after each test for test isolation."""
    reset_semaphores()
    yield
    reset_semaphores()


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
            "min_segment_duration": 0.1,
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
