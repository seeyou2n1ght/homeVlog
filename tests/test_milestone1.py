import threading
import time
from pathlib import Path

import pytest
import yaml

from src.utils import (
    load_config,
    get_nv_semaphore,
    get_qsv_semaphore,
    get_disk_semaphore,
    reset_semaphores,
    WorkStealingManager,
)
from src.pipeline import StreamingOrchestrator


class TestWorkStealingManager:
    """Tests for Milestone 1 WorkStealingManager adaptive state machine."""

    def test_default_state_and_routing(self):
        config = {
            "hardware": {"device": "cuda:0", "max_nv_decoders": 1},
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
            },
        }
        mgr = WorkStealingManager(config)
        assert mgr.state == "NORMAL_DECOUPLED"
        assert not mgr.is_render_active
        assert mgr.active_nv_decoders == 0

        # When queue backlog is low (< 10), default to QSV
        assert mgr.get_analysis_device(queue_size=0) == "qsv"
        assert mgr.get_analysis_device(queue_size=5) == "qsv"
        assert mgr.get_analysis_device(queue_size=9) == "qsv"
        assert mgr.state == "NORMAL_DECOUPLED"

    def test_high_watermark_cooperative_burst(self):
        config = {
            "hardware": {"device": "cuda:0", "max_nv_decoders": 1},
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
            },
        }
        mgr = WorkStealingManager(config)

        # Backlog reaches high watermark -> should select CUDA
        device = mgr.get_analysis_device(queue_size=10)
        assert device == "cuda"
        assert mgr.state == "COOPERATIVE_BURST"

        # Higher backlog also selects CUDA
        assert mgr.get_analysis_device(queue_size=25) == "cuda"

    def test_low_watermark_hysteresis(self):
        config = {
            "hardware": {"device": "cuda:0", "max_nv_decoders": 1},
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
            },
        }
        mgr = WorkStealingManager(config)

        # Trigger burst
        assert mgr.get_analysis_device(queue_size=12) == "cuda"
        assert mgr.state == "COOPERATIVE_BURST"

        # In hysteresis zone (between low 3 and high 10), should maintain COOPERATIVE_BURST
        assert mgr.get_analysis_device(queue_size=7) == "cuda"
        assert mgr.get_analysis_device(queue_size=4) == "cuda"

        # Reaching low watermark (<= 3) -> return to NORMAL_DECOUPLED (QSV)
        assert mgr.get_analysis_device(queue_size=3) == "qsv"
        assert mgr.state == "NORMAL_DECOUPLED"

        # Now at 5 it stays QSV because it was reset
        assert mgr.get_analysis_device(queue_size=5) == "qsv"

    def test_render_preemption_yield(self):
        config = {
            "hardware": {"device": "cuda:0", "max_nv_decoders": 1},
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
            },
        }
        mgr = WorkStealingManager(config)

        # Trigger high backlog
        assert mgr.get_analysis_device(queue_size=20) == "cuda"

        # Render starts -> register_render_start()
        mgr.register_render_start()
        assert mgr.is_render_active
        assert mgr.state == "RENDER_PREEMPTION_YIELD"

        # High backlog MUST yield to QSV while render is active
        assert mgr.get_analysis_device(queue_size=50) == "qsv"
        assert mgr.get_analysis_device(queue_size=100) == "qsv"
        assert mgr.acquire_nvdec_slot() is False

        # Render ends -> register_render_end()
        mgr.register_render_end()
        assert not mgr.is_render_active

        # After render ends, high backlog can resume CUDA work-stealing
        assert mgr.get_analysis_device(queue_size=20) == "cuda"
        assert mgr.state == "COOPERATIVE_BURST"

    def test_slot_limits_and_lease_context_manager(self):
        config = {
            "hardware": {"device": "cuda:0", "max_nv_decoders": 1},
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
            },
        }
        mgr = WorkStealingManager(config)

        # Worker 1 leases device with high backlog
        with mgr.lease_device(queue_size=15) as dev1:
            assert dev1 == "cuda"
            assert mgr.active_nv_decoders == 1

            # Worker 2 tries to lease concurrently -> max_nv_decoders reached, falls back to QSV
            with mgr.lease_device(queue_size=15) as dev2:
                assert dev2 == "qsv"
                assert mgr.active_nv_decoders == 1

        # Worker 1 exited context manager -> slot released
        assert mgr.active_nv_decoders == 0

        # Exception safety
        try:
            with mgr.lease_device(queue_size=15) as dev:
                assert dev == "cuda"
                assert mgr.active_nv_decoders == 1
                raise RuntimeError("Simulated analysis error")
        except RuntimeError:
            pass

        assert mgr.active_nv_decoders == 0

    def test_cpu_or_no_cuda_fallback(self):
        config = {
            "hardware": {"device": "cpu"},
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
            },
        }
        mgr = WorkStealingManager(config)
        assert mgr.get_analysis_device(queue_size=100) == "qsv"


class TestHardwareSemaphores:
    """Tests for hardware concurrency semaphores."""

    def setup_method(self):
        reset_semaphores()

    def teardown_method(self):
        reset_semaphores()

    def test_semaphore_capacities(self):
        sem_nv = get_nv_semaphore()
        sem_qsv = get_qsv_semaphore()
        sem_disk = get_disk_semaphore()

        # Verify NV semaphore capacity (3)
        assert sem_nv._value == 3
        # Verify QSV semaphore capacity (8)
        assert sem_qsv._value == 8
        # Verify Disk semaphore capacity (8)
        assert sem_disk._value == 8

    def test_semaphore_acquire_release(self):
        sem_nv = get_nv_semaphore()
        assert sem_nv.acquire(timeout=0.1) is True
        assert sem_nv._value == 2
        sem_nv.release()
        assert sem_nv._value == 3


class TestSettingsSchema:
    """Tests for config/settings.yaml integrity and completeness."""

    def test_settings_load_and_keys(self):
        cfg = load_config()
        assert isinstance(cfg, dict)

        # Check critical top-level sections
        assert "paths" in cfg
        assert "hardware" in cfg
        assert "pipeline" in cfg
        assert "scheduler" in cfg
        assert "recovery" in cfg
        assert "detection" in cfg
        assert "segment" in cfg
        assert "yolo" in cfg
        assert "pass2" in cfg
        assert "render" in cfg
        assert "output" in cfg
        assert "logging" in cfg

        # Check Milestone 1 specific settings
        assert cfg["pipeline"]["prescreen_gpu_policy"] == "qsv_only"
        assert cfg["scheduler"]["watermark_high"] == 10
        assert cfg["scheduler"]["watermark_low"] == 3
        assert cfg["scheduler"]["nvdec_cooperative"] is True
        assert cfg["scheduler"]["max_nv_decoders"] == 1
        assert cfg["hardware"]["max_nv_concurrency"] == 3
        assert cfg["hardware"]["max_qsv_concurrency"] == 8
        assert cfg["hardware"]["max_io_concurrency"] == 8


class TestPipelineOrchestratorInit:
    """Tests for StreamingOrchestrator initialization with WorkStealingManager."""

    def test_orchestrator_initializes_work_stealing(self):
        cfg = load_config()
        # Mock database
        class MockDB:
            pass

        orchestrator = StreamingOrchestrator(
            db=MockDB(),
            date="20260901",
            cam_index=0,
            config=cfg,
            render_enabled=True,
        )
        assert hasattr(orchestrator, "work_stealing")
        assert isinstance(orchestrator.work_stealing, WorkStealingManager)
        assert orchestrator.work_stealing.watermark_high == 10
        assert orchestrator.work_stealing.watermark_low == 3
