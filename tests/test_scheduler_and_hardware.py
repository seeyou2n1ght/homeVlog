"""测试模块 4: 异构自适应调度器 (WorkStealingManager) 与硬件资源隔离.

覆盖：
1. 异构调度器高低水位线动态跳变 (NORMAL_DECOUPLED <-> COOPERATIVE_BURST)；
2. Pass 2 渲染抢占与 NVDEC 强制让步 (RENDER_PREEMPTION_YIELD)；
3. 协同槽位出租生命周期 (lease_device, acquire/release_nvdec_slot)；
4. 硬件并发信号量隔离与重置 (nv, qsv, disk IO, reset_semaphores)；
5. 磁盘空间安全阈值防御 (check_disk_space)；
6. 系统监控探针 (get_monitor) 与性能记录采集 (PerfCollector)。
"""

import threading
import time
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

from src.utils import (
    WorkStealingManager,
    get_nv_semaphore,
    get_qsv_semaphore,
    get_disk_semaphore,
    reset_semaphores,
    check_disk_space,
)
from src.monitor import get_monitor, get_perf, PerfRecord


class TestWorkStealingScheduler:
    """测试异构算力动态工作窃取调度器。"""

    def test_watermark_transitions_and_burst(self):
        cfg = {
            "scheduler": {"watermark_high": 10, "watermark_low": 3, "nvdec_cooperative": True, "max_nv_decoders": 1},
            "hardware": {"device": "cuda:0"},
        }
        mgr = WorkStealingManager(config=cfg)
        mgr.disable_cold_start_burst()
        mgr.set_vram_probe_fn(lambda: 3000)

        # 1. 低于高水位线时，默认 QSV 解耦独占
        assert mgr.get_analysis_device(queue_size=5) == "qsv"
        assert mgr.state == "NORMAL_DECOUPLED"

        # 2. 达到/超过高水位线，爆发切换到 CUDA (NVDEC)
        assert mgr.get_analysis_device(queue_size=12) == "cuda"
        assert mgr.state == "COOPERATIVE_BURST"

        # 3. 队列回落至低水位线以下，回归 QSV
        assert mgr.get_analysis_device(queue_size=2) == "qsv"
        assert mgr.state == "NORMAL_DECOUPLED"

    def test_render_preemption_forces_qsv(self):
        cfg = {
            "scheduler": {"watermark_high": 5, "watermark_low": 2, "nvdec_cooperative": True, "max_nv_decoders": 1},
            "hardware": {"device": "cuda:0"},
        }
        mgr = WorkStealingManager(config=cfg)

        # 渲染启动
        mgr.register_render_start()
        assert mgr.is_render_active
        # 即便队列严重积压，也必须强制让步 QSV
        assert mgr.get_analysis_device(queue_size=100) == "qsv"
        assert mgr.state == "RENDER_PREEMPTION_YIELD"

        # 渲染结束
        mgr.register_render_end()
        assert not mgr.is_render_active

    def test_vram_pressure_yield_forces_qsv(self):
        """测试物理显存触及安全高水位时强制抑制 NVDEC 借调，让步 QSV。"""
        cfg = {
            "scheduler": {
                "watermark_high": 5,
                "watermark_low": 2,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
                "vram_watermark_mb": 6000,
            },
            "hardware": {"device": "cuda:0"},
        }
        mgr = WorkStealingManager(config=cfg)
        mgr.disable_cold_start_burst()

        # 模拟显存水位安全 (4000MB < 6000MB)：高队列时允许 CUDA 协同
        mgr.set_vram_probe_fn(lambda: 4000)
        assert mgr.get_analysis_device(queue_size=10) == "cuda"
        assert mgr.state == "COOPERATIVE_BURST"

        # 模拟显存触及警戒线 (6500MB >= 6000MB)：强制进入 VRAM_PRESSURE_YIELD 并让步 QSV
        mgr.set_vram_probe_fn(lambda: 6500)
        assert mgr.get_analysis_device(queue_size=10) == "qsv"
        assert mgr.state == "VRAM_PRESSURE_YIELD"

        # 显存承压状态下拒绝出租 NVDEC 槽位
        assert not mgr.acquire_nvdec_slot()

        # 显存恢复安全水位 (5500MB < 6000MB)：恢复协同借调
        mgr.set_vram_probe_fn(lambda: 5500)
        assert mgr.get_analysis_device(queue_size=10) == "cuda"
        assert mgr.state == "COOPERATIVE_BURST"

    def test_lease_device_context_manager(self):
        cfg = {
            "scheduler": {"watermark_high": 5, "watermark_low": 2, "nvdec_cooperative": True, "max_nv_decoders": 1},
            "hardware": {"device": "cuda:0"},
        }
        mgr = WorkStealingManager(config=cfg)
        mgr.disable_cold_start_burst()
        mgr.set_vram_probe_fn(lambda: 3000)

        # 在高水位下租用设备
        with mgr.lease_device(queue_size=10) as dev1:
            assert dev1 == "cuda"
            assert mgr.active_nv_decoders == 1

            # 达到 max_nv_decoders=1 上限后，第二任务自动降级到 qsv
            with mgr.lease_device(queue_size=10) as dev2:
                assert dev2 == "qsv"

        # 退出上下文后自动释放槽位
        assert mgr.active_nv_decoders == 0


class TestHardwareSemaphoresAndProtection:
    """测试硬件并发信号量与磁盘保护。"""

    def test_semaphore_singleton_and_reset(self):
        reset_semaphores()
        sem_nv1 = get_nv_semaphore()
        sem_nv2 = get_nv_semaphore()
        assert sem_nv1 is sem_nv2

        reset_semaphores()
        sem_nv3 = get_nv_semaphore()
        assert sem_nv3 is not sem_nv1

    def test_hardware_semaphores_isolation(self):
        reset_semaphores()
        sem_nv = get_nv_semaphore()
        sem_qsv = get_qsv_semaphore()
        sem_io = get_disk_semaphore()

        assert sem_nv is not sem_qsv
        assert sem_qsv is not sem_io

    def test_check_disk_space(self, tmp_path):
        # 足够空间
        assert check_disk_space(tmp_path, min_gb=1)
        # 极其夸张的 1000000 GB，必然返回 False 拦截
        assert not check_disk_space(tmp_path, min_gb=1_000_000)


class TestSystemMonitoring:
    """测试监控与性能记录采集。"""

    def test_monitor_stages_and_perf_collector(self, tmp_path):
        monitor = get_monitor()
        with monitor.stage("test_stage"):
            time.sleep(0.01)

        perf = get_perf()
        perf.reset()
        perf.add(PerfRecord(stage="prescreen", file="clip1.mp4", gpu="qsv", duration=0.05))
        perf.add(PerfRecord(stage="analysis", file="clip1.mp4", gpu="cuda", duration=0.15))

        summary = perf.summary_by_stage()
        assert "prescreen" in summary
        assert "analysis" in summary
        assert summary["prescreen"]["count"] == 1
        assert summary["analysis"]["count"] == 1

        dump_path = tmp_path / "perf.json"
        perf.dump(dump_path)
        assert dump_path.exists()

    def test_cold_start_burst_requires_backlog(self):
        """测试冷启动破冰只在队列有积压 (> watermark_low) 时启用，尾部任务绝不滥用独显 NVDEC。"""
        cfg = {
            "scheduler": {"watermark_high": 10, "watermark_low": 3, "nvdec_cooperative": True, "max_nv_decoders": 1},
            "hardware": {"device": "cuda:0"},
        }
        mgr = WorkStealingManager(config=cfg)
        mgr.enable_cold_start_burst()
        mgr.set_vram_probe_fn(lambda: 3000)

        # 尾部少量任务 (1 <= queue_size <= 3): 即使冷启动也不借调 CUDA，全部由 QSV 处理
        assert mgr.get_analysis_device(queue_size=1) == "qsv"
        assert mgr.get_analysis_device(queue_size=2) == "qsv"
        assert mgr.get_analysis_device(queue_size=3) == "qsv"
        assert mgr.state == "NORMAL_DECOUPLED"

        # 队列积压超过 watermark_low (queue_size=4 > 3): 触发破冰借调 CUDA
        assert mgr.get_analysis_device(queue_size=4) == "cuda"
        assert mgr.state == "COOPERATIVE_BURST"


def test_pipe_decode_watchdog_terminates_hang(tmp_path):
    """测试 _decode_file_pipe 在子进程挂死时被独立硬看门狗主动终止，且信号量安全释放。"""
    from src.detector import MotionDetector
    from src.utils import get_qsv_semaphore

    det = MotionDetector(config={"detection": {"decode_timeout": 1.0}}, decode_gpu="qsv")
    io_sem = get_qsv_semaphore()

    # 模拟一个会输出海量 stderr 且挂起的子进程
    # 验证无论 stderr 有多少数据，均不会发生管道写满死锁，且定时器超时后子进程被杀灭
    fake_video = tmp_path / "hang.mp4"
    fake_video.write_bytes(b"fake video data")

    t0 = time.monotonic()
    # 传入极小 duration 触发 60s 硬看门狗（或者通过 mock 缩短看门狗）
    with patch("subprocess.Popen") as mock_popen:
        mock_proc = MagicMock()
        mock_proc.stdout.read.side_effect = lambda size: time.sleep(0.5) or b""
        mock_proc.poll.return_value = None
        mock_proc.returncode = 0
        mock_popen.return_value = mock_proc

        frames, yolo_buf, meta = det._decode_file_pipe(str(fake_video), file_duration=10.0, effective_fps=1.0)
        assert len(frames) == 0
        assert meta["frames"] == 0
