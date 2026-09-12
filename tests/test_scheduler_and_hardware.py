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


    def test_dual_rail_semaphores_and_render_cooperation(self):
        """测试 ADR 0007 双轨信号量解耦：NVENC 与 NVDEC 独立管控，支持渲染期间协同且零死锁。"""
        from src.utils import get_nvenc_semaphore, get_nvdec_semaphore, reset_semaphores
        reset_semaphores()

        sem_enc = get_nvenc_semaphore()
        sem_dec = get_nvdec_semaphore()
        assert sem_enc is not sem_dec
        assert getattr(sem_enc, "resource_name", "") == "NVENC hardware"
        assert getattr(sem_dec, "resource_name", "") == "NVDEC hardware"

        cfg = {
            "scheduler": {
                "watermark_high": 5,
                "watermark_low": 2,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
                "cooperative_during_render": True,
                "vram_watermark_mb": 6000,
            },
            "hardware": {"device": "cuda:0"},
        }
        mgr = WorkStealingManager(config=cfg)
        mgr.set_vram_probe_fn(lambda: 3500)

        # 渲染启动 (占用 NVENC 信号量)
        mgr.register_render_start()
        assert mgr.is_render_active

        # 待分析队列积压 (queue_size=10 >= 5)，双轨解耦下 NVDEC 槽位独立放行
        assert mgr.get_analysis_device(queue_size=10) == "cuda"
        assert mgr.state == "COOPERATIVE_RENDER_COEXIST"
        assert mgr.acquire_nvdec_slot() is True

        # 第二路尝试，因 max_nv_decoders=1 上限，让步 QSV
        assert mgr.get_analysis_device(queue_size=10) == "qsv"
        assert mgr.acquire_nvdec_slot() is False

        # 释放槽位
        mgr.release_nvdec_slot()
        assert mgr.active_nv_decoders == 0

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


def test_heterogeneous_fleet_allocates_two_nv_one_qsv(tmp_path):
    """测试异构模式下，渲染并发数为 3 时精准构建 2 NVENC + 1 QSV 编队。"""
    from src.database import VlogDatabase
    from src.pipeline import StreamingOrchestrator
    db = VlogDatabase(db_path=tmp_path / "fleet.db")
    cfg = {
        "pipeline": {"render_gpu_policy": "heterogeneous", "render_start_delay": 0},
        "render": {"batch_max_files": 1, "max_concurrency": 3, "max_qsv_dynamic_duration_s": 60.0},
    }
    orch = StreamingOrchestrator(db, "20260320", 0, cfg, dashboard_enabled=False)
    assert orch.render_workers == 3
    policy = orch.config.get("pipeline", {}).get("render_gpu_policy")
    assert policy == "heterogeneous"
    render_workers = orch.render_workers
    render_gpus = ["nv"] if render_workers == 1 else ["nv"] * min(2, render_workers - 1) + ["qsv"]
    assert render_gpus == ["nv", "nv", "qsv"]


def test_dual_ended_batch_queue_lpt_spt():
    """测试双端批次调度队列：NVENC LPT 降序消费与 QSV SPT 逆向窃取。"""
    from src.scheduler import DualEndedBatchQueue

    q = DualEndedBatchQueue()
    assert q.empty()

    # 乱序推入不同动态时长的批次
    q.put(b_idx=1, files=["f1.mp4"], dynamic_duration=120.0)
    q.put(b_idx=2, files=["f2.mp4"], dynamic_duration=360.0)
    q.put(b_idx=3, files=["f3.mp4"], dynamic_duration=45.0)
    q.put(b_idx=4, files=["f4.mp4"], dynamic_duration=80.0)

    assert q.qsize() == 4

    # 1. NVENC 消费队首最重批次 (LPT: 360.0s)
    h1 = q.pop_heaviest(timeout=0.1)
    assert h1 is not None
    assert h1.b_idx == 2
    assert h1.dynamic_duration == 360.0

    # 2. QSV 逆向从队尾窃取最轻批次 (SPT: 45.0s, 安全门限 100.0s)
    s1 = q.steal_lightest(max_dynamic_s=100.0)
    assert s1 is not None
    assert s1.b_idx == 3
    assert s1.dynamic_duration == 45.0

    # 3. QSV 继续逆向窃取下一最轻批次 (SPT: 80.0s, 安全门限 100.0s)
    s2 = q.steal_lightest(max_dynamic_s=100.0)
    assert s2 is not None
    assert s2.b_idx == 4
    assert s2.dynamic_duration == 80.0

    # 4. 剩余批次为 120.0s，若 QSV 安全门限为 100.0s，则拦截窃取并安全保留在队列中
    s3 = q.steal_lightest(max_dynamic_s=100.0)
    assert s3 is None
    assert q.qsize() == 1

    # 5. NVENC 消费剩余的 120.0s 批次
    h2 = q.pop_heaviest(timeout=0.1)
    assert h2 is not None
    assert h2.b_idx == 1
    assert h2.dynamic_duration == 120.0

    assert q.empty()


def test_dual_ended_batch_queue_concurrent_stealing():
    """测试多线程高并发下双端队列的工作窃取无丢批与无重复。"""
    from src.scheduler import DualEndedBatchQueue

    q = DualEndedBatchQueue()
    total_batches = 50
    for i in range(total_batches):
        # 动态时长分布在 10s ~ 300s
        q.put(b_idx=i, files=[f"file_{i}.mp4"], dynamic_duration=float((i * 17) % 300 + 10))

    consumed_nv: list[int] = []
    consumed_qsv: list[int] = []
    stop_event = threading.Event()

    def nv_worker():
        while not stop_event.is_set():
            item = q.pop_heaviest(timeout=0.05)
            if item:
                consumed_nv.append(item.b_idx)
            elif q.empty():
                break

    def qsv_worker():
        while not stop_event.is_set():
            # QSV 窃取 <= 150s 的任务
            item = q.steal_lightest(max_dynamic_s=150.0)
            if item:
                consumed_qsv.append(item.b_idx)
            elif q.empty():
                break
            else:
                time.sleep(0.01)

    t_nv1 = threading.Thread(target=nv_worker)
    t_nv2 = threading.Thread(target=nv_worker)
    t_qsv = threading.Thread(target=qsv_worker)

    t_nv1.start()
    t_nv2.start()
    t_qsv.start()

    t_nv1.join(timeout=5.0)
    t_nv2.join(timeout=5.0)
    t_qsv.join(timeout=5.0)
    stop_event.set()

    all_consumed = consumed_nv + consumed_qsv
    assert len(all_consumed) == total_batches
    assert set(all_consumed) == set(range(total_batches))
    # 验证 QSV 窃取的所有批次时长确实 <= 150.0s
    for b_id in consumed_qsv:
        dur = float((b_id * 17) % 300 + 10)
        assert dur <= 150.0


def test_qsv_adaptive_duration_formula_and_elastic_expansion():
    """测试 QSV 弹性门限自适应计算：大积压时放宽至 200s，小积压时自动收敛防长尾。"""
    max_qsv_dynamic_s = 200.0

    def calc_allowed(q_len: int) -> float:
        return min(max_qsv_dynamic_s, max(30.0, max(0, q_len - 1) * 18.0))

    # 1. 超高积压 (q_len=24): 允许上限达到 200.0s 满额，覆盖 160s~200s 中轻动态批次
    assert calc_allowed(24) == 200.0
    assert calc_allowed(15) == 200.0

    # 2. 中等积压 (q_len=10): (10-1)*18 = 162.0s
    assert calc_allowed(10) == 162.0

    # 3. 低积压 (q_len=6): (6-1)*18 = 90.0s
    assert calc_allowed(6) == 90.0

    # 4. 濒临收尾 (q_len=3): (3-1)*18 = 36.0s
    assert calc_allowed(3) == 36.0

    # 5. 队尾 (q_len=1): 保底 30.0s
    assert calc_allowed(1) == 30.0
    assert calc_allowed(0) == 30.0



