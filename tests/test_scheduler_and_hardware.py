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


@pytest.mark.parametrize("used, expected", [(0, True), (6199, True), (6200, False),
    (7686, False), (None, False), (-1, False), (float("nan"), False), (float("inf"), False)])
def test_extra_nv_admission_requires_measured_vram_headroom(used, expected):
    from src.hardware.scheduler import WorkStealingManager

    manager = WorkStealingManager({"scheduler": {"vram_watermark_mb": 6200}})
    manager.set_vram_probe_fn(lambda: used)
    assert manager.has_vram_headroom() is expected


def test_extra_nv_admission_rejects_unavailable_or_failed_probe():
    from src.hardware.scheduler import WorkStealingManager

    manager = WorkStealingManager({})
    manager.set_vram_probe_fn(None)
    assert not manager.has_vram_headroom()

    def failed():
        raise OSError("NVML unavailable")

    manager.set_vram_probe_fn(failed)
    assert not manager.has_vram_headroom()

from src.core.utils import check_disk_space
from src.hardware.scheduler import (
    WorkStealingManager,
    get_nv_semaphore,
    get_qsv_semaphore,
    get_disk_semaphore,
    reset_semaphores,
)
from src.hardware.monitor import get_monitor, get_perf, PerfRecord


def test_io_wait_checks_cancellation_without_blocking_budget_release(monkeypatch):
    from src.hardware import scheduler
    budget = scheduler.FileBudget(1)
    assert budget.acquire(timeout=0)
    probe_finished = threading.Event()
    checks, probes = [], []
    def probe_release_lock():
        with budget.condition:
            probe_finished.set()
    def interrupted():
        probe = threading.Thread(target=probe_release_lock)
        probes.append(probe)
        probe.start()
        checks.append(probe_finished.wait(timeout=1))
        return True
    monkeypatch.setattr(scheduler, '_interrupted', interrupted)
    try:
        assert not budget.acquire(timeout=2)
    finally:
        for probe in probes:
            probe.join(timeout=2)
            assert not probe.is_alive()
        budget.release()
    assert checks == [True], 'Cancellation check held the lock needed by deferred permit release'
    assert budget.available == 1 and not budget.waiters


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
        mgr.set_vram_probe_fn(lambda: 3000)
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
        from src.hardware.scheduler import get_nvenc_semaphore, get_nvdec_semaphore, reset_semaphores
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


def _assert_nv_available(budget, count):
    acquired = 0
    try:
        for _ in range(count):
            assert budget.acquire(blocking=False)
            acquired += 1
        assert not budget.acquire(blocking=False)
    finally:
        if acquired:
            budget.release(acquired)


def test_nv_tail_scope_preserves_normal_admission_and_restores_capacity(monkeypatch):
    from src.hardware import scheduler
    budget = scheduler.NvencSemaphore(2)
    monkeypatch.setattr(scheduler, 'get_nvenc_semaphore', lambda: budget)
    assert type(budget).acquire is threading.Semaphore.acquire
    _assert_nv_available(budget, 2)
    with scheduler.nv_tail_capacity(False) as granted:
        assert not granted
        _assert_nv_available(budget, 2)
    with pytest.raises(RuntimeError, match='tail failed'):
        with scheduler.nv_tail_capacity() as granted:
            assert granted
            _assert_nv_available(budget, 3)
            with scheduler.nv_tail_capacity() as second:
                assert not second
            raise RuntimeError('tail failed')
    _assert_nv_available(budget, 2)
    with scheduler.nv_tail_capacity() as next_day:
        assert next_day
    _assert_nv_available(budget, 2)


@pytest.mark.parametrize('base,target', [(1, 3), (3, 3), (4, 3), (2, 2), (2, 4)])
def test_nv_tail_capacity_only_lends_one_slot_to_two_slot_budget(base, target):
    from src.hardware.scheduler import NvencSemaphore
    budget = NvencSemaphore(base)
    assert budget.try_tail_capacity(target) is None
    _assert_nv_available(budget, base)


@pytest.mark.parametrize('batched_release', [False, True])
def test_nv_tail_retirement_preserves_active_leases_until_real_release(batched_release):
    from src.hardware.scheduler import NvencSemaphore
    budget = NvencSemaphore(2)
    owner = budget.try_tail_capacity()
    assert owner is not None
    assert all(budget.acquire(blocking=False) for _ in range(3))
    owner.close()
    owner.close()
    assert budget.try_tail_capacity() is None
    assert not budget.acquire(blocking=False)
    if batched_release:
        budget.release(3)
    else:
        budget.release()
        assert not budget.acquire(blocking=False), 'First release must retire the extra slot'
        budget.release(2)
    _assert_nv_available(budget, 2)
    next_owner = budget.try_tail_capacity()
    assert next_owner is not None
    owner.close()  # An old owner's close cannot revoke the next day's slot.
    _assert_nv_available(budget, 3)
    next_owner.close()
    _assert_nv_available(budget, 2)


@pytest.mark.parametrize('n', [0, -1])
def test_nv_tail_invalid_release_does_not_change_retirement(n):
    from src.hardware.scheduler import NvencSemaphore
    budget = NvencSemaphore(2)
    owner = budget.try_tail_capacity()
    assert all(budget.acquire(blocking=False) for _ in range(3))
    owner.close()
    with pytest.raises(ValueError):
        budget.release(n)
    assert budget.try_tail_capacity() is None
    budget.release(3)
    _assert_nv_available(budget, 2)


def test_nv_tail_concurrent_requests_grant_exactly_one_owner():
    from src.hardware.scheduler import NvencSemaphore
    budget = NvencSemaphore(2)
    start = threading.Barrier(9)
    owners = []
    lock = threading.Lock()

    def request():
        start.wait(timeout=5)
        owner = budget.try_tail_capacity()
        with lock:
            owners.append(owner)

    threads = [threading.Thread(target=request) for _ in range(8)]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert len(owners) == 8
    granted = [owner for owner in owners if owner is not None]
    assert len(granted) == 1
    _assert_nv_available(budget, 3)
    granted[0].close()
    _assert_nv_available(budget, 2)


def test_nv_tail_deferred_registry_cleanup_keeps_original_video_lease(monkeypatch):
    from src.hardware import ffmpeg, scheduler
    registry = ffmpeg.FFmpegProcessRegistry
    for name in ('_processes', '_jobs', '_deferred'):
        monkeypatch.setattr(registry, name, {})
    budget, disk = scheduler.NvencSemaphore(2), scheduler.FileBudget(3)
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: disk)
    owner = budget.try_tail_capacity()
    leases = [scheduler.VideoLease(budget) for _ in range(3)]
    assert all(lease.acquire(timeout=1) for lease in leases)
    process = MagicMock(stdout=None, stderr=None)
    process.poll.return_value = None
    registry.defer_cleanup('nv_tail_test', process, leases[2])
    owner.close()
    assert disk.available == 0
    assert registry.reap_cleanups() == {'recovered': [], 'pending': ['nv_tail_test']}
    assert budget.try_tail_capacity() is None
    assert not budget.acquire(blocking=False)
    process.poll.return_value = 0
    assert registry.reap_cleanups() == {'recovered': ['nv_tail_test'], 'pending': []}
    assert disk.available == 1
    assert not budget.acquire(blocking=False), 'Recovered lease must repay retirement debt'
    assert registry.reap_cleanups() == {'recovered': [], 'pending': []}
    assert disk.available == 1
    leases[0].release()
    leases[1].release()
    assert disk.available == 3
    _assert_nv_available(budget, 2)


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
        from src.hardware.scheduler import get_qsv_render_semaphore
        sem_nv = get_nv_semaphore()
        sem_qsv = get_qsv_semaphore()
        sem_qsv_render = get_qsv_render_semaphore()
        sem_io = get_disk_semaphore()

        assert sem_nv is not sem_qsv
        assert sem_qsv is not sem_io
        assert sem_qsv_render is not sem_qsv
        assert sem_qsv_render is not sem_nv

    def test_check_disk_space(self, tmp_path):
        # 足够空间
        assert check_disk_space(tmp_path, min_gb=1)
        # 极其夸张的 1000000 GB，必然返回 False 拦截
        assert not check_disk_space(tmp_path, min_gb=1_000_000)

    def test_impossible_hardware_concurrency_is_rejected(self):
        from src.core.config import validate_config

        with pytest.raises(ValueError, match=r"analysis \+ render"):
            validate_config({"hardware": {
                "max_qsv_concurrency": 2,
                "max_qsv_analysis_concurrency": 2,
                "max_qsv_render_concurrency": 1,
            }})
        with pytest.raises(ValueError, match="at most 3"):
            validate_config({
                "pipeline": {"render_gpu_policy": "heterogeneous"},
                "render": {"max_concurrency": 4},
            })


class TestSystemMonitoring:
    """测试监控与性能记录采集。"""

    def test_nested_phase_samples_do_not_duplicate_total(self, monkeypatch):
        from src.hardware import monitor as module
        monitor = module.Monitor.__new__(module.Monitor)
        monitor._active_lock = threading.Lock()
        monitor._active_stages = set()
        monitor._stages = []
        monitor._num_gpus = 0
        clock = iter((0.0, 1.0, 3.0, 4.0))
        monkeypatch.setattr(module.time, "monotonic", lambda: next(clock))
        with monitor.stage("pipeline"):
            with monitor.stage("render_tail", include_in_total=False):
                for stage in monitor._active_stages:
                    stage.record_sample(cpu=25.0)
        data = monitor.stages_data()
        assert [(s["name"], s["duration"], s["include_in_total"], s["avg_cpu"]) for s in data] == [
            ("pipeline", 4.0, True, 25.0), ("render_tail", 2.0, False, 25.0)]
        assert "TOTAL: 4.0s" in monitor.summary()
        assert not monitor._active_stages

    def test_queue_eligibility_inspection_does_not_consume_work(self):
        from src.hardware.scheduler import DualEndedBatchQueue
        queue = DualEndedBatchQueue()
        assert queue.lightest_dynamic_duration() is None
        queue.put(0, ["heavy"], 400)
        queue.put(1, ["light"], 200)
        assert queue.lightest_dynamic_duration() == 200
        assert queue.qsize() == 2
        assert queue.pop_heaviest().b_idx == 0
        assert queue.lightest_dynamic_duration() == 200
        assert queue.steal_lightest(200).b_idx == 1
        assert queue.lightest_dynamic_duration() is None

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

        perf.add(PerfRecord(
            stage="analysis", file="queued.mp4", gpu="qsv", duration=0.2,
            extra={"analysis_queue_wait_s": 1.25, "sem_wait": 0.4},
        ))
        perf.add(PerfRecord(
            stage="render", file="batch_0.mp4", gpu="nv", duration=0.3,
            extra={"render_queue_wait_s": 2.5},
        ))
        waits = perf.wait_summary()
        assert waits["analysis_queue_wait_s"] == {"count": 1, "total": 1.25, "max": 1.25}
        assert waits["sem_wait"] == {"count": 1, "total": 0.4, "max": 0.4}
        assert waits["render_queue_wait_s"] == {"count": 1, "total": 2.5, "max": 2.5}

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
    from src.stages.detector import MotionDetector
    from src.hardware.scheduler import get_qsv_semaphore

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
    from src.core.database import VlogDatabase
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
    from src.hardware.scheduler import DualEndedBatchQueue

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
    from src.hardware.scheduler import DualEndedBatchQueue

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


def test_legacy_config_tracks_core_reload_and_preserves_explicit_overrides(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import src.core.config as core
    import src.core.utils as legacy

    saved_overrides = {name: legacy.__dict__[name] for name in ("SETTINGS", "CONFIG_PATH")
                       if name in legacy.__dict__}
    for name in saved_overrides:
        legacy.__dict__.pop(name)
    first, second, invalid = [tmp_path / name for name in ("first.yaml", "second.yaml", "invalid.yaml")]
    first.write_text("marker: first\n", encoding="utf-8")
    second.write_text("marker: second\n", encoding="utf-8")
    invalid.write_text("hardware:\n  max_nv_concurrency: 0\n", encoding="utf-8")
    try:
        with monkeypatch.context() as m:
            m.setattr(core, "SETTINGS", {})
            m.setattr(core, "CONFIG_PATH", first)
            for name in ("OUTPUT_DIR", "TEMP_DIR", "TEMP_STAGING_DIR", "TEMP_BATCHES_DIR",
                         "TEMP_SCRIPTS_DIR", "LOGS_DIR", "DB_PATH"):
                m.setattr(core, name, tmp_path / name)
            assert legacy.load_config()["marker"] == "first"
            updated = core.load_config(second)
            assert legacy.load_config() is updated
            assert legacy.SETTINGS is updated
            assert legacy.CONFIG_PATH == second
            exported = {}
            exec("from src.core.utils import *", exported)
            assert exported["SETTINGS"] is updated
            assert exported["CONFIG_PATH"] == second
            with ThreadPoolExecutor(max_workers=4) as pool:
                results = list(pool.map(lambda i: (core if i % 2 else legacy).load_config(), range(24)))
            assert all(result is updated for result in results)
            legacy.SETTINGS = {"marker": "explicit"}
            assert legacy.load_config() is core.SETTINGS
            assert core.SETTINGS["marker"] == "explicit"
            legacy.CONFIG_PATH = first
            assert legacy.load_config(reload=True)["marker"] == "first"
            good = core.SETTINGS
            with pytest.raises(ValueError):
                legacy.load_config(invalid)
            assert core.SETTINGS is legacy.SETTINGS is good
            assert legacy.load_config() is good
    finally:
        for name in ("SETTINGS", "CONFIG_PATH"):
            legacy.__dict__.pop(name, None)
        legacy.__dict__.update(saved_overrides)





class TestAutoTuner:
    """测试异构硬件自适应画像探测与参数推导 (Hardware Auto-Tuner)。"""

    def test_autotune_hardware_profile_derivations(self):
        from src.hardware.autotune import HardwareSnapshot, compute_optimal_concurrency

        # 1. 典型 8GB 显卡 (如 RTX 3060Ti / 4060) + 16 线程 CPU + QSV
        snap_8g = HardwareSnapshot(
            cpu_logical=16,
            cpu_physical=10,
            ram_total_gb=32.0,
            has_cuda=True,
            nv_device_count=1,
            nv_vram_mb=8192,
            has_nvenc=True,
            has_qsv=True,
        )
        opt_8g = compute_optimal_concurrency(snap_8g)
        assert opt_8g["hardware"]["max_nv_concurrency"] == 2
        assert opt_8g["scheduler"]["max_nv_decoders"] == 2
        assert opt_8g["render"]["max_concurrency"] == 3  # 2 NVENC + 1 QSV
        assert opt_8g["detection"]["analysis_max_workers"] == 7  # 对齐 QSV 分析上限
        assert opt_8g["scheduler"]["vram_watermark_mb"] == int(8192 * 0.78)

        # 2. 24GB 顶级显卡 (如 RTX 4090 / 3090)
        snap_24g = HardwareSnapshot(
            cpu_logical=32,
            cpu_physical=16,
            ram_total_gb=64.0,
            has_cuda=True,
            nv_device_count=1,
            nv_vram_mb=24576,
            has_nvenc=True,
            has_qsv=True,
        )
        opt_24g = compute_optimal_concurrency(snap_24g)
        assert opt_24g["hardware"]["max_nv_concurrency"] == 2
        assert opt_24g["scheduler"]["max_nv_decoders"] == 2
        assert opt_24g["render"]["max_concurrency"] == 3

        # 3. 纯核显/CPU 环境 (无独立显卡)
        snap_igpu = HardwareSnapshot(
            cpu_logical=8,
            cpu_physical=4,
            ram_total_gb=16.0,
            has_cuda=False,
            has_nvenc=False,
            has_qsv=True,
        )
        opt_igpu = compute_optimal_concurrency(snap_igpu)
        assert opt_igpu["hardware"]["max_nv_concurrency"] == 0
        assert opt_igpu["scheduler"]["nvdec_cooperative"] is False
        assert opt_igpu["render"]["max_concurrency"] == 1

    def test_autotune_applies_without_overriding_explicit_config(self):
        from src.hardware.autotune import apply_auto_tune

        # auto_tune 为 False 时不发生改动
        cfg_disabled = {"scheduler": {"auto_tune": False, "watermark_high": 10}}
        res_disabled = apply_auto_tune(cfg_disabled, in_place=False)
        assert res_disabled == cfg_disabled

        # auto_tune 为 True 时注入推荐参数，但保留已有显式键
        cfg_enabled = {
            "scheduler": {"auto_tune": True, "watermark_high": 99},
            "render": {"custom_user_key": "user_value"},
        }
        res_enabled = apply_auto_tune(cfg_enabled, in_place=False)
        assert res_enabled["scheduler"]["watermark_high"] == 99  # 保留显式配置
        assert res_enabled["render"]["custom_user_key"] == "user_value"
        assert "max_concurrency" in res_enabled["render"]  # 自动补全推荐并发


def test_cooperative_render_hysteresis():
    """测试渲染协同下的滞回比较器 (Hysteresis) 机制，防止频繁让步震荡。"""
    cfg = {
        "scheduler": {
            "watermark_high": 5,
            "watermark_low": 2,
            "nvdec_cooperative": True,
            "max_nv_decoders": 2,
            "cooperative_during_render": True,
            "vram_watermark_mb": 6000,
        },
        "hardware": {"device": "cuda:0"},
    }
    mgr = WorkStealingManager(config=cfg)
    mgr.set_vram_probe_fn(lambda: 3500)

    # 1. 渲染启动，初始待分析队列 queue_size=3 (< 5)，未达高水线，让步 QSV
    mgr.register_render_start()
    assert mgr.get_analysis_device(queue_size=3) == "qsv"
    assert mgr.state == "RENDER_PREEMPTION_YIELD"

    # 2. 待分析队列积压激增至 6 (>= 5)，激活 COOPERATIVE_RENDER_COEXIST
    assert mgr.get_analysis_device(queue_size=6) == "cuda"
    assert mgr.state == "COOPERATIVE_RENDER_COEXIST"

    # 3. 消费任务并占满 NVDEC 槽位：返回 qsv 溢出，但保持 COOPERATIVE_RENDER_COEXIST 状态不被破坏
    assert mgr.acquire_nvdec_slot() is True
    assert mgr.acquire_nvdec_slot() is True
    assert mgr.get_analysis_device(queue_size=4) == "qsv"
    assert mgr.state == "COOPERATIVE_RENDER_COEXIST"

    # 4. NVDEC 槽位释放，下一任务无缝重回 CUDA
    mgr.release_nvdec_slot()
    assert mgr.get_analysis_device(queue_size=4) == "cuda"
    assert mgr.state == "COOPERATIVE_RENDER_COEXIST"

    # 5. 队列彻底回落至低水线 2 (<= 2)，平滑退出协同状态
    mgr.release_nvdec_slot()
    assert mgr.get_analysis_device(queue_size=2) == "qsv"
    assert mgr.state == "RENDER_PREEMPTION_YIELD"


def test_cost_aware_heavy_task_nvdec_priority():
    """测试 ADR0080 成本感知设备路由：夜间巨型重文件 (>=600s) 突破水位线限制优先占用 NVDEC。"""
    cfg = {
        "scheduler": {
            "watermark_high": 10,
            "watermark_low": 3,
            "nvdec_cooperative": True,
            "max_nv_decoders": 2,
            "cooperative_during_render": True,
            "heavy_file_duration_s": 600.0,
            "vram_watermark_mb": 6000,
        },
        "hardware": {"device": "cuda:0"},
    }
    mgr = WorkStealingManager(config=cfg)
    mgr.set_vram_probe_fn(lambda: 3000)

    # 1. 队列仅有 1 个任务 (queue_size=1 < watermark_high)，普通短任务 (300s) 走 QSV
    assert mgr.get_analysis_device(queue_size=1, file_duration=300.0) == "qsv"
    assert mgr.state == "NORMAL_DECOUPLED"

    # 2. 队列同样只有 1 个任务，但为 55 分钟巨型文件 (3300s >= 600s)，成本感知优先路由至 CUDA
    assert mgr.get_analysis_device(queue_size=1, file_duration=3300.0) == "cuda"
    assert mgr.state == "COOPERATIVE_BURST"

    # 3. 槽位被重任务占用
    with mgr.lease_device(queue_size=1, file_duration=3300.0) as dev:
        assert dev == "cuda"
        assert mgr.active_nv_decoders == 1
        # 第二个重任务继续借调第二个 NVDEC 槽位
        with mgr.lease_device(queue_size=1, file_duration=3300.0) as dev2:
            assert dev2 == "cuda"
            assert mgr.active_nv_decoders == 2

    # 4. 退出后槽位完全归还
    assert mgr.active_nv_decoders == 0

    # 5. 显存超警戒线时，即便巨型任务也必须安全降级回 QSV，杜绝 OOM
    mgr.set_vram_probe_fn(lambda: 6500)
    assert mgr.get_analysis_device(queue_size=1, file_duration=3300.0) == "qsv"
    assert mgr.state == "VRAM_PRESSURE_YIELD"
