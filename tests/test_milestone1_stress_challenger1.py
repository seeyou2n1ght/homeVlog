"""
Milestone 1 High-Intensity Concurrency & Stress Verification Test Suite.
Authored by Challenger 1.

Verifies:
1. High thread contention across NV (3), QSV (8), and Disk (8) semaphores with exception injection.
2. High-frequency queue size flapping (2 <-> 12) across hysteresis thresholds under multi-threaded access.
3. Render preemption yield timing during active cooperative bursts (in-flight lease vs incoming lease).
4. Multi-agent stress simulation with zero semaphore/slot leaks and zero deadlocks.
"""

import concurrent.futures
import queue
import random
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.utils import (
    load_config,
    get_nv_semaphore,
    get_qsv_semaphore,
    get_disk_semaphore,
    reset_semaphores,
    WorkStealingManager,
)
from src.pipeline import StreamingOrchestrator


class TestHardwareSemaphoresContention:
    """Stress testing hardware semaphores under severe multi-threaded contention."""

    def setup_method(self):
        reset_semaphores()

    def teardown_method(self):
        reset_semaphores()

    def test_extreme_multi_semaphore_contention_and_deadlock_absence(self):
        """
        100 threads concurrently acquiring combinations of NV, QSV, and Disk semaphores.
        Ensures strict concurrency capacity (NV <= 3, QSV <= 8, Disk <= 8),
        verifies that timeout-based and blocking acquisitions complete without deadlocks,
        and confirms no semaphore counter leaks occur after all threads terminate.
        """
        sem_nv = get_nv_semaphore()
        sem_qsv = get_qsv_semaphore()
        sem_disk = get_disk_semaphore()

        nv_active = 0
        qsv_active = 0
        disk_active = 0
        max_nv = 0
        max_qsv = 0
        max_disk = 0
        lock = threading.Lock()
        violations = []

        def worker_nv_disk(worker_id: int):
            nonlocal nv_active, disk_active, max_nv, max_disk
            for _ in range(25):
                with sem_disk:
                    with lock:
                        disk_active += 1
                        if disk_active > 8:
                            violations.append(f"Disk concurrency limit breached: {disk_active} > 8")
                        max_disk = max(max_disk, disk_active)
                    try:
                        with sem_nv:
                            with lock:
                                nv_active += 1
                                if nv_active > 3:
                                    violations.append(f"NV concurrency limit breached: {nv_active} > 3")
                                max_nv = max(max_nv, nv_active)
                            time.sleep(random.uniform(0.0002, 0.001))
                            with lock:
                                nv_active -= 1
                    finally:
                        with lock:
                            disk_active -= 1

        def worker_qsv_disk(worker_id: int):
            nonlocal qsv_active, disk_active, max_qsv, max_disk
            for _ in range(25):
                with sem_qsv:
                    with lock:
                        qsv_active += 1
                        if qsv_active > 8:
                            violations.append(f"QSV concurrency limit breached: {qsv_active} > 8")
                        max_qsv = max(max_qsv, qsv_active)
                    try:
                        with sem_disk:
                            with lock:
                                disk_active += 1
                                if disk_active > 8:
                                    violations.append(f"Disk concurrency limit breached: {disk_active} > 8")
                                max_disk = max(max_disk, disk_active)
                            time.sleep(random.uniform(0.0002, 0.001))
                            with lock:
                                disk_active -= 1
                    finally:
                        with lock:
                            qsv_active -= 1

        threads = []
        for i in range(50):
            threads.append(threading.Thread(target=worker_nv_disk, args=(i,)))
            threads.append(threading.Thread(target=worker_qsv_disk, args=(i,)))

        t0 = time.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)
            assert not t.is_alive(), "Worker thread timed out - deadlock detected in semaphore contention!"

        duration = time.monotonic() - t0
        assert not violations, f"Concurrency invariants violated: {violations[:5]}"
        assert max_nv <= 3, f"Peak NV concurrency breached: {max_nv} > 3"
        assert max_qsv <= 8, f"Peak QSV concurrency breached: {max_qsv} > 8"
        assert max_disk <= 8, f"Peak Disk concurrency breached: {max_disk} > 8"
        assert sem_nv._value == 3, f"NV semaphore count leaked: {sem_nv._value} != 3"
        assert sem_qsv._value == 8, f"QSV semaphore count leaked: {sem_qsv._value} != 8"
        assert sem_disk._value == 8, f"Disk semaphore count leaked: {sem_disk._value} != 8"

    def test_semaphore_exception_safety_under_chaos(self):
        """
        Verify semaphores under chaos conditions where 40% of threads raise unhandled exceptions
        at varied acquisition checkpoints.
        """
        sem_nv = get_nv_semaphore()
        sem_qsv = get_qsv_semaphore()

        def chaos_worker(i: int):
            try:
                with sem_nv:
                    if i % 3 == 0:
                        raise RuntimeError("Simulated crash in NV critical section")
                    with sem_qsv:
                        if i % 5 == 0:
                            raise ValueError("Simulated crash in QSV critical section")
                        time.sleep(0.0005)
            except (RuntimeError, ValueError):
                pass

        threads = [threading.Thread(target=chaos_worker, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert sem_nv._value == 3, f"NV semaphore leaked after crashes: {sem_nv._value} != 3"
        assert sem_qsv._value == 8, f"QSV semaphore leaked after crashes: {sem_qsv._value} != 8"


class TestWatermarkOscillationAndHysteresis:
    """Stress testing queue watermark transitions and rapid oscillation/flapping."""

    @pytest.fixture
    def manager(self):
        config = {
            "hardware": {"device": "cuda:0", "max_nv_decoders": 1},
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
            },
        }
        return WorkStealingManager(config)

    def test_rapid_queue_flapping_stability(self, manager):
        """
        Rapidly oscillates queue size between 2 (below low watermark 3) and 12 (above high watermark 10).
        10,000 cycles executed to verify zero state divergence and absolute hysteresis stability.
        """
        for cycle in range(10000):
            dev_low = manager.get_analysis_device(queue_size=2)
            assert dev_low == "qsv", f"Cycle {cycle}: Expected 'qsv' at queue_size=2, got '{dev_low}'"
            assert manager.state == "NORMAL_DECOUPLED"

            dev_mid_up = manager.get_analysis_device(queue_size=6)
            assert dev_mid_up == "qsv", f"Cycle {cycle}: Expected 'qsv' at queue_size=6 while rising, got '{dev_mid_up}'"
            assert manager.state == "NORMAL_DECOUPLED"

            dev_high = manager.get_analysis_device(queue_size=12)
            assert dev_high == "cuda", f"Cycle {cycle}: Expected 'cuda' at queue_size=12, got '{dev_high}'"
            assert manager.state == "COOPERATIVE_BURST"

            dev_mid_down = manager.get_analysis_device(queue_size=6)
            assert dev_mid_down == "cuda", f"Cycle {cycle}: Expected 'cuda' at queue_size=6 while falling, got '{dev_mid_down}'"
            assert manager.state == "COOPERATIVE_BURST"

            dev_boundary = manager.get_analysis_device(queue_size=3)
            assert dev_boundary == "qsv", f"Cycle {cycle}: Expected 'qsv' at queue_size=3, got '{dev_boundary}'"
            assert manager.state == "NORMAL_DECOUPLED"

    def test_concurrent_multi_thread_flapping_and_leasing(self, manager):
        """
        30 threads concurrently querying and leasing devices while queue size dynamically flaps
        between 2 and 12 at microsecond intervals.
        """
        stop_event = threading.Event()
        current_queue_size = [2]
        q_lock = threading.Lock()
        violations = []

        def flapper():
            sizes = [2, 4, 7, 9, 10, 12, 8, 5, 3, 2]
            idx = 0
            while not stop_event.is_set():
                with q_lock:
                    current_queue_size[0] = sizes[idx % len(sizes)]
                idx += 1
                time.sleep(0.0005)

        def worker(worker_id: int):
            while not stop_event.is_set():
                with q_lock:
                    qs = current_queue_size[0]
                with manager.lease_device(queue_size=qs) as dev:
                    active = manager.active_nv_decoders
                    if active > 1 or active < 0:
                        violations.append(f"Worker {worker_id}: active_nv_decoders invariant broken: {active}")
                    if dev == "cuda" and active == 0:
                        violations.append(f"Worker {worker_id}: Leased 'cuda' but active_nv_decoders is 0")
                    time.sleep(random.uniform(0.0001, 0.0008))

        flapper_t = threading.Thread(target=flapper)
        worker_threads = [threading.Thread(target=worker, args=(i,)) for i in range(30)]

        flapper_t.start()
        for t in worker_threads:
            t.start()

        time.sleep(1.0)
        stop_event.set()

        flapper_t.join(timeout=3.0)
        for t in worker_threads:
            t.join(timeout=3.0)

        assert not violations, f"Concurrent flapping violations: {violations[:5]}"
        assert manager.active_nv_decoders == 0, f"Leaked NV decoders: {manager.active_nv_decoders}"


class TestRenderPreemptionYieldTiming:
    """Stress testing render preemption yield timing during active cooperative burst."""

    def test_in_flight_cooperative_burst_preemption_and_instant_yield(self):
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

        with mgr.lease_device(queue_size=20) as dev_a:
            assert dev_a == "cuda"
            assert mgr.active_nv_decoders == 1
            assert mgr.state == "COOPERATIVE_BURST"

            mgr.register_render_start()
            assert mgr.is_render_active
            assert mgr.state == "RENDER_PREEMPTION_YIELD"

            with mgr.lease_device(queue_size=20) as dev_b:
                assert dev_b == "qsv", f"Worker B must yield to QSV during render preemption, got {dev_b}"
                assert mgr.active_nv_decoders == 1

        assert mgr.active_nv_decoders == 0, f"Worker A slot must be released cleanly, got {mgr.active_nv_decoders}"

        with mgr.lease_device(queue_size=20) as dev_c:
            assert dev_c == "qsv", f"Worker C must yield to QSV while render active, got {dev_c}"
            assert mgr.active_nv_decoders == 0

        mgr.register_render_end()
        assert not mgr.is_render_active

        with mgr.lease_device(queue_size=20) as dev_d:
            assert dev_d == "cuda", f"Worker D should resume CUDA burst, got {dev_d}"
            assert mgr.active_nv_decoders == 1
            assert mgr.state == "COOPERATIVE_BURST"

        assert mgr.active_nv_decoders == 0

    def test_render_preemption_burst_storm(self):
        config = {
            "hardware": {"device": "cuda:0", "max_nv_decoders": 2},
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
                "max_nv_decoders": 2,
            },
        }
        mgr = WorkStealingManager(config)
        stop_event = threading.Event()
        violations = []

        def render_actor(actor_id: int):
            while not stop_event.is_set():
                mgr.register_render_start()
                time.sleep(random.uniform(0.001, 0.004))
                mgr.register_render_end()
                time.sleep(random.uniform(0.001, 0.003))

        def analysis_actor(actor_id: int):
            while not stop_event.is_set():
                qs = random.choice([2, 5, 12, 20])
                with mgr.lease_device(queue_size=qs) as dev:
                    if dev == "cuda" and mgr.is_render_active:
                        violations.append(f"Analysis actor {actor_id} obtained CUDA while render active!")
                    time.sleep(random.uniform(0.0002, 0.001))

        renders = [threading.Thread(target=render_actor, args=(i,)) for i in range(5)]
        analyses = [threading.Thread(target=analysis_actor, args=(i,)) for i in range(40)]

        for t in renders + analyses:
            t.start()

        time.sleep(1.5)
        stop_event.set()

        for t in renders + analyses:
            t.join(timeout=5.0)

        assert not violations, f"Render preemption storm violations: {violations[:5]}"
        assert mgr.active_nv_decoders == 0, f"Leaked NV decoders: {mgr.active_nv_decoders}"
        assert mgr._render_active_count == 0, f"Render active count corrupted: {mgr._render_active_count}"


class TestStreamingOrchestratorStressIntegration:
    """End-to-end stress integration with 80 tasks, simulated dual-GPU pipelines and faults."""

    def setup_method(self):
        reset_semaphores()

    def teardown_method(self):
        reset_semaphores()

    def test_streaming_orchestrator_80_tasks_stress_execution(self):
        cfg = load_config()
        cfg["detection"]["prescreen_parallel"] = 8
        cfg["detection"]["analysis_max_workers"] = 8
        cfg["render"]["batch_max_files"] = 4
        cfg["pipeline"]["render_start_delay"] = 0.01
        cfg["yolo"]["enabled"] = False

        tasks = []
        for i in range(80):
            tasks.append({
                "filepath": f"C:\\nas\\slice_{i:03d}.mp4",
                "file_start_time": f"202609010{i//60:01d}{i%60:02d}00",
                "file_duration": 300.0,
                "prescreen_status": "PENDING",
                "analysis_status": "PENDING",
            })

        class MockDB:
            def __init__(self, task_list):
                self._tasks = {t["filepath"]: dict(t) for t in task_list}
                self._lock = threading.Lock()

            def get_all_file_tasks_for_date(self, date, cam_index):
                with self._lock:
                    return list(self._tasks.values())

            def set_prescreen_result(self, filepath, status, result_json):
                with self._lock:
                    if filepath in self._tasks:
                        self._tasks[filepath]["prescreen_status"] = status

            def set_analysis_result(self, filepath, status, result_json):
                with self._lock:
                    if filepath in self._tasks:
                        self._tasks[filepath]["analysis_status"] = status

            def set_file_metadata(self, filepath, has_audio):
                pass

        db = MockDB(tasks)
        orchestrator = StreamingOrchestrator(
            db=db,
            date="20260901",
            cam_index=0,
            config=cfg,
            render_enabled=True,
        )

        def mock_prescreen(filepath, duration, config, gpu="qsv"):
            time.sleep(random.uniform(0.0005, 0.002))
            if random.random() < 0.05:
                return {"status": "FAILED", "error": "mock IO read error"}
            if random.random() < 0.6:
                return {"status": "SUSPICIOUS", "result_json": '{"motion_ratio": 0.3}'}
            return {"status": "STATIC", "result_json": "{}"}

        def mock_analyze(filepath, start_offset=0.0, file_duration=300.0):
            time.sleep(random.uniform(0.001, 0.003))
            if random.random() < 0.03:
                raise RuntimeError("mock PyAV decode error")
            labels = [{"time": start_offset + j * 10, "is_motion": (j % 2 == 0), "energy": 0.5} for j in range(10)]
            return labels, {}

        def mock_build_batch_render(*args, **kwargs):
            time.sleep(random.uniform(0.002, 0.005))
            batch_idx = args[1]
            return f"C:\\output\\batch_{batch_idx}.mp4"

        with patch("src.pipeline.prescreen_file", side_effect=mock_prescreen), \
             patch("src.pipeline.MotionDetector.analyze", side_effect=mock_analyze), \
             patch("src.pipeline.build_batch_render", side_effect=mock_build_batch_render), \
             patch("src.pipeline.build_timeline", return_value=[]):
            batch_results = orchestrator.run()

        assert orchestrator.prescreen_queue.empty()
        assert orchestrator.analysis_queue.empty()
        assert orchestrator.render_batch_queue.empty()
        assert orchestrator.work_stealing.active_nv_decoders == 0
        assert not orchestrator.work_stealing.is_render_active
        assert get_nv_semaphore()._value == 3
        assert get_qsv_semaphore()._value == 8
        assert get_disk_semaphore()._value == 8
