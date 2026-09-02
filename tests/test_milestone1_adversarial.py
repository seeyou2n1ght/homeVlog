import concurrent.futures
import queue
import random
import threading
import time
from pathlib import Path
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


class TestAdversarialWorkStealing:
    """Adversarial stress test harness for WorkStealingManager concurrency, watermarks, and preemption."""

    def test_high_concurrency_slot_invariants(self):
        """Stress test with 100 concurrent threads acquiring/leasing devices.
        
        Invariants checked:
        1. active_nv_decoders never exceeds max_nv_decoders (1).
        2. active_nv_decoders never drops below 0.
        3. All slots are released when workers exit.
        """
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
        violations = []
        stop_test = threading.Event()

        def worker(worker_id: int):
            for _ in range(50):
                if stop_test.is_set():
                    break
                q_size = random.choice([0, 5, 10, 15, 25, 50])
                with mgr.lease_device(q_size) as dev:
                    active = mgr.active_nv_decoders
                    if active > 1 or active < 0:
                        violations.append(f"Worker {worker_id}: Invalid active_nv_decoders = {active}")
                    if dev == "cuda" and active != 1:
                        violations.append(f"Worker {worker_id}: Device cuda leased but active = {active}")
                    # Simulate light task processing
                    time.sleep(random.uniform(0.0001, 0.001))

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert not violations, f"Slot invariant violations detected: {violations[:5]}"
        assert mgr.active_nv_decoders == 0, f"Leaked slots: {mgr.active_nv_decoders}"

    def test_rapid_render_preemption_chaos(self):
        """Simulate rapid, chaotic render start/end events while workers constantly lease devices.
        
        Invariants checked:
        1. When render is active, no CUDA slots can be acquired.
        2. No slot leaks or count drift occur during intense concurrent preemption toggling.
        """
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
        violations = []
        stop_event = threading.Event()

        def render_toggler(toggler_id: int):
            while not stop_event.is_set():
                mgr.register_render_start()
                time.sleep(random.uniform(0.001, 0.003))
                mgr.register_render_end()
                time.sleep(random.uniform(0.001, 0.003))

        def analysis_worker(worker_id: int):
            while not stop_event.is_set():
                # Backlog is high (20), would normally steal CUDA
                with mgr.lease_device(queue_size=20) as dev:
                    if dev == "cuda" and mgr.is_render_active:
                        violations.append(
                            f"Worker {worker_id} acquired CUDA while render was active!"
                        )
                    time.sleep(random.uniform(0.0005, 0.001))

        toggler_threads = [threading.Thread(target=render_toggler, args=(i,)) for i in range(5)]
        worker_threads = [threading.Thread(target=analysis_worker, args=(i,)) for i in range(20)]

        for t in toggler_threads + worker_threads:
            t.start()

        time.sleep(1.5)
        stop_event.set()

        for t in toggler_threads + worker_threads:
            t.join(timeout=5.0)

        assert not violations, f"Preemption violations detected: {violations[:5]}"
        assert mgr.active_nv_decoders == 0, f"Leaked slots: {mgr.active_nv_decoders}"
        assert mgr._render_active_count == 0, f"Render active count not reset: {mgr._render_active_count}"

    def test_worker_exception_crash_and_slot_leak(self):
        """Simulate severe worker crashes/exceptions inside lease_device context manager.
        
        Invariants checked:
        1. Every exception raised during task execution cleanly triggers slot release.
        2. Zero slot leaks occur after 200 exception cycles.
        """
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

        def crashing_worker():
            try:
                with mgr.lease_device(queue_size=15) as dev:
                    # Randomly crash on 70% of runs
                    if random.random() < 0.7:
                        raise RuntimeError("Simulated unhandled worker crash")
                    time.sleep(0.0005)
            except RuntimeError:
                pass

        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(crashing_worker) for _ in range(200)]
            concurrent.futures.wait(futures)

        assert mgr.active_nv_decoders == 0, f"Slots leaked after crashes: {mgr.active_nv_decoders}"

    def test_watermark_fuzzing_transitions(self):
        """Fuzz queue size transitions across 2,000 steps to verify state machine correctness."""
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

        # Baseline: start at 0 -> NORMAL_DECOUPLED
        assert mgr.get_analysis_device(0) == "qsv"
        assert mgr.state == "NORMAL_DECOUPLED"

        # Fuzz through varied queue transitions
        for _ in range(2000):
            q_size = random.randint(0, 30)
            dev = mgr.get_analysis_device(q_size)

            if q_size >= 10:
                assert dev == "cuda"
                assert mgr.state == "COOPERATIVE_BURST"
            elif q_size <= 3:
                assert dev == "qsv"
                assert mgr.state == "NORMAL_DECOUPLED"
            else:
                # In hysteresis band [4, 9]: device must match current state
                if mgr.state == "COOPERATIVE_BURST":
                    assert dev == "cuda"
                else:
                    assert dev == "qsv"


class TestAdversarialHardwareSemaphores:
    """Stress test hardware semaphores under extreme contention and exception injection."""

    def setup_method(self):
        reset_semaphores()

    def teardown_method(self):
        reset_semaphores()

    def test_semaphore_high_contention_and_exception_recovery(self):
        """50 concurrent threads acquiring semaphores with random exception crashes."""
        sem_nv = get_nv_semaphore()
        initial_val = sem_nv._value

        def worker():
            for _ in range(20):
                acquired = sem_nv.acquire(timeout=0.5)
                if acquired:
                    try:
                        if random.random() < 0.3:
                            raise ValueError("Simulated NV session crash")
                        time.sleep(0.0002)
                    except ValueError:
                        pass
                    finally:
                        sem_nv.release()

        threads = [threading.Thread(target=worker) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert sem_nv._value == initial_val, f"NV semaphore leaked! Expected {initial_val}, got {sem_nv._value}"


class TestAdversarialStreamingOrchestrator:
    """End-to-end chaos stress test for StreamingOrchestrator pipeline."""

    def setup_method(self):
        reset_semaphores()

    def teardown_method(self):
        reset_semaphores()

    def test_pipeline_chaos_stress_no_deadlock_no_leak(self):
        """Simulate a streaming pipeline execution with 40 tasks and random worker failures."""
        cfg = load_config()
        cfg["pipeline"]["prescreen_gpu_policy"] = "qsv_only"
        cfg["detection"]["prescreen_parallel"] = 4
        cfg["detection"]["analysis_max_workers"] = 4
        cfg["render"]["batch_max_files"] = 3
        cfg["pipeline"]["render_start_delay"] = 0.01
        cfg["yolo"]["enabled"] = False

        # Build mock database with 40 tasks
        tasks = []
        for i in range(40):
            tasks.append({
                "filepath": f"C:\\test\\video_{i:03d}.mp4",
                "file_start_time": f"2026090100{i:02d}00",
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

        # Mock prescreen_file, MotionDetector, and build_batch_render with chaos delays & failures
        def mock_prescreen(filepath, duration, config, gpu="qsv"):
            time.sleep(random.uniform(0.001, 0.005))
            if random.random() < 0.1:
                return {"status": "FAILED", "error": "mock prescreen failure"}
            if random.random() < 0.6:
                return {"status": "SUSPICIOUS", "result_json": '{"motion_ratio": 0.25}'}
            return {"status": "STATIC", "result_json": "{}"}

        def mock_analyze(filepath, start_offset=0.0, file_duration=300.0):
            time.sleep(random.uniform(0.002, 0.008))
            if random.random() < 0.05:
                raise RuntimeError("mock PyAV decoder failure")
            # Return 5 segments
            labels = [{"time": start_offset + j * 10, "is_motion": (j % 2 == 0), "energy": 0.5} for j in range(10)]
            return labels, {}

        def mock_build_batch_render(*args, **kwargs):
            time.sleep(random.uniform(0.005, 0.01))
            batch_idx = args[1]
            return f"C:\\output\\batch_{batch_idx}.mp4"

        import src.pipeline
        with patch("src.pipeline.prescreen_file", side_effect=mock_prescreen), \
             patch.object(src.pipeline.MotionDetector, "analyze", side_effect=mock_analyze), \
             patch("src.pipeline.build_batch_render", side_effect=mock_build_batch_render), \
             patch("src.pipeline.build_timeline", return_value=[]):
            batch_results = orchestrator.run()

        # Invariants after run:
        assert orchestrator.work_stealing.active_nv_decoders == 0, "Active NV decoders slot leaked!"
        assert not orchestrator.work_stealing.is_render_active, "Render preemption state not cleared!"
        assert get_nv_semaphore()._value == 3, "NV semaphore leaked!"
        assert get_qsv_semaphore()._value == 8, "QSV semaphore leaked!"
        assert get_disk_semaphore()._value == 8, "Disk semaphore leaked!"


class TestAdversarialEdgeCases:
    """Corner cases and invariant bounds testing."""

    def test_underflow_protection(self):
        """Ensure render_end and release_slot cannot drive counts negative."""
        mgr = WorkStealingManager()
        for _ in range(10):
            mgr.register_render_end()
            mgr.release_nvdec_slot()

        assert mgr._render_active_count == 0
        assert mgr.active_nv_decoders == 0

    def test_nested_concurrent_render_batches(self):
        """Ensure preemption holds until the last active render batch completes."""
        mgr = WorkStealingManager({"scheduler": {"watermark_high": 5}})
        
        # Batch 1 starts
        mgr.register_render_start()
        assert mgr.is_render_active
        assert mgr.get_analysis_device(queue_size=20) == "qsv"

        # Batch 2 starts concurrently
        mgr.register_render_start()
        assert mgr._render_active_count == 2
        assert mgr.get_analysis_device(queue_size=20) == "qsv"

        # Batch 1 finishes
        mgr.register_render_end()
        assert mgr.is_render_active
        assert mgr._render_active_count == 1
        assert mgr.get_analysis_device(queue_size=20) == "qsv"

        # Batch 2 finishes
        mgr.register_render_end()
        assert not mgr.is_render_active
        assert mgr._render_active_count == 0
        assert mgr.get_analysis_device(queue_size=20) == "cuda"

    def test_disabled_cooperative_or_zero_slots(self):
        """When disabled or max_nv_decoders is 0, never allocate cuda."""
        mgr1 = WorkStealingManager({"scheduler": {"nvdec_cooperative": False}})
        assert mgr1.get_analysis_device(queue_size=100) == "qsv"

        mgr2 = WorkStealingManager({"scheduler": {"max_nv_decoders": 0}})
        assert mgr2.get_analysis_device(queue_size=100) == "qsv"
        assert mgr2.acquire_nvdec_slot() is False

    def test_strict_semaphore_concurrency_ceiling(self):
        """Empirically verify that NV and QSV semaphores strictly limit active concurrent threads."""
        reset_semaphores()
        sem_nv = get_nv_semaphore()
        sem_qsv = get_qsv_semaphore()

        # Test NV ceiling (3)
        nv_concurrent = 0
        max_observed_nv = 0
        nv_lock = threading.Lock()
        nv_start = threading.Event()

        def nv_task():
            nonlocal nv_concurrent, max_observed_nv
            nv_start.wait()
            with sem_nv:
                with nv_lock:
                    nv_concurrent += 1
                    if nv_concurrent > max_observed_nv:
                        max_observed_nv = nv_concurrent
                time.sleep(0.05)
                with nv_lock:
                    nv_concurrent -= 1

        threads = [threading.Thread(target=nv_task) for _ in range(20)]
        for t in threads:
            t.start()
        nv_start.set()
        for t in threads:
            t.join()

        assert max_observed_nv == 3, f"NV concurrency ceiling violated: max {max_observed_nv} > 3"

        # Test QSV ceiling (8)
        qsv_concurrent = 0
        max_observed_qsv = 0
        qsv_lock = threading.Lock()
        qsv_start = threading.Event()

        def qsv_task():
            nonlocal qsv_concurrent, max_observed_qsv
            qsv_start.wait()
            with sem_qsv:
                with qsv_lock:
                    qsv_concurrent += 1
                    if qsv_concurrent > max_observed_qsv:
                        max_observed_qsv = qsv_concurrent
                time.sleep(0.05)
                with qsv_lock:
                    qsv_concurrent -= 1

        threads = [threading.Thread(target=qsv_task) for _ in range(30)]
        for t in threads:
            t.start()
        qsv_start.set()
        for t in threads:
            t.join()

        assert max_observed_qsv == 8, f"QSV concurrency ceiling violated: max {max_observed_qsv} > 8"

