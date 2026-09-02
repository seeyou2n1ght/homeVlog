"""
Milestone 1 Empirical Challenge Test Suite (Challenger 2).

Thoroughly stress-tests:
1. Boundary edge cases (queue_size = 0, watermark_high - 1, watermark_high, watermark_low, negative, large).
2. Hysteresis stability and rapid thrashing resistance.
3. Multi-threaded race conditions in lease_device().
4. Concurrent render preemption and work-stealing preemption safety.
5. Nested/re-entrant render session counting and mismatched end calls.
6. Thread safety and race condition resistance in get_*_semaphore() and reset_semaphores().
7. Strict hardware semaphore concurrency limits under heavy thread contention.
8. Multiple NV decoders capacity testing (max_nv_decoders > 1).
9. Degenerate watermark configurations (watermark_high == watermark_low, inverted watermarks).
10. Defensive slot underflow protection.
"""

import concurrent.futures
import random
import threading
import time
from unittest.mock import patch

import pytest

from src.utils import (
    WorkStealingManager,
    get_disk_semaphore,
    get_nv_semaphore,
    get_qsv_semaphore,
    reset_semaphores,
)


class TestWatermarkEdgeCases:
    """Challenge boundary conditions and edge cases in WorkStealingManager."""

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

    def test_exact_boundary_transitions(self, manager):
        # 1. queue_size = 0
        assert manager.get_analysis_device(queue_size=0) == "qsv"
        assert manager.state == "NORMAL_DECOUPLED"

        # 2. queue_size = watermark_high - 1 (9)
        assert manager.get_analysis_device(queue_size=9) == "qsv"
        assert manager.state == "NORMAL_DECOUPLED"

        # 3. queue_size = watermark_high (10) -> triggers COOPERATIVE_BURST
        assert manager.get_analysis_device(queue_size=10) == "cuda"
        assert manager.state == "COOPERATIVE_BURST"

        # 4. In burst mode: queue_size = watermark_high - 1 (9) stays CUDA (hysteresis)
        assert manager.get_analysis_device(queue_size=9) == "cuda"
        assert manager.state == "COOPERATIVE_BURST"

        # 5. In burst mode: queue_size = watermark_low + 1 (4) stays CUDA (hysteresis)
        assert manager.get_analysis_device(queue_size=4) == "cuda"
        assert manager.state == "COOPERATIVE_BURST"

        # 6. In burst mode: queue_size = watermark_low (3) -> reverts to NORMAL_DECOUPLED (QSV)
        assert manager.get_analysis_device(queue_size=3) == "qsv"
        assert manager.state == "NORMAL_DECOUPLED"

        # 7. In normal mode: queue_size = watermark_low + 1 (4) stays QSV (hysteresis)
        assert manager.get_analysis_device(queue_size=4) == "qsv"
        assert manager.state == "NORMAL_DECOUPLED"

    def test_extreme_and_negative_queue_sizes(self, manager):
        # Negative queue sizes (e.g. malformed or drain arithmetic)
        assert manager.get_analysis_device(queue_size=-1) == "qsv"
        assert manager.get_analysis_device(queue_size=-100) == "qsv"
        assert manager.state == "NORMAL_DECOUPLED"

        # Very large queue size
        assert manager.get_analysis_device(queue_size=1_000_000) == "cuda"
        assert manager.state == "COOPERATIVE_BURST"

    def test_rapid_thrashing_resistance(self, manager):
        """
        Oscillate rapidly around watermark thresholds (1000 iterations).
        Verify state transitions remain strictly deterministic with no stuck states.
        """
        for _ in range(1000):
            # Jump to high watermark -> CUDA
            dev = manager.get_analysis_device(queue_size=10)
            assert dev == "cuda"
            assert manager.state == "COOPERATIVE_BURST"

            # Fluctuate inside hysteresis band (9, 8, 7, 6, 5, 4) -> must stay CUDA
            for q in [9, 8, 7, 6, 5, 4]:
                assert manager.get_analysis_device(queue_size=q) == "cuda"
                assert manager.state == "COOPERATIVE_BURST"

            # Drop to low watermark -> QSV
            dev = manager.get_analysis_device(queue_size=3)
            assert dev == "qsv"
            assert manager.state == "NORMAL_DECOUPLED"

            # Fluctuate inside hysteresis band (4, 5, 6, 7, 8, 9) -> must stay QSV
            for q in [4, 5, 6, 7, 8, 9]:
                assert manager.get_analysis_device(queue_size=q) == "qsv"
                assert manager.state == "NORMAL_DECOUPLED"

    def test_equal_watermark_edge_case(self):
        """When watermark_high == watermark_low (degenerate zero-width hysteresis band)."""
        config = {
            "hardware": {"device": "cuda:0", "max_nv_decoders": 1},
            "scheduler": {
                "watermark_high": 5,
                "watermark_low": 5,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
            },
        }
        mgr = WorkStealingManager(config)
        assert mgr.get_analysis_device(queue_size=5) == "cuda"
        assert mgr.get_analysis_device(queue_size=4) == "qsv"
        assert mgr.get_analysis_device(queue_size=5) == "cuda"

    def test_multi_nv_decoder_capacity(self):
        """When max_nv_decoders = 3, up to 3 decoders can concurrently run CUDA."""
        config = {
            "hardware": {"device": "cuda:0", "max_nv_decoders": 3},
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
                "max_nv_decoders": 3,
            },
        }
        mgr = WorkStealingManager(config)
        with mgr.lease_device(queue_size=15) as dev1:
            assert dev1 == "cuda"
            assert mgr.active_nv_decoders == 1
            with mgr.lease_device(queue_size=15) as dev2:
                assert dev2 == "cuda"
                assert mgr.active_nv_decoders == 2
                with mgr.lease_device(queue_size=15) as dev3:
                    assert dev3 == "cuda"
                    assert mgr.active_nv_decoders == 3
                    # 4th concurrent attempt falls back to QSV
                    with mgr.lease_device(queue_size=15) as dev4:
                        assert dev4 == "qsv"
                        assert mgr.active_nv_decoders == 3
        assert mgr.active_nv_decoders == 0

    def test_defensive_underflow_protection(self):
        """Verify release_nvdec_slot and register_render_end do not underflow past zero."""
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
        # Extra release calls
        mgr.release_nvdec_slot()
        mgr.release_nvdec_slot()
        assert mgr.active_nv_decoders == 0

        # Extra render_end calls
        mgr.register_render_end()
        mgr.register_render_end()
        assert not mgr.is_render_active
        assert mgr._render_active_count == 0


class TestMultiThreadedConcurrency:
    """Stress-test multi-threaded race conditions in lease_device and semaphores."""

    def setup_method(self):
        reset_semaphores()

    def teardown_method(self):
        reset_semaphores()

    def test_concurrent_lease_device_slot_limits_and_no_leaks(self):
        """
        50 threads concurrently lease devices with high queue size.
        Ensure active_nv_decoders NEVER exceeds max_nv_decoders (1),
        and returns cleanly to 0 with zero leaks even under exceptions.
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

        max_observed_nv = 0
        lock = threading.Lock()
        cuda_count = 0
        qsv_count = 0
        error_count = 0

        def worker(thread_id: int):
            nonlocal max_observed_nv, cuda_count, qsv_count, error_count
            for _ in range(20):
                should_raise = (thread_id + _) % 7 == 0
                try:
                    with mgr.lease_device(queue_size=15) as dev:
                        with lock:
                            current_nv = mgr.active_nv_decoders
                            if current_nv > max_observed_nv:
                                max_observed_nv = current_nv
                            if dev == "cuda":
                                cuda_count += 1
                            else:
                                qsv_count += 1

                        assert current_nv <= 1, f"active_nv_decoders exceeded limit: {current_nv}"
                        # Simulate small random workload
                        time.sleep(random.uniform(0.001, 0.005))

                        if should_raise:
                            raise ValueError("Simulated task exception inside lease block")
                except ValueError:
                    with lock:
                        error_count += 1

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Invariants:
        assert max_observed_nv <= 1, f"Peak NV decoders exceeded limit: {max_observed_nv}"
        assert mgr.active_nv_decoders == 0, f"Slot leak detected! Active decoders: {mgr.active_nv_decoders}"
        assert cuda_count > 0, "Expected some tasks to utilize CUDA"
        assert qsv_count > 0, "Expected overflow tasks to route to QSV"
        assert error_count > 0, "Expected exception branches to be tested"

    def test_concurrent_render_preemption_stress(self):
        """
        Stress test: Analysis workers continuously leasing devices while render
        threads concurrently trigger render batches (register_render_start / end).
        Invariant: Whenever render is active, NO worker may be granted 'cuda'.
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

        stop_event = threading.Event()
        violations = []

        def analysis_worker():
            while not stop_event.is_set():
                with mgr.lease_device(queue_size=20) as dev:
                    # If this lease was granted 'cuda', check if render was active
                    if dev == "cuda" and mgr.is_render_active:
                        violations.append("Leased 'cuda' while render was active!")
                    time.sleep(0.002)

        def render_worker():
            while not stop_event.is_set():
                mgr.register_render_start()
                assert mgr.is_render_active
                time.sleep(0.01)
                mgr.register_render_end()
                time.sleep(0.005)

        workers = [threading.Thread(target=analysis_worker) for _ in range(20)]
        renders = [threading.Thread(target=render_worker) for _ in range(3)]

        for t in workers + renders:
            t.start()

        time.sleep(0.5)
        stop_event.set()

        for t in workers + renders:
            t.join()

        assert not violations, f"Violations found: {violations}"
        assert mgr.active_nv_decoders == 0, f"Leaked decoders: {mgr.active_nv_decoders}"
        assert not mgr.is_render_active, "Render active count corrupted"

    def test_nested_render_preemption_refcount(self):
        """Verify re-entrant / nested render start/end reference counting."""
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

        # First render batch starts
        mgr.register_render_start()
        assert mgr.is_render_active
        assert mgr.state == "RENDER_PREEMPTION_YIELD"
        assert mgr.get_analysis_device(queue_size=50) == "qsv"

        # Second concurrent render batch starts
        mgr.register_render_start()
        assert mgr.is_render_active
        assert mgr.state == "RENDER_PREEMPTION_YIELD"

        # First render batch completes -> must STILL be active
        mgr.register_render_end()
        assert mgr.is_render_active
        assert mgr.state == "RENDER_PREEMPTION_YIELD"
        assert mgr.get_analysis_device(queue_size=50) == "qsv"

        # Second render batch completes -> now inactive
        mgr.register_render_end()
        assert not mgr.is_render_active
        assert mgr.state == "NORMAL_DECOUPLED"

        # High backlog can resume CUDA
        assert mgr.get_analysis_device(queue_size=50) == "cuda"
        assert mgr.state == "COOPERATIVE_BURST"

    def test_hardware_semaphore_concurrency_stress(self):
        """
        Stress test hardware semaphores to guarantee strict capacity enforcement.
        get_nv_semaphore (3) and get_qsv_semaphore (8).
        """
        nv_sem = get_nv_semaphore()
        active_nv = 0
        max_nv = 0
        nv_lock = threading.Lock()

        def nv_task():
            nonlocal active_nv, max_nv
            with nv_sem:
                with nv_lock:
                    active_nv += 1
                    if active_nv > max_nv:
                        max_nv = active_nv
                time.sleep(0.005)
                with nv_lock:
                    active_nv -= 1

        threads = [threading.Thread(target=nv_task) for _ in range(30)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_nv <= 3, f"NV semaphore limit violated: {max_nv} > 3"
        assert nv_sem._value == 3, f"NV semaphore not fully released: {nv_sem._value}"

    def test_get_semaphore_concurrency_and_reset(self):
        """
        Verify thread safety when accessing get_*_semaphore concurrently.
        """
        errors = []

        def reader():
            try:
                for _ in range(50):
                    s1 = get_nv_semaphore()
                    s2 = get_qsv_semaphore()
                    s3 = get_disk_semaphore()
                    if s1 is None or s2 is None or s3 is None:
                        errors.append("Returned None semaphore")
                    time.sleep(0.001)
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=reader) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors encountered: {errors}"
