"""
Empirical Adversarial Challenge & Stress Test Suite for Milestone 2 (R2).
Challenger 1: Concurrency races, min-heap integrity, and admission policy stress.

Mission:
1. AnalysisQueue.steal_sublongest and min-heap integrity:
   - Concurrently interleave hundreds of put(), get(), get_heavy(), and steal_sublongest()
     across multiple threads with randomized durations (10s to 5000s) and identical start times.
   - Assert:
     (a) zero tasks are lost or duplicated;
     (b) min-heap invariant is 100% preserved at all times;
     (c) unfinished_tasks count exactly equals zero after all task_done() calls;
     (d) steal_sublongest never returns tasks >= max_duration or < min_duration.
2. WorkStealingManager.can_worker_0_steal and four-gate admission policy:
   - Stress-test rapid state transitions under simulated heavy task arrivals,
     queue backlog fluctuations, and VRAM margin swings.
   - Verify that when heavy_tasks_cleared=False, admission is unconditionally rejected.
   - Verify that when free VRAM drops below 3072 MB, admission is unconditionally rejected.
"""

import sys
import time
import random
import threading
import queue
import unittest
from typing import Any

from src.stages.pipeline_types import AnalysisQueue, PipelineTask
from src.hardware.scheduler import WorkStealingManager


def is_min_heap(heap_list):
    """Verify standard binary min-heap property on the queue list."""
    n = len(heap_list)
    for i in range(n):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n and heap_list[i][0] > heap_list[left][0]:
            return False
        if right < n and heap_list[i][0] > heap_list[right][0]:
            return False
    return True


class TestAnalysisQueueAdversarialStress(unittest.TestCase):
    """Adversarial concurrent stress testing for AnalysisQueue."""

    def test_01_concurrent_interleaved_heavy_steal_and_get(self):
        """Stress test 600 tasks across multiple producer & consumer threads.

        Tasks have randomized durations between 10s and 5000s and IDENTICAL start times.
        Simultaneously tests put(), get(), get_heavy(), and steal_sublongest().
        """
        q = AnalysisQueue()
        q.cost_priority = True

        total_tasks = 600
        min_steal_dur = 300.0
        max_steal_dur = 1200.0
        heavy_dur = 1200.0
        shared_start_time = "20260320_120000"

        # Pre-generate tasks with randomized durations and identical start time
        random.seed(42)
        durations = [random.uniform(10.0, 5000.0) for _ in range(total_tasks)]
        tasks = [
            {
                "id": i,
                "filepath": f"video_{i:04d}.mp4",
                "file_duration": durations[i],
                "file_start_time": shared_start_time,
            }
            for i in range(total_tasks)
        ]

        extracted_tasks = []
        extracted_lock = threading.Lock()
        heap_violations = []
        steal_violations = []

        producers_done = threading.Event()

        # Producer function
        def producer(task_slice):
            for t in task_slice:
                q.put(t)
                # Verify min-heap invariant under queue mutex periodically
                with q.mutex:
                    if not is_min_heap(q.queue):
                        heap_violations.append("Violation during producer put")
                time.sleep(random.uniform(0.00001, 0.0002))

        # Consumer worker simulating QSV workers calling get()
        def standard_consumer():
            while True:
                try:
                    task = q.get(timeout=0.05)
                except queue.Empty:
                    if producers_done.is_set() and q.empty():
                        break
                    continue

                with extracted_lock:
                    extracted_tasks.append((task, "get"))
                # Simulate light processing
                time.sleep(random.uniform(0.00001, 0.0001))
                q.task_done()

        # Consumer worker simulating Worker 0 consuming night giants via get_heavy()
        def heavy_consumer():
            while True:
                try:
                    # Check if heavy task exists
                    if q.has_heavy_tasks(min_duration=heavy_dur):
                        task = q.get_heavy(timeout=0.05, min_duration=heavy_dur)
                    else:
                        task = q.get(timeout=0.05)
                except queue.Empty:
                    if producers_done.is_set() and q.empty():
                        break
                    continue

                with extracted_lock:
                    extracted_tasks.append((task, "get_heavy"))
                time.sleep(random.uniform(0.00001, 0.0001))
                q.task_done()

        # Consumer worker simulating Worker 0 stealing medium tasks via steal_sublongest()
        def stealer_consumer():
            while True:
                stolen = q.steal_sublongest(
                    max_duration=max_steal_dur,
                    min_duration=min_steal_dur,
                )
                if stolen is not None:
                    # Verify duration boundary assertion (d)
                    s_dur = float(stolen.get("file_duration") or 0.0)
                    if not (min_steal_dur <= s_dur < max_steal_dur):
                        steal_violations.append(
                            f"Stolen task duration {s_dur} outside [{min_steal_dur}, {max_steal_dur})"
                        )
                    if not stolen.get("is_stolen"):
                        steal_violations.append("Stolen task is_stolen flag not True")

                    # Verify heap invariant after steal
                    with q.mutex:
                        if not is_min_heap(q.queue):
                            heap_violations.append("Violation after steal_sublongest")

                    with extracted_lock:
                        extracted_tasks.append((stolen, "steal"))
                    time.sleep(random.uniform(0.00001, 0.0001))
                    q.task_done()
                else:
                    if producers_done.is_set() and q.empty():
                        break
                    time.sleep(random.uniform(0.0001, 0.0005))

        # Launch 4 producers
        chunk_size = total_tasks // 4
        producers = [
            threading.Thread(target=producer, args=(tasks[i * chunk_size : (i + 1) * chunk_size],))
            for i in range(4)
        ]

        # Launch 6 consumers: 2 standard, 2 heavy, 2 stealers
        consumers = [
            threading.Thread(target=standard_consumer) for _ in range(2)
        ] + [
            threading.Thread(target=heavy_consumer) for _ in range(2)
        ] + [
            threading.Thread(target=stealer_consumer) for _ in range(2)
        ]

        for p in producers:
            p.start()
        for c in consumers:
            c.start()

        for p in producers:
            p.join(timeout=10.0)
        producers_done.set()

        for c in consumers:
            c.join(timeout=15.0)

        # Assertions
        # (a) Zero tasks lost or duplicated
        self.assertEqual(
            len(extracted_tasks),
            total_tasks,
            f"Expected {total_tasks} extracted tasks, got {len(extracted_tasks)}",
        )
        extracted_ids = [t[0]["id"] for t in extracted_tasks]
        self.assertEqual(
            len(set(extracted_ids)),
            total_tasks,
            "Duplicate tasks detected in extracted tasks!",
        )
        self.assertEqual(
            set(extracted_ids),
            set(range(total_tasks)),
            "Mismatch in task IDs (some tasks were lost)!",
        )

        # (b) Min-heap invariant is 100% preserved
        self.assertEqual(
            len(heap_violations),
            0,
            f"Min-heap invariant violated: {heap_violations[:5]}",
        )

        # (c) unfinished_tasks count exactly equals zero after all task_done() calls
        self.assertEqual(
            q.unfinished_tasks,
            0,
            f"unfinished_tasks must be 0, got {q.unfinished_tasks}",
        )
        # q.join() should return immediately without blocking
        q.join()

        # (d) steal_sublongest duration constraints
        self.assertEqual(
            len(steal_violations),
            0,
            f"Steal duration violations: {steal_violations[:5]}",
        )

        # Stats breakdown
        methods = [m for _, m in extracted_tasks]
        n_get = methods.count("get")
        n_heavy = methods.count("get_heavy")
        n_steal = methods.count("steal")
        print(f"\n[Interleaved Stress Test] Total={total_tasks}: get={n_get}, get_heavy={n_heavy}, steal={n_steal}")
        self.assertGreater(n_steal, 0, "Stealer should have stolen at least one qualifying task")

    def test_02_steal_sublongest_boundary_conditions(self):
        """Adversarially test boundary conditions of steal_sublongest."""
        q = AnalysisQueue()
        q.cost_priority = True

        # 1. Exactly on min_duration (300.0) -> must be eligible
        q.put({"id": 1, "filepath": "f1.mp4", "file_duration": 300.0, "file_start_time": "120000"})
        stolen = q.steal_sublongest(max_duration=1200.0, min_duration=300.0)
        self.assertIsNotNone(stolen)
        self.assertEqual(stolen["file_duration"], 300.0)
        self.assertTrue(is_min_heap(q.queue))
        q.task_done()

        # 2. Strictly below min_duration (299.99) -> must NOT be stolen
        q.put({"id": 2, "filepath": "f2.mp4", "file_duration": 299.99, "file_start_time": "120000"})
        self.assertIsNone(q.steal_sublongest(max_duration=1200.0, min_duration=300.0))
        item = q.get()
        self.assertEqual(item["id"], 2)
        q.task_done()

        # 3. Exactly on max_duration (1200.0) -> must NOT be stolen (strictly < max_duration)
        q.put({"id": 3, "filepath": "f3.mp4", "file_duration": 1200.0, "file_start_time": "120000"})
        self.assertIsNone(q.steal_sublongest(max_duration=1200.0, min_duration=300.0))
        item = q.get()
        self.assertEqual(item["id"], 3)
        q.task_done()

        # 4. Strictly below max_duration (1199.99) -> must be eligible
        q.put({"id": 4, "filepath": "f4.mp4", "file_duration": 1199.99, "file_start_time": "120000"})
        stolen = q.steal_sublongest(max_duration=1200.0, min_duration=300.0)
        self.assertIsNotNone(stolen)
        self.assertEqual(stolen["file_duration"], 1199.99)
        q.task_done()

        # 5. Multiple candidates: select maximum duration
        q.put({"id": 5, "filepath": "f5.mp4", "file_duration": 400.0, "file_start_time": "120000"})
        q.put({"id": 6, "filepath": "f6.mp4", "file_duration": 950.0, "file_start_time": "120000"})
        q.put({"id": 7, "filepath": "f7.mp4", "file_duration": 600.0, "file_start_time": "120000"})
        stolen = q.steal_sublongest(max_duration=1200.0, min_duration=300.0)
        self.assertIsNotNone(stolen)
        self.assertEqual(stolen["id"], 6)  # 950.0 is the maximum medium candidate
        self.assertTrue(is_min_heap(q.queue))
        q.task_done()

        # 6. Tie-breaking on identical durations: earlier start time wins
        q.put({"id": 8, "filepath": "f8.mp4", "file_duration": 800.0, "file_start_time": "120010"})
        q.put({"id": 9, "filepath": "f9.mp4", "file_duration": 800.0, "file_start_time": "120000"})  # earlier
        stolen = q.steal_sublongest(max_duration=1200.0, min_duration=300.0)
        self.assertIsNotNone(stolen)
        self.assertEqual(stolen["id"], 9)
        self.assertTrue(is_min_heap(q.queue))
        q.task_done()

        # 7. Drain remaining
        while not q.empty():
            q.get()
            q.task_done()
        self.assertEqual(q.unfinished_tasks, 0)

    def test_03_min_heap_invariant_under_rapid_steal_and_pop(self):
        """Perform 500 sequential operations mixing random put, steal, and heappop."""
        q = AnalysisQueue()
        q.cost_priority = True

        for i in range(500):
            op = random.choice(["put", "steal", "pop"])
            if op == "put" or q.empty():
                dur = random.uniform(50.0, 3000.0)
                q.put({"id": i, "filepath": f"f_{i}.mp4", "file_duration": dur, "file_start_time": "120000"})
            elif op == "steal":
                q.steal_sublongest(max_duration=1200.0, min_duration=300.0)
            else:
                q.get()

            # Verify min-heap invariant after every single operation
            self.assertTrue(is_min_heap(q.queue), f"Heap invariant broken at iteration {i}")

    def test_04_pipelinetask_objects_and_cost_priority_false(self):
        """Stress-test with strongly-typed PipelineTask objects and cost_priority=False."""
        q = AnalysisQueue()
        q.cost_priority = False  # chronological order instead of duration order

        tasks = [
            PipelineTask(
                id=i,
                filepath=f"cam0_{i:03d}.mp4",
                file_start_time=f"20260320_00{i:02d}00",
                file_duration=float(100 + (i * 37) % 1500),
            )
            for i in range(100)
        ]

        for t in tasks:
            q.put(t)

        self.assertTrue(is_min_heap(q.queue))

        stolen_count = 0
        while True:
            stolen = q.steal_sublongest(max_duration=1200.0, min_duration=300.0)
            if stolen is None:
                break
            stolen_count += 1
            self.assertTrue(300.0 <= stolen.file_duration < 1200.0)
            self.assertTrue(stolen.is_stolen)
            self.assertTrue(is_min_heap(q.queue))
            q.task_done()

        # Drain rest
        while not q.empty():
            item = q.get()
            q.task_done()

        self.assertEqual(q.unfinished_tasks, 0)
        self.assertGreater(stolen_count, 0)


class TestFourGateAdmissionAdversarialStress(unittest.TestCase):
    """Adversarial stress testing for WorkStealingManager four-gate admission policy."""

    def setUp(self):
        self.config = {
            "hardware": {
                "device": "cuda:0",
                "vram_watermark_mb": 6200,
                "max_nv_decoders": 1,
            },
            "scheduler": {
                "steal_enabled": True,
                "steal_backlog_threshold": 2,
                "steal_vram_margin_mb": 3072,
                "heavy_file_duration_s": 600.0,
                "nvdec_cooperative": True,
            },
        }

    def test_01_heavy_tasks_cleared_absolute_veto(self):
        """When heavy_tasks_cleared is False, admission must be unconditionally rejected.

        Even if Worker 0 is idle, queue backlog is huge, and free VRAM is 8000 MB!
        """
        mgr = WorkStealingManager(self.config)
        # Mock unlimited free VRAM and low used VRAM
        mgr.set_vram_margin_probe_fn(lambda gpu_index=0: 8000)
        mgr.set_vram_probe_fn(lambda: 1000)

        # Vary queue backlog from 2 to 1000
        for q_size in [2, 5, 10, 50, 100, 1000]:
            admitted = mgr.can_worker_0_steal(
                queue_size=q_size,
                heavy_tasks_cleared=False,  # Heavy tasks remain!
            )
            self.assertFalse(
                admitted,
                f"Gate 2 VETO failed: admitted when heavy_tasks_cleared=False at qsize={q_size}",
            )

    def test_02_vram_margin_boundary_and_swings(self):
        """When free VRAM drops below 3072 MB, admission must be unconditionally rejected.

        Stress-test exact boundary: 3071 vs 3072, and rapid dynamic swings.
        """
        mgr = WorkStealingManager(self.config)
        mgr.set_vram_probe_fn(lambda: 2000)  # used = 2000 MB < 6200 watermark

        # 1. Exact boundary test
        # 3071 MB -> Reject
        mgr.set_vram_margin_probe_fn(lambda gpu_index=0: 3071)
        self.assertFalse(
            mgr.can_worker_0_steal(queue_size=5, heavy_tasks_cleared=True),
            "Failed to reject free VRAM == 3071 MB (< 3072 MB)",
        )

        # 3072 MB -> Admit
        mgr.set_vram_margin_probe_fn(lambda gpu_index=0: 3072)
        self.assertTrue(
            mgr.can_worker_0_steal(queue_size=5, heavy_tasks_cleared=True),
            "Failed to admit free VRAM == 3072 MB (>= 3072 MB)",
        )

        # 3073 MB -> Admit
        mgr.set_vram_margin_probe_fn(lambda gpu_index=0: 3073)
        self.assertTrue(
            mgr.can_worker_0_steal(queue_size=5, heavy_tasks_cleared=True),
            "Failed to admit free VRAM == 3073 MB",
        )

        # 2. Dynamic swing stress test across 5,000 rapid fluctuations
        random.seed(1337)
        for _ in range(5000):
            free_vram = random.randint(0, 8192)
            used_vram = random.randint(500, 8000)
            q_size = random.randint(0, 10)
            heavy_cleared = random.choice([True, False])
            active_decoders = random.choice([0, 1, 2])

            mgr.set_vram_margin_probe_fn(lambda gpu_index=0, f=free_vram: f)
            mgr.set_vram_probe_fn(lambda u=used_vram: u)
            mgr._active_nv_decoders = active_decoders

            result = mgr.can_worker_0_steal(
                queue_size=q_size,
                heavy_tasks_cleared=heavy_cleared,
            )

            # Ground truth verification of all five gates
            expected = (
                mgr.steal_enabled
                and heavy_cleared
                and (active_decoders < mgr.max_nv_decoders)
                and (q_size >= mgr.steal_backlog_threshold)
                and (free_vram >= 3072)
                and (used_vram < mgr.vram_watermark_mb)
            )

            self.assertEqual(
                result,
                expected,
                f"Mismatch in 5-gate admission: got {result}, expected {expected} | "
                f"free={free_vram}, used={used_vram}, qsize={q_size}, heavy={heavy_cleared}, decoders={active_decoders}",
            )

    def test_03_concurrent_admission_evaluations_and_render_lifecycle(self):
        """Concurrently evaluate admission across 8 threads during simulated render lifecycle."""
        mgr = WorkStealingManager(self.config)
        stop_event = threading.Event()
        errors = []

        # Thread 1: Simulates render lifecycle (starts and ends batches)
        def render_simulator():
            while not stop_event.is_set():
                mgr.register_render_start()
                time.sleep(random.uniform(0.0005, 0.002))
                mgr.register_render_end()
                time.sleep(random.uniform(0.0005, 0.002))

        # Thread 2: Simulates VRAM fluctuations
        def vram_simulator():
            while not stop_event.is_set():
                free_val = random.choice([2500, 3071, 3072, 4000, 5000])
                used_val = random.choice([2000, 4000, 6199, 6200, 7000])
                mgr.set_vram_margin_probe_fn(lambda gpu_index=0, f=free_val: f)
                mgr.set_vram_probe_fn(lambda u=used_val: u)
                time.sleep(random.uniform(0.0001, 0.0005))

        # Threads 3-8: Evaluate can_worker_0_steal and acquire/release lease
        def worker_simulator(w_id):
            while not stop_event.is_set():
                q_len = random.randint(0, 5)
                heavy_cleared = random.choice([True, False])
                can_steal = mgr.can_worker_0_steal(q_len, heavy_tasks_cleared=heavy_cleared)

                # If heavy_tasks_cleared is False, must NEVER be allowed
                if not heavy_cleared and can_steal:
                    errors.append(f"Worker {w_id}: admitted with heavy_tasks_cleared=False")

                # If admitted, simulate leasing device for stolen task
                if can_steal:
                    try:
                        with mgr.lease_device(queue_size=q_len, is_stolen=True) as dev:
                            if dev == "cuda":
                                # active decoders must not exceed max
                                if mgr.active_nv_decoders > mgr.max_nv_decoders:
                                    errors.append(f"active_nv_decoders {mgr.active_nv_decoders} > max {mgr.max_nv_decoders}")
                            time.sleep(random.uniform(0.0001, 0.0005))
                    except Exception as exc:
                        errors.append(f"Worker {w_id} exception during lease: {exc}")

        threads = [
            threading.Thread(target=render_simulator),
            threading.Thread(target=vram_simulator),
        ] + [
            threading.Thread(target=worker_simulator, args=(i,)) for i in range(6)
        ]

        for t in threads:
            t.start()

        time.sleep(2.0)  # Run concurrent stress for 2.0 seconds
        stop_event.set()

        for t in threads:
            t.join(timeout=5.0)

        self.assertEqual(len(errors), 0, f"Errors during concurrent stress: {errors[:5]}")
        # Active decoders must be cleanly restored to 0
        self.assertEqual(mgr.active_nv_decoders, 0, "active_nv_decoders was not restored to 0!")


if __name__ == "__main__":
    unittest.main()
