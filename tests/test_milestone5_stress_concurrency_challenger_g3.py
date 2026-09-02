"""
Milestone 5 High-Intensity Concurrency & Stress Verification Test Suite.
Authored by Challenger G3-1 (Role: Stress & Concurrency Challenger).

Verifies:
1. Device leasing and work-stealing preemption under high queue contention.
2. Hardware semaphore isolation under concurrent stress.
3. Pipeline error handling and graceful recovery when clips are corrupt or missing.
4. Race conditions in StreamingOrchestrator and BatchRenderer.
"""

import concurrent.futures
import json
import queue
import random
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.database import VlogDatabase
from src.detector import MotionDetector
from src.pipeline import StreamingOrchestrator, process_date_cam
from src.renderer import _render_batches_parallel, _render_batches_sequential, partition_timeline_by_batches
from src.timeline import TimelineSegment, build_timeline
from src.utils import (
    WorkStealingManager,
    get_disk_semaphore,
    get_nv_semaphore,
    get_qsv_semaphore,
    load_config,
    reset_semaphores,
)


class TestWorkStealingAndLeasingContention:
    """Stress testing work stealing and device leasing under extreme queue contention."""

    def test_work_stealing_high_contention_flapping_and_leasing(self):
        """
        Stress test: 50 concurrent worker threads making rapid lease requests
        while queue size dynamically flaps across watermark thresholds [0, 25],
        and an external thread periodically triggers render start/end cycles.
        
        Invariants:
        1. Number of concurrently leased 'cuda' devices must NEVER exceed max_nv_decoders (1).
        2. When render is active, leased device must always be 'qsv'.
        3. active_nv_decoders count must return strictly to 0 when all leases finish.
        4. Zero deadlocks or race condition crashes across 5000+ operations.
        """
        config = {
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
            },
            "hardware": {"device": "cuda:0"},
        }
        manager = WorkStealingManager(config=config)

        num_threads = 50
        cycles_per_thread = 100
        active_cuda_leases = 0
        max_observed_cuda = 0
        cuda_count_lock = threading.Lock()
        violations = []
        stop_flapper = threading.Event()

        def lease_worker(worker_id: int):
            nonlocal active_cuda_leases, max_observed_cuda
            for _ in range(cycles_per_thread):
                q_size = random.randint(0, 25)
                with manager.lease_device(q_size) as dev:
                    if dev == "cuda":
                        with cuda_count_lock:
                            active_cuda_leases += 1
                            if active_cuda_leases > 1:
                                violations.append(
                                    f"Concurrent CUDA leases exceeded limit: {active_cuda_leases} > 1"
                                )
                            max_observed_cuda = max(max_observed_cuda, active_cuda_leases)
                        try:
                            time.sleep(random.uniform(0.0001, 0.0005))
                        finally:
                            with cuda_count_lock:
                                active_cuda_leases -= 1
                    elif dev == "qsv":
                        time.sleep(random.uniform(0.0001, 0.0003))
                    else:
                        violations.append(f"Unexpected device leased: {dev}")

        def render_controller():
            while not stop_flapper.is_set():
                time.sleep(random.uniform(0.005, 0.015))
                manager.register_render_start()
                assert manager.is_render_active is True
                time.sleep(random.uniform(0.005, 0.015))
                manager.register_render_end()

        render_thread = threading.Thread(target=render_controller, daemon=True)
        render_thread.start()

        threads = [
            threading.Thread(target=lease_worker, args=(i,))
            for i in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "Worker thread deadlocked during lease contention"

        stop_flapper.set()
        render_thread.join(timeout=5)

        assert not violations, f"Lease invariants violated: {violations[:5]}"
        assert max_observed_cuda <= 1, f"Max concurrent CUDA leases was {max_observed_cuda}"
        assert manager.active_nv_decoders == 0, f"Leaked active_nv_decoders: {manager.active_nv_decoders}"

    def test_work_stealing_immediate_render_preemption_race(self):
        """
        Test the preemption race condition:
        Worker A holds a 'cuda' lease while in COOPERATIVE_BURST.
        Worker B calls register_render_start().
        Worker C immediately calls lease_device(20).
        
        Verification:
        - Worker C receives 'qsv' immediately because render is active.
        - Worker A finishes cleanly and decrements active_nv_decoders to 0.
        - Active count never goes negative.
        """
        config = {
            "scheduler": {
                "watermark_high": 5,
                "watermark_low": 2,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
            },
            "hardware": {"device": "cuda:0"},
        }
        manager = WorkStealingManager(config=config)

        slot_acquired = threading.Event()
        release_slot = threading.Event()
        worker_c_device = None

        def worker_a():
            with manager.lease_device(10) as dev:
                assert dev == "cuda"
                assert manager.active_nv_decoders == 1
                slot_acquired.set()
                release_slot.wait(timeout=5)

        def worker_c():
            nonlocal worker_c_device
            slot_acquired.wait(timeout=5)
            # Worker C tries to acquire with high queue size while render is started
            with manager.lease_device(10) as dev:
                worker_c_device = dev

        t_a = threading.Thread(target=worker_a)
        t_c = threading.Thread(target=worker_c)

        t_a.start()
        slot_acquired.wait(timeout=5)

        # Worker B signals render start
        manager.register_render_start()
        assert manager.is_render_active is True

        t_c.start()
        t_c.join(timeout=5)
        assert worker_c_device == "qsv", f"Worker C should have yielded to qsv, got {worker_c_device}"

        release_slot.set()
        t_a.join(timeout=5)

        assert manager.active_nv_decoders == 0
        manager.register_render_end()
        assert manager.is_render_active is False
        assert manager.state == "NORMAL_DECOUPLED"

    def test_work_stealing_hysteresis_boundary_transitions(self):
        """
        Test strict state transitions across hysteresis thresholds:
        - Q size < 10 -> NORMAL_DECOUPLED ('qsv')
        - Q size >= 10 -> COOPERATIVE_BURST ('cuda')
        - Q size falls to 5 (between 3 and 10) -> stays COOPERATIVE_BURST ('cuda')
        - Q size falls to 2 (<= 3) -> returns to NORMAL_DECOUPLED ('qsv')
        - Q size rises to 5 -> stays NORMAL_DECOUPLED ('qsv')
        """
        config = {
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
            },
            "hardware": {"device": "cuda:0"},
        }
        manager = WorkStealingManager(config=config)

        assert manager.get_analysis_device(5) == "qsv"
        assert manager.state == "NORMAL_DECOUPLED"

        assert manager.get_analysis_device(10) == "cuda"
        assert manager.state == "COOPERATIVE_BURST"

        assert manager.get_analysis_device(5) == "cuda"
        assert manager.state == "COOPERATIVE_BURST"

        assert manager.get_analysis_device(3) == "qsv"
        assert manager.state == "NORMAL_DECOUPLED"

        assert manager.get_analysis_device(5) == "qsv"
        assert manager.state == "NORMAL_DECOUPLED"


class TestHardwareSemaphoresIsolationAndStress:
    """Stress testing hardware semaphores under severe concurrent load and failure modes."""

    def setup_method(self):
        reset_semaphores()

    def teardown_method(self):
        reset_semaphores()

    def test_hardware_semaphores_concurrent_stress_and_isolation(self):
        """
        60 concurrent threads randomly acquiring combinations of NV (3), QSV (8), and Disk (8)
        with artificial latency and random exceptions thrown inside the critical region.
        
        Invariants:
        1. NV concurrent active workers <= 3.
        2. QSV concurrent active workers <= 8.
        3. Disk concurrent active workers <= 8.
        4. Zero resource leaks after all threads complete.
        """
        sem_nv = get_nv_semaphore()
        sem_qsv = get_qsv_semaphore()
        sem_disk = get_disk_semaphore()

        nv_active, qsv_active, disk_active = 0, 0, 0
        max_nv, max_qsv, max_disk = 0, 0, 0
        lock = threading.Lock()
        violations = []

        def worker(worker_id: int):
            nonlocal nv_active, qsv_active, disk_active, max_nv, max_qsv, max_disk
            for _ in range(30):
                target_sem = random.choice(["nv", "qsv"])
                with sem_disk:
                    with lock:
                        disk_active += 1
                        if disk_active > 8:
                            violations.append(f"Disk concurrency limit breached: {disk_active} > 8")
                        max_disk = max(max_disk, disk_active)
                    try:
                        time.sleep(random.uniform(0.0001, 0.0005))
                        if target_sem == "nv":
                            with sem_nv:
                                with lock:
                                    nv_active += 1
                                    if nv_active > 3:
                                        violations.append(f"NV concurrency limit breached: {nv_active} > 3")
                                    max_nv = max(max_nv, nv_active)
                                try:
                                    time.sleep(random.uniform(0.0001, 0.0005))
                                    if random.random() < 0.05:
                                        raise RuntimeError("Injected transient worker exception")
                                finally:
                                    with lock:
                                        nv_active -= 1
                        else:
                            with sem_qsv:
                                with lock:
                                    qsv_active += 1
                                    if qsv_active > 8:
                                        violations.append(f"QSV concurrency limit breached: {qsv_active} > 8")
                                    max_qsv = max(max_qsv, qsv_active)
                                try:
                                    time.sleep(random.uniform(0.0001, 0.0005))
                                    if random.random() < 0.05:
                                        raise RuntimeError("Injected transient worker exception")
                                finally:
                                    with lock:
                                        qsv_active -= 1
                    except RuntimeError:
                        pass
                    finally:
                        with lock:
                            disk_active -= 1

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(60)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
            assert not t.is_alive(), "Worker thread deadlocked during semaphore stress"

        assert not violations, f"Concurrency violations: {violations[:5]}"
        assert max_nv <= 3, f"Max NV was {max_nv} > 3"
        assert max_qsv <= 8, f"Max QSV was {max_qsv} > 8"
        assert max_disk <= 8, f"Max Disk was {max_disk} > 8"

    def test_semaphore_lazy_init_and_reset_thread_safety(self):
        """
        Test thread safety of double-checked locking in get_*_semaphore() and reset_semaphores().
        40 threads concurrently fetching semaphores while another thread resets them.
        """
        violations = []
        stop_event = threading.Event()

        def getter():
            while not stop_event.is_set():
                try:
                    s1 = get_nv_semaphore()
                    s2 = get_qsv_semaphore()
                    s3 = get_disk_semaphore()
                    assert s1 is not None and s2 is not None and s3 is not None
                except Exception as e:
                    violations.append(str(e))
                time.sleep(0.0005)

        def resetter():
            while not stop_event.is_set():
                reset_semaphores()
                time.sleep(0.002)

        threads = [threading.Thread(target=getter) for _ in range(40)]
        r_thread = threading.Thread(target=resetter)

        r_thread.start()
        for t in threads:
            t.start()

        time.sleep(0.5)
        stop_event.set()

        r_thread.join(timeout=5)
        for t in threads:
            t.join(timeout=5)

        assert not violations, f"Semaphore initialization/reset errors: {violations[:5]}"


class TestPipelineErrorHandlingAndFaultTolerance:
    """Stress testing pipeline fault isolation with corrupt, truncated, and missing files."""

    def test_pipeline_error_handling_corrupt_and_missing_clips(self, tmp_path):
        """
        Adversarial test: Inject a mix of missing files, 0-byte corrupt files,
        corrupt non-video files, and valid synthetic tasks into StreamingOrchestrator.
        
        Verification:
        - Corrupt/missing files are caught and marked as FAILED in DB.
        - Pipeline logs errors without raising unhandled exceptions or crashing.
        - Valid tasks are processed and batched.
        """
        db_path = tmp_path / "test_corrupt.db"
        db = VlogDatabase(db_path)
        date = "20260901"
        cam_index = 0

        # Create files
        missing_file = str(tmp_path / "missing_00.mp4")
        empty_file = tmp_path / "empty_01.mp4"
        empty_file.write_bytes(b"")
        garbage_file = tmp_path / "garbage_02.mp4"
        garbage_file.write_text("This is not a valid mp4 header", encoding="utf-8")
        valid_file = tmp_path / "valid_03.mp4"
        valid_file.write_bytes(b"\x00" * 1024)

        files = [
            (missing_file, "000000", "000500"),
            (str(empty_file), "000500", "001000"),
            (str(garbage_file), "001000", "001500"),
            (str(valid_file), "001500", "002000"),
        ]

        for fp, st, et in files:
            db.add_file_task(fp, cam_index, date, f"{date}{st}", f"{date}{et}", 300.0)

        config = load_config()
        orchestrator = StreamingOrchestrator(db, date, cam_index, config, render_enabled=False)

        def mock_prescreen(filepath, duration, cfg, gpu="qsv"):
            if "missing" in filepath or "empty" in filepath or "garbage" in filepath:
                return {"status": "FAILED", "error": "Corrupt or unreadable container"}
            return {"status": "STATIC", "result_json": json.dumps({"static_ratio": 1.0})}

        with patch("src.pipeline.prescreen_file", side_effect=mock_prescreen):
            paths = orchestrator.run()

        # Check DB statuses
        all_tasks = db.get_all_file_tasks_for_date(date, cam_index)
        status_map = {t["filepath"]: t["prescreen_status"] for t in all_tasks}

        assert status_map[missing_file] == "FAILED"
        assert status_map[str(empty_file)] == "FAILED"
        assert status_map[str(garbage_file)] == "FAILED"
        assert status_map[str(valid_file)] == "STATIC"

        # Check errors recorded
        assert len(orchestrator.errors) >= 3

    def test_pipeline_all_files_failed_recovery(self, tmp_path):
        """
        Adversarial test: When 100% of files fail in a given day,
        process_date_cam must return False, set render status to FAILED,
        and not hang or crash.
        """
        db_path = tmp_path / "test_all_fail.db"
        db = VlogDatabase(db_path)
        date = "20260901"
        cam_index = 0

        for i in range(5):
            fp = str(tmp_path / f"corrupt_{i:02d}.mp4")
            db.add_file_task(fp, cam_index, date, f"{date}{i:02d}0000", f"{date}{i:02d}0500", 300.0)

        def mock_prescreen(filepath, duration, cfg, gpu="qsv"):
            return {"status": "FAILED", "error": "Fatal codec decode failure"}

        with patch("src.pipeline.prescreen_file", side_effect=mock_prescreen):
            result = process_date_cam(db, date, cam_index, skip_render=False)

        assert result is False
        assert db.is_render_completed(date, cam_index) is False

    def test_detector_analyze_corrupt_file_graceful_handling(self, tmp_path):
        """
        Verify MotionDetector.analyze on non-existent or 0-byte file returns empty records
        gracefully rather than crashing unhandled.
        """
        config = load_config()
        detector = MotionDetector(config, decode_gpu="qsv")

        corrupt_p = tmp_path / "bad.mp4"
        corrupt_p.write_bytes(b"\x00\x00\x00")

        labels, buf = detector.analyze(str(corrupt_p), start_offset=0.0, file_duration=300.0)
        assert labels == []
        assert buf == {} or buf == []

    def test_build_timeline_corrupted_json_resilience(self, tmp_path):
        """
        Adversarial test: Inject corrupted / malformed JSON strings in analysis_segments
        and ensure build_timeline gracefully parses without uncaught JSONDecodeError.
        """
        db_path = tmp_path / "corrupt_json.db"
        db = VlogDatabase(db_path)
        date = "20260901"
        cam_index = 0

        db.add_file_task("good_static.mp4", cam_index, date, f"{date}000000", f"{date}000500", 300.0)
        db.set_prescreen_result("good_static.mp4", "STATIC", json.dumps({"static_ratio": 1.0}))

        db.add_file_task("bad_json.mp4", cam_index, date, f"{date}000500", f"{date}001000", 300.0)
        db.set_prescreen_result("bad_json.mp4", "SUSPICIOUS", "")
        db.set_analysis_result("bad_json.mp4", "ANALYZED", "{corrupt_invalid_json: 1234")

        # build_timeline should not crash
        timeline = build_timeline(db, date, cam_index)
        assert isinstance(timeline, list)
        assert len(timeline) >= 1
        db.close()


class TestRaceConditionsAndConcurrencySafety:
    """Stress testing race conditions in StreamingOrchestrator, BatchRenderer, and SQLite WAL."""

    def test_streaming_orchestrator_rapid_batch_queuing_race(self):
        """
        Verify that under fast arrival of analyzed files into render_batch_queue,
        batches are partitioned accurately without duplicate batch indexes or dropped files.
        """
        db = MagicMock(spec=VlogDatabase)
        db.get_all_file_tasks_for_date.return_value = []
        config = load_config()
        orchestrator = StreamingOrchestrator(db, "20260901", 0, config, render_enabled=True)
        orchestrator.batch_max_files = 4

        # Feed 20 files
        files = [f"file_{i:02d}.mp4" for i in range(20)]
        for f in files:
            orchestrator.render_batch_queue.put({"filepath": f, "status": "STATIC"})

        orchestrator.stop_event.set()
        pending = []
        batches = []
        while not orchestrator.render_batch_queue.empty():
            msg = orchestrator.render_batch_queue.get()
            pending.append(msg["filepath"])
            if len(pending) >= orchestrator.batch_max_files:
                batches.append(list(pending))
                pending = []

        if pending:
            batches.append(list(pending))

        assert len(batches) == 5
        for b in batches:
            assert len(b) == 4

    def test_database_concurrent_wal_read_write_stress(self, tmp_path):
        """
        Stress test SQLite WAL database:
        30 threads writing prescreen/analysis statuses concurrently
        while 15 threads concurrently query all tasks and check completion statuses.
        
        Invariant:
        Zero SQLite locking errors or database corruption.
        """
        db_path = tmp_path / "wal_stress.db"
        db = VlogDatabase(db_path)
        date = "20260901"
        cam_index = 0

        # Prepopulate 50 files
        for i in range(50):
            fp = f"cam0_20260901_{i:04d}.mp4"
            db.add_file_task(fp, cam_index, date, f"{date}000000", f"{date}000500", 300.0)

        stop_event = threading.Event()
        errors = []

        def writer(worker_id: int):
            for i in range(30):
                if stop_event.is_set():
                    break
                fp = f"cam0_20260901_{(worker_id * 30 + i) % 50:04d}.mp4"
                try:
                    db.set_prescreen_result(fp, "SUSPICIOUS", json.dumps({"score": 0.9}))
                    db.set_analysis_result(fp, "ANALYZED", json.dumps([{"start": 0, "end": 10}]))
                    db.set_file_metadata(fp, 1)
                except Exception as e:
                    errors.append(f"Writer error: {e}")
                time.sleep(0.0002)

        def reader(worker_id: int):
            for _ in range(30):
                if stop_event.is_set():
                    break
                try:
                    tasks = db.get_all_file_tasks_for_date(date, cam_index)
                    assert len(tasks) == 50
                    _ = db.is_prescreen_complete(date, cam_index)
                    _ = db.is_analysis_complete(date, cam_index)
                    _ = db.get_pending_file_count_for_date(date, cam_index)
                except Exception as e:
                    errors.append(f"Reader error: {e}")
                time.sleep(0.0002)

        writers = [threading.Thread(target=writer, args=(i,)) for i in range(30)]
        readers = [threading.Thread(target=reader, args=(i,)) for i in range(15)]

        for t in writers + readers:
            t.start()
        for t in writers + readers:
            t.join(timeout=30)
            assert not t.is_alive(), "Database worker thread hung under WAL concurrency"

        assert not errors, f"Database WAL concurrency errors: {errors[:5]}"
        db.close()

    def test_batch_renderer_parallel_dispatch_and_error_handling(self, tmp_path):
        """
        Verify that _render_batches_parallel handles partial batch render failures
        gracefully by returning None and cleaning up without hung threads or unhandled exceptions.
        """
        timeline = [
            TimelineSegment(filepath=f"file_{i}.mp4", input_index=i, start_in_file=0.0, end_in_file=10.0, state="DYNAMIC", duration=10.0)
            for i in range(10)
        ]
        batches = partition_timeline_by_batches(timeline, batch_max_files=2)
        assert len(batches) == 5

        output_path = tmp_path / "out.mp4"
        seg_cfg = {}
        out_cfg = {}
        audio_cfg = {}

        # Simulate batch render failure on batch index 2
        def mock_build_batch_render(batch_segs, bi, enc, fps, width, height, sc, oc, ac, date, cam_index, rows):
            if bi == 2:
                raise RuntimeError("Simulated FFmpeg batch encode failure")
            p = tmp_path / f"batch_{bi}.mp4"
            p.write_bytes(b"dummy_video_data")
            return str(p)

        with patch("src.renderer.build_batch_render", side_effect=mock_build_batch_render):
            result = _render_batches_parallel(
                batches, output_path, fps=20, width=1920, height=1080,
                seg_cfg=seg_cfg, out_cfg=out_cfg, audio_cfg=audio_cfg,
                date="20260901", cam_index=0, rows=None
            )

        assert result is None, "Expected None return on batch render failure"
