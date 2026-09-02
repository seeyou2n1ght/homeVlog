"""
Adversarial Challenge & Empirical Stress-Test Suite for Milestone 5:
End-to-End Performance Benchmark & Acceptance Hardening.

Empirical verification harness targeting:
1. Performance log schema compliance, quantile consistency, and adversarial mutation stress.
2. Hardware resource bounds: Concurrency semaphores, WorkStealingManager state machine, and thread-safe monitor/perf collection.
3. Multi-file stress scenarios: Large-scale timeline assembly, cross-file boundary partitioning, duration conservation, database concurrency, and complex filtergraph label closures.
"""

import json
import math
import random
import threading
import time
from pathlib import Path
from typing import Any
import pytest

from src.database import VlogDatabase
from src.monitor import Monitor, PerfCollector, PerfRecord, StageStats, get_monitor, get_perf
from src.pipeline import StreamingOrchestrator, WorkStealingManager
from src.segment import (
    Segment,
    build_segments,
    merge_cross_file,
    split_segments_at_file_boundaries,
    segments_to_json,
    segments_from_json,
)
from src.timeline import (
    TimelineSegment,
    build_concat_filter,
    calculate_speed_ramping_curve,
    partition_timeline_by_batches,
)
from src.utils import (
    get_nv_semaphore,
    get_qsv_semaphore,
    get_disk_semaphore,
    reset_semaphores,
    load_config,
    ts_to_unix,
)
from tests.helpers import (
    validate_perf_json_schema,
    parse_ffmpeg_filtergraph,
    verify_filtergraph_labels_closure,
)


# ============================================================================
# 1. Performance Log Schema & Metrics Challenge
# ============================================================================

class TestPerfLogSchemaAdversarialChallenge:
    """Stress tests and adversarial challenges on the performance log schema and collector."""

    def test_actual_perf_logs_in_repository_validity(self):
        """Verify all existing perf_*.json files in logs/ strictly pass schema validation."""
        logs_dir = Path("logs")
        perf_files = list(logs_dir.glob("perf_*.json"))
        assert len(perf_files) > 0, "Expected at least one perf_*.json log in logs/"

        for pf in perf_files:
            with open(pf, "r", encoding="utf-8") as f:
                data = json.load(f)
            valid, msg = validate_perf_json_schema(data)
            assert valid is True, f"Perf log {pf.name} failed schema validation: {msg}"
            assert data["pipeline_duration"] > 0
            assert isinstance(data["monitor_summary"], list)
            assert isinstance(data["records"], list)

        # Specifically check the full 81-clip benchmark log
        bench_log_path = logs_dir / "perf_20260901_cam0_20260902_062848.json"
        if bench_log_path.exists():
            with open(bench_log_path, "r", encoding="utf-8") as f:
                bench_data = json.load(f)
            assert len(bench_data["records"]) == 136
            assert len(bench_data["monitor_summary"]) >= 1

            summary = bench_data.get("perf_summary", {})
            assert "prescreen" in summary
            assert "analysis" in summary
            assert "render" in summary
            for stage, stats in summary.items():
                assert stats["min"] <= stats["p50"] <= stats["p95"] <= stats["max"], (
                    f"Quantile ordering violated in stage {stage}: "
                    f"min={stats['min']}, p50={stats['p50']}, p95={stats['p95']}, max={stats['max']}"
                )
                assert stats["total"] >= stats["min"] * stats["count"]
                assert stats["avg"] >= stats["min"]
                assert stats["avg"] <= stats["max"]

    @pytest.mark.parametrize("missing_key", [
        "date", "cam", "pipeline_duration", "monitor_summary", "perf_summary", "records"
    ])
    def test_schema_rejection_missing_top_level_keys(self, missing_key: str):
        """Schema validator must reject any log missing a mandatory top-level key."""
        valid_sample = {
            "date": "20260901",
            "cam": 0,
            "pipeline_duration": 123.45,
            "monitor_summary": [{"name": "stage1", "duration": 10.0, "avg_cpu": 50.0, "avg_ram": 40.0}],
            "perf_summary": {"prescreen": {"count": 1, "total": 1.0, "avg": 1.0, "p50": 1.0, "p95": 1.0, "min": 1.0, "max": 1.0}},
            "records": [{"stage": "prescreen", "file": "f1.mp4", "gpu": "qsv", "duration": 1.0}],
        }
        del valid_sample[missing_key]
        valid, msg = validate_perf_json_schema(valid_sample)
        assert valid is False
        assert f"Missing top-level key: {missing_key}" in msg

    @pytest.mark.parametrize("bad_duration", [-1.0, -100.5, "fast", None])
    def test_schema_rejection_invalid_pipeline_duration(self, bad_duration: Any):
        """Schema validator must reject negative or non-numeric pipeline durations."""
        sample = {
            "date": "20260901",
            "cam": 0,
            "pipeline_duration": bad_duration,
            "monitor_summary": [],
            "perf_summary": {},
            "records": [],
        }
        valid, msg = validate_perf_json_schema(sample)
        assert valid is False
        assert "pipeline_duration" in msg

    def test_perf_collector_heavy_concurrent_add_and_summary(self):
        """Stress-test PerfCollector with 50 threads concurrently writing and reading."""
        collector = PerfCollector()
        stages = ["prescreen", "analysis", "render"]
        gpus = ["qsv", "cuda", "nv"]
        num_threads = 50
        records_per_thread = 200

        def worker_task(thread_id: int):
            for i in range(records_per_thread):
                stage = stages[(thread_id + i) % len(stages)]
                gpu = gpus[(thread_id + i) % len(gpus)]
                dur = round(random.uniform(0.1, 5.0), 3)
                collector.add(
                    PerfRecord(
                        stage=stage,
                        file=f"file_{thread_id}_{i}.mp4",
                        gpu=gpu,
                        duration=dur,
                        frames=int(dur * 20),
                        fps=20.0,
                        extra={"thread": thread_id, "idx": i},
                    )
                )
                if i % 50 == 0:
                    summary = collector.summary_by_stage()
                    assert isinstance(summary, dict)

        threads = [threading.Thread(target=worker_task, args=(tid,)) for tid in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        summary = collector.summary_by_stage()
        total_records = sum(stats["count"] for stats in summary.values())
        assert total_records == num_threads * records_per_thread

        for stg, stats in summary.items():
            assert stats["count"] > 0
            assert stats["min"] <= stats["p50"] <= stats["p95"] <= stats["max"]
            assert stats["total"] > 0
            assert math.isclose(stats["avg"], stats["total"] / stats["count"], rel_tol=1e-2)

    def test_stage_stats_multi_gpu_summary_integrity(self):
        """StageStats must handle arbitrary numbers of GPUs without index errors or null refs."""
        stats = StageStats(name="multi_gpu_stage", start_ts=100.0, end_ts=120.0)
        stats.cpu_pct = [45.0, 55.0, 60.0]
        stats.ram_pct = [70.0, 72.0, 71.0]
        stats.gpu_load = [[30, 40, 50], [10, 20, 30]]
        stats.gpu_mem = [[2048, 3072, 4096], [1024, 1024, 1024]]
        stats.gpu_enc_load = [[0, 10, 20], [0, 0, 0]]
        stats.gpu_dec_load = [[50, 60, 70], [0, 0, 0]]

        assert math.isclose(stats.duration, 20.0)
        assert math.isclose(stats.avg_cpu, 53.333, rel_tol=1e-2)
        assert math.isclose(stats.avg_ram, 71.0, rel_tol=1e-2)

        g0 = stats.gpu_summary(0)
        assert math.isclose(g0["avg_load"], 40.0)
        assert g0["peak_mem_mb"] == 4096
        assert math.isclose(g0["avg_enc"], 10.0)
        assert math.isclose(g0["avg_dec"], 60.0)

        g1 = stats.gpu_summary(1)
        assert math.isclose(g1["avg_load"], 20.0)
        assert g1["peak_mem_mb"] == 1024

        # Non-existent GPU index must gracefully return zeroed metrics
        g99 = stats.gpu_summary(99)
        assert g99["avg_load"] == 0.0
        assert g99["peak_mem_mb"] == 0


# ============================================================================
# 2. Hardware Resource Bounds & Concurrency Challenge
# ============================================================================

class TestHardwareResourceBoundsChallenge:
    """Stress tests and boundary checks on hardware semaphores and WorkStealingManager."""

    def test_hardware_semaphores_strict_concurrency_bounds(self):
        """Semaphores must strictly enforce bounded concurrency under 30+ simultaneous threads."""
        reset_semaphores()
        nv_sem = get_nv_semaphore()
        qsv_sem = get_qsv_semaphore()
        disk_sem = get_disk_semaphore()

        # Check configured limits from settings
        cfg = load_config()
        max_nv = cfg.get("hardware", {}).get("max_nv_concurrency", 3)
        max_qsv = cfg.get("hardware", {}).get("max_qsv_concurrency", 8)

        active_nv = 0
        max_seen_nv = 0
        active_qsv = 0
        max_seen_qsv = 0
        lock = threading.Lock()

        def nv_task():
            nonlocal active_nv, max_seen_nv
            with nv_sem:
                with lock:
                    active_nv += 1
                    max_seen_nv = max(max_seen_nv, active_nv)
                time.sleep(0.01)
                with lock:
                    active_nv -= 1

        def qsv_task():
            nonlocal active_qsv, max_seen_qsv
            with qsv_sem:
                with lock:
                    active_qsv += 1
                    max_seen_qsv = max(max_seen_qsv, active_qsv)
                time.sleep(0.01)
                with lock:
                    active_qsv -= 1

        threads = []
        for _ in range(25):
            threads.append(threading.Thread(target=nv_task))
            threads.append(threading.Thread(target=qsv_task))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert max_seen_nv <= max_nv, f"NV concurrency exceeded limit: {max_seen_nv} > {max_nv}"
        assert max_seen_qsv <= max_qsv, f"QSV concurrency exceeded limit: {max_seen_qsv} > {max_qsv}"

    def test_work_stealing_manager_state_machine_hysteresis(self):
        """Verify WorkStealingManager transitions through low, hysteresis, and high watermarks."""
        config = {
            "scheduler": {
                "watermark_high": 10,
                "watermark_low": 3,
                "nvdec_cooperative": True,
                "max_nv_decoders": 2,
            },
            "hardware": {"device": "cuda:0"},
        }
        wsm = WorkStealingManager(config=config)

        # Baseline: normal decoupled
        assert wsm.state == "NORMAL_DECOUPLED"
        assert wsm.get_analysis_device(queue_size=2) == "qsv"
        assert wsm.state == "NORMAL_DECOUPLED"

        # High watermark triggers cooperative burst
        assert wsm.get_analysis_device(queue_size=12) == "cuda"
        assert wsm.state == "COOPERATIVE_BURST"

        # Hysteresis band (queue_size=6) retains COOPERATIVE_BURST until low watermark
        assert wsm.get_analysis_device(queue_size=6) == "cuda"
        assert wsm.state == "COOPERATIVE_BURST"

        # Low watermark returns to NORMAL_DECOUPLED
        assert wsm.get_analysis_device(queue_size=2) == "qsv"
        assert wsm.state == "NORMAL_DECOUPLED"

        # Inside hysteresis band when coming from NORMAL_DECOUPLED stays "qsv"
        assert wsm.get_analysis_device(queue_size=6) == "qsv"
        assert wsm.state == "NORMAL_DECOUPLED"

    def test_work_stealing_manager_render_preemption_yield(self):
        """Pass 2 render start must immediately preempt NVDEC work-stealing."""
        config = {
            "scheduler": {
                "watermark_high": 5,
                "watermark_low": 2,
                "nvdec_cooperative": True,
                "max_nv_decoders": 2,
            },
            "hardware": {"device": "cuda:0"},
        }
        wsm = WorkStealingManager(config=config)

        # High watermark gives cuda
        assert wsm.get_analysis_device(queue_size=10) == "cuda"

        # Render starts -> forces qsv even with massive queue backlog
        wsm.register_render_start()
        assert wsm.is_render_active is True
        assert wsm.state == "RENDER_PREEMPTION_YIELD"
        assert wsm.get_analysis_device(queue_size=50) == "qsv"
        assert wsm.acquire_nvdec_slot() is False

        # Multiple concurrent render registrations
        wsm.register_render_start()
        wsm.register_render_end()
        assert wsm.is_render_active is True
        assert wsm.get_analysis_device(queue_size=50) == "qsv"

        # Final render ends -> unblocks work-stealing
        wsm.register_render_end()
        assert wsm.is_render_active is False
        assert wsm.get_analysis_device(queue_size=50) == "cuda"

    def test_work_stealing_lease_device_exception_safety(self):
        """lease_device context manager must strictly release NVDEC slots even if exception occurs."""
        config = {
            "scheduler": {
                "watermark_high": 5,
                "watermark_low": 2,
                "nvdec_cooperative": True,
                "max_nv_decoders": 1,
            },
            "hardware": {"device": "cuda:0"},
        }
        wsm = WorkStealingManager(config=config)

        assert wsm.active_nv_decoders == 0

        # Simulate exception during leased cuda execution
        try:
            with wsm.lease_device(queue_size=10) as dev:
                assert dev == "cuda"
                assert wsm.active_nv_decoders == 1
                raise RuntimeError("Simulated analysis decoder crash")
        except RuntimeError:
            pass

        # Slot must be safely released
        assert wsm.active_nv_decoders == 0
        # Subsequent lease should succeed without slot leakage
        with wsm.lease_device(queue_size=10) as dev:
            assert dev == "cuda"
            assert wsm.active_nv_decoders == 1
        assert wsm.active_nv_decoders == 0


# ============================================================================
# 3. Multi-File Stress Scenarios & Boundary Partitioning
# ============================================================================

class TestMultiFileStressScenarios:
    """Stress tests on large multi-file timelines, boundary partitioning, and duration conservation."""

    def test_multi_file_concurrent_database_operations(self, isolated_db):
        """Stress-test SQLite database under 40 concurrent workers inserting/updating tasks."""
        db = isolated_db
        num_workers = 40
        files_per_worker = 25

        def db_worker(worker_id: int):
            for i in range(files_per_worker):
                filepath = f"/surveillance/cam0/clip_{worker_id:02d}_{i:03d}.mp4"
                db.add_file_task(
                    filepath=filepath,
                    cam_index=0,
                    date="20260901",
                    file_start_time=f"2026090100{i % 60:02d}00",
                    file_end_time=f"2026090100{(i+1) % 60:02d}00",
                    file_duration=300.0,
                )
                db.set_prescreen_result(filepath, "SUSPICIOUS", json.dumps({"score": 0.85}))
                db.set_file_metadata(filepath, has_audio=bool(i % 2 == 0))
                db.set_analysis_result(filepath, "ANALYZED", json.dumps([]))

        threads = [threading.Thread(target=db_worker, args=(wid,)) for wid in range(num_workers)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        all_tasks = db.get_all_file_tasks_for_date("20260901", 0)
        assert len(all_tasks) == num_workers * files_per_worker
        assert all(t["analysis_status"] == "ANALYZED" for t in all_tasks)

    def test_multi_file_segment_boundary_partitioning_duration_conservation(self):
        """Cross-file segments spanning 50+ files must conserve exact source duration when partitioned."""
        files = [f"/surveillance/clip_{i:03d}.mp4" for i in range(50)]
        # 50 contiguous files of 300s each -> total timeline 0.0s to 15,000.0s
        files_info = {
            files[i]: (float(i * 300.0), float((i + 1) * 300.0))
            for i in range(50)
        }

        # Create one giant dynamic segment spanning all 50 files (0s to 15,000s)
        giant_seg = Segment(
            start_time=0.0,
            end_time=15000.0,
            state="DYNAMIC",
            source_file=files[0],
            file_start_offset=0.0,
            max_energy=0.9,
        )

        partitioned = split_segments_at_file_boundaries([giant_seg], files_info)

        # Must split into exactly 50 segments, each exactly 300.0s long
        assert len(partitioned) == 50
        total_partitioned_dur = sum((s.end_time - s.start_time) for s in partitioned)
        assert math.isclose(total_partitioned_dur, 15000.0, abs_tol=1e-4)

        for idx, seg in enumerate(partitioned):
            assert seg.source_file == files[idx]
            assert math.isclose(seg.start_time, float(idx * 300.0), abs_tol=1e-4)
            assert math.isclose(seg.end_time, float((idx + 1) * 300.0), abs_tol=1e-4)
            assert math.isclose(seg.file_start_offset, float(idx * 300.0), abs_tol=1e-4)
            assert seg.state == "DYNAMIC"

    def test_large_scale_timeline_batch_partitioning_and_batch_limits(self):
        """Partitioning a 100-file timeline with batch_max_files=8 must produce disjoint bounded batches."""
        files = [f"/camera/file_{i:03d}.mp4" for i in range(100)]
        timeline = []
        for i, f in enumerate(files):
            timeline.append(TimelineSegment(
                filepath=f,
                input_index=i,
                start_in_file=0.0,
                end_in_file=100.0,
                state="STATIC",
                duration=100.0,
            ))
            timeline.append(TimelineSegment(
                filepath=f,
                input_index=i,
                start_in_file=100.0,
                end_in_file=200.0,
                state="DYNAMIC",
                duration=100.0,
            ))

        batches = partition_timeline_by_batches(timeline, batch_max_files=8)
        assert len(batches) == math.ceil(100 / 8)  # 13 batches

        all_batched_segs = []
        for b_idx, b in enumerate(batches):
            unique_files_in_batch = {s.filepath for s in b}
            assert len(unique_files_in_batch) <= 8, f"Batch {b_idx} exceeded max files limit: {len(unique_files_in_batch)}"
            all_batched_segs.extend(b)

        assert len(all_batched_segs) == len(timeline)

    def test_complex_filtergraph_label_closure_and_mixed_audio_streams(self):
        """Construct filtergraph for 60 segments with alternating audio presence and verify closed labels."""
        num_files = 15
        files = [f"/cams/file_{i:02d}.mp4" for i in range(num_files)]
        rows = [{"filepath": f, "has_audio": int(i % 2 == 0)} for i, f in enumerate(files)]

        timeline = []
        for i in range(60):
            f_idx = (i // 4) % num_files
            timeline.append(
                TimelineSegment(
                    filepath=files[f_idx],
                    input_index=f_idx,
                    start_in_file=float((i % 4) * 25.0),
                    end_in_file=float((i % 4 + 1) * 25.0),
                    state="DYNAMIC" if i % 3 == 0 else "STATIC",
                    duration=25.0,
                )
            )

        filtergraph = build_concat_filter(
            timeline=timeline,
            rows=rows,
            speed_ramping=True,
            output_fps=20,
            output_width=1920,
            output_height=1080,
        )

        assert len(filtergraph) > 0
        valid, msg = verify_filtergraph_labels_closure(filtergraph)
        assert valid is True, f"Filtergraph closure validation failed: {msg}"

        parsed = parse_ffmpeg_filtergraph(filtergraph)
        assert parsed["concat_n"] == 60
        assert len(parsed["concat_inputs"]) == 120  # 60 video + 60 audio

    def test_speed_ramping_extreme_duration_and_acceleration_stress(self):
        """Test speed ramping curve calculation under extreme micro and macro durations."""
        # Extreme micro duration: 0.005s (shorter than any ramp zone)
        micro_info = calculate_speed_ramping_curve(
            dur=0.005,
            v_fast=60.0,
            has_ramp_in=True,
            has_ramp_out=True,
            ramp_duration_s=1.0,
        )
        assert micro_info.target_display_dur > 0
        assert micro_info.cruise_src_dur == 0.0
        assert micro_info.ramp_in_src_dur == 0.0025
        assert micro_info.ramp_out_src_dur == 0.0025

        # Extreme macro duration: 86,400s (24 hours static video)
        macro_info = calculate_speed_ramping_curve(
            dur=86400.0,
            v_fast=60.0,
            has_ramp_in=True,
            has_ramp_out=True,
            ramp_duration_s=1.0,
        )
        assert macro_info.target_display_dur < 86400.0 / 50.0
        assert macro_info.cruise_src_dur > 86000.0
        assert "if(lt(" in macro_info.pts_expr

    def test_timeline_gap_handling_and_no_merging_across_temporal_discontinuities(self):
        """Discontinuous recordings (> 1.0s gap) must not be merged into continuous segments."""
        # File 1 ends at t=300.0, File 2 starts at t=310.0 (10.0s recording gap)
        seg1 = Segment(
            start_time=0.0,
            end_time=300.0,
            state="STATIC",
            source_file="f1.mp4",
            file_start_offset=0.0,
        )
        seg2 = Segment(
            start_time=310.0,
            end_time=610.0,
            state="STATIC",
            source_file="f2.mp4",
            file_start_offset=310.0,
        )

        merged = merge_cross_file([seg1, seg2], gap_tolerance=0.5)
        # Must NOT merge because gap is 10.0s > 0.5s tolerance
        assert len(merged) == 2
        assert merged[0].end_time == 300.0
        assert merged[1].start_time == 310.0

    def test_device_fallback_when_cooperative_disabled_or_cpu_only(self):
        """When nvdec_cooperative is disabled, WorkStealingManager must never allocate cuda."""
        cfg_disabled = {
            "scheduler": {
                "watermark_high": 2,
                "watermark_low": 1,
                "nvdec_cooperative": False,
                "max_nv_decoders": 2,
            },
            "hardware": {"device": "cuda:0"},
        }
        wsm_disabled = WorkStealingManager(config=cfg_disabled)
        assert wsm_disabled.get_analysis_device(queue_size=100) == "qsv"
        assert wsm_disabled.state == "NORMAL_DECOUPLED"

        cfg_cpu = {
            "scheduler": {
                "watermark_high": 2,
                "watermark_low": 1,
                "nvdec_cooperative": True,
                "max_nv_decoders": 2,
            },
            "hardware": {"device": "cpu"},
        }
        wsm_cpu = WorkStealingManager(config=cfg_cpu)
        assert wsm_cpu.get_analysis_device(queue_size=100) == "qsv"
        assert wsm_cpu.state == "NORMAL_DECOUPLED"

    def test_perf_collector_quantile_mathematical_precision(self):
        """Verify summary_by_stage computes mathematically precise percentiles."""
        collector = PerfCollector()
        # Insert known deterministic durations: 1.0, 2.0, ..., 100.0
        for i in range(1, 101):
            collector.add(PerfRecord(stage="analysis", file=f"f{i}.mp4", gpu="qsv", duration=float(i)))

        summary = collector.summary_by_stage()
        stg = summary["analysis"]
        assert stg["count"] == 100
        assert stg["total"] == sum(range(1, 101))
        assert stg["avg"] == 50.5
        assert stg["min"] == 1.0
        assert stg["max"] == 100.0
        assert stg["p50"] == 51.0  # index 50 in 0-indexed sorted list
        assert stg["p95"] == 96.0  # index 95 in 0-indexed sorted list

