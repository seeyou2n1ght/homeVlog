"""Efficiency changes must preserve temporal and classification behavior."""
import time
from unittest.mock import patch

import numpy as np
import pytest

from src.filters import SpatialGridMotionFilter
from src.detector import MotionDetector


def test_worker_total_does_not_double_count_render_children():
    from scripts.analyze_perf import primary_worker_time

    summary = {stage: {"total": value} for stage, value in {
        "prescreen": 2, "analysis": 10, "render": 20, "final_concat": 3,
        "render_enc": 18, "render_stage": 5,
    }.items()}
    assert primary_worker_time(summary) == 35
    assert primary_worker_time({"render_enc": {"total": 18}}) == 0


def test_decoder_comparison_rejects_missing_or_different_frames():
    from scripts.compare_analysis_decoders import compare_frames

    frame = np.zeros((4, 4), dtype=np.uint8)
    assert compare_frames([frame, frame], [frame, frame])["equivalent"]
    assert not compare_frames([frame, frame], [frame])["equivalent"]
    assert not compare_frames([frame], [frame + 1])["equivalent"]
    assert not compare_frames([frame], [frame[:2]])["equivalent"]
    assert not compare_frames([], [])["equivalent"]


def test_perf_log_json_schema_validity(tmp_path):
    import json
    from src.monitor import PerfCollector, PerfRecord
    from tests.helpers import validate_perf_json_schema

    perf = PerfCollector()
    for stage, device, duration in [("prescreen", "qsv", .04), ("analysis", "cuda", .12), ("render", "nv", 1.5)]:
        perf.add(PerfRecord(stage=stage, file="clip.mp4", gpu=device, duration=duration))
    path = tmp_path / "perf.json"
    perf.dump(path, metadata={
        "date": "20260901", "cam": 0, "pipeline_duration": 1.66,
        "monitor_summary": [{"name": "pipeline", "duration": 1.66, "avg_cpu": 25.0, "avg_ram": 8.0}],
        "perf_summary": perf.summary_by_stage(),
    })
    valid, message = validate_perf_json_schema(json.loads(path.read_text(encoding="utf-8")))
    assert valid, message


class ReferenceGrid(SpatialGridMotionFilter):
    def extract_grid_energies(self, image):
        h, w = image.shape
        ys = np.linspace(0, h, self.grid_rows + 1, dtype=int)
        xs = np.linspace(0, w, self.grid_cols + 1, dtype=int)
        result = np.zeros((self.grid_rows, self.grid_cols), np.float32)
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                cell = image[ys[r]:ys[r + 1], xs[c]:xs[c + 1]]
                result[r, c] = cell.mean() if cell.size else 0
        return result


def test_late_prescreen_completion_gets_analysis_priority():
    from src.pipeline import AnalysisQueue
    pending = AnalysisQueue()
    late_day = {"file_start_time": "20260320120000"}
    early_day = {"file_start_time": "20260320000000"}
    pending.put(late_day)
    pending.put(early_day)
    pending.put(dict(early_day))
    assert pending.get() is early_day
    pending.task_done()
    assert pending.get() == early_day
    pending.task_done()
    assert pending.get() is late_day
    pending.task_done()
    pending.join()


@pytest.mark.parametrize("shape", [(198, 332), (234, 416), (3, 5)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_integral_grid_matches_reference_state(shape, dtype):
    fast, reference = SpatialGridMotionFilter(), ReferenceGrid()
    rng = np.random.default_rng(23)
    for i in range(80):
        image = rng.random(shape).astype(dtype)
        if i % 7 == 0:
            image *= 25  # Global flashes.
        elif i % 3 == 0:
            image[:shape[0] // 2, :shape[1] // 2] += 8  # Local motion.
        actual, expected = fast.process_frame(image, .5), reference.process_frame(image, .5)
        assert actual[1] == expected[1]
        assert actual[0] == pytest.approx(expected[0], rel=2e-6, abs=1e-6)
        np.testing.assert_array_equal(fast.active_grid, reference.active_grid)
        np.testing.assert_allclose(fast.noise_floor_grid, reference.noise_floor_grid, rtol=2e-6)
        np.testing.assert_array_equal(fast.confidence_grid, reference.confidence_grid)


def test_frame_budget_failure_does_not_decode_source_twice():
    detector = MotionDetector({})
    with patch.object(detector, "_decode_file_pipe", side_effect=MemoryError("budget")), \
         patch.object(detector, "_decode_file_pyav") as fallback:
        with pytest.raises(MemoryError):
            detector._decode_file("source.mp4", 3600)
        fallback.assert_not_called()


@pytest.mark.parametrize("fps", [.5, 1, 2, 5])
@pytest.mark.parametrize("ema", [True, False])
def test_streamed_motion_preserves_labels_and_closed_timeline(fps, ema):
    from src.motion_trace import MotionTrace
    from src.frame_pool import FramePool
    cfg = {"detection": {"analysis_resolution": "96x64", "ema_background_enabled": ema},
           "audio_vad": {"enabled": False}}
    detector = MotionDetector(cfg)
    reference = MotionDetector(cfg)
    reference_grid = reference.create_grid_filter()
    reference_grid.__class__ = ReferenceGrid
    reference.create_grid_filter = lambda: reference_grid
    trace, pool = MotionTrace(detector, fps), FramePool(4 * 1024 * 1024)
    rng = np.random.default_rng(19)
    frame = np.empty((64, 96), np.uint8)
    for i in range(100):
        frame[:] = rng.integers(0, 5, frame.shape, dtype=np.uint8)
        if 20 <= i <= 50:
            frame[20:45, i:i + 10] = 150
        trace.append(frame)
        pool.append(frame)
    actual, _ = detector.analyze_frames(trace, start_offset=83, file_duration=100/fps, fps=fps)
    expected, _ = reference.analyze_frames(pool, start_offset=83, file_duration=100/fps, fps=fps)
    assert len(actual) == len(expected)
    for a, e in zip(actual, expected):
        for key in a:
            if isinstance(a[key], float):
                assert a[key] == pytest.approx(e[key], rel=2e-6, abs=1e-6)
            else:
                assert a[key] == e[key]
    assert actual[-1]["time"] == 83 + 100/fps
    assert len(trace.energies) * trace.energies.itemsize == 800


@pytest.mark.parametrize("workers", [1, 3])
def test_render_concurrency_controls_real_workers(tmp_path, workers):
    import threading
    from src.database import VlogDatabase
    from src.pipeline import StreamingOrchestrator
    db = VlogDatabase(db_path=tmp_path / "workers.db")
    cfg = {"pipeline": {"render_gpu_policy": "nv_only", "render_start_delay": 0},
           "render": {"batch_max_files": 1, "max_concurrency": workers}}
    orch = StreamingOrchestrator(db, "20260320", 0, cfg, dashboard_enabled=False)
    for i in range(workers):
        fp = f"file{i}.mp4"
        db.add_file_task(fp, 0, "20260320", f"202603200{i}0000", f"202603200{i}0100", 60)
        db.set_prescreen_result(fp, "STATIC")
        orch.render_batch_queue.put({"filepath": fp, "status": "STATIC"})
    barrier = threading.Barrier(workers, timeout=5)
    identities = set()
    lock = threading.Lock()

    def render(segs, bi, gpu, *args, **kwargs):
        assert gpu == "nv"
        assert kwargs["config"] is cfg
        with lock:
            identities.add(threading.get_ident())
        barrier.wait()
        return str(tmp_path / f"batch{bi}.mp4")

    orch.stop_event.set()
    try:
        with patch("src.pipeline.build_batch_render", side_effect=render):
            orch._render_manager()
        assert not orch.errors
        assert len(identities) == workers
        assert len(orch.batch_paths) == workers
    finally:
        db.close()


def test_profile_decode_cost_receipt_structure(tmp_path):
    """Verify non-invasive decode profiler produces structured 6-stage diagnostic receipt."""
    from scripts.profile_decode_cost import resolve_media_path, profile_pass15_analysis
    from src.core.config import load_config
    import json

    raw_media = "scratch/complex_batch_diagnosis/e35443eff91f4b11a0ba328cadffdb19/00_20260328002805_20260328013503.mp4"
    try:
        media_path = resolve_media_path(raw_media)
    except FileNotFoundError:
        pytest.skip(f"Target media {raw_media} not available in current environment")

    cfg = load_config()
    out_json = tmp_path / "decode_breakdown.json"

    # Profile 30 frames with CUDA or CPU
    report = profile_pass15_analysis(
        video_path=media_path,
        config=cfg,
        max_frames=30,
        hwaccel_mode="cuda",
        analysis_res="416x234",
        analysis_fps=5.0,
        enable_yolo=False,
    )

    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    loaded = json.loads(out_json.read_text(encoding="utf-8"))

    # 1. Structure contract validation
    assert loaded["report_version"] == "1.0"
    assert "input_media" in loaded
    assert loaded["input_media"]["native_resolution"] == "3840x2160"
    assert loaded["execution_summary"]["video_frames_demuxed"] == 30

    # 2. 6-stage breakdown validation
    stages = loaded["stages"]
    assert len(stages) == 6
    stage_keys = [s["key"] for s in stages]
    assert stage_keys == [
        "demux_and_video_decode",
        "grayscale_and_resize",
        "ema_motion_analysis",
        "candidate_jpeg_encoding",
        "audio_demux_and_vad",
        "yolo_verification",
    ]

    # 3. Normalized per-10k frames timing
    for s in stages[:5]:
        assert s["total_ms"] > 0
        assert s["time_per_10k_frames_s"] > 0
        assert 0.0 <= s["pct_of_analyzed"] <= 100.0

    total_pct = sum(s["pct_of_analyzed"] for s in stages)
    assert abs(total_pct - 100.0) < 0.5

    # 4. Hardware metrics validation
    hw = loaded["hardware_utilization"]
    assert "cpu_avg_pct" in hw
    assert "ram_peak_mb" in hw
    assert "gpu_name" in hw


if __name__ == "__main__":
    image = np.random.default_rng(23).random((198, 332))
    for implementation in (ReferenceGrid, SpatialGridMotionFilter):
        grid = implementation()
        start = time.perf_counter()
        for _ in range(5000):
            grid.extract_grid_energies(image)
        print(implementation.__name__, round(time.perf_counter() - start, 3), "seconds / 5000 frames")


@pytest.mark.parametrize("mode", ["render_only", "full"])
def test_day_benchmark_seeds_explicit_isolated_workload(tmp_path, mode):
    from scripts.benchmark_day import seed_workload
    from src.core.database import VlogDatabase
    source = str(tmp_path / "00_20260829000000_20260829000020.mp4")
    task = {"id": 7, "filepath": source, "date": "20260829", "cam_index": 0,
            "camera_id": "camera", "file_start_time": "20260829000000", "file_end_time": "20260829000010",
            "file_duration": 9.4, "prescreen_status": "SUSPICIOUS", "analysis_status": "ANALYZED",
            "analysis_segments": "[]", "duration_verified": 1, "processing_fingerprint": "historical", "has_audio": 1}
    segment = {"id": 19, "file_id": 7, "filepath": source, "date": "20260829", "cam_index": 0,
               "start_time": 0, "end_time": 9.4, "duration": 9.4, "state": "DYNAMIC"}
    review = {"filepath": source, "start_time": 2, "end_time": 3, "manual_label": "CONFIRMED_MOTION"}
    workload = {"tasks": [task], "segments": [segment], "human_reviews": [review],
                "camera_registry": [{"camera_id": "camera", "cam_index": 0}]}
    db = VlogDatabase(tmp_path / "isolated.sqlite")
    try:
        seed_workload(db, workload, mode)
        saved = dict(db.conn.execute("SELECT * FROM file_tasks").fetchone())
        assert db.conn.execute("SELECT count(*) FROM human_reviews").fetchone()[0] == 1
        if mode == "render_only":
            assert saved["file_duration"] == 9.4 and saved["processing_fingerprint"] == "historical"
            assert db.conn.execute("SELECT count(*) FROM segments").fetchone()[0] == 1
        else:
            assert saved["file_duration"] == 20 and saved["file_end_time"] == "20260829000020"
            assert saved["duration_verified"] == 0 and saved["processing_fingerprint"] is None
            assert saved["prescreen_status"] == saved["analysis_status"] == "PENDING"
            assert db.conn.execute("SELECT count(*) FROM segments").fetchone()[0] == 0
        assert task["file_duration"] == 9.4 and task["processing_fingerprint"] == "historical"
        with pytest.raises(ValueError, match="empty database"):
            seed_workload(db, workload, mode)
    finally:
        db.close()


def test_day_benchmark_invalid_seed_is_transactional(tmp_path):
    from scripts.benchmark_day import seed_workload
    from src.core.database import VlogDatabase
    db = VlogDatabase(tmp_path / "rollback.sqlite")
    try:
        workload = {"camera_registry": [{"camera_id": "camera", "cam_index": 0}],
                    "tasks": [{"unknown_column": 1}], "human_reviews": [], "segments": []}
        with pytest.raises(ValueError, match="Unknown workload columns"):
            seed_workload(db, workload, "render_only")
        assert db.conn.execute("SELECT count(*) FROM camera_registry").fetchone()[0] == 0
        assert db.conn.execute("SELECT count(*) FROM file_tasks").fetchone()[0] == 0
    finally:
        db.close()
