"""Efficiency changes must preserve temporal and classification behavior."""
import time
from unittest.mock import patch

import numpy as np
import pytest

from src.filters import SpatialGridMotionFilter
from src.detector import MotionDetector


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

    def render(segs, bi, gpu, *args):
        assert gpu == "nv"
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


if __name__ == "__main__":
    image = np.random.default_rng(23).random((198, 332))
    for implementation in (ReferenceGrid, SpatialGridMotionFilter):
        grid = implementation()
        start = time.perf_counter()
        for _ in range(5000):
            grid.extract_grid_energies(image)
        print(implementation.__name__, round(time.perf_counter() - start, 3), "seconds / 5000 frames")
