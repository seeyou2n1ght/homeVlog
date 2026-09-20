"""Behavioral regressions for the 2026-09-08 architecture review."""
import copy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import av
import numpy as np
import pytest

from src.database import VlogDatabase
from src.detector import MotionDetector
from src.prescreen import _prescreen_keyframes
from src.segment import Segment
from src.yolo_verifier import YoloVerifier


def make_video(path, seconds=60, fps=1, motion_at=20):
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = stream.height = 128
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 1
        for i in range(seconds * fps):
            pixels = np.zeros((128, 128, 3), dtype=np.uint8)
            if i / fps >= motion_at:
                pixels[24:104, 24:104] = 220 if i % 2 else 70
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


def test_prescreen_reaches_late_motion(tmp_path):
    path = make_video(tmp_path / "late.mp4")
    assert _prescreen_keyframes(str(path), 60, 10, 8)["status"] == "SUSPICIOUS"


def test_one_keyframe_is_unknown_not_static(tmp_path):
    path = make_video(tmp_path / "one.mp4", seconds=1)
    assert _prescreen_keyframes(str(path), 1, 10, 8)["status"] == "SUSPICIOUS"


def test_late_motion_survives_legacy_early_stop_setting(mock_config):
    config = copy.deepcopy(mock_config)
    frames = []
    for i in range(300):
        frame = np.zeros((234, 416), dtype=np.uint8)
        if i >= 100:
            x = 60 + (i * 9) % 240
            frame[60:170, x:x+60] = 220
        frames.append(frame)
    labels, meta = MotionDetector(config).analyze_frames(frames, file_duration=60, fps=5)
    assert not meta["early_terminated"]
    assert sum(label["is_motion"] for label in labels if label["time"] >= 20) > 100
    assert labels[-1]["time"] == 60


def verifier(model, batch=2):
    instance = YoloVerifier.__new__(YoloVerifier)
    instance.enabled = True
    instance.confidence = 0.3
    instance.sample_fps = 2
    instance.target_classes = {0}
    instance.device = "cpu"
    instance.batch_size = batch
    instance.model = model
    return instance


def test_missing_samples_never_become_negative():
    model = lambda frames, **kw: [SimpleNamespace(boxes=None) for _ in frames]
    segments = [Segment(0, 4, "DYNAMIC", "x"), Segment(100, 104, "DYNAMIC", "x")]
    result = verifier(model).verify("x", segments, frames_buffer={0: np.zeros((32,32,3), np.uint8)}, analysis_fps=2)
    assert all(s.state == "DYNAMIC" and s.needs_review for s in result)


def test_low_confidence_enters_predict_and_microbatches_are_bounded():
    import torch
    sizes = []
    def predict(frames, **kwargs):
        sizes.append(len(frames))
        # Mimic upstream filtering: the old implementation silently used 0.25.
        boxes = SimpleNamespace(cls=torch.tensor([0]), conf=torch.tensor([0.18])) if kwargs.get("conf", .25) <= .18 else None
        return [SimpleNamespace(boxes=boxes) for _ in frames]
    frames = {i: np.zeros((32,32,3), np.uint8) for i in range(40)}
    result = verifier(predict).verify("x", [Segment(0,20,"DYNAMIC","x")], frames_buffer=frames, analysis_fps=2)
    assert result[0].state == "DYNAMIC"
    assert result[0].avg_confidence == .18
    assert max(sizes) <= 2 and sum(sizes) == 8


def test_partial_decode_falls_back_and_cancel_does_not(mock_config):
    detector = MotionDetector(mock_config)
    partial = ([np.zeros((1,1))], {}, {"complete": False, "aborted": False})
    fallback = ([], {}, np.array([]), {"complete": False})
    with patch.object(detector, "_decode_file_pipe", return_value=partial), patch.object(detector, "_decode_file_pyav", return_value=fallback) as retry:
        assert detector._decode_file("fake", 10) is fallback
        retry.assert_called_once()
    partial[2]["aborted"] = True
    with patch.object(detector, "_decode_file_pipe", return_value=partial), patch.object(detector, "_decode_file_pyav") as retry:
        assert detector._decode_file("fake", 10)[0] == []
        retry.assert_not_called()


def test_valid_small_video_and_cache_identity(tmp_path):
    from src.render_cache import valid_video, render_fingerprint, reusable, save_manifest
    path = make_video(tmp_path / "short.mp4", seconds=2)
    assert path.stat().st_size < 512 * 1024
    assert valid_video(path, 2)
    assert not valid_video(path, 20)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    def identity(graph="original"):
        return render_fingerprint([source], graph, "nv", 20, {}, {}, {})
    fingerprint = identity()
    save_manifest(path, fingerprint)
    assert reusable(path, identity(), 2)
    assert not reusable(path, identity("changed-human-label"), 2)
    source.write_bytes(b"changed-source")
    assert not reusable(path, identity(), 2)


def test_valid_video_duration_tolerance(tmp_path):
    from src.render_cache import valid_video
    path = make_video(tmp_path / "tol.mp4", seconds=10)
    # 期望 10s: 匹配 -> True
    assert valid_video(path, 10.0)
    # 期望 13.0s: 偏差 3.0s (<= 5s 容差下限) -> True (保护轻微断流视频)
    assert valid_video(path, 13.0)
    # 期望 20.0s: 偏差 10.0s (> 5s 且 > 5%) -> False
    assert not valid_video(path, 20.0)


def test_valid_video_checkpoints_sequential_and_parallel(tmp_path):
    from src.render_cache import valid_video
    path = make_video(tmp_path / "checkpoints.mp4", seconds=20, fps=5)
    # 1. 无接缝校验
    assert valid_video(path, 20.0, checkpoints=())
    # 2. 顺序快速路径 (<= 4 个接缝)
    assert valid_video(path, 20.0, checkpoints=[3.0, 7.0, 12.0])
    # 3. 多线程并行校验路径 (> 4 个接缝)
    assert valid_video(path, 20.0, checkpoints=[2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0])
    # 4. 超出时长的非法接缝正确拦截
    assert not valid_video(path, 20.0, checkpoints=[2.0, 50.0])




def test_camera_indices_persist_across_scan_order(tmp_path):
    dbpath = tmp_path / "db.sqlite"
    files = [str(tmp_path / f"XiaomiCamera_01_{mac}" / "00_20260901000000_20260901000100.mp4")
             for mac in ["AABBCCDDEE01", "AABBCCDDEE02"]]
    db = VlogDatabase(dbpath)
    for fp in files:
        db.add_file_task(fp, 0, "20260901", "20260901000000", "20260901000100", 60)
    before = dict(db.conn.execute("SELECT filepath,cam_index FROM file_tasks"))
    assert len(set(before.values())) == 2
    db.close()
    db = VlogDatabase(dbpath)
    for fp in reversed(files):
        assert not db.add_file_task(fp, 0, "20260901", "20260901000000", "20260901000100", 60)
    assert dict(db.conn.execute("SELECT filepath,cam_index FROM file_tasks")) == before
    db.close()


def test_human_interval_survives_reanalysis_and_smoothing(tmp_path):
    from src.timeline import build_timeline_from_rows
    db = VlogDatabase(tmp_path / "review.sqlite")
    db.add_file_task("x.mp4",0,"20260901","20260901000000","20260901000100",60)
    db.set_prescreen_result("x.mp4", "SUSPICIOUS")
    db.set_analysis_result("x.mp4", [Segment(20,21,"STATIC","x.mp4")])
    db.update_segment_review(db.get_segments_for_file("x.mp4")[0]["id"], "FN")
    db.set_analysis_result("x.mp4", [Segment(0,60,"STATIC","x.mp4")])
    timeline = build_timeline_from_rows(db.get_all_file_tasks_for_date("20260901",0), "20260901", config={})
    assert [(s.start_in_file,s.end_in_file) for s in timeline if s.state == "DYNAMIC"] == [(20,21)]
    db.close()


def test_failed_database_write_propagates(tmp_path):
    db = VlogDatabase(tmp_path / "bad.sqlite")
    db.conn.execute("DROP TABLE segments")
    db.add_file_task("x",0,"20260901","20260901000000","20260901000100",60)
    with pytest.raises(Exception):
        db.set_analysis_result("x", [Segment(0,60,"STATIC","x")])
    assert db.conn.execute("SELECT analysis_status FROM file_tasks").fetchone()[0] == "PENDING"
    db.close()


def test_worker_failure_signals_abort(tmp_path):
    from src.pipeline import StreamingOrchestrator
    db = VlogDatabase(tmp_path / "worker.sqlite")
    orch = StreamingOrchestrator(db,"20260901",0,{},render_enabled=False,dashboard_enabled=False)
    def fail():
        raise RuntimeError("worker failed")
    orch._guard_worker(fail)
    assert orch.abort_event.is_set() and orch.render_finished_event.is_set()
    db.close()


@pytest.mark.parametrize("label", ["FP", "TN", "FALSE_ALARM", "CONFIRMED_STATIC"])
def test_all_negative_labels_are_empty(tmp_path, label):
    from scripts.export_dataset import DatasetItem, generate_yolo_labels_for_item
    target = tmp_path / "label.txt"
    model = MagicMock()
    item = DatasetItem(tmp_path/"image.jpg", label, 1, "x", 0)
    assert generate_yolo_labels_for_item(item,target,model) == 0
    assert target.read_bytes() == b""
    model.assert_not_called()


def test_frame_pool_is_lossless_and_bounded():
    from src.frame_pool import FramePool
    frame = np.random.default_rng(3).integers(0,256,(128,128),dtype=np.uint8)
    pool = FramePool(20000)
    pool.append(frame)
    np.testing.assert_array_equal(next(iter(pool)), frame)
    with pytest.raises(MemoryError):
        pool.append(frame)


def test_io_budget_reserves_multi_input_atomically():
    from src.scheduler import FileBudget
    budget = FileBudget(8)
    assert budget.acquire(weight=6,timeout=0)
    assert not budget.acquire(weight=3,timeout=0)
    assert budget.available == 2
    budget.release(weight=6)
    assert budget.acquire(weight=8,timeout=0)
    budget.release(weight=8)
    with pytest.raises(ValueError):
        budget.release()


def test_streaming_audio_matches_full_pcm():
    from src.audio_features import AudioFeatures
    from src.filters import AudioEnergyVAD
    samples = np.zeros(16000 * 5, np.float32)
    samples[16000:48000] = .15 * np.sin(np.arange(32000) * 2 * np.pi * 220 / 16000)
    features = AudioFeatures()
    for i in range(0,len(samples),997):
        features.append(samples[i:i+997])
    vad = AudioEnergyVAD()
    raw, stats = vad.detect_events(samples)
    streamed, stream_stats = vad.detect_events(features)
    np.testing.assert_allclose([x[:2] for x in streamed], [x[:2] for x in raw], atol=1e-5)
    assert stats == stream_stats


def test_analysis_json_matches_relational_segments(tmp_path):
    import json
    db = VlogDatabase(tmp_path / 'segments.sqlite')
    db.add_file_task('x', 0, '20260901', '20260901000000', '20260901000100', 60)
    db.set_analysis_result('x', [Segment(0, 60, 'DYNAMIC', 'x')])
    row = db.get_all_file_tasks_for_date('20260901', 0)[0]
    assert json.loads(row['analysis_segments'])[0]['state'] == 'DYNAMIC'
    assert row['segments'][0]['state'] == 'DYNAMIC'
    db.close()


@pytest.mark.parametrize("settings", [
    'hardware:\n  max_nv_concurrency: 0\n',
    'output:\n  qsv:\n    global_quality: 28\n    maxrate: 4M\n',
])
def test_invalid_config_does_not_poison_cache(tmp_path, monkeypatch, settings):
    import src.utils as utils
    config = tmp_path / 'invalid.yaml'
    config.write_text(settings)
    previous = {'hardware': {'max_nv_concurrency': 2}}
    monkeypatch.setattr(utils, 'SETTINGS', previous)
    monkeypatch.setattr(utils, 'CONFIG_PATH', config)
    with pytest.raises(ValueError):
        utils.load_config(reload=True)
    assert utils.SETTINGS is previous


def test_pyav_fallback_completes_and_cancel_is_incomplete(tmp_path):
    from src.renderer import FFmpegProcessRegistry
    source = make_video(tmp_path / 'fallback.mp4', seconds=4, fps=2)
    detector = MotionDetector({'detection': {'analysis_resolution': '128x128',
                              'analysis_fps_adaptive': False, 'analysis_fps': 2},
                              'audio_vad': {'enabled': False}}, decode_gpu='qsv')
    with patch('av.codec.hwaccel.HWAccel', side_effect=RuntimeError('software fallback')):
        frames, _, _, meta = detector._decode_file_pyav(str(source), 4)
        assert len(frames) == 8 and meta['complete']
        FFmpegProcessRegistry.mark_interrupted()
        frames, _, _, meta = detector._decode_file_pyav(str(source), 4)
        assert not meta.get('complete')


def test_stream_consumer_failure_releases_process_and_io(monkeypatch):
    from src.ffmpeg import run_ffmpeg
    from src.renderer import FFmpegProcessRegistry
    from src.scheduler import FileBudget
    budget = FileBudget(1)
    monkeypatch.setattr('src.utils.get_disk_semaphore', lambda: budget)
    observed = []
    def fail(chunk):
        observed.extend(FFmpegProcessRegistry._processes.values())
        raise RuntimeError('consumer failed')
    with pytest.raises(RuntimeError, match='consumer failed'):
        run_ffmpeg(['-f', 'lavfi', '-i', 'sine=duration=2', '-f', 'f32le', '-'],
                   timeout=10, stdout_consumer=fail)
    assert observed and all(p.poll() is not None for p in observed)
    assert not FFmpegProcessRegistry._processes and budget.available == 1


def test_disabling_vad_preserves_audio_stream_metadata(monkeypatch):
    detector = MotionDetector({'audio_vad': {'enabled': False}})
    detector.has_audio_detected = 1
    monkeypatch.setattr(detector, '_decode_file_pipe', lambda *a: ([np.zeros((2,2))], {}, {'complete': True}))
    _, _, audio, meta = detector._decode_file('synthetic', 1)
    assert audio.size == 0 and meta['has_audio'] == 1


def test_render_batch_reserves_analysis_io_slots(tmp_path):
    from src.pipeline import StreamingOrchestrator
    db = VlogDatabase(tmp_path / "budget.sqlite")
    config = {"hardware": {"max_io_concurrency": 8},
              "detection": {"analysis_max_workers": 4, "prescreen_parallel": 8},
              "render": {"batch_max_files": 8, "max_concurrency": 2}}
    orch = StreamingOrchestrator(db, "20260901", 0, config,
                                 render_enabled=False, dashboard_enabled=False)
    assert orch.batch_max_files == 1
    db.close()


def test_ultra_long_tier_uses_bounded_sampling():
    detector = MotionDetector({"detection": {
        "analysis_fps": 5, "analysis_fps_adaptive": True,
        "analysis_fps_tiers": {"short": 5, "medium": 2, "long": 2, "ultra_long": 0.5},
        "analysis_fps_tier_thresholds": {"short_max": 120, "medium_max": 600, "long_max": 1800}}})
    assert detector._resolve_effective_fps(3600) == 0.5


def test_yolo_candidate_sampling_is_bounded_for_long_files():
    detector = MotionDetector({"detection": {"analysis_fps": 2},
                               "yolo": {"enabled": True, "sample_fps": 2,
                                        "max_frames_per_file": 512}})
    effective_fps = 1.0
    interval = max(1, int(round(effective_fps / detector.yolo_sample_fps)),
                   int(np.ceil(1356 * effective_fps / detector.yolo_max_frames)))
    assert interval == 3 and int(np.ceil(1356 / interval)) <= 512


def test_static_audio_gate_keeps_audio_events_conservative(monkeypatch):
    from src.detector import detect_audio_activity
    cfg = {"audio_vad": {"enabled": True}}
    class FakeDetector:
        def __init__(self, config, decode_gpu):
            self.vad = self
            self.has_audio_detected = 0
        def _decode_audio_pipe(self, filepath, duration):
            return self
        @property
        def size(self):
            return 1
        def detect_events(self, features):
            return [(1.0, 2.0, "audio")], {"noise_floor_db": -40}
    monkeypatch.setattr("src.detector.MotionDetector", FakeDetector)
    events, _ = detect_audio_activity("ignored", 2, cfg)
    assert events
