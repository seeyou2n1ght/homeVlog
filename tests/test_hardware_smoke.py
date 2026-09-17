"""Opt-in real QSV/NVDEC/NVENC decode and render, without touching user media."""
import os
from pathlib import Path

import av
import numpy as np
import pytest

from src.detector import MotionDetector
from src.renderer import build_batch_render
from src.render_cache import valid_video
from src.timeline import TimelineSegment

pytestmark = pytest.mark.skipif(os.environ.get("HOMEVLOG_HARDWARE_TESTS") != "1",
                                reason="Set HOMEVLOG_HARDWARE_TESTS=1 on QSV + NVIDIA host")


@pytest.mark.parametrize("gpu", ["qsv", "nv"])
@pytest.mark.parametrize("kind", ["mixed", "static"])
def test_real_hardware_decode_render(tmp_path, monkeypatch, gpu, kind):
    source = tmp_path / "source.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 128
        stream.pix_fmt = "yuv420p"
        for i in range(1200 if kind == "static" else 60):
            pixels = np.zeros((128,128,3), dtype=np.uint8)
            pixels[20:80, (i % 40):60+(i % 40)] = 180
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    config = {"detection": {"analysis_resolution": "128x128", "analysis_fps_adaptive": False,
                            "analysis_fps": 2}, "audio_vad": {"enabled": False},
              "render": {"speed_ramping_enabled": False}}
    monkeypatch.setattr("src.renderer.TEMP_DIR", tmp_path)
    monkeypatch.setattr("src.utils.TEMP_DIR", tmp_path)
    monkeypatch.setattr("src.utils.load_config", lambda: config)
    detector = MotionDetector(config, decode_gpu="qsv" if gpu == "qsv" else "cuda")
    duration = 120 if kind == "static" else 6
    frames, _, meta = detector._decode_file_pipe(str(source), duration, 2)
    assert meta["complete"] and len(frames) == duration * 2
    timeline = [TimelineSegment(str(source),0,0,3,"DYNAMIC",3),
                TimelineSegment(str(source),0,3,6,"STATIC",3)]
    if kind == "static":
        timeline = [TimelineSegment(str(source),0,0,120,"STATIC",120)]
    path = build_batch_render(timeline,0,gpu,10,640,360,{}, {}, {},"20260901",0,
                              [{"filepath":str(source),"has_audio":0}])
    assert path and valid_video(Path(path), 2 if kind == "static" else 4.5)


def test_real_yolo_microbatch():
    from src.utils import PROJECT_ROOT
    from src.yolo_verifier import YoloVerifier
    from src.segment import Segment
    model = PROJECT_ROOT / "models" / "yolo11m.pt"
    if not model.exists():
        pytest.skip("Local weights are required; hardware smoke never downloads weights")
    verifier = YoloVerifier({"yolo": {"enabled": True, "model_path": str(model),
                                      "device": "cuda:0", "batch_size": 2}})
    segments = [Segment(0,4,"DYNAMIC","synthetic")]
    result = verifier.verify("synthetic", segments, frames_buffer={
        i: np.zeros((234,416,3),np.uint8) for i in range(8)}, analysis_fps=2)
    assert result[0].state == "STATIC" and result[0].needs_review


def test_real_mixed_encoder_concat_preserves_every_frame(tmp_path, monkeypatch):
    import hashlib
    from src.renderer import concat_output_files
    from src.utils import load_config
    source = tmp_path / "source.mp4"
    with av.open(str(source), "w") as container:
        stream = container.add_stream("libx264", rate=20)
        stream.width, stream.height, stream.pix_fmt = 640, 360, "yuv420p"
        for i in range(40):
            pixels = np.zeros((360, 640, 3), dtype=np.uint8)
            pixels[30:200, i*4:i*4+100] = (180, 90, 60)
            for packet in stream.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    monkeypatch.setattr("src.renderer.TEMP_DIR", tmp_path)
    out_cfg = load_config()["output"]
    clips = []
    for i, gpu in enumerate(["nv", "qsv"]):
        path = build_batch_render([TimelineSegment(str(source), 0, 0, 2, "DYNAMIC", 2)],
                                  i, gpu, 20, 640, 360, {}, out_cfg, {}, "20260901", 0,
                                  [{"filepath": str(source), "has_audio": 0}])
        assert path
        clips.append(Path(path))
    def frames(path):
        with av.open(str(path)) as container:
            return [hashlib.sha256(f.to_ndarray(format="rgb24").tobytes()).hexdigest()
                    for f in container.decode(video=0)]
    expected = frames(clips[0]) + frames(clips[1])
    output = tmp_path / "joined.mp4"
    assert concat_output_files(clips, output)
    assert len(expected) == 80
    assert frames(output) == expected
