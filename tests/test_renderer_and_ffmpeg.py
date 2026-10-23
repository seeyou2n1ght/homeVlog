"""测试模块 5: 渲染引擎 (Renderer) 与 FFmpeg 进程生命周期管理.

覆盖：
1. FFmpeg 进程全局注册表 (FFmpegProcessRegistry) 追踪、注销与孤儿进程强杀；
2. 复杂滤镜图 (build_concat_filter) 生成、PTS 表达式与输入输出标签完全闭包；
3. 大时间线批次划分 (partition_timeline_by_batches) 连续性与守恒律；
4. 成片拼接器 (concat_output_files) 错误与空列表防御。
"""

import subprocess
import time
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.mark.parametrize("failure", [None, "corrupt", "missing_pts", "seek", "decode"])
def test_checkpoint_timings_preserve_validation_failures(monkeypatch, failure):
    from fractions import Fraction
    from types import SimpleNamespace
    import av
    from src.hardware import render_cache
    stream = SimpleNamespace(average_rate=20, time_base=Fraction(1, 20))

    class Container:
        streams = SimpleNamespace(video=[stream])

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def seek(self, offset, **kwargs):
            if failure == "seek":
                raise OSError("seek failed")
            self.offset = offset

        def decode(self, _stream):
            if failure == "decode":
                raise RuntimeError("decode failed")
            for i in range(21):
                yield SimpleNamespace(is_corrupt=failure == "corrupt",
                    pts=None if failure == "missing_pts" else self.offset + i,
                    time_base=stream.time_base)

    monkeypatch.setattr(av, "open", lambda *a, **k: Container())
    details = {}
    assert render_cache._validate_checkpoint_chunk([1, 2], "media", 3, diagnostics=details) is (failure is None)
    assert details["open_s"] >= 0 and details["total_s"] >= details["open_s"]
    assert details["seek_s"] >= 0
    if failure != "seek":
        assert details["decode_s"] >= 0
        assert details["windows_attempted"] == (2 if failure is None else 1)
        assert details["decoded_frames"] == (42 if failure is None else 0 if failure == "decode" else 1)
    if failure in ("seek", "decode"):
        assert details["error_type"] in ("OSError", "RuntimeError")


@pytest.mark.parametrize("kind,rate,frame_samples", [
    ("dense", 48000, 4096), ("legacy", 48000, 137), ("simple", 44100, 4096),
])
def test_final_audio_clock_uses_exact_samples_for_long_variable_frames(
        tmp_path, monkeypatch, kind, rate, frame_samples):
    import shutil
    import av
    import src.stages.timeline as timeline
    from src.hardware.ffmpeg import run_ffmpeg

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required for AAC packet clock validation")
    segments = [timeline.TimelineSegment("source", 0, 0, 10, "NIGHT_STATIONARY", 10),
                timeline.TimelineSegment("source", 0, 10, 300, "DYNAMIC", 290)]
    if kind == "simple":
        segments = [timeline.TimelineSegment("source", 0, 0, 300, "DYNAMIC", 300)]
    if kind == "legacy":
        monkeypatch.setattr(timeline, "_single_pass_dense_video", lambda *args, **kwargs: None)
    graph = timeline.build_concat_filter(
        segments, [{"filepath": "source", "has_audio": 1}], scale_mode="cpu",
        audio_sample_rate=rate, simple_dynamic=kind == "simple", speed_ramping=False,
        config={"presence": {"night_speed_factor": 16}})
    # Exercise the actual terminal chain with long inputs and differing frame boundaries.
    audio = graph.rsplit(";", 1)[-1].replace("[a_tmp]", "[0:a]")
    output = tmp_path / "clock.m4a"
    result = run_ffmpeg([
        "-f", "lavfi", "-i", f"sine=sample_rate={rate}:duration=300,asetnsamples=n={frame_samples}:p=0",
        "-filter_complex", audio, "-map", "[a]", "-c:a", "aac", "-b:a", "96k", str(output),
    ], timeout=60)
    assert result.returncode == 0 and not result.timed_out, result.stderr_text[-2000:]
    with av.open(str(output)) as container:
        stream = container.streams.audio[0]
        packets = [p for p in container.demux(stream) if p.size]
        assert len(packets) > 12000  # Includes long-range clock rounding boundaries.
        for index, packet in enumerate(packets):
            assert packet.pts == packet.dts == (index - 1) * 1024
        assert all(a.pts + a.duration == b.pts for a, b in zip(packets, packets[1:]))
        assert packets[-1].pts + packets[-1].duration == stream.duration


@pytest.mark.parametrize("offset,end", [(0, 45), (.7, 45), (0, 48)])
def test_single_pass_dense_video_matches_legacy_pixels_pts_and_audio(tmp_path, monkeypatch, offset, end):
    import shutil
    import src.stages.timeline as timeline
    from src.hardware.ffmpeg import run_ffmpeg
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required for real clock/pixel/audio equivalence")
    source = tmp_path / "input.mkv"
    generated = run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=128x128:rate=15:duration=45",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=45",
        "-c:v", "libx264", "-crf", "0", "-c:a", "pcm_s16le", "-shortest", str(source)], timeout=30)
    assert generated.returncode == 0, generated.stderr_text[-1000:]
    cfg = {"render": {"speed_ramping_enabled": False}, "presence": {"night_speed_factor": 16},
           "micro_motion": {"cruise_speed": 16}}
    intervals = [(0, 7.025, "NIGHT_STATIONARY"), (7.025, 10.275, "DYNAMIC"),
                 (10.275, 39.333, "MICRO_MOTION"), (39.333, end, "NIGHT_STATIONARY")]
    segments = [timeline.TimelineSegment(str(source), 0, start, stop, state, stop-start)
                for start, stop, state in intervals]
    kwargs = {"rows": [{"filepath": str(source), "has_audio": 1}], "output_fps": 20,
              "output_width": 64, "output_height": 64, "scale_mode": "cpu", "config": cfg}
    optimized = timeline.build_concat_filter(segments, **kwargs)
    with monkeypatch.context() as context:
        context.setattr(timeline, "_single_pass_dense_video", lambda *args, **kwargs: None)
        legacy = timeline.build_concat_filter(segments, **kwargs)
    assert "split=" not in optimized and "split=" in legacy
    for index, graph in enumerate((legacy, optimized)):
        script = tmp_path / f"graph{index}.txt"
        script.write_text(graph, encoding="utf-8")
        result = run_ffmpeg(["-itsoffset", str(offset), "-i", str(source), "-/filter_complex", str(script),
            "-map", "[v]", "-c:v", "rawvideo", "-pix_fmt", "yuv420p", "-f", "framemd5", str(tmp_path/f"{index}.frames"),
            "-map", "[a]", "-c:a", "pcm_s16le", "-f", "hash", "-hash", "sha256", str(tmp_path/f"{index}.audio")], timeout=30)
        assert result.returncode == 0, result.stderr_text[-1000:]
    assert (tmp_path/"0.frames").read_bytes() == (tmp_path/"1.frames").read_bytes()
    assert (tmp_path/"0.audio").read_bytes() == (tmp_path/"1.audio").read_bytes()


@pytest.mark.parametrize("unsupported", ["static", "osd", "multiple_files", "subframe", "overlap"])
def test_single_pass_dense_video_preserves_legacy_for_unsupported_slices(unsupported):
    import src.stages.timeline as timeline
    segments = [timeline.TimelineSegment("source.mp4", 0, 0, 10, "NIGHT_STATIONARY", 10),
                timeline.TimelineSegment("source.mp4", 0, 10, 15, "DYNAMIC", 5)]
    kwargs = {"rows": [{"filepath": "source.mp4", "has_audio": 0}], "scale_mode": "cpu"}
    if unsupported == "static":
        segments[0].state = "STATIC"
    elif unsupported == "osd":
        kwargs["timecode_osd"] = True
    elif unsupported == "multiple_files":
        segments[1].filepath = "other.mp4"
        segments[1].input_index = 1
    elif unsupported == "subframe":
        segments[0].end_in_file = .025
        segments[0].duration = .025
    else:
        segments[1].start_in_file = 9.5
    graph = timeline.build_concat_filter(segments, **kwargs)
    assert "[v_tmp]" in graph  # original paired audio/video concat remains


@pytest.mark.parametrize("cache_hit", [False, True])
def test_batch_inspection_uses_real_cache_decision_without_encoding(tmp_path, monkeypatch, cache_hit):
    from src.stages import renderer, timeline
    from src.hardware import render_cache
    monkeypatch.setattr(renderer, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(render_cache, "reusable", lambda *args, **kwargs: cache_hit)
    monkeypatch.setattr(renderer.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Inspection launched encoder"))
    segment = timeline.TimelineSegment(str(tmp_path / "source.mp4"), 0, 0, 120, "STATIC", 120)
    descriptor = tmp_path / "_edl_batch0_20261012_cam0.ffconcat"
    descriptor.write_text("preserve an existing task descriptor", encoding="utf-8")
    reports = []
    cfg = {"render": {"virtual_concat_enabled": True}}
    renderer.build_batch_render([segment], 0, "nv", 20, 128, 128, {}, {}, {}, "20261012", 0,
                                [{"filepath": segment.filepath, "has_audio": 0}], config=cfg,
                                inspect_only=True, cache_observer=reports.append)
    assert len(reports) == 1 and reports[0]["cache_hit"] is cache_hit
    assert len(reports[0]["fingerprint"]) == 64
    assert reports[0]["expected_duration_s"] > 0
    assert descriptor.read_text(encoding="utf-8") == "preserve an existing task descriptor"
    assert not list(tmp_path.glob("*.mp4"))


def test_explicit_camera_config_reaches_filter_validation_and_subtitles(tmp_path, monkeypatch):
    import src.stages.renderer as renderer
    import src.stages.timeline as timeline
    cfg = {"output": {"fps": 20}, "presence": {"speed_factor": 8},
           "render": {"virtual_concat_enabled": False, "virtual_concat_mixed_enabled": False,
                      "speed_ramping_enabled": False}}
    segment = timeline.TimelineSegment("source.mp4", 0, 0, 20, "PRESENCE", 20)
    captured = {}
    def capture(*args, **kwargs):
        captured.update(kwargs)
        captured["filter"] = args[1]
        return str(tmp_path / "batch.mp4")
    monkeypatch.setattr(renderer, "_run_batch_render", capture)
    monkeypatch.setattr(timeline, "load_config", lambda: {"presence": {"speed_factor": 4}})
    renderer.build_batch_render([segment], 0, "nv", 20, 128, 128, {}, {}, {}, "20260901", 0,
                                [{"filepath": "source.mp4", "has_audio": 0}], config=cfg)
    assert captured["config"] is cfg
    assert captured["expected_duration"] == 2.5
    assert "end_frame=50" in captured["filter"]
    srt = timeline.generate_timecode_subtitles([segment], config=cfg)
    assert "00:00:02,500" in srt
    assert "00:00:05,000" not in srt


def test_virtual_batch_unconfirmed_stop_preserves_inputs_and_never_falls_back(tmp_path, monkeypatch):
    import src.stages.renderer as renderer
    from src.stages.timeline import TimelineSegment
    monkeypatch.setattr(renderer,'TEMP_DIR',tmp_path)
    source=tmp_path/'source.mp4';source.write_bytes(b'original preserved')
    staged=tmp_path/'staged.mp4';staged.write_bytes(b'staged reader input')
    monkeypatch.setattr(renderer,'_stage_source_for_render',lambda *a,**k: staged)
    calls=[]
    def fail(*args,**kwargs):
        calls.append(args)
        assert kwargs['prepare_input']()
        raise renderer.RenderTerminationUnconfirmed('retained virtual reader')
    monkeypatch.setattr(renderer,'_run_batch_render',fail)
    segment=TimelineSegment(str(source),0,0,60,'STATIC',60)
    cfg={'render':{'virtual_concat_enabled':True,'local_staging_enabled':True}}
    with pytest.raises(renderer.RenderTerminationUnconfirmed,match='retained virtual reader'):
        renderer.build_batch_render([segment],0,'nv',20,128,128,{}, {}, {},'20260901',0,
            [{'filepath':str(source),'has_audio':0}],config=cfg)
    assert len(calls)==1
    assert source.read_bytes()==b'original preserved'
    assert staged.read_bytes()==b'staged reader input'
    assert (tmp_path/'_edl_batch0_20260901_cam0.ffconcat').exists()

from src.renderer import (
    FFmpegProcessRegistry,
    _startup_watchdog_timeout,
    build_compact_virtual_concat_plan,
    build_virtual_concat_plan,
    _normalize_file_timeline,
    build_concat_filter,
    concat_output_files,
)
from src.timeline import (
    TimelineSegment,
    partition_timeline_by_batches,
)
from tests.helpers import verify_filtergraph_labels_closure


def test_fast_forward_keeps_pitch_and_audio_event_timing():
    import shutil
    import numpy as np
    from src.timeline import build_retimed_audio

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for the audio integration check")
    graph = build_retimed_audio("0:a", "a", 0, 16, 4)
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
        r"aevalsrc=if(between(t\,8\,12)\,0.5*sin(2*PI*440*t)\,0):s=48000:d=16",
        "-filter_complex", graph, "-map", "[a]", "-f", "f32le", "pipe:1",
    ], capture_output=True, timeout=30, check=True)
    audio = np.frombuffer(result.stdout, dtype=np.float32)
    assert len(audio) == 4 * 48000
    assert np.max(np.abs(audio[12000:36000])) < 0.001
    event = audio[108000:132000]  # Source 9..11 seconds appears at display 2.25..2.75.
    assert np.sqrt(np.mean(event ** 2)) > 0.1
    frequency = np.argmax(np.abs(np.fft.rfft(event))) * 48000 / len(event)
    assert abs(frequency - 440) <= 2


def test_continuous_static_video_and_audio_share_display_duration(tmp_path, monkeypatch):
    import shutil
    import av
    import numpy as np

    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg is required for the media integration check")
    monkeypatch.setattr("src.stages.timeline.load_config", lambda: {"render": {"static_mode": "continuous"}})
    segments = [TimelineSegment("source", 0, 0, 4, "DYNAMIC", 4),
                TimelineSegment("source", 0, 4, 20, "STATIC", 16)]
    graph = build_concat_filter(segments, [{"filepath": "source", "has_audio": 1}],
                               output_fps=10, output_width=160, output_height=90,
                               static_keyframe_interval=1, keyframe_display_duration=0.25,
                               speed_ramping=False, audio_input_offset=1)
    target = tmp_path / "mixed.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=20:duration=20",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=20",
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        "-c:v", "ffv1", "-c:a", "pcm_s16le", str(target),
    ], capture_output=True, timeout=30, check=True)
    with av.open(str(target)) as container:
        frames = [f.to_ndarray(format="gray") for f in container.decode(video=0)]
    with av.open(str(target)) as container:
        samples = np.concatenate([f.to_ndarray().ravel() for f in container.decode(audio=0)])
    assert len(frames) == 80  # 4 seconds at 1x, then 16 seconds at 4x.
    assert len(samples) == 8 * 48000
    assert np.mean(np.abs(frames[45].astype(float) - frames[65])) > 5
    assert np.max(np.abs(samples[5*48000:7*48000])) > 100


@pytest.mark.parametrize("fps", [20, 24])
@pytest.mark.parametrize("state", ["DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE", "NIGHT_STATIONARY", "MICRO_MOTION", "STATIC"])
def test_fractional_segments_keep_one_video_audio_clock(tmp_path, fps, state):
    import av
    from src.timeline import compute_display_plans

    segments = [TimelineSegment("source", 0, i * 1.026, (i + 1) * 1.026,
                                state if i % 2 else "DYNAMIC", 1.026) for i in range(20)]
    plans = compute_display_plans(segments, output_fps=fps, speed_ramping=True)
    expected_frames = sum(round(duration * fps) for duration, _ in plans)
    graph = build_concat_filter(
        segments, [{"filepath": "source", "has_audio": True}],
        output_fps=fps, output_width=64, output_height=64,
        speed_ramping=True, audio_input_offset=1,
    )
    target = tmp_path / "fractional.nut"
    graph_path = tmp_path / "filter.txt"
    graph_path.write_text(graph, encoding="utf-8")
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=24:duration=22",
        # Real camera audio may contain overlapping timestamps. Sample counts,
        # rather than source PTS span, must determine the output segment clock.
        "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=22,asetpts=0.98*PTS",
        "-/filter_complex", str(graph_path), "-map", "[v]", "-map", "[a]",
        "-c:v", "ffv1", "-c:a", "pcm_s16le", str(target),
    ], capture_output=True, timeout=30, check=True)
    with av.open(str(target)) as container:
        timestamps = [float(f.pts * f.time_base) for f in container.decode(video=0)]
    with av.open(str(target)) as container:
        samples = sum(f.samples for f in container.decode(audio=0))
    assert len(timestamps) == expected_frames
    assert all(b - a == pytest.approx(1 / fps) for a, b in zip(timestamps, timestamps[1:]))
    assert samples == round(expected_frames / fps * 48000)


def test_dynamic_eof_is_padded_to_shared_video_audio_clock(tmp_path):
    import av

    segment = TimelineSegment("clip.mp4", 0, 0.0, 12.0, "DYNAMIC", 12.0)
    graph = build_concat_filter(
        [segment], [{"filepath": "clip.mp4", "has_audio": 1}],
        output_fps=20, output_width=160, output_height=90, simple_dynamic=True,
    )
    source = tmp_path / "source.mp4"
    subprocess.run([
        "ffmpeg", "-v", "error",
        "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=20:duration=8",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
        "-shortest", "-c:v", "mpeg4", "-c:a", "aac", str(source),
    ], capture_output=True, timeout=30, check=True)
    target = tmp_path / "dynamic_eof.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-i", str(source),
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        "-c:v", "ffv1", "-c:a", "pcm_s16le", str(target),
    ], capture_output=True, timeout=30, check=True)
    with av.open(str(target)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 240
    with av.open(str(target)) as container:
        assert sum(frame.samples for frame in container.decode(audio=0)) == 12 * 48000


def test_empty_final_static_slice_closes_shared_clock(tmp_path):
    import av

    segments = [
        TimelineSegment("source", 0, 0.0, 8.0, "DYNAMIC", 8.0),
        TimelineSegment("source", 0, 8.0, 8.1, "STATIC", 0.1),
    ]
    graph = build_concat_filter(
        segments, [{"filepath": "source", "has_audio": 1}],
        output_fps=20, output_width=160, output_height=90,
        speed_ramping=False, audio_input_offset=1,
    )
    target = tmp_path / "empty_tail.mkv"
    subprocess.run([
        "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=20:duration=8",
        "-f", "lavfi", "-i", "sine=sample_rate=48000:duration=8.1",
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        "-c:v", "ffv1", "-c:a", "pcm_s16le", str(target),
    ], capture_output=True, timeout=30, check=True)
    with av.open(str(target)) as container:
        assert sum(1 for _ in container.decode(video=0)) == 162
    with av.open(str(target)) as container:
        assert sum(frame.samples for frame in container.decode(audio=0)) == 162 * 48000 // 20


class TestRenderWatchdog:
    def test_multi_input_startup_gets_longer_grace(self):
        assert _startup_watchdog_timeout(8, 120.0, 600.0, 4800.0) == 600.0

    def test_startup_grace_never_exceeds_config_or_render_timeout(self):
        assert _startup_watchdog_timeout(8, 120.0, 300.0, 4800.0) == 300.0
        assert _startup_watchdog_timeout(8, 120.0, 600.0, 240.0) == 240.0

    def test_startup_grace_never_weaker_than_steady_state(self):
        assert _startup_watchdog_timeout(1, 120.0, 30.0, 3600.0) == 120.0

    def test_single_input_startup_respects_configured_timeout(self):
        assert _startup_watchdog_timeout(1, 120.0, 600.0, 3600.0) == 600.0


class TestFFmpegProcessRegistry:
    """测试 FFmpeg 进程追踪与清理守护机制。"""

    def test_registry_lifecycle_and_cleanup(self):
        # 启动一个虚拟的长生命周期休眠进程模拟 ffmpeg
        proc = subprocess.Popen(["powershell", "-Command", "Start-Sleep -Seconds 10"])
        FFmpegProcessRegistry.register("proc1", proc)

        # 验证在注册表中
        with FFmpegProcessRegistry._lock:
            assert "proc1" in FFmpegProcessRegistry._processes

        # 测试强杀清理
        FFmpegProcessRegistry.kill_all()
        time.sleep(0.1)
        assert proc.poll() is not None

        # 再次清理：注册表为空
        with FFmpegProcessRegistry._lock:
            assert len(FFmpegProcessRegistry._processes) == 0

    def test_unregister_cleans_safely(self):
        proc = subprocess.Popen(["powershell", "-Command", "Start-Sleep -Seconds 2"])
        FFmpegProcessRegistry.register("proc2", proc)
        FFmpegProcessRegistry.deregister("proc2")
        with FFmpegProcessRegistry._lock:
            assert "proc2" not in FFmpegProcessRegistry._processes
        proc.kill()


class TestFiltergraphGenerationAndClosure:
    """测试 FFmpeg Filtergraph 语法有效性与闭包不变量。"""

    def test_build_concat_filter_dynamic_and_static_closure(self):
        t1 = TimelineSegment(
            filepath="clip1.mp4",
            input_index=0,
            start_in_file=0.0,
            end_in_file=10.0,
            state="DYNAMIC",
            duration=10.0,
        )
        t2 = TimelineSegment(
            filepath="clip1.mp4",
            input_index=0,
            start_in_file=10.0,
            end_in_file=40.0,
            state="STATIC",
            duration=30.0,
        )

        rows = [{"filepath": "clip1.mp4", "has_audio": 1}]
        filter_str = build_concat_filter(
            timeline=[t1, t2],
            rows=rows,
            output_fps=20,
            output_width=1920,
            output_height=1080,
        )

        assert "concat=n=2:v=1:a=1" in filter_str
        assert "async=1000:first_pts=0,apad,atrim=end_sample=" in filter_str
        is_closed, reason = verify_filtergraph_labels_closure(filter_str)
        assert is_closed, f"Filtergraph not closed: {reason}"

    def test_presence_segment_pads_video_tail_with_tpad(self):
        """验证 PRESENCE 等变速快进段在视频末尾注入 tpad 尾帧补齐，杜绝因源切片提前断流导致音画时长漂移。"""
        t1 = TimelineSegment(
            filepath="clip1.mp4",
            input_index=0,
            start_in_file=0.0,
            end_in_file=120.0,
            state="PRESENCE",
            duration=120.0,
        )
        rows = [{"filepath": "clip1.mp4", "has_audio": 1}]
        filter_str = build_concat_filter(
            timeline=[t1],
            rows=rows,
            output_fps=20,
            output_width=1920,
            output_height=1080,
            speed_ramping=True,
        )
        assert "tpad=stop_mode=clone" in filter_str
        assert "trim=end_frame=600" in filter_str
        is_closed, reason = verify_filtergraph_labels_closure(filter_str)
        assert is_closed, f"Filtergraph not closed: {reason}"

    def test_dynamic_segment_also_closes_to_planned_duration(self):
        segment = TimelineSegment(
            filepath="clip1.mp4", input_index=0, start_in_file=0.0,
            end_in_file=10.0, state="DYNAMIC", duration=10.0,
        )
        graph = build_concat_filter(
            timeline=[segment], rows=[{"filepath": "clip1.mp4", "has_audio": 1}],
            output_fps=20, output_width=1920, output_height=1080,
        )
        assert "tpad=stop_mode=clone:stop_duration=10.000" in graph
        assert "fps=fps=20,trim=end_frame=200" in graph

    def test_keyframe_fastpath_pure_static_file(self, monkeypatch):
        """纯静态长文件走 select 抽帧快路径，滤镜图标签保持闭包。"""
        monkeypatch.setattr("src.stages.timeline.load_config", lambda: {"render": {"static_mode": "hybrid_keyframe"}})
        t1 = TimelineSegment(
            filepath="night.mp4",
            input_index=0,
            start_in_file=0.0,
            end_in_file=300.0,
            state="STATIC",
            duration=300.0,
        )
        rows = [{"filepath": "night.mp4", "has_audio": 0}]
        filter_str = build_concat_filter(
            timeline=[t1],
            rows=rows,
            output_fps=20,
            output_width=1920,
            output_height=1080,
            scale_mode="cuda_passthrough",
            static_keyframe_interval=30.0,
        )
        # 快路径：select 抽帧置于 scale/hwdownload 之前
        assert "select='isnan(prev_selected_t)+gte(t-prev_selected_t\\,30.0)'" in filter_str
        assert filter_str.index("select=") < filter_str.index("scale_cuda=")
        is_closed, reason = verify_filtergraph_labels_closure(filter_str)
        assert is_closed, f"Filtergraph not closed: {reason}"

    def test_keyframe_fastpath_not_applied_to_mixed_or_short(self):
        """混排动态段或短静态段不启用快路径，保证 trim 区间有帧。"""
        segs = [
            TimelineSegment(
                filepath="mix.mp4", input_index=0,
                start_in_file=0.0, end_in_file=120.0,
                state="STATIC", duration=120.0,
            ),
            TimelineSegment(
                filepath="mix.mp4", input_index=0,
                start_in_file=120.0, end_in_file=130.0,
                state="DYNAMIC", duration=10.0,
            ),
        ]
        rows = [{"filepath": "mix.mp4", "has_audio": 0}]
        filter_str = build_concat_filter(
            timeline=segs, rows=rows, output_fps=20,
            output_width=1920, output_height=1080,
            scale_mode="cuda_passthrough",
        )
        assert "select='isnan(prev_selected_t)" not in filter_str

    def test_sparse_mixed_drops_static_frames_before_scale(self):
        segs = [
            TimelineSegment("mix.mp4", 0, 0.0, 120.0, "STATIC", 120.0),
            TimelineSegment("mix.mp4", 0, 120.0, 150.0, "DYNAMIC", 30.0),
            TimelineSegment("mix.mp4", 0, 150.0, 270.0, "STATIC", 120.0),
        ]
        graph = build_concat_filter(
            timeline=segs,
            rows=[{"filepath": "mix.mp4", "has_audio": 1}],
            output_fps=20,
            output_width=1920,
            output_height=1080,
            scale_mode="cuda_passthrough",
            sparse_mixed=True,
            static_sample_window_s=0.25,
        )
        assert "between(t\\,119.950\\,150.050)" in graph
        assert graph.index("select=") < graph.index("scale_cuda=")
        assert "tpad=stop_mode=clone" in graph
        assert ",trim=end_frame=" in graph
        is_closed, reason = verify_filtergraph_labels_closure(graph)
        assert is_closed, f"Filtergraph not closed: {reason}"

    def test_virtual_concat_plan_preserves_dynamic_and_sparsifies_static(self):
        segs = [
            TimelineSegment("clip.mp4", 0, 10.0, 130.0, "STATIC", 120.0),
            TimelineSegment("clip.mp4", 0, 130.0, 140.0, "DYNAMIC", 10.0),
        ]
        plan = build_virtual_concat_plan(
            segs, static_sample_window_s=0.25,
        )
        assert plan.entries == 2
        assert plan.static_entries == 1
        assert "duration 120.000000" in plan.text
        assert "inpoint 130.000000" in plan.text
        assert "outpoint 140.000000" in plan.text

    def test_compact_virtual_plan_maps_sparse_source_to_dense_pts(self):
        segs = [
            TimelineSegment("clip.mp4", 0, 0.0, 120.0, "STATIC", 120.0),
            TimelineSegment("clip.mp4", 0, 120.0, 130.0, "DYNAMIC", 10.0),
            TimelineSegment("clip.mp4", 0, 130.0, 250.0, "STATIC", 120.0),
        ]
        plan = build_compact_virtual_concat_plan(
            segs, static_sample_window_s=10.0, dynamic_coalesce_gap_s=0.0,
        )
        assert plan.entries == 4
        assert plan.static_entries == 2
        assert [round(s.end_in_file - s.start_in_file, 3) for s in plan.mapped_timeline] == [10.0, 10.0, 10.0]
        assert plan.mapped_timeline[0].start_in_file == pytest.approx(10.0)
        assert [s.duration for s in plan.source_timeline] == [120.0, 10.0, 120.0]
        graph = build_concat_filter(
            list(plan.mapped_timeline),
            [{"filepath": "clip.mp4", "has_audio": 1}],
            preselected_static=True,
            concat_demuxer=True,
            audio_input_offset=1,
            source_timeline=list(plan.source_timeline),
        )
        assert "trim=start=20.000:end=30.000" in graph
        assert "[1:a]atrim=start=120.000:end=130.000" in graph

    def test_normalize_file_timeline_clips_historical_overlap(self):
        segs = [
            TimelineSegment("clip.mp4", 0, 0.0, 0.5, "STATIC", 0.5),
            TimelineSegment("clip.mp4", 0, 0.0, 375.0, "STATIC", 375.0),
            TimelineSegment("clip.mp4", 0, 375.0, 378.5, "DYNAMIC_AUDIO", 3.5),
        ]
        normalized = _normalize_file_timeline(segs)
        assert [(s.start_in_file, s.end_in_file) for s in normalized] == [
            (0.0, 375.0), (375.0, 378.5)
        ]
        next_file = TimelineSegment("next.mp4", 1, 0.0, 10.0, "DYNAMIC", 10.0)
        combined = _normalize_file_timeline(normalized + [next_file])
        assert combined[-1] == next_file
        assert sum(s.duration for s in combined) == pytest.approx(388.5)

    def test_virtual_concat_filter_selects_demuxer_ranges_before_scale(self):
        segs = [
            TimelineSegment("clip.mp4", 0, 0.0, 120.0, "STATIC", 120.0),
            TimelineSegment("clip.mp4", 0, 120.0, 130.0, "DYNAMIC", 10.0),
        ]
        graph = build_concat_filter(
            timeline=segs,
            rows=[{"filepath": "clip.mp4", "has_audio": 1}],
            scale_mode="cuda_passthrough",
            preselected_static=True,
            concat_demuxer=True,
            audio_input_offset=1,
        )
        # Concat inpoints include GOP preroll; segment metadata clips it before
        # scale and the timeline trim owns final segment boundaries.
        assert "select=concatdec_select" not in graph
        assert "[1:a]atrim=" in graph
        assert "select='isnan(prev_selected_t)" not in graph
        assert "tpad=stop_mode=clone" in graph
        assert ",trim=end_frame=" in graph

    def test_simple_dynamic_filter_bypasses_concat_graph(self):
        seg = TimelineSegment("clip.mp4", 0, 0.0, 30.0, "DYNAMIC", 30.0)
        graph = build_concat_filter(
            [seg], [{"filepath": "clip.mp4", "has_audio": 0}],
            scale_mode="cuda_passthrough", simple_dynamic=True,
        )
        assert "concat=n=" not in graph
        assert "split=" not in graph
        assert "scale_cuda=1920:1080" in graph
        assert "tpad=stop_mode=clone:stop_duration=30.000" in graph

        short_static = [
            TimelineSegment(
                filepath="short.mp4", input_index=0,
                start_in_file=0.0, end_in_file=45.0,
                state="STATIC", duration=45.0,
            ),
        ]
        rows = [{"filepath": "short.mp4", "has_audio": 0}]
        filter_str = build_concat_filter(
            timeline=short_static, rows=rows, output_fps=20,
            output_width=1920, output_height=1080,
            scale_mode="cuda_passthrough",
            static_keyframe_interval=30.0,
        )
        # 45s < 2*30s 阈值，不启用快路径
        assert "select='isnan(prev_selected_t)" not in filter_str

    def test_partition_timeline_by_batches_preserves_count(self):
        segments = [
            TimelineSegment(
                filepath=f"file_{i // 3}.mp4",
                input_index=i // 3,
                start_in_file=0.0,
                end_in_file=5.0,
                state="DYNAMIC",
                duration=5.0,
            )
            for i in range(12)
        ]
        # 4 个不同文件，batch_max_files=2，预期分为 2 个 batch
        batches = partition_timeline_by_batches(timeline=segments, batch_max_files=2)
        assert len(batches) == 2
        total_segs = sum(len(b) for b in batches)
        assert total_segs == 12




class TestConcatOutputFiles:
    """测试切片拼接器的边界条件。"""

    @pytest.mark.parametrize("clip_count", [2, 6])
    def test_audio_padding_cannot_shift_video_seams(self, tmp_path, clip_count):
        import av
        import hashlib
        import numpy as np

        clips = []
        for i, color in enumerate(["red", "blue"] * (clip_count // 2)):
            path = tmp_path / f"clip{i}.mp4"
            subprocess.run([
                "ffmpeg", "-v", "error", "-f", "lavfi", "-i",
                f"color=c={color}:s=64x64:r=20:d=2", "-f", "lavfi", "-i",
                f"sine=frequency={440*(i+1)}:sample_rate=48000:duration=2.042",
                "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(path),
            ], capture_output=True, timeout=30, check=True)
            clips.append(path)

        def video(path):
            with av.open(str(path)) as container:
                return [(float(f.pts * f.time_base), hashlib.sha256(f.to_ndarray().tobytes()).hexdigest())
                        for f in container.decode(video=0)]

        expected_hashes = [h for clip in clips for _, h in video(clip)]
        output = tmp_path / "joined.mp4"
        timings = {"stale": 1}
        assert concat_output_files(clips, output, timings=timings)
        assert {"probe_s", "ffmpeg_s", "validation_s", "checkpoints", "promote_s"} <= set(timings)
        assert timings["status"] == "completed"
        assert timings["failure_stage"] is None
        assert timings["checkpoints"] == clip_count - 1
        assert all(timings[key] >= 0 for key in ("probe_s", "ffmpeg_s", "validation_s", "promote_s"))
        frames = video(output)
        assert [h for _, h in frames] == expected_hashes
        assert len(frames) == clip_count * 40
        assert all(b[0] - a[0] == pytest.approx(0.05) for a, b in zip(frames, frames[1:]))
        with av.open(str(output)) as container:
            samples = np.concatenate([f.to_ndarray().ravel() for f in container.decode(audio=0)])
        for start, expected_frequency in [(48000, 440), (120000, 880)]:
            window = samples[start:start+24000]
            assert np.sqrt(np.mean(window**2)) > 0.04
            frequency = np.argmax(np.abs(np.fft.rfft(window))) * 2
            assert abs(frequency - expected_frequency) <= 2

    def test_concat_output_files_empty_list(self, tmp_path):
        out = tmp_path / "final.mp4"
        assert not concat_output_files([], out)
        assert not out.exists()

    def test_invalid_join_preserves_previous_output_and_batches(self, tmp_path):
        from src.hardware.ffmpeg import FFmpegResult
        out, batch = tmp_path / "out.mp4", tmp_path / "batch.mp4"
        out.write_bytes(b"previous")
        batch.write_bytes(b"batch")
        def encode(args, **kwargs):
            Path(args[-1]).write_bytes(b"invalid join")
            return FFmpegResult(0, b"", b"encoder diagnostic")
        with patch("src.renderer.get_duration", return_value=2), patch("src.renderer.run_ffmpeg", side_effect=encode), patch("src.render_cache.valid_video", return_value=False):
            assert not concat_output_files([batch, batch], out)
        assert out.read_bytes() == b"previous"
        assert batch.read_bytes() == b"batch"
        assert not out.with_name("out.tmp.mp4").exists()
        diagnostic = tmp_path / ".concat_failures" / "out"
        assert (diagnostic / "failed.mp4").read_bytes() == b"invalid join"
        assert (diagnostic / "stderr.log").read_text() == "encoder diagnostic"
        import json
        assert json.loads((diagnostic / "failure.json").read_text())["failure_stage"] == "validation"

    @pytest.mark.parametrize("cancelled", [False, True])
    def test_concat_failure_records_phase_and_preserves_first_evidence(self, tmp_path, cancelled):
        import json
        from src.hardware.ffmpeg import FFmpegResult
        output = tmp_path / "out.mp4"
        batch = tmp_path / "batch.mp4"
        batch.write_bytes(b"original batch")
        timings = {}
        def encode(args, **kwargs):
            Path(args[-1]).write_bytes(b"partial output")
            if cancelled:
                raise KeyboardInterrupt
            return FFmpegResult(-9, b"", b"timeout evidence", timed_out=True)
        with patch("src.renderer.run_ffmpeg", side_effect=encode):
            if cancelled:
                with pytest.raises(KeyboardInterrupt):
                    concat_output_files([batch, batch], output, durations=[2, 2], timings=timings)
            else:
                assert not concat_output_files([batch, batch], output, durations=[2, 2], timings=timings)
        assert timings["status"] == ("cancelled" if cancelled else "failed")
        assert timings["failure_stage"] == "ffmpeg"
        assert timings["ffmpeg_s"] >= 0
        assert timings["total_s"] >= timings["ffmpeg_s"]
        diagnostic = output.parent / ".concat_failures" / output.stem
        first = (diagnostic / "failure.json").read_bytes()
        assert json.loads(first)["status"] == timings["status"]
        assert (diagnostic / "failed.mp4").read_bytes() == b"partial output"
        assert batch.read_bytes() == b"original batch"
        # Later failures cannot overwrite the first retained artifact.
        with patch("src.renderer.run_ffmpeg", return_value=FFmpegResult(1, b"", b"new error")):
            assert not concat_output_files([batch, batch], output, durations=[2, 2])
        assert (diagnostic / "failure.json").read_bytes() == first

    def test_concat_failure_media_budget_keeps_report(self, tmp_path):
        from src.stages.renderer import _preserve_concat_failure
        temporary = tmp_path / "out.tmp.mp4"
        temporary.write_bytes(b"partial")
        real_stat = Path.stat
        def stat(path, *args, **kwargs):
            if path == temporary:
                from types import SimpleNamespace
                return SimpleNamespace(st_size=17 * 1024**3, st_mode=real_stat(path).st_mode)
            return real_stat(path, *args, **kwargs)
        report = {"failure_stage": "ffmpeg"}
        with patch.object(Path, "stat", stat):
            _preserve_concat_failure(tmp_path / "out.mp4", temporary, tmp_path / "missing.txt", report, "故障\r\n" * 100000)
        diagnostic = tmp_path / ".concat_failures" / "out"
        assert not (diagnostic / "failed.mp4").exists()
        assert (diagnostic / "stderr.log").stat().st_size <= 65536
        assert (diagnostic / "stderr.log").read_text(encoding="utf-8").endswith("故障\n")
        assert "media_retention_skipped" in report

    def test_timeline_normalization_preserves_clipped_confidence_and_peak(self):
        segments = [
            TimelineSegment("clip.mp4", 0, 0, 4, "DYNAMIC", 4, 10, 0.2),
            TimelineSegment("clip.mp4", 0, 2, 10, "DYNAMIC", 8, 30, 0.8),
        ]
        actual = _normalize_file_timeline(segments)
        assert len(actual) == 1
        assert actual[0].duration == 10
        assert actual[0].max_energy == 30
        assert actual[0].avg_confidence == pytest.approx(0.56)

    def test_consumed_prefetch_does_not_recopy_source(self, tmp_path, monkeypatch):
        from src.renderer import _stage_source_for_render
        source = tmp_path / "source.mp4"
        source.write_bytes(b"source")
        monkeypatch.setattr("src.renderer.TEMP_DIR", tmp_path / "cache")
        first = _stage_source_for_render(source)
        assert first and first.read_bytes() == b"source"
        first.unlink()
        with patch("src.renderer.shutil.copyfile", side_effect=AssertionError("late prefetch copied")):
            assert _stage_source_for_render(source, needed=lambda: False) is None

    def test_qsv_rejects_silent_rate_control_fallback(self):
        from src.renderer import _build_enc_args
        with pytest.raises(ValueError, match="ICQ"):
            _build_enc_args("qsv", {"qsv": {"global_quality": 28, "maxrate": "4M"}})

    def test_batch_render_reuse_existing_file(self, tmp_path):
        from src.renderer import _run_batch_render
        fake_batch = tmp_path / "_batch0_20260901_cam0.mp4"
        # 写入 600KB 伪中间视频文件
        fake_batch.write_bytes(b"0" * (600 * 1024))
        res = _run_batch_render(
            input_files=[],
            filter_complex="",
            output_path=fake_batch,
            encoder="nv",
            fps=20,
            out_cfg={},
            audio_cfg={},
            date="20260901",
            cam_index=0,
            batch_idx=0,
        )
        assert res is None  # Size alone never authorizes reuse of a corrupt video.


class TestBuildEncArgs:
    """测试异构编码参数对齐与 dump_extra 带内参数集注入。"""

    def test_nvenc_enc_args_aligned_and_dump_extra(self):
        from src.renderer import _build_enc_args
        out_cfg = {
            "nv": {"preset": "p1", "cq": 28, "maxrate": "4M", "bufsize": "8M", "pix_fmt": "nv12"},
            "qsv": {"preset": "fast", "global_quality": 28, "pix_fmt": "nv12"},
        }
        nv_args = _build_enc_args("nv", out_cfg)
        assert "-c:v" in nv_args
        assert "hevc_nvenc" in nv_args
        assert "-pix_fmt" in nv_args
        idx_pix = nv_args.index("-pix_fmt")
        assert nv_args[idx_pix + 1] == "nv12"
        assert "-bsf:v" in nv_args
        idx_bsf = nv_args.index("-bsf:v")
        assert nv_args[idx_bsf + 1] == "dump_extra"
        assert "-forced-idr" in nv_args
        assert nv_args[nv_args.index("-forced-idr") + 1] == "1"
        assert "-g" in nv_args
        assert nv_args[nv_args.index("-g") + 1] == "60"

    def test_qsv_enc_args_aligned_and_dump_extra(self):
        from src.renderer import _build_enc_args
        out_cfg = {
            "nv": {"preset": "p1", "cq": 28, "maxrate": "4M", "bufsize": "8M", "pix_fmt": "nv12"},
            "qsv": {"preset": "fast", "global_quality": 28, "pix_fmt": "nv12"},
        }
        qsv_args = _build_enc_args("qsv", out_cfg)
        assert "-c:v" in qsv_args
        assert "hevc_qsv" in qsv_args
        assert "-pix_fmt" in qsv_args
        idx_pix = qsv_args.index("-pix_fmt")
        assert qsv_args[idx_pix + 1] == "nv12"
        assert "-bsf:v" in qsv_args
        idx_bsf = qsv_args.index("-bsf:v")
        assert qsv_args[idx_bsf + 1] == "dump_extra"
        assert "-forced_idr" in qsv_args
        assert qsv_args[qsv_args.index("-forced_idr") + 1] == "1"
        assert "-g" in qsv_args
        assert qsv_args[qsv_args.index("-g") + 1] == "60"

    def test_advanced_enc_args_support(self):
        from src.renderer import _build_enc_args
        out_cfg = {
            "gop": 120,
            "nv": {
                "preset": "p4",
                "cq": 28,
                "rc_lookahead": 32,
                "spatial_aq": 1,
                "temporal_aq": 1,
                "aq_strength": 8,
                "b_ref_mode": "middle",
            },
            "qsv": {
                "gop": 120,
                "look_ahead_depth": 32,
            },
        }
        nv_args = _build_enc_args("nv", out_cfg)
        assert nv_args[nv_args.index("-g") + 1] == "120"
        assert nv_args[nv_args.index("-preset") + 1] == "p4"
        assert nv_args[nv_args.index("-rc-lookahead") + 1] == "32"
        assert nv_args[nv_args.index("-spatial-aq") + 1] == "1"
        assert nv_args[nv_args.index("-temporal-aq") + 1] == "1"
        assert nv_args[nv_args.index("-aq-strength") + 1] == "8"
        assert nv_args[nv_args.index("-b_ref_mode") + 1] == "middle"

        qsv_args = _build_enc_args("qsv", out_cfg)
        assert qsv_args[qsv_args.index("-g") + 1] == "120"
        assert qsv_args[qsv_args.index("-look_ahead_depth") + 1] == "32"



def test_delivery_mapping_covers_all_states_on_integer_batch_clock():
    from src.stages.companion import build_delivery_mapping
    from src.stages.timeline import TimelineSegment, compute_display_plans, display_plan_options
    timeline = [TimelineSegment("same.mp4",0,start,start+duration,state,duration) for start,duration,state in
                [(0,3,"DYNAMIC"),(3,10,"PRESENCE"),(13,40,"STATIC"),(53,10,"MICRO_MOTION"),(63,10,"NIGHT_STATIONARY")]]
    config = {"output":{"fps":20},"render":{"batch_max_files":1}}
    mapping = build_delivery_mapping(timeline,config)
    expected = sum(round(duration*20) for duration,_ in compute_display_plans(timeline,**display_plan_options(config)))
    assert mapping["planned_frames"] == expected
    assert len(mapping["segments"]) == 5
    assert mapping["segments"][0]["output_start_frame"] == 0
    assert mapping["segments"][-1]["output_end_frame"] == expected
    assert all(a["output_end_frame"] == b["output_start_frame"] for a,b in zip(mapping["segments"],mapping["segments"][1:]))
    assert [item.state for item in timeline] == ["DYNAMIC","PRESENCE","STATIC","MICRO_MOTION","NIGHT_STATIONARY"]


@pytest.mark.parametrize("case,available", [
    ("nominal", True), ("average_rate_drift", True), ("b_frame_order", True),
    ("wrong_frames", False), ("missing_pts", False), ("interior_gap", False),
    ("duplicate_pts", False), ("missing_packet", False), ("extra_packet", False),
    ("late_origin", False), ("changed_media", False), ("io_denied", False),
])
def test_delivery_mapping_verifies_packet_clock(tmp_path, monkeypatch, case, available):
    from fractions import Fraction
    from types import SimpleNamespace
    import threading
    import av
    import src.hardware.scheduler as scheduler
    from src.stages.companion import bind_delivery_mapping
    from src.stages.timeline import TimelineSegment

    movie = tmp_path / "movie.mp4"
    movie.write_bytes(b"unchanged")
    timeline = [TimelineSegment("source.mp4", 0, 0, 3.05, "DYNAMIC", 3.05)]
    config = {"output": {"fps": 20}, "render": {"batch_max_files": 1},
              "segment": {"dynamic_speed": 1}}
    frames = 61
    stream = SimpleNamespace(frames=frames, average_rate=Fraction(20), time_base=Fraction(1, 10240))
    timestamps = [215 + index * 512 for index in range(frames)]
    if case == "average_rate_drift":
        stream.average_rate = Fraction(frames * 20, frames - 1)
    elif case == "b_frame_order":
        timestamps[1:4] = [timestamps[3], timestamps[1], timestamps[2]]
    elif case == "wrong_frames":
        stream.frames -= 1
    elif case == "missing_pts":
        timestamps[20] = None
    elif case == "interior_gap":
        timestamps[20] += 256  # Same average rate and endpoints, invalid interior clock.
    elif case == "duplicate_pts":
        timestamps[20] = timestamps[19]
    elif case == "missing_packet":
        timestamps.pop()
    elif case == "extra_packet":
        timestamps.append(timestamps[-1] + 512)
    elif case == "late_origin":
        timestamps = [pts + 1024 for pts in timestamps]

    class Container:
        streams = SimpleNamespace(video=[stream])
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def demux(self, selected):
            assert selected is stream
            for pts in timestamps:
                yield SimpleNamespace(size=10, pts=pts)
            yield SimpleNamespace(size=0, pts=None)
            if case == "changed_media":
                movie.write_bytes(b"changed size")

    budget = threading.BoundedSemaphore(1)
    if case == "io_denied":
        budget.acquire()
        monkeypatch.setattr(scheduler, "acquire_with_retry", lambda *args, **kwargs: False)
    monkeypatch.setattr(scheduler, "get_disk_semaphore", lambda: budget)
    monkeypatch.setattr(av, "open", lambda *args, **kwargs: Container())
    result = bind_delivery_mapping(movie, timeline, config)
    assert (result["status"] == "AVAILABLE") is available, result
    if available:
        assert result["planned_frames"] == frames
        assert result["media_identity"]["fps"] == result["fps"] == 20
        assert result["clock_validation"]["packets"] == frames
    else:
        assert result["reason"]
    if case == "io_denied":
        assert not budget.acquire(blocking=False)
        budget.release()
    assert budget.acquire(blocking=False), "I/O lease was not returned after verification"
    budget.release()
@pytest.mark.parametrize("io_timeout", [None, .05])
def test_managed_ffmpeg_admission_options_preserve_defaults(monkeypatch, io_timeout):
    from src.hardware import ffmpeg, scheduler
    calls = []
    monkeypatch.setattr(scheduler, "get_disk_semaphore", lambda: object())
    def denied(semaphore, **kwargs):
        calls.append(kwargs)
        return False
    monkeypatch.setattr(scheduler, "acquire_with_retry", denied)
    monkeypatch.setattr(ffmpeg.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Admission denial spawned FFmpeg"))
    result = ffmpeg.run_ffmpeg(["-version"], io_timeout=io_timeout)
    assert result.returncode != 0
    assert calls == ([{}] if io_timeout is None else [{"timeout": .05, "retries": 1}])


@pytest.mark.parametrize("io_timeout", [0, -1, float("nan"), float("inf")])
def test_managed_ffmpeg_rejects_invalid_admission_timeout(io_timeout):
    from src.hardware.ffmpeg import run_ffmpeg
    with pytest.raises(ValueError, match="finite and positive"):
        run_ffmpeg(["-version"], io_timeout=io_timeout)
@pytest.mark.skipif(__import__('os').name!='nt',reason='Windows owned process Job')
@pytest.mark.parametrize('failure',['create','assign'])
def test_registry_ownership_failure_stops_new_process(monkeypatch,failure):
    import sys
    from src.hardware import ffmpeg
    class FailedJob:
        def __init__(self):
            if failure=='create':raise OSError('Controlled Job creation failure')
        def assign(self,proc):raise OSError('Controlled Job assignment failure')
        def close(self):pass
    monkeypatch.setattr(ffmpeg,'OwnedProcessJob',FailedJob)
    proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        with pytest.raises(OSError,match='Controlled Job'):
            ffmpeg.FFmpegProcessRegistry.register('ownership_failure',proc)
        assert proc.poll() is not None
        assert 'ownership_failure' not in ffmpeg.FFmpegProcessRegistry._processes
        assert 'ownership_failure' not in ffmpeg.FFmpegProcessRegistry._jobs
    finally:
        if proc.poll() is None:proc.kill()
        proc.wait(timeout=5)


def test_registry_duplicate_does_not_replace_owned_process():
    import sys
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    first=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    second=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    FFmpegProcessRegistry.reset_interrupted()
    try:
        FFmpegProcessRegistry.register('same_key',first)
        FFmpegProcessRegistry.register('same_key',first)  # Idempotent same instance.
        with pytest.raises(RuntimeError,match='already owned'):
            FFmpegProcessRegistry.register('same_key',second)
        assert first.poll() is None and second.poll() is not None
        assert FFmpegProcessRegistry._processes['same_key'] is first
    finally:
        FFmpegProcessRegistry.deregister('same_key')
        for proc in (first,second):
            if proc.poll() is None:proc.kill()
            proc.wait(timeout=5)


def test_registry_deregister_relinquishes_live_process_without_killing():
    import sys
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    proc=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        FFmpegProcessRegistry.register('detach_live',proc)
        FFmpegProcessRegistry.deregister('detach_live')
        assert proc.poll() is None
        assert 'detach_live' not in FFmpegProcessRegistry._jobs
        assert 'detach_live' not in FFmpegProcessRegistry._processes
    finally:
        if proc.poll() is None:proc.kill()
        proc.wait(timeout=5)


def test_registry_failed_job_close_keeps_diagnosable_ownership(monkeypatch):
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    class Process:
        def kill(self):pass
    class Job:
        def close(self):raise OSError('controlled close failure')
    proc,job=Process(),Job()
    monkeypatch.setattr(FFmpegProcessRegistry,'_processes',{'failed_close':proc})
    monkeypatch.setattr(FFmpegProcessRegistry,'_jobs',{'failed_close':job})
    FFmpegProcessRegistry.kill_all()
    assert FFmpegProcessRegistry._processes['failed_close'] is proc
    assert FFmpegProcessRegistry._jobs['failed_close'] is job
    FFmpegProcessRegistry.reset_interrupted()
@pytest.mark.skipif(__import__('os').name!='nt',reason='Windows registered FFmpeg ownership')
@pytest.mark.parametrize('detach',[False,True,'before_registration'])
def test_registry_parent_death_terminates_only_registered_ffmpeg(tmp_path,detach):
    import _winapi
    import json
    import os
    import shutil
    import sys
    from src.core.visual_index import reader_process_state
    if not shutil.which('ffmpeg'):pytest.skip('FFmpeg required')
    record=tmp_path/'registered.json'
    program=r'''
import json,subprocess,sys,time
from pathlib import Path
from src.hardware.ffmpeg import FFmpegProcessRegistry
from src.core.visual_index import reader_process_state
if sys.argv[2]=='before_registration':
    def pause_before_registration(key,child):
        Path(sys.argv[1]).write_text(json.dumps(reader_process_state(child.pid,handle=child._handle)))
        time.sleep(30)
    FFmpegProcessRegistry.register=pause_before_registration
p=FFmpegProcessRegistry.start('parent_test',['ffmpeg','-nostdin','-loglevel','error','-re','-f','lavfi','-i','testsrc2=size=64x64:rate=2','-t','30','-f','null','-'],stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
if sys.argv[2]=='detach':FFmpegProcessRegistry.deregister('parent_test')
Path(sys.argv[1]).write_text(json.dumps(reader_process_state(p.pid,handle=p._handle)))
time.sleep(30)
'''
    env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    independent=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    mode='before_registration' if detach=='before_registration' else ('detach' if detach else 'registered')
    parent=subprocess.Popen([sys.executable,'-c',program,str(record),mode],env=env)
    handle=None
    try:
        deadline=time.monotonic()+5
        while not record.exists() and parent.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert record.exists()
        child=json.loads(record.read_text());handle=_winapi.OpenProcess(0x101001,False,child['pid'])
        assert reader_process_state(child['pid'],handle=handle)['created']==child['created']
        parent.kill();parent.wait(timeout=5)
        result=_winapi.WaitForSingleObject(handle,300 if detach is True else 5000)
        assert result==(_winapi.WAIT_TIMEOUT if detach is True else _winapi.WAIT_OBJECT_0)
        assert independent.poll() is None
    finally:
        if parent.poll() is None:parent.kill()
        parent.wait(timeout=5)
        if handle is not None:
            if _winapi.WaitForSingleObject(handle,0)!=_winapi.WAIT_OBJECT_0:_winapi.TerminateProcess(handle,1)
            assert _winapi.WaitForSingleObject(handle,5000)==_winapi.WAIT_OBJECT_0
            _winapi.CloseHandle(handle)
        independent.kill();independent.wait(timeout=5)


@pytest.mark.skipif(__import__('os').name!='nt',reason='Windows creation-time Job ownership')
@pytest.mark.parametrize('merged',[False,True])
def test_owned_start_preserves_popen_streams_and_single_job(tmp_path,merged):
    import os
    import sys
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    FFmpegProcessRegistry.reset_interrupted()
    env=os.environ.copy();env['HOMEVLOG_OWNED_TEST']='测试'
    proc=FFmpegProcessRegistry.start('owned_stdio',[sys.executable,'-c',
        "import sys,os,json;print(json.dumps([os.getcwd(),os.environ['HOMEVLOG_OWNED_TEST']]));print('stderr',file=sys.stderr)"],
        stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.STDOUT if merged else subprocess.PIPE,cwd=tmp_path,env=env)
    try:
        assert FFmpegProcessRegistry._jobs['owned_stdio'] is proc.owned_job
        out,err=proc.communicate(timeout=5)
        assert proc.returncode==0 and b'stderr' in (out if merged else err)
        assert str(tmp_path).replace('\\','\\\\').encode() in out and b'\\u6d4b\\u8bd5' in out
    finally:
        if proc.poll() is None:proc.kill();proc.wait(timeout=5)
        FFmpegProcessRegistry.deregister('owned_stdio')
    assert 'owned_stdio' not in FFmpegProcessRegistry._jobs


@pytest.mark.skipif(__import__('os').name!='nt',reason='Windows creation-time Job ownership')
def test_owned_start_job_creation_failure_preserves_registry(tmp_path,monkeypatch):
    from src.hardware import ffmpeg
    jobs=[]
    class FailedJob(ffmpeg.OwnedProcessJob):
        def __init__(self):super().__init__();jobs.append(self)
        def create_process(self,*args,**kwargs):raise OSError('Controlled native creation failure')
    monkeypatch.setattr(ffmpeg,'OwnedProcessJob',FailedJob)
    with pytest.raises(OSError,match='Controlled native creation'):
        ffmpeg.FFmpegProcessRegistry.start('native_failure',['missing'],stdout=subprocess.PIPE)
    assert jobs and jobs[0].handle is None
    assert 'native_failure' not in ffmpeg.FFmpegProcessRegistry._processes
    assert 'native_failure' not in ffmpeg.FFmpegProcessRegistry._jobs


@pytest.mark.skipif(__import__('os').name!='nt',reason='Windows creation-time Job ownership')
def test_owned_start_registration_failure_stops_created_child(monkeypatch):
    import sys
    from src.hardware import ffmpeg
    children=[]
    class RecordingPopen(ffmpeg._OwnedFFmpegPopen):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);children.append(self)
    monkeypatch.setattr(ffmpeg,'_OwnedFFmpegPopen',RecordingPopen)
    def refuse(*args):raise RuntimeError('Controlled registry refusal')
    monkeypatch.setattr(ffmpeg.FFmpegProcessRegistry,'register',refuse)
    with pytest.raises(RuntimeError,match='registry refusal'):
        ffmpeg.FFmpegProcessRegistry.start('refused',[sys.executable,'-c','import time;time.sleep(30)'])
    assert len(children)==1 and children[0].poll() is not None
    assert children[0].owned_job.handle is None


@pytest.mark.skipif(__import__('os').name!='nt',reason='Windows creation-time Job ownership')
def test_owned_start_cancel_and_detach_reregister_contract():
    import sys
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    FFmpegProcessRegistry.reset_interrupted()
    proc=FFmpegProcessRegistry.start('owned_detach',[sys.executable,'-c','import time;time.sleep(30)'])
    try:
        FFmpegProcessRegistry.deregister('owned_detach')
        assert proc.poll() is None and proc.owned_job.handle is None
        FFmpegProcessRegistry.register('owned_detach',proc)
        assert FFmpegProcessRegistry._jobs['owned_detach'].handle
        FFmpegProcessRegistry.kill_all()
        proc.wait(timeout=5)
        assert proc.poll() is not None
        interrupted=FFmpegProcessRegistry.start('already_cancelled',[sys.executable,'-c','import time;time.sleep(30)'])
        try:interrupted.wait(timeout=5);assert interrupted.poll() is not None
        finally:FFmpegProcessRegistry.deregister('already_cancelled')
    finally:
        if proc.poll() is None:proc.kill();proc.wait(timeout=5)
        FFmpegProcessRegistry.deregister('owned_detach')
        FFmpegProcessRegistry.reset_interrupted()


@pytest.mark.parametrize('outcome', ['success', 'nonzero', 'empty', 'invalid', 'timeout', 'cancel', 'spawn'])
def test_metadata_probe_lifecycle_and_existing_results(monkeypatch, outcome):
    import subprocess
    from unittest.mock import MagicMock
    from src.hardware import ffmpeg, scheduler
    registry = ffmpeg.FFmpegProcessRegistry
    monkeypatch.setattr(registry, '_processes', {})
    monkeypatch.setattr(registry, '_jobs', {})
    semaphore = MagicMock()
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: semaphore)
    monkeypatch.setattr(scheduler, 'acquire_with_retry', lambda sem: True)
    process = MagicMock()
    process.__enter__.return_value = process
    process.poll.return_value = None
    process.returncode = 1 if outcome == 'nonzero' else 0

    def communicate(*args, timeout=None):
        assert process in registry._processes.values(), 'metadata reader bypassed lifecycle registry'
        if outcome == 'timeout' and process.communicate.call_count == 1:
            raise subprocess.TimeoutExpired('ffprobe', timeout, output=b'partial')
        if outcome == 'cancel':
            registry.kill_all()
            process.returncode = -9
        process.poll.return_value = process.returncode
        output = {'empty': b'', 'invalid': b'{broken'}.get(outcome, b'{"format":{"duration":"2.0"}}')
        return output, b''

    process.communicate.side_effect = communicate
    if outcome == 'spawn':
        monkeypatch.setattr(ffmpeg.subprocess, 'Popen', MagicMock(side_effect=OSError('missing ffprobe')))
    else:
        monkeypatch.setattr(ffmpeg.subprocess, 'Popen', MagicMock(return_value=process))
    try:
        if outcome in ('timeout', 'spawn'):
            with pytest.raises(subprocess.TimeoutExpired if outcome == 'timeout' else OSError):
                ffmpeg.run_ffprobe('isolated.mp4', timeout=.2)
        else:
            result = ffmpeg.run_ffprobe('isolated.mp4', timeout=.2)
            assert result == ({'format': {'duration': '2.0'}} if outcome == 'success' else None)
        assert not registry._processes and not registry._jobs
        semaphore.release.assert_called_once_with()
        if outcome in ('timeout', 'cancel'):
            process.kill.assert_called()
    finally:
        registry.reset_interrupted()


def test_metadata_probe_admission_failure_does_not_spawn(monkeypatch):
    from unittest.mock import MagicMock
    from src.hardware import ffmpeg, scheduler
    semaphore = MagicMock()
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: semaphore)
    monkeypatch.setattr(scheduler, 'acquire_with_retry', lambda sem: False)
    spawn = MagicMock()
    monkeypatch.setattr(ffmpeg.subprocess, 'Popen', spawn)
    assert ffmpeg.run_ffprobe('unread.mp4') is None
    spawn.assert_not_called()
    semaphore.release.assert_not_called()


def test_metadata_probe_unconfirmed_stop_retains_reader_and_permit(monkeypatch):
    import subprocess
    from unittest.mock import MagicMock
    from src.hardware import ffmpeg, scheduler
    registry = ffmpeg.FFmpegProcessRegistry
    monkeypatch.setattr(registry, '_processes', {})
    monkeypatch.setattr(registry, '_jobs', {})
    monkeypatch.setattr(registry, '_deferred', {})
    semaphore, process = MagicMock(), MagicMock()
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: semaphore)
    monkeypatch.setattr(scheduler, 'acquire_with_retry', lambda sem: True)
    monkeypatch.setattr(ffmpeg.subprocess, 'Popen', MagicMock(return_value=process))
    process.poll.return_value = None
    process.communicate.side_effect = subprocess.TimeoutExpired('ffprobe', .1)
    process.wait.side_effect = subprocess.TimeoutExpired('ffprobe cleanup', 5)
    with pytest.raises(ffmpeg.MediaTerminationUnconfirmed, match='cleanup'):
        ffmpeg.run_ffprobe('controlled.mp4', .1)
    assert process in registry._processes.values()
    semaphore.release.assert_not_called()
    process.stdout.close.assert_not_called()


def test_metadata_probe_real_local_duration_and_invalid_file(tmp_path):
    import shutil
    import wave
    from src.hardware.ffmpeg import FFmpegProcessRegistry, get_duration, run_ffprobe
    if not shutil.which('ffprobe'):
        pytest.skip('ffprobe required')
    source = tmp_path / 'one_second.wav'
    with wave.open(str(source), 'wb') as output:
        output.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
        output.writeframes(b'\0\0' * 8000)
    assert run_ffprobe(str(source), 5)['streams'][0]['codec_name'] == 'pcm_s16le'
    assert get_duration(str(source)) == pytest.approx(1.0)
    assert run_ffprobe(str(tmp_path / 'missing.mp4'), 5) is None
    assert not FFmpegProcessRegistry._processes and not FFmpegProcessRegistry._jobs


@pytest.mark.parametrize('cancel', [False, True])
def test_metadata_probe_real_stalled_network_timeout_or_cancel(cancel):
    import shutil
    import socket
    import subprocess
    import threading
    from src.hardware.ffmpeg import FFmpegProcessRegistry, run_ffprobe
    from src.hardware.scheduler import get_disk_semaphore
    if not shutil.which('ffprobe'):
        pytest.skip('ffprobe required')
    results, errors = [], []
    semaphore = get_disk_semaphore()
    available = semaphore.available
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)  # HTTP peer accepts TCP but never sends a response.
        def probe():
            try:
                results.append(run_ffprobe(f'http://127.0.0.1:{listener.getsockname()[1]}/blocked', 5 if cancel else .3))
            except BaseException as error:
                errors.append(error)
        worker = threading.Thread(target=probe, daemon=True)
        worker.start()
        child = None
        try:
            deadline = time.monotonic() + 3
            while child is None and worker.is_alive() and time.monotonic() < deadline:
                with FFmpegProcessRegistry._lock:
                    child = next(iter(FFmpegProcessRegistry._processes.values()), None)
                time.sleep(.01)
            assert child is not None
            if cancel:
                FFmpegProcessRegistry.kill_all()
            worker.join(timeout=6)
            assert not worker.is_alive()
            assert child.poll() is not None
            assert (results == [None] and not errors) if cancel else (
                not results and len(errors) == 1 and isinstance(errors[0], subprocess.TimeoutExpired))
            assert not FFmpegProcessRegistry._processes and not FFmpegProcessRegistry._jobs
            assert semaphore.available == available
        finally:
            if worker.is_alive():
                FFmpegProcessRegistry.kill_all()
                worker.join(timeout=6)
            FFmpegProcessRegistry.reset_interrupted()


@pytest.mark.skipif(__import__('os').name != 'nt', reason='Windows ffprobe creation-time ownership')
@pytest.mark.parametrize('before_registration', [False, True])
def test_metadata_probe_parent_death_stops_reader(tmp_path, before_registration):
    import _winapi
    import json
    import os
    import shutil
    import socket
    import sys
    from src.core.visual_index import reader_process_state
    if not shutil.which('ffprobe'):
        pytest.skip('ffprobe required')
    record = tmp_path / 'probe_identity.json'
    program = r'''
import json,sys,time
from pathlib import Path
from src.hardware.ffmpeg import FFmpegProcessRegistry,run_ffprobe
from src.core.visual_index import reader_process_state
register=FFmpegProcessRegistry.register
def pause(key,child):
    if sys.argv[3]=='after':register(key,child)
    Path(sys.argv[1]).write_text(json.dumps(reader_process_state(child.pid,handle=child._handle)))
    time.sleep(30)
FFmpegProcessRegistry.register=pause
run_ffprobe(sys.argv[2],30)
'''
    env = os.environ.copy()
    env['PYTHONPATH'] = str(Path(__file__).resolve().parents[1])
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        independent = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
        parent = subprocess.Popen([sys.executable, '-c', program, str(record),
                                   f'http://127.0.0.1:{listener.getsockname()[1]}/blocked',
                                   'before' if before_registration else 'after'], env=env)
        handle = None
        try:
            deadline = time.monotonic() + 5
            while not record.exists() and parent.poll() is None and time.monotonic() < deadline:
                time.sleep(.02)
            assert record.exists()
            identity = json.loads(record.read_text())
            handle = _winapi.OpenProcess(0x101001, False, identity['pid'])
            state = reader_process_state(identity['pid'], handle=handle)
            assert state['created'] == identity['created'] and not state['kernel_ended']
            parent.kill()
            parent.wait(timeout=5)
            assert _winapi.WaitForSingleObject(handle, 5000) == _winapi.WAIT_OBJECT_0
            assert independent.poll() is None
        finally:
            if parent.poll() is None:
                parent.kill()
            parent.wait(timeout=5)
            if handle is not None:
                if _winapi.WaitForSingleObject(handle, 0) != _winapi.WAIT_OBJECT_0:
                    _winapi.TerminateProcess(handle, 1)
                assert _winapi.WaitForSingleObject(handle, 5000) == _winapi.WAIT_OBJECT_0
                _winapi.CloseHandle(handle)
            independent.kill()
            independent.wait(timeout=5)


@pytest.mark.skipif(__import__('os').name != 'nt', reason='Windows owned descendants')
@pytest.mark.parametrize('mode', ['capture', 'stream', 'probe', 'probe_dead_parent'])
def test_managed_timeout_stops_descendant_pipe_owner(tmp_path, monkeypatch, mode):
    import json
    import sys
    from src.hardware import ffmpeg
    from src.core.visual_index import reader_process_state
    registry = ffmpeg.FFmpegProcessRegistry
    original_start = registry.start
    record = tmp_path / 'descendant.json'
    program = r'''
import json,subprocess,sys,time
from pathlib import Path
from src.core.visual_index import reader_process_state
child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(4)'])
Path(sys.argv[1]).write_text(json.dumps(reader_process_state(child.pid,handle=child._handle)))
if sys.argv[2]!='dead':time.sleep(30)
'''
    def start(key, command, **kwargs):
        return original_start(key, [sys.executable, '-c', program, str(record),
                                   'dead' if mode == 'probe_dead_parent' else 'alive'], **kwargs)
    monkeypatch.setattr(registry, 'start', start)
    independent = subprocess.Popen([sys.executable, '-c', 'import time;time.sleep(30)'])
    try:
        began = time.monotonic()
        if mode.startswith('probe'):
            with pytest.raises(subprocess.TimeoutExpired):
                ffmpeg.run_ffprobe('controlled.mp4', timeout=.8)
        else:
            result = ffmpeg.run_ffmpeg([], timeout=.8, stdout_consumer=(lambda block: None) if mode == 'stream' else None)
        elapsed = time.monotonic() - began
        assert record.exists()
        identity = json.loads(record.read_text())
        state = reader_process_state(identity['pid'])
        assert state.get('missing') or state.get('kernel_ended') or state.get('created') != identity['created']
        if not mode.startswith('probe'):
            assert result.timed_out and result.returncode == -9
        assert elapsed < 2.5, f'Timeout waited for descendant natural exit: {elapsed:.3f}s'
        assert independent.poll() is None
        assert not registry._processes and not registry._jobs
    finally:
        registry.kill_all()
        registry.reset_interrupted()
        independent.kill()
        independent.wait(timeout=5)


def test_streaming_cleanup_error_still_checks_reader_before_release(monkeypatch):
    from unittest.mock import MagicMock
    from src.hardware import ffmpeg, scheduler
    registry = ffmpeg.FFmpegProcessRegistry
    monkeypatch.setattr(registry, '_processes', {})
    monkeypatch.setattr(registry, '_jobs', {})
    monkeypatch.setattr(registry, '_deferred', {})
    process, permit, reader, timer = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    process._handle = None
    process.stdout_thread = process.stderr_thread = None
    process.stdout.read.return_value = b''
    process.poll.return_value = None
    process.wait.side_effect = OSError('controlled execution wait error')
    reader.is_alive.return_value = True
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: permit)
    monkeypatch.setattr(scheduler, 'acquire_with_retry', lambda sem: True)
    monkeypatch.setattr(ffmpeg.subprocess, 'Popen', MagicMock(return_value=process))
    terminate = MagicMock(side_effect=[OSError('first cleanup failed'), None])
    monkeypatch.setattr(registry, 'terminate', terminate)
    monkeypatch.setattr(registry, 'wait_for_exit', MagicMock())
    class ControlledReader:
        def __new__(cls, *args, **kwargs):
            return reader
    with patch('threading.Thread', ControlledReader), patch('threading.Timer', return_value=timer):
        with pytest.raises(ffmpeg.MediaTerminationUnconfirmed):
            ffmpeg.run_ffmpeg([], timeout=1, stdout_consumer=lambda block: None)
    assert terminate.call_count == 2
    reader.join.assert_called_once_with(timeout=5)
    permit.release.assert_not_called()
    item = next(iter(registry._deferred.values()))
    assert item['proc'] is process and item['io_sem'] is permit and reader in item['readers']
    reader.is_alive.return_value = False
    process.poll.return_value = 0
    assert len(registry.reap_cleanups()['recovered']) == 1
    permit.release.assert_called_once()
    assert not registry.reap_cleanups()['recovered']


def test_managed_pipe_cleanup_failure_keeps_registry_and_permit(monkeypatch):
    from unittest.mock import MagicMock
    from src.hardware import ffmpeg, scheduler
    registry = ffmpeg.FFmpegProcessRegistry
    monkeypatch.setattr(registry, '_processes', {})
    monkeypatch.setattr(registry, '_jobs', {})
    monkeypatch.setattr(registry, '_deferred', {})
    semaphore, process = MagicMock(), MagicMock()
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: semaphore)
    monkeypatch.setattr(scheduler, 'acquire_with_retry', lambda sem: True)
    monkeypatch.setattr(ffmpeg.subprocess, 'Popen', MagicMock(return_value=process))
    process.poll.return_value = None
    process.communicate.side_effect = [subprocess.TimeoutExpired('ffmpeg', .1),
                                       subprocess.TimeoutExpired('pipe cleanup', 5)]
    with pytest.raises(ffmpeg.MediaTerminationUnconfirmed, match='pipe cleanup'):
        ffmpeg.run_ffmpeg([], .1)
    assert process in registry._processes.values()
    assert process.communicate.call_args.kwargs['timeout'] == 5
    semaphore.release.assert_not_called()
    assert next(iter(registry._deferred.values()))['io_sem'] is semaphore


@pytest.mark.skipif(__import__('os').name != 'nt', reason='Windows per-key Job termination')
def test_registry_terminate_preserves_unrelated_registered_task():
    import sys
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    first = registry.start('terminate_first', [sys.executable, '-c', 'import time;time.sleep(30)'])
    second = registry.start('keep_second', [sys.executable, '-c', 'import time;time.sleep(30)'])
    try:
        registry.terminate('terminate_first')
        first.wait(timeout=5)
        assert first.poll() is not None and second.poll() is None
        assert not registry.is_interrupted()
        assert registry._processes['terminate_first'] is first
        assert registry._processes['keep_second'] is second
        assert registry._jobs['keep_second'].handle
    finally:
        for key, child in [('terminate_first', first), ('keep_second', second)]:
            registry.terminate(key)
            child.wait(timeout=5)
            registry.deregister(key)


def test_registry_terminate_failed_job_close_retains_ownership(monkeypatch):
    from unittest.mock import MagicMock
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    process, job = MagicMock(), MagicMock()
    job.close.side_effect = OSError('controlled Job close failure')
    monkeypatch.setattr(registry, '_processes', {'failure': process})
    monkeypatch.setattr(registry, '_jobs', {'failure': job})
    with pytest.raises(OSError, match='Job close failure'):
        registry.terminate('failure')
    assert registry._processes['failure'] is process and registry._jobs['failure'] is job
    assert not registry.is_interrupted()


@pytest.mark.parametrize('pending_reason', ['process', 'reader', 'job', 'ready', 'close_failure', 'release_failure'])
def test_deferred_cleanup_requires_all_terminal_evidence(monkeypatch, pending_reason):
    from unittest.mock import MagicMock
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    process, permit, reader, job = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    process.poll.return_value = None if pending_reason == 'process' else 0
    reader.is_alive.return_value = pending_reason == 'reader'
    job.handle = 123 if pending_reason == 'job' else None
    if pending_reason == 'close_failure':
        process.stdout.close.side_effect = OSError('controlled pipe close failure')
    if pending_reason == 'release_failure':
        permit.release.side_effect = ValueError('controlled permit release failure')
    monkeypatch.setattr(registry, '_processes', {'deferred': process})
    monkeypatch.setattr(registry, '_jobs', {'deferred': job})
    monkeypatch.setattr(registry, '_deferred', {})
    registry.defer_cleanup('deferred', process, permit, [reader])
    registry.defer_cleanup('deferred', process, permit, [reader])
    if pending_reason == 'ready':
        assert registry.reap_cleanups() == {'recovered': ['deferred'], 'pending': []}
        assert registry.reap_cleanups() == {'recovered': [], 'pending': []}
        permit.release.assert_called_once_with()
        assert not registry._processes and not registry._jobs
    else:
        assert registry.reap_cleanups() == {'recovered': [], 'pending': ['deferred']}
        assert registry._processes['deferred'] is process
        if pending_reason != 'release_failure':
            permit.release.assert_not_called()
        with pytest.raises(RuntimeError, match='still deferred'):
            registry.deregister('deferred')
    assert not registry.is_interrupted()


def test_deferred_cleanup_identity_conflict_and_concurrent_recovery(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import MagicMock
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    process, permit = MagicMock(), MagicMock()
    process.poll.return_value = 0
    monkeypatch.setattr(registry, '_processes', {})
    monkeypatch.setattr(registry, '_jobs', {})
    monkeypatch.setattr(registry, '_deferred', {})
    registry.defer_cleanup('once', process, permit)
    with pytest.raises(RuntimeError, match='identity conflict'):
        registry.defer_cleanup('once', process, MagicMock())
    with ThreadPoolExecutor(max_workers=4) as pool:
        reports = list(pool.map(lambda _: registry.reap_cleanups(), range(4)))
    assert sum(len(x['recovered']) for x in reports) == 1
    permit.release.assert_called_once_with()
    assert not registry._deferred


def test_deferred_cleanup_survives_global_cancel_and_reset(monkeypatch):
    from unittest.mock import MagicMock
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    process, permit = MagicMock(), MagicMock()
    process.poll.return_value = None
    monkeypatch.setattr(registry, '_processes', {})
    monkeypatch.setattr(registry, '_jobs', {})
    monkeypatch.setattr(registry, '_deferred', {})
    registry.defer_cleanup('pending', process, permit)
    registry.kill_all()
    from src.hardware.ffmpeg import MediaTerminationUnconfirmed
    with pytest.raises(MediaTerminationUnconfirmed, match='still pending'):
        registry.reset_interrupted()
    assert registry.is_interrupted()
    assert registry._processes['pending'] is process
    assert registry.reap_cleanups()['pending'] == ['pending']
    permit.release.assert_not_called()
    process.poll.return_value = 0
    assert registry.reap_cleanups()['recovered'] == ['pending']
    permit.release.assert_called_once_with()
    registry.reset_interrupted()
    assert not registry.is_interrupted()
    permit.release.assert_called_once_with()


def test_cancellation_reset_refuses_an_unresolved_start_claim(monkeypatch):
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry, MediaTerminationUnconfirmed
    monkeypatch.setattr(registry, '_starting', {'old-start': {'proc': None}})
    with pytest.raises(MediaTerminationUnconfirmed, match='old-start'):
        registry.reset_interrupted()
    assert registry.is_interrupted() and 'old-start' in registry._starting
    registry._starting.clear()
    registry.require_cleanups()
    assert registry.is_interrupted()  # A read-only admission check never resumes cancellation.
    registry.reset_interrupted()
    assert not registry.is_interrupted()


@pytest.mark.skipif(__import__('os').name != 'nt', reason='Windows actual deferred readers')
def test_deferred_cleanup_real_reader_recovers_on_next_probe(tmp_path, monkeypatch):
    import _winapi
    import shutil
    import sys
    from src.hardware import ffmpeg, scheduler
    if not shutil.which('ffprobe'):
        pytest.skip('ffprobe required')
    registry = ffmpeg.FFmpegProcessRegistry
    monkeypatch.setattr(registry, '_processes', {})
    monkeypatch.setattr(registry, '_jobs', {})
    monkeypatch.setattr(registry, '_deferred', {})
    original_start, original_terminate = registry.start, registry.terminate
    disk = scheduler.get_disk_semaphore()
    available = disk.available
    children = []
    def start(key, command, **kwargs):
        child = original_start(key, [sys.executable, '-c', 'import time;time.sleep(30)'], **kwargs)
        children.append(child)
        return child
    monkeypatch.setattr(registry, 'start', start)
    monkeypatch.setattr(registry, 'terminate', lambda key: (_ for _ in ()).throw(OSError('controlled stop failure')))
    key = None
    try:
        with pytest.raises(ffmpeg.MediaTerminationUnconfirmed, match='cleanup') as raised:
            ffmpeg.run_ffprobe('controlled.mp4', .15)
        assert isinstance(raised.value.__cause__, OSError)
        key = next(iter(registry._deferred))
        child = children[0]
        pending = registry._deferred[key]
        assert child.poll() is None and pending['io_sem'] is disk
        assert len(pending['readers']) == 2
        assert disk.available == available - 1
        assert registry.reap_cleanups()['pending'] == [key]
        with pytest.raises(ffmpeg.MediaTerminationUnconfirmed, match='still pending'):
            registry.reset_interrupted()
        assert disk.available == available - 1
        original_terminate(key)
        child.wait(timeout=5)
        assert _winapi.WaitForSingleObject(child._handle, 5000) == _winapi.WAIT_OBJECT_0
        for reader in pending['readers']:
            reader.join(timeout=2)
            assert not reader.is_alive()
        monkeypatch.setattr(registry, 'start', original_start)
        monkeypatch.setattr(registry, 'terminate', original_terminate)
        registry.reset_interrupted()
        assert ffmpeg.run_ffprobe(str(tmp_path/'missing.mp4'), 5) is None
        assert not registry._deferred and not registry._processes and not registry._jobs
        assert disk.available == available
        assert registry.reap_cleanups() == {'recovered': [], 'pending': []}
        assert disk.available == available
    finally:
        if key is not None and key in registry._processes:
            original_terminate(key)
            children[0].wait(timeout=5)
            for reader in registry._deferred.get(key, {}).get('readers', []):
                reader.join(timeout=2)
            registry.reap_cleanups()


@pytest.fixture
def local_batch_lifecycle(tmp_path, monkeypatch, request):
    """Exercise actual Windows ownership with local Python, no media or GPU."""
    import json
    import os
    import sys
    import threading
    from types import SimpleNamespace
    from src.hardware import render_cache, scheduler
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    from src.stages import renderer

    if os.name != 'nt':
        pytest.skip('Windows actual batch ownership and kernel identity')
    import _winapi
    from src.core.visual_index import reader_process_state

    monkeypatch.setattr(registry, '_processes', {})
    monkeypatch.setattr(registry, '_jobs', {})
    monkeypatch.setattr(registry, '_deferred', {})
    original_start, original_terminate = registry.start, registry.terminate
    children, leases, disk_releases, hardware_releases, observations = [], [], [], [], []

    class Budget(scheduler.FileBudget):
        def release(self, weight=1):
            super().release(weight)
            disk_releases.append(weight)

    class Hardware(threading.BoundedSemaphore):
        def release(self, n=1):
            super().release(n)
            hardware_releases.append(n)

    class Clock:
        def __init__(self):
            self.value = 0.0
        def monotonic(self):
            self.value += .25
            return self.value

    disk, hardware = Budget(3), Hardware(1)
    state = SimpleNamespace(mode='timeout', fail_stop=False, output=tmp_path/'batch.mp4')
    temporary = tmp_path/'batch.tmp.mp4'
    script = tmp_path/'_fc_batch0_20261014_cam0.txt'
    monkeypatch.setattr(renderer, 'TEMP_DIR', tmp_path)
    monkeypatch.setattr(renderer, 'time', Clock())
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: disk)
    monkeypatch.setattr(scheduler, 'get_nvenc_semaphore', lambda: hardware)
    monkeypatch.setattr(render_cache, 'reusable', lambda *args, **kwargs: False)
    monkeypatch.setattr(render_cache, 'render_fingerprint', lambda *args, **kwargs: 'local-batch-identity')

    def lease_factory(resource, inputs):
        lease = scheduler.VideoLease(resource, inputs)
        leases.append(lease)
        return lease

    def progress(*args):
        if state.mode == 'exception':
            raise RuntimeError('controlled local render observation failure')
        return {'enc_frames': -1}

    def start(key, command, **kwargs):
        program = 'pass' if state.mode == 'success' else 'import time;time.sleep(30)'
        child = original_start(key, [sys.executable, '-c', program], **kwargs)
        original_wait = child.wait
        children.append((key, child, original_wait))
        temporary.write_bytes(b'controlled incomplete output')
        def wait(timeout=None):
            if timeout == 2.0 and state.mode != 'success':
                raise subprocess.TimeoutExpired('controlled render progress wait', timeout)
            if state.fail_stop and child.poll() is None:
                raise subprocess.TimeoutExpired('controlled unconfirmed stop wait', timeout)
            return original_wait(timeout=timeout)
        child.wait = wait
        return child

    def terminate(key):
        if state.fail_stop:
            raise OSError('controlled batch Job stop failure')
        return original_terminate(key)

    def observe(stage, child):
        identity = reader_process_state(child.pid, handle=child._handle)
        observations.append({'stage': stage, **identity,
                             'same_process_handle': int(child._handle),
                             'kernel_wait': _winapi.WaitForSingleObject(child._handle, 0),
                             'disk_available': disk.available,
                             'hardware_available': hardware._value,
                             'registry_processes': list(registry._processes),
                             'registry_jobs': list(registry._jobs),
                             'registry_deferred': list(registry._deferred),
                             'interrupted': registry.is_interrupted(),
                             'disk_releases': list(disk_releases),
                             'hardware_releases': list(hardware_releases)})
        return observations[-1]

    def run():
        return renderer._run_batch_render(
            ['local-a.mp4', 'local-b.mp4'], 'controlled graph', state.output, 'nv', 20,
            {}, {}, '20261014', 0,
            config={'render': {'inactivity_timeout_s': .1, 'startup_inactivity_timeout_s': .1}})

    monkeypatch.setattr(renderer, 'VideoLease', lease_factory)
    monkeypatch.setattr(renderer, '_parse_ffmpeg_progress', progress)
    monkeypatch.setattr(registry, 'start', start)
    monkeypatch.setattr(registry, 'terminate', terminate)
    workspace = SimpleNamespace(run=run, state=state, registry=registry, renderer=renderer,
                                children=children, leases=leases, disk=disk, hardware=hardware,
                                disk_releases=disk_releases, hardware_releases=hardware_releases,
                                temporary=temporary, script=script, observe=observe,
                                original_terminate=original_terminate, winapi=_winapi)
    try:
        yield workspace
    finally:
        state.fail_stop = False
        for key, child, original_wait in children:
            child.wait = original_wait
            if key in registry._processes:
                original_terminate(key)
            elif _winapi.WaitForSingleObject(child._handle, 0) != _winapi.WAIT_OBJECT_0:
                child.kill()
            original_wait(timeout=5)
            assert _winapi.WaitForSingleObject(child._handle, 5000) == _winapi.WAIT_OBJECT_0
            observe('finally_kernel_ended', child)
        registry.reap_cleanups()
        for key in list(registry._processes):
            registry.deregister(key)
        request.node.user_properties.append(('local_batch_kernel_evidence',
                                             json.dumps(observations, sort_keys=True)))


@pytest.mark.parametrize('failure', ['timeout', 'exception'])
def test_batch_render_unconfirmed_stop_retains_original_weighted_lease_and_recovers(
        local_batch_lifecycle, failure):
    w = local_batch_lifecycle
    w.state.mode, w.state.fail_stop = failure, True
    with pytest.raises(w.renderer.RenderTerminationUnconfirmed):
        w.run()
    key, child, original_wait = w.children[0]
    pending = w.registry._deferred[key]
    assert pending['proc'] is child and pending['io_sem'] is w.leases[0]
    assert pending['io_sem'].disk is w.disk and pending['io_sem'].hardware is w.hardware
    assert pending['io_sem'].inputs == 2
    assert w.registry._processes[key] is child and w.registry._jobs[key].handle
    assert w.observe('unconfirmed_still_live', child)['kernel_wait'] == w.winapi.WAIT_TIMEOUT
    assert w.disk.available == 1 and not w.hardware.acquire(blocking=False)
    assert not w.disk_releases and not w.hardware_releases
    assert w.temporary.read_bytes() == b'controlled incomplete output'
    previous_script = w.script.read_bytes()
    with pytest.raises(w.renderer.RenderTerminationUnconfirmed):
        w.run()
    assert len(w.children) == 1 and len(w.leases) == 1
    assert w.temporary.read_bytes() == b'controlled incomplete output'
    assert w.script.read_bytes() == previous_script
    assert w.disk.available == 1 and not w.disk_releases and not w.hardware_releases

    w.state.fail_stop = False
    w.original_terminate(key)
    original_wait(timeout=5)
    assert w.winapi.WaitForSingleObject(child._handle, 5000) == w.winapi.WAIT_OBJECT_0
    assert w.observe('explicitly_stopped_permit_still_held', child)['kernel_wait'] == w.winapi.WAIT_OBJECT_0
    assert w.disk.available == 1 and not w.disk_releases and not w.hardware_releases
    w.state.mode = 'timeout'
    assert w.run() is None  # The next renderer entry reaps the old exact lease.
    assert len(w.children) == 2
    assert w.disk.available == 3 and w.disk_releases == [2, 2]
    assert w.hardware_releases == [1, 1]
    assert w.hardware._value == 1
    assert not w.registry._processes and not w.registry._jobs and not w.registry._deferred
    assert not w.registry.is_interrupted()
    before = list(w.disk_releases), list(w.hardware_releases)
    assert w.registry.reap_cleanups() == {'recovered': [], 'pending': []}
    assert before == (w.disk_releases, w.hardware_releases)
    assert not w.temporary.exists() and not w.script.exists()


@pytest.mark.parametrize('mode', ['timeout', 'exception', 'success'])
def test_batch_render_confirmed_cleanup_releases_weighted_lease_once(local_batch_lifecycle, monkeypatch, mode):
    from src.hardware import render_cache
    w = local_batch_lifecycle
    w.state.mode = mode
    if mode == 'success':
        monkeypatch.setattr(render_cache, 'valid_video', lambda *args, **kwargs: True)
        monkeypatch.setattr(render_cache, 'save_manifest', lambda *args, **kwargs: None)
    result = w.run()
    assert result == (str(w.state.output) if mode == 'success' else None)
    _, child, _ = w.children[0]
    assert w.observe('confirmed_cleanup', child)['kernel_wait'] == w.winapi.WAIT_OBJECT_0
    assert w.disk.available == 3 and w.disk_releases == [2]
    assert w.hardware_releases == [1]
    assert w.hardware._value == 1
    assert not w.registry._processes and not w.registry._jobs and not w.registry._deferred
    assert not w.registry.is_interrupted()
    assert not w.script.exists()


def test_batch_render_cached_returncode_does_not_release_live_kernel_process(local_batch_lifecycle):
    w = local_batch_lifecycle
    w.state.fail_stop = True
    with pytest.raises(w.renderer.RenderTerminationUnconfirmed):
        w.run()
    key, child, _ = w.children[0]
    child.returncode = 0  # Controlled cached-value injection, not a real SMB failure.
    assert w.observe('cached_zero_kernel_still_live', child)['kernel_wait'] == w.winapi.WAIT_TIMEOUT
    assert not w.registry.has_exited(child)
    assert w.registry.reap_cleanups() == {'recovered': [], 'pending': [key]}
    assert w.disk.available == 1 and not w.disk_releases and not w.hardware_releases
    assert w.registry._deferred[key]['io_sem'] is w.leases[0]
    w.state.fail_stop = False
    w.original_terminate(key)
    assert w.winapi.WaitForSingleObject(child._handle, 5000) == w.winapi.WAIT_OBJECT_0
    assert w.registry.has_exited(child)
    assert w.disk.available == 1  # Exit confirmation alone does not replace lease recovery.
    assert w.run() is None
    assert w.disk.available == 3 and w.disk_releases == [2, 2] and w.hardware_releases == [1, 1]
    assert not w.registry._processes and not w.registry._jobs and not w.registry._deferred


@pytest.fixture
def native_startup_failure(tmp_path, monkeypatch, request):
    """Real owned children; stop failures and short native-wait injection are controlled."""
    import json
    import os
    import sys
    import threading
    from types import SimpleNamespace
    from src.core.visual_index import reader_process_state
    from src.hardware import ffmpeg, scheduler, render_cache
    from src.stages import detector, prescreen, renderer
    if os.name != 'nt':
        pytest.skip('Windows actual startup ownership')
    import _winapi
    registry = ffmpeg.FFmpegProcessRegistry
    for field in ('_processes', '_jobs', '_deferred', '_starting'):
        monkeypatch.setattr(registry, field, {})
    monkeypatch.setattr(registry, '_interrupted', False)
    original_popen = ffmpeg._OwnedFFmpegPopen
    original_register, original_wait = registry.register, registry.wait_for_exit
    original_stop = registry.terminate
    children, leases, releases, observations = [], [], [], []
    state = SimpleNamespace(cached=False, fail_cleanup=True, route=None, block_creation=False,
                            reached=threading.Event(), allow=threading.Event())

    class Budget(scheduler.FileBudget):
        def release(self, weight=1):
            super().release(weight)
            releases.append(weight)

    disk, hardware = Budget(3), threading.BoundedSemaphore(1)
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: disk)
    for name in ('get_qsv_semaphore', 'get_nv_semaphore', 'get_nvdec_semaphore', 'get_nvenc_semaphore'):
        monkeypatch.setattr(scheduler, name, lambda: hardware)
    monkeypatch.setattr('src.core.utils.TEMP_DIR', tmp_path)
    monkeypatch.setattr(renderer, 'TEMP_DIR', tmp_path)
    monkeypatch.setattr(render_cache, 'reusable', lambda *args, **kwargs: False)
    monkeypatch.setattr(render_cache, 'render_fingerprint', lambda *args, **kwargs: 'startup-controlled')
    output = tmp_path/'batch.mp4'
    temporary = tmp_path/'batch.tmp.mp4'
    script = tmp_path/'_fc_batch0_20261015_cam0.txt'

    def make_lease(resource, inputs=1):
        lease = scheduler.VideoLease(resource, inputs)
        leases.append(lease)
        return lease
    for module in (detector, prescreen, renderer):
        monkeypatch.setattr(module, 'VideoLease', make_lease)

    class LocalPopen(original_popen):
        def __init__(self, command, **kwargs):
            super().__init__([sys.executable, '-c', 'import time;time.sleep(30)'], **kwargs)
            job = self.owned_job
            children.append(dict(proc=self, job=job, kill=self.kill, wait=self.wait,
                                 close=job.close, key=None))
            if state.cached:
                self.returncode = 0
            if state.fail_cleanup:
                def failed_kill():
                    raise OSError('controlled startup child kill failure')
                def failed_close():
                    raise OSError('controlled startup Job close failure')
                self.kill, job.close = failed_kill, failed_close
            if state.route == 'renderer':
                temporary.write_bytes(b'preserve unconfirmed startup output')
            if state.block_creation:
                state.reached.set()
                if not state.allow.wait(timeout=3):
                    raise RuntimeError('controlled startup claim barrier expired')

    def reject(key, proc):
        children[-1]['key'] = key
        raise RuntimeError('controlled registration failure')

    def bounded_wait(proc, timeout=5):
        if state.fail_cleanup and _winapi.WaitForSingleObject(proc._handle, 0) == _winapi.WAIT_TIMEOUT:
            # Avoid 5s per case; this tests real liveness, not an actual deadline.
            raise subprocess.TimeoutExpired(proc.args, timeout)
        return original_wait(proc, timeout)

    def observe(stage, proc):
        observations.append(dict(stage=stage, **reader_process_state(proc.pid, handle=proc._handle),
                                 handle=int(proc._handle),
                                 kernel_wait=_winapi.WaitForSingleObject(proc._handle, 0),
                                 disk_available=disk.available, hardware_available=hardware._value,
                                 releases=list(releases), owned=list(registry._processes),
                                 deferred=list(registry._deferred), starting=list(registry._starting)))
        return observations[-1]

    monkeypatch.setattr(ffmpeg, '_OwnedFFmpegPopen', LocalPopen)
    monkeypatch.setattr(registry, 'register', reject)
    monkeypatch.setattr(registry, 'wait_for_exit', bounded_wait)

    def run(route):
        state.route = route
        if route == 'ffmpeg':
            return ffmpeg.run_ffmpeg(['controlled-no-media'], timeout=.1)
        if route == 'ffprobe':
            return ffmpeg.run_ffprobe('controlled-no-media', .1)
        if route == 'extract':
            return prescreen._extract_frame('controlled-no-media', 0, 64, 48, timeout=.1)
        if route == 'prescreen':
            return prescreen._prescreen_stream_fps('controlled-no-media', 2, 1, 64, 48, 8, 'qsv', .1)
        if route == 'detector':
            instance = detector.MotionDetector(
                {'detection': {'analysis_resolution': '64x48', 'analysis_fps_adaptive': False},
                 'yolo': {'enabled': False}, 'audio_vad': {'enabled': False}}, decode_gpu='qsv')
            monkeypatch.setattr(instance, '_decode_file_pyav',
                                lambda *a: pytest.fail('Unconfirmed startup reached a second decoder'))
            return instance._decode_file('controlled-no-media', 2)
        return renderer._run_batch_render(
            ['local-a', 'local-b'], 'controlled startup graph', output, 'nv', 20, {}, {}, '20261015', 0,
            config={'render': {}})

    workspace = SimpleNamespace(run=run, state=state, registry=registry, ffmpeg=ffmpeg,
                                renderer=renderer, disk=disk, hardware=hardware, releases=releases,
                                leases=leases, children=children, observe=observe, winapi=_winapi,
                                original_register=original_register, original_wait=original_wait,
                                original_stop=original_stop, output=output, temporary=temporary, script=script)
    try:
        yield workspace
    finally:
        state.allow.set()
        for entry in children:
            proc, job = entry['proc'], entry['job']
            proc.kill, proc.wait, job.close = entry['kill'], entry['wait'], entry['close']
            job.close()
            if _winapi.WaitForSingleObject(proc._handle, 5000) != _winapi.WAIT_OBJECT_0:
                _winapi.TerminateProcess(proc._handle, 1)
            assert _winapi.WaitForSingleObject(proc._handle, 5000) == _winapi.WAIT_OBJECT_0
            original_wait(proc, timeout=5)
            observe('finally_kernel_ended', proc)
        registry.reap_cleanups()
        for key, proc in list(registry._processes.items()):
            registry.deregister(key, proc)
        for entry in children:
            for stream in (entry['proc'].stdout, entry['proc'].stderr):
                if stream is not None:
                    stream.close()
        assert not registry._starting
        request.node.user_properties.append(('native_startup_evidence', json.dumps(observations, sort_keys=True)))


@pytest.mark.parametrize('route', ['ffmpeg', 'ffprobe', 'detector', 'prescreen', 'renderer', 'extract'])
@pytest.mark.parametrize('cached', [False, True], ids=['kill_close_failure', 'cached_zero_close_failure'])
def test_startup_unconfirmed_child_returns_to_exact_original_lease(native_startup_failure, route, cached):
    w = native_startup_failure
    w.state.cached = cached
    error_type = w.renderer.RenderTerminationUnconfirmed if route == 'renderer' else w.ffmpeg.MediaTerminationUnconfirmed
    with pytest.raises(error_type) as caught:
        w.run(route)
    assert len(w.children) == 1
    entry = w.children[0]
    proc, job, key = entry['proc'], entry['job'], entry['key']
    contexts, pending_errors = [], [caught.value]
    while pending_errors:
        error = pending_errors.pop()
        if any(error is old for old in contexts):
            continue
        contexts.append(error)
        pending_errors.extend(x for x in (error.__cause__, error.__context__) if x is not None)
    start_error = next(error for error in contexts if isinstance(error, w.ffmpeg.MediaStartTerminationUnconfirmed))
    assert start_error.process is proc and start_error.process_key == key
    assert w.registry._processes[key] is proc and w.registry._jobs[key] is job and job.handle
    held = w.registry._deferred[key]['io_sem']
    weight = 2 if route == 'renderer' else 1
    assert held is (w.disk if route in ('ffmpeg', 'ffprobe') else w.leases[0])
    if route not in ('ffmpeg', 'ffprobe'):
        assert held.disk is w.disk and held.hardware is w.hardware and held.inputs == weight
    assert w.observe('unreturned_start_still_live', proc)['kernel_wait'] == w.winapi.WAIT_TIMEOUT
    assert w.disk.available == 3-weight and not w.releases and not w.registry._starting
    if route == 'renderer':
        previous = w.temporary.read_bytes(), w.script.read_bytes()
        with pytest.raises(w.renderer.RenderTerminationUnconfirmed):
            w.run(route)
        assert len(w.children) == 1 and len(w.leases) == 1
        assert previous == (w.temporary.read_bytes(), w.script.read_bytes())
    proc.kill, proc.wait, job.close = entry['kill'], entry['wait'], entry['close']
    w.original_stop(key)
    w.original_wait(proc, timeout=5)
    assert w.observe('stopped_before_original_lease_reap', proc)['kernel_wait'] == w.winapi.WAIT_OBJECT_0
    assert w.disk.available == 3-weight and not w.releases
    assert w.registry.reap_cleanups() == {'recovered': [key], 'pending': []}
    assert w.disk.available == 3 and w.hardware._value == 1 and w.releases == [weight]
    assert w.registry.reap_cleanups() == {'recovered': [], 'pending': []}
    assert w.releases == [weight]


def test_startup_confirmed_registration_failure_preserves_ordinary_contract(native_startup_failure):
    w = native_startup_failure
    w.state.fail_cleanup = False
    with pytest.raises(RuntimeError, match='controlled registration failure'):
        w.run('ffmpeg')
    proc = w.children[0]['proc']
    assert w.observe('confirmed_start_failure', proc)['kernel_wait'] == w.winapi.WAIT_OBJECT_0
    assert w.disk.available == 3 and w.releases == [1]
    assert not w.registry._processes and not w.registry._jobs and not w.registry._deferred and not w.registry._starting


def test_startup_creation_failure_returns_spawn_error_without_leaking_claim(native_startup_failure, monkeypatch):
    w = native_startup_failure
    def missing(*args, **kwargs):
        raise FileNotFoundError('controlled missing executable')
    monkeypatch.setattr(w.ffmpeg, '_OwnedFFmpegPopen', missing)
    result = w.run('ffmpeg')
    assert result.returncode == -1 and 'spawn failed' in result.stderr_text
    assert not w.children and w.disk.available == 3 and w.releases == [1]
    assert not w.registry._starting and not w.registry._processes


def test_startup_existing_owner_and_wrong_identity_cannot_be_deregistered(native_startup_failure, monkeypatch):
    from types import SimpleNamespace
    w = native_startup_failure
    w.state.fail_cleanup = False
    monkeypatch.setattr(w.registry, 'register', w.original_register)
    args = ['controlled-existing-key']
    key = f'ffmpeg:{id(args)}:1.0'
    owner = w.registry.start(key, ['local-python'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    with pytest.raises(RuntimeError, match='identity mismatch'):
        w.registry.deregister(key, object())
    assert w.registry._processes[key] is owner
    # Exercise caller proc=None finally against an already-owned generated key.
    monkeypatch.setattr(w.ffmpeg, '_time', SimpleNamespace(monotonic=lambda: 1.0))
    with pytest.raises(RuntimeError, match='already owned'):
        w.ffmpeg.run_ffmpeg(args)
    assert len(w.children) == 1 and w.registry._processes[key] is owner
    assert w.observe('existing_owner_preserved', owner)['kernel_wait'] == w.winapi.WAIT_TIMEOUT
    assert w.disk.available == 3 and w.releases == [1]


def test_startup_concurrent_same_key_has_one_child_and_one_claim(native_startup_failure, monkeypatch):
    import threading
    w = native_startup_failure
    w.state.fail_cleanup, w.state.block_creation = False, True
    monkeypatch.setattr(w.registry, 'register', w.original_register)
    outcomes, errors = [], []
    def start_first():
        try:
            outcomes.append(w.registry.start('same-start', ['local-python'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=start_first)
    thread.start()
    try:
        assert w.state.reached.wait(timeout=2)
        assert w.registry.is_owned('same-start')
        with pytest.raises(RuntimeError, match='already owned'):
            w.registry.start('same-start', ['must-not-create'])
        with pytest.raises(RuntimeError, match='being started'):
            w.registry.register('same-start', object())
        with pytest.raises(RuntimeError, match='startup still pending'):
            w.registry.deregister('same-start', object())
        assert len(w.children) == 1
    finally:
        w.state.allow.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
    assert not errors and len(outcomes) == 1
    assert w.registry._processes['same-start'] is outcomes[0] and not w.registry._starting
    assert w.observe('concurrent_start_single_owner', outcomes[0])['kernel_wait'] == w.winapi.WAIT_TIMEOUT


@pytest.mark.parametrize('cleanup_would_fail', [False, True])
def test_concat_unconfirmed_reader_preserves_original_failure_and_live_inputs(tmp_path, monkeypatch, cleanup_would_fail):
    from src.stages import renderer
    from src.hardware.ffmpeg import MediaTerminationUnconfirmed
    output = tmp_path/'delivery.pending.mp4'
    output.write_bytes(b'retained old candidate')
    batch = tmp_path/'batch.mp4'
    batch.write_bytes(b'retained complete batch')
    temporary = output.with_name(output.stem+'.tmp.mp4')
    listing = output.with_name('.concat_'+output.stem+'.txt')
    fatal = MediaTerminationUnconfirmed('concat reader still owns files')
    def encode(*args, **kwargs):
        temporary.write_bytes(b'live partial output')
        raise fatal
    monkeypatch.setattr(renderer, 'run_ffmpeg', encode)
    original_unlink = Path.unlink
    def unlink(path, *args, **kwargs):
        if cleanup_would_fail and path in (temporary, listing):
            raise PermissionError('still in use')
        return original_unlink(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    timings = {}
    with pytest.raises(MediaTerminationUnconfirmed) as caught:
        renderer.concat_output_files([batch, batch], output, durations=[2, 2], timings=timings)
    assert caught.value is fatal
    assert temporary.read_bytes() == b'live partial output' and listing.is_file()
    assert output.read_bytes() == b'retained old candidate' and batch.read_bytes() == b'retained complete batch'
    assert not (tmp_path/'.concat_failures').exists()
    assert timings['status'] == 'termination_unconfirmed' and timings['retained_temporary'] == str(temporary)


def test_concat_pending_reader_refuses_before_rewriting_retained_inputs(tmp_path, monkeypatch):
    from unittest.mock import MagicMock
    from src.stages import renderer
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry, MediaTerminationUnconfirmed
    for field in ('_processes', '_jobs', '_deferred', '_starting'):
        monkeypatch.setattr(registry, field, {})
    process = MagicMock()
    process.poll.return_value = None
    permit = MagicMock()
    registry.defer_cleanup('old-concat-reader', process, permit)
    output = tmp_path/'pending.mp4'
    listing = output.with_name('.concat_'+output.stem+'.txt')
    temporary = output.with_name(output.stem+'.tmp.mp4')
    for path in (output, listing, temporary):
        path.write_bytes(b'owned retained asset')
    encode = MagicMock(side_effect=AssertionError('Pending owner admitted another concat'))
    monkeypatch.setattr(renderer, 'run_ffmpeg', encode)
    with pytest.raises(MediaTerminationUnconfirmed, match='still pending'):
        renderer.concat_output_files([output], output, durations=[2])
    assert all(path.read_bytes() == b'owned retained asset' for path in (output, listing, temporary))
    permit.release.assert_not_called()
    encode.assert_not_called()


def test_diagnostic_extract_frame_uses_managed_real_media(tmp_path):
    import shutil
    from scripts.inspect_detection import extract_frame
    from src.hardware.ffmpeg import run_ffmpeg, FFmpegProcessRegistry
    if not shutil.which("ffmpeg"):
        pytest.skip("FFmpeg required")
    source = tmp_path / "source.mp4"
    result = run_ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=1",
                         "-c:v", "libx264", str(source)], timeout=10)
    assert result.returncode == 0, result.stderr_text
    output = tmp_path / "frame.png"
    assert extract_frame(str(source), .2, output)
    import cv2
    frame = cv2.imread(str(output))
    assert frame is not None and frame.shape[:2] == (64, 64)
    assert not FFmpegProcessRegistry._processes
    assert not FFmpegProcessRegistry._deferred


def test_diagnostic_extract_frame_propagates_unknown_stop_and_reports_failure(tmp_path, monkeypatch):
    from scripts import inspect_detection as tool
    from src.hardware.ffmpeg import FFmpegResult, MediaTerminationUnconfirmed
    def failed(args, **kwargs):
        assert kwargs == {"timeout": 15, "io_timeout": 3}
        return FFmpegResult(-1, b"", b"timeout", timed_out=True)
    monkeypatch.setattr(tool, "run_ffmpeg", failed)
    assert not tool.extract_frame("source.mp4", 0, tmp_path / "frame.png")
    def unknown(*args, **kwargs):
        raise MediaTerminationUnconfirmed("reader still pending")
    monkeypatch.setattr(tool, "run_ffmpeg", unknown)
    with pytest.raises(MediaTerminationUnconfirmed, match="pending"):
        tool.extract_frame("source.mp4", 0, tmp_path / "frame.png")


@pytest.mark.parametrize("count,rate,fps,has_audio,eligible", [
    (179, 48000, 20, True, False), (180, 48000, 20, True, True),
    (183, 48000, 20, True, True), (215, 48000, 20, True, True),
    (283, 48000, 20, True, True), (397, 48000, 20, True, True),
    (180, 44100, 20, True, False), (180, 48000, 29.97, True, False),
    (180, 48000, 20, False, False),
])
def test_grouped_audio_eligibility_preserves_sample_closure(count, rate, fps, has_audio, eligible):
    from src.stages import timeline
    segments = [timeline.TimelineSegment("source", 0, i * 2, (i + 1) * 2,
        "DYNAMIC" if i % 2 else "NIGHT_STATIONARY", 2) for i in range(count)]
    plans = []
    timeline.build_concat_filter(segments, [{"filepath": "source", "has_audio": has_audio}],
        scale_mode="cpu", output_fps=fps, audio_sample_rate=rate,
        speed_ramping=False, audio_groups_out=plans)
    assert bool(plans) is eligible
    if eligible:
        assert len(plans) == 1
        assert sum(samples for _, samples in plans[0].groups) == plans[0].samples
        assert len(plans[0].groups) == (count + 47) // 48
        assert plans[0].samples * 4 <= timeline.GROUPED_AUDIO_MAX_BYTES


@pytest.mark.parametrize("failure", ["short", "overflow", "nonzero", "timeout", "cancel", "unknown"])
def test_grouped_audio_preparation_reports_incomplete_or_unknown_reader(tmp_path, monkeypatch, failure):
    from types import SimpleNamespace
    from src.stages import renderer, timeline
    monkeypatch.setattr(renderer.FFmpegProcessRegistry, "is_interrupted", lambda: failure == "cancel")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        if failure == "unknown":
            raise renderer.MediaTerminationUnconfirmed("reader alive")
        kwargs["stdout_consumer"](b"x" * (8 if failure == "overflow" else 2 if failure == "short" else 4))
        return SimpleNamespace(returncode=1 if failure == "nonzero" else 0,
            timed_out=failure == "timeout", stderr_text="diagnostic")

    monkeypatch.setattr(renderer, "run_ffmpeg", run)
    plan = timeline.GroupedAudioPlan("", (("graph", 1),), "", 48000, 1)
    error = renderer.RenderInterrupted if failure == "cancel" else (
        renderer.MediaTerminationUnconfirmed if failure == "unknown" else RuntimeError)
    with pytest.raises(error):
        renderer._prepare_grouped_audio("source", plan, tmp_path / "audio.f32", tmp_path / "graph.txt")
    assert bool(calls) is (failure != "cancel")


@pytest.mark.parametrize("reason", ["disk", "cap"])
def test_grouped_audio_resource_limit_does_not_open_source(tmp_path, monkeypatch, reason):
    from types import SimpleNamespace
    from src.stages import renderer, timeline
    monkeypatch.setattr(renderer.shutil, "disk_usage",
        lambda _path: SimpleNamespace(free=0 if reason == "disk" else 2**40))
    monkeypatch.setattr(renderer, "run_ffmpeg", lambda *a, **k: pytest.fail("unsupported audio launched"))
    plan = timeline.GroupedAudioPlan("", (("graph", 1),), "", 48000,
        timeline.GROUPED_AUDIO_MAX_BYTES if reason == "cap" else 1)
    pcm = tmp_path / "audio.f32"
    assert not renderer._prepare_grouped_audio("source", plan, pcm, tmp_path / "graph.txt")
    assert not pcm.exists()


@pytest.mark.parametrize("mode", ["success", "fallback", "unknown", "cancel", "cache"])
def test_grouped_audio_dispatch_preserves_graph_lease_and_failure_contract(tmp_path, monkeypatch, mode):
    from src.stages import renderer, timeline
    from src.hardware import render_cache
    retained = {}
    monkeypatch.setattr(renderer, "_retained_batch_locks", retained)
    monkeypatch.setattr(renderer, "TEMP_DIR", tmp_path)
    monkeypatch.setattr(render_cache, "reusable", lambda *a, **k: mode == "cache")
    monkeypatch.setattr(render_cache, "render_fingerprint", lambda *a, **k: "identity")
    calls = []

    def prepare(source, plan, pcm, script):
        calls.append("prepare")
        pcm.write_bytes(b"pcm")
        script.write_text("group")
        if mode == "unknown":
            raise renderer.MediaTerminationUnconfirmed("live reader")
        if mode == "cancel":
            raise renderer.RenderInterrupted("cancelled")
        if mode == "fallback":
            raise RuntimeError("confirmed failed preparation")
        return True

    def acquire(lease, **kwargs):
        calls.append("admission")
        assert lease.inputs == (2 if mode == "success" else 1)
        graph = (tmp_path / "_fc_batch0_20260328_cam0.txt").read_text()
        assert graph == ("video;audio" if mode == "success" else "original")
        return False

    monkeypatch.setattr(renderer, "_prepare_grouped_audio", prepare)
    monkeypatch.setattr(renderer, "acquire_with_retry", acquire)
    output = tmp_path / "batch.mp4"
    plan = timeline.GroupedAudioPlan("video", (("group", 1),), "audio", 48000, 1)
    args = (["source"], "original", output, "nv", 20, {}, {}, "20260328", 0)
    try:
        if mode in ("unknown", "cancel"):
            error = renderer.RenderTerminationUnconfirmed if mode == "unknown" else renderer.RenderInterrupted
            with pytest.raises(error):
                renderer._run_batch_render(*args, config={}, audio_plan=plan)
            assert calls == ["prepare"]
        else:
            result = renderer._run_batch_render(*args, config={}, audio_plan=plan)
            assert result == (str(output) if mode == "cache" else None)
            assert calls == ([] if mode == "cache" else ["prepare", "admission"])
        assert (tmp_path / "batch.audio.f32").exists() is (mode == "unknown")
    finally:
        for stack in retained.values():
            stack.close()


@pytest.mark.parametrize("manifest_state", ["valid", "missing", "legacy", "stale", "malformed", "zero"])
def test_concat_duration_cache_falls_back_and_keeps_full_validation(tmp_path, monkeypatch, manifest_state):
    import json
    from fractions import Fraction
    from types import SimpleNamespace
    import av
    from src.hardware import render_cache
    from src.hardware.ffmpeg import FFmpegResult
    from src.stages import renderer
    batch, output = tmp_path / "batch.mp4", tmp_path / "final.mp4"
    batch.write_bytes(b"original batch")
    render_cache.save_manifest(batch, "identity", duration=2)
    manifest_path = render_cache._manifest_path(batch)
    data = json.loads(manifest_path.read_text())
    if manifest_state == "missing":
        manifest_path.unlink()
    elif manifest_state == "malformed":
        manifest_path.write_text("{")
    else:
        if manifest_state == "legacy":
            del data["duration"]
        elif manifest_state == "stale":
            data["output"][1] += 1
        elif manifest_state == "zero":
            data["duration"] = 0
        manifest_path.write_text(json.dumps(data))
    probes, validations = [], []

    class Container:
        duration = 2 * av.time_base
        streams = SimpleNamespace(video=[SimpleNamespace(duration=40, time_base=Fraction(1, 20))])

        def __enter__(self):
            probes.append(True)
            return self

        def __exit__(self, *args):
            pass

    def encode(args, **kwargs):
        descriptor = Path(args[args.index("-i") + 1]).read_text()
        assert descriptor.count("duration 2.000000000") == 2
        Path(args[-1]).write_bytes(b"joined media")
        return FFmpegResult(0, b"", b"")

    def validate(path, expected, **kwargs):
        validations.append((expected, kwargs["checkpoints"], kwargs["require_audio"]))
        kwargs["diagnostics"]["stage"] = "completed"
        return True

    monkeypatch.setattr(av, "open", lambda *a, **k: Container())
    monkeypatch.setattr(renderer, "run_ffmpeg", encode)
    monkeypatch.setattr(render_cache, "valid_video", validate)
    assert renderer.concat_output_files([batch, batch], output)
    assert len(probes) == (0 if manifest_state == "valid" else 2)
    assert validations == [(4, [2], True)]
    assert output.read_bytes() == b"joined media"
    assert batch.read_bytes() == b"original batch"


def test_watch_export_real_frame_boundaries_and_integer_audio(tmp_path,monkeypatch):
    import av
    import threading
    import numpy as np
    from scripts.audit_tool.watch_export import encode_daily
    from src.hardware.ffmpeg import run_ffmpeg
    source=tmp_path/'parent.mp4'
    result=run_ffmpeg(['-f','lavfi','-i','testsrc2=size=320x180:rate=20:duration=6',
        '-f','lavfi','-i','sine=frequency=440:sample_rate=48000:duration=6',
        '-c:v','libx264','-c:a','aac','-output_ts_offset','0.021',str(source)],timeout=30)
    assert result.returncode==0,result.stderr
    with av.open(str(source)) as c:
        parent_frames=list(c.decode(video=0))
        origin=float(parent_frames[0].pts*parent_frames[0].time_base)
    pending=tmp_path/'pending';pending.mkdir()
    from scripts.audit_tool import watch_export as w
    # Exercise the real trim/concat/validation graph without requiring GPU CI.
    def software_encode(args,**kwargs):
        args=list(args)
        if 'h264_nvenc' in args:
            args[args.index('h264_nvenc')]='libx264'
            args[args.index('p4')]='fast'
        return run_ffmpeg(args,**kwargs)
    monkeypatch.setattr(w,'run_ffmpeg',software_encode)
    intervals=[(21,47),(87,115)]
    items=[{'disposition':'daily','daily_start_frame':cursor,'output_start_frame':a,'output_end_frame':b,'frames':b-a}
        for cursor,(a,b) in zip((0,26),intervals)]
    frozen={'plan':{'fps':20,'items':items,'selected_frames':54},
            'binding':{'parent':{'video_origin_s':origin,'path':str(source)}}}
    result=encode_daily(frozen,pending,threading.Event(),lambda **kwargs:None)
    assert result['decoded_video_frames']==54 and result['all_joins_clock_checked']
    with av.open(str(pending/'daily.mp4')) as c: daily=list(c.decode(video=0))
    for new,old in ((0,21),(25,46),(26,87),(53,114)):
        a=parent_frames[old].to_ndarray(format='gray').astype(float)
        b=daily[new].to_ndarray(format='gray').astype(float)
        assert np.mean((a-b)**2)<25  # Independent boundary correspondence, beyond total frame count.


def test_run_ffmpeg_task_cancel_stops_owned_process_only(tmp_path):
    import threading
    import time
    from src.hardware.ffmpeg import FFmpegProcessRegistry,run_ffmpeg
    event=threading.Event();result=[]
    before=set(FFmpegProcessRegistry._processes)
    worker=threading.Thread(target=lambda:result.append(run_ffmpeg([
        '-re','-f','lavfi','-i','testsrc2=size=64x64:rate=20','-t','60',
        '-c:v','libx264',str(tmp_path/'cancel.mp4')],timeout=20,cancel_event=event)))
    worker.start()
    deadline=time.monotonic()+5
    while set(FFmpegProcessRegistry._processes)==before and time.monotonic()<deadline:time.sleep(.02)
    assert set(FFmpegProcessRegistry._processes)!=before
    event.set();worker.join(8)
    assert not worker.is_alive() and result[0].returncode!=0
    assert set(FFmpegProcessRegistry._processes)==before
    assert not FFmpegProcessRegistry.is_interrupted()
