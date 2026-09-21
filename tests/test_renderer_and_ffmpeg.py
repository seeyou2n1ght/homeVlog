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
        "-filter_complex_script", str(graph_path), "-map", "[v]", "-map", "[a]",
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


class TestRenderWatchdog:
    def test_multi_input_startup_gets_longer_grace(self):
        assert _startup_watchdog_timeout(8, 120.0, 600.0, 4800.0) == 600.0

    def test_startup_grace_never_exceeds_config_or_render_timeout(self):
        assert _startup_watchdog_timeout(8, 120.0, 300.0, 4800.0) == 300.0
        assert _startup_watchdog_timeout(8, 120.0, 600.0, 240.0) == 240.0

    def test_startup_grace_never_weaker_than_steady_state(self):
        assert _startup_watchdog_timeout(1, 120.0, 30.0, 3600.0) == 120.0


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
        assert "async=1000:first_pts=0,asetpts=N/SR/TB[a]" in filter_str
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

    def test_audio_padding_cannot_shift_video_seams(self, tmp_path):
        import av
        import hashlib
        import numpy as np

        clips = []
        for i, color in enumerate(["red", "blue"]):
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
        assert concat_output_files(clips, output)
        frames = video(output)
        assert [h for _, h in frames] == expected_hashes
        assert len(frames) == 80
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
        from types import SimpleNamespace
        out, batch = tmp_path / "out.mp4", tmp_path / "batch.mp4"
        out.write_bytes(b"previous")
        batch.write_bytes(b"batch")
        def encode(args, **kwargs):
            Path(args[-1]).write_bytes(b"invalid join")
            return SimpleNamespace(returncode=0)
        with patch("src.renderer.get_duration", return_value=2), patch("src.renderer.run_ffmpeg", side_effect=encode), patch("src.render_cache.valid_video", return_value=False):
            assert not concat_output_files([batch, batch], out)
        assert out.read_bytes() == b"previous"
        assert batch.read_bytes() == b"batch"
        assert not out.with_name("out.tmp.mp4").exists()

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
