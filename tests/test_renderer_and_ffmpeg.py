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

    def test_keyframe_fastpath_pure_static_file(self):
        """纯静态长文件走 select 抽帧快路径，滤镜图标签保持闭包。"""
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
        assert "trim=duration=" in graph
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
        assert "trim=duration=" in graph

    def test_simple_dynamic_filter_bypasses_concat_graph(self):
        seg = TimelineSegment("clip.mp4", 0, 0.0, 30.0, "DYNAMIC", 30.0)
        graph = build_concat_filter(
            [seg], [{"filepath": "clip.mp4", "has_audio": 0}],
            scale_mode="cuda_passthrough", simple_dynamic=True,
        )
        assert "concat=n=" not in graph
        assert "split=" not in graph
        assert "scale_cuda=1920:1080" in graph

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

    def test_concat_output_files_empty_list(self, tmp_path):
        out = tmp_path / "final.mp4"
        assert not concat_output_files([], out)
        assert not out.exists()

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
            "qsv": {"preset": "fast", "global_quality": 28, "maxrate": "4M", "bufsize": "8M", "pix_fmt": "nv12"},
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
            "qsv": {"preset": "fast", "global_quality": 28, "maxrate": "4M", "bufsize": "8M", "pix_fmt": "nv12"},
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
