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
    build_concat_filter,
    concat_output_files,
)
from src.timeline import (
    TimelineSegment,
    partition_timeline_by_batches,
)
from tests.helpers import verify_filtergraph_labels_closure


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
        assert res == str(fake_batch)

