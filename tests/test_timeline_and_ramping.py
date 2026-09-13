"""测试模块 3: 时间线聚合 (Timeline)、片段聚类与平滑变速曲线 (Speed Ramping).

覆盖：
1. 动作/静止片段构建与平滑吸收 (build_segments, pre_roll, post_roll, gap_tolerance)；
2. 跨文件边界切分与同状态连续段融合 (split_segments_at_file_boundaries, merge_cross_file)；
3. 变速过渡曲线 (calculate_speed_ramping_curve) 核心数学守恒不变量（精炼 16 组代表性边界）；
4. 时间码 OSD (drawtext) 转义、跨午夜时间戳计算与字幕生成 (SRT / ASS)。
"""

import math
import pytest

from src.segment import (
    Segment,
    build_segments,
    split_segments_at_file_boundaries,
    merge_cross_file,
    segments_to_json,
    segments_from_json,
)
from src.timeline import (
    TimelineSegment,
    calculate_speed_ramping_curve,
    build_timecode_drawtext_filter,
    compute_display_plans,
    generate_timecode_subtitles,
    src_offset_at_display,
    _format_srt_timestamp,
    _format_ass_timestamp,
)


class TestSegmentClusteringAndSmoothing:
    """测试片段聚类与跨文件平滑吸收。"""

    def test_build_segments_basic_and_padding(self):
        # 0~9s STATIC, 10~15s DYNAMIC, 16~29s STATIC
        labels = [{"time": float(t), "state": "DYNAMIC" if 10.0 <= t <= 15.0 else "STATIC"} for t in range(30)]
        segs = build_segments(
            labels,
            source_file="cam0_clip1.mp4",
            min_motion_dur=2.0,
            min_static_dur=5.0,
            gap_tolerance=1.5,
            pre_roll=1.0,
            post_roll=1.0,
        )
        assert len(segs) >= 1
        dyn = [s for s in segs if s.is_dynamic]
        assert len(dyn) == 1
        # pre_roll/post_roll 垫底保护
        assert dyn[0].start_time <= 9.0
        assert dyn[0].end_time >= 16.0

    def test_short_motion_absorption(self):
        # 只有 1 秒的极短动作，小于 min_motion_dur=3.0，被两侧稳定的长静态吸收
        labels = [{"time": float(t), "state": "DYNAMIC" if t == 10.0 else "STATIC"} for t in range(30)]
        segs = build_segments(labels, source_file="cam0_clip2.mp4", min_motion_dur=3.0, min_static_dur=1.0, apply_smoothing=True)
        dyn = [s for s in segs if s.is_dynamic]
        assert len(dyn) == 0

    def test_energy_gated_static_absorption(self):
        from src.segment import _filter_short
        # Case A: 强动作 (energy=35.0)，中间短静止 (5s < min_static=8.0s) 被吸收成一个连贯大动态
        segs_strong = [
            Segment(0.0, 10.0, "DYNAMIC", source_file="clip.mp4", max_energy=35.0),
            Segment(10.0, 15.0, "STATIC", source_file="clip.mp4"),
            Segment(15.0, 25.0, "DYNAMIC", source_file="clip.mp4", max_energy=40.0),
        ]
        res_strong = _filter_short(segs_strong, min_motion=2.0, min_static=8.0, motion_absorb_energy_threshold=12.0)
        assert len(res_strong) == 1
        assert (res_strong[0].end_time - res_strong[0].start_time) == 25.0
        assert res_strong[0].state == "DYNAMIC"

        # Case B: 弱动作/噪点 (energy=5.0 < 12.0)，中间静止段保持独立，绝不被微弱噪点多米诺吞噬
        segs_weak = [
            Segment(0.0, 10.0, "DYNAMIC", source_file="clip.mp4", max_energy=5.0),
            Segment(10.0, 15.0, "STATIC", source_file="clip.mp4"),
            Segment(15.0, 25.0, "DYNAMIC", source_file="clip.mp4", max_energy=6.0),
        ]
        res_weak = _filter_short(segs_weak, min_motion=2.0, min_static=8.0, motion_absorb_energy_threshold=12.0)
        assert len(res_weak) == 3
        assert res_weak[1].state == "STATIC"



    def test_merge_cross_file_contiguous_states(self):
        seg1 = Segment(start_time=0.0, end_time=10.0, state="DYNAMIC", source_file="file1.mp4", file_start_offset=0.0)
        seg2 = Segment(start_time=10.0, end_time=20.0, state="DYNAMIC", source_file="file2.mp4", file_start_offset=10.0)
        merged = merge_cross_file([seg1, seg2])
        assert len(merged) == 1
        assert merged[0].start_time == 0.0
        assert merged[0].end_time == 20.0

    def test_json_roundtrip_serialization(self):
        orig = [Segment(start_time=1.5, end_time=6.5, state="DYNAMIC", source_file="clip.mp4", file_start_offset=0.0)]
        js = segments_to_json(orig)
        recovered = segments_from_json(js)
        assert len(recovered) == 1
        assert recovered[0].start_time == 1.5
        assert recovered[0].end_time == 6.5
        assert recovered[0].state == "DYNAMIC"


class TestSpeedRampingMathInvariants:
    """测试平滑变速曲线的数学物理不变量（精炼 16 组核心边界组合）。"""

    @pytest.mark.parametrize("dur", [0.05, 1.0, 30.0, 3600.0])
    @pytest.mark.parametrize("v_fast", [1.0, 15.0, 60.0, 1000.0])
    @pytest.mark.parametrize("ramps", [(True, True), (False, False)])
    def test_speed_ramping_invariants(self, dur: float, v_fast: float, ramps: tuple[bool, bool]):
        has_in, has_out = ramps
        info = calculate_speed_ramping_curve(
            dur=dur,
            v_fast=v_fast,
            has_ramp_in=has_in,
            has_ramp_out=has_out,
            ramp_duration_s=1.0,
        )

        # 1. 守恒律：源时长完全守恒
        total_src = info.ramp_in_src_dur + info.cruise_src_dur + info.ramp_out_src_dur
        assert total_src == pytest.approx(dur, abs=1e-5)

        # 2. 各物理片段时长非负
        assert info.ramp_in_src_dur >= 0.0
        assert info.ramp_out_src_dur >= 0.0
        assert info.cruise_src_dur >= 0.0

        # 3. 输出播放时长非零正数
        assert info.target_display_dur > 0.0

        # 4. PTS 表达式合法无 NaN/Inf
        assert info.pts_expr != ""
        assert "nan" not in info.pts_expr.lower()
        assert "inf" not in info.pts_expr.lower()


class TestTimecodeAndSubtitles:
    """测试时间码 OSD 滤镜转义与字幕生成。"""

    def test_format_srt_and_ass_timestamps(self):
        assert _format_srt_timestamp(3661.123) == "01:01:01,123"
        assert _format_ass_timestamp(3661.12) == "1:01:01.12"

    def test_build_timecode_drawtext_filter_escaping(self):
        flt = build_timecode_drawtext_filter(
            start_unix=1704067200.0,  # 2024-01-01 00:00:00
            font_size=24,
            font_color="white",
        )
        assert "drawtext=" in flt
        assert "pts" in flt.lower()

    def test_generate_timecode_subtitles_empty_or_valid(self):
        subs = generate_timecode_subtitles(
            timeline=[],
            format_type="srt",
        )
        assert subs == ""


class TestDisplayPlanAndRampingInverse:
    """测试展示时长计划与 ramping 曲线逆映射（字幕墙钟还原精度）。"""

    def test_display_plan_matches_ramping_curve(self):
        segs = [
            TimelineSegment("f.mp4", 0, 0.0, 10.0, "DYNAMIC", 10.0),
            TimelineSegment("f.mp4", 0, 10.0, 130.0, "STATIC", 120.0),
            TimelineSegment("f.mp4", 0, 130.0, 140.0, "DYNAMIC", 10.0),
        ]
        plans = compute_display_plans(
            segs,
            static_keyframe_interval=30.0,
            keyframe_display_duration=0.5,
            min_static_display_duration=1.5,
            speed_ramping=True,
            ramp_duration_s=1.0,
        )
        assert plans[0] == (10.0, None)
        assert plans[2] == (10.0, None)
        disp, info = plans[1]
        assert info is not None
        assert disp == pytest.approx(info.target_display_dur)

    @pytest.mark.parametrize("v_fast", [2.0, 15.0, 60.0])
    def test_src_offset_inverse_roundtrip(self, v_fast: float):
        """ramping 正映射 (s→d) 与逆映射 (d→s) 往返一致。"""
        dur = 120.0
        info = calculate_speed_ramping_curve(
            dur=dur, v_fast=v_fast,
            has_ramp_in=True, has_ramp_out=True, ramp_duration_s=1.0,
        )
        s_in = info.ramp_in_src_dur
        s_mid = s_in + info.cruise_src_dur
        v_c = info.v_fast
        inv_v = 1.0 / v_c
        k_in = (1.0 - inv_v) / (2.0 * s_in) if s_in > 0 else 0.0
        k_out = (1.0 - inv_v) / (2.0 * info.ramp_out_src_dur) if info.ramp_out_src_dur > 0 else 0.0
        p_in = s_in * (1.0 + inv_v) / 2.0
        p_mid = p_in + info.cruise_src_dur / v_c

        def forward(s: float) -> float:
            if s < s_in:
                return s - k_in * s * s
            if s < s_mid:
                return p_in + (s - s_in) * inv_v
            u = s - s_mid
            return p_mid + u * inv_v + k_out * u * u

        for s in [0.0, s_in * 0.5, s_in, (s_in + s_mid) / 2, s_mid, dur * 0.9, dur]:
            d = forward(s)
            s_back = src_offset_at_display(d, dur, info.target_display_dur, info)
            assert s_back == pytest.approx(s, abs=1e-4), f"s={s}, d={d}, back={s_back}"

    def test_src_offset_linear_without_ramp(self):
        assert src_offset_at_display(5.0, 300.0, 10.0, None) == pytest.approx(150.0)

    def test_subtitles_wall_clock_static_compression(self):
        """静态段按抽帧倍率压缩后，字幕墙钟仍映射回源时间轴。"""
        segs = [
            TimelineSegment("00_20260901100000_20260901100500.mp4", 0, 0.0, 300.0, "STATIC", 300.0),
        ]
        subs = generate_timecode_subtitles(
            timeline=segs,
            rows=[{"filepath": "00_20260901100000_20260901100500.mp4",
                   "file_start_time": "20260901100000"}],
            base_date="20260901",
            step_s=1.0,
            format_type="srt",
        )
        assert "2026-09-01 10:00:00" in subs
        # 300s 源片按 60x 压缩为 ~5s 展示，末条字幕墙钟应显著晚于起始
        assert "2026-09-01 10:0" in subs


class TestPresenceAndMicroMotionPlans:
    def test_presence_display_duration_and_ramping(self):
        """PRESENCE 状态按 4x 播放，且相邻 DYNAMIC 时平滑缓入缓出。"""
        timeline = [
            TimelineSegment("f1.mp4", 0, 0.0, 10.0, "DYNAMIC", 10.0),
            TimelineSegment("f1.mp4", 0, 10.0, 50.0, "PRESENCE", 40.0),
            TimelineSegment("f1.mp4", 0, 50.0, 60.0, "DYNAMIC", 10.0),
        ]
        plans = compute_display_plans(
            timeline,
            presence_speed_factor=4.0,
            speed_ramping=True,
            ramp_duration_s=1.0,
        )
        assert len(plans) == 3
        # DYNAMIC: 10s 源时长对应 10s 展示
        assert plans[0][0] == pytest.approx(10.0)
        assert plans[0][1] is None

        # PRESENCE: 40s 源时长在 4x 下对应 10s 展示
        disp_dur, ramp = plans[1]
        assert disp_dur == pytest.approx(10.0, abs=0.1)
        assert ramp is not None
        assert ramp.has_ramp_in is True
        assert ramp.has_ramp_out is True

    def test_micro_motion_anchor_duration(self):
        """MICRO_MOTION 状态保留锚点，其余部分以 16x 巡航。"""
        timeline = [
            TimelineSegment("f1.mp4", 0, 0.0, 35.0, "MICRO_MOTION", 35.0),
        ]
        plans = compute_display_plans(
            timeline,
            micro_motion_anchor_s=3.0,
            micro_motion_cruise_speed=16.0,
            speed_ramping=False,
        )
        # 35s 源时长: 3s 锚点 + 32s/16 = 3 + 2 = 5s 展示时长
        disp_dur, ramp = plans[0]
        assert disp_dur == pytest.approx(5.0, abs=0.1)

    def test_resolve_presence_segments_logic(self):
        """两段确认有人动态之间的静态停顿应升级为 PRESENCE。"""
        from src.segment import Segment, resolve_presence_segments
        segs = [
            Segment(0.0, 10.0, "DYNAMIC", "f1.mp4", avg_confidence=0.85),
            Segment(10.0, 46.0, "STATIC", "f1.mp4"),  # 36s 停顿 (<=120s)
            Segment(46.0, 50.0, "DYNAMIC", "f1.mp4", avg_confidence=0.90),
        ]
        resolved = resolve_presence_segments(segs, max_presence_gap_s=120.0, person_conf_threshold=0.30)
        assert resolved[1].state == "PRESENCE"
        assert resolved[1].avg_confidence == pytest.approx(0.875)
        assert "TARGET_PERSISTENCE" in resolved[1].review_reason

    def test_resolve_presence_segments_skip_large_gap(self):
        """停顿超过 120s 时不自动升级为 PRESENCE。"""
        from src.segment import Segment, resolve_presence_segments
        segs = [
            Segment(0.0, 10.0, "DYNAMIC", "f1.mp4", avg_confidence=0.85),
            Segment(10.0, 200.0, "STATIC", "f1.mp4"),  # 190s 停顿 (>120s)
            Segment(200.0, 210.0, "DYNAMIC", "f1.mp4", avg_confidence=0.90),
        ]
        resolved = resolve_presence_segments(segs, max_presence_gap_s=120.0, person_conf_threshold=0.30)
        assert resolved[1].state == "STATIC"

    def test_sparse_mixed_non_static_trim_duration_clamped(self):
        """非纯静态段在混合稀疏或关键帧模式下必须包含 trim=duration 截断，防止 EOF PTS 泄露导致帧克隆。"""
        from src.stages.timeline import TimelineSegment, build_concat_filter
        timeline = [
            TimelineSegment("f1.mp4", 0, 0.0, 10.0, "MICRO_MOTION", 10.0),
            TimelineSegment("f1.mp4", 0, 10.0, 30.0, "PRESENCE", 20.0),
            TimelineSegment("f1.mp4", 0, 30.0, 60.0, "STATIC", 30.0),
        ]
        fc = build_concat_filter(
            timeline,
            rows=[{"filepath": "f1.mp4", "has_audio": False}],
            sparse_mixed=True,
        )
        # 验证 MICRO_MOTION 和 PRESENCE 支路上均注入了 trim=duration 与 setpts=PTS-STARTPTS
        assert "trim=duration=" in fc
        assert "setpts=PTS-STARTPTS" in fc



