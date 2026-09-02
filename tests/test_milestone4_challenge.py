"""
Adversarial Challenge & Stress-Test Suite for Milestone 4 (R4: Smooth Speed Ramping & Strict Timecode Closure).

Empirical verification harness targeting:
1. 100+ Parameter variations matrix for calculate_speed_ramping_curve (durations, speeds, ramp durations, ramp flags).
2. Dense PTS monotonicity and non-inversion verification (10,000+ sample points across extreme ratios 1x - 1000x).
3. Micro-static segments (< 0.5s down to 0.001s) with proportional scaling and zero cruise zones.
4. Audio cross-fading (afade) clamping, envelope bounds, and zero/negative fade handling on short dynamic segments.
5. Consecutive dynamic pulses with micro-static gaps and mixed audio stream configurations.
6. Real-world wall-clock timecode OSD escaping, 24-hour midnight rollover, and SRT/ASS subtitle structural validity.
7. Multi-file batch boundary splitting, duration conservation, and distinct file batch limits.
"""

import math
import re
import pytest
from pathlib import Path

from src.segment import (
    Segment,
    split_segments_at_file_boundaries,
    merge_cross_file,
)
from src.timeline import (
    TimelineSegment,
    SpeedRampInfo,
    calculate_speed_ramping_curve,
    evaluate_pts_easing,
    build_timecode_drawtext_filter,
    generate_timecode_subtitles,
    save_timecode_subtitles,
    partition_timeline_by_batches,
    build_timeline,
    build_concat_filter,
    _format_srt_timestamp,
    _format_ass_timestamp,
)
from src.utils import ts_to_unix
from tests.helpers import parse_ffmpeg_filtergraph, verify_filtergraph_labels_closure


# ============================================================================
# 1. 100+ Parameter Variations Matrix for calculate_speed_ramping_curve
# ============================================================================

class TestSpeedRampingParameterVariationsMatrix:
    """Exhaustive matrix stress testing of calculate_speed_ramping_curve across 100+ parameter variations."""

    @pytest.mark.parametrize("dur", [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 15.0, 30.0, 60.0, 300.0, 3600.0])
    @pytest.mark.parametrize("v_fast", [1.0, 1.5, 2.0, 5.0, 15.0, 30.0, 60.0, 120.0, 240.0, 1000.0])
    @pytest.mark.parametrize("ramp_dur", [0.1, 0.5, 1.0, 2.0])
    @pytest.mark.parametrize("ramps", [(True, True), (True, False), (False, True), (False, False)])
    def test_parameter_matrix_invariants(
        self,
        dur: float,
        v_fast: float,
        ramp_dur: float,
        ramps: tuple[bool, bool],
    ):
        """Verify mathematical invariants across 1,760 parameter combinations."""
        has_in, has_out = ramps
        info = calculate_speed_ramping_curve(
            dur=dur,
            v_fast=v_fast,
            has_ramp_in=has_in,
            has_ramp_out=has_out,
            ramp_duration_s=ramp_dur,
        )

        # Invariant 1: duration components are non-negative
        assert info.ramp_in_src_dur >= 0.0
        assert info.ramp_out_src_dur >= 0.0
        assert info.cruise_src_dur >= 0.0

        # Invariant 2: source duration conservation
        total_src = info.ramp_in_src_dur + info.cruise_src_dur + info.ramp_out_src_dur
        assert total_src == pytest.approx(dur, abs=1e-5)

        # Invariant 3: target display duration bounds and effective speed
        if not has_in and not has_out:
            assert info.target_display_dur == pytest.approx(dur / v_fast, rel=1e-3)
            assert info.effective_speed == pytest.approx(v_fast, rel=1e-3)
        else:
            assert 0.0 < info.target_display_dur <= max(0.05, dur)
            if dur >= 0.05:
                assert info.effective_speed >= 0.999
            else:
                assert info.target_display_dur == pytest.approx(0.05, abs=1e-5)
                assert info.effective_speed == pytest.approx(dur / 0.05, rel=1e-3)

        # Invariant 4: PTS expression is non-empty and contains no NaN / Inf
        assert info.pts_expr != ""
        assert "nan" not in info.pts_expr.lower()
        assert "inf" not in info.pts_expr.lower()

        # Invariant 5: Flag conformance
        assert info.has_ramp_in == has_in
        assert info.has_ramp_out == has_out
        if not has_in:
            assert info.ramp_in_src_dur == 0.0
        if not has_out:
            assert info.ramp_out_src_dur == 0.0


# ============================================================================
# 2. Dense PTS Monotonicity & Inversion-Free Stress Harness
# ============================================================================

class TestPTSMonotonicityDenseHarness:
    """Dense evaluation harness checking for timestamp regression and monotonicity across 10,000+ points."""

    @pytest.mark.parametrize("dur", [0.05, 0.2, 1.0, 10.0, 60.0, 600.0])
    @pytest.mark.parametrize("v_fast", [2.0, 15.0, 60.0, 120.0, 500.0])
    @pytest.mark.parametrize("ramps", [(True, True), (True, False), (False, True), (False, False)])
    def test_monotonicity_and_derivatives(self, dur: float, v_fast: float, ramps: tuple[bool, bool]):
        """Sample 200 points per curve to verify strict monotonicity and non-negative derivatives."""
        has_in, has_out = ramps
        info = calculate_speed_ramping_curve(dur, v_fast, has_in, has_out, ramp_duration_s=1.0)

        steps = 200
        prev_pts = -1.0
        for i in range(steps + 1):
            t = (i / steps) * dur
            pts, speed = evaluate_pts_easing(t, dur, v_fast, has_in, has_out, ramp_duration_s=1.0)

            # Monotonicity check
            assert pts >= prev_pts, f"PTS inversion at t={t}: {pts} < {prev_pts} for dur={dur}, v_fast={v_fast}"

            # Derivative check: numerical dP/dt >= 0
            if i > 0:
                delta_t = dur / steps
                deriv = (pts - prev_pts) / delta_t
                assert deriv >= 0.0, f"Negative numerical derivative at t={t}: {deriv}"

            prev_pts = pts

        # Boundary checks
        pts_start, _ = evaluate_pts_easing(0.0, dur, v_fast, has_in, has_out, 1.0)
        assert pts_start == pytest.approx(0.0, abs=1e-5)

        if has_in:
            _, sp0 = evaluate_pts_easing(0.0, dur, v_fast, has_in, has_out, 1.0)
            assert sp0 == pytest.approx(1.0, rel=1e-2)

        if has_out:
            _, sp_end = evaluate_pts_easing(dur, dur, v_fast, has_in, has_out, 1.0)
            assert sp_end == pytest.approx(1.0, rel=1e-2)


# ============================================================================
# 2.5 Extreme Speed Ratios & Clamping
# ============================================================================

class TestExtremeSpeedRatios:
    """Stress tests on extreme and invalid speed ratios (sub-1.0, zero, negative)."""

    @pytest.mark.parametrize("v_fast", [0.5, 0.0, -1.0, -100.0])
    def test_invalid_sub_one_speed_ratios_clamping(self, v_fast: float):
        """Verify sub-1.0 and negative speeds are clamped to 1.0x without division by zero or negative PTS."""
        dur = 10.0
        # No ramps: linear 1.0x -> pts_mid at t=5.0 should be 5.0
        pts_mid, sp_mid = evaluate_pts_easing(5.0, dur, v_fast, False, False, 1.0)
        assert pts_mid == pytest.approx(5.0, abs=1e-5)
        assert sp_mid == pytest.approx(1.0, abs=1e-5)

        # With ramps: ensure monotonicity and non-negative PTS
        steps = 50
        prev_pts = -1.0
        for i in range(steps + 1):
            t = (i / steps) * dur
            pts, sp = evaluate_pts_easing(t, dur, v_fast, True, True, 1.0)
            assert pts >= prev_pts, f"PTS inversion at t={t} for v_fast={v_fast}"
            assert pts >= 0.0
            assert sp >= 0.99
            prev_pts = pts


# ============================================================================
# 3. Micro-Static Segments (< 0.5s down to 0.001s)
# ============================================================================

class TestMicroStaticSegments:
    """Stress-test speed ramping and filtergraph generation on micro-static segments."""

    @pytest.mark.parametrize("dur", [0.49, 0.25, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001])
    def test_micro_static_durations_monotonicity(self, dur: float):
        """Verify micro-static segments scale proportionally without division by zero or NaN."""
        v_fast = 30.0
        ramp_info = calculate_speed_ramping_curve(
            dur=dur,
            v_fast=v_fast,
            has_ramp_in=True,
            has_ramp_out=True,
            ramp_duration_s=1.0,
        )
        assert ramp_info.has_ramp_in
        assert ramp_info.has_ramp_out
        assert ramp_info.cruise_src_dur == 0.0
        assert ramp_info.ramp_in_src_dur + ramp_info.ramp_out_src_dur == pytest.approx(dur, abs=1e-6)

        steps = 100
        prev_pts = -1.0
        for i in range(steps + 1):
            t = (i / steps) * dur
            pts, speed = evaluate_pts_easing(t, dur, v_fast, True, True, 1.0)
            assert pts >= prev_pts, f"PTS decreased at t={t}: {pts} < {prev_pts} for dur={dur}"
            assert speed >= 0.99, f"Instantaneous speed dropped below 1.0x at t={t}: {speed}"
            prev_pts = pts

    @pytest.mark.parametrize("dur", [0.4, 0.1, 0.05, 0.01])
    def test_micro_static_filtergraph_generation(self, dur: float):
        """Verify build_concat_filter generates syntactically valid filtergraph with micro-static segments."""
        timeline = [
            TimelineSegment(filepath="cam1.mp4", input_index=0, start_in_file=0.0, end_in_file=2.0, state="DYNAMIC", duration=2.0),
            TimelineSegment(filepath="cam1.mp4", input_index=0, start_in_file=2.0, end_in_file=2.0 + dur, state="STATIC", duration=dur),
            TimelineSegment(filepath="cam1.mp4", input_index=0, start_in_file=2.0 + dur, end_in_file=4.0 + dur, state="DYNAMIC", duration=2.0),
        ]
        rows = [{"filepath": "cam1.mp4", "has_audio": 1}]
        fc = build_concat_filter(timeline=timeline, rows=rows, speed_ramping=True)
        valid, msg = verify_filtergraph_labels_closure(fc)
        assert valid, f"Filtergraph label closure failed for dur={dur}: {msg}"

        parsed = parse_ffmpeg_filtergraph(fc)
        assert parsed["concat_n"] == 3
        assert len(parsed["video_trims"]) == 3
        assert len(parsed["audio_trims"]) == 2
        assert len(parsed["audio_nulls"]) == 1


# ============================================================================
# 4. Audio Linear Cross-Fading (afade) Clamping & Bounds
# ============================================================================

class TestAudioCrossFadeRobustness:
    """Stress tests on audio fade boundaries, short segment clamping, and edge cases."""

    @pytest.mark.parametrize("dur", [10.0, 5.0, 2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
    def test_afade_clamping_on_short_segments(self, dur: float):
        """Verify fade duration is strictly clamped to min(audio_fade_duration_s, dur/2.0)."""
        requested_fade = 0.15
        expected_fade = min(requested_fade, dur / 2.0)
        expected_fade_out_st = dur - expected_fade

        timeline = [
            TimelineSegment(filepath="clip.mp4", input_index=0, start_in_file=0.0, end_in_file=dur, state="DYNAMIC", duration=dur),
        ]
        rows = [{"filepath": "clip.mp4", "has_audio": 1}]
        fc = build_concat_filter(timeline=timeline, rows=rows, audio_fade_duration_s=requested_fade)
        parsed = parse_ffmpeg_filtergraph(fc)
        assert len(parsed["audio_trims"]) == 1
        a_trim = parsed["audio_trims"][0]

        assert f"afade=t=in:ss=0:d={expected_fade:.3f}" in a_trim
        assert f"afade=t=out:st={expected_fade_out_st:.3f}:d={expected_fade:.3f}" in a_trim

    def test_zero_fade_duration_omits_afade(self):
        """Verify audio_fade_duration_s <= 0 produces clean atrim without afade."""
        timeline = [
            TimelineSegment(filepath="clip.mp4", input_index=0, start_in_file=0.0, end_in_file=5.0, state="DYNAMIC", duration=5.0),
        ]
        rows = [{"filepath": "clip.mp4", "has_audio": 1}]
        fc = build_concat_filter(timeline=timeline, rows=rows, audio_fade_duration_s=0.0)
        parsed = parse_ffmpeg_filtergraph(fc)
        assert len(parsed["audio_trims"]) == 1
        assert "afade" not in parsed["audio_trims"][0]

    def test_silent_audio_fallback(self):
        """Verify that files with has_audio=0 generate anullsrc audio."""
        timeline = [
            TimelineSegment(filepath="silent.mp4", input_index=0, start_in_file=0.0, end_in_file=3.0, state="DYNAMIC", duration=3.0),
        ]
        rows = [{"filepath": "silent.mp4", "has_audio": 0}]
        fc = build_concat_filter(timeline=timeline, rows=rows)
        parsed = parse_ffmpeg_filtergraph(fc)
        assert len(parsed["audio_trims"]) == 0
        assert len(parsed["audio_nulls"]) == 1
        assert "anullsrc=" in parsed["audio_nulls"][0]


# ============================================================================
# 5. Consecutive Dynamic Pulses with Micro-Static Gaps
# ============================================================================

class TestConsecutiveDynamicPulsesMicroGaps:
    """Stress-test alternating rapid dynamic pulses and micro static gaps."""

    def test_rapid_alternating_pulses_filtergraph(self):
        """Test a series of 10 alternating dynamic pulses (0.2s) and micro static gaps (0.1s)."""
        timeline = []
        cur_t = 0.0
        for _ in range(10):
            timeline.append(TimelineSegment(
                filepath="cam1.mp4",
                input_index=0,
                start_in_file=cur_t,
                end_in_file=cur_t + 0.2,
                state="DYNAMIC",
                duration=0.2,
            ))
            cur_t += 0.2
            timeline.append(TimelineSegment(
                filepath="cam1.mp4",
                input_index=0,
                start_in_file=cur_t,
                end_in_file=cur_t + 0.1,
                state="STATIC",
                duration=0.1,
            ))
            cur_t += 0.1

        rows = [{"filepath": "cam1.mp4", "has_audio": 1}]
        fc = build_concat_filter(
            timeline=timeline,
            rows=rows,
            speed_ramping=True,
            audio_fade_duration_s=0.15,
        )

        valid, msg = verify_filtergraph_labels_closure(fc)
        assert valid, f"Rapid pulse filtergraph closure error: {msg}"

        parsed = parse_ffmpeg_filtergraph(fc)
        assert parsed["concat_n"] == 20
        assert len(parsed["video_trims"]) == 20
        assert len(parsed["audio_trims"]) == 10
        assert len(parsed["audio_nulls"]) == 10

        for a_trim in parsed["audio_trims"]:
            assert "afade=t=in:ss=0:d=0.100" in a_trim
            assert "afade=t=out:st=0.100:d=0.100" in a_trim


# ============================================================================
# 6. Real-World Wall-Clock Timecode OSD & Subtitle Generator Stress
# ============================================================================

class TestTimecodeOSDAndSubtitleStress:
    """Stress tests on timecode formatting, midnight rollover, and multi-hour timelines."""

    def test_timestamp_formatting_helpers(self):
        """Test SRT and ASS timestamp formatters across hours, minutes, seconds, milliseconds."""
        assert _format_srt_timestamp(0.0) == "00:00:00,000"
        assert _format_ass_timestamp(0.0) == "0:00:00.00"

        t = 3600.0 + 23 * 60.0 + 45.0 + 0.678
        assert _format_srt_timestamp(t) == "01:23:45,678"
        assert _format_ass_timestamp(t) == "1:23:45.68"

        t_25h = 25 * 3600.0 + 12.34
        assert _format_srt_timestamp(t_25h) == "25:00:12,340"
        assert _format_ass_timestamp(t_25h) == "25:00:12.34"

    def test_srt_subtitles_syntax_and_line_numbering(self, tmp_path: Path):
        """Generate SRT subtitles for 50 segments and verify structural correctness."""
        timeline = []
        for i in range(50):
            timeline.append(TimelineSegment(
                filepath="cam.mp4",
                input_index=0,
                start_in_file=i * 10.0,
                end_in_file=(i + 1) * 10.0,
                state="DYNAMIC" if i % 2 == 0 else "STATIC",
                duration=10.0,
            ))
        rows = [{"filepath": "cam.mp4", "file_start_time": "20260901080000"}]
        srt_content = generate_timecode_subtitles(timeline=timeline, rows=rows, base_date="20260901", format_type="srt")
        assert "-->" in srt_content

        out_srt = tmp_path / "test.srt"
        save_timecode_subtitles(timeline, out_srt, rows=rows, base_date="20260901", format_type="srt")
        assert out_srt.exists()
        assert out_srt.stat().st_size > 0

        # Verify saving pre-rendered string content directly
        out_srt2 = tmp_path / "test2.srt"
        save_timecode_subtitles(srt_content, out_srt2)
        assert out_srt2.exists()
        assert out_srt2.read_text(encoding="utf-8") == srt_content

    def test_ass_subtitles_header_and_events(self, tmp_path: Path):
        """Generate ASS subtitles and verify header, style, and Dialogue events."""
        timeline = [
            TimelineSegment(filepath="cam.mp4", input_index=0, start_in_file=0.0, end_in_file=5.0, state="DYNAMIC", duration=5.0)
        ]
        rows = [{"filepath": "cam.mp4", "file_start_time": "20260901120000"}]
        ass_content = generate_timecode_subtitles(timeline=timeline, rows=rows, format_type="ass")
        assert "[Script Info]" in ass_content
        assert "[V4+ Styles]" in ass_content
        assert "[Events]" in ass_content
        assert "Dialogue: 0," in ass_content
        assert "2026-09-01 12:00:00" in ass_content


# ============================================================================
# 7. Batch Boundary Splitting & Multi-File Partitioning Stress
# ============================================================================

class TestBatchBoundaryPartitioningStress:
    """Stress-test segment splitting across 10+ file boundaries and batch partitioning."""

    def test_multi_file_boundary_splitting_10_files(self):
        """A single long dynamic segment spanning 10 files is cleanly split into 10 bounded segments."""
        n_files = 10
        files_info = []
        for i in range(n_files):
            files_info.append({
                "filepath": f"clip_{i:02d}.mp4",
                "file_start_offset": i * 60.0,
                "file_end_offset": (i + 1) * 60.0,
                "duration": 60.0,
            })

        merged_seg = [
            Segment(start_time=0.0, end_time=600.0, state="DYNAMIC", source_file="clip_00.mp4", file_start_offset=0.0)
        ]

        split_res = split_segments_at_file_boundaries(merged_seg, files_info)
        assert len(split_res) == 10

        for i, piece in enumerate(split_res):
            assert piece.source_file == f"clip_{i:02d}.mp4"
            assert piece.start_time == pytest.approx(i * 60.0, abs=1e-5)
            assert piece.end_time == pytest.approx((i + 1) * 60.0, abs=1e-5)
            assert piece.file_start_offset == pytest.approx(i * 60.0, abs=1e-5)
            dur_in_file = piece.end_time - piece.start_time
            assert dur_in_file == pytest.approx(60.0, abs=1e-5)

    def test_partition_large_timeline_into_batches(self):
        """Partition 200 segments from 50 files into batches with batch_max_files=8."""
        timeline = []
        for f_idx in range(50):
            fn = f"file_{f_idx:03d}.mp4"
            for s_idx in range(4):
                timeline.append(TimelineSegment(
                    filepath=fn,
                    input_index=0,
                    start_in_file=s_idx * 15.0,
                    end_in_file=(s_idx + 1) * 15.0,
                    state="DYNAMIC" if s_idx % 2 == 0 else "STATIC",
                    duration=15.0,
                ))

        batches = partition_timeline_by_batches(timeline, batch_max_files=8)
        assert len(batches) > 1

        total_segs = 0
        for batch_idx, batch in enumerate(batches):
            distinct_files = set(seg.filepath for seg in batch)
            assert len(distinct_files) <= 8, f"Batch {batch_idx} exceeded max files: {len(distinct_files)} > 8"
            max_input_idx = max(seg.input_index for seg in batch)
            assert max_input_idx == len(distinct_files) - 1
            for seg in batch:
                assert 0 <= seg.input_index < len(distinct_files)
            total_segs += len(batch)

        assert total_segs == 200, f"Segment count mismatch: {total_segs} != 200"
