import math
import re
import time
from pathlib import Path
import pytest

from src.segment import (
    Segment,
    split_segments_at_file_boundaries,
    merge_cross_file,
    segments_to_json,
    segments_from_json,
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
from src.utils import ts_to_unix, load_config
from tests.helpers import parse_ffmpeg_filtergraph, verify_filtergraph_labels_closure


# ============================================================================
# 1. Non-linear Speed Ramping & PTS Easing Mathematical Verification
# ============================================================================

class TestSpeedRampingMathematicsAndCurves:
    """Rigorous tests for speed ramping curve calculations and PTS easing monotonicity."""

    def test_pts_strict_monotonicity(self):
        """Verify that output PTS P(t) is strictly monotonically increasing for any duration and speed."""
        v_fast = 60.0
        dur = 30.0
        ramp_info = calculate_speed_ramping_curve(dur=dur, v_fast=v_fast, has_ramp_in=True, has_ramp_out=True, ramp_duration_s=1.0)
        assert ramp_info.has_ramp_in
        assert ramp_info.has_ramp_out
        assert ramp_info.target_display_dur > 0

        # Sample 1000 points across [0, dur]
        steps = 1000
        prev_pts = -1.0
        for i in range(steps + 1):
            t = (i / steps) * dur
            pts, speed = evaluate_pts_easing(t, dur, v_fast, has_ramp_in=True, has_ramp_out=True, ramp_duration_s=1.0)
            assert pts > prev_pts or (i == 0 and pts == 0.0), f"PTS inversion at t={t}: {pts} <= {prev_pts}"
            assert speed >= 0.99, f"Instantaneous speed dropped below 1.0x at t={t}: {speed}"
            prev_pts = pts

    def test_boundary_speed_matching(self):
        """Verify that instantaneous speed at boundaries matches 1.0x (adjacent dynamic segment speed)."""
        v_fast = 40.0
        dur = 20.0
        # t=0 should have speed=1.0x
        pts0, speed0 = evaluate_pts_easing(0.0, dur, v_fast, has_ramp_in=True, has_ramp_out=True, ramp_duration_s=1.0)
        assert pytest.approx(speed0, rel=1e-3) == 1.0
        assert pytest.approx(pts0, abs=1e-6) == 0.0

        # t=dur should have speed=1.0x
        pts_end, speed_end = evaluate_pts_easing(dur, dur, v_fast, has_ramp_in=True, has_ramp_out=True, ramp_duration_s=1.0)
        assert pytest.approx(speed_end, rel=1e-3) == 1.0

    def test_continuity_across_zone_boundaries(self):
        """Verify that P(t) has C0 and C1 continuity at ramp-in/cruise and cruise/ramp-out boundaries."""
        v_fast = 50.0
        dur = 50.0
        ramp_info = calculate_speed_ramping_curve(dur, v_fast, has_ramp_in=True, has_ramp_out=True, ramp_duration_s=1.0)
        s_in = ramp_info.ramp_in_src_dur
        s_mid = s_in + ramp_info.cruise_src_dur

        # Boundary 1: s_in
        eps = 1e-5
        pts_before, sp_before = evaluate_pts_easing(s_in - eps, dur, v_fast, True, True, 1.0)
        pts_after, sp_after = evaluate_pts_easing(s_in + eps, dur, v_fast, True, True, 1.0)
        assert pytest.approx(pts_before, abs=1e-3) == pts_after
        assert pytest.approx(sp_before, rel=1e-2) == v_fast
        assert pytest.approx(sp_after, rel=1e-2) == v_fast

        # Boundary 2: s_mid
        pts_mid_b, sp_mid_b = evaluate_pts_easing(s_mid - eps, dur, v_fast, True, True, 1.0)
        pts_mid_a, sp_mid_a = evaluate_pts_easing(s_mid + eps, dur, v_fast, True, True, 1.0)
        assert pytest.approx(pts_mid_b, abs=1e-3) == pts_mid_a
        assert pytest.approx(sp_mid_b, rel=1e-2) == v_fast
        assert pytest.approx(sp_mid_a, rel=1e-2) == v_fast

    def test_short_segment_proportional_scaling(self):
        """Verify that static segment shorter than nominal ramp zones scales proportionally without crash or inversion."""
        v_fast = 60.0
        dur = 0.5  # Much shorter than total_needed (2 * 30.5s)
        ramp_info = calculate_speed_ramping_curve(dur, v_fast, has_ramp_in=True, has_ramp_out=True, ramp_duration_s=1.0)
        assert ramp_info.cruise_src_dur == 0.0
        assert ramp_info.ramp_in_src_dur + ramp_info.ramp_out_src_dur == pytest.approx(dur, abs=1e-6)
        assert 0.0 < ramp_info.target_display_dur <= dur

        # Verify monotonicity on short segment
        prev_pts = -1.0
        for i in range(50):
            t = (i / 49.0) * dur
            pts, speed = evaluate_pts_easing(t, dur, v_fast, True, True, 1.0)
            assert pts >= prev_pts
            prev_pts = pts

    def test_asymmetric_ramps(self):
        """Verify ramp-in only (start of video) and ramp-out only (end of video)."""
        v_fast = 30.0
        dur = 40.0
        # Ramp-in only
        r_in = calculate_speed_ramping_curve(dur, v_fast, has_ramp_in=True, has_ramp_out=False, ramp_duration_s=1.0)
        assert r_in.has_ramp_in
        assert not r_in.has_ramp_out
        assert r_in.ramp_out_src_dur == 0.0
        assert "if(lt(T-STARTT" in r_in.pts_expr

        # Ramp-out only
        r_out = calculate_speed_ramping_curve(dur, v_fast, has_ramp_in=False, has_ramp_out=True, ramp_duration_s=1.0)
        assert not r_out.has_ramp_in
        assert r_out.has_ramp_out
        assert r_out.ramp_in_src_dur == 0.0
        assert "if(lt(T-STARTT" in r_out.pts_expr

        # Cruise only (isolated static segment)
        r_none = calculate_speed_ramping_curve(dur, v_fast, has_ramp_in=False, has_ramp_out=False, ramp_duration_s=1.0)
        assert not r_none.has_ramp_in
        assert not r_none.has_ramp_out
        assert r_none.ramp_in_src_dur == 0.0
        assert r_none.ramp_out_src_dur == 0.0
        assert r_none.target_display_dur == pytest.approx(dur / v_fast, abs=1e-4)


# ============================================================================
# 2. Audio Linear Cross-Fading (afade) Tests
# ============================================================================

class TestAudioLinearCrossFading:
    """Tests for audio linear cross-fading filter generation and boundary handling."""

    def test_afade_filter_generation(self):
        """Verify afade=t=in and afade=t=out filters are generated on dynamic segments with audio."""
        segs = [
            TimelineSegment(filepath="clip.mp4", input_index=0, start_in_file=0.0, end_in_file=10.0, state="DYNAMIC", duration=10.0),
        ]
        rows = [{"filepath": "clip.mp4", "has_audio": 1}]
        fc = build_concat_filter(
            timeline=segs,
            rows=rows,
            audio_fade_duration_s=0.15,
        )
        parsed = parse_ffmpeg_filtergraph(fc)
        assert len(parsed["audio_trims"]) == 1
        a_trim = parsed["audio_trims"][0]
        assert "afade=t=in:ss=0:d=0.150" in a_trim
        assert "afade=t=out:st=9.850:d=0.150" in a_trim

    def test_short_audio_fade_clamping(self):
        """Verify fade duration is clamped to dur/2.0 for short dynamic segments."""
        dur = 0.20  # fade_d=0.15 would exceed dur/2.0 (0.10)
        segs = [
            TimelineSegment(filepath="short.mp4", input_index=0, start_in_file=0.0, end_in_file=dur, state="DYNAMIC", duration=dur),
        ]
        rows = [{"filepath": "short.mp4", "has_audio": 1}]
        fc = build_concat_filter(
            timeline=segs,
            rows=rows,
            audio_fade_duration_s=0.15,
        )
        parsed = parse_ffmpeg_filtergraph(fc)
        assert len(parsed["audio_trims"]) == 1
        a_trim = parsed["audio_trims"][0]
        # Should be clamped to 0.100
        assert "afade=t=in:ss=0:d=0.100" in a_trim
        assert "afade=t=out:st=0.100:d=0.100" in a_trim

    def test_zero_fade_duration_disables_afade(self):
        """Verify audio_fade_duration_s=0.0 generates clean atrim without afade."""
        segs = [
            TimelineSegment(filepath="clip.mp4", input_index=0, start_in_file=0.0, end_in_file=10.0, state="DYNAMIC", duration=10.0),
        ]
        rows = [{"filepath": "clip.mp4", "has_audio": 1}]
        fc = build_concat_filter(
            timeline=segs,
            rows=rows,
            audio_fade_duration_s=0.0,
        )
        parsed = parse_ffmpeg_filtergraph(fc)
        assert len(parsed["audio_trims"]) == 1
        assert "afade" not in parsed["audio_trims"][0]

    def test_no_audio_falls_back_to_anullsrc(self):
        """Verify dynamic segments from video without audio stream use anullsrc."""
        segs = [
            TimelineSegment(filepath="mute.mp4", input_index=0, start_in_file=0.0, end_in_file=5.0, state="DYNAMIC", duration=5.0),
        ]
        rows = [{"filepath": "mute.mp4", "has_audio": 0}]
        fc = build_concat_filter(timeline=segs, rows=rows)
        parsed = parse_ffmpeg_filtergraph(fc)
        assert len(parsed["audio_trims"]) == 0
        assert len(parsed["audio_nulls"]) == 1
        assert "anullsrc=r=48000:cl=mono:d=5.000" in parsed["audio_nulls"][0]


# ============================================================================
# 3. Batch Boundary Partitioning & Cross-File Splitting Tests
# ============================================================================

class TestBatchBoundaryPartitioningPreservation:
    """Tests for cross-file segment splitting and batch partition integrity."""

    def test_split_multi_file_merged_segment(self):
        """Verify that a segment spanning 3 files is split into 3 segments strictly at file boundaries."""
        files_info = [
            {"filepath": "f0.mp4", "file_start_offset": 0.0, "file_end_offset": 100.0, "duration": 100.0},
            {"filepath": "f1.mp4", "file_start_offset": 100.0, "file_end_offset": 250.0, "duration": 150.0},
            {"filepath": "f2.mp4", "file_start_offset": 250.0, "file_end_offset": 400.0, "duration": 150.0},
        ]
        # Single long dynamic segment from 50.0s to 350.0s
        merged_seg = [
            Segment(start_time=50.0, end_time=350.0, state="DYNAMIC", source_file="f0.mp4", file_start_offset=0.0)
        ]

        split_res = split_segments_at_file_boundaries(merged_seg, files_info)
        assert len(split_res) == 3

        # Piece 0: in f0 [50.0, 100.0]
        assert split_res[0].source_file == "f0.mp4"
        assert split_res[0].start_time == 50.0
        assert split_res[0].end_time == 100.0
        assert split_res[0].file_start_offset == 0.0

        # Piece 1: in f1 [100.0, 250.0]
        assert split_res[1].source_file == "f1.mp4"
        assert split_res[1].start_time == 100.0
        assert split_res[1].end_time == 250.0
        assert split_res[1].file_start_offset == 100.0

        # Piece 2: in f2 [250.0, 350.0]
        assert split_res[2].source_file == "f2.mp4"
        assert split_res[2].start_time == 250.0
        assert split_res[2].end_time == 350.0
        assert split_res[2].file_start_offset == 250.0

    def test_partition_timeline_by_batches_file_limit(self):
        """Verify that partition_timeline_by_batches limits distinct files per batch to batch_max_files."""
        timeline = []
        for i in range(25):
            fp = f"/surveillance/cam0_file_{i:02d}.mp4"
            timeline.append(TimelineSegment(filepath=fp, input_index=i, start_in_file=0.0, end_in_file=60.0, state="STATIC", duration=60.0))
            timeline.append(TimelineSegment(filepath=fp, input_index=i, start_in_file=60.0, end_in_file=120.0, state="DYNAMIC", duration=60.0))

        batches = partition_timeline_by_batches(timeline, batch_max_files=8)
        assert len(batches) == 4  # 8 + 8 + 8 + 1 = 25 files

        for bi, batch in enumerate(batches):
            files_in_batch = set(s.filepath for s in batch)
            assert len(files_in_batch) <= 8
            if bi < 3:
                assert len(files_in_batch) == 8
            else:
                assert len(files_in_batch) == 1

    def test_build_timeline_cross_file_split_and_offsets(self, isolated_db):
        """Verify build_timeline integrates database tasks, cross-file merge, and boundary splitting."""
        db = isolated_db
        # Insert 3 contiguous tasks
        durations = [120.0, 180.0, 150.0]
        current_offset = 0.0
        for i, dur in enumerate(durations):
            s_ts = f"2026090100{i:02d}00"
            e_ts = f"2026090100{i+1:02d}00"
            fp = f"/nas/surveillance_cam0_{i}.mp4"
            db.add_file_task(
                filepath=fp,
                cam_index=0,
                date="20260901",
                file_start_time=s_ts,
                file_end_time=e_ts,
                file_duration=dur,
            )
            segs = [
                Segment(start_time=current_offset, end_time=current_offset + dur, state="STATIC", source_file=fp, file_start_offset=current_offset)
            ]
            db.set_analysis_result(fp, "ANALYZED", segments_to_json(segs))
            current_offset += dur

        timeline = build_timeline(db, date="20260901", cam_index=0)
        # All 3 files are STATIC -> merged across files -> split back strictly at boundaries -> 3 TimelineSegments
        assert len(timeline) == 3
        for i, seg in enumerate(timeline):
            assert seg.start_in_file == 0.0
            assert seg.end_in_file == pytest.approx(durations[i], abs=1e-4)
            assert seg.duration == pytest.approx(durations[i], abs=1e-4)


# ============================================================================
# 4. Real-World Wall-Clock Timecode OSD & Subtitle Generator Tests
# ============================================================================

class TestRealWorldTimecodeOSDAndSubtitles:
    """Tests for drawtext filter and SRT/ASS subtitle generation with real-world timecodes."""

    def test_build_timecode_drawtext_filter_format(self):
        """Verify drawtext filter string matches FFmpeg localtime syntax."""
        start_unix = 1788220800.0  # 2026-09-01 00:00:00
        filter_str = build_timecode_drawtext_filter(
            start_unix=start_unix,
            font_size=28,
            font_color="yellow",
            x="30",
            y="h-th-30",
            box=True,
            box_color="black@0.6",
            box_border_w=4,
        )
        assert "drawtext=" in filter_str
        assert "localtime\\:1788220800" in filter_str
        assert "fontsize=28" in filter_str
        assert "fontcolor=yellow" in filter_str
        assert "x=30" in filter_str
        assert "y=h-th-30" in filter_str
        assert "box=1" in filter_str
        assert "boxcolor=black@0.6" in filter_str
        assert "boxborderw=4" in filter_str

    def test_generate_timecode_subtitles_srt_format(self):
        """Verify SRT subtitle generation produces valid syntax and monotonic timestamps."""
        timeline = [
            TimelineSegment(filepath="00_20260901080000_20260901081000.mp4", input_index=0, start_in_file=0.0, end_in_file=60.0, state="DYNAMIC", duration=60.0),
            TimelineSegment(filepath="00_20260901080000_20260901081000.mp4", input_index=0, start_in_file=60.0, end_in_file=360.0, state="STATIC", duration=5.0),
        ]
        srt_content = generate_timecode_subtitles(
            timeline=timeline,
            base_date="20260901",
            step_s=1.0,
            format_type="srt",
        )
        assert srt_content != ""
        lines = srt_content.strip().split("\n")
        assert lines[0] == "1"
        assert "-->" in lines[1]
        assert "2026-09-01 08:00:00" in lines[2]

        # Check that subtitle entries match total display duration (60s + 5s = 65s)
        timecode_matches = re.findall(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", srt_content)
        assert len(timecode_matches) == 65
        assert timecode_matches[0][0] == "00:00:00,000"
        assert timecode_matches[-1][1] == "00:01:05,000"

    def test_generate_timecode_subtitles_ass_format(self):
        """Verify ASS subtitle generation includes script headers and dialogue events."""
        timeline = [
            TimelineSegment(filepath="00_20260901120000_20260901121000.mp4", input_index=0, start_in_file=0.0, end_in_file=10.0, state="DYNAMIC", duration=10.0),
        ]
        ass_content = generate_timecode_subtitles(
            timeline=timeline,
            base_date="20260901",
            step_s=2.0,
            format_type="ass",
        )
        assert "[Script Info]" in ass_content
        assert "[V4+ Styles]" in ass_content
        assert "[Events]" in ass_content
        assert "Dialogue: 0,0:00:00.00,0:00:02.00,Default,,0,0,0,,2026-09-01 12:00:00" in ass_content

    def test_save_timecode_subtitles_file_creation(self, tmp_path):
        """Verify save_timecode_subtitles writes file correctly."""
        timeline = [
            TimelineSegment(filepath="00_20260901000007_20260901005727.mp4", input_index=0, start_in_file=0.0, end_in_file=5.0, state="DYNAMIC", duration=5.0),
        ]
        srt_file = tmp_path / "test_timecode.srt"
        res_p = save_timecode_subtitles(timeline, srt_file, base_date="20260901")
        assert res_p.exists()
        assert srt_file.exists()
        content = srt_file.read_text(encoding="utf-8")
        assert "2026-09-01 00:00:07" in content


# ============================================================================
# 5. Filtergraph Integration & Edge Case Tests
# ============================================================================

class TestFiltergraphIntegrationAndSpeedRamping:
    """End-to-end filtergraph integration tests with speed ramping and OSD enabled."""

    def test_speed_ramping_enabled_generates_nonlinear_pts(self):
        """Verify build_concat_filter with speed_ramping=True creates non-linear PTS easing."""
        timeline = [
            TimelineSegment(filepath="f.mp4", input_index=0, start_in_file=0.0, end_in_file=20.0, state="DYNAMIC", duration=20.0),
            TimelineSegment(filepath="f.mp4", input_index=0, start_in_file=20.0, end_in_file=120.0, state="STATIC", duration=1.67),
            TimelineSegment(filepath="f.mp4", input_index=0, start_in_file=120.0, end_in_file=140.0, state="DYNAMIC", duration=20.0),
        ]
        rows = [{"filepath": "f.mp4", "has_audio": 1}]
        fc = build_concat_filter(
            timeline=timeline,
            rows=rows,
            speed_ramping=True,
            ramp_duration_s=1.0,
            audio_fade_duration_s=0.15,
            timecode_osd=True,
            base_date="20260901",
        )
        parsed = parse_ffmpeg_filtergraph(fc)
        assert len(parsed["video_trims"]) == 3
        # Dynamic seg 0
        assert "setpts=PTS-STARTPTS" in parsed["video_trims"][0]
        assert "drawtext=" in parsed["video_trims"][0]
        # Static seg 1 with speed ramping
        assert "if(lt(T-STARTT" in parsed["video_trims"][1]
        assert "drawtext=" in parsed["video_trims"][1]
        # Dynamic seg 2
        assert "setpts=PTS-STARTPTS" in parsed["video_trims"][2]

        # Verify filtergraph label closure
        ok, msg = verify_filtergraph_labels_closure(fc)
        assert ok, msg

    def test_scale_modes_and_hardware_passthrough(self):
        """Verify CUDA and QSV scale filters format correctly."""
        timeline = [
            TimelineSegment(filepath="f.mp4", input_index=0, start_in_file=0.0, end_in_file=10.0, state="DYNAMIC", duration=10.0),
        ]
        rows = [{"filepath": "f.mp4", "has_audio": 0}]

        # CUDA
        fc_cuda = build_concat_filter(timeline, rows, output_width=1280, output_height=720, scale_mode="cuda")
        assert "hwupload_cuda,scale_cuda=1280:720,hwdownload,format=nv12" in fc_cuda

        # CUDA passthrough
        fc_passthrough = build_concat_filter(timeline, rows, output_width=1280, output_height=720, scale_mode="cuda_passthrough")
        assert "scale_cuda=1280:720,hwdownload,format=nv12" in fc_passthrough

        # QSV
        fc_qsv = build_concat_filter(timeline, rows, output_width=1280, output_height=720, scale_mode="qsv")
        assert "scale_qsv=w=1280:h=720,hwdownload,format=nv12" in fc_qsv

        # Skip
        fc_skip = build_concat_filter(timeline, rows, scale_mode="skip")
        assert "null" in fc_skip


# ============================================================================
# 6. Settings Schema & Config Compliance Tests
# ============================================================================

class TestSettingsSchemaCompliance:
    """Verify configuration settings schema for Milestone 4 parameters."""

    def test_settings_schema_milestone4_keys(self):
        """Verify settings.yaml contains all required M4 config parameters with valid values."""
        import yaml
        from src.utils import CONFIG_PATH
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        render_cfg = cfg.get("render", {})
        seg_cfg = cfg.get("segment", {})

        # speed_ramping_enabled
        assert "speed_ramping_enabled" in render_cfg or "speed_ramping_enabled" in seg_cfg
        assert bool(render_cfg.get("speed_ramping_enabled", seg_cfg.get("speed_ramping_enabled"))) is True

        # ramp_duration_s
        assert float(render_cfg.get("ramp_duration_s", seg_cfg.get("ramp_duration_s", 0.0))) >= 0.5

        # audio_fade_duration_s
        assert float(render_cfg.get("audio_fade_duration_s", seg_cfg.get("audio_fade_duration_s", 0.0))) > 0.0

        # batch_max_files
        assert int(render_cfg.get("batch_max_files", 0)) == 8
