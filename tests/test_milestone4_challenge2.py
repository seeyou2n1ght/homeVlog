"""
Adversarial Challenge & Stress-Test Suite 2 for Milestone 4 (R4: Timeline Multi-File Boundary Cuts & Subtitle/OSD Robustness).

Scope:
1. Adversarially stress test split_segments_at_file_boundaries with synthetic multi-file scenarios across 2-5 file boundaries.
2. Adversarially stress test SRT and ASS timestamp syntax, precision, edge cases, and wall-clock time mapping.
3. Adversarially stress test FFmpeg filter generation (build_concat_filter) across multi-file inputs, scale modes, audio cross-fading, and label closure.
"""

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
from src.utils import ts_to_unix
from tests.helpers import parse_ffmpeg_filtergraph, verify_filtergraph_labels_closure


# ============================================================================
# 1. Multi-File Boundary Cuts Synthetic Scenarios (2 to 5 File Boundaries)
# ============================================================================

class TestMultiFileBoundarySplittingSynthetic:
    """Rigorous stress-testing of split_segments_at_file_boundaries across 2 to 5 file boundaries."""

    def test_two_file_boundary_cut_mid_segment(self):
        """Segment spanning 2 files with cut right in the middle."""
        files_info = [
            {"filepath": "file_0.mp4", "file_start_offset": 0.0, "file_end_offset": 100.0, "duration": 100.0},
            {"filepath": "file_1.mp4", "file_start_offset": 100.0, "file_end_offset": 200.0, "duration": 100.0},
        ]
        # Segment from 60.0 to 140.0 (40s in file 0, 40s in file 1)
        segs = [
            Segment(start_time=60.0, end_time=140.0, state="DYNAMIC", source_file="file_0.mp4", file_start_offset=0.0, max_energy=18.5)
        ]

        result = split_segments_at_file_boundaries(segs, files_info)
        assert len(result) == 2

        # Piece 0 in file 0: [60.0, 100.0]
        assert result[0].source_file == "file_0.mp4"
        assert result[0].start_time == 60.0
        assert result[0].end_time == 100.0
        assert result[0].file_start_offset == 0.0
        assert result[0].state == "DYNAMIC"
        assert result[0].max_energy == 18.5

        # Piece 1 in file 1: [100.0, 140.0]
        assert result[1].source_file == "file_1.mp4"
        assert result[1].start_time == 100.0
        assert result[1].end_time == 140.0
        assert result[1].file_start_offset == 100.0
        assert result[1].state == "DYNAMIC"
        assert result[1].max_energy == 18.5

        # Total duration conserved
        total_dur = sum(s.end_time - s.start_time for s in result)
        assert total_dur == pytest.approx(80.0, abs=1e-6)

    def test_three_file_boundary_cut_middle_swallowed(self):
        """Segment spanning 3 files completely swallowing the middle file."""
        files_info = [
            {"filepath": "cam_f0.mp4", "file_start_offset": 0.0, "file_end_offset": 60.0, "duration": 60.0},
            {"filepath": "cam_f1.mp4", "file_start_offset": 60.0, "file_end_offset": 120.0, "duration": 60.0},
            {"filepath": "cam_f2.mp4", "file_start_offset": 120.0, "file_end_offset": 180.0, "duration": 60.0},
        ]
        # Single long static segment [20.0, 160.0]
        segs = [
            Segment(start_time=20.0, end_time=160.0, state="STATIC", source_file="cam_f0.mp4", file_start_offset=0.0)
        ]

        result = split_segments_at_file_boundaries(segs, files_info)
        assert len(result) == 3

        # Piece 0: [20, 60] (40s)
        assert result[0].source_file == "cam_f0.mp4"
        assert result[0].start_time == 20.0
        assert result[0].end_time == 60.0
        assert result[0].file_start_offset == 0.0

        # Piece 1: [60, 120] (60s)
        assert result[1].source_file == "cam_f1.mp4"
        assert result[1].start_time == 60.0
        assert result[1].end_time == 120.0
        assert result[1].file_start_offset == 60.0

        # Piece 2: [120, 160] (40s)
        assert result[2].source_file == "cam_f2.mp4"
        assert result[2].start_time == 120.0
        assert result[2].end_time == 160.0
        assert result[2].file_start_offset == 120.0

        total_dur = sum(s.end_time - s.start_time for s in result)
        assert total_dur == pytest.approx(140.0, abs=1e-6)

    def test_four_file_boundary_cut_alternating_states(self):
        """Alternating DYNAMIC and STATIC segments cutting across 4 file boundaries."""
        files_info = [
            {"filepath": "part1.mp4", "file_start_offset": 0.0, "file_end_offset": 100.0, "duration": 100.0},
            {"filepath": "part2.mp4", "file_start_offset": 100.0, "file_end_offset": 200.0, "duration": 100.0},
            {"filepath": "part3.mp4", "file_start_offset": 200.0, "file_end_offset": 300.0, "duration": 100.0},
            {"filepath": "part4.mp4", "file_start_offset": 300.0, "file_end_offset": 400.0, "duration": 100.0},
        ]
        # Segments:
        # Seg 1: DYNAMIC [50, 150] (spans part1 and part2)
        # Seg 2: STATIC [150, 250] (spans part2 and part3)
        # Seg 3: DYNAMIC_AUDIO [250, 350] (spans part3 and part4)
        segs = [
            Segment(start_time=50.0, end_time=150.0, state="DYNAMIC", source_file="part1.mp4", file_start_offset=0.0),
            Segment(start_time=150.0, end_time=250.0, state="STATIC", source_file="part2.mp4", file_start_offset=100.0),
            Segment(start_time=250.0, end_time=350.0, state="DYNAMIC_AUDIO", source_file="part3.mp4", file_start_offset=200.0, max_energy=30.0),
        ]

        result = split_segments_at_file_boundaries(segs, files_info)
        assert len(result) == 6

        # Seg 1 split:
        assert result[0].source_file == "part1.mp4" and result[0].start_time == 50.0 and result[0].end_time == 100.0 and result[0].state == "DYNAMIC"
        assert result[1].source_file == "part2.mp4" and result[1].start_time == 100.0 and result[1].end_time == 150.0 and result[1].state == "DYNAMIC"

        # Seg 2 split:
        assert result[2].source_file == "part2.mp4" and result[2].start_time == 150.0 and result[2].end_time == 200.0 and result[2].state == "STATIC"
        assert result[3].source_file == "part3.mp4" and result[3].start_time == 200.0 and result[3].end_time == 250.0 and result[3].state == "STATIC"

        # Seg 3 split:
        assert result[4].source_file == "part3.mp4" and result[4].start_time == 250.0 and result[4].end_time == 300.0 and result[4].state == "DYNAMIC_AUDIO"
        assert result[5].source_file == "part4.mp4" and result[5].start_time == 300.0 and result[5].end_time == 350.0 and result[5].state == "DYNAMIC_AUDIO"

        # Verify all durations conserved
        assert sum(s.end_time - s.start_time for s in result) == pytest.approx(300.0, abs=1e-6)

    def test_five_file_boundary_cut_full_day_span(self):
        """Giant segment spanning 5 consecutive physical files."""
        n_files = 5
        dur_per_file = 1800.0  # 30 mins each
        files_info = []
        for i in range(n_files):
            files_info.append({
                "filepath": f"surveillance_{i:02d}.mp4",
                "file_start_offset": i * dur_per_file,
                "file_end_offset": (i + 1) * dur_per_file,
                "duration": dur_per_file,
            })

        # Giant segment from 900s (middle of file 0) to 8100s (middle of file 4)
        segs = [
            Segment(start_time=900.0, end_time=8100.0, state="DYNAMIC", source_file="surveillance_00.mp4", file_start_offset=0.0, max_energy=42.0)
        ]

        result = split_segments_at_file_boundaries(segs, files_info)
        assert len(result) == 5

        for i, piece in enumerate(result):
            assert piece.source_file == f"surveillance_{i:02d}.mp4"
            assert piece.file_start_offset == i * dur_per_file
            assert piece.state == "DYNAMIC"
            assert piece.max_energy == 42.0
            if i == 0:
                assert piece.start_time == 900.0 and piece.end_time == 1800.0
            elif i == 4:
                assert piece.start_time == 7200.0 and piece.end_time == 8100.0
            else:
                assert piece.start_time == i * dur_per_file and piece.end_time == (i + 1) * dur_per_file

        assert sum(s.end_time - s.start_time for s in result) == pytest.approx(8100.0 - 900.0, abs=1e-6)

    def test_files_info_various_input_data_structures(self):
        """Verify split_segments_at_file_boundaries supports dict, dict-of-dicts, list-of-dicts, list-of-tuples."""
        seg = [Segment(start_time=50.0, end_time=150.0, state="DYNAMIC", source_file="f1.mp4", file_start_offset=0.0)]

        # 1. Dict of tuples: {filepath: (start, end)}
        d_tuples = {"f1.mp4": (0.0, 100.0), "f2.mp4": (100.0, 200.0)}
        res1 = split_segments_at_file_boundaries(seg, d_tuples)
        assert len(res1) == 2
        assert res1[0].source_file == "f1.mp4" and res1[1].source_file == "f2.mp4"

        # 2. Dict of dicts: {filepath: {file_start_offset, file_end_offset}}
        d_dicts = {
            "f1.mp4": {"file_start_offset": 0.0, "file_end_offset": 100.0},
            "f2.mp4": {"file_start_offset": 100.0, "file_end_offset": 200.0},
        }
        res2 = split_segments_at_file_boundaries(seg, d_dicts)
        assert len(res2) == 2

        # 3. List of tuples: [(filepath, start, end)]
        l_tuples = [("f1.mp4", 0.0, 100.0), ("f2.mp4", 100.0, 200.0)]
        res3 = split_segments_at_file_boundaries(seg, l_tuples)
        assert len(res3) == 2

        # 4. List of dicts: [{filepath, start_offset, duration}]
        l_dicts = [
            {"filepath": "f1.mp4", "start_offset": 0.0, "duration": 100.0},
            {"filepath": "f2.mp4", "start_offset": 100.0, "duration": 100.0},
        ]
        res4 = split_segments_at_file_boundaries(seg, l_dicts)
        assert len(res4) == 2

    def test_unsorted_files_info_handled_gracefully(self):
        """Verify out-of-order files_info is correctly sorted and segments are split chronologically."""
        unsorted_files = [
            {"filepath": "f3.mp4", "file_start_offset": 200.0, "file_end_offset": 300.0, "duration": 100.0},
            {"filepath": "f1.mp4", "file_start_offset": 0.0, "file_end_offset": 100.0, "duration": 100.0},
            {"filepath": "f2.mp4", "file_start_offset": 100.0, "file_end_offset": 200.0, "duration": 100.0},
        ]
        seg = [Segment(start_time=50.0, end_time=250.0, state="DYNAMIC", source_file="f1.mp4", file_start_offset=0.0)]
        res = split_segments_at_file_boundaries(seg, unsorted_files)
        assert len(res) == 3
        assert res[0].source_file == "f1.mp4"
        assert res[1].source_file == "f2.mp4"
        assert res[2].source_file == "f3.mp4"

    def test_discontinuous_gaps_between_files(self):
        """Verify that segments covering gaps between files only produce pieces within physical files."""
        files_info = [
            {"filepath": "f1.mp4", "file_start_offset": 0.0, "file_end_offset": 50.0, "duration": 50.0},
            {"filepath": "f2.mp4", "file_start_offset": 70.0, "file_end_offset": 120.0, "duration": 50.0},  # 20s gap [50, 70]
        ]
        seg = [Segment(start_time=20.0, end_time=100.0, state="DYNAMIC", source_file="f1.mp4", file_start_offset=0.0)]
        res = split_segments_at_file_boundaries(seg, files_info)
        assert len(res) == 2
        assert res[0].start_time == 20.0 and res[0].end_time == 50.0 and res[0].source_file == "f1.mp4"
        assert res[1].start_time == 70.0 and res[1].end_time == 100.0 and res[1].source_file == "f2.mp4"

    def test_empty_or_none_inputs_safety(self):
        """Verify empty segment list or None files_info returns safely without exception."""
        assert split_segments_at_file_boundaries([], [{"filepath": "f.mp4", "file_start_offset": 0.0, "file_end_offset": 10.0}]) == []
        segs = [Segment(start_time=0.0, end_time=5.0, state="DYNAMIC", source_file="f.mp4", file_start_offset=0.0)]
        assert split_segments_at_file_boundaries(segs, None) == segs
        assert split_segments_at_file_boundaries(segs, []) == segs


# ============================================================================
# 2. SRT & ASS Timestamp Syntax & Subtitle Robustness
# ============================================================================

class TestSRTAndASSTimestampSyntaxRobustness:
    """Stress tests for SRT and ASS timestamp syntax, precision, edge cases, and wall-clock mapping."""

    @pytest.mark.parametrize("seconds,expected_srt,expected_ass", [
        (0.0, "00:00:00,000", "0:00:00.00"),
        (0.001, "00:00:00,001", "0:00:00.00"),
        (0.010, "00:00:00,010", "0:00:00.01"),
        (0.999, "00:00:00,999", "0:00:01.00"),
        (1.0, "00:00:01,000", "0:00:01.00"),
        (59.999, "00:00:59,999", "0:01:00.00"),
        (60.0, "00:01:00,000", "0:01:00.00"),
        (3599.999, "00:59:59,999", "1:00:00.00"),
        (3600.0, "01:00:00,000", "1:00:00.00"),
        (86399.999, "23:59:59,999", "24:00:00.00"),
        (86400.0, "24:00:00,000", "24:00:00.00"),
        (100000.555, "27:46:40,555", "27:46:40.56"),
    ])
    def test_timestamp_formatting_values(self, seconds: float, expected_srt: str, expected_ass: str):
        """Verify SRT and ASS formatting at critical boundaries and rollover points."""
        assert _format_srt_timestamp(seconds) == expected_srt
        assert _format_ass_timestamp(seconds) == expected_ass

    def test_srt_regex_strict_conformance(self):
        """Verify SRT timestamp syntax strictly matches standard regex `HH:MM:SS,mmm`."""
        srt_pattern = re.compile(r"^\d{2,}:\d{2}:\d{2},\d{3}$")
        for sec in [0.0, 0.123, 59.456, 3600.789, 72000.100]:
            formatted = _format_srt_timestamp(sec)
            assert srt_pattern.match(formatted), f"SRT timestamp '{formatted}' failed regex match"

    def test_ass_regex_strict_conformance(self):
        """Verify ASS timestamp syntax strictly matches standard regex `H:MM:SS.cc`."""
        ass_pattern = re.compile(r"^\d+:\d{2}:\d{2}\.\d{2}$")
        for sec in [0.0, 0.12, 59.45, 3600.78, 72000.10]:
            formatted = _format_ass_timestamp(sec)
            assert ass_pattern.match(formatted), f"ASS timestamp '{formatted}' failed regex match"

    def test_generate_timecode_subtitles_srt_full_structure(self, tmp_path: Path):
        """Verify SRT generation produces contiguous, non-overlapping cues matching presentation duration."""
        timeline = [
            TimelineSegment(filepath="00_20260901080000_20260901081000.mp4", input_index=0, start_in_file=0.0, end_in_file=10.0, state="DYNAMIC", duration=10.0),
            TimelineSegment(filepath="00_20260901080000_20260901081000.mp4", input_index=0, start_in_file=10.0, end_in_file=70.0, state="STATIC", duration=2.0),
        ]
        rows = [{"filepath": "00_20260901080000_20260901081000.mp4", "file_start_time": "20260901080000"}]
        srt_content = generate_timecode_subtitles(timeline=timeline, rows=rows, base_date="20260901", step_s=1.0, format_type="srt")
        
        # Verify cue entries
        blocks = [b.strip() for b in srt_content.strip().split("\n\n") if b.strip()]
        assert len(blocks) == 12  # 10s dynamic + 2s static = 12 1-second steps

        for i, block in enumerate(blocks):
            lines = block.split("\n")
            assert len(lines) == 3
            cue_num = int(lines[0])
            assert cue_num == i + 1
            m = re.match(r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})", lines[1])
            assert m is not None
            # Content should be date string
            assert lines[2].startswith("2026-09-01 08:")

        # Test save_timecode_subtitles
        out_file = tmp_path / "vlog.srt"
        res_p = save_timecode_subtitles(timeline=timeline, output_path=out_file, rows=rows, base_date="20260901")
        assert res_p.exists()
        assert res_p.stat().st_size > 0

    def test_generate_timecode_subtitles_ass_full_structure(self, tmp_path: Path):
        """Verify ASS generation produces valid Script Info, Styles, and Events headers."""
        timeline = [
            TimelineSegment(filepath="00_20260901100000_20260901101000.mp4", input_index=0, start_in_file=0.0, end_in_file=5.0, state="DYNAMIC", duration=5.0),
        ]
        ass_content = generate_timecode_subtitles(timeline=timeline, base_date="20260901", step_s=1.0, format_type="ass")
        assert "[Script Info]" in ass_content
        assert "ScriptType: v4.00+" in ass_content
        assert "[V4+ Styles]" in ass_content
        assert "Style: Default" in ass_content
        assert "[Events]" in ass_content
        assert "Dialogue: 0," in ass_content
        assert "2026-09-01 10:00:00" in ass_content

        out_ass = tmp_path / "vlog.ass"
        res_p = save_timecode_subtitles(timeline=timeline, output_path=out_ass, format_type="ass")
        assert res_p.exists()
        assert res_p.read_text(encoding="utf-8").startswith("[Script Info]")

    def test_wall_clock_speed_progression(self):
        """Verify wall-clock time in subtitles advances at accelerated rate during static segments."""
        # 60s source file compressed into 2.0s display duration (30x speed)
        timeline = [
            TimelineSegment(filepath="00_20260901000000_20260901000100.mp4", input_index=0, start_in_file=0.0, end_in_file=60.0, state="STATIC", duration=2.0),
        ]
        srt_content = generate_timecode_subtitles(timeline=timeline, base_date="20260901", step_s=1.0, format_type="srt")
        blocks = [b.strip() for b in srt_content.strip().split("\n\n") if b.strip()]
        assert len(blocks) == 2

        # First second: 00:00:00 -> wall clock 00:00:00
        assert "2026-09-01 00:00:00" in blocks[0]
        # Second second: 00:00:01 -> wall clock should have jumped by 30s -> 00:00:30
        assert "2026-09-01 00:00:30" in blocks[1]


# ============================================================================
# 3. FFmpeg Filter Generation Across Multi-File Topologies
# ============================================================================

class TestFFmpegFilterGenerationStress:
    """Stress testing FFmpeg complex filtergraph generation across multi-file inputs, scale modes, and audio cross-fading."""

    def test_multi_file_filtergraph_label_closure_5_files(self):
        """Verify filtergraph label closure across 5 distinct files with multiple segments each."""
        timeline = []
        rows = []
        for f_idx in range(5):
            fn = f"surveillance_cam0_part{f_idx}.mp4"
            rows.append({"filepath": fn, "has_audio": 1, "file_start_time": f"2026090100{f_idx:02d}00"})
            # 2 segments per file: 1 static, 1 dynamic
            timeline.append(TimelineSegment(filepath=fn, input_index=f_idx, start_in_file=0.0, end_in_file=20.0, state="STATIC", duration=1.0))
            timeline.append(TimelineSegment(filepath=fn, input_index=f_idx, start_in_file=20.0, end_in_file=40.0, state="DYNAMIC", duration=20.0))

        fc = build_concat_filter(
            timeline=timeline,
            rows=rows,
            speed_ramping=True,
            audio_fade_duration_s=0.15,
            timecode_osd=True,
            base_date="20260901",
        )

        ok, msg = verify_filtergraph_labels_closure(fc)
        assert ok, f"Label closure failed: {msg}"

        parsed = parse_ffmpeg_filtergraph(fc)
        assert parsed["concat_n"] == 10  # 5 files * 2 segments = 10 segments
        assert len(parsed["scales"]) == 5  # 1 scale per file
        assert len(parsed["splits"]) == 5  # 1 split per file (since each file has 2 segs)
        assert len(parsed["video_trims"]) == 10
        assert len(parsed["audio_trims"]) == 5  # 5 dynamic segments with audio
        assert len(parsed["audio_nulls"]) == 5  # 5 static segments with null audio

    @pytest.mark.parametrize("scale_mode,expected_token", [
        ("cpu", "scale=1920:1080"),
        ("cuda", "hwupload_cuda,scale_cuda=1920:1080,hwdownload,format=nv12"),
        ("cuda_passthrough", "scale_cuda=1920:1080,hwdownload,format=nv12"),
        ("qsv", "scale_qsv=w=1920:h=1080,hwdownload,format=nv12"),
        ("skip", "null"),
    ])
    def test_all_hardware_scale_modes(self, scale_mode: str, expected_token: str):
        """Verify scale filter generation across CPU, CUDA, CUDA-passthrough, QSV, and Skip modes."""
        timeline = [
            TimelineSegment(filepath="cam.mp4", input_index=0, start_in_file=0.0, end_in_file=10.0, state="DYNAMIC", duration=10.0)
        ]
        rows = [{"filepath": "cam.mp4", "has_audio": 0}]
        fc = build_concat_filter(timeline=timeline, rows=rows, scale_mode=scale_mode)
        assert expected_token in fc
        ok, msg = verify_filtergraph_labels_closure(fc)
        assert ok, msg

    def test_audio_fade_clamping_boundary_values(self):
        """Verify audio fade clamping under extreme durations."""
        # Case 1: Ultra short segment 0.04s, fade_duration 0.15s -> clamped to dur/2 = 0.020s
        segs = [
            TimelineSegment(filepath="clip.mp4", input_index=0, start_in_file=0.0, end_in_file=0.04, state="DYNAMIC", duration=0.04)
        ]
        rows = [{"filepath": "clip.mp4", "has_audio": 1}]
        fc = build_concat_filter(timeline=segs, rows=rows, audio_fade_duration_s=0.15)
        parsed = parse_ffmpeg_filtergraph(fc)
        assert len(parsed["audio_trims"]) == 1
        assert "afade=t=in:ss=0:d=0.020" in parsed["audio_trims"][0]
        assert "afade=t=out:st=0.020:d=0.020" in parsed["audio_trims"][0]

        # Case 2: Zero fade duration -> afade omitted
        fc_nofade = build_concat_filter(timeline=segs, rows=rows, audio_fade_duration_s=0.0)
        parsed_nofade = parse_ffmpeg_filtergraph(fc_nofade)
        assert "afade" not in parsed_nofade["audio_trims"][0]

    def test_timecode_drawtext_osd_filter_escaping(self):
        """Verify drawtext filter formatting, colon escaping, and text template."""
        filter_str = build_timecode_drawtext_filter(
            start_unix=1788220800.0,
            font_size=24,
            font_color="white",
            x="w-tw-30",
            y="30",
            box=True,
            box_color="black@0.5",
            box_border_w=5,
        )
        assert "drawtext=text='%{pts\\:localtime\\:1788220800\\:%Y-%m-%d %H\\\\\\:%M\\\\\\:%S}'" in filter_str
        assert "x=w-tw-30:y=30:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.5:boxborderw=5" in filter_str


# ============================================================================
# 4. Batch Partitioning Preservation
# ============================================================================

class TestBatchPartitioningPreservation:
    """Stress tests on partition_timeline_by_batches."""

    def test_partition_timeline_by_batches_exact_boundaries(self):
        """Verify that batch partitioning never exceeds batch_max_files distinct files per batch."""
        timeline = []
        for i in range(19):
            fn = f"file_{i:02d}.mp4"
            timeline.append(TimelineSegment(filepath=fn, input_index=i, start_in_file=0.0, end_in_file=10.0, state="STATIC", duration=10.0))
            timeline.append(TimelineSegment(filepath=fn, input_index=i, start_in_file=10.0, end_in_file=20.0, state="DYNAMIC", duration=10.0))

        # 19 files with max 8 per batch -> 3 batches (8, 8, 3)
        batches = partition_timeline_by_batches(timeline, batch_max_files=8)
        assert len(batches) == 3

        batch0_files = set(s.filepath for s in batches[0])
        batch1_files = set(s.filepath for s in batches[1])
        batch2_files = set(s.filepath for s in batches[2])

        assert len(batch0_files) == 8
        assert len(batch1_files) == 8
        assert len(batch2_files) == 3

        # Disjoint file sets across batches
        assert batch0_files.isdisjoint(batch1_files)
        assert batch1_files.isdisjoint(batch2_files)
        assert batch0_files.isdisjoint(batch2_files)

        # Total segments preserved
        total_segs = sum(len(b) for b in batches)
        assert total_segs == 38
