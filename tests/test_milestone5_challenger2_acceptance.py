import json
import sqlite3
import subprocess
from pathlib import Path
import pytest
import av

from tests.helpers import validate_perf_json_schema

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_VIDEO_PATH = PROJECT_ROOT / "output" / "DailyVlog_20260901_cam0.mp4"
DB_PATH = PROJECT_ROOT / "data" / "vlog.db"
PERF_LOG_PATH = PROJECT_ROOT / "logs" / "perf_20260901_cam0_20260902_062848.json"
TESTSAMPLE_DIR = PROJECT_ROOT / "testsample"


class TestMilestone5AcceptanceChallenger2:
    """Adversarial Acceptance Suite for Milestone 5 by Challenger 2."""

    def test_output_video_file_exists_and_non_empty(self):
        """Verify output video file exists, is non-zero, and matches condensed size."""
        assert OUTPUT_VIDEO_PATH.exists(), f"Output video does not exist: {OUTPUT_VIDEO_PATH}"
        stat = OUTPUT_VIDEO_PATH.stat()
        assert stat.st_size > 100 * 1024 * 1024, f"Output video file is unexpectedly small: {stat.st_size} bytes"
        assert stat.st_size < 10 * 1024 * 1024 * 1024, f"Output video file is unexpectedly huge: {stat.st_size} bytes"

    def test_output_video_streams_with_pyav(self):
        """Verify video and audio streams using PyAV demuxer."""
        container = av.open(str(OUTPUT_VIDEO_PATH))
        assert len(container.streams.video) >= 1, "Output video missing video stream"
        assert len(container.streams.audio) >= 1, "Output video missing audio stream"

        v_stream = container.streams.video[0]
        a_stream = container.streams.audio[0]

        # Video stream checks
        assert v_stream.codec_context.name in ("hevc", "h265"), f"Unexpected video codec: {v_stream.codec_context.name}"
        assert v_stream.width == 1920, f"Unexpected width: {v_stream.width}"
        assert v_stream.height == 1080, f"Unexpected height: {v_stream.height}"
        assert abs(float(v_stream.average_rate) - 20.0) < 0.05, f"Unexpected video fps: {v_stream.average_rate}"

        # Audio stream checks
        assert a_stream.codec_context.name == "aac", f"Unexpected audio codec: {a_stream.codec_context.name}"
        assert a_stream.codec_context.sample_rate == 48000, f"Unexpected sample rate: {a_stream.codec_context.sample_rate}"
        assert a_stream.codec_context.channels == 1, f"Unexpected channels: {a_stream.codec_context.channels}"

        # Duration checks
        v_dur = float(v_stream.duration * v_stream.time_base) if v_stream.duration else 0.0
        a_dur = float(a_stream.duration * a_stream.time_base) if a_stream.duration else 0.0
        container_dur = float(container.duration) / 1000000.0 if container.duration else 0.0

        assert container_dur > 1800.0, f"Unexpectedly short container duration: {container_dur}s"
        assert v_dur > 1800.0, f"Unexpectedly short video duration: {v_dur}s"
        assert a_dur > 1800.0, f"Unexpectedly short audio duration: {a_dur}s"

        container.close()

    def test_output_video_av_sync_and_pts_continuity(self):
        """Verify AV sync, start time alignment, and stream duration delta."""
        probe_cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "stream=index,codec_type,codec_name,start_time,duration,nb_frames",
            "-show_entries", "format=duration,start_time,bit_rate",
            "-of", "json",
            str(OUTPUT_VIDEO_PATH)
        ]
        res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        probe = json.loads(res.stdout)

        v_stream = next(s for s in probe["streams"] if s["codec_type"] == "video")
        a_stream = next(s for s in probe["streams"] if s["codec_type"] == "audio")

        v_dur = float(v_stream["duration"])
        a_dur = float(a_stream["duration"])
        v_start = float(v_stream.get("start_time", 0.0))
        a_start = float(a_stream.get("start_time", 0.0))

        # Start time sync
        assert abs(v_start - a_start) < 0.1, f"Start time drift between audio and video: v={v_start}, a={a_start}"

        # Total duration sync (AV duration delta must be within 0.5s for a 5.3hr video)
        dur_delta = abs(v_dur - a_dur)
        assert dur_delta < 0.5, f"AV duration discrepancy too large: |{v_dur} - {a_dur}| = {dur_delta:.4f}s"

    def test_packet_level_pts_monotonicity_across_timeline(self):
        """Adversarial check: Seek to multiple points across 5.31 hours and verify packet DTS and decoded frame PTS ordering."""
        container = av.open(str(OUTPUT_VIDEO_PATH))
        v_stream = container.streams.video[0]
        a_stream = container.streams.audio[0]

        # Test checkpoints at 0s, 3600s, 7200s, 10800s, 14400s, 18000s
        checkpoints_sec = [0, 3600, 7200, 10800, 14400, 18000]
        for cp in checkpoints_sec:
            target_pts = int(cp / float(v_stream.time_base))
            container.seek(target_pts, stream=v_stream)

            last_v_dts = None
            last_a_dts = None
            v_count = 0
            a_count = 0

            for packet in container.demux(v_stream, a_stream):
                if packet.dts is None:
                    continue
                if packet.stream == v_stream:
                    if last_v_dts is not None:
                        assert packet.dts >= last_v_dts, f"Non-monotonic video DTS at seek {cp}s: {packet.dts} < {last_v_dts}"
                    last_v_dts = packet.dts
                    v_count += 1
                elif packet.stream == a_stream:
                    if last_a_dts is not None:
                        assert packet.dts >= last_a_dts, f"Non-monotonic audio DTS at seek {cp}s: {packet.dts} < {last_a_dts}"
                    last_a_dts = packet.dts
                    a_count += 1

                if v_count >= 30 and a_count >= 30:
                    break

            assert v_count > 0, f"No video packets found near seek {cp}s"
            assert a_count > 0, f"No audio packets found near seek {cp}s"

        container.close()

    def test_database_task_log_integrity_81_files(self):
        """Verify database task log integrity across all 81 surveillance files."""
        assert DB_PATH.exists(), f"Database does not exist: {DB_PATH}"

        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        # 1. Total file_tasks count must match files in testsample
        sample_files = sorted(list(TESTSAMPLE_DIR.glob("*.mp4")))
        n_expected = len(sample_files)
        assert n_expected >= 80, f"Expected at least 80 files in testsample, found {n_expected}"

        total_tasks = c.execute("SELECT COUNT(*) FROM file_tasks").fetchone()[0]
        assert total_tasks == n_expected, f"Expected {n_expected} file_tasks in DB, got {total_tasks}"

        rows = c.execute("SELECT * FROM file_tasks ORDER BY filepath ASC").fetchall()
        assert len(rows) == n_expected

        prescreen_static_count = 0
        prescreen_suspicious_count = 0

        for row, fpath in zip(rows, sample_files):
            assert Path(row["filepath"]).resolve() == fpath.resolve(), f"DB filepath mismatch: {row['filepath']} vs {fpath}"
            assert row["cam_index"] == 0, f"Unexpected cam_index: {row['cam_index']}"
            assert row["date"] == "20260901", f"Unexpected date: {row['date']}"
            assert row["file_duration"] is not None and row["file_duration"] > 0, f"Invalid duration for {fpath}"
            assert row["prescreen_status"] in ("STATIC", "SUSPICIOUS"), f"Invalid prescreen_status: {row['prescreen_status']}"

            if row["prescreen_status"] == "STATIC":
                prescreen_static_count += 1
                assert row["analysis_status"] == "PENDING", f"Expected PENDING analysis_status for STATIC file {fpath}, got {row['analysis_status']}"
            else:
                prescreen_suspicious_count += 1
                assert row["analysis_status"] == "ANALYZED", f"Expected ANALYZED analysis_status for SUSPICIOUS file {fpath}, got {row['analysis_status']}"

            assert row["retry_count"] == 0, f"Unexpected retry_count: {row['retry_count']}"
            assert row["error_msg"] is None, f"Unexpected error_msg in DB: {row['error_msg']}"

            # Check analysis segments JSON if present
            if row["analysis_segments"]:
                segments = json.loads(row["analysis_segments"])
                assert isinstance(segments, list), f"analysis_segments not a list for {fpath}"
                for seg in segments:
                    assert "start_time" in seg and "end_time" in seg and "state" in seg
                    assert seg["state"] in ("STATIC", "DYNAMIC", "DYNAMIC_AUDIO")
                    assert seg["start_time"] <= seg["end_time"]

        assert prescreen_static_count + prescreen_suspicious_count == n_expected, f"Expected total {n_expected} files, got {prescreen_static_count + prescreen_suspicious_count}"
        assert prescreen_static_count > 0, "Expected at least 1 STATIC file"
        assert prescreen_suspicious_count > 0, "Expected at least 1 SUSPICIOUS file"

        # 3. Check render_tasks table
        render_rows = c.execute("SELECT * FROM render_tasks").fetchall()
        assert len(render_rows) >= 1, "No render_tasks found in DB"
        r_row = render_rows[0]
        assert r_row["date"] == "20260901"
        assert r_row["cam_index"] == 0
        assert r_row["status"] in ("COMPLETED", "DONE", "SUCCESS"), f"Render task status not completed: {r_row['status']}"
        assert r_row["retry_count"] == 0
        assert r_row["error_msg"] is None
        assert Path(r_row["output_file"]).name == "DailyVlog_20260901_cam0.mp4"

        conn.close()

    def test_perf_json_schema_and_metrics_integrity(self):
        """Verify performance metrics log structure and resource bounds."""
        perf_logs = sorted(list((PROJECT_ROOT / "logs").glob("perf_20260901_cam0_*.json")))
        assert len(perf_logs) > 0, "No perf log found"
        latest_perf_log = perf_logs[-1]

        with open(latest_perf_log, "r", encoding="utf-8") as f:
            perf_data = json.load(f)

        valid, msg = validate_perf_json_schema(perf_data)
        assert valid, f"Perf JSON schema invalid: {msg}"

        assert perf_data["date"] == "20260901"
        assert perf_data["cam"] == 0
        assert perf_data["pipeline_duration"] > 0

        records = perf_data.get("records", [])
        assert len(records) > 0, "Expected non-empty records in perf JSON"

        prescreen_recs = [r for r in records if r.get("stage") == "prescreen"]
        analysis_recs = [r for r in records if r.get("stage") == "analysis"]
        render_recs = [r for r in records if r.get("stage") == "render"]

        assert len(prescreen_recs) >= 80, f"Expected at least 80 prescreen records, got {len(prescreen_recs)}"
        assert len(analysis_recs) > 0, "Expected non-zero analysis records"
        assert len(render_recs) > 0, "Expected non-zero render records"

        # Verify monitor summary metrics
        monitors = perf_data.get("monitor_summary", [])
        assert len(monitors) > 0, "No monitor summary in perf JSON"
        for m in monitors:
            assert 0 <= m.get("avg_cpu", 0) <= 100
            assert 0 <= m.get("avg_ram", 0) <= 100
            gpu0 = m.get("gpu0", {})
            assert 0 <= gpu0.get("avg_load", 0) <= 100
            assert gpu0.get("peak_mem_mb", 0) <= 8192, f"VRAM exceeded 8GB: {gpu0.get('peak_mem_mb')}"
