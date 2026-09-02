"""
Tier 4: Real-World Workload Acceptance & Perf Log Schema Test Suite for HomeVlog.
Covers:
- testsample/ 84-clip dataset structural integrity, timestamp continuity & natural gaps
- Scanner ingestion of 84 real surveillance clips into SQLite WAL database
- PerfCollector and Monitor metrics generation and strict JSON schema validation
- FFmpegProcessRegistry lifecycle, process tracking & deep resource GC cleanup
"""

import json
import os
import re
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.database import VlogDatabase
from src.monitor import Monitor, PerfCollector, PerfRecord, StageStats
from src.renderer import FFmpegProcessRegistry
from src.scanner import FILENAME_RE, parse_filename, scan_directory, get_date_cam_groups
from src.utils import cleanup_resources, ts_to_unix
from tests.helpers import validate_perf_json_schema


class TestTestsampleRealWorldDataset:
    """Acceptance tests for the real 84-clip surveillance dataset in testsample/."""

    @pytest.fixture
    def testsample_files(self):
        root = Path(__file__).resolve().parent.parent
        testsample_dir = root / "testsample"
        if not testsample_dir.exists():
            pytest.skip("testsample/ directory not present in workspace")
        files = sorted(list(testsample_dir.glob("*.mp4")))
        return files

    def test_testsample_has_84_mp4_files(self, testsample_files):
        assert len(testsample_files) == 84, f"Expected 84 clips in testsample/, found {len(testsample_files)}"

    def test_testsample_naming_and_metadata_parsing(self, testsample_files):
        for f in testsample_files:
            info = parse_filename(f.name)
            assert info is not None, f"File {f.name} failed filename regex parse"
            assert info["cam_index"] == 0, f"Expected cam 0, got {info['cam_index']} for {f.name}"
            assert info["date"] == "20260901", f"Expected date 20260901, got {info['date']} for {f.name}"
            assert info["end_ts"] > info["start_ts"], f"Invalid duration for {f.name}"

    def test_testsample_timestamp_ordering_and_2_gaps(self, testsample_files):
        parsed = [parse_filename(f.name) for f in testsample_files]

        gaps = []
        for i in range(len(parsed) - 1):
            cur = parsed[i]
            nxt = parsed[i + 1]
            gap = nxt["start_ts"] - cur["end_ts"]
            # Check timestamps are non-decreasing
            assert nxt["start_ts"] >= cur["start_ts"], f"Timestamp ordering violation between {testsample_files[i].name} and {testsample_files[i+1].name}"
            if gap > 1.0:
                gaps.append((i, gap, testsample_files[i].name, testsample_files[i+1].name))

        # Dataset specification confirms exactly 2 natural gaps (> 1.0s)
        assert len(gaps) == 2, f"Expected exactly 2 natural gaps, found {len(gaps)}: {gaps}"

        # Gap 1: ~07:14 (around index 8-9)
        assert gaps[0][0] == 8
        assert 1.5 <= gaps[0][1] <= 2.5

        # Gap 2: ~15:44 (around index 54-55)
        assert gaps[1][0] == 54
        assert 1.5 <= gaps[1][1] <= 2.5

    def test_testsample_scanner_ingestion_into_database(self, testsample_files, isolated_db, mock_config):
        root = Path(__file__).resolve().parent.parent
        testsample_dir = str(root / "testsample")
        mock_config["paths"]["input_dir"] = testsample_dir

        with patch("src.scanner.load_config", return_value=mock_config):
            result = scan_directory(isolated_db, input_dir=testsample_dir)
            assert result.added == 84
            assert result.skipped == 0
            assert result.frozen_pending == 0

            groups = get_date_cam_groups(isolated_db)
            assert len(groups) == 1
            assert groups[0] == ("20260901", 0)

            all_tasks = isolated_db.get_all_file_tasks_for_date("20260901", 0)
            assert len(all_tasks) == 84
            for task in all_tasks:
                assert task["prescreen_status"] in ("PENDING", "STATIC")


class TestPerfLogJsonSchemaAndMetrics:
    """Acceptance tests for structured performance log format and schema conformance."""

    def test_perf_collector_metrics_summary(self):
        collector = PerfCollector()
        collector.add(PerfRecord(stage="prescreen", file="f1.mp4", gpu="qsv", duration=0.35))
        collector.add(PerfRecord(stage="prescreen", file="f2.mp4", gpu="qsv", duration=0.45))
        collector.add(PerfRecord(stage="analysis", file="f1.mp4", gpu="cuda", duration=4.8))
        collector.add(PerfRecord(stage="render", file="batch0.mp4", gpu="nv", duration=25.0))

        summary = collector.summary_by_stage()
        assert "prescreen" in summary
        assert summary["prescreen"]["count"] == 2
        assert summary["prescreen"]["avg"] == 0.40
        assert summary["analysis"]["count"] == 1
        assert summary["render"]["count"] == 1

    def test_perf_log_json_dump_and_schema_validation(self, temp_test_dir):
        collector = PerfCollector()
        for i in range(5):
            collector.add(PerfRecord(
                stage="prescreen",
                file=f"00_20260901000{i}00_20260901000{i+1}00.mp4",
                gpu="qsv",
                duration=0.3 + 0.05 * i,
                extra={"status": "SUSPICIOUS"},
            ))

        monitor = Monitor()
        with monitor.stage("pipeline_20260901_cam0"):
            time.sleep(0.05)

        perf_file = temp_test_dir / "logs" / "perf_20260901_cam0_test.json"
        metadata = {
            "date": "20260901",
            "cam": 0,
            "pipeline_duration": 12.5,
            "monitor_summary": monitor.stages_data(),
            "perf_summary": collector.summary_by_stage(),
        }
        collector.dump(perf_file, metadata=metadata)
        assert perf_file.exists()

        with open(perf_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        valid, msg = validate_perf_json_schema(data)
        assert valid, f"Perf log JSON schema violation: {msg}"


class TestResourceRegistryAndCleanup:
    """Tests for process tracking, deep GC, and resource cleanup safety."""

    def test_ffmpeg_process_registry_lifecycle(self):
        mock_proc1 = MagicMock()
        mock_proc2 = MagicMock()

        FFmpegProcessRegistry.register("batch_0", mock_proc1)
        FFmpegProcessRegistry.register("batch_1", mock_proc2)

        # Deregister batch 0
        FFmpegProcessRegistry.deregister("batch_0")

        # kill_all should kill remaining processes (batch 1)
        FFmpegProcessRegistry.kill_all()
        mock_proc2.kill.assert_called_once()
        mock_proc1.kill.assert_not_called()

    def test_cleanup_resources_safety(self):
        # Verify cleanup_resources runs without unhandled exceptions
        cleanup_resources()
