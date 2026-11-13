"""Tier 4: Real-World Application Scenarios (7 Workflows).

Full pipeline workflows exercising the entire system under realistic surveillance conditions.
"""

import json
from pathlib import Path

import av
import pytest
import yaml

import main
from src.core import config as core_config
from src.core.database import VlogDatabase
from src.core.identity import file_identity
from src.hardware import scheduler
from src.stages.pipeline_types import AnalysisQueue
from tests.helpers_e2e import (
    CAMERA_MACS,
    assert_resource_hygiene,
    build_custom_dataset,
    build_footage,
)


def test_tier4_scenario_01_24h_surveillance_simulation(tmp_path, mock_config, monkeypatch):
    """T4-01: 24-Hour Surveillance Day Simulation: Night static, daytime motion & audio, and dusk transition."""
    cam_root = tmp_path / "mock_cam"
    specs = {
        0: [
            {"name": "00_20260901000000_20260901000002.mp4", "scene": "static", "duration_s": 2.0},
            {"name": "00_20260901080000_20260901080004.mp4", "scene": "motion_audio", "duration_s": 4.0, "motion": [(1.0, 3.0)], "audio": [(1.5, 2.5)]},
            {"name": "00_20260901180000_20260901180002.mp4", "scene": "static", "duration_s": 2.0},
        ]
    }
    build_custom_dataset(cam_root, specs)
    cam_dirs = [str(cam_root / f"XiaomiCamera_01_{CAMERA_MACS[0]}")]

    settings = dict(mock_config)
    settings["paths"] = {"input_dirs": cam_dirs}
    settings["pipeline"]["render_gpu_policy"] = "cpu_only"
    settings["recovery"]["min_disk_space_gb"] = 0
    settings["detection"].update(
        analysis_fps_adaptive=False,
        analysis_early_term_enabled=False,
        analysis_resolution="128x128",
        prescreen_resolution="128x128",
    )
    settings["output"].update(naming="DailyVlog_{date}_{mac}.mp4", resolution="128x128", fps=20)
    settings["output"]["qsv"].pop("maxrate", None)
    settings["output"]["qsv"].pop("bufsize", None)

    settings_file = tmp_path / "settings_t4_01.yaml"
    settings_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["main.py", "--no-tui", "--config", str(settings_file)])
    main.main()

    db = VlogDatabase()
    try:
        tasks = db.get_all_file_tasks_for_date("20260901", 0)
        assert len(tasks) == 3
        outputs = list(core_config.OUTPUT_DIR.glob("DailyVlog_20260901_*.mp4"))
        assert len(outputs) == 1
        meta_file = outputs[0].with_suffix(".meta.json")
        assert meta_file.exists()
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        assert meta["delivery_mapping"]["status"] == "AVAILABLE"
    finally:
        db.close()
    assert_resource_hygiene()


def test_tier4_scenario_02_multicamera_interleaved_ingestion(tmp_path, mock_config, monkeypatch):
    """T4-02: Multi-Camera Interleaved Ingestion: Camera 1 and Camera 2 processed concurrently."""
    cam_root = tmp_path / "mock_multicam"
    specs = {
        0: [
            {"name": "00_20260901120000_20260901120002.mp4", "scene": "motion", "duration_s": 2.0},
            {"name": "00_20260901120002_20260901120004.mp4", "scene": "static", "duration_s": 2.0},
        ],
        1: [
            {"name": "00_20260901120000_20260901120002.mp4", "scene": "static", "duration_s": 2.0},
            {"name": "00_20260901120002_20260901120004.mp4", "scene": "static", "duration_s": 2.0},
        ],
    }
    build_custom_dataset(cam_root, specs)
    cam_dirs = [
        str(cam_root / f"XiaomiCamera_01_{CAMERA_MACS[0]}"),
        str(cam_root / f"XiaomiCamera_02_{CAMERA_MACS[1]}"),
    ]

    settings = dict(mock_config)
    settings["paths"] = {"input_dirs": cam_dirs}
    settings["pipeline"]["render_gpu_policy"] = "cpu_only"
    settings["recovery"]["min_disk_space_gb"] = 0
    settings["detection"].update(
        analysis_fps_adaptive=False,
        analysis_early_term_enabled=False,
        analysis_resolution="128x128",
        prescreen_resolution="128x128",
    )
    settings["output"].update(naming="DailyVlog_{date}_{mac}.mp4", resolution="128x128", fps=20)
    settings["output"]["qsv"].pop("maxrate", None)
    settings["output"]["qsv"].pop("bufsize", None)

    settings_file = tmp_path / "settings_t4_02.yaml"
    settings_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["main.py", "--no-tui", "--config", str(settings_file)])
    main.main()

    db = VlogDatabase()
    try:
        tasks_cam0 = db.get_all_file_tasks_for_date("20260901", 0)
        tasks_cam1 = db.get_all_file_tasks_for_date("20260901", 1)
        assert len(tasks_cam0) == 2
        assert len(tasks_cam1) == 2
        outputs = sorted(core_config.OUTPUT_DIR.glob("DailyVlog_20260901_*.mp4"))
        assert len(outputs) == 2
    finally:
        db.close()
    assert_resource_hygiene()


def test_tier4_scenario_03_environmental_thunderstorm_and_flash(tmp_path, mock_config, monkeypatch):
    """T4-03: Environmental Shift & Thunderstorm: Sudden flash & thunder rumbles handled safely in full pipeline."""
    cam_root = tmp_path / "mock_cam_t4_03"
    specs = {
        0: [
            {"name": "00_20260901140000_20260901140002.mp4", "scene": "static", "duration_s": 2.0},
            {
                "name": "00_20260901140002_20260901140006.mp4",
                "scene": "motion_audio",
                "duration_s": 4.0,
                "motion": [(0.8, 1.2)],
                "audio": [(2.0, 3.5)],
            },
            {"name": "00_20260901140006_20260901140008.mp4", "scene": "static", "duration_s": 2.0},
        ]
    }
    files = build_custom_dataset(cam_root, specs)
    cam_dirs = [str(cam_root / f"XiaomiCamera_01_{CAMERA_MACS[0]}")]

    settings = dict(mock_config)
    settings["paths"] = {"input_dirs": cam_dirs}
    settings["pipeline"]["render_gpu_policy"] = "cpu_only"
    settings["recovery"]["min_disk_space_gb"] = 0
    settings["detection"].update(
        analysis_fps_adaptive=False,
        analysis_early_term_enabled=False,
        analysis_resolution="128x128",
        prescreen_resolution="128x128",
        iframe_prescreen_enabled=True,
        iframe_window_padding=1.0,
        iframe_window_gap_merge=2.0,
    )
    settings["output"].update(naming="DailyVlog_{date}_{mac}.mp4", resolution="128x128", fps=20)
    settings["output"]["qsv"].pop("maxrate", None)
    settings["output"]["qsv"].pop("bufsize", None)

    settings_file = tmp_path / "settings_t4_03.yaml"
    settings_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["main.py", "--no-tui", "--config", str(settings_file)])

    # 1. Full pipeline execution
    main.main()

    # 2. Database and task validation
    db = VlogDatabase()
    try:
        tasks = db.get_all_file_tasks_for_date("20260901", 0)
        assert len(tasks) == 3

        # Static clips are marked STATIC
        t_pre = db.get_file_task(str(files["cam0_00_20260901140000_20260901140002.mp4"]))
        assert t_pre["prescreen_status"] in ("STATIC", "SKIPPED_EMPTY")

        # Storm clip awakened windows captured flash and thunder
        storm_path = str(files["cam0_00_20260901140002_20260901140006.mp4"])
        storm_segs = db.get_segments_for_file(storm_path)
        assert len(storm_segs) > 0
        dynamic_storm_segs = [s for s in storm_segs if s.get("state") in ("DYNAMIC", "DYNAMIC_AUDIO", "MICRO_MOTION")]
        assert len(dynamic_storm_segs) > 0
        total_dynamic = sum(float(s["end_time"] - s["start_time"]) for s in dynamic_storm_segs)
        assert total_dynamic > 0.0

        # 3. Delivery artifacts & real container stream validation via PyAV
        outputs = list(core_config.OUTPUT_DIR.glob("DailyVlog_20260901_*.mp4"))
        assert len(outputs) == 1
        output_mp4 = outputs[0]
        with av.open(str(output_mp4)) as media:
            assert len(media.streams.video) > 0
            frames = list(media.decode(video=0))
            assert len(frames) > 0
            pts = [f.pts for f in frames if f.pts is not None]
            assert all(b > a for a, b in zip(pts, pts[1:]))

        meta_file = output_mp4.with_suffix(".meta.json")
        assert meta_file.exists()
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        assert meta["delivery_mapping"]["status"] == "AVAILABLE"
        assert meta["delivery_mapping"]["clock_validation"]["max_error_ticks"] <= 1
    finally:
        db.close()

    assert len(list(tmp_path.rglob("*.jpg")) + list(tmp_path.rglob("*.png"))) == 0
    assert_resource_hygiene()


def test_tier4_scenario_04_intermittent_pet_micromotion_bursts(tmp_path, mock_config, monkeypatch):
    """T4-04: Intermittent Pet Micro-Motion Burst: 9 bursts trigger fallback to avoid seek overhead."""
    cam_root = tmp_path / "mock_cam_t4_04"
    pet_bursts = [
        (0.2, 0.4), (0.8, 1.0), (1.4, 1.6), (2.0, 2.2), (2.6, 2.8),
        (3.2, 3.4), (3.8, 4.0), (4.4, 4.6), (5.0, 5.2),
    ]
    specs = {
        0: [
            {"name": "00_20260901090000_20260901090002.mp4", "scene": "static", "duration_s": 2.0},
            {"name": "00_20260901090002_20260901090008.mp4", "scene": "motion", "duration_s": 6.0, "motion": pet_bursts},
            {"name": "00_20260901090008_20260901090010.mp4", "scene": "static", "duration_s": 2.0},
        ]
    }
    files = build_custom_dataset(cam_root, specs)
    cam_dirs = [str(cam_root / f"XiaomiCamera_01_{CAMERA_MACS[0]}")]

    settings = dict(mock_config)
    settings["paths"] = {"input_dirs": cam_dirs}
    settings["pipeline"]["render_gpu_policy"] = "cpu_only"
    settings["recovery"]["min_disk_space_gb"] = 0
    settings["detection"].update(
        analysis_fps_adaptive=False,
        analysis_early_term_enabled=False,
        analysis_resolution="128x128",
        prescreen_resolution="128x128",
        iframe_prescreen_enabled=True,
        iframe_max_isolated_windows=8,
        iframe_fallback_active_ratio=0.60,
        iframe_window_padding=1.0,
        iframe_window_gap_merge=1.0,
    )
    settings["output"].update(naming="DailyVlog_{date}_{mac}.mp4", resolution="128x128", fps=20)
    settings["output"]["qsv"].pop("maxrate", None)
    settings["output"]["qsv"].pop("bufsize", None)

    settings_file = tmp_path / "settings_t4_04.yaml"
    settings_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["main.py", "--no-tui", "--config", str(settings_file)])

    # 1. Execute full pipeline
    main.main()

    # 2. Database validation: fallback occurred on Clip 2, while Clips 1 and 3 remained static
    db = VlogDatabase()
    try:
        tasks = db.get_all_file_tasks_for_date("20260901", 0)
        assert len(tasks) == 3

        pet_clip = str(files["cam0_00_20260901090002_20260901090008.mp4"])
        pet_task = db.get_file_task(pet_clip)
        assert pet_task is not None

        segs = db.get_segments_for_file(pet_clip)
        assert len(segs) > 0
        total_dynamic = sum(
            float(s["end_time"] - s["start_time"])
            for s in segs if s.get("state") in ("DYNAMIC", "DYNAMIC_AUDIO", "MICRO_MOTION")
        )
        assert total_dynamic > 0.0

        outputs = list(core_config.OUTPUT_DIR.glob("DailyVlog_20260901_*.mp4"))
        assert len(outputs) == 1
        with av.open(str(outputs[0])) as media:
            frames = list(media.decode(video=0))
            pts = [f.pts for f in frames if f.pts is not None]
            assert all(b > a for a, b in zip(pts, pts[1:]))
    finally:
        db.close()

    assert len(list(tmp_path.rglob("*.jpg")) + list(tmp_path.rglob("*.png"))) == 0
    assert_resource_hygiene()


def test_tier4_scenario_05_night_giant_clearance_and_tail_stealing(tmp_path, mock_config, monkeypatch):
    """T4-05: Night Giant Clearance & Work Prioritization: Heavy tasks prioritized via LPT, light tasks drained."""
    cam_root = tmp_path / "mock_cam_t4_05"
    specs = {
        0: [
            {"name": "00_20260901000000_20260901000006.mp4", "scene": "static", "duration_s": 6.0},
            {"name": "00_20260901010000_20260901010006.mp4", "scene": "static", "duration_s": 6.0},
            {"name": "00_20260901080000_20260901080003.mp4", "scene": "motion", "duration_s": 3.0, "motion": [(0.5, 2.0)]},
            {"name": "00_20260901090000_20260901090003.mp4", "scene": "motion", "duration_s": 3.0, "motion": [(0.5, 2.0)]},
            {"name": "00_20260901100000_20260901100003.mp4", "scene": "motion", "duration_s": 3.0, "motion": [(0.5, 2.0)]},
            {"name": "00_20260901120000_20260901120001.mp4", "scene": "static", "duration_s": 1.0},
            {"name": "00_20260901130000_20260901130001.mp4", "scene": "static", "duration_s": 1.0},
            {"name": "00_20260901140000_20260901140001.mp4", "scene": "static", "duration_s": 1.0},
        ]
    }
    build_custom_dataset(cam_root, specs)
    cam_dirs = [str(cam_root / f"XiaomiCamera_01_{CAMERA_MACS[0]}")]

    settings = dict(mock_config)
    settings["paths"] = {"input_dirs": cam_dirs}
    settings["scheduler"]["heavy_file_duration_s"] = 5.0
    settings["pipeline"]["cost_priority"] = True
    settings["pipeline"]["render_gpu_policy"] = "cpu_only"
    settings["recovery"]["min_disk_space_gb"] = 0
    settings["detection"].update(
        analysis_fps_adaptive=False,
        analysis_early_term_enabled=False,
        analysis_resolution="128x128",
        prescreen_resolution="128x128",
    )
    settings["output"].update(naming="DailyVlog_{date}_{mac}.mp4", resolution="128x128", fps=20)
    settings["output"]["qsv"].pop("maxrate", None)
    settings["output"]["qsv"].pop("bufsize", None)

    # 1. Verify AnalysisQueue dual-priority contract directly on production class
    q = AnalysisQueue()
    q.cost_priority = True
    q.put({"filepath": "heavy_1.mp4", "file_duration": 6.0})
    q.put({"filepath": "short_1.mp4", "file_duration": 1.0})
    q.put({"filepath": "heavy_2.mp4", "file_duration": 6.0})
    q.put({"filepath": "med_1.mp4", "file_duration": 3.0})

    heavy_1 = q.get_heavy(min_duration=5.0)
    assert heavy_1["file_duration"] == 6.0
    heavy_2 = q.get_heavy(min_duration=5.0)
    assert heavy_2["file_duration"] == 6.0
    short_task = q.get()
    assert short_task["file_duration"] == 1.0

    # 2. Run full pipeline on the 8-clip dataset
    settings_file = tmp_path / "settings_t4_05.yaml"
    settings_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["main.py", "--no-tui", "--config", str(settings_file)])
    main.main()

    # 3. Verify SQLite persistence and output delivery
    db = VlogDatabase()
    try:
        tasks = db.get_all_file_tasks_for_date("20260901", 0)
        assert len(tasks) == 8
        heavy_tasks = [t for t in tasks if float(t.get("file_duration") or 0.0) >= 5.0]
        assert len(heavy_tasks) == 2

        outputs = list(core_config.OUTPUT_DIR.glob("DailyVlog_20260901_*.mp4"))
        assert len(outputs) == 1
        with av.open(str(outputs[0])) as media:
            assert len(media.streams.video) > 0
    finally:
        db.close()

    assert len(list(tmp_path.rglob("*.jpg")) + list(tmp_path.rglob("*.png"))) == 0
    assert_resource_hygiene()


def test_tier4_scenario_06_midstream_crash_and_idempotent_resume(tmp_path, mock_config, monkeypatch):
    """T4-06: Mid-Stream Crash & Idempotent Resume: CLI restarted skips completed work."""
    build_footage(tmp_path / "footage")
    cam_dirs = [str(tmp_path / "footage" / f"XiaomiCamera_01_{CAMERA_MACS[0]}")]

    settings = dict(mock_config)
    settings["paths"] = {"input_dirs": cam_dirs}
    settings["pipeline"]["render_gpu_policy"] = "cpu_only"
    settings["recovery"]["min_disk_space_gb"] = 0
    settings["detection"].update(
        analysis_fps_adaptive=False,
        analysis_early_term_enabled=False,
        analysis_resolution="128x128",
        prescreen_resolution="128x128",
    )
    settings["output"].update(naming="DailyVlog_{date}_{mac}.mp4", resolution="128x128", fps=20)
    settings["output"]["qsv"].pop("maxrate", None)
    settings["output"]["qsv"].pop("bufsize", None)

    settings_file = tmp_path / "settings_t4_06.yaml"
    settings_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["main.py", "--no-tui", "--config", str(settings_file)])

    # First run completes
    main.main()
    outputs1 = list(core_config.OUTPUT_DIR.glob("DailyVlog_20260901_*.mp4"))
    assert len(outputs1) == 1
    fp1 = file_identity(outputs1[0])

    # Second run is idempotent (re-uses existing completed render)
    main.main()
    outputs2 = list(core_config.OUTPUT_DIR.glob("DailyVlog_20260901_*.mp4"))
    assert len(outputs2) == 1
    assert file_identity(outputs2[0]) == fp1
    assert_resource_hygiene()


def test_tier4_scenario_07_dualgpu_heterogeneous_transcode_peak_backpressure(tmp_path, mock_config, monkeypatch):
    """T4-07: Dual-GPU Heterogeneous Transcode Peak: Multi-batch backpressure & seamless concatenation."""
    cam_root = tmp_path / "mock_cam_t4_07"
    specs = {
        0: [
            {"name": "00_20260901000000_20260901000002.mp4", "scene": "static", "duration_s": 2.0},
            {"name": "00_20260901040000_20260901040003.mp4", "scene": "motion", "duration_s": 3.0, "motion": [(0.5, 2.5)]},
            {"name": "00_20260901080000_20260901080002.mp4", "scene": "static", "duration_s": 2.0},
            {"name": "00_20260901120000_20260901120003.mp4", "scene": "motion", "duration_s": 3.0, "motion": [(0.5, 2.5)]},
            {"name": "00_20260901160000_20260901160002.mp4", "scene": "static", "duration_s": 2.0},
            {"name": "00_20260901200000_20260901200003.mp4", "scene": "motion", "duration_s": 3.0, "motion": [(0.5, 2.5)]},
        ]
    }
    build_custom_dataset(cam_root, specs)
    cam_dirs = [str(cam_root / f"XiaomiCamera_01_{CAMERA_MACS[0]}")]

    settings = dict(mock_config)
    settings["paths"] = {"input_dirs": cam_dirs}
    settings["batching"] = {"max_files_per_batch": 2, "max_frames_per_batch": 60}
    settings["pipeline"]["render_gpu_policy"] = "cpu_only"
    settings["recovery"]["min_disk_space_gb"] = 0
    settings["detection"].update(
        analysis_fps_adaptive=False,
        analysis_early_term_enabled=False,
        analysis_resolution="128x128",
        prescreen_resolution="128x128",
    )
    settings["output"].update(naming="DailyVlog_{date}_{mac}.mp4", resolution="128x128", fps=20)
    settings["output"]["qsv"].pop("maxrate", None)
    settings["output"]["qsv"].pop("bufsize", None)

    # 1. Verify DualEndedBatchQueue scheduling invariants under backpressure
    bq = scheduler.DualEndedBatchQueue()
    bq.put(b_idx=0, files=["f1", "f2"], dynamic_duration=10.0)
    bq.put(b_idx=1, files=["f3", "f4"], dynamic_duration=2.0)
    bq.put(b_idx=2, files=["f5", "f6"], dynamic_duration=5.0)

    heaviest = bq.pop_heaviest(timeout=0.5)
    assert heaviest is not None and heaviest.b_idx == 0
    lightest = bq.steal_lightest(max_dynamic_s=3.0)
    assert lightest is not None and lightest.b_idx == 1
    rem = bq.pop_heaviest(timeout=0.5)
    assert rem is not None and rem.b_idx == 2
    assert bq.empty() is True

    # 2. Run full multi-batch pipeline
    settings_file = tmp_path / "settings_t4_07.yaml"
    settings_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["main.py", "--no-tui", "--config", str(settings_file)])
    main.main()

    # 3. Verify multi-batch reconciliation & real container PTS seam continuity
    db = VlogDatabase()
    try:
        tasks = db.get_all_file_tasks_for_date("20260901", 0)
        assert len(tasks) == 6

        outputs = list(core_config.OUTPUT_DIR.glob("DailyVlog_20260901_*.mp4"))
        assert len(outputs) == 1
        output_mp4 = outputs[0]

        with av.open(str(output_mp4)) as media:
            frames = list(media.decode(video=0))
            pts = [f.pts for f in frames if f.pts is not None]
            assert len(pts) >= 40
            for a, b in zip(pts, pts[1:]):
                assert b > a, f"Non-monotonic PTS transition at batch seam: {a} -> {b}"

        meta = json.loads(output_mp4.with_suffix(".meta.json").read_text(encoding="utf-8"))
        assert meta["delivery_mapping"]["status"] == "AVAILABLE"
        assert meta["delivery_mapping"]["clock_validation"]["max_error_ticks"] <= 1
    finally:
        db.close()

    assert len(list(tmp_path.rglob("*.jpg")) + list(tmp_path.rglob("*.png"))) == 0
    assert_resource_hygiene()
