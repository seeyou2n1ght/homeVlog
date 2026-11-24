"""Exercise the production CLI against local media through complete software delivery."""

import hashlib
import json
import re
import sys

import av
import pytest
import yaml

from src.core import config
from src.core.database import VlogDatabase
from src.core.identity import file_identity
from src.hardware import scheduler
from src.hardware.ffmpeg import FFmpegProcessRegistry
from src.hardware.render_cache import _manifest_path
from src.stages.scanner import get_date_cam_groups
from tests.helpers import validate_perf_json_schema
from tests.helpers_e2e import CAMERA_MACS, CLIP_SECONDS, DATE, FPS, build_footage


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_manifest(media):
    manifest = _read_json(_manifest_path(media))
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["fingerprint"])
    assert manifest["output"] == file_identity(media)


def _subtitle_seconds(stamp):
    hours, minutes, seconds, millis = map(int, re.split("[:,]", stamp))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000


def _assert_delivery(output):
    _assert_manifest(output)
    metadata = _read_json(output.with_suffix(".meta.json"))
    mapping = metadata["delivery_mapping"]
    assert mapping["status"] == "AVAILABLE"
    assert mapping["fps"] == FPS
    with av.open(str(output)) as media:
        video = media.streams.video[0]
        audio = media.streams.audio[0]
        assert (video.codec_context.name, video.width, video.height) == ("h264", 128, 128)
        assert audio.codec_context.name == "aac"
        assert video.average_rate == FPS
        assert len(list(media.decode(video=0))) == mapping["planned_frames"]
        video_seconds = float(video.duration * video.time_base)
        audio_seconds = float(audio.duration * audio.time_base)
    assert video_seconds == pytest.approx(mapping["planned_frames"] / FPS)
    assert abs(video_seconds - audio_seconds) <= 0.05

    cursor = 0
    for segment in mapping["segments"]:
        assert segment["output_start_frame"] == cursor
        cursor = segment["output_end_frame"]
        assert cursor > segment["output_start_frame"]
        source_seconds = segment["source_end_s"] - segment["source_start_s"]
        display_seconds = (cursor - segment["output_start_frame"]) / FPS
        if segment["state"] in {"DYNAMIC", "DYNAMIC_AUDIO"}:
            assert display_seconds == pytest.approx(source_seconds, abs=1 / FPS)
        elif segment["state"] == "STATIC" and source_seconds == CLIP_SECONDS:
            assert display_seconds < source_seconds
    assert cursor == mapping["planned_frames"]

    subtitles = output.with_suffix(".srt").read_text(encoding="utf-8")
    ranges = [tuple(map(_subtitle_seconds, line.split(" --> ")))
              for line in subtitles.splitlines() if " --> " in line]
    assert ranges
    assert ranges[0][0] == 0
    assert all(start < end for start, end in ranges)
    assert all(end == next_start for (_, end), (next_start, _) in zip(ranges, ranges[1:]))
    assert ranges[-1][1] == pytest.approx(video_seconds, abs=1 / FPS)
    return metadata


def _snapshot(paths):
    return {path: (path.stat().st_mtime_ns, hashlib.sha256(path.read_bytes()).hexdigest())
            for path in paths}


def _assert_resources_released():
    io = scheduler.get_disk_semaphore()
    assert io.available == io.limit
    assert not io.waiters
    assert scheduler._nvdec_semaphore is None
    assert scheduler._nvenc_semaphore is None
    assert scheduler._qsv_semaphore is None
    assert scheduler._qsv_render_semaphore is None
    assert not FFmpegProcessRegistry._processes
    assert not FFmpegProcessRegistry._jobs
    assert not FFmpegProcessRegistry._starting
    assert not FFmpegProcessRegistry._deferred


def test_synthetic_cli_delivers_and_reuses_two_cameras(tmp_path, mock_config, monkeypatch):
    import main
    from src import pipeline

    footage = build_footage(tmp_path / "mock_smb_share")
    camera_dirs = list(dict.fromkeys(str(path.parent) for path in footage.values()))
    settings = mock_config
    settings["paths"] = {"input_dirs": camera_dirs}
    settings["pipeline"]["render_gpu_policy"] = "cpu_only"
    settings["recovery"]["min_disk_space_gb"] = 0
    settings["detection"].update(
        analysis_fps_adaptive=False, analysis_early_term_enabled=False,
        analysis_resolution="128x128", prescreen_resolution="128x128",
    )
    settings["segment"].update(
        min_motion_duration=0.5, min_static_duration=0.5, gap_tolerance=0.1,
        apply_smoothing=False, pre_roll=0.2, post_roll=0.2, action_coalesce_gap=0.2,
    )
    settings["audio_vad"] = {"enabled": True, "prescreen_audio_gate": True}
    settings["render"].update(
        max_concurrency=1, batch_max_files=1, hw_decode=False, generate_subtitles=True,
        local_staging_enabled=True,
    )
    settings["output"].update(naming="DailyVlog_{date}_{mac}.mp4", resolution="128x128", fps=FPS)
    settings["output"]["qsv"].pop("maxrate")
    settings["output"]["qsv"].pop("bufsize")
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["main.py", "--no-tui", "--config", str(settings_file)])
    main.setup_logging()
    main.main()

    db = VlogDatabase()
    try:
        groups = get_date_cam_groups(db)
        assert len(groups) == 2
        assert all(date == DATE for date, _ in groups)
        assert sorted(len(db.get_all_file_tasks_for_date(*group)) for group in groups) == [1, 5]
        tasks = {scene: db.get_file_task(str(path)) for scene, path in footage.items()}
        assert all(row["file_duration"] == CLIP_SECONDS for row in tasks.values())
        assert tasks["empty"]["prescreen_status"] == "SKIPPED_EMPTY"
        assert tasks["static"]["prescreen_status"] == "STATIC"
        assert tasks["secondary"]["prescreen_status"] == "STATIC"
        assert tasks["motion_audio"]["prescreen_status"] == "SUSPICIOUS"
        for scene in ("motion_audio", "arrival", "motion"):
            assert tasks[scene]["analysis_status"] == "ANALYZED"
            completion = json.loads(tasks[scene]["decode_completion"])
            assert completion["complete"]
            assert completion["reason"] == "NORMAL_COMPLETION"
        assert {"DYNAMIC", "DYNAMIC_AUDIO"} <= {
            segment["state"] for segment in db.get_segments_for_file(str(footage["motion_audio"]))
        }
        assert all(db.is_render_completed(*group) for group in groups)
    finally:
        db.close()

    outputs = [config.OUTPUT_DIR / f"DailyVlog_{DATE}_{mac}.mp4" for mac in CAMERA_MACS]
    primary, secondary = [_assert_delivery(path) for path in outputs]
    assert [row["source_file"] for row in primary["skipped_sources"]] == [footage["empty"].name]
    assert primary["skipped_sources"][0]["reason"] == "empty_file"
    assert not secondary["skipped_sources"]
    assert {row["state"] for row in primary["timeline_highlights"]} >= {"DYNAMIC", "DYNAMIC_AUDIO"}
    assert not secondary["timeline_highlights"]

    batches = sorted(config.TEMP_DIR.glob("_batch*.mp4"))
    assert len(batches) == 5
    for batch in batches:
        _assert_manifest(batch)
    for staging in (config.TEMP_STAGING_DIR, config.TEMP_DIR / "_render_stage"):
        assert not any(path.is_file() for path in staging.rglob("*"))
    assert not any(path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in tmp_path.rglob("*"))
    assert not list(config.OUTPUT_DIR.glob(".delivery_backup_*"))
    assert not list(config.OUTPUT_DIR.glob("*.pending.*"))

    perf_files = sorted((config.LOGS_DIR / "perf").glob("*.json"))
    assert len(perf_files) == 2
    events = [json.loads(line) for event_file in (config.LOGS_DIR / "events").glob("*.jsonl*")
              for line in event_file.read_text(encoding="utf-8").splitlines()]
    run_ids = set()
    for path in perf_files:
        perf = _read_json(path)
        rid = perf["run_context"]["run_id"]
        assert rid not in run_ids
        run_ids.add(rid)
        group = [e for e in events if e.get("run_id") == rid]
        assert all((e["date"], e["cam"]) == (perf["date"], perf["cam"]) for e in group)
        terminal, = [e for e in group if e.get("event") == "run_finished"]
        encoded = [e for e in group if e.get("event") == "batch_encoded"]
        assert encoded and all(isinstance(e["batch"], int) and e["worker"] for e in encoded)
        assert terminal["status"] == "completed" and terminal["counts"]["encoded"] == len(encoded)
        assert [e["phase"] for e in group if e.get("event") == "phase_changed"][-4:] == [
            "最终拼接中", "时钟校验与元数据生成中", "交付提交中", "已交付",
        ]
        valid, message = validate_perf_json_schema(perf)
        assert valid, message
        stages = {row["stage"] for row in perf["records"]}
        assert {"prescreen", "render", "final_concat"} <= stages
        assert all(row["gpu"] == "cpu" for row in perf["records"])
    _assert_resources_released()

    assets = [path for path in config.OUTPUT_DIR.rglob("*") if path.is_file()]
    tracked = assets + batches + [_manifest_path(batch) for batch in batches] + perf_files
    first = _snapshot(tracked)

    def unexpected_transcode(*args, **kwargs):
        pytest.fail("Completed deliveries must not create another streaming orchestrator")

    monkeypatch.setattr(pipeline, "StreamingOrchestrator", unexpected_transcode)
    main.main()
    assert _snapshot(tracked) == first
    assert sorted((config.LOGS_DIR / "perf").glob("*.json")) == perf_files
    _assert_resources_released()
