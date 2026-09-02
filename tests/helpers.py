"""
Test Helper Utilities for HomeVlog E2E and Tier 1-4 Test Suites.
"""

import json
import math
import os
import re
import struct
import wave
from pathlib import Path
from typing import Any


def validate_perf_json_schema(data: dict[str, Any]) -> tuple[bool, str]:
    """
    Validate the structured performance log against the HomeVlog perf.json schema.
    """
    required_top_keys = ["date", "cam", "pipeline_duration", "monitor_summary", "perf_summary", "records"]
    for key in required_top_keys:
        if key not in data:
            return False, f"Missing top-level key: {key}"

    if not isinstance(data["date"], str) or len(data["date"]) != 8:
        return False, f"Invalid 'date' format, expected 8-char string, got {data.get('date')!r}"

    if not isinstance(data["cam"], int):
        return False, f"Invalid 'cam' format, expected int, got {type(data.get('cam'))}"

    if not isinstance(data["pipeline_duration"], (int, float)) or data["pipeline_duration"] < 0:
        return False, f"Invalid 'pipeline_duration', expected non-negative number, got {data.get('pipeline_duration')}"

    if not isinstance(data["monitor_summary"], list):
        return False, f"'monitor_summary' must be a list, got {type(data.get('monitor_summary'))}"

    for idx, stage in enumerate(data["monitor_summary"]):
        if not isinstance(stage, dict):
            return False, f"monitor_summary[{idx}] must be a dict"
        for m_key in ["name", "duration", "avg_cpu", "avg_ram"]:
            if m_key not in stage:
                return False, f"monitor_summary[{idx}] missing key '{m_key}'"

    if not isinstance(data["perf_summary"], dict):
        return False, f"'perf_summary' must be a dict, got {type(data.get('perf_summary'))}"

    for stage_name, stage_stats in data["perf_summary"].items():
        if not isinstance(stage_stats, dict):
            return False, f"perf_summary['{stage_name}'] must be a dict"
        for stat_key in ["count", "total", "avg", "p50", "p95", "min", "max"]:
            if stat_key not in stage_stats:
                return False, f"perf_summary['{stage_name}'] missing stat key '{stat_key}'"
            if not isinstance(stage_stats[stat_key], (int, float)):
                return False, f"perf_summary['{stage_name}']['{stat_key}'] must be numeric"

    if not isinstance(data["records"], list):
        return False, f"'records' must be a list, got {type(data.get('records'))}"

    for idx, rec in enumerate(data["records"]):
        if not isinstance(rec, dict):
            return False, f"records[{idx}] must be a dict"
        for r_key in ["stage", "file", "gpu", "duration"]:
            if r_key not in rec:
                return False, f"records[{idx}] missing record key '{r_key}'"

    return True, "Schema valid"


def parse_ffmpeg_filtergraph(filter_str: str) -> dict[str, Any]:
    """
    Parse a complex FFmpeg filtergraph string and return its structured components.
    """
    statements = [s.strip() for s in filter_str.split(";") if s.strip()]
    scales = []
    splits = []
    video_trims = []
    audio_trims = []
    audio_nulls = []
    concat_clause = None

    for stmt in statements:
        if (":v]scale" in stmt or ":v]hwupload" in stmt or ":v]null[" in stmt or "scale_cuda=" in stmt or "scale_qsv=" in stmt):
            scales.append(stmt)
        elif "split=" in stmt:
            splits.append(stmt)
        elif "atrim=" not in stmt and "trim=" in stmt and "setpts=" in stmt:
            video_trims.append(stmt)
        elif "atrim=" in stmt and "asetpts=" in stmt:
            audio_trims.append(stmt)
        elif "anullsrc=" in stmt:
            audio_nulls.append(stmt)
        elif "concat=" in stmt:
            concat_clause = stmt

    concat_inputs = []
    concat_n = 0
    if concat_clause:
        m = re.search(r"concat=n=(\d+):v=(\d+):a=(\d+)", concat_clause)
        if m:
            concat_n = int(m.group(1))
        # Find all leading [vN][aN] inputs
        concat_inputs = re.findall(r"\[([a-zA-Z0-9_]+)\]", concat_clause.split("concat=")[0])

    return {
        "raw_statements": statements,
        "scales": scales,
        "splits": splits,
        "video_trims": video_trims,
        "audio_trims": audio_trims,
        "audio_nulls": audio_nulls,
        "concat_clause": concat_clause,
        "concat_n": concat_n,
        "concat_inputs": concat_inputs,
    }


def verify_filtergraph_labels_closure(filter_str: str) -> tuple[bool, str]:
    """
    Verify that every label consumed in the filtergraph is defined.
    """
    parsed = parse_ffmpeg_filtergraph(filter_str)
    concat_inputs = parsed["concat_inputs"]
    concat_n = parsed["concat_n"]

    expected_v_labels = [f"v{i}" for i in range(concat_n)]
    expected_a_labels = [f"a{i}" for i in range(concat_n)]

    for v_lbl in expected_v_labels:
        if v_lbl not in concat_inputs:
            return False, f"Missing video label [{v_lbl}] in concat inputs"

    for a_lbl in expected_a_labels:
        if a_lbl not in concat_inputs:
            return False, f"Missing audio label [{a_lbl}] in concat inputs"

    return True, "Filtergraph labels properly closed"


def create_synthetic_wav(
    filepath: Path,
    duration_sec: float = 1.0,
    sample_rate: int = 48000,
    frequency: float = 440.0,
    silence: bool = False,
) -> Path:
    """
    Generate a synthetic WAV file for testing audio extraction and VAD.
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    num_samples = int(sample_rate * duration_sec)

    with wave.open(str(filepath), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for i in range(num_samples):
            if silence:
                value = 0
            else:
                t = i / sample_rate
                value = int(16000.0 * math.sin(2.0 * math.pi * frequency * t))
            frames.extend(struct.pack("<h", value))
        wav_file.writeframes(frames)

    return filepath
