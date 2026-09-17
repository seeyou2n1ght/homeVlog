"""Fail early for settings that would strand workers or produce invalid frames."""
import math
import threading
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_STAGING_DIR = TEMP_DIR / "staging"
TEMP_BATCHES_DIR = TEMP_DIR / "batches"
TEMP_SCRIPTS_DIR = TEMP_DIR / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_PATH = PROJECT_ROOT / "data" / "vlog.db"

SETTINGS: dict = {}
_config_lock = threading.Lock()


def load_config(config_path: str | Path | None = None, reload: bool = False) -> dict:
    """Thread-safe configuration loader with validation and required directory creation."""
    global SETTINGS, CONFIG_PATH
    with _config_lock:
        if config_path is not None:
            CONFIG_PATH = Path(config_path).resolve()
            reload = True
        if SETTINGS and not reload:
            return SETTINGS
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"config not found: {CONFIG_PATH}")
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            candidate = yaml.safe_load(f) or {}
        validate_config(candidate)
        SETTINGS = candidate

        for d in [OUTPUT_DIR, TEMP_DIR, TEMP_STAGING_DIR, TEMP_BATCHES_DIR, TEMP_SCRIPTS_DIR, LOGS_DIR, DB_PATH.parent]:
            d.mkdir(parents=True, exist_ok=True)

        return SETTINGS



def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    qsv = config.get("output", {}).get("qsv", {})
    if qsv.get("maxrate") or qsv.get("bufsize"):
        raise ValueError("QSV ICQ uses global_quality; remove maxrate/bufsize to avoid CQP fallback")
    positive = {
        "hardware": ("max_nv_concurrency", "max_qsv_concurrency", "max_qsv_analysis_concurrency", "max_qsv_render_concurrency", "max_io_concurrency"),
        "detection": ("prescreen_parallel", "analysis_max_workers", "analysis_buffer_mb", "prescreen_segments", "analysis_fps", "prescreen_diff_threshold"),
        "render": (
            "batch_max_files", "max_concurrency", "inactivity_timeout_s",
            "qsv_timeout_s", "static_sample_window_s", "dynamic_window_max_s",
        ),
        "segment": (
            "min_motion_duration", "min_static_duration", "static_keyframe_interval",
            "keyframe_display_duration", "min_static_display_duration", "max_static_display_duration",
        ),
        "audio_vad": ("window_ms", "sample_rate"),
        "yolo": ("batch_size", "sample_fps", "max_frames_per_file"),
        "output": ("fps",),
    }
    for section, keys in positive.items():
        for key in keys:
            if key in config.get(section, {}):
                if key == "max_static_display_duration" and config[section][key] is None:
                    continue
                value = float(config[section][key])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"{section}.{key} must be positive and finite")
    modes = {
        ("detection", "prescreen_mode"): {"keyframes", "auto", "stream_fps", "legacy_seek"},
        ("pipeline", "prescreen_gpu_policy"): {"qsv_only", "cuda_only", "alternating"},
        ("pipeline", "render_gpu_policy"): {"heterogeneous", "nv_only", "qsv_only"},
    }
    for (section, key), allowed in modes.items():
        value = config.get(section, {}).get(key)
        if value is not None and value not in allowed:
            raise ValueError(f"Unsupported {section}.{key}: {value}")
    render_policy = config.get("pipeline", {}).get("render_gpu_policy")
    render_conc = config.get("render", {}).get("max_concurrency")
    if render_policy == "heterogeneous" and render_conc is not None:
        if int(render_conc) < 2:
            raise ValueError("pipeline.render_gpu_policy 'heterogeneous' requires render.max_concurrency >= 2")
    hw = config.get("hardware", {})
    qsv_total = hw.get("max_qsv_concurrency")
    qsv_analysis = hw.get("max_qsv_analysis_concurrency")
    if qsv_total is not None and qsv_analysis is not None and float(qsv_analysis) > float(qsv_total):
        raise ValueError("hardware.max_qsv_analysis_concurrency cannot exceed max_qsv_concurrency")
    seg_cfg = config.get("segment", {})
    min_disp = seg_cfg.get("min_static_display_duration")
    max_disp = seg_cfg.get("max_static_display_duration")
    if min_disp is not None and max_disp is not None:
        if float(min_disp) > float(max_disp):
            raise ValueError("segment.min_static_display_duration cannot exceed max_static_display_duration")
    for name, rate in config.get("detection", {}).get("analysis_fps_tiers", {}).items():
        if not .5 <= float(rate) <= 5:
            raise ValueError(f"analysis_fps_tiers.{name} must be between 0.5 and 5")
    coalesce_gap = config.get("render", {}).get("dynamic_coalesce_gap_s")
    if coalesce_gap is not None and (
        not math.isfinite(float(coalesce_gap)) or float(coalesce_gap) < 0
    ):
        raise ValueError("render.dynamic_coalesce_gap_s must be non-negative and finite")
    decode_ratio = config.get("render", {}).get("virtual_concat_max_decode_ratio")
    if decode_ratio is not None and (
        not math.isfinite(float(decode_ratio)) or not 0 < float(decode_ratio) <= 1
    ):
        raise ValueError("render.virtual_concat_max_decode_ratio must be in (0, 1]")
    for section, key in (("detection", "analysis_resolution"), ("detection", "prescreen_resolution"), ("output", "resolution")):
        value = config.get(section, {}).get(key)
        if value:
            width, height = (int(v) for v in value.lower().split("x"))
            if min(width, height) <= 0 or width % 2 or height % 2:
                raise ValueError(f"{section}.{key} requires positive even dimensions")
