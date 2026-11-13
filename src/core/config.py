"""Fail early for settings that would strand workers or produce invalid frames."""
import math
import threading
from collections.abc import Callable
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "output"
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_STAGING_DIR = TEMP_DIR / "staging"
TEMP_BATCHES_DIR = TEMP_DIR / "batches"
TEMP_SCRIPTS_DIR = TEMP_DIR / "scripts"
LOGS_DIR = PROJECT_ROOT / "logs"
DB_PATH = PROJECT_ROOT / "data" / "vlog.db"

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "settings.yaml"
CONFIG_PATH = DEFAULT_CONFIG_PATH

SETTINGS: dict = {}
_config_lock = threading.RLock()
SOFTWARE_RENDER_POLICIES = {"cpu_only", "software"}


def uses_software_pipeline(config: dict) -> bool:
    return config.get("pipeline", {}).get("render_gpu_policy") in SOFTWARE_RENDER_POLICIES


def load_config(
    config_path: str | Path | None = None,
    reload: bool = False,
    *,
    prepare: Callable[[dict], dict] | None = None,
) -> dict:
    """Load, optionally prepare, and validate settings before publishing them."""
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
        from src.core.profiles import resolve_profile_config
        candidate = resolve_profile_config(candidate)
        if prepare is not None:
            candidate = prepare(candidate)
        validate_config(candidate)
        SETTINGS = candidate

        for d in [OUTPUT_DIR, TEMP_DIR, TEMP_STAGING_DIR, TEMP_BATCHES_DIR, TEMP_SCRIPTS_DIR, LOGS_DIR, DB_PATH.parent]:
            d.mkdir(parents=True, exist_ok=True)

        return SETTINGS


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a mapping")
    # R1: I 帧关键帧跳帧预筛选治理规则
    det_cfg = config.get("detection", {})
    if isinstance(det_cfg, dict):
        if "iframe_prescreen_enabled" in det_cfg:
            val = det_cfg["iframe_prescreen_enabled"]
            if not isinstance(val, bool):
                raise ValueError("detection.iframe_prescreen_enabled must be a boolean")

        if "iframe_min_file_duration" in det_cfg:
            val = det_cfg["iframe_min_file_duration"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or float(val) < 0.0:
                raise ValueError("detection.iframe_min_file_duration must be a non-negative finite number")

        iframe_floats = [
            ("iframe_energy_margin", 0.10, 1.00),
            ("iframe_burst_ratio", 1.00, 5.00),
            ("iframe_window_padding", 1.0, 10.0),
            ("iframe_window_gap_merge", 1.0, 20.0),
            ("iframe_fallback_active_ratio", 0.20, 0.90),
        ]
        for key, min_val, max_val in iframe_floats:
            if key in det_cfg:
                val = det_cfg[key]
                if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(float(val)) or not (min_val <= float(val) <= max_val):
                    raise ValueError(f"detection.{key} must be between {min_val} and {max_val}")

        if "iframe_max_isolated_windows" in det_cfg:
            val = det_cfg["iframe_max_isolated_windows"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or int(val) != float(val) or not (1 <= int(val) <= 32):
                raise ValueError("detection.iframe_max_isolated_windows must be an integer between 1 and 32")

    # R2: 尾段自适应工作窃取 (Work Stealing) 配置治理规则
    sched_cfg = config.get("scheduler", {})
    if isinstance(sched_cfg, dict):
        if "work_stealing_enabled" in sched_cfg:
            val = sched_cfg["work_stealing_enabled"]
            if not isinstance(val, bool):
                raise ValueError("scheduler.work_stealing_enabled must be a boolean")
        if "steal_enabled" in sched_cfg:
            val = sched_cfg["steal_enabled"]
            if not isinstance(val, bool):
                raise ValueError("scheduler.steal_enabled must be a boolean")
        if "steal_backlog_threshold" in sched_cfg:
            val = sched_cfg["steal_backlog_threshold"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or int(val) != float(val) or int(val) < 1:
                raise ValueError("scheduler.steal_backlog_threshold must be a positive integer")
        if "steal_vram_margin_mb" in sched_cfg:
            val = sched_cfg["steal_vram_margin_mb"]
            if isinstance(val, bool) or not isinstance(val, (int, float)) or int(val) != float(val) or int(val) <= 0:
                raise ValueError("scheduler.steal_vram_margin_mb must be a positive integer")
        min_dur = sched_cfg.get("steal_sublongest_min_duration")
        max_dur = sched_cfg.get("steal_sublongest_max_duration")
        if min_dur is not None:
            if isinstance(min_dur, bool) or not isinstance(min_dur, (int, float)) or not math.isfinite(float(min_dur)) or float(min_dur) <= 0:
                raise ValueError("scheduler.steal_sublongest_min_duration must be positive and finite")
        if max_dur is not None:
            if isinstance(max_dur, bool) or not isinstance(max_dur, (int, float)) or not math.isfinite(float(max_dur)) or float(max_dur) <= 0:
                raise ValueError("scheduler.steal_sublongest_max_duration must be positive and finite")
        if min_dur is not None and max_dur is not None:
            if float(min_dur) >= float(max_dur):
                raise ValueError("scheduler.steal_sublongest_min_duration cannot be greater than or equal to steal_sublongest_max_duration")

    qsv = config.get("output", {}).get("qsv", {})
    if qsv.get("maxrate") or qsv.get("bufsize"):
        raise ValueError("QSV ICQ uses global_quality; remove maxrate/bufsize to avoid CQP fallback")
    positive = {
        "hardware": ("max_nv_concurrency", "max_qsv_concurrency", "max_qsv_analysis_concurrency", "max_qsv_render_concurrency", "max_io_concurrency"),
        "detection": ("prescreen_parallel", "analysis_max_workers", "analysis_buffer_mb", "prescreen_segments", "analysis_fps", "prescreen_diff_threshold", "motion_sensitivity"),
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
    from src.core.profiles import PROFILES
    allowed_profiles = set(PROFILES.keys())
    det_profile = config.get("detection", {}).get("profile")
    if det_profile is not None and det_profile not in allowed_profiles:
        raise ValueError(f"Unsupported detection.profile: {det_profile}")
    cam_cfg = config.get("cameras", {})
    if isinstance(cam_cfg, dict):
        for cam_key, cam_val in cam_cfg.items():
            if isinstance(cam_val, dict):
                p = cam_val.get("profile")
                if p is not None and p not in allowed_profiles:
                    raise ValueError(f"cameras.{cam_key}.profile unsupported: {p}")
                sens = cam_val.get("motion_sensitivity")
                if sens is not None:
                    sens_val = float(sens)
                    if not math.isfinite(sens_val) or sens_val <= 0:
                        raise ValueError(f"cameras.{cam_key}.motion_sensitivity must be positive")
            elif isinstance(cam_val, str) and cam_val in allowed_profiles:
                pass
    modes = {
        ("detection", "prescreen_mode"): {"keyframes", "auto", "stream_fps", "legacy_seek"},
        ("pipeline", "prescreen_gpu_policy"): {"qsv_only", "cuda_only", "alternating"},
        ("pipeline", "render_gpu_policy"): {"heterogeneous", "nv_only", "qsv_only"} | SOFTWARE_RENDER_POLICIES,
    }
    for (section, key), allowed in modes.items():
        value = config.get(section, {}).get(key)
        if value is not None and value not in allowed:
            raise ValueError(f"Unsupported {section}.{key}: {value}")
    render_policy = config.get("pipeline", {}).get("render_gpu_policy")
    tail_concurrency = config.get("render", {}).get("nv_tail_concurrency")
    if "nv_tail_concurrency" in config.get("render", {}) and (
        isinstance(tail_concurrency, bool) or tail_concurrency not in (2, 3)
    ):
        raise ValueError("render.nv_tail_concurrency must be 2 or 3")
    render_conc = config.get("render", {}).get("max_concurrency")
    if uses_software_pipeline(config) and render_conc is not None and float(render_conc) < 1:
        raise ValueError("software rendering requires render.max_concurrency >= 1")
    if render_policy == "heterogeneous" and render_conc is not None:
        if int(render_conc) < 2:
            raise ValueError("pipeline.render_gpu_policy 'heterogeneous' requires render.max_concurrency >= 2")
    hw = config.get("hardware", {})
    qsv_total = hw.get("max_qsv_concurrency")
    qsv_analysis = hw.get("max_qsv_analysis_concurrency")
    if qsv_total is not None and qsv_analysis is not None and float(qsv_analysis) > float(qsv_total):
        raise ValueError("hardware.max_qsv_analysis_concurrency cannot exceed max_qsv_concurrency")
    qsv_render = hw.get("max_qsv_render_concurrency")
    if qsv_total is not None and qsv_analysis is not None and qsv_render is not None:
        if int(qsv_analysis) + int(qsv_render) > int(qsv_total):
            raise ValueError("QSV analysis + render concurrency cannot exceed max_qsv_concurrency")
    integer_keys = {
        "hardware": ("max_nv_concurrency", "max_qsv_concurrency", "max_qsv_analysis_concurrency",
                     "max_qsv_render_concurrency", "max_io_concurrency"),
        "detection": ("prescreen_parallel", "analysis_max_workers"),
        "render": ("batch_max_files", "max_concurrency"),
        "logging": ("retention_days",),
    }
    for section, keys in integer_keys.items():
        for key in keys:
            if key in config.get(section, {}):
                val = config[section][key]
                if isinstance(val, bool) or int(val) != float(val):
                    raise ValueError(f"{section}.{key} must be an integer")
                if section == "logging" and int(val) < 1:
                    raise ValueError(f"{section}.{key} must be a positive integer")
    for section, key in (("recovery", "skip_corrupt"), ("render", "cleanup_batches_on_success")):
        if key in config.get(section, {}) and not isinstance(config[section][key], bool):
            raise ValueError(f"{section}.{key} must be a boolean")
    if render_policy == "heterogeneous" and render_conc is not None and int(render_conc) > 3:
        raise ValueError("heterogeneous rendering supports at most 3 workers (2 NVENC + 1 QSV)")
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
