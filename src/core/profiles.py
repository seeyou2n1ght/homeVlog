"""Built-in camera scene profiles and sensitivity resolution engine.

Provides hierarchical configuration governance:
Layer 1: Built-in scenario profiles (living_room, baby_room, outdoor_corridor)
Layer 2: Macro sensitivity knob (motion_sensitivity 0.5 ~ 2.0)
Layer 3: Explicit configuration and expert overrides (highest priority)
"""

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any


class ResolvedConfig(dict):
    """Dictionary-compatible effective config with original explicit provenance."""
    def __init__(self, values: dict, explicit: dict):
        super().__init__(values)
        self.explicit = copy.deepcopy(explicit)
        self.expanded = copy.deepcopy(values)


def _explicit_config(config: dict) -> dict:
    if not isinstance(config, ResolvedConfig):
        return copy.deepcopy(config)
    explicit = copy.deepcopy(config.explicit)
    # Preserve runtime overrides made after expansion, including CLI settings.
    for section, value in config.items():
        old = config.expanded.get(section)
        if isinstance(value, dict) and isinstance(old, dict):
            for key, item in value.items():
                if key not in old or item != old[key]:
                    explicit.setdefault(section, {})[key] = copy.deepcopy(item)
            for key in old.keys() - value.keys():
                explicit.get(section, {}).pop(key, None)
        elif section not in config.expanded or value != old:
            explicit[section] = copy.deepcopy(value)
    for section in config.expanded.keys() - config.keys():
        explicit.pop(section, None)
    return explicit

# Empirical defaults; independent recognition-quality acceptance is pending.
PROFILES: dict[str, dict[str, Any]] = {
    "living_room": {
        "name": "living_room",
        "description": "客厅标准场景：平衡动作捕捉与光影漫射抑制，启用有人驻留4x温和快进与夜间睡眠16x延时",
        "detection": {
            "min_motion_threshold": 2.5,
            "fg_threshold": 12.0,
            "timestamp_margin": 0.5,
            "ambient_drift_suppress": True,
            "ambient_drift_active_ratio": 0.35,
            "ambient_drift_max_energy": 5.0,
            "camera_motion_suppress": True,
            "camera_motion_active_ratio": 0.65,
            "camera_motion_max_cov": 0.60,
            "camera_motion_max_focal_ratio": 3.5,
        },
        "audio_vad": {
            "enabled": True,
            "noise_margin_db": 14.0,
            "min_dbfs": -32.0,
            "window_ms": 50,
            "sample_rate": 16000,
            "prescreen_audio_gate": True,
            "speech_min_dur": 0.5,
        },
        "segment": {
            "apply_smoothing": True,
            "min_motion_duration": 2.0,
            "min_static_duration": 3.0,
            "motion_absorb_energy_threshold": 12.0,
            "gap_tolerance": 1.5,
            "action_coalesce_gap": 1.2,
            "pre_roll": 1.0,
            "post_roll": 1.5,
            "static_keyframe_interval": 55.0,
            "keyframe_display_duration": 0.25,
            "min_static_display_duration": 0.3,
            "max_static_display_duration": 1.5,
        },
        "presence": {
            "enabled": True,
            "speed_factor": 4.0,
            "max_presence_gap_s": 180.0,
            "person_conf_threshold": 0.20,
            "night_stationary_enabled": True,
            "night_hours": [23, 7],
            "night_speed_factor": 16.0,
            "min_stationary_duration_s": 180.0,
            "stationary_energy_max": 2.5,
        },
        "micro_motion": {
            "enabled": True,
            "cruise_speed": 16.0,
            "day_energy_threshold": 16.0,
            "day_cell_threshold": 35.0,
            "night_energy_threshold": 18.0,
            "night_cell_threshold": 45.0,
            "min_active_cells": 2,
            "hysteresis_low_ratio": 0.65,
            "max_maintain_duration": 2.0,
            "max_coalesce_duration": 8.0,
        },
    },
    "baby_room": {
        "name": "baby_room",
        "description": "婴儿房高敏场景：极低微动触发门限，全黑红外夜视不误抑微动，高敏啼哭唤醒，较早入睡时间段",
        "detection": {
            "min_motion_threshold": 1.8,
            "fg_threshold": 8.0,
            "timestamp_margin": 0.5,
            "ambient_drift_suppress": False,  # 红外微光无室外阳光漫射，关闭大面积软抑制避免吞噬婴儿翻身
            "ambient_drift_active_ratio": 0.40,
            "ambient_drift_max_energy": 5.0,
            "camera_motion_suppress": False,  # 固定机位无云台巡航
            "camera_motion_active_ratio": 0.70,
            "camera_motion_max_cov": 0.60,
            "camera_motion_max_focal_ratio": 3.5,
        },
        "audio_vad": {
            "enabled": True,
            "noise_margin_db": 10.0,  # 提升声音灵敏度
            "min_dbfs": -38.0,        # 捕捉微弱婴儿啼哭与哼唧
            "window_ms": 50,
            "sample_rate": 16000,
            "prescreen_audio_gate": True,
            "speech_min_dur": 0.3,
        },
        "segment": {
            "apply_smoothing": True,
            "min_motion_duration": 1.5,
            "min_static_duration": 3.0,
            "motion_absorb_energy_threshold": 8.0,
            "gap_tolerance": 1.5,
            "action_coalesce_gap": 1.5,
            "pre_roll": 1.0,
            "post_roll": 1.5,
            "static_keyframe_interval": 55.0,
            "keyframe_display_duration": 0.25,
            "min_static_display_duration": 0.3,
            "max_static_display_duration": 1.5,
        },
        "presence": {
            "enabled": True,
            "speed_factor": 4.0,
            "max_presence_gap_s": 180.0,
            "person_conf_threshold": 0.15,
            "night_stationary_enabled": True,
            "night_hours": [21, 8],   # 婴儿作息更早睡、更晚起
            "night_speed_factor": 16.0,
            "min_stationary_duration_s": 120.0,
            "stationary_energy_max": 2.0,
        },
        "micro_motion": {
            "enabled": True,
            "cruise_speed": 16.0,
            "day_energy_threshold": 10.0,
            "day_cell_threshold": 20.0,
            "night_energy_threshold": 12.0,
            "night_cell_threshold": 25.0,
            "min_active_cells": 2,
            "hysteresis_low_ratio": 0.65,
            "max_maintain_duration": 2.0,
            "max_coalesce_duration": 8.0,
        },
    },
    "outdoor_corridor": {
        "name": "outdoor_corridor",
        "description": "室外与走廊场景：高抗噪与抗光影车灯漂移，较高运动门限，关闭室内驻留与睡眠延时",
        "detection": {
            "min_motion_threshold": 3.5,
            "fg_threshold": 15.0,
            "timestamp_margin": 0.5,
            "ambient_drift_suppress": True,
            "ambient_drift_active_ratio": 0.30,
            "ambient_drift_max_energy": 6.0,
            "camera_motion_suppress": True,
            "camera_motion_active_ratio": 0.60,
            "camera_motion_max_cov": 0.60,
            "camera_motion_max_focal_ratio": 3.5,
        },
        "audio_vad": {
            "enabled": True,
            "noise_margin_db": 18.0,  # 抗风噪与道路背景噪声
            "min_dbfs": -28.0,
            "window_ms": 50,
            "sample_rate": 16000,
            "prescreen_audio_gate": False,  # 室外风噪不单独作为视频唤醒依据
            "speech_min_dur": 0.5,
        },
        "segment": {
            "apply_smoothing": True,
            "min_motion_duration": 2.5,
            "min_static_duration": 4.0,
            "motion_absorb_energy_threshold": 15.0,
            "gap_tolerance": 1.5,
            "action_coalesce_gap": 1.0,
            "pre_roll": 1.0,
            "post_roll": 1.5,
            "static_keyframe_interval": 55.0,
            "keyframe_display_duration": 0.25,
            "min_static_display_duration": 0.3,
            "max_static_display_duration": 1.5,
        },
        "presence": {
            "enabled": False,  # 走廊无熟睡或驻留陪伴
            "speed_factor": 4.0,
            "max_presence_gap_s": 180.0,
            "person_conf_threshold": 0.35,
            "night_stationary_enabled": False,
            "night_hours": [23, 7],
            "night_speed_factor": 16.0,
            "min_stationary_duration_s": 180.0,
            "stationary_energy_max": 2.5,
        },
        "micro_motion": {
            "enabled": False,  # 室外无需夜间微动
            "cruise_speed": 16.0,
            "day_energy_threshold": 20.0,
            "day_cell_threshold": 45.0,
            "night_energy_threshold": 25.0,
            "night_cell_threshold": 55.0,
            "min_active_cells": 2,
            "hysteresis_low_ratio": 0.65,
            "max_maintain_duration": 2.0,
            "max_coalesce_duration": 8.0,
        },
    },
}


def apply_motion_sensitivity(profile_dict: dict[str, Any], sensitivity: float = 1.0) -> dict[str, Any]:
    """根据主旋钮 motion_sensitivity 动态缩放核心门限。

    反比缩放规则：Effective Threshold = Base Threshold / sensitivity
    灵敏度越高 (如 1.5)，门限越低，检出更灵敏；
    灵敏度越低 (如 0.8)，门限越高，更抗干扰。
    """
    sens = float(sensitivity)
    if sens <= 0:
        sens = 1.0

    scaled = copy.deepcopy(profile_dict)
    if abs(sens - 1.0) < 1e-4:
        return scaled

    # 需要缩放的门限字段及其安全下界
    scalers = [
        ("detection", "min_motion_threshold", 0.5),
        ("detection", "fg_threshold", 2.0),
        ("audio_vad", "noise_margin_db", 3.0),
        ("micro_motion", "day_energy_threshold", 2.0),
        ("micro_motion", "day_cell_threshold", 5.0),
        ("micro_motion", "night_energy_threshold", 3.0),
        ("micro_motion", "night_cell_threshold", 5.0),
    ]
    for section, key, min_val in scalers:
        if section in scaled and key in scaled[section]:
            base_val = float(scaled[section][key])
            eff_val = max(min_val, round(base_val / sens, 2))
            scaled[section][key] = eff_val

    return scaled


def resolve_profile_config(
    base_config: dict[str, Any],
    profile_name: str | None = None,
    sensitivity: float | None = None,
) -> dict[str, Any]:
    """分层解析配置：合并场景 Profile、主旋钮 sensitivity 与显式专家覆盖。

    合并优先级（从低到高）：
    1. Layer 1: 内置场景 Profile 默认值
    2. Layer 2: 宏观主旋钮 sensitivity 缩放后的参数
    3. Layer 3: base_config 中显式配置的参数（专家级最高优先级）
    """
    explicit = _explicit_config(base_config)
    result = copy.deepcopy(explicit)

    # 提取 Profile 名称
    if not profile_name:
        profile_name = result.get("detection", {}).get("profile")
    if not profile_name or profile_name not in PROFILES:
        profile_name = "living_room"

    # 提取 sensitivity
    if sensitivity is None:
        sensitivity = result.get("detection", {}).get("motion_sensitivity", 1.0)
    try:
        sensitivity = float(sensitivity)
    except (ValueError, TypeError):
        sensitivity = 1.0

    base_profile = PROFILES.get(profile_name, PROFILES["living_room"])
    scaled_profile = apply_motion_sensitivity(base_profile, sensitivity)

    # 逐 Section 合并，保留 base_config 中的显式参数
    for sec_name, sec_dict in scaled_profile.items():
        if sec_name in ("name", "description"):
            continue
        if not isinstance(sec_dict, dict):
            continue

        if sec_name not in result:
            result[sec_name] = copy.deepcopy(sec_dict)
        else:
            current_sec = result[sec_name]
            if isinstance(current_sec, dict):
                for k, v in sec_dict.items():
                    if k not in current_sec:
                        # 仅填充未显式声明的参数
                        current_sec[k] = copy.deepcopy(v)

    # 确保宏观元信息回填
    det_sec = result.setdefault("detection", {})
    det_sec["profile"] = profile_name
    det_sec["motion_sensitivity"] = sensitivity

    return ResolvedConfig(result, explicit)


def resolve_camera_config(
    config: dict[str, Any],
    camera_id: str | int | None = None,
    filepath: str | None = None,
    cam_index: int | None = None,
) -> dict[str, Any]:
    """为特定机位解析专有配置，支持机位映射、场景绑定与独立覆盖。"""
    cam_cfg = _explicit_config(config)
    cameras_map = cam_cfg.get("cameras", {}) or {}

    # 识别 MAC 地址或机位 Key
    target_mac = None
    if filepath:
        from src.core.identity import parse_camera_dir
        cam_info = parse_camera_dir(filepath)
        target_mac = cam_info.get("mac")
        if not target_mac:
            import re
            m = re.search(r"(?<![A-Fa-f0-9])[A-Fa-f0-9]{12}(?![A-Fa-f0-9])", Path(filepath).as_posix())
            if m:
                target_mac = m.group().upper()

    if not target_mac and isinstance(camera_id, str):
        target_mac = camera_id.upper() if len(camera_id) == 12 else camera_id

    # 在 cameras_map 中查找机位配置
    cam_entry = None
    if target_mac and target_mac in cameras_map:
        cam_entry = cameras_map[target_mac]
    elif camera_id is not None and str(camera_id) in cameras_map:
        cam_entry = cameras_map[str(camera_id)]
    elif cam_index is not None and str(cam_index) in cameras_map:
        cam_entry = cameras_map[str(cam_index)]

    if cam_entry is None:
        profile_name = cam_cfg.get("detection", {}).get("profile", "living_room")
        cam_sensitivity = cam_cfg.get("detection", {}).get("motion_sensitivity", 1.0)
        return resolve_profile_config(cam_cfg, profile_name=profile_name, sensitivity=cam_sensitivity)

    # 解析机位对应的 Profile 名称与 sensitivity
    profile_name = None
    cam_sensitivity = None
    cam_overrides: dict[str, Any] = {}

    if isinstance(cam_entry, str):
        # 简写形式：cameras: { "B888805AA3CD": "baby_room" }
        if cam_entry in PROFILES:
            profile_name = cam_entry
        else:
            profile_name = cam_cfg.get("detection", {}).get("profile", "living_room")
    elif isinstance(cam_entry, dict):
        profile_name = cam_entry.get("profile")
        cam_sensitivity = cam_entry.get("motion_sensitivity")
        cam_overrides = copy.deepcopy(cam_entry.get("overrides", {}))
        for sec in ("detection", "audio_vad", "segment", "presence", "micro_motion", "yolo"):
            if sec in cam_entry and isinstance(cam_entry[sec], dict):
                cam_overrides.setdefault(sec, {}).update(cam_entry[sec])

    if not profile_name:
        profile_name = cam_cfg.get("detection", {}).get("profile", "living_room")

    if cam_sensitivity is None:
        cam_sensitivity = cam_cfg.get("detection", {}).get("motion_sensitivity", 1.0)

    # Defaults are reconstructed from explicit input, never reverse-inferred
    # from the expanded global dictionary.
    cam_cfg = resolve_profile_config(cam_cfg, profile_name=profile_name, sensitivity=cam_sensitivity)

    # 应用机位特有的 overrides (最高优先级)
    for sec_name, sec_dict in cam_overrides.items():
        if sec_name in cam_cfg and isinstance(cam_cfg[sec_name], dict) and isinstance(sec_dict, dict):
            cam_cfg[sec_name].update(sec_dict)
        elif isinstance(sec_dict, dict):
            cam_cfg[sec_name] = copy.deepcopy(sec_dict)

    return cam_cfg


def get_available_profiles() -> list[str]:
    """返回所有已支持的场景 Profile 列表。"""
    return list(PROFILES.keys())


def get_profile_description(profile_name: str) -> dict[str, Any]:
    """返回特定 Profile 的详细说明与核心参数摘要。"""
    if profile_name not in PROFILES:
        raise KeyError(f"Unknown profile: {profile_name}. Available: {get_available_profiles()}")
    p = PROFILES[profile_name]
    return {
        "name": p.get("name", profile_name),
        "description": p.get("description", ""),
        "min_motion_threshold": p.get("detection", {}).get("min_motion_threshold"),
        "ambient_drift_suppress": p.get("detection", {}).get("ambient_drift_suppress"),
        "audio_noise_margin_db": p.get("audio_vad", {}).get("noise_margin_db"),
        "presence_enabled": p.get("presence", {}).get("enabled"),
        "micro_motion_enabled": p.get("micro_motion", {}).get("enabled"),
    }
