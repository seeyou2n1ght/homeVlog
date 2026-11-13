"""HomeVlog 分层配置治理与场景 Profile 验收测试套件 (Profiles & Config Governance Suite)

验证:
1. 内置场景预设矩阵契约完整性 (living_room, baby_room, outdoor_corridor)
2. 主旋钮 motion_sensitivity 反比动态缩放一致性与边界保护
3. 机位绑定 (MAC / 路径 / 别名) 与分层解析优先级 (Layer 1 -> Layer 2 -> Layer 3)
5. 负面异常拦截 (无效 Profile、非正 sensitivity 等)
"""

import copy
import pytest

from src.core.config import load_config, validate_config
from src.core.profiles import (
    PROFILES,
    apply_motion_sensitivity,
    get_available_profiles,
    get_profile_description,
    resolve_camera_config,
    resolve_profile_config,
)
from src.stages.scanner import resolve_camera_identity, resolve_camera_ui_name


def test_config_load_prepares_candidate_before_validation(tmp_path):
    settings = tmp_path / "settings.yaml"
    settings.write_text(
        "pipeline:\n  render_gpu_policy: heterogeneous\nrender:\n  max_concurrency: 1\n",
        encoding="utf-8",
    )

    def prepare(candidate):
        candidate["render"]["max_concurrency"] = 2
        return candidate

    with pytest.raises(ValueError, match="requires render.max_concurrency >= 2"):
        load_config(settings)
    prepared = load_config(settings, prepare=prepare)
    assert prepared["render"]["max_concurrency"] == 2
    assert load_config() is prepared


@pytest.mark.parametrize("policy", ["cpu_only", "software"])
def test_software_render_policy_allows_one_worker(policy):
    config = {
        "pipeline": {"render_gpu_policy": policy},
        "render": {"max_concurrency": 1},
    }
    validate_config(config)
    config["render"]["max_concurrency"] = 0.5
    with pytest.raises(ValueError, match="render.max_concurrency >= 1"):
        validate_config(config)


@pytest.mark.parametrize("value", [None, 0, 1, 4, 2.5, True, "3"])
def test_nv_tail_capacity_rejects_invalid_configuration(value):
    with pytest.raises(ValueError, match="nv_tail_concurrency"):
        validate_config({"render": {"nv_tail_concurrency": value}})


@pytest.mark.parametrize("value", [2, 3])
def test_nv_tail_capacity_accepts_bounded_configuration(value):
    validate_config({"render": {"nv_tail_concurrency": value}})


def test_available_profiles_and_contract():
    """验证场景预设列表与字段契约完整性。"""
    profiles = get_available_profiles()
    assert "living_room" in profiles
    assert "baby_room" in profiles
    assert "outdoor_corridor" in profiles

    required_sections = {"detection", "audio_vad", "segment", "presence", "micro_motion"}

    for name in profiles:
        prof = PROFILES[name]
        assert prof["name"] == name
        assert len(prof["description"]) > 0
        for sec in required_sections:
            assert sec in prof, f"Profile {name} 缺失必要 Section: {sec}"
            assert isinstance(prof[sec], dict), f"Profile {name} Section {sec} 必须为字典"

        # 验证每个预设单独都能通过配置校验契约
        dummy_cfg = {
            "output": {"fps": 20},
            "detection": prof["detection"],
            "audio_vad": prof["audio_vad"],
            "segment": prof["segment"],
            "presence": prof["presence"],
            "micro_motion": prof["micro_motion"],
        }
        validate_config(dummy_cfg)

        desc = get_profile_description(name)
        assert desc["name"] == name
        assert "min_motion_threshold" in desc


def test_motion_sensitivity_scaling_dynamics():
    """验证主旋钮 motion_sensitivity 反比缩放及其下界截断安全保护。"""
    base_profile = PROFILES["living_room"]
    base_thresh = base_profile["detection"]["min_motion_threshold"]  # 2.5
    base_fg = base_profile["detection"]["fg_threshold"]              # 12.0
    base_day = base_profile["micro_motion"]["day_energy_threshold"]  # 16.0

    # 1. 灵敏度 1.0 (基准): 保持原样
    scaled_1 = apply_motion_sensitivity(base_profile, sensitivity=1.0)
    assert scaled_1["detection"]["min_motion_threshold"] == base_thresh
    assert scaled_1["detection"]["fg_threshold"] == base_fg
    assert scaled_1["micro_motion"]["day_energy_threshold"] == base_day

    # 2. 灵敏度 2.0 (高敏): 门限减半
    scaled_2 = apply_motion_sensitivity(base_profile, sensitivity=2.0)
    assert scaled_2["detection"]["min_motion_threshold"] == round(base_thresh / 2.0, 2)
    assert scaled_2["detection"]["fg_threshold"] == round(base_fg / 2.0, 2)
    assert scaled_2["micro_motion"]["day_energy_threshold"] == round(base_day / 2.0, 2)

    # 3. 灵敏度 0.5 (钝感): 门限翻倍
    scaled_half = apply_motion_sensitivity(base_profile, sensitivity=0.5)
    assert scaled_half["detection"]["min_motion_threshold"] == round(base_thresh / 0.5, 2)
    assert scaled_half["detection"]["fg_threshold"] == round(base_fg / 0.5, 2)
    assert scaled_half["micro_motion"]["day_energy_threshold"] == round(base_day / 0.5, 2)

    # 4. 极端值保护: sensitivity <= 0 安全回退为 1.0，不崩溃也不产生除以零
    scaled_zero = apply_motion_sensitivity(base_profile, sensitivity=0.0)
    assert scaled_zero["detection"]["min_motion_threshold"] == base_thresh
    scaled_neg = apply_motion_sensitivity(base_profile, sensitivity=-2.5)
    assert scaled_neg["detection"]["min_motion_threshold"] == base_thresh

    # 5. 安全下界截断: 极高灵敏度 (如 100.0) 不会跌破物理安全下界
    scaled_extreme = apply_motion_sensitivity(base_profile, sensitivity=100.0)
    assert scaled_extreme["detection"]["min_motion_threshold"] >= 0.5
    assert scaled_extreme["detection"]["fg_threshold"] >= 2.0


def test_resolve_profile_config_layering():
    """验证分层解析优先级：专家级显式参数 > 宏观主旋钮 > 预设默认值。"""
    base_cfg = {
        "detection": {
            "profile": "baby_room",
            "motion_sensitivity": 1.2,
            "fg_threshold": 15.5,  # 专家显式覆盖
        },
        "segment": {
            "min_motion_duration": 4.2,  # 专家显式覆盖
        },
    }

    resolved = resolve_profile_config(base_cfg)

    # 1. 专家级显式覆盖保持最高优先级不变
    assert resolved["detection"]["fg_threshold"] == 15.5
    assert resolved["segment"]["min_motion_duration"] == 4.2

    # 2. 未显式指定的参数由 baby_room 预设补全并按 1.2 缩放
    baby_base_thresh = PROFILES["baby_room"]["detection"]["min_motion_threshold"]  # 1.8
    assert resolved["detection"]["min_motion_threshold"] == round(baby_base_thresh / 1.2, 2)

    # 3. 缺失的 section (audio_vad, presence, micro_motion) 完整补全
    assert "audio_vad" in resolved
    assert resolved["audio_vad"]["noise_margin_db"] == round(PROFILES["baby_room"]["audio_vad"]["noise_margin_db"] / 1.2, 2)
    assert resolved["presence"]["night_hours"] == [21, 8]


def test_camera_resolution_preserves_explicit_global_values_after_expansion():
    raw = {"detection": {"profile": "living_room", "fg_threshold": 19.0},
           "segment": {"min_motion_duration": 4.2},
           "cameras": {"CAM": {"profile": "baby_room", "motion_sensitivity": 1.5}}}
    global_cfg = resolve_profile_config(raw)
    global_cfg.setdefault("recovery", {})["skip_corrupt"] = True
    cam = resolve_camera_config(global_cfg, camera_id="CAM")
    assert cam["detection"]["fg_threshold"] == 19.0
    assert cam["segment"]["min_motion_duration"] == 4.2
    assert cam["detection"]["min_motion_threshold"] == 1.2
    assert cam["recovery"]["skip_corrupt"] is True
    assert resolve_camera_config(copy.deepcopy(global_cfg), camera_id="CAM") == cam
    assert resolve_camera_config(cam, camera_id="CAM") == cam
    # Selecting a different profile rebuilds implicit defaults while retaining experts.
    switched = resolve_profile_config(global_cfg, profile_name="baby_room", sensitivity=2)
    assert switched["detection"]["profile"] == "baby_room"
    assert switched["detection"]["min_motion_threshold"] == 0.9
    assert switched["detection"]["fg_threshold"] == 19.0


def test_camera_profile_resolution_and_hierarchy():
    """验证机位级绑定与配置解析的精准度。"""
    config = {
        "detection": {
            "profile": "living_room",
            "motion_sensitivity": 1.0,
        },
        "cameras": {
            # 形式 A: 字典配置
            "B888805AA3CD": {
                "profile": "baby_room",
                "alias": "baby_room",
                "motion_sensitivity": 1.5,
                "overrides": {
                    "detection": {"fg_threshold": 9.9},
                },
            },
            # 形式 B: 字符串简写
            "B888806BD537": "living_room",
        },
    }

    # 1. 针对婴儿房机位解析
    baby_cfg = resolve_camera_config(config, camera_id="B888805AA3CD")
    assert baby_cfg["detection"]["profile"] == "baby_room"
    assert baby_cfg["detection"]["motion_sensitivity"] == 1.5
    # 门限按 1.5 缩放: 1.8 / 1.5 = 1.2
    assert baby_cfg["detection"]["min_motion_threshold"] == 1.2
    # 机位特有覆盖生效
    assert baby_cfg["detection"]["fg_threshold"] == 9.9
    # 红外夜视不误抑微动
    assert baby_cfg["detection"]["ambient_drift_suppress"] is False

    # 2. 针对客厅机位解析
    living_cfg = resolve_camera_config(config, camera_id="B888806BD537")
    assert living_cfg["detection"]["profile"] == "living_room"
    assert living_cfg["detection"]["min_motion_threshold"] == 2.5
    assert living_cfg["detection"]["ambient_drift_suppress"] is True

    # 3. 针对文件路径解析 MAC
    file_path = "\\\\192.168.5.8\\LyuShare\\XiaomiCamera_01_B888805AA3CD\\00_20260320060513_20260320061157.mp4"
    path_cfg = resolve_camera_config(config, filepath=file_path)
    assert path_cfg["detection"]["profile"] == "baby_room"
    assert path_cfg["detection"]["min_motion_threshold"] == 1.2

    # 4. 未知机位优雅回退到全局默认预设
    unknown_cfg = resolve_camera_config(config, camera_id="UNKNOWN_MAC_12")
    assert unknown_cfg["detection"]["profile"] == "living_room"
    assert unknown_cfg["detection"]["min_motion_threshold"] == 2.5


def test_load_config_end_to_end_integration():
    """验证 production settings.yaml 端到端加载与校验合规性。"""
    cfg = load_config(reload=True)

    # 1. 必要段落无一遗漏
    for sec in ["paths", "cameras", "hardware", "pipeline", "scheduler", "recovery", "detection", "yolo", "render", "output", "logging", "audio_vad", "segment", "presence", "micro_motion"]:
        assert sec in cfg, f"加载结果缺少核心段: {sec}"

    # 2. 全局预设与主旋钮校验
    assert cfg["detection"]["profile"] in get_available_profiles()
    assert cfg["detection"]["motion_sensitivity"] > 0

    # 3. 负向校验: 无效预设抛出 ValueError
    bad_cfg = copy.deepcopy(cfg)
    bad_cfg["detection"]["profile"] = "non_existent_preset"
    with pytest.raises(ValueError, match="Unsupported detection.profile"):
        validate_config(bad_cfg)

    # 4. 负向校验: 非正 sensitivity 抛出 ValueError
    bad_sens_cfg = copy.deepcopy(cfg)
    bad_sens_cfg["detection"]["motion_sensitivity"] = -1.0
    with pytest.raises(ValueError, match="detection.motion_sensitivity must be positive"):
        validate_config(bad_sens_cfg)

    # 5. 负向校验: 机位无效预设抛出 ValueError
    bad_cam_cfg = copy.deepcopy(cfg)
    bad_cam_cfg["cameras"] = {"CAM01": {"profile": "invalid_profile"}}
    with pytest.raises(ValueError, match="cameras.CAM01.profile unsupported"):
        validate_config(bad_cam_cfg)


def test_scanner_camera_identity_compatibility():
    """验证扫描器对字典型机位配置与字符串简写均具备 100% 兼容性。"""
    config = {
        "cameras": {
            "B888805AA3CD": {
                "profile": "baby_room",
                "alias": "baby_room",
            },
            "B888806BD537": "living_room",
        }
    }

    dir_baby = "D:\\Footage\\XiaomiCamera_01_B888805AA3CD"
    display, identifier = resolve_camera_identity(dir_baby, cam_index=0, config=config)
    assert "baby_room" in display
    assert identifier == "baby_room"
    assert resolve_camera_ui_name(dir_baby, cam_index=0, config=config) == "baby_room"

    dir_living = "D:\\Footage\\XiaomiCamera_00_B888806BD537"
    display_l, identifier_l = resolve_camera_identity(dir_living, cam_index=1, config=config)
    assert "living_room" in display_l
    assert identifier_l == "living_room"
    assert resolve_camera_ui_name(dir_living, cam_index=1, config=config) == "living_room"


class TestIFramePrescreenConfigGovernance:
    """验证 M1 (F1): iframe_prescreen_enabled 及其超参数的分层配置治理与边界校验规则。"""

    def test_default_iframe_prescreen_config_passes_validation(self):
        """验证符合规格的默认配置顺利通过校验。"""
        valid_cfg = {
            "output": {"fps": 20},
            "detection": {
                "iframe_prescreen_enabled": True,
                "iframe_energy_margin": 0.80,
                "iframe_burst_ratio": 1.8,
                "iframe_window_padding": 3.0,
                "iframe_window_gap_merge": 6.0,
                "iframe_fallback_active_ratio": 0.60,
                "iframe_max_isolated_windows": 8,
            },
        }
        validate_config(valid_cfg)

    @pytest.mark.parametrize("invalid_val", ["true", 1, 0, "yes", None, [True]])
    def test_iframe_prescreen_enabled_rejects_non_bool(self, invalid_val):
        """严格校验 iframe_prescreen_enabled 必须为原生 bool 类型，拒绝字符串或整型。"""
        cfg = {"detection": {"iframe_prescreen_enabled": invalid_val}}
        with pytest.raises(ValueError, match="detection.iframe_prescreen_enabled must be a boolean"):
            validate_config(cfg)

    @pytest.mark.parametrize("key,valid_vals,invalid_vals,err_pattern", [
        ("iframe_energy_margin", [0.10, 0.80, 1.00], [0.05, 1.05, -0.5, "0.8", True], "iframe_energy_margin must be between 0.1 and 1.0"),
        ("iframe_burst_ratio", [1.0, 1.8, 5.0], [0.8, 5.5, 0.0, -1.0, True], "iframe_burst_ratio must be between 1.0 and 5.0"),
        ("iframe_window_padding", [1.0, 3.0, 10.0], [0.5, 12.0, 0.0, -2.0, True], "iframe_window_padding must be between 1.0 and 10.0"),
        ("iframe_window_gap_merge", [1.0, 6.0, 20.0], [0.5, 25.0, 0.0, -5.0, True], "iframe_window_gap_merge must be between 1.0 and 20.0"),
        ("iframe_fallback_active_ratio", [0.20, 0.60, 0.90], [0.10, 0.95, 0.0, 1.5, True], "iframe_fallback_active_ratio must be between 0.2 and 0.9"),
    ])
    def test_iframe_float_parameters_range_and_type_bounds(self, key, valid_vals, invalid_vals, err_pattern):
        """验证浮点型超参数的区间上下界与类型合法性。"""
        for v in valid_vals:
            validate_config({"output": {"fps": 20}, "detection": {key: v}})
        for inv in invalid_vals:
            with pytest.raises(ValueError, match=err_pattern):
                validate_config({"detection": {key: inv}})

    @pytest.mark.parametrize("invalid_val", [0, 33, -1, 8.5, "8", True, False])
    def test_iframe_max_isolated_windows_integer_and_range_bounds(self, invalid_val):
        """验证最大独立窗口数必须为 [1, 32] 之间的整数。"""
        with pytest.raises(ValueError, match="detection.iframe_max_isolated_windows must be an integer between 1 and 32"):
            validate_config({"detection": {"iframe_max_isolated_windows": invalid_val}})

    def test_motion_detector_parses_iframe_settings(self):
        """验证 MotionDetector 正确解析并绑定配置参数。"""
        from src.stages.detector import MotionDetector
        custom_cfg = {
            "detection": {
                "iframe_prescreen_enabled": False,
                "iframe_energy_margin": 0.75,
                "iframe_burst_ratio": 2.2,
                "iframe_window_padding": 4.5,
                "iframe_window_gap_merge": 8.0,
                "iframe_fallback_active_ratio": 0.50,
                "iframe_max_isolated_windows": 12,
            }
        }
        detector = MotionDetector(custom_cfg)
        assert detector.iframe_prescreen_enabled is False
        assert detector.iframe_energy_margin == 0.75
        assert detector.iframe_burst_ratio == 2.2
        assert detector.iframe_window_padding == 4.5
        assert detector.iframe_window_gap_merge == 8.0
        assert detector.iframe_fallback_active_ratio == 0.50
        assert detector.iframe_max_isolated_windows == 12
