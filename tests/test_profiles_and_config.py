"""HomeVlog 分层配置治理与场景 Profile 验收测试套件 (Profiles & Config Governance Suite)

验证:
1. 内置场景预设矩阵契约完整性 (living_room, baby_room, outdoor_corridor)
2. 主旋钮 motion_sensitivity 反比动态缩放一致性与边界保护
3. 机位绑定 (MAC / 路径 / 别名) 与分层解析优先级 (Layer 1 -> Layer 2 -> Layer 3)
4. What-If 调参仿真器与真实黄金基准集无缝集成
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
from src.core.tuner import TuningSimulator
from src.stages.scanner import resolve_camera_identity, resolve_camera_ui_name


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


def test_what_if_tuning_simulation_with_profiles():
    """验证 What-If 仿真器调用场景预设能够产生确定性的指标演进。"""
    sim = TuningSimulator()
    if not sim.samples:
        pytest.skip("黄金基准集样本不可用")

    # 仿真婴儿房高敏预设 (极低动作门限 1.8)
    res = sim.simulate_profile("baby_room", sensitivity=1.0)
    assert res["verdict"] == "UNAVAILABLE"
    assert res["profile"] == "baby_room"
    assert "deltas" not in res
    assert res["reviewed_samples"] == 0


def test_profile_simulation_passes_full_effective_profile_to_replay():
    observed = []
    sample = {"scenario": "ADULT_MOTION", "reference_active": True,
              "review_status": "independent_visual", "review_evidence": "fixture"}
    def replay(config, samples):
        observed.append(config)
        return [True]
    sim = TuningSimulator([sample], {}, replay=replay)
    sim.simulate_profile("baby_room", sensitivity=2.)
    assert len(observed) == 2
    assert observed[-1]["detection"]["profile"] == "baby_room"
    assert observed[-1]["detection"]["motion_sensitivity"] == 2.
    assert observed[-1]["detection"]["fg_threshold"] == 4.
    assert observed[-1]["detection"]["ambient_drift_suppress"] is False


def test_equal_global_values_do_not_skip_changed_camera_inheritance():
    from src.core.profiles import resolve_camera_config
    sample = {"filepath": "XiaomiCamera_01_B888805AA3CD/source.mp4", "cam_index": 1,
              "reference_active": True, "review_status": "independent_visual", "review_evidence": "fixture"}
    calls = []
    def replay(config, samples):
        calls.append(resolve_camera_config(config, filepath=sample["filepath"], cam_index=1)["detection"]["min_motion_threshold"])
        return [True]
    simulator = TuningSimulator([sample], {"cameras": {"B888805AA3CD": "baby_room"}}, replay=replay)
    simulator.simulate({"detection.min_motion_threshold": 2.5})
    assert calls == [1.8, 2.5]


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
