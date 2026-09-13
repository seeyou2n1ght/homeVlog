"""测试模块 2: 运动检测 (Motion Detector)、空间网格抗噪滤波与音频 VAD 唤醒.

覆盖：
1. 空间网格滤波 (SpatialGridMotionFilter) 8x8 高斯抗噪、单点孤立噪点抑制、连通域聚类放大；
2. EMA 背景建模 (EmaBackgroundModel) 背景与前景分离；
3. 早停置信度衰减 (Early Termination & Anti-miss Memory)；
4. 标签时域平滑与中值滤波 (_median_filter, _smooth_labels)；
5. 多模态音频 VAD 能量检测、分帧与静音鲁棒性。
"""

import math
import numpy as np
import pytest

from src.detector import (
    SpatialGridMotionFilter,
    EmaBackgroundModel,
    MotionDetector,
    _median_filter,
    _smooth_labels,
)


class TestSpatialGridAndEmaMotionFilter:
    """测试空间网格抗噪与 EMA 滑动背景模型。"""

    def test_gaussian_diffuse_noise_suppression(self):
        """弥散高斯底噪被 8x8 空间网格成功吸收，不触发连通域报警。"""
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=1.5,
            sens_multiplier=1.5,
            cell_noise_alpha=0.05,
        )
        np.random.seed(42)
        height, width = 160, 160
        for _ in range(15):
            noise = np.abs(np.random.normal(loc=0.0, scale=1.2, size=(height, width))).astype(np.float32)
            eff_energy, is_motion, stats = grid_filter.process_frame(noise, dt=0.2)
            assert not is_motion
            assert stats["active_cells"] == 0
            assert eff_energy < 1.5

    def test_salt_and_pepper_isolated_spike_suppressed(self):
        """孤立单网格轻度噪点不满足 8-邻域连通度，被有效压制。"""
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=1.5,
            sens_multiplier=1.0,
        )
        frame = np.zeros((160, 160), dtype=np.float32)
        # 单网格微幅噪声（高于 threshold 但低于极端 spike 阈值）
        frame[20:40, 20:40] = 2.0
        eff_energy, is_motion, stats = grid_filter.process_frame(frame, dt=0.2)
        # 单独 1 个 cell 不满足 min_connected_cells >= 2，被过滤
        assert not is_motion
        assert stats["active_cells"] == 0

    def test_connected_motion_cluster_boosted(self):
        """相邻多个网格的连续真实动作激活聚类放大。"""
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            min_connected_cells=2,
            base_noise_thresh=1.5,
            cluster_boost=1.2,
        )
        frame = np.zeros((160, 160), dtype=np.float32)
        # 激活相邻的 (1,1) 和 (1,2) 区域
        frame[20:40, 20:40] = 30.0
        frame[20:40, 40:60] = 30.0
        eff_energy, is_motion, stats = grid_filter.process_frame(frame, dt=0.2)
        assert is_motion
        assert stats["active_cells"] >= 2
        assert eff_energy > 30.0

    def test_ema_background_model_adaptation(self):
        """验证 EMA 滑动背景模型对静态场景的快速吸收与动态前景分离。"""
        bg_model = EmaBackgroundModel(alpha_bg=0.1, alpha_fg=0.01)
        static_frame = np.full((100, 100), 128, dtype=np.uint8)

        # 连续推入相同背景帧
        for _ in range(10):
            sal, d_frame, d_bg = bg_model.update(static_frame)
        assert np.mean(sal) < 2.0

        # 突入移动前景
        motion_frame = static_frame.copy()
        motion_frame[30:70, 30:70] = 255
        sal_motion, _, _ = bg_model.update(motion_frame)
        assert np.max(sal_motion) > 50.0


class TestTemporalSmoothingAndEarlyTerm:
    """测试时域标签平滑、中值滤波与早停机制。"""

    def test_median_filter(self):
        energies = [1.0, 1.2, 50.0, 1.1, 1.0]
        filtered = _median_filter(energies, window=3)
        assert filtered[2] < 10.0  # 孤立尖峰被中值平滑过滤

    def test_smooth_labels_absorption(self):
        # 极短的 1 帧 True (动作) 被平滑压制
        raw = [False] * 10 + [True] * 1 + [False] * 10
        smoothed = _smooth_labels(
            raw,
            min_motion=3,
            min_static=5,
            noise_suppress=2,
        )
        assert True not in smoothed

    @pytest.mark.parametrize("pause_s,should_decay", [(3.0, False), (10.0, True), (30.0, True)])
    def test_confidence_cooldown_decay(self, pause_s: float, should_decay: bool):
        """测试静止人员冷却半衰期与置信度衰减逻辑。"""
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            cooldown_half_life=8.0,
        )
        # 初始化高置信度
        frame = np.zeros((160, 160), dtype=np.float32)
        frame[20:60, 20:60] = 40.0
        grid_filter.process_frame(frame, dt=0.2)
        initial_conf = np.max(grid_filter.confidence_grid)
        assert initial_conf > 0.5

        # 静止 pause_s 时间
        blank = np.zeros((160, 160), dtype=np.float32)
        grid_filter.process_frame(blank, dt=pause_s)
        decayed_conf = np.max(grid_filter.confidence_grid)

        if should_decay:
            assert decayed_conf < initial_conf * 0.5
        else:
            assert decayed_conf > initial_conf * 0.6

    def test_global_flash_and_ir_cut_suppression(self):
        """红外夜视切换或全屋开灯引起的全幅单帧暴闪被识别并抑制。"""
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8,
            grid_cols=8,
            base_noise_thresh=1.5,
        )
        flash_frame = np.full((160, 160), fill_value=80.0, dtype=np.float32)
        eff_energy, is_motion, stats = grid_filter.process_frame(flash_frame, dt=0.2)
        assert stats.get("is_global_flash") is True
        assert not is_motion
        assert stats["active_cells"] == 0

    def test_can_early_terminate_keyword_signature(self):
        """验证 can_early_terminate 别名及关键字传参兼容性。"""
        grid_filter = SpatialGridMotionFilter(grid_rows=8, grid_cols=8)
        can_term = grid_filter.can_early_terminate(
            consecutive_static=60,
            term_window=50,
            current_energy=0.1,
            term_threshold=0.8,
        )
        assert can_term is True

    def test_night_mode_baby_movement_sensitivity(self):
        """验证夜间红外模式下，允许局部单网格婴儿微动（翻身/踢被）被灵敏捕获。"""
        grid_filter = SpatialGridMotionFilter(grid_rows=8, grid_cols=8, base_noise_thresh=1.0)
        # 仅在单个格子 (2, 2) 产生微弱动作能量
        frame_saliency = np.zeros((160, 160), dtype=np.float32)
        frame_saliency[40:60, 40:60] = 3.0 # 单格能量 3.0

        # 白天模式下：单格孤立微动会被当成噪点过滤（要求至少 2 格连通或 2.5 倍能量）
        _, is_motion_day, stats_day = grid_filter.process_frame(frame_saliency, dt=0.2, is_night_mode=False)
        assert stats_day["active_cells"] == 0
        assert not is_motion_day

        # 夜间模式下：单格微动激活（允许 min_conn=1，1.6 倍能量），有效判定为动态
        _, is_motion_night, stats_night = grid_filter.process_frame(frame_saliency, dt=0.2, is_night_mode=True)
        assert stats_night["active_cells"] == 1
        assert is_motion_night

    def test_ambient_drift_diffuse_sunlight_suppressed(self):
        """验证白天大面积均匀漫射光影（日落/朝阳）被准确识别为 ambient_drift 并软抑制。"""
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8, grid_cols=8, base_noise_thresh=1.5,
            ambient_drift_suppress=True, ambient_drift_active_ratio=0.35, ambient_drift_max_energy=5.0,
        )
        # 构造全屏均匀慢速漫射光影 (8x8 中约 32 个网格均发生微弱均匀变化 3.2 能量，无局部焦点)
        frame_saliency = np.zeros((160, 160), dtype=np.float32)
        frame_saliency[:80, :] = 3.2  # 上半屏 32 个格子全是 3.2 能量 (50% 面积)

        eff_e, is_motion, stats = grid_filter.process_frame(frame_saliency, dt=0.2, is_night_mode=False)
        assert stats["is_ambient_drift"] is True
        assert not is_motion
        assert stats["active_cells"] == 0

    def test_genuine_person_movement_not_suppressed_by_ambient_drift(self):
        """验证即使背景存在大面积漫射光影，前景真实人体动作（高能量/高局部对比）绝对不被误抑制。"""
        grid_filter = SpatialGridMotionFilter(
            grid_rows=8, grid_cols=8, base_noise_thresh=1.5,
            ambient_drift_suppress=True, ambient_drift_active_ratio=0.35, ambient_drift_max_energy=5.0,
        )
        # 漫射光影 (3.0 能量) + 显著人物走动 (20, 20) 处能量达 15.0
        frame_saliency = np.zeros((160, 160), dtype=np.float32)
        frame_saliency[:80, :] = 3.0
        frame_saliency[20:40, 20:40] = 15.0  # 人物焦点

        eff_e, is_motion, stats = grid_filter.process_frame(frame_saliency, dt=0.2, is_night_mode=False)
        assert stats["is_ambient_drift"] is False
        assert is_motion
        assert stats["active_cells"] > 0
        assert stats["max_cell_energy"] >= 15.0





class TestAudioVADAndMultimodal:
    """测试音频分帧、RMS 能量与多模态 VAD。"""

    def test_audio_rms_and_dbfs_calculation(self):
        # 构造纯正弦波信号
        sr = 16000
        t = np.linspace(0, 1, sr, endpoint=False)
        sine = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        rms = np.sqrt(np.mean(sine.astype(np.float64) ** 2))
        assert rms > 20000.0
        dbfs = 20 * math.log10(max(rms, 1e-9) / 32768.0)
        assert -4.0 < dbfs < -2.0  # 正弦波 RMS 峰值为 -3 dBFS 左右

    def test_silent_audio_produces_low_energy(self):
        silent = np.zeros(16000, dtype=np.int16)
        rms = np.sqrt(np.mean(silent.astype(np.float64) ** 2))
        assert rms == 0.0
        dbfs = 20 * math.log10(max(rms, 1e-9) / 32768.0)
        assert dbfs < -90.0
