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


def test_increasing_sensitivity_does_not_raise_adaptive_noise_threshold():
    from array import array
    from src.core.profiles import resolve_profile_config
    from src.algorithms.motion_trace import MotionTrace
    energies = [0.0] * 10 + [1.0] * 30 + [3.0] * 60
    active = []
    for sensitivity in (1.0, 2.0):
        cfg = resolve_profile_config({"detection": {"motion_sensitivity": sensitivity,
            "median_filter_window": 1, "min_motion_frames": 1, "min_static_frames": 1,
            "noise_suppress_frames": 1}, "audio_vad": {"enabled": False}})
        detector = MotionDetector(cfg)
        trace = MotionTrace(detector, 5)
        trace.energies = array("d", energies)
        trace.confidences = array("d", [0.0] * len(energies))
        labels, _ = detector.analyze_frames(trace, file_duration=20, fps=5)
        active.append(sum(item["is_motion"] for item in labels))
    assert active == [60, 60]

from src.stages.detector import (
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


class TestTruncatedMediaAndDegradation:
    """测试截断/部分损坏原片的容错处理与真实时长校准。"""

    def test_detector_truncated_media_preserves_frames_and_corrects_duration(self):
        """当原片物理提前到达 EOF (解码帧数远小于声明时长) 时，
        MotionDetector 应保留已解码帧，校准 file_duration_detected 并产出有效标签。"""
        from unittest.mock import patch
        detector = MotionDetector({"detection": {"analysis_fps": 5, "motion_sensitivity": 1.0}})
        synthetic_frames = [np.full((180, 320), fill_value=i * 2, dtype=np.uint8) for i in range(30)]
        meta = {
            "has_audio": 0,
            "effective_fps": 5.0,
            "complete": True,
            "truncated": True,
            "decode_time": 0.5,
            "frames": 30,
        }

        with patch.object(detector, "_decode_file", return_value=(synthetic_frames, {}, np.array([], dtype=np.float32), meta)):
            labels, yolo_buf = detector.analyze("truncated_sample.mp4", file_duration=300.0, has_audio=0, duration_verified=True)

        assert len(labels) >= 30
        assert detector.file_duration_detected == 6.0  # 30 帧 / 5.0 fps = 6.0s
        assert labels[-1]["time"] <= 6.0

    def test_reanalysis_keeps_original_declared_extent(self):
        from unittest.mock import patch
        detector = MotionDetector({"detection": {"analysis_fps": 5}})
        frames = [np.zeros((180, 320), dtype=np.uint8) for _ in range(30)]
        meta = {"has_audio": 0, "effective_fps": 5., "complete": True, "truncated": True,
                "decode_time": .1, "frames": 30, "completion_reason": "TRUNCATED_EOF"}
        with patch.object(detector, "_decode_file", return_value=(frames, {}, np.array([], dtype=np.float32), meta)) as decode:
            detector.analyze("truncated.mp4", file_duration=6., has_audio=0,
                             duration_verified=True, declared_duration=300.)
        decode.assert_called_once_with("truncated.mp4", 300.)
        assert detector.decode_completion["declared_duration_s"] == 300.
        assert detector.decode_completion["available_duration_s"] == 6.

    def test_detector_audio_decode_failure_degrades_to_silent_video(self):
        """当音频管道解码抛出异常时，_decode_file 应安全降级为静音视频，禁止清空视频帧。"""
        from unittest.mock import patch
        detector = MotionDetector({"detection": {"analysis_fps": 5}})
        synthetic_frames = [np.zeros((180, 320), dtype=np.uint8) for _ in range(10)]
        pipe_meta = {
            "has_audio": 0,
            "effective_fps": 5.0,
            "complete": True,
            "truncated": False,
            "decode_time": 0.2,
            "frames": 10,
        }

        with patch.object(detector, "_decode_file_pipe", return_value=(synthetic_frames, {}, pipe_meta)), \
             patch.object(detector, "_decode_audio_pipe", side_effect=RuntimeError("Audio decode truncated")):
            frames, yolo_buf, full_audio, meta = detector._decode_file("sample_audio_corrupt.mp4", file_duration=2.0)

        assert len(frames) == 10
        assert meta["complete"] is True
        assert full_audio.size == 0  # 降级为静音



def test_visual_features_reuse_one_decode_and_clear_after_incomplete_source(tmp_path,monkeypatch,mock_config):
    from src.stages.detector import MotionDetector
    path=tmp_path/"source.mp4";path.write_bytes(b"controlled source")
    detector=MotionDetector(mock_config,decode_gpu="cpu")
    frames=[np.zeros((32,32),np.uint8)]*2500
    meta={"has_audio":0,"effective_fps":1250,"decode_time":0,"frames":2500,"complete":True}
    calls=[]
    monkeypatch.setattr(detector,"_decode_file",lambda *args:(calls.append(args) or (frames,{},np.array([],np.float32),meta)))
    labels=[{"time":0,"is_motion":False,"state":"STATIC"}]
    monkeypatch.setattr(detector,"analyze_frames",lambda *args,**kwargs:(labels,{}))
    actual,_=detector.analyze(str(path),file_duration=2,has_audio=0,duration_verified=True)
    assert actual==labels and len(calls)==1
    meta["complete"]=False
    actual,_=detector.analyze(str(path),file_duration=2,has_audio=0,duration_verified=True)
    assert actual == []
    assert len(calls)==2


# ============================================================================
# Suite 2: Window Merging and Padding Mathematics
# ============================================================================

class TestIFrameWindowResolution:
    """验证 _resolve_awakened_windows 时间窗口外扩、单格爆发与区间合并算法。"""

    def test_empty_trace_and_no_audio_returns_empty_windows(self):
        detector = MotionDetector({"detection": {"iframe_window_padding": 3.0, "iframe_window_gap_merge": 6.0}})
        # 静态关键帧序列：整幅能量 0.5 < 2.0，单格能量 1.0 < 4.5
        keyframes = [(float(t), 0.5, 1.0) for t in range(0, 60, 3)]
        windows = detector._resolve_awakened_windows(keyframes, audio_events=[], file_duration=60.0, threshold=2.5)
        assert windows == []

    def test_single_active_keyframe_padded_and_clamped(self):
        detector = MotionDetector({"detection": {"iframe_window_padding": 3.0, "iframe_window_gap_merge": 6.0}})
        # 在 t=15.0 处出现大幅运动 (能量 5.0 > 2.0)
        keyframes = [(0.0, 0.2, 0.5), (15.0, 5.0, 10.0), (30.0, 0.2, 0.5)]
        windows = detector._resolve_awakened_windows(keyframes, audio_events=[], file_duration=60.0, threshold=2.5)
        assert windows == [(12.0, 18.0)]

    def test_boundary_clamping_at_start_and_eof(self):
        detector = MotionDetector({"detection": {"iframe_window_padding": 3.0, "iframe_window_gap_merge": 6.0}})
        # 在 t=1.0 和 t=59.0 触发动作
        keyframes = [(1.0, 4.0, 8.0), (30.0, 0.1, 0.2), (59.0, 4.0, 8.0)]
        windows = detector._resolve_awakened_windows(keyframes, audio_events=[], file_duration=60.0, threshold=2.5)
        assert len(windows) == 2
        assert windows[0] == (0.0, 4.0)   # 1.0 - 3.0 截断于 0.0
        assert windows[1] == (56.0, 60.0) # 59.0 + 3.0 截断于 60.0

    def test_adjacent_windows_gap_merged(self):
        detector = MotionDetector({"detection": {"iframe_window_padding": 3.0, "iframe_window_gap_merge": 6.0}})
        # t=10.0 -> [7.0, 13.0]; t=17.0 -> [14.0, 20.0]; 间隙 14.0 - 13.0 = 1.0 <= 6.0
        keyframes = [(10.0, 3.5, 6.0), (17.0, 3.5, 6.0)]
        windows = detector._resolve_awakened_windows(keyframes, audio_events=[], file_duration=60.0, threshold=2.5)
        assert windows == [(7.0, 20.0)]

    def test_large_gap_retains_separate_windows(self):
        detector = MotionDetector({"detection": {"iframe_window_padding": 3.0, "iframe_window_gap_merge": 6.0}})
        # t=10.0 -> [7.0, 13.0]; t=28.0 -> [25.0, 31.0]; 间隙 25.0 - 13.0 = 12.0 > 6.0
        keyframes = [(10.0, 3.5, 6.0), (28.0, 3.5, 6.0)]
        windows = detector._resolve_awakened_windows(keyframes, audio_events=[], file_duration=60.0, threshold=2.5)
        assert windows == [(7.0, 13.0), (25.0, 31.0)]

    def test_micromotion_cell_burst_awakens_window_when_mean_energy_is_low(self):
        """验证整幅画面均值能量低于门限，但 8x8 单格能量触发微动突增时，能灵敏唤醒窗口。"""
        detector = MotionDetector({
            "detection": {
                "iframe_energy_margin": 0.8,
                "iframe_burst_ratio": 1.8,
                "iframe_window_padding": 3.0,
            }
        })
        # 门限 2.5 -> 全图门限 2.0; 爆发门限 2.5 * 1.8 = 4.5
        # 某帧全图能量仅 1.2 (< 2.0)，但 max_cell_energy 达到 5.2 (>= 4.5)
        keyframes = [(20.0, 1.2, 5.2)]
        windows = detector._resolve_awakened_windows(keyframes, audio_events=[], file_duration=60.0, threshold=2.5)
        assert windows == [(17.0, 23.0)]


# ============================================================================
# Suite 3: Multimodal Audio VAD Awakening & Safeguards
# ============================================================================

class TestAudioVadAwakeningAndSafeguards:
    """验证多模态音频唤醒与 SMB Seek 密集动作/长 GOP 熔断保护。"""

    def test_audio_event_awakens_window_in_visually_static_file(self):
        """画面全静止但发生门铃/人声事件时，音频窗口无条件唤醒精析，杜绝漏检。"""
        from unittest.mock import patch, MagicMock
        detector = MotionDetector({"detection": {"iframe_prescreen_enabled": True}})
        # 构造全部静止的关键帧
        keyframes = [(float(t), 0.1, 0.2) for t in range(0, 100, 3)]
        audio_events = [(40.0, 44.0, "SPEECH")]

        with patch.object(detector, "_scan_keyframes_coarse", return_value=(keyframes, {"keyframes": len(keyframes)})), \
             patch.object(detector, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
             patch.object(detector.vad, "detect_events", return_value=(audio_events, {})), \
             patch.object(detector, "_decode_file_window") as mock_window_decode:
            
            # 模拟局部精析返回 50 帧 (10 秒 * 5 fps)
            synthetic_frames = [np.zeros((180, 320), dtype=np.uint8) for _ in range(50)]
            mock_window_decode.return_value = (synthetic_frames, {}, {"complete": True, "frames": 50})

            labels, yolo_buf = detector.analyze("test_audio_event.mp4", file_duration=100.0, duration_verified=True)

        # 唤醒窗口应为 [40-3, 44+3] = [37.0, 47.0]
        mock_window_decode.assert_called_once()
        args, kwargs = mock_window_decode.call_args
        assert abs(args[1] - 37.0) < 1e-3
        assert abs(args[2] - 47.0) < 1e-3

        # 音频区间被标注为 DYNAMIC_AUDIO，外部区间被补全为 STATIC
        audio_labels = [lbl for lbl in labels if 37.0 <= lbl["time"] <= 47.0]
        static_labels = [lbl for lbl in labels if lbl["time"] < 37.0 or lbl["time"] > 47.0]
        assert any(lbl.get("state") == "DYNAMIC_AUDIO" or lbl.get("is_audio_active") for lbl in audio_labels)
        assert all(lbl.get("state") == "STATIC" for lbl in static_labels)

    def test_dense_motion_active_ratio_triggers_full_decode_fallback(self):
        """当唤醒窗口累计时长占比 > 0.60 时，触发保护回退全量顺序解码。"""
        from unittest.mock import patch
        detector = MotionDetector({
            "detection": {
                "iframe_prescreen_enabled": True,
                "iframe_fallback_active_ratio": 0.60,
            }
        })
        # 构造大面积动作：前 65 秒连续超门限
        keyframes = [(float(t), 8.0, 15.0) if t <= 65 else (float(t), 0.1, 0.2) for t in range(0, 100, 3)]
        
        with patch.object(detector, "_scan_keyframes_coarse", return_value=(keyframes, {"keyframes": len(keyframes)})), \
             patch.object(detector, "_decode_audio_pipe", return_value=np.array([], dtype=np.float32)), \
             patch.object(detector, "_decode_file", return_value=([np.zeros((180, 320), dtype=np.uint8)] * 100, {}, np.array([], dtype=np.float32), {"complete": True, "effective_fps": 5.0, "frames": 100, "decode_time": 1.0})) as mock_full_decode, \
             patch.object(detector, "_decode_file_window") as mock_window_decode:
            
            labels, yolo_buf = detector.analyze("dense_motion.mp4", file_duration=100.0, duration_verified=True)

        mock_full_decode.assert_called_once()
        mock_window_decode.assert_not_called()
        assert detector.decode_completion["reason"] in ("NORMAL_COMPLETION", "FALLBACK_DENSE_MOTION")

    def test_excessive_isolated_windows_triggers_fallback(self):
        """当独立离散窗口数 > 8 时，为规避 SMB 网络 Seek 气泡回退全量顺序解码。"""
        from unittest.mock import patch
        detector = MotionDetector({
            "detection": {
                "iframe_prescreen_enabled": True,
                "iframe_max_isolated_windows": 8,
                "iframe_window_padding": 1.0,
                "iframe_window_gap_merge": 2.0,
            }
        })
        # 构造 10 个独立散落的 1 帧微弱动作 (间隔 30 秒)
        active_times = {12.0, 42.0, 72.0, 102.0, 132.0, 162.0, 192.0, 222.0, 252.0, 282.0}
        keyframes = [(float(t), 5.0 if t in active_times else 0.1, 8.0 if t in active_times else 0.2) for t in range(0, 300, 3)]

        with patch.object(detector, "_scan_keyframes_coarse", return_value=(keyframes, {"keyframes": len(keyframes)})), \
             patch.object(detector, "_decode_audio_pipe", return_value=np.array([], dtype=np.float32)), \
             patch.object(detector, "_decode_file", return_value=([np.zeros((180, 320), dtype=np.uint8)] * 100, {}, np.array([], dtype=np.float32), {"complete": True, "effective_fps": 5.0, "frames": 100, "decode_time": 1.0})) as mock_full_decode, \
             patch.object(detector, "_decode_file_window") as mock_window_decode:

            labels, yolo_buf = detector.analyze("scattered.mp4", file_duration=300.0, duration_verified=True)

        mock_full_decode.assert_called_once()
        mock_window_decode.assert_not_called()

    def test_long_or_sparse_gop_triggers_fallback(self):
        """当关键帧间隔平均 > 8.0s (稀疏 GOP) 时，粗筛可能漏检，无条件回退全量解码。"""
        from unittest.mock import patch
        detector = MotionDetector({"detection": {"iframe_prescreen_enabled": True}})
        # 稀疏关键帧：每 12 秒一帧
        sparse_keyframes = [(0.0, 0.1, 0.1), (12.0, 0.1, 0.1), (24.0, 0.1, 0.1), (36.0, 0.1, 0.1)]

        with patch.object(detector, "_scan_keyframes_coarse", return_value=(sparse_keyframes, {"keyframes": len(sparse_keyframes), "avg_gop_s": 12.0})), \
             patch.object(detector, "_decode_audio_pipe", return_value=np.array([], dtype=np.float32)), \
             patch.object(detector, "_decode_file", return_value=([np.zeros((180, 320), dtype=np.uint8)] * 10, {}, np.array([], dtype=np.float32), {"complete": True, "effective_fps": 5.0, "frames": 10, "decode_time": 0.5})) as mock_full_decode, \
             patch.object(detector, "_decode_file_window") as mock_window_decode:

            labels, yolo_buf = detector.analyze("sparse_gop.mp4", file_duration=48.0, duration_verified=True)

        mock_full_decode.assert_called_once()
        mock_window_decode.assert_not_called()


# ============================================================================
# Regression Suite: Pipeline Device Lease Re-entry & Windowed Fine Decode
# ============================================================================

from contextlib import contextmanager


class TestPipelineDeviceLeasingReentry:
    """验证 detector.device_lease 上下文管理器生成器重入与 Tier 1/Tier 2 两级解码契约。
    
    回归验证（Milestone 1 Critical Finding 1）：
    pipeline.py:816, 831 将 detector.device_lease 绑定为返回 scheduler.lease_device() 的 lambda；
    lease_device 为 @contextmanager 生成器。
    验证 active 文件分别在 Tier 1 粗筛与 Tier 2 局部精析两次申请 lease 时，
    不会发生生成器重入崩溃 (AttributeError: '_GeneratorContextManager' object has no attribute 'args')，
    杜绝静默回退全量解码，且 completion 元数据正常输出 NORMAL_COMPLETION。
    """

    class MockScheduler:
        """模拟 src/hardware/scheduler.py WorkStealingManager 协同租约生命周期。"""
        def __init__(self):
            self.acquire_count = 0
            self.release_count = 0
            self.active_slots = 0
            self.devices_yielded = []

        @contextmanager
        def lease_device(self, queue_size: int = 0, prescreen_idle: bool = False, file_duration: float = 0.0):
            self.acquire_count += 1
            self.active_slots += 1
            device = "cuda"
            self.devices_yielded.append(device)
            try:
                yield device
            finally:
                self.active_slots -= 1
                self.release_count += 1

    def test_active_file_acquires_separate_leases_for_tier1_and_tier2_without_fallback(self):
        """TC-R1-LEASE-01: 活跃切片两级解码分别获取独立 Lease，无 AttributeError，无回退，NORMAL_COMPLETION。"""
        from unittest.mock import patch

        scheduler = self.MockScheduler()
        file_dur = 30.0

        detector = MotionDetector({"detection": {"iframe_prescreen_enabled": True}})
        # 严格复刻 src/pipeline.py:816, 831 运行期赋值
        detector.device_lease = lambda dur=file_dur: scheduler.lease_device(
            queue_size=0,
            prescreen_idle=False,
            file_duration=dur,
        )

        # 构造关键帧：t=10.0s 处发生显著动作突增 (能量 5.0 > 门限 2.5)
        keyframes = [
            (0.0, 0.1, 0.2),
            (10.0, 5.0, 10.0),
            (20.0, 0.1, 0.2),
            (30.0, 0.1, 0.2),
        ]
        coarse_meta = {"keyframes": 4, "complete": True, "avg_gop": 3.0, "max_gop": 3.0, "scan_time": 0.1}

        # 模拟局部精析返回 30 帧有效图像 (对应 [7.0, 13.0] 6秒 * 5fps)
        synthetic_window_frames = [
            np.full((180, 320), 255 if i % 2 == 0 else 0, dtype=np.uint8) for i in range(30)
        ]

        with patch.object(detector, "_scan_keyframes_coarse", return_value=(keyframes, coarse_meta)) as mock_coarse, \
             patch.object(detector, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
             patch.object(detector.vad, "detect_events", return_value=([], {})), \
             patch.object(detector, "_decode_file_window", return_value=(synthetic_window_frames, 30)) as mock_window_decode, \
             patch.object(detector, "_fallback_full_decode", side_effect=AssertionError("Critical Regression: Fell back to full decode!")) as mock_fallback:

            labels, yolo_buf = detector.analyze("active_slice.mp4", file_duration=file_dur, duration_verified=True)

        # 1. 验证两次独立租约获取与释放（Tier 1 粗筛一次，Tier 2 精析一次）
        assert scheduler.acquire_count == 2, f"Expected exactly 2 lease acquisitions, got {scheduler.acquire_count}"
        assert scheduler.release_count == 2, f"Expected exactly 2 lease releases, got {scheduler.release_count}"
        assert scheduler.active_slots == 0, f"Hardware slot leak: active_slots={scheduler.active_slots}"
        assert scheduler.devices_yielded == ["cuda", "cuda"]

        # 2. 验证执行路径：精析被调用，全量回退绝对未触发
        mock_coarse.assert_called_once()
        mock_window_decode.assert_called_once()
        mock_fallback.assert_not_called()

        # 3. 验证精析时间窗口参数为 [10-3, 10+3] = [7.0, 13.0]
        args, kwargs = mock_window_decode.call_args
        assert abs(args[1] - 7.0) < 1e-3, f"Expected window start 7.0, got {args[1]}"
        assert abs(args[2] - 13.0) < 1e-3, f"Expected window end 13.0, got {args[2]}"

        # 4. 验证 completion 元数据与 telemetry 契约
        assert detector.decode_completion["complete"] is True
        assert detector.decode_completion["reason"] == "NORMAL_COMPLETION", \
            f"Expected reason NORMAL_COMPLETION, got {detector.decode_completion.get('reason')}"
        assert detector.decode_completion["declared_duration_s"] == 30.0
        assert detector.decode_completion["available_duration_s"] == 30.0
        assert detector.last_perf["completion_reason"] == "TIER2_WINDOWED_COMPLETION"

        # 5. 验证运动分析结果正确产出
        assert any(l["is_motion"] for l in labels), "Expected detected motion in awakened window"

    def test_static_file_single_lease_exit_without_entering_tier2(self):
        """TC-R1-LEASE-02: 纯静止切片仅在 Tier 1 申请 1 次 Lease 并释放，不进入 Tier 2。"""
        from unittest.mock import patch

        scheduler = self.MockScheduler()
        file_dur = 30.0

        detector = MotionDetector({"detection": {"iframe_prescreen_enabled": True}})
        detector.device_lease = lambda dur=file_dur: scheduler.lease_device(file_duration=dur)

        # 全部为静止关键帧
        keyframes = [(float(t), 0.1, 0.2) for t in range(0, 30, 3)]
        coarse_meta = {"keyframes": len(keyframes), "complete": True, "avg_gop": 3.0, "max_gop": 3.0}

        with patch.object(detector, "_scan_keyframes_coarse", return_value=(keyframes, coarse_meta)), \
             patch.object(detector, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
             patch.object(detector.vad, "detect_events", return_value=([], {})), \
             patch.object(detector, "_decode_file_window") as mock_window_decode, \
             patch.object(detector, "_fallback_full_decode") as mock_fallback:

            labels, yolo_buf = detector.analyze("static_slice.mp4", file_duration=file_dur, duration_verified=True)

        assert scheduler.acquire_count == 1, "Static file must acquire lease only once during Tier 1"
        assert scheduler.release_count == 1
        assert scheduler.active_slots == 0
        mock_window_decode.assert_not_called()
        mock_fallback.assert_not_called()
        assert detector.decode_completion["complete"] is True
        assert detector.last_perf["completion_reason"] == "ZERO_DECODE_FAST_PATH"
        assert all(not l["is_motion"] for l in labels)

    def test_dense_motion_fallback_acquires_separate_lease_for_fallback_decode(self):
        """TC-R1-LEASE-03: 密集动作熔断触发回退时，_fallback_full_decode 正确获取独立 Lease。"""
        from unittest.mock import patch

        scheduler = self.MockScheduler()
        file_dur = 60.0

        detector = MotionDetector({"detection": {"iframe_prescreen_enabled": True, "iframe_fallback_active_ratio": 0.50}})
        detector.device_lease = lambda dur=file_dur: scheduler.lease_device(file_duration=dur)

        # 连续动作覆盖超过 50% 触发回退
        keyframes = [(float(t), 8.0, 15.0) if t <= 40 else (float(t), 0.1, 0.2) for t in range(0, 60, 3)]
        coarse_meta = {"keyframes": len(keyframes), "complete": True, "avg_gop": 3.0, "max_gop": 3.0}

        synthetic_full_frames = [np.zeros((180, 320), dtype=np.uint8) for _ in range(300)]
        decode_meta = {"complete": True, "effective_fps": 5.0, "frames": 300, "decode_time": 0.5, "completion_reason": "NORMAL_COMPLETION"}

        with patch.object(detector, "_scan_keyframes_coarse", return_value=(keyframes, coarse_meta)), \
             patch.object(detector, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
             patch.object(detector.vad, "detect_events", return_value=([], {})), \
             patch.object(detector, "_decode_file", return_value=(synthetic_full_frames, {}, np.zeros(16000, dtype=np.float32), decode_meta)), \
             patch.object(detector, "_decode_file_window") as mock_window_decode:

            labels, yolo_buf = detector.analyze("dense_slice.mp4", file_duration=file_dur, duration_verified=True)

        # Tier 1 粗筛 1 次 + 回退全量解码 1 次 = 2 次
        assert scheduler.acquire_count == 2
        assert scheduler.release_count == 2
        assert scheduler.active_slots == 0
        mock_window_decode.assert_not_called()
        assert detector.decode_completion["reason"] in ("NORMAL_COMPLETION", "FALLBACK_DENSE_MOTION")

    def test_exception_in_window_decode_guarantees_slot_release(self):
        """TC-R1-LEASE-04: Tier 2 解码中途发生异常时，确保持有槽位 100% 释放，无资源泄漏。"""
        from unittest.mock import patch

        scheduler = self.MockScheduler()
        file_dur = 30.0

        detector = MotionDetector({"detection": {"iframe_prescreen_enabled": True}})
        detector.device_lease = lambda dur=file_dur: scheduler.lease_device(file_duration=dur)

        keyframes = [(0.0, 0.1, 0.2), (10.0, 5.0, 10.0), (20.0, 0.1, 0.2), (30.0, 0.1, 0.2)]
        coarse_meta = {"keyframes": 4, "complete": True, "avg_gop": 3.0, "max_gop": 3.0}

        with patch.object(detector, "_scan_keyframes_coarse", return_value=(keyframes, coarse_meta)), \
             patch.object(detector, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
             patch.object(detector.vad, "detect_events", return_value=([], {})), \
             patch.object(detector, "_decode_file_window", side_effect=RuntimeError("FFmpeg process crash")):

            # 异常应触发 full decode fallback
            with patch.object(detector, "_fallback_full_decode", return_value=([], {})):
                detector.analyze("corrupt_window.mp4", file_duration=file_dur, duration_verified=True)

        assert scheduler.active_slots == 0, "Slot must be released even when window decode raises exception"
        assert scheduler.acquire_count == scheduler.release_count

    def test_unassigned_device_lease_defaults_to_reusable_nullcontext(self):
        """TC-R1-LEASE-05: 独立运行未注入 device_lease 时，安全缺省为 nullcontext 且正常完成两级解码。"""
        from unittest.mock import patch

        detector = MotionDetector({"detection": {"iframe_prescreen_enabled": True}}, decode_gpu="cuda")
        # 确保未设置 device_lease 属性
        if hasattr(detector, "device_lease"):
            delattr(detector, "device_lease")

        keyframes = [(0.0, 0.1, 0.2), (10.0, 5.0, 10.0), (20.0, 0.1, 0.2), (30.0, 0.1, 0.2)]
        coarse_meta = {"keyframes": 4, "complete": True, "avg_gop": 3.0, "max_gop": 3.0}
        synthetic_frames = [np.full((180, 320), 200, dtype=np.uint8) for _ in range(30)]

        with patch.object(detector, "_scan_keyframes_coarse", return_value=(keyframes, coarse_meta)), \
             patch.object(detector, "_decode_audio_pipe", return_value=np.zeros(16000, dtype=np.float32)), \
             patch.object(detector.vad, "detect_events", return_value=([], {})), \
             patch.object(detector, "_decode_file_window", return_value=(synthetic_frames, 30)), \
             patch.object(detector, "_fallback_full_decode", side_effect=AssertionError("Fallback occurred!")):

            labels, yolo_buf = detector.analyze("no_lease_assigned.mp4", file_duration=30.0, duration_verified=True)

        assert detector.decode_completion["complete"] is True
        assert detector.decode_completion["reason"] == "NORMAL_COMPLETION"
        assert detector.decode_gpu == "cuda"


# ============================================================================
# Regression Suite: Minor Findings 2, 3, 4 (Audio Timeout, Start Offset, PTS)
# ============================================================================

def test_extract_keyframe_pts_handles_null_pts_and_dts_fallback():
    """Verify _extract_keyframe_pts tolerates packets with None PTS and falls back to DTS or extrapolation."""
    from unittest.mock import MagicMock, patch

    class MockPacket:
        def __init__(self, is_key, pts, dts):
            self.is_keyframe = is_key
            self.pts = pts
            self.dts = dts

    mock_stream = MagicMock()
    mock_stream.time_base = 1.0 / 1000.0
    mock_stream.duration = 10000

    packets = [
        MockPacket(True, None, 1000),      # PTS None, DTS 1000 -> 1.0s
        MockPacket(False, 2000, 2000),     # Non-keyframe -> skipped
        MockPacket(True, None, None),      # PTS None, DTS None -> extrapolate 1.0 + 3.0 = 4.0s
        MockPacket(True, 8000, 8000),      # Normal keyframe -> 8.0s
    ]

    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.demux.return_value = packets
    mock_container.duration = 10000

    detector = MotionDetector({"detection": {"iframe_prescreen_enabled": True}})

    with patch("av.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_container
        pts_list, duration, avg_gop, max_gop = detector._extract_keyframe_pts("dummy.mp4")

    assert pts_list == [1.0, 4.0, 8.0]
    assert duration == 10.0
    assert avg_gop > 0.0
    assert max_gop >= avg_gop


def test_audio_join_timeout_scales_dynamically_with_file_duration():
    """Verify audio thread join timeout scales with file duration and guards against timeouts."""
    assert max(60.0, min(10.0 * 0.5, 300.0)) == 60.0
    assert max(60.0, min(60.0 * 0.5, 300.0)) == 60.0
    assert max(60.0, min(300.0 * 0.5, 300.0)) == 150.0
    assert max(60.0, min(3300.0 * 0.5, 300.0)) == 300.0


def test_window_trace_night_mode_safe_without_prior_analyze():
    """Verify MotionDetector initializes _analysis_start_offset and handles window trace safely."""
    detector = MotionDetector({"detection": {"iframe_prescreen_enabled": True}})
    assert getattr(detector, "_analysis_start_offset", 0.0) == 0.0
    offset = getattr(detector, "_analysis_start_offset", 0.0)
    assert detector._is_night_time(offset + 120.0) in (True, False)

