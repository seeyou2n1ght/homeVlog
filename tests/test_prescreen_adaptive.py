"""tests/test_prescreen_adaptive.py

验证预筛选阶段自适应动态阈值、4x4网格空间集中度算子、时间邻域先验保护
以及 tune_thresholds 超参数寻优工具的逻辑准确性。
"""

import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.prescreen import (
    _calc_dynamic_threshold,
    _calc_spatial_concentration,
    _prescreen_keyframes,
    prescreen_file,
)
from scripts.tune_thresholds import (
    PrescreenSample,
    evaluate_grid,
    run_grid_search,
    simulate_prediction,
)


def test_calc_dynamic_threshold():
    base_th = 8.0

    # 1. 正常白天光照 (mean_luma=100) -> 保持基准阈值
    th_day = _calc_dynamic_threshold(base_th, mean_luma=100.0, is_prior_active=False)
    assert th_day == 8.0

    # 2. 暗光/夜视微光 (mean_luma=30) -> 下调至 40% (3.20)
    th_night = _calc_dynamic_threshold(base_th, mean_luma=30.0, is_prior_active=False)
    assert th_night == 3.20

    # 3. 极暗光保底 (mean_luma=5.0, base_th=3.0) -> 下限保底 2.0
    th_min = _calc_dynamic_threshold(3.0, mean_luma=5.0, is_prior_active=False)
    assert th_min == 2.0

    # 4. 逆光/强光噪点密集 (mean_luma=200.0) -> 适度上浮 25% (10.0)
    th_bright = _calc_dynamic_threshold(base_th, mean_luma=200.0, is_prior_active=False)
    assert th_bright == 10.0

    # 5. 时间邻域先验保护生效 (is_prior_active=True) -> 下调 35%
    th_prior_day = _calc_dynamic_threshold(base_th, mean_luma=100.0, is_prior_active=True)
    assert th_prior_day == round(8.0 * 0.65, 2)  # 5.20

    th_prior_night = _calc_dynamic_threshold(base_th, mean_luma=30.0, is_prior_active=True)
    assert th_prior_night == round(3.20 * 0.65, 2)  # 2.08

    # 6. 先验保护下限保底 1.8
    th_prior_floor = _calc_dynamic_threshold(2.0, mean_luma=20.0, is_prior_active=True)
    assert th_prior_floor == 1.8


def test_spatial_concentration_diffuse_vs_localized():
    # 1. 全画幅弥散性光影跳变 (整图均匀变化)
    diffuse_map = np.full((180, 320), fill_value=10.0, dtype=np.float32)
    conc_diffuse = _calc_spatial_concentration(diffuse_map)
    # 所有 4x4 网格均值相同，集中度比值应为 1.0
    assert abs(conc_diffuse - 1.0) < 1e-3

    # 2. 局部高能前景运动 (仅左上角一个网格单元剧烈变化，其余全为 0)
    local_map = np.zeros((180, 320), dtype=np.float32)
    local_map[0:45, 0:80] = 50.0  # 局部能量密集
    conc_local = _calc_spatial_concentration(local_map)
    # 空间集中度应显著高于 1.35
    assert conc_local > 3.0


def test_prescreen_night_motion_not_dropped_as_static():
    """测试暗光下 max_diff 处于 (夜视阈值, 全局阈值) 区间的微弱运动不会在收尾时被误判为 STATIC"""
    # 模拟视频帧数据：两帧均为暗光（均值约 25），前后帧绝对差分均值 4.5
    # 若无动态阈值贯穿首尾判定，4.5 < 8.0 会被当场漏检为 STATIC
    fake_frame1 = np.full((180, 320), fill_value=25, dtype=np.uint8)
    fake_frame2 = np.full((180, 320), fill_value=25, dtype=np.uint8)
    # 在 1/4 画面制造高能前景微动 (均值差分 5.0, 介于夜视阈值 3.2 与全局阈值 8.0 之间)
    fake_frame2[0:90, 0:160] = 45

    # 构造 PyAV mock 解码
    mock_av_frame1 = MagicMock()
    mock_av_frame1.planes = [fake_frame1.tobytes()]
    mock_av_frame1.width = 320
    mock_av_frame1.height = 180
    mock_av_frame1.pts = 0

    mock_av_frame2 = MagicMock()
    mock_av_frame2.planes = [fake_frame2.tobytes()]
    mock_av_frame2.width = 320
    mock_av_frame2.height = 180
    mock_av_frame2.pts = 1000

    mock_stream = MagicMock()
    mock_stream.time_base = 0.001
    mock_stream.codec_context = MagicMock()

    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.streams.audio = []
    mock_container.decode.return_value = [mock_av_frame1, mock_av_frame2]

    with patch("av.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_container

        res = _prescreen_keyframes(
            filepath="dummy.mp4",
            duration=60.0,
            max_keyframes=5,
            threshold=8.0,
            is_prior_active=False,
        )

        assert res["status"] == "SUSPICIOUS"
        result_data = json.loads(res["result_json"])
        assert result_data["mean_luma"] < 50.0
        # 验证使用的判定阈值已自适应下调 (8.0 * 0.4 = 3.2)
        assert result_data["threshold"] == 3.2


def test_prescreen_diffuse_light_suppressed_as_static():
    """测试全画幅均匀弥散性光影跳变（如傍晚曝光变化）被空间集中度算子正确抑制，判定为 STATIC"""
    # 模拟两帧全图均匀变暗或变亮（全图差分均为 9.0，高于基础阈值 8.0，但无局部高能）
    fake_frame1 = np.full((180, 320), fill_value=120, dtype=np.uint8)
    fake_frame2 = np.full((180, 320), fill_value=129, dtype=np.uint8)  # 全图均差 9.0

    mock_av_frame1 = MagicMock()
    mock_av_frame1.planes = [fake_frame1.tobytes()]
    mock_av_frame1.width = 320
    mock_av_frame1.height = 180
    mock_av_frame1.pts = 0

    mock_av_frame2 = MagicMock()
    mock_av_frame2.planes = [fake_frame2.tobytes()]
    mock_av_frame2.width = 320
    mock_av_frame2.height = 180
    mock_av_frame2.pts = 1000

    mock_stream = MagicMock()
    mock_stream.time_base = 0.001
    mock_stream.codec_context = MagicMock()

    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.streams.audio = []
    mock_container.decode.return_value = [mock_av_frame1, mock_av_frame2]

    with patch("av.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_container

        res = _prescreen_keyframes(
            filepath="dusk_diffuse.mp4",
            duration=60.0,
            max_keyframes=5,
            threshold=8.0,
            is_prior_active=False,
        )

        assert res["status"] == "STATIC"
        result_data = json.loads(res["result_json"])
        assert result_data["early_stop"] is False
        assert result_data["max_diff"] == 9.0
        assert result_data["concentration"] < 1.2


def test_tune_thresholds_simulation_and_grid_search():
    """验证超参数寻优算法对暗光漏检和强光误报样本的正确打分与最优推荐"""
    samples = [
        # 样本1: 暗光真实人体移动 (luma 28, max_diff 4.0, 集中度高) -> 真值 MOTION
        PrescreenSample(
            file_id=1,
            filepath="night_motion.mp4",
            ground_truth="MOTION",
            max_diff=4.0,
            mean_luma=28.0,
            concentration=1.55,
            diffs=[1.0, 4.0],
        ),
        # 样本2: 傍晚光照渐变 (luma 120, max_diff 9.5, 但全图弥漫 concentration 1.05) -> 真值 STATIC
        PrescreenSample(
            file_id=2,
            filepath="dusk_light_shift.mp4",
            ground_truth="STATIC",
            max_diff=9.5,
            mean_luma=120.0,
            concentration=1.05,
            diffs=[2.0, 9.5],
        ),
        # 样本3: 白天正常人物穿行 (luma 130, max_diff 14.0, concentration 1.6) -> 真值 MOTION
        PrescreenSample(
            file_id=3,
            filepath="day_motion.mp4",
            ground_truth="MOTION",
            max_diff=14.0,
            mean_luma=130.0,
            concentration=1.6,
            diffs=[1.0, 14.0],
        ),
    ]

    # 1. 验证单一预测：
    # 当 base_threshold=8.0, night_factor=0.4 时：
    # 样本1 的有效阈值是 8.0*0.4=3.2，4.0 > 3.2 且 concentration=1.55 >= 1.35 -> 识别为 MOTION (TP)
    pred_sample1 = simulate_prediction(samples[0], base_threshold=8.0, night_factor=0.4, concentration_thresh=1.35)
    assert pred_sample1 == "MOTION"

    # 若未开启暗光自适应 (night_factor=1.0)：
    # 样本1 的阈值为 8.0，4.0 < 8.0 -> 误判为 STATIC (FN 漏检！)
    pred_sample1_rigid = simulate_prediction(samples[0], base_threshold=8.0, night_factor=1.0, concentration_thresh=1.35)
    assert pred_sample1_rigid == "STATIC"

    # 2. 验证网格评估打分：
    eval_adaptive = evaluate_grid(samples, base_threshold=8.0, night_factor=0.4, concentration_thresh=1.35)
    eval_rigid = evaluate_grid(samples, base_threshold=8.0, night_factor=1.0, concentration_thresh=1.35)

    # 开启自适应时 0 漏检 (fn=0, recall=1.0)
    assert eval_adaptive.fn == 0
    assert eval_adaptive.recall == 1.0

    # 未开启自适应时有 1 处严重漏检 (fn=1)
    assert eval_rigid.fn == 1

    # 成本打分对比：自适应方案显著优于僵化方案
    assert eval_adaptive.cost_score < eval_rigid.cost_score

    # 3. 验证全网格搜索能够推选出正确的最优解
    results = run_grid_search(samples, base_thresholds=[6.0, 8.0, 10.0], night_factors=[0.4, 0.9])
    best = results[0]
    assert best.night_factor == 0.4
    assert best.fn == 0
