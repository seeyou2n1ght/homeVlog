"""Historical dataset contracts, current replay gating, and review precedence.

The legacy 100 labels are unverified hypotheses, not release truth. Static
algo_state statistics do not exercise today's detector.
"""

import json
from pathlib import Path
import pytest

from src.core.database import VlogDatabase
from src.timeline import build_timeline

BENCHMARK_FILE = Path(__file__).resolve().parent / "benchmark_data" / "gold_ground_truth.json"


def test_gold_benchmark_schema_and_distribution():
    """验证黄金基准评测数据集的存在性、字段契约与类别分布均衡性。"""
    assert BENCHMARK_FILE.exists(), f"黄金基准文件未找到: {BENCHMARK_FILE}"

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 规模门禁: 至少有 50 个高代表性样本
    assert len(data) >= 50, f"黄金基准样本量不足 ({len(data)} < 50)"

    required_keys = {
        "segment_id", "filepath", "filename", "start_time", "end_time",
        "duration", "algo_state", "max_energy", "gold_label"
    }

    labels = set()
    for item in data:
        # 2. 字段完整性
        for k in required_keys:
            assert k in item, f"样本 {item.get('segment_id')} 缺失字段: {k}"

        assert item["end_time"] >= item["start_time"], f"样本时序逆转: {item}"
        labels.add(item["gold_label"])

    # 3. 必须覆盖全部四大基准分类
    expected_categories = {"CONFIRMED_MOTION", "FALSE_ALARM", "MISSED_MOTION", "CONFIRMED_STATIC"}
    assert expected_categories.issubset(labels), f"基准分类覆盖不全: 缺少 {expected_categories - labels}"


def test_historical_confusion_matrix_arithmetic():
    """仅验证历史记录的统计计算，不代表当前算法识别质量。"""
    assert BENCHMARK_FILE.exists()

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    tp = 0
    fp = 0
    fn = 0
    tn = 0

    for item in data:
        algo = item["algo_state"]
        gold = item["gold_label"]
        is_algo_active = algo in ("DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE")

        if gold == "CONFIRMED_MOTION":
            if is_algo_active:
                tp += 1
            else:
                fn += 1
        elif gold == "FALSE_ALARM":
            if is_algo_active:
                fp += 1
            else:
                tn += 1
        elif gold == "MISSED_MOTION":
            if not is_algo_active:
                fn += 1
            else:
                tp += 1
        elif gold == "CONFIRMED_STATIC":
            if not is_algo_active:
                tn += 1
            else:
                fp += 1

    total = tp + fp + fn + tn
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    print("\n" + "=" * 50)
    print("      HomeVlog Gold Benchmark Report")
    print("=" * 50)
    print(f" Total Benchmark Samples: {total}")
    print(f" Confusion Matrix: TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f" Precision: {precision:.2f}%")
    print(f" Recall:    {recall:.2f}%")
    print(f" F1-Score:  {f1:.2f}%")
    print("=" * 50)

    # 4. 验证基准数据集的识别判据具备有效区分度
    assert total == len(data)
    assert tp + fn > 0, "正样本总数不能为 0"
    assert tn + fp > 0, "负样本总数不能为 0"


def test_historical_scenario_distribution():
    """历史分层数据可以汇总，但未核验的标签不能通过质量门禁。"""
    from src.core.benchmark import load_benchmark_dataset, evaluate_stratified_metrics

    samples, meta = load_benchmark_dataset(BENCHMARK_FILE)
    assert len(samples) >= 50
    assert meta.get("version"), "基准元数据必须包含版本标识"

    stratified = evaluate_stratified_metrics(samples, group_key="scenario")

    print("\n" + "=" * 55)
    print("      HomeVlog Stratified Scenario Quality Report")
    print("=" * 55)
    for sc, metrics in sorted(stratified.items()):
        print(f" [{sc:14s}] N={metrics['total']:2d} | TP={metrics['tp']:2d} FP={metrics['fp']:2d} FN={metrics['fn']:2d} TN={metrics['tn']:2d} | F1={metrics['f1']:5.1f}%")
    print("=" * 55)

    assert "LIGHT_SHADOW" in stratified, "基准集必须覆盖光影误报挑战集"
    assert "INFANT_MOTION" in stratified, "基准集必须覆盖微动漏报挑战集"
    from src.core.benchmark import assess_replay_quality
    gate = assess_replay_quality(samples, [True] * len(samples))
    assert gate["verdict"] == "UNAVAILABLE"
    assert gate["verified_samples"] == 0


def test_current_predictions_enforce_all_four_quality_limits():
    from src.core.benchmark import assess_replay_quality
    samples = [{"scenario": scenario, "reference_active": active,
                "review_status": "independent_visual", "review_evidence": "controlled reference",
                "split": "holdout", "filepath": f"{scenario}.mp4"}
               for scenario, active in (("ADULT_MOTION", True), ("INFANT_MOTION", True),
                                         ("EMPTY_ROOM", False), ("LIGHT_SHADOW", False))]
    assert assess_replay_quality(samples, [True, True, False, False], min_per_scenario=1)["verdict"] == "PASS"
    for predictions, failed in (([False, False, False, False], ["ADULT_MOTION", "INFANT_MOTION"]),
                                ([True, True, True, True], ["EMPTY_ROOM", "LIGHT_SHADOW"])):
        gate = assess_replay_quality(samples, predictions, min_per_scenario=1)
        assert gate["verdict"] == "FAIL"
        assert gate["failed_scenarios"] == failed


@pytest.mark.parametrize('variant', ['tuning', 'missing_split', 'used', 'same_source', 'same_event', 'duplicates'])
def test_quality_gate_rejects_leaked_or_correlated_reference_counts(variant):
    from src.core.benchmark import assess_replay_quality
    samples, predictions = [], []
    for scenario, active in [('ADULT_MOTION', True), ('INFANT_MOTION', True), ('EMPTY_ROOM', False), ('LIGHT_SHADOW', False)]:
        for index in range(10):
            row = dict(scenario=scenario, reference_active=active, review_status='independent_visual',
                review_evidence='controlled fixture', split='holdout', filepath=f'{scenario}_{index}.mp4',
                start_time=index*10, end_time=index*10+6)
            if variant == 'tuning': row['split'] = 'tuning'
            elif variant == 'missing_split': row.pop('split')
            elif variant == 'used': row['used_for_tuning'] = True
            elif variant == 'same_source': row['filepath'] = f'{scenario}.mp4'
            elif variant == 'same_event': row['independence_group'] = 'one event'
            elif variant == 'duplicates': row.update(filepath=f'{scenario}.mp4', start_time=0, end_time=6)
            samples.append(row)
            predictions.append(active)
    gate = assess_replay_quality(samples, predictions)
    assert gate['verdict'] == 'UNAVAILABLE'
    assert len(gate['insufficient_scenarios']) == 4
    assert all(value <= 1 for value in gate['independent_counts'].values())
    if variant == 'duplicates':
        assert gate['verified_samples'] == 4 and gate['excluded_samples']['duplicates'] == 36


def test_quality_gate_accepts_distinct_holdout_sources_and_excludes_conflicts():
    from src.core.benchmark import assess_replay_quality
    samples, predictions = [], []
    for scenario, active in [('ADULT_MOTION', True), ('INFANT_MOTION', True), ('EMPTY_ROOM', False), ('LIGHT_SHADOW', False)]:
        for index in range(10):
            samples.append(dict(scenario=scenario, reference_active=active, review_status='independent_human',
                review_evidence='controlled fixture', split='holdout', filepath=f'{scenario}_{index}.mp4'))
            predictions.append(active)
    assert assess_replay_quality(samples, predictions)['verdict'] == 'PASS'
    samples.append(dict(samples[0], reference_active=False))
    predictions.append(False)
    gate = assess_replay_quality(samples, predictions)
    assert gate['verdict'] == 'UNAVAILABLE'
    assert gate['independent_counts']['ADULT_MOTION'] == 9
    assert gate['excluded_samples']['conflicts'] == 1


@pytest.mark.parametrize('active', [True, False])
def test_quality_gate_weights_events_instead_of_easy_window_count(active):
    from src.core.benchmark import assess_replay_quality
    samples, predictions = [], []
    for scenario, truth in [('ADULT_MOTION', True), ('INFANT_MOTION', True), ('EMPTY_ROOM', False), ('LIGHT_SHADOW', False)]:
        for index in range(10):
            row = dict(scenario=scenario, reference_active=truth, review_status='independent_human',
                review_evidence='controlled fixture', split='holdout', filepath=f'{scenario}_{index}.mp4',
                start_time=0, end_time=6)
            samples.append(row)
            predictions.append(not truth if truth == active and index else truth)
        if truth == active:
            easy = dict(samples[-10])
            for index in range(1, 301):
                samples.append(dict(easy, start_time=index*10, end_time=index*10+6))
                predictions.append(truth)
    gate = assess_replay_quality(samples, predictions)
    assert gate['verdict'] == 'FAIL'
    for scenario in (['ADULT_MOTION', 'INFANT_MOTION'] if active else ['EMPTY_ROOM', 'LIGHT_SHADOW']):
        assert gate['independent_counts'][scenario] == 10
        assert scenario in gate['failed_scenarios']


@pytest.mark.parametrize('interval', [(None, 6), (0, float('nan')), (False, 6), (6, 0), ([0], 6)])
def test_quality_gate_excludes_invalid_interval_identity(interval):
    from src.core.benchmark import assess_replay_quality
    row = dict(scenario='ADULT_MOTION', reference_active=True, review_status='independent_visual',
        review_evidence='fixture', split='holdout', filepath='source.mp4', start_time=interval[0], end_time=interval[1])
    gate = assess_replay_quality([row], [True], min_per_scenario=1)
    assert gate['verified_samples'] == 0 and gate['excluded_samples']['invalid_interval'] == 1


def test_human_feedback_remedy_guarantee(tmp_path):
    """验证人机协同复核机制对争议切片具有 100% 纠偏保护力 (Timeline Invariant)."""
    from src.segment import Segment

    db = VlogDatabase(tmp_path / "remedy.sqlite")
    fp = "sample_clip.mp4"
    db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901101000", 600)

    # 模拟 1 个被误判为 DYNAMIC 的光影假阳性切片
    seg = Segment(10.0, 30.0, "DYNAMIC", fp, max_energy=18.0, avg_confidence=0.0)
    db.set_analysis_result(fp, [seg])

    # 初始成片时间轴: 该切片处于 DYNAMIC 原速
    initial_tl = build_timeline(db, "20260901", 0)
    assert any(t.state == "DYNAMIC" for t in initial_tl)

    # 模拟人工工作台将其标记为 FALSE_ALARM (光影误报)
    db_segs = db.get_segments_for_file(fp)
    seg_id = db_segs[0]["id"]
    db.update_segment_review(seg_id, "FALSE_ALARM", notes="树影晃动纠偏")

    # 复核后成片时间轴: build_timeline 必须保证纠偏生效，降级为 STATIC 快进
    corrected_tl = build_timeline(db, "20260901", 0)
    assert not any(t.state == "DYNAMIC" for t in corrected_tl)
    assert all(t.state == "STATIC" for t in corrected_tl)

    db.close()

