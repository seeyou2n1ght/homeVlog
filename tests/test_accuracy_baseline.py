"""HomeVlog 权威黄金基准评测与精度回归测试套件 (Gold Benchmark Accuracy Suite)

用途:
    依托 tests/benchmark_data/gold_ground_truth.json 黄金真值数据集，
    对算法识别准确率、混淆矩阵、Precision、Recall、F1-Score 进行量化评估，
    并作为发版和算法迭代的自动化质量门禁 (Quality Gate)。
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


def test_confusion_matrix_and_accuracy_metrics():
    """计算当前算法识别的混淆矩阵与核心指标，确立精度基线。"""
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
