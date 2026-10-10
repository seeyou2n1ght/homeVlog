"""测试基准数据集管理引擎与 What-If 调参仿真器 (Benchmark & Tuner Test Suite)"""

import json
from pathlib import Path
import pytest
import yaml

from src.core.benchmark import (
    load_benchmark_dataset,
    save_benchmark_dataset,
    merge_benchmark_samples,
    evaluate_confusion_matrix,
    evaluate_stratified_metrics,
    normalize_benchmark_label,
)
from src.core.tuner import (
    TuningSimulator,
    generate_config_diff,
    apply_config_safe,
    rollback_config,
)


@pytest.fixture
def sample_benchmark():
    return [
        {
            "segment_id": 1,
            "filepath": "cam1/clip1.mp4",
            "start_time": 0.0,
            "end_time": 10.0,
            "duration": 10.0,
            "algo_state": "DYNAMIC",
            "max_energy": 25.0,
            "avg_confidence": 0.85,
            "gold_label": "CONFIRMED_MOTION",
            "scenario": "ADULT_MOTION",
        },
        {
            "segment_id": 2,
            "filepath": "cam1/clip2.mp4",
            "start_time": 10.0,
            "end_time": 15.0,
            "duration": 5.0,
            "algo_state": "DYNAMIC",
            "max_energy": 3.0,
            "avg_confidence": 0.0,
            "gold_label": "FALSE_ALARM",
            "scenario": "LIGHT_SHADOW",
        },
        {
            "segment_id": 3,
            "filepath": "cam1/clip3.mp4",
            "start_time": 20.0,
            "end_time": 25.0,
            "duration": 5.0,
            "algo_state": "STATIC",
            "max_energy": 2.2,
            "avg_confidence": 0.0,
            "gold_label": "MISSED_MOTION",
            "scenario": "INFANT_MOTION",
        },
        {
            "segment_id": 4,
            "filepath": "cam1/clip4.mp4",
            "start_time": 30.0,
            "end_time": 40.0,
            "duration": 10.0,
            "algo_state": "STATIC",
            "max_energy": 0.5,
            "avg_confidence": 0.0,
            "gold_label": "CONFIRMED_STATIC",
            "scenario": "EMPTY_ROOM",
        },
    ]


def test_label_normalization():
    assert normalize_benchmark_label("tp") == "CONFIRMED_MOTION"
    assert normalize_benchmark_label("false_alarm") == "FALSE_ALARM"
    assert normalize_benchmark_label("fn") == "MISSED_MOTION"
    assert normalize_benchmark_label("tn") == "CONFIRMED_STATIC"
    assert normalize_benchmark_label("light_shadow") == "FALSE_ALARM"


def test_confusion_matrix_and_stratified(sample_benchmark):
    metrics = evaluate_confusion_matrix(sample_benchmark)
    assert metrics["total"] == 4
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision"] == 50.0
    assert metrics["recall"] == 50.0
    assert metrics["f1"] == 50.0

    stratified = evaluate_stratified_metrics(sample_benchmark, group_key="scenario")
    assert "ADULT_MOTION" in stratified
    assert stratified["ADULT_MOTION"]["tp"] == 1
    assert stratified["LIGHT_SHADOW"]["fp"] == 1
    assert stratified["INFANT_MOTION"]["fn"] == 1
    assert stratified["EMPTY_ROOM"]["tn"] == 1


def test_save_and_load_benchmark_dataset(sample_benchmark, tmp_path):
    out_file = tmp_path / "test_gold.json"
    save_benchmark_dataset(sample_benchmark, output_path=out_file, version="1.1", archive_version_copy=False)

    assert out_file.exists()
    meta_file = out_file.with_name(out_file.stem + ".meta.json")
    assert meta_file.exists()

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["version"] == "1.1"
    assert meta["total_samples"] == 4

    loaded_samples, loaded_meta = load_benchmark_dataset(out_file)
    assert len(loaded_samples) == 4
    assert loaded_samples[0]["segment_id"] in [1, 2, 3, 4]


def test_merge_benchmark_samples(sample_benchmark):
    new_samples = [
        # 更新已存在的样本 2
        {
            "segment_id": 2,
            "filepath": "cam1/clip2.mp4",
            "gold_label": "CONFIRMED_STATIC",
            "notes": "修正为纯静止",
        },
        # 新增样本 5
        {
            "segment_id": 5,
            "filepath": "cam1/clip5.mp4",
            "start_time": 50.0,
            "end_time": 60.0,
            "duration": 10.0,
            "algo_state": "DYNAMIC",
            "gold_label": "CONFIRMED_MOTION",
        },
    ]

    merged = merge_benchmark_samples(sample_benchmark, new_samples)
    assert len(merged) == 5

    s2 = next(s for s in merged if s["segment_id"] == 2)
    assert s2["gold_label"] == "CONFIRMED_STATIC"
    assert s2["notes"] == "修正为纯静止"


def test_tuning_simulator_what_if(sample_benchmark):
    config = {
        "detection": {"min_motion_threshold": 2.5},
        "segment": {"min_motion_duration": 2.0},
        "yolo": {"confidence": 0.3},
        "presence": {"person_conf_threshold": 0.20},
    }
    sim = TuningSimulator(benchmark_samples=sample_benchmark, config=config)

    # 模拟调高阈值至 3.5 (样本 2 动能为 3.0，将被过滤，消除误报 FP -> TN)
    res = sim.simulate({"detection.min_motion_threshold": 3.5})

    assert res["baseline"]["fp"] == 1
    assert res["simulated"]["fp"] == 0
    assert res["deltas"]["fp_reduction"] == 1
    assert res["verdict"] in ("IMPROVED", "FAVORABLE_TRADEOFF")


def test_config_diff_and_safe_apply(tmp_path):
    config_data = {
        "detection": {
            "min_motion_threshold": 2.5,
            "analysis_resolution": "416x234",
            "prescreen_parallel": 4,
            "analysis_max_workers": 4,
        },
        "render": {
            "batch_max_files": 1,
            "max_concurrency": 1,
        },
        "segment": {
            "min_motion_duration": 2.0,
        },
    }
    cfg_file = tmp_path / "settings.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    # 1. 验证 Diff 生成
    diff = generate_config_diff({"detection.min_motion_threshold": 3.2}, config_path=cfg_file)
    assert diff["total_changes"] == 1
    assert diff["diffs"][0]["current"] == 2.5
    assert diff["diffs"][0]["proposed"] == 3.2

    # 2. 验证安全写入与备份
    apply_res = apply_config_safe(
        {"detection.min_motion_threshold": 3.2},
        config_path=cfg_file,
        create_backup=True,
    )
    assert apply_res["success"] is True
    backup_file = Path(apply_res["backup_path"])
    assert backup_file.exists()

    with open(cfg_file, "r", encoding="utf-8") as f:
        new_cfg = yaml.safe_load(f)
    assert new_cfg["detection"]["min_motion_threshold"] == 3.2

    # 3. 验证回滚
    rollback_res = rollback_config(backup_file, config_path=cfg_file)
    assert rollback_res["success"] is True

    with open(cfg_file, "r", encoding="utf-8") as f:
        restored_cfg = yaml.safe_load(f)
    assert restored_cfg["detection"]["min_motion_threshold"] == 2.5
