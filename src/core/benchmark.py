"""HomeVlog 评测基准数据集管理与分层评估引擎 (Benchmark Manager)

负责黄金真值数据集 (Gold Ground Truth) 的版本化存储、增量合并去重、
指标度量 (Confusion Matrix / F1-Score) 以及按场景 (Scenario) 的细分评估。
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("homevlog.benchmark")

DEFAULT_BENCHMARK_FILE = (
    Path(__file__).resolve().parents[2] / "tests" / "benchmark_data" / "gold_ground_truth.json"
)
VERSIONS_DIR = DEFAULT_BENCHMARK_FILE.parent / "versions"


def normalize_benchmark_label(label: str | None) -> str:
    """标准化真值标签为四大基准类别之一。"""
    if not label:
        return "UNKNOWN"
    lbl = str(label).strip().upper()
    if lbl in ("CONFIRMED_MOTION", "VERIFIED_MOTION", "TP", "DYNAMIC"):
        return "CONFIRMED_MOTION"
    if lbl in ("FALSE_ALARM", "FP", "LIGHT_SHADOW", "CURTAIN"):
        return "FALSE_ALARM"
    if lbl in ("MISSED_MOTION", "FN", "INFANT_MOTION"):
        return "MISSED_MOTION"
    if lbl in ("CONFIRMED_STATIC", "TN", "STATIC", "EMPTY_ROOM"):
        return "CONFIRMED_STATIC"
    return lbl


def load_benchmark_dataset(file_path: Path | str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """加载基准评测数据集，向后兼容纯列表格式与带元数据头的字典格式。

    返回:
        (samples_list, metadata_dict)
    """
    path = Path(file_path) if file_path else DEFAULT_BENCHMARK_FILE
    if not path.exists():
        raise FileNotFoundError(f"基准数据集文件不存在: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    if isinstance(raw_data, list):
        samples = raw_data
        metadata = {
            "version": "1.0",
            "format": "legacy_list",
            "total_samples": len(samples),
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
    elif isinstance(raw_data, dict):
        samples = raw_data.get("samples", [])
        metadata = raw_data.get("metadata", {})
    else:
        raise ValueError(f"无效的基准数据结构: {type(raw_data)}")

    # 统一清洗和填充必要默认值
    for s in samples:
        s["gold_label"] = normalize_benchmark_label(s.get("gold_label") or s.get("manual_label"))
        if not s.get("scenario"):
            # 若无场景标记，尝试从 notes 或类别中推断
            notes = str(s.get("notes") or "").upper()
            if "光影" in notes or "树影" in notes or "HEADLIGHT" in notes:
                s["scenario"] = "LIGHT_SHADOW"
            elif "微动" in notes or "婴儿" in notes:
                s["scenario"] = "INFANT_MOTION"
            elif "空镜" in notes or "静止" in notes:
                s["scenario"] = "EMPTY_ROOM"
            else:
                s["scenario"] = "GENERAL"

    return samples, metadata


def save_benchmark_dataset(
    samples: list[dict[str, Any]],
    output_path: Path | str | None = None,
    version: str = "1.0",
    archive_version_copy: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """持久化基准数据集，支持元数据记录与历史版本快照归档。"""
    target_path = Path(output_path) if output_path else DEFAULT_BENCHMARK_FILE
    target_path.parent.mkdir(parents=True, exist_ok=True)

    distribution: dict[str, int] = {}
    scenario_dist: dict[str, int] = {}
    for s in samples:
        lbl = s.get("gold_label", "UNKNOWN")
        sc = s.get("scenario", "GENERAL")
        distribution[lbl] = distribution.get(lbl, 0) + 1
        scenario_dist[sc] = scenario_dist.get(sc, 0) + 1

    meta = {
        "version": version,
        "format": "homevlog_gold_benchmark_v2",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(samples),
        "distribution": distribution,
        "scenario_distribution": scenario_dist,
    }
    if extra_metadata:
        meta.update(extra_metadata)

    # 规范化纯净字典格式，同时将 samples 排列有序
    sorted_samples = sorted(
        samples,
        key=lambda x: (x.get("date", ""), x.get("cam_index", 0), x.get("start_time", 0.0))
    )

    # 写入主基准文件（保持直接为列表或带 metadata 的列表对象以兼顾既有代码）
    # 为保证所有历史测试用例直接 json.load() 即是 list，在 list 首项或直接保存为标准 list 格式：
    # 既有代码 tests/test_accuracy_baseline.py 期望直接得到 list，因此写入标准的 samples 列表
    # 但每个 sample 包含标准字段，并在同一目录保存伴随元数据 .meta.json
    with open(target_path, "w", encoding="utf-8") as f:
        json.dump(sorted_samples, f, ensure_ascii=False, indent=2)

    meta_file = target_path.with_name(target_path.stem + ".meta.json")
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # 历史版本快照归档
    if archive_version_copy:
        VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
        v_stem = f"gold_ground_truth_v{version}"
        v_file = VERSIONS_DIR / f"{v_stem}.json"
        shutil.copy2(target_path, v_file)
        v_meta = VERSIONS_DIR / f"{v_stem}.meta.json"
        shutil.copy2(meta_file, v_meta)
        logger.info("基准版本已归档至: %s", v_file)

    logger.info("基准数据集成功保存: %s (样本数: %d)", target_path, len(samples))
    return target_path


def merge_benchmark_samples(
    base_samples: list[dict[str, Any]],
    new_samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """增量合并两批基准样本，依据 segment_id 或 (filepath, start_time) 权威去重更新。"""
    merged_map: dict[str, dict[str, Any]] = {}

    def get_key(s: dict[str, Any]) -> str:
        if s.get("segment_id"):
            return f"id:{s['segment_id']}"
        fp = s.get("filepath") or s.get("filename") or ""
        st = round(float(s.get("start_time", 0.0)), 2)
        return f"loc:{fp}_{st}"

    for s in base_samples:
        k = get_key(s)
        merged_map[k] = dict(s)

    for s in new_samples:
        k = get_key(s)
        if k in merged_map:
            # 增量更新已有人工标注字段
            merged_map[k].update(s)
        else:
            merged_map[k] = dict(s)

    return list(merged_map.values())


def evaluate_confusion_matrix(
    samples: list[dict[str, Any]],
    is_active_decision_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, Any]:
    """计算样本集的混淆矩阵、Precision、Recall、F1-Score 与 Accuracy。

    参数:
        samples: 样本列表
        is_active_decision_fn: 判断算法判定是否为动作的函数。默认为判断 algo_state in ("DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE")
    """
    tp = 0
    fp = 0
    fn = 0
    tn = 0

    if is_active_decision_fn is None:
        def default_decision(s: dict[str, Any]) -> bool:
            return s.get("algo_state") in ("DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE")
        is_active_decision_fn = default_decision

    for s in samples:
        gold = normalize_benchmark_label(s.get("gold_label"))
        is_active = bool(is_active_decision_fn(s))

        if gold == "CONFIRMED_MOTION":
            if is_active:
                tp += 1
            else:
                fn += 1
        elif gold == "FALSE_ALARM":
            if is_active:
                fp += 1
            else:
                tn += 1
        elif gold == "MISSED_MOTION":
            if is_active:
                tp += 1
            else:
                fn += 1
        elif gold == "CONFIRMED_STATIC":
            if is_active:
                fp += 1
            else:
                tn += 1

    total = tp + fp + fn + tn
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = ((tp + tn) / total) * 100 if total > 0 else 0.0

    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "f1": round(f1, 2),
        "accuracy": round(accuracy, 2),
    }


def evaluate_stratified_metrics(
    samples: list[dict[str, Any]],
    group_key: str = "scenario",
    is_active_decision_fn: Callable[[dict[str, Any]], bool] | None = None,
) -> dict[str, dict[str, Any]]:
    """按场景 (scenario) 或其他分组字段计算切片分层指标。"""
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in samples:
        k = str(s.get(group_key) or "GENERAL")
        groups.setdefault(k, []).append(s)

    results: dict[str, dict[str, Any]] = {}
    for grp_name, grp_samples in groups.items():
        results[grp_name] = evaluate_confusion_matrix(
            grp_samples, is_active_decision_fn=is_active_decision_fn
        )

    return results
