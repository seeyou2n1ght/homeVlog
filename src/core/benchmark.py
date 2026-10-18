"""HomeVlog 评测基准数据集管理与分层评估引擎 (Benchmark Manager)

负责黄金真值数据集 (Gold Ground Truth) 的版本化存储、增量合并去重、
指标度量 (Confusion Matrix / F1-Score) 以及按场景 (Scenario) 的细分评估。
"""

import json
import hashlib
import logging
import math
import os
import re
import tempfile
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("homevlog.benchmark")

DEFAULT_BENCHMARK_FILE = (
    Path(__file__).resolve().parents[2] / "tests" / "benchmark_data" / "gold_ground_truth.json"
)
VERSIONS_DIR = DEFAULT_BENCHMARK_FILE.parent / "versions"
_dataset_lock = threading.RLock()


def assess_replay_quality(samples: list[dict], predictions: list[bool], *, min_per_scenario: int = 10) -> dict:
    """Gate current predictions, refusing historical or unreviewed references.

    Explicit holdout references only; tuning evidence remains diagnostic.
    Recall >=90% for adult and infant activity; specificity >=95% for empty
    rooms and light/shadow negatives. Count and score independent event groups
    conservatively; absent event metadata uses one group per source file.
    This is sample-level acceptance, not full-day temporal recall.
    """
    if min_per_scenario < 1:
        raise ValueError("min_per_scenario must be positive")
    if len(samples) != len(predictions) or any(type(p) is not bool for p in predictions):
        raise ValueError("One current prediction is required per sample")
    admitted = {}
    conflicts = set()
    excluded = {"unreviewed": 0, "not_holdout": 0, "used_for_tuning": 0,
                "missing_source": 0, "invalid_interval": 0, "duplicates": 0, "conflicts": 0}
    for sample, prediction in zip(samples, predictions):
        if (sample.get("review_status") not in ("independent_human", "independent_visual")
                or not isinstance(sample.get("reference_active"), bool) or not sample.get("review_evidence")):
            excluded["unreviewed"] += 1
            continue
        if sample.get("split") != "holdout":
            excluded["not_holdout"] += 1
            continue
        if sample.get("used_for_tuning") not in (None, False):
            excluded["used_for_tuning"] += 1
            continue
        filepath = sample.get("filepath")
        if not isinstance(filepath, str) or not filepath.strip():
            excluded["missing_source"] += 1
            continue
        source = os.path.normcase(os.path.normpath(filepath))
        declared = sample.get("independence_group")
        group = ("declared", declared.strip()) if isinstance(declared, str) and declared.strip() else ("source", source)
        start, end = sample.get("start_time"), sample.get("end_time")
        if (start is not None or end is not None) and (
                any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end))
                or end <= start):
            excluded["invalid_interval"] += 1
            continue
        key = (source, start, end)
        signature = (sample.get("scenario"), sample["reference_active"], prediction, group)
        if key in admitted:
            excluded["duplicates"] += 1
            if admitted[key][0] != signature:
                conflicts.add(key)
            continue
        admitted[key] = (signature, dict(sample, replay_active=prediction, quality_independence_group=group,
            gold_label="CONFIRMED_MOTION" if sample["reference_active"] else "CONFIRMED_STATIC"))
    excluded["conflicts"] = len(conflicts)
    verified = [value[1] for key, value in admitted.items() if key not in conflicts]
    metrics = evaluate_stratified_metrics(verified, group_key="scenario",
                                          is_active_decision_fn=lambda sample: sample["replay_active"])
    groups = {}
    for sample in verified:
        group_key = (sample.get("scenario"), sample["quality_independence_group"], sample["reference_active"])
        groups.setdefault(group_key, []).append(sample)
    independent = []
    for (scenario, _, active), members in groups.items():
        # A long, repeatedly sampled easy event cannot hide missed hard events.
        decision = all(s["replay_active"] for s in members) if active else any(s["replay_active"] for s in members)
        independent.append(dict(scenario=scenario, replay_active=decision,
            gold_label="CONFIRMED_MOTION" if active else "CONFIRMED_STATIC"))
    independent_metrics = evaluate_stratified_metrics(independent, group_key="scenario",
        is_active_decision_fn=lambda sample: sample["replay_active"])
    failures, missing, independent_counts = [], [], {}
    for scenario in ("ADULT_MOTION", "INFANT_MOTION", "EMPTY_ROOM", "LIGHT_SHADOW"):
        result = independent_metrics.get(scenario, {})
        positive = scenario in ("ADULT_MOTION", "INFANT_MOTION")
        denominator = (result.get("tp", 0) + result.get("fn", 0) if positive
                       else result.get("tn", 0) + result.get("fp", 0))
        independent_counts[scenario] = len({s["quality_independence_group"] for s in verified
            if s.get("scenario") == scenario and s["reference_active"] is positive})
        if independent_counts[scenario] < min_per_scenario:
            missing.append(scenario)
        elif positive:
            if result["recall"] < 90:
                failures.append(scenario)
        elif 100 * result["tn"] / denominator < 95:
            failures.append(scenario)
    return {"verdict": "FAIL" if failures else ("UNAVAILABLE" if missing else "PASS"),
            "failed_scenarios": failures, "insufficient_scenarios": missing,
            "verified_samples": len(verified), "total_samples": len(samples), "metrics": metrics,
            "independent_counts": independent_counts, "independent_metrics": independent_metrics,
            "excluded_samples": excluded, "required_independent_groups": min_per_scenario}


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

    with _dataset_lock:
        data_bytes = path.read_bytes()
        raw_data = json.loads(data_bytes)
        sidecar = path.with_name(path.stem + ".meta.json")
        stored_meta = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else None
        if stored_meta and stored_meta.get("samples_sha256"):
            if hashlib.sha256(data_bytes).hexdigest() != stored_meta["samples_sha256"]:
                raise ValueError(f"Benchmark data/metadata mismatch: {path}")

    if isinstance(raw_data, list):
        samples = raw_data
        metadata = {
            "version": "1.0",
            "format": "legacy_list",
            "total_samples": len(samples),
            "updated_at": datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
        }
        if stored_meta:
            metadata.update(stored_meta)
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
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,64}", version):
        raise ValueError("Benchmark version must contain only letters, digits, dot, underscore or hyphen")
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

    payload = json.dumps(sorted_samples, ensure_ascii=False, indent=2).encode("utf-8")
    meta["samples_sha256"] = hashlib.sha256(payload).hexdigest()
    meta["snapshot_id"] = uuid.uuid4().hex
    metadata_bytes = json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8")
    meta_file = target_path.with_name(target_path.stem + ".meta.json")
    with _dataset_lock:
        # Immutable snapshots are the recovery source, even if publishing fails.
        if archive_version_copy:
            VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
            v_stem = f"gold_ground_truth_v{version}_{meta['snapshot_id']}"
            for suffix, content in ((".json", payload), (".meta.json", metadata_bytes)):
                with (VERSIONS_DIR / (v_stem + suffix)).open("xb") as f:
                    f.write(content)
                    f.flush()
                    os.fsync(f.fileno())
        previous = [(p, p.read_bytes() if p.exists() else None) for p in (target_path, meta_file)]
        try:
            _atomic_write(target_path, payload)
            # Hash-bearing metadata is the commit marker for external readers.
            _atomic_write(meta_file, metadata_bytes)
        except BaseException:
            for path, old in previous:
                if old is None:
                    path.unlink(missing_ok=True)
                else:
                    _atomic_write(path, old)
            raise

    logger.info("基准数据集成功保存: %s (样本数: %d)", target_path, len(samples))
    return target_path


def _atomic_write(path: Path, payload: bytes) -> None:
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
            temp = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        temp.replace(path)
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


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
