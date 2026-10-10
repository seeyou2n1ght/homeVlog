"""HomeVlog 离线参数仿真器与配置安全拟合引擎 (What-If Tuning Simulator)

基于黄金真值评测集 (Gold Benchmark)，离线仿真候选参数变更对
混淆矩阵、Precision、Recall、F1-Score 及各场景误报/漏报的量化影响。
提供参数 Diff 预览、原子备份回滚与安全配置落地保障。
"""

import copy
import logging
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from src.core.benchmark import (
    evaluate_confusion_matrix,
    evaluate_stratified_metrics,
    load_benchmark_dataset,
    normalize_benchmark_label,
)
from src.core.config import CONFIG_PATH, load_config, validate_config

logger = logging.getLogger("homevlog.tuner")


def _get_nested_param(d: dict[str, Any], key_path: str, default: Any = None) -> Any:
    """按点号分段键获取嵌套字典值，如 detection.min_motion_threshold"""
    keys = key_path.split(".")
    curr = d
    for k in keys:
        if not isinstance(curr, dict) or k not in curr:
            return default
        curr = curr[k]
    return curr


def _set_nested_param(d: dict[str, Any], key_path: str, value: Any) -> None:
    """按点号分段键设置嵌套字典值"""
    keys = key_path.split(".")
    curr = d
    for k in keys[:-1]:
        if k not in curr or not isinstance(curr[k], dict):
            curr[k] = {}
        curr = curr[k]
    curr[keys[-1]] = value


class TuningSimulator:
    """What-If 离线参数灵敏度仿真引擎"""

    def __init__(self, benchmark_samples: list[dict[str, Any]] | None = None, config: dict[str, Any] | None = None):
        if benchmark_samples is None:
            try:
                self.samples, _ = load_benchmark_dataset()
            except Exception as e:
                logger.warning("未能自动加载基准数据集: %s", e)
                self.samples = []
        else:
            self.samples = benchmark_samples

        self.config = config or load_config()

    def simulate(self, candidate_params: dict[str, Any]) -> dict[str, Any]:
        """对候选参数执行 What-If 离线推演并返回详尽的量化收益分析。

        支持的仿真参数:
            - detection.min_motion_threshold (float)
            - segment.min_motion_duration (float)
            - yolo.confidence (float)
            - presence.person_conf_threshold (float)
        """
        if not self.samples:
            return {
                "error": "No benchmark samples available for simulation",
                "verdict": "UNAVAILABLE",
            }

        # 1. 计算基准表现 (Baseline)
        baseline_metrics = evaluate_confusion_matrix(self.samples)
        baseline_stratified = evaluate_stratified_metrics(self.samples, group_key="scenario")

        # 2. 提取当前生效参数与候选参数
        cur_det_thresh = float(_get_nested_param(self.config, "detection.min_motion_threshold", 2.5))
        cur_min_dur = float(_get_nested_param(self.config, "segment.min_motion_duration", 2.0))
        cur_yolo_conf = float(_get_nested_param(self.config, "yolo.confidence", 0.3))
        cur_person_conf = float(_get_nested_param(self.config, "presence.person_conf_threshold", 0.20))

        sim_det_thresh = float(candidate_params.get("detection.min_motion_threshold", cur_det_thresh))
        sim_min_dur = float(candidate_params.get("segment.min_motion_duration", cur_min_dur))
        sim_yolo_conf = float(candidate_params.get("yolo.confidence", cur_yolo_conf))
        sim_person_conf = float(candidate_params.get("presence.person_conf_threshold", cur_person_conf))

        # 3. 逐样本推演新状态
        def simulated_decision(s: dict[str, Any]) -> bool:
            orig_state = s.get("algo_state", "STATIC")
            is_originally_active = orig_state in ("DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE")

            energy = float(s.get("max_energy", 0.0) or 0.0)
            conf = float(s.get("avg_confidence", 0.0) or 0.0)
            dur = float(s.get("duration", 0.0) or 0.0)

            # 规则 A: 目标置信度驱动
            # 若置信度达到新 YOLO/驻留门限，强制认定为真实活动
            has_strong_object = conf >= sim_yolo_conf or conf >= sim_person_conf
            has_weak_object = conf > 0.0 and (conf >= min(sim_yolo_conf, sim_person_conf) * 0.8)

            # 规则 B: 短碎片过滤 (时长惩罚)
            if dur < sim_min_dur and not has_strong_object:
                # 过于短暂的毛刺碎片被平滑为静态
                return False

            # 规则 C: 能量门限判断
            if is_originally_active:
                # 原为激活态: 如果没有高确信度物体，且动能跌破新门限，则被滤除
                if not has_strong_object and energy < sim_det_thresh:
                    return False
                return True
            else:
                # 原为静态: 若动能超过了降低后的新门限，且达到微动或有弱目标，则被召回
                if energy >= sim_det_thresh and (has_weak_object or energy >= sim_det_thresh * 1.2):
                    return True
                return False

        sim_metrics = evaluate_confusion_matrix(self.samples, is_active_decision_fn=simulated_decision)
        sim_stratified = evaluate_stratified_metrics(
            self.samples, group_key="scenario", is_active_decision_fn=simulated_decision
        )

        # 4. 指标差值计算
        delta_f1 = round(sim_metrics["f1"] - baseline_metrics["f1"], 2)
        delta_prec = round(sim_metrics["precision"] - baseline_metrics["precision"], 2)
        delta_recall = round(sim_metrics["recall"] - baseline_metrics["recall"], 2)
        fp_reduction = baseline_metrics["fp"] - sim_metrics["fp"]  # 正数表示误报减少
        fn_increase = sim_metrics["fn"] - baseline_metrics["fn"]    # 正数表示漏报增加

        # 5. 场景细分收益度量
        scenario_deltas = {}
        for sc in set(list(baseline_stratified.keys()) + list(sim_stratified.keys())):
            b_sc = baseline_stratified.get(sc, {"fp": 0, "fn": 0, "f1": 0.0})
            s_sc = sim_stratified.get(sc, {"fp": 0, "fn": 0, "f1": 0.0})
            scenario_deltas[sc] = {
                "fp_delta": s_sc.get("fp", 0) - b_sc.get("fp", 0),
                "fn_delta": s_sc.get("fn", 0) - b_sc.get("fn", 0),
                "f1_delta": round(s_sc.get("f1", 0.0) - b_sc.get("f1", 0.0), 2),
            }

        # 6. 决策评估与解释性结论
        if delta_f1 > 0 and fn_increase <= 1:
            verdict = "IMPROVED"
            reason = f"综合 F1 提升 +{delta_f1}%，成功压制 {fp_reduction} 处误报且未显著破坏动作召回"
        elif fp_reduction >= 3 and delta_f1 >= -0.5:
            verdict = "FAVORABLE_TRADEOFF"
            reason = f"显著压制 {fp_reduction} 处误报，虽略有 {fn_increase} 处漏报代价，但整体成片率大幅净化"
        elif fn_increase >= 3 or delta_f1 < -1.0:
            verdict = "REGRESSED"
            reason = f"参数过于激进：增加 {fn_increase} 处漏报，F1 下降 {delta_f1}%，存在关键动态漏剪风险"
        else:
            verdict = "NEUTRAL"
            reason = "指标变化微弱或正负效应相互抵消"

        return {
            "baseline": baseline_metrics,
            "simulated": sim_metrics,
            "deltas": {
                "f1": delta_f1,
                "precision": delta_prec,
                "recall": delta_recall,
                "fp_reduction": fp_reduction,
                "fn_increase": fn_increase,
            },
            "scenario_impacts": scenario_deltas,
            "applied_params": candidate_params,
            "verdict": verdict,
            "rationale": reason,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }


def generate_config_diff(
    new_params: dict[str, Any],
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """生成候选参数与当前 settings.yaml 的结构化 Diff 对比。"""
    cfg_file = Path(config_path) if config_path else CONFIG_PATH
    if not cfg_file.exists():
        raise FileNotFoundError(f"配置文件未找到: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        current_cfg = yaml.safe_load(f) or {}

    diffs = []
    for k, new_v in new_params.items():
        curr_v = _get_nested_param(current_cfg, k)
        if curr_v != new_v:
            diffs.append({
                "param": k,
                "current": curr_v,
                "proposed": new_v,
                "type": type(curr_v).__name__ if curr_v is not None else type(new_v).__name__,
            })

    return {
        "config_path": str(cfg_file),
        "total_changes": len(diffs),
        "diffs": diffs,
    }


def apply_config_safe(
    new_params: dict[str, Any],
    config_path: Path | str | None = None,
    create_backup: bool = True,
) -> dict[str, Any]:
    """安全校验并原子写回新配置，自动创建带时间戳的备份文件。"""
    cfg_file = Path(config_path) if config_path else CONFIG_PATH
    if not cfg_file.exists():
        raise FileNotFoundError(f"配置文件未找到: {cfg_file}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f) or {}

    # 1. 深度拷贝并在内存中应用新参数
    candidate_cfg = copy.deepcopy(raw_cfg)
    for k, v in new_params.items():
        _set_nested_param(candidate_cfg, k, v)

    # 2. 强校验: 必须通过核心契约 validate_config
    validate_config(candidate_cfg)

    # 3. 创建安全备份
    backup_path = None
    if create_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = cfg_file.with_name(f"{cfg_file.name}.bak_{ts}")
        shutil.copy2(cfg_file, backup_path)
        logger.info("已生成配置安全备份: %s", backup_path)

    # 4. 原子安全写回
    temp_target = cfg_file.with_name(f"{cfg_file.name}.tmp")
    with open(temp_target, "w", encoding="utf-8") as f:
        yaml.safe_dump(candidate_cfg, f, allow_unicode=True, sort_keys=False)

    shutil.move(str(temp_target), str(cfg_file))

    # 5. 触发配置缓存刷新
    load_config(config_path=cfg_file, reload=True)

    logger.info("配置已成功更新并热重载: %s (修改项: %d)", cfg_file, len(new_params))
    return {
        "success": True,
        "config_path": str(cfg_file),
        "applied_count": len(new_params),
        "backup_path": str(backup_path) if backup_path else None,
        "updated_params": new_params,
    }


def rollback_config(
    backup_path: Path | str,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """从备份文件安全回滚配置。"""
    b_file = Path(backup_path)
    if not b_file.exists():
        raise FileNotFoundError(f"指定的备份文件不存在: {b_file}")

    target_file = Path(config_path) if config_path else CONFIG_PATH

    # 先验证备份文件的合法性
    with open(b_file, "r", encoding="utf-8") as f:
        candidate_cfg = yaml.safe_load(f) or {}
    validate_config(candidate_cfg)

    # 复制覆盖
    shutil.copy2(b_file, target_file)
    load_config(config_path=target_file, reload=True)

    logger.info("配置已成功从备份回滚: %s -> %s", b_file, target_file)
    return {
        "success": True,
        "restored_from": str(b_file),
        "target_config": str(target_file),
    }
