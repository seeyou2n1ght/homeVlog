"""Compare independent references using complete production replay receipts.

Report observed retention and display contributions, never infer accuracy from
historical summary fields. Provide config previews, conflict checks and rollback.
"""

import copy
from contextlib import contextmanager
import hashlib
import logging
import math
import os
import tempfile
import uuid
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
from src.core import config as core_config
from src.core.config import load_config, validate_config

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
    """Compare current production replays against independently reviewed evidence.

    A replay callable receives (effective_config, samples) and returns one boolean
    decision per sample, in order. Historical algo_state/energy/confidence are
    never substituted for running the detector. UI without a replay provider
    reports unavailable rather than invented quality gains.
    """

    def __init__(self, benchmark_samples=None, config=None, *, replay=None):
        if benchmark_samples is None:
            try:
                self.samples, _ = load_benchmark_dataset()
            except (OSError, ValueError) as error:
                logger.warning("未能加载基准数据集: %s", error)
                self.samples = []
        else:
            self.samples = benchmark_samples
        self.config = config if config is not None else load_config()
        self.replay = replay

    def simulate(self, candidate_params):
        from src.core.profiles import resolve_profile_config, resolve_camera_config, _explicit_config
        reviewed = [s for s in self.samples
                    if s.get("review_status") in ("independent_human", "independent_visual")
                    and isinstance(s.get("reference_active"), bool)
                    and s.get("review_evidence")]
        if not reviewed or self.replay is None:
            return {"verdict": "UNAVAILABLE", "applied_params": candidate_params,
                    "error": "需要独立核验的参考标签与当前生产算法重放；历史状态和摘要不能预测调参收益",
                    "reviewed_samples": len(reviewed), "total_samples": len(self.samples)}
        baseline_cfg = resolve_profile_config(self.config)
        raw_candidate = _explicit_config(self.config)
        for key, value in candidate_params.items():
            _set_nested_param(raw_candidate, key, value)
        candidate_cfg = resolve_profile_config(raw_candidate)
        same_effective = all(
            resolve_camera_config(baseline_cfg, filepath=sample.get("filepath"), cam_index=sample.get("cam_index"))
            == resolve_camera_config(candidate_cfg, filepath=sample.get("filepath"), cam_index=sample.get("cam_index"))
            for sample in reviewed
        )

        def decisions(cfg):
            measure = getattr(self.replay, "measure", None)
            measured = measure(cfg, reviewed) if callable(measure) else None
            result = measured["predictions"] if measured is not None else self.replay(cfg, reviewed)
            if len(result) != len(reviewed) or any(type(v) is not bool for v in result):
                raise ValueError("Replay must return a boolean for every reviewed sample")
            return result, measured

        from src.core.replay_evidence import ReplayUnavailable
        try:
            baseline, base_measurement = decisions(baseline_cfg)
            candidate, new_measurement = (baseline, base_measurement) if same_effective else decisions(candidate_cfg)
            if base_measurement is not None and new_measurement is not None and base_measurement["identity"] != new_measurement["identity"]:
                raise ReplayUnavailable("基线与候选的代码、模型、运行环境或源集合身份不一致，不能比较")
        except ReplayUnavailable as error:
            return {"verdict": "UNAVAILABLE", "error": str(error), "applied_params": candidate_params,
                    "reviewed_samples": len(reviewed), "total_samples": len(self.samples)}

        def metrics(predictions):
            rows = [dict(s, gold_label="CONFIRMED_MOTION" if s["reference_active"] else "CONFIRMED_STATIC",
                         replay_active=active) for s, active in zip(reviewed, predictions)]
            decision = lambda s: s["replay_active"]
            return (evaluate_confusion_matrix(rows, is_active_decision_fn=decision),
                    evaluate_stratified_metrics(rows, group_key="scenario", is_active_decision_fn=decision))

        base_metrics, base_scenarios = metrics(baseline)
        new_metrics, new_scenarios = metrics(candidate)
        deltas = {k: round(new_metrics[k] - base_metrics[k], 2) for k in ("f1", "precision", "recall")}
        deltas.update(fp_reduction=base_metrics["fp"] - new_metrics["fp"],
                      fn_increase=new_metrics["fn"] - base_metrics["fn"])
        impacts = {key: {"fp_delta": new_scenarios[key]["fp"] - value["fp"],
                         "fn_delta": new_scenarios[key]["fn"] - value["fn"],
                         "f1_delta": round(new_scenarios[key]["f1"] - value["f1"], 2)}
                   for key, value in base_scenarios.items()}
        # An observed metric change is not release acceptance: missing negative
        # examples or unreviewed difficult scenarios leave quality unestablished.
        from src.core.benchmark import assess_replay_quality
        quality_gate = assess_replay_quality(reviewed, candidate)
        rows = [{"segment_id": sample.get("segment_id", sample.get("id")),
                 "filepath": sample.get("filepath"), "date": sample.get("date"),
                 "cam_index": sample.get("cam_index"), "start_time": sample.get("start_time"),
                 "end_time": sample.get("end_time"), "scenario": sample.get("scenario", "UNKNOWN"),
                 "reference_active": sample["reference_active"],
                 "baseline_retained": before, "candidate_retained": after,
                 "retention_changed": before != after}
                for sample, before, after in zip(reviewed, baseline, candidate)]
        details = getattr(self.replay, "details", None)
        if base_measurement is not None or callable(details):
            try:
                base_details = base_measurement["details"] if base_measurement is not None else details(baseline_cfg, reviewed)
                new_details = new_measurement["details"] if new_measurement is not None else (base_details if same_effective else details(candidate_cfg, reviewed))
                if len(base_details) != len(rows) or len(new_details) != len(rows):
                    raise ValueError("Replay details must cover every reference")
                for row, before, after in zip(rows, base_details, new_details):
                    row.update(baseline_display=before, candidate_display=after)
            except ReplayUnavailable as error:
                # Do not keep a metric result if source/receipt identity changed
                # between retention and detailed display-clock reads.
                return {"verdict": "UNAVAILABLE", "error": str(error), "applied_params": candidate_params,
                        "reviewed_samples": len(reviewed), "total_samples": len(self.samples)}
        return {"baseline": base_metrics, "simulated": new_metrics, "deltas": deltas,
                "scenario_impacts": impacts, "applied_params": candidate_params,
                "verdict": "NEUTRAL" if baseline == candidate else "MEASURED_CHANGE",
                "rationale": "当前生产算法重放对照；是否满足质量门禁需独立验收",
                "reviewed_samples": len(reviewed), "total_samples": len(self.samples),
                "quality_gate": quality_gate, "sample_changes": rows,
                "baseline_receipt_sha256": base_measurement.get("receipt_sha256") if base_measurement else None,
                "candidate_receipt_sha256": new_measurement.get("receipt_sha256") if new_measurement else None,
                "evidence_basis": "production_replay", "timestamp": datetime.now().isoformat()}

    def simulate_profile(self, profile_name, sensitivity=1.0, overrides=None):
        params = {"detection.profile": profile_name, "detection.motion_sensitivity": sensitivity}
        params.update(overrides or {})
        result = self.simulate(params)
        result.update(profile=profile_name, sensitivity=sensitivity)
        return result


def generate_config_diff(
    new_params: dict[str, Any],
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """生成候选参数与当前 settings.yaml 的结构化 Diff 对比。"""
    cfg_file = Path(config_path) if config_path else core_config.CONFIG_PATH
    if not cfg_file.exists():
        raise FileNotFoundError(f"配置文件未找到: {cfg_file}")

    original = cfg_file.read_bytes()
    current_cfg = yaml.safe_load(original) or {}

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
        "config_sha256": hashlib.sha256(original).hexdigest(),
        "total_changes": len(diffs),
        "diffs": diffs,
    }


@contextmanager
def _config_writer(path):
    from src.core.process_lock import exclusive_file_lock
    lock = path.with_name("." + path.name + ".writer.lock")
    # Translate admission conflicts only; errors inside the transaction retain
    # their original context and rollback semantics.
    from contextlib import ExitStack
    with ExitStack() as stack:
        try:
            stack.enter_context(exclusive_file_lock(lock, conflict="Configuration writer already active"))
        except RuntimeError as error:
            raise ConfigurationConflict(str(error)) from error
        yield


def apply_config_safe(
    new_params: dict[str, Any],
    config_path: Path | str | None = None,
    create_backup: bool = True,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """安全校验并原子写回新配置，自动创建带时间戳的备份文件。"""
    with core_config._config_lock:
        cfg_file = (Path(config_path) if config_path else core_config.CONFIG_PATH).resolve()
        with _config_writer(cfg_file):
            original = cfg_file.read_bytes()
            if expected_sha256 is not None and hashlib.sha256(original).hexdigest() != expected_sha256:
                raise ConfigurationConflict("Configuration changed since preview; refresh the diff before applying")
            candidate_cfg = yaml.safe_load(original) or {}
            for k, v in new_params.items():
                _set_nested_param(candidate_cfg, k, v)
            from src.core.profiles import resolve_profile_config
            validate_config(resolve_profile_config(candidate_cfg))
            proposed = yaml.safe_dump(candidate_cfg, allow_unicode=True, sort_keys=False).encode("utf-8")
            backup_path = _commit_config(cfg_file, proposed, original, create_backup)
            return {"success": True, "config_path": str(cfg_file), "applied_count": len(new_params),
                    "backup_path": str(backup_path) if backup_path else None, "updated_params": new_params}


class ConfigurationConflict(ValueError):
    """A writer owns the configuration or a preview refers to an older version."""


def _atomic_config_write(path: Path, payload: bytes) -> None:
    """Replace a single file from a unique, flushed same-directory temporary."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as f:
            temporary = Path(f.name)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _commit_config(path: Path, proposed: bytes, original: bytes, create_backup: bool) -> Path | None:
    """Caller owns the config lock across read, validation, commit and reload."""
    backup = None
    if create_backup:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup = path.with_name(f"{path.name}.bak_{stamp}_{uuid.uuid4().hex}")
        with backup.open("xb") as f:
            f.write(original)
            f.flush()
            os.fsync(f.fileno())
    previous_path, previous_settings = core_config.CONFIG_PATH, core_config.SETTINGS
    _atomic_config_write(path, proposed)
    try:
        load_config(config_path=path, reload=True)
    except BaseException:
        try:
            _atomic_config_write(path, original)
        except OSError:
            logger.exception("Configuration rollback failed; preserve backup at %s", backup)
            raise
        finally:
            core_config.CONFIG_PATH, core_config.SETTINGS = previous_path, previous_settings
        raise
    return backup


def rollback_config(
    backup_path: Path | str,
    config_path: Path | str | None = None,
) -> dict[str, Any]:
    """从备份文件安全回滚配置。"""
    with core_config._config_lock:
        b_file = Path(backup_path).resolve()
        target_file = (Path(config_path) if config_path else core_config.CONFIG_PATH).resolve()
        with _config_writer(target_file):
            proposed = b_file.read_bytes()
            from src.core.profiles import resolve_profile_config
            validate_config(resolve_profile_config(yaml.safe_load(proposed) or {}))
            current = target_file.read_bytes()
            backup = _commit_config(target_file, proposed, current, True)
            return {"success": True, "restored_from": str(b_file), "target_config": str(target_file),
                    "backup_path": str(backup)}
