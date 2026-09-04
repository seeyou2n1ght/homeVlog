#!/usr/bin/env python3
"""scripts/tune_thresholds.py

基于人工打标历史数据与预筛指标缓存的超参数网格搜索与自动化寻优工具。
针对家庭监控场景（低照度微动与全屏光影突变）优化 Pass 1 预筛选阈值。
"""

import argparse
import json
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台编码防护
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("tune_thresholds")


@dataclass
class PrescreenSample:
    file_id: int
    filepath: str
    ground_truth: str  # "MOTION" or "STATIC"
    max_diff: float
    mean_luma: float
    concentration: float
    diffs: list[float]
    has_prior: bool = False


@dataclass
class TuningResult:
    base_threshold: float
    night_factor: float
    concentration_thresh: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f2_score: float
    cost_score: float


def load_labeled_samples(db_path: Path) -> list[PrescreenSample]:
    """从数据库加载已人工复核的文件样本及预筛指标。"""
    if not db_path.exists():
        logger.warning("数据库文件不存在: %s", db_path)
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # 获取所有有复核记录的文件
    cursor = conn.execute(
        """
        SELECT 
            ft.id as file_id,
            ft.filepath,
            ft.prescreen_result,
            s.manual_label
        FROM file_tasks ft
        JOIN segments s ON ft.id = s.file_id
        WHERE s.manual_label IS NOT NULL 
          AND s.manual_label != 'UNREVIEWED'
        """
    )
    rows = cursor.fetchall()
    conn.close()

    if not rows:
        return []

    file_labels: dict[int, dict[str, Any]] = {}
    for r in rows:
        fid = r["file_id"]
        if fid not in file_labels:
            file_labels[fid] = {
                "filepath": r["filepath"],
                "prescreen_result": r["prescreen_result"],
                "labels": set(),
            }
        file_labels[fid]["labels"].add(r["manual_label"])

    samples: list[PrescreenSample] = []
    for fid, data in file_labels.items():
        lbls = data["labels"]
        if "VERIFIED_MOTION" in lbls or "MISSED_MOTION" in lbls:
            gt = "MOTION"
        elif "FALSE_ALARM" in lbls:
            gt = "STATIC"
        else:
            continue

        raw_res = data["prescreen_result"]
        if not raw_res:
            continue

        try:
            res = json.loads(raw_res)
        except Exception:
            continue

        max_diff = float(res.get("max_diff", 0.0))
        mean_luma = float(res.get("mean_luma", 100.0))
        concentration = float(res.get("concentration", 1.4))
        diffs = [float(x) for x in res.get("diffs", [])]

        samples.append(
            PrescreenSample(
                file_id=fid,
                filepath=data["filepath"],
                ground_truth=gt,
                max_diff=max_diff,
                mean_luma=mean_luma,
                concentration=concentration,
                diffs=diffs,
            )
        )

    return samples


def simulate_prediction(
    sample: PrescreenSample,
    base_threshold: float,
    night_factor: float,
    concentration_thresh: float,
) -> str:
    """模拟单样本在指定参数下的预筛输出。"""
    if sample.mean_luma < 50.0:
        dyn_th = max(2.0, base_threshold * night_factor)
    elif sample.mean_luma > 180.0:
        dyn_th = base_threshold * 1.25
    else:
        dyn_th = base_threshold

    if sample.has_prior:
        dyn_th = max(1.8, dyn_th * 0.65)

    d = sample.max_diff
    if d > dyn_th:
        if sample.concentration >= concentration_thresh or d >= dyn_th * 1.6:
            return "MOTION"
    return "STATIC"


def evaluate_grid(
    samples: list[PrescreenSample],
    base_threshold: float,
    night_factor: float,
    concentration_thresh: float,
    fn_weight: float = 5.0,
    fp_weight: float = 1.0,
) -> TuningResult:
    """在给定样本集上评估特定参数组合的表现。"""
    tp = fp = tn = fn = 0
    for s in samples:
        pred = simulate_prediction(s, base_threshold, night_factor, concentration_thresh)
        if s.ground_truth == "MOTION":
            if pred == "MOTION":
                tp += 1
            else:
                fn += 1
        else:
            if pred == "MOTION":
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f2 = (1 + 4) * (precision * recall) / (4 * precision + recall) if (4 * precision + recall) > 0 else 0.0
    cost = fn * fn_weight + fp * fp_weight

    return TuningResult(
        base_threshold=base_threshold,
        night_factor=night_factor,
        concentration_thresh=concentration_thresh,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f2_score=f2,
        cost_score=cost,
    )


def run_grid_search(
    samples: list[PrescreenSample],
    base_thresholds: list[float] | None = None,
    night_factors: list[float] | None = None,
    concentration_thresholds: list[float] | None = None,
) -> list[TuningResult]:
    """执行笛卡尔积全网格搜索。"""
    if base_thresholds is None:
        base_thresholds = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0]
    if night_factors is None:
        night_factors = [0.3, 0.4, 0.5, 0.6]
    if concentration_thresholds is None:
        concentration_thresholds = [1.2, 1.35, 1.5]

    results: list[TuningResult] = []
    for b_th in base_thresholds:
        for n_fac in night_factors:
            for c_th in concentration_thresholds:
                res = evaluate_grid(samples, b_th, n_fac, c_th)
                results.append(res)

    results.sort(key=lambda r: (r.cost_score, -r.f2_score, -r.recall))
    return results


def print_dataset_summary(db_path: Path) -> None:
    """输出当前数据库中预筛数据的宏观分布统计。"""
    if not db_path.exists():
        print(f"[-] 数据库未找到: {db_path}")
        return

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    total_tasks = c.execute("SELECT count(*) FROM file_tasks").fetchone()[0]
    prescreen_status = dict(
        c.execute("SELECT prescreen_status, count(*) FROM file_tasks GROUP BY prescreen_status").fetchall()
    )
    total_segments = c.execute("SELECT count(*) FROM segments").fetchone()[0]
    reviewed_segments = c.execute(
        "SELECT manual_label, count(*) FROM segments WHERE manual_label != 'UNREVIEWED' GROUP BY manual_label"
    ).fetchall()
    conn.close()

    print("\n" + "=" * 65)
    print(" [Stats] HomeVlog 数据库数据概览")
    print("=" * 65)
    print(f" 总文件任务数: {total_tasks}")
    print(f" 预筛状态分布: {prescreen_status}")
    print(f" 视频片段总数: {total_segments}")
    print(f" 人工打标记录: {dict(reviewed_segments) if reviewed_segments else '暂无 (0 条)'}")
    print("=" * 65 + "\n")


def apply_best_config(config_path: Path, best: TuningResult) -> bool:
    """安全写回推荐的超参数至 settings.yaml（创建 .bak 备份）。"""
    if not config_path.exists():
        logger.error("配置文件不存在: %s", config_path)
        return False

    try:
        import yaml
    except ImportError:
        logger.error("缺少 pyyaml 依赖，无法更新配置文件")
        return False

    backup_path = config_path.with_suffix(".yaml.bak")
    try:
        content = config_path.read_text(encoding="utf-8")
        backup_path.write_text(content, encoding="utf-8")
        logger.info("已创建配置文件备份: %s", backup_path)

        data = yaml.safe_load(content)
        if "detection" not in data:
            data["detection"] = {}

        data["detection"]["prescreen_diff_threshold"] = float(best.base_threshold)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

        logger.info("[Success] 成功更新 %s: prescreen_diff_threshold = %.1f", config_path.name, best.base_threshold)
        return True
    except Exception as e:
        logger.error("更新配置文件失败: %s", e)
        return False


def main():
    parser = argparse.ArgumentParser(description="HomeVlog 预筛自适应阈值网格调优工具")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "vlog.db", help="数据库路径")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "settings.yaml", help="配置文件路径")
    parser.add_argument("--apply", action="store_true", help="自动将搜索得到的最优参数写回配置文件")
    parser.add_argument("--dry-run", action="store_true", help="仅查看数据统计，不执行调参")

    args = parser.parse_args()

    print_dataset_summary(args.db)

    if args.dry_run:
        return

    samples = load_labeled_samples(args.db)
    if not samples:
        print("[!] 数据库中未找到任何已复核的有效样本。")
        print("[i] 使用提示:")
        print("    1. 请先启动 Web 审核工作台: uv run python scripts/audit_tool/app.py")
        print("    2. 在浏览器中标记若干疑难视频（标记为 动态确认、误报、漏检）")
        print("    3. 再次运行本工具执行超参数调优。")
        return

    pos_count = sum(1 for s in samples if s.ground_truth == "MOTION")
    neg_count = sum(1 for s in samples if s.ground_truth == "STATIC")
    print(f"[*] 加载有效真值样本: 总计 {len(samples)} (真实动态: {pos_count}, 纯静止: {neg_count})")

    print("[*] 开始执行多维网格寻优搜索...")
    results = run_grid_search(samples)

    print("\n" + "=" * 80)
    print(" [Top 5] 前 5 名最优超参数组合排名")
    print("=" * 80)
    print(f"{'排名':<4} {'基础阈值':<8} {'暗光系数':<8} {'网格集中度':<10} {'召回率':<8} {'精确率':<8} {'F2-分':<8} {'漏检FN':<6} {'误报FP':<6}")
    print("-" * 80)

    for idx, r in enumerate(results[:5], 1):
        print(
            f"{idx:<4} {r.base_threshold:<8.1f} {r.night_factor:<8.2f} {r.concentration_thresh:<10.2f} "
            f"{r.recall * 100:<7.1f}% {r.precision * 100:<7.1f}% {r.f2_score:<8.3f} {r.fn:<6} {r.fp:<6}"
        )
    print("=" * 80 + "\n")

    best = results[0]
    print(f"[Recommendation] 推荐最佳配置: 基础阈值 = {best.base_threshold:.1f}, 暗光系数 = {best.night_factor:.2f}")

    if args.apply:
        apply_best_config(args.config, best)


if __name__ == "__main__":
    main()
