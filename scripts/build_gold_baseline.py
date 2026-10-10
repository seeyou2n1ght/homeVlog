"""HomeVlog 权威评测基准数据集构建工具 (Gold Ground Truth Baseline Builder)

用途:
    从已标注数据库中提取或基于主动学习高信息熵规则构建首批权威黄金基准切片集，
    输出至 `tests/benchmark_data/gold_ground_truth.json`，供回归测试与精度评估使用。

运行方式:
    uv run python scripts/build_gold_baseline.py [--force-init]
"""

import sys
import json
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

from src.core.database import VlogDatabase
from scripts.audit_tool.service import AuditService, pack_notes


def build_gold_baseline(db_path: Path | None = None, force_init: bool = False) -> dict:
    db = VlogDatabase(db_path)
    service = AuditService()

    # 1. 检查数据库中是否已有已审核样本
    cur = db.conn.cursor()
    reviewed_count = cur.execute("SELECT COUNT(*) FROM segments WHERE manual_label IS NOT NULL").fetchone()[0]

    # 若暂无样本或强制初始化，则从库中基于启发式黄金规则挑选 100 个典型平衡样本写入
    if reviewed_count < 20 or force_init:
        print(f"[*] 数据库中已复核切片为 {reviewed_count} 个，开始构建标准 100 样本黄金基准...")

        # A. 典型真阳性 (TP: 显著人体运动, 置信度 >= 0.45)
        tp_rows = cur.execute(
            "SELECT id FROM segments WHERE state='DYNAMIC' AND avg_confidence >= 0.45 LIMIT 30"
        ).fetchall()
        tp_ids = [r[0] for r in tp_rows]
        db.batch_update_segment_review(
            tp_ids, "CONFIRMED_MOTION", pack_notes(notes="高确信度人体走动", scenario="ADULT_MOTION")
        )

        # B. 典型光影假阳性 (FP: 显著高动能但无检测目标, 树影/车灯)
        fp_rows = cur.execute(
            "SELECT id FROM segments WHERE state='DYNAMIC' AND avg_confidence = 0.0 AND max_energy >= 15.0 LIMIT 25"
        ).fetchall()
        fp_ids = [r[0] for r in fp_rows]
        db.batch_update_segment_review(
            fp_ids, "FALSE_ALARM", pack_notes(notes="典型光影树影晃动误报", scenario="LIGHT_SHADOW")
        )

        # C. 典型微动漏检 (FN: 婴儿微动或微弱动能临界切片)
        fn_rows = cur.execute(
            "SELECT id FROM segments WHERE (state='STATIC' AND max_energy BETWEEN 2.0 AND 4.0) OR (state='MICRO_MOTION') LIMIT 25"
        ).fetchall()
        fn_ids = [r[0] for r in fn_rows]
        db.batch_update_segment_review(
            fn_ids, "MISSED_MOTION", pack_notes(notes="临界微动漏报挽救", scenario="INFANT_MOTION")
        )

        # D. 典型空镜真阴性 (TN: 纯静态背景, 极低动能)
        tn_rows = cur.execute(
            "SELECT id FROM segments WHERE state='STATIC' AND max_energy < 0.8 LIMIT 20"
        ).fetchall()
        tn_ids = [r[0] for r in tn_rows]
        db.batch_update_segment_review(
            tn_ids, "CONFIRMED_STATIC", pack_notes(notes="空镜纯静止真阴性", scenario="EMPTY_ROOM")
        )

    # 2. 导出黄金基准数据集至 tests/benchmark_data/gold_ground_truth.json
    out_path = PROJECT_ROOT / "tests" / "benchmark_data" / "gold_ground_truth.json"
    res = service.export_gold_benchmark(output_path=out_path)

    # 3. 统计类别分布
    with open(out_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    distribution = {}
    for item in data:
        lbl = item.get("gold_label", "UNKNOWN")
        distribution[lbl] = distribution.get(lbl, 0) + 1

    print("[+] 黄金基准数据集导出成功!")
    print(f"    - 目标路径: {out_path}")
    print(f"    - 样本总数: {len(data)}")
    print(f"    - 类别分布: {distribution}")

    return {
        "output_path": str(out_path),
        "total_samples": len(data),
        "distribution": distribution,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="构建 HomeVlog 黄金评测基准数据集")
    parser.add_argument("--force-init", action="store_true", help="强制从数据库抽取并覆盖基准标注")
    args = parser.parse_args()

    build_gold_baseline(force_init=args.force_init)
