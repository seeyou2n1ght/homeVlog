#!/usr/bin/env python3
"""scripts/verify_accuracy.py

全量监控素材识别质量自动化核查与深度审计 CLI 工具。
对数据库中的全部切片、动静判定、YOLO 置信度及音频唤醒进行多维度量化分析，
输出量化质量指标及详细审计报告 (docs/ACCURACY_AUDIT_REPORT.md)。
"""

import argparse
import logging
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

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
logger = logging.getLogger("verify_accuracy")


@dataclass
class QualityMetrics:
    total_files: int
    analyzed_files: int
    static_prescreen_files: int
    total_segments: int
    dynamic_count: int
    dynamic_audio_count: int
    static_count: int
    fp_suspects: int
    fn_suspects: int
    jitter_count: int
    consecutive_same_state_count: int
    zero_conf_dynamic_count: int
    avg_segment_duration: float
    report_text: str = ""


def audit_database(db_path: Path, date: str | None = None, cam_index: int | None = None) -> QualityMetrics:
    """执行数据库质量审计与指标计算。"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    where_files = []
    params_files = []
    where_segs = []
    params_segs = []

    if date:
        where_files.append("date = ?")
        params_files.append(date)
        where_segs.append("date = ?")
        params_segs.append(date)
    if cam_index is not None:
        where_files.append("cam_index = ?")
        params_files.append(cam_index)
        where_segs.append("cam_index = ?")
        params_segs.append(cam_index)

    wf_sql = f"WHERE {' AND '.join(where_files)}" if where_files else ""
    ws_sql = f"WHERE {' AND '.join(where_segs)}" if where_segs else ""

    total_files = conn.execute(f"SELECT COUNT(*) FROM file_tasks {wf_sql}", params_files).fetchone()[0]
    analyzed_files = conn.execute(
        f"SELECT COUNT(*) FROM file_tasks {wf_sql} {'AND' if wf_sql else 'WHERE'} analysis_status='ANALYZED'",
        params_files,
    ).fetchone()[0]
    static_prescreen_files = conn.execute(
        f"SELECT COUNT(*) FROM file_tasks {wf_sql} {'AND' if wf_sql else 'WHERE'} prescreen_status='STATIC'",
        params_files,
    ).fetchone()[0]

    total_segments = conn.execute(f"SELECT COUNT(*) FROM segments {ws_sql}", params_segs).fetchone()[0]

    dynamic_count = conn.execute(
        f"SELECT COUNT(*) FROM segments {ws_sql} {'AND' if ws_sql else 'WHERE'} state='DYNAMIC'",
        params_segs,
    ).fetchone()[0]

    dynamic_audio_count = conn.execute(
        f"SELECT COUNT(*) FROM segments {ws_sql} {'AND' if ws_sql else 'WHERE'} state='DYNAMIC_AUDIO'",
        params_segs,
    ).fetchone()[0]

    static_count = conn.execute(
        f"SELECT COUNT(*) FROM segments {ws_sql} {'AND' if ws_sql else 'WHERE'} state='STATIC'",
        params_segs,
    ).fetchone()[0]

    fp_suspects = conn.execute(
        f"SELECT COUNT(*) FROM segments {ws_sql} {'AND' if ws_sql else 'WHERE'} state='DYNAMIC' AND avg_confidence=0.0 AND max_energy < 8.0",
        params_segs,
    ).fetchone()[0]

    fn_suspects = conn.execute(
        f"SELECT COUNT(*) FROM segments {ws_sql} {'AND' if ws_sql else 'WHERE'} state='STATIC' AND max_energy >= 1.5 AND max_energy <= 4.0",
        params_segs,
    ).fetchone()[0]

    jitter_count = conn.execute(
        f"SELECT COUNT(*) FROM segments {ws_sql} {'AND' if ws_sql else 'WHERE'} duration < 2.0 AND state='DYNAMIC'",
        params_segs,
    ).fetchone()[0]

    zero_conf_dynamic_count = conn.execute(
        f"SELECT COUNT(*) FROM segments {ws_sql} {'AND' if ws_sql else 'WHERE'} state='DYNAMIC' AND avg_confidence = 0.0",
        params_segs,
    ).fetchone()[0]

    avg_dur_row = conn.execute(
        f"SELECT AVG(duration) FROM segments {ws_sql}",
        params_segs,
    ).fetchone()[0]
    avg_segment_duration = float(avg_dur_row or 0.0)

    # 拓扑碎片化检查：连续相邻同状态切片对数
    cursor = conn.execute(
        f"SELECT file_id, state, start_time, end_time FROM segments {ws_sql} ORDER BY file_id, start_time",
        params_segs,
    )
    rows = cursor.fetchall()
    conn.close()

    consecutive_same_state_count = 0
    prev_file_id = None
    prev_state = None
    for r in rows:
        fid = r["file_id"]
        st = r["state"]
        if fid == prev_file_id and st == prev_state:
            consecutive_same_state_count += 1
        prev_file_id = fid
        prev_state = st

    return QualityMetrics(
        total_files=total_files,
        analyzed_files=analyzed_files,
        static_prescreen_files=static_prescreen_files,
        total_segments=total_segments,
        dynamic_count=dynamic_count,
        dynamic_audio_count=dynamic_audio_count,
        static_count=static_count,
        fp_suspects=fp_suspects,
        fn_suspects=fn_suspects,
        jitter_count=jitter_count,
        consecutive_same_state_count=consecutive_same_state_count,
        zero_conf_dynamic_count=zero_conf_dynamic_count,
        avg_segment_duration=round(avg_segment_duration, 2),
    )


def generate_markdown_report(metrics: QualityMetrics, out_path: Path, scope_desc: str = "全量监控素材"):
    """Report rule-based audit signals without claiming measured precision/recall."""
    from datetime import date
    dynamic_total = metrics.dynamic_count + metrics.dynamic_audio_count
    audio_share = metrics.dynamic_audio_count / dynamic_total * 100 if dynamic_total else 0
    rows = [
        ("扫描文件", metrics.total_files), ("已精析文件", metrics.analyzed_files),
        ("预筛静态文件", metrics.static_prescreen_files), ("总切片", metrics.total_segments),
        ("视觉动态切片", metrics.dynamic_count), ("音频动态切片", metrics.dynamic_audio_count),
        ("静态切片", metrics.static_count), ("疑似误报线索", metrics.fp_suspects),
        ("疑似漏报线索", metrics.fn_suspects), ("短时抖动线索", metrics.jitter_count),
        ("相邻同状态切片对", metrics.consecutive_same_state_count),
        ("零 YOLO 置信度动态切片", metrics.zero_conf_dynamic_count),
    ]
    md = f"# 识别质量待审线索\n\n生成日期：{date.today()}；范围：{scope_desc}。\n\n"
    md += "这些是数据库规则统计，不是人工真值准确率。未进入 segments 的预筛静态文件也必须抽检。\n\n| 指标 | 数量 |\n|---|---:|\n"
    md += "".join(f"| {label} | {value} |\n" for label,value in rows)
    md += f"\n音频动态占动态**切片数量**的 {audio_share:.1f}%，不是时长比例。\n"
    md += "\n零置信度可能来自缺少样本、模型降级或画外声音，不能自动归因为数据断链；相邻同状态需结合人工边界判断。\n"
    md += "\n使用 sample_accuracy.py 从原始时间范围抽样，记录人工视觉/音频事件，再评估召回。当前报告不推断故障成因，也不把零疑似误报写成 100% 准确。\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md,encoding="utf-8")


def fix_database_fragmentation(db_path: Path) -> tuple[int, int]:
    """遍历数据库中已分析的文件，对相邻同状态切片执行原子熔断合并，消除历史碎片化。"""
    from src.segment import Segment, _merge_same_state, segments_to_json
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    cursor = conn.execute("SELECT id, filepath, cam_index, date FROM file_tasks WHERE analysis_status='ANALYZED'")
    files = cursor.fetchall()

    total_before = 0
    total_after = 0

    for f in files:
        fid = f["id"]
        fp = f["filepath"]
        cam = f["cam_index"]
        dt = f["date"]

        seg_rows = conn.execute(
            "SELECT * FROM segments WHERE file_id=? ORDER BY start_time ASC", (fid,)
        ).fetchall()
        if not seg_rows:
            continue
        if any(r["manual_label"] for r in seg_rows if "manual_label" in r.keys()):
            total_before += len(seg_rows)
            total_after += len(seg_rows)
            continue

        total_before += len(seg_rows)
        segs = [
            Segment(
                start_time=r["start_time"],
                end_time=r["end_time"],
                state=r["state"],
                source_file=fp,
                file_start_offset=r["file_start_offset"],
                max_energy=r["max_energy"],
                avg_confidence=r["avg_confidence"],
            )
            for r in seg_rows
        ]

        merged = _merge_same_state(segs, gap_tolerance=1.5)
        total_after += len(merged)

        if len(merged) != len(segs):
            conn.execute("DELETE FROM segments WHERE file_id=?", (fid,))
            new_records = [
                (
                    fid, fp, cam, dt,
                    s.start_time, s.end_time, max(0.0, s.end_time - s.start_time),
                    s.state, s.max_energy, s.avg_confidence, s.file_start_offset
                )
                for s in merged
            ]
            conn.executemany(
                """INSERT OR REPLACE INTO segments
                   (file_id, filepath, cam_index, date, start_time, end_time, duration,
                    state, max_energy, avg_confidence, file_start_offset)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                new_records
            )
            conn.execute(
                "UPDATE file_tasks SET analysis_segments=? WHERE id=?",
                (segments_to_json(merged), fid)
            )

    conn.commit()
    conn.close()
    return total_before, total_after


def main():
    parser = argparse.ArgumentParser(description="Verify accuracy and audit detection quality")
    parser.add_argument("--db-path", type=str, default="data/vlog.db", help="Path to SQLite database")
    parser.add_argument("--report-out", type=str, default="docs/ACCURACY_AUDIT_REPORT.md", help="Output Markdown report path")
    parser.add_argument("--date", type=str, default=None, help="Optional filter by date (YYYYMMDD)")
    parser.add_argument("--cam-index", type=int, default=None, help="Optional filter by cam index")
    parser.add_argument("--fix-fragmentation", action="store_true", help="Atomically merge adjacent same-state fragments in database")
    args = parser.parse_args()

    db_p = Path(args.db_path)
    if not db_p.exists():
        logger.error("Database not found: %s", db_p)
        sys.exit(1)

    if args.fix_fragmentation:
        logger.info("正在执行历史碎片切片原子熔断合并...")
        before_cnt, after_cnt = fix_database_fragmentation(db_p)
        logger.info("碎片熔断完成: 切片总数从 %d 优化至 %d (减少 %d 个碎切片)", before_cnt, after_cnt, before_cnt - after_cnt)

    out_p = Path(args.report_out)
    metrics = audit_database(db_p, date=args.date, cam_index=args.cam_index)

    scope = f"日期: {args.date}" if args.date else "全量历史素材"
    generate_markdown_report(metrics, out_p, scope_desc=scope)

    print("\n" + "=" * 60)
    print(f"质量核查概览 ({scope})")
    print("=" * 60)
    print(f"总文件数: {metrics.total_files} (已分析: {metrics.analyzed_files})")
    print(f"总切片数: {metrics.total_segments}")
    print(f"  - 动态段 (DYNAMIC):       {metrics.dynamic_count}")
    print(f"  - 音频唤醒 (AUDIO):       {metrics.dynamic_audio_count}")
    print(f"  - 静态段 (STATIC):        {metrics.static_count}")
    print(f"  - 疑似错检 (FP Suspect):  {metrics.fp_suspects}")
    print(f"  - 疑似漏检 (FN Suspect):  {metrics.fn_suspects}")
    print(f"  - 未合并碎片切片对数:     {metrics.consecutive_same_state_count}")
    print(f"  - YOLO零置信度切片数:     {metrics.zero_conf_dynamic_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
