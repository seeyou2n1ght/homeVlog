#!/usr/bin/env python3
"""scripts/reprocess_accuracy.py

全量监控素材质量重算与成片重建 CLI 工具。
用于一键基于最新识别算法（门控双差分背景模型、硬件底噪硬阈值下限、多模态音频验证闭环）
重新核算切片、净化历史数据库、并计算成片最终展示时长，确保零漏检且无长时间静止常速播放。

Usage:
  uv run python scripts/reprocess_accuracy.py --date 20260321 --audit
  uv run python scripts/reprocess_accuracy.py --date 20260321 --clean-empty-audio
  uv run python scripts/reprocess_accuracy.py --date 20260321 --reanalyze
"""

import argparse
import logging
import sqlite3
import sys
import time
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

from src.database import VlogDatabase
from src.segment import (
    _merge_same_state,
    _filter_short,
    segments_to_json,
    segments_from_json,
)
from src.timeline import build_timeline_from_rows, compute_display_plans
from src.utils import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reprocess_accuracy")


def calculate_timeline_metrics(tasks: list[dict], date: str, config: dict) -> dict:
    """基于任务切片构建时间轴并计算展示时长指标。"""
    seg_cfg = config.get("segment", {})
    kf_interval = float(seg_cfg.get("static_keyframe_interval", 45.0))
    kf_dur = float(seg_cfg.get("keyframe_display_duration", 0.35))
    min_static_disp = float(seg_cfg.get("min_static_display_duration", 0.4))
    max_static_disp = float(seg_cfg.get("max_static_display_duration", 2.0))

    timeline = build_timeline_from_rows(tasks, date, config=config)
    if not timeline:
        return {
            "total_disp": 0.0,
            "dyn_disp": 0.0,
            "dyn_audio_disp": 0.0,
            "static_disp": 0.0,
            "segments_count": 0,
            "dynamic_count": 0,
            "static_count": 0,
            "audio_count": 0,
        }

    plans = compute_display_plans(
        timeline,
        static_keyframe_interval=kf_interval,
        keyframe_display_duration=kf_dur,
        min_static_display_duration=min_static_disp,
        max_static_display_duration=max_static_disp,
        output_fps=float(config.get("output", {}).get("fps", 20)),
    )

    tot_disp = sum(p[0] for p in plans)
    dyn_disp = sum(p[0] for s, p in zip(timeline, plans) if s.state == "DYNAMIC")
    dyn_a_disp = sum(p[0] for s, p in zip(timeline, plans) if s.state == "DYNAMIC_AUDIO")
    static_disp = sum(p[0] for s, p in zip(timeline, plans) if s.state == "STATIC")

    return {
        "total_disp": tot_disp,
        "dyn_disp": dyn_disp,
        "dyn_audio_disp": dyn_a_disp,
        "static_disp": static_disp,
        "segments_count": len(timeline),
        "dynamic_count": sum(1 for s in timeline if s.state == "DYNAMIC"),
        "static_count": sum(1 for s in timeline if s.state == "STATIC"),
        "audio_count": sum(1 for s in timeline if s.state == "DYNAMIC_AUDIO"),
    }


def clean_empty_audio_in_db(db_path: Path, date: str | None = None, cam_index: int = 0) -> tuple[int, int]:
    """将数据库中无目标确认的纯背景音切片原子降级为 STATIC 并拓扑融合。"""
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    where_clause = "WHERE cam_index = ?"
    params: list = [cam_index]
    if date:
        where_clause += " AND date = ?"
        params.append(date)

    file_tasks = conn.execute(
        f"SELECT id, filepath, date, cam_index, file_duration, analysis_segments FROM file_tasks {where_clause} AND analysis_status='ANALYZED'",
        params,
    ).fetchall()

    total_downgraded = 0
    tasks_updated = 0

    for f in file_tasks:
        fid = f["id"]
        fp = f["filepath"]
        dt = f["date"]
        cam = f["cam_index"]
        ar_json = f["analysis_segments"]
        if not ar_json:
            continue

        segs = segments_from_json(ar_json)
        changed = False
        for s in segs:
            if s.state == "DYNAMIC_AUDIO" and (s.avg_confidence or 0.0) == 0.0:
                s.state = "STATIC"
                s.needs_review = True
                s.review_reason = "MULTIMODAL_AUDIO_NO_TARGET: 音画冲突（音频唤醒但画面无目标确认）"
                total_downgraded += 1
                changed = True

        if changed:
            merged = _merge_same_state(segs, gap_tolerance=1.5)
            merged = _filter_short(merged, min_motion=2.0, min_static=8.0, gap_tolerance=1.5)

            # 更新 file_tasks
            conn.execute(
                "UPDATE file_tasks SET analysis_segments=? WHERE id=?",
                (segments_to_json(merged), fid),
            )

            # 更新 segments 表
            conn.execute("DELETE FROM segments WHERE file_id=?", (fid,))
            new_records = [
                (
                    fid, fp, cam, dt,
                    s.start_time, s.end_time, max(0.0, s.end_time - s.start_time),
                    s.state, s.max_energy, s.avg_confidence, s.file_start_offset,
                    1 if s.needs_review else 0, s.review_reason,
                )
                for s in merged
            ]
            conn.executemany(
                """INSERT OR REPLACE INTO segments
                   (file_id, filepath, cam_index, date, start_time, end_time, duration,
                    state, max_energy, avg_confidence, file_start_offset,
                    needs_review, review_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                new_records,
            )
            tasks_updated += 1

    conn.commit()
    conn.close()
    return total_downgraded, tasks_updated


def list_review_queue(db: VlogDatabase, date: str | None = None, cam_index: int = 0, export_path: str | None = None) -> list[dict]:
    """查询并展示待人工判断的典型疑难时间戳切片清单。"""
    items = db.get_anomaly_segments(category="needs_review", date=date, cam_index=cam_index, limit=300)
    if not items:
        items = db.get_anomaly_segments(category="all", date=date, cam_index=cam_index, limit=300)

    from src.utils import ts_to_unix
    enriched = []
    for it in items:
        dt_str = it.get("date", "")
        st = float(it.get("start_time", 0.0))
        et = float(it.get("end_time", 0.0))
        base_unix = 0.0
        if dt_str and len(dt_str) == 8:
            try:
                base_unix = ts_to_unix(dt_str + "000000")
            except Exception:
                pass
        w_start = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(base_unix + st)) if base_unix > 0 else f"+{st:.1f}s"
        w_end = time.strftime("%H:%M:%S", time.localtime(base_unix + et)) if base_unix > 0 else f"+{et:.1f}s"
        in_file_s = round(max(0.0, st - float(it.get("file_start_offset", 0.0))), 1)
        in_file_e = round(max(0.0, et - float(it.get("file_start_offset", 0.0))), 1)
        it["wall_start"] = w_start
        it["wall_end"] = w_end
        it["in_file_range"] = f"{in_file_s:.1f}s ~ {in_file_e:.1f}s"
        enriched.append(it)

    print(f"\n{'=' * 85}")
    print(f"📋 待人工判断典型场景时间戳清单 (共 {len(enriched)} 项)")
    print(f"{'=' * 85}")
    print(f"{'序号':<4} | {'绝对时间戳区间':<28} | {'文件':<18} | {'片内偏移':<14} | {'状态':<7} | {'疑难原因'}")
    print("-" * 85)
    for idx, it in enumerate(enriched[:50], 1):
        fp_name = Path(it.get("filepath", "")).name[:16]
        time_span = f"{it['wall_start']} ~ {it['wall_end']}"
        reason = it.get("reason_desc", it.get("review_reason", "待复核"))
        print(f"{idx:<4} | {time_span:<28} | {fp_name:<18} | {it['in_file_range']:<14} | {it.get('state',''):<7} | {reason}")
    if len(enriched) > 50:
        print(f"... 剩余 {len(enriched) - 50} 条未展示，可指定 --export-review-queue 导出完整清单")
    print(f"{'=' * 85}\n")

    if export_path:
        out_p = Path(export_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# HomeVlog 待人工判断切片时间戳清单",
            f"- 生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"- 待审总数: {len(enriched)} 条",
            "",
            "| 序号 | 绝对时间区间 | 源文件 | 文件内起止 | 算法状态 | 能量峰值 | YOLO置信度 | 典型场景原因 | 人工判定标记 |",
            "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
        ]
        for idx, it in enumerate(enriched, 1):
            fp_name = Path(it.get("filepath", "")).name
            time_span = f"{it['wall_start']} ~ {it['wall_end']}"
            r_desc = it.get("reason_desc", it.get("review_reason", ""))
            energy = f"{float(it.get('max_energy', 0.0)):.2f}"
            conf = f"{float(it.get('avg_confidence', 0.0)):.2f}"
            lines.append(f"| {idx} | {time_span} | `{fp_name}` | {it['in_file_range']} | `{it.get('state','')}` | {energy} | {conf} | {r_desc} | [ ] |")

        out_p.write_text("\n".join(lines), encoding="utf-8")
        logger.info("已将待人工判断时间戳清单成功导出至: %s", out_p)

    return enriched


def main():
    parser = argparse.ArgumentParser(description="HomeVlog 准确度重算与净化工具")
    parser.add_argument("--db-path", type=str, default="data/vlog.db", help="数据库路径")
    parser.add_argument("--date", type=str, default=None, help="目标处理日期 (YYYYMMDD)")
    parser.add_argument("--cam", type=int, default=0, help="机位索引 (默认: 0)")
    parser.add_argument("--clean-empty-audio", action="store_true", help="将数据库中无目标确认的纯背景音切片降为 STATIC 并标记待审")
    parser.add_argument("--list-review-queue", action="store_true", help="列出待人工判断的不确定/争议切片时间戳列表")
    parser.add_argument("--export-review-queue", type=str, default=None, help="导出待审时间戳清单为 Markdown 文件")
    parser.add_argument("--reanalyze", action="store_true", help="重置分析状态以使用最新算法全量重新精析")
    parser.add_argument("--rerender", action="store_true", help="基于净化后的时间轴触发重渲染")
    args = parser.parse_args()

    db_path = Path(args.db_path)
    if not db_path.exists():
        logger.error("数据库文件不存在: %s", db_path)
        sys.exit(1)

    config = load_config()
    db = VlogDatabase(db_path)
    dates = [args.date] if args.date else ["20260320", "20260321"]

    for d in dates:
        tasks = db.get_all_file_tasks_for_date(d, args.cam)
        if not tasks:
            continue

        print(f"\n{'=' * 65}")
        print(f"📊 监控素材识别质量与成片时长审计 (日期: {d}, Cam {args.cam})")
        print(f"{'=' * 65}")

        # 1. 净化前指标
        m_before = calculate_timeline_metrics(tasks, d, config)
        print(f"【当前成片展示时长】: {m_before['total_disp']:.1f}s ({m_before['total_disp'] / 60:.1f} 分钟 / {m_before['total_disp'] / 3600:.2f} 小时)")
        print(f"  - 真实动态展示时长:   {m_before['dyn_disp']:.1f}s ({m_before['dyn_disp'] / 60:.1f} 分钟)")
        print(f"  - 音频唤醒展示时长:   {m_before['dyn_audio_disp']:.1f}s ({m_before['dyn_audio_disp'] / 60:.1f} 分钟)")
        print(f"  - 静态快进展示时长:   {m_before['static_disp']:.1f}s ({m_before['static_disp'] / 60:.1f} 分钟)")
        print(f"  - 切片总数:           {m_before['segments_count']} (动: {m_before['dynamic_count']}, 音: {m_before['audio_count']}, 静: {m_before['static_count']})")

        if args.clean_empty_audio:
            downgraded, files_upd = clean_empty_audio_in_db(db_path, date=d, cam_index=args.cam)
            logger.info("已将 %d 个无目标空房背景音切片安全降级为 STATIC (涉及 %d 个文件)", downgraded, files_upd)
            
            # 重新加载任务并计算净化后指标
            tasks_after = db.get_all_file_tasks_for_date(d, args.cam)
            m_after = calculate_timeline_metrics(tasks_after, d, config)
            print(f"\n【净化后成片展示时长】: {m_after['total_disp']:.1f}s ({m_after['total_disp'] / 60:.1f} 分钟 / {m_after['total_disp'] / 3600:.2f} 小时)")
            print(f"  - 真实动态展示时长:   {m_after['dyn_disp']:.1f}s ({m_after['dyn_disp'] / 60:.1f} 分钟)")
            print(f"  - 音频唤醒展示时长:   {m_after['dyn_audio_disp']:.1f}s ({m_after['dyn_audio_disp'] / 60:.1f} 分钟)")
            print(f"  - 静态快进展示时长:   {m_after['static_disp']:.1f}s ({m_after['static_disp'] / 60:.1f} 分钟)")
            print(f"  - 净减少无效常速播放: {m_before['total_disp'] - m_after['total_disp']:.1f}s (消灭 {(m_before['total_disp'] - m_after['total_disp'])/60:.1f} 分钟空房静止画面)")

        if args.reanalyze:
            conn = sqlite3.connect(str(db_path))
            conn.execute(
                "UPDATE file_tasks SET analysis_status='PENDING', analysis_segments=NULL WHERE date=? AND cam_index=? AND prescreen_status='SUSPICIOUS'",
                (d, args.cam),
            )
            conn.execute("DELETE FROM segments WHERE date=? AND cam_index=?", (d, args.cam))
            conn.commit()
            conn.close()
            logger.info("已重置 %s 的已分析任务状态。运行 uv run python main.py --date %s 将自动使用最新模型重跑并生成最终 DailyVlog。", d, d)

        if args.list_review_queue or args.export_review_queue:
            list_review_queue(db, date=d, cam_index=args.cam, export_path=args.export_review_queue)

        if args.rerender:
            logger.info("正在启动轻量秒级重渲染...")
            from scripts.audit_tool.rerender import ReRenderManager
            mgr = ReRenderManager()
            res = mgr.start_rerender(db, d, args.cam, output_version="accuracy_optimized")
            logger.info("重渲染已提交: %s", res)

    db.close()
    print(f"{'=' * 65}\n")


if __name__ == "__main__":
    main()
