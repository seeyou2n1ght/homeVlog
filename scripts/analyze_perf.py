"""perf_*.json 性能产物聚合分析工具.

将每日 perf JSON 汇总为瓶颈评估视图：
1. 单日头条：处理倍速 (speedup_x)、浓缩率、产出体积；
2. 分阶段耗时分布与估算墙钟占比（识别当前瓶颈阶段）；
3. 分析阶段 decode/analysis 拆解、渲染 staging/编码与最终合并统计；
4. 信号量等待 (sem_wait) 汇总（饥饿探测）；
5. 各阶段 Top-N 最慢记录（长尾定位）。

用法:
    uv run python scripts/analyze_perf.py [--top 5] [--glob "logs/perf_*.json"]
"""

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


def _fmt_s(v: float) -> str:
    return f"{v:.1f}s" if v < 120 else f"{v/60:.1f}min"


def analyze_file(path: Path, top_n: int) -> None:
    d = json.load(open(path, encoding="utf-8"))
    date = d.get("date", "?")
    wall = d.get("pipeline_duration", 0)
    headline = d.get("headline") or {}
    summary = d.get("perf_summary", {})
    records = d.get("records", [])
    # monitor stage 跨日累积：取与本文件日期匹配的阶段，兜底取最后一个
    stages = d.get("monitor_summary") or [{}]
    mon = next(
        (s for s in stages if s.get("name") == f"pipeline_{date}_cam{d.get('cam', 0)}"),
        stages[-1],
    )

    print("=" * 78)
    print(f"📅 {date}  总耗时 {_fmt_s(wall)}", end="")
    if headline:
        print(
            f"  │  输入 {_fmt_s(headline.get('input_dur_s', 0))}"
            f"  │  倍速 {headline.get('speedup_x', '?')}x"
            f"  │  浓缩 {headline.get('condensation_x', '?')}x"
            f"  │  产出 {headline.get('output_size_mb', '?')} MB"
        )
    else:
        print("  │  (无 headline，旧格式)")
    gpu0 = mon.get("gpu0") or {}
    igpu = mon.get("igpu") or {}
    print(
        f"   硬件: CPU {mon.get('avg_cpu', 0):.0f}%  RAM {mon.get('avg_ram', 0):.0f}%"
        f"  │  dGPU load {gpu0.get('avg_load', 0):.0f}% enc {gpu0.get('avg_enc', 0):.0f}%"
        f" dec {gpu0.get('avg_dec', 0):.0f}% mem_peak {gpu0.get('peak_mem_mb', 0)}MB"
        + (f"  │  iGPU {igpu}" if igpu else "")
    )

    # 累计工时与并发度
    total_worker_time = sum(s.get("total", 0.0) for s in summary.values())
    concurrency_x = total_worker_time / max(wall, 0.1)
    print(f"   ⏱ 工时汇总: 累计工时 {_fmt_s(total_worker_time)}  │  端到端壁钟 {_fmt_s(wall)}  │  并发重叠系数 {concurrency_x:.2f}x")

    # 分阶段统计 (注: 累计工时为各 Worker 并发总时长，由于多路并行，该和值大于端到端壁钟总耗时)
    print(f"   {'阶段 (Stage)':<16} {'数量(n)':>6} {'累计工时(Worker)':>16} {'均值(avg)':>10} {'P95':>10} {'最大(max)':>10}")
    for st in ("prescreen", "analysis", "render_stage", "render_enc", "render", "final_concat"):
        s = summary.get(st)
        if s:
            print(
                f"   {st:<16} {s['count']:>6} {_fmt_s(s['total']):>16}"
                f" {_fmt_s(s['avg']):>10} {_fmt_s(s['p95']):>10} {_fmt_s(s['max']):>10}"
            )

    # 分析阶段 decode / motion / yolo 细粒度拆解
    ana = [r for r in records if r.get("stage") == "analysis" and r.get("extra")]
    if ana:
        dec = [r["extra"].get("decode_time_s", r["extra"].get("decode_time", 0)) for r in ana]
        mot = [r["extra"].get("analysis_time_s", r["extra"].get("analysis_time", 0)) for r in ana]
        yolo_durs = [r["extra"].get("yolo_time_s", 0) for r in ana]
        overheads = [r["extra"].get("overhead_s", 0) for r in ana]
        total_dec = sum(dec)
        total_mot = sum(mot)
        total_yolo = sum(yolo_durs)
        total_ovh = sum(overheads)
        print(
            f"   分析工时拆解: 解码 {_fmt_s(total_dec)} (均 {statistics.mean(dec):.1f}s)"
            f" │ 运动检测 {_fmt_s(total_mot)} (均 {statistics.mean(mot):.2f}s)"
            f" │ YOLO验证 {_fmt_s(total_yolo)} (均 {statistics.mean(yolo_durs):.2f}s)"
            f" │ 拓扑/归档/DB {_fmt_s(total_ovh)}"
        )

    # YOLO 验证汇总
    yolo_sum = d.get("yolo_summary")
    if not yolo_sum:
        # 兼容从 records 动态回算
        yolo_recs = [r for r in ana if "yolo_frames" in r["extra"]]
        if yolo_recs:
            tf = sum(r["extra"].get("yolo_frames", 0) for r in yolo_recs)
            ti = sum(r["extra"].get("yolo_infer_time", 0.0) for r in yolo_recs)
            tw = sum(r["extra"].get("yolo_lock_wait", 0.0) for r in yolo_recs)
            td = sum(r["extra"].get("yolo_duration", 0.0) for r in yolo_recs)
            c_segs = sum(r["extra"].get("yolo_candidate_segments", 0) for r in yolo_recs)
            cf_segs = sum(r["extra"].get("yolo_confirmed_segments", 0) for r in yolo_recs)
            rj_segs = sum(r["extra"].get("yolo_rejected_segments", 0) for r in yolo_recs)
            yolo_sum = {
                "files_evaluated": len(yolo_recs),
                "total_duration_s": round(td, 2),
                "total_lock_wait_s": round(tw, 2),
                "total_infer_s": round(ti, 2),
                "total_frames": tf,
                "avg_infer_ms_per_frame": round((ti * 1000 / tf), 2) if tf > 0 else 0.0,
                "candidate_segments": c_segs,
                "confirmed_segments": cf_segs,
                "rejected_segments": rj_segs,
                "suppression_rate_pct": round(rj_segs / max(1, c_segs) * 100, 1),
            }
    if yolo_sum:
        print(
            f"   🎯 YOLO 细测: 验证文件 {yolo_sum['files_evaluated']}个"
            f" │ 帧数 {yolo_sum['total_frames']} (纯推理 {_fmt_s(yolo_sum['total_infer_s'])}, 均 {yolo_sum['avg_infer_ms_per_frame']}ms/帧)"
            f" │ 锁等待 {_fmt_s(yolo_sum['total_lock_wait_s'])}"
            f" │ 动作段 {yolo_sum['candidate_segments']} -> 确认 {yolo_sum['confirmed_segments']} / 误报压制 {yolo_sum['rejected_segments']}"
            f" (压制率 {yolo_sum['suppression_rate_pct']}%)"
        )

    # 渲染 Worker 繁忙度与工作窃取
    w_stats = d.get("worker_stats") or {}
    if w_stats:
        print("   ⚙️ 渲染编队利用率 (Worker Utilization):")
        for wid, ws in sorted(w_stats.items()):
            extra_info = ""
            if ws.get("qsv_stolen_batches") or ws.get("qsv_light_batches"):
                extra_info = f" (轻批次 {ws.get('qsv_light_batches', 0)} │ 逆向窃取 {ws.get('qsv_stolen_batches', 0)} │ 窃取拒收 {ws.get('qsv_steal_rejected', 0)})"
            print(
                f"      [{wid:<7}] 繁忙 {_fmt_s(ws['busy_time_s'])} / 空闲 {_fmt_s(ws['idle_time_s'])}"
                f" │ 利用率 {ws['utilization_pct']:>5.1f}%"
                f" │ 产出 {ws['batches_rendered']:>3} 批次"
                f" ({_fmt_s(ws['dynamic_sec_rendered'])} 动态素材){extra_info}"
            )

    # 渲染编码吞吐
    enc = [r for r in records if r.get("stage") == "render_enc" and r.get("extra", {}).get("encode_speed_x")]
    if enc:
        speeds = [r["extra"]["encode_speed_x"] for r in enc]
        print(
            f"   渲染编码: speed 均值 {statistics.mean(speeds):.2f}x"
            f"  最低 {min(speeds):.2f}x  最高 {max(speeds):.2f}x"
        )

    # 信号量等待汇总（饥饿探测）
    sem = {}
    for r in records:
        w = (r.get("extra") or {}).get("sem_wait")
        if w:
            sem[r["stage"]] = sem.get(r["stage"], 0.0) + w
    if sem:
        print(f"   信号量等待总计: {dict((k, _fmt_s(v)) for k, v in sem.items())}")

    # 阶段流水线起止时间轴分析 (若包含时间戳)
    timed_recs = [r for r in records if r.get("start_time") and r.get("end_time")]
    if timed_recs:
        t_base = min(r["start_time"] for r in timed_recs)
        stages_spans = {}
        for r in timed_recs:
            st = r["stage"]
            stages_spans.setdefault(st, []).append((r["start_time"] - t_base, r["end_time"] - t_base))
        print("   📈 流水线重叠时间轴 (Timeline Spans):")
        for st in ("prescreen", "analysis", "render_stage", "render", "final_concat"):
            if st in stages_spans:
                s_min = min(s[0] for s in stages_spans[st])
                s_max = max(s[1] for s in stages_spans[st])
                print(f"      {st:<14} T+{_fmt_s(s_min):<8} -> T+{_fmt_s(s_max):<8} (跨度 {_fmt_s(s_max - s_min)})")

    # 瓶颈自动诊断与优化建议 (RCA)
    rca = []
    if w_stats:
        nv_idles = [ws["idle_time_s"] for wid, ws in w_stats.items() if ws.get("gpu") == "nv"]
        qsv_ws = [ws for wid, ws in w_stats.items() if ws.get("gpu") == "qsv"]
        if nv_idles and statistics.mean(nv_idles) > wall * 0.25:
            rca.append(f"NVENC 渲染端存在明显等待饥饿 (平均空闲 {_fmt_s(statistics.mean(nv_idles))}, 占壁钟 {statistics.mean(nv_idles)/wall*100:.1f}%)，瓶颈在前期分析产出速度。")
        if qsv_ws and qsv_ws[0]["utilization_pct"] < 45.0:
            rca.append(f"QSV 核显利用率偏低 ({qsv_ws[0]['utilization_pct']:.1f}%)，建议评估调优工作窃取门限以分担更多渲染。")
        if qsv_ws and qsv_ws[0].get("qsv_steal_rejected", 0) > 15:
            rca.append(f"QSV 尝试窃取动态批次被拒收 {qsv_ws[0]['qsv_steal_rejected']} 次，表明队列动态时长偏重或门限约束较紧。")
    if yolo_sum and yolo_sum.get("total_lock_wait_s", 0) > 30.0:
        rca.append(f"YOLO 推理锁争用严重 (累计排队 {_fmt_s(yolo_sum['total_lock_wait_s'])})，建议加大分析端帧批量或降低 YOLO 采样率。")
    if ana:
        total_dec = sum(r["extra"].get("decode_time_s", r["extra"].get("decode_time", 0)) for r in ana)
        if total_dec > total_worker_time * 0.40:
            rca.append(f"视频解码占总工时 {total_dec/total_worker_time*100:.1f}%，是分析阶段最大物理开销。")

    if rca:
        print("   🔍 瓶颈自动研判 (Bottleneck RCA):")
        for item in rca:
            print(f"      • {item}")

    # Top-N 最慢
    for st in ("prescreen", "analysis", "render"):
        recs = sorted(
            (r for r in records if r.get("stage") == st),
            key=lambda r: r.get("duration", 0), reverse=True,
        )[:top_n]
        if recs:
            tops = ", ".join(f"{r['file'][:26]}({_fmt_s(r['duration'])})" for r in recs)
            print(f"   {st} Top{top_n}: {tops}")


def main():
    ap = argparse.ArgumentParser(description="HomeVlog perf 聚合瓶颈分析")
    ap.add_argument("--top", type=int, default=5, help="各阶段展示最慢记录数")
    ap.add_argument("--glob", default="logs/**/perf_*.json", help="perf JSON 匹配模式")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob, recursive=True))
    if not files:
        print(f"未找到 perf 文件: {args.glob}")
        return
    for f in files:
        try:
            analyze_file(Path(f), args.top)
        except Exception as e:
            print(f"{f}: 解析失败 {e}")
    print("=" * 78)


if __name__ == "__main__":
    main()
