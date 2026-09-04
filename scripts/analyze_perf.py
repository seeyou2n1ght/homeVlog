"""perf_*.json 性能产物聚合分析工具.

将每日 perf JSON 汇总为瓶颈评估视图：
1. 单日头条：处理倍速 (speedup_x)、浓缩率、产出体积；
2. 分阶段耗时分布与估算墙钟占比（识别当前瓶颈阶段）；
3. 分析阶段 decode/analysis 拆解 与 渲染 encode_speed_x 统计；
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

    # 分阶段统计
    print(f"   {'stage':<12} {'n':>4} {'total':>9} {'avg':>8} {'p95':>8} {'max':>9}")
    for st in ("prescreen", "analysis", "render", "render_enc"):
        s = summary.get(st)
        if s:
            print(
                f"   {st:<12} {s['count']:>4} {_fmt_s(s['total']):>9}"
                f" {_fmt_s(s['avg']):>8} {_fmt_s(s['p95']):>8} {_fmt_s(s['max']):>9}"
            )

    # 分析阶段 decode/analysis 拆解
    ana = [r for r in records if r.get("stage") == "analysis" and r.get("extra")]
    if ana:
        dec = [r["extra"].get("decode_time", 0) for r in ana]
        cpu = [r["extra"].get("analysis_time", 0) for r in ana]
        print(
            f"   分析拆解: decode 均值 {statistics.mean(dec):.1f}s"
            f" vs CPU 分析均值 {statistics.mean(cpu):.1f}s"
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
    ap.add_argument("--glob", default="logs/perf_*.json", help="perf JSON 匹配模式")
    args = ap.parse_args()

    files = sorted(glob.glob(args.glob))
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
