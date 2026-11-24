"""HomeVlog 终端富文本呈现与实时仪表盘模块 (Rich Terminal UI & Dashboard).

包含：
- PipelineDashboard: 负责多阶段并发流水线的实时 Live 仪表盘，整合进度条、队列水位、调度状态、显存监测与活动追踪；
- PlainProgressTracker: 针对非 TTY / 无头环境 (--no-tui / CI) 的流式心跳进度汇报器；
- 启动面板 (Startup Banner / Batch Startup Banner) 与动态硬件感知；
- 完成总结看板 (Summary Card) 与多日全景汇总大表 (Batch Summary Table)；
- 数据库归档总览看板 (Status Table) 与扫描结果表格 (Scan Results Table)；
- 错误汇总卡片 (Error Summary)。
"""

import json
import logging
from datetime import datetime, timedelta
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.core.run_observation import current_run
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Windows 控制台编码防护 (安全向下兼容 UTF-8)
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    try:
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

console = Console()

# DEFAULT_DEVICE_INFO removed in favor of dynamic hardware summary


def is_interactive_terminal() -> bool:
    """判断当前环境是否为支持交互式光标刷新的终端。"""
    return console.is_terminal and not console.is_dumb_terminal


def get_hardware_summary_string() -> str:
    """动态探测当前平台 CPU 与 GPU 算力组合描述，取代静态写例文案。"""
    try:
        import torch
        if torch.cuda.is_available():
            dev_name = torch.cuda.get_device_name(0)
            dev_clean = dev_name.replace("NVIDIA GeForce ", "").replace("NVIDIA ", "")
            return f"独显 {dev_clean}（AI 分析+渲染）＋ 核显/CPU（粗筛解码）"
    except Exception:
        pass
    return "核显/CPU（解码+粗筛）＋ 独立算力（AI 分析+渲染）"


def print_batch_startup_banner(
    total_groups: int,
    date_range: Tuple[str, str],
    cameras: List[str],
    output_dir: str = "output",
    device_info: str | None = None,
) -> None:
    """在多天批量浓缩任务启动时，打印宏观全局规划 Banner。"""
    if not device_info:
        device_info = get_hardware_summary_string()

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="white")

    start_date, end_date = date_range
    date_str = f"{start_date} ~ {end_date}" if start_date != end_date else start_date
    table.add_row("📅 任务规划:", f"共发现 [bold yellow]{total_groups}[/] 个归档任务 (日期跨度: [bold white]{date_str}[/])")

    cam_str = ", ".join(cameras) if cameras else "全机位"
    table.add_row("📷 监控机位:", f"[bold green]{cam_str}[/]")
    table.add_row("⚡ 硬件分工:", f"[bold magenta]{device_info}[/]")
    table.add_row("📦 归档输出:", f"[dim]{output_dir}/ (成片按日自动命名归档)[/dim]")

    panel = Panel(
        table,
        title="[bold blue]🚀 HomeVlog 多天监控批量浓缩任务启动[/bold blue]",
        subtitle="[dim]Batch Video Summarization Engine[/dim]",
        border_style="bright_blue",
        padding=(1, 2),
    )
    console.print(panel)


def print_startup_banner(
    date: str,
    cam_index: int,
    total_files: int,
    total_duration_s: float,
    device_info: str | None = None,
    output_path: Optional[str] = None,
    cam_name: Optional[str] = None,
    batch_progress: Optional[str] = None,
) -> None:
    """以高质感 Rich 面板展示流水线启动配置与硬件就绪状态。"""
    if not device_info:
        device_info = get_hardware_summary_string()

    hours = total_duration_s / 3600.0

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="white")

    cam_label = cam_name or f"Cam {cam_index}"
    progress_tag = f" [bold cyan]{batch_progress}[/]" if batch_progress else ""
    table.add_row("📅 处理日期:", f"[bold white]{date}[/]{progress_tag}")
    table.add_row("📷 监控机位:", f"[bold yellow]{cam_label}[/]")

    table.add_row(
        "🎬 输入素材:",
        f"[bold green]{total_files}[/] 个监控切片 | 原始总时长: [bold yellow]{hours:.2f}[/] 小时",
    )
    table.add_row("⚡ 硬件分工:", f"[bold magenta]{device_info}[/]")
    if output_path:
        table.add_row("📦 成片目标:", f"[dim]{output_path}[/dim]")

    panel = Panel(
        table,
        title="[bold blue]🚀 HomeVlog 智能监控浓缩流水线[/bold blue]",
        subtitle="[dim]Heterogeneous Hardware Orchestration Engine[/dim]",
        border_style="bright_blue",
        padding=(1, 2),
    )
    console.print(panel)


def print_summary_card(
    date: str,
    cam_index: int,
    total_files: int,
    total_input_dur: float,
    output_path: Path,
    elapsed_wall: float,
    stage_durations: Optional[Dict[str, float]] = None,
    cam_name: Optional[str] = None,
    vlog_dur: Optional[float] = None,
    dynamic_dur: Optional[float] = None,
    static_dur: Optional[float] = None,
) -> None:
    """输出美观的流水线完成报告卡片与指标分解表。"""
    file_sz_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
    speedup = (total_input_dur / elapsed_wall) if elapsed_wall > 0 else 0.0
    el_m, el_s = divmod(int(elapsed_wall), 60)

    # 尝试从伴随 .meta.json 中提取真实时长与指标（若未显式传入）
    meta_path = output_path.with_suffix(".meta.json")
    by_state = {}
    m_data = {}
    if meta_path.exists():
        try:
            m_data = json.loads(meta_path.read_text(encoding="utf-8"))
            metrics = m_data.get("metrics", {})
            if vlog_dur is None:
                vlog_dur = metrics.get("vlog_duration_s")
                dynamic_dur = metrics.get("dynamic_duration_s")
                static_dur = metrics.get("static_duration_s")
            by_state = metrics.get("source_duration_by_state_s", {})
        except (OSError, ValueError, AttributeError) as exc:
            m_data = {}
            logging.getLogger("homevlog").warning("Delivery summary metadata unreadable: %s", exc)

    # 核心指标统计表
    metric_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    metric_table.add_column("指标类别", style="bold", width=18)
    metric_table.add_column("数值与明细", style="white")
    metric_table.add_row("监控机位", cam_name or f"Cam {cam_index}")

    metric_table.add_row(
        "输入素材",
        f"[bold]{total_files}[/] 个文件 (总计 [yellow]{total_input_dur / 3600:.2f}[/] 小时)",
    )
    metric_table.add_row(
        "输出成片",
        f"[bold green]{output_path.name}[/] ([cyan]{file_sz_mb:.2f} MB[/cyan])",
    )

    metric_table.add_row("本地成片", "存在" if output_path.is_file() else "未找到")
    mapping = m_data.get("delivery_mapping", {})
    clock = mapping.get("clock_validation", {})
    if mapping.get("status") == "AVAILABLE" and clock.get("method") == "packet_pts":
        metric_table.add_row("交付时时钟", f"{clock['packets']:,} 个视频包 · 最大误差 {clock['max_error_ticks']} tick")
    else:
        metric_table.add_row("交付时时钟", "无校验记录")
    if "unavailable_source_ranges" in m_data and "skipped_sources" in m_data:
        gaps = m_data["unavailable_source_ranges"]
        skipped = m_data["skipped_sources"]
        missing_s = sum(item["missing_duration_s"] for item in gaps)
        skipped_s = sum(item.get("nominal_duration_s") or 0 for item in skipped)
        metric_table.add_row("源素材缺口", f"缺失 {len(gaps)} 段 / {missing_s:.1f} 秒 · 跳过 {len(skipped)} 个文件 / 名义 {skipped_s:.1f} 秒")
        for item in gaps:
            start = datetime.strptime(item["source_start"], "%Y%m%d%H%M%S")
            left = start + timedelta(seconds=item["missing_start_offset_s"])
            right = start + timedelta(seconds=item["missing_end_offset_s"])
            metric_table.add_row("缺失源时段", f"{left:%m-%d %H:%M:%S} – {right:%m-%d %H:%M:%S}")
        for item in skipped:
            metric_table.add_row("跳过源时段", f"{item['source_start']} – {item['source_end']}")
    if vlog_dur and vlog_dur > 0:
        condense_ratio = total_input_dur / max(vlog_dur, 0.1)
        v_m, v_s = divmod(int(vlog_dur), 60)
        metric_table.add_row(
            "成片时长与浓缩比",
            f"[bold green]{condense_ratio:.1f}× 浓缩[/bold green] (成片时长: [cyan]{v_m:02d}分{v_s:02d}秒[/cyan])",
        )

    if by_state:
        names = {"DYNAMIC": "动态", "DYNAMIC_AUDIO": "声音活动", "STATIC": "静态",
                 "PRESENCE": "有人驻留", "NIGHT_STATIONARY": "夜间驻留", "MICRO_MOTION": "微动"}
        metric_table.add_row("源时段分布", " | ".join(
            f"{names.get(state, state)}: {duration:.1f}s" for state, duration in by_state.items()))
    elif dynamic_dur is not None and static_dur is not None and (dynamic_dur + static_dur) > 0:
        total_content = dynamic_dur + static_dur
        dyn_pct = (dynamic_dur / total_content) * 100
        sta_pct = (static_dur / total_content) * 100
        metric_table.add_row(
            "高光/静态子集分布",
            f"[yellow]高光动态: {dyn_pct:.1f}% ({dynamic_dur:.1f}s)[/] | [dim]静态快进: {sta_pct:.1f}% ({static_dur:.1f}s)[/dim]",
        )

    metric_table.add_row(
        "总耗时与处理速度",
        f"[bold yellow]{el_m:02d}分{el_s:02d}秒[/bold yellow] ([bold red]{speedup:.1f}× 实时[/bold red])",
    )

    # 伴随资产检查
    srt_path = output_path.with_suffix(".srt")
    assets = []
    if srt_path.exists():
        assets.append(f"[green]✔ 字幕 ({srt_path.name})[/green]")
    if meta_path.exists():
        assets.append(f"[green]✔ 元数据 ({meta_path.name})[/green]")
    if assets:
        metric_table.add_row("伴随资产", " | ".join(assets))

    metric_table.add_row(
        "输出路径",
        f"{str(output_path)}",
    )

    # 若有分阶段用时，追加阶段分解表
    renderables: List[Any] = [metric_table]
    if stage_durations:
        stage_table = Table(title="[bold dim]⏱️ 阶段累计耗时统计[/bold dim]", border_style="dim", padding=(0, 2))
        stage_table.add_column("阶段", style="cyan")
        stage_table.add_column("总用时", justify="right", style="yellow")
        stage_table.add_column("占总时长比率", justify="right", style="magenta")

        sum_stages = sum(stage_durations.values()) or 1.0
        for name, dur in stage_durations.items():
            pct = (dur / sum_stages) * 100.0
            stage_table.add_row(name.capitalize(), f"{dur:.2f}s", f"{pct:.1f}%")
        renderables.append(stage_table)

    panel = Panel(
        Group(*renderables),
        title=f"[bold green]✨ HomeVlog 处理完成 | {date}[/bold green]",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)


def print_batch_summary_table(results: List[Dict[str, Any]]) -> None:
    """输出多任务批处理全局全景汇总看板。"""
    if not results:
        return

    table = Table(
        title="[bold blue]📊 HomeVlog 批量浓缩任务全局全景汇总[/bold blue]",
        border_style="bright_blue",
        show_footer=True,
    )
    table.add_column("序号", justify="center", style="dim", footer="总计")
    table.add_column("归档日期", justify="center", style="bold yellow")
    table.add_column("监控机位", justify="center", style="green")
    table.add_column("切片数", justify="right", style="cyan", footer=str(sum(r.get("total_files", 0) for r in results)))

    total_in_dur = sum(r.get("input_duration_s", 0.0) for r in results)
    total_vlog_dur = sum(r.get("vlog_duration_s", 0.0) for r in results)
    total_size_mb = sum(r.get("output_size_mb", 0.0) for r in results)
    total_wall_s = sum(r.get("wall_clock_s", 0.0) for r in results)
    avg_speedup = (total_in_dur / total_wall_s) if total_wall_s > 0 else 0.0

    table.add_column("原始时长", justify="right", style="yellow", footer=f"{total_in_dur / 3600:.2f}h")
    table.add_column("成片时长", justify="right", style="cyan", footer=f"{total_vlog_dur / 60:.1f}m")
    table.add_column("浓缩比", justify="right", style="bold green", footer=f"{(total_in_dur / max(total_vlog_dur, 0.1)):.1f}×")
    table.add_column("成片体积", justify="right", style="magenta", footer=f"{total_size_mb:.1f} MB")
    table.add_column("处理壁钟", justify="right", style="yellow", footer=f"{int(total_wall_s // 60)}分{int(total_wall_s % 60)}秒")
    table.add_column("处理速度", justify="right", style="bold red", footer=f"{avg_speedup:.1f}×")
    table.add_column("状态", justify="center", footer=f"{sum(1 for r in results if r.get('status') == 'SUCCESS')}/{len(results)} 成功")

    for idx, r in enumerate(results, 1):
        d = r.get("date", "-")
        cam = r.get("cam_name", f"Cam {r.get('cam_index', 0)}")
        n_files = str(r.get("total_files", 0))
        in_h = f"{r.get('input_duration_s', 0.0) / 3600:.2f}h"
        v_m = f"{r.get('vlog_duration_s', 0.0) / 60:.1f}m"
        condense = f"{r.get('condensation_ratio', 0.0):.1f}×" if r.get("condensation_ratio") else "-"
        sz = f"{r.get('output_size_mb', 0.0):.1f} MB"
        w_s = int(r.get("wall_clock_s", 0.0))
        w_str = f"{w_s // 60:02d}:{w_s % 60:02d}"
        spd = f"{r.get('speedup_x', 0.0):.1f}×"
        st = {"SUCCESS": "[bold green]✔ 完成[/bold green]",
              "SKIPPED": "[cyan]已交付跳过[/cyan]"}.get(r.get("status"), "[bold red]✖ 失败[/bold red]")

        table.add_row(str(idx), d, cam, n_files, in_h, v_m, condense, sz, w_str, spd, st)

    console.print(table)


def print_status_table(db: Any, camera_display_names: Optional[Dict[int, str]] = None) -> None:
    """展示当前数据库中全部日期/机位的归档任务处理进度与成片状态。"""
    rows = db.conn.execute(
        """SELECT 
            date, 
            cam_index, 
            COUNT(*) as total_files,
            SUM(CASE WHEN prescreen_status != 'PENDING' THEN 1 ELSE 0 END) as prescreen_done,
            SUM(CASE WHEN prescreen_status = 'SUSPICIOUS' THEN 1 ELSE 0 END) as suspicious_count,
            SUM(CASE WHEN analysis_status != 'PENDING' AND prescreen_status = 'SUSPICIOUS' THEN 1 ELSE 0 END) as analysis_done,
            SUM(file_duration) as total_duration
        FROM file_tasks 
        GROUP BY date, cam_index
        ORDER BY date, cam_index"""
    ).fetchall()

    if not rows:
        console.print("[yellow]数据库中暂无任何监控切片任务记录。可运行 'uv run python main.py --scan' 扫描素材。[/yellow]")
        return

    render_tasks = {}
    try:
        r_rows = db.conn.execute("SELECT date, cam_index, status, output_file, error_msg FROM render_tasks").fetchall()
        for r in r_rows:
            render_tasks[(r["date"], r["cam_index"])] = (r["status"], r["output_file"], r["error_msg"])
    except Exception:
        pass

    table = Table(
        title="[bold cyan]📋 HomeVlog 素材归档与处理进度总览[/bold cyan]",
        border_style="dim",
        expand=False,
    )
    table.add_column("序号", justify="center", style="dim", no_wrap=True)
    table.add_column("归档日期", justify="center", style="bold yellow", no_wrap=True)
    table.add_column("监控机位", justify="center", style="green", no_wrap=True)
    table.add_column("切片数", justify="right", style="cyan", no_wrap=True)
    table.add_column("素材时长", justify="right", style="yellow", no_wrap=True)
    table.add_column("预筛选进度", justify="center", style="white", no_wrap=True)
    table.add_column("AI精析进度", justify="center", style="white", no_wrap=True)
    table.add_column("成片状态", justify="center", no_wrap=True)
    table.add_column("成片体积", justify="right", style="magenta", no_wrap=True)

    failures = []
    for idx, r in enumerate(rows, 1):
        d = r["date"]
        cam = r["cam_index"]
        cam_text = camera_display_names.get(cam, f"Cam {cam}") if camera_display_names else f"Cam {cam}"
        if " (" in cam_text:
            cam_text = cam_text.split(" (")[0]
        total_f = r["total_files"]
        dur_h = (r["total_duration"] or 0.0) / 3600.0

        p_done = r["prescreen_done"] or 0
        p_str = f"{p_done}/{total_f} ({int(p_done/total_f*100)}%)"

        sus_cnt = r["suspicious_count"] or 0
        a_done = r["analysis_done"] or 0
        if sus_cnt == 0:
            a_str = "[dim]0 疑点 (纯静态)[/dim]"
        else:
            a_str = f"{a_done}/{sus_cnt} ({int(a_done/sus_cnt*100)}%)"

        status, out_file, error_msg = render_tasks.get((d, cam), ("PENDING", None, None))
        out_p = Path(out_file) if out_file else None
        has_out = out_p.exists() if out_p else False
        out_size_mb = (out_p.stat().st_size / (1024 * 1024)) if has_out else 0.0

        if status == "FAILED":
            st_text = "[bold red]✖ 渲染失败[/bold red]"
            sz_text = "-"
            if error_msg:
                failures.append(f"{d} · {cam_text}: {error_msg}")
        elif has_out:
            st_text = "[bold green]✔ 已就绪[/bold green]"
            sz_text = f"{out_size_mb:.1f} MB"
        elif status == "COMPLETED":
            st_text = "[green]✔ 已渲染[/green]"
            sz_text = "[dim]记录的路径下未找到文件[/dim]" if out_file else "[dim]未知路径[/dim]"
        else:
            st_text = "[yellow]⏳ 待处理[/yellow]"
            sz_text = "-"

        table.add_row(str(idx), str(d), cam_text, str(total_f), f"{dur_h:.2f} 小时", p_str, a_str, st_text, sz_text)

    console.print(table)
    if failures:
        print_error_summary(failures)


def print_error_summary(errors: List[str]) -> None:
    """输出醒目的失败警示卡片。"""
    if not errors:
        return

    text = Text()
    for idx, err in enumerate(errors, 1):
        text.append(f" {idx}. {err}\n", style="bold red")

    panel = Panel(
        text,
        title=f"[bold red]⚠️ 运行告警与异常报告 (共 {len(errors)} 项)[/bold red]",
        border_style="red",
        padding=(1, 2),
    )
    console.print(panel)


def print_scan_results(
    groups: List[Tuple[str, int]],
    camera_display_names: Optional[Dict[int, str]] = None,
    group_stats: Optional[Dict[Tuple[str, int], Dict[str, Any]]] = None,
) -> None:
    """以整洁的表格展示扫描出来的日期与机位组合（兼容轻量与增强统计模式）。"""
    table = Table(title="[bold cyan]🔍 监控素材归档组扫描结果[/bold cyan]", border_style="dim")
    table.add_column("序号", justify="center", style="dim", width=6)
    table.add_column("归档日期 (Date)", justify="center", style="bold yellow")
    table.add_column("监控机位 (Camera)", justify="center", style="bold green")

    has_stats = bool(group_stats)
    if has_stats:
        table.add_column("切片数量", justify="right", style="cyan")
        table.add_column("素材时长", justify="right", style="yellow")
        table.add_column("处理状态", justify="center", style="magenta")

    for idx, (date, cam) in enumerate(groups, 1):
        cam_text = camera_display_names.get(cam, f"Cam {cam}") if camera_display_names else f"Cam {cam}"
        if has_stats and group_stats and (date, cam) in group_stats:
            st = group_stats[(date, cam)]
            n_f = str(st.get("total_files", "-"))
            dur_str = f"{st.get('duration_s', 0)/3600:.2f} 小时" if "duration_s" in st else "-"
            status_str = st.get("status_tag", "[yellow]待处理[/yellow]")
            table.add_row(str(idx), str(date), cam_text, n_f, dur_str, status_str)
        else:
            table.add_row(str(idx), str(date), cam_text)

    console.print(table)


class PlainProgressTracker:
    """针对无头环境 (--no-tui / CI / 管道重定向) 的流式单行心跳进度汇报器。"""

    def __init__(self, date: str, cam_index: int, total_files: int, interval_s: float = 12.0, cam_name: str | None = None):
        self.date = date
        self.cam_index = cam_index
        self.cam_name = cam_name or f"Cam {cam_index}"
        self.total_files = max(1, total_files)
        self.interval_s = interval_s
        self._last_report = 0.0
        self._start_time = time.monotonic()
        self._lock = threading.Lock()

    def heartbeat(
        self,
        prescreen_done: int,
        prescreen_total: int,
        analysis_done: int,
        analysis_total: int,
        render_done: int,
        render_total: int,
        force: bool = False,
        detail: str = "",
    ) -> None:
        now = time.monotonic()
        with self._lock:
            if not force and (now - self._last_report) < self.interval_s:
                return
            self._last_report = now
            elapsed = int(now - self._start_time)
            m, s = divmod(elapsed, 60)

            p_pct = int((prescreen_done / max(1, prescreen_total)) * 100)
            a_pct = int((analysis_done / max(1, analysis_total)) * 100) if analysis_total > 0 else 0
            r_pct = int((render_done / max(1, render_total)) * 100) if render_total > 0 else 0

            now_str = time.strftime("%H:%M:%S")
            render_progress = "" if detail else f"③渲染: {render_done}/{render_total} ({r_pct}%) | "
            line = (
                f"[{now_str}] [HomeVlog {self.date} | 机位 {self.cam_name}] "
                f"①粗筛: {prescreen_done}/{prescreen_total} ({p_pct}%) | "
                f"②精析: {analysis_done}/{analysis_total} ({a_pct}%) | "
                f"{render_progress}"
                f"耗时: {m:02d}:{s:02d}"
            )
            console.print(Text(line + (" | " + detail if detail else ""), style="dim"))


class PipelineDashboard:
    """
    终极多任务终端动态仪表盘 (Unified Multi-Stage Terminal Dashboard).

    集成：
    - 三阶段进度管理 (Prescreen, Analysis, Render)
    - 队列积压水位实时监视
    - 异构调度器硬件状态徽标 (QSV 独占 / NVDEC 协同 / 渲染抢占)
    - VRAM 显存实时监控指示
    - 活跃事件与告警滚动追踪
    - 非 TTY / 无头环境的流式心跳优雅降级 (PlainProgressTracker)
    """

    def __init__(
        self,
        date: str,
        cam_index: int,
        total_prescreen: int = 0,
        render_enabled: bool = True,
        enabled: bool = True,
        cam_name: str | None = None,
    ):
        self.date = date
        self.cam_index = cam_index
        self.cam_name = cam_name or f"Cam {cam_index}"
        self.render_enabled = render_enabled
        self.enabled = enabled and is_interactive_terminal()

        self._lock = threading.Lock()
        self._start_time = time.monotonic()

        # 状态指标与工序追踪
        self.total_files = max(1, total_prescreen)
        self.prescreen_done = 0
        self.prescreen_total = total_prescreen
        self.analysis_done = 0
        self.analysis_cached = 0
        self.analysis_total = 0
        self.render_done = 0
        self.render_total = 0

        self.last_prescreen_speed = ""
        self.last_analysis_speed = ""
        self.last_render_speed = ""

        self.prescreen_queue_size = 0
        self.analysis_queue_size = 0
        self.render_queue_size = 0
        self.scheduler_state = "NORMAL_DECOUPLED"
        self.active_nv_decoders = 0
        self.max_nv_decoders = 1

        self.last_prescreen_file = ""
        self.last_analysis_file = ""
        self.last_render_batch = ""
        self.recent_alerts: List[str] = []

        # 渲染批次文件级真实计数
        self.render_done_files = 0
        self.render_inflight_files = 0

        # 无头/非交互环境流式汇报器
        self.tracker: Optional[PlainProgressTracker] = (
            PlainProgressTracker(date, cam_index, self.total_files, cam_name=self.cam_name) if not self.enabled else None
        )

        # 视图刷新节流与脉冲定时器
        self._cached_view: Optional[Panel] = None
        self._last_view_build = 0.0
        self._view_min_interval = 0.12
        self._pulse_stop = threading.Event()
        self._pulse_thread: Optional[threading.Thread] = None

        self.observation = current_run.get()
        self._paint_lock = threading.RLock()
        self._memory_at = float("-inf")
        self._memory = None
        self._alert_counts = {}
        self.live: Optional[Live] = None
        if self.enabled:
            try:
                self._cached_view = self._generate_view()
                self.live = Live(self._cached_view, console=console, auto_refresh=False, transient=False)
            except Exception as exc:
                self._fallback(exc)

    def _fallback(self, exc):
        from src.core.utils import unregister_dashboard
        live, self.live = self.live, None
        self.enabled = False
        unregister_dashboard()
        self.tracker = PlainProgressTracker(self.date, self.cam_index, self.total_files, cam_name=self.cam_name)
        if live:
            try:
                live.stop()
            except Exception as stop_exc:
                logging.getLogger("homevlog").debug("Live cleanup failed: %s", stop_exc)
        logging.getLogger("homevlog").warning("TUI unavailable (%s); using plain progress", type(exc).__name__,
                                               extra={"event": "tui_fallback"})

    def start(self) -> None:
        self._pulse_stop.clear()
        if self.live:
            try:
                self.live.start()
            except Exception as exc:
                self._fallback(exc)
        self.update_view(force=True)
        self._pulse_thread = threading.Thread(target=self._pulse_loop, daemon=True)
        self._pulse_thread.start()

    def stop(self) -> None:
        self._pulse_stop.set()
        if self._pulse_thread:
            self._pulse_thread.join(timeout=1.0)
            self._pulse_thread = None
        self.update_view(force=True)
        if self.live:
            try:
                self.live.stop()
            except Exception as exc:
                self._fallback(exc)
                self.update_view(force=True)
            self.live = None

    def _pulse_loop(self) -> None:
        """周期性强制重建视图或发送无头心跳。"""
        while not self._pulse_stop.wait(0.5 if self.enabled else 10.0):
            self.update_view(force=self.enabled)

    def update_prescreen(self, completed: int, total: Optional[int] = None, latest_file: str = "", speed_str: str = "") -> None:
        """更新预筛阶段进度。"""
        with self._lock:
            self.prescreen_done = completed
            if total is not None:
                self.prescreen_total = total
                self.total_files = max(self.total_files, total)
            if latest_file:
                self.last_prescreen_file = latest_file
            if speed_str:
                self.last_prescreen_speed = speed_str
        self.update_view()

    def update_analysis(self, completed: int, total: Optional[int] = None, latest_file: str = "", speed_str: str = "") -> None:
        """更新精析阶段进度。"""
        with self._lock:
            self.analysis_done = completed
            if total is not None:
                self.analysis_total = total
            if latest_file:
                self.last_analysis_file = latest_file
            if speed_str:
                self.last_analysis_speed = speed_str
        self.update_view()

    def update_render(self, completed: int, total: Optional[int] = None, latest_batch: str = "", speed_str: str = "") -> None:
        """更新渲染阶段进度。"""
        if not self.render_enabled:
            return
        with self._lock:
            self.render_done = completed
            if total is not None:
                self.render_total = total
            if latest_batch:
                self.last_render_batch = latest_batch
            if speed_str:
                self.last_render_speed = speed_str
        self.update_view()

    def render_batch_dispatched(self, n_files: int) -> None:
        """批次投递渲染队列：在飞渲染文件数增加。"""
        with self._lock:
            self.render_inflight_files += max(0, n_files)
        self.update_view()

    def render_batch_finished(self, n_files: int) -> None:
        """批次渲染终结（成功/空批次/最终失败）：在飞转已成片计数。"""
        with self._lock:
            n = max(0, n_files)
            self.render_inflight_files = max(0, self.render_inflight_files - n)
            self.render_done_files += n
        self.update_view()

    def set_queue_status(self, prescreen_q: int, analysis_q: int, render_q: int) -> None:
        """更新当前积压队列水位。"""
        with self._lock:
            self.prescreen_queue_size = prescreen_q
            self.analysis_queue_size = analysis_q
            self.render_queue_size = render_q
        self.update_view()

    def set_scheduler_state(self, state: str, active_nv: int = 0, max_nv: int = 1) -> None:
        """更新工作窃取调度器的硬件协同状态。"""
        with self._lock:
            self.scheduler_state = state
            self.active_nv_decoders = active_nv
            self.max_nv_decoders = max_nv
        self.update_view()

    def add_alert(self, message: str, key=None) -> None:
        with self._lock:
            key = key or message
            self._alert_counts[key] = (self._alert_counts.get(key, (0, ""))[0] + 1, message)
            item = self._alert_counts.pop(key)
            self._alert_counts[key] = item
            if len(self._alert_counts) > 3:
                del self._alert_counts[next(iter(self._alert_counts))]
            self.recent_alerts = [f"{msg} ×{n}" if n > 1 else msg for n, msg in self._alert_counts.values()]
        self.update_view()

    def _status(self):
        run = self.observation
        counts, active = run.snapshot() if run else ({}, [])
        phase = run.phase if run else "流水处理"
        ready = sum(counts.get(key, 0) for key in ("encoded", "reused", "ready")) if run else self.render_done
        batch_text = (f"就绪 {ready} 批 · 编码 {counts.get('encoded', 0)} · 复用 {counts.get('reused', 0)}"
                      f" · 跳过 {counts.get('skipped', 0)} · 失败 {counts.get('failed', 0)}")
        detail = ""
        if phase in ("已取消", "失败"):
            detail = f"本轮确认保留 {ready} 批，可续跑"
        elif active and phase in ("流水处理", "渲染收尾中"):
            idx, item = active[0]
            now = time.monotonic()
            detail = f"批次 {idx} · {item['worker']} · 执行 {int(now - item['started'])} 秒"
            if item.get("note"):
                detail += f" · {item['note']}"
            if "progress_at" in item:
                detail += f" · 第 {item['frames']} 帧，{int(now - item['progress_at'])} 秒前推进"
        return phase, counts, batch_text, detail

    def update_view(self, force: bool = False) -> None:
        if not self._paint_lock.acquire(blocking=force):
            return
        try:
            if self.live:
                now = time.monotonic()
                if not force and now - self._last_view_build < self._view_min_interval:
                    return
                try:
                    self._cached_view = self._generate_view()
                    self.live.update(self._cached_view, refresh=True)
                    self._last_view_build = now
                    return
                except Exception as exc:
                    self._fallback(exc)
                    force = True
            if self.tracker:
                if not force and time.monotonic() - self.tracker._last_report < self.tracker.interval_s:
                    return
                phase, _, batches, detail = self._status()
                if force and self.observation:
                    detail = (detail + " · " if detail else "") + "run_id=" + self.observation.identity["run_id"]
                self.tracker.heartbeat(self.prescreen_done, self.prescreen_total, self.analysis_done,
                                       self.analysis_total, self.render_done, self.render_total, force=force,
                                       detail=" | ".join(filter(None, (phase, batches if self.render_enabled else "", detail))))

        finally:
            self._paint_lock.release()

    def _generate_view(self) -> Panel:
        with self._lock:
            phase, counts, batches, detail = self._status()
            wide = console.width >= 100
            elapsed = int(time.monotonic() - self._start_time)
            header = Text(f"日期: {self.date}   机位: {self.cam_name}   {phase}   {elapsed//60:02d}:{elapsed%60:02d}", style="bold cyan")
            rows = Table.grid(padding=(0, 2))
            rows.add_column(no_wrap=True)
            rows.add_column()
            rows.add_row("粗筛", f"{self.prescreen_done}/{self.prescreen_total} 个源文件")
            analysis = f"已处理 {self.analysis_done}/{self.analysis_total} 个已发现文件" if self.analysis_total else "暂无待析文件"
            if self.analysis_cached:
                analysis += f" · 沿用 {self.analysis_cached} 个已有结果"
            rows.add_row("精析", analysis)
            if self.render_enabled:
                rows.add_row("批次渲染", batches)
                if phase in ("流水处理", "渲染收尾中"):
                    rows.add_row("调度", f"排队 {counts.get('queued', 0) + counts.get('retry', 0)} · 执行 {counts.get('running', 0)}")
                gates = ("最终拼接中", "时钟校验与元数据生成中", "交付提交中", "已交付")
                if phase in gates:
                    for i, label in enumerate(("最终拼接", "时钟与元数据", "交付提交")):
                        rows.add_row(label, ("完成" if i < gates.index(phase) else "进行中" if i == gates.index(phase) else "待开始"))
            elements = [header, rows]
            if wide and phase in ("流水处理", "渲染收尾中"):
                now = time.monotonic()
                if now - self._memory_at >= 2:
                    from src.hardware.scheduler import get_nv_vram_info
                    self._memory = get_nv_vram_info()
                    self._memory_at = now
                memory = f"   NVIDIA 0 整卡显存 {self._memory['used_mb']}/{self._memory['total_mb']} MiB" if self._memory else ""
                elements.append(Text(f"粗筛队列 {self.prescreen_queue_size}   精析队列 {self.analysis_queue_size}"
                                     f"   NV 解码 {self.active_nv_decoders}/{self.max_nv_decoders}   {self.scheduler_state}{memory}", style="dim"))
            if detail:
                elements.append(Text(detail, style="bold"))
            if self.observation and phase in ("已交付", "失败", "已取消", "分析完成"):
                elements.append(Text("run_id=" + self.observation.identity["run_id"], style="dim"))
            for alert in self.recent_alerts:
                elements.append(Text(alert, style="yellow"))
            return Panel(Group(*elements), title="HomeVlog", border_style="blue")
