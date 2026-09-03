"""HomeVlog 终端富文本呈现与实时仪表盘模块 (Rich Terminal UI & Dashboard).

包含：
- PipelineDashboard: 负责多阶段并发流水线的实时 Live 仪表盘，整合进度条、队列水位、调度状态与活动追踪；
- 启动面板 (Startup Banner)、完成总结看板 (Summary Card) 与错误汇总卡片 (Error Summary)；
- 扫描结果表格 (Scan Results Table)；
- 自动检测 TTY 并支持 Headless/CI 优雅降级。
"""

import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional

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



def is_interactive_terminal() -> bool:
    """判断当前环境是否为支持交互式光标刷新的终端。"""
    return console.is_terminal and not console.is_dumb_terminal


def print_batch_startup_banner(
    total_groups: int,
    date_range: tuple[str, str],
    cameras: list[str],
    output_dir: str = "output",
    device_info: str = "Intel UHD 770 (QSV 粗筛/解码) + RTX 3060Ti (NVENC 渲染/YOLO)",
) -> None:
    """在多天批量浓缩任务启动时，打印宏观全局规划 Banner。"""
    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="white")

    start_date, end_date = date_range
    date_str = f"{start_date} ~ {end_date}" if start_date != end_date else start_date
    table.add_row("📅 任务规划:", f"共发现 [bold yellow]{total_groups}[/] 个归档任务 (日期跨度: [bold white]{date_str}[/])")

    cam_str = ", ".join(cameras) if cameras else "全机位"
    table.add_row("📷 监控机位:", f"[bold green]{cam_str}[/]")
    table.add_row("⚡ 硬件协同:", f"[bold magenta]{device_info}[/]")
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
    device_info: str = "Intel UHD 770 (QSV 粗筛/解码) + RTX 3060Ti (NVENC 渲染/YOLO)",
    output_path: Optional[str] = None,
    cam_name: Optional[str] = None,
    batch_progress: Optional[str] = None,
) -> None:
    """以高质感 Rich 面板展示流水线启动配置与硬件就绪状态。"""
    hours = total_duration_s / 3600.0

    table = Table.grid(padding=(0, 2))
    table.add_column(style="bold cyan", justify="right")
    table.add_column(style="white")

    cam_label = f"{cam_name} (Cam {cam_index})" if cam_name and cam_name != f"cam{cam_index}" else f"Cam {cam_index}"
    progress_tag = f" [bold cyan]{batch_progress}[/]" if batch_progress else ""
    table.add_row("📅 处理日期:", f"[bold white]{date}[/] (机位: [bold yellow]{cam_label}[/]){progress_tag}")


    table.add_row(
        "🎬 输入素材:",
        f"[bold green]{total_files}[/] 个监控切片 | 原始总时长: [bold yellow]{hours:.2f}[/] 小时 ([dim]{total_duration_s:.1f} 秒[/dim])",
    )
    table.add_row("⚡ 硬件协同:", f"[bold magenta]{device_info}[/]")
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
    stage_durations: Optional[dict[str, float]] = None,
    cam_name: Optional[str] = None,
) -> None:
    """输出美观的流水线完成报告卡片与指标分解表。"""
    file_sz_mb = output_path.stat().st_size / (1024 * 1024) if output_path.exists() else 0.0
    speedup = (total_input_dur / elapsed_wall) if elapsed_wall > 0 else 0.0
    el_m, el_s = divmod(int(elapsed_wall), 60)

    # 核心指标统计表
    metric_table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    metric_table.add_column("指标类别", style="bold", width=18)
    metric_table.add_column("数值与明细", style="white")

    metric_table.add_row(
        "原始素材规模",
        f"[bold]{total_files}[/] 个文件 (总计 [yellow]{total_input_dur / 3600:.2f}[/] 小时 / [dim]{total_input_dur:.1f}s[/dim])",
    )
    metric_table.add_row(
        "生成浓缩成片",
        f"[bold green]{output_path.name}[/] ([cyan]{file_sz_mb:.2f} MB[/cyan])",
    )
    metric_table.add_row(
        "全流程总耗时",
        f"[bold yellow]{el_m:02d}分{el_s:02d}秒[/bold yellow] (物理耗时 [dim]{elapsed_wall:.2f}s[/dim])",
    )
    metric_table.add_row(
        "等效加速倍率",
        f"[bold red]{speedup:.2f}x 实时加速[/bold red]",
    )
    metric_table.add_row(
        "文件输出路径",
        f"{str(output_path)}",
    )

    # 若有分阶段用时，追加阶段分解表
    renderables: list[Any] = [metric_table]
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

    cam_label = f"{cam_name} (Cam {cam_index})" if cam_name and cam_name != f"cam{cam_index}" else f"Cam {cam_index}"
    panel = Panel(
        Group(*renderables),
        title=f"[bold green]✨ HomeVlog 浓缩流水线处理圆满完成 | {date} {cam_label}[/bold green]",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)



def print_error_summary(errors: list[str]) -> None:
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


def print_scan_results(groups: list[tuple[str, int]], camera_display_names: Optional[dict[int, str]] = None) -> None:
    """以整洁的表格展示扫描出来的日期与机位组合。"""
    table = Table(title="[bold cyan]🔍 监控素材归档组扫描结果[/bold cyan]", border_style="dim")
    table.add_column("序号", justify="center", style="dim", width=6)
    table.add_column("归档日期 (Date)", justify="center", style="bold yellow")
    table.add_column("监控机位 (Camera)", justify="center", style="bold green")

    for idx, (date, cam) in enumerate(groups, 1):
        cam_text = camera_display_names.get(cam, f"Cam {cam}") if camera_display_names else f"Cam {cam}"
        table.add_row(str(idx), str(date), cam_text)

    console.print(table)



class PipelineDashboard:
    """
    终极多任务终端动态仪表盘 (Unified Multi-Stage Terminal Dashboard).

    集成：
    - 三阶段进度管理 (Prescreen, Analysis, Render)
    - 队列积压水位实时监视
    - 异构调度器硬件状态徽标 (QSV 独占 / NVDEC 协同 / 渲染抢占)
    - 活跃事件与告警滚动追踪
    - 非 TTY / 无头环境的优雅降级
    """

    def __init__(
        self,
        date: str,
        cam_index: int,
        total_prescreen: int = 0,
        render_enabled: bool = True,
        enabled: bool = True,
    ):

        self.date = date
        self.cam_index = cam_index
        self.render_enabled = render_enabled
        self.enabled = enabled and is_interactive_terminal()

        self._lock = threading.Lock()
        self._start_time = time.monotonic()

        # 状态指标与工序追踪
        self.total_files = max(1, total_prescreen)
        self.prescreen_done = 0
        self.prescreen_total = total_prescreen
        self.analysis_done = 0
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
        self.recent_alerts: list[str] = []

        # 渲染批次文件级真实计数 (驱动堆叠分布条，杜绝比率推算失真)
        self.render_done_files = 0
        self.render_inflight_files = 0

        # 视图刷新节流与脉冲定时器
        self._cached_view: Optional[Panel] = None
        self._last_view_build = 0.0
        self._view_min_interval = 0.12
        self._pulse_stop = threading.Event()
        self._pulse_thread: Optional[threading.Thread] = None

        self.live: Optional[Live] = None
        if self.enabled:
            self._cached_view = self._generate_view()
            self.live = Live(
                self._cached_view,
                console=console,
                refresh_per_second=5,
                transient=False,
                auto_refresh=True,
            )

    def start(self) -> None:
        """启动实时仪表盘与 0.5s 脉冲刷新线程。"""
        if self.live:
            self.live.start()
            self._pulse_stop.clear()
            self._pulse_thread = threading.Thread(target=self._pulse_loop, daemon=True)
            self._pulse_thread.start()

    def stop(self) -> None:
        """安全停止脉冲线程并关闭仪表盘。"""
        self._pulse_stop.set()
        if self._pulse_thread:
            self._pulse_thread.join(timeout=1.0)
            self._pulse_thread = None
        if self.live:
            self.update_view(force=True)
            self.live.stop()
            self.live = None

    def _pulse_loop(self) -> None:
        """周期性强制重建视图：保证耗时钟走字与节流期内的脏状态最终落屏。"""
        while not self._pulse_stop.wait(0.5):
            self.update_view(force=True)

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

    def add_alert(self, message: str) -> None:
        """向仪表盘添加一条告警或异常追踪记录。"""
        with self._lock:
            now_str = time.strftime("%H:%M:%S")
            self.recent_alerts.append(f"[{now_str}] {message}")
            if len(self.recent_alerts) > 3:
                self.recent_alerts.pop(0)
        self.update_view()

    def update_view(self, force: bool = False) -> None:
        """触发界面刷新（0.12s 节流，force 强制重建）。"""
        if not self.live:
            return
        now = time.monotonic()
        if not force and (now - self._last_view_build) < self._view_min_interval:
            return
        try:
            self._cached_view = self._generate_view()
            self._last_view_build = now
            self.live.update(self._cached_view, refresh=False)
        except Exception:
            pass

    def _generate_view(self) -> Panel:
        """组合并渲染符合方案 A (全生命周期堆叠状态条) 的现代仪表盘。"""
        with self._lock:
            # 1. 顶部硬件调度与队列监控表 (2:2:3 配比防折行)
            hw_table = Table.grid(padding=(0, 2), expand=True)
            hw_table.add_column("Col1", ratio=2)
            hw_table.add_column("Col2", ratio=2)
            hw_table.add_column("Col3", ratio=3)

            # 格式化调度器状态徽标 (短文本防折行)
            if self.scheduler_state == "COOPERATIVE_BURST":
                sched_badge = f"[bold yellow on black] ⚡ BURST (NVDEC {self.active_nv_decoders}/{self.max_nv_decoders}+QSV) [/]"
            elif self.scheduler_state == "RENDER_PREEMPTION_YIELD":
                sched_badge = "[bold magenta on black] 🎬 RENDER PREEMPT [/]"
            else:
                sched_badge = "[bold green on black] 🍃 NORMAL (QSV) [/]"

            elapsed_s = int(time.monotonic() - self._start_time)
            mins, secs = divmod(elapsed_s, 60)
            time_str = f"{mins:02d}:{secs:02d}"

            q_info = (
                f"[dim]待筛:[/] [cyan]{self.prescreen_queue_size}[/] | "
                f"[dim]待析:[/] [yellow]{self.analysis_queue_size}[/] | "
                f"[dim]待编:[/] [magenta]{self.render_queue_size}[/]"
            )

            hw_table.add_row(
                f"[bold cyan]日期:[/] {self.date} (Cam {self.cam_index})",
                f"[bold cyan]调度策略:[/] {sched_badge}",
                f"[bold cyan]队列水位:[/] {q_info} [dim](耗时 {time_str})[/dim]",
            )

            # 2. 方案 A: 各工序切片分布 (文件级真实计数，守恒归一化)
            #    桶定义严格互斥且总和 = total，杜绝比率推算造成的开局虚报
            total = max(1, self.total_files)

            # (1) 已成片: 已完成渲染批次覆盖的真实文件数
            done = min(total, max(0, self.render_done_files))

            # (2) 压制中: 已投递未完成的渲染批次覆盖的真实文件数
            rendering = min(total - done, max(0, self.render_inflight_files))

            # (3) YOLO 精析中: 待析队列 + 分析在飞
            yolo_pending = max(0, self.analysis_total - self.analysis_done)
            yolo = min(total - done - rendering, yolo_pending)

            # (4) 静态快进就绪: 已过筛/已分析但尚未进入渲染批次的文件
            static_candidates = max(0, self.prescreen_done - yolo_pending - done - rendering)
            static = max(0, min(total - done - rendering - yolo, static_candidates))

            # (5) QSV 粗筛在飞: 总待筛 - 已完成 - 队列中等待
            prescreen_inflight = max(0, self.prescreen_total - self.prescreen_done - self.prescreen_queue_size)
            prescreen = min(total - done - rendering - yolo - static, prescreen_inflight)

            # (6) 待输入: 守恒残差
            pending = max(0, total - (done + rendering + yolo + static + prescreen))

            # 3. 构造 48 字符等宽堆叠条 (针对 Windows Terminal 完美等宽对齐)
            bar_len = 48
            w_done = int(round(done / total * bar_len))
            w_rend = int(round(rendering / total * bar_len))
            w_yolo = int(round(yolo / total * bar_len))
            w_stat = int(round(static / total * bar_len))
            w_pres = int(round(prescreen / total * bar_len))
            w_pend = max(0, bar_len - (w_done + w_rend + w_yolo + w_stat + w_pres))

            t_bar = Text()
            t_bar.append("  [", style="bold white")
            if w_done > 0:
                t_bar.append("█" * w_done, style="bold green")
            if w_rend > 0:
                t_bar.append("█" * w_rend, style="bold magenta")
            if w_yolo > 0:
                t_bar.append("█" * w_yolo, style="bold yellow")
            if w_stat > 0:
                t_bar.append("▒" * w_stat, style="cyan")
            if w_pres > 0:
                t_bar.append("░" * w_pres, style="bold blue")
            if w_pend > 0:
                t_bar.append("┈" * w_pend, style="dim white")
            pct = (done / total) * 100
            t_bar.append(f"]  {done}/{total} ({pct:.1f}% 闭环)", style="bold white")

            legend_text = Text.from_markup(
                f"  [bold green]█ 已成片 {done}[/] │ "
                f"[bold magenta]█ 压制中 {rendering}[/] │ "
                f"[bold yellow]█ YOLO精析 {yolo}[/] │ "
                f"[cyan]▒ 静态快进 {static}[/] │ "
                f"[bold blue]░ QSV粗筛 {prescreen}[/] │ "
                f"[dim white]┈ 待输入 {pending}[/]"
            )

            # 4. 微观阶段进展表 (极简紧凑 3 行)
            p_pct = int(self.prescreen_done / max(1, self.prescreen_total) * 100)
            a_pct = int(self.analysis_done / max(1, self.analysis_total) * 100) if self.analysis_total > 0 else 0
            r_pct = int(self.render_done / max(1, self.render_total) * 100) if self.render_total > 0 else 0

            micro_table = Table.grid(padding=(0, 2), expand=True)
            micro_table.add_column("Stage", style="bold cyan", width=14)
            micro_table.add_column("Progress", style="white", width=20)
            micro_table.add_column("Detail", style="dim")

            p_detail = self.last_prescreen_speed
            if self.last_prescreen_file:
                p_detail += f" ({Path(self.last_prescreen_file).name[:20]})"
            micro_table.add_row(
                "• Pass 1 粗筛",
                f"[{p_pct:>3}%] {self.prescreen_done}/{self.prescreen_total}",
                p_detail or "[dim]准备就绪[/dim]"
            )

            a_detail = self.last_analysis_speed
            if self.last_analysis_file:
                a_detail += f" ({Path(self.last_analysis_file).name[:20]})"
            micro_table.add_row(
                "• Pass 1.5精析",
                f"[{a_pct:>3}%] {self.analysis_done}/{self.analysis_total}" if self.analysis_total > 0 else "[  0%] 等待疑点",
                a_detail or ("[dim]等待疑点入队[/dim]" if self.analysis_total == 0 else "[dim]推断中[/dim]")
            )

            if self.render_enabled:
                r_detail = self.last_render_speed
                if self.last_render_batch:
                    r_detail += f" ({self.last_render_batch})"
                micro_table.add_row(
                    "• Pass 2 渲染",
                    f"[{r_pct:>3}%] {self.render_done}/{self.render_total} 批" if self.render_total > 0 else "[  0%] 准备批次",
                    r_detail or "[dim]等待批次合成[/dim]"
                )

            # 组装整个面板
            elements: list[Any] = [
                hw_table,
                Text(""),
                Text("🎬 全天素材工序全景流动分布 [横轴总长 = 全天素材 100%]:", style="bold cyan"),
                t_bar,
                legend_text,
                Text(""),
                micro_table,
            ]

            # 5. 底部告警追踪 (如有)
            if self.recent_alerts:
                elements.append(Text(""))
                alert_text = Text("⚠️ 实时告警追踪:\n", style="bold red")
                for item in self.recent_alerts[-3:]:
                    alert_text.append(f"  • {item}\n", style="yellow")
                elements.append(alert_text)

            return Panel(
                Group(*elements),
                title="[bold blue]🎛️ HomeVlog 流式管线实时监控[/bold blue]",
                border_style="bright_blue",
                padding=(0, 1),
            )

