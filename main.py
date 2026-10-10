"""HomeVlog — 家庭监控素材智能浓缩与异构渲染系统.

Usage:
  uv run python main.py                       # 完整流水线 (自动扫描全量未处理素材并运行)
  uv run python main.py --doctor              # 系统与硬件就绪体检 (FFmpeg/CUDA/显存/NAS/模型)
  uv run python main.py --status              # 查看数据库内所有日期与机位的处理进度大表
  uv run python main.py --scan                # 仅执行素材目录扫描入库
  uv run python main.py --date 2026-03-20     # 处理指定日期 (支持 20260320 或 2026-03-20)
  uv run python main.py --date-range 20260320..20260324 # 批量处理日期范围
  uv run python main.py --days 3              # 处理最近 3 天素材
  uv run python main.py --no-render           # 仅执行预筛选与 AI 精析，跳过渲染
  uv run python main.py --dry-run             # 规划预览模式，不启动实际转码
  uv run python main.py --no-tui              # 禁用交互式仪表盘，使用流式心跳日志
"""

import logging
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

if __name__ == "__main__":
    print("HomeVlog 正在加载配置和运行环境…", flush=True)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import setup_logging, load_config, reset_semaphores, cleanup_resources, cleanup_temp_artifacts, get_input_dirs, check_disk_space
from src.pipeline import run_pipeline, process_date_cam
from src.stages.companion import build_batch_summary_item
from src.database import VlogDatabase
from src.scanner import scan_directory, get_date_cam_groups, resolve_camera_ui_name
from src.ui import (
    build_arg_parser,
    resolve_cli_dates,
    print_doctor_report,
    print_status_table,
    print_scan_results,
    print_batch_startup_banner,
    print_batch_summary_table,
    console,
)

logger = setup_logging()


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.debug:
        setup_logging(level_override=logging.DEBUG)

    if args.config:
        load_config(args.config, reload=True)
        reset_semaphores()

    config = load_config()
    if getattr(args, "skip_corrupt", False):
        config.setdefault("recovery", {})["skip_corrupt"] = True
    log_path = next((h.baseFilename for h in logger.handlers
                     if getattr(h, "homevlog_role", None) == "run"), "logs/runs")
    console.print(f"HomeVlog 已初始化，运行日志: {log_path}", markup=False)

    # 1. 独立系统与硬件就绪体检 (--doctor)
    if args.doctor:
        if not print_doctor_report(console):
            sys.exit(1)
        return

    # 输入目录解析（支持多次 --input-dir 指定与 settings.yaml 配置混合）
    if args.input_dir:
        input_dirs = args.input_dir
    else:
        input_dirs = get_input_dirs(config)

    dashboard_enabled = not args.no_tui
    skip_render = args.no_render or args.stage == "analyze"

    # 3. 数据库状态速查总览表 (--status)
    if args.status:
        db = VlogDatabase()
        try:
            cam_map = {}
            groups = get_date_cam_groups(db)
            for d, cam in groups:
                if cam not in cam_map:
                    tasks = db.get_all_file_tasks_for_date(d, cam)
                    s_dir = str(Path(tasks[0]["filepath"]).parent) if tasks else (input_dirs[0] if input_dirs else "")
                    cam_map[cam] = resolve_camera_ui_name(s_dir, cam_index=cam, config=config)
            print_status_table(db, camera_display_names=cam_map)
        finally:
            db.close()
        return

    # 4. 重析缓存重置 (--reanalyze)
    target_dates = []
    try:
        target_dates = resolve_cli_dates(args)
    except ValueError as e:
        console.print(f"[bold red]参数错误:[/] {e}")
        sys.exit(1)

    if args.reanalyze and not args.dry_run:
        db = VlogDatabase()
        try:
            groups = get_date_cam_groups(db, dates=target_dates, cam_index=args.cam)
            with db._lock:
                if not args.force_render and any(
                    db.conn.execute("SELECT 1 FROM render_tasks WHERE date=? AND cam_index=? AND status='COMPLETED'", (d, c)).fetchone()
                    for d, c in groups
                ):
                    parser.error("所选范围包含已交付日期；重析请同时使用 --force-render（尚未重置任何数据）")
                try:
                    for d, cam in groups:
                        db.conn.execute(
                            """UPDATE file_tasks SET prescreen_status='PENDING', prescreen_result=NULL,
                               analysis_status='PENDING', analysis_segments=NULL, processing_fingerprint=NULL,
                               retry_count=0, error_msg=NULL WHERE date=? AND cam_index=?""", (d, cam),
                        )
                        db.conn.execute("DELETE FROM segments WHERE date=? AND cam_index=?", (d, cam))
                    db.conn.commit()
                except BaseException:
                    db.conn.rollback()
                    raise
            console.print(f"[bold cyan]ℹ 已重置所选 {len(groups)} 个日期机位组的分析缓存；人工审核保留。[/bold cyan]")
        finally:
            db.close()

    # 2. 独立或前置清理临时批次文件 (--clean-temp)
    if args.clean_temp and not args.dry_run:
        cleanup_temp_artifacts(clean_batches=True)
        console.print("[bold green]✔ 临时渲染批次与中间碎片已全部清理完毕。[/bold green]")
        # 若未指定任何其他操作或目标，清理完成后直接退出
        if not (args.date or args.date_range or args.days or args.scan or args.status or args.reanalyze):
            return

    try:
        # 5. 纯扫描模式 (--scan)
        if args.scan:
            db = VlogDatabase()
            try:
                scan_res = scan_directory(db, input_dir=input_dirs)
                groups = get_date_cam_groups(db)
                if groups:
                    cam_map = {}
                    group_stats = {}
                    for d, cam in groups:
                        tasks = db.get_all_file_tasks_for_date(d, cam)
                        s_dir = str(Path(tasks[0]["filepath"]).parent) if tasks else (input_dirs[0] if input_dirs else "")
                        if cam not in cam_map:
                            cam_map[cam] = resolve_camera_ui_name(s_dir, cam_index=cam, config=config)
                        dur = sum(t.get("file_duration", 0.0) or 0.0 for t in tasks)
                        done_c = sum(1 for t in tasks if t.get("prescreen_status") != "PENDING")
                        st_tag = "[green]已就绪[/green]" if done_c == len(tasks) else f"[yellow]{done_c}/{len(tasks)}[/yellow]"
                        group_stats[(d, cam)] = {
                            "total_files": len(tasks),
                            "duration_s": dur,
                            "status_tag": st_tag,
                        }
                    console.print(f"[bold green]扫描入库完成:[/] 新增切片 {scan_res.added} 个, 跳过 {scan_res.skipped} 个 (共 {len(groups)} 个归档组)")
                    print_scan_results(groups, camera_display_names=cam_map, group_stats=group_stats)
                else:
                    console.print("[yellow]未在指定输入目录中发现符合命名格式的监控切片文件。[/yellow]")
            finally:
                db.close()
            return

        # 6. 规划预览模式 (--dry-run)
        if args.dry_run:
            from rich.panel import Panel
            from rich.table import Table
            db = VlogDatabase()
            try:
                scan_directory(db, input_dir=input_dirs)
                active_groups = get_date_cam_groups(db, dates=target_dates, cam_index=args.cam)

                table = Table.grid(padding=(0, 2))
                table.add_column(style="bold cyan", justify="right")
                table.add_column(style="white")
                table.add_row("🎯 规划执行组:", f"共 [bold yellow]{len(active_groups)}[/] 个日期机位组")
                table.add_row("⚙️ 执行阶段:", f"[bold magenta]{args.stage.upper()}[/] (跳过渲染: {skip_render})")
                table.add_row("🖥️ 终端显示:", "实时仪表盘 (Live TUI)" if dashboard_enabled else "无头流式心跳 (Plain Text)")

                total_files_sum = 0
                total_dur_sum = 0.0
                for d, c in active_groups:
                    t_list = db.get_all_file_tasks_for_date(d, c)
                    total_files_sum += len(t_list)
                    total_dur_sum += sum(t.get("file_duration", 0.0) or 0.0 for t in t_list)

                table.add_row("📊 预计素材总量:", f"[bold green]{total_files_sum}[/] 个切片 (总计 [bold yellow]{total_dur_sum/3600:.2f}[/] 小时)")
                panel = Panel(table, title="[bold blue]🔍 HomeVlog 规划预览 (Dry Run)[/bold blue]", border_style="bright_blue", padding=(1, 2))
                console.print(panel)
            finally:
                db.close()
            return

        # 7. 指定日期或日期范围批处理 (--date, --date-range, --days)
        if target_dates or args.cam is not None:
            db = VlogDatabase()
            monitor = None
            try:
                scan_directory(db, input_dir=input_dirs)
                active_groups = get_date_cam_groups(db, dates=target_dates, cam_index=args.cam)
                if not active_groups:
                    console.print("[yellow]未在任务库中找到所选日期/机位对应的监控切片。[/yellow]")
                    return

                from src.monitor import get_monitor
                from src.config import OUTPUT_DIR
                monitor = get_monitor()
                monitor.start()

                dates_sorted = sorted(set(d for d, _ in active_groups))
                camera_names = set()
                for d, cam in active_groups:
                    tasks = db.get_all_file_tasks_for_date(d, cam)
                    sample_dir = str(Path(tasks[0]["filepath"]).parent) if tasks else ""
                    camera_names.add(resolve_camera_ui_name(sample_dir, cam_index=cam, config=config))
                cams_sorted = sorted(camera_names)
                if len(active_groups) > 1:
                    print_batch_startup_banner(
                        total_groups=len(active_groups),
                        date_range=(dates_sorted[0], dates_sorted[-1]),
                        cameras=cams_sorted,
                        output_dir=str(OUTPUT_DIR),
                    )

                batch_summary_list = []
                ok_count, fail_count, skip_count = 0, 0, 0
                stopped_reason = None

                for d, cam in active_groups:
                    already_completed = not args.force_render and db.is_render_completed(d, cam)
                    if not already_completed and not check_disk_space(OUTPUT_DIR):
                        stopped_reason = "LOW_DISK_SPACE"
                        break
                    t0 = time.monotonic()
                    ok = process_date_cam(
                        db, d, cam,
                        skip_render=skip_render,
                        dashboard_enabled=dashboard_enabled,
                        force_render=getattr(args, "force_render", False),
                        config=config,
                    )
                    wall_s = time.monotonic() - t0
                    if ok:
                        if already_completed:
                            skip_count += 1
                        else:
                            ok_count += 1
                    else:
                        fail_count += 1

                    status = "SKIPPED" if ok and already_completed else ("SUCCESS" if ok else "FAILED")
                    batch_summary_list.append(build_batch_summary_item(
                        db, d, cam, OUTPUT_DIR, config, wall_s, status, skip_render=skip_render,
                    ))
                    cleanup_resources()

                logging.shutdown()

                if len(active_groups) > 1:
                    print_batch_summary_table(batch_summary_list)
                elif batch_summary_list:
                    res_item = batch_summary_list[0]
                    status_color = "red" if res_item["status"] == "FAILED" else "green"
                    status_label = {"SUCCESS": "✔ 处理完成", "SKIPPED": "已交付跳过"}.get(res_item["status"], "✘ 处理失败")
                    console.print(f"[bold {status_color}]{status_label}: {res_item['date']} ({res_item['cam_name']})[/bold {status_color}]")
                if stopped_reason:
                    remaining = len(active_groups) - ok_count - fail_count - skip_count
                    console.print(f"[yellow]流水线提前停止: {stopped_reason}，未处理={remaining}[/yellow]")
                if fail_count or stopped_reason:
                    sys.exit(1)
            finally:
                if monitor is not None:
                    monitor.shutdown()
                db.close()
            return

        # 8. 全量常规批处理 (run_pipeline)
        result = run_pipeline(skip_render=skip_render, input_dir=input_dirs, dashboard_enabled=dashboard_enabled,
                              force_render=args.force_render, config=config)
        console.print(f"[bold cyan]流水线总览:[/] 处理组合总数={result['total']}, 成功={result['ok']}, 失败={result['failed']}, 已交付跳过={result.get('skipped', 0)}, 未处理={result.get('remaining', 0)}")
        if result.get("stopped_reason"):
            console.print(f"[yellow]流水线提前停止: {result['stopped_reason']}[/yellow]")
        if result["failed"] or result.get("remaining") or result.get("stopped_reason"):
            sys.exit(1)

    except KeyboardInterrupt:
        from src.renderer import FFmpegProcessRegistry
        FFmpegProcessRegistry.mark_interrupted()
        FFmpegProcessRegistry.kill_all()
        console.print("\n[bold yellow]⚠ 任务已中断。已终止受管子进程与硬件会话，保留已保存的分析结果和有效批次。下次启动将自动断点续传。[/bold yellow]")
        cleanup_resources()
        sys.exit(130)


if __name__ == "__main__":
    main()
