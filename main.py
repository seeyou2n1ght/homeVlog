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

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils import setup_logging, load_config, reset_semaphores, cleanup_resources, cleanup_temp_artifacts, get_input_dirs
from src.pipeline import run_pipeline, process_date_cam
from src.database import VlogDatabase
from src.scanner import scan_directory, get_date_cam_groups, resolve_camera_identity, resolve_output_filename
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

    # 1. 独立系统与硬件就绪体检 (--doctor)
    if args.doctor:
        print_doctor_report(console)
        return

    # 输入目录解析（支持多次 --input-dir 指定与 settings.yaml 配置混合）
    if args.input_dir:
        input_dirs = args.input_dir
    else:
        input_dirs = get_input_dirs(config)

    dashboard_enabled = not args.no_tui
    skip_render = args.no_render or (args.stage in ("prescreen", "analyze"))

    # 2. 独立或前置清理临时批次文件 (--clean-temp)
    if args.clean_temp:
        cleanup_temp_artifacts(clean_batches=True)
        console.print("[bold green]✔ 临时渲染批次与中间碎片已全部清理完毕。[/bold green]")
        # 若未指定任何其他操作或目标，清理完成后直接退出
        if not (args.date or args.date_range or args.days or args.scan or args.status or args.reanalyze):
            return

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
                    disp, _ = resolve_camera_identity(s_dir, cam_index=cam, config=config)
                    cam_map[cam] = disp
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

    if args.reanalyze:
        db = VlogDatabase()
        try:
            if target_dates:
                for d in target_dates:
                    if args.cam is not None:
                        db.conn.execute(
                            "UPDATE file_tasks SET prescreen_status='PENDING', analysis_status='PENDING', analysis_segments=NULL WHERE date=? AND cam_index=?",
                            (d, args.cam),
                        )
                        db.conn.execute("DELETE FROM segments WHERE date=? AND cam_index=?", (d, args.cam))
                    else:
                        db.conn.execute(
                            "UPDATE file_tasks SET prescreen_status='PENDING', analysis_status='PENDING', analysis_segments=NULL WHERE date=?",
                            (d,),
                        )
                        db.conn.execute("DELETE FROM segments WHERE date=?", (d,))
                dates_str = ", ".join(target_dates)
                cam_str = f"机位 {args.cam}" if args.cam is not None else "全机位"
                console.print(f"[bold cyan]ℹ 已重置日期 [{dates_str}] ({cam_str}) 的分析缓存，将重新精析并渲染。[/bold cyan]")
            else:
                db.conn.execute(
                    "UPDATE file_tasks SET prescreen_status='PENDING', analysis_status='PENDING', analysis_segments=NULL"
                )
                db.conn.execute("DELETE FROM segments")
                console.print("[bold cyan]ℹ 已重置数据库中所有任务的分析缓存，将全局重新精析并渲染。[/bold cyan]")
            db.conn.commit()
        finally:
            db.close()

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
                            disp, _ = resolve_camera_identity(s_dir, cam_index=cam, config=config)
                            cam_map[cam] = disp
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
                all_groups = get_date_cam_groups(db)
                if target_dates:
                    active_groups = [(d, c) for d, c in all_groups if d in target_dates and (args.cam is None or c == args.cam)]
                elif args.cam is not None:
                    active_groups = [(d, c) for d, c in all_groups if c == args.cam]
                else:
                    active_groups = all_groups

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
        if target_dates:
            db = VlogDatabase()
            try:
                scan_directory(db, input_dir=input_dirs)
                all_groups = get_date_cam_groups(db)
                active_groups = [
                    (d, c) for d, c in all_groups
                    if d in target_dates and (args.cam is None or c == args.cam)
                ]
                if not active_groups:
                    console.print(f"[yellow]未在任务库中找到日期 {target_dates} 对应的监控切片。[/yellow]")
                    return

                from src.monitor import get_monitor
                from src.config import OUTPUT_DIR
                monitor = get_monitor()
                monitor.start()

                dates_sorted = sorted(set(d for d, _ in active_groups))
                cams_sorted = sorted(set(f"Cam {c}" for _, c in active_groups))
                if len(active_groups) > 1:
                    print_batch_startup_banner(
                        total_groups=len(active_groups),
                        date_range=(dates_sorted[0], dates_sorted[-1]),
                        cameras=cams_sorted,
                        output_dir=str(OUTPUT_DIR),
                    )

                batch_summary_list = []
                ok_count, fail_count = 0, 0

                for d, cam in active_groups:
                    t0 = time.monotonic()
                    ok = process_date_cam(db, d, cam, skip_render=skip_render, dashboard_enabled=dashboard_enabled)
                    wall_s = time.monotonic() - t0
                    if ok:
                        ok_count += 1
                    else:
                        fail_count += 1

                    # 收集指标用于多任务总结
                    tasks = db.get_all_file_tasks_for_date(d, cam)
                    s_dir = str(Path(tasks[0]["filepath"]).parent) if tasks else (input_dirs[0] if input_dirs else "")
                    cam_disp, _ = resolve_camera_identity(s_dir, cam_index=cam, config=config)
                    out_name = resolve_output_filename(
                        config.get("output", {}).get("naming", "DailyVlog_{date}_{mac}.mp4"),
                        d, cam, tasks[0]["filepath"] if tasks else None, config=config
                    )
                    out_p = Path(OUTPUT_DIR) / out_name
                    meta_p = out_p.with_suffix(".meta.json")

                    item = {
                        "date": d,
                        "cam_index": cam,
                        "cam_name": cam_disp,
                        "total_files": len(tasks),
                        "input_duration_s": sum(t.get("file_duration", 0.0) or 0.0 for t in tasks),
                        "vlog_duration_s": 0.0,
                        "output_size_mb": round(out_p.stat().st_size / (1024 * 1024), 2) if out_p.exists() else 0.0,
                        "wall_clock_s": round(wall_s, 2),
                        "status": "SUCCESS" if ok else "FAILED",
                    }
                    if meta_p.exists():
                        try:
                            import json
                            m_d = json.loads(meta_p.read_text(encoding="utf-8"))
                            m_m = m_d.get("metrics", {})
                            item["vlog_duration_s"] = m_m.get("vlog_duration_s", 0.0)
                            item["condensation_ratio"] = m_m.get("condensation_ratio", 0.0)
                            item["speedup_x"] = m_d.get("performance", {}).get("speedup_x", 0.0)
                        except Exception:
                            pass
                    batch_summary_list.append(item)
                    cleanup_resources()

                monitor.shutdown()
                logging.shutdown()

                if len(active_groups) > 1:
                    print_batch_summary_table(batch_summary_list)
                else:
                    res_item = batch_summary_list[0]
                    status_color = "green" if res_item["status"] == "SUCCESS" else "red"
                    console.print(f"[bold {status_color}]✔ 处理完成: {res_item['date']} ({res_item['cam_name']})[/bold {status_color}]")
            finally:
                db.close()
            return

        # 8. 全量常规批处理 (run_pipeline)
        result = run_pipeline(skip_render=skip_render, input_dir=input_dirs, dashboard_enabled=dashboard_enabled)
        console.print(f"[bold cyan]流水线总览:[/] 处理组合总数={result['total']}, 成功={result['ok']}, 失败={result['failed']}")

    except KeyboardInterrupt:
        from src.renderer import FFmpegProcessRegistry
        FFmpegProcessRegistry.mark_interrupted()
        FFmpegProcessRegistry.kill_all()
        console.print("\n[bold yellow]⚠ 用户中断 (Ctrl+C)。已平稳终止子进程与硬件会话，当前已分析任务与批次已安全落盘。下次启动将自动断点续传。[/bold yellow]")
        cleanup_resources()
        sys.exit(130)


if __name__ == "__main__":
    main()
