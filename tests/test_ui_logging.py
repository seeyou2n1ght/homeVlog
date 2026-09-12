import json
import logging
import time
from pathlib import Path

import pytest

from src.ui import (
    PipelineDashboard,
    print_startup_banner,
    print_summary_card,
    print_error_summary,
    print_scan_results,
)
from src.utils import (
    setup_logging,
    get_logger,
    register_dashboard,
    unregister_dashboard,
    LOGS_DIR,
)


def test_subsystem_adapter_and_contextual_formatter(tmp_path):
    """验证子系统上下文 Logger 能够正确注入 subsystem 标签并被格式化器解析。"""
    logger = get_logger("my_subsystem")
    assert logger.extra["subsystem"] == "my_subsystem"
    logger.info("Test contextual message from subsystem")


def test_error_log_and_jsonl_separation(tmp_path):
    """验证主日志、独立错误日志及 JSONL 事件流的三路分离机制。"""
    setup_logging()

    # 寻找最新的日志文件集合（支持子目录分区）
    main_logs = sorted(LOGS_DIR.rglob("homevlog_*.log"), key=lambda p: p.stat().st_mtime)
    err_logs = sorted(LOGS_DIR.rglob("error_*.log"), key=lambda p: p.stat().st_mtime)
    jsonl_logs = sorted(LOGS_DIR.rglob("events_*.jsonl"), key=lambda p: p.stat().st_mtime)

    assert len(main_logs) > 0
    assert len(err_logs) > 0
    assert len(jsonl_logs) > 0

    latest_main = main_logs[-1]
    latest_err = err_logs[-1]
    latest_jsonl = jsonl_logs[-1]

    sub_logger = get_logger("diagnostics")
    test_token_info = f"TEST_INFO_{time.time_ns()}"
    test_token_warn = f"TEST_WARN_{time.time_ns()}"

    sub_logger.info(test_token_info)
    sub_logger.warning(test_token_warn)

    # 刷新 handlers
    base_logger = logging.getLogger("homevlog")
    for h in base_logger.handlers:
        h.flush()

    main_text = latest_main.read_text(encoding="utf-8")
    err_text = latest_err.read_text(encoding="utf-8")
    jsonl_text = latest_jsonl.read_text(encoding="utf-8")

    # 验证主日志包含 INFO 和 WARNING
    assert test_token_info in main_text
    assert test_token_warn in main_text
    assert "[diagnostics]" in main_text

    # 验证错误日志仅包含 WARNING，严格隔离掉普通的 INFO
    assert test_token_info not in err_text
    assert test_token_warn in err_text

    # 验证 JSONL 结构化解析
    lines = [json.loads(line) for line in jsonl_text.strip().splitlines() if line.strip()]
    tokens_in_jsonl = [item["message"] for item in lines]
    assert test_token_info in tokens_in_jsonl
    assert test_token_warn in tokens_in_jsonl


def test_dashboard_alert_bridge():
    """验证当仪表盘激活时，WARNING/ERROR 日志通过桥接器无缝送入仪表盘告警区。"""
    dash = PipelineDashboard(
        date="20260903",
        cam_index=0,
        total_prescreen=10,
        render_enabled=True,
        enabled=False,  # 测试环境非交互式
    )
    register_dashboard(dash)
    try:
        sub_logger = get_logger("scheduler")
        alert_msg = f"Alert_test_{time.time_ns()}"
        sub_logger.warning(alert_msg)

        found = any(alert_msg in a for a in dash.recent_alerts)
        assert found, "Warning log was not routed to dashboard recent_alerts"
    finally:
        unregister_dashboard()


def test_ui_components_render(tmp_path):
    """验证 Startup Banner, Summary Card, Error Summary 与 Scan Results 表格渲染无崩溃。"""
    # 1. 批量启动 Banner 与 单日启动 Banner
    from src.ui import print_batch_startup_banner
    print_batch_startup_banner(
        total_groups=7,
        date_range=("20260825", "20260901"),
        cameras=["baby_room (B888805AA3CD)"],
        output_dir="output",
    )
    print_startup_banner(
        date="20260903",
        cam_index=0,
        total_files=42,
        total_duration_s=3600.0,
        output_path="C:/test/out.mp4",
        batch_progress="[1/7]",
    )


    # 2. 成果汇总卡片 (带/不带阶段耗时)
    dummy_out = tmp_path / "test_out.mp4"
    dummy_out.write_bytes(b"dummy content 12345")
    print_summary_card(
        date="20260903",
        cam_index=0,
        total_files=42,
        total_input_dur=3600.0,
        output_path=dummy_out,
        elapsed_wall=45.2,
        stage_durations={"prescreen": 10.0, "analysis": 25.0, "render": 10.2},
    )

    # 3. 错误总结面板
    print_error_summary(["Error 1: Corrupt slice detected", "Error 2: QSV session timeout"])
    print_error_summary([])  # 空列表应静默无输出

    # 4. 扫描列表展示
    print_scan_results([("20260901", 0), ("20260902", 1)])


def test_pipeline_dashboard_lifecycle():
    """验证 PipelineDashboard 的更新逻辑、队列水位更新及生命周期开关。"""
    dash = PipelineDashboard(
        date="20260903",
        cam_index=0,
        total_prescreen=50,
        render_enabled=True,
        enabled=False,
    )
    dash.start()
    dash.update_prescreen(completed=10, total=50, latest_file="test_001.mp4", speed_str="12.0文件/s")
    dash.update_analysis(completed=5, total=15, latest_file="test_002.mp4", speed_str="均速 1.5s/文件")
    dash.update_render(completed=2, total=4, latest_batch="Batch 1 on NVENC", speed_str="均速 8.2s/批")
    dash.set_queue_status(prescreen_q=30, analysis_q=10, render_q=2)
    dash.set_scheduler_state("COOPERATIVE_BURST", active_nv=1, max_nv=1)
    dash.add_alert("Test warning message")
    dash.stop()

    assert dash.prescreen_queue_size == 30
    assert dash.analysis_queue_size == 10
    assert dash.render_queue_size == 2
    assert dash.scheduler_state == "COOPERATIVE_BURST"
    assert len(dash.recent_alerts) >= 1


def test_cli_parser_and_date_normalization():
    """验证 CLI 解析器配置、日期容差格式归一化与日期范围计算。"""
    from src.ui import (
        build_arg_parser,
        normalize_date_string,
        parse_date_range,
        get_recent_dates,
        resolve_cli_dates,
    )
    import datetime

    # 1. 常见日期格式容差归一化
    assert normalize_date_string("20260320") == "20260320"
    assert normalize_date_string("2026-03-20") == "20260320"
    assert normalize_date_string("2026/03/20") == "20260320"
    assert normalize_date_string(" 2026.03.20 ") == "20260320"

    with pytest.raises(ValueError):
        normalize_date_string("not_a_date")

    # 2. 日期范围解析
    dates = parse_date_range("20260320..20260322")
    assert dates == ["20260320", "20260321", "20260322"]

    # 倒序输入自适应调正
    dates_rev = parse_date_range("20260322:20260320")
    assert dates_rev == ["20260320", "20260321", "20260322"]

    # 3. 相对最近 N 天计算
    base = datetime.date(2026, 3, 22)
    recent = get_recent_dates(3, base_date=base)
    assert recent == ["20260320", "20260321", "20260322"]

    # 4. 参数解析器测试
    parser = build_arg_parser()
    args1 = parser.parse_args(["--date", "2026-03-20", "--dry-run", "--stage", "analyze"])
    assert args1.date == "2026-03-20"
    assert args1.dry_run is True
    assert args1.stage == "analyze"
    assert resolve_cli_dates(args1) == ["20260320"]

    args2 = parser.parse_args(["--date-range", "20260320..20260321", "--doctor", "--clean-temp"])
    assert args2.doctor is True
    assert args2.clean_temp is True
    assert resolve_cli_dates(args2) == ["20260320", "20260321"]


def test_system_doctor_and_report():
    """验证环境体检医生模块数据采集与报告面板呈现。"""
    from src.ui import run_system_doctor, print_doctor_report
    report = run_system_doctor()
    assert "python" in report
    assert "ffmpeg" in report
    assert "gpu" in report
    assert "storage" in report
    assert "model" in report

    # 打印报告卡片不崩溃
    ok = print_doctor_report()
    assert isinstance(ok, bool)


def test_batch_summary_table_and_status_table(tmp_path):
    """验证多日批处理全景大表与数据库任务状态大表正常渲染。"""
    from src.ui import print_batch_summary_table, print_status_table
    from src.database import VlogDatabase

    # 1. 批量全景表
    mock_batch = [
        {
            "date": "20260320",
            "cam_index": 0,
            "cam_name": "baby_room",
            "total_files": 142,
            "input_duration_s": 89280.0,
            "vlog_duration_s": 1302.0,
            "condensation_ratio": 68.6,
            "output_size_mb": 2850.0,
            "wall_clock_s": 1302.0,
            "speedup_x": 68.6,
            "status": "SUCCESS",
        },
        {
            "date": "20260321",
            "cam_index": 0,
            "cam_name": "baby_room",
            "total_files": 140,
            "input_duration_s": 83880.0,
            "vlog_duration_s": 1670.0,
            "condensation_ratio": 50.3,
            "output_size_mb": 3200.0,
            "wall_clock_s": 1670.0,
            "speedup_x": 50.3,
            "status": "SUCCESS",
        }
    ]
    print_batch_summary_table(mock_batch)
    print_batch_summary_table([])  # 空表安全

    # 2. 状态总览表
    db_file = tmp_path / "test_status.db"
    db = VlogDatabase(db_path=db_file)
    try:
        db.add_file_task(
            filepath="C:/dummy/01_20260320100000_20260320101000.mp4",
            cam_index=0,
            date="20260320",
            file_start_time="20260320100000",
            file_end_time="20260320101000",
            file_duration=600.0,
        )
        print_status_table(db, camera_display_names={0: "baby_room (B888805AA3CD)"})
    finally:
        db.close()


def test_plain_progress_tracker():
    """验证无头模式流式进度心跳器触发无异常。"""
    from src.ui import PlainProgressTracker
    tracker = PlainProgressTracker(date="20260320", cam_index=0, total_files=100, interval_s=0.01)
    tracker.heartbeat(
        prescreen_done=50,
        prescreen_total=100,
        analysis_done=10,
        analysis_total=20,
        render_done=2,
        render_total=5,
        force=True,
    )

