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

    # 寻找最新的日志文件集合
    main_logs = sorted(LOGS_DIR.glob("homevlog_*.log"), key=lambda p: p.stat().st_mtime)
    err_logs = sorted(LOGS_DIR.glob("error_*.log"), key=lambda p: p.stat().st_mtime)
    jsonl_logs = sorted(LOGS_DIR.glob("events_*.jsonl"), key=lambda p: p.stat().st_mtime)

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
