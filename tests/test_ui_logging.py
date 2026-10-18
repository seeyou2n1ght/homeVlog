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
    size_metrics,
)


def test_size_metrics_keeps_decimal_and_binary_units_explicit():
    metrics = size_metrics(8_003_885_496)
    assert metrics["output_size_bytes"] == 8_003_885_496
    assert metrics["output_size_mb"] == pytest.approx(8003.89)
    assert metrics["output_size_mib"] == pytest.approx(7633.10)
    assert metrics["output_size_gb"] == pytest.approx(8.004)


def test_subsystem_adapter_and_contextual_formatter(tmp_path):
    """验证子系统上下文 Logger 能够正确注入 subsystem 标签并被格式化器解析。"""
    logger = get_logger("my_subsystem")
    assert logger.extra["subsystem"] == "my_subsystem"
    logger.info("Test contextual message from subsystem")


def test_error_log_and_jsonl_separation(tmp_path):
    """验证主日志、独立错误日志及 JSONL 事件流的三路分离机制。"""
    setup_logging()
    from src.core.utils import LOGS_DIR

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


def test_debug_reconfiguration_preserves_error_log_and_updates_existing_handlers(tmp_path, monkeypatch):
    from io import StringIO
    from rich.console import Console
    import src.core.utils as utils
    isolated = logging.Logger("diagnostic-test", logging.DEBUG)
    real_get_logger = logging.getLogger
    monkeypatch.setattr(utils.logging, "getLogger", lambda name=None: isolated if name == "homevlog" else real_get_logger(name))
    monkeypatch.setattr(utils, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(utils, "load_config", lambda: {})
    output = StringIO()
    try:
        utils.setup_logging()
        bridge = next(h for h in isolated.handlers if isinstance(h, utils.RichConsoleBridgeHandler))
        bridge._console = Console(file=output)
        isolated.info("ordinary-info")
        isolated.info("scan-visible", extra={"startup_progress": True})
        assert "ordinary-info" not in output.getvalue()
        assert "scan-visible" in output.getvalue()
        handlers = list(isolated.handlers)
        utils.setup_logging(logging.DEBUG)
        isolated.debug("debug-visible")
        for handler in isolated.handlers:
            handler.flush()
        assert isolated.handlers == handlers
        assert "debug-visible" in output.getvalue()
        assert "debug-visible" in next((tmp_path / "runs").glob("*.log")).read_text(encoding="utf-8")
        assert "debug-visible" in next((tmp_path / "events").glob("*.jsonl")).read_text(encoding="utf-8")
        assert "debug-visible" not in next((tmp_path / "error").glob("*.log")).read_text(encoding="utf-8")
    finally:
        for handler in isolated.handlers:
            handler.close()


@pytest.mark.parametrize("passed", [False, True])
def test_doctor_cli_exit_status(monkeypatch, passed):
    import main as cli
    monkeypatch.setattr("sys.argv", ["main.py", "--doctor"])
    monkeypatch.setattr(cli, "print_doctor_report", lambda _console: passed)
    if passed:
        cli.main()
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 1


def test_doctor_disk_warning_is_failure(monkeypatch):
    from io import StringIO
    from rich.console import Console
    import src.ui.doctor as doctor
    report = {
        "python": {"passed": True, "version": "3.12", "packages": {}},
        "ffmpeg": {"passed": True, "version": "test", "nvenc": True, "qsv": True},
        "gpu": {"cuda_available": False, "devices": []},
        "storage": {"inputs": [], "output_free_gb": 1, "output_min_gb": 20, "disk_ok": False},
        "model": {"exists": True, "model_path": "model.pt", "size_mb": 1},
    }
    monkeypatch.setattr(doctor, "run_system_doctor", lambda: report)
    output = StringIO()
    assert not doctor.print_doctor_report(Console(file=output, width=180))
    assert "全部核心环境已就绪" not in output.getvalue()


def test_ui_components_render(tmp_path, capsys):
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
        cam_name="育儿房",
        total_files=42,
        total_duration_s=3600.0,
        output_path="C:/test/out.mp4",
        batch_progress="[1/7]",
    )
    banner = capsys.readouterr().out
    assert "处理日期" in banner and "监控机位" in banner
    assert "育儿房" in banner and "Cam 0" not in banner


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
    output = capsys.readouterr().out
    assert "20260903" in output and "20260902" in output
    assert "Corrupt slice detected" in output and "QSV session timeout" in output


def test_summary_discloses_presence_and_micro_motion_instead_of_false_dynamic_percentage(tmp_path, capsys):
    output = tmp_path / "summary.mp4"
    output.write_bytes(b"test media placeholder")
    output.with_suffix(".meta.json").write_text(json.dumps({"metrics": {
        "vlog_duration_s": 25., "dynamic_duration_s": 4., "static_duration_s": 0.,
        "source_duration_by_state_s": {"DYNAMIC": 4., "PRESENCE": 80., "MICRO_MOTION": 16.},
    }}), encoding="utf-8")
    print_summary_card("20260610", 0, 1, 100., output, 10.)
    displayed = capsys.readouterr().out
    assert "源时段分布" in displayed
    assert "有人驻留" in displayed and "80.0s" in displayed
    assert "微动" in displayed and "16.0s" in displayed
    assert "100.0%" not in displayed


def test_pipeline_dashboard_lifecycle():
    """验证 PipelineDashboard 的更新逻辑、队列水位更新及生命周期开关。"""
    from rich.console import Console

    dash = PipelineDashboard(
        date="20260903",
        cam_index=0,
        cam_name="育儿房",
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
    console = Console(record=True, width=120)
    console.print(dash._generate_view())
    view = console.export_text()
    assert "日期: 20260903" in view
    assert "机位: 育儿房" in view
    assert "Cam 0" not in view
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


def test_system_doctor_and_report(tmp_path, monkeypatch):
    """验证环境体检医生模块数据采集与报告面板呈现。"""
    from src.ui import run_system_doctor, print_doctor_report
    # Routine regression must not depend on the availability of a production NAS.
    input_dir=tmp_path/'doctor_input';input_dir.mkdir()
    monkeypatch.setattr('src.core.utils.get_input_dirs',lambda config:[str(input_dir)])
    report = run_system_doctor()
    assert "python" in report
    assert "ffmpeg" in report
    assert "gpu" in report
    assert "storage" in report
    assert "model" in report
    assert report['storage']['inputs'][0]['exists'] is True

    # 打印报告卡片不崩溃
    ok = print_doctor_report()
    assert isinstance(ok, bool)


def test_batch_summary_table_and_status_table(tmp_path, capsys, monkeypatch):
    """验证多日批处理全景大表与数据库任务状态大表正常渲染。"""
    from src.ui import print_batch_summary_table, print_status_table
    from src.ui import console
    monkeypatch.setattr(console, "width", 220)
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
        incomplete_output = tmp_path / "incomplete.mp4"
        incomplete_output.write_bytes(b"partial")
        db.set_render_status("20260320", 0, "FAILED", output_file=str(incomplete_output),
                             error_msg="required companion assets failed")
        print_status_table(db, camera_display_names={0: "baby_room (B888805AA3CD)"})
        output = capsys.readouterr().out
        assert "20260320" in output and "20260321" in output
        assert "baby_room" in output
        assert "渲染失败" in output and "required companion assets failed" in output
    finally:
        db.close()


def test_plain_progress_tracker(capsys):
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
    output = capsys.readouterr().out
    assert "20260320" in output and "50/100" in output and "2/5" in output


@pytest.mark.parametrize("success", [False, True])
def test_cli_reports_actual_run_outcome(tmp_path, monkeypatch, capsys, success):
    from unittest.mock import MagicMock
    import main as cli
    from src.database import VlogDatabase

    db = VlogDatabase(tmp_path / "cli.db")
    db.add_file_task("clip.mp4", 0, "20260320", "20260320000000", "20260320000100", 60)
    monkeypatch.setattr("sys.argv", ["main.py", "--date", "20260320", "--no-tui"])
    monkeypatch.setattr(cli, "VlogDatabase", lambda: db)
    monkeypatch.setattr(cli, "scan_directory", lambda *a, **k: None)
    monkeypatch.setattr(cli, "process_date_cam", lambda *a, **k: success)
    monkeypatch.setattr(cli, "cleanup_resources", lambda: None)
    monkeypatch.setattr(cli, "check_disk_space", lambda *_a: True)
    monkeypatch.setattr("src.monitor.get_monitor", lambda: MagicMock())
    monkeypatch.setattr("src.config.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(cli.logging, "shutdown", lambda: None)
    if success:
        cli.main()
    else:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 1
    output = capsys.readouterr().out
    assert ("处理完成" in output) is success
    assert ("处理失败" in output) is not success


@pytest.mark.parametrize("selected", [False, True])
def test_cli_disk_stop_is_not_success(tmp_path, monkeypatch, capsys, selected):
    from unittest.mock import MagicMock
    import main as cli
    from src.database import VlogDatabase
    db = VlogDatabase(tmp_path / "disk_cli.db")
    db.add_file_task("clip.mp4", 0, "20260320", "20260320000000", "20260320000100", 60)
    monkeypatch.setattr("sys.argv", ["main.py", "--no-tui"] + (["--date", "20260320"] if selected else []))
    monkeypatch.setattr(cli, "VlogDatabase", lambda: db)
    monkeypatch.setattr(cli, "scan_directory", lambda *a, **k: None)
    processor = MagicMock()
    monkeypatch.setattr(cli, "process_date_cam", processor)
    monkeypatch.setattr(cli, "check_disk_space", lambda *_a: False)
    monkeypatch.setattr(cli, "cleanup_resources", lambda: None)
    monkeypatch.setattr(cli, "run_pipeline", lambda **_k: {"total": 1, "ok": 0, "failed": 0,
                          "remaining": 1, "stopped_reason": "LOW_DISK_SPACE"})
    monkeypatch.setattr("src.monitor.get_monitor", MagicMock())
    monkeypatch.setattr("src.config.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(cli.logging, "shutdown", lambda: None)
    try:
        with pytest.raises(SystemExit) as caught:
            cli.main()
        assert caught.value.code == 1
        processor.assert_not_called()
        output = capsys.readouterr().out
        assert "LOW_DISK_SPACE" in output and "未处理=1" in output
        assert "处理完成" not in output
    finally:
        if not db.is_closed:
            db.close()


@pytest.fixture
def cli_workspace(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    import main as cli
    from src.database import VlogDatabase
    from src.segment import Segment
    db = VlogDatabase(tmp_path / "selection.db")
    original_close = db.close
    for date in ("20260901", "20260902"):
        for cam in (0, 1):
            source = f"cam{cam}_{date}.mp4"
            db.add_file_task(source, cam, date, date + "000000", date + "000100", 60)
            db.set_prescreen_result(source, "SUSPICIOUS")
            db.set_analysis_result(source, [Segment(0, 60, "STATIC", source)])
            seg_id = db.get_segments_for_file(source)[0]["id"]
            db.update_segment_review(seg_id, "FN")
    processor = MagicMock(return_value=True)
    cleanup = MagicMock()
    monitor = MagicMock()
    monkeypatch.setattr(cli, "VlogDatabase", lambda: db)
    monkeypatch.setattr(db, "close", lambda: None)
    monkeypatch.setattr(cli, "load_config", lambda *a, **k: {})
    monkeypatch.setattr(cli, "scan_directory", lambda *a, **k: None)
    monkeypatch.setattr(cli, "process_date_cam", processor)
    monkeypatch.setattr(cli, "cleanup_resources", lambda: None)
    monkeypatch.setattr(cli, "cleanup_temp_artifacts", cleanup)
    monkeypatch.setattr(cli, "check_disk_space", lambda *_a: True)
    monkeypatch.setattr("src.monitor.get_monitor", lambda: monitor)
    monkeypatch.setattr("src.config.OUTPUT_DIR", tmp_path)
    monkeypatch.setattr(cli.logging, "shutdown", lambda: None)
    try:
        yield SimpleNamespace(cli=cli, db=db, processor=processor, cleanup=cleanup, monitor=monitor)
    finally:
        original_close()


@pytest.mark.parametrize("selector,expected", [
    (["--cam", "1"], [("20260901", 1), ("20260902", 1)]),
    (["--date", "20260902", "--cam", "0"], [("20260902", 0)]),
    (["--date-range", "20260901..20260902", "--cam", "0"], [("20260901", 0), ("20260902", 0)]),
])
def test_cli_selection_matches_processing_and_reanalysis(cli_workspace, monkeypatch, selector, expected):
    w = cli_workspace
    monkeypatch.setattr("sys.argv", ["main.py", "--no-tui", "--reanalyze", *selector])
    w.cli.main()
    assert [(call.args[1], call.args[2]) for call in w.processor.call_args_list] == expected
    for row in w.db.conn.execute("SELECT filepath,date,cam_index,analysis_status FROM file_tasks"):
        selected = (row["date"], row["cam_index"]) in expected
        assert row["analysis_status"] == ("PENDING" if selected else "ANALYZED")
        assert bool(w.db.get_segments_for_file(row["filepath"])) is not selected
    assert w.db.conn.execute("SELECT COUNT(*) FROM human_reviews").fetchone()[0] == 4
    w.monitor.shutdown.assert_called_once()


def test_cli_dry_run_does_not_reset_or_clean(cli_workspace, monkeypatch):
    w = cli_workspace
    w.db.set_render_status("20260901", 1, "COMPLETED")
    before = list(w.db.conn.iterdump())
    monkeypatch.setattr("sys.argv", ["main.py", "--cam", "1", "--dry-run", "--reanalyze", "--clean-temp"])
    w.cli.main()
    assert list(w.db.conn.iterdump()) == before
    w.processor.assert_not_called()
    w.cleanup.assert_not_called()


def test_cli_reanalysis_cannot_reset_an_owned_day(cli_workspace,external_day_owner,monkeypatch,tmp_path):
    from src.stages import renderer
    w=cli_workspace
    external_day_owner(w.db.db_path,renderer.TEMP_DIR,tmp_path/'out.mp4','20260901',1)
    before=list(w.db.conn.iterdump())
    monkeypatch.setattr('sys.argv',['main.py','--date','20260901','--cam','1','--reanalyze','--no-tui'])
    with pytest.raises(RuntimeError,match='already owned'):
        w.cli.main()
    assert before==list(w.db.conn.iterdump())
    w.processor.assert_not_called()


@pytest.mark.parametrize("force", [False, True])
def test_cli_reanalysis_requires_force_for_completed_groups(cli_workspace, monkeypatch, force):
    w = cli_workspace
    w.db.set_render_status("20260901", 1, "COMPLETED")
    before = list(w.db.conn.iterdump())
    monkeypatch.setattr("sys.argv", ["main.py", "--cam", "1", "--reanalyze", "--no-tui"] + (["--force-render"] if force else []))
    if force:
        w.cli.main()
        assert w.processor.call_count == 2
        assert all(call.kwargs["force_render"] for call in w.processor.call_args_list)
        assert w.db.conn.execute("SELECT COUNT(*) FROM human_reviews").fetchone()[0] == 4
    else:
        with pytest.raises(SystemExit) as caught:
            w.cli.main()
        assert caught.value.code == 2
        assert list(w.db.conn.iterdump()) == before
        w.processor.assert_not_called()


@pytest.mark.parametrize("flags", [["--stage", "prescreen"], ["--stage", "render"], ["--yes"], ["-y"]])
def test_cli_rejects_unimplemented_options(flags):
    from src.ui import build_arg_parser
    with pytest.raises(SystemExit) as caught:
        build_arg_parser().parse_args(flags)
    assert caught.value.code == 2


@pytest.fixture
def cli_pending_media(monkeypatch):
    from unittest.mock import MagicMock
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    for field in ('_processes', '_jobs', '_deferred', '_starting'):
        monkeypatch.setattr(registry, field, {})
    process, permit = MagicMock(), MagicMock()
    process.poll.return_value = None
    def defer():
        registry.defer_cleanup('previous-day', process, permit)
    return registry, permit, defer


@pytest.mark.parametrize('flags', [[], ['--reanalyze'], ['--clean-temp'], ['--config', 'unused.yaml']])
def test_cli_pending_media_rejects_before_mutation(cli_workspace, cli_pending_media, monkeypatch, capsys, flags):
    w = cli_workspace
    registry, permit, defer = cli_pending_media
    before = list(w.db.conn.iterdump())
    defer()
    monkeypatch.setattr('sys.argv', ['main.py', '--date', '20260901', '--no-tui', *flags])
    with pytest.raises(SystemExit) as stopped:
        w.cli.main()
    assert stopped.value.code == 1
    assert list(w.db.conn.iterdump()) == before
    w.processor.assert_not_called()
    w.cleanup.assert_not_called()
    permit.release.assert_not_called()
    assert registry.is_interrupted()
    assert 'MEDIA_TERMINATION_UNCONFIRMED' in capsys.readouterr().out


@pytest.mark.parametrize('outcome', ['exception', 'pending_false', 'ordinary_false'])
def test_cli_stops_cross_day_after_unconfirmed_media(cli_workspace, cli_pending_media, monkeypatch, capsys, outcome):
    from src.hardware.ffmpeg import MediaTerminationUnconfirmed
    w = cli_workspace
    registry, permit, defer = cli_pending_media
    before = list(w.db.conn.iterdump())
    def process(*args, **kwargs):
        if w.processor.call_count == 1:
            if outcome == 'exception':
                raise MediaTerminationUnconfirmed('controlled previous reader')
            if outcome == 'pending_false':
                defer()
            return False
        return True
    w.processor.side_effect = process
    monkeypatch.setattr('sys.argv', ['main.py', '--cam', '1', '--no-tui'])
    with pytest.raises(SystemExit) as stopped:
        w.cli.main()
    assert stopped.value.code == 1
    assert list(w.db.conn.iterdump()) == before
    assert w.processor.call_count == (2 if outcome == 'ordinary_false' else 1)
    output = capsys.readouterr().out
    if outcome != 'ordinary_false':
        assert 'MEDIA_TERMINATION_UNCONFIRMED' in output and '未处理=1' in output
    else:
        assert 'MEDIA_TERMINATION_UNCONFIRMED' not in output
    permit.release.assert_not_called()
