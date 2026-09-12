"""HomeVlog 命令行参数构建与规范化模块 (CLI Parser & Argument Normalization).

特性：
- 结构化参数分组 (Target, Pipeline, Diagnostics, Maintenance, UI/Logging)；
- 智能日期容差：自动将 YYYY-MM-DD 归一化为 YYYYMMDD；
- 日期范围与相对日期支持：--date-range (START..END) 与 --days N；
- 规划预览支持 (--dry-run)；
- 状态查看与系统诊断入参支持 (--status, --doctor)。
"""

import argparse
import datetime
import re
from typing import Optional


def normalize_date_string(date_str: str) -> str:
    """将多种常见日期格式归一化为标准 8 位数字格式 YYYYMMDD。
    
    支持格式：
    - 20260320 -> 20260320
    - 2026-03-20 -> 20260320
    - 2026/03/20 -> 20260320
    - 2026.03.20 -> 20260320
    """
    clean = re.sub(r"[-_/\.]", "", date_str.strip())
    if len(clean) == 8 and clean.isdigit():
        # 简单校验合法日期
        try:
            datetime.datetime.strptime(clean, "%Y%m%d")
            return clean
        except ValueError:
            raise ValueError(f"无效的日期数值: {date_str}")
    raise ValueError(f"无法识别的日期格式: '{date_str}', 请使用 YYYYMMDD 或 YYYY-MM-DD")


def parse_date_range(range_str: str) -> list[str]:
    """解析日期范围字符串，支持 'START..END'、'START:END' 或 'START-END'（若为8位-8位）。
    
    返回连续的 YYYYMMDD 列表（闭区间）。
    """
    parts = re.split(r"\.\.|:|(?:(?<=\d{8})-(?=\d{8}))", range_str.strip())
    if len(parts) != 2:
        raise ValueError(f"无效的日期范围表达式: '{range_str}'，正确格式例如: 20260320..20260325")

    start_date = normalize_date_string(parts[0])
    end_date = normalize_date_string(parts[1])

    dt_start = datetime.datetime.strptime(start_date, "%Y%m%d")
    dt_end = datetime.datetime.strptime(end_date, "%Y%m%d")

    if dt_start > dt_end:
        dt_start, dt_end = dt_end, dt_start

    dates = []
    curr = dt_start
    while curr <= dt_end:
        dates.append(curr.strftime("%Y%m%d"))
        curr += datetime.timedelta(days=1)
    return dates


def get_recent_dates(num_days: int, base_date: Optional[datetime.date] = None) -> list[str]:
    """获取以 base_date（默认今天）为基准前推 num_days 天的 YYYYMMDD 列表。"""
    if num_days <= 0:
        return []
    base = base_date or datetime.date.today()
    dates = []
    for i in range(num_days - 1, -1, -1):
        d = base - datetime.timedelta(days=i)
        dates.append(d.strftime("%Y%m%d"))
    return dates


class FormattedHelpFormatter(argparse.HelpFormatter):
    """优化后的命令行帮助格式化器，具备更清晰的列宽和排版。"""
    def __init__(self, prog):
        super().__init__(prog, max_help_position=32, width=100)


def build_arg_parser() -> argparse.ArgumentParser:
    """构建分组清晰、体验友好的 HomeVlog 统一命令行参数解析器。"""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="🎬 HomeVlog — 家庭监控素材智能浓缩与异构渲染系统",
        formatter_class=FormattedHelpFormatter,
        add_help=True,
    )

    # 1. 目标范围过滤
    group_target = parser.add_argument_group("🎯 目标范围过滤 (Target Selection)")
    group_target.add_argument(
        "--date",
        type=str,
        default=None,
        metavar="YYYYMMDD",
        help="指定处理单个日期 (支持 YYYYMMDD 或 YYYY-MM-DD)",
    )
    group_target.add_argument(
        "--date-range",
        type=str,
        default=None,
        metavar="START..END",
        help="指定处理日期范围 (例如: 20260320..20260324)",
    )
    group_target.add_argument(
        "--days",
        type=int,
        default=None,
        metavar="N",
        help="处理最近 N 天的素材 (例如: --days 3)",
    )
    group_target.add_argument(
        "--cam",
        type=int,
        default=None,
        metavar="INDEX",
        help="指定监控机位逻辑编号 (未指定时处理该日期下所有机位)",
    )
    group_target.add_argument(
        "--input-dir",
        action="append",
        default=None,
        metavar="PATH",
        help="覆盖或追加监控切片输入目录 (支持多次指定)",
    )

    # 2. 流水线阶段与运行控制
    group_pipeline = parser.add_argument_group("⚙️ 流水线阶段控制 (Pipeline & Execution)")
    group_pipeline.add_argument(
        "--stage",
        type=str,
        choices=["all", "prescreen", "analyze", "render"],
        default="all",
        help="指定执行阶段: all(完整流程), prescreen(仅预筛选), analyze(至精析), render(仅渲染)",
    )
    group_pipeline.add_argument(
        "--no-render",
        "--skip-render",
        action="store_true",
        dest="no_render",
        help="跳过 Pass 2 视频渲染 (等价于 --stage analyze)",
    )
    group_pipeline.add_argument(
        "--force-render",
        action="store_true",
        help="强制重新渲染，忽略已存在的同名成片",
    )
    group_pipeline.add_argument(
        "--dry-run",
        action="store_true",
        help="规划预览模式：仅统计待处理素材与阶段计划，不启动实际转码",
    )

    # 3. 状态速查与系统诊断
    group_diag = parser.add_argument_group("📊 状态速查与诊断 (Status & Diagnostics)")
    group_diag.add_argument(
        "--status",
        action="store_true",
        help="展示数据库内各日期/机位的归档处理进度总览表",
    )
    group_diag.add_argument(
        "--scan",
        action="store_true",
        help="仅执行输入目录扫描入库并输出切片统计，不执行后续处理",
    )
    group_diag.add_argument(
        "--doctor",
        action="store_true",
        help="执行系统与硬件环境体检 (FFmpeg硬件编解码/CUDA/显存/NAS延迟/模型权重)",
    )

    # 4. 维护与缓存重置
    group_maint = parser.add_argument_group("🧹 运维与缓存重置 (Maintenance & Cleanup)")
    group_maint.add_argument(
        "--clean-temp",
        action="store_true",
        help="清理临时渲染批次视频与临时脚本 (可独立运行)",
    )
    group_maint.add_argument(
        "--reanalyze",
        action="store_true",
        help="重置指定日期/机位的分析缓存，强制使用最新模型重新精析与渲染",
    )
    group_maint.add_argument(
        "--yes", "-y",
        action="store_true",
        help="对维护或重析等操作自动确认，跳过交互式二次询问",
    )

    # 5. 终端交互与系统配置
    group_ui = parser.add_argument_group("🖥️ 终端显示与系统配置 (UI & Configuration)")
    group_ui.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="指定自定义 settings.yaml 配置文件路径",
    )
    group_ui.add_argument(
        "--no-tui",
        "--plain",
        action="store_true",
        dest="no_tui",
        help="禁用动态 Live 仪表盘，启用纯文本流式心跳日志 (适合无头/后台运行)",
    )
    group_ui.add_argument(
        "--debug",
        action="store_true",
        help="开启详细 DEBUG 级日志输出",
    )

    return parser


def resolve_cli_dates(args: argparse.Namespace) -> list[str]:
    """根据 CLI 参数解析出规范化的待处理日期列表。"""
    dates = []
    if getattr(args, "date", None):
        dates.append(normalize_date_string(args.date))
    elif getattr(args, "date_range", None):
        dates.extend(parse_date_range(args.date_range))
    elif getattr(args, "days", None):
        dates.extend(get_recent_dates(args.days))
    return sorted(list(set(dates)))
