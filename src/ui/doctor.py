"""HomeVlog 系统环境与硬件就绪体检医生模块 (System Readiness Doctor).

用于在执行长时间批量视频浓缩流水线前，全面自检：
1. Python 依赖与版本；
2. FFmpeg / FFprobe 及其硬件编码器 (hevc_nvenc / hevc_qsv)；
3. NVIDIA CUDA、驱动与显存预算；
4. 存储路径、NAS 连通性与磁盘剩余空间；
5. YOLO 模型权重文件完整性。
"""

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.core.config import load_config, PROJECT_ROOT, OUTPUT_DIR, TEMP_DIR


def check_python_environment() -> Dict[str, Any]:
    """检测 Python 版本与关键算法扩展包。"""
    version_str = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    passed = sys.version_info >= (3, 10)

    packages = {}
    for pkg in ["torch", "av", "ultralytics", "cv2", "rich", "psutil"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "installed")
        except ImportError:
            packages[pkg] = None

    all_pkgs_ok = all(v is not None for v in packages.values())
    return {
        "passed": passed and all_pkgs_ok,
        "version": version_str,
        "packages": packages,
    }


def check_ffmpeg_and_hardware_encoders() -> Dict[str, Any]:
    """检测 ffmpeg、ffprobe 可执行文件及硬件加速编码器支持。"""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if not ffmpeg_path or not ffprobe_path:
        return {
            "passed": False,
            "ffmpeg_path": ffmpeg_path,
            "ffprobe_path": ffprobe_path,
            "version": "未找到",
            "nvenc": False,
            "qsv": False,
            "error": "未在环境变量 PATH 中找到 ffmpeg 或 ffprobe",
        }

    # 获取版本与编码器列表
    try:
        res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5)
        encoders_output = res.stdout or ""
        nvenc_supported = "hevc_nvenc" in encoders_output
        qsv_supported = "hevc_qsv" in encoders_output
    except Exception as e:
        nvenc_supported = False
        qsv_supported = False
        encoders_output = str(e)

    # 获取版本号
    try:
        ver_res = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=5)
        first_line = (ver_res.stdout or "").splitlines()[0] if ver_res.stdout else "已安装"
    except Exception:
        first_line = "未知版本"

    passed = bool(ffmpeg_path and ffprobe_path and (nvenc_supported or qsv_supported))
    return {
        "passed": passed,
        "ffmpeg_path": ffmpeg_path,
        "ffprobe_path": ffprobe_path,
        "version": first_line,
        "nvenc": nvenc_supported,
        "qsv": qsv_supported,
    }


def check_gpu_and_cuda() -> Dict[str, Any]:
    """检测 NVIDIA 独显、CUDA 算力与显存状态。"""
    info: Dict[str, Any] = {
        "cuda_available": False,
        "device_count": 0,
        "devices": [],
        "driver_version": "N/A",
    }

    try:
        import torch
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["device_count"] = torch.cuda.device_count()
            for i in range(info["device_count"]):
                name = torch.cuda.get_device_name(i)
                total_mem_mb = torch.cuda.get_device_properties(i).total_memory / (1024 * 1024)
                info["devices"].append({
                    "index": i,
                    "name": name,
                    "total_memory_mb": round(total_mem_mb, 1),
                })
    except Exception:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        info["driver_version"] = pynvml.nvmlSystemGetDriverVersion().decode() if isinstance(pynvml.nvmlSystemGetDriverVersion(), bytes) else pynvml.nvmlSystemGetDriverVersion()
    except Exception:
        pass

    return info


def check_storage_and_paths() -> Dict[str, Any]:
    """检测素材输入目录、输出磁盘空间与临时目录状态。"""
    config = load_config()
    from src.core.utils import get_input_dirs, check_disk_space

    input_dirs = get_input_dirs(config)
    input_status = []
    for d in input_dirs:
        t0 = time.monotonic()
        exists = Path(d).exists()
        latency_ms = round((time.monotonic() - t0) * 1000, 1) if exists else None
        input_status.append({
            "path": d,
            "exists": exists,
            "latency_ms": latency_ms,
        })

    # 输出盘空间
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(out_dir).free
    free_gb = round(free_bytes / (1024 ** 3), 2)
    min_gb = config.get("recovery", {}).get("min_disk_space_gb", 20)
    disk_ok = free_gb >= min_gb

    # 临时目录
    temp_dir = Path(TEMP_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)

    return {
        "inputs": input_status,
        "output_dir": str(out_dir),
        "output_free_gb": free_gb,
        "output_min_gb": min_gb,
        "disk_ok": disk_ok,
    }


def check_model_weights() -> Dict[str, Any]:
    """检测 YOLO 目标检测模型权重文件状态。"""
    config = load_config()
    model_path_str = config.get("yolo", {}).get("model_path", "models/yolo11m.pt")
    model_path = Path(model_path_str)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    exists = model_path.is_file()
    size_mb = round(model_path.stat().st_size / (1024 * 1024), 2) if exists else 0.0

    return {
        "model_path": str(model_path),
        "exists": exists,
        "size_mb": size_mb,
    }


def run_system_doctor() -> Dict[str, Any]:
    """执行全部就绪性检查并返回诊断结果字典。"""
    return {
        "python": check_python_environment(),
        "ffmpeg": check_ffmpeg_and_hardware_encoders(),
        "gpu": check_gpu_and_cuda(),
        "storage": check_storage_and_paths(),
        "model": check_model_weights(),
    }


def print_doctor_report(console: Console = None) -> bool:
    """以 Rich 格式在控制台打印高质感系统就绪诊断卡片。
    
    返回: True (全部就绪), False (存在关键阻塞项)
    """
    if console is None:
        console = Console()

    report = run_system_doctor()
    table = Table(show_header=True, header_style="bold cyan", border_style="dim")
    table.add_column("检查项目", style="bold", width=22)
    table.add_column("状态", justify="center", width=10)
    table.add_column("详细诊断与配置信息", style="white")

    all_passed = True

    # 1. Python 环境
    py_data = report["python"]
    if py_data["passed"]:
        table.add_row("Python 与算法库", "[bold green]✔ 正常[/bold green]", f"Python {py_data['version']} (PyTorch, PyAV, YOLO, OpenCV 就绪)")
    else:
        all_passed = False
        missing = [k for k, v in py_data["packages"].items() if v is None]
        table.add_row("Python 与算法库", "[bold red]✖ 异常[/bold red]", f"缺少依赖包: {', '.join(missing)}")

    # 2. FFmpeg 与硬件编解码
    ff_data = report["ffmpeg"]
    if ff_data["passed"]:
        encoders = []
        if ff_data["nvenc"]:
            encoders.append("NVENC HEVC")
        if ff_data["qsv"]:
            encoders.append("QSV HEVC")
        table.add_row("FFmpeg 硬件编解码", "[bold green]✔ 就绪[/bold green]", f"{ff_data['version'][:35]} | 硬件加速: {', '.join(encoders)}")
    else:
        all_passed = False
        table.add_row("FFmpeg 硬件编解码", "[bold red]✖ 缺失[/bold red]", ff_data.get("error", "未检测到 hevc_nvenc 或 hevc_qsv 硬件加速编码器"))

    # 3. GPU 与 CUDA
    gpu_data = report["gpu"]
    if gpu_data["cuda_available"] and gpu_data["devices"]:
        dev_desc = ", ".join([f"{d['name']} ({d['total_memory_mb']/1024:.1f} GB)" for d in gpu_data["devices"]])
        table.add_row("GPU 算力与显存", "[bold green]✔ 就绪[/bold green]", f"{dev_desc} (驱动: {gpu_data['driver_version']})")
    else:
        table.add_row("GPU 算力与显存", "[yellow]⚠ 降级[/yellow]", "未检测到可用 CUDA GPU，将以 CPU 模式运行")

    # 4. 存储与 NAS
    st_data = report["storage"]
    storage_msgs = []
    for inp in st_data["inputs"]:
        if inp["exists"]:
            storage_msgs.append(f"[green]✔[/green] {Path(inp['path']).name} ({inp['latency_ms']}ms)")
        else:
            storage_msgs.append(f"[red]✖ 未挂载:[/] {inp['path']}")
            all_passed = False
    storage_msgs.append(f"输出盘剩余空间: [bold]{st_data['output_free_gb']} GB[/bold] (阈值 {st_data['output_min_gb']} GB)")

    st_status = "[bold green]✔ 就绪[/bold green]" if st_data["disk_ok"] and all(i["exists"] for i in st_data["inputs"]) else "[bold red]✖ 警告[/bold red]"
    table.add_row("存储与网络挂载", st_status, " | ".join(storage_msgs))

    # 5. YOLO 权重
    md_data = report["model"]
    if md_data["exists"]:
        table.add_row("YOLO 视觉模型", "[bold green]✔ 就绪[/bold green]", f"{Path(md_data['model_path']).name} ({md_data['size_mb']} MB)")
    else:
        all_passed = False
        table.add_row("YOLO 视觉模型", "[bold red]✖ 缺失[/bold red]", f"模型文件不存在: {md_data['model_path']}")

    title_color = "green" if all_passed else "yellow"
    panel_title = f"[bold {title_color}]🩺 HomeVlog 系统就绪体检报告[/bold {title_color}]"
    panel_sub = "[bold green]✨ 全部核心环境已就绪，可执行高速异构浓缩流水线[/bold green]" if all_passed else "[bold yellow]⚠ 检测到潜在环境风险，请根据上述提示排查[/bold yellow]"

    panel = Panel(table, title=panel_title, subtitle=panel_sub, border_style=title_color, padding=(1, 2))
    console.print(panel)
    return all_passed
