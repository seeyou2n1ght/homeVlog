"""异构硬件自适应画像探测与并发参数推荐引擎 (Hardware Auto-Tuner).

负责：
1. 深入探测当前主机的物理硬件拓扑 (CPU 核心数、系统内存、NVIDIA GPU 显存/驱动、Intel QSV 支持)；
2. 基于排队平衡与流水线 Makespan 极小化目标，数学推导计算最优硬件并发向量；
3. 支持动态注入有效配置与体检医生 (--doctor) 集成展示。
"""

from dataclasses import dataclass, field
import logging
import math
import shutil
import subprocess
import sys
from typing import Any

logger = logging.getLogger("homevlog")


@dataclass
class HardwareSnapshot:
    """系统硬件与加速器规格画像。"""
    cpu_logical: int = 4
    cpu_physical: int = 4
    ram_total_gb: float = 8.0
    has_cuda: bool = False
    nv_device_count: int = 0
    nv_device_name: str = ""
    nv_vram_mb: int = 0
    nv_driver_version: str = ""
    has_nvenc: bool = False
    has_qsv: bool = False
    details: dict[str, Any] = field(default_factory=dict)


def probe_hardware() -> HardwareSnapshot:
    """深入探测当前主机的 CPU、RAM、NVIDIA GPU 与 Intel QSV 硬件特征。"""
    import psutil

    cpu_logical = psutil.cpu_count(logical=True) or 4
    cpu_physical = psutil.cpu_count(logical=False) or max(1, cpu_logical // 2)
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 1)

    has_cuda = False
    device_count = 0
    gpu_name = ""
    vram_mb = 0
    driver_version = ""

    try:
        import torch
        if torch.cuda.is_available():
            has_cuda = True
            device_count = torch.cuda.device_count()
            if device_count > 0:
                gpu_name = torch.cuda.get_device_name(0)
                vram_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    except Exception:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        drv = pynvml.nvmlSystemGetDriverVersion()
        driver_version = drv.decode() if isinstance(drv, bytes) else str(drv)
        if device_count > 0 and vram_mb == 0:
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            vram_mb = int(mem.total / (1024 * 1024))
            name = pynvml.nvmlDeviceGetName(h)
            gpu_name = name.decode() if isinstance(name, bytes) else str(name)
    except Exception:
        pass

    # 探测 FFmpeg 编码器加速支持
    has_nvenc = False
    has_qsv = False
    try:
        res = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True, timeout=5)
        enc_out = res.stdout or ""
        has_nvenc = "hevc_nvenc" in enc_out
        has_qsv = "hevc_qsv" in enc_out
    except Exception:
        pass

    return HardwareSnapshot(
        cpu_logical=cpu_logical,
        cpu_physical=cpu_physical,
        ram_total_gb=ram_gb,
        has_cuda=has_cuda,
        nv_device_count=device_count,
        nv_device_name=gpu_name,
        nv_vram_mb=vram_mb,
        nv_driver_version=driver_version,
        has_nvenc=has_nvenc,
        has_qsv=has_qsv,
    )


def compute_optimal_concurrency(snapshot: HardwareSnapshot) -> dict[str, Any]:
    """根据硬件画像数学推导计算最优并发配置字典。

    优化目标:
    min Makespan = min max(T_upstream, T_render) + T_concat
    在防 OOM (VRAM <= 85%) 与硬件 ASIC 饱和点内实现吞吐最大化。
    """
    rec: dict[str, Any] = {
        "hardware": {},
        "scheduler": {},
        "render": {},
        "detection": {},
    }

    # 1. 独显 NVENC / NVDEC 槽位决策 (遵循 ADR 0007 双轨信号量与异构主线铁律)
    if snapshot.has_cuda and snapshot.has_nvenc and snapshot.nv_vram_mb > 0:
        if snapshot.nv_vram_mb >= 7500:
            # 8GB 及以上显卡 (RTX 3060Ti, 3070, 3080, 4060, 4070, 4090)
            base_nv_enc = 2
            tail_nv_enc = 3
            max_nv_dec = 2
            vram_watermark = int(snapshot.nv_vram_mb * 0.78)
        elif snapshot.nv_vram_mb >= 5500:
            # 6GB 显卡 (RTX 2060, 3060 6G)
            base_nv_enc = 2
            tail_nv_enc = 2
            max_nv_dec = 1
            vram_watermark = int(snapshot.nv_vram_mb * 0.75)
        else:
            # 4GB 及以下入门卡
            base_nv_enc = 1
            tail_nv_enc = 2
            max_nv_dec = 1
            vram_watermark = int(snapshot.nv_vram_mb * 0.70)

        rec["hardware"]["device"] = "cuda:0"
        rec["hardware"]["max_nv_concurrency"] = base_nv_enc
        rec["scheduler"]["nvdec_cooperative"] = True
        rec["scheduler"]["max_nv_decoders"] = max_nv_dec
        rec["scheduler"]["vram_watermark_mb"] = vram_watermark
        rec["scheduler"]["cooperative_during_render"] = True
        rec["scheduler"]["watermark_high"] = 2
        rec["scheduler"]["watermark_low"] = 0
        rec["render"]["nv_tail_concurrency"] = tail_nv_enc
    else:
        base_nv_enc = 0
        tail_nv_enc = 2
        max_nv_dec = 0
        rec["hardware"]["device"] = "cpu"
        rec["hardware"]["max_nv_concurrency"] = 0
        rec["scheduler"]["nvdec_cooperative"] = False
        rec["scheduler"]["max_nv_decoders"] = 0

    # 2. 核显 QSV 槽位决策
    if snapshot.has_qsv:
        rec["hardware"]["max_qsv_concurrency"] = 8
        rec["hardware"]["max_qsv_analysis_concurrency"] = 7
        rec["hardware"]["max_qsv_render_concurrency"] = 1
        qsv_render_workers = 1
        qsv_analysis_slots = 7
    else:
        rec["hardware"]["max_qsv_concurrency"] = 0
        rec["hardware"]["max_qsv_analysis_concurrency"] = 0
        rec["hardware"]["max_qsv_render_concurrency"] = 0
        qsv_render_workers = 0
        qsv_analysis_slots = 0

    # 3. 渲染总并发决策 (主渲染阶段 2 NVENC + 1 QSV = 3 workers 铁律，尾段由 nv_tail_0 冲刺)
    total_render = base_nv_enc + qsv_render_workers
    rec["render"]["max_concurrency"] = max(1, min(3, total_render))

    # 4. 分析 Worker 并发决策 (严格对齐 QSV 独占兜底容量，杜绝降级回落时的信号量排队饥饿)
    if qsv_analysis_slots > 0:
        recommended_workers = min(max(2, snapshot.cpu_logical - 2), qsv_analysis_slots)
    elif max_nv_dec > 0:
        recommended_workers = max(1, max_nv_dec)
    else:
        recommended_workers = max(1, snapshot.cpu_logical // 2)
    rec["detection"]["analysis_max_workers"] = recommended_workers

    # 5. I/O 预算推荐
    rec["hardware"]["max_io_concurrency"] = min(64, max(16, snapshot.cpu_logical * 2))

    return rec


def apply_auto_tune(config: dict[str, Any], in_place: bool = True) -> dict[str, Any]:
    """若配置启用了 scheduler.auto_tune: true，自动将硬件推导参数合并补全至配置中。

    规则：已有专家显式配置保持最高优先级，仅对未定义或默认的字段进行针对性补全优化。
    """
    import copy
    target = config if in_place else copy.deepcopy(config)
    sched_cfg = target.get("scheduler", {})
    from src.core.config import uses_software_pipeline
    if not sched_cfg.get("auto_tune", False) or uses_software_pipeline(target):
        return target

    snapshot = probe_hardware()
    optimal = compute_optimal_concurrency(snapshot)

    for section, sec_dict in optimal.items():
        if section not in target:
            target[section] = {}
        for k, v in sec_dict.items():
            # 仅在配置未指定或处于自动推断时注入
            if k not in target[section]:
                target[section][k] = v

    return target
