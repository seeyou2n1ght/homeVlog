# HomeVlog

HomeVlog 是一个自动化的家庭监控视频浓缩系统。它能够智能识别原始录像中的运动片段，并将多段素材合并、浓缩成一段每日 Vlog。

## 核心特性

- **多级流水线架构**：
    - **Pass 1 (Prescreening)**：基于硬件加速的快速抽帧预筛，过滤掉完全静止的文件。支持**环境光自适应阈值**（红外夜视自动下调阈值）和**首帧比对**（消除盲区）。
    - **Pass 1.5 (Analysis)**：对可疑文件进行全量分析。采用 **Robust Noise Floor Estimation**（鲁棒底噪估计）算法，即使在画面大部分时间都在运动的情况下也能精准识别。
    - **Pass 2 (Rendering)**：利用硬件加速（CUDA/QSV）进行高效的分辨率缩放和编码，将动态片段以原速保留，静态片段以快进/幻灯片形式浓缩。
- **全局平滑处理**：动作片段的平滑处理在全局时间线（Global Timeline）阶段进行，有效解决了跨文件边界处的“动作截断”问题。
- **时钟漂移补偿**：基于文件真实时长和解出的总帧数进行等比映射，消除变帧率（VFR）或掉帧引起的音画不同步问题。
- **并发处理**：支持多 GPU（如 NVIDIA CUDA + Intel QSV）并行分析，最大化处理效率。

## 快速开始

### 环境准备

项目使用 [uv](https://github.com/astral-sh/uv) 管理 Python 环境。

```powershell
uv sync
```

### 配置文件

修改 `config/settings.yaml` 以匹配你的本地环境：
- `paths.input_dir`: 摄像机原始录像存储路径。
- `detection.roi_crop`: 设置感兴趣区域（ROI），屏蔽时间戳或无关区域。
- `output`: 设置输出分辨率、帧率及编码参数。

### 运行

```powershell
uv run main.py
```

## 调试工具

你可以使用内置的检测脚本来可视化分析特定视频的检测效果：

```powershell
uv run python scripts/inspect_detection.py "path/to/video.mp4" --extract-frames --csv --open-dir
```

该脚本会生成：
- **片段列表**：展示最终判定的动态/静态段。
- **截帧图片**：在状态切换点自动截帧，方便人工核对。
- **逐帧 CSV**：记录每一帧的运动判定结果。

## 技术栈

- **语言**：Python 3.12
- **核心库**：NumPy, SQLite3
- **多媒体引擎**：FFmpeg (需要支持硬件加速)
- **环境管理**：uv
