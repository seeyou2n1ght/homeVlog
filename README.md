# HomeVlog

HomeVlog 是一个高度自动化的家庭监控视频浓缩系统。它能够智能识别原始录像中的运动片段，利用异构计算资源（NVIDIA CUDA + Intel QSV）并行处理，将海量原始素材合并浓缩为紧凑、流畅的每日 Vlog。

## 🚀 核心架构：三级级联流水线

项目采用级联式（Cascade）架构，通过分阶段过滤极大提升了处理吞吐量：

### 1. Pass 1: 预筛 (Prescreening)
- **目标**：以极低开销（约 200x 速度）过滤掉 80% 以上无动作的文件。
- **技术**：基于 FFmpeg 的快速抽帧比对。
- **特性**：
  - **环境光自适应阈值**：针对红外夜视自动下调检测阈值。
  - **首帧比对**：通过与前一文件末帧比对，消除文件衔接处的盲区。

### 2. Pass 1.5: 分析与验证 (Deep Analysis)
- **目标**：对可疑文件进行像素级运动检测与 AI 目标过滤。
- **技术**：
  - **Robust Noise Floor Estimation**：基于分位数（P5-P20）的鲁棒底噪估计，有效对抗树叶晃动、光影闪烁。
  - **YOLO 级联验证**：对识别出的“运动段”引入 YOLO (YOLO11n) 进行目标检测，仅保留包含人、车、动物等目标的片段。
  - **多 GPU 抢占式调度**：NVDEC (NVIDIA) 与 QSV (Intel) 异构 Worker 同时从任务队列抢占任务，最大化硬件利用率。

### 3. Pass 2: 渲染与生成 (Smart Rendering)
- **目标**：生成最终视频。
- **技术**：
  - **动态时钟映射**：消除变帧率（VFR）导致的音画不同步。
  - **分段渲染 (Batch Render)**：将超长视频拆分为批次并行渲染，降低 VRAM 压力。
  - **异构渲染**：支持多个 GPU 分别渲染不同的批次，最后无损合并。

---

## 🛠️ 性能与健壮性优化 (2026-05-11 更新)

针对大规模处理场景，系统已完成深度重构：
- **资源解耦**：将全局 IO 锁拆分为 `disk_io_sem`、`nv_render_sem` 和 `qsv_render_sem`，实现真正的计算并行。
- **数据库 WAL 模式**：开启 SQLite 预写日志 (Write-Ahead Logging)，解决多线程写入冲突与锁死。
- **内存安全**：YOLO 模型采用线程安全单例模式，避免显存溢出 (OOM)；子进程文件描述符显式关闭，防止句柄泄露。
- **可视化进度**：集成 `tqdm` 进度管理，通过三个独立进度条实时展示预筛、分析、渲染进度。

---

## 📦 技术栈

- **Runtime**: Python 3.12 (managed by [uv](https://github.com/astral-sh/uv))
- **Engine**: FFmpeg (HWAccel Required)
- **Compute**: NumPy, PyTorch (YOLO11), OpenCV
- **Storage**: SQLite3 (WAL Mode)

---

## 🚦 快速开始

### 环境安装
```powershell
uv sync
uv add tqdm  # 确保进度条组件已安装
```

### 核心配置 (`config/settings.yaml`)
- `paths.input_dir`: 监控录像原始路径 (支持 NAS 挂载路径)。
- `detection.roi_crop`: [x, y, w, h] 比例，用于规避屏幕时间戳干扰。
- `yolo.enabled`: 开启 AI 目标过滤，仅生成有意义的动态片段。
- `output.batch_max_files`: 调整单次渲染的文件数，以适应显存大小。

### 启动运行
```powershell
uv run main.py
```

---

## 🔍 调试与检查工具

项目提供 `inspect_detection.py` 脚本用于深度调优：
```powershell
uv run python scripts/inspect_detection.py "path/to/video.mp4" --extract-frames --csv --open-dir
```
- **可视化验证**：在动作起始点自动截取 PNG 图片。
- **数据分析**：输出逐帧运动能量分布 CSV，辅助调整敏感度。

---

## 📂 项目结构

```text
├── config/             # 配置文件 (YAML)
├── src/                # 核心逻辑
│   ├── pipeline.py     # 并发流水线编排
│   ├── detector.py     # 运动检测算法
│   ├── yolo_verifier.py # YOLO AI 验证
│   ├── renderer.py     # 硬件加速渲染
│   └── database.py     # 任务状态持久化
├── scripts/            # 调试脚本
└── data/               # SQLite 数据库
```
