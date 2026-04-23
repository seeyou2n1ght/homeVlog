# HomeVlog 项目 AI Coding 开发对齐指南 (V2.2 性能狂飙版)

本指南为 HomeVlog 自动化浓缩流水线的唯一参考标准。V2.2 性能狂飙版在 V2.1 的极简解耦管道的基础上，引入了降维解析管道、Batch推理缓冲池和多线程并发 I/O 切割三大核心外挂，进一步榨干了异构计算硬件的潜能。

---

## 1. 项目定义与目标 (Project Scope)

- **输入**：4K HEVC 监控素材（20fps, 3Mbps, 128MB/文件，存放于 NAS/SMB）。
- **目标**：通过 YOLO 语义检测识别目标，结合追踪去抖动与静止抽稀算法，快速产出每日浓缩 Vlog。
- **硬件基准**：Intel i5-12600K (QSV) + NVIDIA RTX 3060 Ti (8GB VRAM)。

---

## 2. 核心机制与性能外挂 (Architecture & Performance Hacks)

### 2.1 降维打击管道 (Pipe Downscaling)
- **挑战**：4K RGB24 数据流极大消耗 Python 内存总线与 `queue` 同步时间。
- **对策**：彻底摒弃在 Python 层接收 4K 原图。在 `Decoder` 底层强制注入 FFmpeg `scale=640:640,fps=5` 滤镜。数据量瞬间缩减 98%，同时完美规避 VFR 变帧率导致的 PTS 漂移难题。

### 2.2 纯血 Batch 推理 (Batched Inference)
- **挑战**：单帧 `for` 循环推理无法有效填饱 TensorRT 张量核心，GPU 处于饥饿状态。
- **对策**：在主循环中引入 `batch_frames` 缓冲池，攒够 `batch_size` (如 16) 后一次性送入 TensorRT 引擎，吞吐量成倍暴增。

### 2.3 并发物理切割 (Concurrent I/O Cutting)
- **挑战**：数十个碎片切片串行切割时，CPU 绝大部分时间在阻塞等待磁盘同步。
- **对策**：引入 `ThreadPoolExecutor` (并发度由 `parallel_jobs` 决定)，并发发起多个 `-c copy` 子进程，利用多线程压榨 NVMe SSD 的极致 IOPS。

### 2.4 SMB 写入锁定与探测 (Active Write Detection)
直接读取内网 NAS (SMB) 时，必须防止读取到摄像头正在持续写入的“热文件”。
- **策略**：**多重采样校验**。在 `Scanner` 阶段，针对最新发现的文件，记录其最后修改时间（mtime）和文件大小（size）。间隔 N 秒（由配置决定，如 5 秒）后再次采样，只有当 `mtime` 和 `size` 均无变化时，才标记为“锁定完成”并推入处理队列。

### 2.2 视频输出的“三保”红线 (Audio & Watermark Preservation)
- **保留音频**：家庭录像的音频包含重要情绪价值，必须保留。
- **保留时间水印**：大部分监控摄像头已在原始视频流中硬编码（Burnt-in）了时间戳。
- **强制指令**：为满足“三保”且不损失速度，`Cutter` 阶段严禁重编码（Re-encode），必须且只能使用 `ffmpeg -c copy`（全流拷贝）。该指令天然会无损保留源文件的所有音频轨和硬编码水印。

### 2.3 脉冲抽稀与去抖动逻辑 (Configurable Thinning)
单纯依靠单帧 IoU 会引发极度碎片化，必须引入：
1. **轻量级追踪 (Tracker)**：为目标分配 Track ID，应对遮挡。
2. **状态去抖 (Debounce Window)**：状态切换必须持续超过指定帧数才生效。
3. **脉冲抽稀 (Pulse Thinning)**：静止期内，按照 `static_interval_ms` 设定的周期（如 60s），仅保留 1 个单位的抽帧视频。

---

## 3. 环境隔离与依赖契约 (UV Environment Isolation)

为了彻底摆脱系统级库的“环境地狱”，项目严格规定使用 `uv` 进行极速构建，并利用 NVIDIA 官方 PyPI 轮子实现项目级的 CUDA 隔离（仅依赖宿主机显卡驱动）。

### pyproject.toml 核心规范：
必须显式声明依赖 NVIDIA 官方的运行时与 TensorRT 包，确保环境在任何只要有 NVIDIA Driver 的机器上都能通过 `uv sync` 一键拉起。

```toml
[project]
name = "homevlog"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    # 核心张量与视觉库
    "torch>=2.0",
    "numpy",
    "opencv-python-headless", # 服务器环境无需 GUI
    
    # 彻底隔离的 GPU 运行时 (通过 PyPI 而非系统级安装)
    "tensorrt",
    "nvidia-cudnn-cu12",
    "nvidia-cublas-cu12",
    
    # 工具链
    "pyyaml",
    "tqdm",
]
```

---

## 4. 全局配置化管理 (Configuration)

所有涉及阈值、时间、路径和硬件调度的变量，禁止硬编码（Hardcode），必须通过统一的 `config.yaml` 驱动：

```yaml
# 1. 环境与路径
paths:
  nas_input: "\\\\NAS\\surveillance"
  local_output: "D:\\vlogs"
  sqlite_db: "./data/homevlog.db"

# 2. SMB 探测策略
scanner:
  write_detect_interval_sec: 5      # 两次采样的间隔时间
  write_detect_stable_count: 2      # 连续几次采样不变视为写入完成

# 3. 动态帧检测与抽稀参数
detection:
  model_path: "./models/yolov11s.trt"
  classes: ["person", "baby", "cat", "dog"]
  confidence_threshold: 0.5
  iou_threshold: 0.80               # 追踪判定的同源目标 IoU 阈值
  debounce_frames: 5                # 状态平滑窗口（防抖动所需帧数）
  static_interval_ms: 60000         # 静止期脉冲采样间隔 (毫秒)
  
# 4. 视频切割参数
cutting:
  keyframe_padding_ms: 2500         # -c copy 的关键帧宽容度补偿 (前后各拓宽时间)

# 5. 硬件与性能
hardware:
  ffmpeg_hwaccel: "qsv"             # 硬件解码器
  batch_size: 16                    # TensorRT 推理批次
```

---

## 5. SQLite 数据分析层 (Analytics & Observability)

引入 SQLite 不仅是为了幂等性（防重复运行）和断点恢复，更是为了支撑后续的**用户数据分析**和**系统性能大盘**。必须采用 `WAL` 模式确保高并发。

核心表结构设计要求：

### 表 1: `processed_files` (业务流水表)
记录素材处理状态，用于断点恢复和幂等验证。
- 关键字段：`source_path`, `hash`, `start_time`, `end_time`, `status`。

### 表 2: `analytics_events` (用户数据分析)
记录视频中发生的实际业务事件，便于后续生成“一周萌娃图表”等高维特征。
- 关键字段：`date`, `class_name` (人/猫/狗), `motion_duration_ms` (运动时长), `static_duration_ms` (静止时长)。

### 表 3: `performance_metrics` (系统性能探针)
用于诊断 I/O 瓶颈和 GPU 利用率。
- 关键字段：`file_name`, `decode_fps`, `infer_fps`, `total_process_time_ms`, `gpu_mem_peak_mb`。

---

## 6. 数据流与模块管道 (Data Flow)

```mermaid
graph TD
    NAS[NAS SMB 目录] -->|mtime/size 双重采样过滤| Decoder
    
    subgraph Pipeline["处理管道 (Worker)"]
        Decoder[FFmpeg QSV 解码] -->|Stdout Pipe (队列满则阻塞)| ReaderThread[ Daemon 队列消费]
        ReaderThread -->|Bounded Queue| Detector[TensorRT FP16 检测]
        Detector --> Tracker[追踪与去抖动平滑]
        Tracker --> Aggregator[脉冲抽稀聚合]
    end

    subgraph Output["无损落盘与统计"]
        Aggregator -->|-c copy 关键帧 Padding| Cutter[高速无损切割]
        Cutter --> Merger[每日 Vlog 合成]
        Merger --> SSD_Out[本地输出]
    end

    SQLite[(SQLite WAL)] -.记录耗时、事件、状态.- Pipeline
```

---

## 7. 盲点防范与防御性编程 (Defensive Programming & Blind Spots)

在实际开发与部署中，必须对以下 5 个极易忽视的系统级/业务级盲点进行防御性处理，这被视为本项目的 P0 级红线：

### 7.1 变帧率 (VFR) 与时间轴漂移
- **隐患**：根据实际探测元数据 (`avg_frame_rate=58860000/2987171` 等异常小数)，监控视频通常为变帧率 (VFR)。通过“帧序号/平均帧率”计算出的时间轴会产生严重漂移。
- **对策**：**绝对禁止使用帧序号计算时间**。在解码管道提取帧时，必须让 FFmpeg 同时输出该帧的**绝对 PTS (Presentation Time Stamp)**，后续所有的逻辑判断、抽稀、裁剪，全部精准锚定 PTS 进行计算。

### 7.2 Padding 导致的区间重叠 (Overlapping Segments)
- **隐患**：由于 Cutter 阶段必须依赖 `ffmpeg -c copy` 和 `keyframe_padding_ms`（例如前后增加 2.5 秒的补偿），时间较近的两个独立事件，其 Padding 极易发生时间轴重叠，导致同一画面被重复切割合并。
- **对策**：在 `Aggregator` 生成最终切片任务前，必须执行**“区间合并 (Merge Overlapping Intervals)”**操作。当发现相邻的两个区间在加上 Padding 后产生交集时，直接将它们融合成一个连续的长区间。

### 7.3 夜视红外模式 (IR Mode) 的识别率下降
- **隐患**：YOLOv11s 等 COCO 预训练模型在面对缺乏色彩特征的高噪点黑白红外画面时，检出率 (Recall) 可能会下降。
- **对策**：架构层面上暂不干预，接受夜间素材的合理漏检。预留数据上报通道，收集红外失效的负样本，在 V1.1+ 中通过对模型进行微调 (Fine-tuning) 来解决，保持整体代码架构的封闭与稳定。

### 7.4 NAS 网络死锁与僵尸进程 (Zombie Process Deadlock)
- **隐患**：SMB 直连受网络波动影响，若 FFmpeg 由于 I/O 挂起而死锁，将导致 Python 读取管道陷入永久阻塞 (Deadlock)。
- **对策**：引入**独立的心跳看门狗 (Watchdog)**。设定 `io_timeout_sec`（如 30 秒）。如果在此时长内未能从 FFmpeg `stdout` 读取到任何新帧，必须通过 `SIGKILL` 强杀该子进程，并在 SQLite 记录异常后，拉起重试机制。

### 7.5 开发期 Mock 数据的过度静态化
- **隐患**：为了解决开发机（核显笔记本）无 GPU 的问题，如果 `MockBackend` 仅返回死板的静态坐标，将导致状态机（追踪、抽稀、判定切换）无法在开发期得到真实流转测试。
- **对策**：`MockBackend` 必须内置一套简易的**“事件发生器状态机”**。动态生成“目标移动 (IoU 变化)” -> “停滞 (IoU=1.0)” -> “消失”的模拟生命周期数据，确保即便在核显轻薄本上，依然能端到端跑通复杂的时间序列切分逻辑。