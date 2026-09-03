
# HomeVlog

HomeVlog 是一个用于家庭主机闲时批量处理室内监控素材的 DailyVlog 生成工具。它会扫描 NAS 或本地目录中的 H.265/MP4 监控录像，按日期和摄像头分组，检测运动片段，压缩静态片段，并输出按天合并的精简视频。

当前代码以 `StreamingOrchestrator` 为主流程：预筛选（Prescreen）、精细分析（Analysis）、批次渲染（Render）三个阶段重叠流式运行，并通过 SQLite 记录状态，支持增量防重、故障自愈与断点恢复。

## 适用硬件与性能极限

目标生产环境：

- **CPU**: Intel i5-12600K (16 线程全负载)
- **iGPU**: UHD 770 / Intel QSV (双 VDBox 高并发解码)
- **dGPU**: RTX 3060Ti / NVDEC + NVENC (CUDA 算力全开)
- **输入**: NAS/SMB 上的 4K H.265 素材 (支持高延迟网络环境)

**性能指标 (12600K + UHD 770 + RTX 3060Ti 实测)**：
- **全天监控处理耗时**: 24.37 小时 (87,730 秒) 监控素材仅需 **26 分 47 秒**（等效 **54.57x 极限实时倍速**）。
- **成片高保真极速浓缩**: 24 小时 4K 原始文件压缩为 **1.09 GB (1,118 MB)** DailyVlog，婴儿夜视微动与啼哭 100% 原速保真。
- **GPU 算力利用率**: 解码利用率稳定 ~95%，RTX 3060Ti 显存严格控制在 6.3GB 黄金安全水位，彻底消除 WDDM 换页减速。
- **自动化测试矩阵**: **100 / 100 核心业务用例通过（1 项环境依赖跳过），全量回归约 5 秒**。

## 核心架构与优化策略

### 1. 异构自适应工作窃取调度器 (Work-Stealing Scheduler)
- **Intel UHD 770 (QSV)**: 专职承担预筛（Prescreen）与主干分析（Analysis）硬件解码，充分利用双 Gen12 VDBox 吞吐。
- **NVIDIA RTX 3060Ti (NVENC/CUDA)**: 专职承担 YOLOv11 Tensor Core 张量批推理与双路 NVENC 硬件渲染。
- **超长切片分片并发解码 (Intra-File Chunking)**: 针对 5 分钟以上中长动态切片，自动拆分为 2~4 个独立分片并行 Seek 解码，消灭长尾木桶效应。
- **动态工作窃取 (`WorkStealingManager`)**: 队列积压超过高水位线时动态向 CUDA 租借 NVDEC 槽位；渲染启动时毫秒级原子让步（`RENDER_PREEMPTION_YIELD`），彻底根除硬件争用与死锁。

### 2. 小米摄像头多机位 MAC 自动别名映射 (Camera MAC & Alias)
- **物理目录智能解析 (`parse_camera_dir`)**: 自动解析 NAS 备份目录形如 `XiaomiCamera_01_B888805AA3CD` 中的唯一硬件 MAC 地址，根治文件名同名 `00_*.mp4` 冲突。
- **人类友好别名支持**: 配置文件支持将 MAC 映射为人类可读名称（如 `baby_room`），成片自动命名为 `DailyVlog_{date}_{camera}.mp4`，并在终端与日志中优雅透出。

### 3. 时空连通域抗噪与多模态感知 (Filters & VAD)
- **轻量选择性 EMA 滑动背景**: 双差分显著图融合，前景低速吸收、背景高速更新，精准捕获静坐等微动作。
- **8×8 空间连通域滤波 (`SpatialGridMotionFilter`)**: 动态跟踪 64 个网格单元底噪，8-邻域连通分量过滤孤立红外夜视噪点，聚类放大连续动作。
- **AudioEnergyVAD 声音事件唤醒**: 内存流 50ms 短时 RMS 包络与一阶自相关分析，交谈/啼哭等声音事件自动锁定 1x 原速原声。

### 4. 平滑变速过渡与时间码字幕 (Speed Ramping & OSD)
- **静态段关键帧抽取快路径 (Keyframe Fast-Path)**: 纯静态文件在解码侧以 `select` 按 `static_keyframe_interval` 抽帧，置于 scale/hwdownload 之前，仅关键帧进入缩放与显存回下载，消除静态段全帧解码瓶颈。
- **变速过渡曲线 (`calculate_speed_ramping_curve`)**: 静态片段计算 $C^1$ 连续平滑非线性过渡 PTS 曲线，消除跳帧顿挫感。
- **动作前后平滑缓冲 (Pre/Post-Roll)**: 动态动作前置扩展 1.0s，后置顺延 1.5s，完整保留动作起势与余波。
- **音频 afade 防爆音淡入淡出**: 动态段音轨自动进行 0.25s 线性双向交叉淡入淡出。
- **时间码 OSD 与字幕**: 支持硬字幕滤镜实时烧录真实墙上时间戳，或导出外挂 SRT / ASS 字幕文件；字幕墙钟映射与渲染滤镜图共用同一展示时长计划，并按 ramping 曲线精确逆映射还原。

### 5. 现代终端交互与三路日志体系
- **Rich 动态控制台终端 (`PipelineDashboard`)**: 实时呈现多阶段进度、积压队列深度、调度器硬件状态徽标与跑马灯告警。
- **三路分流日志**: 主运行日志 (`homevlog_*.log`)、独立异常日志 (`error_*.log`) 与结构化审计事件流 (`events_*.jsonl`)。

## 快速开始

安装依赖：

```powershell
uv sync
```

运行完整流程（默认开启 Rich 动态仪表盘）：

```powershell
uv run python main.py
```

CLI 常用选项：
```powershell
uv run python main.py --no-tui    # 禁用 Rich Live 仪表盘，使用标准文本输出（适合后台守护进程）
uv run python main.py --debug     # 开启调试级别日志输出
uv run python main.py --scan      # 扫描当前素材目录并展示机位与归档列表
uv run python main.py --date 20260901 --cam 0  # 仅处理指定日期与机位
```

执行全量自动化测试（100 个核心用例）：

```powershell
uv run pytest tests/
```


## 项目结构

```text
config/settings.yaml              系统核心参数配置文件
main.py                           CLI 运行入口
models/yolo11n.pt                 目标检测模型权重
src/
  ├── pipeline.py                 流式并发编排引擎 (StreamingOrchestrator)
  ├── scheduler.py                异构算力工作窃取调度器与硬件并发信号量
  ├── filters.py                  EMA 背景建模、8x8空间网格滤波抗噪与音频VAD
  ├── prescreen.py                Pass 1 关键帧跳跃极速粗筛
  ├── detector.py                 Pass 1.5 视频解码驱动与检测协调器
  ├── yolo_verifier.py            Pass 1.8 Tensor Core YOLOv11 动态批验证
  ├── segment.py                  动作片段聚类、平滑吸收与序列化
  ├── timeline.py                 平滑变速 PTS 曲线与 Filtergraph 滤镜图构建
  ├── renderer.py                 Pass 2 多批次并发硬件渲染与拼接
  ├── database.py                 SQLite WAL 任务状态持久化与并发管理
  ├── ui.py                       Rich 终端动态仪表盘与汇总卡片
  ├── monitor.py                  系统性能指标采样与 PerfRecord 采集
  ├── ffmpeg.py                   底层 FFmpeg 子进程执行封装
  └── utils.py                    配置加载、路径常量与三路日志体系
tests/                            9大业务领域核心测试套件 (100 用例，约 5s)
docs/                             系统架构设计、性能压测与测试基础设施文档

```

