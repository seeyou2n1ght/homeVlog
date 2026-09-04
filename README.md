# HomeVlog

HomeVlog 是专为家庭 NAS 与本地主机打造的室内监控智能浓缩工具。它可以自动扫描并处理全天的 4K H.265/MP4 监控视频，利用硬件加速过滤纯静止与弥散光影时段，精细保留人物/宠物的动态与家庭声音，最终生成平滑过渡的家庭 DailyVlog 成片。

系统基于 Intel 核显 (QSV) 与 NVIDIA 独显 (CUDA/NVENC) 异构协同，配合 Active Learning 质检审核平台，支持秒级时间轴修正与专属机位模型自进化。

---

## 硬件表现与实测战报

测试平台：Intel i5-12600K + Intel UHD 770 (核显) + NVIDIA RTX 3060Ti (8GB 独显)

- **全天监控处理耗时**: 24 小时 (89,500+ 秒) 4K 监控素材平均处理耗时 **16.5 分钟** (实测 13.4 ~ 22.7 分钟)，等效处理倍速 **81x ~ 110x**。
- **高压缩率与保真度**: 24 小时 40GB~60GB 监控录像浓缩为 **2.3GB ~ 3.8GB** 成片；有效动态与对话事件 100% 原速原声保留，静态段 30x~60x 平滑快进。
- **高稳态自动化测试**: 125 项全量核心测试用例自动化回归约 9 秒全绿。

---

## 快速上手

### 1. 安装依赖

本项目使用 uv 管理虚拟环境与依赖（已预置 PyTorch CUDA 12.4 支持）：

`powershell
uv sync
`

### 2. 运行主流水线

默认启动流式浓缩流水线，并展示 Rich 终端动态仪表盘：

`powershell
uv run python main.py
`

常用命令行参数：
`powershell
# 仅扫描素材目录并输出机位与日期清单，不执行分析渲染
uv run python main.py --scan

# 指定处理特定日期与机位 (camera_id 支持 MAC 后缀或别名)
uv run python main.py --date 20260901 --cam 0

# 无 TUI 终端模式 (适合作为后台 Windows 服务或无头任务运行)
uv run python main.py --no-tui

# 开启调试详细日志
uv run python main.py --debug
`

---

## 质检审核与秒级重浓缩平台

系统自带独立的 Web 端人工质检审核工作台，用于复核识别结果、挖掘难样本并实现零重复解码的秒级重新渲染。

### 启动审核平台
`powershell
uv run python scripts/audit_tool/app.py
`
- 服务默认启动在 http://127.0.0.1:8765 并自动打开浏览器。
- 可选参数：--port 8888，--no-browser。

### 平台主要功能
1. **疑难样本排查 (Active Learning)**：自动筛选并置顶疑似光影误报段（动态但无目标）与疑似微动漏判段（能量临界区）。
2. **快捷键一键标注**：
   - 1: 标记为确认有效动态 (TP)
   - 2: 标记为误判假动态 (FP - 窗帘/树影/光斑)
   - 3: 标记为漏判微动 (FN - 实际有人)
   - 4: 标记为确认纯静止 (TN)
   - J / K (或上下方向键): 快速切换前后切片
3. **真实反馈帧物理归档**：标注时自动从 4K 原片抽取变动瞬间原图与差分图，写入 data/archives/ 目录与 manifest.jsonl，用于算法优化。
4. **秒级即时重浓缩**：在界面直接点击「即时重浓缩成片」，系统复用数据库已分析的时间轴与人工修正结果，直接调用 Pass 2 NVENC 硬件渲染，数十秒内生成修正版成片。

---

## 专属机位模型微调与调参闭环

针对特定机位特有的复杂光影、晃动窗帘或特殊视角，系统支持全流程无代码自进化闭环：

### 1. 导出机位标注数据集
将人工审核打标的历史切片与归档原图导出为标准 YOLO 数据集（包含 8:2 划分与负样本支持）：
`powershell
uv run python scripts/export_dataset.py --output data/datasets/my_cam --camera B888805AA3CD
`

### 2. 本地微调 YOLOv11 权重
使用本地 RTX 3060Ti 对骨干网络执行冻结微调（reeze=10, AMP 混合精度），训练机位专属权重：
`powershell
# 训练 15 个 epoch，并在训练完成后直接热替换更新当前系统的模型
uv run python scripts/train_yolo.py --data data/datasets/my_cam/data.yaml --epochs 15 --apply
`

### 3. 预筛选超参数寻优
根据标注结果自动通过网格搜索调优当前机位的最佳时空预筛选阈值：
`powershell
uv run python scripts/tune_thresholds.py --camera B888805AA3CD
`

---

## 核心文档导航

- **[AGENTS.md](file:///c:/Users/seeyo/code/homeVlog/AGENTS.md)**：AI Agent 开发守则、硬件信号量单次原则、优雅停机与时间轴闭环铁律。
- **[docs/ARCHITECTURE.md](file:///c:/Users/seeyo/code/homeVlog/docs/ARCHITECTURE.md)**：系统三级流式流水线拓扑、异构工作窃取调度器、EMA/VAD 动静识别算子与 SQLite 表结构设计。
- **[docs/BENCHMARK.md](file:///c:/Users/seeyo/code/homeVlog/docs/BENCHMARK.md)**：连续 5 天 120 小时生产环境实测数据、各演进阶段性能瓶颈定位与 RCA 记录。

---

## 项目结构

`	ext
config/settings.yaml              系统核心参数配置
main.py                           主流水线 CLI 入口
models/                           目标检测模型权重仓库 (内置 yolo11n / yolo11s / yolo11m)
data/
  ├── vlog.db                     SQLite WAL 任务状态与切片标记数据库
  └── archives/                   真实人工反馈原图归档仓库与索引清单
src/
  ├── pipeline.py                 流式并发编排引擎 (StreamingOrchestrator)
  ├── scheduler.py                异构硬件调度器与并发信号量管理
  ├── prescreen.py                Pass 1 关键帧跳跃粗筛 (自适应空间集中度算子)
  ├── detector.py                 Pass 1.5 解码驱动与 EMA/连通域检测
  ├── filters.py                  EMA 背景建模、8x8 连通域抗噪与 AudioEnergyVAD
  ├── yolo_verifier.py            Pass 1.8 Tensor Core YOLO 动态批验证
  ├── timeline.py                 平滑变速 PTS 曲线、重浓缩时间轴修正与滤镜构建
  ├── renderer.py                 Pass 2 多批次并发硬件渲染与拼接
  ├── archiver.py                 真实反馈帧原图抽取与原子归档模块
  ├── database.py                 SQLite WAL 任务管理与 segments 切片持久化
  ├── ui.py                       Rich 终端动态仪表盘
  └── ffmpeg.py                   FFmpeg 子进程封装与生命周期托管注册表
scripts/
  ├── audit_tool/                 人机协同 Web 二次审核与秒级重浓缩平台
  ├── export_dataset.py           审核样本与归档帧导出 YOLO 数据集工具
  ├── train_yolo.py               专属机位 YOLOv11 本地轻量微调工具
  └── tune_thresholds.py          预筛选超参数网格搜索寻优脚本
tests/                            125 项自动化业务测试套件
docs/                             系统架构设计与性能基准测试文档
`
