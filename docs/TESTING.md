# HomeVlog 测试体系与自动化验证规范

本文档为 HomeVlog 智能监控浓缩系统的权威测试指南，涵盖测试架构、领域划分、执行指令与质量验收基准。

---

## 一、 测试架构与设计方法学

测试套件采用**高内聚业务领域驱动**架构，彻底摒弃了碎片化的里程碑用例，划分为 8 个核心业务测试模块，共收敛为 **87 个高价值测试用例（86 passed + 1 skipped）**，全套回归耗时约 **4.4 秒**。

### 1. 测试方法学
- **等价类划分与格式覆盖**：覆盖 H.264、H.265 (HEVC)、单声道/双声道 AAC 与完全静音音轨的自适应探测。
- **极限边界值校验**：
  - 片段极短时长（0.1s）与极长视频（3600.0s）的时间轴闭环；
  - 跨日边界：覆盖 `00:00:00` 起点与 `23:59:59` 终点的绝对对齐；
  - 磁盘保护：预警空间强制阈值（20.0 GB）。
- **并发与竞态模拟**：
  - 异构信号量隔离（NVENC: 2、QSV: 8、Disk I/O: 8）的争用与释放；
  - 多线程与工作窃取高低水位线动态切换的原子性。
- **零外部硬件依赖 (Mock & Decoupling)**：
  - 借助合成正弦波音频测试 AudioEnergyVAD；
  - 借助纯内存合成图像阵列测试 8x8 网格空间抗噪与连通域聚类；
  - 使得所有测试均可在无独立显卡的 CI 容器环境中快速执行。

---

## 二、 业务领域测试模块映射表

| 测试模块文件 | 包含用例数 | 覆盖业务范围与关键断言 |
| :--- | :--- | :--- |
| [`tests/test_database_and_scanner.py`](../tests/test_database_and_scanner.py) | 9 passed | SQLite WAL 读写解耦、信号量超时重试、NAS 小米摄像头 MAC 解析与目录提取、时间戳解析、多目录扫描 |
| [`tests/test_motion_and_vad.py`](../tests/test_motion_and_vad.py) | 14 passed | `EmaBackgroundModel` 双差分、`SpatialGridMotionFilter` 8x8 空间连通域、暗光婴儿微动敏感度、IR-Cut 全局闪光抑制、`AudioEnergyVAD` 50ms RMS 分帧 |
| [`tests/test_timeline_and_ramping.py`](../tests/test_timeline_and_ramping.py) | 39 passed | $C^1$ 连续平滑非线性变速过渡 PTS 曲线、动态段 1.0s/1.5s 前后延展、外挂 SRT/ASS 与硬字幕滤镜 |
| [`tests/test_scheduler_and_hardware.py`](../tests/test_scheduler_and_hardware.py) | 7 passed | `WorkStealingManager` 三态流转（NORMAL / BURST / PREEMPTION）、硬件信号量争用与重置 |
| [`tests/test_renderer_and_ffmpeg.py`](../tests/test_renderer_and_ffmpeg.py) | 6 passed | 2路 NVENC 满载并发渲染调度 (防 8GB 显存溢出换页)、批次断点秒级复用、Filtergraph 指令构建、防死锁重定向 |
| [`tests/test_pipeline_streaming.py`](../tests/test_pipeline_streaming.py) | 4 passed | 流式多阶段重叠编排引擎、物理时序保序滑动窗口分发（抗乱序完成）、队列积压溢出反压保护 |



| [`tests/test_ui_logging.py`](../tests/test_ui_logging.py) | 5 passed | Rich 终端 Live 仪表盘刷新、三路日志分流（运行/错误/审计流）、无 TUI 模式降级 |
| [`tests/test_acceptance_e2e.py`](../tests/test_acceptance_e2e.py) | 2 passed (1 skipped) | 端到端全天全切片真实集成浓缩流程、成片可读性与元数据校验 |

---

## 三、 测试运行指南

### 1. 全量回归测试
```powershell
uv run pytest tests/
```

### 2. 按业务领域单模块快速调试
```powershell
# 运行运动滤波与音频检测测试
uv run pytest tests/test_motion_and_vad.py

# 运行时间轴变速与平滑测试
uv run pytest tests/test_timeline_and_ramping.py

# 运行异构调度器与硬件并发测试
uv run pytest tests/test_scheduler_and_hardware.py

# 运行终端交互与日志体系测试
uv run pytest tests/test_ui_logging.py
```

### 3. 代码覆盖率报告生成
```powershell
uv run pytest --cov=src tests/ --cov-report=term-missing
```

---

## 四、 持续集成与准入基线

1. **零破坏性回归 (Zero-Regression)**：任何新增功能或重构提交，必须保证上述 **86 项核心用例 100% 绿灯**（1 项 E2E 用例按硬件条件跳过）。
2. **执行时效基线**：全套单元与集成测试在现代 8 核心 CPU 上的运行时间不得超过 **5 秒**。
3. **代码静态检查**：代码在提交前必须通过 `python -m compileall main.py src` 语法校验。
