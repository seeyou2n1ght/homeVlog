# HomeVlog 项目 AI 辅助开发上下文 (GEMINI.md)

## 项目概述
**HomeVlog (V2.4 异构并行版)** 是一款专为家庭安防设计的智能视频浓缩系统。其核心目标是将全天 24 小时的 4K HEVC 海量监控录像，自动提炼并合成为包含关键动态（如人、婴儿、猫、狗）的每日浓缩 Vlog。

项目基于**四级全局异步流水线** (Decoder → GPU → Logic → Cutter) 和**异构硬件分摊**设计：
- **核显 (QSV)**: 负责硬件视频解码与缩放 (`ffmpeg_hwaccel: "qsv"`)。
- **独显 (NVIDIA CUDA/TensorRT)**: 负责 YOLO 模型的高速并行推理 (`batch_size` 优化)。
- **CPU**: 负责目标追踪算法与复杂任务编排。

## 技术栈与依赖
- **语言/环境**: Python >= 3.12 (使用 **`uv`** 作为环境和依赖管理器)。
- **视觉与深度学习**: PyTorch (CU121), Torchvision, Ultralytics (YOLO), TensorRT, OpenCV (headless)。
- **持久化**: SQLite (WAL 模式)。
- **多媒体处理**: FFmpeg (需支持 QSV)。

## 目录结构
```text
homeVlog/
├── config.example.yaml          # 全局配置模板（需要复制为 config.yaml）
├── pyproject.toml               # uv 依赖配置文件
├── docs/                        # 深入了解系统原理的详细文档
│   ├── architecture.md          # 架构解析 (流水线与异构设计)
│   ├── development.md           # 开发规范
│   └── performance.md           # 性能与硬件调优
├── src/homevlog/                # 核心源代码目录
│   ├── main.py                  # 全局流水线编排入口
│   ├── config.py                # 配置加载与管理
│   ├── scanner.py               # 文件扫描策略
│   ├── hal/                     # 硬件抽象层 (TensorRT 后端、Mock 后端等)
│   ├── pipeline/                # 核心处理管道 (Decoder, Tracker, Aggregator 等)
│   ├── database/                # 数据库持久化层 (SQLite 交互)
│   └── utils/                   # 工具类 (FFmpeg 工具、时间处理等)
├── inputs/                      # 原始监控视频存放目录 (自动扫描)
├── outputs/                     # 最终生成的 Vlog 输出目录
└── models/                      # 模型文件目录 (如 TensorRT Engine)
```

## 开发与运行指令

```powershell
# 1. 环境安装与同步
uv sync

# 2. 运行系统 (需确保 config.yaml 存在并配置正确)
uv run python -m homevlog.main
```

## AI 编码规范与契约

1. **语言要求**:
   - 使用**中文**进行代码注释、思维链展示、方案设计和会话交互。
   - 技术术语和标准名词保留英文以确保精确。

2. **代码规范与质量**:
   - **严格类型安全**: 强制要求静态类型注解，非正当理由禁止使用 `any`。
   - **防御性编码**: 必须优雅地处理异常和重试（假设网络或外部 IO 随时会失败）。
   - **模块化**: 保持各个流水线模块的隔离与解耦，"小步快跑" 进行重构或迭代。
   - **未知 API**: 在调用不熟悉的库或系统 API 前，必须先阅读项目文档或官方文档，**禁止猜测参数**。

3. **规划优先原则 (Artifacts)**:
   - 在开始大段编码前，必须先生成 Markdown 格式的**中文设计方案 (Artifact)** 概述实现计划。
   - 评估方案是否违反 SOLID 原则，性能是否达标（特别是异步流水线的阻塞问题）。
   - 如果变更影响超过 3 个文件，必须等待用户明确审批后再行动。
   - 不要一味附和。如果发现用户的思路存在设计缺陷、性能瓶颈或安全风险，**必须暂停并提出反对意见**，同时给出优化后的替代方案。

4. **测试与 Git 规范**:
   - 永远通过日志输出或终端执行结果来判断代码是否生效，**不要靠推测**。
   - 执行测试前，优先执行静态检查或 Lint，拦截低级错误。
   - Git 提交严格遵循 **约定式提交 (Conventional Commits)**。
   - 单次提交需保证职责单一，严禁混杂不同类型的改动。
   - 除非用户明确要求，否则不要主动向远程仓库推送 (Push)。
