# HomeVlog V2.4: 智能监控浓缩 Vlog 流水线 (异构高性能版)

**HomeVlog** 是一款专为家庭安防设计的智能视频浓缩系统。它能将全天 24 小时的 4K 海量监控录像，自动提炼并合成为包含关键动态的每日浓缩 Vlog。

V2.4 版本引入了 **全局异步流水线** 与 **异构硬件平衡** 技术，实现了解码、推理与逻辑分析的深度重叠。

---

## 🚀 核心特性

- **四级全局流水线 (V2.4)**：Decoder → GPU → Logic → Cutter 四层解耦。当 A 文件在切割时，B 文件已经在推理，C 文件在预扫。
- **自适应微动捕获 (New)**：引入基于动态背景建模的纯 CPU 帧差算法，与 YOLO 语义融合，完美捕获夜视红外噪点下的婴儿极缓微动。
- **异构硬件分摊 (The "Sweet Spot")**：
  - **核显 (QSV)**: 100% 负责解码与缩放。
  - **独显 (NVIDIA)**: 100% 负责 TensorRT 高速推理。
  - **CPU**: 负责目标追踪与任务编排。
- **GPU 深度优化**：手写向量化预处理 + 纯 GPU NMS，吞吐量高达 **120-150 FPS**。
- **VFR 精确同步**：基于硬件 PTS 预扫描，彻底解决监控视频变帧率导致的剪辑偏移问题。
- **健壮性设计**：内置资源自动清理钩子，完美应对 Windows 下的任务中断与异常。

---

## 🛠️ 快速开始

本项目使用 **[uv](https://github.com/astral-sh/uv)** 管理环境。

```powershell
# 1. 环境初始化
uv sync

# 2. 配置修改 (设置 QSV/CUDA 及路径)
cp config.example.yaml config.yaml

# 3. 运行全量流水线
# 会自动清理旧数据并重新扫描 inputs/ 目录
uv run python -m homevlog.main
```

---

## 📂 技术文档 (Deep Dive)

为了深入了解系统原理，请参考 `docs/` 目录：

- [🌊 架构解析 - Architecture](./docs/architecture.md): 深入理解异步流水线与异构设计。
- [🛠️ 开发规范 - Development](./docs/development.md): 编码守则、模块职责与 Git 提交规范。
- [📊 性能 Benchmark - Performance](./docs/performance.md): 硬件占用分析与性能调优建议。

---

## 📂 项目结构

```text
homeVlog/
├── docs/                        # 技术文档
├── src/homevlog/
│   ├── hal/                     # 硬件抽象层 (TensorRT)
│   ├── pipeline/                # 核心管道 (Decoder, Tracker, Aggregator)
│   ├── database/                # 持久化层 (SQLite WAL)
│   └── main.py                  # 全局流水线编排入口
├── config.yaml                  # 配置文件
└── pyproject.toml               # 依赖管理
```

---

## 📝 开源许可
MIT License.