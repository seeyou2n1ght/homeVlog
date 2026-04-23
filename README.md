# HomeVlog V2.2: 智能监控浓缩 Vlog 流水线 (性能狂飙版)

**HomeVlog** 是一个高性能、全国产化（架构逻辑）设计的家庭安防视频智能浓缩系统。它旨在将 24 小时不间断的 4K HEVC 监控录像（往往充满大量的静止或无意义的画面），自动化提炼成每日 1 分钟左右的“高光 Vlog”。

本项目专为解决“多设备异构计算”与“环境依赖地狱”而设计，采用了基于 `uv` 的极速隔离方案，并严格遵循 **硬件解码 + TensorRT张量计算 + 无损剪辑输出** 的红线。在 V2.2 版本中，更引入了全方位的性能榨汁外挂。

---

## 🌟 核心特性 (V2.2 Performance Architecture)

* **降维打击管道 (Pipe Downscaling)**：(V2.2新增) 摒弃了将 4K 原图压入管道的粗暴做法。在底层 FFmpeg 强制注入 `scale=640:640,fps=5` 滤镜，将 Python 管道的内存带宽压力瞬间暴降 98%，并让 VFR 漂移不攻自破。
* **纯血 Batch 推理 (Batched Inference)**：(V2.2新增) 在 Python 层维护推理缓冲池，攒够 `batch_size` 一次性喂给 TensorRT 引擎，彻底释放 GPU 张量核心的矩阵并行能力。
* **并发物理切割 (Concurrent I/O)**：(V2.2新增) 使用 `ThreadPoolExecutor` 搭配多线程调用 `-c copy`，彻底榨干 NVMe 固态硬盘的并发写入潜能。
* **无损输出 (Lossless Output)**：红线级约束。严格通过 `ffmpeg -c copy` 执行音视频流直接拷贝，保证 100% 原始画质、原始音频以及所有硬编码（Burnt-in）的时间水印。
* **精准防漂移 (VFR Anti-drift)**：绝不依赖帧序号。通过多线程解析 FFmpeg `showinfo` 提取绝对时间戳（PTS）。
* **轻量级去抖动 (Tracker Debounce)**：自研状态机，通过 `IoU` 计算和状态滑动窗口，抹平 YOLO 在复杂遮挡下产生的瞬间漏检，防止碎片化切割。
* **智能抽稀 (Pulse Thinning)**：在漫长的静止无人时段，以固定脉冲（如每分钟）采样 1 秒“心跳画面”。
* **SQLite 分析探针 (WAL DB)**：底层采用 SQLite 预写日志（WAL）模式，不仅用于幂等防重跑，还完整记录每次推理速度、检出事件（萌娃/宠物），为您未来的图表分析提供数据湖。

---

## 🛠️ 安装与部署 (Installation)

本项目全面抛弃传统的 `pip` 和 `conda`，采用新一代极速包管理器 **[uv](https://github.com/astral-sh/uv)** 以实现环境隔离。

### 1. 开发环境 (无 NVIDIA 显卡或纯 CPU)
如果您在轻薄本上开发，此模式会自动启用内置的 `MockBackend` 状态机，无需安装庞大的 CUDA 库。

```bash
# 1. 安装项目轻量依赖 (只包含 Numpy, OpenCV, Pydantic)
uv sync

# 2. 复制配置模板并修改路径 (可选)
cp config.example.yaml config.yaml

# 3. 运行冒烟测试
uv run python -m homevlog.main
```

### 2. 生产环境 (带 NVIDIA 显卡)
在您的服务器（如 i5-12600K + RTX 3060 Ti）上，您**不需要**在操作系统层面安装庞大复杂的 CUDA Toolkit。

```bash
# 1. 同步完整生产依赖 (一键拉取 NVIDIA 官方 PyPI Runtime 轮子)
uv sync --extra gpu

# 2. 修改配置: 将 config.yaml 中 hardware.use_mock 改为 false

# 3. 启动流水线
uv run python -m homevlog.main
```

---

## ⚙️ 架构全景图

```mermaid
graph TD
    NAS[NAS/SMB 目录] -->|双重采样防写锁定| Scanner[文件扫描器]
    
    subgraph 核心处理管道
        Scanner -->|HW QSV| Decoder[FFmpeg 解码 (带反压)]
        Decoder -->|队列 (FPS: 100+)| Detector[TensorRT FP16 / Mock]
        Detector -->|BBox & Confidence| Tracker[去抖动与追踪状态机]
        Tracker -->|Motion / Static| Aggregator[脉冲抽稀与区间合并]
    end

    subgraph 后期与统计
        Aggregator -->|-c copy 关键帧宽容| FFmpeg[高速无损切割合并]
        FFmpeg --> VLOG[本地浓缩 Vlog]
        
        DB[(SQLite WAL)]
        Tracker -.记录事件.-> DB
        FFmpeg -.记录性能.-> DB
    end
```

---

## 📝 开发守则

1. **绝对禁止污染**：开发期的测试数据、产生的日志与 `.mp4` 文件切勿提交至代码库（已在 `.gitignore` 配置）。
2. **严禁 Bypass 类型推导**：核心管道严格遵守 Python Type Hints。不要为了走捷径禁用 `Pydantic` 校验，它能在启动的第一秒挡住灾难性的配置错误。
3. **单向不可逆**：所有的业务状态流转必须由 `config.yaml` 驱动，不要在 Python 代码里散落 Hardcode。

*Built with ❤️ via Gemini CLI AI Pair Programming.*