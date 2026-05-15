# HomeVlog

HomeVlog 是一个用于家庭主机闲时批量处理室内监控素材的 DailyVlog 生成工具。它会扫描 NAS 或本地目录中的 H.265/MP4 监控录像，按日期和摄像头分组，检测运动片段，压缩静态片段，并输出按天合并的精简视频。

当前代码以 `StreamingOrchestrator` 为主流程：prescreen、analysis、render 三个阶段可以重叠运行，最终通过 SQLite 记录状态，支持重复运行和部分恢复。

## 适用硬件与性能极限

目标生产环境：

- **CPU**: Intel i5-12600K (16 线程全负载)
- **iGPU**: UHD 770 / Intel QSV (双 VDBox 高并发解码)
- **dGPU**: RTX 3060Ti / NVDEC + NVENC (CUDA 算力全开)
- **输入**: NAS/SMB 上的 4K H.265 素材 (支持高延迟网络环境)

**性能指标 (12600K + 3060Ti 极限配置)**：
- **GPU 解码利用率**: ~88% (NVDEC 接近饱和)
- **处理速度**: 24 小时高清监控素材处理约需 **8-10 分钟**。
- **扫描响应**: 秒级启动（得益于 Lazy Metadata 缓存策略）。

## 核心优化策略

### 1. 延迟元数据探测 (Lazy Metadata Probing)
针对 NAS 环境设计的极致优化。系统不再在启动时全局扫描数千个文件的元数据，而是：
- 在 **Analysis 阶段**（文件首次打开解码时）通过 PyAV 自动提取音频流和编码参数。
- 提取后的元数据自动持久化到 SQLite 数据库。
- **渲染阶段**直接从本地数据库读取，彻底消除了网络超时导致的“假死”和渲染崩溃。

### 2. YOLO 批量推理 (Batch Inference)
- **吞吐量提升**: 弃用逐帧推理模式，改用 **Segment-level Batch 推理**。
- **GPU 加速**: 充分利用显卡张量核心，将一组帧一次性送入 CUDA 推理，速度提升 300% 以上。

### 3. 单次解码流水线 (Single-Pass Pipeline)
- 彻底废弃子进程 `subprocess.Popen("ffmpeg")` 模式。
- 使用 **PyAV (FFmpeg C API)** 原生硬件解码，解码帧在内存中以 NumPy 数组形式存在。
- **数据流向**：
    1.  **解码帧** → **NumPy 向量化运动分析** (uint8 空间计算帧差能量)。
    2.  **关键帧采样** → **驻留内存字典** (frames_buffer)。
    3.  **YOLO 验证器** → 从字典读取帧进行 **Batch 推理** (Zero-IO)。
- 极大地减少了 CPU 负载、PCIe 带宽占用以及内存拷贝开销。

### 4. 数据库与持久化 (Persistence)
- **SQLite 核心作用**：不仅记录任务状态，还作为元数据缓存中心。
- **自动迁移**：系统启动时会自动检测并补全 `has_audio` 等字段，确保版本平滑升级。
- **断点续传**：基于数据库状态，支持随时中断并从上次进度恢复，且不会重复探测已完成的文件。

### 4. 硬件自适应调度 (Smart Scheduling)
- **算力最大化**: 系统根据 `max_nv_concurrency` 智能限制分析 Worker，为 NVENC 渲染预留空间，并自动切换核显 (QSV) 处理剩余分析任务。
- **自适应 FPS**: 针对长视频动态降低分析采样率（如 1fps），在不影响检测率的前提下减少 60%+ 的解码压力。

## 快速开始

安装依赖（自动配置 CUDA 环境）：

```powershell
uv sync
```

运行完整流程：

```powershell
uv run python main.py
```

处理指定日期和摄像头：

```powershell
uv run python main.py --date 20260320 --cam 0
```

## 关键配置 (`config/settings.yaml`)

- `hardware.max_nv_concurrency`: 限制 NVENC 并发（通常为 3）。
- `hardware.max_qsv_concurrency`: 压榨 UHD 770 性能，建议设为 6-8。
- `detection.analysis_max_workers`: 建议设为 8-10 以充分利用多核 CPU。
- `detection.analysis_early_term_enabled`: 开启早停逻辑，检测到静止画面后立即终止解码。

## 故障排除与安全回退

如果遇到硬件冲突，可调低 `analysis_max_workers` 或切回 CPU 模式：

```yaml
hardware:
  device: "cpu"
detection:
  analysis_max_workers: 2
```

## 项目结构

```text
config/settings.yaml              极限性能配置文件
main.py                           CLI 入口
src/pipeline.py                   流式管线编排与 Worker 管理
src/detector.py                   向量化运动检测与早停逻辑
src/yolo_verifier.py              Batch 模式 YOLO 验证器
src/database.py                   元数据缓存与状态持久化
src/timeline.py                   跨文件时间轴构建 (修复了早停间隙)
src/renderer.py                   多 batch 硬件并行渲染
```

## 注意事项

- **环境要求**: 请确保使用 `uv run` 执行，以确保加载了正确的 CUDA 版 PyTorch。
- **早停逻辑**: 修复了之前版本早停导致的时间轴空洞 Bug，现在强制映射至文件末尾。
- **磁盘空间**: 渲染前需预留 20GB 以上可用空间。
