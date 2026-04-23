# HomeVlog 架构解析

HomeVlog V2.4 采用 **全局异步流水线 (Global Asynchronous Pipeline)** 与 **异构硬件加速 (Heterogeneous Hardware Acceleration)** 设计，旨在实现 4K 监控视频的极致浓缩效率。

## 1. 核心设计哲学

### 1.1 异构资源物理隔离
为了消除硬件资源争用（Resource Contention），我们将不同的任务分配给最适合的物理单元：
- **核显 (Intel QSV)**: 负责 4K 原始视频的高性能解码与缩放。
- **独显 (NVIDIA Tensor Core)**: 100% 算力专注 TensorRT 推理与 GPU 后处理。
- **CPU (Intel P-Core)**: 负责轻量级逻辑分析与异步切割编排。
- **NVMe I/O**: 纯流拷贝切割，避免重复编码。

### 1.2 全局异步流水线
系统通过四个解耦的 Worker 线程进行编排，每个环节通过**有界背压队列 (Bounded Queue)** 连接：

```mermaid
graph LR
    A[DecoderWorker] -- 帧队列 --> B[GPUWorker]
    B -- 检测结果 --> C[LogicWorker]
    C -- 切割任务 --> D[CutterWorker]
```

- **重叠处理 (Overlap)**: 当文件 A 正在切割时，解码器已经在预读文件 B，推理引擎在处理文件 B 的 Batch。这种设计消除了文件切换时的“冷启动”时间，GPU 利用率接近 100%。

## 2. 关键技术细节

### 2.1 手写向量化预处理
绕过传统的 `for` 循环单帧处理，直接在内存中完成：
1. `np.stack(12帧)` 形成批次。
2. 一次性 DMA 传输至 GPU。
3. GPU 内部执行通道翻转 (BGR->RGB) 与归一化。
**性能提升**: 预处理耗时从单帧 ~15ms 降至 Batch 平均 ~0.4ms。

### 2.2 纯 GPU NMS 后处理
传统的 NMS 在 CPU 上运行，会造成严重的数据回传阻塞。V2.4 采用 `torchvision.ops.nms` 在 GPU 上直接处理：
- 引入类别偏移实现类间独立过滤。
- 只有极少量的最终 BBox 数据（每帧 < 1KB）返回 CPU。

### 2.3 VFR 时间轴对齐
针对监控摄像头常见的变帧率 (VFR) 导致的音画不同步或时长漂移：
- 启动前利用 `ffprobe` 快速预扫所有数据包的 PTS。
- 构建查找表，确保推理结果的时间戳与原始流绝对对齐。

### 2.4 有界背压机制
所有队列均设置 `maxsize`。如果 GPU 推理变慢，帧队列会满，从而阻塞解码线程，反压（Backpressure）会使 FFmpeg 减速。这确保了在高并发下系统显存和内存占用始终处于可控范围，不会崩溃。

## 3. 持久化层设计
采用 **SQLite WAL (Write-Ahead Logging)** 模式：
- **幂等性**: 记录已处理文件，支持断点续传。
- **多维分析**: 自动记录解码、推理 FPS 及处理耗时，便于性能审计。
- **日期聚合**: 数据库层完成日期维度的汇总逻辑，简化主程序代码。
