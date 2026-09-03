# AGENTS.md

This file provides guidance to Codex or other coding agents when working in this repository.

## 上线前注意事项 (2026-09-03 更新)

- **性能极限状态**: 当前系统已达到硬件饱和状态（~95% GPU 解码利用率，NVENC 并发上限 2）。修改 `src/filters.py`、`src/detector.py` 或 `src/yolo_verifier.py` 的算法逻辑时，必须通过 `uv run python scripts/benchmark.py --operator` 确认是否引起每帧延迟恶化。
- **Lazy Metadata 强制约束**: 禁止在 `src/scanner.py` 中增加任何阻塞式的文件读写（如 `ffprobe`）。所有元数据探测必须在 Analysis 阶段通过 PyAV 懒加载完成，并存入 DB。
- **时间轴闭环**: 修改 `src/detector.py` 的 `analyze()` 返回值时，必须确保最后一帧的时间戳严格等于 `start_offset + file_duration`，杜绝渲染出的 Vlog 出现时间轴空洞或跳秒。
- **硬件并发信号量与调度**: 算力并发控制必须通过 `src.scheduler` 中的 `get_nv_semaphore()`、`get_qsv_semaphore()` 与 `WorkStealingManager` 协调，防止 NVENC 超出驱动并发限制（当前生产配置 `max_nv_concurrency: 2`，8GB 显存安全上限）或 QSV 句柄耗尽崩溃。
- **信号量超时强制约束**: 所有 `io_sem.acquire()` 调用必须带 `timeout` 参数（推荐 30s），并通过重试循环（最多3次）防止 prescreen 与 analysis 争用 QSV 信号量时发生死锁。禁止使用无限阻塞的 `acquire()`。
- **信号量释放单次原则**: `io_sem.release()` 必须在 `finally` 块中恰好调用一次。禁止在 `return` 前显式调用 `release()` 后又在 `finally` 中重复释放，否则信号量计数膨胀导致并发失控。

## 核心开发哲学

1. **单次解码 (Single-Pass)**: 一次解码，多重消费。严禁为 YOLO 或运动分析单独开启重复的解码进程。
2. **零拷贝 (Zero-IO)**: 候选帧在内存 JPEG 切片压缩池中流转，禁止在中间阶段频繁写入磁盘零碎临时文件。
3. **容错重于完美**: 在 NAS 环境下，网络抖动是常态。对于失败的切片，记录 Warning 并跳过，确保最终 Vlog 能够生成。

## 环境与测试规范

- 统一使用 `uv` 管理虚拟环境与依赖。
- **CUDA 支持**: `pyproject.toml` 已配置 PyTorch CUDA 索引，`uv sync` 会自动安装支持 RTX 显卡的版本。禁止手动更改为普通 CPU 版 torch。
- 提交前全量测试回归：
  ```powershell
  uv run pytest tests/
  ```
- 语法与静态检查：
  ```powershell
  uv run python -m compileall main.py src
  ```

## 命名与目录规范

- 统一使用 `Path` 对象处理路径，严禁使用字符串拼接，以确保跨 NAS 环境的兼容性。
- 日志时间戳统一采用 `localtime`。
- 模型权重统一归属于 `models/` 目录。

