# AGENTS.md

This file provides guidance to Codex or other coding agents when working in this repository.

## 上线前注意事项 (2026-05-15 更新)

- **性能极限状态**: 当前代码已达到硬件饱和状态（88% GPU 解码利用率）。修改 `detector.py` 或 `yolo_verifier.py` 的处理逻辑时，必须通过 `perf.json` 确认是否引起帧率下降。
- **Lazy Metadata 强制约束**: 禁止在 `scanner.py` 中增加任何阻塞式的文件读写（如 `ffprobe`）。所有元数据探测必须在 Analysis 阶段通过 PyAV 懒加载完成，并存入 DB。
- **时间轴闭环**: 修改 `detector.py` 的 `analyze()` 返回值时，必须确保 `early_term` 触发时最后一帧的时间戳严格等于 `start_offset + file_duration`，否则会导致渲染出的 Vlog 出现“跳秒”现象。
- **硬件锁 (Semaphores)**: 分析与渲染任务必须通过 `src.utils` 中的 `get_nv_semaphore()` 和 `get_qsv_semaphore()` 获取锁，以防止 NVENC 超出驱动并发限制或 QSV 句柄耗尽崩溃。

## 核心开发哲学

1. **单次解码 (Single-Pass)**: 一次解码，多重消费。严禁为 YOLO 或运动分析单独开启重复的解码进程。
2. **零拷贝 (Zero-IO)**: 帧数据在显存/内存字典中流转，禁止在中间阶段写入磁盘临时文件。
3. **容错重于完美**: 在 NAS 环境下，网络抖动是常态。对于失败的片段或文件，记录 Warning 并跳过，确保最终 Vlog 能够生成，哪怕缺失少量镜头。

## 环境与依赖管理

- 必须使用 `uv` 管理。
- **CUDA 支持**: `pyproject.toml` 已配置 Pytorch CUDA 索引，`uv sync` 会自动安装支持 RTX 显卡的版本。禁止手动更改为普通 CPU 版 torch。
- 修改代码后至少执行：
  ```powershell
  uv run python -m compileall main.py src
  ```

## 命名与目录规范

- 统一使用 `Path` 对象处理路径，严禁使用字符串拼接，以确保跨 NAS 环境的兼容性。
- 日志时间戳统一采用 `localtime`。

## 历史缺陷教训

- **2026-05-15**: 修复了早停逻辑导致的片段丢失（Gap）。修复了扫描阶段因 `ffprobe` 过多导致的网络假死。修复了 YOLO `Conv.bn` 属性缺失错误（已通过 `model.fuse()` 解决）。
