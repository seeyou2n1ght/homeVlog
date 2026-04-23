# HomeVlog 开发守则与规范

本项目的目标是构建一个健壮、高性能且易于维护的视频处理系统。开发者需遵守以下规范。

## 1. 环境管理与依赖

项目使用 **[uv](https://github.com/astral-sh/uv)** 作为唯一的包管理器。

- **依赖同步**: 运行 `uv sync` 确保环境与 `uv.lock` 一致。
- **添加依赖**: 使用 `uv add <package>`，严禁手动修改 `pyproject.toml`。
- **Python 版本**: 锁定为 Python 3.10+。

## 2. 核心架构守则

为了保证高性能和稳定性，开发时必须遵循以下红线：

1.  **内存管理 (No Frames on Disk)**：分析阶段的图像数据仅允许在内存管道（Queue/SharedMemory）中流动，绝对禁止写入磁盘临时文件。
2.  **GPU 独占性**：`GPUWorker` 是操作 TensorRT 后端的唯一入口，禁止在其他线程中直接调用检测模型，防止 CUDA Context 竞争和死锁。
3.  **零拷贝原则**：尽量利用 PyTorch 的 CUDA Tensor 进行原地操作（In-place），减少数据在 CPU 和 GPU 之间的往返拷贝。
4.  **资源回收到位**：所有子进程（FFmpeg）必须在 `try...finally` 块中通过 `close()` 方法显式释放，或注册到 `CleanupManager` 中。

## 3. 代码质量与规范

- **静态类型**: 核心接口必须包含 Python Type Hints。
- **模块隔离**:
  - `hal/`: 仅负责底层硬件交互（TRT, CUDA）。
  - `pipeline/`: 仅负责算法逻辑（Tracker, Aggregator）。
  - `utils/`: 仅负责无状态的工具函数。
- **错误处理**:
  - 严禁捕获并忽略所有异常。
  - 对于可恢复的错误（如单帧损坏），应记录 Log 并跳过。
  - 对于不可恢复错误（如 GPU OOM），应立即触发 `stop_event` 优雅停机。

## 4. Git 提交规范

项目遵循 **[Conventional Commits](https://www.conventionalcommits.org/)** 规范。

格式：`<type>(<scope>): <description>`
- `feat`: 新功能（如：增加 QSV 硬件支持）
- `fix`: 修补 BUG（如：修复 FFmpeg 进程残留）
- `docs`: 文档变更
- `refactor`: 重构（既不修复错误也不添加功能的代码更改）
- `perf`: 性能提升的改动

## 5. 健壮性设计

项目内置了针对 Windows 环境和异常中断的保护机制：
- **atexit 钩子**: 自动清理残留的 FFmpeg 进程。
- **stop_event**: 支持主程序随时通过停止信号安全退出，不再处理后续队列任务。
