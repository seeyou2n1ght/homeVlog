# HomeVlog

将家庭摄像头备份素材按日期和机位扫描、分析并浓缩为带字幕和元数据的 Vlog。时间轴由算法和场景配置确定。支持 Windows、NAS、Intel QSV 和 NVIDIA CUDA/NVENC。

## 安装与运行

需要Python 3.12+、uv、包含QSV/NVENC/CUDA支持的FFmpeg及Intel/NVIDIA驱动。FFmpeg须在PATH中。PyTorch沿用uv.lock和配置的CUDA索引。

```powershell
uv sync --locked
ffmpeg -version
uv run python main.py --scan
uv run python main.py --no-tui
```

先在[配置文件](config/settings.yaml)设置 `paths.input_dirs` 和本地 `yolo.model_path`，当前生产模型为models/yolo11m.pt。扫描读取输入目录的直接文件，不递归；文件名需符合 `00_YYYYMMDDhhmmss_YYYYMMDDhhmmss.mp4`。
机位按目录MAC/路径及通道登记；使用 `--scan` 显示的cam_index选择范围。数据库默认data/vlog.db，成片默认output/。

无可用 GPU 时，将 `pipeline.render_gpu_policy` 设置为 `cpu_only`，并确保 FFmpeg 支持 libx264。旧版数据库需先备份并移出运行目录，再启动并扫描素材以初始化新库。

```powershell
# 指定日期与机位；只分析可加 --no-render
uv run python main.py --date 20260321 --cam 0 --no-tui
# 显式重新分析并重新交付
uv run python main.py --date 20260321 --cam 0 --reanalyze --force-render --no-tui
# 使用其他配置和素材目录
uv run python main.py --config config/settings.yaml --input-dir D:\CameraBackup
uv run python main.py --help
```

`--dry-run`会扫描入库并统计计划，不执行重分析重置、清理批次或编码。`--stage analyze`与`--no-render`等价；`--cam`可单独限定该机位。`recovery.skip_today`可跳过当天尚未备份完整的素材。

## 正确性、资源与恢复

- 关键帧预筛可能漏掉采样间隔内的短事件。采样不足、推理失败或阴性证据不足时，流水线会继续分析或保留动态区间。Vlog 不能替代原始录像。
- 分析复用单次灰度解码与内存候选；时间轴、字幕及映射共用 `compute_display_plans()`。
- 默认7个分析worker、NV作业预算2及QSV协同；硬件和I/O由共享调度器管理。预算不是显存或驱动会话上限保证。
- 批次先写临时媒体并验证再原子发布；交付包括媒体、元数据、字幕和manifest。完成记录提交失败、中断或取消不得伪装成功。
- `render_tasks.COMPLETED`为历史交付记录，普通运行不会因产物已移走而自动重做。备份交付时同时备份数据库；明确重做使用 `--force-render`。
- 默认保留有效批次供恢复；`render.cleanup_batches_on_success`可控制成功后清理。`--clean-temp`会显式清理历史批次，只在确认不再需要恢复时使用。
- `.delivery_backup_*` 和 `recovery.json` 用于恢复失败的交付。同一共享工作区通过内核认领限制并发处理，避免多个进程改写同一批次和交付资产。

## 维护与验证

维护工具包括 `analyze_perf.py`（汇总性能日志）、`benchmark_day.py`（隔离运行冻结负载）和 `inspect_detection.py`（单源诊断）。

```powershell
uv run ruff check main.py src scripts tests
uv run python -m compileall main.py src scripts
uv run python -m pytest tests/
```

按变更范围选专项；交付前一次全库足够，不重复累加通过数。硬件专项须显式开启HOMEVLOG_HARDWARE_TESTS，不能由绿色单测推导真实NAS、硬件性能或识别质量。

文档：[架构](docs/ARCHITECTURE.md)、[状态](docs/PROGRESS.md)、[决策](docs/DECISIONS.md)、[测试](docs/TESTING.md)、[性能历史](docs/BENCHMARK.md)。
