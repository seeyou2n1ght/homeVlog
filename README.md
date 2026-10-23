# HomeVlog

将家庭摄像头备份素材按日期、机位分析为完整Vlog，并按需生成不超过20分钟的日常观看版。支持人工事件复核与重渲，面向Windows、NAS、Intel QSV与NVIDIA CUDA/NVENC。

当前产品范围为：扫描 → 预筛/分析 → 完整成片 → 日常版 → 人工复核/重渲。项目内模型训练、目标框标注、交互调参、基准管理及旧原型界面已退役。当前验证结果见[验收报告](docs/ACCEPTANCE.md)。

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

```powershell
# 指定日期与机位；只分析可加 --no-render
uv run python main.py --date 20260321 --cam 0 --no-tui
# 显式重新分析、重新交付；持久人工审核仍优先
uv run python main.py --date 20260321 --cam 0 --reanalyze --force-render --no-tui
# 使用其他配置和素材目录
uv run python main.py --config config/settings.yaml --input-dir D:\CameraBackup
uv run python main.py --help
```

`--dry-run`会扫描入库并统计计划，不执行重分析重置、清理批次或编码。`--stage analyze`与`--no-render`等价；`--cam`可单独限定该机位。`recovery.skip_today`可跳过当天尚未备份完整的素材。

## 观看与人工复核

```powershell
uv run python scripts/audit_tool/app.py
```

打开终端打印的本地地址，默认http://127.0.0.1:8765。支持完整库存分页、筛选、精选、固定种子抽查及视觉相似检索；原片可按真实区间正常速度播放并听声音。代表图和视觉相似不等于事件真值。

审核支持TP/FP/FN/TN、场景/备注/观看依据及撤销。人工修正按原始素材区间持久化，优先于算法；重渲前查看批次计划和缓存复用情况，失败或取消保留旧交付和有效批次。

展开“日常版／争议复核／完整版本”，输入日期与机位：先预览，再生成，等待COMPLETED。已有版本可查找、播放和下载。日常版按父成片的完整映射段选择，不再次加速；人工阳性优先，人工阴性相交段转入争议复核，优先段超过预算会明确拒绝。

日常MP4可以单独观看；完整追溯还须保留本包selection/review/manifest、父成片及meta、源身份对应的原片。完整版本和原声复核不能脱离这些依赖。20分钟是观看预算，不是识别、精彩程度或感知质量承诺。
操作与依赖详见[工作台说明](scripts/audit_tool/README.md)。当前output为空，没有随本次源码交付可用的历史日常包；原先依赖scratch的演示包和实验附件已不在。

## 正确性、资源与恢复

- 关键帧预筛不能证明关键帧间没有短事件；采样不足、推理失败或阴性证据不足会保守进入分析/保留动态并标记复核。仍可能漏检，Vlog不能替代原始录像。
- 分析复用单次灰度解码与内存候选；时间轴、字幕及映射共用 `compute_display_plans()`。
- 默认7个分析worker、NV作业预算2及QSV协同；硬件和I/O由共享调度器管理。预算不是显存或驱动会话上限保证。
- 批次先写临时媒体并验证再原子发布；交付包括媒体、元数据、字幕和manifest。完成记录提交失败、中断或取消不得伪装成功。
- `render_tasks.COMPLETED`为历史交付记录，普通运行不会因产物已移走而自动重做。备份交付时同时备份数据库；明确重做使用 `--force-render`。
- 默认保留有效批次供恢复；`render.cleanup_batches_on_success`可控制成功后清理。`--clean-temp`会显式清理历史批次，只在确认不再需要恢复时使用。
- `.delivery_backup_*`和recovery.json是失败恢复资产，不能作为普通缓存删除。同一共享工作区不支持跨进程同时处理不同日期；工作台与主流程通过内核认领拒绝冲突。

## 维护与验证

保留analyze_perf.py（汇总性能日志）、benchmark_day.py（冻结负载的隔离完整流水线测量）、inspect_detection.py（单源诊断）、sample_accuracy.py（人工抽查窗口）和index_review_visuals.py（视觉索引补建）。它们不自动改变模型或参数。

```powershell
uv run ruff check main.py src scripts tests
uv run python -m compileall main.py src scripts
uv run python -m pytest tests/
```

按变更范围选专项；交付前一次全库足够，不重复累加通过数。硬件专项须显式开启HOMEVLOG_HARDWARE_TESTS，不能由绿色单测推导真实NAS、硬件性能或识别质量。

文档：[架构](docs/ARCHITECTURE.md)、[状态](docs/PROGRESS.md)、[决策](docs/DECISIONS.md)、[测试](docs/TESTING.md)、[性能历史](docs/BENCHMARK.md)。
