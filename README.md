# HomeVlog

将家庭摄像头备份素材浓缩为按日期、机位分组的 DailyVlog：动态与有效声音常速保留，静态按展示时长计划快进。面向 Windows、NAS、Intel 核显 QSV 与 NVIDIA 显卡；当前开发主机为 i5-12600K + RTX 3060 Ti 8GB。

识别采用关键帧预筛、连续灰度运动分析和 YOLO 候选验证。采样、遮挡、夜视和模型误检都可能导致漏检，因此不能把它作为原始录像的替代品，也不承诺事件 100% 保留。

## 安装与首次运行

需要 Python 3.12+、uv、支持 QSV/NVENC/CUDA 滤镜的 FFmpeg，以及可用的 Intel/NVIDIA 驱动。FFmpeg 必须在 PATH 中。核显需要启用。项目使用 uv.lock 固定依赖，PyTorch 使用配置中的 CUDA 索引。

```powershell
uv sync --locked
ffmpeg -version
uv run python main.py --scan
uv run python main.py --no-tui
```

运行前在 [config/settings.yaml](config/settings.yaml) 设置 `paths.input_dirs`，确认 `yolo.model_path` 指向已有本地权重。输出默认位于 `output/`，数据库在 `data/vlog.db`。素材文件名需符合 `00_YYYYMMDDhhmmss_YYYYMMDDhhmmss.mp4`；当前扫描器读取输入目录的直接文件，不递归扫描。

机位使用目录 MAC（无 MAC 时用目录路径）和通道生成持久身份。数据库为每个身份分配唯一 `cam_index`，后续按 `--scan` 显示的索引操作；不同目录内的同名通道不再混成同一日视频。

## 常用命令

```powershell
# 指定日期与数据库机位索引
uv run python main.py --date 20260321 --cam 0 --no-tui

# 只分析，不渲染
uv run python main.py --no-render --no-tui

# 指定配置与输入目录
uv run python main.py --config config/settings.yaml --input-dir D:\CameraBackup

# 强制重新预筛、分析；持久人工修正仍在原时间范围生效
uv run python main.py --date 20260321 --cam 0 --reanalyze --no-tui

# 显式清理全部历史批次；通常不需要
uv run python main.py --clean-temp

uv run python main.py --help
```

`recovery.skip_today` 可跳过尚未备份完整的当天素材。周末批处理可使用无 TUI 命令运行；不要同时启动多个处理相同日期/机位的主程序进程，当前硬件预算和进程注册表为进程内共享。

## 处理流程与资源预算

1. **预筛**：默认 `keyframes` 用 PyAV 软件解码关键帧，发现疑似动作即可进入精析；判定静态前必须读到 EOF。采样不足或预算到期转入精析。关键帧间的短事件仍有漏检风险。
2. **精析**：通过 QSV/NVDEC 输出单通道灰度，不为 YOLO 再开视频解码。整文件持续分析，不以开头静态推断未来静态。分析容器元数据懒加载后回填数据库。
3. **YOLO**：复用内存 JPEG 候选，显式传入低置信度入口阈值，逐个小批推理。没有帧、只有不足的阴性证据或推理失败时保留动态并标记复核；画外声音不因缺少画内目标而自动快进。
4. **渲染**：保序组织批次，默认两路 NVENC，QSV 保留给分析；展示时长和字幕使用同一计划。纯静态长片段在缩放前稀疏选帧。

生产配置保留 `max_nv_concurrency: 2`，它是本项目安全作业预算，不是驱动会话上限或显存保证。QSV 作业也共用一个硬件预算；I/O 按输入文件数原子申请，渲染批次不会绕过 NAS 预算。

生产配置为 8 个分析 worker，按录制时间优先处理较早文件，避免后续短文件阻塞首批渲染。运动特征在解码时连续计算，不保存完整灰度帧序列；每帧能量与置信度占 16 字节，另保留固定大小的背景模型。每任务 `analysis_buffer_mb` 约束 JPEG 候选、音频特征与诊断帧池，不能视为整个进程的 RSS 上限。`render.max_concurrency` 控制实际渲染 worker 数，硬件信号量仍独立限制在途作业。

当前性能证据、完整复测和测试条件见 [2026-09-08 性能记录](docs/PERFORMANCE_20260908.md)。

`yolo.batch_size` 控制真正送入模型的小批大小。提高并发或模型大小前，应同时测显存、CPU、NAS 吞吐与人工事件召回；高 GPU 占用率本身不是优化目标。

## 恢复、配置变化与人工修正

- 批次先写 `.tmp.mp4`，验证容器、视频流和展示时长后原子替换；合法的小 MP4 不受固定 512 KiB 门槛限制。
- 批次与最终成片都有指纹清单。素材大小/mtime、配置、模型版本或时间轴变化后不复用旧结果。
- 检测配置、模型或素材变化会重置旧分析状态。第一次使用本次修复版本会重建无指纹的历史分析；旧成片在新结果成功替换前保留。
- 解码不完整、取消或 DB 写入失败不再伪装为分析成功。坏文件按重试策略降级跳过；渲染批次失败阻止合并，保留已成功批次。
- 人工标签独立存于 `human_reviews`，按原始素材和时间范围覆盖算法结果；重分析改变切片边界也不会扩大或丢掉人工修正。
- 所有直接启动的 FFmpeg 子进程纳入注册表；取消会终止进程，资源等待也会响应中断。

## 审核、数据导出与微调

```powershell
uv run python scripts/audit_tool/app.py
uv run python scripts/inspect_detection.py "path/to/sample.mp4" --csv
uv run python scripts/verify_accuracy.py --report-out docs/latest_quality.md

# 按日期、机位、预筛状态分层抽取原始时间窗，包含预筛静态区域
uv run python scripts/sample_accuracy.py --count 100 --output data/review_windows.csv

# 导出到新的空目录，可按 MAC 或配置别名过滤
uv run python scripts/export_dataset.py --output-dir data/datasets/my_cam --camera B888805AA3CD
uv run python scripts/train_yolo.py --data data/datasets/my_cam/data.yaml --epochs 15
```

审核快捷键与界面操作以当前工作台为准，TP/FP/FN/TN 短码和完整标签统一归一化。`FrameArchiver` 将反馈帧保存到 `data/feedback_archive/images/` 并追加 `manifest.jsonl`；原片不可达时尝试近期内存帧及已有审核缓存，失败记录 Warning。

TN/FP 导出空标注，正样本缺少可信框时排除并记录，不能当作背景训练。注意“静态”不等于“没有人”：静坐人物仍需要 person 框，标注负样本前必须确认没有目标。已人工校正的同名 `.txt` 框文件优先于伪标签。类别沿用 COCO 的 person=0、cat=15、dog=16，避免自定义类别与基础模型错位。

训练/验证按源视频隔离并记录 provenance。导出的伪标签验证集不能证明真实准确率。训练强制 `freeze=10`；要自动应用新模型，必须另提供与训练素材不重叠、人工校正的 holdout，并通过召回与精度不退化检查：

```powershell
uv run python scripts/train_yolo.py --data data/datasets/my_cam/data.yaml --epochs 15 --apply --validation-data data/datasets/holdout/data.yaml
```

`--apply` 写配置，并非运行中热切换。新任务加载所选权重。无人工标注时审核指标显示 N/A；自动疑似 FP/FN 仅为待审线索。

## 开发验证与性能评估

```powershell
uv run ruff check main.py src scripts tests
uv run python -m compileall main.py src scripts
uv run python -m pytest tests/

# 本机真实 QSV/NVDEC/NVENC 与本地 YOLO 权重冒烟测试
$env:HOMEVLOG_HARDWARE_TESTS = '1'
uv run python -m pytest tests/test_hardware_smoke.py

uv run python scripts/analyze_perf.py --top 5
uv run python scripts/benchmark.py --all
```

CI 在 Windows 执行 locked 依赖同步、Ruff、语法检查和回归。Ruff 当前启用致命错误与未定义名称规则，尚未实施全仓库严格类型检查。硬件测试显式开启，普通单元测试不能替代真实硬件验收。

2026-09-08 修复验证与限制见 [实施记录](docs/IMPLEMENTATION_20260908.md)。[历史基准](docs/BENCHMARK.md) 来自旧算法与素材分布，不能作为新版速度或召回保证；修复预筛/早停后处理量可能上升，应重新测量。

## 文档导航

- [架构与数据契约](docs/ARCHITECTURE.md)
- [开发约束](AGENTS.md)
- [审查发现与原始复现](docs/REVIEW_20260908.md)
- [实施及验收记录](docs/IMPLEMENTATION_20260908.md)
- [历史质量线索](docs/ACCURACY_AUDIT_REPORT.md)
