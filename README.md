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

# 重做指定日期，重新预筛、分析；持久人工修正仍在原时间范围生效
uv run python main.py --date 20260321 --cam 0 --reanalyze --force-render --no-tui

# 显式清理全部历史批次；通常不需要
uv run python main.py --clean-temp

uv run python main.py --help
```

`recovery.skip_today` 可跳过尚未备份完整的当天素材。周末批处理可使用无 TUI 命令运行；主流程与审核重渲通过数据库日期/机位、共享临时工作区及交付路径的内核认领拒绝冲突；同临时工作区也不支持跨进程并行不同日期。硬件预算和进程注册表仍是进程内共享。

`--cam` 可单独选择该机位的所有日期；预览、重析和实际处理使用相同范围。`--stage` 支持 `all`、`analyze`，后者与 `--no-render` 相同，执行预筛和精析。仅分析遇到失败或未完成素材也返回非零退出码。

`--dry-run` 会扫描入库并统计计划，不执行重析重置、临时批次清理或转码，即使同时传入 `--reanalyze` / `--clean-temp`。已交付日期默认跳过；重析范围包含已交付日期时，须同时使用 `--force-render`，否则在重置前拒绝执行。

## 处理流程与资源预算

1. **预筛**：默认 `keyframes` 用 PyAV 软件解码关键帧，发现疑似动作即可进入精析；判定静态前必须读到 EOF。采样不足或预算到期转入精析。关键帧间的短事件仍有漏检风险。
2. **精析**：通过 QSV/NVDEC 输出单通道灰度，不为 YOLO 再开视频解码。整文件持续分析，不以开头静态推断未来静态。分析容器元数据懒加载后回填数据库。
3. **YOLO**：复用内存 JPEG 候选，显式传入低置信度入口阈值，逐个小批推理。没有帧、只有不足的阴性证据或推理失败时保留动态并标记复核；画外声音不因缺少画内目标而自动快进。
4. **渲染**：默认两路 NVENC 与一路 QSV 协同；批次最终按录制顺序拼接，展示时长和字幕使用同一计划。纯静态长片段在缩放前稀疏选帧。

生产配置保留 `max_nv_concurrency: 2`，它是本项目安全作业预算，不是驱动会话上限或显存保证。QSV 作业也共用一个硬件预算；I/O 按输入文件数原子申请，渲染批次不会绕过 NAS 预算。

生产配置为 7 个分析 worker；单文件渲染批次采用短作业优先，多文件批次按录制顺序处理。运动特征在解码时连续计算，不保存完整灰度帧序列；每帧能量与置信度占 16 字节，另保留固定大小的背景模型。每任务 `analysis_buffer_mb` 约束 JPEG 候选、音频特征与诊断帧池，不能视为整个进程的 RSS 上限。`render.max_concurrency` 控制实际渲染 worker 数，硬件信号量仍独立限制在途作业。

实测遥测与验收边界见 [docs/BENCHMARK.md](docs/BENCHMARK.md)，当前阻塞见 [docs/PROGRESS.md](docs/PROGRESS.md)。

`yolo.batch_size` 控制真正送入模型的小批大小。提高并发或模型大小前，应同时测显存、CPU、NAS 吞吐与人工事件召回；高 GPU 占用率本身不是优化目标。

## 恢复、配置变化与人工修正

- 批次先写 `.tmp.mp4`，验证容器、视频流和展示时长后原子替换；合法的小 MP4 不受固定 512 KiB 门槛限制。
- 批次与最终成片都有指纹清单。未交付日期会校验素材大小/mtime、配置、模型和时间轴；检测配置或素材变化会使旧分析失效。
- `render_tasks.COMPLETED` 保存历史交付记录，普通运行不因配置变化、产物移走或新增素材自动重做。备份产物时须保留并备份 `data/vlog.db`；显式 `--force-render` 才重新处理。
- 成片先写 `.pending.mp4` 并生成字幕/元数据，manifest 和完成记录提交成功后才清理本次批次。默认保留有效批次，可设置 `render.cleanup_batches_on_success: true` 在成功后清理。
- 普通提交异常或取消会恢复上一份资产；恢复本身失败时，旧文件保留在输出目录 `.delivery_backup_*/`，`recovery.json` 记录原路径。新事务包由下次持有认领的入口核对并恢复；旧仅含 `recovery.json` 的包会阻止对应历史完成记录被直接跳过，须人工检查，不直接删除。文件分别原子替换，不承诺断电时整个资产包同时提交。
- 解码不完整、取消或 DB 写入失败不再伪装为分析成功。坏文件按重试策略降级跳过；渲染批次失败阻止合并，保留已成功批次。
- 人工标签独立存于 `human_reviews`，按原始素材和时间范围覆盖算法结果；撤销当前区间同步撤销持久标签，区间外的审核和已归档图片保留。短区间保持真实源边界，输出最小帧数由展示计划决定。
- 所有直接启动的 FFmpeg 子进程纳入注册表；取消会终止进程，资源等待也会响应中断。

## 审核、数据导出与微调

```powershell
uv run python scripts/audit_tool/app.py
uv run python scripts/inspect_detection.py "path/to/sample.mp4" --csv
uv run python scripts/verify_accuracy.py --report-out reports/latest_quality.md

# 按日期、机位、预筛状态分层抽取原始时间窗，包含预筛静态区域
uv run python scripts/sample_accuracy.py --count 100 --output data/review_windows.csv

# 导出到新的空目录，可按 MAC 或配置别名过滤
uv run python scripts/export_dataset.py --output-dir data/datasets/my_cam --camera B888805AA3CD
uv run python scripts/train_yolo.py --data data/datasets/my_cam/data.yaml --epochs 15
```

审核快捷键与界面操作以当前工作台为准，TP/FP/FN/TN 短码和完整标签统一归一化。`FrameArchiver` 将反馈帧保存到 `data/feedback_archive/images/` 并追加 `manifest.jsonl`；原片不可达时尝试近期内存帧及已有审核缓存，失败记录 Warning。

训练只接受绑定图片 SHA 和来源身份、完成全部目标复核的 `.targets.json`。事件误报/静止、旧 `.txt`、草稿及模型预测都不能充当目标框真值；只有明确复核无目标才允许空框。类别沿用 COCO person=0、cat=15、dog=16。

导出按来源/日期分为训练、调优和独立测试组，默认约60/20/20；不足三个合格组时拒绝导出。训练固定 `freeze=10`，每次生成独立候选与训练收据，不覆盖已有候选。可先使用 `--dry-run` 检查数据和基底：

```powershell
uv run python scripts/train_yolo.py --data data/datasets/my_cam/data.yaml --dry-run
# 对已有候选评估独立 test 分组，不训练或写生产配置
uv run python scripts/train_yolo.py --data data/datasets/my_cam/data.yaml --base-model models/yolo11n.pt --evaluate-model models/my_candidate.pt --validation-data data/datasets/my_cam/data.yaml
```

`--apply` 已停用。图片指标通过仍需独立原片识别和动作验收；完整复核、来源分组与候选评估操作见[工作台说明](scripts/audit_tool/README.md)。无合格人工参考时指标为不可用，自动疑似 FP/FN 仅为待审线索。

旧 `tune_thresholds.py` 只保留历史摘要诊断，不推荐或写入生产阈值；参数比较使用工作台实际重放。`repair_audio_metadata.py` 只读统计疑似记录，不推断音轨存在、不修改数据库，旧 `--clean-batches` 已停用。

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

生产基准实测遥测与多日数据见 [docs/PROGRESS.md](docs/PROGRESS.md)。

## 文档导航

- [开发约束与治理规则](AGENTS.md)
- [系统架构与数据契约](docs/ARCHITECTURE.md)
- [重大架构决策记录 (ADR)](docs/DECISIONS.md)
- [项目进度与实测遥测](docs/PROGRESS.md)
- [测试策略与质量验收规范](docs/TESTING.md)
- [性能与质量基准](docs/BENCHMARK.md)
