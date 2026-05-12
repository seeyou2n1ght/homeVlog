# AGENTS.md

This file provides guidance to Codex or other coding agents when working in this repository.

## 上线前注意事项（2026-05-12）

- `--no-render` 只能执行到 prescreen/analysis，不应启动 render worker。
- 任一 prescreen、analysis 或 render batch 失败时，当前 date-cam 任务必须失败，不能用部分 batch concat 后标记 `COMPLETED`。
- prescreen 默认模式是 `legacy_seek`；`stream_fps` 仍作为家庭主机 A/B 测试选项。
- `pipeline.render_start_delay` 默认值是 5 秒。
- 动态片段必须兼容没有 audio stream 的输入，必要时生成静音轨。
- 仓库内文档和代码注释使用中文，QSV/NVDEC/NVENC/FFmpeg 等专有名词可保留英文。

## 项目定位

HomeVlog 是一个家庭监控素材 DailyVlog 生成工具。它扫描 NAS 或本地目录中的 H.265/MP4 监控录像，按日期和摄像头分组，通过运动检测裁掉或快进静态片段，输出精简的每日 vlog。

当前实现以代码为准。旧设计文档可能过时，尤其是 prescreen multi-seek、YOLO 是否参与 streaming、批渲染配置来源等描述，必须以 `src/` 和 `config/settings.yaml` 的最新实现判断。

## 常用命令

```powershell
# 完整流程: scan -> prescreen -> analyze -> render
uv run python main.py

# 仅扫描
uv run python main.py --scan

# 指定日期和摄像头，跳过 render
uv run python main.py --date 20260428 --cam 0 --no-render

# 指定日期和摄像头完整流程
uv run python main.py --date 20260428 --cam 0

# 运行测试
uv run pytest

# 开发机静态编译检查，不运行硬件加速或素材处理
python -m compileall main.py src

# 使用项目虚拟环境检查 YAML
.\.venv\Scripts\python.exe -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('config/settings.yaml').read_text(encoding='utf-8')); print('yaml ok')"
```

开发机通常无法验证 QSV/CUDA/NVENC 和真实素材性能。不要在开发机上运行硬件加速压测来推断家庭主机表现。

## 当前主流程

入口：

- `main.py` 调用 `src.pipeline.run_pipeline()`。
- 指定 `--date` 时调用 `src.pipeline.process_date_cam()`。
- 当前实际处理由 `StreamingOrchestrator` 编排。

Streaming 主流程：

1. `scanner.py` 扫描 `paths.input_dir`，写入 `file_tasks`。
2. prescreen worker 从队列读取 PENDING 文件，调用 `prescreen_file()`。
3. SUSPICIOUS 文件进入 analysis queue。
4. analysis worker 调用 `MotionDetector.analyze()` 生成 segment。
5. 若 `yolo.enabled` 且 `yolo.streaming_verify` 均为 true，streaming analysis 会调用 `YoloVerifier.verify()`。
6. render manager 收集已完成文件，按 batch 调用 `build_batch_render()`。
7. 多个 batch 最后通过 `concat_output_files()` 无损合并。

## 关键模块

| 文件 | 作用 |
| --- | --- |
| `src/pipeline.py` | streaming 主流程编排，连接 prescreen、analysis、render。 |
| `src/scanner.py` | 扫描输入目录，解析 `{cam}_{start}_{end}.mp4` 文件名，写入 DB。 |
| `src/prescreen.py` | Pass 1 预筛，支持 `legacy_seek` 和 `stream_fps`。 |
| `src/detector.py` | Pass 1.5 精细运动检测，低分辨率灰度帧差、阈值估计、平滑。 |
| `src/yolo_verifier.py` | 可选 YOLO 二次验证，用于过滤无目标对象的 DYNAMIC segment。 |
| `src/segment.py` | Segment dataclass、segment 构建、合并和短段过滤。 |
| `src/timeline.py` | 从 DB 构建 timeline，生成 FFmpeg filter graph。 |
| `src/renderer.py` | Batch render、NV/QSV 渲染路径、最终 concat。 |
| `src/database.py` | SQLite 状态持久化，WAL，线程锁保护。 |
| `src/ffmpeg.py` | FFmpeg/ffprobe wrapper 和硬件解码参数构建。 |
| `src/monitor.py` | NVML/CPU/RAM 采样和 perf JSON 记录。 |
| `src/utils.py` | 配置加载、目录常量、信号量、工具函数。 |

## 配置重点

配置文件是 `config/settings.yaml`。

### Prescreen

- `detection.prescreen_mode`
  - `legacy_seek`: 每个采样点单独执行一次 FFmpeg seek。
  - `stream_fps`: 单个低 FPS FFmpeg 流式抽帧，发现 motion 后早停。
- `detection.prescreen_segments`: 采样密度。
- `detection.prescreen_parallel`: prescreen worker 数。
- `pipeline.prescreen_gpu_policy`
  - `qsv_only`: 默认行为，所有 prescreen worker 使用 QSV。
  - `alternating`: prescreen worker 在 QSV/CUDA 间交替。

### Analysis

- `detection.analysis_fps`: 精细分析 FPS。
- `detection.analysis_resolution`: 精细分析分辨率。
- `detection.motion_sensitivity`: 自适应阈值灵敏度。
- `detection.median_filter_window`: 中值滤波窗口。
- `detection.analysis_max_workers`: analysis worker 上限。
- `detection.qsv_fallback_threshold`: 文件数超过该阈值时启用 QSV analysis worker。

### YOLO

- `yolo.enabled`: 是否启用 YOLO 能力。
- `yolo.streaming_verify`: streaming 主流程是否实际执行 YOLO verification。默认 false。
- `yolo.device`: 推理设备，例如 `cuda:0`。

### Render

- `pass2.batch_max_files`: streaming render batch 大小。
- `pass2.use_streaming_batch_config`: true 时 streaming 使用 `pass2.batch_max_files`；false 时回退旧的 `output.batch_max_files` 查找。
- `pipeline.render_start_delay`: render manager 启动前等待 prescreen/analysis 预热的秒数。
- `output.nv` / `output.qsv`: NVENC/QSV 编码参数。

## 性能优化原则

- 先加观测，再改算法。
- 所有高风险优化必须有 config fallback。
- 家庭主机 A/B 测试必须固定同一天、同摄像头、同一批文件。
- 不只看总耗时，也要检查输出质量、误检漏检、失败率和 perf JSON。
- 不要直接删除 legacy 路径，除非已有家庭主机数据证明新路径稳定更快。

当前性能优化实施文档：

- `docs/PERFORMANCE_OPTIMIZATION_PLAN.md`

## FFmpeg 和 GPU 注意事项

- NV 路径使用 CUDA/NVDEC decode、`scale_cuda`、NVENC。
- QSV 路径使用 QSV decode、`scale_qsv`、QSV encode。
- 当前 render filter graph 在 GPU scale 后会 `hwdownload` 到系统内存，再执行 trim/setpts/concat。
- STATIC segment 当前使用 `setpts` 快进，输出时长减少，但输入帧仍可能需要大量 decode/scale。
- 后续若优化 STATIC render 策略，必须保留当前策略作为 fallback。

## 状态与恢复

- SQLite DB 位于 `data/vlog.db`。
- `file_tasks` 记录 per-file prescreen/analysis 状态。
- `render_tasks` 记录 per date/cam render 状态。
- `process_date_cam()` 会跳过已经 COMPLETED 且无新增 PENDING 文件的 date/cam。
- rerun 应尽量保持幂等，不要破坏已有 COMPLETED/ANALYZED 数据。

## 输入命名

```text
{cam_index:02d}_{YYYYMMDDHHMMSS}_{YYYYMMDDHHMMSS}.mp4
```

示例：

```text
00_20260428080000_20260428084600.mp4
```

跨午夜文件归入开始日期。

## 开发约束

- 不要在开发机假设硬件加速可用。
- 不要把硬件测试失败当成代码逻辑失败，除非家庭主机也复现。
- 修改代码后至少做 Python 编译检查。
- 修改 YAML 后用项目 `.venv` Python 解析配置。
- 代码注释和文档使用中文，GPU、FFmpeg、streaming、batch、fallback 等专有名词可保留英文。
