# HomeVlog

HomeVlog 是一个用于家庭主机闲时批量处理室内监控素材的 DailyVlog 生成工具。它会扫描 NAS 或本地目录中的 H.265/MP4 监控录像，按日期和摄像头分组，检测运动片段，压缩静态片段，并输出按天合并的精简视频。

当前代码以 `StreamingOrchestrator` 为主流程：prescreen、analysis、render 三个阶段可以重叠运行，最终通过 SQLite 记录状态，支持重复运行和部分恢复。

## 适用硬件

目标生产环境：

- CPU: Intel i5-12600K
- iGPU: UHD 770 / Intel QSV
- dGPU: RTX 3060Ti / NVDEC + NVENC
- 输入: NAS/SMB 或本地目录中的 4K H.265 MP4

开发机不要求具备上述硬件。开发机上建议只做静态检查、编译检查和配置解析；真实性能必须回到家庭主机用固定素材集验证。

## 处理流程

### 1. Scan

`scanner.py` 扫描 `paths.input_dir`，识别文件名：

```text
{cam_index:02d}_{YYYYMMDDHHMMSS}_{YYYYMMDDHHMMSS}.mp4
```

示例：

```text
00_20260428080000_20260428084600.mp4
```

文件按开始时间归入日期。跨午夜文件归属开始日期。

### 2. Pass 1: Prescreen

`prescreen.py` 对每个文件做低成本运动预筛，输出：

- `STATIC`: 认为整段基本静止，跳过 Pass 1.5。
- `SUSPICIOUS`: 进入精细运动分析。
- `FAILED`: 记录失败，流水线继续处理其他文件。

当前支持两种预筛模式：

- `legacy_seek`: 旧模式，每个采样点单独执行一次 FFmpeg seek。
- `stream_fps`: 单个低 FPS FFmpeg 流式抽帧，发现 motion 后早停。

`stream_fps` 的目标是减少大量小 FFmpeg 进程和 SMB 随机 seek 压力。若家庭主机测试发现长静态文件变慢，可在配置中切回 `legacy_seek`。

### 3. Pass 1.5: Analysis

`detector.py` 对 SUSPICIOUS 文件做精细帧差分析：

- 硬件解码到低分辨率灰度帧。
- 基于 P5/P20 噪声底估计自适应阈值。
- 使用 median filter 抑制 H.265 GOP 边界尖峰。
- 生成 DYNAMIC/STATIC segment。

streaming 主流程会根据文件数量启动 CUDA/QSV analysis worker。不要在开发机上用性能结果推断家庭主机吞吐，真实瓶颈可能在 NAS I/O、QSV、NVDEC、rawvideo pipe 或 CPU filtergraph。

### 4. Optional YOLO Verification

`yolo_verifier.py` 可对 DYNAMIC segment 做二次目标验证，用来把“没有目标对象但帧差明显”的片段转回 STATIC。

当前 streaming 主流程通过配置控制：

```yaml
yolo:
  enabled: true
  streaming_verify: false
```

默认 `streaming_verify: false`，保持当前轻量行为。开启后需要在家庭主机验证总耗时是否下降，因为 YOLO 会与 NVDEC/NVENC/CUDA analysis 争用 RTX 3060Ti。

### 5. Pass 2: Render

`renderer.py` 按 batch 渲染 timeline：

- NV 路径: CUDA/NVDEC decode + `scale_cuda` + NVENC。
- QSV 路径: QSV decode + `scale_qsv` + QSV encode。
- batch 完成后用 concat demuxer 无损合并。

当前 GPU render 路径会在 scale 后 `hwdownload` 到系统内存，再做 trim/setpts/concat。STATIC segment 使用 `setpts` 快进，输出时长会缩短，但仍可能需要解码和缩放较多输入帧。后续若优化 STATIC 渲染策略，必须保留配置 fallback。

## 快速开始

安装依赖：

```powershell
uv sync
```

配置输入目录：

```yaml
paths:
  input_dir: '\\192.168.5.8\homelab\XiaomiCamera_00_B888805AA3CD'
```

运行完整流程：

```powershell
uv run python main.py
```

仅扫描：

```powershell
uv run python main.py --scan
```

处理指定日期和摄像头，但不渲染：

```powershell
uv run python main.py --date 20260428 --cam 0 --no-render
```

处理指定日期和摄像头完整流程：

```powershell
uv run python main.py --date 20260428 --cam 0
```

## 关键配置

所有配置在 [config/settings.yaml](config/settings.yaml)。

### 路径

- `paths.input_dir`: 输入素材目录，必填。

### Prescreen

- `detection.prescreen_segments`: 每文件采样密度。越高盲区越小，但成本越高。
- `detection.prescreen_mode`: `legacy_seek` 或 `stream_fps`。
- `detection.prescreen_resolution`: 预筛抽帧分辨率。
- `detection.prescreen_diff_threshold`: 预筛帧差阈值。
- `detection.prescreen_parallel`: prescreen worker 数。

### Analysis

- `detection.analysis_fps`: 精细分析抽帧 FPS。
- `detection.analysis_resolution`: 精细分析分辨率。
- `detection.motion_sensitivity`: 自适应阈值灵敏度。
- `detection.median_filter_window`: 时序中值滤波窗口。
- `detection.analysis_max_workers`: analysis worker 上限。
- `detection.qsv_fallback_threshold`: 文件数超过该阈值时启用 QSV analysis worker。

### Render

- `pass2.batch_max_files`: streaming render 每个 batch 的最大文件数。
- `pass2.use_streaming_batch_config`: 是否让 streaming 使用 `pass2.batch_max_files`。
- `output.resolution`: 输出分辨率。
- `output.fps`: 输出帧率。
- `output.nv`: NVENC 编码参数。
- `output.qsv`: QSV 编码参数。

### Streaming

- `pipeline.render_start_delay`: render manager 启动前等待 prescreen/analysis 预热的秒数。
- `pipeline.max_io_concurrency`: 全局 FFmpeg/ffprobe I/O 并发限制。
- `pipeline.prescreen_gpu_policy`: `qsv_only` 或 `alternating`。

### YOLO

- `yolo.enabled`: 是否加载 YOLO 验证能力。
- `yolo.streaming_verify`: streaming 主流程是否实际执行 YOLO verification。
- `yolo.sample_fps`: YOLO 验证抽帧 FPS。
- `yolo.device`: YOLO 推理设备，例如 `cuda:0` 或 `cpu`。

## 性能优化与回退

当前性能优化分支的详细实施方案见：

- [docs/PERFORMANCE_OPTIMIZATION_PLAN.md](docs/PERFORMANCE_OPTIMIZATION_PLAN.md)

建议家庭主机 A/B 测试固定同一天、同一 camera、同一批输入文件。最低限度测试：

| 测试 | prescreen | YOLO streaming | 目的 |
| --- | --- | --- | --- |
| A | `legacy_seek` | `false` | 当前行为基线 |
| B | `stream_fps` | `false` | 验证单进程预筛收益 |
| C | `stream_fps` | `true` | 验证 YOLO 是否净收益 |

若优化不及预期，优先通过配置回退：

```yaml
detection:
  prescreen_mode: "legacy_seek"

yolo:
  streaming_verify: false

pass2:
  use_streaming_batch_config: false

pipeline:
  prescreen_gpu_policy: "qsv_only"
```

## 上线前修复说明

- `--no-render` 现在只执行 scan/prescreen/analysis，不再启动 render manager，也不会生成临时 batch 输出。
- 任一 prescreen、analysis 或 render batch 失败都会使当前 date-cam 任务失败，避免只合并成功 batch 后误标记为 `COMPLETED`。
- `detection.prescreen_mode` 默认回到 `legacy_seek`。`stream_fps` 保留为实验选项，必须在家庭主机用固定素材 A/B 验证后再启用。
- `pipeline.render_start_delay` 默认改为 5 秒，避免每个 date-cam 无条件增加 60 秒等待。
- 动态片段如果输入文件没有 audio stream，会自动补静音轨，避免 FFmpeg filter graph 因 `[idx:a]` 不存在而失败。
- concat demuxer 的 batch 路径会加引号，兼容路径中包含空格或特殊字符的情况。

## 性能记录

运行结束后会在 `logs/` 写入 perf JSON，包含：

- pipeline 总耗时。
- prescreen 每文件耗时和 `ffmpeg_calls`。
- analysis decode/analysis time、frames、motion ratio。
- render batch 耗时、GPU 类型、输入数量。
- NVML 可见的 NVIDIA GPU 利用率摘要。

Intel QSV 利用率当前不通过 NVML 采集，需要结合外部工具或 QSV worker 耗时侧面判断。

## 开发机验证

开发机不运行硬件加速和素材测试时，可做：

```powershell
python -m compileall main.py src
```

配置语法检查可使用项目虚拟环境：

```powershell
.\.venv\Scripts\python.exe -c "import yaml, pathlib; yaml.safe_load(pathlib.Path('config/settings.yaml').read_text(encoding='utf-8')); print('yaml ok')"
```

如果本机 `uv run python` 会触发依赖准备或环境解析超时，可以先用 `.venv` 或系统 Python 做静态检查。

## 项目结构

```text
config/settings.yaml              配置文件
main.py                           CLI 入口
src/pipeline.py                   streaming 主流程编排
src/scanner.py                    输入扫描和文件名解析
src/prescreen.py                  Pass 1 预筛
src/detector.py                   Pass 1.5 精细运动分析
src/segment.py                    segment 构建、合并和短段过滤
src/timeline.py                   timeline 构建和 FFmpeg filter graph
src/renderer.py                   Pass 2 batch render 和 concat
src/yolo_verifier.py              可选 YOLO 二次验证
src/database.py                   SQLite 状态持久化
src/monitor.py                    性能采样和 perf JSON
docs/PERFORMANCE_OPTIMIZATION_PLAN.md  性能优化实施方案
```

## 注意事项

- 文档可能过时，行为判断以最新代码为准。
- 家庭主机性能测试前建议保留当前可运行版本的 Git tag 或 commit。
- 不要只看总耗时，必须同时检查输出质量、误检漏检、失败率和 perf JSON。
- `AGENTS.md` 是给自动化编码代理的工作说明，不是用户操作手册。
