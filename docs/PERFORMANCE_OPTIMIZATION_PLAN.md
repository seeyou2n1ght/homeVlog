# HomeVlog 性能优化实施方案

本文档记录 HomeVlog 在家庭主机上的性能优化计划。目标硬件为 Intel
i5-12600K UHD 770 Quick Sync 与 NVIDIA RTX 3060Ti。当前代码版本视为
“可运行基线”，以最新代码为准，旧文档只作参考。

开发机无法实际运行硬件加速和 4K 素材压测，因此本分支只做可静态验证的代码开发、
配置接线、编译检查和低风险结构调整。真实性能结论必须回到家庭主机用固定素材集验证。

## 优化目标

- 降低单日 4K 监控素材生成 DailyVlog 的 wall-clock 时间。
- 保持输出效果与当前可运行版本基本一致。
- 尽量释放 UHD 770 QSV 与 RTX 3060Ti 的解码、缩放、编码能力。
- 所有高风险优化都必须能通过 `config/settings.yaml` 回退。
- 生成足够细的 perf JSON，方便家庭主机测试后定位下一轮瓶颈。

## 基线与回退

在家庭主机测试本分支前，建议先固定当前可运行版本：

1. 在当前可运行版本上创建 commit 或 tag。
2. 所有性能优化只在 `codex/` 子分支开发。
3. 高风险行为保留配置开关。
4. 每次 A/B 测试使用同一天、同摄像头、同一批输入文件。

回退方式：

- Git 回退：切回基线分支或 tag。
- 配置回退：把新增模式切回 legacy/default 值。
- 单项回退：按 commit 粒度 revert 某个优化。

## 测试约束

开发机不能验证：

- FFmpeg QSV/CUDA/NVENC 实际可用性。
- NAS/SMB 读写吞吐。
- 4K H.265 长素材真实性能。
- Intel iGPU 利用率。
- 输出画面质量和误检漏检。

开发机只做：

- Python 编译检查。
- 静态代码检查。
- 配置路径和 import 路径检查。
- 不依赖素材的单元测试。

## 家庭主机测量协议

固定样本：

- 同一日期。
- 同一 camera。
- 同一批输入文件。
- 同一输出分辨率、FPS、质量参数和 audio 配置。
- 对比完整流程时，清理或隔离数据库状态，避免复用旧分析结果影响结论。

必须记录：

- 总 wall-clock 时间。
- prescreen 文件数、平均耗时、p50、p95、最大值。
- 每文件 `ffmpeg_calls`、`checked_pairs`、是否 early stop。
- analysis 的 decode time、Python analysis time、frames、motion ratio。
- render 每个 batch 的 GPU、输入文件数、segment 数、耗时。
- STATIC/DYNAMIC 时长占比。
- NVIDIA NVML 的 encoder/decoder 利用率。
- Intel QSV 利用率，若无法采集，则用 QSV worker 耗时侧面判断。
- 输出文件大小和肉眼可接受度。
- 误动态段、漏动态段样例。

## 当前瓶颈假设

### Pass 1 prescreen

旧 prescreen 路径会对每个采样点启动一次 FFmpeg seek。当前
`prescreen_segments: 120` 时，一个完全 STATIC 文件最坏约 121 次 FFmpeg
进程调用，再加一次 ffprobe。150 个文件会放大为大量进程启动与 SMB 随机 seek。

本分支新增：

- `detection.prescreen_mode: legacy_seek | stream_fps`
- `legacy_seek` 保留旧行为。
- `stream_fps` 使用单个低 FPS FFmpeg 流式抽帧，并在发现 suspicious diff 时早停。

预期收益：

- 降低 FFmpeg 进程启动次数。
- 降低 NAS 随机 seek 压力。
- 让 prescreen perf JSON 能直接反映每文件调用成本。

风险：

- 对极长且完全静态的文件，连续低 FPS decode 可能比随机 seek 慢。
- 若家庭主机验证变慢，直接切回 `legacy_seek`。

### Pass 1.5 analysis

当前 streaming 路径根据任务数量使用 CUDA/QSV worker。瓶颈可能在硬件解码、rawvideo
pipe、`hwdownload` 或 NAS I/O，不应盲目增加 worker。

本分支策略：

- 继续记录每文件 decode time、analysis time、frames。
- 不在开发机上调整 worker 默认值。
- 通过家庭主机 perf JSON 决定是否增加 CUDA 或 QSV 并发。

### YOLO verification

YOLO 的目标是把“无目标对象但帧差明显”的动态段转回 STATIC，减少后续渲染工作量。
它只有在节省的 render 成本大于 YOLO 抽帧和推理成本时才是净收益。

本分支新增：

- `yolo.streaming_verify: true | false`
- 默认 false，保持当前 streaming 行为。
- 开启后在 streaming Pass1.5 中执行 YOLO verification。
- perf JSON 记录 YOLO 前后 segment 数量。

风险：

- YOLO 使用 `cuda:0` 时会与 NVDEC/NVENC/CUDA analysis 争用 RTX 3060Ti。
- 当前 YOLO shared model 有锁，推理可能串行。
- 若总耗时增加，切回 false。

### Pass 2 render

当前 GPU render 路径使用硬件 decode/scale 后 `hwdownload` 到系统内存，再执行 trim、
setpts、concat。STATIC 段用 `setpts` 快进，输出时长减少，但仍可能解码和缩放大量输入帧。

本阶段只做低风险接线：

- streaming 批大小读取 `pass2.batch_max_files`。
- 保留旧的 `output.batch_max_files` fallback。
- 暂不改变 STATIC render 策略，避免影响输出观感。

后续候选：

- STATIC 段抽帧或 hybrid keyframe 模式。
- 更细粒度的 render batch metadata。
- 根据 NV/QSV batch 实测速度调整 batch size。

## 新增配置项

### `pass2.use_streaming_batch_config`

- true：streaming 主流程使用 `pass2.batch_max_files`。
- false：回退旧查找逻辑，优先 `output.batch_max_files`，缺失时默认 8。
- 用途：让 Pass2 配置在实际主流程中生效。

### `pipeline.prescreen_gpu_policy`

- `qsv_only`：保持当前 streaming 行为，所有 prescreen worker 走 QSV。
- `alternating`：prescreen worker 在 QSV/CUDA 间交替。
- 用途：家庭主机确认 NAS I/O 未饱和后，测试 RTX 3060Ti 是否能帮助 prescreen。

### `detection.prescreen_mode`

- `legacy_seek`：旧实现，每个采样点一次 FFmpeg seek。
- `stream_fps`：单进程低 FPS 流式抽帧。
- 用途：降低 FFmpeg 进程启动与 SMB 随机 seek 成本。

### `yolo.streaming_verify`

- true：streaming Pass1.5 后执行 YOLO verification。
- false：保持当前 streaming 行为。
- 用途：评估 YOLO 是否能通过减少动态段降低总 render 成本。

## 当前分支已实现内容

截至当前实现，本分支已经完成以下改动：

- streaming 主流程读取 `pass2.batch_max_files`，并通过 `pass2.use_streaming_batch_config` 保留旧逻辑 fallback。
- streaming render manager 使用 `pipeline.render_start_delay`，不再写死 5 秒。
- streaming prescreen worker 支持 `pipeline.prescreen_gpu_policy`，默认仍为 `qsv_only`。
- `prescreen_file()` 支持 `detection.prescreen_mode: legacy_seek | stream_fps`。
- 优化了 `stream_fps` 的错误处理与 stdout 鲁棒性。
- `build_concat_filter` 支持 `render.static_mode: hybrid_keyframe`，显著降低静态素材的渲染压力。
- streaming prescreen 会保存 `prescreen_result`，并写入 perf records。
- perf records 会记录 prescreen 的 `mode`、`ffmpeg_calls`、`checked_pairs`、`early_stop`、`sample_fps` 等字段。
- streaming analysis 可通过 `yolo.streaming_verify` 控制是否执行 YOLO verification，默认关闭。
- analysis perf records 记录 YOLO 前后 segment 数量。

开发机已完成的验证：

- Python 编译检查通过。
- AST 静态解析通过。
- 使用项目 `.venv` Python 解析 `config/settings.yaml` 通过。

开发机未执行、也不应作为结论的验证：

- FFmpeg QSV/CUDA/NVENC 实际运行。
- 4K 素材处理。
- NAS/SMB 吞吐。
- 家庭主机端输出质量和性能对比。

## 实施阶段

### 阶段 1：配置接线与观测

- 新增配置项和中文注释。
- streaming 读取 `pipeline.render_start_delay`。
- streaming 读取 `pass2.batch_max_files`。
- streaming 保存 prescreen `result_json`。
- streaming 写入 prescreen perf records。

### 阶段 2：prescreen 单进程模式

- 实现 `detection.prescreen_mode: stream_fps`。
- 保留 `legacy_seek`。
- perf 中记录 `ffmpeg_calls`、`checked_pairs`、`sample_fps`、`early_stop`。

### 阶段 3：YOLO streaming 开关

- 实现 `yolo.streaming_verify`。
- 记录 YOLO 前后 segment 数。
- 默认关闭，等待家庭主机 A/B。

### 阶段 4：render STATIC 策略实验

- 暂不在本轮默认启用。
- 后续再比较 current setpts、sampled static、hybrid keyframe。
- 每种模式都必须保留 fallback。

## 家庭主机 A/B 矩阵

| 测试 | prescreen | YOLO streaming | render | 目的 |
| --- | --- | --- | --- | --- |
| A | legacy_seek | false | current | 当前行为基线 |
| B | stream_fps | false | current | 验证 prescreen 优化 |
| C | stream_fps | true | current | 验证 YOLO 净收益 |
| D | stream_fps | false | 调整 batch size | 验证批大小和双 GPU 调度 |

## 成功标准

- 不增加 FAILED 文件或 FAILED render。
- motion 选择质量不下降。
- 同样输入集总 wall-clock 时间降低。
- perf JSON 能解释哪个阶段变快或变慢。
- 任何高风险优化都能通过配置或 Git 快速回退。
# 上线前修复记录（2026-05-12）

本轮修复以下 review 发现的问题：

- render batch 失败不再被吞掉；任一 batch 缺失、返回空路径或抛异常，当前 date-cam 会标记为 `FAILED`。
- `--no-render` 不再启动 render manager，只跑到 prescreen/analysis，用于开发机和家庭主机预检查。
- `render_tasks` 状态写入改为 upsert，避免没有预先创建记录时 `FAILED` 状态丢失。
- 默认 `detection.prescreen_mode` 回退为 `legacy_seek`，`stream_fps` 保留为实验开关。
- 默认 `pipeline.render_start_delay` 从 60 秒降为 5 秒，降低每个 date-cam 的固定等待成本。
- 动态片段缺少 audio stream 时使用 `anullsrc` 补静音，避免 filter graph 引用不存在的 `[idx:a]`。
- `stream_fps` prescreen 会检查 FFmpeg return code，异常退出不再按 partial frame 结果误分类。
- concat demuxer 列表中的路径改为带引号写入，兼容空格和特殊字符。

当前开发机仍不具备硬件加速和真实素材验证条件；上线前必须在家庭主机用固定 date-cam 素材做 A/B 测试。
