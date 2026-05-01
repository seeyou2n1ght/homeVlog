# 复盘文档 — 设计到开发经验总结

## 元信息

| 字段 | 值 |
|------|-----|
| 更新日期 | 2026-04-28 |
| 对应版本 | v0.1.0 (Phase 1-5 完成) |
| 素材规模 | 10 样本文件 (~3.2h, ~1.3GB) |
| 更新触发 | 每完成一个 Phase 或有重大经验发现时更新 |

---

## 1. 设计阶段预判偏差

### 1.1 分析分辨率与 pipe 数据量

**预判**: analysis 用 1280×720@5fps。  
**实测**: rawvideo pipe 数据量 = `1280×720×3 × 5fps × 340s ≈ 4.7GB/文件`，直接卡死。  
**调整**: 640×360 gray (单通道, 230KB/帧)，12.8s/文件，motion detection 准确率无下降。

**教训**: pipe 传输 raw video 时先算数据量。分辨率选择不只是精度问题，是 IO 可行性问题。

### 1.2 文件名时长 vs ffprobe 实际时长

**预判**: 文件名时间戳差值足够准确。  
**实测**: 文件名 dur=599s，ffprobe=598.64s，差 0.36s 导致最后时间戳 seek 出界，2/10 文件预筛失败。  
**调整**: prescreen_file 内部用 `get_duration()` (ffprobe) 获取准确时长，且 `_calc_sample_timestamps` 最后点用 `duration - 0.5s` 安全边距。

**教训**: 文件边界时间戳不能依赖元数据估算。要么安全边距 >= 1s，要么花 0.1s 调 ffprobe。

### 1.3 静态段关键帧 vs 加速播放

**预判**: select filter 做 keyframe extraction + 0.5s 展示。  
**实测**: select + setpts + loop filter 组合在 filter_complex 中实现复杂，且 concat 要求统一分辨率。  
**调整**: v1 用 `setpts=PTS/60` (60x 加速) 等价替代 30s→0.5s 压缩比。效果接近，实现简单。

**教训**: filter_complex 中复杂帧选择逻辑应先在单条命令行测试可行性，再集成到生成器。

---

## 2. 开发阶段 FFmpeg 踩坑

### 2.1 stdout binary decode (已修复)

**问题**: `run_ffmpeg` 把 rawvideo stdout decode 为 utf-8 字符串。  
**症状**: `len(解码后字符串) ≠ len(原始bytes)` → frame reshape 失败。  
**修复**: `FFmpegResult.stdout: bytes`，调用方按需 `.decode()`。

### 2.2 concat filter pad 顺序 (已修复)

**问题**: `[v0][v1]...[a0][a1]...concat` — 所有 video pad 在前，audio 在后。  
**症状**: "Media type mismatch between Parsed_scale filter output pad 0 (video) and Parsed_concat filter input pad 1 (audio)"。  
**修复**: `labels = "".join(f"[v{i}][a{i}]" for ...)` — 交替排列。

**原理**: concat filter 的 segment 模式 (`concat=n=N:v=1:a=1`) 要求 pads 严格按 `[v0][a0][v1][a1]...[vN][aN]` 排列。不同于 `concat=n=N:v=0:a=1` (纯音频)。

### 2.3 硬件解码 + CPU scale 不兼容 (已修复)

**问题**: renderer 用 `-hwaccel cuda -hwaccel_output_format cuda`，filtergraph 用 `scale` (CPU filter)。  
**症状**: "Input link in0:v0 parameters (size 3840x2160) do not match output link (320x180)"。实际是 scale 在 CUDA 帧上没生效。  
**修复**: 移除 renderer 的 `-hwaccel`，软件解码 + 硬件编码。NVENC/QSV 编码不需要 hwaccel 输入。

### 2.4 scale 必须在 trim 之前 (已修复)

**问题**: `trim → scale` 链中 concat 读到的是 trim 前分辨率。  
**修复**: `scale → trim`。FFmpeg filtergraph metadata 传播：`scale` 输出帧携带新的 width/height，`trim` 透传该 metadata。反之 trim 透传输入文件的原始分辨率。

### 2.5 anullsrc channel_layout 语法

**当前状态**: 使用 `anullsrc=r=48000:cl=mono:d=DUR`。  
**注意**: FFmpeg 7.x `channel_layout=` → `cl=`。`channel_layout=mono` 在旧版可用。`cl=mono` 是简写。需在目标 FFmpeg 版本实测。

### 2.6 CUDA OOM: per-input hwaccel 大规模文件渲染 (2026-04-29)

**问题**: 98个输入文件，每个带 `-hwaccel cuda -hwaccel_output_format cuda -i`，filter graph 包含 98 个 `hwupload_cuda` 实例。渲染阶段 `cuCtxCreate` 失败：`CUDA_ERROR_OUT_OF_MEMORY`（RTX 3060Ti 8GB，空闲 6.8GB）。

**调试过程**（4次迭代）：
1. 怀疑AAC编码器格式问题 → 加 `aformat` 统一采样率/声道 → 无效（错误从AAC变成NVENC，但根因相同）
2. 怀疑stderr截断丢失关键信息 → 扩展日志：保留完整stderr、filter_complex、ffmpeg命令 → 发现 `CUDA_ERROR_OUT_OF_MEMORY`
3. 怀疑 `-hwaccel_output_format cuda` 隐式hwupload失败 → 去掉output_format，filter链显式 `hwupload_cuda,` → 仍然OOM
4. 怀疑98个per-file CUDA decoder上下文耗尽显存 → 去掉所有 `-hwaccel`，纯软解 → 仍然OOM（`hwupload_cuda` filter 初始化时也需要CUDA context）

**根因**: 没有显式 `-init_hw_device` 时，每个 `hwupload_cuda` filter 实例可能尝试独立初始化CUDA context。98个filter × 各自cuCtxCreate → 即使单个context只占少量显存，98次创建请求叠加消耗驱动内部资源（非可见VRAM），或触发碎片化导致后续创建失败。

**修复**: 
```bash
ffmpeg -init_hw_device cuda=gpu:0 \   # 显式创建共享CUDA device
  -i f1 -i f2 ... -i f98 \            # 全部软解
  -filter_hw_device gpu \              # filter graph使用此device
  -filter_complex_script fc.txt ...
```
一个CUDA context，98个filter共享。`hwupload_cuda` 自动从 `gpu` device派生。

**附带修复**: concat filter音频格式不匹配（DYNAMIC段保留原始采样率，STATIC段`anullsrc`输出48000Hz mono）→ DYNAMIC音频链加 `aformat=sample_rates=48000:channel_layouts=mono`。

**教训**: 
- 多文件 + GPU filter graph 场景，始终用 `-init_hw_device` 显式创建device，不要依赖FFmpeg自动推导
- Windows NVIDIA驱动在进程频繁创建/销毁CUDA context后可能留下"retired contexts"，即使子进程已退出，VRAM未必立即可用
- stderr截断600字符不够看（原型错误被截在之前），失败日志应保留完整stderr
- `hevc_nvenc` 编码器可用系统内存帧（yuv420p），不需要GPU decode前置

---

## 3. 算法设计迭代

### 3.1 段合并算法 (3 次重写)

**v1** (原地翻转): 遍历列表，短 DYNAMIC → STATIC，短 STATIC → DYNAMIC。前一个段 flip 后改变了邻居状态，后续判断基于脏数据。

**v2** (两阶段 flip+merge): 先收集 flip 决策（基于原始邻居状态），再批量应用 flip，再 merge。但连续 3 个交替短段时，左右都翻、中间被迫孤立 (0.8s 的 STATIC 残留在结果中)。

**v3** (迭代吸收): 遍历列表，短段直接并入相邻的不同态段 (修改其 end_time 或 start_time)，pop 短段，continue 同一索引。外层 while 循环到 stable。

**教训**: 段合并是局部贪心问题，不要用全局 flip + merge 两步走。滑动窗口看左右邻居，短就吃掉，迭代到不动。v3 代码最少、逻辑最直观、结果最正确。

### 3.2 帧差分 ROI 裁切顺序

**当前做法**: `frame[roi_y:roi_y+h, roi_x:roi_x+w]` 后再 `np.mean(axis=2)` 转灰度。裁切后再转灰度比先转灰度再裁切节省约 3× 的 roi 外像素计算。

### 3.3 红外运动检测 + H.265 GOP 伪影 (2026-04-29)

**发现**: H.265 编码的 I-frame/P-frame 差异在 5fps 连续解码时产生周期性 `mean(abs diff)` 尖峰。46min 红外文件中 694 帧（5%）energy ≈ 1.24-1.30，每 20 帧（4.0s）一次。红外宝宝翻身同样 energy ≈ 1.25，SNR=1:1。自适应 IQR 阈值无法区分——降低 threshold 引入 694 帧 GOP 误报，提高 threshold 漏检宝宝。

**关键差异不在幅值而在时域形态**:
- GOP 伪影: 孤立尖峰（1 帧 peak + 2 帧衰减尾 = 3 帧）, 周期性 4.0s
- 真实运动: 持续高原（≥10 帧 elevated, 虽然幅值只有 0.08-0.24）

**解决**: 时序中值滤波 (window=7) 作为 IQR 阈值的前置步骤。window=7 的 median 覆盖 GOP 尖峰全宽度（3 帧两侧各有 ≥2 帧 baseline），尖峰被替换为 baseline。持续高原的中值仍为 elevated 值，得以保留。

**参数联动**: 中值滤波后 energy 分布紧致（Q1≈0.036, Q3≈0.068, IQR≈0.032, max≈0.18），IQR sensitivity 需从 8.0 降至 2.0（≈3.4σ）。运动段滤波后 duration 被压缩（10 帧→6 帧过阈值），`min_motion_duration` 从 2.0s 降至 1.0s。

**教训**: 监控视频的 H.265 GOP 伪影在低对比度场景（红外）中与真实运动不可区分，必须利用时域结构特征（孤立 vs 持续）而非仅依靠幅值统计。中值滤波是最简单的时域结构提取器。

---

## 4. 工程实践

### 4.1 SQLite WAL 断点续跑

用 `is_render_completed` 一行检查实现 crash-safe，比最初计划的纯文件标记更可靠。用户坚持引入 SQLite 是正确的。

### 4.2 增量测试节奏

每写完一个模块立刻用真实样本测试。Scanner → Prescreen → Detector → Segment → Timeline → Renderer，每步发现问题立即修，bug 没跨模块累积。

### 4.3 uv 包管理

pynvml 弃用 → 换成 nvidia-ml-py，一行 `uv sync` 无冲突。

### 4.4 Pass2 优化经验

**渲染管线架构**: 软件解码(CPU) + GPU缩放 + 硬件编码。`-init_hw_device cuda=gpu:0` 创建共享CUDA device，`-filter_hw_device gpu` 让filter chain继承。避免per-input `-hwaccel` 导致的大规模CUDA context OOM（见2.6）。

**GPU scale + hwdownload 限制**: `scale_cuda`/`scale_qsv` 后必须 `hwdownload,format=nv12` 把帧从 GPU 显存拷回系统内存。这之后的 CPU filter（如 `fps`）与 CUDA 帧不兼容——`fps=fps=1/30` 在 hwdownload 后产 0 帧。GPU 路径解决方案：静止段用 `setpts=PTS/60` 快进，CPU 路径保留关键帧幻灯片。

**split-render 实验**: 完整的双半渲染+无损 concat 已实现。10 文件测试中 NVENC 半耗时 2m13s (15段)，QSV 半耗时 5m5s (2段但各46min 静态)。两静态文件落到 QSV 导致总耗时 7m5s > 单路 4m34s。负载不均衡是核心问题，暂时关闭。

**预设调优**: NVENC preset p4→p1 几乎不影响画质（CQ 模式下 preset 主要影响编码速度），QSV medium→fast。可调参数集中到 settings.yaml。

### 4.5 双 GPU Pass1.5

按 `file_duration` 降序 + 交替分片（interleave）优于简单的 N/2 前后切。原因：文件时长可能不均，前半全是长文件导致一个 GPU 跑完另一个还在跑。交替分配让两个 GPU 都吃到长短混合文件。

SQLite 并发安全：DB 写有 `threading.Lock`（database.py），WAL 模式支持并发读+串行写。两个 worker 各自调 `db.set_analysis_result()`，锁争用可忽略。

---

## 5. 已解决的问题

### 2026-04-28 (Phase 优化)

| 问题 | 解决方案 | 涉及文件 |
|------|---------|---------|
| 双 GPU Pass1.5 分析 | SUSPICIOUS 文件按 duration 降序交替分片，NVDEC+QSV 双线程并行。实测 1.7x 加速 | `detector.py` |
| Pass2 软解慢 (8m26s) | 加 `-hwaccel cuda/qsv` 每个 `-i` 前 (之前不知道需要 per-input)，GPU scale 替代 CPU scale。1.9x 加速 (4m34s) | `renderer.py`, `timeline.py` |
| CUDA OOM: 大规模文件渲染 (98文件) | `-init_hw_device cuda=gpu:0` + `-filter_hw_device gpu` 共享CUDA device，软解替代per-file hwaccel。audio concat格式对齐加 `aformat`。stderr完整捕获+fc保留便于排查 | `renderer.py`, `timeline.py` |
| CUDA OOM: 496 segments GPU scale (2026-04-29) | 原因：每个segment一条 `hwupload_cuda,scale_cuda,hwdownload` 链 → 496次CUDA分配。改为per-file scale（98次）仍OOM——NVIDIA驱动无法同时初始化98个filter context（6.6GB空闲）。最终方案：CPU scale + NVENC encode。timeline.py重构为per-file scale+split架构，GPU路径留作未来优化。 | `timeline.py`, `renderer.py` |
| `fps` filter 与 hwdownload 帧不兼容 | GPU 路径静止段用 `setpts=PTS/SPEED` 快进替代关键帧幻灯片，视觉等价 | `timeline.py` |
| split-render 负载不均衡 | 已实现完整代码，默认关闭 (`pass2.split_render: false`)。QSV 分到长静态文件时严重拖慢 | `renderer.py` |
| 编码预设保守 | NVENC p4→p1, QSV medium→fast | `settings.yaml` |
| Unicode `→` 在 Windows cp1252 日志崩溃 | 全部替换为 `->` | `renderer.py`, `pipeline.py` |
| prescreen 线程安全 | `_process_one` 返回 local dict，主线程聚合 | `prescreen.py` |
| SQL f-string 注入 | `mark_file_failed` column 参数加白名单 `_ALLOWED_FAIL_COLUMNS` | `database.py` |
| rendering 中断恢复: RENDERING→PENDING 回滚 | `_recover_interrupted()` 在 pipeline 启动时检查并回滚所有 RENDERING 状态 | `pipeline.py:63-70` |
| `anullsrc` 在 FFmpeg 7.x 格式兼容性 | 确认 `anullsrc=r=48000:cl=mono:d=DUR` 在 FFmpeg 8.1 可用 | `timeline.py` |
| 静态段关键帧幻灯片替代全帧快进 | CPU 路径: `fps=fps=1/{kf_interval}` 取关键帧 + 0.5s 展示 | `timeline.py` |
| 红外运动检测漏检（宝宝翻身判全静态） | ① IQR 自适应阈值替代 hardcode `diff_threshold`+`motion_ratio` ② 时序中值滤波 (window=7) 压制 H.265 GOP 边界伪影（每 4s 尖峰 energy ~1.25，与红外运动同量级）③ sensitivity=2.0 + min_motion_duration=1.0s | `detector.py`, `settings.yaml` |
| 496 segments GPU scale CUDA OOM | timeline.py 重构为 per-file scale + split 架构。98 个 hwupload_cuda 仍触发驱动 OOM（6.6GB 空闲，驱动 context 数限制）。改为 CPU scale + NVENC encode，性能损失小（scale 占比低）。GPU scale 代码保留，待 driver 突破后可切回 | `timeline.py`, `renderer.py` |

## 6. 待解决问题

| 问题 | 优先级 | 说明 |
|------|--------|------|
| split-render 负载均衡 | 中 | 当前按 STATIC 边界切分，两半时长可能不均。应加上时长加权 |
| Pass2 QSV 编码槽闲置 (单摄) | 低 | 硬件约束，单路输出不可拆分。双摄解决 |
| GPU scale CUDA OOM workaround | 中 | 当前用 CPU scale。NVIDIA 驱动对同时初始化的 hwupload_cuda filter 实例数有限制（~98 个 OOM，6.6GB 空闲）。可能的突破方向：① 单 input + select filter 替代多 input + trim ② FFmpeg 批量初始化优化 ③ 更换驱动版本 |
| 空间方差法未充分探索 | 低 | Grid-based `std(cell_means)` 过滤 GOP 伪影 98%，当前被中值滤波方案替代。中值滤波若退化可切换 |

---

## 7. 性能基准 (10 文件, 2026-04-28)

### 旧方案 (串行 NVDEC, 软件解码)

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Scan | <1s | |
| Pass1 预筛 (QSV×4) | 13s | |
| Pass1.5 分析 (NVDEC 串行) | 92s | 640×360 gray, 5fps |
| Pass2 渲染 (NVENC, 纯软解+CPU缩放) | 506s | 17 segments, 7 files → 1 output |
| **总计** | **611s** | |

### 新方案 (双 GPU Pass1.5 + 软解+GPU缩放 Pass2)

| 阶段 | 耗时 | vs 旧方案 |
|------|------|-----------|
| Scan | <1s | — |
| Pass1 预筛 (QSV×4) | 13s | 不变 |
| Pass1.5 分析 (NVDEC+QSV 并行) | 54s | 1.7x |
| Pass2 渲染 (软解+scale_cuda+NVENC) | 274s | 1.9x |
| **总计** | **341s** | **1.8x** |

### 单日 150 文件外推

| 阶段 | 旧方案 | 新方案 |
|------|--------|--------|
| Pass1 | ~3.5min | ~3.5min |
| Pass1.5 | ~23min | ~13min |
| Pass2 | ~60min | ~45min |
| **总计** | **~88min** | **~62min** |


### 最终架构演进 (2026-05-01)
*   **极致 Prescreen 提速 (Lazy Fast-Fail)**：
    *   **问题**：监控视频中存在大量长切片，导致硬件全解超时。
    *   **解决**：抛弃全量并发提取，采用单帧延迟提取（\`ffmpeg -ss ... -vframes 1\`），一旦发现动静立刻提早终止。此举将耗时大幅缩减。
*   **修复静态漏检 Bug**：
    *   **问题**：由于提早断定静态，导致大量包含后半段运动的可疑视频被错抛（漏检）。
    *   **解决**：移除对 STATIC 判定的早停逻辑，100%覆盖全视频采样点。
*   **极致调度 Work-Stealing**：
    *   双卡在统一队列中动态消费。速度更快的独显自动分配了近3倍的工作量，从而实现两张卡的无缝对齐。
