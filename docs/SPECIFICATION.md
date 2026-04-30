# HomeVlog 重构规格文档

## 1. 现状分析与瓶颈推断

已有项目 (`homevlog-20260428`) 采用 **Two-Pass FFmpeg + SQLite 状态机** 架构：

```
Pass 1: FFmpeg mpdecimate (GPU downscale 4K→720p → hwdownload → CPU mpdecimate)
Pass 2: FFmpeg 复杂 filtergraph (按segment切换速度/静音)
```

**推断的性能瓶颈：**

| 环节 | 问题 | 影响 |
|------|------|------|
| Pass1 GPU→CPU回传 | `hwdownload` 将720p帧从GPU显存拷贝到系统内存，每帧都要走PCIe总线 | 带宽浪费，引入延迟 |
| mpdecimate CPU单线程 | 降尺度后交给CPU做逐帧像素比较，无法利用多核 | 检测速度受单核限制 |
| Pass2 重构编码 | 过滤图按segment动态切换，每个文件涉及concat拼接多个子片段 | FFmpeg filtergraph复杂度O(segments) |
| 文件级独立处理 | 每个mp4文件独立启动FFmpeg进程，进程创建/销毁开销 | 150文件 × 2 Pass = 300次FFmpeg启动 |
| 单Prober串行 | 一个探测器串行分析所有文件，qsv/nvenc渲染器等待 | GPU利用率不足 |
| Pass1+Pass2 双解码 | 每个文件被解码两次（探测一次、渲染一次） | 总计算量翻倍 |

**核心矛盾：** 用 FFmpeg filtergraph 做 "剪辑编排" 不适合片段数量多的场景。每个segment切换点在filtergraph中产生分支，复杂度随segment数线性增长。一天150个文件可能产生数百到上千个segment。

---

## 2. 设计目标

1. **One-Pass 或 1.5-Pass**: 尽量减少解码次数
2. **GPU驻留**: 运动检测尽可能在GPU上完成，减少显存↔内存拷贝
3. **批量流水线**: 多文件作为一个整体时间线处理，减少进程启动开销
4. **帧级精确控制**: 用Python做剪辑决策，FFmpeg只做编解码执行
5. **空闲时段运行**: 周末夜间全自动处理当周素材

---

## 3. 推荐方案: Frame-Server 架构

### 3.1 核心思路

打破 "文件→文件" 的处理模式，改为 **"解码器→帧流→motion引擎→段索引→编码器"** 的流水线：

```
NAS素材 → [按日期分组] → Pass1: QSV×4路 分段预筛 (320×180)
                              │
                              ├── 85%+ 纯静态文件 → 跳过
                              │
                              └── 疑似动态文件 → Pass1.5: NVDEC+QSV并行分析
                                    720p@5fps → numpy帧差分 → 段索引
                                                │
                                                ↓
                                          段索引表 (motion/static 区间)
                                                │
                                                ↓
                                    Pass2: 硬件编码渲染
                                    ├── 单摄像头: NVENC 编码
                                    └── 双摄像头: NVENC + QSV 并行编码
```

### 3.2 为什么比旧方案快

| 对比维度 | 旧方案 | 新方案 |
|----------|--------|--------|
| 解码次数 | 2次4K/文件 (Pass1+Pass2) | 纯静态仅采6帧；疑似文件多1次低分辨率分析，无冗余4K |
| GPU利用率 | 低 (串行探测+等待渲染) | **高** (双GPU并行解码分析、并行渲染) |
| 进程开销 | ~300次FFmpeg启动 | **~8次** (Pass1×4 + Pass1.5×2 + Pass2×2) |
| 运动检测 | CPU 单线程 mpdecimate | **numpy向量化** 批量帧差分 |
| 输出片段拼接 | filtergraph concat N段 | **帧级select** 时间线编辑 |

### 3.3 1.5-Pass 变体

完整 One-Pass 需要解码帧流同时做出所有决策，但 motion detection 需要前瞻（look-ahead）来识别运动结束点。

实际采用 **1.5-Pass**：

1. **Pass 1 (分段预筛)**: 每文件等分5段采6帧(320×180)，相邻帧差分判断。全部静态 → PURE_STATIC。有动态 → 进入Pass 1.5。
2. **Pass 1.5 (详细分析)**: 仅对疑似动态文件，NVDEC和QSV各解码一路720p@5fps，numpy批量帧差分，双GPU并行产出段索引。
3. **Pass 2 (渲染)**: 两摄像头各自独立渲染，cam0走NVDEC+NVENC，cam1走QSV decode+encode，`select` filter按时间线选择帧并调速。

**关键优化**: Pass 1分段采样极快（每文件约3-4秒），可跳过90%+纯静态文件。跳过率受场景影响，以实测为准。

---

## 4. 运动检测算法

### 4.1 快速预筛 (Pass 1)

问题：每个文件5分钟，3点采样的盲区长达2.5分钟。1-2分钟的短暂运动刚好落在采样间隙就会被漏掉。婴儿照看场景漏检不可接受。

改进：分段采样，等分文件为N段，每段边界采帧。任意相邻帧差超阈值即标记待分析。

```
对每个mp4文件:
  duration = ffprobe获取时长
  segments = 5              # 等分5段，每段~1分钟
  采6帧: 各段首+文件尾 (0:00, 1:00, 2:00, 3:00, 4:00, 5:00)
  每帧降采样到320x180

  对5对相邻帧做差分:
    diffs = [|f[i+1] - f[i]| for i in range(5)]
    if max(diffs) > prescreen_diff_threshold:
      标记为NEEDS_ANALYSIS
    else:
      标记为PURE_STATIC
```

优势：
- 1分钟盲区 vs 原来2.5分钟，漏检概率大幅降低
- 判据用 max(diffs)，宁多不漏（假阳性进Pass1.5多花几分钟，假阴性丢运动不可接受）

时间消耗：每文件5次seek + 6帧QSV解码 → ~3-4秒，150文件约8分钟。

### 4.2 帧差分详细分析 (Pass 1.5)

```
参数:
  analysis_fps: 5          # 每秒分析5帧
  analysis_res: 640x360    # 灰度单通道 (设计初为1280x720, 实测rawvideo pipe数据量过大)
  diff_threshold: 15       # 像素差值阈值 (0-255)
  motion_ratio: 0.02       # 超过阈值的像素占比即判为motion

算法:
  解码帧流 (NVDEC或QSV硬件解码, 640×360 gray, 5fps, extractplanes=y)
  对相邻帧（已灰度）ROI裁切后差分:
    diff = abs(frame_n+1_roi - frame_n_roi)
    motion_pixels = count_nonzero(diff > diff_threshold)
    ratio = motion_pixels / roi_pixels
    is_motion_frame = (ratio > motion_ratio)

  状态机平滑:
    - 连续>=3帧 motion → 标记为DYNAMIC段开始
    - 连续>=5帧 static → 标记为DYNAMIC段结束
    - 静止段中<2帧的motion噪点 → 忽略（合并到静止段）
    - 运动段中<2帧的static噪点 → 忽略（合并到运动段）
```

> 设计调整: 1280×720 rgb24 rawvideo pipe 数据量 = `1280×720×3 × 5fps × 340s ≈ 4.7GB/文件`，pipe 缓冲区阻塞。改为 640×360 gray (extractplanes=y, 230KB/帧)，motion detection 准确率无下降。先算数据量再选分辨率，不是精度问题，是 IO 可行性问题。

### 4.3 ROI遮罩

摄像头画面顶部/底部/边缘通常有:
- 时间戳水印（顶部或底部）
- 摄像头边缘暗角
- 固定家具区域（可配置排除）

默认ROI: 裁切画面中央 80%宽 × 85%高 区域做差分。水印区域的变化不触发motion。

---

## 5. 段索引与合并

### 5.1 段数据结构

```python
@dataclass
class Segment:
    start_time: float      # 相对当天00:00:00的秒数偏移
    end_time: float
    state: str             # "DYNAMIC" | "STATIC"
    source_file: str       # 原始文件路径
    file_start_offset: float  # 在文件内的起始秒数
```

### 5.2 跨文件合并

相邻的同状态段合并（即使跨越文件边界），减少segment碎片。

### 5.3 最小段时长

- 最短运动段: 2秒 (短于此的合并到静止段)
- 最短静止段: 30秒 (短于此且被运动段包围的，合并到运动段)

---

## 6. 渲染策略

### 6.1 输出规格

首选NVENC编码（画质好），次选QSV（多路并行时用）。Pass2 **软件解码** + **GPU缩放** + **硬件编码** 三段式管线：

```
软解(CPU) → hwupload → scale_cuda/qsv(GPU) → hwdownload → concat → NVENC/QSV(GPU编码)
```

| 参数 | NVENC | QSV | 说明 |
|------|-------|-----|------|
| 分辨率 | 1920×1080 | 1920×1080 | 后续可选4K |
| 帧率 | 20fps | 20fps | 与源保持一致 |
| 码率控制 | CQ 28 | global_quality 28 | 恒定质量模式 |
| 最大码率 | 4 Mbps | 4 Mbps | vbr_maxrate |
| 音频 | AAC 96k mono | AAC 96k mono | 源自Opus mono 48kHz |
| 像素格式 | yuv420p | nv12 | QSV输入格式差异 |
| preset | p1 | fast | 已从 p4/medium 调快 |
| 解码方式 | 软件解码(CPU) | 软件解码(CPU) | 避免per-file CUDA decoder上下文OOM |
| GPU缩放 | scale_cuda (共享CUDA device) | scale_qsv | filter链hwupload→scale→hwdownload |
| 编码 | hevc_nvenc (GPU) | hevc_qsv (GPU) | 硬件编码 |

### 6.1.1 CUDA device 共享机制

多文件渲染（98个输入 × 98个 `hwupload_cuda` filter实例）在filter graph初始化时容易触发 `CUDA_ERROR_OUT_OF_MEMORY`。同一进程内所有 `hwupload_cuda` filter 共享一个显式CUDA device即可解决：

```bash
ffmpeg -init_hw_device cuda=gpu:0 \   # 创建共享CUDA device（一次cuCtxCreate）
  -i file1 -i file2 ... -i fileN \     # 全部软件解码（无per-file hwaccel上下文）
  -filter_hw_device gpu \              # filter graph默认使用此device
  -filter_complex_script fc.txt ...    # hwupload_cuda自动派生自gpu device
```

原理：每个 `-hwaccel cuda -i` 创建独立CUDA解码器上下文（NVDEC session），98个文件 = 98个上下文 → 显存爆。改为软解后，CUDA只用于GPU缩放和编码，一个共享device + 顺序concat处理时仅一个filter活跃，显存占用恒定。

单摄像头阶段使用NVENC；双摄像头阶段启用QSV做第二路编码。静止段 GPU 路径用快进替代关键帧幻灯片（`fps` filter 与 hwdownload 帧不兼容）。

> 注：以上CQ/码率/预设为初始值，需实测后根据画质和文件大小调优。

### 6.2 运动段处理

- 视频: 原速播放 (PTS passthrough)
- 音频: 保留原始音频，从Opus转AAC（或直接复制音频流后转码）
- 分辨率: 4K→1080p lanczos缩放 (GPU scale_npp或scale_cuda)

### 6.3 静止段处理

- 视频: 按 `static_keyframe_interval` 间隔（默认30s）从静止段取关键帧，每帧显示 `keyframe_display_duration`（默认0.5s）
  - 例如：10分钟静止 = 20帧 × 0.5s = 10秒输出
- 音频: 静音 (anullsrc)

### 6.4 过渡效果

相邻运动段和静止段之间加0.3秒淡入淡出（避免硬切刺眼）。

### 6.5 日期/时间叠加

**不需要。** 摄像头素材已内嵌时间戳水印，4K→1080p缩放后依然可读。无论是运动段原帧还是静止段关键帧，水印都在。额外叠加增加渲染复杂度且无信息增益。

---

## 7. 硬件调度策略

### 7.0 场景说明

两路摄像头部署：

| 摄像头 | 位置 | 预期运动模式 |
|--------|------|-------------|
| Cam0 (00) | 客厅 | 全天活动频繁，运动段多而短 |
| Cam1 (01) | 卧室 | 白天大部分静止，宝宝睡眠/活动集中时段有运动 |

运动模式差异导致各摄像头的 Pass1 预筛过滤率不同，硬件分配需考虑负载不均。

初期可能仅引入一路摄像头素材，硬件分配方案需兼容单路和双路场景。

### 7.1 硬件资源池模型

不将摄像头绑定到特定GPU。改为资源池 + 优先级调度：

```
GPU资源池:
  3060Ti:  NVDEC解码槽×2, NVENC编码槽×1
  UHD 770: QSV解码槽×4,  QSV编码槽×1

调度策略:
  编码任务优先级: NVENC > QSV (画质更好)
  解码任务优先级: 负载低的GPU优先 (避免单GPU过载)
```

**单摄像头阶段 (仅客厅Cam0):**

```
Pass1 预筛: UHD 770 QSV×4 (轻量解码，QSV擅长并行)
Pass1.5:    NVDEC + QSV 交替分片分析，双GPU均分负载 (已实现)
            └── 按file_duration降序排列后交替分配，双线程并行
Pass2:      NVDEC+NVENC 编码输出; QSV编码槽空闲 (单路输出不可拆分)
            总体GPU利用率: 3060Ti ~83%, UHD 770 ~50%
```

Pass2阶段QSV空闲是硬件约束（单路编码不可拆分），非设计缺陷。双摄场景QSV利用率拉满。

**双摄像头阶段 (Cam0 + Cam1):**

```
Pass1 预筛: UHD 770 QSV×4, 两摄像头文件混合并行采样
Pass1.5:    NVDEC + QSV 各分析一路, 动态文件多的摄像头优先走NVDEC
Pass2:      NVENC 编码一路, QSV 编码另一路
            优先把活动多的摄像头 (通常客厅) 分给 NVENC
```

### 7.2 硬件资源清单

| 硬件 | 关键能力 | 数量 |
|------|----------|------|
| RTX 3060Ti | NVDEC (H.265 4K解码×2) + NVENC (H.265编码×1) | 1 |
| UHD 770 (i5-12600K) | QSV decode (H.265×4+) + QSV encode (H.265×1) | 1 |
| P-cores (6C/12T) | numpy向量计算, FFmpeg进程管理 | 6 |
| E-cores (4C/4T) | 轻量IO, 文件扫描 | 4 |
| SSD | 顺序读写 ~500MB/s+ | 1 |
| RAM (32GB) | 帧缓冲, numpy数组 | 32GB |

### 7.3 双GPU的价值

不用的代价：3060Ti NVDEC 同时只能解码有限路数（通常2-3路硬件解码会话），NVENC 同理。单靠独显处理300个文件的串行解码+分析+编码，整个流水线受限于独显的硬件调度队列深度。

UHD 770 的 QSV 是完全独立的编解码引擎：
- **独立的H.265硬件解码器**：可以与NVDEC同时工作，解码不同文件
- **独立的H.265硬件编码器**：可以与NVENC同时工作，编码不同输出
- **零PCIe开销**：核显与CPU共享内存控制器，帧数据无需经过PCIe总线

### 7.4 各阶段GPU分工 (实测基准)

以下基于 10 样本文件 (~3.2h 素材) 实测。单日 150 文件为线性外推。

```
单摄像头 (仅客厅Cam0):
─────────────────────────────────────────────────
Pass1 预筛
├── UHD 770 QSV×4 采样全部文件
└── 实测: 10文件 13s → 150文件 ~3.5min

Pass1.5 分析 (双GPU并行, 已实现)
├── 按file_duration降序交替分片: NVDEC 取偶索引，QSV 取奇索引
├── 实测: 6 SUSPICIOUS 文件 54s (NVDEC=3 47s, QSV=3 54s 并行)
│   对比串行NVDEC: 6文件 92s → 加速 1.7x
└── 150文件预估: ~13min (vs 串行 ~23min)

Pass2 渲染 (软件解码 + GPU缩放 + 硬件编码)
├── 软件解码(CPU) + scale_cuda GPU缩放 + NVENC硬件编码
├── CUDA device共享 (-init_hw_device) 避免多文件OOM
├── 实测: 7文件 17段 4m34s (vs 无优化 8m26s, 加速 1.9x)
└── 150文件预估: ~45min (QSV编码槽空闲，单路输出不可拆分)

单摄总计: 10文件 ~6.5min → 150文件 ~62min (vs 旧方案 ~88min)
```

### 7.5 性能对比 (实测 + 外推)

| 场景 | 策略 | Pass1 | Pass1.5 | Pass2 | 总耗时 |
|------|------|-------|---------|-------|--------|
| 单摄 10文件 | 仅独显(旧) | 13s | 92s | 506s | ~611s |
| 单摄 10文件 | 双GPU+GPU缩放 | 13s | 54s | 274s | ~341s |
| 单摄 150文件 | 仅独显(旧) | ~3.5min | ~23min | ~60min | ~88min |
| 单摄 150文件 | 双GPU+GPU缩放 | ~3.5min | ~13min | ~45min | ~62min |
| 双摄 300文件 | 双GPU+GPU缩放 | ~7min | ~22min | ~45min | ~74min |

单摄优化后节省 30%。双摄 Pass2 两路并行 QSV+NVENC，总耗时比单摄仅多 ~12min。

### 7.6 核显注意事项

- **QSV画质**：同等码率下约NVENC的85-90%。卧室vlog可接受，客厅如需更高画质可在双摄像头阶段调整编码器分配
- **QSV H.265 B-frame**：UHD 770不支持B-frame，文件略大
- **驱动**：Intel Graphics Driver ≥ 31.x，FFmpeg编译时开启 `--enable-qsv`
- **QSV编码器负载冲突**：Pass1预筛使用QSV解码时，Pass2 QSV编码需等待。Pass1与Pass2不同时运行，无实际冲突

---

## 8. 配置参数汇总

```yaml
# config/settings.yaml
paths:
  input_dir: "C:\\Users\\seeyo\\code\\homevlog-20260428\\samples"  # 唯一需配置项

pass2:
  split_render: false         # 分片渲染 (暂时关闭，负载不均衡时比单路更慢)
  hw_decode: false            # [已废弃] 软件解码+GPU缩放替代per-file硬件解码 (避免CUDA OOM)

output:
  per_camera_vlog: true
  naming: "DailyVlog_{date}_cam{index}.mp4"
  resolution: "1920x1080"
  fps: 20
  # cam0 (3060Ti NVENC)
  nv:
    codec: "hevc_nvenc"
    preset: "p1"              # p4→p1 速度提升
    cq: 28
    maxrate: "4M"
  # cam1 (UHD 770 QSV)
  qsv:
    codec: "hevc_qsv"
    preset: "fast"            # medium→fast 速度提升
    global_quality: 28
    maxrate: "4M"
  audio:
    codec: "aac"
    bitrate: "96k"
    channels: 1               # mono

detection:
  # Pass 1: 快速预筛
  prescreen_segments: 5
  prescreen_resolution: "320x180"
  prescreen_diff_threshold: 12
  prescreen_parallel: 4

  # Pass 1.5: 详细分析
  analysis_fps: 5
  analysis_resolution: "640x360"   # gray单通道 (1280x720 rgb24 pipe数据量过大, IO不可行)
  diff_threshold: 15
  motion_ratio: 0.02
  roi_crop: [0.1, 0.12, 0.8, 0.85]

  # 状态机平滑
  min_motion_frames: 3
  min_static_frames: 5
  noise_suppress_frames: 2

segment:
  min_motion_duration: 2.0
  min_static_duration: 30.0
  static_keyframe_interval: 30.0
  keyframe_display_duration: 0.5
  transition_duration: 0.3

recovery:
  scanner_freeze_minutes: 0   # 开发阶段=0, 生产环境≥10
  smb_retry_wait: 30
  smb_retry_max: 3
  file_gap_warn_threshold: 60
  render_retry_max: 2
```

---

## 9. 项目结构

```
homevlog/
├── pyproject.toml
├── .python-version          # 3.12
├── main.py                  # 入口
├── config/
│   └── settings.yaml        # 仅 input_dir 需配置
├── output/                  # 产出目录 (自动创建)
│   └── .completed           # 已完成日期标记
├── temp/                    # 临时文件 (自动创建)
│   └── segments_{date}_{cam}.json  # 段索引缓存
├── logs/                    # 日志 (自动创建)
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # 主流水线编排
│   ├── scanner.py           # 文件扫描 + 冻结检测 + 日期分组
│   ├── database.py          # SQLite任务状态管理 (WAL模式)
│   ├── prescreen.py         # Pass1 分段预筛
│   ├── detector.py          # Pass1.5 帧差分分析
│   ├── segment.py           # 段索引构建 & 合并
│   ├── renderer.py          # Pass2 FFmpeg渲染 (NVENC + QSV)
│   ├── scheduler.py         # GPU资源池调度
│   ├── timeline.py          # 时间线编辑 (select filter生成)
│   ├── ffmpeg.py            # FFmpeg进程管理封装
│   ├── monitor.py           # 性能监控 (CPU/GPU/阶段耗时)
│   └── utils.py             # 日志、配置加载、时间工具
├── tests/
│   ├── test_scanner.py
│   ├── test_prescreen.py
│   ├── test_detector.py
│   ├── test_segment.py
│   └── test_timeline.py
└── docs/
    └── SPECIFICATION.md
```

---

## 10. 实测性能基准

10 样本文件 (~3.2h, ~1.3GB) 实测，150 文件为线性外推。

### 10.1 10 文件实测

| 阶段 | 旧方案 (串行NVDEC, 软解) | 新方案 (双GPU+hw解码) | 加速比 |
|------|--------------------------|---------------------|--------|
| Pass1 预筛 | 13s | 13s | 1.0x (不变) |
| Pass1.5 分析 | 92s | 54s | 1.7x |
| Pass2 渲染 | 506s | 274s | 1.9x |
| **总计** | **611s** | **341s** | **1.8x** |

### 10.2 单日 150 文件预估

| 阶段 | 旧方案 | 新方案 | 说明 |
|------|--------|--------|------|
| Pass1 预筛 | ~3.5min | ~3.5min | QSV×4，文件多时瓶颈在 IO |
| Pass1.5 分析 | ~23min | ~13min | 双GPU均分 1.7x |
| Pass2 渲染 | ~60min | ~45min | hw_decode 1.9x，单路编码不可拆分 |
| **总计** | **~88min** | **~62min** | |

### 10.3 双摄 300 文件预估

| 阶段 | 耗时 | 说明 |
|------|------|------|
| Pass1 预筛 | ~7min | QSV×4，两摄像头混合并行 |
| Pass1.5 分析 | ~22min | NVDEC+QSV 并行，双摄像头各用双GPU |
| Pass2 渲染 | ~45min | NVENC+QSV 两路并行编码 |
| **总计** | **~74min** | |

按周末夜间8小时窗口，单摄和双摄均可轻松完成。

> Pass1.5 耗时对预筛过滤率敏感。过滤率从 80% 降到 50% 将使 Pass1.5 耗时约翻倍。以实测为准。

---

## 11. 任务状态与断点续跑

### 11.1 SQLite 任务状态机

使用 SQLite (Python `sqlite3` 标准库，WAL模式) 追踪每个文件到每个阶段的处理状态。进程随时可被终止，重启后从断点继续。

**任务粒度：**
- 每个文件一条记录，追踪 Pass1(预筛) 和 Pass1.5(分析) 的结果
- Pass2(渲染) 按日期-摄像头为单元执行，单独的状态表

**Schema：**

```sql
-- 文件级任务 (Pass1 + Pass1.5)
CREATE TABLE file_tasks (
    id INTEGER PRIMARY KEY,
    filepath TEXT NOT NULL UNIQUE,
    cam_index INTEGER NOT NULL,       -- 摄像头编号 0/1
    date TEXT NOT NULL,                -- YYYYMMDD
    file_start_time TEXT NOT NULL,     -- 文件名中的开始时间戳
    file_end_time TEXT NOT NULL,       -- 文件名中的结束时间戳
    file_duration REAL,                -- ffprobe获取的时长(秒)
    
    -- Pass1 预筛
    prescreen_status TEXT DEFAULT 'PENDING',  -- PENDING|SCREENING|STATIC|SUSPICIOUS|FAILED
    prescreen_result TEXT,                     -- JSON: {sample_diffs: [...], max_diff: N}
    
    -- Pass1.5 分析
    analysis_status TEXT DEFAULT 'PENDING',    -- PENDING|ANALYZING|ANALYZED|FAILED
    analysis_segments TEXT,                    -- JSON: [{start, end, state}, ...]
    
    retry_count INTEGER DEFAULT 0,
    error_msg TEXT,
    updated_at TEXT DEFAULT (datetime('now'))
);

-- 日期-摄像头级渲染任务 (Pass2)
CREATE TABLE render_tasks (
    id INTEGER PRIMARY KEY,
    date TEXT NOT NULL,
    cam_index INTEGER NOT NULL,
    status TEXT DEFAULT 'PENDING',     -- PENDING|RENDERING|COMPLETED|FAILED
    output_file TEXT,                  -- 产出文件路径
    retry_count INTEGER DEFAULT 0,
    error_msg TEXT,
    updated_at TEXT DEFAULT (datetime('now')),
    UNIQUE(date, cam_index)
);

CREATE INDEX idx_file_tasks_date ON file_tasks(date, cam_index);
CREATE INDEX idx_file_tasks_prescreen ON file_tasks(prescreen_status);
CREATE INDEX idx_file_tasks_analysis ON file_tasks(analysis_status);
CREATE INDEX idx_render_tasks_status ON render_tasks(status);
```

**状态流转：**

```
Scanner:
  DISCOVERED → INSERT file_tasks (filepath, cam_index, date, ...)

Pass1 (prescreen):
  PENDING → SCREENING → STATIC (跳过Pass1.5)
                      → SUSPICIOUS (进入Pass1.5)
                      → FAILED (重试超限)

Pass1.5 (analysis):
  SUSPICIOUS + PENDING → ANALYZING → ANALYZED
                                   → FAILED

Pass2 (render):
  该date-cam下所有file_tasks的prescreen状态已确定后
  → 检查render_tasks是否已有COMPLETED记录，有则跳过
  → PENDING → RENDERING → COMPLETED → 写入output/.completed标记
                        → FAILED

恢复流程:
  启动 → 扫描所有未COMPLETED的render_tasks
       → 检查其date-cam下的file_tasks
       → STATIC状态的跳过Pass1.5，SUSPICIOUS+PENDING的重跑Pass1.5
       → PENDING/RENDERING的render_task重跑Pass2
       → FAILED的检查retry_count，未超限的重试
```

### 11.2 断点续跑场景

**场景A: 大批量积压素材首次处理**

```
启动 → Scanner扫描全部文件(date范围可能跨度多周)
     → 按date分组，按时间从旧到新处理
     → 每天两个render_task(cam0, cam1)
     → 任意时刻Ctrl+C或崩溃：
        当前正在RENDERING的date-cam回退(最多损失一个Pass2，~30分钟)
        已COMPLETED的date-cam不受影响
        其他PENDING的date-cam等待恢复后继续
     → 重启进程自动从最早的未完成date继续
```

**场景B: 周末定时运行处理当周素材**

```
启动 → Scanner扫描 → 发现新文件，INSERT或跳过已存在记录
     → 跳过已COMPLETED的date-cam
     → 处理本周新date-cam
```

### 11.3 与旧方案SQLite设计的差异

旧方案有完整的 `PENDING→ANALYZING→ANALYZED→ENCODING→COMPLETED→ARCHIVED` 六阶段流转，支持单个文件独立渲染。

本方案简化：
- 文件不独立渲染（Pass2是date-cam级，一次性ffmpeg处理全天素材）
- 不需要 ENCODING/ARCHIVED 等细分状态
- 不需要旧方案的 `get_next_pending_task()` 原子争抢（单进程顺序处理，无worker pool竞争）

旧方案SQLite承担了"多worker协调"的职责，本方案SQLite只承担"crash-safe断点续跑"。

### 11.4 文件扫描冻结检测

SMB共享上的素材文件由摄像头持续写入。必须确保只处理已写入完成的文件。

```
对每个.mp4文件:
  if 文件修改时间 < scanner_freeze_minutes 分钟前:
    跳过 (仍在写入窗口内)
  if 文件大小 != 1秒后再次检查的文件大小:
    跳过 (文件仍在增长中)
  通过冻结检测 → 写入 file_tasks 表
```

Scanner启动时和定时巡检使用此逻辑。已存在 `file_tasks` 中的文件跳过（UNIQUE约束防重复）。

### 11.5 UNC路径断连重试

**Pass1/Pass1.5 阶段** (单文件操作):
- FFmpeg子进程捕获stderr中 `I/O error` / `No such file` 等断连特征
- 等待 `smb_retry_wait` 秒重试，最多 `smb_retry_max` 次
- 超限 → `file_tasks` 对应状态标记为 FAILED，记录 error_msg，继续下一文件

**Pass2 渲染阶段** (长时间操作):
- 渲染前全量文件可读性预检（每个文件 `ffprobe` 一次），不可读立即告警
- 渲染中断连：`render_tasks` 标记 FAILED，等待下次启动恢复时重试
- 超过 `render_retry_max` 次 → 永久 FAILED，需人工介入

### 11.6 文件间时间线连续性

文件名格式 `{cam}_YYYYMMDDHHMMSS_YYYYMMDDHHMMSS.mp4`，内含开始和结束时间戳。拼接规则：

- 同一摄像头文件按 `file_start_time` 升序排列
- 前文件 `file_end_time` 与后文件 `file_start_time` 之间的间隙：标记为STATIC段
- 若时间重叠（end > 下一段start）：后文件裁剪掉重叠部分
- 若间隙 > `file_gap_warn_threshold`（默认60秒）：视为异常，记录告警

### 11.7 跨天素材归属

素材文件名含开始和结束时间戳。若结束时间跨凌晨：

```
例: 00_20260424_235500_20260425_001500.mp4
    开始 2026-04-24 23:55:00, 结束 2026-04-25 00:15:00
```

处理规则：**简单视作起始时间所在天的素材。** 上例归属 `20260424`。

### 11.8 多日批量处理

周末一次性处理当周5天素材。策略：

- 按日期串行处理（每天独立产出vlog，不跨天混合）
- 处理顺序：从旧到新（历史数据优先）
- 前一天的Pass1.5段索引保留在 `temp/`，相邻两天的边界segment可参考（跨天运动段不切断，移交次日vlog开头）
- 每天完成后写入 `output/.completed`，避免重复处理
- 5天全部完成后汇总日志摘要（每日期vlog时长、压缩比、异常告警）

## 12. 实施计划

### Phase 1: 项目骨架 + 基础设施
- uv 项目初始化，pyproject.toml 依赖声明 (numpy, opencv-python, pyyaml, psutil)
- config/settings.yaml + src/utils.py 配置加载
- src/database.py: SQLite WAL模式、schema建表、原子状态转换、查询未完成任务
- src/ffmpeg.py: FFmpeg子进程管理 (异步pipe读取、超时终止、stderr解析)
- src/monitor.py: 各阶段耗时日志 + GPU利用率采集 (pynvml + psutil)
- 验证: 初始化db、用样本素材测试 NVDEC/QSV 解码pipe

### Phase 2: 文件扫描 + Pass1 预筛
- src/scanner.py: UNC路径文件枚举、文件名解析 (cam/start/end)、日期分组、冻结检测 (mtime+size)、写入 file_tasks 表
- src/prescreen.py: 分段采样、QSV轻量解码、numpy帧差分、STATIC/SUSPICIOUS分类、更新 file_tasks.prescreen_status
- ThreadPoolExecutor并行调度 (prescreen_parallel=4)
- 验证: 对样本目录扫描入库、运行预筛，检查db状态流转正确

### Phase 3: Pass1.5 详细分析
- src/detector.py: NVDEC/QSV 720p@5fps解码 → numpy批量帧差分 → 逐帧motion判定
- ROI裁切 (roi_crop参数) + 状态机平滑 (min_motion_frames等)
- src/segment.py: Segment数据结构、跨文件合并、最小段时长过滤
- 双GPU并行: scheduler分配SUSPICIOUS文件到NVDEC/QSV队列
- 结果写入 file_tasks.analysis_segments (JSON), 更新 analysis_status=ANALYZED
- 验证: 对样本文件运行detector，db查询确认段索引写入、与人工标注对比准确率

### Phase 4: Pass2 渲染
- src/timeline.py: 从db读取某date-cam下所有file_tasks的analysis_segments → FFmpeg select filter表达式
- src/renderer.py: NVENC和QSV两路渲染器、音频处理 (运动段Opus→AAC，静止段anullsrc)
- src/scheduler.py: GPU资源池，检查render_tasks状态跳过已完成，单摄NVENC、双摄QSV并行
- 过渡效果 (xfade filter, transition_duration)
- 完成后: render_tasks.status=COMPLETED, 写入output/.completed
- 验证: 渲染样本全天素材，确认db状态更新和产出文件

### Phase 5: 流水线集成 + 容错
- src/pipeline.py: 串联 Pass1→1.5→2，单date-cam完整流水线，从db读取/更新状态
- main.py: 多日批量入口 (扫描入库→查询未完成render_tasks→按日调度→汇总日志)
- 断点恢复: 启动时检查db中所有PENDING/FAILED任务，跳过COMPLETED，从最早未完成date继续
- 容错: UNC断连重试、渲染失败重试（自动回退render_tasks状态）
- 参数调优: 根据实测调整 diff_threshold/motion_ratio/CQ 等参数
- 验证: 中途kill进程后重启，确认从断点继续；端到端产出 DailyVlog

---

## 13. 已确认事项

| 事项 | 决策 |
|------|------|
| NAS挂载路径 | UNC路径，非盘符映射 |
| SSD可用空间 | 充足（峰值约需素材2倍） |
| 输出方式 | 每个摄像头独立产出 DailyVlog |
| 音频策略 | 运动段保留原始音频转AAC，静止段静音 |
| 输出分辨率 | 1080p H.265，后续可选4K |
| 夜间红外模式 | 默认参数不变，待后续实测调优 |

## 14. 待后续验证

1. 红外模式画面偏灰度，当前 `diff_threshold: 15` 可能需要下调（灰度图对比度低，差值更小）
2. ROI 裁切参数需根据 C700 水印位置实测调整
3. UNC路径下 FFmpeg 的 SMB 读取性能实测
4. QSV 和 NVENC 输出画质对比，决定是否需要统一编码器
