# HomeVlog Architecture Decisions

本文档记录 HomeVlog 中需要跨会话长期保留的架构与工程决策。

它回答：

> 为什么系统采用当前设计，以及哪些替代方案已经被验证、否决或替代。

当前有效架构事实以以下文档为准：

```text
docs/ARCHITECTURE.md
```

性能测试过程、硬件基准和生产遥测以：

```text
docs/PROGRESS.md
```

和：

```text
docs/ARCHITECTURE.md
```

为准。

ADR 记录决策历史，不应被当作当前实现说明书。

---

# 1. ADR Status

允许的状态：

| Status | 含义 |
|---|---|
| Proposed | 尚未正式采用 |
| Accepted | 当前仍然有效 |
| Superseded | 已被后续决策替代 |
| Deprecated | 仍可能存在，但不应继续扩展 |
| Rejected | 已评估但明确不采用 |

当一个 ADR 被后续 ADR 替代时，不删除旧记录。

应显式维护：

```text
Superseded by: ADR xxxx
```

以保留设计演进过程。

---

# 2. Decision Index

| ADR | Decision | Status | Date | Relationship |
|---|---|---|---|---|
| ADR 0001 | 单次灰度解码与内存候选帧流转 | Accepted | 2026-03 | — |
| ADR 0002 | 单一 NVENC 同构渲染策略 | Superseded | 2026-03 | Superseded by ADR 0008 |
| ADR 0003 | 人机协同主动学习闭环 | Accepted | 2026-03 | — |
| ADR 0004 | 原子渲染批次与确定性时间轴 | Accepted | 2026-03 | — |
| ADR 0005 | 硬件并发对齐、NVDEC 借调与强类型 PipelineTask | Partially Superseded | 2026-03 | Scheduling 部分演进至 ADR 0007；PipelineTask 保持有效 |
| ADR 0006 | 命名模板、低照度抗噪与 Filter Graph 健壮性 | Accepted | 2026-04 | — |
| ADR 0007 | NVENC/NVDEC 资源解耦与流水线错峰 | Accepted | 2026-04 | Evolves ADR 0005 |
| ADR 0008 | NVENC + QSV 异构流式渲染 | Accepted | 2026-04 | Supersedes ADR 0002 |
| ADR 0009 | 双端渲染工作窃取与基于真实成本的调度 | Accepted | 2026-04 | Evolves ADR 0008 |
| ADR 0010 | Analysis SJF 与 QSV 弹性窃取策略 | Accepted | 2026-04 | Evolves ADR 0009 |
| ADR 0011 | 进程生命周期下沉、配置解耦与架构瘦身治理 | Accepted | 2026-04 | 解耦 FFmpegProcessRegistry 与配置加载器 |
| ADR 0012 | 四级自适应阶梯速率模型与防漏检保全 | Accepted | 2026-09 | — |
| ADR 0013 | 五级自适应睡眠浓缩与滤镜图硬截断不变量 | Accepted | 2026-09 | Evolves ADR 0004, ADR 0012 |
| ADR 0014 | 异构分析与渲染 QSV 硬件信号量池隔离 | Accepted | 2026-09 | Evolves ADR 0008, ADR 0010 |
| ADR 0015 | 白天漫射光影偏转软抑制与吸收短动态审计保全 | Accepted | 2026-09 | Evolves ADR 0006, ADR 0012 |
| ADR 0016 | 流式批次时间轴稳定性与源任务完成门禁 | Accepted | 2026-09 | Evolves ADR 0004, ADR 0008 |
| ADR 0017 | 离线高质量编码重构与多特征迟滞动作判定 | Accepted | 2026-09 | Evolves ADR 0008, ADR 0012, ADR 0013 |
| ADR 0023 | 交付产物原子提交与共享媒体契约 | Accepted | 2026-09 | Evolves ADR 0004, ADR 0013, ADR 0020, ADR 0022 |

---

# 3. ADR 0001 — Single-Pass Grayscale Analysis

**Status:** Accepted  
**Date:** 2026-03

## Context

早期分析流水线存在多个重复成本：

- Prescreen 多次 Seek；
- RGB 解码后再由 CPU 转灰度；
- YOLO 重新读取并解码源视频；
- 候选帧通过大量临时图片在阶段之间流转。

这使 NAS 读取、视频解码和磁盘 I/O 成为重复开销。

## Decision

分析流水线采用：

1. 单通道灰度解码作为运动分析数据契约；
2. 候选帧以内存压缩表示流转；
3. YOLO 复用分析阶段产生的候选帧，而不是重新完整解码源文件；
4. 灰度帧直接进入时域与空间运动分析。

架构目标是：

```text
decode once
→ derive multiple analysis signals
→ selectively invoke expensive inference
```

## Consequences

### Positive

- 减少重复媒体读取与解码；
- 降低 CPU 色彩空间转换成本；
- 避免中间临时文件 I/O；
- 为多消费者复用建立统一分析数据流。

### Constraints

后续新增分析消费者应优先复用已有解码输出。

独立重新完整解码源视频需要明确的性能或隔离理由。

## Evidence

性能结果与基准测试记录于：

```text
docs/PROGRESS.md
```

---

# 4. ADR 0002 — NVENC-Only Rendering

**Status:** Superseded  
**Date:** 2026-03  
**Superseded by:** ADR 0008

## Context

早期尝试同时使用：

```text
NVENC
+
QSV
```

生成可直接拼接的 HEVC 批次。

实际测试中，不同硬件编码器生成的视频参数集和像素格式不一致。

在 MP4 stream-copy 合并过程中，后续批次所需的参数集未可靠保留，导致播放器在硬件编码器切换位置无法继续正确解码。

## Decision

当时采用：

```text
render_gpu_policy = nv_only
```

统一使用 NVENC 生成最终可拼接批次。

## Reasoning

该策略牺牲部分可用核显算力，以换取：

- 编码参数一致；
- 拼接稳定；
- 播放器兼容性；
- 更简单的故障模型。

## Superseded

后续实验确认问题根因并非“不同硬件编码器天生不能拼接”，而是参数集与像素格式处理方式。

ADR 0008 通过：

```text
consistent pixel format
+
in-band parameter-set injection
```

解决这一限制。

因此：

> “最终渲染必须只使用 NVENC”已经失效。

旧 ADR 保留用于避免未来重新误判该问题。

---

# 5. ADR 0003 — Human-in-the-Loop Active Learning

**Status:** Accepted  
**Date:** 2026-03

## Context

家庭监控环境存在：

- 红外噪声；
- 光照突变；
- 边缘小目标；
- 弱运动；
- 音画判断冲突。

这些情况无法仅通过不断调整固定阈值可靠解决。

同时，用户人工审核产生了高价值监督信号。

## Decision

建立完整反馈闭环：

```text
algorithm
→ uncertain / suspicious samples
→ human review
→ persistent human label
→ timeline correction
→ optional training dataset
```

核心原则：

1. 主动召回疑难样本，而不是随机抽样；
2. 人工判断作为独立持久事实保存；
3. Timeline 重建时人工判断优先于算法判断；
4. 人工样本可以进入后续模型训练数据集；
5. 数据集必须包含负样本，防止只强化正检测能力。

## Consequences

系统由一次性算法流水线演进为持续反馈系统。

人工标签必须与算法状态解耦，否则重新分析会破坏监督数据。

## Training Parameters

具体训练参数，例如：

```text
freeze=10
```

属于当前训练配置，而不是本 ADR 的永久架构要求。

本 ADR 保证的是：

> 训练过程必须有防止误报退化的负样本与验证机制。

---

# 6. ADR 0004 — Atomic Streaming Render and Deterministic Timeline

**Status:** Accepted  
**Date:** 2026-03

## Context

将全天视频作为单个长渲染任务存在明显风险：

```text
long render
→ interruption
→ entire output lost
```

同时，早期时间轴计算存在：

- 源视频末端覆盖不完整；
- 展示时长计算重复；
- 字幕和渲染时间线不同步；
- 静态与动态切换突兀。

## Decision

### Atomic Render Unit

将最终渲染拆成独立可验证批次。

```text
_batchX.tmp.mp4
→ validation
→ atomic promotion
→ _batchX.mp4
```

### Resume

已经验证完成的正式批次在普通清理或重新运行过程中保留。

### Timeline SSOT

所有展示时间相关消费者共享统一 Display Plan。

```text
compute_display_plans()
```

负责产生展示时长计划。

### Timeline Closure

分析产生的最后边界必须完整闭合到源媒体有效时间范围。

### Physical Reconciliation

最终合并前必须保证：

```text
dispatched
==
produced ∪ terminal
```

## Consequences

渲染成为可恢复事务流，而不是一次性长作业。

“所有任务已经发送”不能被视为“任务已经完成”。

---

# 7. ADR 0005 — Resource Alignment and Typed Pipeline Tasks

**Status:** Partially Superseded  
**Date:** 2026-03  
**Scheduling evolved by:** ADR 0007

## Context

早期系统暴露三个问题：

1. Worker 数量大于实际硬件资源并发能力；
2. NVDEC 在部分阶段处于闲置状态；
3. Pipeline 阶段之间通过松散字典传递任务状态。

第三点曾导致不同阶段对默认字段解释不一致。

## Decision

### Hardware Concurrency Alignment

Worker 数量必须与实际受控硬件并发能力协调。

禁止通过增加线程数绕过硬件资源信号量。

### Cooperative Decode

允许在安全条件下使用闲置 NVDEC 补充分析解码能力。

该部分的具体资源模型后来由 ADR 0007 进一步演进。

### PipelineTask

建立强类型跨阶段任务对象：

```text
PipelineTask
```

替代自由结构的裸字典。

跨阶段共享的重要字段应：

- 明确定义；
- 强类型；
- 具有一致默认语义；
- 尽可能保持不可变。

## Current Validity

以下部分仍然有效：

```text
PipelineTask contract
hardware concurrency must be centrally controlled
```

早期 NV 统一信号量模型已被 ADR 0007 替代。

---

# 8. ADR 0006 — Robust Prescreen and Render Graph Safety

**Status:** Accepted  
**Date:** 2026-04

## Context

多个相互独立的问题暴露出局部实现缺乏健壮性：

- 输出文件命名规则散落；
- 红外夜视噪声触发大量错误运动；
- 超复杂 FFmpeg `select` 表达式造成递归深度异常；
- 视频提前 EOF 导致过于严格的有效性判断；
- SQLite 并发写入存在锁竞争。

## Decision

### Centralized Output Naming

所有最终成片名称通过统一解析函数产生。

避免主程序、审核工具等模块独立复制命名逻辑。

### Spatial Noise Discrimination

低照度运动判断同时考虑：

```text
global energy
+
spatial concentration
+
local peak energy
```

而不是简单抬高全局阈值。

### Filter Graph Complexity Bound

复杂 `select` 条件首先执行区间融合，然后构造平衡表达式。

当复杂度超过安全范围时：

```text
optimized sparse decode
→ fallback
→ conventional decode
```

系统稳定性优先于局部优化。

### Storage Robustness

数据库连接与视频合法性检查允许合理的运行环境容差。

## Consequences

该 ADR 建立了一个更通用的原则：

> 优化路径必须存在复杂度上限和安全退化路径。

具体阈值属于实现和配置，不属于永久架构约束。

---

# 9. ADR 0007 — Separate NVENC/NVDEC Resource Domains

**Status:** Accepted  
**Date:** 2026-04  
**Evolves:** ADR 0005

## Context

早期单一 NVIDIA 信号量同时管理：

```text
NVENC
NVDEC
```

实际硬件遥测显示，两类工作可以在受控条件下并行执行。

统一锁导致：

- 编码与解码发生不必要互斥；
- Analysis Queue 积压；
- 硬件空转；
- 无意义等待。

此外，过大的动态区间合并阈值导致大量实际静态时间被作为动态视频正常速度渲染。

## Decision

### Resource Separation

将 GPU 资源抽象拆分为：

```text
NVENC
NVDEC
```

分别管理。

业务模块不得重新将二者通过单一互斥资源绑定。

### VRAM Safety

虽然编码和解码资源独立，但共享显存，因此仍保留全局显存保护机制。

### Dynamic Gap Semantics

只应合并语义上连续的短运动间隙。

长静态区间必须重新交给静态快路径，而不能因为调度便利被视为动态画面。

### Pipeline Overlap

分析与渲染允许在资源安全时重叠执行，而不是强制严格阶段串行。

## Consequences

Scheduler 从：

```text
device-level lock
```

演进为：

```text
resource-domain scheduling
```

这是后续异构 Work-Stealing 的基础。

---

# 10. ADR 0008 — Heterogeneous NVENC/QSV Streaming Render

**Status:** Accepted  
**Date:** 2026-04  
**Supersedes:** ADR 0002

## Context

仅使用 NVENC 后，分析阶段完成时：

- Intel QSV 长时间闲置；
- NVENC 仍有大量积压批次；
- Pipeline makespan 由尾部渲染决定。

此前禁止 NVENC/QSV 混合输出，是因为批次切换时存在 HEVC 参数集兼容问题。

后续实验定位到实际问题来自：

- 参数集未可靠随流携带；
- 像素格式不统一。

## Decision

### Cross-Encoder Compatibility

NVENC 和 QSV 输出统一关键编码契约，包括像素格式。

需要确保 HEVC 参数集能够随流携带，使播放器在批次切换时可以完成 Decoder Reconfiguration。

### Heterogeneous Rendering

允许：

```text
2 × NVENC
+
1 × QSV
```

作为并行渲染资源。

### Cost-Aware Assignment

较慢的 QSV 不应接管可能形成全局尾部的大任务。

高速 NVENC 优先承担重任务。

### Tail Guard

在接近 Pipeline 结束时，允许 QSV 停止继续获取新任务，使高速 Worker 完成最后收尾。

## Consequences

系统优化目标从：

```text
maximize individual device utilization
```

转变为：

```text
minimize end-to-end makespan
```

这成为后续 ADR 0009 调度策略的基础。

---

# 11. ADR 0009 — Dual-Ended Cost-Aware Render Scheduling

**Status:** Accepted  
**Date:** 2026-04  
**Evolves:** ADR 0008

## Context

初始异构调度虽然激活了 QSV，但存在明显队头阻塞：

- 任务成本使用源视频总时长估算；
- QSV 遇到超出自身能力的大任务后反复拒绝；
- 后方实际适合 QSV 的轻量任务无法被获取；
- NVENC 和 QSV 的速度差异没有充分体现在队列结构中。

生产遥测证明，源文件总时长不能代表实际渲染成本。fileciteturn0file0L183-L196

## Decision

### Render Cost

渲染成本以：

```text
dynamic_duration
```

等更接近实际编码工作量的指标计算，而不是使用源文件完整时长。

### Dual-Ended Queue

使用双端成本队列：

```text
heavy end ←────────────→ light end
     ↑                         ↑
   NVENC                      QSV
```

NVENC 优先：

```text
pop_heaviest()
```

QSV 优先：

```text
static
→ steal_lightest()
```

### Adaptive Eligibility

QSV 可以承担的最大任务成本根据：

- 当前队列深度；
- 剩余任务数量；
- 硬件速度关系；

动态变化。

### Tail Protection

尾部任务量较低时，限制 QSV 获取可能拖慢全局结束时间的新任务。

## Consequences

调度由固定类型分配：

```text
static → QSV
dynamic → NVENC
```

演进为：

```text
actual cost
+
worker capability
+
queue state
+
remaining makespan
```

联合决策。

具体阈值属于调优参数。

---

# 12. ADR 0010 — Analysis SJF and Elastic QSV Stealing

**Status:** Accepted  
**Date:** 2026-04  
**Evolves:** ADR 0009

## Context

端到端 Pipeline 性能并不仅由单个阶段吞吐决定。

生产遥测发现：

### Analysis Head-of-Line Blocking

Analysis Queue 曾优先处理超长视频。

少量超长任务长期占用全部分析 Worker，使短视频不能及时完成分析。

结果：

```text
render queue starvation
→ NVENC idle
```

### QSV Tail Underutilization

QSV 的任务准入阈值没有正确反映：

- 队列剩余规模；
- QSV 实际处理速度；
- 动态视频时长与墙钟执行时间之间的关系。

因此在还有大量适合工作的情况下提前闲置。fileciteturn0file0L207-L223

## Decision

### Analysis Queue Uses SJF

Analysis Queue 优先处理较短素材。

核心目标不是：

```text
minimize average analysis task time
```

而是：

```text
produce renderable work early
→ keep downstream hardware fed
```

超长素材可以在后续持续处理。

### Elastic QSV Eligibility

QSV 可接受任务的成本上限随渲染队列积压动态变化：

```text
large backlog
→ allow moderately larger QSV work

small backlog
→ tighten QSV eligibility
```

具体公式属于当前实现和调优参数，不作为架构永久约束。

### Shared-Bus Awareness

QSV 渲染还必须考虑当前 Analysis Queue 对 Intel 媒体资源和共享总线的使用情况。

## Consequences

Pipeline Scheduler 不再单独优化：

```text
analysis throughput
```

或：

```text
render throughput
```

而是优化：

```text
end-to-end pipeline makespan
```

这意味着局部看似“不公平”的策略，例如 SJF、Tail Guard 或主动让较慢 Worker 闲置，在降低最终完成时间时是合理的。

---

# 13. ADR 0011 — Process Lifecycle Separation, Config Isolation & Technical Debt Remediation

**Status:** Accepted  
**Date:** 2026-04  

## Context

在长周期的异构多阶段演进过程中，代码库积累了较为显著的局部技术债：
1. **依赖倒置与隐式循环引用**：`FFmpegProcessRegistry` 最初随渲染器原型定义在顶层业务模块 `src/renderer.py` 中。然而由于基础设施进程管控的铁律（Ctrl+C 安全中断、所有子进程生命周期统一治理），底层的 `src.ffmpeg`、`src.scheduler`、`src.detector`、`src.prescreen` 及 `main.py` 均需要引用该类，导致底层向顶层依赖倒置，只能在函数体内使用局部延迟导入 (`from src.renderer import FFmpegProcessRegistry`) 兜底。
2. **配置层耦合**：`load_config` 与路径常量定义在 `src.utils`，而 `src.utils` 导出了 `scheduler` 的并发信号量，造成模块互相交叉引用。
3. **算法与数据模型冗余**：`src/segment.py` 的短分段平滑逻辑中存在两段完全一致的相邻同状态合并代码；`src/timeline.py` 的 `build_timeline_from_rows` 中对普通分段、分析分段及静态回退的解析与解构逻辑存在数十行重复。
4. **边缘 NAS 探测回退漏检**：`src/detector.py` 在判断文件元数据缓存时，对无音频文件（`has_audio == 0`）误触发 NAS `av.open` 远程容器重探测。
5. **数据库初始化开销**：`_migrate()` 在每次数据库连接初始化时均执行 `BEGIN IMMEDIATE` 与 `PRAGMA table_info`，且 `get_all_file_tasks_for_date` 对全表 `human_reviews` 进行全表扫描。

## Decision

1. **下沉基础生命周期管控**：将 `FFmpegProcessRegistry` 统一下沉至底层封装模块 `src/ffmpeg.py`，保持基础设施自包含与干净闭环；`src/renderer.py` 仅作为向后兼容 re-export。
2. **独立配置核心层**：将配置加载器 `load_config`、验证逻辑与基础工程路径常量整合至专职模块 `src/config.py`，彻底消除 `utils` 与 `scheduler` 间的循环引用。
3. **纯函数抽取与 DRY 规范**：
   - `src/segment.py` 严格统一复用 `_merge_same_state`；
   - `src/timeline.py` 抽取单一职责的 `_parse_row_segments` 统筹行记录提取与人工审核标签优先级判定。
4. **数据库模式与查询收敛**：
   - 基础建表语句包含全部业务字段；
   - 增加 `PRAGMA user_version = 2` 版本门禁，跳过无谓的每次初始化迁移检查；
   - 查询通过 `filepath IN (...)` 参数化精准过滤。
5. **修复边界探测条件**：严格区分“未获取音频状态”与“已确认为无音频（0）”，彻底杜绝 NAS 冗余 I/O。

## Consequences

- 根除了代码库各核心模块函数内数十处临时的 `from src.xxx import ...` 延迟导入，静态编译与测试加载时间显著缩短。
- 逻辑单点化，消除分段合并与时间轴重构可能出现的分歧与漂移。
- 真实素材压测证实无任何性能倒退（20260320 保持 55.6x~57.4x 高速稳态，端到端产出 100% 完整）。

---

# 14. ADR 0012 — Four-Tier Adaptive Rate Model and Anti-Leakage Video Preservation

**Status:** Accepted  
**Date:** 2026-09  

## Context

在 HomeVlog 的核心愿景中，DailyVlog 生成完毕后用户需要能够安全物理删除 NAS 上的海量原始素材以释放备份压力。
在初始实现中，系统采用极端的“动静二值化”策略：
- 判定为 `DYNAMIC` 的素材以 1x 常速原画保留；
- 判定为 `STATIC` 的素材以 55s 抽 1 帧的幻灯片速率粗暴压缩。

在对生产素材（`20260320`，24.81 小时）的穿透式实测审计中，发现 44 起 P0 级严重漏检：
1. **静坐陪伴被丢弃**：看护人坐定看手机或照料婴儿停顿超过 8 秒时，被粗暴切入 STATIC 抽帧，人物在成片中凭空消失数十秒；
2. **夜间微动/遮挡未保全**：红外夜视下睡眠翻身或手足轻微活动虽然能量显著（$\text{energy} \ge 2.5$），但因 YOLO 无法框选人形而被悲观降级为 STATIC 丢弃；
3. **物理切片边界截断**：由于视频录制每 5 分钟切分物理文件，单文件独立计算时，位于文件交界处的静坐停顿因丢失上下文而被误判为独立静态。

若直接将所有静态帧保留，成片时长将膨胀数倍；若不加固，则严禁删除原始素材。

## Decision

1. **确立四级自适应阶梯浓缩模型**：
   - `DYNAMIC` / `DYNAMIC_AUDIO`: **1.0x 常速原画**（运动与声音核心事件，无损保全）；
   - `PRESENCE`: **4.0x 温和快进**（基于时序因果链，识别并在实体解码流中保全静坐、看书、陪伴状态）；
   - `MICRO_MOTION`: **16.0x 巡航 + 3.0s 动作锚点**（差分能量 $\ge 2.5$ 的有效物理微动，事件驱动浓缩）；
   - `STATIC`: **55.0s 抽 1 帧**（真正无人、无声、深夜深度睡眠静止时段，极限浓缩）。
2. **时序因果链与跨文件传递 (Cross-File Presence Propagation)**：
   - 升级 `resolve_presence_segments`，将因果链前置判定扩展至所有活动事件（`ACTIVE_STATES = {"DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE", "MICRO_MOTION"}`）；
   - 在时间线聚合层 (`src/stages/timeline.py`) 针对日内全局序列执行跨文件因果链传递，彻底消除 5 分钟文件切片边界造成的上下文截断；
   - 因果链计算完毕后，通过 `split_segments_at_file_boundaries` 严格投影回物理文件边界，保障 Virtual Concat 寻道安全。
3. **渲染端实体集扩充**：
   - `ACTIVE_STATES` 统一纳入 Virtual Concat 连续与混合解码流程，禁止将 PRESENCE 与 MICRO_MOTION 视作静态忽略。

## Consequences

- **P0 致命漏检彻底归零**：`20260320` 实测审计中，高危静态切片 P0 漏检从 44 起骤降至 0 起，最终裁决正式转为 `PASS_SAFE_TO_DELETE`（允许安全删除原素材）。
- **成片时长严格受控**：加固后全天展示时长从 18,193s (5.05h) 微调至 20,508s (5.70h)，增幅仅为 +12.73% (+38.6 分钟)，完美落在 5%~15% 预设控温区间内。
- **用户信心保障**：建立起“自动化分层超敏审计套件 (`scripts/audit_leakage_20260320.py`)”，提供客观量化数据支撑物理删除决策。

---

# 15. ADR 0013 — Five-Tier Adaptive Sleep Compression and Filter Graph Duration Invariance

**Status:** Accepted  
**Date:** 2026-09  
**Evolves:** ADR 0004, ADR 0012

## Context

在多日真实生产素材（2026-03-20 至 2026-04-05，共 384.3 小时 4K 素材）连续实测中暴露两大关键设计痛点：

1. **夜间熟睡过度膨胀成片 (Sleep Inflation)**：
   实测数据显示，全量日均 55.3 小时的 `TARGET_PERSISTENCE` 中有 **62.3%（38.11 小时）发生于夜间 23:00~07:00**。按照原四级速率模型的 `PRESENCE`（4.0x 温和快进）渲染，导致每天产出 60~90 分钟的静止熟睡录像，每日成片膨胀至 5~7 小时（单片 6~8 GB），极大地降低了家庭 Vlog 的观赏价值与流媒体分享便利性。
2. **稀疏混合解码中的 FFmpeg EOF 时间戳外溢 (EOF PTS Leakage Bug)**：
   在 20260331 渲染 Batch 113 时，切片因输入流经 `sparse_mixed` 跳过了无运动区间，非纯静态段（`PRESENCE` / `MICRO_MOTION`）后续挂载 `fps` 滤镜时，`trim=start=s:end=e` 遇到跳跃区间发射 EOF 时将下游链路的 `eof_pts` 传递为跳跃后的未来时间戳，导致 `fps` 滤镜克隆末帧数百次（单段异常膨胀 16.6 秒），造成整批展示时长超限被 `valid_video` 丢弃并中断全日合成。

## Decision

1. **演进五级自适应阶梯浓缩模型 (Five-Tier Adaptive Rate Model)**：
   - `DYNAMIC` / `DYNAMIC_AUDIO`: **1.0x 常速原画**（运动与声音核心事件，无损保全）；
   - `PRESENCE`: **4.0x 温和快进**（白天有人陪伴、静坐看护状态）；
   - `NIGHT_STATIONARY`: **16.0x 高倍浓缩**（夜间 23:00~07:00 熟睡静止或全天持续低能量 $\ge 180$s 静止，兼具 speed ramping 平滑缓入缓出）；
   - `MICRO_MOTION`: **16.0x 巡航 + 3.0s 动作锚点**（翻身、肢体微动事件驱动保留）；
   - `STATIC`: **55.0s 抽 1 帧**（真正无人静止）。
2. **时间轴滤镜图时长硬截断不变量 (Duration Clamping Invariant)**：
   在 `src/stages/timeline.py:build_concat_filter` 中，对于所有非纯静态段，在 `fps` 滤镜之后显式追加 `trim=duration={actual_display_dur:.3f},setpts=PTS-STARTPTS`，硬性切断 FFmpeg 的 EOF 时间戳外溢，确保成片时长与 `compute_display_plans` 达到毫秒级数学同源。

## Consequences

- **成片虚胖彻底根治**：多日全量回测显示，4,295 个夜间熟睡段落（38.11 小时）成功升级为 `NIGHT_STATIONARY`，成片累计节省 7.15 小时无效熟睡视频（减少 46.7% 的驻留时长），每日成片精炼至 1.5~2.5 小时。
- **20260331 断点无损恢复**：修复后 Batch 113 渲染精确度达到 100.0%，并一键成功合成了缺失的 `DailyVlog_20260331_B888805AA3CD.mp4`（6.87 GB）。

---

# 16. ADR 0014 — Dedicated QSV Hardware Semaphore Slot Isolation

**Status:** Accepted  
**Date:** 2026-09  
**Evolves:** ADR 0008, ADR 0010

## Context

在多日生产压测中，分析阶段配置为 `detection.analysis_max_workers: 8`，硬件配置为 `hardware.max_qsv_concurrency: 8`。当 8 个分析 Worker 全速并发且遇到密集 4K 解码时，全局 QSV 信号量租约完全耗尽，导致 `qsv_0` 渲染 Worker 无法申请到解码租约，频繁在日志中产生 30s~90s 的超时排队告警，严重拖慢了流水线的端到端吞吐。

## Decision

1. **信号量槽位物理隔离**：
   在 `src/hardware/scheduler.py` 中拆分 QSV 信号量域：
   - 导出 `get_qsv_render_semaphore()`（上限固定为 1），专门供 `qsv_0` 渲染 Worker 独占；
   - 将分析阶段使用的 `get_qsv_semaphore()` 上限定为 `max(1, max_qsv_concurrency - 1) = 7`。
2. **保底租约优先权**：
   无论分析 Worker 并发多么饱和，始终为渲染通道保留至少 1 个硬件上下文，根除跨阶段硬件锁死排队。

## Consequences

- 渲染与分析实现真正的错峰流水线并行，消除 30s/90s 超时排队告警。
- 系统在 NAS 和高压 4K 批次下保持高度线性的 56.6x 稳态渲染吞吐。

---

# 17. ADR 0015 — Ambient Light Drift Soft-Suppression and Absorbed Motion Audit Preservation

**Status:** Accepted  
**Date:** 2026-09  
**Evolves:** ADR 0006, ADR 0012

## Context

1. **日出日落大面积慢速光影干扰**：
   生产日志审计发现，全量数据中存在 101 起 `YOLO_NEGATIVE_REQUIRES_AUDIT` 审核疑点（6.29 小时），其中 **60.8% 集中于朝阳（06:00~09:00）与夕阳（16:00~19:00）**，平均单段长达 211.8 秒。这是由于太阳偏转或云层掠过造成全屋大面积、低频且均匀的漫射光变化，触发了运动检测但 YOLO 无法识别到人体。
2. **静态吸收短动态的静默漏审风险**：
   当短于 `min_motion_duration`（2.0s）的短运动被静态段吸收时，静态段继承了该运动的 `max_energy`（甚至 $\ge 6.0$），但原算法未将其标记为 `needs_review`，导致高能瞬态动作在审核后台不可见。

## Decision

1. **动态空间网格慢速漫射光影软抑制**：
   在 `SpatialGridMotionFilter` 中新增漫射光影检测：
   - 当激活网格占比 $\ge 35\%$ 且非夜间模式；
   - 全局网格能量最高值 $< 5.0$（安全红线：任何真实人体动作必定 $\ge 6.0 \sim 25.0$）；
   - 空间网格变异系数 $\text{CoV} < 0.35$ 且聚焦比 $< 1.75$（均匀漫射无局部运动焦点）；
   判定为 `is_ambient_drift` 并软抑制，促使底噪与 EMA 背景模型迅速自适应光线，杜绝虚假动态段落生成。
2. **吸收短动态审计保全铁律**：
   任何静态段若吸收了能量 $\ge 2.2$ 的动作，必须打上 `needs_review = 1` 并在后台显示明确的 `review_reason`（`ABSORBED_HIGH_ENERGY_MOTION` 或 `BORDERLINE_MICRO_MOTION`），保证人机协同审核队列对所有异常切片 100% 穿透。

## Consequences

- 日出日落漫射光影误报大幅压降，审核队列信噪比显著提升。
- 历史与实时数据库全量回测确证：$P0 = 0$ 致命漏检为 0，高能静态切片漏审为 0。

---

# 18. Current Decision Hierarchy

当前几个调度 ADR 的关系为：

```text
ADR 0005
Central resource control
        │
        ▼
ADR 0007
NVENC / NVDEC resource separation
        │
        ▼
ADR 0008
NVENC + QSV heterogeneous rendering
        │
        ▼
ADR 0009
Cost-aware dual-ended work stealing
        │
        ▼
ADR 0010
Pipeline-aware SJF + elastic stealing
```

它们不是五套独立调度模型。

后续 ADR 在保留前序有效原则的基础上继续演进。

当前实现应以：

```text
docs/ARCHITECTURE.md
```

描述的最终状态为准。

---

# 14. When to Create a New ADR

以下变化通常需要新 ADR：

- Pipeline Stage 增删或职责发生明显变化；
- 核心数据契约变化；
- Scheduler 的资源模型变化；
- Timeline 语义变化；
- 持久化模型变化；
- Failure / Retry / Resume 语义变化；
- Human Feedback 优先级变化；
- 跨硬件编码或合并策略变化；
- 引入影响多个模块的新基础抽象；
- 推翻已有 Accepted ADR 的核心假设。

以下情况通常不需要 ADR：

- 参数微调；
- 单个 Bug 修复；
- 局部重构；
- 单个配置项新增；
- Benchmark 数值变化；
- 不改变外部行为的性能优化。

---

# 17. ADR 0016 — Stable Streaming Timeline and Source Completion Gate

**Status:** Accepted
**Date:** 2026-09

## Context

流式渲染可能早于所有文件分析完成。此时跨文件 presence 推断会随着邻居任务完成顺序改变，导致同一批次在不同运行时得到不同时间轴；同时，单个渲染批次成功不能证明所有源文件都已成功分析。

## Decision

- 流式渲染批次只使用已持久化的单文件分析结果，暂不执行跨文件 presence 重判。
- 日期/机位在合并成片前必须通过源任务完成门禁；`PENDING`、`FAILED` 或未完成的可疑分析会阻止发布。
- 全量时间轴构建路径仍可在分析结果稳定后执行跨文件 presence 传播。

## Consequences

流式阶段牺牲部分 presence 跨文件优化，以换取批次时间轴确定性和失败可恢复性。源任务失败会保留已成功批次，等待修复后续跑。

---

# 15. ADR Writing Template

新增 ADR 使用以下结构：

```markdown
# ADR xxxx — Title

**Status:** Proposed | Accepted | Superseded | Deprecated | Rejected
**Date:** YYYY-MM
**Supersedes:** ADR xxxx
**Superseded by:** ADR xxxx

## Context

需要解决什么问题？

为什么现有设计不足？

哪些事实或约束驱动这个决策？

## Decision

决定采用什么设计？

只记录稳定的设计原则。

避免把临时参数调优写成架构原则。

## Alternatives

评估过哪些主要替代方案？

为什么没有选择？

## Consequences

### Positive

带来哪些长期收益？

### Negative

增加哪些复杂度、约束或维护成本？

## Evidence

相关 Benchmark、生产遥测、Issue 或实验在哪里？

## Revisit When

什么条件变化时需要重新评估本 ADR？
```

---

# 16. Maintenance Rules

ADR 应保持历史真实性。

不要因为当前实现发生变化而直接重写旧 ADR，使其看起来像当初就做出了今天的设计。

正确方式：

```text
old decision
→ mark Superseded
→ create new ADR
```

如果只是补充背景、修正事实错误或添加交叉引用，可以编辑已有 ADR。

性能指标应尽量引用：

```text
docs/PROGRESS.md
```

而不是在多个 ADR 中复制同一组生产数字。

---

# 19. ADR 0017 — Offline High-Efficiency Encoding and Multi-Feature Hysteresis Action Model

**Status:** Accepted  
**Date:** 2026-09  
**Evolves:** ADR 0008, ADR 0012, ADR 0013

## Context

在五级浓缩模型和驻留优化落地后，系统运行遥测暴露两项深层矛盾：
1. **静态压缩收益见顶与编码模式错配**：`20260320` 生产数据显示静态段在最终成片中展示时长仅约 1%，继续挤压静态段无法带来显著体积缩减；此前尝试单纯将 CQ 调高至 31/32 换取体积，导致抽样出现严重的“绿色解码损坏”。后续在 P4 下仍复现损坏：异构 HEVC 参数集被 hvc1 封装剥离是已验证原因，不能将损坏归因于提高 QP 或 P1。
2. **动作判定单变量硬阈值与空间信息浪费**：`refine_activity_segments` 仅凭 `raw_energy >= 5.5` 做出 1x/4x 二值决策，单帧噪点极易被前后扩展缓冲 (`pre_roll 1.0s` + `post_roll 1.5s`) 放大为数秒常速巨石；与此同时，`SpatialGridMotionFilter` 已经计算出丰富的 8×8 连通单元数 (`active_cells`) 与局部单格能量峰值 (`max_cell_energy`)，但 `MotionTrace` 将其完全丢弃。

## Decision

### 1. 离线高质量编码参数重构
- 将 NVENC 从超低延迟推流模式切为离线高质量压制预设：启用 `preset: p4`，彻底移除 `tune: ll`；
- 将批次硬编码的 GOP=60 提升为可配置的 GOP=120（20fps 下 6.0s，契合独立切片 concat 规范）；
- 开启硬件高级压缩特性：`-rc-lookahead 32`、`-spatial-aq 1`、`-temporal-aq 1`、`-aq-strength 8`；
- 保留 `-strict_gop 1` 与 `-no-scenecut 1`；这些选项不能替代跨编码器参数集保留与拼接完整性验证；
- QSV 同步开放配置并对齐 GOP=120。

### 2. 空间网格特征管道持久化与多特征迟滞状态机
- 在 `MotionTrace` 内存流中打通 `active_cells` 与 `max_cell_energies` 存储通道，经由 `detector.py` 传递给时序分段层；
- 在 `refine_activity_segments` 中引入高低双门限 ($T_{high}=5.5, T_{low}=3.5$) 迟滞状态机与局部聚类触发：
  - 音频事件 100% 强制直通 `DYNAMIC_AUDIO` 1x 保全；
  - 触发门限：全局突变能量 $\ge 5.5$ 或局部聚类连通高能量 (`active_cells >= 2` 且 `max_cell_energy >= 7.0`) 触发 1x；
  - 维持门限：在动作余波内，只要能量 $\ge 3.5$ 或局部 $\ge 4.2$ 即平滑维持，杜绝单帧抖动引起的眨眼式变速与常速膨胀。

### 3. 调度器滑动平均动态竞价
- 将异构工作窃取中写死的 42s 批次耗时经验常数升级为基于已完成批次真实耗时的在线滑动平均模型，自适应匹配当前素材负载。

## Consequences

- P4 和迟滞状态机保留为当前实现；其净体积、耗时与识别收益尚无有效同条件 A/B 证明。
- 旧 6.832GB 产物的异构边界存在解码错误，不能作为质量验收结果；基准口径见 `docs/BENCHMARK.md`。

# ADR 0018 — Preserve heterogeneous HEVC parameters and render timeline identity

**Status:** Accepted  
**Date:** 2026-09-16

## Decision

- HEVC 批次使用 hev1 并保留 in-band 参数集。最终合成在原子替换前检查各批次边界；失败时保留原成片与可恢复批次。渲染缓存版本升级，分析缓存不受该编码变更影响。
- 流式渲染及伴随资产统一消费 `build_timeline_from_rows(..., resolve_presence=False)` 的逐文件分析结果与人工纠正；不跨文件重新分类或合并。展示时间继续来自 `compute_display_plans()`；区间归一化共用 timeline 实现，并对各文件独立维护游标，变速过渡不跨文件边界。
- 渲染接管文件时消费预取所有权；预取在源文件锁内复核所有权，避免批次清理后重新复制。
- QSV 使用明确的 ICQ 配置，拒绝同时传入会改变当前本机码控语义的 maxrate/bufsize。

## Consequences

- hev1 需要目标播放器兼容性验收；边界检查增加解码成本，不等价于逐帧全片验证。
- 时间轴不再因伴随资产生成而写回或变动；历史跨文件 presence 重分类产物不能充当相同时间轴基线。
- QSV ICQ28 的体积与画质需要重新测量；不从质量参数数值推导实际收益。

---

# ADR 0019 — Human-Centric Spatial Gating, Adaptive Day/Night Thresholds, and PTZ Cruise Suppression

**Status:** Accepted  
**Date:** 2026-09-18  
**Evolves:** ADR 0012, ADR 0013, ADR 0017

## Context

在 `20260320` 真实全天素材中，成片膨胀至 6.11 小时 / 6.81 GB（历史早期仅约 2.96 小时 / 2.41~2.85 GB）。诊断发现 93.6%（20,576 秒）的成片被锁定在 1.0x `DYNAMIC` 常速播放，大量应以 4x 快进（`PRESENCE`）或 16x 延时（`NIGHT_STATIONARY`）的片段失控膨胀：
1. **迟滞底噪下限与人体呼吸微动失配**：ADR 0017 设定的空间网格门限 $7.0$ 与全局门限 $5.5$ 处于人体静坐呼吸底噪（单格能量 10~17，全局能量 6~8）之下，导致看护人静坐 6 分钟被 100% 误判为常速剧烈运动；
2. **事件无界连环合并**：在 `action_coalesce_gap = 2.0s` 缺乏时长上限约束时，微弱呼吸噪点连锁吞噬整段静止；
3. **夜间低照度低置信度回退错误**：静止间隙在未能高置信度检出人体时，错误回退至 `parent.state`（1x DYNAMIC），造成夜间大量睡眠被当作 1x 播放；
4. **云台自适应巡航误报**：全景巡航平移导致 60/64 单元均匀激活，伪造假动态；
5. **室外类别污染**：室内机位检测汽车/自行车/摩托车造成假目标干扰。

## Decision

1. **人体空间位移门控 (Human-Centric Motion Gating)**：
   - 在 `YoloVerifier.verify` 中追踪人体采样包围盒序列；
   - 当连续采样帧位移 $\text{max\_disp} < 0.08$、$\text{min\_iou} > 0.55$ 且能量 $< 25.0$（无音频事件）时，确认为静坐/静卧陪伴，精准升级为 `PRESENCE`（日间 4x）或 `NIGHT_STATIONARY`（夜间 16x），彻底剥离 1x 动态。
2. **室内检测目标净化**：
   - 室内机位剥离 COCO 1/2/3 类（汽车、自行车、摩托车），锁定目标类别为 `target_classes: [0, 15, 16]`（人、猫、狗）。
3. **云台巡航与全画幅相机平移抑制 (PTZ Cruise Gating)**：
   - 在 `SpatialGridMotionFilter` 中增加全画幅平移抑制：当激活单元比 $\ge 0.65$ 且方差变异系数 $\text{cov} < 0.60$、聚焦比 $\text{focal\_ratio} < 3.5$ 时，判定为全景平移而非主体活动，予以软抑制。
4. **昼夜自适应多特征迟滞状态机与合并边界收敛**：
   - 区分日间触发门限（全局 $e \ge 10.0$ 或局部 $\text{max\_cell\_e} \ge 28.0$ 且连通数 $\ge 2$）与夜间微动保护门限（全局 $e \ge 12.0$ 或局部 $\text{max\_cell\_e} \ge 35.0$）；
   - 迟滞维持设最大窗口 `max_maintain_duration = 2.5s`；
   - 动作合并设最大单体时长 `max_coalesce_duration = 8.0s`（非强动作/音频禁止无限连环合并）；
   - 修复静止间隙回退优先级：优先保留 `YOLO_FAILED` 故障兜底，夜间强制回归 `NIGHT_STATIONARY` (16x)，日间有人回归 `PRESENCE` (4x)，无人回归 `STATIC`。

## Consequences

- 日间静坐测试素材（如 `140727`、`143054`）75% 以上时长精准进入 `PRESENCE` (4x)，展示时长减少 20%~26%；
- 夜间睡眠素材（如 `000658` 50分钟视频）99.2% 时长精准落入 `NIGHT_STATIONARY` (16x)，展示时长减少 30.7%；
- 全量 243 项自动化单元测试 100% 保持通过，零漏检（$P0 = 0$）防护机制持续成立。

---

# ADR 0020 — Virtual Concat Local Staging Dynamic Binding, QSV Heterogeneous Retry, and SMB Probing Bypass

**Status:** Accepted  
**Date:** 2026-09-19  
**Evolves:** ADR 0004, ADR 0015, ADR 0016

## Context

在对全天 24.81 小时 `20260320` 素材执行全流程复现时，管线在 141/142 批次时遇到中断：
1. **虚拟转码音频流远程阻塞**：`_prepare_virtual_input` 对源视频切片执行了本地 SSD 缓存（`staged_source`），但 filtergraph 伴生音频输入 `-i` 仍硬编码绑定至远程 SMB NAS 原始路径 `files[0]`。当多路分析并发冲击 NAS 磁盘吞吐时，FFmpeg 启动探针阻塞超过 120s 被看门狗终止；
2. **纯静态虚拟批次硬失败**：纯静态虚拟批次重试条件仅检查 `virtual_mixed`，导致单文件纯静态批次在启动异常时无法降级至 continuous 稀疏路径；
3. **异构单批次偶发失败阻断全局**：当 QSV 批次偶发超时或底层驱动卡顿时，缺少异构（NVENC）容错重试路径，直接导致全天合成被中止；
4. **冗余 NAS 探测读锁放大**：`detector.analyze` 对每个已由数据库记录 `file_duration` 与 `has_audio` 的切片依然无差别发起 `av.open` 远端嗅探，造成全天 95 次重复远程网络 I/O；
5. **硬件并发信号量不对齐**：`analysis_max_workers: 8` 争抢 `max_qsv_analysis_concurrency: 7` 导致信号量获取频繁超时抖动。

## Decision

1. **虚拟渲染本地暂存动态全流绑定**：
   - 在 `_prepare_virtual_input` 中，当源文件已暂存至本地 SSD（`staged_source`）时，动态将 `input_args` 中的音频输入流 `-i` 替换为 `staged_source` 本地路径，确保音视频流解码全部消费本地 SSD。
2. **虚拟批次自愈降级**：
   - 将虚拟拼接异常重试条件放宽至 `not virtual_enabled`，所有虚拟批次（含纯静态批次）在遭遇启动超时或校验失败时均能平滑回退至 continuous 稀疏渲染模式。
3. **异构渲染跨硬件自动重试**：
   - 在 `StreamingOrchestrator` 批次收集流程中，若批次在 `qsv` 下未能产出有效输出，自动使用 `nv` 触发一次无感重试，保障单批次硬件偶发异常不影响整日交付。
4. **数据库缓存感知并跳过冗余远端探测**：
   - 在 `MotionDetector.analyze` 中判断若入参已具备 `file_duration > 0` 且 `has_audio is not None`，直接使用已知元数据，彻底绕过跨 SMB 网络的 `av.open`。
5. **硬件并发信号量严格对齐**：
   - 生产配置中 `analysis_max_workers` 与 `max_qsv_analysis_concurrency` 统一收敛为 7，避免过载争用。

## Consequences

- 20260320 全量 142 个批次 100% 成功交付并完成最终合成；
- 成片 `DailyVlog_20260320_B888805AA3CD.mp4`（4.21 GB / 226分56秒 / 6.56x 浓缩 / 49.5x 实时）一次性生成，音画同步误差小于 0.021s；
- 全量自动化单元测试 243 passed，7 skipped，零回归风险。

---

# ADR 0021 — Clean Code Architecture Layering, Zero Dependency Inversion, and Companion Decoupling

**Status:** Accepted  
**Date:** 2026-09-19  
**Evolves:** ADR 0011, ADR 0014

## Context

随着多阶段流式管线演进，系统出现以下代码结构坏味道：
1. **分层依赖反转**：底座核心 `src/core/database.py` 逆向导入顶层阶段 `src/stages/scanner.py`（用于机位别名解析），引发下层依赖上层的架构反转；
2. **伴随资产与编排混杂**：`src/pipeline.py` 中充斥着大量的字幕（SRT）生成、元数据（meta.json）序列化及性能图表持久化逻辑，流水线编排核心职责膨胀；
3. **状态常量发散**：切片状态与活动集合在各模块中以字面量元组形式硬编码定义，存在拼写错误与契约漂移风险；
4. **内部垫片残留**：内部生产模块间依然存在通过根目录垫片（`from src.utils import ...`）引用的历史遗留，未彻底收敛至分层子包规范。

## Decision

1. **新建机位与文件名规则层 `src/core/identity.py`**：
   - 将 `resolve_camera_identity`、`resolve_output_filename` 等纯规则算子下沉至 `src.core.identity`，`database.py` 与 `scanner.py` 均单向依赖核心层，彻底终结架构倒置。
2. **伴随资产生成逻辑独立封装 `src/stages/companion.py`**：
   - 提取 `save_companion_assets`、`build_srt_subtitles`、`build_vlog_metadata`，`pipeline.py` 仅关注阶段调度编排。
3. **领域状态枚举与活动集合契约收敛**：
   - 在 `src/algorithms/segment.py` 统一定义 `SegmentState(StrEnum)` 与 `ACTIVE_STATES`，全库统一引入。
4. **内部生产导入卫生全面治理**：
   - 全库生产代码全面迁移为 `src.core.*`, `src.hardware.*`, `src.algorithms.*`, `src.stages.*`, `src.ui.*`，生产代码对根层垫片引用清零；顶层透明 Facade 严格保留以保证外部脚本与单测 100% 兼容。

## Consequences

- 根除所有隐式循环依赖与架构反转；
- 代码内聚性与模块边界显著清晰，流水线编排模块代码精简 ~160 行；
- 243 项自动化单元测试 100% 保持绿色通过。

---

# ADR 0022 — QSV Tail Load Balancing, Parallel Checkpoint Validation, and Critical Path Operator Acceleration

**Status:** Partially Superseded by ADR 0023
**Date:** 2026-09-20  
**Evolves:** ADR 0015, ADR 0019, ADR 0020

## Context

对 `20260320` 真实 4K 全天素材（142 切片 / 24.81 小时）的性能分析暴露了三大瓶颈：
1. **QSV 尾部饥饿与负载失衡**：刚性 420s 动态上限与过激的退出守卫导致尾部队列仅剩少量重任务时 QSV 提前退出，空转闲置 **168.16 秒**（近 3 分钟），全由 NVENC 孤军收尾；
2. **最终拼接串行接缝校验延迟**：4.4GB 成片中 141 个接缝由单线程逐一 Seek 解码校验，耗时高达 **~39.5 秒**；
3. **关键算子微观损耗**：
   - 预筛选与精析逐帧将 YUV420p 转 Gray（FFmpeg `to_ndarray` 需 10.6ms/帧）；
   - YOLO 推理每帧执行 3 次独立 `.cpu().numpy()` 导致冗余 CUDA 同步；
   - 候选帧反复执行 `rgb24` $\rightarrow$ `cv2.cvtColor(BGR)` 内存拷贝。

## Decision

1. **QSV 尾部动态窃取与负载均衡**：
   - 退出守卫调整为仅当队列为空或剩余任务 $\le$ 活跃 NV 工人数时退出；
   - 当 NV 工人全忙时，动态放宽 QSV 窃取门限 1.5×（由 420s 放宽至 630s），实现毫秒级同步收尾。
2. **拼接点全量多线程并行校验**：
   - 坚持全量接缝校验原则（杜绝抽样风险）；
   - 超过 4 个接缝时采用 `ThreadPoolExecutor(max_workers=min(8, N))` 分块并行校验，各线程独立持有解码容器且 Seek 单向递增。
3. **关键路径零拷贝算子加速**：
   - 预筛选与运动分析直接从 `frame.planes[0]` 读取 Y 分量零拷贝切片（0.16ms vs 11.2ms，71x 提速）；
   - YOLO 单次搬移连续张量 `boxes.data.cpu().numpy()`；
   - 候选帧直出 `bgr24`，消除中间内存与色彩转换。

## Consequences

- 20260320 端到端壁钟由 **30分03秒 (1803.14s)** 缩短至 **24分12秒 (1452.84s)**，**净提速 5分50秒 (-19.43%)**，吞吐达 **61.48× 实时**；
- QSV 尾部空转由 168.16s 降至 **0.56s**（利用率 100%），三工人总空转损耗由 337s 降至 **2.61s**（消除 99.2%）；
- 拼接接缝校验由 39.5s 压缩至 **10.66s (3.7x 提速)**；
- 独显显存峰值降低 **-31.7%**（5,198MB $\rightarrow$ 3,551MB），全库 244 项单测 100% 通过。

---

# ADR 0023 — Delivery Artifact Atomic Commit and Shared Media Contracts

**Status:** Accepted
**Date:** 2026-09-21
**Evolves:** ADR 0004, ADR 0013, ADR 0020, ADR 0022

## Context

交付审查发现“可打开媒体”“批次成功”和“整日交付完成”使用了不同门槛：短片、无音轨、接缝内部断帧或缺伴随资产仍可能进入完成状态；Prescreen 与 Analysis 对 YUV 灰度的解释也不一致。QSV 重试和审核重渲染另有独立完成语义，导致恢复与遥测失真。

## Decision

1. Prescreen 与 Analysis 共用 8 位全范围灰度转换；旧的任意 Y 平面直读优化撤销，只有原生 `gray` 可直接读平面。
2. 文件名跨度保持名义值，容器探测后以 `duration_verified` 标记真实媒体时长。所有状态的视频用 `tpad/fps/trim`、音频用 `apad/atrim` 闭合到共享 display plan。
3. `valid_video` 同时验证音轨、严格计划时长、音视频末端、接缝内部 PTS 连续性与尾部可解码性。
4. 视频、字幕、元数据和 manifest 全部成功后才提交 `COMPLETED`；恢复必须验证同一资产集合。
5. QSV 准入上限随已取任务传递，失败批次回到 NV 专用队列。审核重渲染复用只读逐文件 timeline、完整源文件门禁和任务级取消。
6. 每次性能记录携带 run id、代码版本/脏状态、配置与分析快照指纹；批次只查询目标文件。

## Consequences

- ADR 0022 的“任意 Y 平面零拷贝”决策被撤销；历史性能数字保留为历史证据，必须用新正确性契约重新 A/B 才能成为当前结论。
- 单元测试与软件 FFmpeg 媒体回归可验证代码契约，但不能替代整日逐帧解码、目标播放器与人物识别 holdout。

---

# ADR 0024 — Integer-Frame Display Plans and Cancellation Outcome

**Status:** Accepted
**Date:** 2026-09-21

连续秒数分段分别经过 fps 后，音视频 concat 按较长轨道推进，导致分段误差累积。2026-09-21 日志中的时长拒绝已通过真实软件媒体复现。

`compute_display_plans(output_fps=...)` 将每段目标向上量化为整数输出帧，保留源尾部；视频按帧数截断，音频补齐到同一计划，字幕、高光和校验也传入同一输出帧率。每段增加不足一帧展示时间，不修改识别标签或负判定门槛。快进曲线按量化后的目标重新求解。

主动取消保留完成批次并保存 status=cancelled 的性能记录，不作为静默丢批报错；正常结束的缺批与真实失败仍阻止合成。CLI 对失败返回非零退出码，不能以进程正常退出代表成片完成。

真实复验补充：源音频可能有重叠 PTS，所有音频段按整数样本数截断，变速子段拼接后也须闭合总样本数。最终 concat 显式使用视频 duration，复制视频并对齐、重新编码音频；单纯复制音频会在批次边界产生 non-monotonic DTS。额外音频编码的成本和质量必须进入整日验收。

