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
| ADR 0018 | 异构 HEVC 参数与渲染时间轴身份 | Accepted | 2026-09 | — |
| ADR 0019 | 人体空间门控、昼夜门限与云台巡航抑制 | Accepted | 2026-09 | Evolves ADR 0012, ADR 0013, ADR 0017 |
| ADR 0020 | 虚拟拼接暂存绑定、QSV 重试与 SMB 探测边界 | Accepted | 2026-09 | Evolves ADR 0004, ADR 0015, ADR 0016 |
| ADR 0021 | 分层依赖、兼容入口与伴随资产职责 | Accepted | 2026-09 | Evolves ADR 0011, ADR 0014 |
| ADR 0022 | QSV 尾部负载、并行接缝校验与算子优化 | Partially Superseded by ADR 0023 | 2026-09 | Evolves ADR 0015, ADR 0019, ADR 0020 |
| ADR 0023 | 交付产物原子提交与共享媒体契约 | Accepted | 2026-09 | Evolves ADR 0004, ADR 0013, ADR 0020, ADR 0022 |
| ADR 0024 | 整数帧展示计划与取消结果 | Accepted | 2026-09 | Evolves ADR 0004 |
| ADR 0025 | 预筛末关键帧覆盖 | Accepted | 2026-09 | Evolves ADR 0006, ADR 0015 |
| ADR 0026 | 交付元数据保留分析证据 | Accepted | 2026-09 | Evolves ADR 0023 |
| ADR 0027 | 看门狗配置语义闭环与 NVENC 瞬态重试兜底 | Accepted | 2026-09 | Evolves ADR 0004, ADR 0008, ADR 0011, ADR 0023 |
| ADR 0028 | 已交付日期保留离线备份完成记录 | Accepted | 2026-10 | — |
| ADR 0029 | 显式零字节源文件排除 | Accepted | 2026-10 | — |
| ADR 0030 | 局部识别证据与显式批处理停止 | Accepted | 2026-10 | — |
| ADR 0031 | 亚帧源区间必须输出计划帧 | Accepted | 2026-10 | — |
| ADR 0032 | 可见启动与完整扫描门禁 | Accepted | 2026-10 | — |
| ADR 0033 | 配置状态单一所有权与旧入口兼容 | Accepted | 2026-10 | Evolves ADR 0011, ADR 0021 |
| ADR 0034 | 最终拼接故障证据与完整接缝回退 | Accepted | 2026-10 | Evolves ADR 0027 |
| ADR 0035 | 持久审核撤销与精确源区间 | Accepted | 2026-10 | Evolves ADR 0003, ADR 0024 |
| ADR 0036 | 可恢复资产提交与真实运行结果 | Accepted | 2026-10 | Evolves ADR 0023, ADR 0028 |

后续决策索引（状态与演进范围以各节正文为准）：

| ADR | Decision |
|---|---|
| [ADR 0037](#adr-0037--propagate-conclusive-render-interruptions) | Propagate Conclusive Render Interruptions |
| [ADR 0038](#adr-0038--triad-human-in-the-loop-architecture-benchmark-priority-re-rendering-and-tuning-simulator) | Triad Human-in-the-Loop Architecture: Benchmark, Priority Re-rendering, and Tuning Simulator |
| [ADR 0039](#adr-0039--hierarchical-configuration-governance-and-empirical-scenario-profiles) | Hierarchical Configuration Governance and Empirical Scenario Profiles |
| [ADR 0040](#adr-0040--审计修复显式配置来源与审核资产提交边界) | 审计修复：显式配置来源与审核资产提交边界 |
| [ADR 0041](#adr-0041--解码完成证据与不可用源范围) | 解码完成证据与不可用源范围 |
| [ADR 0042](#adr-0042--单文件稠密加速时间轴的视频单次变换) | 单文件稠密加速时间轴的视频单次变换 |
| [ADR 0043](#adr-0043--审核库存事件保留与播放清晰度分开验收) | 审核库存、事件保留与播放清晰度分开验收 |
| [ADR 0044](#adr-0044--审核重渲使用共享缓存决策与启动快照) | 审核重渲使用共享缓存决策与启动快照 |
| [ADR 0045](#adr-0045--复核归因与实测对照保持来源和确认边界) | 复核归因与实测对照保持来源和确认边界 |
| [ADR 0046](#adr-0046--审核队列区分库存代表分组和随机抽查) | 审核队列区分库存、代表分组和随机抽查 |
| [ADR 0047](#adr-0047--成片对比使用交付绑定映射及成功审核历史) | 成片对比使用交付绑定映射及成功审核历史 |
| [ADR 0048](#adr-0048--视觉近似索引复用解码数据并与人工真值隔离) | 视觉近似索引复用解码数据并与人工真值隔离 |
| [ADR 0049](#adr0049隔离分析重放的正式检查点与续跑) | 隔离分析重放的正式检查点与续跑 |
| [ADR 0050](#adr0050事件审核不作为训练目标真值) | 事件审核不作为训练目标真值 |
| [ADR 0051](#adr0051历史视觉补建使用受父进程管理的独立源读者) | 历史视觉补建使用受父进程管理的独立源读者 |
| [ADR 0052](#adr0052主流程工作台与维护入口共用内核认领) | 主流程、工作台与维护入口共用内核认领 |
| [ADR 0053](#adr0053交付包骤停恢复与完成标记一致性) | 交付包骤停恢复与完成标记一致性 |
| [ADR 0054](#adr0054成片映射绑定验证后的计划帧时钟2026-10-13) | 成片映射绑定验证后的计划帧时钟（2026-10-13） |
| [ADR 0055](#adr0055智能体视觉建议与人工目标真值分别持久化2026-10-13) | 智能体视觉建议与人工目标真值分别持久化（2026-10-13） |
| [ADR 0056](#adr0056粗筛邻域状态按源顺序确定2026-10-14) | 粗筛邻域状态按源顺序确定（2026-10-14） |
| [ADR 0057](#adr0057工作台预览复用托管ffmpeg与有界io准入2026-10-14) | 工作台预览复用托管FFmpeg与有界I/O准入（2026-10-14） |
| [ADR 0058](#adr0058工作台预览身份读取复用拥有的元数据读者2026-10-14) | 工作台预览身份读取复用拥有的元数据读者（2026-10-14） |
| [ADR 0059](#adr0059已注册ffmpeg复用共享windows进程job2026-10-14) | 已注册FFmpeg复用共享Windows进程Job（2026-10-14） |
| [ADR 0060](#adr0060媒体启动时建立windows-job归属2026-10-14) | 媒体启动时建立Windows Job归属（2026-10-14） |
| [ADR 0061](#adr0061按任务停止拥有树与有界管道回收2026-10-14) | 按任务停止拥有树与有界管道回收（2026-10-14） |
| [ADR 0062](#adr0062未确认清理的原io许可转交与终态恢复2026-10-14) | 未确认清理的原I/O许可转交与终态恢复（2026-10-14） |
| [ADR 0063](#adr0063识别质量门禁只接受留出集及独立事件组2026-10-14) | 识别质量门禁只接受留出集及独立事件组（2026-10-14） |
| [ADR 0064](#adr0064逐条声明事件复核依据冻结时保守准入2026-10-14) | 逐条声明事件复核依据，冻结时保守准入（2026-10-14） |
| [ADR 0065](#adr0065批次停止未确认保留租约并阻止重试2026-10-14) | 批次停止未确认保留租约并阻止重试（2026-10-14） |
| [ADR 0066](#adr0066直接源读者未确认停止的资源保留与失败传播2026-10-15) | 直接源读者未确认停止的资源保留与失败传播（2026-10-15） |
| [ADR 0067](#adr0067启动登记失败的实际进程交接与同key认领2026-10-15) | 启动登记失败的实际进程交接与同key认领（2026-10-15） |
| [ADR 0068](#adr0068跨日期准入与致命媒体失败保留2026-10-15) | 跨日期准入与致命媒体失败保留（2026-10-15） |
| [ADR 0069](#adr0069人工排除交付准备与配置写者边界2026-10-16) | 人工排除、交付准备与配置写者边界 |
| [ADR 0070](#adr0070整数音频时钟与有界分组2026-10-17) | 整数音频时钟与有界分组 |
| [ADR 0071](#adr0071非侵入式分析耗时微观分解与交付-io-审查2026-10-18) | 非侵入式分析耗时微观分解与交付 I/O 审查（2026-10-18） |

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
- **用户信心保障**：建立起“自动化分层超敏审计套件（如 `scripts/audit_multiday_backtest.py` 与回测套件）”，提供客观量化数据支撑物理删除决策。

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

2026-09-24 补充：若最后一个计划切片完全落在物理视频 EOF 之后，逐段 `tpad` 接收空视频，无法克隆末帧。应在每个输入完成缩放或抽帧后、分给各计划切片之前补源末帧，并在批次 `concat` 出口按共享计划的总帧数和音频样本数闭合。源视频全程不可解码仍须失败。

---

# ADR 0025 — Cover the Final Decoded Keyframe in Prescreen

**Status:** Accepted
**Date:** 2026-09-25
**Evolves:** ADR 0006, ADR 0015

关键帧预筛虽解码到 EOF，稀疏步长可能跳过最后关键帧，使尾部人物入镜被判 `STATIC`。负判定前补查已解码但未采样的最后关键帧，复用原有差分、阈值和空间集中度规则；不再增加相邻文件先验或更改 YOLO 色彩输入。126 个生产 `STATIC` 文件及另 60 个不同日期的 `STATIC` 文件只读对照共翻转 3 个：两段经原片和精析确认的末尾成人入镜，以及一段夜间有人驻留。已知空床光照误报保持 `STATIC`。这证明具体漏检得到修复，不等于独立人工 holdout 或整日成片耗时验收。

---

# ADR 0026 — Preserve Analysis Evidence in Delivery Metadata

**Status:** Accepted
**Date:** 2026-09-29

数据库中的置信度和能量在转换为 `TimelineSegment` 时丢失，使伴随元数据将所有高光字段默认为零。时间轴现携带这两个证据字段，虚拟 EDL 映射也予以保留；相邻同状态区间合并时取最大能量和按保留时长加权的置信度。字段仅用于交付解释，不改变分类、展示计划、编码或人工审核优先级。缺少模型分数仍保留零，不根据人工确认状态伪造分数。

既有成片元数据回填必须先备份并逐条验证高光数量、状态、源文件和展示起止时间不变，确认媒体身份未变化后再原子替换；不重跑识别，不写生产数据库。

---

# ADR 0027 — Watchdog Timeout Configuration Semantics and NVENC Transient Retry Fallback

**Status:** Accepted  
**Date:** 2026-09-30  
**Evolves:** ADR 0004, ADR 0008, ADR 0011, ADR 0023  

## Context

在全量连续生产批处理场景（如 `20260320` 真实全天素材）中，暴露了流水线超时预算与调度容错的重大脆弱点：

1. **配置语义被下层计算覆盖失效**：
   `config/settings.yaml` 显式设置的 `startup_inactivity_timeout_s: 600.0` 在旧代码中被单文件批次公式 `per_input_budget = 90.0s` 强行覆盖，再叠加上稳态下限逻辑后，启动看门狗被硬编码钳位在 120s。
2. **冷启动与网络抖动导致单点崩溃**：
   流水线刚启动时，前两个批次（文件物理跨度 >3200s）同时由 2 路 NVENC 直接读取 NAS，恰逢主线程 PyTorch YOLO 加载 CUDA 模型初始化，发生驱动与 SMB 握手竞争，在 120.6s 时因无数据产出被看门狗作为超时强行终止。
3. **缺乏 NVENC 级别的退避重试兜底**：
   调度器中仅 QSV 失败批次具备退避回退至 `nv_retry_queue` 的保护，NV 批次若发生瞬态抖动超时则直接报错，导致耗时 24 小时的整日拼接因单个单点失败被完全拦截，违背 ADR 0004 事务恢复与 AGENTS.md 弹性容错原则。
4. **素材冻结时间边界抖动**：
   扫描器入库时，素材冻结判定对 `age_min < freeze_minutes` 缺乏 `freeze_minutes > 0` 前置条件保护。在 Windows NTFS 高精度时钟微秒抖动下偶发出现 `age_min < 0`，被错误当成“未脱水冻结中”而漏扫入库。

## Decision

1. **看门狗配置真实语义恢复**：
   重构 `_startup_watchdog_timeout`，以配置声明的 `startup_inactivity_timeout_s` 作为基线预算，多输入批次在配置基线上按 `per_input_budget` 增量叠加，确保配置声明的 600s 启动预算真正具有运行时语义。
2. **NVENC 瞬态失败有限退避重试**：
   在 `_render_manager` 中引入批次级 `batch_retries` 计数器。NVENC 批次首次发生非优雅异常退出时，记录警告日志并重新放回 `nv_retry_queue`，在活跃 NV 工人空闲时进行至多 1 次重试；仅在重试再次失败时才作为终端硬故障阻止最终合成。该队列仅由 NV 工人消费；异构模式 QSV 失败可转交 NV，`qsv_only` 模式失败则直接记录为终态，避免无人消费的重试阻塞收尾。
3. **扫描器零冻结判定防御**：
   在 `src/stages/scanner.py` 中增加 `freeze_minutes > 0` 前置判定，确保零冻结时间配置下任何新产生或微秒抖动素材均立即入库，根除间歇性漏扫。

## Consequences

- `20260320` 故障成功自愈，秒级复用 140 个既有批次并自愈修复 Batch 4/Batch 6，全日成片通过 141 个接缝切片关键帧探测校验顺利交付（4,736.93 MB，时长 246分28秒）；
- 全量 290 项自动化测试全部通过，杜绝了 NAS 抖动与 CUDA 冷启动导致的长流程单点崩溃；
- 保持了事务化批次不可篡改与端到端时间轴闭环的核心架构不变量。

---

# ADR 0028 — Delivered Days Survive Offline Backup

**Status:** Accepted

**Date:** 2026-10-03

用户会将交付产物移至其他介质；将本地文件存在性作为日任务跳过条件会导致重复渲染。

`render_tasks.COMPLETED` 作为日期/机位的持久交付记录，普通运行（含仅分析阶段）在配置失效处理前跳过，不再读取本地成片或 manifest。已有完成记录直接兼容，无需迁移；必须保留和备份 `data/vlog.db`。新增待处理素材仅提示，配置变化不自动重开已交付日期。用户通过 `--force-render` 显式重做，指定日期和全量入口均传递此参数；有效中间批次仍可复用。

首次交付的校验与提交顺序不变，未完成任务仍执行原恢复逻辑。显式重做一旦进入处理，失败按现有规则记录为 FAILED，后续可恢复；本次不引入并发任务认领或跨进程排他机制。跳过仅证明历史交付完成，不证明外部备份介质上的文件当前完整。

---

# ADR 0029 — Explicit Empty Source Exclusion

**Status:** Accepted

**Date:** 2026-10-03

生产素材 `00_20260504210814_20260504211329.mp4` 实测为 0 字节，导致 20260504 交付被完整性保护阻止。用户明确授权跳过空文件并打印警告。

使用现有任务状态列保存 `SKIPPED_EMPTY`，仅用于确认大小为零的普通文件。检查发生在未完成日期的配置失效处理之后、失败任务重试之前，不修改源文件。渲染保序队列将此状态作为无输出终态推进；时间轴不生成占位片段；完成保护允许该显式排除，其他失败仍阻止交付。伴随元数据保存文件名、原因、名义起止时间和时长，避免静默丢失。若全日无可渲染素材，仍不生成或标记成片完成。

每次重试重新 stat：非空文件回到预筛，无法访问的原空文件回到失败。状态更新使用现有数据库锁和事务；部分更新中断可逐条重查恢复。原始文件仍应遵守扫描冻结规则，不新增对正在写入源文件的跨进程锁。历史已完成日期保持跳过，后补素材需显式重做。

---

# ADR 0030 — Local Recognition Evidence and Explicit Batch Stops

**Status:** Accepted

**Date:** 2026-10-06

原片已复现长动态父区间均匀抽 8 帧漏掉末段人物；模型失败以外的证据不足也可能在后续平滑/活动细化中变为静态。改为遍历现有受限缓存，并将父区间按至多 8 个候选帧的局部窗口验证，避免一次阳性覆盖整段长录像。保留原模型、阈值、推理批大小、解码采样与设备策略。`YOLO_FAILED`、`YOLO_UNAVAILABLE`、`INSUFFICIENT_YOLO_EVIDENCE` 不作为阴性，动态回退继续常速并待审。分析指纹升级至 `analysis-v4-window-evidence`；未完成任务按原失效流程重新分析，历史交付仍遵循 ADR 0028。该修复不构成独立人物召回率验收，且增加推理量。

音频门控结果嵌入原 `visual_result`，保留两类依据；缓存活动状态使用与新分析相同的源时长队列权重。只修正恢复一致性并补充计量，不改变 QSV 门槛、并发或 SJF 策略。

全量批处理显式返回 `skipped`、`remaining`、`stopped_reason`，磁盘不足为 `LOW_DISK_SPACE`；全量和指定日期 CLI 都返回非零退出码。使用现有 `recovery.min_disk_space_gb`；已交付跳过不受启动空间检查阻挡，也不纳入本轮性能收益。停止不改变待处理任务或历史交付状态。

QSV/CUDA 共用 CPU area 缩放的候选虽通过一个实片的像素一致性检查，但会改变运动边界并触发额外驻留推断，未满足识别验收，已撤回；不得将设备一致性等同于识别质量不变。

---

# ADR 0031 — Sub-frame Source Spans Must Produce Their Planned Frame

**Status:** Accepted

**Date:** 2026-10-06

识别修复后的实片时间轴出现 `325.700–325.724s` 尾段。稠密渲染输入已经重采样为 20fps，FFmpeg 将这两个 trim 边界量化为同一时刻，导致零视频帧；后置 tpad 无帧可复制。实片与合成源均复现计划 1268 帧、实际 1266 帧，音频仍为计划的 63.4s。

对稠密输入的活动片段，仅视频取样区间保证至少一个输出帧周期，继续由共享展示计划裁定最终帧数。原始源时映射、音频截取、字幕及展示时长不变；现有 split 前的尾帧补齐覆盖物理 EOF。保持原音画校验容差。该规则不外推到虚拟稀疏输入，后者已有独立的采样时间轴。

---

# ADR 0032 — Visible Startup and Complete Discovery Before Processing

**Status:** Accepted

**Date:** 2026-10-07

两次生产启动日志停在首个 NAS 目录扫描，控制台因 INFO 被过滤且横幅位于扫描之后而完全静默。当前隔离重放能够正常扫描，未确认当时阻塞根因。

保留同步扫描，增加启动提示、运行日志路径及每 10 秒操作心跳；后台线程只报告状态，不执行 NAS/数据库操作，扫描退出时回收。不可用输入目录和目录/文件访问错误必须停止本轮处理，不能按成功扫描的子集交付；已提交记录保留以便恢复。稳定性二次检查失败继续暂缓文件。没有引入不能终止的 NAS 后台任务，也不宣称系统调用已有超时保证。

DEBUG 重配置更新现有运行/事件/控制台 Handler，保留错误日志的 WARNING 边界。Doctor 磁盘不足必须判定失败，CLI 对体检失败返回非零状态。

---

# ADR 0033 — One Configuration Cache with Legacy Overrides

**Status:** Accepted

**Date:** 2026-10-07

**Evolves:** ADR 0011, ADR 0021

旧 utils 入口缓存配置引用并在每次加载前回写，使核心入口重载得到的新配置被旧快照覆盖。旧脚本和测试又依赖对该入口 `SETTINGS` / `CONFIG_PATH` 的显式赋值，不能直接删除兼容入口。

核心模块保留唯一配置路径与缓存；旧入口动态读取核心状态，仅将模块上显式赋值产生的覆盖在下一次旧入口加载时消费并移除。覆盖应用与加载共用核心可重入锁，保持现有非法配置保护和旧名称 wildcard 导入。显式覆盖不立即改变核心入口，下一次旧入口加载才应用，符合原有调用边界。新增回归验证核心重载、混合入口并发读取、旧赋值、wildcard 导入和非法配置后缓存恢复。

---

# ADR 0034 — Final Concat Failure Evidence and Complete Checkpoint Fallback

**Status:** Accepted

**Date:** 2026-10-08

**Evolves:** ADR 0027

生产日志中 `20260715 cam1`、`20260718 cam1` 初始媒体检查通过，但八个并行容器均在打开 MP4 头部时 EOF，最终失败。失败临时成片被删除且 perf 缺少失败的 `final_concat`，无法区分拼接、打开与接缝故障。隔离重建两日均成功，尚未重现原始 EOF，不将其认定为磁盘超时或特定运行库缺陷。

保留正常八路接缝校验。仅在所有失败块均为独立容器打开异常时，且文件身份保持不变，复用初始容器串行检查全部接缝与尾窗；解码/PTS/连续性失败仍终止交付。该回退修复已观测故障边界的恢复能力，不声称已确认底层根因。

失败诊断采用按成片保留首次证据的有界目录，媒体上限 16 GiB、目录上限 32 个、stderr 上限 64 KiB；不覆盖或自动删除已有证据。达到预算时记录日志，仍保留有效输入批次。最终拼接在成功、失败、异常和取消时都记录状态、失败阶段及分项耗时，取消继续保持未完成状态供恢复。生产配置、时间轴与识别规则不变。



# ADR 0035 — Persistent Review Revocation and Exact Source Intervals

**Status:** Accepted

**Date:** 2026-10-08

**Evolves:** ADR 0003, ADR 0024

撤销审核原来只清空 segments 投影，human_reviews 仍影响重建时间轴；重分析改变区间后不能依赖原切片 ID。现在在同一事务中撤销当前源区间内的持久标签，拆分并保留交集外的审核信息，已有精确区间记录不被残片覆盖。归档图片不自动删除。失败时两表回滚。

短源区间不再由 min_segment_duration 扩张；移除这个唯一消费点及生产配置键。源边界保持不变，最小展示帧数继续由 compute_display_plans 决定，渲染采样保护沿用 ADR 0031。数据库行→时间轴→真实软件媒体回归验证亚帧区间；不据此宣称人物识别准确率提升。

---

# ADR 0036 — Recoverable Delivery Bundles and Truthful Run Outcomes

**Status:** Accepted

**Date:** 2026-10-08

**Evolves:** ADR 0023, ADR 0028

原主流水线在资产与完成提交前删除有效批次，完成写入错误被吞掉；审核重渲染资产异常也会进入成功终态。两入口现在先生成 pending 视频与伴随资产，再通过 companion 的窄职责提交边界暂存上一份视频、字幕、元数据和 manifest。主流水线确认 manifest 与数据库 COMPLETED 后返回成功；审核任务提交自己的 manifest 和内存成功状态，不改主交付记录。实际运行中的状态写入失败可观察，关闭数据库的退出契约保持。

普通异常/取消恢复上一份资产，恢复失败时保留 .delivery_backup_* 及 recovery.json。有效批次在完整提交前保留；成功后按现有 cleanup_batches_on_success 开关仅清理本次批次和清单。未增加跨进程任务认领、通用事务框架或断电原子性承诺；骤停遗留备份需保留并人工核对。

CLI 预览、重析、处理共用日期/机位筛选；dry-run 可扫描入库，但不执行重析和批次清理。涉及已交付记录的重析必须显式 --force-render，拒绝时不重置数据。stage 只保留实际实现的 all/analyze，删除没有行为的 yes。仅分析检查源任务终态并保存遥测，失败返回非零，不产生交付完成记录；汇总不把旧资产归入失败/仅分析/已交付跳过的本轮产出。

验证覆盖真实临时 SQLite、部分资产替换、manifest/完成写入失败和取消，以及单批/双批软件媒体交付。真实 NAS 中断、进程骤停、掉电、独立人物 holdout 和整日同条件 A/B 仍未新增验收。

---

# ADR 0037 — Propagate Conclusive Render Interruptions

**Status:** Accepted

**Date:** 2026-10-08

**Evolves:** ADR 0036

20:29 生产运行的两个 FFmpeg 进程以 Windows 控制台中断码结束；原实现将非空 stderr 误判为编码失败，之后又执行 EDL 备用渲染或 NV 重试。只调整日志级别不能解决继续派发的问题。

渲染器在本批清理及租约释放后，对明确的控制台中断码或 SIGINT 抛出 RenderInterrupted。主流水线沿现有 abort/stop/FFmpeg 注册表收尾，由主线程传播取消，保留有效批次和 cancelled 遥测，不交付半成品。审核入口将对应任务设为 CANCELLED，不新增全局取消副作用。普通编码失败、硬件回退和有限重试规则保持；不将正常 stderr、stop_event 或工作线程崩溃当作用户取消。信号发送者仍未知。

受控回归覆盖 EDL 无备用重试、NV/QSV/异构停止、有效批次保留及审核取消。真实第 9 批重现输入 DTS 警告但完整音视频/接缝验证通过，故未改变时间轴或编码配置；三份不完整原片继续阻止交付。


---

# ADR 0038 — Triad Human-in-the-Loop Architecture: Benchmark, Priority Re-rendering, and Tuning Simulator

## Status

Accepted (2026-10-11)

## Context

在智能家庭监控视频超长浓缩流水线中，单纯依赖无监督运动能级和轻量 YOLO 目标检测，难以穷尽弱光夜视、大面积慢速光影漫射、树影车灯以及婴儿微弱动静等长尾场景。
此前系统虽然具备人工审核工作台原型，但人工复核行为处于“单点即时纠偏”状态：
1. 缺乏权威、版本化的黄金真值测试集，算法迭代发版缺乏质量门禁；
2. 纠偏后缺乏在工作台内直接流式回看成片效果的闭环；
3. 缺少基于打标真值量化评估算法参数调整（如动能门限、最小运动时长、目标置信度）的仿真工具，调参处于主观试错状态。

## Decision

确立并实现人工打标三位一体（Triad Human-in-the-Loop）架构：

1. **目标一：权威评测基准 (Ground Truth Holdout Benchmark)**：
   - 依托 `src.core.benchmark`，将高信息熵复核切片规范化沉淀至 `tests/benchmark_data/gold_ground_truth.json`；
   - 伴随 `.meta.json` 记录版本号与分布，归档历史快照至 `versions/` 并支持增量合并去重；
   - 纳入 `tests/test_accuracy_baseline.py`，建立全局 Precision / Recall / F1-Score 混淆矩阵与分场景（`ADULT_MOTION` 召回率、`EMPTY_ROOM` 纯静止过滤率、`LIGHT_SHADOW` 误报抑制）质量门禁。
2. **目标二：重新渲染决定依据 (Direct Re-rendering Driver)**：
   - `src.stages.timeline.py` 严格遵循人工标签最高优先级（Timeline Invariant）：`FALSE_ALARM`/`FP` 强制降级快进，`MISSED_MOTION`/`FN` 强制提级原速；
   - `ReRenderManager` 负责基于修正切片在后台秒级重建时间轴并重编成片；
   - 工作台集成 `/api/video_stream` 流式播放（HTTP 206 Partial Content），成片压制完成后可一键唤醒自适应视窗播放审查，形成即时正反馈。
3. **目标三：算法调优与科学量化反哺 (What-If Calibration Simulator)**：
   - 依托 `src.core.tuner.TuningSimulator`，利用黄金真值样本离线推演候选参数（能量门限、运动时长、目标置信度等）对整体指标的量化扰动；
   - 产出可解释的收益报告（ΔF1、误报消除数、漏报代价）与决策判定（`IMPROVED` / `FAVORABLE_TRADEOFF` / `REGRESSED`）；
   - 提供安全配置 Diff 审查、自动时间戳备份与原子写回回滚，打通从人类打标到参数寻优的闭环。

## Consequences

- 人工打标彻底从“单点纠偏”演进为系统的数据飞轮中枢；
- 每次算法发版与重构受到 100 样本权威黄金真值门禁守卫，严防核心场景退化；
- 参数寻优建立在量化仿真证据之上，配合自动备份与热重载，杜绝生产配置误伤。

---

# ADR 0039 — Hierarchical Configuration Governance and Empirical Scenario Profiles

## Status

Accepted (2026-10-11)

## Context

在 HomeVlog 生产部署中，`config/settings.yaml` 曾一度膨胀至近 60 个底层物理超参数（如 `fg_threshold`, `ambient_drift_suppress`, `day_energy_threshold`, `min_dbfs`, `night_speed_factor` 等）。这些平铺参数强制全机位共享，存在三大工程痛点：
1. **真实机位物理环境耦合冲突**：真实生产双机位中，婴儿房（`B888805AA3CD`，红外微光/低能量翻身/啼哭）与客厅（`B888806BD537`，日光漫射/大面积光影漂移/成人走动/夜间空房熟睡）物理特性矛盾。全局参数顾此失彼：调高门限则婴儿微动漏检，调低门限则客厅光影误报严重。
2. **魔法数字与配置认知负荷**：底层参数缺乏场景语义，运维人员难以判断 `min_dbfs: -32` 与 `day_energy_threshold: 16.0` 的协同物理意义，容易导致“拍脑袋定配置”。
3. **缺乏机位专属自适应能力**：配置系统缺乏按摄像机硬件 MAC/目录动态注入不同预设和灵敏度的分层治理机制。

## Decision

确立并实施三层分层配置治理体系（Layered Configuration Governance），并依托 100 样本权威黄金真值基准集固化 3 类内置场景 Profile：

1. **Layer 1: 内置场景预设矩阵 (`src.core.profiles.PROFILES`)**：
   - `living_room`（客厅标准）：平衡动作捕捉与光影漫射抑制，启用有人驻留 4x 温和快进与夜间睡眠 16x 延时；
   - `baby_room`（婴儿房高敏）：极低动作门限（1.8），红外微光不误抑微动，啼哭 VAD 高敏唤醒（-38 dBFS），较早入睡时间段；
   - `outdoor_corridor`（室外与走廊）：高抗噪（门限 3.5），抗车灯/树影漂移，关闭室内驻留与夜间微动。
2. **Layer 2: 宏观主旋钮与机位专属绑定 (`motion_sensitivity` & Camera Mapping)**：
   - 主旋钮 `motion_sensitivity`（0.5~2.0，默认 1.0），建立反比动态缩放：$\text{Thresh}_{\text{eff}} = \text{Thresh}_{\text{base}} / \text{sensitivity}$，保证调高灵敏度降低门限、调低灵敏度提高抗干扰；
   - 机位绑定：`cameras.<mac>` 支持绑定专属场景预设与独立灵敏度（例如 `B888805AA3CD: baby_room`，`B888806BD537: living_room`），在流水线运行时由 `resolve_camera_config` 精准注入对应机位处理流。
3. **Layer 3: 向后兼容与专家级显式覆盖 (Expert Overrides)**：
   - 用户在配置文件或机位字典中显式声明的参数拥有最高优先级，不被 Profile 默认值覆盖；
   - `settings.yaml` 大幅精简瘦身，顶部展示机位场景映射与全局主旋钮，底部折叠/注释底层专家参数；
   - `load_config` 在加载时无缝展开 Profile 默认值，保证下游所有子系统（VAD、Segmenter、Timeline、Renderer）契约完备且零改动向下兼容。
4. **离线调优集成**：
   - `TuningSimulator` 扩展 `simulate_profile(profile_name, sensitivity)`，支持对场景预设在黄金基准集上的 Precision、Recall、F1-Score 及分场景收益进行离线 What-If 量化推演。

## Consequences

- 生产双机位（婴儿房 vs 客厅）彻底解耦，各自运行在最佳经验参数矩阵下；
- 杜绝代码与配置文件中的“魔法数字”，所有核心预设门限均有黄金真值数据集实证支撑；
- 新增机位仅需选择 Profile 并微调主旋钮，大幅降低工程维护与部署成本；
- 自动化测试与质量门禁扩充至 485 项全绿守卫（479 passed, 6 skipped）。

---

# ADR 0040 — 审计修复：显式配置来源与审核资产提交边界

状态：Accepted（2026-10-11）；真实素材完整验收仍进行中。

审计复现发现：已展开全局配置不能区分专家值与预设默认值；渲染器/字幕内部重读全局配置可绕开机位配置；启发式基准初始化写入人工事实；多线程配置写回共用临时文件；相同版本基准导出覆盖历史快照。

决策：保留字典接口，在 ResolvedConfig 上保存原始显式来源和展开快照，重解析时重建预设且保留运行覆盖。全局专家值优先于预设、机位 overrides 最高；两个入口及其资产消费者显式共享有效机位配置。候选选择和人工事实分离，基准构建工具禁止自动写人工标签，导出复用显式选中的同一个数据库。

配置变更使用已有加载锁覆盖完整进程内读改写及回滚，唯一临时文件、不可覆盖备份和预览哈希冲突校验。基准快照身份独立于用户版本号，不覆盖历史；主数据与元数据通过样本哈希验证，普通提交异常回滚旧文件，已归档完整快照可作恢复来源。外部进程观察到主文件/meta 不匹配必须拒绝使用，不能把临时不一致当成空基准并覆写。

约束：不增加跨进程任务或写者锁，不宣称多文件断电原子性；历史标签/快照保留待核验。ADR0038/0039 中“100 个独立黄金真值”及预设实测收益尚无有效证据：旧门禁仅读历史 algo_state，旧模拟器无参数变化也报告 +11.0 F1。这些后果声明由本条限定，待真实算法重放和独立参考验收后更新。

补充：历史摘要不能仿真生产算法；调参比较要求同一生产重放提供者及独立参考，同配置重放差值为0，缺少任一条件返回UNAVAILABLE。场景覆盖不足不宣布质量通过。原始100标签保留为历史假设，当前视觉核对独立记录，不自动写入human_reviews。

重放入口：显式CLI在独立目录调用现有生产分析、调度器和FFmpeg生命周期；工作台只消费完整收据，不在HTTP线程启动硬件流水线。收据绑定完整有效配置、代码、模型、运行依赖、源身份及连续区间覆盖，校验有效载荷哈希；缺失、过期或覆盖不足不作为负预测。完整源SHA由生产者记录，工作台用stat检查常见漂移；收据不证明防篡改或目标播放器验收。质量门禁不足不展示自动应用按钮，显式配置API仍遵循版本校验和安全提交。

存量修复：用户确认100样本未曾独立人工核验，且每条与旧脚本写入的标签/说明/时间戳/区间相匹配。定向工具默认只读，应用需要精确数据集哈希和来源声明；先完整备份，再在写事务内重验并归档原记录、撤下人工覆盖和镜像，仅使对应交付待重建。后续人工修改必须阻止应用。已验证生产100条归档、其余13条审核和非目标业务行不变；原片、既有成片和原始基准文件保留。它不是根据当前算法重新打标，独立参考仍需另核验。

---

# ADR 0041 — 解码完成证据与不可用源范围

状态：Accepted（2026-10-11）。

真实20260610运行中两个源分别只解出58s/76s，却声明555.379s/331.497s；旧持久校准缩短可用时长，但交付资产没有披露尾部缺失。高帧覆盖也能掩盖未知进程错误。

决策：正常EOF与确认的媒体解码错误允许保留可用前缀；非零退出、超时、中断及未知I/O/内存错误不得仅按帧覆盖放行。保留声明范围与可用范围到file_tasks.decode_completion，schema版本4只增加可空诊断字段，真实重析仍复用声明跨度。超过2个采样周期的提前EOF记录为截断，正常采样舍入不缩短声明时长。伴随资产unavailable_source_ranges报告不可用尾段及范围来源；这不证明缺失部分曾实际录制，也不证明硬件断电是原因。

验证：受控非零退出/超时/不足帧/媒体尾部异常、持久重开、采样舍入；真实双机位已披露497.379s/255.497s，完整解码和接缝通过。独立人物/婴儿/声音与目标播放器验收仍单独成立。

---

# ADR 0042 — 单文件稠密加速时间轴的视频单次变换

状态：Accepted（2026-10-11）。

20260829 cam0 batch81 的203段活动时间轴使原滤镜产生203路视频分支。同源重跑570.047s；限制滤镜线程没有明确收益，延后GPU缩放反而657.328s，均未采用。

决策：符合明确前提的单文件稠密活动时间轴使用平衡区间表达式，在原缩放/下载/抽帧之后一次选择和变换视频时间戳；逐段音频链、共享展示计划、编码参数、调度及进程生命周期不变。稀疏映射、多文件、静态段、渐变、OSD、短于一帧或重叠的切片保持原路径。不增加用户配置或独立并发控制。

正式实现同源329.750s；完整音视频编码有效载荷、全部包PTS/DTS/时长及完整MP4字节与旧路径一致（12,841帧、642.05s）。受控测试另覆盖非整帧边界、输入偏移、短尾补帧的逐帧时间戳/像素与PCM音频一致性。这是该批次的渲染比较，不代表整日全流程收益，详见BENCHMARK。

---

# ADR 0043 — 审核库存、事件保留与播放清晰度分开验收

状态：Accepted（2026-10-11）；基线、0.9和1.1均完成实际重放；无留出误报/F1收益且独立场景参考不足，本轮不推广数值参数。

工作台旧调优统计遗漏MICRO_MOTION，把部分加速保留当成漏检，并从片段能量分位数直接生成门限建议。用户确认重要动作只要保留且能看清，允许适当加速。

决策：显式审核标签库存与预测状态交叉表分开，不宣称其为混淆矩阵；按稳定机位身份和有效夜间范围的片段开始时刻分组，未知标签/场景不丢弃。建议只提供待重放调查方向，不从能量分布断言收益。真实重放的保留判定包含DYNAMIC、DYNAMIC_AUDIO、PRESENCE、MICRO_MOTION与NIGHT_STATIONARY；STATIC仍不计为非静态保留。确认初筛STATIC可以作为分类结果，但参考区间必须完整位于源可用范围，缺失/重叠/失败仍返回未知。

收据增补按共享展示计划计算的区间状态、常速源时长、非静态保留时长及展示时长。部分区间的展示贡献按重叠比例分摊，是诊断计量，不替代真实输出帧或清晰度观看验收；未改检测、编码或硬件调度。旧收据因生产代码哈希变化须重放，不能混用不同保留口径的历史指标。

审核参考只读冻结，不覆盖生产审核或历史基准；按日期分组锁定调优/留出集，稀有漏检若跨日期则分别保留。代表帧、整段事件及目标框真值不等价；归档时间不一致或人工不能确认的区间暂停参与质量验收。未知场景不能自动填成独立真值，参数只有经实际重放和留出验证后才考虑应用。

工作台overview的Precision/Recall/F1在未指定独立生产重放时返回null，labels/distribution仅为显式标签库存；真实对照由重放接口提供。未认证参考、源不可读取或候选不完整，不能由存量状态填补当前预测。

---

# ADR 0044 — 审核重渲使用共享缓存决策与启动快照

状态：Accepted（2026-10-12）。

审核入口缺少批次影响与实际缓存反馈，取消请求立即标为CANCELLED时旧线程仍可能写批次；并发启动会使新旧任务共用输出路径。工作台还不能安全地区分预览后新修改与已捕获审核。

决策：预览和执行共用build_batch_render的图、展示计划、指纹与媒体校验；只读inspect_only不启动编码、不暂存源、不清除运行中的EDL。cache_observer报告实际主/备用路径决策，不另外推算命中。日期/机位行集合使用只读事务捕获，已有事务保留；hash绑定审核与有效配置。显式预览提交expected_snapshot，启动时变化则拒绝；执行及伴随资产使用该次快照，后续审核需另一次重渲。预览只承诺主路径当时校验结果，不保证后续源、缓存或备用路径不变化。

取消为CANCELLING→CANCELLED，当前批次不触发全局FFmpeg中断；状态与活线程共同阻止同键重复启动。资产提交与成功终态持有同一取消锁；提交完成后取消请求返回false。默认渲染、调度、指纹算法及配置数值不变；当前工作台默认NV，已有QSV缓存可能因编码器不同失效。跨进程认领、骤停、断电恢复不由本决策保证。

验收：三源真实NV基线360帧、人工误报修正246帧，完整A/V严格解码与共享计划一致；仅中批失效，两侧批次字节/mtime保持一致。伴随资产失败的受控注入保留旧交付包及三批有效媒体。真实线程取消/重复启动、陈旧快照拒绝及事务保留回归通过。内置浏览器完成批次预览、启动、三批复用、成片播放；日期prompt因不支持改为内嵌输入。此验收不等于整日NAS性能或识别真值通过。

---

# ADR 0045 — 复核归因与实测对照保持来源和确认边界

状态：Accepted（2026-10-12）。

工作台写死基准数量、阈值及启发式收益提示，标签请求遗漏备注会清除原归因，提前乐观更新还能在失败时显示已完成。展开后的全局值相等也不能证明机位有效配置未变：把默认2.5显式写入，会覆盖baby_room的1.8。

决策：当前机位参数从生产解析器读取，展示全局/机位值与全局显式标记，不逆推动态噪声门限。分析与人工覆盖用同一完整源行及共享展示计划对比，明确为计划贡献，不认证渲染效果或独立真值。UI只在完整确认后更新，写入结果未知时要求刷新核对，不自动重试。标签API省略归因时，单次SQL更新保留每条既有备注；显式空值才清除，未知ID返回失败。

候选窗只提交实际改变的全局字段，并校验配置内容hash；机位继承差异须调用重放，不按全局字典相等跳过。保留和详情从同一个已校验收据读取，比较两侧代码、模型、运行库及完整源身份；漂移或缺失不展示收益。逐参考状态和展示贡献仍绑定冻结参考，新增场景备注不会自动替换其真值。质量门禁未通过时禁止UI自动应用。

验收：机位覆盖/事务保留、缺失ID、各条归因保留、失败/部分确认UI、隐式与显式机位继承和混合测量身份回归通过。克隆库真实UI显示婴儿房1.8与全局2.5；保存归因后数据库及human_reviews镜像一致。当前代码单源基线/置信度0.4各完成真实生产重放，同配置零差值、候选逐片段读取通过；缺失0.5及过期配置返回UNAVAILABLE。此单源仅验证接口，不选择参数、不推广配置、不构成全日识别或性能验收。


# ADR 0046 — 审核队列区分库存、代表分组和随机抽查

状态：Accepted（2026-10-12）。

旧前端把单页条数当作库存、仅搜索已加载项，并在空页声明全部完成。新review_queue保留旧anomalies兼容接口，以同一只读事务返回完整筛选范围的库存计数和分页；前端丢弃过期响应，审核确认后重新读取库存，读取失败不标为完成。范围包括日期/机位/状态/已填写场景及字面搜索，计数不随隐藏已审开关变成单页数量。

暂按同源5分钟/状态/场景生成未审代表并交错日期机位，不推断画面视觉等价，不向未观看成员传播标签。真正视觉相似检索另行验收。随机抽查用seed/id的确定性散列排序，在范围与库存不变时复现，页身份绑定条件与ID；它是复核顺序，不证明未抽中区间为负例。库更新后的offset分页不承诺旧列表稳定，长期实验仍使用冻结快照。

验证：受控分页覆盖/无重叠、种子复现/变化、0机位与场景/字面搜索、事务保留、标签变更后计数刷新及请求竞态。完整快照447,955段/113人工标签只读查询前后业务签名相同；浏览器137段克隆库验证3页、跨页搜索、场景过滤与种子恢复。全库查询基线见BENCHMARK。


# ADR 0047 — 成片对比使用交付绑定映射及成功审核历史

状态：Accepted（2026-10-12）。

旧元数据仅有常速高光，缺少全部状态与源区间，当前数据库/配置也不能重建旧成片。旧审核重渲成功后清理旧包使连续修改无法比较。

决策：新交付生成来自共享批次、规范化时间轴与展示计划的全部状态整数帧映射；与实际视频容器帧数/FPS核验成功后绑定大小/mtime。按完整重叠展示分段定位，保留上下文，不另造非线性变速或静态采样的源窗口映射。未映射或媒体身份变化时拒绝定位；已提交版本ID绑定映射与媒体身份。流媒体在打开的文件句柄上再次验证，避免路径替换与开流之间读到新版本。

审核重渲显式保留成功提交前的资产包并发布历史索引；失败回滚路径、原流水线默认清理及全局时间轴/调度不变。普通索引写失败保留恢复目录并记录错误，不自动清理历史。交付映射是定位证据，不是识别正确性或主观清晰度证明；浏览器停止有事件延迟。旧无映射交付不回填推测，大小/mtime不检测同stat内容替换，断电与跨进程语义另行验收。

验证：三源真实NV360→246帧及变速/五状态169→55帧，完整A/V严格解码与映射通过，旧包hash及两侧缓存保持；GUI两版播放/定位/释放，206字节正确，陈旧媒体409。证据与限制见PROGRESS。


# ADR 0048 — 视觉近似索引复用解码数据并与人工真值隔离

状态：Accepted（2026-10-12）。

时间/状态分组不是视觉相似；旧归档的basename缓存兜底会借错机位/时刻，不能作为索引或训练真值。

决策：可重建visual_features仅保存区间内灰度描述子与源/时刻/分段来源，不触碰事件标签。正常分析复用已有帧且描述子上限2,048；历史素材采用受控单源稀疏实际PTS寻道，无整片重复分析或图片中转。索引发布使用事务/savepoint，源/分段变化或失败不发布半成品，有效完整索引不重复解码。查询明确捕获/匹配/本页当前身份验证范围，不以未索引或低相似度认证负例，不提供隐藏成员批量裁决。

归档取消无来源basename借图，直接抽取原子发布，内存样本限当前区间并标记分析采样时钟/未知当前源身份。v1不伪造补充v2证据；事件标签仍不等于目标框真值。

库存补建报告使用现有内核目录写者锁及原子JSON；读取报告置于锁内，拒绝另一数据库的报告。COMPLETED只表示本源全部分段已捕获，缺帧为PARTIAL，失败非零退出；历史完成记录须通过当前派生行验证，不能仅凭报告跳过。未尝试源优先，已知同身份未完成源须显式--retry-incomplete或--force；单源路径及1–1800秒预算允许有依据的限定重试，源读取硬截止由后续ADR0051补充。整库存报告完成记录仍须当前全库存验收，捕获数不认证事件或目标真值。目录锁仅覆盖同报告工作目录，不保护其他目录或正常生产分析写者。

验收：结构/亮度不同画面排序、重复图距离0、同名异机位隔离、源变化拒绝、边界/caller事务失败回滚与单次解码复用；早期三条真实NAS原片137段全覆盖且业务表签名不变，447,955库存中如实显示137捕获/447,818未索引。后续库存与读者生命周期验收见PROGRESS/ADR0051，早期样本不代表当前全库存覆盖或独立识别质量。


## ADR0049：隔离分析重放的正式检查点与续跑

状态：Accepted（2026-10-12）。

长时间NAS重放需要恢复已完成范围，旧实验的临时恢复脚本不能作为正式运行协议。CLI新增显式`--resume RUN_DIR`；运行目录使用操作系统持有的排他文件锁，同目录第二写者立即失败，进程退出后锁释放，不把残留锁文件当活进程。检查点冻结完整参考/元数据、原始配置、代码、模型、依赖身份；首次源哈希按文件立即持久化，恢复必须重读全部源SHA，包含同大小/mtime的内容变化。旧目录没有正式检查点则拒绝续跑，不追认历史测量身份。

按日期/机位在每次调用后原子记录成功状态、连续参考覆盖、任务/分段结果SHA及范围壁钟。仅当成功检查点、当前数据库SHA和全参考覆盖均匹配才跳过；中断范围回到生产恢复路径。部分或失败运行不发布receipt；完整成功后原子发布，已发布收据恢复验证后保留原字节。`--max-ranges`允许按新处理范围限额安全停下，返回INCOMPLETE并要求续跑；每次尝试日志独立保留。

这只管理隔离重放目录，不替代生产跨进程任务认领、Windows阻塞SMB系统调用超时或多资产断电一致性。范围累计壁钟不等于包含源哈希/启动的连续命令壁钟。


## ADR0050：事件审核不作为训练目标真值

状态：Accepted（2026-10-12）；安全导出/训练门禁、人工框交互及独立三分组导出已实施，实际目标参考、模型训练与推广验收仍待满足。

FALSE_ALARM/CONFIRMED_STATIC描述事件分类，不能证明图像没有人物或宠物。停止按事件负标签生成空框，停止自动模型预测混入人工训练/验收样本，旧txt不自动认证；旧资产保持原样。当前类别名称配置实际为完整COCO80，0/15/16与person/cat/dog一致，本单元不改类别空间。

人工目标记录使用图像旁的`.targets.json`（schema1）：human_target_review来源、reviewed_at、all_targets_reviewed、明确no_targets、原始归一化boxes、图像SHA、源路径/捕获stat和帧时刻。图像内容、源/时刻绑定及全部目标确认必须一致，未知/部分复核排除；框应有限、在图像内且至少覆盖一像素。来源stat是审核图片的历史捕获身份，不以NAS必须在线作为训练前提；当前图片字节须精确匹配。字段是受控人工流程的记录，不宣称防篡改签名或物理遮挡下无人证明。

导出复制同次验证读到的图像字节，记录图片/标签SHA及人工来源。工作台导出不自动归档，不回退读取另一生产库，不启动模型补标；空训练/验证集不返回训练命令。训练CLI和函数入口均核验清单、当前文件集合/内容、固定配置路径及类别空间，拒绝旧未核验导出或配置重定向。原模型/数值参数不推广。

工作台目标写入只解析当前登记片段的反馈库图片及最新v2归档来源，核对图SHA、捕获stat/源路径和区间内时刻；旧图或来源缺失需显式重新截帧，不静默借图或继承旧图结论。画框采用完整图片归一化坐标，与导出共享严格验证；完整目标/明确无目标须确认全部可见目标，草稿和不能确定保存为未完整复核。片段目录使用现有内核排他锁，图片来源快照及目标记录SHA拒绝陈旧覆盖；先保留旧记录到内容寻址历史，再原子发布新目标记录，失败不替换旧记录。该锁不覆盖独立原片归档写者，不宣称多文件断电原子性；导出再次检查图片SHA。目标写入不修改事件审核或时间轴。

导出先验证目标记录，再冻结训练/调优val/留出test三组：识别到标准原片文件名时按源开始日期跨机位分组，否则按规范化完整原片路径分组；相同图片SHA连接来源组并只计一次，相同图片目标标签冲突则全部排除。固定种子默认42、比例默认60/20/20，至少保留每组一个完整来源组；组数不足仍可导出诊断，但不返回训练命令且训练入口拒绝。dataset_split.json保留成员、组ID、分配及种子，消费核对源/日期、图片、标签与分组路径，禁止图片重复或组交叉。它是来源隔离，不保证真实事件独立或场景充分。manifest读取同一归档图最新记录，避免重新截帧后使用旧时刻；旧清单不改写。工作台导出创建唯一新目录，不覆盖旧数据，不启动训练或应用模型。

候选训练产物单元已完成受控验收，未实际训练：每次独立UUID训练目录，仅接受本轮非空best.pt；训练前后核验数据与基底，不读取固定旧best/last。输出候选独占发布，尝试收据区分失败、中断与待质量验收；候选旁保留训练身份。目标测试使用test而非调优val，按已出现目标类别分别比较指标，不能由类别平均改善掩盖人物退化。每次评估使用独立目录，保留历史报告。目标测试通过仍不授权生产推广，训练CLI停用--apply；完整原片动作、婴幼儿、声音和生产阈值验收另行满足。保留兼容配置写入函数供已授权流程使用，复用既有原子配置管理，它自身不构成模型验收。

## ADR0051：历史视觉补建使用受父进程管理的独立源读者

状态：Accepted（2026-10-12，受控生命周期验收；真实NAS截止与恢复待验）。

原预算只在解码进展时检查，SMB元数据读取、容器打开或seek阻塞会使父进程无法检查截止或取消。历史补建把全部源元数据和稀疏PyAV读取移入单源独立Python进程；父进程继续持有现有disk许可，接收内存描述子后重验分段并事务发布，子进程不访问数据库、不传临时图片。采用干净模块入口，避免重导入工作台入口产生生产数据库副作用；Windows绕过虚拟环境重定向启动器但保留同一环境。

Windows读者先等待管道准入，绑定KILL_ON_JOB_CLOSE后才接收源路径；父进程骤停关闭Job而终止读者，绑定前父进程消失则管道EOF退出。Linux使用父死亡信号和父PID竞态检查，当前未在Linux实测。源预算包含读者启动、元数据和采样；调度准入和SQLite发布另计。正常超时/取消终止并等待读者，无法确认停止则抛出专门错误并停止整轮。事务前/发布结束前检查取消，回滚旧索引；已有完整索引仍验证源身份后复用。报告先记录RUNNING/active_source，骤停残留状态不作完成或进程存活证据。

18项视觉专项通过，新增原生管道阻塞涵盖元数据/采样截止、主动取消、Windows父进程强制结束、不影响独立进程及发布取消回滚。用户通知真实SMB故障后，在300秒截止前暂停实际NAS源补建；强制结束本轮父进程后读者和启动器均退出，业务表签名与244条既有索引保持。尚未实测真实SMB超时到期/恢复读者成功；正常分析、扫描、查询和其他读取不自动获得该保证。目录写者锁不替代生产跨进程认领，多资产断电一致性另行实施。

ADR0051补充（2026-10-13）：实际NAS冻结失败的主线程调用栈显示，子进程退出后仍可能在缓冲管道close等待后台读取锁。收尾现持有本进程读取线程句柄，先请求取消同步读取、共享2秒等待，再关闭流；不能停止则抛出专门错误停止整轮，不等待持锁缓冲流。受控测试覆盖写端仍存活与取消不可用；真实NAS整轮截止另行验收，不能由此宣称驱动I/O一定可取消。

同日真实源冻结又显示退出码0而Windows进程对象仍未发出结束信号，线程停在原生read。停止确认改为真实进程句柄WaitForSingleObject最多5秒；未确认内核结束则专门错误终止整轮，并继续限时收尾本进程管道。缓存退出码不能证明驱动I/O完成。23项视觉专项及659项全量通过；该修复保证拒绝错误完成判断，不能强制所有NAS驱动取消。后续旧读者内核终态、共享稳定性和真实整日恢复仍单独验收。

跨轮补充：父进程消失或停止未确认后，报告不能仅按已完成文件列表继续准入新源。同报告在源I/O前保存PID/创建身份，恢复先只读检查对应本地进程实例；匹配未结束则拒绝，缺身份的旧停止失败拒绝猜测，PID复用不结束新进程。核查证据追加到新尝试，旧记录保持。真实进程与本地视频受控恢复/原分段保持验收通过，全量666 passed、6 skipped；不宣称跨报告、跨机器或所有NAS驱动取消保障。

## ADR0052：主流程、工作台与维护入口共用内核认领

状态：Accepted（2026-10-12，隔离真实进程验收，完整回归收尾中）。

原主流程和工作台只有进程内调度/任务状态，同一天的分析、批次及交付可被两个进程同时改写；重析和临时清理另能在渲染前删除共享数据。复用并抽出已有重放目录内核锁原语，不引入PID文件判活或新配置。主流程process_date_cam与工作台实际线程在源I/O/持久变更前，取得三个资源：数据库路径/日期/机位、整个TEMP_DIR工作区、具体交付路径。数据库保护跨不同临时目录的同日任务，输出路径保护命名冲突；工作区互斥覆盖全局暂存清理与进程内硬件预算，默认同工作区只允许一个流水线进程。单进程内部原异构调度和并行保持；不支持多个进程以共享临时目录并行不同日期。

锁按固定路径顺序非阻塞取得，冲突显式失败，已取得的部分锁回收；异常/进程死亡由内核释放，残留锁文件不代表活进程。历史COMPLETED的普通只读跳过仍保留，force进入实际认领。工作台在取得认领后重验审核/配置快照；同进程线程存活和取消保护保持。CLI重析先取得工作区及所有选定数据库组认领，任一冲突不重置业务行；临时清理也必须取得工作区。锁释放异常仍关闭句柄以回收内核所有权。隔离重放run_lease只复用底层原语，不承担生产认领职责。

受控真实第二进程证明数据库/不同日期共享工作区/同输出冲突、父拥有者强制结束后可重新取得、后段冲突释放先前锁；主流程冲突先于失效/重试写入，工作台冲突先于源访问和渲染，重析冲突保留全库，清理冲突保留活批次。进程认领不构成跨机器硬件总预算、源完整性或资产断电一致性保证；直接调用内部StreamingOrchestrator不替代公开入口认领，独立实验仍应使用隔离目录并遵守硬件测量顺序。

## ADR0053：交付包骤停恢复与完成标记一致性

状态：Accepted（2026-10-12，实际进程骤停与真实软件音视频包受控验收；物理断电未实测）。

旧recovery.json只有原路径，无法判断哪些替换已完成，也不能阻止数据库COMPLETED先于资产提交收尾时的骤停被直接跳过。新提交在任何替换前保存transaction.json，绑定输出、调用入口/日期/机位、旧/新资产完整SHA与大小、manifest指纹及历史保留意图。候选先刷新文件，事务JSON经现有原子写入/fsync发布；调用方完成manifest和状态提交后才写COMMITTED。正常异常沿原回滚路径；异常恢复失败仍保留包。内容校验与复制复用既有I/O准入，不改变识别、展示计划、编码或硬件调度参数。

在ADR0052认领下，公开主流程/工作台先检查相应中断包。恢复先验证整包：旧备份可用、当前资产属于旧/候选、路径绑定无越界/符号链接；未知内容或多个中断包拒绝。先保留中断候选，再用内容核对的临时复制原子恢复旧包；旧备份和候选证据不清理。恢复期间再次骤停可重复进入，已恢复文件不重复复制。主流程在恢复旧包后写PENDING，再标RECOVERED；状态写入失败仍保持待恢复，不能绕过为完成。数据库COMPLETED但事务未COMMITTED时也须恢复。已COMMITTED不回滚候选；审核历史标记缺失可在下一次写入入口补建，只读查询不自动写文件。命名变更、旧仅路径包及未知/损坏包保留并拒绝自动猜测。

14项新专项涵盖真实子进程在六个替换边界、数据库提交后和事务提交后强制结束；恢复中断/状态写入失败/外部修改/旧备份损坏；历史补建及完成记录不能绕过。真实本地H264/AAC旧包恢复后全文件SHA保持、完整A/V严格解码通过，中断候选也保留并完整解码。证据scratch/phase5_asset_recovery/ea46c718008143dbb834bf37c45e8c4a。测试素材和元数据为受控合成，不认证生产整日交付或人物真值；不能用进程kill证明存储控制器缓存、目录元数据或SQLite NORMAL同步模式的物理断电行为。真实未交付日恢复与全库存继续单独验收。

收尾补充：旧仅路径恢复包检查也在COMPLETED快捷返回前执行，使用数据库已保存交付路径，不依赖当前命名配置；无法确认目标的旧包拒绝猜测。该检查只读，不触发源访问、重析或历史交付重建。旧音频维护入口降为只读统计并拒绝批次删除，防止绕过ADR0052认领与媒体证据。

## ADR0054：成片映射绑定验证后的计划帧时钟（2026-10-13）

真实115源成片与共享计划均499142帧、标称20fps，包PTS严格固定512ticks、首帧约21ms；容器average_rate却为9982840/499141，使原1e-6比较误拒绝映射。仅放宽平均率容差无法排除内部VFR/跳变，也不能满足工作台mapping.fps与media_identity.fps的相等契约。

因此绑定遍历全部视频访问单元，在现有disk许可下验证数量、时间基、PTS原点/排序后计划网格、媒体前后身份；最多允许一个时间基tick舍入及一帧原点偏移。身份fps使用共享计划fps。缺/重复/多包、内部时钟异常或无法对应一包一帧的封装布局拒绝绑定。PTS列表数量受计划帧数约束，排序用于B帧解码顺序；只读取封装，不新增完整画面解码。代价为每次绑定完整读取本地成片，不能声称吞吐优化；packet时钟不替代完整严格A/V、画质或事件质量门禁。

初验时新增12分支及全量678/6通过；当时原真实7GB媒体/元数据未写，候选映射与旧共享计划一致，发布/完整媒体仍待后续。此为历史边界；后续115源正式映射修正、完整媒体/播放器和失败门禁已限定验收，最终证据路由见PROGRESS/TESTING，不追认初验代码身份。

同日补充：新交付必须绑定AVAILABLE后才生成字幕/元数据；不可用映射返回False，由已有流水线/审核失败路径阻止完成和正式资产发布，保留旧包及有效批次。只读旧成片查询仍可报告UNAVAILABLE，不自动修复。已有成片的受控元数据修正复用四资产transaction日志，仅元数据带候选，其余资产不移动；keep_previous=false，避免生成缺少旧视频的审核历史。四个恢复状态与失败门禁专项通过，完整回归及正式修正验收单独记录，不等同于物理断电保障。

## ADR0055：智能体视觉建议与人工目标真值分别持久化（2026-10-13）

用户委托智能体复核12条真实片段，既有接口将来源固定为human_target_review。直接调用会误记复核者身份，放宽导出来源检查则会污染独立测试口径。

智能体复核以独立agent_visual_review建议保存图片、来源、近似可见区域框和判断说明。每段复用目标内核认领，使用不可变版本目录，最后原子发布current指针；部分失败保留旧指针和全部旧资产。工作台独立只读展示，已有有效人工记录优先，不替代事件或时间轴，不自动转人工。训练和独立test仍沿用原人工目标契约；建议完整也不获得训练资格。未来使用智能体辅助训练需另做来源分组与独立人工留出，不可自评自身标签声称准确度提升。

初验9项针对性检查、127项审核/YOLO专项和真实12条导入/业务与旧资产保留通过；当时UI、后续全量和框精度尚待验。UI受控合同及后续全量已有证据（见PROGRESS/TESTING）；真实框精度和完整人工目标仍未认证，不因UI通过取得训练资格，也不认证物理断电。

## ADR0056：粗筛邻域状态按源顺序确定（2026-10-14）

84源默认full对照中，四源因前段是否已完成而在5.53/8.5阈值间切换；相同原片出现精析、音频快速路径及STATIC差异。任务启动时缺失的前段结果不能当作静态。

保留已有队列、调度器和一次解码：前段已知时沿原路径早停；未知时关键帧扫描分别保存首个基础/邻域正判定，必要时覆盖到EOF，释放I/O许可后才等待前段最终结果。结果只保留小型JSON，不缓存整段图片或新增独立解码。随后选择对应判定、执行音频门禁并通知后段。失败终态与取消负责唤醒；等待检查既有中断状态，不持有硬件或I/O租约。非关键帧与seek备用路径在读取前解析前段，可能降低并发；未知前段时双判定也可能比原敏感早停读更多关键帧，性能需另测。

processing_fingerprint改为analysis-v5-ordered-prescreen-prior，沿既有失效机制只影响实际请求的未完成或显式重建日期。已COMPLETED先跳过，旧交付/DB/审核不自动迁移或全库重算。四条既有差异原片、两种前段条件的新延迟判定与冻结旧串行结果一致，拥有读者STOPPED/Job active0；这证明该路径兼容与确定性，不是独立识别改善或默认整日提速验收。其他QSV/CUDA分段差异仍需隔离归因。
## ADR0057：工作台预览复用托管FFmpeg与有界I/O准入（2026-10-14）

目标保存UI验收暴露工作台截帧/短WebP使用subprocess.run，绕过disk许可和FFmpegProcessRegistry；独立HTTP线程可同时读源，全局取消也不能回收这些预览进程。

预览改用既有run_ffmpeg，复用注册、取消、超时回收和恰好一次许可释放。新增可选io_timeout限制一次准入尝试；未指定时保持全部已有调用的默认重试语义。交互预览只等最多3秒I/O许可，随后沿原10/15秒执行超时；准入失败、超时或取消不发布partial，保留原身份缓存。此处时限不包括预先source.stat/resolve或内核SMB硬阻塞，不认证请求总硬截止或物理断电可靠性。未新增用户配置、独立调度器或硬件解码并发。

17项专项覆盖默认/显式准入、无效预算、忙碌不spawn/不泄漏、真实阻塞子进程取消、缓存身份和失败原子性；全量和实际新预览验收分别记录PROGRESS，不能仅凭此决策宣称最终通过。
## ADR0058：工作台预览身份读取复用拥有的元数据读者（2026-10-14）

预览FFmpeg已托管，但直接source.stat/is_file/resolve仍可在NAS系统调用中阻塞HTTP线程。只把stat移到线程或只返回超时不保证旧读者退场，也不能安全继续读源。

新增inspect_source_identity_owned复用视觉模块的_SourceJob、管道读者、内核PID/创建身份与退出核查，不增加媒体解码或新进程协议。预览前/后身份查询使用本地cache/source_identity状态目录：内核目录认领先解析前次读者；身份记录原子落盘后才发送源路径；同一3秒等待预算覆盖认领、disk准入和身份响应，清理另有既有预算并可明确失败。内核退出未确认时拒绝后续查询，当前进程保留I/O许可；实际退场后在下一次认领中核查并恰好一次归还。跨进程通过持久身份拒绝旧活读者，不能将旧RUNNING记录或退出0当活跃/终止证明。所有旧尝试记录保留，既有缓存键及源前后核对语义不变。

元数据等待/清理不等于整个HTTP请求硬截止；FFmpeg执行及SMB驱动硬阻塞仍有独立边界。候选受控复现曾提前释放未退场读者的许可，修正后仅在内核确认后恢复；正常本地查询约0.125秒，三并发请求约0.406秒，增加进程/落盘成本，不声称提速。两个既有NAS源仅元数据核对通过，未读视频或认证真实掉线停止。生产接口、回归与实际预览验收见PROGRESS，推广范围仅工作台预览身份，扫描/分析和其他NAS访问不自动获得保证。
## ADR0059：已注册FFmpeg复用共享Windows进程Job（2026-10-14）

实际合成FFmpeg在父进程强制结束后仍存活，说明内存Registry与kill_all不能覆盖父死亡。已有元数据_SourceJob已解决相同的Windows拥有者生命周期，因此将该实现移到process_lock.OwnedProcessJob共享，视觉模块保留_SourceJob兼容别名；不复制第二个Job实现、不增加调度器或并发许可。

注册真实Windows Popen时把尚未结束的进程分配给kill-on-close Job；父进程死亡关闭句柄后终止其成员。短命令可能在分配前结束，只有内核确已终止才允许该竞态；创建/分配失败且子仍活时停止新进程并报错。同键同实例幂等，异实例拒绝且停止新实例，避免覆盖旧拥有者。kill_all关闭Job；失败保留诊断引用并记录错误。显式deregister沿旧合同解除拥有关系，不杀仍活成员：先清除kill-on-close策略再关闭句柄，失败不遗忘其Job/进程。

保证范围是成功注册到解除追踪之间。Popen到register的窗口、真实SMB驱动不能完成终止、物理断电及非Windows父死亡仍不由此认证；常规调用继续在完成后deregister。合成实际父死亡/独立进程保留、解除追踪后存活、注册失败与重复键、现有元数据回收及NV/QSV正常媒体通过，结果见PROGRESS。此为可靠性改动，未修改编码命令/时间轴/模型/参数，不声称提速或新整日质量验收。

## ADR0060：媒体启动时建立Windows Job归属（2026-10-14）

补充元数据探测范围：run_ffprobe原subprocess.run绕过Registry，受控检查证实运行期间无登记。现复用同一Registry.start，ffprobe与四个FFmpeg入口获得相同的Windows创建时归属，不新增Job实现或调度器。成功/None及探测超时/启动异常合同保持；退出清理另有5秒等待，未确认终止不解除追踪或归还I/O许可，而是隔离并传播带上下文的清理失败，防止新读者叠加。该分支是安全降级而非自动恢复完成；恢复必须核对内核终态，不能直接reset许可。真实本地探测、受控HTTP不响应的超时/全局取消及注册前后父死亡检查通过；不外推NAS驱动硬阻塞或物理断电。详见PROGRESS及metadata_probe专项。

未注册父死亡的实际合成FFmpeg仍存活，ADR0059的注册后分配不能补齐窗口。四个媒体入口统一使用FFmpegProcessRegistry.start：先创建OwnedProcessJob，通过CreateProcessW的PROC_THREAD_ATTRIBUTE_JOB_LIST在内核创建时归入Job，再登记同一个拥有对象。创建或登记失败停止新子并关闭Job；不以CREATE_SUSPENDED后分配、全局替换_winapi或将整个主进程加入Job来伪称消除窗口。

保留Popen的I/O、超时/退出码和调用方清理，通过媒体专用_OwnedFFmpegPopen替换其私有Windows创建钩子；只接受既有调用的shell=False/close_fds=True合同，显式标准句柄白名单去重（含stderr=STDOUT），Job句柄不继承。参数、工作目录、Unicode环境及sys.audit保留；钩子签名是Python升级时必须验证的兼容边界。非Windows和注入Popen测试替身沿原路径，手动register外部子仍只有注册后保护，不对其扩大声明。deregister解除追踪而不杀活进程，解除后重新register须新建Job；中断或重复键按原拒绝/停止语义处理。四入口编码命令、硬件/I/O准入、时间轴及媒体缓存不变。

原型和兼容验收只使用本地合成输入；并发标准流、不可继承无关句柄、注册前/后父死亡及独立进程、登记失败、取消和正常NV/QSV媒体的当前证据见PROGRESS。不据退出0、短媒体或进程kill认证默认整日提速、真实SMB驱动硬阻塞或物理断电。后两项保持独立验证边界。

## ADR0061：按任务停止拥有树与有界管道回收（2026-10-14）

受控真实Python进程父/后代继承输出管道，经生产run_ffmpeg的启动/I/O/超时路径验证：0.8s超时仅kill直接父，普通捕获4.218s、流式4.203s才在后代自然退出后返回；直接父已退出但后代持管道的run_ffprobe分支同样4.235s。这是本地生命周期缺口，不是已观察到FFmpeg编码器生成后代或NAS驱动故障，不将这些功能等待时间当性能基准。

复用既有Registry/OwnedProcessJob增加terminate(key)：仅关闭该Job并停止直接子，保留登记引用直到调用方确认、解除追踪，不设置全局interrupted或影响其他登记任务。run_ffmpeg超时、流式定时取消和异常退出，以及run_ffprobe通信异常采用该入口。直接父已退出也不能据poll推断管道无人持有。普通捕获和异常探测的communicate回收各有5秒预算；退出wait和流式stderr线程join各另5秒，不声称整个调用硬截止。未确认退出/管道/线程回收时不关闭仍被线程使用的管道，不deregister或归还I/O许可；保留上下文并报错，须核查旧读者终态后恢复，不能reset调度器伪称恢复。

四类实际受控进程场景（捕获、流式、探测、探测直接父先退）均确认后代退场，另有实际两登记任务只停止指定键及失败保留引用/许可检查；常规返回/探测异常及全局kill_all合同保持，检查见PROGRESS/TESTING。非Windows未认证后代树停止，直接detector/prescreen/renderer局部kill不自动继承新合同。无NAS断网/物理断电注入，不扩大Windows Job关闭请求为驱动硬终止证明；不更改时间轴/编码命令/模型/配置或推广性能方案。

## ADR0062：未确认清理的原I/O许可转交与终态恢复（2026-10-14）

ADR0061能保留未确认读者与I/O占用，但原许可对象只存在调用局部变量；函数异常返回后Registry没有对应许可，读者后来终止也无法精确归还。不能通过reset_semaphores绕过旧读者，亦不能仅poll直接父就关闭正在被线程读取的管道。

在现有Registry内记录_deferred，持有同一proc、原io_sem与明确的管道读线程；重复同身份幂等，冲突拒绝，未完成deregister拒绝。正常完成仍由调用方释放；只有异常未确认分支转交。kill_all保留延迟引用，取消标记重置不释放。非阻塞reap_cleanups在run_ffmpeg/run_ffprobe准入前核对进程终态、Job已关闭与全部读线程结束，关闭管道、归还原许可一次后才移除；不停止重试、不创建线程/调度器、不重置取消。Job仍开放须显式停止；失败记录保持pending，不为继续执行丢弃引用。保持同一Registry锁串行核对/释放，沿现有FileBudget释放验证前不修改预算的合同，避免多调用方重复归还。

合成状态检查覆盖进程/线程/Job未结束、关闭与释放失败、身份冲突、并发回收和全局取消/重置。实际Windows拥有Python读者注入第一次停止失败，真实两个communicate线程继续读取，原许可保持；撤除受控失败、显式停止并核对终态后，下一实际ffprobe调用自动核对回收，预算恢复且重复核对不重复释放。初次验收发现假设的线程字段名错误（0个读者），按当前CPython stdout_thread/stderr_thread修正后通过，旧失败XML保留；Python升级必须重验实际线程合同。此为进程内I/O许可恢复，不认证真实NAS驱动硬阻塞、非Windows后代树或跨进程持久化许可恢复；参数/模型/数据库/时间轴不改。

## ADR0063：识别质量门禁只接受留出集及独立事件组（2026-10-14）

受控复现将同来源重复为四场景各10条，并声明split=tuning，旧assess_replay_quality仍PASS/verified40。它仅检查review_status/reference_active/evidence，不检查留出或相关事件，文档宣称“独立验收”但实现不能阻止调优泄漏和重复量膨胀。TuningSimulator及正式replay共同调用该入口；它们的全参考诊断指标与历史收据不得被删除或改写。

当前门禁要求明确holdout、未声明used_for_tuning、可追溯filepath和有效区间（完整源参考可以没有区间）。相同规范化来源/区间去重，同区间标签/场景/预测/组冲突全部排除。以声明的independence_group聚合，否则保守一源一组；每场景至少10个组。只去重组数仍可由大量容易的窗口掩盖其他事件，故按组评分：阳性任一窗口漏检记该组FN，阴性任一窗口误报记FP；成人/婴幼儿组召回90%、空镜/光影组特异度95%保持。窗口metrics继续诊断，另给independent_metrics/counts及排除理由；无明确划分/来源不默认为可验收。声明分组不证明真实事件独立或盲性，仍须来源冻结和复核证据，不以该准入代替输入质量。

本单元不写参数、模型或人工标签，不训练。冻结100条参考与基线/0.9/1.1历史预测重新检查：27条holdout准入、72条非holdout仅留诊断、1条未确认排除；全部scenario UNKNOWN，四层组数均0，三份仍UNAVAILABLE。这是历史预测的当前准入审计，不是当前代码新识别结果；旧收据SHA保持。合成检查保护调优泄漏、无split/已调优、同源/同事件/重复、冲突、独立40源通过、窗口膨胀掩盖困难事件、非法区间及原四阈值边界；工作台/调优路径与全量回归见PROGRESS/TESTING。

## ADR0064：逐条声明事件复核依据，冻结时保守准入（2026-10-14）

人工事件标签优先于算法重渲，但只看代表图不能核验整个区间。原prepare_review_benchmark会把所有裁决直接标为independent_human，notes未说明观看范围。现复用现有配置归因面板和review_notes JSON增加review_basis，不增表或新审核流程；保留scenario/notes旧合同，旧文本未知，快捷/批量裁决保持各条原依据。单条提交非法依据或缺少明确归因字段先拒绝，批量不传播观看声明。

新冻结活动参考仅接受完整视频声明的阳性、完整视频及可用声音声明的阴性；代表图、未知、不能确定或未听审的阴性保留原裁决与gold_label，reference_active=null/pending_confirmation。完整观看声明与参考资格分开，不能把未听审阴性写成未观看。声明是复核者提供的范围记录，不证明场景、独立事件、盲留出或真实人工目标框；ADR0063的分组门禁继续执行。旧冻结和生产历史不自动修改，无新模型、参数或训练。

隔离真实SQLite、生产单条API/服务和实际浏览器保存→重新读取完成受控验收，模型和源捕获被夹具禁用；七类观看/标签组合验证冻结资格和数据库不变。完整回归及后续字段专项身份、原失败辅助记录和验证边界见PROGRESS/TESTING。

## ADR0065：批次停止未确认保留租约并阻止重试（2026-10-14）

真实Python进程进入生产_run_batch_render，受控kill无效/wait失败：返回None后同句柄内核仍活，Registry/Job引用丢失，原硬件/I/O许可已可再取得。None及普通异常还能触发EDL备用/NV重试，不能仅视作NAS认证边界。

复用Registry.terminate/defer_cleanup/reap_cleanups，未知终态保留同一VideoLease和临时输入/媒体，抛非重试RenderTerminationUnconfirmed。同输出仍pending先于文件修改拒绝；主流程停止新工作并归入已有render batch失败门禁，不拼接/COMPLETED；审核失败保留旧包。虚拟输入不在未确认finally删除。显式记录wait自然完成，Job停止返回0不等于批次正常完成。

Windows终态等待/延迟回收核对真实句柄，缓存returncode=0但内核仍活不归还许可；无句柄替身及非Windows沿Popen合同。真实CPU控制故障、租约恢复/调用方失败门禁和顺序NV/QSV正常2秒HEVC/AAC通过，见PROGRESS。初验尚未覆盖的detector/prescreen已由后续ADR0066补充；仍不声明真实SMB硬停止、物理断电或新整日性能/质量验收。

## ADR0066：直接源读者未确认停止的资源保留与失败传播（2026-10-15）

ADR0065未覆盖prescreen/detector局部清理：实际受控Windows子进程证明停止失败或缓存退出码0时，旧路径可能丢失原租约、允许备用读者或不发看门狗停止。单靠共享ffmpeg保留disk也不能保护_extract_frame外层先归还的硬件许可；普通Exception还会被音频/解码容错吞掉。

采用公共MediaTerminationUnconfirmed，批次专用异常继承它；共享FFmpeg/ffprobe及直接源读者在核对内核终态、拥有Job及读线程后才解除/释放，未知转交现有Registry的同一租约和线程。run_ffmpeg接收可选未取得io_lease，_extract_frame提交现有VideoLease，不另建许可或改变默认调用语义。streaming最终清理补齐内层异常跳过的读线程核对；detector看门狗不信缓存0。普通确认失败的备用与重试保持。

显式类型穿过PyAV备用、keyframes备用、音频静默降级及prescreen音频门禁。流水线先建立致命停止状态，再尽力写FAILED，写库失败仍不准继续、拼接或完成；记录第二错误便于修复持久化。保留旧交付、已成批次和必要诊断，不改分析命令/采样、阈值、模型或时间轴。实际CPU故障、数据库失败、当前回归及本地正常媒体证据以TESTING/PROGRESS最终收据为准；不认证真实SMB硬阻塞、物理断电或吞吐/识别收益。

## ADR0067：启动登记失败的实际进程交接与同key认领（2026-10-15）

ADR0066的调用方只有在Registry.start返回后才拿到proc。真实Windows子进程在登记拒绝且kill/Job关闭失败、或缓存0但内核仍活时，start抛普通异常，调用方局部proc仍None，原许可被归还且读者失去追踪；只保护运行后的finally不足。

复用Registry锁为创建→登记→失败收尾保留同key启动认领，既有owner/启动者先于创建新子拒绝；register只允许认领的实际proc，解除也可校验proc身份。清理失败保留同一proc/Job，抛MediaStartTerminationUnconfirmed并携带原key/proc。五个生产入口显式接回实际进程，再沿ADR0066转交原租约；renderer将该异常转为非重试渲染失败并保留临时资产，同输出已占用或pending时在修改资产前拒绝。普通创建失败没有proc时不得按同名key解除其他owner；确认停止的普通启动失败仍保持原错误合同。

启动认领不取代上游日任务/工作区/输出的持久认领，也不新增硬件许可或覆盖渲染准备前的独立调用竞争。范围为已创建且Popen已交回实际对象的登记失败；不扩大到未复现的Popen构造中途未知状态、真实SMB硬停止或物理断电。初验不能继承此前850项回归；后续866及当前892检查点各绑定自己的代码身份，当前验证状态以PROGRESS/TESTING对应收据为准。

## ADR0068：跨日期准入与致命媒体失败保留（2026-10-15）

真实本地Windows旧deferred读者仍持许可时，新StreamingOrchestrator清除中断，下一run_ffmpeg创建第二个实际子进程；两原句柄曾同时WAIT_TIMEOUT。另在实际Registry/FileBudget锁与确定性交错下复现两线程循环：reaper持Registry等budget.condition，等待者持condition查询Registry中断。证据分别在phase5_parallel既有根的cross_day_deferred_admission与registry_budget_lock_cycle，不冒充真实SMB或完整CLI故障。

复用原Registry锁与reaper，公开写入入口先require_cleanups，构造器最先reset_interrupted；重置须在同一锁内先确认无deferred/启动认领，不能凭剩余预算继续。历史COMPLETED且无待恢复包的API跳过保持只读，force或恢复包仍受门禁。取消查询移到FileBudget.condition之外，票据、FIFO加权准入和释放仍由原condition维护，不新增调度器或持久状态。

致命类型从render worker经orchestrator、concat及companion到公开API/跨组循环保持，写FAILED/诊断失败不掩盖原错误；不能因reap后来成功就把本轮致命错误转为普通False继续。当前组失败、后续remaining明确，普通确认失败可继续；只有核对原读者并显式新执行才恢复。未知停止时保留可能仍占用的candidate及原租约，旧正式包保持。该决策补充ADR0062的reset合同，不改变采样/阈值/模型/时间轴；功能与本地恢复收据以PROGRESS/TESTING为准，不认证NAS硬截止、物理断电或默认全流程提速。

concat内部也保留原位置的临时MP4及ffconcat清单，不经失败归档移动/删除；受控复现说明移动会丢失仍使用的路径、unlink的PermissionError还会覆盖致命类型。timings记录保留路径，下一concat在改写前通过同一准入门禁。普通已确认失败/取消的既有诊断预算不变。

## ADR0069：人工排除、交付准备与配置写者边界（2026-10-16）

状态：Accepted。双SQLite连接复现排除被迟到结果覆盖；交付准备hash异常留下旧格式阻塞包；双配置写者在相同预览版本下互相覆盖。第三项扩展ADR0040原先明确限定的进程内合同，前两项补充ADR0052/0053。

源排除在数据库入口取得已有日期/机位内核认领，拒绝活动流水线或审核重建期间修改；工作台显示409。预筛和分析结果UPDATE同时检查两个状态均非SKIPPED_EXCLUDED，零更新不写分段，保留人工排除及原因。防止迟到结果复活不意味着允许活动任务期间修改已捕获时间轴。

交付先建立独立preparing目录并写齐完整事务，再原子重命名为backup目录，之后才替换任何旧资产。普通准备失败只清理本次preparing目录；进程骤停的preparing不作为恢复包，保留待定向检查。完整PREPARED发布后的中断沿用原恢复协议，旧无身份包继续拒绝猜测；不宣称物理断电原子性。

配置应用/回滚在现有进程内锁下，按resolve后的目标文件取得同目录内核写者锁，覆盖读、SHA校验、提交、加载与回滚。冲突快速失败，不自动以旧预览重试；后续显式重试必须刷新版本。使用既有exclusive_file_lock，不新增锁库；不覆盖编辑器、绕过API的写者或跨主机文件系统语义。残留锁文件不删除，不代表活跃写者。

验证包括实际双连接、真实HTTP、真实子进程竞争及准备骤停，既有资产替换阶段骤停与回滚回归仍须通过。完整验收身份见PROGRESS/TESTING。

## ADR0070：整数音频时钟与有界分组（2026-10-17）

状态：整数音频时钟 Accepted；当前有界分组接入 Rejected（2026-10-18）。

批次末端统一使用`asettb=1/<sample_rate>,asetpts=N`，消除长音频浮点时间戳的一采样点舍入，不改变段内DSP、展示计划或分析结果。三个实际AAC长音频回归覆盖稠密、传统与简单图及不同帧边界。

397段本地批次分组有局部收益，但完整NAS日对照中，新增声道ffprobe在30秒超时，分组尚未开始便回退原图。流水线调用墙钟3298.937s → 3431.843s，慢4.03%；不能把差值全部归因于探测，也不能据此否定分组算法本身。另有3批编码器分配变化，整片严格包等价未通过；源/分析/展示映射及目标批次包一致。当前接入未通过，已撤回分组计划、PCM准备、配套输出锁与候选专用测试，仅保留整数时钟修复；候选代码、测试、异常和媒体证据冻结在scratch。

重新评估前必须先证明声道元数据可在已有探测/缓存或受控I/O准入中可靠取得，且实际NAS路径能执行分组。不能仅延长超时、跳过声道校验，或重复无变化的整日试验。单批通过后仍需同输入整日墙钟和内容验收。历史CUDA/QSV及滤镜线程否定结论不受本次改变；不涉及检测、生产配置或质量门槛。

## ADR0071：非侵入式分析耗时微观分解与交付 I/O 审查（2026-10-18）

状态：Accepted。

根据工程不变量与下一阶段待办，对 Pass 1.5 Analysis 解码与分析各微观子阶段及交付阶段 I/O 进行了非侵入式度量与审查。

### 背景与决策

1. **非侵入式微观耗时分解**：
   - 避免在生产核心代码中插入调试打点；开发了独立的诊断工具 `scripts/profile_decode_cost.py`，全链路采用 `time.perf_counter_ns()` 高精度度量；
   - 在 `tests/test_performance_regression.py` 中增加契约测试 `test_profile_decode_cost_receipt_structure`，确保诊断收据结构受自动化回归保护；
   - 确立了 6 大微观子阶段（容器解复用与视频解码、灰度提取与双线性缩放、EMA 运动分析、候选帧 JPEG 编码、音频解复用与 VAD、YOLO 批推理）的标准衡量契约。

2. **交付与收尾 I/O 审查**：
   - 实测否定了“大文件 SHA256 顺序读引发交付瓶颈”的假设（6.68GB 成片仅需 5.8s）；
   - 确认交付资产采用原子重命名，常规交付路径无物理大文件拷贝；
   - 将 `final_concat` 历史瓶颈定位至并发写盘与检查点 seek-decode 产生的 NVMe 写入缓冲区饱和。

### 后续约束

- 优化 Pass 1.5 解码应优先考虑 GPU 显存内直接下采样（如管道路径 `scale_cuda`），避免在 CPU 端进行 4K NV12 格式转换；
- 交付与收尾阶段无需更改既有的 SHA256 校验和原子事务协议。
