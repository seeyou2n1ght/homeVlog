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

ADR记录决策历史，不应被当作当前实现说明书。当前产品边界以ADR0076为准：训练/目标框、基准管理、交互调参、旧原型及历史实验观察适配已退役；涉及这些功能的旧Accepted段落仅保留历史，相关恢复/人工审核/身份校验等共享合同继续有效。

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
| ADR 0001 | 单次灰度解码与内存候选帧流转 | [ACTIVE] Accepted | 2026-03 | — |
| ADR 0002 | 单一 NVENC 同构渲染策略 | Superseded | 2026-03 | Superseded by ADR 0008 |
| ADR 0003 | 人机协同主动学习闭环 | Superseded | 2026-03 | Superseded by ADR 0082 |
| ADR 0004 | 原子渲染批次与确定性时间轴 | [ACTIVE] Accepted | 2026-03 | — |
| ADR 0005 | 硬件并发对齐、NVDEC 借调与强类型 PipelineTask | Partially Superseded | 2026-03 | Scheduling 部分演进至 ADR 0007；PipelineTask 保持有效 |
| ADR 0006 | 命名模板、低照度抗噪与 Filter Graph 健壮性 | [ACTIVE] Accepted | 2026-04 | — |
| ADR 0007 | NVENC/NVDEC 资源解耦与流水线错峰 | [ACTIVE] Accepted | 2026-04 | Evolves ADR 0005 |
| ADR 0008 | NVENC + QSV 异构流式渲染 | [ACTIVE] Accepted | 2026-04 | Supersedes ADR 0002 |
| ADR 0009 | 双端渲染工作窃取与基于真实成本的调度 | [ACTIVE] Accepted | 2026-04 | Evolves ADR 0008 |
| ADR 0010 | Analysis SJF 与 QSV 弹性窃取策略 | [ACTIVE] Accepted | 2026-04 | Evolves ADR 0009 |
| ADR 0011 | 进程生命周期下沉、配置解耦与架构瘦身治理 | [ACTIVE] Accepted | 2026-04 | 解耦 FFmpegProcessRegistry 与配置加载器 |
| ADR 0012 | 四级自适应阶梯速率模型与防漏检保全 | [ACTIVE] Accepted | 2026-09 | — |
| ADR 0013 | 五级自适应睡眠浓缩与滤镜图硬截断不变量 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0004, ADR 0012 |
| ADR 0014 | 异构分析与渲染 QSV 硬件信号量池隔离 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0008, ADR 0010 |
| ADR 0015 | 白天漫射光影偏转软抑制与吸收短动态审计保全 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0006, ADR 0012 |
| ADR 0016 | 流式批次时间轴稳定性与源任务完成门禁 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0004, ADR 0008 |
| ADR 0017 | 离线高质量编码重构与多特征迟滞动作判定 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0008, ADR 0012, ADR 0013 |
| ADR 0018 | 异构 HEVC 参数与渲染时间轴身份 | [ACTIVE] Accepted | 2026-09 | — |
| ADR 0019 | 人体空间门控、昼夜门限与云台巡航抑制 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0012, ADR 0013, ADR 0017 |
| ADR 0020 | 虚拟拼接暂存绑定、QSV 重试与 SMB 探测边界 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0004, ADR 0015, ADR 0016 |
| ADR 0021 | 分层依赖、兼容入口与伴随资产职责 | Superseded | 2026-09 | Superseded by ADR 0082 |
| ADR 0022 | QSV 尾部负载、并行接缝校验与算子优化 | Partially Superseded | 2026-09 | Evolves ADR 0015, ADR 0019, ADR 0020; 部分被 ADR 0023 替代 |
| ADR 0023 | 交付产物原子提交与共享媒体契约 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0004, ADR 0013, ADR 0020, ADR 0022 |
| ADR 0024 | 整数帧展示计划与取消结果 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0004 |
| ADR 0025 | 预筛末关键帧覆盖 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0006, ADR 0015 |
| ADR 0026 | 交付元数据保留分析证据 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0023 |
| ADR 0027 | 看门狗配置语义闭环与 NVENC 瞬态重试兜底 | [ACTIVE] Accepted | 2026-09 | Evolves ADR 0004, ADR 0008, ADR 0011, ADR 0023 |
| ADR 0028 | 已交付日期保留离线备份完成记录 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0029 | 显式零字节源文件排除 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0030 | 局部识别证据与显式批处理停止 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0031 | 亚帧源区间必须输出计划帧 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0032 | 可见启动与完整扫描门禁 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0033 | 配置状态单一所有权与旧入口兼容 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0034 | 最终拼接故障证据与完整接缝回退 | [ACTIVE] Accepted | 2026-10 | Evolves ADR 0027 |
| ADR 0035 | 持久审核撤销与精确源区间 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0036 | 可恢复资产提交与真实运行结果 | [ACTIVE] Accepted | 2026-10 | Evolves ADR 0023, ADR 0028 |
| ADR 0037 | 明确的渲染中断传播 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0038 | 人在回路三支柱架构（工作台/重渲/调参模拟器） | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0039 | 分层配置治理与实证场景 Profile | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0040 | 审计修复：显式配置来源与审核资产提交边界 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0041 | 解码完成证据与不可用源范围 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0042 | 单文件稠密加速时间轴的视频单次变换 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0043 | 审核库存、事件保留与播放清晰度分开验收 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0044 | 审核重渲使用共享缓存决策与启动快照 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0045 | 复核归因与实测对照保持来源和确认边界 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0046 | 审核队列区分库存、代表分组和随机抽查 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0047 | 成片对比使用交付绑定映射及成功审核历史 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0048 | 视觉近似索引复用解码数据并与人工真值隔离 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0049 | 隔离分析重放的正式检查点与续跑 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0050 | 事件审核不作为训练目标真值 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0051 | 历史视觉补建使用受父进程管理的独立源读者 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0052 | 主流程、工作台与维护入口共用内核认领 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0053 | 交付包骤停恢复与完成标记一致性 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0054 | 成片映射绑定验证后的计划帧时钟 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0055 | 智能体视觉建议与人工目标真值分别持久化 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0056 | 粗筛邻域状态按源顺序确定 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0057 | 工作台预览复用托管 FFmpeg 与有界 I/O 准入 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0058 | 工作台预览身份读取复用拥有的元数据读者 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0059 | 已注册 FFmpeg 复用共享 Windows 进程 Job | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0060 | 媒体启动时建立 Windows Job 归属 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0061 | 按任务停止拥有树与有界管道回收 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0062 | 未确认清理的原 I/O 许可转交与终态恢复 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0063 | 识别质量门禁只接受留出集及独立事件组 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0064 | 逐条声明事件复核依据，冻结时保守准入 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0065 | 批次停止未确认保留租约并阻止重试 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0066 | 直接源读者未确认停止的资源保留与失败传播 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0067 | 启动登记失败的实际进程交接与同 key 认领 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0068 | 跨日期准入与致命媒体失败保留 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0069 | 人工排除、交付准备与配置写者边界 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0070 | 整数音频时钟与有界分组 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0071 | 非侵入式分析耗时微观分解与交付 I/O 审查 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0072 | 可选 NV 尾段接管调度 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0073 | 调用线程拥有接缝校验容器 | [ACTIVE] Accepted | 2026-10 | — |
| ADR 0074 | 驻留必须区分人物观测与推导状态 | Rejected | 2026-10 | — |
| ADR 0075 | 在交付帧序列上按需生成日常观看版 | Superseded | 2026-10 | Superseded by ADR 0076 / ADR 0082 |
| ADR 0076 | 聚焦纯视频处理与流水线精炼 | Superseded | 2026-10 | Superseded by ADR 0082 |
| ADR 0077 | 仅在上游完成后借用第三 NV 尾段许可 | [ACTIVE] Accepted | 2026-10 | Evolves into ADR 0087 |
| ADR 0078 | 硬件自适应画像探测与并发参数推荐 | [ACTIVE] Accepted | 2026-11 | — |
| ADR 0079 | 异构协同滞回调度控制 | [ACTIVE] Accepted | 2026-11 | — |
| ADR 0080 | 成本感知设备路由与重任务 NVDEC 优先 | [ACTIVE] Accepted | 2026-11 | — |
| ADR 0081 | 双端非对称优先调度与 NVMe 本地预暂存直通 | [ACTIVE] Accepted | 2026-11 | — |
| ADR 0082 | 纯命令行流水线收敛、整洁架构与规范对齐 | [ACTIVE] Accepted | 2026-11 | Supersedes ADR 0003, 0021, 0033, 0035, 0038, 0040, 0043-0048, 0050, 0051, 0055, 0057, 0058, 0064, 0075, 0076 |
| ADR 0083 | I-Frame 关键帧跳帧预筛选与时域窗口精析 | [ACTIVE] Accepted | 2026-11 | — |
| ADR 0084 | 尾段自适应工作窃取与次长切片双 GPU 负载均衡 | [ACTIVE] Accepted | 2026-11 | — |
| ADR 0085 | 关键帧粗扫根因校准、异步 YOLO 流水线重叠与双阶段均衡收敛 | [ACTIVE] Accepted | 2026-11 | — |
| ADR 0086 | 关键帧视觉观测暂存与前源先验结算 | Rejected | 2026-11 | — |
| ADR 0087 | 两阶段流水线渲染减负与异构去争用 | [ACTIVE] Accepted | 2026-11 | — |
| ADR 0088 | 分析并发度收敛、全解码渲染权重校准与中间批产物精简 | [ACTIVE] Accepted | 2026-11 | — |
| ADR 0089 | 全彩 640×360 YOLO 管道直通、高精确率分段门控与长时静止对数封顶浓缩 | [ACTIVE] Accepted | 2026-11 | Evolves ADR 0001, ADR 0013, ADR 0019 |
| ADR 0090 | CUDA 分析解码抗锯齿下采样统一与无视觉纯音频切片边界收紧 | [ACTIVE] Accepted | 2026-11 | Evolves ADR 0089 |

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

**Status:** Superseded by ADR 0082 — 人机协同主动学习与反馈回流已于 pure-CLI 收敛时退役，生产时间轴 100% 由确定性算法生成。

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

**Status:** Superseded by ADR 0082 — 历史 src/*.py 兼容入口及审核相关分层过渡约束已清理，现统一使用明确包路径与 pure-CLI 分层。

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

**Status:** Superseded by ADR 0082 — 旧 Web 审核入口的配置覆盖与单进程缓存兼容路径已随 Web 审核工具一并退役。

# ADR 0034 — Final Concat Failure Evidence and Complete Checkpoint Fallback

**Status:** Accepted

**Date:** 2026-10-08

**Evolves:** ADR 0027

生产日志中 `20260715 cam1`、`20260718 cam1` 初始媒体检查通过，但八个并行容器均在打开 MP4 头部时 EOF，最终失败。失败临时成片被删除且 perf 缺少失败的 `final_concat`，无法区分拼接、打开与接缝故障。隔离重建两日均成功，尚未重现原始 EOF，不将其认定为磁盘超时或特定运行库缺陷。

保留正常八路接缝校验。仅在所有失败块均为独立容器打开异常时，且文件身份保持不变，复用初始容器串行检查全部接缝与尾窗；解码/PTS/连续性失败仍终止交付。该回退修复已观测故障边界的恢复能力，不声称已确认底层根因。

失败诊断采用按成片保留首次证据的有界目录，媒体上限 16 GiB、目录上限 32 个、stderr 上限 64 KiB；不覆盖或自动删除已有证据。达到预算时记录日志，仍保留有效输入批次。最终拼接在成功、失败、异常和取消时都记录状态、失败阶段及分项耗时，取消继续保持未完成状态供恢复。生产配置、时间轴与识别规则不变。



# ADR 0035 — Persistent Review Revocation and Exact Source Intervals

**Status:** Superseded by ADR 0082 — 审核撤销与人工精确源区间覆盖流程已随 Web 审核工具退出生产系统。

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

**Status:** Superseded by ADR 0082 — 人在回路工作台、优先级重渲与调参模拟器三支柱架构已退役，详见 ADR 0082。

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

**Status:** Superseded by ADR 0082 — 审核工作台配置来源与审核资产提交边界已随 Web 工作台整体清理。

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

**Status:** Superseded by ADR 0082 — 审核库存与播放清晰度双轨验收流程已随审核体系一并退役。

# ADR 0044 — 审核重渲使用共享缓存决策与启动快照

**Status:** Superseded by ADR 0082 — 审核重渲路径及启动快照契约已随审核重渲能力退役。

# ADR 0045 — 复核归因与实测对照保持来源和确认边界

**Status:** Superseded by ADR 0082 — 人工复核归因边界已随人工复核流程退役。

# ADR 0046 — 审核队列区分库存、代表分组和随机抽查

**Status:** Superseded by ADR 0082 — 审核队列分层抽样策略已随工作台整体退役。

# ADR 0047 — 成片对比使用交付绑定映射及成功审核历史

**Status:** Superseded by ADR 0082 — 成片人工审核对比视图映射已随 Web 工具链删除。

# ADR 0048 — 视觉近似索引复用解码数据并与人工真值隔离

**Status:** Superseded by ADR 0082 — 视觉近似索引（visual_index）与感知哈希模块已整体清理，详见 ADR 0082。

## ADR0049：隔离分析重放的正式检查点与续跑

> 当前边界由ADR0076局部替代：训练/调参或历史实验选择功能已退役；共享恢复、人工审核与身份合同保持。

状态：Accepted（2026-10-12）。

长时间NAS重放需要恢复已完成范围，旧实验的临时恢复脚本不能作为正式运行协议。CLI新增显式`--resume RUN_DIR`；运行目录使用操作系统持有的排他文件锁，同目录第二写者立即失败，进程退出后锁释放，不把残留锁文件当活进程。检查点冻结完整参考/元数据、原始配置、代码、模型、依赖身份；首次源哈希按文件立即持久化，恢复必须重读全部源SHA，包含同大小/mtime的内容变化。旧目录没有正式检查点则拒绝续跑，不追认历史测量身份。

按日期/机位在每次调用后原子记录成功状态、连续参考覆盖、任务/分段结果SHA及范围壁钟。仅当成功检查点、当前数据库SHA和全参考覆盖均匹配才跳过；中断范围回到生产恢复路径。部分或失败运行不发布receipt；完整成功后原子发布，已发布收据恢复验证后保留原字节。`--max-ranges`允许按新处理范围限额安全停下，返回INCOMPLETE并要求续跑；每次尝试日志独立保留。

这只管理隔离重放目录，不替代生产跨进程任务认领、Windows阻塞SMB系统调用超时或多资产断电一致性。范围累计壁钟不等于包含源哈希/启动的连续命令壁钟。


## ADR0050：事件审核不作为训练目标真值

**Status:** Superseded by ADR 0082 — 人工事件审核与主动学习目标真值分离规则已随审核闭环退出生产。

## ADR0051：历史视觉补建使用受父进程管理的独立源读者

**Status:** Superseded by ADR 0082 — 历史视觉近似索引补建工具（visual_index_backfill）已退役。

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

**Status:** Superseded by ADR 0082 — 智能体视觉建议与人工真值分离持久化已随 rchiver.py 与 isual_index.py 一并移除。

## ADR0056：粗筛邻域状态按源顺序确定（2026-10-14）

84源默认full对照中，四源因前段是否已完成而在5.53/8.5阈值间切换；相同原片出现精析、音频快速路径及STATIC差异。任务启动时缺失的前段结果不能当作静态。

保留已有队列、调度器和一次解码：前段已知时沿原路径早停；未知时关键帧扫描分别保存首个基础/邻域正判定，必要时覆盖到EOF，释放I/O许可后才等待前段最终结果。结果只保留小型JSON，不缓存整段图片或新增独立解码。随后选择对应判定、执行音频门禁并通知后段。失败终态与取消负责唤醒；等待检查既有中断状态，不持有硬件或I/O租约。非关键帧与seek备用路径在读取前解析前段，可能降低并发；未知前段时双判定也可能比原敏感早停读更多关键帧，性能需另测。

processing_fingerprint改为analysis-v5-ordered-prescreen-prior，沿既有失效机制只影响实际请求的未完成或显式重建日期。已COMPLETED先跳过，旧交付/DB/审核不自动迁移或全库重算。四条既有差异原片、两种前段条件的新延迟判定与冻结旧串行结果一致，拥有读者STOPPED/Job active0；这证明该路径兼容与确定性，不是独立识别改善或默认整日提速验收。其他QSV/CUDA分段差异仍需隔离归因。
## ADR0057：工作台预览复用托管FFmpeg与有界I/O准入（2026-10-14）

**Status:** Superseded by ADR 0082 — 审核工作台 FFmpeg 预览与准入通路已随 scripts/audit_tool/ 整体删除。

## ADR0058：工作台预览身份读取复用拥有的元数据读者（2026-10-14）

**Status:** Superseded by ADR 0082 — 审核工作台预览身份读取路径已随 scripts/audit_tool/ 整体删除。

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

> 当前边界由ADR0076局部替代：训练/调参或历史实验选择功能已退役；共享恢复、人工审核与身份合同保持。

受控复现将同来源重复为四场景各10条，并声明split=tuning，旧assess_replay_quality仍PASS/verified40。它仅检查review_status/reference_active/evidence，不检查留出或相关事件，文档宣称“独立验收”但实现不能阻止调优泄漏和重复量膨胀。TuningSimulator及正式replay共同调用该入口；它们的全参考诊断指标与历史收据不得被删除或改写。

当前门禁要求明确holdout、未声明used_for_tuning、可追溯filepath和有效区间（完整源参考可以没有区间）。相同规范化来源/区间去重，同区间标签/场景/预测/组冲突全部排除。以声明的independence_group聚合，否则保守一源一组；每场景至少10个组。只去重组数仍可由大量容易的窗口掩盖其他事件，故按组评分：阳性任一窗口漏检记该组FN，阴性任一窗口误报记FP；成人/婴幼儿组召回90%、空镜/光影组特异度95%保持。窗口metrics继续诊断，另给independent_metrics/counts及排除理由；无明确划分/来源不默认为可验收。声明分组不证明真实事件独立或盲性，仍须来源冻结和复核证据，不以该准入代替输入质量。

本单元不写参数、模型或人工标签，不训练。冻结100条参考与基线/0.9/1.1历史预测重新检查：27条holdout准入、72条非holdout仅留诊断、1条未确认排除；全部scenario UNKNOWN，四层组数均0，三份仍UNAVAILABLE。这是历史预测的当前准入审计，不是当前代码新识别结果；旧收据SHA保持。合成检查保护调优泄漏、无split/已调优、同源/同事件/重复、冲突、独立40源通过、窗口膨胀掩盖困难事件、非法区间及原四阈值边界；工作台/调优路径与全量回归见PROGRESS/TESTING。

## ADR0064：逐条声明事件复核依据，冻结时保守准入（2026-10-14）

**Status:** Superseded by ADR 0082 — 人工事件复核依据声明门禁已随人工复核工作流退役。

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

状态：整数音频时钟Accepted；2026-10-18的有界分组接入Rejected。当前代码于2026-10-19重新接入并下调至180段；本轮保留已有实现，固定历史分析重建单对缩短17.8381%，145批及整片带时戳包恒等、完整相对帧钟及音频通过。原零起点门禁仍失败于两版共有偏移，尚未Accepted为默认新分析整日性能优化；详细证据和限制见BENCHMARK。

批次末端统一使用`asettb=1/<sample_rate>,asetpts=N`，消除长音频浮点时间戳的一采样点舍入，不改变段内DSP、展示计划或分析结果。三个实际AAC长音频回归覆盖稠密、传统与简单图及不同帧边界。

397段本地批次分组有局部收益，但完整NAS日对照中，新增声道ffprobe在30秒超时，分组尚未开始便回退原图。流水线调用墙钟3298.937s → 3431.843s，慢4.03%；不能把差值全部归因于探测，也不能据此否定分组算法本身。另有3批编码器分配变化，整片严格包等价未通过；源/分析/展示映射及目标批次包一致。当前接入未通过，已撤回分组计划、PCM准备、配套输出锁与候选专用测试，仅保留整数时钟修复；候选代码、测试、异常和媒体证据冻结在scratch。

当前接入移除了新增声道探测，只按已有音频标记及单声道输出准入。2026-10-19两种源声道、两种采样率的180段控制输入及真实183段NAS单批均包等价，但它们不能继承为所有声道布局、并发负载或整日认证。源声道语义与元数据前提仍须由代表性保真验收解决，不能以删除超时探测本身作为等价证明。单批通过后仍需同输入整日墙钟和内容验收。历史CUDA/QSV及滤镜线程否定结论不受本次改变；不涉及检测或质量门槛。

验收边界补充：渲染占用与默认分析设备选择通过现有WorkStealingScheduler关联。检测源码不变并不证明默认新分析的输入设备不变；固定分析重建与默认新分析的速度／识别保持须分别验证，不从render-only自动推广。

2026-10-19有限评估决策：不改默认QSV并发，固定七源5并发虽快但缺混合整日验证；不采纳只通过受控行为测试的ETA尾盘候选。原速率/声道PCM缓存虽快23.8764%但样本不等价，音频直拷贝出现接缝样本重叠，均拒绝推广；保留现有时长缓存和完整拼接校验。没有新的运行时或调度合同变更。

后续实机筛选补充（`scratch/scheduler_evaluation_20261019_v2/final_evaluation.json`）：21源新分析/渲染中，5并发候选已出现5个整源的解码设备、灰度/YOLO候选及持久状态区间变化，拒绝默认推广，不沿用固定设备七源收益。尾盘候选的NV→QSV改派在同时间线160帧无损参考下使固定事件样本SSIM0.983716→0.942148，也拒绝；其正常触发在上游分析完成之后，失败点为成片质量而非本轮识别。两项基线完整、候选有证据提前停止并清理，均未通过进入整日A/B的门禁；没有遗留待重跑任务或生产改动。

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
   - 2026-10-19复核代码与原始perf：FFmpeg拼接与检查点校验串行，旧“并发写盘导致NVMe饱和”的归因撤回；171.984s的FFmpeg阶段尚需CPU/I/O细分。

### 后续约束

- 默认FFmpeg分析已使用GPU缩放；独立PyAV工具的微观占比不能外推默认路径。后续须在实际管道、固定每源设备与采样下测吞吐及争用，保持灰度和候选合同；
- 交付与收尾阶段无需更改既有的 SHA256 校验和原子事务协议。

## ADR0072：在既有NV预算内复用闲置QSV尾段线程（2026-10-19）

状态（2026-10-20）：实现保留，默认推广暂停，配置已恢复预算2。固定12批两次配对、140源完整历史分析配对及修复后完整AV校验通过，单对pipeline少104.579秒；但随后新分析基线仅两NV的尾段显存采样峰值已达7686MiB/8192MiB，历史分析的资源边界不能覆盖新分析。原正向与失败收据均保持，详见BENCHMARK及nv_tail_promotion.json。

异构编队保留两NV和一QSV。只有所有源已派发、最轻剩余任务严格超过QSV最大收尾准入、队列尚未触发末尾硬保护、`hardware.max_nv_concurrency >= 3`且现有显存探针取得低于水位的有效非负有限值时，QSV完成原worker的计数和统计后，在同一线程进入原NV worker循环。未知水位拒绝额外准入，后续循环可随水位恢复继续判断；原两NV仍正常清空队列。总线程数仍3，使用原NVENC和加权I/O租约，不创建独立并发控制，也不把NV任务改派QSV。转换后的worker单列`nv_tail_0`；取消、未知停止、有限重试和批次对账沿用原路径。新增水位保护经过专项回归，不追认旧完整日期为新保护版本的整日验收；水位快照也不保证未来显存峰值。

正常路径的all_dispatched晚于分析队列排空。所有输入已到齐且剩余最小成本超过`min(max_qsv_dynamic_duration_s,180)`，保证基线QSV此后也不能领取这些批次；这是转换的设备保真前提，不能仅以QSV暂时空闲作为条件。当前默认预算2保持不启用；其他配置须遵守同样的资源、设备、媒体及墙钟验收边界。重新推广必须先覆盖真实暖模型显存状态，不以单次准入水位替代峰值验收。

调用方约束：NV粗筛的`_extract_frame`与`_prescreen_stream_fps`也经`get_nv_semaphore()`使用同一预算。初始渲染编队保持两NV，不证明预算3在新分析上游完全无影响；固定历史分析对照不能替代新分析的粗筛/设备/灰度/候选/分段等价验证。

当前配置的限定：`prescreen_gpu_policy=qsv_only`由`StreamingOrchestrator.run()`向全部粗筛worker传入qsv，未走上述NV粗筛分支；精析NVDEC采用独立预算。按当前调用方检查，第三路NV在all_dispatched之后才启动，不能把cuda_only/alternating粗筛的影响直接套在此配置，也不能由静态检查宣称新分析整日收益已实测。

## ADR0073：调用线程拥有接缝校验容器（2026-10-20）

状态：Accepted；验收范围为交付校验容器生命周期及完整校验组件性能。

本机torchvision导入时启用PyAV的ERROR日志回调；同片None→ERROR→None受控复现独立容器并发打开的显著等待及EOF，保持ERROR的串行打开原型两次配对更快，全部139接缝、首帧及尾窗的解码PTS/像素恒等。新生产运行8个完成日期也反复记录打开等待。证据见BENCHMARK；这是已复现的交互因素，不把原生锁机制推断写成已定位结论。

`valid_video`由调用线程通过ExitStack顺序打开最多8个独立容器，再由原线程池并行seek/解码。工作线程借用容器而不关闭；池结束后由调用线程关闭，部分打开失败、池构造/工作异常和关闭异常均不得泄漏读者或报告成功。独立打开失败仍允许使用初始容器重验全部接缝，并核对回退前后文件身份；解码/PTS/连续性失败保持拒绝。小检查点路径、交错分块、窗口、容差及默认解码线程不变。

不关闭进程级日志、不减少接缝覆盖、不新增并发预算或配置。改动仅影响只读交付校验，不触及分析输入、检测或编码内容；性能收益按完整校验组件实测，不外推默认新分析整日百分比。

## ADR0074：驻留必须区分人物观测与推导状态（2026-10-20）

**Status:** Rejected — 未进入生产实现；驻留语义仍遵循 ADR 0019 / ADR 0082 的纯算法状态转移。

## ADR0075：在交付帧序列上按需生成日常观看版（2026-10-21）

**Status:** Superseded by ADR 0076 / ADR 0082 — 按需日常观看版及衍生产物路径已退出 pure-CLI 主线。

## ADR0076：收敛为视频处理、观看与人工事件复核（2026-10-21）

**Status:** Superseded by ADR 0082 — 人工事件复核界面收敛过渡方案已由 ADR 0082 彻底替代并清除。

## ADR0077：仅在上游完成后借用第三NV尾段许可（2026-10-24）

状态：Experimental，未默认推广。用户授权单日提速并保持识别。

全程修改hardware.max_nv_concurrency会同时影响共用许可的NV预筛；先前整日对照出现分析设备和分段漂移。因此保留原共享预算，使用render.nv_tail_concurrency作为尾段目标预算。省略该键保持原行为，显式值限定2/3。上游完成必须由两队列unfinished_tasks归零且未取消证明，不能用empty或stop_event代替；再满足原all_dispatched、QSV无法接收剩余批次及显存准入，才借用一个许可。

许可属于原NV semaphore；作用域退出有空闲额外许可则撤销，否则记债，由真实媒体租约归还偿还。不等待未知进程终态、不提前释放其资源，未清债时拒绝新借用。原预算3不再扩至4。约束适用于当前按日/机位串行入口，不推广为多个并行Orchestrator的全局屏障。

12批隔离筛选已保持逐件媒体SHA相同，性能见BENCHMARK；全日新分析收益、受控比较和独立验收仍待完成。两次动态调度运行即使源码相同也可能分段不同，须明确冻结设备的受控比较与默认真实运行的不同证据范围。

## ADR0078：硬件自适应画像探测与并发参数推荐（2026-11-10）

状态：Accepted。

为解决静态硬编码并发配置在异构硬件（不同 CPU 线程、核显规格、独立显卡显存）与不同视频输入规格下的脆弱性与调优成本，系统引入 `src/hardware/autotune.py`：
1. 启动期或 `--doctor` 体检时深入探测系统底层硬件画像（CPU 逻辑/物理核心、物理内存、CUDA 设备及 VRAM 大小、NVIDIA 驱动版本、FFmpeg QSV 与 NVENC 加速能力）；
2. 依据排队论平衡（Makespan 极小化）与硬件显存安全红线（VRAM 85% 防御），动态推导并推荐最优并发向量（包含 NVENC 渲染路数、NVDEC 解码协同路数、QSV 分析与渲染配额、分析 Worker 总数及 I/O 预算）；
3. 当配置启用 `scheduler.auto_tune: true` 时，自动对缺省并发参数执行安全自适应注入，且严格遵循分层治理原则——用户显式配置具有最高优先级，决不覆盖用户专家设定。

## ADR0079：异构协同滞回调度控制（2026-11-10）

状态：Accepted。

为消除此前在主渲染阶段由于微小队列波动导致 NVDEC 频繁让步震荡（Flapping）、致使大量解码任务单侧拥堵于 Intel QSV（造成上游分析耗时卡在 33 分钟物理上限）的瓶颈：
1. 在 `WorkStealingManager.get_analysis_device` 的渲染运行分支中引入滞回比较器（Hysteresis Controller）；
2. 当待分析队列达到高水位线（watermark_high）且显存与槽位安全时，激活 `COOPERATIVE_RENDER_COEXIST` 借调 NVDEC 槽位；
3. 进入协同状态后，只要队列未完全回落至低水位线（watermark_low）以下且未超显存警戒线，持续保持 NVDEC 借调状态；
4. 该机制彻底打通了 2 路 NVDEC 与 5 路 QSV 的双轨并行解码能力，将整体解码理论吞吐提升至 48~50x 实时速度，有效规避单卡饱和排队与死锁风险。


## ADR0080：成本感知设备路由与重任务 NVDEC 优先（2026-11-10）

状态：Accepted。

针对实测遥测中揭示的“算力倒挂”瓶颈（单片耗时 505s 的 55 分钟夜间长视频被机械派发给慢速核显 QSV，而快 5.35 倍的 RTX 3060 Ti NVDEC 却因低积压水位闲置，导致 QSV 累计负载高达 10,057 秒并成为全局 Makespan 木桶短板）：
1. 在 `WorkStealingManager` 调度决策链路引入 `file_duration` 任务成本感知（Cost-Aware Scheduling）；
2. 凡录制时长超过 `heavy_file_duration_s`（默认 600s，实测夜间文件达 3300s~3500s）的重型载荷，在 NVDEC 槽位可用且显存安全前提下，**突破队列水位线限制，绝对优先调度至 CUDA (NVDEC)**；
3. `acquire_nvdec_slot` 支持带超时等待机制（重型任务等待数十秒借调 CUDA，相比盲目落入 QSV 承受 8 分钟单片阻塞，总周转时间缩减 4~5 倍）；
4. 结合 3 路 NVDEC 饱和解码支持，将单日夜间巨型文件的分析耗时从 3,500 秒压缩至不足 300 秒，使上游全流程分析耗时下探至 18~20 分钟内。


## ADR0081：双端非对称优先调度与 NVMe 本地预暂存直通（2026-11-10）

状态：Accepted。

为攻破单日 24.82 小时素材（142 切片）处理总用时 30 分钟极限门槛，基于多次全量实测遥测数据中的根因诊断，落地两项关键架构演进：

### 1. 非对称加速比与双端优先级队列（LPT / SJF）
- **根因**：55 分钟 4K 夜间长切片（3,323s）在 RTX 3060 Ti NVDEC 上解码仅需 100 秒（35× 实时），而在 Intel UHD 770 QSV 上需 850 秒（14.2 分钟），加速比高达 8.5×；而 3~5 分钟日间切片在两者间差距仅数秒。若普通 QSV Worker 误领长文件，会导致长达 14 分钟的线程挂起并瘫痪核显编队。
- **决策**：
  - 在 `src/pipeline.py` 的 `AnalysisQueue` 中实现双端出队：Worker 0 作为 CUDA NVDEC 专属通道，通过 `get_heavy()` 优先消费最长作业（LPT），在 11.6 分钟内串行出清全部 7 个 55 分钟夜间长切片；
  - 其余 6 个 QSV Worker 采用短作业优先（SJF）消费 3~5 分钟轻中度日间切片（15 秒/个），并在队列中仅剩长文件时主动让步等待（QSV Yield），严禁长文件沉沦核显；
  - 彻底消除了分析阶段的非对称瓶颈，使上游全天 142 源分析完成时间由 37 分钟压缩至 26.5 分钟。

### 2. NVMe 本地 SSD 预暂存直通
- **决策**：扩展 `src/stages/renderer.py`，打通本地 NVMe SSD 预暂存机制，使全天所有单文件批次（含夜间 55 分钟大文件）在渲染前均自动通过后台异步 Prefetch 命中本地高速缓存，以 500~680 MB/s 极速读入，彻底消除 SMB 跨网络 Seek 阻塞与 I/O 气泡（单批次渲染用时从 402.2s 骤降至 92.1s）。
- **成效**：全天处理总用时最终由 53 分 33 秒压缩至 **28 分 58 秒（1,738.9 秒）**，达成 51.4× 实时处理速度，全库 891 项测试 100% 通过。


## ADR 0082 — 纯命令行流水线与关系表收敛（2026-11-10）

状态：Accepted

人工审核工作台、主动学习归档、视觉索引和人工标签覆盖退出生产流程。算法、场景配置和传感器指标决定时间轴，继续使用 `compute_display_plans()` 统一渲染、字幕及源时间映射。

删除根目录兼容入口。文件身份和处理指纹由 `src/core/identity.py` 持有，媒体读者进程状态由 `src/hardware/ffmpeg.py` 持有。移除 `typeperf.exe` 轮询，CPU 采样改为非阻塞，遥测按设备增量聚合。

交付恢复位于阶段层，硬件资源清理由硬件层负责。运行入口将硬件调优函数交给配置加载器，配置层只处理准备、验证和缓存。

新库使用五张业务表。九列 `segments` 关系表保存算法结果，时长从起止时间计算，源记录删除时级联删除切片。已完成的空切片结果按任务状态解释。移除分析 JSON 副本、视觉索引表和审核触发器。`human_reviews` 仅保存历史事实供基准导入，生产分析和时间轴不访问它。旧库启动时拒绝套用新 schema，保留原库并重新扫描到新库。

`pipeline.render_gpu_policy: cpu_only`（别名 `software`）使用软件预筛、分析和编码，仍通过共享 I/O 调度、FFmpeg 进程注册和事务性交付。它用于无 GPU 环境和合成端到端测试。硬件策略保留既有调度路径。

替代 ADR 0003、0021、0033、0035、0038、0040、0043–0048、0050、0051、0055、0057、0058、0064、0075、0076。


---

## ADR0083：I-Frame 关键帧跳帧预筛选与时域窗口精析（2026-11-12）

状态：Accepted。

### 背景与根因
在 24.8 小时 4K 监控视频中，90% 以上的画面为纯静态无活动场景。原有 `detector.py` 采用全量逐帧解码模式，即使画面毫无变动，依然对 55 分钟文件全量解压缩 66,469 帧，不仅耗费数十 GB 的视频内存带宽，而且在单文件上消耗 100~850 秒的解码时间，使得上游分析耗时卡在 2,305 秒，占到流水线总壁钟的 91.8%。

### 核心决策
1. **Tier 1 极速扫视（Coarse I-Frame Scan）**：
   - 引入 FFmpeg `-skip_frame nokey` 参数，在预筛选阶段仅解码关键帧（I 帧）。对于 3.0s GOP 的监控流，解码负载直接削减 60 倍；
   - 在 CUDA NVDEC 与 QSV 上实现极速扫视，并计算 8×8 空间单元网格能量与 EMA 背景差分。
2. **Zero-Decode 快速直通**：
   - 对全片未检测到视觉能量跃迁、微动脉冲且无音频 VAD 触发的切片，判定为全静态，直接在内存中合成 `[0.0, duration]` 的静态区间，彻底免除二次精析解码。
3. **Tier 2 局部时域窗口精析（Awakened Window Fine Analysis）**：
   - 仅对被视觉运动或音频 VAD 唤醒的 I 帧展开时域边界扩充（前向/后向 3.0 秒），将 $\le 6.0\text{s}$ 的临近窗口自动融合；
   - 仅对唤醒区间执行精确 Seek 解码，其余区间自动闭合为静态。
4. **完备保底退化机制**：
   - 若唤醒区间累计超过全片 60% 或离散区间数超过 8 个，自动切换为单次完整文件解码，杜绝 SMB 网络 Seek 开销反噬。
5. **架构不变量守卫**：
   - 候选帧严格保持在内存压缩池，磁盘 0 临时图片落地；时间轴保证确定性严格闭合。

---

## ADR0084：尾段自适应工作窃取与次长切片双 GPU 负载均衡（2026-11-12）

状态：Accepted。

### 背景与根因
在流水线调度中，RTX 3060 Ti（Worker 0）依靠 LPT 策略在约 700 秒内极速出清 7 个 55 分钟夜间巨型长任务后，算力陷入闲置；而 10~20 分钟的中型切片（“次长切片”）由于堆顶出队顺序仍被留在 QSV 队列中由慢速核显单线程串行爬行（4.1× 实时，每个需耗时 150~300 秒）。同时，旧有 `RENDER_PREEMPTION_YIELD` 会在渲染并发启动时强制剥夺 CUDA 分析租约，导致 RTX 3060 Ti 坐拥 3.5GB 空闲显存却无法支援 QSV，造成尾段严重失衡与上游排空拖长。

### 核心决策
1. **四门禁自适应准入控制（Four-Gate Admission Policy）**：
   - 在 `WorkStealingManager` 中新增 `can_steal_analysis_task`，必须同时满足：① Worker 0 算力未满；② 夜间重任务已出清；③ NVML 物理空闲显存严格 $> 3072\text{ MB}$（3GB 裕度）；④ QSV 队列积压超过阈值。
2. **次长切片原子提取（Atomic Sub-Longest Stealing）**：
   - 在 `AnalysisQueue` 中实现 `steal_sublongest`，在互斥锁下从堆中定位并弹出时长在 `[min_duration, heavy_duration)` 区间内的最大任务，并执行 `heapq.heapify` 重建最小堆。
3. **安全 CUDA NVDEC 租赁授权**：
   - 明确工作窃取属于协同出清，授予被盗任务使用 CUDA NVDEC 的合法权限，绕过 `RENDER_PREEMPTION_YIELD` 限制。
4. **严格保全成片接缝与时钟同步**：
   - NVENC 与 QSV 渲染输出统一采用 nv12、GOP 120、`-bsf:v dump_extra` 规范，确保 141 个切片拼接接缝 100% 平滑连续，时钟漂移控制在 $\le 1$ tick 内。

---

## ADR0085：关键帧粗扫根因校准、异步 YOLO 流水线重叠与双阶段均衡收敛（2026-11-13）

状态：Accepted。

### 背景与根因
在 ADR0083/0084 初版纯冷跑（`2,768.92s`）的逐文件遥测审计中暴露出三处工程瓶颈：
1. **Tier 1 粗扫开销反噬与高回退率**：初版 `_extract_keyframe_pts` 对 1.5GB 4K 文件执行了二次 SMB 全包扫描，且 `scale_cuda` 9.23× 下采样最近邻混叠把红外静态噪点放大至 `3.21`（超过 `2.2` 阈值），叠加 108 个日间短切片（`< 1200s`）因高活动占比 100% 触发 `FALLBACK_DENSE_MOTION`，白白消耗了 `1,674.5s` 的预扫开销；
2. **解码与 YOLO 推理串行等待**：每个分析 Worker 在完成视频解码后必须同步等待 YOLO 推理与数据库持久化完成才能拉取下一文件，导致硬件解码器在文件切换间隙空转；
3. **尾段重文件边界与 QSV 渲染窃取口径失配**：`heavy_file_duration_s=600.0` 使 Worker 0 在完成 12 个超长文件后仍需独自串行消化 4 个 10~12 分钟中等文件；同时提交给渲染队列的 `dynamic_duration` 误将 `PRESENCE`/`MICRO_MOTION`/`NIGHT_STATIONARY` 计入常速动态时长（达 `510s~691s > 420s`），导致 QSV 渲染器拒收并提前 204 秒退出。

### 核心决策
1. **单遍 Demuxer 跳包粗扫与短文件直通门控 (`src/stages/detector.py`, `src/core/config.py`)**：
   - 对 `>= 180s` 文件启用 `-discard nokey -skip_frame nokey` 并在单遍 FFmpeg 管道内通过双线性 `scale=w:h,showinfo` 同时提取抗混叠灰度帧与精确 PTS；
   - 按关键帧实际步长补偿 EMA 学习率，按文件时长比例扩展 `iframe_max_isolated_windows`，并在超长文件（`> 1200s`）采用 `45.0s` 窗口合并；回退全量解码时通过 `_predecoded_audio` 复用已解码音频；
   - 引入 `detection.iframe_min_file_duration: 1200.0`（默认 `0.0`），`< 1200s` 日间短片直接走单遍硬件全解，`>= 1200s` 长切片 100% 走窗口精析或零解码直通。
2. **Per-Worker 异步 YOLO 后处理流水线重叠 (`src/pipeline.py`)**：
   - `detector.analyze_file(..., defer_post=True)` 在解码结束后返回后处理闭包，由每个分析 Worker 的专属 `ThreadPoolExecutor(max_workers=1)` 异步完成分段构建、YOLO 验证、DB 持久化与渲染入队，主线程立即启动下一文件解码。
3. **精析与渲染双阶段尾段均衡 (`config/settings.yaml`, `src/pipeline.py`)**：
   - 将 `heavy_file_duration_s` 与 `steal_sublongest_max_duration` 统一设为 `1200.0s`，使 7 路精析 Worker 在 `1,826.3s ~ 1,912.2s`（极差 `85.9s`）内齐步收尾；
   - 将渲染入队 `effective_dyn_dur` 对齐为实际 `1x` 常速动态时长（无常速段时取 `min(60.0, active_source_duration)`），使 `qsv_0`、`nv_0`、`nv_1` 三路渲染器在 `2,054.86s ~ 2,060.00s`（极差仅 `5.14s`）内同步收尾，20260320 纯冷跑总耗时降至 **2,193.52 秒（36分33秒，40.73× 实时）**。

---

## ADR0086：关键帧视觉观测暂存与前源先验结算（2026-11-14）

状态为 Rejected。撤回候选，恢复原预筛调度。

前源最终状态依赖视觉判定和适用的音频门禁。原预筛 Worker 在视觉扫描和 I/O 许可归还后仍同步等待前源，后续文件不能利用这些空出的 I/O 机会。12 个真实源的计时探针确认了视觉先验等待，样本内未触发音频门禁。

候选将前源状态尚未知的双阈值观测交给有界待结算表，允许现有 Worker 继续扫描，前源状态发布后结算。12 源局部预筛墙钟减少 23.45%，整日空应用缓存对照的流水线墙钟由 2,274.02 秒增至 2,294.86 秒（增加 20.84 秒）。142 源预筛结果一致，1 源的后续分析分段边界和类别发生变化，未通过整日性能与等价性验收。

按文件时长跳过视觉预筛会丢失前源状态对后继阈值的影响，也可能绕过音频门禁和静态快路。两组对照的条件与结果见 [BENCHMARK](BENCHMARK.md)。

---

## ADR0087：两阶段流水线渲染减负与异构去争用（2026-11-15）

状态：Accepted。

### 核心决策
1. **NightSparsePlan 与静态关键帧跳包渲染（`src/stages/renderer.py`, `src/stages/timeline.py`）**：
   - 当批次含时长 `>= render.night_sparse_min_duration_s`（默认 `16.0s`）的 `NIGHT_STATIONARY` 长静态段时，启用 `NightSparsePlan`：有活跃短窗口则以双输入（`_edl_dense` 全解活跃窗口 + 主输入 `-discard nokey` 仅解关键帧并保持独立连续音频），纯稀疏则单输入 `-discard nokey`（失败或受限时自动回退常规连续解码）；`pure_static_files` 同步附加 `-discard nokey`。
   - 保持 `build_batch_render(..., cpu_decode=False)` 外部 `"nv"` 契约，内部对 `nv_tail_0` 启用 CPU 解码 + `hevc_nvenc`，避免第 3 路渲染再争用单颗 NVDEC。
2. **点播 Seek 预筛（`src/stages/prescreen.py`）**：
   - 对 `>= 30.0s` 切片采用 `_iter_sparse_keyframes` 按目标时间戳点播 seek 抽样（尾锚 `duration - 2.0s`），不支持 seek 或不足 2 帧时回退顺序读帧；`20260326` 实测预筛累计耗时减半（`2,510.11s -> 1,246.30s`）。
3. **上游优先的异构去争用策略（`src/pipeline.py`, `config/settings.yaml`）**：
   - 上游 `analysis_queue` 或 `prescreen_queue` 尚有积压时，`qsv_0` 渲染线程仅消化 `light_queue`（纯静态轻批次），让渡核显全额算力给 `ana_1..6`；
   - 当预筛与 `< heavy_file_duration_s` 短任务全部清空且 `ana_0` NVDEC 繁忙时，放行 `[heavy_file_duration_s, 1800s)` 中重型切片由 QSV 并行分析，`>= 1800s` 巨型文件仍优先保留给 NVDEC；
   - **实验证伪保留项**：`detection.cpu_analysis_workers` 默认设为 `0`（实测 `2` 路软解与 GPU 编队并发会把 CPU 推至 `78.4%` 并拖慢 CUDA/YOLO 导致总壁钟反慢 `+51.8s`，故默认禁用并保留可配置能力）；`output.audio.concat_codec` 默认保留 `"aac"`（实测 `aac_mf` 会破坏 `<1ms` 音画闭环）。


## ADR0088：分析并发度收敛、全解码渲染权重校准与中间批产物精简（2026-11-15）

状态：Accepted。

### 背景
在 [ADR0087](DECISIONS.md#adr0087两阶段流水线渲染减负与异构去争用2026-11-15) 将 `20260326 / cam0` 冷跑降至 `1,873.53s` 后，实测遥测仍显示两处主要瓶颈：
1. 7 路分析并发（`1 CUDA + 6 QSV`）使 Intel UHD 770 单文件平均精析耗时升至 `82.3s` 并产生 `46.48s` YOLO 锁等待；
2. 队列排序及 QSV 窃取准入仅按 `1x DYNAMIC` 或 `>=1800s` 超长文件加权，导致 `qsv_0` 窃取了含 `300~597s` 日间 `PRESENCE`/`MICRO_MOTION` 全解码片段的慢批（`42~55s`），而低解码 `virtual_concat_mixed` 批次反被留给 `nv_0`/`nv_1`。

### 核心决策
1. **收敛精析 Worker 并发度至吞吐最优值（`config/settings.yaml`）**：
   - 将 `detection.analysis_max_workers` 从 `7` 调整为 `3`（`1 CUDA + 2 QSV`），保留 `hardware.max_qsv_analysis_concurrency: 7` 以维持前端 10 路 Seek 预筛的并发突击能力。
   - 经全量冷跑消融对比（`3` 路 `1,785.19s` vs `4` 路 `1,795.97s` vs `7` 路 `1,873.53s`），`1 CUDA + 2 QSV` 避开 UHD 770 多路 QSV 媒体引擎拥塞与 NVDEC/YOLO 争用放大，将精析累计算力压缩 `-50.6%`（`10,304.46s -> 5,094.05s`）并将 YOLO 锁等待减半（`46.48s -> 22.80s`）。
2. **统一全解码负载队列权重 `_compute_qsv_queue_weight`（`src/pipeline.py`）**：
   - 在 `_finalize_post` 入队估算与 `_render_worker` 的 QSV 二次准入检查中共享 `_compute_qsv_queue_weight`，当非 `1x` 全解码时段（`PRESENCE`、`MICRO_MOTION` 以及不满足 `NightSparsePlan` 的 `NIGHT_STATIONARY`）总长 `>= 120.0s` 时按 `0.35×` 计入调度权重；保持 `<120s` 文件原有权重不变，不改变 `DualEndedBatchQueue.put` 三参契约与 `180s` 窃取门限。
   - 该权重使重型日间全解码批自动保留给 NVDEC/NVENC，`qsv_0` 专注快速 `virtual_concat` 与稀疏夜间批次（单批均耗 `8.47s`），渲染尾段从 `255.16s` 降至 `38.80s`（`-84.8%`）。
3. **移除中间批产物冗余 `faststart` 并完善 `cpu_nvenc` 回退路径（`src/stages/renderer.py`）**：
   - 仅由最终拼接 `concat_output_files` 生成 `faststart` MOOV 前置，移除 `_run_batch_render_owned` 为临时 `batch_*.mp4` 执行的第二遍重封装 `-movflags +faststart`；
   - 为 `cpu_nvenc` 虚拟拼接解码分支补齐 `-threads` 参数注入，并在 `build_batch_render` 与 `_run_batch_render_owned` 中对齐 `cpu_nvenc` 分组音频资格。
4. **已验证但明确不采纳的折中方向**：
   - 降采样 `yolo.imgsz=416`（改变 YOLO 检出与抑制数）、缩窄 Tier-2 `coalesce_gap`（Seek 重启增耗）、降低 `nv` 并发或用 `cpu_nvenc` 替代主路 `nv` 渲染，以及重新启用 `cpu_analysis_workers`。

### 结果
- `20260326 / cam0` 全量零缓存冷跑降至 **`1,785.19s`（`29m45s` / `48.40×` 实时）**，`verify_cold_run_audit.py 20260326` 验证 296,919 个媒体包最大误差 `1 tick`、129/129 接缝 0 断裂。


## ADR0089：全彩 640×360 YOLO 管道直通、高精确率分段门控与长时静止对数封顶浓缩（2026-11-16）

状态：Accepted。

### 背景
全量数据库（2,885 源 / 545.65 小时）与 NAS 实片对照诊断暴露三类系统性精度与时长膨胀问题：
1. **灰度低分辨率 YOLO 失真**：`416x234` 单通道 `gray` 输入导致日间侧身/坐姿人物漏检（真值 `640_RGB` / 4K 原图均 `0.87–0.90` 稳定命中）且在纯空房间出现低置信误检；
2. **空画面误膨为 1x 常速及假驻留**：`MotionTrace` 冷却置信度泄漏进 `Segment.avg_confidence`、`YoloVerifier` 误读不存在的 `micro_motion.energy_threshold` 且向子窗继承父段最高能量、`refine_activity_segments` 将 YOLO 否决切片回升为 1x `DYNAMIC`、`resolve_presence_segments` 无证据桥接两段 `conf=0.0` `DYNAMIC` 并硬造 `0.35` 置信度、Tier-1 长片过长运动拉高 `p20` 噪声门限、`prescreen_audio_gate` 跳过长串音频验证，以及 `baby_room` 日间关闭 `ambient_drift_suppress` 与过低语音门限造成大量无人常速片段；
3. **线性长静止浓缩效率不足**：夜间 `NIGHT_STATIONARY`（16x）与日间 `PRESENCE`（4x）在长时连续静止下仍产出单日 `~64 分钟` 平铺慢放。

### 核心决策
1. **单遍解码彩色 YOLO + Y 平面运动分析双轨复用（`config/settings.yaml`, `src/stages/detector.py`）**：
   - 默认 `detection.analysis_resolution` 调至 `640x360`；启用 `yolo.enabled` 且分辨率宽高为偶数时，FFmpeg 管道与 Tier-2 活跃窗口解码输出 `yuvj420p`（`scale=out_range=full`），运动追踪零复制切片直接消费 `Y` 平面，YOLO 抽样帧经 `COLOR_YUV2BGR_I420` 编码为 JPEG（q=85）进入 `yolo_buffer`。
2. **全链路高精确率/高召回率修复（`filters.py`, `profiles.py`, `prescreen.py`, `detector.py`, `yolo_verifier.py`, `segment.py`, `pipeline.py`）**：
   - 修复 `find_connected_components` 重复追加、`AudioEnergyVAD` voiced 救济补齐 `snr_floor_db` 门限；
   - 预筛升级至 `8x8` 空间网格（小图回退 `4x4`）且 $\ge 180\text{s}$ 长片按时长扩展关键帧样本数（至多 48 帧）；
   - 限制自适应 `p20` 噪声扩展上界并用 `active_cells >= 2` 保护持续动作，修复 Tier-2 窗口 `yolo_sample_interval` 分母口径为解码窗口总帧数；
   - `YoloVerifier` 通过传入的帧级 `labels` 计算每个 8 帧子窗的真实局部能量、移除白天 `max_energy >= 3.5 -> 0.20` 放宽、约束多帧场景单帧弱峰值误触发、否定切片 `avg_confidence` 归零，并按昼夜读取 `day_energy_threshold` / `night_energy_threshold`；
   - `build_segments` 不再继承运动冷却置信度（初值恒为 `0.0`），`refine_activity_segments` 不再将 `YOLO_NEGATIVE` 的 `MICRO_MOTION` 或日间非极端视觉能量重启为 1x `DYNAMIC`，`resolve_presence_segments` 取消 `conf=0` 对的无证据桥接及伪造 `0.35`，`pipeline` 对 `>12` 段或 `>45.0s` 的声音门禁切片转入完整视觉+YOLO 校验。
3. **分段对数封顶静止浓缩策略（`src/stages/timeline.py`, `src/core/config.py`, `src/core/profiles.py`, `config/settings.yaml`）**：
   - 新增 `segment.adaptive_stationary_speed`（默认 `true`，经 `display_plan_options()`、`compute_display_plans()` 与 `build_concat_filter()` 统一消费）：短暂停留在拐点内维持原线性倍速（`PRESENCE <= 30s @ 4x`、`NIGHT_STATIONARY <= 60s @ 16x`、`MICRO_MOTION <= 40s @ 16x`），超过拐点后按指数饱和曲线收敛至对应封顶时长（分别按基准倍速等比缩放于 `15s`、`10s`、`8s`），保持单调且严格闭合音画帧数契约。


## ADR0090：CUDA 分析解码抗锯齿下采样统一与无视觉纯音频切片边界收紧（2026-11-17）

状态：Accepted。

### 背景
基于 ADR0089 的 7 日（`20260320`–`20260326`）重跑审计暴露两项底层偏差：
1. **CUDA 分析下采样莫尔条纹（`src/stages/detector.py`）**：`scale_cuda` 在 `3840×2160 -> 640×360`（6× 下采样）下缺少抗锯齿低通滤波，实测空婴儿床蚊帐切片 `20260320200620_20260320201135` 上 `CUDA` 与 `CPU`/`QSV` 产生 `15.85–16.79` 像素均差、帧间虚假能量升至 `16.52`（vs `CPU/QSV 3.60`）并在 241/315 帧误检 `cat`（最高 `0.799`），而 `CPU`/`QSV` 正确为 `0` 误检；
2. **无视觉纯音频 `DYNAMIC_AUDIO (conf=0)` 3× 时长膨胀（`src/algorithms/segment.py`, `src/pipeline.py`）**：在 `0.5 fps` 长片与低能量静音缝隙下，`refine_activity_segments` 沿两侧相邻帧额外扩展了 `±2.0s` 采样步长 + `1.0s/1.5s` 前后滚，且 `build_segments`/`_filter_short` 对 `max_energy == 0.0` 或 `< threshold` 的纯音频段吸收最长 `8.0s` 静音缝隙，使 `73s–110s` 音频脉冲膨胀至 `237s–294s` 1× 播放。

### 核心决策
1. **NVDEC 抽帧后置 CPU `swscale`（`src/stages/detector.py`）**：
   - `_decode_file_pipe` 与 `_decode_file_window` 的 CUDA 分析分支改为 `fps=...,hwdownload,format=nv12,scale={w}:{h}`（仅将 `fps` 抽选后的稀疏帧回传并经抗锯齿 `swscale` 下采样，保留渲染端 1080p `scale_cuda` 不变）；
   - 实测 `cuda_swscale` 与 `CPU` 像素差 `0.00`、`30` 帧解码 `1.53s` 略快于 `scale_cuda` `1.56s`，且 `20260320200620` 与 `20260326143025` 在 `CUDA` 与 `QSV` 下实现 `0.1s` 级完全一致结果。
2. **收紧无视觉目标 `DYNAMIC_AUDIO` 的边界与间隙吸收（`src/algorithms/segment.py`, `src/pipeline.py`）**：
   - `refine_activity_segments` 限制邻居扩展步长上界（`<= 1.0s`），并对无确认人物且纯声触发事件使用紧边界（`<= 0.35s / 0.50s`、`coalesce <= 0.6s`）；已确认人体（`confirmed(parent)`）或视觉高能事件仍维持完整 `pre_roll` / `post_roll`；
   - `build_segments` 与 `_filter_short` 停止把 `max_energy == 0.0` 快捷豁免用于无视觉目标的 `DYNAMIC_AUDIO`，仅允许吸收 `<= min(1.5s, gap_tolerance)` 短抖动，`pipeline.py` `audio_gate_fast_path` 采用 `(0.4s, 0.6s, 1.0s)` 有界前后滚。

## ADR0091：批次发布与恢复共用计划帧时钟（2026-11-23）

三个生产批次的最后一个 PRESENCE 片段均短于一个输出帧。虚拟输入上没有足够的视频帧时，音轨仍能到达计划终点；原有两帧时长容差允许短批次进入清单。批次首次发布与缓存复用都必须核对共享展示计划的整数视频帧数。末尾活跃区间短于一个输出帧时走连续输入，由既有的密集帧裁切规则处理，保留真实源区间和音轨。成片继续执行逐包 PTS 校验。

异构恢复允许使用另一受支持后端已完成的批次，条件是本次源身份、配置、该后端实际滤镜图对应的指纹、媒体身份及验证均匹配。检查在批次认领内完成；旧清单可以直接通过实测帧数验证。
