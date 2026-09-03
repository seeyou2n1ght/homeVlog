交代一下项目背景：
我需要开发一个程序项目，用于利用家用消费级主机（12600K + 3060Ti )不定期批量运行处理家庭室内监控摄像头生产的 4K 视频备份素材。目标是把素材片段（按 128MB 大小保存）浓缩成按每天保存的 DailyVlog 视频，对静态画面的处理做等比例抽帧浓缩，动态画面常速保留。摄像头素材包含夜间画面，微光全彩画面。
需求是在主机上准确快速完成处理，我可能会在周末夜间运行机器处理当周累积素材。
现在给出你的架构设计和处理流程，通过严谨的推理和测算分析，不得随意给出魔法数字夸大效果和效率

# GPT的建议

这是一个很有价值的工程问题——关键点在于用严谨的数字说话，而不是凭感觉给"几倍提速"之类的结论。我先把硬件能力、监控码率这些基础参数核实清楚，再给出架构设计和测算。

先做几组针对性调研。

基础参数已经有眉目了：3060Ti（GA104）NVENC 第7代，支持 HEVC 4K/10bit，但不支持 AV1 编码；NVDEC 第5代支持 AV1/H.265 硬解。监控码率方面，4K H.265 大约 6–8 Mbps。我再核实一下 NVENC 编码吞吐和 12600K 核显 QSV 的能力，这决定我整个测算的准确性。

编码侧数据很关键：4K HEVC 下 NVENC 约 60–95 fps（P4/P1），QSV（UHD 770）H.264 4K 约 57 fps、HEVC 相当。这些已足够支撑"编码不是瓶颈"的结论。我再确认 NVDEC 的解码吞吐，这是分析阶段的瓶颈所在。

解码与编码的吞吐数据基本齐了。还有两个关键点需要落实：GA104（3060Ti）媒体引擎的具体数量（决定并发模型），以及 UHD 770 的硬解能力（是否可作为第二条分析通道）。

媒体引擎数据齐了：3060Ti（GA104）是第7代 NVENC + 第5代 NVDEC，UHD 770 支持 4K@60 HEVC 硬编解码、可并行多路。现在开始整理测算模型，并先读取可视化 Skill 用于绘制架构图。

我准备用 HTML/SVG 静态信息图来呈现架构与流程。先读取该模式要求的参考文件。

核心数据都核到了。最后确认一个影响并发模型的细节——桌面版 3060Ti（GA104）的 NVENC 引擎数量。

数据核实完毕，可以给出严谨的架构设计和测算。先看整体结论，再逐项推导。

---

## 0. 结论先行（关键数字）

基于 **12600K + 3060Ti + UHD 770 核显** 的实测能力区间和 4K 监控常见码率（4–8 Mbps H.265），我测算后的核心结论：

- **单夜可处理量**：一个周末夜（约 10–12h 可用窗口）**稳妥处理 1 台相机一周（7 天）**；2 台相机一周需"高吞吐配置 + NVENC/QSV 双编码通道"勉强可行；3 台一周基本超单夜，需拆两夜或输出降为 1080p。
- **真正的瓶颈不是解码分析，而是"动态画面的全帧率 4K 编码"**——这是需求本身决定的不可压缩工作量。
- **输出体积**：静态段按 1fps 抽帧后，静态 24h 只压缩成约 **1–2 分钟** 播放时长；输出大小主要被动态段决定（按 8Mbps 计，动态占比 15% 时约 13GB/相机/天）。
- **3060Ti 不支持 AV1 编码**（第 7 代 NVENC 仅 H.264/HEVC），所以输出格式定为 **HEVC**；核显 UHD 770 是现成的第二条独立编解码通道，用于并发和冗余。

> 
> ⚠️ 下面所有数字都给出推导公式与假设，属于"量级级测算"，**首个批次必须先做一次本机校准**（用 `ffmpeg -benchmark` 实测真实解码/编码 fps），再冻结并发参数。

---

## 1. 硬件能力核实（先定天花板，不拍脑袋）

| 硬件 | 与任务相关的能力 | 影响 |
| --- | --- | --- |
| **RTX 3060Ti**（GA104，第7代 NVENC / 第5代 NVDEC） | HEVC 4K **编码** 支持（含 10bit）；**不支持 AV1 编码**；NVDEC 硬解 AV1/H.265/H.264/VP9，支持多路并发解码会话（官方消费卡并发会话上限约 8–12 路） | 输出必须走 HEVC；解码/编码可同时多路跑 |
| **NVENC 4K HEVC 编码吞吐** | 4K 实测约 **P1≈90fps / P4≈60–70fps**（单会话） | 编码不是"秒出"，要按输出帧数算账 |
| **UHD 770 核显（QSV）** | 4K@60 HEVC 硬编解码；12600K 实测 H.264 4K 编码约 **57fps**（HEVC 同量级） | 可作**第二条编码/解码通道**，与独显并行 |
| **12600K CPU**（6P+4E） | 活动度计算、分段、调度、I/O 全在 CPU；4K 降采样后 640×360 的帧差计算每帧 <0.5ms | CPU 远不是瓶颈 |

**关键推论**：分析阶段（解码→降采样→活动度）由 NVDEC/QSV 承担，吞吐高；编码阶段是算账的核心。

---

## 2. 架构设计（分四层 + 控制面）

整体是一个**"逐分片分析 → 逐分片重编码 → 按日组装"的流水线**，128MB 分片（约 2.4 分钟/片）天然是最佳任务粒度，利于并发和断点续跑。

架构图如下（静态信息图，说明四层流水线与控制面、硬件分工）：

流水线架构图如下（说明四层处理链路、控制面与硬件分工）：

```
<html style="margin:0;padding:0;">
<div style="background-color:transparent;box-sizing:border-box;--accent:#A3D5E8;font-family:'Roboto','PingFang SC','Segoe UI',Arial,sans-serif;color:#1A1B1C;">
  <div style="padding:12px;box-sizing:border-box;">
    <!-- 控制面 -->
    <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 12px;background:rgba(163,213,232,0.16);border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
      <span style="font-size:13px;font-weight:600;margin-right:2px;">控制面</span>
      <span style="font-size:12px;color:#4A5568;background:#fff;border:0.5px solid rgba(0,0,0,0.08);border-radius:8px;padding:4px 8px;">任务调度器 · 并发队列</span>
      <span style="font-size:12px;color:#4A5568;background:#fff;border:0.5px solid rgba(0,0,0,0.08);border-radius:8px;padding:4px 8px;">SQLite 状态库 · 断点续跑</span>
      <span style="font-size:12px;color:#4A5568;background:#fff;border:0.5px solid rgba(0,0,0,0.08);border-radius:8px;padding:4px 8px;">失败重试 · 原子落盘</span>
    </div>

    <!-- 阶段1 -->
    <div style="display:flex;flex-direction:column;align-items:center;margin:10px 0 0 0;">
      <div style="width:100%;padding:12px;background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px;">
          <span style="font-size:13px;font-weight:600;">① 盘点 / 清单</span>
          <span style="font-size:11px;color:#4A5568;background:rgba(163,213,232,0.25);border-radius:8px;padding:3px 8px;">CPU · ffprobe</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">扫描 128MB 分片</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">探测编码/分辨率/帧率/时长</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">按 相机×日期 归类</span>
        </div>
        <div style="font-size:11px;color:#6B7280;margin-top:8px;">产物 → Manifest（分片→相机/日期映射）</div>
      </div>
      <div style="color:#9EACEA;font-size:16px;line-height:1;margin:4px 0;">↓</div>

      <!-- 阶段2 -->
      <div style="width:100%;padding:12px;background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px;">
          <span style="font-size:13px;font-weight:600;">② 分析（解码 + 活动度）</span>
          <span style="font-size:11px;color:#4A5568;background:rgba(163,213,232,0.25);border-radius:8px;padding:3px 8px;">NVDEC / QSV + CPU</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">硬解（GPU 内保持）</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">GPU 降采样 640×360</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">帧差活动度 + 噪声底自适应</span>
        </div>
        <div style="font-size:11px;color:#6B7280;margin-top:8px;">产物 → 每分片分段图 segment_map.json（静态/动态区间）</div>
      </div>
      <div style="color:#9EACEA;font-size:16px;line-height:1;margin:4px 0;">↓</div>

      <!-- 阶段3 -->
      <div style="width:100%;padding:12px;background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px;">
          <span style="font-size:13px;font-weight:600;">③ 编码（帧选择 + HEVC）</span>
          <span style="font-size:11px;color:#4A5568;background:rgba(163,213,232,0.25);border-radius:8px;padding:3px 8px;">NVENC + QSV 双通道</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">静态段：抽 1/30（→1fps 浓缩）</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">动态段：全帧率 常速</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">边界保留 1–2s 缓冲</span>
        </div>
        <div style="font-size:11px;color:#6B7280;margin-top:8px;">产物 → 每分片 chunk_out.mp4（HEVC 4K）</div>
      </div>
      <div style="color:#9EACEA;font-size:16px;line-height:1;margin:4px 0;">↓</div>

      <!-- 阶段4 -->
      <div style="width:100%;padding:12px;background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px;">
          <span style="font-size:13px;font-weight:600;">④ 组装 / QC / 索引</span>
          <span style="font-size:11px;color:#4A5568;background:rgba(163,213,232,0.25);border-radius:8px;padding:3px 8px;">CPU</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">按日 concat（同参数）</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">帧数/时长/黑帧校验</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">动态事件索引 + 缩略图</span>
        </div>
        <div style="font-size:11px;color:#6B7280;margin-top:8px;">产物 → DailyVlog_YYYY-MM-DD.mp4 + 元数据 / 事件清单</div>
      </div>
    </div>
  </div>
</div>
</html>
```

> 
> 架构要点：**以 128MB 分片为任务粒度**（约 2.4 分钟/片），分析（②）与编码（③）可对**不同分片并行**，用工作队列消解流水线气泡；每片落盘一次状态，断电/崩溃后下个周末可续跑。

**组件职责与数据流**：Manifest 负责把散片归到"相机×日期"；分段图 JSON 是分析产物，同时驱动编码（帧选择）和最终索引（列出当天每个动态事件的时间戳，方便"跳看今天发生了什么"）。QC 在最终交付前拦截坏片。

---

## 3. 核心算法与处理流程

### 3.1 活动度度量（动静判定）

分析层不对 4K 全分辨率计算，而是 **NVDEC/QSV 硬解 → GPU 内降采样到 640×360 → 只把缩小帧送回 CPU** 做帧差，成本极低：

- **度量**：亮度帧差 `d_t = |Y_t − Y_{t−1}|`，先做 3×3 空间模糊 + 3–5 帧时间平均抑制闪烁；活动度 `a_t` = 帧差直方图的 **高百分位均值**（如 top-5%），对局部运动敏感、对全局亮度漂移不敏感。
- **夜间/微光全彩噪声处理（关键）**：低光下增益放大导致传感器噪声，即使画面静止帧差也不为零。做法是**逐分片估计噪声底**——用一段已判定为静止的区间，取帧差的 MAD（中位绝对偏差）作为 σ；自适应阈值 `T = μ_noise + k·σ_noise`（k≈4–6）。这样夜间阈值自动抬高，白天自动回落，不靠魔法常数。

### 3.2 动静分段状态机（带滞回）

判定采用带**滞回**的状态机，防止边界抖动：

```
<html style="margin:0;padding:0;">
<div style="background-color:transparent;box-sizing:border-box;--accent:#A3D5E8;font-family:'Roboto','PingFang SC','Segoe UI',Arial,sans-serif;color:#1A1B1C;">
  <div style="padding:12px;box-sizing:border-box;">
    <div style="font-size:13px;font-weight:600;margin-bottom:10px;">动静分段状态机（滞回判定）</div>
    <!-- 阈值示意图 -->
    <svg viewBox="0 0 640 190" width="100%" style="display:block;max-width:640px;margin:0 auto 12px auto;">
      <rect x="0" y="0" width="640" height="190" fill="rgba(148,212,208,0.10)" rx="8"/>
      <line x1="40" y1="132" x2="620" y2="132" stroke="#94D4D0" stroke-width="2" stroke-dasharray="5 4"/>
      <line x1="40" y1="96" x2="620" y2="96" stroke="#EAA7B2" stroke-width="2" stroke-dasharray="5 4"/>
      <text x="624" y="136" text-anchor="end" font-size="11" fill="#4A5568">T_lo（进入静态阈值）</text>
      <text x="624" y="100" text-anchor="end" font-size="11" fill="#4A5568">T_hi（进入动态阈值）</text>
      <polyline fill="none" stroke="#1A1B1C" stroke-width="2" points="40,140 90,138 120,142 150,120 175,80 200,62 230,70 260,90 300,108 330,130 360,136 400,132 440,120 470,88 495,66 520,74 550,112 580,132 620,138"/>
      <circle cx="175" cy="80" r="4" fill="#EAA7B2"/>
      <circle cx="470" cy="88" r="4" fill="#94D4D0"/>
      <text x="175" y="64" text-anchor="middle" font-size="11" fill="#1A1B1C" font-weight="600">上升过 T_hi → 动态</text>
      <text x="470" y="72" text-anchor="middle" font-size="11" fill="#1A1B1C" font-weight="600">回落过 T_lo → 静态</text>
      <text x="40" y="22" font-size="11" fill="#6B7280">活动度 a_t</text>
      <text x="560" y="178" font-size="11" fill="#6B7280">时间 →</text>
    </svg>
    <!-- 状态 -->
    <div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center;">
      <div style="flex:1 1 220px;padding:12px;background:rgba(148,212,208,0.14);border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="font-size:13px;font-weight:600;">静态态</div>
        <div style="font-size:12px;color:#4A5568;margin-top:6px;">抽帧 1/30 → 输出 1fps 浓缩<br/>每 30 帧保留 1 帧，30× 时间压缩</div>
      </div>
      <div style="display:flex;flex-direction:column;justify-content:center;align-items:center;font-size:12px;color:#6B7280;">
        <span style="background:#fff;border:0.5px solid rgba(0,0,0,0.08);border-radius:8px;padding:3px 8px;margin:2px 0;">a &gt; T_hi 持续 ≥ τ_on（0.5–1s）</span>
        <span style="color:#9EACEA;">⇄</span>
        <span style="background:#fff;border:0.5px solid rgba(0,0,0,0.08);border-radius:8px;padding:3px 8px;margin:2px 0;">a &lt; T_lo 持续 ≥ τ_off（3–5s）</span>
      </div>
      <div style="flex:1 1 220px;padding:12px;background:rgba(234,167,178,0.14);border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="font-size:13px;font-weight:600;">动态态</div>
        <div style="font-size:12px;color:#4A5568;margin-top:6px;">全帧率 30fps 常速保留<br/>转换边界前后补 1–2s 常速缓冲</div>
      </div>
    </div>
    <div style="font-size:11px;color:#6B7280;margin-top:10px;line-height:1.6;">滞回：进入动态用高阈值 T_hi，退出用低阈值 T_lo（T_hi &gt; T_lo），配合最小持续时长 τ_on/τ_off 消抖，避免画面在阈值附近反复跳变。</div>
  </div>
</div>
</html>
```

> 
> 参数（T、k、τ_on/τ_off、静态输出帧率、边界缓冲）都应做成配置文件；首跑用一小段夜间+白天样片校准 k 与 τ，之后全自动。

### 3.3 编码与组装

- **帧选择**：每分片用分段图生成 `select` 表达式（静态区间内 `mod(n,30)==0`，动态区间全选），配合 `setpts` 重打时间戳——静态段每保留帧只播 1/30s，等效 30× 快进；动态段原速。输出统一 30fps、HEVC。
- **参数建议**：`hevc_nvenc -preset p5 -rc vbr -cq 26 -spatial_aq 1 -temporal_aq 1`（AQ 对夜间噪声场景帮助明显）；可选 10-bit 编码抑制暗部色带。QSV 与 NVENC 用相同参数，保证分片可 concat。
- **音频**：默认策略建议"仅动态段保留"或整体丢弃（见决策点④）。

---

## 4. 吞吐与时间测算（全部可复核推导）

### 4.1 输入量

- 监控 4K H.265 常见码率 **B = 4–8 Mbps**（以 7Mbps 为基准）
- 单机单天体积：`V_day = B × 86400 / 8 / 1000 = 10.8 × B` GB → **B=7 时 ≈ 75.6 GB/天**
- 128MB 分片数：`75.6GB ÷ 128MB ≈ 590 片/天`，单片时长 ≈ **2.4 分钟**
- 一周（2 台 × 7 天）≈ **1.06 TB** 输入

### 4.2 分析与编码吞吐

| 环节 | 通道 | 吞吐假设（保守→现实） |
| --- | --- | --- |
| 分析（解码+活动度） | NVDEC ×3–4 会话（可选 QSV +1–2） | 单会话 ≥8× 实时（保守）→ 12–20×；合计 **24× → 40–60×** |
| 编码（HEVC 4K） | NVENC（P4≈60–70fps）+ QSV（≈57fps） | 单会话 ~60–90fps；双通道 ~120–180fps |

NVDEC 支持多会话并发、切换开销可忽略；NVENC/QSV 为独立硬件，可同时工作。

### 4.3 输出时长与体积（取决于动态占比 S）

公式（F=30fps，静态抽 1/30 → 1fps）：

- 动态帧数 = `(1−S) × 86400 × 30`；静态保留帧 = `S × 86400 / 30`
- 输出时长 = 动态秒数 + `S×86400/900`

| 动态占比 1−S | 输出时长/天 | 输出帧数/天 | 编码耗时(单NVENC@70fps) | 输出体积/天(动态按8Mbps) |
| --- | --- | --- | --- | --- |
| 30% | 7.2h | 78 万帧 | 3.1h | ≈26 GB |
| **15%（基准）** | **3.6h** | **39 万帧** | **1.6h** | **≈13 GB** |
| 10% | 2.4h | 26 万帧 | 1.0h | ≈8.6 GB |
| 5% | 1.2h | 13 万帧 | 0.5h | ≈4.3 GB |

> 
> 注意两个"反直觉"点：① 静态 24h 浓缩后只占播放 1–2 分钟，**输出长度几乎全由动态段决定**；② 因为动态要常速保留，**体积只能压缩到输入的约 1/6**（基准），时间上压缩到 1/6.6——这是需求边界决定的，不是算法差。

### 4.4 周末夜间批量预算（基准：B=7Mbps、S=0.85、静态1fps）

分析/编码对**不同分片并行**，整批墙钟时间 ≈ `max(分析总时长, 编码总时长)` + 管道填充：

| 批量 | 输入 | 分析(24×→48×) | 编码 1 路 | 编码 2–3 路 | 单夜(10–12h)可行性 |
| --- | --- | --- | --- | --- | --- |
| 1 台 × 7 天 | 0.53 TB | 7h → 3.5h | 10.9h | 5.4h → ~3.6h | **✓ 稳妥** |
| 2 台 × 7 天 | 1.06 TB | 14h → 7h | 21.7h | 10.9h → ~7h | 需高速分析+双/三编码，紧张 |
| 3 台 × 7 天 | 1.59 TB | 21h → 10.5h | 32.6h | 16.3h → ~11h | **超单夜**（拆两夜 或 输出1080p） |

**结论**：

- 单夜先按 **1 台相机 × 一周** 配置，最稳；
- 想一晚上跑完 2–3 台，两个杠杆：① 输出降到 **1080p（编码提速约 4 倍，瓶颈直接消失）**；② 动态段增加"超过 N 分钟连续动态再二次抽帧"的可选上限；
- 首跑务必用 `ffmpeg -benchmark` 实测你机器上的真实解码/编码 fps 校准，我上面的 8×/70fps 都是保守下限，实际大概率更快。

### 4.5 I/O 与内存

- 双 pass（先分析后编码）会让**总读取 ≈ 2× 输入体积**：1.06TB → 约 2.1TB/夜 → 持续 ~50MB/s，机械盘即可，SSD 更稳。
- 8GB 显存：4K 解码会话显存占用约 100–200MB/路，4–5 路并发 + 2 路编码远在预算内。
- 输出按日落盘后，原始素材保留一段时间再删，QC 通过前不动原件。

---

## 5. 需要你拍板的 4 个决策点

1. **输出分辨率**：4K 保真 vs 1080p（时间/体积省 ~4×）。DailyVlog 用途下 1080p 通常够看，我建议默认 4K、给 1080p 开关。
2. **动态占比 S 的真实值**：取决于你家"有人在动"的时间占比，直接决定输出长度与单夜容量，跑一次分析就得到实测值。
3. **实际码率/编码格式**：请从相机设置读真实码率（4 还是 8Mbps）和编码（H.264/H.265）——若相机是 H.264 4K（16–32Mbps），输入量会大 2–4 倍，预算需重算。
4. **音频**：保留（仅动态段）还是丢弃。家庭监控音频有隐私考量，建议默认丢弃，需要时再开。

---

## 6. 工程落地建议（技术栈与稳健性）

- **技术栈**：Python 3.10+ + ffmpeg 6.x（`--enable-cuda` + nv-codec-headers）+ PyAV/OpenCV + numpy；SQLite 存 Manifest 与任务状态；`multiprocessing` 起 2–4 个分析 worker + 1–2 个编码 worker（用 NVENC 与 QSV 两条独立通道）。
- **断点与恢复**：以分片为原子单位，输出写 `.part` 再改名，SQLite 记 done；中断后下个周末自动续跑。
- **QC 门**：每片校验帧数/时长；抽样 5–10 帧做亮度直方图查黑帧/花屏；每日 concat 后生成**事件时间线索引 + 关键帧缩略图**（HTML），让你不用翻视频就能看当天有几次活动、各在几点。
- **首跑校准清单**：真实码率、NVDEC 解码 fps、NVENC/QSV 编码 fps、夜间噪声底 σ——跑一次 10 分钟样片即可全部拿到，再冻结参数。

---

## 一句话总结

**这套架构的骨架是"分片级两遍流水线"：NVDEC/QSV 硬解做噪声自适应的动静分段，静态段 30 倍抽帧、动态段常速保留，NVENC+QSV 双通道 HEVC 编码，按日 concat 出 DailyVlog；测算表明单夜稳妥处理 1 台相机一周，瓶颈在动态画面的 4K 编码工作量而非分析，想扩到 2–3 台就上 1080p 输出或拆两夜。**

下一步建议：把相机实际码率/编码/是否连续录制这三个参数发我，我可以把第 4 节预算表按你的真实数据重算一遍，或直接帮你搭一个最小可跑的原型（先支持 1 台相机、能出 DailyVlog 即可）。

# 豆包的建议

这是一个很有价值的工程问题——关键点在于用严谨的数字说话，而不是凭感觉给"几倍提速"之类的结论。我先把硬件能力、监控码率这些基础参数核实清楚，再给出架构设计和测算。

先做几组针对性调研。

基础参数已经有眉目了：3060Ti（GA104）NVENC 第7代，支持 HEVC 4K/10bit，但不支持 AV1 编码；NVDEC 第5代支持 AV1/H.265 硬解。监控码率方面，4K H.265 大约 6–8 Mbps。我再核实一下 NVENC 编码吞吐和 12600K 核显 QSV 的能力，这决定我整个测算的准确性。

编码侧数据很关键：4K HEVC 下 NVENC 约 60–95 fps（P4/P1），QSV（UHD 770）H.264 4K 约 57 fps、HEVC 相当。这些已足够支撑"编码不是瓶颈"的结论。我再确认 NVDEC 的解码吞吐，这是分析阶段的瓶颈所在。

解码与编码的吞吐数据基本齐了。还有两个关键点需要落实：GA104（3060Ti）媒体引擎的具体数量（决定并发模型），以及 UHD 770 的硬解能力（是否可作为第二条分析通道）。

媒体引擎数据齐了：3060Ti（GA104）是第7代 NVENC + 第5代 NVDEC，UHD 770 支持 4K@60 HEVC 硬编解码、可并行多路。现在开始整理测算模型，并先读取可视化 Skill 用于绘制架构图。

我准备用 HTML/SVG 静态信息图来呈现架构与流程。先读取该模式要求的参考文件。

核心数据都核到了。最后确认一个影响并发模型的细节——桌面版 3060Ti（GA104）的 NVENC 引擎数量。

数据核实完毕，可以给出严谨的架构设计和测算。先看整体结论，再逐项推导。

---

## 0. 结论先行（关键数字）

基于 **12600K + 3060Ti + UHD 770 核显** 的实测能力区间和 4K 监控常见码率（4–8 Mbps H.265），我测算后的核心结论：

- **单夜可处理量**：一个周末夜（约 10–12h 可用窗口）**稳妥处理 1 台相机一周（7 天）**；2 台相机一周需"高吞吐配置 + NVENC/QSV 双编码通道"勉强可行；3 台一周基本超单夜，需拆两夜或输出降为 1080p。
- **真正的瓶颈不是解码分析，而是"动态画面的全帧率 4K 编码"**——这是需求本身决定的不可压缩工作量。
- **输出体积**：静态段按 1fps 抽帧后，静态 24h 只压缩成约 **1–2 分钟** 播放时长；输出大小主要被动态段决定（按 8Mbps 计，动态占比 15% 时约 13GB/相机/天）。
- **3060Ti 不支持 AV1 编码**（第 7 代 NVENC 仅 H.264/HEVC），所以输出格式定为 **HEVC**；核显 UHD 770 是现成的第二条独立编解码通道，用于并发和冗余。

> 
> ⚠️ 下面所有数字都给出推导公式与假设，属于"量级级测算"，**首个批次必须先做一次本机校准**（用 `ffmpeg -benchmark` 实测真实解码/编码 fps），再冻结并发参数。

---

## 1. 硬件能力核实（先定天花板，不拍脑袋）

| 硬件 | 与任务相关的能力 | 影响 |
| --- | --- | --- |
| **RTX 3060Ti**（GA104，第7代 NVENC / 第5代 NVDEC） | HEVC 4K **编码** 支持（含 10bit）；**不支持 AV1 编码**；NVDEC 硬解 AV1/H.265/H.264/VP9，支持多路并发解码会话（官方消费卡并发会话上限约 8–12 路） | 输出必须走 HEVC；解码/编码可同时多路跑 |
| **NVENC 4K HEVC 编码吞吐** | 4K 实测约 **P1≈90fps / P4≈60–70fps**（单会话） | 编码不是"秒出"，要按输出帧数算账 |
| **UHD 770 核显（QSV）** | 4K@60 HEVC 硬编解码；12600K 实测 H.264 4K 编码约 **57fps**（HEVC 同量级） | 可作**第二条编码/解码通道**，与独显并行 |
| **12600K CPU**（6P+4E） | 活动度计算、分段、调度、I/O 全在 CPU；4K 降采样后 640×360 的帧差计算每帧 <0.5ms | CPU 远不是瓶颈 |

**关键推论**：分析阶段（解码→降采样→活动度）由 NVDEC/QSV 承担，吞吐高；编码阶段是算账的核心。

---

## 2. 架构设计（分四层 + 控制面）

整体是一个**"逐分片分析 → 逐分片重编码 → 按日组装"的流水线**，128MB 分片（约 2.4 分钟/片）天然是最佳任务粒度，利于并发和断点续跑。

架构图如下（静态信息图，说明四层流水线与控制面、硬件分工）：

流水线架构图如下（说明四层处理链路、控制面与硬件分工）：

```
<html style="margin:0;padding:0;">
<div style="background-color:transparent;box-sizing:border-box;--accent:#A3D5E8;font-family:'Roboto','PingFang SC','Segoe UI',Arial,sans-serif;color:#1A1B1C;">
  <div style="padding:12px;box-sizing:border-box;">
    <!-- 控制面 -->
    <div style="display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:10px 12px;background:rgba(163,213,232,0.16);border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
      <span style="font-size:13px;font-weight:600;margin-right:2px;">控制面</span>
      <span style="font-size:12px;color:#4A5568;background:#fff;border:0.5px solid rgba(0,0,0,0.08);border-radius:8px;padding:4px 8px;">任务调度器 · 并发队列</span>
      <span style="font-size:12px;color:#4A5568;background:#fff;border:0.5px solid rgba(0,0,0,0.08);border-radius:8px;padding:4px 8px;">SQLite 状态库 · 断点续跑</span>
      <span style="font-size:12px;color:#4A5568;background:#fff;border:0.5px solid rgba(0,0,0,0.08);border-radius:8px;padding:4px 8px;">失败重试 · 原子落盘</span>
    </div>

    <!-- 阶段1 -->
    <div style="display:flex;flex-direction:column;align-items:center;margin:10px 0 0 0;">
      <div style="width:100%;padding:12px;background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px;">
          <span style="font-size:13px;font-weight:600;">① 盘点 / 清单</span>
          <span style="font-size:11px;color:#4A5568;background:rgba(163,213,232,0.25);border-radius:8px;padding:3px 8px;">CPU · ffprobe</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">扫描 128MB 分片</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">探测编码/分辨率/帧率/时长</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">按 相机×日期 归类</span>
        </div>
        <div style="font-size:11px;color:#6B7280;margin-top:8px;">产物 → Manifest（分片→相机/日期映射）</div>
      </div>
      <div style="color:#9EACEA;font-size:16px;line-height:1;margin:4px 0;">↓</div>

      <!-- 阶段2 -->
      <div style="width:100%;padding:12px;background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px;">
          <span style="font-size:13px;font-weight:600;">② 分析（解码 + 活动度）</span>
          <span style="font-size:11px;color:#4A5568;background:rgba(163,213,232,0.25);border-radius:8px;padding:3px 8px;">NVDEC / QSV + CPU</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">硬解（GPU 内保持）</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">GPU 降采样 640×360</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">帧差活动度 + 噪声底自适应</span>
        </div>
        <div style="font-size:11px;color:#6B7280;margin-top:8px;">产物 → 每分片分段图 segment_map.json（静态/动态区间）</div>
      </div>
      <div style="color:#9EACEA;font-size:16px;line-height:1;margin:4px 0;">↓</div>

      <!-- 阶段3 -->
      <div style="width:100%;padding:12px;background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px;">
          <span style="font-size:13px;font-weight:600;">③ 编码（帧选择 + HEVC）</span>
          <span style="font-size:11px;color:#4A5568;background:rgba(163,213,232,0.25);border-radius:8px;padding:3px 8px;">NVENC + QSV 双通道</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">静态段：抽 1/30（→1fps 浓缩）</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">动态段：全帧率 常速</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">边界保留 1–2s 缓冲</span>
        </div>
        <div style="font-size:11px;color:#6B7280;margin-top:8px;">产物 → 每分片 chunk_out.mp4（HEVC 4K）</div>
      </div>
      <div style="color:#9EACEA;font-size:16px;line-height:1;margin:4px 0;">↓</div>

      <!-- 阶段4 -->
      <div style="width:100%;padding:12px;background:#fff;border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="display:flex;flex-wrap:wrap;justify-content:space-between;align-items:center;gap:6px;">
          <span style="font-size:13px;font-weight:600;">④ 组装 / QC / 索引</span>
          <span style="font-size:11px;color:#4A5568;background:rgba(163,213,232,0.25);border-radius:8px;padding:3px 8px;">CPU</span>
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:6px;margin-top:8px;">
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">按日 concat（同参数）</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">帧数/时长/黑帧校验</span>
          <span style="font-size:12px;color:#4A5568;background:rgba(26,27,28,0.04);border-radius:6px;padding:3px 8px;">动态事件索引 + 缩略图</span>
        </div>
        <div style="font-size:11px;color:#6B7280;margin-top:8px;">产物 → DailyVlog_YYYY-MM-DD.mp4 + 元数据 / 事件清单</div>
      </div>
    </div>
  </div>
</div>
</html>
```

> 
> 架构要点：**以 128MB 分片为任务粒度**（约 2.4 分钟/片），分析（②）与编码（③）可对**不同分片并行**，用工作队列消解流水线气泡；每片落盘一次状态，断电/崩溃后下个周末可续跑。

**组件职责与数据流**：Manifest 负责把散片归到"相机×日期"；分段图 JSON 是分析产物，同时驱动编码（帧选择）和最终索引（列出当天每个动态事件的时间戳，方便"跳看今天发生了什么"）。QC 在最终交付前拦截坏片。

---

## 3. 核心算法与处理流程

### 3.1 活动度度量（动静判定）

分析层不对 4K 全分辨率计算，而是 **NVDEC/QSV 硬解 → GPU 内降采样到 640×360 → 只把缩小帧送回 CPU** 做帧差，成本极低：

- **度量**：亮度帧差 `d_t = |Y_t − Y_{t−1}|`，先做 3×3 空间模糊 + 3–5 帧时间平均抑制闪烁；活动度 `a_t` = 帧差直方图的 **高百分位均值**（如 top-5%），对局部运动敏感、对全局亮度漂移不敏感。
- **夜间/微光全彩噪声处理（关键）**：低光下增益放大导致传感器噪声，即使画面静止帧差也不为零。做法是**逐分片估计噪声底**——用一段已判定为静止的区间，取帧差的 MAD（中位绝对偏差）作为 σ；自适应阈值 `T = μ_noise + k·σ_noise`（k≈4–6）。这样夜间阈值自动抬高，白天自动回落，不靠魔法常数。

### 3.2 动静分段状态机（带滞回）

判定采用带**滞回**的状态机，防止边界抖动：

```
<html style="margin:0;padding:0;">
<div style="background-color:transparent;box-sizing:border-box;--accent:#A3D5E8;font-family:'Roboto','PingFang SC','Segoe UI',Arial,sans-serif;color:#1A1B1C;">
  <div style="padding:12px;box-sizing:border-box;">
    <div style="font-size:13px;font-weight:600;margin-bottom:10px;">动静分段状态机（滞回判定）</div>
    <!-- 阈值示意图 -->
    <svg viewBox="0 0 640 190" width="100%" style="display:block;max-width:640px;margin:0 auto 12px auto;">
      <rect x="0" y="0" width="640" height="190" fill="rgba(148,212,208,0.10)" rx="8"/>
      <line x1="40" y1="132" x2="620" y2="132" stroke="#94D4D0" stroke-width="2" stroke-dasharray="5 4"/>
      <line x1="40" y1="96" x2="620" y2="96" stroke="#EAA7B2" stroke-width="2" stroke-dasharray="5 4"/>
      <text x="624" y="136" text-anchor="end" font-size="11" fill="#4A5568">T_lo（进入静态阈值）</text>
      <text x="624" y="100" text-anchor="end" font-size="11" fill="#4A5568">T_hi（进入动态阈值）</text>
      <polyline fill="none" stroke="#1A1B1C" stroke-width="2" points="40,140 90,138 120,142 150,120 175,80 200,62 230,70 260,90 300,108 330,130 360,136 400,132 440,120 470,88 495,66 520,74 550,112 580,132 620,138"/>
      <circle cx="175" cy="80" r="4" fill="#EAA7B2"/>
      <circle cx="470" cy="88" r="4" fill="#94D4D0"/>
      <text x="175" y="64" text-anchor="middle" font-size="11" fill="#1A1B1C" font-weight="600">上升过 T_hi → 动态</text>
      <text x="470" y="72" text-anchor="middle" font-size="11" fill="#1A1B1C" font-weight="600">回落过 T_lo → 静态</text>
      <text x="40" y="22" font-size="11" fill="#6B7280">活动度 a_t</text>
      <text x="560" y="178" font-size="11" fill="#6B7280">时间 →</text>
    </svg>
    <!-- 状态 -->
    <div style="display:flex;flex-wrap:wrap;gap:10px;justify-content:center;">
      <div style="flex:1 1 220px;padding:12px;background:rgba(148,212,208,0.14);border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="font-size:13px;font-weight:600;">静态态</div>
        <div style="font-size:12px;color:#4A5568;margin-top:6px;">抽帧 1/30 → 输出 1fps 浓缩<br/>每 30 帧保留 1 帧，30× 时间压缩</div>
      </div>
      <div style="display:flex;flex-direction:column;justify-content:center;align-items:center;font-size:12px;color:#6B7280;">
        <span style="background:#fff;border:0.5px solid rgba(0,0,0,0.08);border-radius:8px;padding:3px 8px;margin:2px 0;">a &gt; T_hi 持续 ≥ τ_on（0.5–1s）</span>
        <span style="color:#9EACEA;">⇄</span>
        <span style="background:#fff;border:0.5px solid rgba(0,0,0,0.08);border-radius:8px;padding:3px 8px;margin:2px 0;">a &lt; T_lo 持续 ≥ τ_off（3–5s）</span>
      </div>
      <div style="flex:1 1 220px;padding:12px;background:rgba(234,167,178,0.14);border:1px solid rgba(0,0,0,0.08);border-radius:12px;box-sizing:border-box;">
        <div style="font-size:13px;font-weight:600;">动态态</div>
        <div style="font-size:12px;color:#4A5568;margin-top:6px;">全帧率 30fps 常速保留<br/>转换边界前后补 1–2s 常速缓冲</div>
      </div>
    </div>
    <div style="font-size:11px;color:#6B7280;margin-top:10px;line-height:1.6;">滞回：进入动态用高阈值 T_hi，退出用低阈值 T_lo（T_hi &gt; T_lo），配合最小持续时长 τ_on/τ_off 消抖，避免画面在阈值附近反复跳变。</div>
  </div>
</div>
</html>
```

> 
> 参数（T、k、τ_on/τ_off、静态输出帧率、边界缓冲）都应做成配置文件；首跑用一小段夜间+白天样片校准 k 与 τ，之后全自动。

### 3.3 编码与组装

- **帧选择**：每分片用分段图生成 `select` 表达式（静态区间内 `mod(n,30)==0`，动态区间全选），配合 `setpts` 重打时间戳——静态段每保留帧只播 1/30s，等效 30× 快进；动态段原速。输出统一 30fps、HEVC。
- **参数建议**：`hevc_nvenc -preset p5 -rc vbr -cq 26 -spatial_aq 1 -temporal_aq 1`（AQ 对夜间噪声场景帮助明显）；可选 10-bit 编码抑制暗部色带。QSV 与 NVENC 用相同参数，保证分片可 concat。
- **音频**：默认策略建议"仅动态段保留"或整体丢弃（见决策点④）。

---

## 4. 吞吐与时间测算（全部可复核推导）

### 4.1 输入量

- 监控 4K H.265 常见码率 **B = 4–8 Mbps**（以 7Mbps 为基准）
- 单机单天体积：`V_day = B × 86400 / 8 / 1000 = 10.8 × B` GB → **B=7 时 ≈ 75.6 GB/天**
- 128MB 分片数：`75.6GB ÷ 128MB ≈ 590 片/天`，单片时长 ≈ **2.4 分钟**
- 一周（2 台 × 7 天）≈ **1.06 TB** 输入

### 4.2 分析与编码吞吐

| 环节 | 通道 | 吞吐假设（保守→现实） |
| --- | --- | --- |
| 分析（解码+活动度） | NVDEC ×3–4 会话（可选 QSV +1–2） | 单会话 ≥8× 实时（保守）→ 12–20×；合计 **24× → 40–60×** |
| 编码（HEVC 4K） | NVENC（P4≈60–70fps）+ QSV（≈57fps） | 单会话 ~60–90fps；双通道 ~120–180fps |

NVDEC 支持多会话并发、切换开销可忽略；NVENC/QSV 为独立硬件，可同时工作。

### 4.3 输出时长与体积（取决于动态占比 S）

公式（F=30fps，静态抽 1/30 → 1fps）：

- 动态帧数 = `(1−S) × 86400 × 30`；静态保留帧 = `S × 86400 / 30`
- 输出时长 = 动态秒数 + `S×86400/900`

| 动态占比 1−S | 输出时长/天 | 输出帧数/天 | 编码耗时(单NVENC@70fps) | 输出体积/天(动态按8Mbps) |
| --- | --- | --- | --- | --- |
| 30% | 7.2h | 78 万帧 | 3.1h | ≈26 GB |
| **15%（基准）** | **3.6h** | **39 万帧** | **1.6h** | **≈13 GB** |
| 10% | 2.4h | 26 万帧 | 1.0h | ≈8.6 GB |
| 5% | 1.2h | 13 万帧 | 0.5h | ≈4.3 GB |

> 
> 注意两个"反直觉"点：① 静态 24h 浓缩后只占播放 1–2 分钟，**输出长度几乎全由动态段决定**；② 因为动态要常速保留，**体积只能压缩到输入的约 1/6**（基准），时间上压缩到 1/6.6——这是需求边界决定的，不是算法差。

### 4.4 周末夜间批量预算（基准：B=7Mbps、S=0.85、静态1fps）

分析/编码对**不同分片并行**，整批墙钟时间 ≈ `max(分析总时长, 编码总时长)` + 管道填充：

| 批量 | 输入 | 分析(24×→48×) | 编码 1 路 | 编码 2–3 路 | 单夜(10–12h)可行性 |
| --- | --- | --- | --- | --- | --- |
| 1 台 × 7 天 | 0.53 TB | 7h → 3.5h | 10.9h | 5.4h → ~3.6h | **✓ 稳妥** |
| 2 台 × 7 天 | 1.06 TB | 14h → 7h | 21.7h | 10.9h → ~7h | 需高速分析+双/三编码，紧张 |
| 3 台 × 7 天 | 1.59 TB | 21h → 10.5h | 32.6h | 16.3h → ~11h | **超单夜**（拆两夜 或 输出1080p） |

**结论**：

- 单夜先按 **1 台相机 × 一周** 配置，最稳；
- 想一晚上跑完 2–3 台，两个杠杆：① 输出降到 **1080p（编码提速约 4 倍，瓶颈直接消失）**；② 动态段增加"超过 N 分钟连续动态再二次抽帧"的可选上限；
- 首跑务必用 `ffmpeg -benchmark` 实测你机器上的真实解码/编码 fps 校准，我上面的 8×/70fps 都是保守下限，实际大概率更快。

### 4.5 I/O 与内存

- 双 pass（先分析后编码）会让**总读取 ≈ 2× 输入体积**：1.06TB → 约 2.1TB/夜 → 持续 ~50MB/s，机械盘即可，SSD 更稳。
- 8GB 显存：4K 解码会话显存占用约 100–200MB/路，4–5 路并发 + 2 路编码远在预算内。
- 输出按日落盘后，原始素材保留一段时间再删，QC 通过前不动原件。

---

## 5. 需要你拍板的 4 个决策点

1. **输出分辨率**：4K 保真 vs 1080p（时间/体积省 ~4×）。DailyVlog 用途下 1080p 通常够看，我建议默认 4K、给 1080p 开关。
2. **动态占比 S 的真实值**：取决于你家"有人在动"的时间占比，直接决定输出长度与单夜容量，跑一次分析就得到实测值。
3. **实际码率/编码格式**：请从相机设置读真实码率（4 还是 8Mbps）和编码（H.264/H.265）——若相机是 H.264 4K（16–32Mbps），输入量会大 2–4 倍，预算需重算。
4. **音频**：保留（仅动态段）还是丢弃。家庭监控音频有隐私考量，建议默认丢弃，需要时再开。

---

## 6. 工程落地建议（技术栈与稳健性）

- **技术栈**：Python 3.10+ + ffmpeg 6.x（`--enable-cuda` + nv-codec-headers）+ PyAV/OpenCV + numpy；SQLite 存 Manifest 与任务状态；`multiprocessing` 起 2–4 个分析 worker + 1–2 个编码 worker（用 NVENC 与 QSV 两条独立通道）。
- **断点与恢复**：以分片为原子单位，输出写 `.part` 再改名，SQLite 记 done；中断后下个周末自动续跑。
- **QC 门**：每片校验帧数/时长；抽样 5–10 帧做亮度直方图查黑帧/花屏；每日 concat 后生成**事件时间线索引 + 关键帧缩略图**（HTML），让你不用翻视频就能看当天有几次活动、各在几点。
- **首跑校准清单**：真实码率、NVDEC 解码 fps、NVENC/QSV 编码 fps、夜间噪声底 σ——跑一次 10 分钟样片即可全部拿到，再冻结参数。

---

## 一句话总结

**这套架构的骨架是"分片级两遍流水线"：NVDEC/QSV 硬解做噪声自适应的动静分段，静态段 30 倍抽帧、动态段常速保留，NVENC+QSV 双通道 HEVC 编码，按日 concat 出 DailyVlog；测算表明单夜稳妥处理 1 台相机一周，瓶颈在动态画面的 4K 编码工作量而非分析，想扩到 2–3 台就上 1080p 输出或拆两夜。**

下一步建议：把相机实际码率/编码/是否连续录制这三个参数发我，我可以把第 4 节预算表按你的真实数据重算一遍，或直接帮你搭一个最小可跑的原型（先支持 1 台相机、能出 DailyVlog 即可）。