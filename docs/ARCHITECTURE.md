# 系统架构设计 (Architecture)

HomeVlog 是面向家庭 NAS 与本地主机的长视频智能浓缩系统。系统输入全天或多日的家庭室内监控视频（通常为 4K H.265 编码），自动识别有效动态与声音事件，压缩大段时间的静止画面，输出高保真、平滑过渡的家庭 DailyVlog。

---

## 1. 全局数据流拓扑

系统遵循 **单次解码 (Single-Pass)** 与 **零零碎磁盘 IO (Zero-IO)** 原则，数据流分为三级流式流水线：

`	ext
[NAS / 本地监控录像] (4K H.265 / MP4)
         │
         ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 1. 预筛选 (Pass 1: Prescreen)                             │
 │    - FFmpeg 关键帧跳跃读取 (I 帧直出)                      │
 │    - 自适应动态阈值 + 4x4 网格空间集中度算子              │
 │    - 剔除纯静态时段 (漫反射光影/车灯抑制)                  │
 └─────────────────────────┬─────────────────────────────────┘
                           │ 候选可疑切片 (SUSPICIOUS)
                           ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 2. 精细分析 (Pass 1.5: Analysis)                          │
 │    - FFmpeg 管道解码出 80x45 单通道灰度流                  │
 │    - EMA 选择性滑动背景更新 + 8x8 连通域抗噪滤波          │
 │    - 短时音频 RMS 能量包络与一阶自相关 (AudioEnergyVAD)    │
 │    - 内存 JPEG 切片压缩池 (Zero-IO 候选帧流转)            │
 └─────────────────────────┬─────────────────────────────────┘
                           │ 触发运动判定
                           ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 3. 语义验证 (Pass 1.8: YOLO Verification)                 │
 │    - 内存中动态组装 Batch，送入 Tensor Core YOLOv11       │
 │    - 人体/宠物目标置信度核验，彻底过滤假动作              │
 └─────────────────────────┬─────────────────────────────────┘
                           │ 结构化切片与时间轴映射
                           ▼
 ┌───────────────────────────────────────────────────────────┐
 │ 4. 硬件渲染 (Pass 2: Render)                              │
 │    - 动态段 1x 原速原声 (afade 音频平滑过渡)              │
 │    - 静态段 30x~60x 快速缩放过渡 (C¹ 连续 PTS 曲线)       │
 │    - 双路 NVENC 硬件并发编码，原子切片输出与无损合并      │
 └─────────────────────────┬─────────────────────────────────┘
                           │
                           ▼
              [最终 DailyVlog 视频成片]
`

---

## 2. 异构硬件调度与并发控制 (src/scheduler.py)

系统设计运行于兼具 Intel 核显（QSV）与 NVIDIA 独显（CUDA/NVENC）的混合异构平台（如 i5-12600K + RTX 3060Ti）。

### 2.1 硬件分工与信号量限制
- **Intel UHD 770 (QSV)**: 专职负责 Pass 1 粗筛与 Pass 1.5 灰度管道解码，充分利用双 Gen12 VDBox 的高并发吞吐。
- **NVIDIA RTX 3060Ti (NVENC/CUDA)**: 专职负责 YOLOv11 批推理加速与 Pass 2 最终成片的 NVENC 视频编码。
- **硬件并发信号量**:
  - 
v_sem: 限制 NVENC 并发编码上限（当前基线为 2，防止显存溢出与驱动层拒绝会话）。
  - qsv_sem: 限制 QSV 解码并发句柄数（默认 4~8），防止驱动句柄耗尽。
  - **释放单次原则**: 所有信号量获取后必须在 	ry...finally 的 inally 块中且仅释放一次，禁止在代码路径分支中多重释放。

### 2.2 工作窃取与动态让步机制 (WorkStealingManager)
1. **动态租借**: 当 QSV 解码队列积压达到高水位时，调度器动态向 CUDA 申请空闲 NVDEC 槽位协助解码。
2. **渲染抢占与让步 (RENDER_PREEMPTION_YIELD)**: 当批次渲染阶段启动需要使用 NVENC/CUDA 算力时，调度器发出让步信号，解码端立即归还借调的 NV 槽位，保障渲染流水线拥有绝对优先算力，杜绝硬件争用死锁。

---


### 2.3 批次调度与收尾物理对账
- **Heavy / Light 双队列**: 批次按运动密集度分流进入 heavy_queue（含动态，优先 NVENC 压制）与 light_queue（纯静态，快速压制）。
- **哨兵安全语义**: _render_worker 中投递的 None 哨兵仅作为唤醒信号；Worker 退出的充要条件为 ll_dispatched && heavy.empty() && light.empty()，防止 heavy 抽空时遗留 light 静态批次。
- **物理收尾对账**: 流水线退出前强制校验 dispatched_batch_ids 与实际落盘批次（produced | terminal）的差集，确保零静默丢批。
- **静态段渲染快路径**: 纯静态且时长满足阈值的切片，在 FFmpeg 滤镜链中使用 select 按关键帧抽帧，置于 scale/hwdownload 之前，消灭纯静态段全帧硬解的显存回传瓶颈。
- **展示时长计划与墙钟字幕单一来源**: 	imeline.compute_display_plans() 输出统一的展示计划，Filtergraph 滤镜图与 SRT 字幕共用该数据结构；字幕时间戳通过 src_offset_at_display() 按变速曲线逆映射还原真实墙钟时间。

## 3. 动静识别算法体系 (src/filters.py, src/prescreen.py)

### 3.1 预筛选空间集中度算子 (_calc_spatial_concentration)
针对家庭摄像头常见的全画幅光影干扰（如早晚太阳漫反射、窗帘大面积受风微动、夜间车灯扫过）：
- 将关键帧差分图划分为 4x4 空间网格（共 16 个单元）。
- 计算局部最高网格能量与全图平均能量的比值：
  Concentration = max(GridEnergy) / (MeanEnergy + epsilon)
- **判定逻辑**：当全图变化能量超过阈值，但 Concentration < 1.35（能量均匀弥散在整个画幅）时，判定为环境光变化，予以直接压制；仅当 Concentration >= 1.35 时才判定为有局部主体移动（人体/宠物入画）。

### 3.2 自适应 EMA 背景更新与 8x8 连通域网格
- **EMA 背景建模**: 双差分显著图融合，前景区域低速吸收，背景区域高速更新，兼顾快速动态捕获与人体静坐微动保留。
- **8x8 连通域滤波 (SpatialGridMotionFilter)**: 动态追踪 64 个网格单元底噪，8-邻域连通分量过滤孤立红外夜视噪点，聚类放大连续肢体动作。
- **音频 VAD 唤醒 (AudioEnergyVAD)**: 在短时 50ms 窗口计算音频 RMS 能量与一阶自相关，家庭对话、婴儿哭声等声音事件可直接唤醒 1x 原速保留。

---

## 4. 人机协同与 Active Learning 闭环

`	ext
[日常流水线运行] ──> [data/vlog.db (segments 表)]
                            │
                            ▼
               [Web 审核工作台 (scripts/audit_tool)]
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
       [疑难样本主动挖掘]             [人工打标 TP/FP/FN/TN]
     (光影假阳 / 微动假阴)                  │
                                           ├─> [物理原图归档 data/archives/]
                                           ├─> [秒级时间轴修正重浓缩]
                                           └─> [专属机位数据集导出]
                                                           │
                                                           ▼
                                            [YOLO11 骨干冻结本地微调]
                                            (scripts/train_yolo.py)
`

1. **疑难样本排查**：Web 工作台自动检索判定为 DYNAMIC 但 YOLO 置信度为 0（疑似光影误报），或判定为 STATIC 但能量处于临界区（疑似微动漏判）的切片置顶。
2. **物理帧归档 (src/archiver.py)**：标记切片时，自动从原始高清视频抽取发生变动瞬间的原图与差分图，保存至 data/archives/{camera}/{date}/，并原子追加更新 manifest.jsonl 索引。
3. **秒级重浓缩 (scripts/audit_tool/api)**：人工修正打标（如将误报段标记为 FALSE_ALARM）后，直接更新 DB 中的时间轴标记。调用重浓缩接口可跳过 Pass 1 与 Pass 1.5 解码，仅重走 Pass 2 渲染，数十秒内输出修正后的新成片。
4. **模型微调闭环 (scripts/train_yolo.py)**：基于归档的难样本自动生成 8:2 训练验证集与 YOLO 格式标注（包括负样本空标注），冻结主干网络（reeze=10）进行轻量微调，生成机位专属权重，实现识别准确率自我演进。

---

## 5. 数据库结构设计与断点续传

系统使用 SQLite WAL 模式管理核心状态 (data/vlog.db)，主要包含三张业务表：

### 5.1 	asks 表
记录素材扫描与文件级处理状态：
- ile_path: 原始素材绝对物理路径（主键）。
- camera_id: 摄像头 MAC 地址（如 B888805AA3CD）。
- date: 素材录制日期（YYYYMMDD）。
- duration: 视频总时长（秒）。
- status: PENDING -> ANALYZING -> ANALYZED -> RENDERING -> COMPLETED。

### 5.2 segments 表
记录切片级动静态判定与人工审核标注：
- id: 自增主键。
- ile_path: 关联的素材路径。
- start_time / end_time: 切片在文件内的相对起止时间戳。
- lgo_status: 算法原始判定（DYNAMIC 或 STATIC）。
- human_label: 人工审核标注（TP, FP, FN, TN，未审核时为 NULL）。
- max_energy: 动作能量峰值。
- yolo_max_conf: YOLO 目标检测最高置信度。
- rchived_frame_path: 关联归档帧的高清图片相对路径。

### 5.3 perf_records 表
记录每阶段（Prescreen、Analysis、Render）的硬件耗时、CPU/GPU 占用、吞吐倍速与丢帧指标，供基准测试与性能诊断。

---

## 6. 进程与容灾规范

1. **子进程托管注册表 (FFmpegProcessRegistry)**：所有通过 subprocess.Popen 派生的后台 FFmpeg 进程必须显式注册。收到终端中断（SIGINT/SIGTERM）时，统一执行 kill_all() 强力释放硬件解码上下文与 NVENC 会话，杜绝产生孤儿进程。
2. **原子批次替换**：批次渲染文件先写入 _batchX.tmp.mp4，校验退出码与非空尺寸后原子重命名为 _batchX.mp4；任务意外中断重启时，校验成功的历史批次直接复用，保障随时中断随时秒级续跑。
