# HomeVlog 测试策略与质量验证规范 (TESTING)

本文档为 HomeVlog 系统的**核心测试策略、验证命令、质量验收准则与测试套件基线规范**。

---

## 1. 测试体系与分层策略

HomeVlog 测试体系旨在保障长时间音视频批处理流水线的高容错、确定性与高并发安全，分为以下四层：

```text
┌─────────────────────────────────────────────────────────┐
│               1. 语法编译与静态检查 (CI 必过)             │
│        uv run python -m compileall main.py src scripts   │
├─────────────────────────────────────────────────────────┤
│               2. 自动化纯软单元与集成测试套件              │
│                 uv run python -m pytest tests/          │
├─────────────────────────────────────────────────────────┤
│            3. 本地真实硬件加速冒烟验收 (隔离环境)          │
│       $env:HOMEVLOG_HARDWARE_TESTS = '1'; pytest ...     │
├─────────────────────────────────────────────────────────┤
│            4. 全链路真实素材回测与生产遥测 (E2E)          │
│          uv run python main.py --date 20260320 ...      │
└─────────────────────────────────────────────────────────┘
```

### 1.1 隔离与 Mock 原则
- **默认无硬件依赖**：普通的 CI 与开发者单元测试默认严禁依赖物理 NVIDIA GPU（NVDEC/NVENC/CUDA）或 Intel 核显（QSV），禁止依赖真实 NAS 网络文件系统。
- **环境隔离守护**：所有依赖物理硬件或本地视频的测试必须标记 `@pytest.mark.skipif(not os.getenv("HOMEVLOG_HARDWARE_TESTS"), ...)`，并在无硬件时安全跳过。
- **子进程与资源清理**：测试用例创建的 FFmpeg 管道、临时 SQLite 数据库或临时目录，必须在 `teardown` 中严格释放与清理，严禁泄漏孤儿进程或锁文件。

---

## 2. 标准验证命令清单

### 2.1 语法编译与导入检查
在进行任何提交前，必须运行语法编译检查，确认无任何语法错误或损坏的 import：
```powershell
uv run python -m compileall main.py src scripts
```

### 2.2 核心自动化测试套件
运行所有单元测试与流式编排集成测试：
```powershell
uv run python -m pytest tests/
```

### 2.3 单模块聚焦验证
按功能域独立验证特定子系统：
```powershell
# 验证数据库与扫描持久化契约
uv run python -m pytest tests/test_database_and_scanner.py

# 验证时间轴构建与映射
uv run python -m pytest tests/test_timeline_and_ramping.py

# 验证异构硬件调度与工作窃取
uv run python -m pytest tests/test_scheduler_and_hardware.py

# 验证多模态算法 (EMA / VAD / YOLO)
uv run python -m pytest tests/test_motion_and_vad.py tests/test_accuracy_and_merging.py

# 验证流式流水线编排与断点续传
uv run python -m pytest tests/test_pipeline_streaming.py tests/test_resumption_and_shutdown.py

# 验证 Web 审核平台与数据接口
uv run python -m pytest tests/test_audit_service.py
```

启动诊断回归位于 `test_database_and_scanner.py` 与 `test_ui_logging.py`：覆盖受控目录枚举阻塞期间的可见心跳、报告线程回收、扫描部分提交后失败及增量恢复、无冻结时避免属性请求、DEBUG Handler 重配置和 Doctor 磁盘失败/退出码。受控阻塞不证明实际 NAS 系统调用可超时或取消；真实启动验证应使用隔离数据库并在分析/渲染边界停止，避免改写生产交付。

配置兼容回归位于 `test_scheduler_and_hardware.py`，覆盖核心重载后旧入口读值、显式覆盖、混合入口并发和失败缓存保护。展示参数回归位于 `test_timeline_and_ramping.py`，以六种状态、自定义倍率、20/24fps 及变速开关核对渲染预算、SRT 终点和元数据高光同钟；渲染执行和媒体探测为受控替身，真实软件媒体回归仍由现有渲染套件覆盖。`test_pipeline_streaming.py` 验证成功路径只探测一次时长、保留未舍入测量，以及资产提交失败不发布测量。

### 2.4 本机真实硬件加速冒烟验收
在具备真实 Intel QSV 与 NVIDIA GPU 的物理机上显式开启冒烟验证：
```powershell
$env:HOMEVLOG_HARDWARE_TESTS = '1'
uv run python -m pytest tests/test_hardware_smoke.py
```

---

## 3. 算法识别与切片质量验收基线 (Quality Baselines)

HomeVlog 基于多模态融合（空间集中度预筛选 + EMA 运动差分 + AudioEnergyVAD + YOLO 目标验证）对长视频进行动静切片，其生产验收标准如下：

| 评估维度 | 衡量指标 | 目标基线 | 实现保障与测试验证 |
| :--- | :--- | :--- | :--- |
| **切片拓扑完整度** | 连续同状态未合并碎片对数 | **0 对 (0.0%)** | YOLO 验证完成后强制触发 `_merge_same_state()` 拓扑熔断；末帧闭合检查。 |
| **置信度数据闭环** | YOLO 动态段零置信度占比 | **0.0%** | 推理得分必须实时双向回填至 `Segment.avg_confidence`。 |
| **误报抑制 (FP)** | 疑似光影刚性动态段占比 | **< 1.0%** | 空间网格集中度门限 + 漫反射光照抑制 + YOLO 目标存在性校验。 |
| **漏检抑制 (FN)** | 婴儿房微光微动漏检率 | **趋近 0%** | 暗光自适应阈值动态下调 (0.4x) + 动态段前后扩展缓冲 (pre/post-roll)。 |
| **极短抖动控制** | $< 2.0\text{s}$ 破碎切片数 | **0 个** | 拓扑平滑窗口 `apply_smoothing: true` 自动吸收。 |
| **音画同步偏差** | 分段帧数、PCM 样本数与最终音视频 PTS | 分段共享整数帧计划；编码后按帧时间基验收 | 20/24fps、六种状态的多段真实媒体回归验证帧数、PTS 连续性和样本数；AAC 编码延迟不能按 2ms 绝对误差要求。 |

以上 FP/FN 数字是验收目标，不是当前单元测试测得的真实召回率。人工合成能量序列的正反例只能证明分段规则；必须另以独立标注的真实人物、微动和声音样本复核。短片不得仅因小于 2 秒就被删除，显著活动的前后保护与可恢复性优先。

默认真实软件媒体测试只要求 FFmpeg/PyAV；硬件测试使用 HOMEVLOG_HARDWARE_TESTS 显式开启。旧 testsample 文件名检查既未解码也未运行流水线，且重复 scanner 解析用例，已移除；perf schema 用例并入性能子系统，不能用这些单元检查充当完整 E2E。

---

## 4. 自动化测试套件执行基线 (Test Baseline)

- **当前基线（2026-10-08 中断传播修复）**：常规全量 **443 passed、6 skipped，63.28s**；随后新增SIGINT用例的停止套件 **16 passed，4.10s**；独立开启真实硬件冒烟 **6 passed，10.16s**。另有真实单批音视频完整解码、软件媒体交付与亚帧回归；测试数量不代表识别质量。
- **测试通过率要求**: **100% Passed**（除特定依赖物理硬件而显式跳过的 Case 外，不得有任何 Error 或 Failure）
- **核心覆盖领域**:
  - `core/`: 数据库 WAL 并发、单行索引、状态机转换、配置隔离与参数校验
  - `hardware/`: 双端工作窃取队列、GA104 物理配额、QSV/NVENC 信号量 lifecycle、FFmpegRegistry 优雅停机
  - `algorithms/`: 80x45 预筛选差分、EMA 滑动平均更新、AudioEnergyVAD 音频切片、YOLO 类别置信度过滤
  - `stages/`: 流式编排、断点恢复续传、原子临时批次生成与提交、丢失文件 NAS 自愈跳过
  - `ui/`: Web 审核 REST API、时间戳双轨计算、代表帧物理抽取与回退兜底、SRT 字幕渲染

### 异构合成回归

`test_real_mixed_encoder_concat_preserves_every_frame` 在实际 NVENC/QSV 上编码两个片段，比较独立解码与拼接解码的全部 80 帧哈希。常规测试还覆盖分析任务成功状态、预取所有权消费与逐文件时间轴一致性。硬件冒烟不证明整日画质或人物召回率。静态检查命令：`uv run python -m ruff check main.py src scripts`。

### 交付审查回归门禁

常规套件必须覆盖：无音轨和时长不符拒绝、接缝内部 PTS 缺口、尾部解码、limited-range YUV 规范灰度、实际媒体时长验证标记、空分析原子替换、所有状态 EOF 音视频同钟、伴随资产提交、审核任务级取消、API 同源令牌及硬件并发配置约束。真实 FFmpeg 回归包括：8 秒源按 12 秒动态计划输出 240 帧视频和 576,000 个 48kHz 音频采样；最终静态切片在物理视频 EOF 后仍按 8.1 秒计划输出 162 帧与 388,800 个音频采样。

最终拼接回归同时覆盖：独立容器打开异常后的全接缝串行回退、真实接缝失败不得回退放行、回退前后文件变化拒绝、首份失败证据保留与容量限制，以及失败/异常/取消均生成 `final_concat`。受控打开异常只验证恢复分支，不证明生产 EOF 根因；短合成媒体不替代大成片检查。

停止/恢复套件覆盖Windows控制台终止码的有符号与无符号形式，并使用含流头/信号信息的非空stderr核对日志分类和异常传播；普通退出码1的编码错误必须保留ERROR。明确中断不得启动EDL备用路径或NV重试；NV/QSV/异构模式须传播到主线程，保留有效批次且不提交完成。审核任务须进入CANCELLED并保留旧交付和批次。该受控回归不确定实际信号发送者，不替代真实控制台骤停验收。

交付提交回归使用真实临时 SQLite 和资产文件，按一批/多批、清理开/关注入资产返回失败、资产异常、部分替换失败、manifest 异常、完成写入失败和取消。必须核对旧视频/字幕/元数据/manifest、批次及清单、无关文件和遥测；审核重渲染覆盖相同资产边界。真实软件媒体回归运行一批/两批交付，检查全部视频帧、音视频流时长、SRT、高光和 manifest 身份；元数据容器时长可以包含 AAC 填充，不能与展示计划要求逐位相等。未模拟掉电或回滚磁盘故障的实际硬件行为。

审核回归在撤销后验证持久标签和重建时间轴，覆盖重分析与部分区间撤销、事务失败回滚。亚帧媒体回归从数据库行构建时间轴，检查源边界保留后再验证输出整数帧/音频样本。CLI 回归以多日双机位的临时数据库核对执行与重置范围、dry-run 不清空/清理、已交付重析的 --force-render 边界，以及未实现参数被拒绝。

整日交付仍需另行验证全片逐帧解码、目标播放器、字幕/元数据同钟、人工人物事件 holdout，以及同数据库、timeline、编码与硬件的 wall-clock A/B。

### 分析解码路径等价性

`uv run python scripts/compare_analysis_decoders.py SOURCE --report scratch/decoder_comparison.json` 以 QSV/CUDA/CUDA/QSV 顺序调用生产解码路径，比较完整灰度帧和固定时段条件下的运动标签。同设备重放用于排除自身漂移；帧数、形状、像素或标签不等价返回 2，未完整解码为错误。支持多个源文件和显式 `--reference`/`--candidate`。报告不含音频、YOLO 或人工 holdout，不能用来量化召回率，也不能把顺序单文件耗时当作整日提速。
