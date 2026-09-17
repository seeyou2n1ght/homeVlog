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
| **音画同步偏差** | 最终成片音视频 PTS 漂移 | **< 2.0 ms** | 统一 C1 平滑速率曲线映射与流式时间对齐。 |

---

## 4. 自动化测试套件执行基线 (Test Baseline)

- **测试用例总数**: 243 项通过，7 项硬件/环境条件跳过（2026-09-16）；另行开启硬件套件 6 项通过
- **常规软测试耗时**: 31.52 秒（本轮 Windows 开发环境）
- **测试通过率要求**: **100% Passed**（除特定依赖物理硬件而显式跳过的 Case 外，不得有任何 Error 或 Failure）
- **核心覆盖领域**:
  - `core/`: 数据库 WAL 并发、单行索引、状态机转换、配置隔离与参数校验
  - `hardware/`: 双端工作窃取队列、GA104 物理配额、QSV/NVENC 信号量 lifecycle、FFmpegRegistry 优雅停机
  - `algorithms/`: 80x45 预筛选差分、EMA 滑动平均更新、AudioEnergyVAD 音频切片、YOLO 类别置信度过滤
  - `stages/`: 流式编排、断点恢复续传、原子临时批次生成与提交、丢失文件 NAS 自愈跳过
  - `ui/`: Web 审核 REST API、时间戳双轨计算、代表帧物理抽取与回退兜底、SRT 字幕渲染

### 异构合成回归

`test_real_mixed_encoder_concat_preserves_every_frame` 在实际 NVENC/QSV 上编码两个片段，比较独立解码与拼接解码的全部 80 帧哈希。常规测试还覆盖分析任务成功状态、预取所有权消费与逐文件时间轴一致性。硬件冒烟不证明整日画质或人物召回率。静态检查命令：`uv run python -m ruff check main.py src scripts`。
