# AGENTS.md

This file provides guidance to Codex or other coding agents when working in this repository.

## 核心开发哲学

1. **单次解码 (Single-Pass)**: 一次解码，多重消费。严禁为 YOLO 或运动分析单独开启重复的解码进程。
2. **零拷贝 (Zero-IO)**: 候选帧在内存 JPEG 切片压缩池中流转，禁止在中间阶段频繁向磁盘写入零碎临时图片。
3. **容错重于完美**: 在 NAS 环境下，网络抖动与文件损坏是常态。对于失败的切片，记录 Warning 并安全降级跳过，确保最终 Vlog 能够顺利生成。

---

## 硬件调度与并发铁律

- **硬件并发信号量**: 算力并发控制必须通过 src.scheduler 中的 get_nv_semaphore()、get_qsv_semaphore() 与 WorkStealingManager 统一协调，防止 NVENC 超出驱动并发限制（当前生产配置 max_nv_concurrency: 2，8GB 显存安全上限）或 QSV 句柄耗尽崩溃。
- **信号量释放单次原则**: io_sem.release() 必须在 try...finally 的 finally 块中恰好调用一次。禁止在 return 前显式调用 release() 后又在 finally 中重复释放，否则信号量计数膨胀导致并发失控。
- **子进程注册与优雅停机约束**: 所有通过 subprocess.Popen 启动的后台 FFmpeg 进程（包含 Pass 2 批次渲染与 Pass 1.5 管道解码），必须通过 FFmpegProcessRegistry.register() 注册并在 finally 块中 deregister()。严禁产生脱离管控的孤儿进程，确保用户按 Ctrl+C 时由 kill_all() 瞬间释放 GPU 会话。
- **分析管道灰度直通**: 管道解码必须严格维持 -pix_fmt gray 单通道灰度直通，帧尺寸为 w * h。严禁在管道中输出 3 通道数据后再由 CPU 转灰度。

---

## 流水线与时间轴规范

- **Lazy Metadata 强制约束**: 禁止在 src/scanner.py 中增加任何阻塞式的文件读写（如 ffprobe）。所有元数据探测必须在 Analysis 阶段通过 PyAV 懒加载完成，并存入 DB。
- **时间轴闭环**: 修改 src/detector.py 的 analyze() 返回值时，必须确保最后一帧的时间戳严格等于 start_offset + file_duration，杜绝渲染出的 Vlog 出现时间轴空洞或跳秒。
- **哨兵语义与退出判定**: _render_worker 中的 None 哨兵仅作为队列唤醒信号。Worker 退出的充要条件必须是 all_dispatched && heavy.empty() && light.empty()。严禁在 heavy 队列拿到 None 时直接退出而丢弃 light 队列中的纯静态批次。
- **确定性渲染等待与物理对账闭环**: StreamingOrchestrator 严禁仅等待单文件队列派发完毕即提前退出，必须等待 render_finished_event 确保所有渲染批次 100% 物理落盘与对账完成；收尾时强制执行 dispatched_batch_ids 对账（dispatched vs produced | terminal），若有缺失批次立即触发 Error 阻断合并。
- **断点续传与原子批次约束**: 批次渲染必须采用 _batchX.tmp.mp4 临时文件原子写入，经退出码与大小校验后原子替换正式文件；cleanup_temp_artifacts(clean_batches=False) 严禁默认删除有效 _batch*.mp4，保障随时中断随时秒级续跑。
- **静态段渲染快路径**: 纯静态且段长 >= 2 * static_keyframe_interval 的文件，在解码侧以 select 按关键帧抽帧，置于 scale/hwdownload 之前，彻底消灭夜间大段纯静态录像的全帧解码瓶颈。
- **展示时长计划单一来源**: 展示时长计划的单一真相来源为 timeline.compute_display_plans()，渲染滤镜图与 SRT 字幕生成共用该计划；字幕真实墙钟时间映射必须通过 src_offset_at_display() 逆映射还原。
- **YOLO 验证帧索引与动态帧率**: YOLO 验证帧索引计算必须基于解码器动态自适应的真实 effective_fps（0.5~5），严禁使用固定的静态配置值。

---

## 配置纪律与代码分支对齐

- **配置键代码消费闭环**: config/settings.yaml 中的每一个配置键必须有真实业务代码消费。新增配置项必须先完成代码接线，删除功能必须同步清理无用配置键，杜绝僵尸键。
- **档位配置与分支代码严格对齐**: 配置文件中预设的枚举档位（如 analysis_fps_tiers.ultra_long）在代码中必须有明确的分支逻辑承接，严禁存在配置已定义但代码分支遗漏的幽灵配置。

---

## 架构演进与文档管理规范 (ADR & Docs Discipline)

- **架构决策 ADR 原则**: 核心架构模式变更（如灰度直通解码、同构硬件渲染、人机主动学习、流式原子批次渲染等）必须遵循 ADR 规范统一沉淀于 docs/ARCHITECTURE.md 的核心决策章节，完整记录背景、驱动因素、具体决策与推论影响。保持 docs/ 目录单一扁平层级。
- **架构文档单一真相来源**: docs/ARCHITECTURE.md 为系统架构与架构决策的唯一事实来源（Single Source of Truth），代码修改影响数据契约或流向时必须保持 100% 对齐更新；所有生产压测基线与多日指标统一归并于 docs/BENCHMARK.md，严禁在 docs/ 下提交零碎日期的临时分析报告。
- **性能指标对账清晰**: 性能分析器（scripts/analyze_perf.py）与日志输出中必须显式区分多路并发的「Worker 累计工时 (Worker Time)」与实际端到端「壁钟耗时 (Wall-clock Time)」，杜绝多卡并行下的统计认知混淆。

---

## 人机协同与 Active Learning 规范

- **真实反馈数据归档约束**: 审核打标产生的物理帧归档由 src.archiver.FrameArchiver 负责原图抽取，并原子追加写入 manifest.jsonl。若原片丢失或 Seek 异常，必须自动降级提取内存缓存帧并记录 Warning，严禁向上抛出未捕获异常阻塞工作台。
- **重浓缩时间轴人工优先**: 在 src/timeline.py 重建渲染时间轴时，数据库 segments 表中的 human_label 优先级严格高于 algo_status（例如 FP 强制重置为静态，FN 强制重置为动态）。
- **模型微调防退化与防虚警**: 使用 scripts/export_dataset.py 导出机位训练集时，必须包含确认纯静态段（TN）与误报段（FP）作为负样本（输出空 txt 标注），且在 scripts/train_yolo.py 中强制冻结骨干网络（freeze=10），杜绝微调后模型在空旷静态场景产生泛化虚警。

---

## 测试套件与工程整洁规范 (Testing & Workspace Hygiene)

- **测试用例内聚收敛**: 严禁为单个边缘测试函数或修复项新建零散微型测试文件。测试套件必须按核心业务领域（数据库与扫描、渲染与 FFmpeg、硬件调度、审核服务、多模态算法等）高度收敛内聚。
- **共享测试基底集中维护**: 合成测试视频生成 helper 与 Mock 工厂统一维护于 tests/conftest.py，杜绝各测试文件之间重复拷贝定义。
- **未暂存工作区防护与轨迹恢复**: 修改或撤销核心代码文件前必须核对 git diff，防止未提交的有效改动被冲刷；若发生未暂存改动意外丢失，必须基于本地 brain 历史轨迹（transcript_full.jsonl）的完整快照精准还原。

---

## 环境与测试规范

- 统一使用 uv 管理虚拟环境与依赖。
- **CUDA 支持**: pyproject.toml 已配置 PyTorch CUDA 索引，uv sync 会自动安装支持 RTX 显卡的版本。禁止手动更改为普通 CPU 版 torch。
- 提交前全量测试回归：
  ```powershell
  uv run python -m pytest tests/
  ```
- 语法与静态检查：
  ```powershell
  uv run python -m compileall main.py src scripts
  ```
