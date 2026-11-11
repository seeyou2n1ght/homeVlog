# 项目状态

## 2026-11-11 性能验证结论

首次运行基线 2522.08 秒。仅将渲染并发从 2 NV + 1 QSV 调为 1 NV + 1 QSV，整日为 2589.48 秒；上游缩短被尾段增加抵消，已恢复原配置。固定夜间批次的软件解码 + NVENC 对照保持成片逐字节一致，但与 CUDA 分析共同运行的墙钟由 183.297 秒增至 273.250 秒，不推广。尚无实测支持空缓存 30 分钟方案。详细条件及证据见 [BENCHMARK](BENCHMARK.md)。

## 阶段性收尾与目录清理（2026-11-11）

重构与性能验证已阶段性收尾，生产并发保留原配置。旧重构方案、实验脚本、代码快照、性能记录和数据库快照已归档到 `reports/cleanup_20261111/evidence.zip`，ZIP 内保留原项目相对路径。重复实验视频、模型副本和批次缓存已移出项目，隔离在 `C:/Users/seeyo/code/homeVlog-cleanup-20261111/`；当前 `data/`、`output/`、`models/` 和性能日志保留。ZIP 归档不包含媒体；本次采用可恢复移出，隔离目录仍保留原文件，尚未释放其磁盘空间。

`src.stages`、`src.hardware`、`src.algorithms` 的未使用通配符重导出已移除，调用方继续使用明确的子模块导入。历史性能条件和最终验证结论统一见 [BENCHMARK](BENCHMARK.md)。

## 产品范围与架构演进（2026-11-10 架构大瘦身与纯算法闭环）

1. **核心产品定位**:
   HomeVlog 定位为纯命令行、高可靠、确定性的家庭监控素材智能浓缩流水线。
   保留全自动目录扫描、关键帧/VAD自适应预筛选、YOLO多目标精析、单一事实源（Single Source of Truth）确定性时间轴、异构/CPU事务性渲染、以及 Companion 伴生资产（SRT字幕、.meta.json、性能遥测）生成。

2. **退役与清理组件**:
   - 彻底废除人工审核与重渲 Web 工具（`scripts/audit_tool/` 全目录及 37 个 HTTP 路由）；
   - 彻底废除本地临时图片落地与主动学习归档模块（`src/stages/archiver.py`）；
   - 彻底废除冗余的二次视觉索引与感知哈希（`src/core/visual_index.py`）；
   - 彻底废除历史向后兼容门面脚本（`src/*.py` 根目录 20 个兼容 Shim）；
   - 彻底废除 `Monitor` 中高开销的 `typeperf.exe` 轮询与启动计数器探测；
   - 数据库 Schema 统一升级至精简 5 表结构，废除所有 SQLite 触发器及 JSON 冗余列。

3. **核心不变量遵循**:
   继续严格坚守 `AGENTS.md` 核心七大不变量：单次解码分析、内存中间帧、容错降级隔离、确定性时间轴、受控硬件并发、托管 FFmpeg 进程生命周期、事务性断点续跑。

---

## 重构实施进展（2026-11-10 大规模架构重构实施路线图）

| 里程碑 | 目标阶段 | 涵盖功能 | 状态 | 成果与验证 |
|---|---|---|:---:|---|
| **M1** | Phase 1: 关键依赖提炼与前置解耦 | F1, F2, F3 | **DONE** | `reader_process_state` 提炼至 `src/hardware/ffmpeg.py`，`source_identity` 提炼至 `src/core/identity.py`，解除层级倒置 |
| **M2** | Phase 2: 辅助模块与工作台裁剪 | F4, F5, F6, F7, F8 | **DONE** | 裁剪 `scripts/audit_tool/`（37个路由）、`archiver.py`、`feedback.py`、`visual_index.py`，移除 2048 帧视觉描述子循环 |
| **M3** | Phase 3: 数据库极简重构与配置优化 | F9, F10, F11, F12, F13 | **DONE** | `segments` 表 9 字段极简 DDL，删除 `visual_features`，参数化 JOIN 动态时长，修复批次折叠与配置项动态化 |
| **M4** | Phase 4: 监控精简与测试套件收敛 | F14, F15, F16, F17 | **DONE** | 剔除 `typeperf.exe` 轮询与阻塞采样，O(1) 流式 StageStats，删除 3 个过时测试套件，收敛至 31 核心算法回归 |
| **M5** | Phase 5: 架构垫片清理与文档对齐 | F18, F19, F20 | **DONE** | 迁移 `processing_fingerprint` 至 `src/core/identity.py`，删除 `src/*.py` 18个门面，更新 160 处引用，同步四大核心文档 |
| **M6** | Final: 合成端到端流水线与总验收 | F21, F22 | **DONE** | CPU 合成端到端、Ruff、编译和 `--doctor` 通过。最终全库回归 749 通过、6 个硬件专项跳过，用时 121.39 秒。20260320 最终冷跑新编码 142 个批次，用时 2,510.08 秒，较历史冷跑少 58.92 秒。 |

新库保留 `human_reviews` 历史事实表，生产流程使用独立的算法结果。检测到旧 schema 时启动会提示先备份旧库。交付恢复归阶段层，硬件调优和资源清理归运行入口及硬件层。

最终冷跑中，1,200 秒重任务阈值使 4 个中等切片转入 QSV。全部源时间区间覆盖一致，片段分类发生变化，识别质量等价尚未用独立真值验证。耗时、成片和缓存比较口径见 [BENCHMARK](BENCHMARK.md)。
