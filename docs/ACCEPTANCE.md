# HomeVlog 核心范围裁剪验收报告

日期：2026-10-21。结论：已批准的裁剪完成，工程回归通过；当前没有随源码交付的生产视频包。

## 执行范围

保留扫描、预筛/分析、完整成片、日常版、人工事件复核/重渲、视觉相似检索和必要诊断。未改变识别算法、硬件调度、共享时间轴、批次事务或完成记录语义。

本轮删除27个文件；运行代码与界面由30,102行降为21,936行，净减8,166行（不含main.py和文档）；测试代码由16,132行降为13,760行，净减2,372行。统计基于本轮开工源码快照，不把此前已有删除计入本轮。

| 已退役 | 实际范围 |
| --- | --- |
| 旧独立工具 | audit_multiday_backtest、reprocess_accuracy、verify_accuracy、tune_thresholds、repair_audio_metadata、benchmark、profile_decode_cost、compare_analysis_decoders、measure_vlog_quality |
| 旧原型 | prototype.html/js/css及对应四个静态路由 |
| 训练/目标标注 | train_yolo、export_dataset、目标框/智能体目标建议画布、服务和API |
| 调参/基准管理 | replay_benchmark、prepare_review_benchmark、build_gold_baseline、quarantine_generated_reviews、core/tuner、core/benchmark及页面/API |
| 历史实验适配 | 固定Rxx案例、实验附件读取、scratch参数和观察库存生成 |
| 专用测试/数据 | 训练/基准套件、旧100条标签JSON及副本、已退役脚本和功能用例；人工审核优先回归迁入review套件 |

保留analyze_perf.py、benchmark_day.py、inspect_detection.py、sample_accuracy.py、index_review_visuals.py。root src兼容入口继续服务现有调用方。

另清理37份没有对应源码的孤立字节码，937,221字节，先保存并验证orphan_bytecode.zip后删除；未清空scratch/temp或扩大删除生产数据。两份配置备份与三份模型权重保持。

## 关键实现

人工审核库存直接统计显式标签，不再经调参模块；Precision/Recall/F1保持null。配置归因保留有效配置、场景/备注/观看依据和共享计划对比，移除配置写入入口。核心config缓存、校验和重新加载保留。

日常选择policy为whole-delivery-segments-round-robin-v2，仅使用父映射、当前分析和持久human_reviews；人工阳性优先、人工阴性相交进入复核、混合裁决和预算冲突保持。新capture/manager接口没有scratch参数，未迁移或改写生产数据库。

replay_journal仍承担原子JSON、哈希、租约和交付恢复。replay_evidence保留内容身份、环境/代码指纹及展示计量；无消费者的调参重放提供者已删除。benchmark_day指纹改为src、main.py和自身，旧负载需重新冻结。

结构变化记录为ADR0076。README、工作台说明、ARCHITECTURE、PROGRESS、TESTING和BENCHMARK已同步；旧ADR保留历史并标出局部替代，当前文档不再发布已删除的交付路径。

## 验证

| 检查 | 结果与证据（相对reports/core_prune_20261021） |
| --- | --- |
| 一次全库pytest | **856 passed、6 skipped、0 failures/errors**，862总项；命令156.31秒，JUnit 156.244秒。regression.xml/log |
| 受影响专项 | 187通过；新增日常capture和退役HTTP入口控制2通过。targeted.xml、changed_paths.log；不与全库累计 |
| 仓库配置Ruff | 通过；仅对已改文件额外清理无用导入 |
| compileall main.py src scripts | 通过 |
| main.py、工作台实际CLI --help | 均成功，main_help.txt、workbench_help.txt |
| 本地导入/文档引用 | 无缺失的本地模块导入；当前Markdown文件链接无失效路径 |
| git diff --check | 通过；既有LF/CRLF转换提示不作为失败 |
| 浏览器真实路径 | 隔离SQLite和2秒合成媒体；队列/预览、只保留六项功能的工具菜单、配置归因面板正常，控制台错误0。ui/core_menu.png、browser_errors.json |
| 日常capture | 真实隔离DB读取；人工阳性/阴性改变选择和快照、父帧守恒、重复捕获确定性均通过 |
| 退役HTTP | 旧原型、目标、基准及调参GET/POST入口经鉴权后均404 |

六项跳过均为未开启HOMEVLOG_HARDWARE_TESTS的真实硬件控制。此次不追加GPU、NAS原片、整日性能、模型训练或生产重渲。
关闭隔离浏览器后服务日志记录一次客户端中断连接10053；浏览器控制台无错误，隔离服务已退出，不把客户端关闭记录改写成新的产品通过证据。

## 状态保护与交付边界

开工时data/vlog.db不存在，scratch和output为空；原6acdd日常包、冻结父片和实验收据已由用户移除。旧984/6、历史媒体/浏览器验收及旧恢复映射不继承当前实现。本轮没有重新生成生产日常包或补做真实全日验收。

开工已有7份配置/写者锁/模型文件的大小、mtime和SHA均保持；未创建生产DB。main.py、pipeline、硬件实现及其他任务外既有源码SHA保持，保留此前未提交工作。只在reports目录使用隔离DB和合成媒体。

识别准确率、全片感知/动作质量、声音语义、默认新分析整日净提速、真实SMB驱动硬阻塞、物理断电和非Windows后代停止仍未认证。历史性能表已明确原证据缺失，不作为当前性能证书。

## 复核与恢复

修改前完整源码快照：reports/core_prune_20261021/before.zip，逐文件SHA为before_sha.json。清单、差异、状态保护、检查结果和截图同目录保存。需要撤回时按清单提取指定文件并核对SHA，不对整个工作区做reset或覆盖后续改动。

所有修改仍位于当前未提交工作区，没有提交、打标签、推送或发布。当前验收绑定当前源码SHA清单implementation_manifest.json；归档和最终收据见verification.json与handoff_evidence.zip。
