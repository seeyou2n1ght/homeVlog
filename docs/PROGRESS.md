# 项目进度与当前状态

更新：2026-10-08。历史性能数字与产物复核见 [BENCHMARK.md](BENCHMARK.md)，架构选择见 [DECISIONS.md](DECISIONS.md)，验证命令与边界见 [TESTING.md](TESTING.md)。

## 当前目标

保持源时间轴正确、人工审核优先和交付可恢复；使实际返回值、持久状态、CLI 帮助、文档与测试一致。吞吐优化必须同时通过识别与整日验收。

## 当前实现

- 三路异构渲染：两路 NVENC、一路 QSV；生产配置7个分析worker。单文件批次使用SJF、多文件批次保序，不更改现有硬件预算。
- 已交付日期按持久完成记录跳过，离线备份仍须保留data/vlog.db。显式force-render重做，未完成批次按指纹和媒体校验恢复。
- 只跳过已确认的零字节普通文件，元数据记录缺失时段。非空损坏、不可访问或未完成分析仍阻止交付。
- YOLO失败/不可用/证据不足仍保留常速待审；复用受限候选缓存，不新增独立解码。QSV/CUDA灰度路径差异仍是识别验收限制。
- 扫描开始/结束和操作心跳可见；扫描不完整阻止本轮处理。底层NAS系统调用仍不保证可超时/取消。
- 审核撤销同步更新持久区间，区间外标签及归档图片保留；短源区间不扩张，输出整数帧由共享展示计划决定（ADR0035）。
- 两交付入口先生成候选资产，再提交manifest/成功状态；正常失败或取消恢复上一份资产，批次仅在完整成功后按配置清理。骤停/回滚失败遗留备份不自动清除（ADR0036）。
- CLI预览、重析和处理共用日期/机位范围；dry-run可扫描入库，但不重置分析或清理批次。已交付重析要求force-render；stage只提供all/analyze。仅分析失败不再报告成功。
- 明确FFmpeg外部中断向主线程传播取消，不再进入EDL备用渲染/NV重试；有效批次保留，主流水线取消以130退出，审核任务标记CANCELLED（ADR0037）。
- 人工标注三位一体闭环 (Triad Human-in-the-Loop Architecture, ADR 0038)：
  1. 基准测试集：实现 `src.core.benchmark`，支持版本化快照归档与增量合并；`tests/test_accuracy_baseline.py` 升级场景分层（`ADULT_MOTION` 召回率、`EMPTY_ROOM` 纯静止过滤、`LIGHT_SHADOW` 误报抑制）质量门禁；
  2. 重渲染驱动闭环：时间轴严格遵循 `human_label` 一票否决权，工作台新增流式视频播放端点 (`/api/video_stream`，HTTP 206 Partial Content) 与重渲染后即时成片自适应预览视窗；
  3. 调优仿真与配置落地：实现 `src.core.tuner.TuningSimulator` 离线推演候选参数对 F1-Score 与场景误报/漏报的净收益；提供结构化 Diff、自动时间戳备份与原子安全写回/回滚接口与工作台面板；
  4. 工作台人机工效与多线程并发升级：`scripts/audit_tool/app.py` 升级为 `ThreadingHTTPServer`，消除微视频流/抽帧与 API 互斥；重构四键模式为原位内嵌无缝平移（In-place Toggle），彻底消除上方微缩帧视窗遮挡碰撞（保持 20px 安全净距）；增加空状态占位与破损图隐蔽防护；通过 Chrome CDP 原生无头浏览器实机测试验证（Zero-Scroll 100vh 单屏、盲打心流、Esc 模态框交互闭环）。

## 当前验证

- 2026-10-11 全面工程审计与健壮性加固 (Auditing & Hardening Delivery)：
  1. 不可逆损坏原片受控跳过与对账放行：新增 `--skip-corrupt` 命令行参数与 `recovery.skip_corrupt` 配置，支持在人工确认或策略授权下将 0 帧/致命损坏废片标记为 `SKIPPED_EXCLUDED` 或 `SKIPPED_CORRUPT`，在 companion manifest 中如实记录缺失区间，解除坏片对全日交付的死锁阻断；工作台新增 `/api/exclude_file` API；
  2. 截断视频时空元数据精准同步：在 `set_file_metadata` 校准真实物理时长时，联动更新 `file_end_time = file_start_time + file_duration`，彻底消除时间轴与外部分析元数据跨度脱节；
  3. 双轨解码器流判定策略收敛：抽取不可变纯函数 `evaluate_stream_completion`，统一 `_decode_file_pipe` 与 `_decode_file_pyav` 的流完成度与截断 EOF 判定策略，杜绝双轨不同步维护陷阱；
  4. 初筛模式收敛与性能优化：代码内核默认统一切至 `keyframes`（单进程多帧快速抽取），降低进程启停开销与越界异常；
  5. 人工标注切片平滑豁免权（Immunity Flag）：`Segment` 数据结构显式支持 `is_human_reviewed`，在 `_filter_short` 时序滤波中获得最高豁免保护，严防微小人工真值动作被算法意外吞噬；
  6. Web 工作台 NAS 远端流式体验优化：`/api/video_stream` 分块读取缓冲区从 256KB 扩充至 1MB，降低高延迟 SMB 网络的 I/O 往返频次。
- 2026-10-11 P3 配置治理与场景预设闭环收敛 (Hierarchical Configuration Governance, ADR 0039)：
  1. 场景 Profile 矩阵内核：实现 `src.core.profiles`，基于 100 样本黄金基准集与真实生产双机位物理特征固化 `living_room`（客厅防光影漫射+长时驻留4x/16x）、`baby_room`（婴儿房低门限1.8+红外微动保全+啼哭高敏VAD）、`outdoor_corridor`（室外走廊高抗噪3.5+抗车灯树影）；
  2. 宏观主旋钮与反比动态缩放：定义 `motion_sensitivity`（0.5~2.0），以反比函数 $\text{Thresh}_{\text{eff}} = \text{Thresh}_{\text{base}} / \text{sensitivity}$ 动态缩放核心门限；支持机位级绑定（`cameras.<mac>.profile = "baby_room"`）与独立灵敏度；
  3. 生产配置文件瘦身与零改动兼容：重构 `config/settings.yaml`，剥离/注释平铺的 60 个魔法数字；`load_config` 自动无缝补全 Profile 默认值，且用户显式声明的专家参数拥有最高优先权；
  4. What-If 离线调优仿真扩展：`TuningSimulator` 扩展 `simulate_profile`，实测 `baby_room` 预设可显著压制 9 处微动漏报（Recall 提升 +16.36%，综合 F1 提升 +11.0%）；
  5. 搭建端到端配置与预设验收测试套件 `tests/test_profiles_and_config.py`，覆盖预设契约、灵敏度缩放、机位继承与异常拦截。
- 全量 485 项测试通过（479 passed, 6 skipped，耗时 74s，0 failure）；Ruff 代码风格检查与 compileall 编译检查 100% 通过。

- 本轮在隔离SQLite和临时资产上验证审核撤销、范围筛选、失败传播、资产回滚和批次保留；真实软件媒体覆盖源边界、整数帧、音频时钟、字幕及manifest。准确测试数量见TESTING当前基线。
- 2026-10-08先前的拼接恢复记录：20260715 cam1、20260718 cam1复用162/151批恢复交付，完整音视频软件解码及全部接缝/尾窗通过；原始并行打开EOF根因未复现。该记录是本轮之前的生产证据，详见BENCHMARK。
- 本轮已验证生产截断原片在初筛与精析阶段的自适应校准与容错。

## 阻塞与未验证边界

- 20260721 cam1中断任务未在本轮恢复。
- 原始并行容器打开EOF及凌晨NAS扫描停滞的底层根因尚未复现。
- 独立人物/微动/声音 100 样本黄金真值基线已构建并纳入门禁；当前整日同条件A/B和目标播放器验收仍待后续生产长测；历史“零漏检”及单元正反例不是删除原片依据。
- 跨进程任务认领、真实NAS掉线、进程骤停和断电期间的资产包一致性未新增验收；当前仅保证逐文件替换和可观察异常的回滚。

## 下一步

1. 恢复不完整源文件并显式续跑未交付任务。
2. 冻结真实输入、分析快照、人工标签和设备分配，验证解码路径差异与识别holdout。
3. 仅在上述边界明确后开展含最终拼接和完整交付的整日A/B；保留未采纳候选及原始测量。
