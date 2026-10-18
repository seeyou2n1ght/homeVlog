# HomeVlog 测试策略与质量验证规范

更新：2026-10-18。[完整历史验证记录](../scratch/closeout/6acff55398704dcba9947b08609b042c/documents_before/TESTING.md)（SHA256 `5f642aaa27579e4e9d694733910fc6975d26c92a48cdb2a398544a6dc28f6faf`）保留所有逐轮通过数、失败及旧待验状态；它们不作为当前待办，也不继承新代码身份。

## 当前验收检查点

- core：`459b7f855178815501a68a4f53381fbaeda66d85790616c2b5f23975cb1e1ffb`；与完成整日对照的clock_baseline相同，指纹不覆盖全部UI/辅助文件。
- 当前整数时钟保留版本：渲染专项153 passed（33.89s），`scratch/audio_group_implementation/018a32d8c29847989b7b4b4e5d29ee3a/clock_only_renderer.xml`。完整回归911 passed、6 skipped、0 failures/errors（135.92s），见同目录`clock_only_regression.xml`，最终收据见`final_acceptance.json`。新增三个实际AAC长音频测试覆盖稠密、传统、简单路径和不同帧边界。
- 已撤回候选的933 passed/6 skipped及原型锁、PCM、回退测试保留为历史证据，不作为当前生产测试数。完整日`day_acceptance.json`确认源/分析/展示映射、目标batch0包一致；另3批设备改变，整片包不等价，严格同设备比较失败。候选声道探测超时回退，未实际执行分组，不构成算法收益结论。未增加完整成片解码、感知质量或新分析认证。
- 仓库配置Ruff及compileall通过；扩展F规则不是当前全仓准入门禁。已去掉renderer重复导入，其他历史静态诊断不自动修复。
- 既有回归覆盖双SQLite连接的排除/迟到结果及时间轴保护、真实HTTP409、真实子进程配置应用/回滚竞争、旧预览拒绝、准备hash/发布失败及实际进程骤停后恢复；替换阶段故障测试保留。不继承旧版本GPU/整日的验收身份。
- 修复前892回归及跨组恢复记录仍保留于`scratch/phase5_cross_day_acceptance/4ba1deec385748b79dfd80a64dbe2f80/acceptance.json`，仅作为其冻结版本的历史证据。
- 上一版908回归与人工排除/配置竞争/交付准备收据保留于`scratch/deep_repairs/85621280a41e42bdb23b2d3821442a67/`。

## 验证层次与隔离

1. 编译、静态检查：确认语法和实际改动范围，不能证明运行行为。
2. 自动化单元/集成：默认不依赖真实NAS、GPU或生产业务状态。本地软件FFmpeg/PyAV媒体用例按工具可用性跳过；Windows原生Job/句柄测试按平台跳过。
3. GPU硬件专项：显式`HOMEVLOG_HARDWARE_TESTS=1`，固定本地媒体、配置和资源预算；不能拿设备存在或退出0当验收。
4. 真实生产路径：隔离配置/SQLite/TEMP/output/logs，冻结输入/代码/模型/有效配置；在已有拥有Job与调度器内执行，设置来源数量、时间及产物预算，硬件/全量串行。只在授权和新验证假设需要时运行，不默认重跑历史日期。

`tests/conftest.py`将数据库、缓存、归档、渲染/审核、日志及已加载兼容消费者隔离。不得用Mock perf污染生产日志，或以旧文件充当本轮产物。fixture退出须确认进程/Job/线程终态及原资源归还；未知停止保留引用、原租约和必要输入，不为测试清理提前释放。

## 常用命令

```powershell
uv run python -m compileall main.py src scripts
uv run python -m pytest tests/

# 时间轴与人工覆盖
uv run python -m pytest tests/test_timeline_and_ramping.py tests/test_review_regression.py
# 媒体生命周期、调度和跨组恢复
uv run python -m pytest tests/test_renderer_and_ffmpeg.py tests/test_scheduler_and_hardware.py tests/test_pipeline_streaming.py tests/test_resumption_and_shutdown.py
# 扫描/持久化、视觉读者和重放恢复
uv run python -m pytest tests/test_database_and_scanner.py tests/test_visual_index.py tests/test_replay_journal.py
# 预筛/解码、配置和参考准入
uv run python -m pytest tests/test_prescreen_adaptive.py tests/test_stream_completion_policy.py tests/test_profiles_and_config.py tests/test_accuracy_baseline.py tests/test_benchmark_and_tuning.py
# 工作台与训练接口
uv run python -m pytest tests/test_audit_service.py tests/test_yolo_finetune.py

# 仅在明确需要真实GPU验证时开启，完成后删除该环境变量
$env:HOMEVLOG_HARDWARE_TESTS = '1'
uv run python -m pytest tests/test_hardware_smoke.py
Remove-Item Env:HOMEVLOG_HARDWARE_TESTS
```

选择与改动相称的最小专项，相关通过后再跑必要全量；没有新变更或未解失败时不反复刷新数字。对改动文件运行已有Ruff/JS语法检查并核对最终diff，历史Lint缺陷不得被汇总为已通过。

## 必须保护的行为

| 子系统 | 验收要点 |
| --- | --- |
| 扫描/持久化 | 元数据懒解析；不完整目录发现阻止交付；部分提交可续扫；确认零字节与不可访问/损坏分别处理 |
| 时间轴/人工审核 | `compute_display_plans`同源；配置fps量化整数帧，音频按样本闭合；审核/撤销持久化且人工优先；子帧源区间不扩张、不吞掉显著短动作 |
| 解码/预筛 | 完成度证据与坏流边界；灰度合同、候选派生和实际采样条件；未知停止不得触发第二解码/keyframes/静音备用 |
| 并发/生命周期 | 同一proc/Job/线程身份、启动认领与跨日期准入；原许可恰好一次归还；终态未知defer，不能reset调度器绕过；FileBudget取消查询不反向持Registry锁 |
| 批次/交付 | 完整批次才可发布；中断/失败不拼接或COMPLETED；旧交付保持；包替换/回滚/恢复及映射AVAILABLE门禁；未知停止保留原位置tmp/清单 |
| 工作台 | 原区间播放、审核/撤销、归因与实际重放、映射/旧审核对照；保存重开、迟到响应/身份漂移拒绝；Canvas坐标/指针/键盘须实际浏览器检查 |
| 训练/建议 | agent建议、人工事件与完整人工目标分别持久化；草稿/不确定/旧txt/伪人工拒绝；来源SHA/完整目标及train/tuning/test隔离，候选训练不切生产模型 |

涉及UI不能只跑服务端测试；涉及FFmpeg/恢复不能只用Mock。当前合同详见[ARCHITECTURE](ARCHITECTURE.md)，因果与替代方案见[DECISIONS](DECISIONS.md)。

## 真实媒体验收

- 计划与源时钟、原始区间/上下文、最终映射逐项对应；读取前后身份与完整SHA冻结范围明确。
- 严格全片音视频解码、计划视频帧数、PTS网格、音频样本/流时长、接缝和尾部均核对。AAC priming/padding按编码时间基解释，不能要求任意2ms绝对偏差。
- 原编码格式在实际播放器检查开头、相关接缝和尾部；逐帧画面/动作、音频语义及连续播放体验分开声明，播放器可打开不证明全片正确。
- 同字节媒体继承播放器内容证据须重新完整SHA并绑定原记录，明确inherited，不能冒充新播放会话。
- 终态要求拥有Job active0、同PID/创建身份或原句柄确认退出、读线程退场、Registry空及原预算恢复。缓存returncode=0或历史PID记录不替代此检查。

有限正常与真实媒体证据入口（相对scratch）：

| 范围 | 收据与边界 |
| --- | --- |
| 当前两日正常实际run_pipeline | `phase5_two_day_normal/600bdd3e5c7f4dd282577dafd7f825c6/`；两个本地合成源、每个40帧/严格AV/两次交付，禁模型与测试时钟控制，不是识别或性能 |
| 当前正常QSV/CUDA读源与NV/QSV短批次 | `phase5_direct_reader_media/953b661cbad4423da4a3ddd20ae8d297/`与`phase4_batch_render_media/68294895cc2f4c7b85e80b250a4cca4e/` |
| 当前配置完整单源分析 | `phase1_current_positive_replay/4ef59dbae8fd446996c8fe93e964b674/acceptance.json`；324.8128778秒完整源，目标4秒True/DYNAMIC_AUDIO/1x计划，不当速度收益 |
| 当前实际幼儿活动短片 | `phase4_positive_render/11cc4ca666a84646b57c2e028c680d8c/acceptance.json`；5.65秒整段113视频/265 AAC帧，271360样本/48kHz含补齐；79源帧像素SHA、80输出帧及VLC抽样检查。动作结论只限约4秒/单DYNAMIC_AUDIO的simple_dynamic；首源PTS24.029归零与20fps重采样已记录，不认证整日mixed等价/真人流畅性/声音语义 |
| 115源正式恢复 | `phase5_daily_recovery/9cbff0efa8304482a19b078384b470c7/runs/3a89af371dcf4b08b4bc94d66a1b8a2c/`；499142帧/全AV/114接缝/源身份/实际播放器及正式映射。属于当时冻结版本，不追认当前代码新整日 |

## 独立质量、调优与训练门禁

历史启发式100条、同源拆窗、已用于调优或旧预测不能冒充独立准确度。准入须有可追溯来源/区间及观看证据、明确holdout、未用于调优、当前有效预测；相同区间去重并拒绝标签/场景/事件组冲突。`used_for_tuning`旧缺失字段按现有兼容合同处理，不声称缺失证明实际盲性。

每类至少10个**独立事件组**：成人/婴幼儿活动组召回≥90%，空镜/光影组特异度≥95%。默认一源一组，显式同事件跨窗口/来源保守合组；阳性任一窗口漏检算组FN，阴性任一窗口误报算组FP。UNKNOWN或组数不足为UNAVAILABLE，不能以单例F1或调低门槛宣布提升。

新冻结阳性须完整区间视觉证据；阴性还须完整区间及可用音频证据。抽样、遮挡、VAD/转录为空或视觉无目标不能证明整段无活动。事件保留、动作清晰度与完整目标框是独立问题。

真实训练只接受完整人工目标合同、来源SHA和可靠分组；agent_visual_review不转human，草稿/不确定/事件标签/旧txt不准入。当前真实完整人工目标0，不训练或切换模型。安全训练接口的受控通过不能证明实际模型收益。

## 性能、恢复边界与证据保留

性能输入冻结与等价比较按[BENCHMARK](BENCHMARK.md)。保留全部必要配置/源身份/SQLite/帧哈希/失败和持久终态；清理同字节重复测试媒体须有逐文件事务、恢复映射、完整SHA和后验。历史验收器如果要求已清实体，先恢复再使用，不改旧报告来伪装实体仍在。

全库存补建永久取消，不把历史447xxx未捕获数字列作未完成比例。真实SMB驱动硬阻塞、物理断电、非Windows后代停止以及CPython私有启动钩子升级仍未认证；受控本地HTTP/管道、Job和进程故障不能扩大为这些保障。当前开放条件与下一步只以[PROGRESS](PROGRESS.md)为准。
