# 项目状态

## 当前目标

完成最终交付前全仓清理、文档收敛与多维回归验证，准备交付合并。

## 已验证交付状态

- **架构形态**：纯命令行确定性视频浓缩流水线（`main.py` -> `src/stages/` / `src/hardware/` / `src/algorithms/` / `src/core/` / `src/ui/`），无 Web 审核台或历史兼容 Shim。
- **最新版本基准（`20260326 / cam0`，130 源 / 1,439.9 分钟 / 零缓存冷跑）**：
  - 端到端壁钟 **`29.75 分钟`（`1,785.19s` / `48.40×` 实时）**，产出 `5,555.78 MB`（`5.8×` 浓缩），对比第 2 轮基线（`1,873.53s` / `46.11×`）再降 **`88.34s`（`-4.7%`）**、累计较初始基线（`2,234.6s` / `38.66×`）缩短 **`449.4s`（`-20.1%`）**；证据：`logs/perf/perf_20260326_cam0_20261115_052353.json`。
  - 主阶段工时：`prescreen` 130 项 `1,171.60s`（均 `9.01s`）、`analysis` 116 项 `5,094.05s`（均 `43.91s`，`-50.6%`）、`render` 130 批 `3,166.45s`（均 `24.36s`，`-14.7%`）、渲染尾段 **`38.80s`**（`-84.8%`）、`final_concat` **`96.48s`**；异构编队均衡：`nv_0` 52 批（`91.1%`）、`nv_1` 62 批（`90.7%`）、`qsv_0` 16 批（均耗时 `8.47s`，无全解码长批误窃）。
  - 成片完整性核验（`uv run python scripts/verify_cold_run_audit.py 20260326`）：**PASS** — `296,919` 包时钟误差 `<= 1 tick`、129/129 接缝 0 断裂、元数据亮点事件 `875` 个。
- **历史跨日参考基线（`20260320 / cam0`，142 源）**：冷跑 `2,193.52s`（`36m33s` / `40.73×` 实时），详见 [BENCHMARK](BENCHMARK.md)。
- **最新实跑与修复验证（2026-11-15 多日生产批跑收口）**：
  - 结合 `20260320`–`20260327` 连续多日实跑日志定位并修复 `hevc_qsv` 缺少 `-bf 0` 导致的异构接缝 DTS 畸形与 `20260321` 结尾校验失败（同步升级 `render_fingerprint` 至 `v4` 防止命中旧 `has_b_frames=2` 缓存、对齐 `valid_video` `math.isclose` 容差边界）；
  - 修复 `Monitor.reset()` 未随 `dump_perf()` 跨天清零、分离 `qsv_upstream_yields` 与 `qsv_steal_rejected` 计数，并补充 `cleanup_temp_artifacts` 对 `temp/.task_locks/batch-*.lock` 及交付提交后陈旧 `.concat_failures/` 诊断目录的自动清理；
  - `20260321 / cam0` 修复交付并经 `verify_cold_run_audit.py 20260321` 及全包检验通过（`410,609` 视频包 `0` 例 DTS/时长异常、`139/139` 接缝连续）；同时同负载消融证伪 4 路并肩渲染（`1,315.6s` vs 3 路 `1,302.2s`），维持 3 路并发甜点，详见 [BENCHMARK](BENCHMARK.md)。
- **自动化验证**：`ruff check`、`compileall` 与 `pytest tests/` 全量通过（6 项 `HOMEVLOG_HARDWARE_TESTS` 硬件专项按预期跳过）；`main.py --doctor` 全项就绪。

## 当前生产默认约束与配置基线

- **关键帧预筛选与精析并发**：`detection.iframe_min_file_duration: 1800.0`、`detection.iframe_fallback_active_ratio: 0.45`；`detection.analysis_max_workers: 3`（1 CUDA + 2 QSV，[ADR0088](DECISIONS.md#adr0088分析并发度收敛全解码渲染权重校准与中间批产物精简2026-11-15)）、`detection.cpu_analysis_workers: 0`（实测多路 CPU 分析线程推高负载且导致整日性能回退）。
- **异构渲染与交付**：`render.nv_tail_concurrency: 3` 配合 `render.nv_tail_use_cpu_decode: true`（[ADR0087](DECISIONS.md)）；`render.cleanup_batches_on_success: true` 交付后自动清理本次临时批次；中间 `batch_*.mp4` 不做 `-movflags +faststart`，仅最终拼接产出前置 MOOV；`output.audio.concat_codec: "aac"`（保证 `< 1 tick` A/V 时钟闭合）。
- **目录与维护约定**：临时运行产物（`temp/`、`logs/`、`data/`）及 `scratch/` 已从交付态剥离；诊断与验证入口为 `main.py --doctor`、`main.py --status`、`scripts/analyze_perf.py`、`scripts/benchmark_day.py`、`scripts/verify_cold_run_audit.py` 与 `scripts/inspect_detection.py`。

## 下一步动作

- 如需针对特定生产日重算并覆盖已交付成片，使用 `uv run python main.py --date <YYYYMMDD> --reanalyze --force-render`。
- 若部署至新主机或更换 GPU/驱动，先运行 `uv run python main.py --doctor` 并核对 `config/settings.yaml` 路径及显存预算。
