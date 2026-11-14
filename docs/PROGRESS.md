# 项目状态

## 当前目标

完成最终交付前全仓清理、文档收敛与多维回归验证，准备交付合并。

## 已验证交付状态

- **架构形态**：纯命令行确定性视频浓缩流水线（`main.py` -> `src/stages/` / `src/hardware/` / `src/algorithms/` / `src/core/` / `src/ui/`），无 Web 审核台或历史兼容 Shim。
- **最新版本基准（`20260326 / cam0`，130 源 / 1,439.9 分钟 / 零缓存冷跑）**：
  - 端到端壁钟 **`31.2 分钟`（`1,873.5s` / `46.11×` 实时）**，产出 `5,388.27 MB`（`5.9×` 浓缩），对比优化前 `20260326` 冷跑基线（`37.2 分钟` / `2,234.6s` / `38.66×`）缩短 `361.1s`（`-16.2%`）；证据：`logs/perf/perf_20260326_cam0_20261115_023847.json`。
  - 主阶段工时：`prescreen` 130 项 `20.8min`（均 `9.6s`）、`analysis` 116 项 `171.7min`（均 `88.8s`）、`render` 130 批 `61.9min`（均 `28.6s`）、`final_concat` `112.2s`；异构编队均衡：`nv_0` 52 批（`95.4%`）、`nv_1` 45 批（`95.4%`）、`qsv_0` 33 批（`23.2%`，逆向窃取 25 批）。
  - 成片完整性核验（`uv run python scripts/verify_cold_run_audit.py 20260326`）：**PASS** — `294,557` 包时钟误差 `<= 1 tick`、129/129 接缝 0 断裂、元数据亮点事件 `843` 个。
- **历史跨日参考基线（`20260320 / cam0`，142 源）**：冷跑 `2,193.52s`（`36m33s` / `40.73×` 实时），详见 [BENCHMARK](BENCHMARK.md)。
- **自动化验证**：`ruff check` 与 `compileall` 全部通过；全库测试 **`1006 passed, 6 skipped`**（6 项 `HOMEVLOG_HARDWARE_TESTS` 硬件专项按预期跳过）；`main.py --doctor` 全项就绪。

## 当前生产默认约束与配置基线

- **关键帧预筛选**：`detection.iframe_min_file_duration: 1800.0`、`detection.iframe_fallback_active_ratio: 0.45`；`detection.cpu_analysis_workers: 0`（实测多路 CPU 分析线程推高负载且导致整日性能回退）。
- **异构渲染与交付**：`render.nv_tail_concurrency: 3` 配合 `render.nv_tail_use_cpu_decode: true`（[ADR0087](DECISIONS.md)）；`render.cleanup_batches_on_success: true` 交付后自动清理本次临时批次；`output.audio.concat_codec: "aac"`（保证 `< 1 tick` A/V 时钟闭合）。
- **目录与维护约定**：临时运行产物（`temp/`、`logs/`、`data/`）及 `scratch/` 已从交付态剥离；诊断与验证入口为 `main.py --doctor`、`main.py --status`、`scripts/analyze_perf.py`、`scripts/benchmark_day.py`、`scripts/verify_cold_run_audit.py` 与 `scripts/inspect_detection.py`。

## 下一步动作

- 如需针对特定生产日重算并覆盖已交付成片，使用 `uv run python main.py --date <YYYYMMDD> --reanalyze --force-render`。
- 若部署至新主机或更换 GPU/驱动，先运行 `uv run python main.py --doctor` 并核对 `config/settings.yaml` 路径及显存预算。
