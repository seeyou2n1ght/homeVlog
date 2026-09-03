# HomeVlog 项目长期工程约定

## 架构事实（2026-09 审计后确立）
- 流式管线 (StreamingOrchestrator) 为唯一主链路；renderer/prescreen 中的旧批处理链路已清除，禁止回流。
- 渲染批次调度：heavy/light 双队列 + 保序滑窗；哨兵（None）仅为唤醒信号，退出条件必须是 `all_dispatched && heavy.empty() && light.empty()`。
- 渲染收尾必须对账：dispatched vs (produced | terminal)，缺失即 error，杜绝静默丢批。
- 静态段渲染快路径：纯静态且段长 >= 2*kf_interval 的文件在 select 抽帧后再 scale/hwdownload；改动需跑 `scripts/benchmark.py --operator`。
- 展示时长计划单一来源：`timeline.compute_display_plans()`，滤镜图与 SRT 字幕共用；字幕墙钟用 `src_offset_at_display()` 逆映射。
- YOLO 验证帧索引一律使用解码真实 `effective_fps`（adaptive 0.5~5），禁止用固定 `detector.fps`。

## 配置纪律
- settings.yaml 中的每个键必须有代码消费；新增功能先接线再加键，删除功能同步删键（2026-09 已清理 5 个僵尸键）。
- 配置档位与代码分支必须一一对应（教训：ultra_long 配置存在但代码缺分支，长期未生效）。

## 测试与文档
- 测试矩阵为 9 模块 100 passed + 1 skipped（E2E 按硬件跳过）；README/TESTING/PROJECT/BENCHMARK 中的用例数须随测试增删同步更新。
- 提交前：`uv run python -m compileall main.py src` + `uv run pytest tests/` 全绿。
