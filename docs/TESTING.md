# 测试与验收

测试围绕纯命令行流水线组织。核心算法回归位于 `tests/test_algorithm_regressions.py`，当前验收状态见 [PROGRESS](PROGRESS.md)。

## 执行

先运行受影响的专项，再做一次全库回归。沿用 uv 配置的 PyTorch 环境。

```powershell
uv run ruff check main.py src scripts tests
uv run python -m compileall main.py src scripts tests
uv run python -m pytest tests/ -v --durations=10
```

`tests/conftest.py` 将数据库、成片、批次和日志放在测试临时目录。真实媒体测试依赖 FFmpeg。合成流程通过 `pipeline.render_gpu_policy: cpu_only` 在无 GPU 环境运行。GPU 专项须显式开启 `HOMEVLOG_HARDWARE_TESTS`。

| 范围 | 验证行为 |
| --- | --- |
| 扫描与持久化 | 懒解析、完整发现、零字节与不可访问区分、失败恢复、历史事实表与生产分析隔离 |
| 分析与预筛 | 灰度合同、迟到短事件、完成度、保守阴性、未知停止不得启动第二读者 |
| 时间轴 | 共享展示计划、整数帧与样本时钟、精确源区间、确定性浓缩 |
| 并发与生命周期 | 许可恰好归还一次，未知停止保留许可，进程和线程终态，跨日期认领 |
| 渲染与恢复 | 软件 FFmpeg 像素、时钟、音频和接缝，失败及取消不标完成，交付回滚与骤停恢复 |
| 合成端到端 | 多机位扫描、动静判定、软件渲染、伴生资产、遥测和重复运行复用 |

合成素材用于验证流程合同。识别准确率、主观质量、真实 SMB 故障和全天硬件性能分别需要相应的实测证据。真实机先运行 `uv run python main.py --doctor`，再按 [BENCHMARK](BENCHMARK.md) 的负载和口径测量。架构决策见 [ADR0082](DECISIONS.md)。
