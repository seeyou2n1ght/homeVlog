# HomeVlog 生产发布与投产完整审计报告

**审计日期**：2026-09-21  
**审计对象**：HomeVlog 主流水线、算法与异构硬件调度系统、Web 审核平台及测试套件  
**目标硬件**：Intel Core i5-12600K (UHD 770 QSV) + NVIDIA GeForce RTX 3060Ti (8GB, NVENC/NVDEC) + SMB/NAS 4K 监控源  
**审计结论**：**准予发布并投入生产运行 (READY FOR PRODUCTION RELEASE)**

---

## 1. 审计综述与发布裁决

经过对代码库架构、核心流水线契约、硬件资源生命周期、故障恢复机制、配置语义、本地 Web 审核安全以及自动化测试套件的多维度穿透式审计，系统在正确性、健壮性和可恢复性方面已达成生产交付标准：

1. **流水线核心契约闭环**：
   - 彻底消除了因源监控切片物理提前 EOF、时间戳漂移、非整数帧及音频采样率引起的接缝音视频时钟脱节。
   - 所有状态（`DYNAMIC`, `DYNAMIC_AUDIO`, `PRESENCE`, `NIGHT_STATIONARY`, `MICRO_MOTION`, `STATIC`）均闭合至统一整数帧与精确 PCM 样本时钟。
2. **硬件调度与生命周期零泄漏**：
   - QSV 工作窃取准入机制与执行守卫门限严格对齐，消除了任务反复退回与无效循环。
   - 信号量租约实现了 `try ... finally` 单次且确定性释放；FFmpeg 子进程注册表在异常、超时和中断下均可保证子进程树彻底清理。
3. **交付物原子性与恢复保障**：
   - 渲染批次、伴随资产（`.srt` / `.meta.json` / `perf.json`）与成片拼接均采用“临时写入 + 原子替换”机制。
   - 中断或局部批次失败时，已渲染完成的高价值有效批次得到严格保全，支持秒级断点续渲。
4. **测试与 CI 全绿通过**：
   - 全量 278 项自动化测试：**272 passed, 6 skipped**（6 项因 CI 环境无物理 GPU 合理跳过）；
   - 物理硬件环境显式冒烟测试：**6 passed**（NVENC/QSV 编解码与逐帧哈希一致性 100% 验证）；
   - CI 环境对齐最新 FFmpeg 8.x CLI 规范（`-/filter_complex`）与 Windows 管道异常防护。

---

## 2. 静态工程质量与架构分层审计

### 2.1 分层架构与依赖边界
- **清晰分层**：
  - `src/core/`：系统配置、持久化数据库、任务身份模型与基础工具；
  - `src/hardware/`：异构 GPU 设备池、硬件感知调度器、渲染批次缓存与 FFmpeg 进程生命周期；
  - `src/algorithms/`：空间网格滤波、EMA 背景分离、AudioEnergyVAD 音频活动检测、YOLO 目标验证与时序平滑；
  - `src/stages/`：轻量扫描器（Scanner）、快速预筛选（Prescreen）、精细分析器（Detector）、确定性时间轴（Timeline）、批次渲染器（Renderer）与伴随资产封装（Companion）；
  - `src/ui/`：Rich 仪表盘、系统体检医生（Doctor）、状态看板（Status）与统一 CLI 解析器。
- **依赖倒置根除**：底层核心库不再逆向依赖上层业务模块；顶层 Facade 垫片仅保留透明向后兼容路由。

### 2.2 静态代码健康度
- **编译检查**：`python -m compileall main.py src scripts` 零语法错误、零损坏导入。
- **代码规范 (Linter)**：`ruff check` 零致命语法违规，且全库零未定义变量引用（`F821` 彻底通过）。
- **敏感数据与凭证安全**：全库无硬编码密码、无明文 API 密钥、无敏感私钥泄漏；NAS 路径采用标准局域网配置。

---

## 3. 流水线核心契约与可靠性审计

| 核心子系统 | 关键审计项 | 审计实现与验证证据 | 状态 |
| :--- | :--- | :--- | :---: |
| **Scanner** | 扫描轻量化与惰性探测 | `src/stages/scanner.py` 仅按文件名规则快速解析机位与名义时间戳，严禁执行阻塞性网络 `av.open`；媒体元数据在后续预筛/分析阶段惰性提取并持久化。 | **合规** |
| **Prescreen** | 快速粗筛与内存候选帧 | `src/stages/prescreen.py` 采用关键帧稀疏采样与 `video_frame_to_gray` 规范灰度转换；候选帧保留于内存张量池，避免磁盘大量临时图片 I/O 轰炸。 | **合规** |
| **Detector** | 多模态检测与异常截断保护 | `src/stages/detector.py` 融合空间集中度、EMA 动态能量、VAD 婴儿啼哭保护与 YOLO 目标过滤。若管道解码不完整（`complete=False`），拒绝提交伪成功分析。 | **合规** |
| **Timeline** | 确定性时间轴与提前 EOF 治理 | `src/stages/timeline.py` 为全状态注入 `tpad=stop_mode=clone` 末帧克隆与 `trim=end_frame` 整数帧截断；音频通过 `atrim=end_sample` 锁死精确采样点数，彻底杜绝视频黑洞与接缝脱节。 | **合规** |
| **Scheduler** | 异构 Makespan 竞价与信号量隔离 | `src/hardware/scheduler.py` 严格区分 NVDEC/NVENC 与 QSV 信号量；工作窃取基于 Makespan ETA 竞价，放宽门限与执行守卫统一传递，杜绝活锁。 | **合规** |
| **Renderer** | 事务性批次与带内参数集注入 | `src/stages/renderer.py` 使用 `-/filter_complex` 脚本传参，注入 `-bsf:v dump_extra` 带内参数集；渲染成片经 PyAV 解码首尾校验后原子提交。 | **合规** |
| **Finalization** | 视频时钟驱动与伴随资产原子化 | `src/pipeline.py` 最终 concat 依据视频 duration 推进；音频重对齐后 AAC 编码。伴随资产（`.srt`, `.meta.json`）失败时阻断标记发布。 | **合规** |

---

## 4. Web 审核工具与人机协同审计

针对 `scripts/audit_tool/` 的安全性与流水线契约一致性，完成以下审计：

1. **网络信任边界防护**：
   - 服务默认且强制绑定 `127.0.0.1` 本地回环；
   - 启用 HTTP `Host` 与 `Origin` 头校验，跨域预检（`do_OPTIONS`）直接阻断；
   - 启动时自动生成高强度随机令牌（`AUDIT_TOKEN = secrets.token_urlsafe(32)`），并注入 `HttpOnly; SameSite=Strict` Cookie；所有写操作与媒体重渲染必须提供合法令牌。
2. **重渲染契约与主流程一致性**：
   - 重渲染复用主流程 `check_source_files_accessibility` 与 `build_timeline_from_rows(..., resolve_presence=False)`，严禁部分素材丢失时静默跳过；
   - 单任务取消（`cancel_task`）采用任务级 `threading.Event`，严禁调用全局 `kill_all()` 或污染全局中断状态。

---

## 5. 运行就绪与投产环境审计

### 5.1 本地就绪体检报告 (`main.py --doctor`)
- **Python 环境**：3.12.13 (PyTorch, PyAV, Ultralytics YOLO, OpenCV, Rich 就绪)；
- **FFmpeg 编解码**：8.1-full_build (启用 NVENC HEVC, QSV HEVC, DXVA2, D3D11VA, D3D12VA)；
- **GPU 算力与显存**：NVIDIA GeForce RTX 3060 Ti (8.0 GB, 驱动 616.92) + Intel UHD 770；
- **网络与存储挂载**：`\\192.168.5.8\LyuShare\XiaomiCamera_01_B888805AA3CD` 挂载网络延迟 15.0ms；输出盘剩余空间 394.85 GB（远高于 20 GB 警戒线）；
- **视觉模型权重**：`yolo11m.pt` (38.8 MB) 本地完整校验通过。

### 5.2 生产配置审计 (`config/settings.yaml`)
- 所有配置项均已接入代码语义，零无用或悬空字段；
- 启用了 QSV 分析并发（7 槽）与独立渲染保留通道（1 槽）隔离，确保 `analysis + render <= max_qsv_concurrency`（8 槽）；
- 异构渲染策略配置为 `render_gpu_policy: heterogeneous`，批次并发设为 2（支持 2 NVENC + 1 QSV 动态调度）。

---

## 6. 自动化测试套件执行证据

### 6.1 纯软及模拟回归套件
```text
uv run python -m pytest tests/
======================= 272 passed, 6 skipped in 38.89s =======================
```
- **测试覆盖面**：数据库与扫描（14 项）、多模态动作与音频 VAD（16 项）、时间轴与速率过渡（53 项）、批次渲染与 FFmpeg 管道（41 项）、异构硬件调度与工作窃取（16 项）、流水线断点续传与关机保护（27 项）、Web 审核服务（40 项）、YOLO 微调（5 项）等；
- **跳过说明**：6 项用例明确标注为真实物理 GPU 依赖，在软件回归环境下按设计安全跳过。

### 6.2 物理真实硬件加速冒烟套件
```text
$env:HOMEVLOG_HARDWARE_TESTS = '1'
uv run python -m pytest tests/test_hardware_smoke.py
============================= 6 passed in 10.38s ==============================
```
- 覆盖真实 NVENC HEVC 与 QSV HEVC 真实双路编码、带内参数集重注入、异构边界逐帧 MD5/SHA256 哈希连续性。

---

## 7. 生产部署与日常运维标准操作规程 (SOP)

### 7.1 常规生产运行指令
```powershell
# 1. 投产前环境自检
uv run python main.py --doctor

# 2. 针对指定日期执行全流程浓缩（常规交互模式）
uv run python main.py --date 20260320

# 3. 后台/无头环境静默执行（流式心跳日志输出，适合定时任务或 CI）
uv run python main.py --date 20260320 --no-tui

# 4. 多日自动批处理（如处理历史三天未完成素材）
uv run python main.py --days 3
```

### 7.2 运维监控与日常维护指令
```powershell
# 查看数据库各机位切片处理与成片状态总览
uv run python main.py --status

# 启动本地 Web 人工协同审核平台（默认绑定 127.0.0.1:8080）
uv run python -m scripts.audit_tool.app

# 清理临时 staging 与历史孤儿中间产物
uv run python main.py --clean-temp
```

### 7.3 异常应急与容灾处置
1. **网络 NAS 瞬时断开或素材损坏**：
   - 流水线会自动记录错误并安全标记该文件任务为 `FAILED`，并阻断当日成片最终合成（防止生成缺漏残卷）；
   - 待 NAS 网络恢复后，直接重新运行相同命令，系统自动跳过已成功的批次，仅对失败项执行断点修复。
2. **磁盘空间不足警戒 (< 20GB)**：
   - 流水线在启动和每个日期批次前均会自检磁盘；若空间不足会自动优雅熔断并等待清理，不造成正在写入的文件截断。
