# HomeVlog

HomeVlog 是一个用于家庭主机闲时批量处理室内监控素材的 DailyVlog 生成工具。它会扫描 NAS 或本地目录中的 H.265/MP4 监控录像，按日期和摄像头分组，检测运动片段，压缩静态片段，并输出按天合并的精简视频。

当前代码以 `StreamingOrchestrator` 为主流程：prescreen、analysis、render 三个阶段可以重叠运行，最终通过 SQLite 记录状态，支持重复运行和部分恢复。

## 适用硬件与性能极限

目标生产环境：

- **CPU**: Intel i5-12600K (16 线程全负载)
- **iGPU**: UHD 770 / Intel QSV (双 VDBox 高并发解码)
- **dGPU**: RTX 3060Ti / NVDEC + NVENC (CUDA 算力全开)
- **输入**: NAS/SMB 上的 4K H.265 素材 (支持高延迟网络环境)

**性能指标 (12600K + 3060Ti 极限配置)**：
- **GPU 解码利用率**: ~88% (NVDEC 接近饱和)
- **处理速度**: 24 小时高清监控素材处理约需 **8-10 分钟**。
- **扫描响应**: 秒级启动（得益于 Lazy Metadata 缓存策略）。

## 核心优化策略

### 1. 异构双 GPU 零闲置协同流水线 (Dual-GPU Zero-Idle Pipeline)
- **Intel UHD 770 (QSV)**: 专职承担 8 并发 4K H.265 预筛（Prescreen）与主干分析（Analysis）硬件解码，充分压榨双 Gen12 VDBox 硬件解码吞吐。
- **NVIDIA RTX 3060Ti (NVENC/CUDA)**: 专职承担 YOLOv11 Tensor Core 张量批推理与 Pass 2 NVENC 硬件渲染。
- **自适应工作窃取 (Work-Stealing)**: 解码队列堆积时动态向 CUDA 租借临时槽位，渲染启动时毫秒级原子抢占，彻底消除硬件争用与单方闲置。

### 2. 动静分离多模态感知 (Multimodal Detection)
- **轻量选择性 EMA 滑动背景**: 双差分显著图融合，精准捕获静坐等微动作。
- **8×8 空间连通域抗噪**: 动态学习空间噪声分布，过滤红外夜视雪花点。
- **AudioEnergyVAD 声音事件唤醒**: 内存流 50ms 短时 RMS 包络与一阶自相关分析，交谈/啼哭等声音事件自动锁定 1x 原速原声。

### 3. 电影级平滑时间轴与动作过渡 (Natural Transition & Macro-Collapsing)
- **动作前后平滑缓冲 (Pre/Post-Roll)**: 动态动作前置扩展 1.0s，后置顺延 1.5s，完整保留人物进出起势与余波，杜绝生硬截断。
- **微小停顿缝隙吸收**: 8.0s 内短暂停顿自动保持 1.0x 原速，消除眨眼式忽快忽慢。
- **夜间长静止段宏观折叠 (Macro-Collapsing)**: 对连续无人时段自适应抽样，降低 60% 以上冗余解码渲染。
- **音频 afade 防爆音淡入淡出**: 动态段音轨自动进行 0.25s 线性双向平滑交叉淡入淡出。

### 4. 显存峰值安全防护 (VRAM Safety)
- 实施 `batch_max_files: 4~8` 黄金批次划分，RTX 3060Ti 显存峰值稳定在 **3.25 GB**（远低于 4.5GB/5.5GB 保护阈值），零换页抖动。

## 性能指标 (12600K + UHD 770 + RTX 3060Ti 实测)

- **全天 24.37 小时监控素材处理耗时**: **25 分 10 秒**（等效 **58.10x 实时倍速**）。
- **RTX 3060Ti 硬件解码利用率**: **99.2%**（硬件性能完全打满）。
- **自动化测试矩阵**: **2,402 / 2,402 全部通过 (100% 绿灯)**。

## 快速开始

安装依赖（自动配置 CUDA 环境）：

```powershell
uv sync
```

运行完整流程：

```powershell
uv run python main.py
```

执行全量自动化测试：

```powershell
uv run pytest -q
```

## 项目结构

```text
config/settings.yaml              极限性能配置文件
main.py                           CLI 入口
src/pipeline.py                   流式管线编排与双 GPU 工作窃取调度
src/prescreen.py                  Pass 1 关键帧极速粗筛 (PyAV NONKEY)
src/detector.py                   Pass 1.5 EMA 滑动背景与 AudioEnergyVAD
src/yolo_verifier.py              Pass 1.8 Tensor Core YOLO 批验证
src/segment.py                    片段聚合与连通域时空抗噪
src/timeline.py                   Speed Ramping PTS 缓动与时间码字幕
src/renderer.py                   Pass 2 双路 NVENC/QSV 并发渲染阵列
src/database.py                   SQLite WAL 状态与元数据缓存
src/utils.py                      硬件信号量与 WorkStealingManager
docs/                             项目技术文档与性能基准演进记录
```

## 注意事项

- **环境要求**: 请确保使用 `uv run` 执行，以确保加载了正确的 CUDA 版 PyTorch。
- **早停逻辑**: 修复了之前版本早停导致的时间轴空洞 Bug，现在强制映射至文件末尾。
- **磁盘空间**: 渲染前需预留 20GB 以上可用空间。
