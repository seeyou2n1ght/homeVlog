# 跨项目通用经验 — HomeVlog 开发沉淀

本文档沉淀 HomeVlog 开发中可复用的设计模式、工程实践和踩坑经验，面向未来视频处理、管线调度类项目。

---

## 1. FFmpeg 集成模式

### 1.1 rawvideo pipe：避免中间文件

**问题**：逐帧处理视频时，写临时帧文件再读取 = 双倍 IO + 磁盘空间。

**模式**：stdout pipe 传 raw video，内存中直接 numpy 处理。

```python
# 解码端：ffmpeg → rawvideo → stdout pipe
proc = subprocess.Popen(
    ["ffmpeg", "-i", file, "-vf", "scale=640:360", "-r", "5",
     "-f", "rawvideo", "-pix_fmt", "gray", "-"],
    stdout=subprocess.PIPE, bufsize=frame_size * 4,
)
while True:
    raw = proc.stdout.read(frame_size)
    if len(raw) < frame_size: break
    frame = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
```

**关键点**：
- `bufsize` 设 `frame_size * 4` 以上，避免 pipe 频繁阻塞
- 灰度图用 `-pix_fmt gray`，数据量 = RGB 的 1/3
- 先算数据量再定分辨率：`W×H×channels × fps × duration`。1280×720×3×5fps×340s ≈ 4.7GB pipe 数据，直接卡死

### 1.2 filter_complex concat：多段拼接无转码

**问题**：多文件/多段合成时，逐段编码再 concat = 多次编码损失 + 时间翻倍。

**模式**：单条 ffmpeg 命令，filter_complex 内 trim + concat 一气呵成。

```python
# v{idx} 和 a{idx} 必须严格交替排列
labels = "".join(f"[v{i}][a{i}]" for i in range(N))
concat = f"{';'.join(parts_v)};{';'.join(parts_a)};{labels}concat=n={N}:v=1:a=1[v][a]"
```

**关键点**：concat filter 的 segment 模式要求 pads 按 `[v0][a0][v1][a1]...[vN][aN]` 排列。video 全在前、audio 全在后 → "Media type mismatch" 报错。

### 1.3 硬件加速 per-input 语法

**问题**：`-hwaccel cuda` 全局只放一次 → 仅对第一个 `-i` 生效。多文件渲染时后面文件仍走软解。

```bash
# 错误：仅 file1 硬解
ffmpeg -hwaccel cuda -hwaccel_output_format cuda -i file1 -i file2 ...

# 正确：每个 -i 前都要放
ffmpeg \
  -hwaccel cuda -hwaccel_output_format cuda -i file1 \
  -hwaccel cuda -hwaccel_output_format cuda -i file2 \
  ...
```

### 1.4 GPU scale → hwdownload → CPU filter 不兼容

GPU filter（`scale_cuda`/`scale_qsv`）后的帧在 GPU 显存。`hwdownload,format=nv12` 拷回系统内存。但之后的 CPU filter（如 `fps`）可能不兼容。

**已知不兼容**：`scale_cuda,hwdownload,format=nv12,fps=fps=1/30` → 产 0 帧。
**替代方案**：① CPU 路径全程用 CPU filter ② GPU 路径用 `setpts` 替代 `fps`

### 1.5 ffprobe 获取准确时长

文件名时间戳差值不完全等于实际视频时长（误差 0.3-1s），seek 接近文件末尾时可能出界。

**模式**：关键边界操作前用 `ffprobe` 取 `format.duration`，加 ≥0.5s 安全边距。

---

## 2. 自适应阈值 vs 硬编码阈值

### 2.1 核心原则

硬编码阈值假设所有场景服从同一分布。光照变化（日间 vs 红外）、摄像机差异（噪声水平、编码参数）都会改变信号分布。**per-file 统计阈值**比全局常量鲁棒得多。

### 2.2 IQR-based 自适应阈值

```python
energies = np.array(energies, dtype=np.float32)
q1 = float(np.percentile(energies, 25))
q3 = float(np.percentile(energies, 75))
iqr = q3 - q1
threshold = q3 + sensitivity * iqr
```

**优于 mean+std 的场景**：分布有长尾或离群值时。IQR 对 25%-75% 之外的数据不敏感，而 mean+std 被离群值严重拉偏。

**sensitivity 选择**：正态假设下 k=2 ≈ 3.4σ → false positive ~0.04%。实际需要根据素材调优，start with k=2-4。

### 2.3 为什么 ratio-based 方法在低对比度场景失效

`count(diff > hard_threshold) / total_pixels` 在日间有效（运动像素 diff 5-15 >> 阈值 15），在红外失效（运动像素 diff 1.3 vs 背景 1.15，>> 15 的像素仅 0.4%）。

**教训**：信号绝对值极低时（< 2），ratio 方法的 SNR 崩溃。改用连续标量（如 `mean(abs diff)`）+ per-file 统计 baseline。

---

## 3. 时域信号处理

### 3.1 GOP 伪影识别

H.265 编码的 I-frame/P-frame 差异在逐帧解码时产生周期性 `mean_diff` 尖峰。关键特征：

| 特征 | GOP 伪影 | 真实运动 |
|------|---------|---------|
| 幅值 | 与弱运动同量级 | — |
| 时域形态 | 孤立尖峰（1-3 帧） | 持续高原（≥10 帧） |
| 周期性 | 每 4 秒（取决于 GOP size） | 无固定周期 |
| 空间分布 | 全帧均匀微涨 | 局部聚集 |

### 3.2 中值滤波压制孤立尖峰

```python
def median_filter(signal, window=7):
    """1D median filter — isolates spikes become neighbors, sustained runs preserved."""
    half = window // 2
    result = np.empty_like(signal)
    for i in range(len(signal)):
        left, right = max(0, i - half), min(len(signal), i + half + 1)
        result[i] = np.median(signal[left:right])
    return result
```

**窗口选择**：window 必须覆盖尖峰总宽度（peak + decay tail）。GOP 尖峰典型宽度 3 帧 → window=7（两侧各 ≥2 帧 baseline buffer）。window=5 覆盖不全，尖峰残留。

**副作用**：持续运动段的边缘被平滑（duration 压缩），需要相应降低 `min_motion_duration` 阈值。

### 3.3 空间方差备选方案

`std(cell_means)` — ROI 切分为 grid，每 cell 算 mean diff，取跨 cell 标准差。GOP 伪影（全局均匀）→ 低方差；真实运动（局部聚集）→ 高方差。过滤率 98%，但信号强度不如中值滤波。作为备选方案保留。

---

## 4. 管线状态机与 Crash-Safe

### 4.1 SQLite WAL 模式断点续跑

```python
# 启动时回滚中断状态
def _recover_interrupted(db):
    rows = db.conn.execute(
        "SELECT date, cam_index FROM render_tasks WHERE status='RENDERING'"
    ).fetchall()
    for r in rows:
        db.set_render_status(r["date"], r["cam_index"], "PENDING")
```

**为什么优于文件标记**：
- 原子性：SQLite write transaction 是原子的
- 查询友好：`WHERE status='RENDERING'` 一次找到所有中断任务
- WAL 模式支持并发读 + 串行写，多 worker 不冲突

### 4.2 阶段式状态流转

```
PENDING → PRESCREEN(STATIC|SUSPICIOUS) → PENDING(ANALYSIS) → ANALYZED|FAILED
                                                                    ↓
                                                              RENDERING → COMPLETED|FAILED
```

每阶段有明确的"已完成"标记（`is_prescreen_complete`, `is_analysis_complete`, `is_render_completed`）。重跑时跳过已完成阶段，幂等安全。

### 4.3 并发写安全

多 worker 写同一 SQLite DB 时，用 `threading.Lock` 保护所有写操作。WAL 模式的 concurrent-read + serialized-write 天然适合此场景。

---

## 5. 双 GPU 负载均衡

### 5.1 Interleave 分片优于顺序切分

**错误**：`tasks[:N//2]` 和 `tasks[N//2:]`
- 短文件全落在一半 → 一个 GPU 空闲，另一个满载

**正确**：`tasks[0::2]` 和 `tasks[1::2]`
- 长短混合 → 两个 GPU 都吃到混合负载

### 5.2 按 duration 降序后再 interleave

```python
sorted_tasks = sorted(tasks, key=lambda t: t.get("file_duration") or 0, reverse=True)
list_nvdec = sorted_tasks[0::2]  # 最长 + 第3长 + ...
list_qsv = sorted_tasks[1::2]    # 第2长 + 第4长 + ...
```

排序确保两个 list 的总时长尽可能接近。

### 5.3 GPU slot 感知

不要超过硬件限制：consumer NVIDIA GPU 通常 2 NVDEC + 1 NVENC；Intel iGPU 4 QSV decode + 1 QSV encode。超出限制 → 驱动排队 → 实际反而更慢。

---

## 6. 配置管理模式

### 6.1 单文件 YAML + 详细注释

```yaml
# 参数名: 默认值  # [可选值] 说明
# 标注 "调优" 的参数对效果影响最大
motion_sensitivity: 2.0  # [1.0 ~ 8.0] IQR 倍数, 调优
```

好处：
- 用户改参数不需要读代码
- 可选值注释降低试错成本
- 调优标注引导用户关注关键参数

### 6.2 消除 hardcode 层次

| 优先级 | 说明 |
|--------|------|
| P0 | 行为阈值（影响检测/渲染结果的数值） |
| P1 | 超时/重试次数 |
| P2 | buffer 大小、thread 数量 |
| P3 | 日志轮转、显示格式 |

P0/P1 必须提取到 config，P2 建议提取，P3 可选。

### 6.3 默认值一致性

同一参数在多个文件中出现时，默认值必须一致。不一致 = 隐蔽 bug。发现不一致后应统一到一个配置源，或至少统一默认值。

---

## 7. 增量开发节奏

### 7.1 模块写完立即用真实数据测

Scanner → Prescreen → Detector → Segment → Timeline → Renderer，每步完成后用真实样本验证，bug 不跨模块累积。

### 7.2 inspect 脚本模式

为关键模块写 `scripts/inspect_*.py`：
- 单文件输入，全流程输出
- 终端 summary + CSV 导出 + 帧提取（可视化验证）
- 比全管线跑快 10-100x，适合参数调优

### 7.3 先探数据再定算法

在写检测逻辑之前，先写 probe 脚本 dump 原始 signal 分布（mean diff、histogram、时间序列）。数据特征决定算法选择，而非反过来。

---

## 8. Python 工程环境

### 8.1 uv 包管理

```toml
[project]
name = "homevlog"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["numpy", "pyyaml", "psutil", "nvidia-ml-py"]

[tool.uv]
dev-dependencies = []
```

`uv sync` 一行安装所有依赖，无冲突。比 pip + venv + requirements.txt 快 10x。

### 8.2 Windows 路径处理

- 配置文件中用双反斜杠或正斜杠：`C:\\path\\to\\dir` 或 `C:/path/to/dir`
- `Path` 对象在任何 OS 上都用 `/`
- subprocess 传递参数时用 `str(path)` 而非 `path.as_posix()`（Windows 上后者可能被 ffmpeg 误解）

### 8.3 Windows 日志编码

Windows console 使用 cp1252 编码，Unicode 字符（如 `→`）可能导致 `UnicodeEncodeError`。全部替换为 ASCII（`->`），或给 StreamHandler 设 `encoding="utf-8"`。

---

## 9. 快速参考：性能数字

| 操作 | 典型耗时 | 瓶颈 |
|------|---------|------|
| QSV 单帧提取 (320×180) | ~0.5s | seek + decode |
| NVDEC 5fps 连续解码 (640×360 gray) | ~12s/10min 文件 | GPU decode |
| NVENC HEVC 编码 (1920×1080, p1, CQ28) | ~2-3x 实时 | GPU encode |
| rawvideo pipe 数据量 | `W×H×ch×fps×dur` bytes | IO bandwidth |

---

*文档更新于 2026-04-29，基于 HomeVlog v0.1.0 开发经验。未来项目遇到 новых 模式可继续扩充。*
