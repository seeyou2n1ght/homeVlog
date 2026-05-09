# HomeVlog YOLO 级联检测重构实施计划

## 1. 架构设计 (Two-Stage Cascade Detection)

将当前的 `MotionDetector` 纯帧差法扫描，重构为“帧差提议 (Stage 1) + YOLO 验证 (Stage 2)”的级联架构。

### 1.1 Stage 1: 帧差提议网络 (Region Proposal)
*   **目标**：作为极速的前置过滤器，捕捉所有可能的运动（宁缺毋滥）。
*   **改动**：
    *   取消或大幅弱化降噪逻辑。
    *   在 `config/settings.yaml` 中，将建议的 `motion_sensitivity` 默认值下调（例如从 `1.2` 改为 `1.0` 或更低），鼓励产生更多的候选动态段。
    *   针对 Pass 1 (预筛) 严重漏检的问题，将 `prescreen_segments` 的默认值从 30 提升至 120，或改为基于固定时间间隔采样。

### 1.2 Stage 2: YOLO 验证网络 (Verification)
*   **目标**：过滤 Stage 1 产生的误报（如光影变化、风吹树叶）。
*   **改动**：
    *   新建 `yolo_verifier.py` 模块。
    *   加载轻量级模型（如 YOLOv11n 或 YOLOv8n），并强制启用 GPU 加速 (TensorRT / CUDA)。
    *   仅接收 Stage 1 判定为 `DYNAMIC` 的时间片段（Segments）。
    *   在每个 `DYNAMIC` 片段中，按一定的采样率（例如 1 FPS 或片段首中尾抽取 3 帧）提取图像送入 YOLO。
    *   如果 YOLO 检测到目标类别（`person`, `car`, `dog`, `cat` 等），则保留该片段的 `DYNAMIC` 状态；若未检测到任何关注目标，则将其翻转回 `STATIC`。

## 2. 实施步骤与模块改造

### 步骤 1：依赖与环境更新
*   更新 `pyproject.toml` 和 `uv.lock`，引入 `ultralytics` 包及其相关依赖。

### 步骤 2：配置项扩充
*   修改 `config/settings.yaml`，新增 `yolo` 配置块：
    *   `model_path`: YOLO 模型路径（默认下载自动获取）。
    *   `target_classes`: 关注的类别 ID 列表（例如 COCO 数据集中的 0:person, 16:dog, 17:cat 等）。
    *   `sample_fps`: 验证阶段的抽帧率（默认 1 或更低）。
    *   `confidence`: YOLO 置信度阈值。

### 步骤 3：核心逻辑开发 (`yolo_verifier.py`)
*   实现 `YoloVerifier` 类，负责模型加载和推理。
*   提供方法：传入视频路径及时间段列表，返回经过验证后的真实动态时间段列表。

### 步骤 4：流水线整合 (`pipeline.py` & `detector.py`)
*   在 Pass 1.5 之后，或者直接在 Timeline 构建之前，插入 `YoloVerifier` 的调用。
*   将 Stage 1 提取的原始 segments 列表传入 YOLO 验证模块，接收清洗后的 segments。
*   执行全局时间轴 (Timeline) 的平滑合并逻辑（考虑 `min_motion_duration` 和 `gap_tolerance`）。

## 3. 性能与风险评估

*   **性能**：YOLO 仅在极少数疑似动态帧上运行，且使用 Nano 模型，总体分析时间增加预计在 10% 以内。
*   **SOLID 原则符合性**：新增的 YOLO 验证模块符合单一职责原则（SRP）和开闭原则（OCP），对原有帧差代码修改较小，作为后置过滤器插入流水线。
*   **容错处理**：若 YOLO 模型加载失败或 OOM，应能优雅降级（Fallback）到纯帧差法结果，保证流水线不中断。

---
**请评估此计划，确认后我将开始分步骤执行。**