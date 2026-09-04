#!/usr/bin/env python3
"""scripts/train_yolo.py

HomeVlog 专属机位小样本迁移学习与轻量微调训练流。
基于 Ultralytics YOLO (如 YOLO11n / YOLO11s)，采用骨干网络冻结 (Backbone Freezing) 策略，
仅针对 Neck 与 Detect Head 进行梯度更新，既能抑制机位专属背景误报并提升微弱动作召回，
又能防止样本量较少时发生过拟合与对通用人形的灾难性遗忘。
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台编码防护
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_yolo")


def check_dataset_health(data_yaml_path: Path) -> bool:
    """检查数据集路径与图片是否存在。"""
    if not data_yaml_path.exists():
        logger.error("数据集配置文件未找到: %s", data_yaml_path)
        return False

    dataset_root = data_yaml_path.parent
    train_images_dir = dataset_root / "images" / "train"
    if not train_images_dir.exists() or len(list(train_images_dir.glob("*.jpg"))) == 0:
        logger.error("训练集图片为空: %s", train_images_dir)
        return False

    return True


def apply_custom_model_to_config(config_path: Path, model_path: Path) -> bool:
    """安全热替换 config/settings.yaml 中的 YOLO 模型配置（创建 .bak 备份）。"""
    if not config_path.exists():
        logger.error("配置文件不存在: %s", config_path)
        return False

    try:
        import yaml
    except ImportError:
        logger.error("缺少 pyyaml 模块，无法写回配置文件")
        return False

    backup_path = config_path.with_suffix(".yaml.bak")
    try:
        content = config_path.read_text(encoding="utf-8")
        backup_path.write_text(content, encoding="utf-8")
        logger.info("已创建配置文件备份: %s", backup_path)

        data = yaml.safe_load(content)
        if "yolo" not in data:
            data["yolo"] = {}

        # 相对路径格式
        try:
            rel_model = str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            rel_model = str(model_path).replace("\\", "/")

        data["yolo"]["enabled"] = True
        data["yolo"]["model_path"] = rel_model

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, sort_keys=False)

        logger.info("[Success] 成功更新 %s: yolo.model_path = %s", config_path.name, rel_model)
        return True
    except Exception as e:
        logger.error("更新配置文件失败: %s", e)
        return False


def run_training(
    data_yaml: Path,
    base_model: Path,
    output_model: Path,
    epochs: int = 30,
    batch: int = 16,
    imgsz: int = 640,
    freeze: int = 10,
    device: str = "0",
) -> Path | None:
    """执行 YOLO 迁移微调训练。"""
    try:
        from ultralytics import YOLO
        import torch
    except ImportError:
        logger.error("环境未安装 ultralytics 或 torch，请执行: uv sync")
        return None

    # 设备可用性自适应探测
    if device in ("0", "cuda:0", "cuda") and not torch.cuda.is_available():
        logger.warning("CUDA 不可用，自动回退到 CPU 训练模式")
        device = "cpu"

    logger.info("正在加载基底权重: %s", base_model)
    model = YOLO(str(base_model))

    project_dir = PROJECT_ROOT / "runs" / "train"
    project_dir.mkdir(parents=True, exist_ok=True)
    exp_name = "homevlog_custom"

    logger.info("🚀 开始微调训练 (epochs=%d, batch=%d, freeze=%d, device=%s)...", epochs, batch, freeze, device)

    try:
        train_res = model.train(
            data=str(data_yaml.resolve()),
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            freeze=freeze,
            device=device,
            amp=True,
            workers=2,
            project=str(project_dir),
            name=exp_name,
            exist_ok=True,
            verbose=True,
            save=True,
            plots=True,
        )
    except Exception as e:
        logger.error("YOLO 训练过程出现异常: %s", e)
        return None

    # 定位 best.pt
    best_weight = project_dir / exp_name / "weights" / "best.pt"
    if not best_weight.exists():
        last_weight = project_dir / exp_name / "weights" / "last.pt"
        best_weight = last_weight if last_weight.exists() else None

    if best_weight and best_weight.exists():
        output_model.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(best_weight), str(output_model))
        logger.info("🎉 最优微调权重已成功导出并落盘: %s (%d KB)", output_model, output_model.stat().st_size // 1024)
        return output_model
    else:
        logger.error("未能找到导出的权重文件 best.pt")
        return None


def main():
    parser = argparse.ArgumentParser(description="HomeVlog 专属机位 YOLO 小样本微调训练流")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "yolo_dataset" / "data.yaml", help="数据集 data.yaml 路径")
    parser.add_argument("--base-model", type=Path, default=PROJECT_ROOT / "models" / "yolo11n.pt", help="预训练基底权重路径")
    parser.add_argument("--output-model", type=Path, default=PROJECT_ROOT / "models" / "yolo11_custom.pt", help="微调输出权重保存路径")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数 (默认 30)")
    parser.add_argument("--batch", type=int, default=16, help="批次大小 (默认 16)")
    parser.add_argument("--freeze", type=int, default=10, help="冻结主干特征层数 (默认 10)")
    parser.add_argument("--device", type=str, default="0", help="训练设备 (0 或 cpu)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "settings.yaml", help="配置文件路径")
    parser.add_argument("--apply", action="store_true", help="训练完成后自动热替换配置文件中的 yolo.model_path")
    parser.add_argument("--dry-run", action="store_true", help="仅验证环境与数据集，不实际启动训练")

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(" [YOLO Fine-Tuning] HomeVlog 本地小样本迁移学习微调工具")
    print("=" * 65)
    print(f" 数据集路径: {args.data}")
    print(f" 基底权重:   {args.base_model}")
    print(f" 输出模型:   {args.output_model}")
    print(f" 训练超参:   epochs={args.epochs}, batch={args.batch}, freeze={args.freeze}, device={args.device}")
    print("=" * 65)

    if not check_dataset_health(args.data):
        print("\n[!] 数据集检查未通过。")
        print("[i] 请先执行数据导出脚本:")
        print("    uv run python scripts/export_dataset.py")
        print("=" * 65 + "\n")
        return

    if not args.base_model.exists():
        print(f"[!] 找不到预训练基底权重: {args.base_model}")
        print("[i] Ultralytics 将在启动时尝试自动下载官方权重，或可手动放置于 models/ 目录。")

    if args.dry_run:
        print("[*] Dry-run 模式：环境与数据集校验通过，未执行实际训练。")
        return

    best_model_path = run_training(
        data_yaml=args.data,
        base_model=args.base_model,
        output_model=args.output_model,
        epochs=args.epochs,
        batch=args.batch,
        freeze=args.freeze,
        device=args.device,
    )

    if best_model_path and best_model_path.exists():
        print("\n" + "=" * 65)
        print(" 🏆 YOLO 专属机位微调模型成功生成！")
        print("=" * 65)
        print(f" 模型路径: {best_model_path}")
        print("=" * 65 + "\n")

        if args.apply:
            apply_custom_model_to_config(args.config, best_model_path)
    else:
        print("\n[!] 训练未完成或未生成有效权重。\n")


if __name__ == "__main__":
    main()
