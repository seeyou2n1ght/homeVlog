#!/usr/bin/env python3
"""scripts/export_dataset.py

从真实反馈归档库 (data/feedback_archive/) 与数据库人工打标记录中，
抽取专属机位难样本（误报 Hard Negatives 与漏报 Hard Positives），
构建符合 Ultralytics YOLO 标准格式的微调训练数据集。
"""

import argparse
import json
import logging
import random
import shutil
import sqlite3
import sys
from dataclasses import dataclass
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
logger = logging.getLogger("export_dataset")


@dataclass
class DatasetItem:
    image_path: Path
    manual_label: str  # "FALSE_ALARM", "MISSED_MOTION", "CONFIRMED_MOTION", etc.
    segment_id: int
    source_video: str
    timestamp: float


def load_archive_items(archive_dir: Path) -> list[DatasetItem]:
    """从本地 feedback_archive 读取已经固化的高清帧和 manifest 记录。"""
    manifest_path = archive_dir / "manifest.jsonl"
    images_dir = archive_dir / "images"

    if not manifest_path.exists() or not images_dir.exists():
        logger.info("未在 %s 找到 manifest.jsonl", archive_dir)
        return []

    items: list[DatasetItem] = []
    seen_images: set[str] = set()

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue

            rel_img = rec.get("archived_image_path", "")
            if not rel_img:
                continue

            img_p = PROJECT_ROOT / rel_img if not Path(rel_img).is_absolute() else Path(rel_img)
            if not img_p.exists() or img_p.stat().st_size == 0:
                # 尝试直接在 images_dir 查找同名文件
                img_p = images_dir / Path(rel_img).name
                if not img_p.exists() or img_p.stat().st_size == 0:
                    continue

            if str(img_p) in seen_images:
                continue
            seen_images.add(str(img_p))

            items.append(
                DatasetItem(
                    image_path=img_p,
                    manual_label=rec.get("manual_label", "UNKNOWN"),
                    segment_id=int(rec.get("segment_id", 0)),
                    source_video=rec.get("filepath", ""),
                    timestamp=float(rec.get("frame_timestamp", 0.0)),
                )
            )

    return items


def load_db_fallback_items(db_path: Path) -> list[DatasetItem]:
    """如果归档目录暂空，从 SQLite 数据库 segments 中扫描已打标记录。"""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT id, filepath, start_time, end_time, manual_label, archived_frame_path
        FROM segments
        WHERE manual_label IS NOT NULL 
          AND manual_label != 'UNREVIEWED'
        """
    ).fetchall()
    conn.close()

    items: list[DatasetItem] = []
    for r in rows:
        arch_p = r["archived_frame_path"]
        if arch_p:
            p = PROJECT_ROOT / arch_p if not Path(arch_p).is_absolute() else Path(arch_p)
            if p.exists() and p.stat().st_size > 0:
                items.append(
                    DatasetItem(
                        image_path=p,
                        manual_label=r["manual_label"],
                        segment_id=r["id"],
                        source_video=r["filepath"],
                        timestamp=round((r["start_time"] + r["end_time"]) / 2.0, 2),
                    )
                )

    return items


def generate_yolo_labels_for_item(
    item: DatasetItem,
    target_txt_path: Path,
    yolo_model: Any = None,
) -> int:
    """为单张图片生成 YOLO 标准格式标注文件 (.txt)。
    
    规则:
    1. FALSE_ALARM (困难负样本):
       生成空的 .txt 文件。告知模型当前机位背景无需产生任何检测框，强力抑制窗帘/树影/扫地机误检！
    2. CONFIRMED_MOTION / VERIFIED_MOTION / MISSED_MOTION (正样本):
       若有 yolo_model，运行推理获取目标的边界框并保存。
    """
    lbl = item.manual_label.upper()

    # 1. 困难负样本：必须写入空文本文件（0 目标）
    if lbl == "FALSE_ALARM":
        target_txt_path.write_text("", encoding="utf-8")
        return 0

    # 2. 正样本：检测并生成目标框 (class_id x y w h)
    box_lines: list[str] = []
    if yolo_model is not None:
        try:
            import cv2
            img = cv2.imread(str(item.image_path))
            if img is not None:
                h_img, w_img = img.shape[:2]
                # 对困难漏报样本使用较低阈值 0.15 捕获微弱轮廓；对确认样本使用 0.25
                conf_th = 0.15 if lbl == "MISSED_MOTION" else 0.25
                results = yolo_model(img, conf=conf_th, verbose=False)
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        # COCO: 0=person, 15=cat, 16=dog
                        if cls_id == 0:
                            target_cls = 0  # person
                        elif cls_id in (15, 16):
                            target_cls = 1  # pet
                        else:
                            continue

                        # 转换为归一化 xywh
                        xywh = box.xywhn[0].cpu().numpy()
                        box_lines.append(f"{target_cls} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}")
        except Exception as e:
            logger.debug("YOLO 预测生成标签异常: %s", e)

    # 写入标签文件
    content = "\n".join(box_lines)
    target_txt_path.write_text(content, encoding="utf-8")
    return len(box_lines)


def build_yolo_dataset(
    items: list[DatasetItem],
    output_dir: Path,
    val_ratio: float = 0.2,
    yolo_model_path: Path | None = None,
) -> dict[str, Any]:
    """将归档图片划分为 train/val 并输出完整 YOLO 数据集结构。"""
    # 初始化输出目录
    img_train_dir = output_dir / "images" / "train"
    img_val_dir = output_dir / "images" / "val"
    lbl_train_dir = output_dir / "labels" / "train"
    lbl_val_dir = output_dir / "labels" / "val"

    for d in (img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 加载推理模型（用于生成正样本伪标签）
    yolo_model = None
    if yolo_model_path and yolo_model_path.exists():
        try:
            from ultralytics import YOLO
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            yolo_model = YOLO(str(yolo_model_path))
            yolo_model.to(device)
            logger.info("已加载 YOLO 模型生成正样本标注: %s (device: %s)", yolo_model_path.name, device)
        except Exception as e:
            logger.warning("未能加载 YOLO 预测模型，将生成默认标签: %s", e)

    # 区分正负样本分层抽样，保证 train 和 val 中正负样本比例均衡
    neg_items = [it for it in items if it.manual_label.upper() == "FALSE_ALARM"]
    pos_items = [it for it in items if it.manual_label.upper() != "FALSE_ALARM"]

    random.seed(42)
    random.shuffle(neg_items)
    random.shuffle(pos_items)

    def split_list(lst: list[DatasetItem], ratio: float):
        n_val = int(len(lst) * ratio)
        return lst[n_val:], lst[:n_val]

    train_neg, val_neg = split_list(neg_items, val_ratio)
    train_pos, val_pos = split_list(pos_items, val_ratio)

    train_set = train_neg + train_pos
    val_set = val_neg + val_pos
    random.shuffle(train_set)
    random.shuffle(val_set)

    logger.info("数据集划分完成: 训练集 %d 张 (正%d/负%d), 验证集 %d 张 (正%d/负%d)",
                len(train_set), len(train_pos), len(train_neg),
                len(val_set), len(val_pos), len(val_neg))

    stats = {
        "train_images": len(train_set),
        "val_images": len(val_set),
        "train_boxes": 0,
        "val_boxes": 0,
        "negatives": len(neg_items),
        "positives": len(pos_items),
    }

    # 输出训练集
    for it in train_set:
        dest_img = img_train_dir / it.image_path.name
        shutil.copy2(str(it.image_path), str(dest_img))
        dest_lbl = lbl_train_dir / f"{it.image_path.stem}.txt"
        n_boxes = generate_yolo_labels_for_item(it, dest_lbl, yolo_model)
        stats["train_boxes"] += n_boxes

    # 输出验证集
    for it in val_set:
        dest_img = img_val_dir / it.image_path.name
        shutil.copy2(str(it.image_path), str(dest_img))
        dest_lbl = lbl_val_dir / f"{it.image_path.stem}.txt"
        n_boxes = generate_yolo_labels_for_item(it, dest_lbl, yolo_model)
        stats["val_boxes"] += n_boxes

    # 生成 data.yaml
    data_yaml_path = output_dir / "data.yaml"
    # 使用规范的绝对或正斜杠相对路径
    yaml_content = f"""# HomeVlog 专属机位微调数据集配置
path: {str(output_dir.resolve()).replace('\\', '/')}
train: images/train
val: images/val

names:
  0: person
  1: pet
"""
    data_yaml_path.write_text(yaml_content, encoding="utf-8")
    logger.info("已生成数据集配置文件: %s", data_yaml_path)

    return stats


def main():
    parser = argparse.ArgumentParser(description="HomeVlog 专属场景难样本导出与 YOLO 数据集构建工具")
    parser.add_argument("--archive-dir", type=Path, default=PROJECT_ROOT / "data" / "feedback_archive", help="反馈归档目录")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "vlog.db", help="数据库路径")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "yolo_dataset", help="YOLO 数据集导出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集切分比例 (默认 0.2)")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "yolo11n.pt", help="基础 YOLO 模型路径")
    parser.add_argument("--dry-run", action="store_true", help="仅预览可用样本数，不执行文件生成")

    args = parser.parse_args()

    print("\n" + "=" * 65)
    print(" [Dataset Export] HomeVlog 专属机位难样本导出工具")
    print("=" * 65)

    # 1. 加载归档项
    items = load_archive_items(args.archive_dir)
    if not items:
        logger.info("归档目录无数据，回退扫描数据库...")
        items = load_db_fallback_items(args.db)

    if not items:
        print("[!] 未找到任何已打标的复核样本。")
        print("[i] 请先在 Web 审核平台 (uv run python scripts/audit_tool/app.py) 中标记若干疑难切片。")
        print("=" * 65 + "\n")
        return

    # 统计分布
    lbl_counts: dict[str, int] = {}
    for it in items:
        lbl_counts[it.manual_label] = lbl_counts.get(it.manual_label, 0) + 1

    print(f" 发现有效候选样本: {len(items)} 张")
    print(f" 标签分布详情: {lbl_counts}")
    print(f" 导出目标路径: {args.output_dir}")
    print("=" * 65)

    if args.dry_run:
        print("[*] Dry-run 模式，未写入任何文件。")
        return

    print("\n[*] 正在构建 YOLO 标准格式数据集...")
    stats = build_yolo_dataset(
        items=items,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        yolo_model_path=args.model,
    )

    print("\n" + "=" * 65)
    print(" ✅ YOLO 数据集导出完成！")
    print("=" * 65)
    print(f" 训练集: {stats['train_images']} 张图片 | {stats['train_boxes']} 个目标标注")
    print(f" 验证集: {stats['val_images']} 张图片 | {stats['val_boxes']} 个目标标注")
    print(f" 困难负样本 (抑制误报): {stats['negatives']} 张 (空 txt 标注)")
    print(f" 配置文件位置: {args.output_dir / 'data.yaml'}")
    print("=" * 65 + "\n")
    print("💡 下一步操作: 运行本地微调训练脚本:")
    print(f"    uv run python scripts/train_yolo.py --data {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
