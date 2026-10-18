#!/usr/bin/env python3
"""scripts/export_dataset.py

从真实反馈归档库 (data/feedback_archive/) 与数据库人工打标记录中，
仅导出绑定图像和捕获来源的完整人工目标复核；事件标签、旧txt和
自动预测不能证明目标框或无人。构建COCO80类别空间的YOLO训练数据。
"""

import argparse
import hashlib
import math
import os
import json
import logging
import random
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

    latest_items: dict[str, DatasetItem] = {}

    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if not isinstance(rec, dict):
                    continue
            except (ValueError, TypeError):
                continue

            rel_img = rec.get("archived_image_path", "")
            if not rel_img:
                continue

            img_p = PROJECT_ROOT / rel_img if not Path(rel_img).is_absolute() else Path(rel_img)
            if not img_p.exists() or img_p.stat().st_size == 0:
                continue  # Basename alone cannot identify the captured source image.

            try:
                latest_items[os.path.normcase(str(img_p.resolve()))] = DatasetItem(
                    image_path=img_p,
                    manual_label=rec.get("manual_label", "UNKNOWN"),
                    segment_id=int(rec.get("segment_id", 0)),
                    source_video=rec.get("filepath", ""),
                    timestamp=float(rec.get("frame_timestamp", 0.0)),
                )
            except (ValueError, TypeError):
                continue

    return [latest_items[key] for key in sorted(latest_items)]


def load_db_fallback_items(db_path: Path) -> list[DatasetItem]:
    """如果归档目录暂空，从 SQLite 数据库 segments 中扫描已打标记录。"""
    if not db_path.exists():
        return []

    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
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


def verified_target_annotation(item: DatasetItem):
    """Read one immutable human target review; event labels/legacy txt are not truth.

    Source identity is the captured provenance of the reviewed image, not a
    promise that the NAS is currently online. The image content itself must
    still match exactly. Automatic model predictions never enter this path.
    """
    review_path = item.image_path.with_suffix(".targets.json")
    if not review_path.is_file():
        return None
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict):
        raise ValueError("Target review must be an object")
    image = item.image_path.read_bytes()
    if (review.get("schema") != 1 or review.get("label_source") != "human_target_review"
            or review.get("all_targets_reviewed") is not True
            or not isinstance(review.get("reviewed_at"), str) or not review["reviewed_at"].strip()
            or review.get("image_sha256") != hashlib.sha256(image).hexdigest()):
        raise ValueError("Target review missing human completeness or current image identity")
    canonical = lambda value: os.path.normcase(str(Path(value).resolve()))
    identity = review.get("source_identity")
    if (not item.source_video or canonical(review.get("source_video", "")) != canonical(item.source_video)
            or not isinstance(identity, list) or len(identity) != 3
            or canonical(identity[0]) != canonical(item.source_video)
            or type(identity[1]) is not int or identity[1] <= 0
            or type(identity[2]) is not int or identity[2] <= 0
            or not math.isfinite(float(review.get("frame_timestamp", float("nan"))))
            or float(review["frame_timestamp"]) != float(item.timestamp)):
        raise ValueError("Target review source/frame identity does not match archive item")
    import cv2
    import numpy as np
    decoded_image = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)
    if decoded_image is None:
        raise ValueError("Reviewed image cannot be decoded")
    boxes = review.get("boxes")
    if (not isinstance(boxes, list) or type(review.get("no_targets")) is not bool
            or review["no_targets"] != (len(boxes) == 0)):
        raise ValueError("Empty target labels require explicit human confirmation")
    lines = validate_target_boxes(boxes, decoded_image.shape[1], decoded_image.shape[0])
    return lines, image, review


def validate_target_boxes(boxes, image_width, image_height):
    """Shared strict validation for human review writes and dataset consumption."""
    if not isinstance(boxes, list):
        raise ValueError("Target boxes must be a list")
    lines = []
    for box in boxes:
        cls = box["class_id"]
        values = [float(box[key]) for key in ("x_center", "y_center", "width", "height")]
        xc, yc, width, height = values
        if (type(cls) is not int or cls not in (0, 15, 16)
                or any(not math.isfinite(value) for value in values)
                or width * image_width < 1 or height * image_height < 1
                or not (0 <= xc-width/2 < xc+width/2 <= 1)
                or not (0 <= yc-height/2 < yc+height/2 <= 1)):
            raise ValueError("Reviewed box has invalid class, finite coordinates or image bounds")
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {width:.6f} {height:.6f}")
    return lines


def generate_yolo_labels_for_item(item: DatasetItem, target_txt_path: Path, yolo_model: Any = None) -> int:
    """Export verified human boxes/explicit no-target review only; never infer truth."""
    try:
        annotation = verified_target_annotation(item)
    except (OSError, ValueError, TypeError, KeyError) as error:
        logger.warning("Target annotation excluded for %s: %s", item.image_path, error)
        return -1
    if annotation is None:
        logger.warning("Target annotation required; event label/legacy txt excluded: %s", item.image_path)
        return -1
    lines, _, _ = annotation
    target_txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def target_source_key(source_video):
    from src.core.identity import parse_filename
    parsed = parse_filename(Path(source_video).name)
    return "date:" + parsed["date"] if parsed else "source:" + os.path.normcase(str(Path(source_video).resolve()))


def build_yolo_dataset(
    items: list[DatasetItem],
    output_dir: Path,
    val_ratio: float = 0.2,
    yolo_model_path: Path | None = None,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, Any]:
    """Freeze reviewed targets into disjoint train/val/test source groups."""
    # 初始化输出目录
    img_train_dir = output_dir / "images" / "train"
    img_val_dir = output_dir / "images" / "val"
    lbl_train_dir = output_dir / "labels" / "train"
    lbl_val_dir = output_dir / "labels" / "val"
    img_test_dir = output_dir / "images" / "test"
    lbl_test_dir = output_dir / "labels" / "test"

    directories = (img_train_dir, img_val_dir, img_test_dir, lbl_train_dir, lbl_val_dir, lbl_test_dir)
    if not (0 < val_ratio < 1 and 0 < test_ratio < 1 and val_ratio + test_ratio < 1):
        raise ValueError("val_ratio and test_ratio must be positive and sum to less than 1")
    if type(seed) is not int or seed < 0:
        raise ValueError("Split seed must be a nonnegative integer")
    for d in directories:
        d.mkdir(parents=True, exist_ok=True)

    # Keep the old model argument for callers, but never run automatic labeling.
    if yolo_model_path is not None:
        logger.info("Model argument ignored: export requires explicit human target annotations")
    from src.feedback import normalize_label
    if any(any(d.iterdir()) for d in directories):
        raise ValueError("Use a new empty dataset directory to avoid stale labels")
    stats = {**{split + "_" + metric: 0 for split in ("train", "val", "test") for metric in ("images", "boxes")},
             "negatives": 0, "positives": 0, "excluded": 0, "duplicates": 0, "conflicts": 0}
    accepted, excluded = [], []
    for it in sorted(items, key=lambda item: (str(item.image_path.resolve()), item.source_video, item.timestamp)):
        try:
            annotation = verified_target_annotation(it)
        except (OSError, ValueError, TypeError, KeyError) as error:
            logger.warning("Target annotation excluded for %s: %s", it.image_path, error)
            annotation = None
        if annotation is None:
            stats["excluded"] += 1
            excluded.append({"segment_id": it.segment_id, "reason": "unverified_target_review"})
            continue
        accepted.append((it, annotation))
    # Source dates keep adjacent files and simultaneous camera views together.
    # Exact duplicate images link groups and are never counted twice. Conflicting
    # target labels for identical bytes are quarantined instead of choosing one.
    from src.core.replay_journal import digest
    def group_for(item):
        return target_source_key(item.source_video)
    parent = {group_for(item): group_for(item) for item, _ in accepted}
    def root(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group
    by_image = {}
    for entry in accepted:
        by_image.setdefault(entry[1][2]["image_sha256"], []).append(entry)
    unique = []
    for entries in by_image.values():
        if len({tuple(sorted(annotation[0])) for _, annotation in entries}) != 1:
            stats["conflicts"] += len(entries)
            stats["excluded"] += len(entries)
            excluded.extend({"segment_id": item.segment_id, "reason": "conflicting_identical_image_targets"} for item, _ in entries)
            continue
        groups = [group_for(item) for item, _ in entries]
        for group in groups[1:]:
            parent[root(group)] = root(groups[0])
        unique.append(entries[0])
        stats["duplicates"] += len(entries)-1
        stats["excluded"] += len(entries)-1
        excluded.extend({"segment_id": item.segment_id, "reason": "identical_image_duplicate"} for item, _ in entries[1:])
    members = {}
    for group in parent:
        members.setdefault(root(group), []).append(group)
    group_ids = {group: digest(sorted(members[root(group)])) for group in parent}
    group_members = {digest(sorted(values)): sorted(values) for values in members.values()}
    groups = sorted({group_ids[group_for(item)] for item, _ in unique})
    random.Random(seed).shuffle(groups)
    n_test = min(max(1, round(len(groups)*test_ratio)), len(groups)-2) if len(groups) >= 3 else 0
    n_val = min(max(1, round(len(groups)*val_ratio)), len(groups)-n_test-1) if len(groups) >= 2 else 0
    assignments = {group: ("test" if index < n_test else "val" if index < n_test+n_val else "train") for index, group in enumerate(groups)}
    manifest = []
    for it, (lines, image_bytes, review) in unique:
        label = normalize_label(it.manual_label)
        source_group = group_ids[group_for(it)]
        split = assignments[source_group]
        suffix = hashlib.sha256(str(it.image_path.resolve()).encode()).hexdigest()[:12]
        name = it.image_path.stem + "_" + suffix
        dest_lbl = output_dir / "labels" / split / (name + ".txt")
        n_boxes = len(lines)
        dest_lbl.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        dest_img = output_dir / "images" / split / (name + it.image_path.suffix)
        dest_img.write_bytes(image_bytes)
        stats[split + "_images"] += 1
        stats[split + "_boxes"] += n_boxes
        stats["negatives" if n_boxes == 0 else "positives"] += 1
        manifest.append({"image": str(dest_img.relative_to(output_dir)), "source_video": it.source_video,
                         "label": label, "split": split, "source_group": source_group,
                         "label_source": "human_target_review", "image_sha256": review["image_sha256"],
                         "label_file": str(dest_lbl.relative_to(output_dir)), "label_sha256": hashlib.sha256(dest_lbl.read_bytes()).hexdigest(),
                         "source_identity": review["source_identity"], "frame_timestamp": review["frame_timestamp"],
                         "all_targets_reviewed": True, "no_targets": review["no_targets"]})
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "excluded_items.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "dataset_split.json").write_text(json.dumps({"schema": 1, "seed": seed, "val_ratio": val_ratio,
        "test_ratio": test_ratio, "grouping": "source_date_else_canonical_path_linked_by_identical_images",
        "groups": [{"id": group, "split": assignments[group], "members": group_members[group]} for group in sorted(groups)]}, ensure_ascii=False, indent=2), encoding="utf-8")

    # 生成 data.yaml
    data_yaml_path = output_dir / "data.yaml"
    # 使用规范的绝对或正斜杠相对路径
    import yaml
    import ultralytics
    coco = yaml.safe_load((Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml").read_text(encoding="utf-8"))
    yaml_content = yaml.safe_dump({"path": str(output_dir.resolve()).replace("\\", "/"),
                                   "train": "images/train", "val": "images/val", "test": "images/test", "names": coco["names"]},
                                  allow_unicode=True, sort_keys=False)
    data_yaml_path.write_text(yaml_content, encoding="utf-8")
    logger.info("已生成数据集配置文件: %s", data_yaml_path)

    stats["target_review_required"] = True
    stats["source_groups"] = len(groups)
    stats["usable"] = all(stats[split+"_images"] > 0 for split in ("train", "val", "test"))
    return stats


def main():
    parser = argparse.ArgumentParser(description="HomeVlog 专属场景难样本导出与 YOLO 数据集构建工具")
    parser.add_argument("--archive-dir", type=Path, default=PROJECT_ROOT / "data" / "feedback_archive", help="反馈归档目录")
    parser.add_argument("--db", type=Path, default=PROJECT_ROOT / "data" / "vlog.db", help="数据库路径")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "yolo_dataset", help="YOLO 数据集导出目录")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="验证集切分比例 (默认 0.2)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="独立留出比例 (默认0.2，按日期/原片分组)")
    parser.add_argument("--seed", type=int, default=42, help="固定划分种子")
    parser.add_argument("--model", type=Path, default=PROJECT_ROOT / "models" / "yolo11n.pt", help="已弃用：自动预测不作为人工目标真值导出")
    parser.add_argument("--camera", help="Camera MAC or configured alias")
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

    if args.camera:
        from src.utils import load_config
        aliases = load_config().get("cameras", {})
        camera = next((mac for mac, alias in aliases.items() if alias == args.camera), args.camera)
        items = [it for it in items if camera.casefold() in it.source_video.casefold()]
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
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("\n" + "=" * 65)
    print(" YOLO 人工目标复核导出完成" if stats["usable"] else " 尚无可用训练/验证集：需要独立人工目标框或明确无目标复核")
    print("=" * 65)
    print(f" 训练集: {stats['train_images']} 张图片 | {stats['train_boxes']} 个目标标注")
    print(f" 验证集: {stats['val_images']} 张图片 | {stats['val_boxes']} 个目标标注")
    print(f" 留出集: {stats['test_images']} 张图片 | {stats['test_boxes']} 个目标标注")
    print(f" 困难负样本 (抑制误报): {stats['negatives']} 张 (空 txt 标注)")
    print(f" 配置文件位置: {args.output_dir / 'data.yaml'}")
    print("=" * 65 + "\n")
    if not stats["usable"]:
        return
    print("💡 下一步操作: 运行本地微调训练脚本:")
    print(f"    uv run python scripts/train_yolo.py --data {args.output_dir / 'data.yaml'}")


if __name__ == "__main__":
    main()
