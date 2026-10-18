#!/usr/bin/env python3
"""scripts/train_yolo.py

HomeVlog 专属机位小样本迁移学习与轻量微调训练流。
基于 Ultralytics YOLO (如 YOLO11n / YOLO11s)，采用骨干网络冻结 (Backbone Freezing) 策略，
仅针对 Neck 与 Detect Head 进行梯度更新；是否改善识别及遗忘风险须独立验收。
"""

import argparse
import hashlib
import json
import logging
import shutil
import sys
import os
import uuid
from pathlib import Path

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

    # Older exports inferred empty object labels from event negatives or mixed
    # automatic predictions with reviewed targets. Preserve them as history,
    # but do not train on that unverified labeling contract.
    try:
        import yaml
        config = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
        names = config.get("names", {})
        if (Path(config.get("path", "")).resolve() != dataset_root.resolve()
                or config.get("train") != "images/train" or config.get("val") != "images/val"
                or config.get("test") != "images/test"
                or "download" in config or len(names) != 80
                or [names[index] for index in (0, 15, 16)] != ["person", "cat", "dog"]):
            raise ValueError("Dataset paths or target class namespace differ from verified export")
        records = json.loads((dataset_root / "manifest.json").read_text(encoding="utf-8"))
        if not isinstance(records, list) or not records:
            raise ValueError("Missing human target export manifest")
        from scripts.export_dataset import target_source_key
        from src.core.replay_journal import digest
        protocol = json.loads((dataset_root / "dataset_split.json").read_text(encoding="utf-8"))
        if protocol.get("schema") != 1:
            raise ValueError("Missing frozen source split protocol")
        groups = {}
        member_groups = {}
        for group in protocol["groups"]:
            if group["id"] != digest(sorted(group["members"])) or group["split"] not in {"train", "val", "test"}:
                raise ValueError("Invalid source split group")
            groups[group["id"]] = group
            for member in group["members"]:
                if member in member_groups:
                    raise ValueError("Source/date crosses dataset groups")
                member_groups[member] = group["id"]
        listed_images, listed_labels, image_hashes = set(), set(), set()
        counts = {split: 0 for split in ("train", "val", "test")}
        for record in records:
            if (record.get("label_source") != "human_target_review"
                    or record.get("all_targets_reviewed") is not True
                    or type(record.get("no_targets")) is not bool):
                raise ValueError("Unverified target labels or automatic predictions")
            image = (dataset_root / record["image"]).resolve()
            label = (dataset_root / record["label_file"]).resolve()
            root = dataset_root.resolve()
            if not image.is_relative_to(root) or not label.is_relative_to(root):
                raise ValueError("Manifest path outside dataset")
            split = record["split"]
            group_id = member_groups.get(target_source_key(record["source_video"]))
            if (group_id != record.get("source_group") or groups.get(group_id, {}).get("split") != split
                    or image.parent != root / "images" / split or label.parent != root / "labels" / split
                    or image in listed_images or label in listed_labels or record["image_sha256"] in image_hashes):
                raise ValueError("Source/date, image or label crosses frozen dataset split")
            if (hashlib.sha256(image.read_bytes()).hexdigest() != record["image_sha256"]
                    or hashlib.sha256(label.read_bytes()).hexdigest() != record["label_sha256"]
                    or record["no_targets"] != (not label.read_text(encoding="utf-8").strip())):
                raise ValueError("Exported image/label identity changed")
            listed_images.add(image)
            listed_labels.add(label)
            image_hashes.add(record["image_sha256"])
            counts[split] += 1
        actual_images = {p.resolve() for p in (dataset_root / "images").rglob("*") if p.is_file()}
        actual_labels = {p.resolve() for p in (dataset_root / "labels").rglob("*.txt")}
        if listed_images != actual_images or listed_labels != actual_labels:
            raise ValueError("Dataset contains unlisted or missing image/label files")
        if not all(counts.values()):
            raise ValueError("Training, tuning and independent test groups are all required")
    except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError) as error:
        logger.error("目标复核数据集未通过来源/内容检查: %s", error)
        return False

    return True


def apply_custom_model_to_config(config_path: Path, model_path: Path) -> bool:
    """安全热替换 config/settings.yaml 中的 YOLO 模型配置（创建 .bak 备份）。"""
    if not config_path.exists() or not model_path.is_file() or model_path.stat().st_size == 0:
        logger.error("配置或候选权重不存在/为空，未更新配置")
        return False

    try:
        content = config_path.read_bytes()

        # 相对路径格式
        try:
            rel_model = str(model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        except ValueError:
            rel_model = str(model_path).replace("\\", "/")

        from src.core.tuner import apply_config_safe
        result = apply_config_safe({"yolo.enabled": True, "yolo.model_path": rel_model},
                                   config_path=config_path, expected_sha256=hashlib.sha256(content).hexdigest())
        logger.info("配置备份: %s", result["backup_path"])

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
    if not check_dataset_health(data_yaml):
        return None
    if (output_model.exists() or output_model.with_suffix(".training.json").exists()
            or not base_model.is_file() or base_model.stat().st_size == 0):
        logger.error("输出模型已存在或基底权重缺失；请选择新输出路径和有效本地基底")
        return None
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

    from src.core.replay_journal import atomic_json, digest
    def dataset_identity():
        return digest({name: hashlib.sha256((data_yaml.parent/name).read_bytes()).hexdigest()
                       for name in (data_yaml.name, "manifest.json", "dataset_split.json")})
    before = dataset_identity()
    base_sha = hashlib.sha256(base_model.read_bytes()).hexdigest()
    project_dir = PROJECT_ROOT / "runs" / "train" / uuid.uuid4().hex
    project_dir.mkdir(parents=True, exist_ok=False)
    exp_name = "fit"
    receipt_path = project_dir / "training_receipt.json"
    receipt = {"status": "RUNNING", "base_model_sha256": base_sha, "dataset_identity": before,
               "data_yaml": str(data_yaml.resolve()), "output_model": str(output_model.resolve()),
               "parameters": {"epochs": epochs, "batch": batch, "imgsz": imgsz, "freeze": freeze, "device": device},
               "quality_gate": "UNAVAILABLE"}
    atomic_json(receipt_path, receipt)

    logger.info("🚀 开始微调训练 (epochs=%d, batch=%d, freeze=%d, device=%s)...", epochs, batch, freeze, device)

    try:
        logger.info("正在加载基底权重: %s", base_model)
        model = YOLO(str(base_model))
        model.train(
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
            exist_ok=False,
            verbose=True,
            save=True,
            plots=True,
        )
        # Only this unique attempt may produce the candidate. A stale best/last
        # from another run must never make an incomplete training look complete.
        best_weight = project_dir / exp_name / "weights" / "best.pt"
        if not best_weight.is_file() or best_weight.stat().st_size == 0:
            raise ValueError("Current training attempt produced no complete best.pt")
        if (not check_dataset_health(data_yaml) or dataset_identity() != before
                or hashlib.sha256(base_model.read_bytes()).hexdigest() != base_sha):
            raise ValueError("Training dataset or base model changed during the attempt")
        output_model.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_model.with_name(output_model.name + "." + uuid.uuid4().hex + ".pending")
        try:
            shutil.copyfile(best_weight, temporary)
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            # Same-directory hard-link publication is exclusive and atomic;
            # unsupported filesystems fail without overwriting an existing model.
            os.link(temporary, output_model)
        finally:
            temporary.unlink(missing_ok=True)
        receipt.update(status="CANDIDATE_EXPORTED_PENDING_QUALITY", model_sha256=hashlib.sha256(output_model.read_bytes()).hexdigest(),
                       best_weight=str(best_weight), training_receipt=str(receipt_path))
        atomic_json(output_model.with_suffix(".training.json"), receipt)
        logger.info("🎉 最优微调权重已成功导出并落盘: %s (%d KB)", output_model, output_model.stat().st_size // 1024)
        return output_model
    except Exception as error:
        receipt.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        logger.exception("YOLO训练或候选发布失败；保留本次产物: %s", project_dir)
        return None
    except BaseException as error:
        receipt.update(status="INTERRUPTED", error_type=type(error).__name__)
        raise
    finally:
        atomic_json(receipt_path, receipt)


def evaluate_candidate(base_model: Path, candidate: Path, validation_data: Path, training_data: Path | None = None) -> bool:
    """Compare targets on source-isolated test images; never authorize deployment."""
    from src.core.replay_journal import atomic_json, digest
    from ultralytics import YOLO
    if training_data is None or not check_dataset_health(training_data) or not check_dataset_health(validation_data):
        raise ValueError("Current verified train/val/test provenance is required")
    receipt = json.loads(candidate.with_suffix(".training.json").read_text(encoding="utf-8"))
    candidate_sha = hashlib.sha256(candidate.read_bytes()).hexdigest()
    base_sha = hashlib.sha256(base_model.read_bytes()).hexdigest()
    training_identity = digest({name: hashlib.sha256((training_data.parent/name).read_bytes()).hexdigest()
                                for name in (training_data.name, "manifest.json", "dataset_split.json")})
    if (receipt.get("status") != "CANDIDATE_EXPORTED_PENDING_QUALITY" or receipt.get("model_sha256") != candidate_sha
            or receipt.get("base_model_sha256") != base_sha or receipt.get("dataset_identity") != training_identity):
        raise ValueError("Candidate differs from its training attempt or supplied training/base inputs")
    manifest = json.loads((validation_data.parent / "manifest.json").read_text(encoding="utf-8"))
    validation = [r for r in manifest if r["split"] == "test"]
    training = [r for r in json.loads((training_data.parent / "manifest.json").read_text(encoding="utf-8")) if r["split"] != "test"]
    def source_members(path, splits):
        protocol = json.loads((path.parent / "dataset_split.json").read_text(encoding="utf-8"))
        return {member for group in protocol["groups"] if group["split"] in splits for member in group["members"]}
    if (source_members(training_data, {"train", "val"}) & source_members(validation_data, {"test"})
            or {r["image_sha256"] for r in training} & {r["image_sha256"] for r in validation}):
        raise ValueError("Test sources/dates/images overlap training or tuning")
    target_classes = set()
    for record in validation:
        for line in (validation_data.parent / record["label_file"]).read_text(encoding="utf-8").splitlines():
            if line.strip():
                target_classes.add(int(line.split()[0]))
    if 0 not in target_classes:
        raise ValueError("Test set has no reviewed person targets; person quality is unavailable")
    def identity(path):
        return digest({name: hashlib.sha256((path.parent/name).read_bytes()).hexdigest()
                       for name in (path.name, "manifest.json", "dataset_split.json")})
    validation_identity = identity(validation_data)
    before = hashlib.sha256((validation_data.parent / "manifest.json").read_bytes()).hexdigest()
    evaluation_dir = candidate.parent / (candidate.name + ".evaluations") / uuid.uuid4().hex
    evaluation_dir.mkdir(parents=True, exist_ok=False)
    baseline = YOLO(str(base_model)).val(data=str(validation_data), split="test", verbose=False,
                                       project=str(evaluation_dir), name="baseline", exist_ok=False)
    updated = YOLO(str(candidate)).val(data=str(validation_data), split="test", verbose=False,
                                     project=str(evaluation_dir), name="candidate", exist_ok=False)
    def metrics(result):
        import math
        values = {}
        for index, class_id in enumerate(result.box.ap_class_index):
            if int(class_id) in target_classes:
                score = tuple(float(value) for value in result.box.class_result(index))
                if len(score) != 4 or not all(math.isfinite(value) for value in score):
                    raise ValueError("Test metrics unavailable or nonfinite")
                values[int(class_id)] = score
        if set(values) != target_classes:
            raise ValueError("Missing per-class test metrics")
        return values
    old, new = metrics(baseline), metrics(updated)
    passed = all(all(current >= previous for current, previous in zip(new[class_id], old[class_id])) for class_id in target_classes)
    if (not check_dataset_health(validation_data) or not check_dataset_health(training_data)
            or identity(training_data) != training_identity or identity(validation_data) != validation_identity
            or hashlib.sha256((validation_data.parent / "manifest.json").read_bytes()).hexdigest() != before
            or hashlib.sha256(candidate.read_bytes()).hexdigest() != candidate_sha
            or hashlib.sha256(base_model.read_bytes()).hexdigest() != base_sha):
        raise ValueError("Evaluation inputs changed")
    report = {"status": "TARGET_TEST_PASSED_PENDING_VIDEO_ACCEPTANCE" if passed else "TARGET_TEST_FAILED",
              "model_sha256": candidate_sha, "base_model_sha256": base_sha, "test_manifest_sha256": before,
              "test_images": len(validation), "explicit_empty_images": sum(record["no_targets"] for record in validation),
              "per_class_before": old, "per_class_after": new, "metrics": ["precision", "recall", "mAP50", "mAP50_95"],
              "quality_gate": "UNAVAILABLE", "deployment_authorized": False,
              "scope": "Selected target images only; metrics use the validator confidence sweep, not fixed production event thresholds or full-video action/infant/audio acceptance"}
    evidence = evaluation_dir / "evaluation.json"
    report["evidence_path"] = str(evidence.resolve())
    atomic_json(evidence, report)
    atomic_json(candidate.with_suffix(".evaluation.json"), report)
    logger.info("Independent target test passed=%s; full-video acceptance still required", passed)
    return passed


def main():
    parser = argparse.ArgumentParser(description="HomeVlog 专属机位 YOLO 小样本微调训练流")
    parser.add_argument("--data", type=Path, default=PROJECT_ROOT / "data" / "yolo_dataset" / "data.yaml", help="数据集 data.yaml 路径")
    parser.add_argument("--base-model", type=Path, default=PROJECT_ROOT / "models" / "yolo11n.pt", help="预训练基底权重路径")
    parser.add_argument("--output-model", type=Path, default=PROJECT_ROOT / "models" / "yolo11_custom.pt", help="微调输出权重保存路径")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数 (默认 30)")
    parser.add_argument("--batch", type=int, default=16, help="批次大小 (默认 16)")
    parser.add_argument("--freeze", type=int, choices=[10], default=10, help="冻结主干特征层数 (强制 10)")
    parser.add_argument("--device", type=str, default="0", help="训练设备 (0 或 cpu)")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "settings.yaml", help="配置文件路径")
    parser.add_argument("--validation-data", type=Path, help="--evaluate-model所需的已核验test数据集")
    parser.add_argument("--apply", action="store_true", help="已停用：候选模型须另做独立原片验收后再推广")
    parser.add_argument("--evaluate-model", type=Path, help="仅评估已生成的候选模型，使用独立test分组，不训练或写生产配置")
    parser.add_argument("--dry-run", action="store_true", help="仅检查数据集与基底文件，不加载模型或启动训练")

    args = parser.parse_args()
    if args.apply:
        parser.error("--apply is disabled: target-image metrics do not establish full-video acceptance; keep production config unchanged")
    if args.evaluate_model:
        if not args.validation_data:
            parser.error("--evaluate-model requires --validation-data")
        if not evaluate_candidate(args.base_model, args.evaluate_model, args.validation_data, args.data):
            raise SystemExit(1)
        return

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
        raise SystemExit(1)

    if not args.base_model.is_file() or args.base_model.stat().st_size == 0:
        print(f"[!] 找不到预训练基底权重: {args.base_model}")
        raise SystemExit(1)

    if args.dry_run:
        print("[*] Dry-run 模式：数据集与本地基底文件检查通过，未加载模型或执行训练。")
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

    else:
        print("\n[!] 训练未完成或未生成有效权重。\n")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
