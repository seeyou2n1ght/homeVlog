"""tests/test_yolo_finetune.py

测试专属机位难样本数据集自动导出 (Ultralytics YOLO 标准格式)、
正负样本分层抽样、空标注负样本生成、骨干冻结微调训练配置与模型热切换。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from scripts.export_dataset import (
    DatasetItem,
    build_yolo_dataset,
    generate_yolo_labels_for_item,
)
from scripts.train_yolo import (
    apply_custom_model_to_config,
    check_dataset_health,
    run_training,
)


def test_export_dataset_structure_and_negative_labels(tmp_path):
    dataset_dir = tmp_path / "test_yolo_dataset"
    archive_dir = tmp_path / "mock_archive"
    archive_dir.mkdir(parents=True, exist_ok=True)

    # 构造 5 个样本：3 个假阳性误报(困难负样本)，2 个漏检(困难正样本)
    items = []
    for i in range(3):
        img_p = archive_dir / f"frame_neg_{i}.jpg"
        img_p.write_bytes(b"\xff\xd8\xff" + b"\x00" * 20)
        items.append(
            DatasetItem(
                image_path=img_p,
                manual_label="FALSE_ALARM",
                segment_id=i,
                source_video=f"video_{i}.mp4",
                timestamp=float(i * 10),
            )
        )

    for i in range(2):
        img_p = archive_dir / f"frame_pos_{i}.jpg"
        img_p.write_bytes(b"\xff\xd8\xff" + b"\x00" * 20)
        items.append(
            DatasetItem(
                image_path=img_p,
                manual_label="MISSED_MOTION",
                segment_id=10 + i,
                source_video=f"video_pos_{i}.mp4",
                timestamp=float(i * 10),
            )
        )

    # 执行数据集构建
    stats = build_yolo_dataset(
        items=items,
        output_dir=dataset_dir,
        val_ratio=0.2,
        yolo_model_path=None,
    )

    # 1. 验证目录规范
    assert (dataset_dir / "images" / "train").exists()
    assert (dataset_dir / "images" / "val").exists()
    assert (dataset_dir / "labels" / "train").exists()
    assert (dataset_dir / "labels" / "val").exists()

    total_images = len(list((dataset_dir / "images" / "train").glob("*.jpg"))) + \
                   len(list((dataset_dir / "images" / "val").glob("*.jpg")))
    assert total_images == 5

    # 2. 验证困难负样本 (FALSE_ALARM) 生成空 txt 标注
    # 查找任一负样本 label 文件
    neg_label_files = list((dataset_dir / "labels" / "train").glob("*neg*.txt")) + \
                      list((dataset_dir / "labels" / "val").glob("*neg*.txt"))
    assert len(neg_label_files) == 3
    for nf in neg_label_files:
        assert nf.read_text(encoding="utf-8").strip() == ""  # 负样本标注为空

    # 3. 验证 data.yaml 配置
    data_yaml = dataset_dir / "data.yaml"
    assert data_yaml.exists()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert cfg["train"] == "images/train"
    assert cfg["val"] == "images/val"
    assert cfg["names"][0] == "person"
    assert cfg["names"][1] == "pet"


def test_generate_yolo_labels_with_mock_model(tmp_path):
    txt_path = tmp_path / "label.txt"
    img_path = tmp_path / "dummy.jpg"
    img_path.write_bytes(b"\xff\xd8\xff")

    # 1. 测试 FALSE_ALARM 返回 0 目标并清空文件
    item_neg = DatasetItem(image_path=img_path, manual_label="FALSE_ALARM", segment_id=1, source_video="", timestamp=0.0)
    boxes_count = generate_yolo_labels_for_item(item_neg, txt_path, yolo_model=None)
    assert boxes_count == 0
    assert txt_path.read_text(encoding="utf-8") == ""

    # 2. 测试带有预测框的 MISSED_MOTION
    mock_box = MagicMock()
    mock_box.cls = [MagicMock(item=lambda: 0)]  # person
    mock_box.xywhn = [MagicMock(cpu=lambda: MagicMock(numpy=lambda: [0.5, 0.5, 0.2, 0.4]))]

    mock_res = MagicMock()
    mock_res.boxes = [mock_box]

    mock_model = MagicMock()
    mock_model.return_value = [mock_res]

    with patch("cv2.imread", return_value=MagicMock(shape=(480, 640, 3))):
        item_pos = DatasetItem(image_path=img_path, manual_label="MISSED_MOTION", segment_id=2, source_video="", timestamp=0.0)
        n = generate_yolo_labels_for_item(item_pos, txt_path, yolo_model=mock_model)
        assert n == 1
        line = txt_path.read_text(encoding="utf-8").strip()
        assert line.startswith("0 0.500000 0.500000 0.200000 0.400000")


def test_check_dataset_health(tmp_path):
    invalid_yaml = tmp_path / "data.yaml"
    assert check_dataset_health(invalid_yaml) is False

    # 创建合法数据集目录
    dataset_dir = tmp_path / "valid_ds"
    train_dir = dataset_dir / "images" / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "sample.jpg").write_bytes(b"image")
    valid_yaml = dataset_dir / "data.yaml"
    valid_yaml.write_text("train: images/train\nval: images/val", encoding="utf-8")

    assert check_dataset_health(valid_yaml) is True


def test_apply_custom_model_to_config(tmp_path):
    cfg_path = tmp_path / "settings.yaml"
    cfg_content = """
detection:
  prescreen_diff_threshold: 8.0
yolo:
  enabled: false
  model_path: models/yolo11n.pt
"""
    cfg_path.write_text(cfg_content, encoding="utf-8")
    new_model_path = tmp_path / "models" / "yolo11_custom.pt"

    ok = apply_custom_model_to_config(cfg_path, new_model_path)
    assert ok is True
    assert (tmp_path / "settings.yaml.bak").exists()

    updated = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert updated["yolo"]["enabled"] is True
    assert "yolo11_custom.pt" in updated["yolo"]["model_path"]


def test_run_training_mock(tmp_path):
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("dummy", encoding="utf-8")
    base_model = tmp_path / "yolo11n.pt"
    base_model.write_bytes(b"dummy_weights")
    output_model = tmp_path / "yolo11_custom.pt"

    mock_yolo_instance = MagicMock()
    mock_train_res = MagicMock()
    mock_yolo_instance.train.return_value = mock_train_res

    # 模拟生成的 best.pt
    mock_best = tmp_path / "runs" / "train" / "homevlog_custom" / "weights" / "best.pt"
    mock_best.parent.mkdir(parents=True, exist_ok=True)
    mock_best.write_bytes(b"best_weights")

    with patch("src.utils.PROJECT_ROOT", tmp_path), \
         patch("scripts.train_yolo.PROJECT_ROOT", tmp_path), \
         patch("ultralytics.YOLO", return_value=mock_yolo_instance):
        out = run_training(
            data_yaml=data_yaml,
            base_model=base_model,
            output_model=output_model,
            epochs=2,
            batch=8,
            freeze=10,
            device="cpu",
        )
        assert out is not None
        assert out.exists()
        assert out.read_bytes() == b"best_weights"
        assert mock_yolo_instance.train.called
        call_kwargs = mock_yolo_instance.train.call_args[1]
        assert call_kwargs["freeze"] == 10
        assert call_kwargs["batch"] == 8
        assert call_kwargs["amp"] is True
