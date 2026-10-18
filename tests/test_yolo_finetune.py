"""tests/test_yolo_finetune.py

测试人工目标来源、隔离三分组、训练产物及候选验收边界。
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
    evaluate_candidate,
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
    assert total_images == 0
    assert stats["excluded"] == 5  # Event labels and automatic boxes are not target truth.
    assert stats["usable"] is False

    # 2. 验证困难负样本 (FALSE_ALARM) 生成空 txt 标注
    # 查找任一负样本 label 文件
    neg_label_files = list((dataset_dir / "labels" / "train").glob("*neg*.txt")) + \
                      list((dataset_dir / "labels" / "val").glob("*neg*.txt"))
    assert len(neg_label_files) == 0
    for nf in neg_label_files:
        assert nf.read_text(encoding="utf-8").strip() == ""  # 负样本标注为空

    # 3. 验证 data.yaml 配置
    data_yaml = dataset_dir / "data.yaml"
    assert data_yaml.exists()
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert cfg["train"] == "images/train"
    assert cfg["val"] == "images/val"
    assert cfg["names"][0] == "person"
    assert cfg["names"][15] == "cat"
    assert cfg["names"][16] == "dog"


def test_generate_yolo_labels_with_mock_model(tmp_path):
    txt_path = tmp_path / "label.txt"
    img_path = tmp_path / "dummy.jpg"
    img_path.write_bytes(b"\xff\xd8\xff")

    # 1. 测试 FALSE_ALARM 返回 0 目标并清空文件
    item_neg = DatasetItem(image_path=img_path, manual_label="FALSE_ALARM", segment_id=1, source_video="", timestamp=0.0)
    boxes_count = generate_yolo_labels_for_item(item_neg, txt_path, yolo_model=None)
    assert boxes_count == -1
    assert not txt_path.exists()

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
        assert n == -1
        mock_model.assert_not_called()
        assert not txt_path.exists()


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

    assert check_dataset_health(valid_yaml) is False  # Legacy/unproven exports cannot train.


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
    assert apply_custom_model_to_config(cfg_path, new_model_path) is False
    assert cfg_path.read_text(encoding="utf-8") == cfg_content
    new_model_path.parent.mkdir()
    new_model_path.write_bytes(b"controlled candidate weights")
    ok = apply_custom_model_to_config(cfg_path, new_model_path)
    assert ok is True
    assert len(list(tmp_path.glob("settings.yaml.bak_*"))) == 1

    updated = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert updated["yolo"]["enabled"] is True
    assert "yolo11_custom.pt" in updated["yolo"]["model_path"]


def test_run_training_mock(tmp_path,verified_split_items):
    build_yolo_dataset(verified_split_items,tmp_path/"dataset")
    data_yaml = tmp_path / "dataset/data.yaml"
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
    def produce_fresh(**kwargs):
        fresh=Path(kwargs['project'])/kwargs['name']/'weights/best.pt'
        fresh.parent.mkdir(parents=True);fresh.write_bytes(b'fresh_weights')
        return mock_train_res
    mock_yolo_instance.train.side_effect=produce_fresh

    with patch("src.utils.PROJECT_ROOT", tmp_path), \
         patch("scripts.train_yolo.PROJECT_ROOT", tmp_path), \
         patch("ultralytics.YOLO", return_value=mock_yolo_instance):
        out = run_training(
            data_yaml=data_yaml,
            base_model=base_model,
            output_model=output_model,
            epochs=2,
            batch=8,
            freeze=5,
            device="cpu",
        )
        assert out is not None
        assert out.exists()
        assert out.read_bytes() == b"fresh_weights"
        assert mock_best.read_bytes() == b"best_weights"
        assert mock_yolo_instance.train.called
        call_kwargs = mock_yolo_instance.train.call_args[1]
        assert call_kwargs["freeze"] == 5
        assert call_kwargs["batch"] == 8
        assert call_kwargs['exist_ok'] is False
        assert call_kwargs["amp"] is True
        import json
        receipt=json.loads((Path(call_kwargs['project'])/'training_receipt.json').read_text())
        assert receipt['status']=='CANDIDATE_EXPORTED_PENDING_QUALITY'
        assert receipt['quality_gate']=='UNAVAILABLE'


@pytest.mark.parametrize('failure',['stale','changed_data','publish_conflict','interrupt'])
def test_training_never_reuses_stale_or_overwrites_output(verified_split_items,tmp_path,monkeypatch,failure):
    import json
    from scripts import train_yolo
    monkeypatch.setattr(train_yolo,'PROJECT_ROOT',tmp_path)
    dataset=tmp_path/'dataset';build_yolo_dataset(verified_split_items,dataset)
    base=tmp_path/'base.pt';base.write_bytes(b'base fixture')
    output=tmp_path/'candidate.pt'
    old=tmp_path/'runs/train/homevlog_custom/weights/best.pt';old.parent.mkdir(parents=True);old.write_bytes(b'old unrelated model')
    model=MagicMock()
    def train(**kwargs):
        if failure=='interrupt':raise KeyboardInterrupt()
        if failure!='stale':
            fresh=Path(kwargs['project'])/kwargs['name']/'weights/best.pt';fresh.parent.mkdir(parents=True);fresh.write_bytes(b'current weights')
        if failure=='changed_data':(dataset/'manifest.json').write_text('[]')
        if failure=='publish_conflict':output.write_bytes(b'other writer model')
    model.train.side_effect=train
    with patch('ultralytics.YOLO',return_value=model):
        if failure=='interrupt':
            with pytest.raises(KeyboardInterrupt):run_training(dataset/'data.yaml',base,output,device='cpu')
        else:assert run_training(dataset/'data.yaml',base,output,device='cpu') is None
    assert old.read_bytes()==b'old unrelated model'
    if failure=='publish_conflict':assert output.read_bytes()==b'other writer model'
    else:assert not output.exists()
    receipt=json.loads(next((tmp_path/'runs/train').glob('*/training_receipt.json')).read_text())
    assert receipt['status']==('INTERRUPTED' if failure=='interrupt' else 'FAILED')
    assert not list(tmp_path.glob('*.pending'))


@pytest.fixture
def target_review_item(tmp_path):
    import cv2
    import numpy as np
    import json
    import hashlib
    source = tmp_path / "source.mp4"
    source.write_bytes(b"captured source identity")
    image = tmp_path / "reviewed.jpg"
    cv2.imwrite(str(image), np.zeros((24,32,3),np.uint8))
    stat = source.stat()
    item = DatasetItem(image, "FALSE_ALARM", 1, str(source), 12.5)
    review = {"schema":1,"label_source":"human_target_review","all_targets_reviewed":True,
              "reviewed_at":"2026-10-12T14:00:00+08:00","image_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
              "source_video":str(source),"source_identity":[str(source.resolve()),stat.st_size,stat.st_mtime_ns],
              "frame_timestamp":12.5,"no_targets":False,
              "boxes":[{"class_id":0,"x_center":.5,"y_center":.5,"width":.2,"height":.4}]}
    def write_review():
        image.with_suffix(".targets.json").write_text(json.dumps(review),encoding="utf-8")
    write_review()
    return item,review,write_review


def test_event_false_alarm_can_have_verified_person_boxes(target_review_item,tmp_path):
    item,review,write_review = target_review_item
    target=tmp_path/"export.txt"
    # Legacy sidecars and event negatives do not override explicit object review.
    item.image_path.with_suffix(".txt").write_text("",encoding="utf-8")
    assert generate_yolo_labels_for_item(item,target)==1
    assert target.read_text().startswith("0 0.500000 0.500000 0.200000 0.400000")
    review.update(no_targets=True,boxes=[])
    write_review()
    assert generate_yolo_labels_for_item(item,target)==0
    assert target.read_bytes()==b""


@pytest.mark.parametrize("mutation", ["image","source","timestamp","automatic","partial","empty","bounds","class","nan","tiny"])
def test_target_review_mismatch_excluded_without_overwriting(target_review_item,tmp_path,mutation):
    item,review,write_review=target_review_item
    target=tmp_path/"previous.txt";target.write_text("preserved")
    if mutation=="image":item.image_path.write_bytes(b"changed image")
    elif mutation=="source":review["source_video"]="another.mp4"
    elif mutation=="timestamp":review["frame_timestamp"]+=.5
    elif mutation=="automatic":review["label_source"]="algorithm"
    elif mutation=="partial":review["all_targets_reviewed"]=False
    elif mutation=="empty":review["boxes"]=[]
    elif mutation=="bounds":review["boxes"][0]["x_center"]=.99
    elif mutation=="class":review["boxes"][0]["class_id"]=3
    elif mutation=="nan":review["boxes"][0]["width"]=float("nan")
    elif mutation=="tiny":review["boxes"][0]["width"]=1e-9
    write_review()
    model=MagicMock()
    assert generate_yolo_labels_for_item(item,target,model)==-1
    assert target.read_text()=="preserved"
    model.assert_not_called()


def test_export_uses_target_counts_and_keeps_source_groups(target_review_item,tmp_path):
    import json
    import hashlib
    import cv2
    import numpy as np
    item,review,_=target_review_item
    items=[]
    for i in range(5):
        image=tmp_path/f"frame_{i}.jpg"
        cv2.imwrite(str(image),np.full((24,32,3),i,np.uint8))
        source=str(tmp_path/f"source_{i//2}.mp4")
        entry=DatasetItem(image,"CONFIRMED_STATIC",i,source,float(i))
        record=dict(review,source_video=source,source_identity=[str(Path(source).resolve()),10,100],
                    frame_timestamp=float(i),image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
        if i==4:record.update(boxes=[],no_targets=True)
        image.with_suffix(".targets.json").write_text(json.dumps(record),encoding="utf-8")
        items.append(entry)
    out=tmp_path/"export"
    stats=build_yolo_dataset(items,out)
    assert stats["positives"]==4 and stats["negatives"]==1 and stats["excluded"]==0
    assert stats["usable"] is True
    assert check_dataset_health(out/"data.yaml") is True
    manifest=json.loads((out/"manifest.json").read_text())
    grouped={}
    for entry in manifest:
        grouped.setdefault(entry["source_video"],set()).add(entry["split"])
        assert entry["label_source"]=="human_target_review"
        assert entry["all_targets_reviewed"] is True
    assert all(len(splits)==1 for splits in grouped.values())


def test_training_rejects_modified_or_unlisted_export_files(verified_split_items,tmp_path):
    item=verified_split_items[0]
    out=tmp_path/"dataset"
    build_yolo_dataset(verified_split_items,out)
    assert check_dataset_health(out/"data.yaml") is True
    label=next((out/"labels/train").glob("*.txt"))
    previous=label.read_bytes();label.write_bytes(b"")
    assert check_dataset_health(out/"data.yaml") is False
    label.write_bytes(previous)
    extra=out/"images/train/unlisted.jpg";extra.write_bytes(item.image_path.read_bytes())
    assert check_dataset_health(out/"data.yaml") is False


def test_training_rejects_yaml_redirect_and_unreviewed_programmatic_input(target_review_item,tmp_path):
    item,_,_=target_review_item
    out=tmp_path/"verified";build_yolo_dataset([item],out)
    cfg=yaml.safe_load((out/"data.yaml").read_text(encoding="utf-8"));cfg["path"]=str(tmp_path/"other")
    (out/"data.yaml").write_text(yaml.safe_dump(cfg),encoding="utf-8")
    assert check_dataset_health(out/"data.yaml") is False
    with patch("ultralytics.YOLO") as model:
        assert run_training(out/"data.yaml",tmp_path/"base.pt",tmp_path/"out.pt",device="cpu") is None
        model.assert_not_called()


@pytest.fixture
def verified_split_items(target_review_item, tmp_path):
    import json
    import hashlib
    import cv2
    import numpy as np
    _, review, _ = target_review_item
    items=[]
    for i in range(3):
        image=tmp_path/f"split_{i}.jpg"
        cv2.imwrite(str(image),np.full((24,32,3),30+i*50,np.uint8))
        source=tmp_path/f"00_202610{10+i}100000_202610{10+i}101000.mp4"
        source.write_bytes(f"controlled source {i}".encode())
        stat=source.stat()
        record=dict(review,source_video=str(source),source_identity=[str(source),stat.st_size,stat.st_mtime_ns],
                    image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
        image.with_suffix('.targets.json').write_text(json.dumps(record),encoding='utf-8')
        items.append(DatasetItem(image,'CONFIRMED_STATIC',i,str(source),12.5))
    return items


def test_dataset_three_splits_deterministic_and_sparse_groups_rejected(verified_split_items,tmp_path):
    import json
    first=tmp_path/'first';second=tmp_path/'second'
    stats=build_yolo_dataset(verified_split_items,first)
    build_yolo_dataset(list(reversed(verified_split_items)),second)
    assert stats['train_images']==stats['val_images']==stats['test_images']==1
    assert stats['source_groups']==3 and stats['usable']
    assert json.loads((first/'manifest.json').read_text())==json.loads((second/'manifest.json').read_text())
    assert (first/'dataset_split.json').read_bytes()==(second/'dataset_split.json').read_bytes()
    assert check_dataset_health(first/'data.yaml')
    few=tmp_path/'few';stats=build_yolo_dataset(verified_split_items[:2],few)
    assert not stats['usable'] and stats['test_images']==0
    with patch('ultralytics.YOLO') as model:
        assert run_training(few/'data.yaml',tmp_path/'base.pt',tmp_path/'out.pt',device='cpu') is None
        model.assert_not_called()


def test_dataset_same_date_and_duplicate_images_link_sources(verified_split_items,tmp_path):
    import json
    import hashlib
    import cv2
    import numpy as np
    items=list(verified_split_items)
    base=items[0]
    def extra(name,source,pixels,boxes=None):
        image=tmp_path/name
        if pixels is None: image.write_bytes(base.image_path.read_bytes())
        else: cv2.imwrite(str(image),np.full((24,32,3),pixels,np.uint8))
        record=json.loads(base.image_path.with_suffix('.targets.json').read_text())
        record.update(source_video=source,source_identity=[str(Path(source).resolve()),10,100],image_sha256=hashlib.sha256(image.read_bytes()).hexdigest())
        if boxes is not None: record.update(boxes=boxes,no_targets=not boxes)
        image.with_suffix('.targets.json').write_text(json.dumps(record),encoding='utf-8')
        return DatasetItem(image,'FALSE_ALARM',10,source,12.5)
    same_date=extra('other.jpg',str(tmp_path/'00_20261010110000_20261010111000.mp4'),220)
    items.append(same_date)
    duplicate=extra('duplicate.jpg',str(tmp_path/'00_20261011110000_20261011111000.mp4'),None)
    items.append(duplicate)
    out=tmp_path/'linked';stats=build_yolo_dataset(items,out)
    assert stats['duplicates']==1 and stats['source_groups']==2 and not stats['usable']
    records=json.loads((out/'manifest.json').read_text())
    assert len({r['split'] for r in records if '20261010' in r['source_video'] or '20261011' in r['source_video']})==1
    conflict=extra('conflict.jpg',str(tmp_path/'fourth.mp4'),None,[])
    conflict_out=tmp_path/'conflicted';stats=build_yolo_dataset([*verified_split_items,conflict],conflict_out)
    assert stats['conflicts']==2
    assert all(r['image_sha256']!=hashlib.sha256(base.image_path.read_bytes()).hexdigest() for r in json.loads((conflict_out/'manifest.json').read_text()))


def test_training_rejects_split_reassignment(verified_split_items,tmp_path):
    import json
    out=tmp_path/'dataset';build_yolo_dataset(verified_split_items,out)
    records=json.loads((out/'manifest.json').read_text())
    records[0]['source_group']=records[1]['source_group']
    (out/'manifest.json').write_text(json.dumps(records),encoding='utf-8')
    assert not check_dataset_health(out/'data.yaml')


def test_archive_loader_uses_latest_bound_frame_record(target_review_item,tmp_path):
    import json
    from scripts.export_dataset import load_archive_items
    item, _, _ = target_review_item
    archive=tmp_path/'archive';(archive/'images').mkdir(parents=True)
    records=[{'segment_id':item.segment_id,'archived_image_path':str(item.image_path),'filepath':item.source_video,
              'manual_label':'FALSE_ALARM','frame_timestamp':old} for old in (10.,item.timestamp)]
    (archive/'manifest.jsonl').write_text('\n'.join(json.dumps(record) for record in [records[0],[],records[1]]),encoding='utf-8')
    loaded=load_archive_items(archive)
    assert len(loaded)==1 and loaded[0].timestamp==item.timestamp
    from scripts.export_dataset import verified_target_annotation
    assert verified_target_annotation(loaded[0]) is not None


@pytest.fixture
def evaluation_inputs(verified_split_items,tmp_path):
    import hashlib
    import json
    from src.core.replay_journal import digest
    root=tmp_path/'evaluation';build_yolo_dataset(verified_split_items,root)
    base=tmp_path/'base.pt';base.write_bytes(b'controlled base')
    candidate=tmp_path/'candidate.pt';candidate.write_bytes(b'controlled candidate')
    data=root/'data.yaml'
    receipt={'status':'CANDIDATE_EXPORTED_PENDING_QUALITY',
             'model_sha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
             'base_model_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),
             'dataset_identity':digest({name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                                        for name in ('data.yaml','manifest.json','dataset_split.json')})}
    candidate.with_suffix('.training.json').write_text(json.dumps(receipt))
    return base,candidate,data


@pytest.mark.parametrize('score,passed',[(.6,True),(.49,False)])
def test_candidate_test_metrics_never_authorize_deployment(evaluation_inputs,score,passed):
    import json
    base,candidate,data=evaluation_inputs
    def result(value):
        metrics=MagicMock();metrics.box.ap_class_index=[0]
        metrics.box.class_result.return_value=(value, .6, .6, .6)
        return metrics
    with patch('ultralytics.YOLO') as model:
        model.return_value.val.side_effect=[result(.5),result(score)]
        assert evaluate_candidate(base,candidate,data,data) is passed
        assert all(call.kwargs['split']=='test' for call in model.return_value.val.call_args_list)
    report=json.loads(candidate.with_suffix('.evaluation.json').read_text())
    assert report['quality_gate']=='UNAVAILABLE' and report['deployment_authorized'] is False
    assert report['test_images']==1
    assert json.loads(Path(report['evidence_path']).read_text())==report


def test_candidate_changed_receipt_inputs_rejected(evaluation_inputs):
    base,candidate,data=evaluation_inputs
    candidate.write_bytes(b'changed')
    with patch('ultralytics.YOLO') as model,pytest.raises(ValueError,match='differs'):
        evaluate_candidate(base,candidate,data,data)
    model.assert_not_called()


def test_candidate_missing_class_metrics_rejected(evaluation_inputs):
    base,candidate,data=evaluation_inputs
    result=MagicMock();result.box.ap_class_index=[]
    with patch('ultralytics.YOLO') as model,pytest.raises(ValueError,match='Missing'):
        model.return_value.val.return_value=result
        evaluate_candidate(base,candidate,data,data)
    assert not candidate.with_suffix('.evaluation.json').exists()


def test_training_cli_apply_disabled(monkeypatch):
    from scripts.train_yolo import main
    monkeypatch.setattr('sys.argv',['train_yolo.py','--apply'])
    with patch('scripts.train_yolo.run_training') as train,pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2
    train.assert_not_called()


def test_candidate_source_overlap_rejected(evaluation_inputs,verified_split_items,tmp_path):
    import json
    base,candidate,data=evaluation_inputs
    original=json.loads((data.parent/'manifest.json').read_text())
    training_sources={r['source_video'] for r in original if r['split']!='test'}
    for seed in range(10):
        other=tmp_path/f'validation_{seed}'
        build_yolo_dataset(verified_split_items,other,seed=seed)
        rows=json.loads((other/'manifest.json').read_text())
        if any(r['split']=='test' and r['source_video'] in training_sources for r in rows):
            break
    else:
        pytest.fail('Controlled alternate split did not exercise source overlap')
    with patch('ultralytics.YOLO') as model,pytest.raises(ValueError,match='overlap'):
        evaluate_candidate(base,candidate,other/'data.yaml',data)
    model.assert_not_called()


def test_candidate_person_regression_cannot_hide_in_other_class(evaluation_inputs):
    import json
    import hashlib
    from src.core.replay_journal import digest
    base,candidate,data=evaluation_inputs
    records=json.loads((data.parent/'manifest.json').read_text())
    test=next(r for r in records if r['split']=='test')
    label=data.parent/test['label_file']
    label.write_text(label.read_text()+'15 0.5 0.5 0.2 0.4\n')
    test['label_sha256']=hashlib.sha256(label.read_bytes()).hexdigest()
    (data.parent/'manifest.json').write_text(json.dumps(records))
    receipt=json.loads(candidate.with_suffix('.training.json').read_text())
    receipt['dataset_identity']=digest({name:hashlib.sha256((data.parent/name).read_bytes()).hexdigest()
                                        for name in ('data.yaml','manifest.json','dataset_split.json')})
    candidate.with_suffix('.training.json').write_text(json.dumps(receipt))
    old=MagicMock();old.box.ap_class_index=[0,15];old.box.class_result.side_effect=[(.5,)*4,(.1,)*4]
    new=MagicMock();new.box.ap_class_index=[0,15];new.box.class_result.side_effect=[(.49,)*4,(.9,)*4]
    with patch('ultralytics.YOLO') as model:
        model.return_value.val.side_effect=[old,new]
        assert not evaluate_candidate(base,candidate,data,data)


def test_candidate_input_changes_during_evaluation_rejected(evaluation_inputs):
    base,candidate,data=evaluation_inputs
    result=MagicMock();result.box.ap_class_index=[0];result.box.class_result.return_value=(.5,)*4
    calls=0
    def change_config(**kwargs):
        nonlocal calls
        calls+=1
        if calls==2:
            data.write_text(data.read_text()+'\n# changed during evaluation\n')
        return result
    with patch('ultralytics.YOLO') as model,pytest.raises(ValueError,match='changed'):
        model.return_value.val.side_effect=change_config
        evaluate_candidate(base,candidate,data,data)
    assert not candidate.with_suffix('.evaluation.json').exists()
