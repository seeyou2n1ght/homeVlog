"""测试基准数据集管理引擎与 What-If 调参仿真器 (Benchmark & Tuner Test Suite)"""

import json
from pathlib import Path
import pytest
import yaml

from src.core.benchmark import (
    load_benchmark_dataset,
    save_benchmark_dataset,
    merge_benchmark_samples,
    evaluate_confusion_matrix,
    evaluate_stratified_metrics,
    normalize_benchmark_label,
)
from src.core.tuner import (
    TuningSimulator,
    generate_config_diff,
    apply_config_safe,
    rollback_config,
)


@pytest.fixture
def sample_benchmark():
    return [
        {
            "segment_id": 1,
            "filepath": "cam1/clip1.mp4",
            "start_time": 0.0,
            "end_time": 10.0,
            "duration": 10.0,
            "algo_state": "DYNAMIC",
            "max_energy": 25.0,
            "avg_confidence": 0.85,
            "gold_label": "CONFIRMED_MOTION",
            "scenario": "ADULT_MOTION",
        },
        {
            "segment_id": 2,
            "filepath": "cam1/clip2.mp4",
            "start_time": 10.0,
            "end_time": 15.0,
            "duration": 5.0,
            "algo_state": "DYNAMIC",
            "max_energy": 3.0,
            "avg_confidence": 0.0,
            "gold_label": "FALSE_ALARM",
            "scenario": "LIGHT_SHADOW",
        },
        {
            "segment_id": 3,
            "filepath": "cam1/clip3.mp4",
            "start_time": 20.0,
            "end_time": 25.0,
            "duration": 5.0,
            "algo_state": "STATIC",
            "max_energy": 2.2,
            "avg_confidence": 0.0,
            "gold_label": "MISSED_MOTION",
            "scenario": "INFANT_MOTION",
        },
        {
            "segment_id": 4,
            "filepath": "cam1/clip4.mp4",
            "start_time": 30.0,
            "end_time": 40.0,
            "duration": 10.0,
            "algo_state": "STATIC",
            "max_energy": 0.5,
            "avg_confidence": 0.0,
            "gold_label": "CONFIRMED_STATIC",
            "scenario": "EMPTY_ROOM",
        },
    ]


def test_label_normalization():
    assert normalize_benchmark_label("tp") == "CONFIRMED_MOTION"
    assert normalize_benchmark_label("false_alarm") == "FALSE_ALARM"
    assert normalize_benchmark_label("fn") == "MISSED_MOTION"
    assert normalize_benchmark_label("tn") == "CONFIRMED_STATIC"
    assert normalize_benchmark_label("light_shadow") == "FALSE_ALARM"


def test_production_receipt_rejects_stale_code_model_source_and_configuration(tmp_path, monkeypatch):
    from src.core import config as core
    from src.core.replay_evidence import (ProductionReplayStore, ReplayUnavailable, content_sha256,
                                          production_code_sha256, replay_receipt, receipt_sha256)
    (tmp_path / "src").mkdir()
    code = tmp_path / "src/detector.py"
    code.write_text("original detector", encoding="utf-8")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts/replay_benchmark.py").write_text("producer", encoding="utf-8")
    monkeypatch.setattr(core, "PROJECT_ROOT", tmp_path)
    model = tmp_path / "model.pt"
    model.write_bytes(b"original model")
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original source")
    samples = [{"filepath": str(source), "date": "20260320", "cam_index": 0,
                "start_time": 0., "end_time": 10., "reference_active": True,
                "review_status": "independent_visual", "review_evidence": "controlled receipt fixture"}]
    config = {"yolo": {"model_path": str(model)}}
    stat = source.stat()
    receipt = replay_receipt(config, samples, [{"filepath": str(source), "size": stat.st_size,
                            "mtime_ns": stat.st_mtime_ns, "sha256": content_sha256(source)}],
                            [{"filepath": str(source), "analysis_status": "ANALYZED"}],
                            [{"filepath": str(source), "start_time": 0., "end_time": 10., "state": "STATIC"}],
                            code_sha256=production_code_sha256(), model_sha256=content_sha256(model))
    receipt["payload_sha256"] = receipt_sha256(receipt)
    store = ProductionReplayStore(tmp_path / "store")
    job = store.directory / "job"
    job.mkdir(parents=True)
    (job / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert store(config, samples) == [False]
    simulator = TuningSimulator(samples, config, replay=store)
    matches = []
    original_match = store.matching_receipt
    def match(config, samples):
        matches.append(True)
        return original_match(config, samples)
    monkeypatch.setattr(store, "matching_receipt", match)
    result = simulator.simulate({})
    assert len(matches) == 1  # retention and display share one source/model validation
    assert result["baseline"]["fn"] == 1
    assert result["sample_changes"][0]["baseline_retained"] is False
    assert result["sample_changes"][0]["candidate_retained"] is False
    assert result["sample_changes"][0]["baseline_display"]["status"] == "UNAVAILABLE"
    assert result["baseline_receipt_sha256"] == receipt["payload_sha256"]
    assert simulator.simulate({"detection.min_motion_threshold": 3.5})["verdict"] == "UNAVAILABLE"
    for path, changed in ((code, b"changed detector"), (model, b"changed model"), (source, b"changed source")):
        original = path.read_bytes()
        original_stat = path.stat()
        path.write_bytes(changed)
        with pytest.raises(ReplayUnavailable):
            store(config, samples)
        path.write_bytes(original)
        import os
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
    assert store(config, samples) == [False]
    receipt["predictions"] = {key: True for key in receipt["predictions"]}
    (job / "receipt.json").write_text(json.dumps(receipt), encoding="utf-8")
    assert store.references() is None


def test_replay_projection_does_not_turn_missing_or_overlapping_coverage_into_a_negative():
    from src.core.replay_evidence import project_predictions, sample_key
    sample = {"filepath": "clip.mp4", "date": "20260320", "cam_index": 0, "start_time": 0., "end_time": 10.}
    task = {"filepath": "clip.mp4", "analysis_status": "ANALYZED"}
    segments = [{"filepath": "clip.mp4", "start_time": 0., "end_time": 6., "state": "STATIC"},
                {"filepath": "clip.mp4", "start_time": 4., "end_time": 8., "state": "DYNAMIC"}]
    # Summed durations equal ten, but overlap and missing tail invalidate coverage.
    assert project_predictions([sample], [task], segments)[sample_key(sample)] is None
    assert project_predictions([sample], [], segments)[sample_key(sample)] is None


def test_replay_retention_distinguishes_fast_activity_and_confirmed_static_prescreen():
    from src.core.replay_evidence import project_predictions, interval_diagnostics, sample_key
    sample = {"filepath": "clip.mp4", "date": "20260320", "cam_index": 0, "start_time": 0., "end_time": 10.}
    task = {"filepath": "clip.mp4", "analysis_status": "ANALYZED", "prescreen_status": "SUSPICIOUS",
            "file_start_time": "20260320000000", "file_end_time": "20260320000010", "file_duration": 10.}
    segments = [{"filepath": "clip.mp4", "start_time": 0., "end_time": 10., "state": "MICRO_MOTION"}]
    cfg = {"micro_motion": {"cruise_speed": 16}}
    assert project_predictions([sample], [task], segments)[sample_key(sample)] is True
    details = interval_diagnostics(cfg, [sample], [task], segments)[sample_key(sample)]
    assert details["retained_source_s"] == 10.
    assert details["normal_source_s"] == 0.
    assert details["display_s"] == .65  # shared integer-frame rounding at 20fps
    assert not details["normal_or_presence"]
    task.update(analysis_status="PENDING", prescreen_status="STATIC")
    assert project_predictions([sample], [task], [])[sample_key(sample)] is False
    sample["end_time"] = 11.
    assert project_predictions([sample], [task], [])[sample_key(sample)] is None


@pytest.mark.parametrize('basis,label,qualified', [
    ('unknown','CONFIRMED_MOTION',False), ('representative_frame','CONFIRMED_MOTION',False),
    ('uncertain','CONFIRMED_STATIC',False), ('video_interval','CONFIRMED_MOTION',True),
    ('video_interval','CONFIRMED_STATIC',False), ('video_audio_interval','CONFIRMED_MOTION',True),
    ('video_audio_interval','CONFIRMED_STATIC',True)])
def test_review_preparation_requires_declared_interval_scope(tmp_path, basis, label, qualified):
    from scripts.prepare_review_benchmark import prepare
    from scripts.audit_tool.service import pack_notes
    from src.core.database import VlogDatabase
    from src.algorithms.segment import Segment
    db = VlogDatabase(tmp_path/'basis.sqlite')
    source = tmp_path/'00_20260501000000_20260501000100.mp4'
    source.write_bytes(b'controlled identity, not media truth')
    try:
        db.add_file_task(str(source),0,'20260501','20260501000000','20260501000100',60)
        db.set_analysis_result(str(source),'ANALYZED',[Segment(start_time=0.,end_time=6.,state='STATIC',source_file=str(source))])
        row = db.get_all_segments_for_date('20260501',0)[0]
        assert db.update_segment_review(row['id'], label, pack_notes('declared fixture','UNKNOWN',basis))
        before = [tuple(r) for r in db.conn.execute('SELECT * FROM human_reviews')]
        report = prepare(db.db_path, tmp_path/'cohort','2000-01-01')
        sample = json.loads((tmp_path/'cohort/all.json').read_text(encoding='utf-8'))['samples'][0]
        assert sample['gold_label'] == label and sample['review_basis'] == basis
        assert sample['review_status'] == ('independent_human' if qualified else 'pending_confirmation')
        assert sample['reference_active'] == ((label=='CONFIRMED_MOTION') if qualified else None)
        assert sample['review_evidence']['full_interval_declared'] == (basis in {'video_interval', 'video_audio_interval'})
        assert report['qualified_references'] == int(qualified)
        assert [tuple(r) for r in db.conn.execute('SELECT * FROM human_reviews')] == before
    finally:
        db.close()


def test_review_preparation_locks_date_groups_and_preserves_database(tmp_path):
    from scripts.prepare_review_benchmark import prepare
    from src.core.database import VlogDatabase
    from src.algorithms.segment import Segment
    db = VlogDatabase(tmp_path / "reviews.db")
    try:
        for date in ("20260501", "20260502"):
            source = tmp_path / f"00_{date}000000_{date}000100.mp4"
            source.write_bytes(b"identity only; no media decoding required")
            db.add_file_task(str(source), 0, date, date+"000000", date+"000100", 60)
            db.set_analysis_result(str(source), "ANALYZED", [Segment(start_time=0.,end_time=1.,state="MICRO_MOTION",source_file=str(source)),
                                                            Segment(start_time=2.,end_time=3.,state="STATIC",source_file=str(source))])
            for row in db.get_all_segments_for_date(date, 0):
                assert db.update_segment_review(row["id"], "MISSED_MOTION" if row["state"] == "MICRO_MOTION" else "CONFIRMED_STATIC", "")
        before = [tuple(r) for r in db.conn.execute("SELECT * FROM human_reviews ORDER BY id")]
        result = prepare(db.db_path, tmp_path/"first", "2000-01-01")
        assert result["total"] == 4 and result["unknown_scenarios"] == 4
        assert result["qualified_references"] == 0 and result["pending_confirmation"] == 4
        data = json.loads((tmp_path/"first/all.json").read_text(encoding="utf-8"))["samples"]
        assert {s["split"] for s in data} == {"tuning", "holdout"}
        assert len({s["date"] for s in data if s["split"] == "tuning"} & {s["date"] for s in data if s["split"] == "holdout"}) == 0
        assert all(len({s["split"] for s in data if s["filepath"] == path}) == 1 for path in {s["filepath"] for s in data})
        assert [tuple(r) for r in db.conn.execute("SELECT * FROM human_reviews ORDER BY id")] == before
        prepare(db.db_path, tmp_path/"second", "2000-01-01")
        assert (tmp_path/"first/all.json").read_bytes() == (tmp_path/"second/all.json").read_bytes()
        with pytest.raises(FileExistsError):
            prepare(db.db_path, tmp_path/"first", "2000-01-01")
    finally:
        db.close()


@pytest.mark.parametrize("changed_review", [False, True])
def test_generated_review_quarantine_archives_and_preserves_user_reviews(tmp_path, changed_review):
    import hashlib
    from src.core.database import VlogDatabase
    from src.algorithms.segment import Segment
    from scripts.quarantine_generated_reviews import quarantine_generated_reviews
    database = tmp_path / "reviews.sqlite"
    db = VlogDatabase(database)
    db.add_file_task("review.mp4", 0, "20260320", "20260320000000", "20260320000100", 60.)
    db.set_analysis_result("review.mp4", [Segment(0., 10., "DYNAMIC", "review.mp4"),
                                           Segment(10., 60., "STATIC", "review.mp4")])
    first, second = db.get_segments_for_file("review.mp4")
    db.update_segment_review(first["id"], "CONFIRMED_MOTION", json.dumps({"scenario": "ADULT_MOTION", "notes": "heuristic"}))
    db.update_segment_review(second["id"], "CONFIRMED_STATIC", "user reviewed")
    original = dict(db.conn.execute("SELECT * FROM segments WHERE id=?", (first["id"],)).fetchone())
    trusted = dict(db.conn.execute("SELECT * FROM segments WHERE id=?", (second["id"],)).fetchone())
    sample = {"segment_id": first["id"], "filepath": "review.mp4", "start_time": 0., "end_time": 10.,
              "raw_label": "CONFIRMED_MOTION", "scenario": "ADULT_MOTION", "notes": "heuristic",
              "reviewed_at": original["reviewed_at"]}
    dataset = tmp_path / "legacy.json"
    dataset.write_text(json.dumps([sample]), encoding="utf-8")
    dataset.with_suffix(".meta.json").write_text(json.dumps({"review_status": "unverified_heuristic"}), encoding="utf-8")
    sha = hashlib.sha256(dataset.read_bytes()).hexdigest()
    preview = quarantine_generated_reviews(database, dataset, sha, tmp_path / "reports")
    assert not preview["applied"]
    assert dict(db.conn.execute("SELECT * FROM segments WHERE id=?", (first["id"],)).fetchone()) == original
    if changed_review:
        db.update_segment_review(first["id"], "CONFIRMED_STATIC", "later user correction")
        with pytest.raises(ValueError, match="provenance mismatch"):
            quarantine_generated_reviews(database, dataset, sha, tmp_path / "reports", apply=True)
        assert db.conn.execute("SELECT count(*) FROM human_reviews").fetchone()[0] == 2
    else:
        receipt = quarantine_generated_reviews(database, dataset, sha, tmp_path / "reports", apply=True)
        assert Path(receipt["backup"]).exists()
        assert db.conn.execute("SELECT manual_label FROM segments WHERE id=?", (first["id"],)).fetchone()[0] is None
        assert dict(db.conn.execute("SELECT * FROM segments WHERE id=?", (second["id"],)).fetchone()) == trusted
        archived = json.loads(db.conn.execute("SELECT original_rows_json FROM quarantined_generated_reviews").fetchone()[0])
        assert archived["segment"] == original
        assert db.conn.execute("SELECT count(*) FROM human_reviews").fetchone()[0] == 1
    db.close()


@pytest.mark.parametrize("force_init", [False, True])
def test_baseline_builder_preserves_reviews_and_uses_selected_database(tmp_path, force_init):
    from scripts.build_gold_baseline import build_gold_baseline
    from src.core.database import VlogDatabase
    from src.algorithms.segment import Segment

    path = tmp_path / "selected.sqlite"
    db = VlogDatabase(path)
    try:
        db.add_file_task("selected.mp4", 0, "20260901", "20260901000000", "20260901000100", 60)
        db.set_analysis_result("selected.mp4", [
            Segment(0, 30, "DYNAMIC", "selected.mp4", max_energy=20, avg_confidence=0),
            Segment(30, 60, "STATIC", "selected.mp4", max_energy=0.2),
        ])
        reviewed_id = db.get_segments_for_file("selected.mp4")[0]["id"]
        db.update_segment_review(reviewed_id, "CONFIRMED_MOTION", "independent review")
        before_segments = [tuple(r) for r in db.conn.execute("SELECT * FROM segments")]
        before_reviews = [tuple(r) for r in db.conn.execute("SELECT * FROM human_reviews")]
        output = tmp_path / "benchmark.json"
        result = build_gold_baseline(path, force_init, output, archive_version=False)
        assert [tuple(r) for r in db.conn.execute("SELECT * FROM segments")] == before_segments
        assert [tuple(r) for r in db.conn.execute("SELECT * FROM human_reviews")] == before_reviews
        exported = json.loads(output.read_text(encoding="utf-8"))
        assert len(exported) == 1
        assert exported[0]["filepath"] == "selected.mp4"
        assert exported[0]["gold_label"] == "CONFIRMED_MOTION"
        candidates = json.loads(Path(result["candidates_path"]).read_text(encoding="utf-8"))["samples"]
        assert len(candidates) == 1
        assert candidates[0]["review_required"] is True
        assert candidates[0]["segment_id"] != reviewed_id
        assert "gold_label" not in candidates[0]
    finally:
        db.close()


def test_baseline_builder_without_reviews_keeps_existing_dataset(tmp_path):
    from scripts.build_gold_baseline import build_gold_baseline
    output = tmp_path / "benchmark.json"
    output.write_text('[{"historical": true}]', encoding="utf-8")
    before = output.read_bytes()
    result = build_gold_baseline(tmp_path / "empty.sqlite", output_path=output, archive_version=False)
    assert result["reviewed_count"] == 0
    assert result["output_path"] is None
    assert output.read_bytes() == before


def test_confusion_matrix_and_stratified(sample_benchmark):
    metrics = evaluate_confusion_matrix(sample_benchmark)
    assert metrics["total"] == 4
    assert metrics["tp"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["tn"] == 1
    assert metrics["precision"] == 50.0
    assert metrics["recall"] == 50.0
    assert metrics["f1"] == 50.0

    stratified = evaluate_stratified_metrics(sample_benchmark, group_key="scenario")
    assert "ADULT_MOTION" in stratified
    assert stratified["ADULT_MOTION"]["tp"] == 1
    assert stratified["LIGHT_SHADOW"]["fp"] == 1
    assert stratified["INFANT_MOTION"]["fn"] == 1
    assert stratified["EMPTY_ROOM"]["tn"] == 1


def test_save_and_load_benchmark_dataset(sample_benchmark, tmp_path):
    out_file = tmp_path / "test_gold.json"
    save_benchmark_dataset(sample_benchmark, output_path=out_file, version="1.1", archive_version_copy=False)

    assert out_file.exists()
    meta_file = out_file.with_name(out_file.stem + ".meta.json")
    assert meta_file.exists()

    with open(meta_file, "r", encoding="utf-8") as f:
        meta = json.load(f)
    assert meta["version"] == "1.1"
    assert meta["total_samples"] == 4

    loaded_samples, loaded_meta = load_benchmark_dataset(out_file)
    assert len(loaded_samples) == 4
    assert loaded_samples[0]["segment_id"] in [1, 2, 3, 4]
    assert loaded_meta["version"] == "1.1"
    assert loaded_meta["samples_sha256"]


def test_benchmark_same_version_keeps_both_snapshots_and_detects_partial_commit(tmp_path, monkeypatch):
    import src.core.benchmark as benchmark
    monkeypatch.setattr(benchmark, "VERSIONS_DIR", tmp_path / "versions")
    out = tmp_path / "gold.json"
    save_benchmark_dataset([{"segment_id": 1, "gold_label": "CONFIRMED_MOTION"}], out, "1.0")
    save_benchmark_dataset([{"segment_id": 2, "gold_label": "FALSE_ALARM"}], out, "1.0")
    snapshots = [p for p in (tmp_path / "versions").glob("*.json") if not p.name.endswith(".meta.json")]
    assert len(snapshots) == 2
    assert {load_benchmark_dataset(p)[0][0]["segment_id"] for p in snapshots} == {1, 2}
    out.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="data/metadata mismatch"):
        load_benchmark_dataset(out)


def test_benchmark_metadata_commit_failure_restores_previous_dataset(tmp_path, monkeypatch):
    import src.core.benchmark as benchmark
    out = tmp_path / "gold.json"
    save_benchmark_dataset([{"segment_id": 1}], out, archive_version_copy=False)
    before = (out.read_bytes(), out.with_suffix(".meta.json").read_bytes())
    write = benchmark._atomic_write
    failed = False
    def fail_metadata_once(path, payload):
        nonlocal failed
        if path.name.endswith(".meta.json") and not failed:
            failed = True
            raise OSError("controlled metadata commit failure")
        write(path, payload)
    monkeypatch.setattr(benchmark, "_atomic_write", fail_metadata_once)
    with pytest.raises(OSError, match="metadata commit failure"):
        save_benchmark_dataset([{"segment_id": 2}], out, archive_version_copy=False)
    assert (out.read_bytes(), out.with_suffix(".meta.json").read_bytes()) == before
    assert load_benchmark_dataset(out)[0][0]["segment_id"] == 1


def test_merge_benchmark_samples(sample_benchmark):
    new_samples = [
        # 更新已存在的样本 2
        {
            "segment_id": 2,
            "filepath": "cam1/clip2.mp4",
            "gold_label": "CONFIRMED_STATIC",
            "notes": "修正为纯静止",
        },
        # 新增样本 5
        {
            "segment_id": 5,
            "filepath": "cam1/clip5.mp4",
            "start_time": 50.0,
            "end_time": 60.0,
            "duration": 10.0,
            "algo_state": "DYNAMIC",
            "gold_label": "CONFIRMED_MOTION",
        },
    ]

    merged = merge_benchmark_samples(sample_benchmark, new_samples)
    assert len(merged) == 5

    s2 = next(s for s in merged if s["segment_id"] == 2)
    assert s2["gold_label"] == "CONFIRMED_STATIC"
    assert s2["notes"] == "修正为纯静止"


def test_tuning_simulator_what_if(sample_benchmark):
    config = {
        "detection": {"min_motion_threshold": 2.5},
        "segment": {"min_motion_duration": 2.0},
        "yolo": {"confidence": 0.3},
        "presence": {"person_conf_threshold": 0.20},
    }
    sim = TuningSimulator(benchmark_samples=sample_benchmark, config=config)

    # 模拟调高阈值至 3.5 (样本 2 动能为 3.0，将被过滤，消除误报 FP -> TN)
    res = sim.simulate({"detection.min_motion_threshold": 3.5})

    assert res["verdict"] == "UNAVAILABLE"
    assert "deltas" not in res


def test_replay_comparison_uses_current_decisions_and_noop_is_zero(sample_benchmark):
    for sample in sample_benchmark:
        sample.update(review_status="independent_visual", review_evidence="controlled fixture",
                      reference_active=sample["gold_label"] in ("CONFIRMED_MOTION", "MISSED_MOTION"))
    calls = []
    def replay(config, samples):
        calls.append(config)
        return [True, False, True, False] if config.get("yolo", {}).get("enabled", True) else [False] * len(samples)
    simulator = TuningSimulator(sample_benchmark, {}, replay=replay)
    unchanged = simulator.simulate({})
    assert unchanged["verdict"] == "NEUTRAL"
    assert all(value == 0 for value in unchanged["deltas"].values())
    assert len(calls) == 1
    disabled = simulator.simulate({"yolo.enabled": False})
    assert disabled["simulated"]["fn"] == 2
    assert disabled["deltas"]["recall"] == -100
    assert disabled["deltas"]["fn_increase"] == 2


def test_replay_comparison_rejects_mixed_measurement_identities():
    class Measurements:
        def measure(self, config, samples):
            return {"predictions": [True], "details": [{"status": "MEASURED"}],
                    "identity": {"code_sha256": "baseline" if config.get("yolo", {}).get("enabled", True) else "different"}}
    sample = {"reference_active": True, "review_status": "independent_visual", "review_evidence": "controlled identity fixture"}
    result = TuningSimulator([sample], {}, replay=Measurements()).simulate({"yolo.enabled": False})
    assert result["verdict"] == "UNAVAILABLE"
    assert "deltas" not in result and "sample_changes" not in result


def test_config_diff_and_safe_apply(tmp_path):
    config_data = {
        "detection": {
            "min_motion_threshold": 2.5,
            "analysis_resolution": "416x234",
            "prescreen_parallel": 4,
            "analysis_max_workers": 4,
        },
        "render": {
            "batch_max_files": 1,
            "max_concurrency": 1,
        },
        "segment": {
            "min_motion_duration": 2.0,
        },
    }
    cfg_file = tmp_path / "settings.yaml"
    with open(cfg_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_data, f)

    # 1. 验证 Diff 生成
    diff = generate_config_diff({"detection.min_motion_threshold": 3.2}, config_path=cfg_file)
    assert diff["total_changes"] == 1
    assert diff["diffs"][0]["current"] == 2.5
    assert diff["diffs"][0]["proposed"] == 3.2

    # 2. 验证安全写入与备份
    apply_res = apply_config_safe(
        {"detection.min_motion_threshold": 3.2},
        config_path=cfg_file,
        create_backup=True,
    )
    assert apply_res["success"] is True
    backup_file = Path(apply_res["backup_path"])
    assert backup_file.exists()

    with open(cfg_file, "r", encoding="utf-8") as f:
        new_cfg = yaml.safe_load(f)
    assert new_cfg["detection"]["min_motion_threshold"] == 3.2

    # 3. 验证回滚
    rollback_res = rollback_config(backup_file, config_path=cfg_file)
    assert rollback_res["success"] is True

    with open(cfg_file, "r", encoding="utf-8") as f:
        restored_cfg = yaml.safe_load(f)
    assert restored_cfg["detection"]["min_motion_threshold"] == 2.5


def test_concurrent_config_edits_preserve_both_changes_and_unique_backups(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("detection:\n  profile: living_room\n", encoding="utf-8")
    start = threading.Barrier(2)
    def apply(params):
        start.wait(timeout=5)
        return apply_config_safe(params, cfg)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(apply, p) for p in (
            {"detection.min_motion_threshold": 1.8}, {"segment.min_motion_duration": 1.5})]
        results = [f.result(timeout=10) for f in futures]
    final = yaml.safe_load(cfg.read_text(encoding="utf-8"))
    assert final["detection"]["min_motion_threshold"] == 1.8
    assert final["segment"]["min_motion_duration"] == 1.5
    assert len({r["backup_path"] for r in results}) == 2
    backups = [yaml.safe_load(Path(r["backup_path"]).read_text(encoding="utf-8")) for r in results]
    assert sum("min_motion_threshold" in b["detection"] for b in backups) <= 1
    assert not list(tmp_path.glob("*.tmp"))


def test_config_commit_rejects_stale_preview_and_restores_reload_failure(tmp_path, monkeypatch):
    import src.core.tuner as tuner
    import src.core.config as core
    cfg = tmp_path / "settings.yaml"
    cfg.write_text("detection:\n  profile: living_room\n", encoding="utf-8")
    checksum = generate_config_diff({}, cfg)["config_sha256"]
    apply_config_safe({"detection.min_motion_threshold": 2.2}, cfg, expected_sha256=checksum)
    before = cfg.read_bytes()
    with pytest.raises(tuner.ConfigurationConflict):
        apply_config_safe({"detection.min_motion_threshold": 3.5}, cfg, expected_sha256=checksum)
    assert cfg.read_bytes() == before
    old_path, old_settings = core.CONFIG_PATH, core.SETTINGS
    def fail_reload(**kwargs):
        raise OSError("controlled reload failure")
    monkeypatch.setattr(tuner, "load_config", fail_reload)
    with pytest.raises(OSError, match="controlled reload failure"):
        apply_config_safe({"detection.min_motion_threshold": 3.5}, cfg)
    assert cfg.read_bytes() == before
    assert core.CONFIG_PATH == old_path and core.SETTINGS is old_settings
    assert not list(tmp_path.glob("*.tmp"))


def test_config_cross_process_writer_conflict_and_stale_preview(tmp_path, monkeypatch):
    import subprocess
    import sys
    import hashlib
    import src.core.tuner as tuner
    cfg = tmp_path / 'settings.yaml'
    cfg.write_text('detection:\n  profile: living_room\n', encoding='utf-8')
    original = cfg.read_bytes()
    checksum = hashlib.sha256(original).hexdigest()
    backup = tmp_path / 'old.yaml'
    backup.write_bytes(original)
    child = '''
import sys
from src.core.tuner import apply_config_safe, rollback_config, ConfigurationConflict
try:
    if sys.argv[3] == 'apply':
        apply_config_safe({'detection.motion_sensitivity': 1.2}, sys.argv[1], expected_sha256=sys.argv[4])
    else:
        rollback_config(sys.argv[2], sys.argv[1])
except ConfigurationConflict:
    sys.exit(23)
sys.exit(0)
'''
    commit = tuner._commit_config
    def interleaved(path, proposed, before, create_backup):
        for operation in ('apply', 'rollback'):
            result = subprocess.run([sys.executable, '-c', child, str(cfg), str(backup), operation, checksum],
                                    capture_output=True, text=True, timeout=30)
            assert result.returncode == 23, result.stderr
            assert cfg.read_bytes() == original
        return commit(path, proposed, before, create_backup)
    with monkeypatch.context() as patch:
        patch.setattr(tuner, '_commit_config', interleaved)
        tuner.apply_config_safe({'detection.min_motion_threshold': 7}, cfg, expected_sha256=checksum)
    final = cfg.read_bytes()
    result = subprocess.run([sys.executable, '-c', child, str(cfg), str(backup), 'apply', checksum],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 23, result.stderr
    assert cfg.read_bytes() == final
    assert yaml.safe_load(final)['detection']['min_motion_threshold'] == 7
    tuner.rollback_config(backup, cfg)
    assert cfg.read_bytes() == original
