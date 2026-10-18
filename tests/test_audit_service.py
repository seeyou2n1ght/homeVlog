"""测试独立审核工作台 AuditService 与 API 逻辑"""

import pytest
from src.database import VlogDatabase
from src.segment import Segment
from scripts.audit_tool.service import AuditService


@pytest.fixture
def target_frame_service(tmp_path, monkeypatch):
    import json
    import hashlib
    from unittest.mock import MagicMock
    import cv2
    import numpy as np
    from scripts.audit_tool import service as module
    archive = tmp_path / "feedback"
    (archive / "images").mkdir(parents=True)
    monkeypatch.setattr(module, "ARCHIVE_DIR", archive)
    image = archive / "images/frame.jpg"
    cv2.imwrite(str(image), np.full((48, 64, 3), 120, np.uint8))
    source = tmp_path / "original.mp4"
    source.write_bytes(b"original bytes preserved")
    stat = source.stat()
    record = {"segment_id": 1, "archived_image_path": str(image), "filepath": str(source),
              "provenance_version": 2, "frame_origin": "ffmpeg_seek", "frame_timestamp": 12.5,
              "timestamp_kind": "requested_seek", "source_identity": [str(source), stat.st_size, stat.st_mtime_ns],
              "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest()}
    def write_manifest():
        (archive / "manifest.jsonl").write_text(json.dumps(record)+"\n", encoding="utf-8")
    write_manifest()
    audit = object.__new__(AuditService)
    audit.db = MagicMock()
    audit.db.get_segment_by_id.return_value = {"id": 1, "filepath": str(source), "start_time": 10.,
                                               "end_time": 15., "archived_frame_path": str(image)}
    return audit, image, source, record, write_manifest


def agent_proposal_record(target_frame_service, decision="no_targets"):
    audit, image, _, provenance, _ = target_frame_service
    return {"schema": 1, "segment_id": 1, "label_source": "agent_visual_review", "decision": decision,
            "all_visible_targets_reviewed": decision != "uncertain", "boxes": [], "notes": "视觉复核建议",
            "image_sha256": provenance["image_sha256"], "human_truth_certified": False,
            "independent_test_eligible": False,
            "source_provenance": {**provenance, "start_time": 10., "end_time": 15.}}


def test_agent_proposal_keeps_human_priority_and_immutable_versions(target_frame_service):
    from scripts.audit_tool import service as module
    audit, image, _, _, _ = target_frame_service
    context = audit.get_target_review(1)
    audit.save_target_review(1, snapshot=context["snapshot"], review_sha256=None,
                            boxes=[], decision="no_targets", all_targets_reviewed=True)
    human_before = image.with_suffix(".targets.json").read_bytes()
    record = agent_proposal_record(target_frame_service)
    first = audit.import_agent_target_proposal(1, record, image.read_bytes())
    second = audit.import_agent_target_proposal(1, {**record, "notes": "第二版"}, image.read_bytes())
    assert first["snapshot"] != second["snapshot"]
    assert second["human_review_has_priority"] and not second["training_eligible"]
    assert not second["independent_test_eligible"]
    assert len(list((module.ARCHIVE_DIR / "agent_proposals/1").glob("*/review.json"))) == 2
    assert image.with_suffix(".targets.json").read_bytes() == human_before
    audit.db.set_segment_archived_path.assert_not_called()


@pytest.mark.parametrize("mutation", ["image", "source", "interval", "human_claim", "empty_positive"])
def test_agent_proposal_rejects_wrong_identity_before_publication(target_frame_service, mutation):
    from scripts.audit_tool import service as module
    audit, image, _, _, _ = target_frame_service
    record = agent_proposal_record(target_frame_service)
    payload = image.read_bytes()
    if mutation == "image": payload += b"changed"
    elif mutation == "source": record["source_provenance"]["filepath"] = "other.mp4"
    elif mutation == "interval": record["source_provenance"]["frame_timestamp"] = 99.
    elif mutation == "human_claim": record["human_truth_certified"] = True
    elif mutation == "empty_positive": record["decision"] = "objects"
    with pytest.raises(ValueError): audit.import_agent_target_proposal(1, record, payload)
    assert not (module.ARCHIVE_DIR / "agent_proposals").exists()


def test_agent_proposal_read_rejects_tampered_image_and_cannot_enter_human_export(target_frame_service):
    import json
    from scripts.audit_tool import service as module
    from scripts.export_dataset import DatasetItem, verified_target_annotation
    audit, image, source, _, _ = target_frame_service
    record = agent_proposal_record(target_frame_service)
    audit.import_agent_target_proposal(1, record, image.read_bytes())
    image.with_suffix(".targets.json").write_text(json.dumps({**record, "all_targets_reviewed": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="human completeness"):
        verified_target_annotation(DatasetItem(image, "UNKNOWN", 1, str(source), 12.5))
    stored = next((module.ARCHIVE_DIR / "agent_proposals/1").glob("*/frame.jpg"))
    stored.write_bytes(stored.read_bytes()+b"tampered")
    with pytest.raises(ValueError): audit.get_agent_target_proposal(1)


def test_agent_proposal_uncertain_and_failed_publication_preserve_pointer(target_frame_service, monkeypatch):
    from src.core import replay_journal
    audit, image, _, _, _ = target_frame_service
    record = agent_proposal_record(target_frame_service, "uncertain")
    first = audit.import_agent_target_proposal(1, record, image.read_bytes())
    assert not first["review"]["all_visible_targets_reviewed"] and not first["training_eligible"]
    original = replay_journal.atomic_json
    def fail_current(path, value):
        if path.name == "current.json": raise OSError("simulated pointer publication failure")
        return original(path, value)
    monkeypatch.setattr(replay_journal, "atomic_json", fail_current)
    with pytest.raises(OSError): audit.import_agent_target_proposal(1, {**record, "notes": "未发布"}, image.read_bytes())
    assert audit.get_agent_target_proposal(1)["snapshot"] == first["snapshot"]


def test_agent_proposal_http_pins_image_revision(target_frame_service, monkeypatch):
    import json
    import threading
    import urllib.request
    import urllib.error
    import http.cookiejar
    from http.server import HTTPServer
    from scripts.audit_tool import app
    audit, image, _, _, _ = target_frame_service
    audit.import_agent_target_proposal(1, agent_proposal_record(target_frame_service), image.read_bytes())
    monkeypatch.setattr(app, "service", audit)
    server = HTTPServer(("127.0.0.1", 0), app.AuditHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    try:
        opener.open(base + "/").read()
        context = json.loads(opener.open(base + "/api/agent_target_review?segment_id=1").read())
        response = opener.open(base + context["image_url"])
        assert response.read() == image.read_bytes() and response.headers["Cache-Control"] == "no-store"
        assert not context["training_eligible"]
        with pytest.raises(urllib.error.HTTPError) as stale:
            opener.open(base + "/api/agent_target_image?segment_id=1&snapshot=stale")
        assert stale.value.code == 409
        with pytest.raises(urllib.error.HTTPError):
            opener.open(base + "/api/agent_target_review?segment_id=999")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_target_review_explicit_truth_draft_and_history(target_frame_service):
    import json
    from scripts.export_dataset import DatasetItem, verified_target_annotation
    audit, image, source, _, _ = target_frame_service
    context = audit.get_target_review(1)
    image_bytes, source_bytes = image.read_bytes(), source.read_bytes()
    boxes = [{"class_id": 0, "x_center": .5, "y_center": .5, "width": .2, "height": .4}]
    def save(decision, boxes, complete):
        nonlocal context
        result = audit.save_target_review(1, snapshot=context["snapshot"], review_sha256=context["review_sha256"],
                                          boxes=boxes, decision=decision, all_targets_reviewed=complete)
        context = result["context"]
        return result
    item = DatasetItem(image, "FALSE_ALARM", 1, str(source), 12.5)
    assert save("draft", boxes, False)["training_eligible"] is False
    with pytest.raises(ValueError, match="completeness"):
        verified_target_annotation(item)
    assert save("objects", boxes, True)["training_eligible"] is True
    assert len(verified_target_annotation(item)[0]) == 1  # Event false alarm is not target absence.
    assert save("no_targets", [], True)["training_eligible"] is True
    assert verified_target_annotation(item)[0] == []
    assert save("uncertain", [], False)["training_eligible"] is False
    with pytest.raises(ValueError):
        verified_target_annotation(item)
    history = list((image.parents[1]/"target_history").rglob("*.json"))
    assert len(history) == 3
    assert any(json.loads(p.read_text())["decision"] == "objects" for p in history)
    assert image.read_bytes() == image_bytes and source.read_bytes() == source_bytes
    audit.db.submit_review.assert_not_called()


@pytest.mark.parametrize("decision,boxes,complete", [
    ("objects", [], True), ("no_targets", [{"class_id": 0, "x_center": .5, "y_center": .5, "width": .2, "height": .4}], True),
    ("no_targets", [], False), ("other", [], True),
    ("objects", [{"class_id": 0, "x_center": .99, "y_center": .5, "width": .2, "height": .4}], True),
])
def test_target_review_invalid_write_does_not_create_truth(target_frame_service, decision, boxes, complete):
    audit, image, _, _, _ = target_frame_service
    context = audit.get_target_review(1)
    with pytest.raises(ValueError):
        audit.save_target_review(1, snapshot=context["snapshot"], review_sha256=None, boxes=boxes,
                                 decision=decision, all_targets_reviewed=complete)
    assert not image.with_suffix(".targets.json").exists()


def test_target_review_stale_and_writer_conflict(target_frame_service):
    from src.core.replay_journal import run_lease
    audit, image, _, _, _ = target_frame_service
    context = audit.get_target_review(1)
    kwargs = dict(snapshot=context["snapshot"], review_sha256=None, boxes=[], decision="no_targets", all_targets_reviewed=True)
    result = audit.save_target_review(1, **kwargs)
    before = image.with_suffix(".targets.json").read_bytes()
    with pytest.raises(ValueError, match="已变化"):
        audit.save_target_review(1, **kwargs)
    kwargs["review_sha256"] = result["context"]["review_sha256"]
    with run_lease(image.parents[1]/".target_locks/1"):
        with pytest.raises(RuntimeError):
            audit.save_target_review(1, **kwargs)
    assert image.with_suffix(".targets.json").read_bytes() == before


@pytest.mark.parametrize("change", ["legacy", "sha", "source", "time", "identity"])
def test_target_frame_requires_bound_provenance(target_frame_service, change):
    audit, _, _, record, write_manifest = target_frame_service
    if change == "legacy": record.pop("provenance_version")
    elif change == "sha": record["image_sha256"] = "changed"
    elif change == "source": record["filepath"] = "other.mp4"
    elif change == "time": record["frame_timestamp"] = 25
    else: record["source_identity"] = None
    write_manifest()
    with pytest.raises(ValueError, match="来源"):
        audit.get_target_review(1)


def test_target_review_http_snapshot_and_explicit_empty(target_frame_service, monkeypatch):
    import json
    import threading
    import http.cookiejar
    import urllib.request
    import urllib.error
    from http.server import HTTPServer
    from scripts.audit_tool import app
    audit, image, _, _, _ = target_frame_service
    monkeypatch.setattr(app, "service", audit)
    server = HTTPServer(("127.0.0.1", 0), app.AuditHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    try:
        opener.open(base + "/").read()
        context = json.loads(opener.open(base + "/api/target_review?segment_id=1").read())
        assert opener.open(base + context["image_url"]).read() == image.read_bytes()
        with pytest.raises(urllib.error.HTTPError) as stale:
            opener.open(base + "/api/target_image?segment_id=1&snapshot=stale")
        assert stale.value.code == 409
        request = dict(segment_id=1, snapshot=context["snapshot"], review_sha256=None,
                       boxes=[], decision="no_targets", all_targets_reviewed=True)
        result = json.loads(opener.open(urllib.request.Request(base + "/api/save_target_review",
                            data=json.dumps(request).encode(), headers={"Content-Type": "application/json"})).read())
        assert result["training_eligible"] is True
        assert result["context"]["review"]["no_targets"] is True
        with pytest.raises(urllib.error.HTTPError) as conflict:
            opener.open(urllib.request.Request(base + "/api/save_target_review", data=json.dumps(request).encode(),
                        headers={"Content-Type": "application/json"}))
        assert conflict.value.code == 409
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_target_review_atomic_failure_preserves_previous(target_frame_service, monkeypatch):
    from src.core import replay_journal
    audit, image, _, _, _ = target_frame_service
    context = audit.get_target_review(1)
    result = audit.save_target_review(1, snapshot=context["snapshot"], review_sha256=None,
                                     boxes=[], decision="no_targets", all_targets_reviewed=True)
    before = image.with_suffix(".targets.json").read_bytes()
    original = replay_journal.atomic_json
    def fail_publish(path, value):
        if str(path).endswith(".targets.json"):
            raise OSError("controlled publication failure")
        return original(path, value)
    monkeypatch.setattr(replay_journal, "atomic_json", fail_publish)
    with pytest.raises(OSError):
        audit.save_target_review(1, snapshot=context["snapshot"], review_sha256=result["context"]["review_sha256"],
                                 boxes=[], decision="uncertain", all_targets_reviewed=False)
    assert image.with_suffix(".targets.json").read_bytes() == before


def test_preview_cache_separates_source_identity_and_exact_seek(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace
    from scripts.audit_tool import service as module
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(module, "CACHE_DIR", cache)
    audit = object.__new__(AuditService)
    monkeypatch.setattr(audit, "_resolve_local_timestamp", lambda fp, t: (t, 300))
    sources = [tmp_path / camera / "same.mp4" for camera in ("cam0", "cam1")]
    for source in sources:
        source.parent.mkdir()
        source.write_bytes(source.parent.name.encode())
    calls = []

    def extract(args, **kwargs):
        calls.append(args)
        assert args[0] != "ffmpeg" and kwargs["io_timeout"] == 3
        Path(args[-1]).write_bytes(Path(args[args.index("-i") + 1]).read_bytes())
        return SimpleNamespace(returncode=0, stderr=b"", timed_out=False)

    monkeypatch.setattr(module, "run_ffmpeg", extract)
    first = audit.extract_frame(str(sources[0]), 1.001)
    second = audit.extract_frame(str(sources[1]), 1.001)
    precise = audit.extract_frame(str(sources[0]), 1.002)
    assert len({first, second, precise}) == 3
    assert first.read_bytes() != second.read_bytes()
    assert audit.extract_frame(str(sources[0]), 1.001) == first
    assert len(calls) == 3
    sources[0].write_bytes(b"changed source")
    assert audit.extract_frame(str(sources[0]), 1.001) != first
    assert not list(cache.glob("preview_pending_*"))


@pytest.mark.parametrize("failure", ["exit", "empty", "source_changed", "timeout", "cancelled"])
def test_preview_failures_never_publish_partial_cache(tmp_path, monkeypatch, failure):
    from pathlib import Path
    from types import SimpleNamespace
    import subprocess
    from scripts.audit_tool import service as module
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    audit = object.__new__(AuditService)
    monkeypatch.setattr(audit, "_resolve_local_timestamp", lambda fp, t: (t, 300))

    def extract(args, **kwargs):
        Path(args[-1]).write_bytes(b"" if failure == "empty" else b"partial")
        if failure == "source_changed":
            source.write_bytes(b"changed")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, 10)
        if failure == "cancelled":
            FFmpegProcessRegistry.mark_interrupted()
        return SimpleNamespace(returncode=1 if failure == "exit" else 0, stderr=b"failure", timed_out=False)

    monkeypatch.setattr(module, "run_ffmpeg", extract)
    try:
        assert audit.extract_frame(str(source), 1) is None
    finally:
        FFmpegProcessRegistry.reset_interrupted()
    assert not list(tmp_path.glob("preview_*"))


def test_preview_clip_does_not_extend_requested_interval(tmp_path, monkeypatch):
    audit = object.__new__(AuditService)
    monkeypatch.setattr(audit, "_resolve_local_timestamp", lambda fp, t: (t, 300))
    calls = []
    monkeypatch.setattr(audit, "_cached_preview", lambda *args: calls.append(args))
    audit.generate_preview_clip("source.mp4", 1, 1.25)
    assert calls[-1][1] == ("clip", "1.000", "0.250")
    audit.generate_preview_clip("source.mp4", 1, 20)
    assert calls[-1][1][-1] == "6.000"
    audit.generate_preview_clip("source.mp4", 2, 1)
    assert len(calls) == 2


def test_source_player_interval_cleanup_and_review_navigation():
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for workbench interaction regression")
    script = r"""
const fs = require('fs'), vm = require('vm'), assert = require('assert');
function element() {
  const classes = new Set(['hidden']);
  return {classList:{add:c=>classes.add(c),remove:c=>classes.delete(c),
    contains:c=>classes.has(c),toggle:(c,on)=>on?classes.add(c):classes.delete(c)},
    pause(){this.paused=true}, play(){this.paused=false;return Promise.resolve()},
    load(){}, removeAttribute(n){delete this[n]}, duration:8, currentTime:0};
}
const elements = Object.fromEntries(['modal-player','modal-whatif','vlog-video-el',
  'player-modal-title','player-interval-hint','btn-replay-interval'].map(id=>[id,element()]));
const context = {document:{addEventListener(){},getElementById:id=>elements[id]},console};
vm.createContext(context); vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
vm.runInContext(`
showToast=()=>{};
openVideoPlayer('source.mp4','source',{start:2,end:4});
`,context);
const video=elements['vlog-video-el'];
video.onloadedmetadata(); assert.equal(video.currentTime,2); assert.equal(video.paused,false);
video.currentTime=4; video.ontimeupdate(); assert.equal(video.paused,true);
vm.runInContext(`
let navigation=[]; let labels=0;
selectNext=direction=>navigation.push(direction);
handleDecision=()=>labels++;
handleKeydown({key:'1',code:'Digit1',target:{tagName:'BODY'},preventDefault(){}});
handleKeydown({key:'j',target:{tagName:'BODY'},preventDefault(){}});
handleKeydown({key:'k',target:{tagName:'BODY'},preventDefault(){}});
if(labels!==0 || navigation.join(',')!=='1,-1') throw Error('Navigation semantics changed');
closeVideoPlayer();
`,context);
assert.equal(video.src,undefined); assert.equal(video.paused,true);
assert.equal(video.onloadedmetadata,null); assert.equal(video.ontimeupdate,null);
vm.runInContext(`openVideoPlayer('source.mp4','source',{start:2,end:9});`,context);
video.onloadedmetadata(); assert.equal(video.paused,true);
assert(elements['player-interval-hint'].innerText.includes('时长不足'));
"""
    source = Path(__file__).resolve().parents[1] / "scripts/audit_tool/static/app.js"
    subprocess.run([node, "-e", script, str(source)], check=True, capture_output=True, timeout=10)


def test_workbench_review_updates_only_after_full_acknowledgement():
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if not node:
        pytest.skip("Node is required for workbench interaction regression")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const context={document:{addEventListener(){}},console,assert};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
vm.runInContext(`
showToast=()=>{};updateCategoryBadges=()=>{};renderQueueList=()=>{};loadQueue=async()=>{};
state.items=[{id:1,manual_label:null},{id:2,manual_label:null}];
state.filteredItems=state.items;state.selectedIndex=0;
let release;
fetch=()=>new Promise(resolve=>release=resolve);
(async()=>{
  const pending=commitReview([1],'CONFIRMED_MOTION');
  assert.equal(state.items[0].manual_label,null);
  assert.equal(await commitReview([1],'FALSE_ALARM'),false);
  release({ok:true,json:async()=>({success:false})});
  assert.equal(await pending,false);assert.equal(state.items[0].manual_label,null);
  fetch=async()=>({ok:true,json:async()=>({success:true,count:1})});
  assert.equal(await commitReview([1,2],'FALSE_ALARM'),false);
  assert(state.items.every(item=>item.manual_label===null));
  assert.equal(await commitReview([1],'CONFIRMED_MOTION',{scenario:'ADULT_MOTION',notes:'full interval'}),true);
  assert.equal(state.items[0].scenario,'ADULT_MOTION');
  fetch=async()=>({ok:true,json:async()=>({success:false})});
  await clearCurrentReview();assert.equal(state.items[0].manual_label,'CONFIRMED_MOTION');
  fetch=async()=>({ok:true,json:async()=>({success:true})});
  await clearCurrentReview();assert.equal(state.items[0].manual_label,null);
})().catch(error=>{console.error(error);process.exitCode=1;});
`,Object.assign(context,{process}));
"""
    source = Path(__file__).resolve().parents[1] / "scripts/audit_tool/static/app.js"
    subprocess.run([node, "-e", script, str(source)], check=True, capture_output=True, timeout=10)


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("cleanup", [False, True])
@pytest.mark.parametrize("failure", ["assets_false", "assets_exception", "manifest", "interrupted", "stop_unconfirmed", "none"])
def test_reviewed_delivery_keeps_batches_and_old_assets_until_commit(delivery_workspace, monkeypatch, count, cleanup, failure):
    from unittest.mock import MagicMock
    import threading
    from scripts.audit_tool import rerender
    w = delivery_workspace
    w.config["render"]["cleanup_batches_on_success"] = cleanup
    monkeypatch.setattr(rerender, "load_config", lambda: w.config)
    monkeypatch.setattr(rerender, "OUTPUT_DIR", w.output.parent)
    monkeypatch.setattr(rerender, "partition_timeline_by_batches", lambda timeline, **kwargs: [[timeline[0]]] * count)
    monkeypatch.setattr(rerender, "build_batch_render", lambda **kwargs: str(w.batches[kwargs["bi"]]))
    monkeypatch.setattr(rerender, "concat_output_files", w.pipeline.concat_output_files)
    if failure.startswith("assets"):
        replacement = MagicMock(return_value=False) if failure == "assets_false" else MagicMock(side_effect=OSError("assets exception"))
        monkeypatch.setattr(w.pipeline, "_save_vlog_companion_assets", replacement)
    elif failure == "manifest":
        monkeypatch.setattr("src.hardware.render_cache.save_manifest", MagicMock(side_effect=OSError("manifest failure")))
    elif failure == "interrupted":
        monkeypatch.setattr(rerender, "build_batch_render", MagicMock(side_effect=rerender.RenderInterrupted("external signal")))
    elif failure == "stop_unconfirmed":
        from src.stages.renderer import RenderTerminationUnconfirmed
        monkeypatch.setattr(rerender, "build_batch_render", MagicMock(side_effect=RenderTerminationUnconfirmed("retained reader")))
    manager = rerender.ReRenderManager()
    key = manager.get_task_key("20260901", 0)
    monkeypatch.setattr(manager, "tasks", {key: {}})
    manager._run_render_worker(w.db, "20260901", 0, "", threading.Event(), key)
    expected_status = "CANCELLED" if failure == "interrupted" else ("COMPLETED" if failure == "none" else "FAILED")
    assert manager.tasks[key]["status"] == expected_status
    if failure != "none":
        assert {path: path.read_bytes() for path in w.original} == w.original
    for index, batch in enumerate(w.batches):
        retained = failure != "none" or not cleanup or index >= count
        assert batch.exists() is retained
        assert batch.with_name(batch.name + ".json").exists() is retained
    assert not w.db.is_render_completed("20260901", 0)  # reviewed copy does not commit the original delivery record


def test_audit_service_overview_and_export(tmp_path):
    db_path = tmp_path / "test_audit.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_cam0_20260901_001.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)

        sample_segs = [
            Segment(start_time=0.0, end_time=10.0, state="STATIC", source_file=fp, file_start_offset=0.0),
            Segment(start_time=10.0, end_time=30.0, state="DYNAMIC", source_file=fp, file_start_offset=0.0, max_energy=4.0),
            Segment(start_time=30.0, end_time=60.0, state="STATIC", source_file=fp, file_start_offset=0.0, max_energy=1.9),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        # 1. 初始 overview 指标
        ov = service.get_overview()
        assert ov["total_files"] == 1
        assert ov["total_segments"] == 3
        assert ov["reviewed_segments"] == 0

        # 2. 疑难排查队列
        anomalies = service.get_anomalies(limit=10)
        assert len(anomalies) >= 1

        # 3. 获取文件分段
        segs_data = service.get_file_segments(filepath=fp)
        assert len(segs_data["segments"]) == 3
        dyn_id = segs_data["segments"][1]["id"]

        # 4. 人工打标: 误报 (FALSE_ALARM)
        assert service.submit_review(dyn_id, "FALSE_ALARM", notes="树影晃动")

        # 5. 再次验证 overview 准确率指标
        ov_updated = service.get_overview()
        assert ov_updated["reviewed_segments"] == 1
        assert ov_updated["labels"]["fp"] == 1

        # 6. 测试报表导出
        csv_str, ctype_csv = service.export_report(fmt="csv")
        assert "FALSE_ALARM" in csv_str
        assert "text/csv" in ctype_csv

        json_str, ctype_json = service.export_report(fmt="json")
        assert "FALSE_ALARM" in json_str
        assert "application/json" in ctype_json

    finally:
        db.close()


def test_review_context_shows_camera_config_and_human_plan_without_writes(tmp_path, monkeypatch):
    import yaml
    from src.core import config as core
    from scripts.audit_tool.service import pack_notes
    raw = {"detection": {"profile": "living_room", "min_motion_threshold": 3.2},
           "yolo": {"confidence": .3}, "output": {"fps": 20},
           "cameras": {"B888805AA3CD": {"profile": "baby_room", "overrides": {"detection": {"min_motion_threshold": 5}}}}}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setattr(core, "CONFIG_PATH", path)
    monkeypatch.setattr("src.utils.CONFIG_PATH", path)
    db = VlogDatabase(db_path=tmp_path / "context.sqlite")
    try:
        source = str(tmp_path / "XiaomiCamera_01_B888805AA3CD" / "source.mp4")
        db.add_file_task(source, 1, "20261012", "20261012000140", "20261012000146", 6)
        db.set_analysis_result(source, "ANALYZED", [Segment(start_time=100, end_time=106, state="PRESENCE", source_file=source, file_start_offset=100)])
        segment_id = db.conn.execute("SELECT id FROM segments").fetchone()[0]
        assert db.update_segment_review(segment_id, "MISSED_MOTION", pack_notes("watched full interval", "ADULT_MOTION", "video_interval"))
        audit = AuditService(db=db)
        before = db.conn.execute("SELECT * FROM segments").fetchall()
        context = audit.get_review_context(segment_id)
        parameters = {row["param"]: row for row in context["parameters"]}
        assert parameters["detection.min_motion_threshold"] == {"param": "detection.min_motion_threshold", "global": 3.2, "camera": 5, "explicit_global": True}
        assert parameters["detection.profile"]["camera"] == "baby_room"
        assert context["scenario"] == "ADULT_MOTION" and context["user_notes"] == "watched full interval"
        assert context["review_basis"] == "video_interval"
        assert context["comparison"]["analysis"]["portions"][0]["state"] == "PRESENCE"
        assert context["comparison"]["reviewed"]["portions"][0]["state"] == "DYNAMIC"
        assert context["comparison"]["reviewed"]["display_s"] == 6
        assert context["comparison_basis"] == "saved_analysis_vs_human_overlay_shared_plan"
        assert not db.conn.in_transaction
        assert db.conn.execute("SELECT * FROM segments").fetchall() == before
        db.conn.execute("BEGIN")
        audit.get_review_context(segment_id)
        assert db.conn.in_transaction
        db.conn.rollback()
        with pytest.raises(ValueError, match="Unknown segment"):
            audit.get_review_context(-1)
    finally:
        db.close()


def test_label_only_reviews_preserve_each_scenario_and_notes(tmp_path, monkeypatch):
    from scripts.audit_tool.service import pack_notes
    monkeypatch.setattr("src.archiver.extract_and_archive_frame", lambda *args, **kwargs: None)
    db = VlogDatabase(db_path=tmp_path / "attribution.sqlite")
    try:
        db.add_file_task("source.mp4", 0, "20261012", "20261012000000", "20261012000006", 6)
        db.set_analysis_result("source.mp4", "ANALYZED", [Segment(start_time=start, end_time=start+2, state="STATIC", source_file="source.mp4", file_start_offset=0) for start in (0, 2)])
        ids = [row[0] for row in db.conn.execute("SELECT id FROM segments ORDER BY id")]
        notes = [pack_notes("visual evidence", "ADULT_MOTION", "representative_frame"), pack_notes("sound uncertain", "UNKNOWN", "uncertain")]
        for segment_id, note in zip(ids, notes):
            assert db.update_segment_review(segment_id, "CONFIRMED_MOTION", note)
        service = AuditService(db=db)
        assert service.submit_review(ids[0], "CONFIRMED_MOTION", notes="visual evidence",
            scenario="ADULT_MOTION", review_basis="representative_frame")
        assert service.get_review_context(ids[0])["review_basis"] == "representative_frame"
        assert service.submit_review(-1, "CONFIRMED_MOTION", preserve_attribution=True) is False
        assert service.submit_review(ids[0], "MISSED_MOTION", preserve_attribution=True)
        assert service.submit_batch_review(ids, "FALSE_ALARM", preserve_attribution=True) == 2
        assert [row[0] for row in db.conn.execute("SELECT review_notes FROM segments ORDER BY id")] == notes
        assert [row[0] for row in db.conn.execute("SELECT review_notes FROM human_reviews ORDER BY start_time")] == notes
        assert service.submit_review(ids[0], "CONFIRMED_STATIC", notes="", scenario="")
        assert db.get_segment_by_id(ids[0])["review_notes"] == ""
    finally:
        db.close()


def test_review_snapshot_preserves_transactions_and_rejects_stale_preview(tmp_path, monkeypatch):
    import threading
    from scripts.audit_tool import rerender
    db = VlogDatabase(db_path=tmp_path / "snapshot.sqlite")
    try:
        source = str(tmp_path / "source.mp4")
        db.add_file_task(source, 0, "20261012", "20261012000000", "20261012000006", 6)
        _, _, first = rerender.capture_review_inputs(db, "20261012", 0)
        assert not db.conn.in_transaction
        db.conn.execute("BEGIN")
        db.conn.execute("UPDATE file_tasks SET file_duration=5")
        _, _, changed = rerender.capture_review_inputs(db, "20261012", 0)
        assert db.conn.in_transaction and changed != first
        db.conn.rollback()
        assert rerender.capture_review_inputs(db, "20261012", 0)[2] == first
        manager = rerender.ReRenderManager()
        key = manager.get_task_key("20261012", 0)
        monkeypatch.setattr(manager, "tasks", {key: {}})
        monkeypatch.setattr(rerender, "check_source_files_accessibility", lambda rows: pytest.fail("Stale preview reached media"))
        manager._run_render_worker(db, "20261012", 0, "reviewed", threading.Event(), key, "0" * 64)
        assert manager.get_status("20261012", 0)["status"] == "FAILED"
        assert "changed after preview" in manager.get_status("20261012", 0)["error"]
    finally:
        db.close()


def test_cancelled_worker_must_exit_before_same_task_can_restart(monkeypatch):
    import threading
    from types import SimpleNamespace
    from scripts.audit_tool.rerender import ReRenderManager
    manager = ReRenderManager()
    key = manager.get_task_key("20261012", 0)
    monkeypatch.setattr(manager, "tasks", {key: {"status": "RENDERING", "batch_results": [{"status": "RUNNING"}]}})
    monkeypatch.setattr(manager, "cancel_flags", {key: threading.Event()})
    monkeypatch.setattr(manager, "active_threads", {key: SimpleNamespace(is_alive=lambda: True)})
    assert manager.cancel_task("20261012", 0)
    assert manager.get_status("20261012", 0)["status"] == "CANCELLING"
    assert manager.start_rerender(None, "20261012", 0)["error"] == "Task already running"
    manager.tasks[key]["status"] = "CANCELLED"
    assert manager.start_rerender(None, "20261012", 0)["error"] == "Task already running"
    status = manager.get_status("20261012", 0)
    status["batch_results"][0]["status"] = "altered copy"
    assert manager.tasks[key]["batch_results"][0]["status"] == "RUNNING"
    assert not manager.cancel_task("20261012", 0)


def test_rerender_cancel_waits_for_live_worker_before_terminal_state(tmp_path, monkeypatch):
    import threading
    from scripts.audit_tool import rerender
    db = VlogDatabase(db_path=tmp_path / "live.sqlite")
    db.add_file_task("controlled.mp4", 0, "20261012", "20261012000000", "20261012000006", 6)
    entered, release = threading.Event(), threading.Event()

    def check(rows):
        entered.set()
        assert release.wait(5)
        return True, [row["filepath"] for row in rows], []

    monkeypatch.setattr(rerender, "check_source_files_accessibility", check)
    manager = rerender.ReRenderManager()
    for name in ("tasks", "cancel_flags", "active_threads"):
        monkeypatch.setattr(manager, name, {})
    key = manager.get_task_key("20261012", 0)
    try:
        assert manager.start_rerender(db, "20261012", 0)["status"] == "CHECKING"
        assert entered.wait(5)
        worker = manager.active_threads[key]
        assert worker.is_alive()
        assert manager.cancel_task("20261012", 0)
        assert manager.get_status("20261012", 0)["status"] == "CANCELLING"
        assert manager.start_rerender(db, "20261012", 0)["error"] == "Task already running"
        release.set()
        worker.join(5)
        assert not worker.is_alive()
        assert manager.get_status("20261012", 0)["status"] == "CANCELLED"
    finally:
        release.set()
        if key in manager.active_threads:
            manager.active_threads[key].join(5)
        db.close()


def test_review_day_conflict_prevents_source_io_and_render(delivery_workspace,external_day_owner,monkeypatch):
    from scripts.audit_tool import rerender
    from src.stages import renderer
    from unittest.mock import MagicMock
    w=delivery_workspace
    external_day_owner(w.db.db_path,renderer.TEMP_DIR,w.output)
    manager=rerender.ReRenderManager()
    monkeypatch.setattr(manager,'tasks',{})
    monkeypatch.setattr(manager,'active_threads',{})
    monkeypatch.setattr(manager,'cancel_flags',{})
    source_check=MagicMock()
    body=MagicMock()
    monkeypatch.setattr(rerender,'capture_review_inputs',lambda *args:([{'filepath':str(w.source)}],w.config,'snapshot'))
    monkeypatch.setattr(rerender,'check_source_files_accessibility',source_check)
    monkeypatch.setattr(manager,'_run_render_worker_owned',body)
    assert manager.start_rerender(w.db,'20260901',0)['status']=='CHECKING'
    manager.active_threads['20260901_cam0'].join(5)
    assert not manager.active_threads['20260901_cam0'].is_alive()
    status=manager.get_status('20260901',0)
    assert status['status']=='FAILED' and 'already owned' in status['error']
    source_check.assert_not_called();body.assert_not_called()
    for path,content in w.original.items():assert path.read_bytes()==content


def test_audit_service_category_filtering(tmp_path):
    db_path = tmp_path / "test_cat.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "cam0_20260901180000.mp4"
        file_start = 18 * 3600.0  # 64800.0
        db.add_file_task(fp, 0, "20260901", "20260901180000", "20260901180500", 300.0)

        sample_segs = [
            # FP 候选: DYNAMIC 但 avg_confidence == 0 (无主体)
            Segment(start_time=file_start + 10.0, end_time=file_start + 25.0, state="DYNAMIC", source_file=fp, file_start_offset=file_start, avg_confidence=0.0, max_energy=0.8),
            # FN 候选: STATIC 但 max_energy >= 1.5
            Segment(start_time=file_start + 30.0, end_time=file_start + 60.0, state="STATIC", source_file=fp, file_start_offset=file_start, max_energy=2.5),
            # 短碎片抖动: duration < 3.0
            Segment(start_time=file_start + 70.0, end_time=file_start + 72.0, state="DYNAMIC", source_file=fp, file_start_offset=file_start, avg_confidence=0.8, max_energy=3.0),
            # 正常切片
            Segment(start_time=file_start + 80.0, end_time=file_start + 110.0, state="DYNAMIC", source_file=fp, file_start_offset=file_start, avg_confidence=0.9, max_energy=5.0),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        # 标记最后一个切片为已复核 CONFIRMED_MOTION
        file_segs = service.get_file_segments(filepath=fp)["segments"]
        last_seg_id = file_segs[-1]["id"]
        service.submit_review(last_seg_id, "CONFIRMED_MOTION")

        # 1. 疑似误报 (FP)
        fps = service.get_anomalies(category="fp_suspect")
        assert len(fps) == 1
        assert fps[0]["anomaly_type"] == "fp_suspect"
        assert "无目标置信度" in fps[0]["reason_desc"]

        # 2. 疑似漏报 (FN)
        fns = service.get_anomalies(category="fn_suspect")
        assert len(fns) == 1
        assert fns[0]["anomaly_type"] == "fn_suspect"
        assert "微动作漏判" in fns[0]["reason_desc"]

        # 3. 碎片跳变
        jitters = service.get_anomalies(category="jitter")
        assert len(jitters) == 1
        assert jitters[0]["duration"] < 3.0

        # 4. 已复核
        revs = service.get_anomalies(category="reviewed")
        assert len(revs) == 1
        assert revs[0]["manual_label"] == "CONFIRMED_MOTION"

        # 5. 全部待排查疑难 (不包含正常的已复核切片)
        all_list = service.get_anomalies(category="all")
        assert len(all_list) == 3
    finally:
        db.close()


def test_resolve_local_timestamp(tmp_path):
    db_path = tmp_path / "test_resolve.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_cam0_resolve.mp4"
        file_start = 65114.0
        db.add_file_task(fp, 0, "20260320", "20260320180514", "20260320181029", 315.0)

        # 场景 A: 传入日内绝对秒数 (65373.5s) -> 应当换算为 259.5s
        t_local, dur = service._resolve_local_timestamp(fp, 65373.5)
        assert abs(t_local - 259.5) < 1e-3
        assert abs(dur - 315.0) < 1e-3

        # 场景 B: 传入相对秒数 (15.0s) -> 应当直接保留为 15.0s
        t_local_rel, _ = service._resolve_local_timestamp(fp, 15.0)
        assert abs(t_local_rel - 15.0) < 1e-3

        # 场景 C: 寻道越界 (350.0s > 315.0s) -> 截断到 314.9s
        t_clamped, _ = service._resolve_local_timestamp(fp, 350.0)
        assert abs(t_clamped - 314.9) < 1e-3

        # 场景 D: 负数截断
        t_neg, _ = service._resolve_local_timestamp(fp, -10.0)
        assert t_neg == 0.0
    finally:
        db.close()


def test_audit_http_export():
    import http.cookiejar
    import threading
    import urllib.error
    import json
    import urllib.request
    from http.server import HTTPServer
    from scripts.audit_tool.app import AuditHandler

    server = HTTPServer(("127.0.0.1", 0), AuditHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        # 1. 测试 CSV 导出 (含 UTF-8 BOM 与命名)
        csv_url = f"http://127.0.0.1:{port}/api/export?format=csv"
        req = urllib.request.Request(csv_url)
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            ctype = resp.headers.get("Content-Type", "")
            cdisp = resp.headers.get("Content-Disposition", "")
            assert "text/csv" in ctype
            assert "homevlog_audit_" in cdisp
            assert ".csv" in cdisp
            raw_bytes = resp.read()
            assert raw_bytes.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM 校验

        # 2. 测试 JSON 导出
        json_url = f"http://127.0.0.1:{port}/api/export?format=json"
        req = urllib.request.Request(json_url)
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 200
            ctype = resp.headers.get("Content-Type", "")
            cdisp = resp.headers.get("Content-Disposition", "")
            assert "application/json" in ctype
            assert "homevlog_audit_" in cdisp
            assert ".json" in cdisp
            raw_bytes = resp.read()
            import json
            parsed = json.loads(raw_bytes.decode("utf-8"))
            assert isinstance(parsed, list)

        # 写接口同时要求同源 Host/Origin 和启动时 token cookie。
        unauth = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/review", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(unauth)
        assert exc.value.code == 403

        opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar())
        )
        opener.open(f"http://127.0.0.1:{port}/").read()
        authenticated = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/review", data=b"{}",
            headers={"Content-Type": "application/json"}, method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            opener.open(authenticated)
        assert exc.value.code == 400

        foreign = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/overview",
            headers={"Origin": "https://example.invalid"},
        )
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(foreign)
        assert exc.value.code == 403
    finally:
        server.shutdown()
        server.server_close()


class TestReRenderWorkflow:
    """测试即时重浓缩引擎 (ReRenderManager) 与物理素材校验"""

    def test_check_source_files_accessibility(self, tmp_path):
        from scripts.audit_tool.rerender import check_source_files_accessibility

        # 创建 2 个真实文件和 1 个假文件
        f1 = tmp_path / "video1.mp4"
        f1.write_bytes(b"dummy video data 1")
        f2 = tmp_path / "video2.mp4"
        f2.write_bytes(b"dummy video data 2")
        f3 = tmp_path / "video3_missing.mp4"

        # 1. 混合测试
        tasks = [
            {"filepath": str(f1)},
            {"filepath": str(f2)},
            {"filepath": str(f3)},
        ]
        can_proceed, valid, missing = check_source_files_accessibility(tasks)
        assert can_proceed is False
        assert len(valid) == 2
        assert len(missing) == 1
        assert str(f3) in missing

        # 2. 全缺失硬阻断测试 (NAS 未挂载场景)
        tasks_all_missing = [{"filepath": str(tmp_path / f"non_existent_{i}.mp4")} for i in range(3)]
        can_proceed_all_missing, _, missing_all = check_source_files_accessibility(tasks_all_missing)
        assert can_proceed_all_missing is False
        assert len(missing_all) == 3

    @pytest.mark.parametrize("prescreen,analysis", [
        ("SKIPPED_EMPTY", "PENDING"), ("SKIPPED_EXCLUDED", "SKIPPED_EXCLUDED"),
        ("SUSPICIOUS", "SKIPPED_CORRUPT"),
    ])
    def test_rerender_exclusions_do_not_require_source_access(self, tmp_path, prescreen, analysis):
        from scripts.audit_tool.rerender import check_source_files_accessibility
        valid = tmp_path / "healthy.mp4"
        valid.write_bytes(b"source")
        excluded = {"filepath": str(tmp_path / "excluded.mp4"),
                    "prescreen_status": prescreen, "analysis_status": analysis}
        assert check_source_files_accessibility([{"filepath": str(valid)}, excluded]) == (True, [str(valid)], [])
        # An ordinary missing source must still block delivery.
        assert check_source_files_accessibility([excluded, {"filepath": str(tmp_path / "missing.mp4")}])[0] is False

    def test_rerender_manager_lifecycle(self, tmp_path):
        from scripts.audit_tool.rerender import ReRenderManager
        from src.database import VlogDatabase

        mgr = ReRenderManager()
        db_path = tmp_path / "test_rerender.db"
        db = VlogDatabase(db_path=db_path)

        try:
            # 空任务状态
            st = mgr.get_status("20260901", 0)
            assert st["status"] == "IDLE"

            # 测试任务启动对无文件任务的拦截
            res = mgr.start_rerender(db, "20260901", 0)
            assert res["status"] == "CHECKING"

            # 稍等片刻，worker 会因为在数据库中找不到 file_tasks 而转为 FAILED
            import time
            for _ in range(20):
                st = mgr.get_status("20260901", 0)
                if st["status"] in ("FAILED", "COMPLETED"):
                    break
                time.sleep(0.05)

            assert st["status"] == "FAILED"
            assert "No file tasks found" in st["error"]

        finally:
            db.close()

    def test_cancel_is_scoped_to_the_rerender_task(self):
        import threading

        from scripts.audit_tool.rerender import ReRenderManager
        from src.hardware.ffmpeg import FFmpegProcessRegistry

        mgr = ReRenderManager()
        mgr._init_manager()
        key = mgr.get_task_key("20260901", 0)
        mgr.cancel_flags[key] = threading.Event()
        mgr.tasks[key] = {"status": "RENDERING"}
        FFmpegProcessRegistry.reset_interrupted()
        assert mgr.cancel_task("20260901", 0)
        assert mgr.tasks[key]["status"] == "CANCELLING"
        assert not FFmpegProcessRegistry.is_interrupted()

    def test_rerender_output_filename_resolution(self):
        from src.scanner import resolve_output_filename
        mock_cfg = {"cameras": {"B888805AA3CD": "living_room"}}
        sample_path = r"\\192.168.5.8\LyuShare\XiaomiCamera_01_B888805AA3CD\00_20260402000000_20260402000500.mp4"

        base = resolve_output_filename("DailyVlog_{date}_{mac}.mp4", "20260402", 0, sample_path, mock_cfg)
        assert base == "DailyVlog_20260402_B888805AA3CD.mp4"

        # 校验带版本标识的重浓缩派生命名
        from pathlib import Path
        output_version = "v2"
        stem = Path(base).stem
        suffix = Path(base).suffix
        final_name = f"{stem}_{output_version}{suffix}"
        assert final_name == "DailyVlog_20260402_B888805AA3CD_v2.mp4"


def test_audit_scenario_and_clear_review(tmp_path):
    """测试场景归因标签、备注打包解析与打标撤销"""
    db_path = tmp_path / "test_scenario.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_scenario_cam0.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)
        sample_segs = [
            Segment(start_time=0.0, end_time=15.0, state="DYNAMIC", source_file=fp, max_energy=5.0),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        file_segs = service.get_file_segments(filepath=fp)["segments"]
        seg_id = file_segs[0]["id"]

        # 1. 提交带场景归因与用户备注的审核
        ok = service.submit_review(seg_id, "FALSE_ALARM", notes="车灯反射白墙", scenario="HEADLIGHT")
        assert ok is True

        # 2. 检验 get_file_segments 与 get_anomalies 是否能正确解包 scenario 和 user_notes
        segs_after = service.get_file_segments(filepath=fp)["segments"]
        assert segs_after[0]["manual_label"] == "FALSE_ALARM"
        assert segs_after[0]["scenario"] == "HEADLIGHT"
        assert segs_after[0]["user_notes"] == "车灯反射白墙"

        # 3. 撤销打标
        assert service.clear_review(seg_id) is True

        segs_cleared = service.get_file_segments(filepath=fp)["segments"]
        assert segs_cleared[0]["manual_label"] is None
        assert segs_cleared[0]["scenario"] == ""
        assert segs_cleared[0]["user_notes"] == ""
        assert not db.get_all_file_tasks_for_date("20260901", 0)[0]["human_reviews"]
    finally:
        db.close()


def test_overview_inventory_does_not_claim_accuracy_from_mutable_states(tmp_path):
    """Inventory is complete, but human overlays are not independent predictions."""
    db_path = tmp_path / "test_matrix.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_matrix_cam0.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)
        sample_segs = [
            # 1. DYNAMIC -> CONFIRMED_MOTION (TP)
            Segment(start_time=0.0, end_time=10.0, state="DYNAMIC", source_file=fp, max_energy=10.0),
            # 2. DYNAMIC -> FALSE_ALARM (FP)
            Segment(start_time=10.0, end_time=20.0, state="DYNAMIC", source_file=fp, max_energy=3.0),
            # 3. STATIC -> MISSED_MOTION (FN)
            Segment(start_time=20.0, end_time=30.0, state="STATIC", source_file=fp, max_energy=2.0),
            # 4. STATIC -> CONFIRMED_STATIC (TN)
            Segment(start_time=30.0, end_time=40.0, state="STATIC", source_file=fp, max_energy=0.5),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        segs = service.get_file_segments(filepath=fp)["segments"]
        service.submit_review(segs[0]["id"], "CONFIRMED_MOTION")
        service.submit_review(segs[1]["id"], "FALSE_ALARM", scenario="LIGHT_SHADOW")
        service.submit_review(segs[2]["id"], "MISSED_MOTION", scenario="INFANT_MOTION")
        service.submit_review(segs[3]["id"], "CONFIRMED_STATIC")

        ov = service.get_overview()
        labels = ov["labels"]
        assert labels["tp"] == 1
        assert labels["fp"] == 1
        assert labels["fn"] == 1
        assert labels["tn"] == 1

        metrics = ov["metrics"]
        assert metrics == {"precision": None, "recall": None, "f1_score": None}
        assert ov["metrics_basis"] == "requires_independent_production_replay"
        assert ov["labels_basis"] == "explicit_review_labels_not_confusion_matrix"
    finally:
        db.close()


def test_save_bounding_box(tmp_path, monkeypatch):
    """测试人工绘制 BBox 并保存为 YOLO 规范的 txt 文件"""
    from pathlib import Path
    db_path = tmp_path / "test_bbox.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    # 隔离归档目录到临时目录
    fake_archive = tmp_path / "feedback_archive"
    fake_archive.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("scripts.audit_tool.service.ARCHIVE_DIR", fake_archive)

    try:
        boxes = [
            {"class_id": 0, "x_center": 0.5, "y_center": 0.6, "width": 0.2, "height": 0.4},
            {"class_id": 15, "x_center": 0.8, "y_center": 0.9, "width": 0.1, "height": 0.1},
        ]
        res = service.save_bounding_box(boxes=boxes, image_name="test_peak.jpg")
        assert res["success"] is True
        assert res["box_count"] == 2
        assert len(res["saved_paths"]) >= 1

        # 检查生成的文件内容
        saved_txt = Path(res["saved_paths"][0])
        assert saved_txt.exists()
        lines = saved_txt.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        assert lines[0] == "0 0.500000 0.600000 0.200000 0.400000"
        assert lines[1] == "15 0.800000 0.900000 0.100000 0.100000"
    finally:
        db.close()


def test_tuning_insights(tmp_path):
    """测试根据人工审核记录量化生成 settings.yaml 调参建议"""
    db_path = tmp_path / "test_tuning.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_tuning_cam0.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)
        sample_segs = [
            Segment(start_time=0.0, end_time=10.0, state="DYNAMIC", source_file=fp, max_energy=3.2),
            Segment(start_time=10.0, end_time=20.0, state="DYNAMIC", source_file=fp, max_energy=3.8),
            Segment(start_time=20.0, end_time=30.0, state="DYNAMIC", source_file=fp, max_energy=4.0),
            Segment(start_time=30.0, end_time=40.0, state="STATIC", source_file=fp, max_energy=2.0),
            Segment(start_time=40.0, end_time=50.0, state="STATIC", source_file=fp, max_energy=1.8),
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        segs = service.get_file_segments(filepath=fp)["segments"]
        # 3 处光影误报
        service.submit_review(segs[0]["id"], "FALSE_ALARM", scenario="LIGHT_SHADOW")
        service.submit_review(segs[1]["id"], "FALSE_ALARM", scenario="LIGHT_SHADOW")
        service.submit_review(segs[2]["id"], "FALSE_ALARM", scenario="LIGHT_SHADOW")
        # 2 处婴儿微动漏报
        service.submit_review(segs[3]["id"], "MISSED_MOTION", scenario="INFANT_MOTION")
        service.submit_review(segs[4]["id"], "MISSED_MOTION", scenario="INFANT_MOTION")

        insights = service.get_tuning_insights()
        assert insights["total_reviewed"] == 5
        assert insights["scenario_counts"]["LIGHT_SHADOW"] == 3
        assert insights["scenario_counts"]["INFANT_MOTION"] == 2

        # 检查是否给出了 min_motion_threshold 与 yolo.confidence 的调参建议
        recs = insights["recommendations"]
        params = [r["param"] for r in recs]
        assert "detection.min_motion_threshold" in params
        assert "yolo.confidence" in params
        assert insights["distribution"] == {"tp": 0, "fp": 3, "fn": 2, "tn": 0}
        assert sum(group["total"] for group in insights["groups"]) == 5
        assert not insights["auto_apply_allowed"]
        assert all(r["suggested_value"] is None and r["status"] == "REQUIRES_REPLAY" for r in recs)
    finally:
        db.close()


def test_tuning_inventory_counts_accelerated_and_unknown_labels(tmp_path):
    db = VlogDatabase(tmp_path / "inventory.db")
    try:
        path = "source.mp4"
        db.add_file_task(path, 0, "20260901", "20260901220000", "20260901220500", 300)
        states = ["MICRO_MOTION", "NIGHT_STATIONARY", "PRESENCE", "STATIC", "DYNAMIC"]
        db.set_analysis_result(path, "ANALYZED", [Segment(start_time=79200+i*10, end_time=79210+i*10,
                               state=state, source_file=path) for i, state in enumerate(states)])
        for row, label in zip(db.get_all_segments_for_date("20260901", 0), ["FN", "FN", "TP", "TN", "UNKNOWN"]):
            db.conn.execute("UPDATE segments SET manual_label=? WHERE id=?", (label, row["id"]))
        db.conn.commit()
        result = AuditService(db).get_tuning_insights()
        assert result["distribution"] == {"tp": 1, "fp": 0, "fn": 2, "tn": 1}
        assert result["unknown_labels"] == 1
        assert len(result["requires_attribution"]) == 2
        assert sum(r["count"] for r in result["prediction_cross_tab"]) == 5
        assert result["scenario_counts"] == {"UNKNOWN": 5}
        overview = AuditService(db).get_overview()
        assert overview["labels"] == result["distribution"]
        assert overview["unknown_labels"] == 1
        assert all(value is None for value in overview["metrics"].values())
    finally:
        db.close()


def test_anomaly_offset_pagination(tmp_path):
    """测试 get_anomalies 的 offset 分页功能"""
    db_path = tmp_path / "test_page.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)

    try:
        fp = "test_page_cam0.mp4"
        db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)
        sample_segs = [
            Segment(start_time=float(i * 10), end_time=float(i * 10 + 2), state="DYNAMIC", source_file=fp)
            for i in range(10)
        ]
        db.set_analysis_result(fp, "ANALYZED", sample_segs)

        page1 = service.get_anomalies(category="jitter", limit=3, offset=0)
        assert len(page1) == 3

        page2 = service.get_anomalies(category="jitter", limit=3, offset=3)
        assert len(page2) == 3

        # 两页的 id 应无交集
        ids1 = {x["id"] for x in page1}
        ids2 = {x["id"] for x in page2}
        assert len(ids1.intersection(ids2)) == 0
    finally:
        db.close()


def test_audit_http_new_endpoints(tmp_path, monkeypatch):
    """测试新增加的 HTTP 端点: /api/tuning_insights, /api/clear_review, /api/save_bbox, /api/export_yolo_dataset"""
    import http.cookiejar
    import json
    import threading
    import urllib.request
    from http.server import HTTPServer
    from scripts.audit_tool.app import AuditHandler

    fake_archive = tmp_path / "feedback_archive"
    fake_archive.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("scripts.audit_tool.service.ARCHIVE_DIR", fake_archive)

    server = HTTPServer(("127.0.0.1", 0), AuditHandler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    try:
        cj = http.cookiejar.CookieJar()
        opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
        opener.open(f"http://127.0.0.1:{port}/").read()

        # GET /api/tuning_insights
        with opener.open(f"http://127.0.0.1:{port}/api/tuning_insights") as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "total_reviewed" in data
            assert "recommendations" in data

        # POST /api/save_bbox
        bbox_payload = json.dumps({
            "boxes": [{"class_id": 0, "x_center": 0.5, "y_center": 0.5, "width": 0.1, "height": 0.2}],
            "image_name": "test_http_bbox.jpg"
        }).encode("utf-8")
        req_bbox = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/save_bbox",
            data=bbox_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with opener.open(req_bbox) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert data["box_count"] == 1

        # POST /api/clear_review
        clear_payload = json.dumps({"segment_id": 99999}).encode("utf-8")
        req_clear = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/clear_review",
            data=clear_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with opener.open(req_clear) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "success" in data

        # POST /api/export_yolo_dataset
        exp_payload = json.dumps({"val_ratio": 0.2, "output_dir": str(tmp_path / "yolo_out")}).encode("utf-8")
        req_exp = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/export_yolo_dataset",
            data=exp_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with opener.open(req_exp) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert "success" in data

    finally:
        server.shutdown()
        server.server_close()


def test_audit_api_benchmark_and_tuning_integration(tmp_path):
    """测试基准版本化、What-If 仿真与流式视频播放的 HTTP 端点集成"""
    import json
    import urllib.request
    import urllib.parse
    import socket
    import http.cookiejar
    from http.server import HTTPServer
    import threading
    from scripts.audit_tool import app
    from src.core.config import OUTPUT_DIR

    db_path = tmp_path / "test_api_bm.db"
    db = VlogDatabase(db_path=db_path)
    test_service = AuditService(db=db)

    # 预置 1 个样本供测试
    fp = "sample_test.mp4"
    db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)
    segs = [
        Segment(start_time=10.0, end_time=20.0, state="DYNAMIC", source_file=fp, max_energy=4.5, avg_confidence=0.8)
    ]
    db.set_analysis_result(fp, "ANALYZED", segs)
    db_segs = db.get_segments_for_file(fp)
    test_service.submit_review(db_segs[0]["id"], "CONFIRMED_MOTION", scenario="ADULT_MOTION")

    # 动态端口
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]

    app.service = test_service
    server = HTTPServer(("127.0.0.1", port), app.AuditHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    cj = http.cookiejar.CookieJar()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

    try:
        # 1. 握手获取 Cookie Token
        req_root = urllib.request.Request(f"http://127.0.0.1:{port}/")
        with opener.open(req_root) as resp:
            assert resp.status == 200

        # 2. POST /api/export_gold_benchmark (版本化导出)
        gold_out = tmp_path / "api_gold.json"
        exp_payload = json.dumps({
            "output_path": str(gold_out),
            "version": "2.0_test",
            "incremental_merge": False,
        }).encode("utf-8")
        req_exp = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/export_gold_benchmark",
            data=exp_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with opener.open(req_exp) as resp:
            assert resp.status == 200
            data = json.loads(resp.read().decode("utf-8"))
            assert data["success"] is True
            assert data["version"] == "2.0_test"
            assert gold_out.exists()

        # 3. GET /api/benchmark_versions
        req_v = urllib.request.Request(f"http://127.0.0.1:{port}/api/benchmark_versions")
        with opener.open(req_v) as resp:
            assert resp.status == 200
            v_data = json.loads(resp.read().decode("utf-8"))
            assert isinstance(v_data, list)

        # 4. POST /api/tuning_simulate
        sim_payload = json.dumps({
            "candidate_params": {"detection.min_motion_threshold": 3.5}
        }).encode("utf-8")
        req_sim = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/tuning_simulate",
            data=sim_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with opener.open(req_sim) as resp:
            assert resp.status == 200
            sim_res = json.loads(resp.read().decode("utf-8"))
            assert "verdict" in sim_res

        # 5. POST /api/tuning_diff
        diff_payload = json.dumps({
            "params": {"detection.min_motion_threshold": 3.5}
        }).encode("utf-8")
        req_diff = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/tuning_diff",
            data=diff_payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with opener.open(req_diff) as resp:
            assert resp.status == 200
            diff_res = json.loads(resp.read().decode("utf-8"))
            assert "total_changes" in diff_res

        # 6. GET /api/video_stream (Range 请求分片播放)
        test_video = OUTPUT_DIR / f"test_stream_{port}.mp4"
        test_video.write_bytes(b"ABCDEFGHIJ" * 50)
        try:
            url = f"http://127.0.0.1:{port}/api/video_stream?filepath={urllib.parse.quote(str(test_video))}"
            payload = test_video.read_bytes()
            for header, start, end in [("bytes=0-9", 0, 9), ("bytes=-4", 496, 499),
                                       ("bytes=498-999", 498, 499), ("bytes=498-", 498, 499),
                                       ("bytes=-999", 0, 499)]:
                req_stream = urllib.request.Request(url, headers={"Range": header})
                with opener.open(req_stream, timeout=5) as resp:
                    assert resp.status == 206
                    assert resp.headers["Content-Range"] == f"bytes {start}-{end}/500"
                    assert int(resp.headers["Content-Length"]) == end - start + 1
                    assert resp.read() == payload[start:end + 1]
            for header in ["bytes=500-", "bytes=9-2", "bytes=-0", "bytes=-", "bytes=abc-def",
                           "bytes=0-1,4-5", "bytes=+1-4", "items=0-1"]:
                with pytest.raises(urllib.error.HTTPError) as rejected:
                    opener.open(urllib.request.Request(url, headers={"Range": header}), timeout=5)
                with rejected.value as resp:
                    assert resp.code == 416
                    assert resp.headers["Content-Range"] == "bytes */500"
                    assert resp.read() == b""
            with opener.open(url, timeout=5) as resp:
                assert resp.status == 200 and resp.read() == payload
            test_video.write_bytes(b"")
            with pytest.raises(urllib.error.HTTPError) as rejected:
                opener.open(urllib.request.Request(url, headers={"Range": "bytes=0-"}), timeout=5)
            with rejected.value as resp:
                assert resp.code == 416 and resp.headers["Content-Range"] == "bytes */0"
                assert resp.read() == b""
        finally:
            test_video.unlink(missing_ok=True)

    finally:
        server.shutdown()
        server.server_close()
        db.close()
        from src.core.benchmark import VERSIONS_DIR
        for p in VERSIONS_DIR.glob("*2.0_test*"):
            p.unlink(missing_ok=True)


def test_audit_service_exclude_file(tmp_path):
    db_path = tmp_path / "test_exclude.db"
    db = VlogDatabase(db_path=db_path)
    service = AuditService(db=db)
    fp = "corrupted_video.mp4"
    db.add_file_task(fp, 0, "20260901", "20260901100000", "20260901100500", 300.0)
    assert service.is_registered_filepath(fp)
    ok = service.exclude_file(fp, reason="PHYSICAL_CORRUPT")
    assert ok
    task = db.get_all_file_tasks_for_date("20260901", 0)[0]
    assert task["prescreen_status"] == "SKIPPED_EXCLUDED"
    assert task["analysis_status"] == "SKIPPED_EXCLUDED"
    assert task["error_msg"] == "PHYSICAL_CORRUPT"




@pytest.fixture
def queue_inventory(tmp_path):
    from scripts.audit_tool.service import pack_notes
    db = VlogDatabase(db_path=tmp_path / "queue.sqlite")
    for index in range(4):
        source = f"source_{index}_%.mp4"
        date = f"202610{11 + index // 2}"
        db.add_file_task(source, index % 2, date, date+"000000", date+"001000", 600)
        db.set_analysis_result(source, "ANALYZED", [
            Segment(start_time=start, end_time=start+2, state="STATIC" if start < 300 else "DYNAMIC", source_file=source, file_start_offset=0, max_energy=start/100)
            for start in (0, 10, 20, 310, 320)])
    ids = [row[0] for row in db.conn.execute("SELECT id FROM segments ORDER BY id")]
    db.conn.execute("UPDATE segments SET manual_label='CONFIRMED_STATIC',review_notes=? WHERE id=?", (pack_notes("<script>literal</script>", "EMPTY"), ids[0]))
    db.conn.execute("UPDATE segments SET review_notes='literal_plain_note' WHERE id=?", (ids[-1],))
    db.conn.commit()
    yield AuditService(db=db), ids
    db.close()


def test_review_queue_inventory_scopes_and_literal_search(queue_inventory):
    service, ids = queue_inventory
    full = service.get_review_queue("all")
    assert full["counts"] == {"all":20,"reviewed":1,"unreviewed":19,"gold_candidates":8}
    assert len(full["items"]) == 20  # Includes unflagged inventory, unlike legacy anomalies.
    assert service.get_review_queue("all", cam_index=0)["total_items"] == 10
    assert service.get_review_queue("all", date="20261012", cam_index=0)["total_items"] == 5
    assert service.get_review_queue("all", search="_%")["total_items"] == 20
    assert service.get_review_queue("all", search="literal_plain_note")["items"][0]["id"] == ids[-1]
    assert service.get_review_queue("all", scenario="EMPTY")["total_items"] == 1
    assert service.get_review_queue("all", scenario="UNKNOWN")["total_items"] == 19
    assert service.get_review_queue("all", hide_reviewed=True)["total_items"] == 19
    assert service.get_review_queue("reviewed")["items"][0]["id"] == ids[0]


@pytest.mark.parametrize("category", ["all","gold_candidates","random"])
def test_review_queue_pages_are_complete_and_read_only(queue_inventory, category):
    service, ids = queue_inventory
    before = [tuple(row) for row in service.db.conn.execute("SELECT * FROM segments ORDER BY id")]
    whole = service.get_review_queue(category, limit=200, seed=83)
    pages = [service.get_review_queue(category, limit=3, offset=offset, seed=83) for offset in range(0, whole["total_items"], 3)]
    assert [item["id"] for page in pages for item in page["items"]] == [item["id"] for item in whole["items"]]
    assert all(page["counts"] == whole["counts"] for page in pages)
    if category == "gold_candidates":
        assert all(not item["manual_label"] and "cluster_ids" not in item for item in whole["items"])
        assert sum(item["represented_count"] for item in whole["items"]) == 19
        assert len({(item["date"],item["cam_index"],item["state"]) for item in whole["items"][:8]}) == 8
    if category == "random":
        assert whole["page_identity"] == service.get_review_queue(category, limit=200, seed=83)["page_identity"]
        assert [item["id"] for item in whole["items"]] != [item["id"] for item in service.get_review_queue(category, limit=200, seed=84)["items"]]
        assert set(item["id"] for item in whole["items"]) == set(ids)
    assert before == [tuple(row) for row in service.db.conn.execute("SELECT * FROM segments ORDER BY id")]
    service.db.conn.execute("BEGIN")
    service.db.conn.execute("UPDATE segments SET review_reason='uncommitted' WHERE id=?", (ids[1],))
    service.get_review_queue(category)
    assert service.db.conn.in_transaction
    service.db.conn.rollback()


@pytest.mark.parametrize("kwargs", [{"category":"invalid"},{"limit":0},{"limit":201},{"offset":-1},{"seed":-1},{"search":"a"*201}])
def test_review_queue_rejects_invalid_requests(queue_inventory, kwargs):
    service, _ = queue_inventory
    with pytest.raises(ValueError):
        service.get_review_queue(**kwargs)


def test_queue_browser_discards_stale_responses_and_uses_inventory_counts():
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const context={document:{addEventListener(){},getElementById(){return null}},URLSearchParams,console,assert};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
vm.runInContext(`
showEmptyPlaceholder=()=>{};updateQueuePage=()=>{};renderQueueList=()=>{};updateCategoryBadges=()=>{};setElementText=()=>{};selectItem=()=>{};
const pending=[];fetch=url=>new Promise(resolve=>pending.push({url,resolve}));
const response=(id,total)=>({ok:true,json:async()=>({items:[{id}],counts:{all:total,reviewed:7,unreviewed:total-7},total_items:total})});
(async()=>{
 const old=loadQueue();state.searchQuery='off_page';state.offset=50;
 const current=loadQueue();assert(pending[1].url.includes('search=off_page'));assert(pending[1].url.includes('offset=50'));
 pending[1].resolve(response(2,166000));await current;
 pending[0].resolve(response(1,1));await old;
 assert.equal(state.items[0].id,2);assert.equal(state.counts.all,166000);assert.equal(state.totalItems,166000);assert.equal(state.loading,false);
 assert.equal(escapeHTML('<img src="x">'), '&lt;img src=&quot;x&quot;&gt;');
 fetch=async()=>{throw Error('controlled read failure')};assert.equal(await loadQueue(),false);assert.equal(state.items.length,0);
 const controls={'queue-seed':{value:'83'},'queue-apply-scope':{addEventListener:(event,handler)=>controls.apply=handler}};
 for(const name of ['date','camera','state','scenario'])controls['queue-'+name]={value:''};
 document.getElementById=id=>controls[id]||null;document.querySelectorAll=()=>[];window={addEventListener(){}};
 controls['queue-seed'].addEventListener=()=>{};loadQueue=()=>{};
 initEventListeners();state.seed=1;controls.apply();assert.equal(state.seed,83);assert.equal(state.offset,0);

})().catch(error=>{console.error(error);process.exitCode=1});
`,Object.assign(context,{process}));
"""
    subprocess.run([node,"-e",script,str(Path(__file__).resolve().parents[1]/"scripts/audit_tool/static/app.js")],check=True,capture_output=True,timeout=10)



def test_review_queue_counts_refresh_after_confirmed_label_change(queue_inventory):
    service, ids = queue_inventory
    original = service.get_review_queue("all")
    assert service.db.update_segment_review(ids[1], "FALSE_ALARM", "controlled fixture only")
    changed = service.get_review_queue("all", hide_reviewed=True)
    assert changed["counts"]["all"] == original["counts"]["all"]
    assert changed["counts"]["reviewed"] == 2
    assert changed["total_items"] == 18
    assert ids[1] not in [item["id"] for item in changed["items"]]
    assert service.db.get_segment_by_id(ids[1])["review_notes"] == "controlled fixture only"



def test_delivery_comparison_uses_bound_artifact_mapping_and_history(tmp_path, monkeypatch):
    import json
    from src.stages.companion import promote_vlog_assets
    from src.core import config as core
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setattr(core, "OUTPUT_DIR", output_dir)
    db = VlogDatabase(db_path=tmp_path / "deliveries.sqlite")
    try:
        source = str(tmp_path / "source.mp4")
        db.add_file_task(source, 0, "20261012", "20261012000000", "20261012000006", 6)
        db.set_analysis_result(source, "ANALYZED", [Segment(start_time=1,end_time=2,state="DYNAMIC",source_file=source)])
        segment_id = db.conn.execute("SELECT id FROM segments").fetchone()[0]
        audit = AuditService(db=db)
        output = output_dir / "delivery.mp4"
        def assets(path, frames, state):
            path.write_bytes(b"controlled media fixture" + bytes([frames]))
            stat = path.stat()
            path.with_suffix(".meta.json").write_text(json.dumps({"date":"20261012","camera":{"cam_index":0},
                "delivery_mapping":{"version":1,"status":"AVAILABLE","fps":20,"planned_frames":frames,
                    "media_identity":{"size":stat.st_size,"mtime_ns":stat.st_mtime_ns,"frames":frames,"fps":20},
                    "segments":[{"source_file":source,"source_start_s":0,"source_end_s":6,"output_start_frame":0,"output_end_frame":frames,"state":state}]}}))
        assets(output,120,"DYNAMIC")
        first = audit.get_delivery_comparison(segment_id)
        assert first["versions"][0]["intervals"][0]["end"] == 6
        candidate = output_dir / "delivery.pending.mp4"
        assets(candidate,6,"STATIC")
        with promote_vlog_assets(candidate,output,{"render":{"generate_subtitles":False}},keep_previous=True):
            pass
        compared = audit.get_delivery_comparison(segment_id)
        assert [item["role"] for item in compared["versions"]] == ["current","previous"]
        assert compared["versions"][0]["intervals"][0]["end"] == .3
        assert compared["versions"][1]["version_id"] != first["versions"][0]["version_id"]  # Archived path is part of identity.
        output.write_bytes(b"replacement with invalid identity")
        changed = audit.get_delivery_comparison(segment_id)
        assert [item["role"] for item in changed["versions"]] == ["previous"]
        assert any("stale mapping" in problem for problem in changed["problems"])
        output.with_suffix(".meta.json").write_text(json.dumps({"date":"20261012","camera":{"cam_index":0}}))
        assert any("lacks a verified" in problem for problem in audit.get_delivery_comparison(segment_id)["problems"])
    finally:
        db.close()



def test_selected_representative_frame_survives_late_clip_response():
    import shutil
    import subprocess
    from pathlib import Path
    node=shutil.which("node")
    if not node:
        pytest.skip("Node required")
    script=r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const elements={};const clips=[];
const context={document:{addEventListener(){},getElementById(id){return elements[id]??=( {classList:{add(){},remove(){}}} );}},Image:class{constructor(){clips.push(this)}},console,assert,clips};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[1],'utf8'),context);
vm.runInContext(`
state.filteredItems=[{id:1,filepath:'source.mp4',start_time:5,end_time:185,visual_frame_time:10,file_start_offset:5}];state.selectedIndex=0;
loadVisualsForSegment(state.filteredItems[0]);
assert(clips[0].src.includes('end=11.00'));
document.getElementById('thumb-mid').onclick();
const pinned=document.getElementById('main-player-img').src;
assert(pinned.includes('t=15.000') && pinned.includes('w=800'));
clips[0].onload();assert.equal(document.getElementById('main-player-img').src,pinned);
assert.equal(state.previewPinnedId,1);
state.filteredItems=[{id:2,filepath:'other.mp4'}];
document.getElementById('thumb-mid').onmouseenter();assert.equal(state.previewPinnedId,1);
` ,context);
"""
    subprocess.run([node,"-e",script,str(Path(__file__).resolve().parents[1]/"scripts/audit_tool/static/app.js")],check=True,capture_output=True,timeout=10)


def test_tuning_quality_summary_preserves_unknown_and_gate_reasons():
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which('node')
    if not node:
        pytest.skip('Node required')
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const start=source.indexOf('function qualityGateSummary('),end=source.indexOf('\n}',start)+2;
const context={};vm.createContext(context);vm.runInContext(source.slice(start,end),context);
const summary=context.qualityGateSummary;
assert(summary(null).includes('资料暂不可用'));
assert(summary({independent_counts:{}}).includes('未知/未知组'));
const gate={verified_samples:27,required_independent_groups:10,
independent_counts:{ADULT_MOTION:0,INFANT_MOTION:9,EMPTY_ROOM:10,LIGHT_SHADOW:0},
excluded_samples:{not_holdout:72,used_for_tuning:0,unreviewed:1,missing_source:0,invalid_interval:0,duplicates:2,conflicts:1},
insufficient_scenarios:['ADULT_MOTION','INFANT_MOTION','LIGHT_SHADOW'],failed_scenarios:['EMPTY_ROOM']};
const text=summary(gate);
for(const word of ['27条','成人活动 0/10组','婴幼儿活动 9/10组','非留出 72条','重复 2条','冲突 1个区间','待补齐：成人活动、婴幼儿活动、光影','未达标：空镜','诊断比较'])assert(text.includes(word),word);
assert(!summary({independent_counts:{ADULT_MOTION:NaN},required_independent_groups:1}).includes('NaN'));
const runStart=source.indexOf('async function runTuningSimulation()');
const runBody=source.slice(runStart,source.indexOf('async function applyTuning',runStart));
assert(runBody.includes("setElementText('sim-quality-details', '')"));
assert(runBody.includes('qualityGateSummary(null)'));
assert(runBody.includes('qualityGateSummary(data.quality_gate)'));
"""
    subprocess.run([node, '-e', script, str(Path(__file__).resolve().parents[1]/'scripts/audit_tool/static/app.js')],
                   check=True, capture_output=True, timeout=10)


@pytest.mark.parametrize('basis', ['unknown','representative_frame','video_interval','video_audio_interval','uncertain'])
def test_review_basis_structured_notes_keep_legacy_contract(basis):
    from scripts.audit_tool.service import pack_notes, unpack_notes, unpack_review_basis
    packed = pack_notes('watched range', 'ADULT_MOTION', basis)
    assert unpack_notes(packed) == ('ADULT_MOTION', 'watched range')
    assert unpack_review_basis(packed) == basis
    assert unpack_review_basis('old plain notes') == 'unknown'
    assert unpack_review_basis('{broken') == 'unknown'
    with pytest.raises(ValueError, match='review_basis'):
        pack_notes('notes', 'EMPTY_ROOM', 'automatically_verified')


def test_review_basis_api_validates_before_writes(monkeypatch):
    import http.cookiejar
    import json
    import threading
    import urllib.error
    import urllib.request
    from http.server import HTTPServer
    from unittest.mock import MagicMock
    from scripts.audit_tool import app
    service = MagicMock()
    service.submit_review.return_value = True
    monkeypatch.setattr(app, 'service', service)
    server = HTTPServer(('127.0.0.1', 0), app.AuditHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    url = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        opener.open(url+'/').read()
        valid = dict(segment_id=1, manual_label='CONFIRMED_MOTION', notes='full interval',
            scenario='ADULT_MOTION', review_basis='video_interval')
        for endpoint, body in [('/api/review', dict(valid, review_basis='auto')),
                               ('/api/review', {k:v for k,v in valid.items() if k!='notes'}),
                               ('/api/batch_review', dict(valid, segment_ids=[1,2]))]:
            request = urllib.request.Request(url+endpoint, data=json.dumps(body).encode(), headers={'Content-Type':'application/json'})
            with pytest.raises(urllib.error.HTTPError) as error:
                opener.open(request)
            assert error.value.code == 400
        service.submit_review.assert_not_called()
        service.submit_batch_review.assert_not_called()
        request = urllib.request.Request(url+'/api/review', data=json.dumps(valid).encode(), headers={'Content-Type':'application/json'})
        assert json.loads(opener.open(request).read())['success']
        assert service.submit_review.call_args.kwargs['review_basis'] == 'video_interval'
        assert service.submit_review.call_args.kwargs['preserve_attribution'] is False
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=3)


def test_target_canvas_coordinates_and_pointer_ownership():
    import shutil
    import subprocess
    from pathlib import Path
    node = shutil.which("node")
    if not node:
        pytest.skip("Node required")
    script = r"""
const fs=require('fs'),vm=require('vm'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const section=source.slice(source.indexOf('const targetEditor ='),source.indexOf("document.addEventListener('DOMContentLoaded'"));
const nodes={},handlers={},ctx={clearRect(){},drawImage(){},strokeRect(){},fillText(){}};
const canvas={width:1920,height:1080,getContext:()=>ctx,getBoundingClientRect:()=>({left:20,top:30,width:802,height:452}),setPointerCapture(){},addEventListener:(name,fn)=>handlers[name]=fn};
nodes['target-canvas']=canvas;
const context={document:{getElementById:id=>nodes[id]??=({value:'0',checked:true,addEventListener(){},classList:{add(){},remove(){}}}),querySelectorAll:()=>[]},getComputedStyle:()=>({borderLeftWidth:'1px',borderRightWidth:'1px',borderTopWidth:'1px',borderBottomWidth:'1px'}),assert};
vm.createContext(context);vm.runInContext(section,context);
vm.runInContext('initTargetReview();targetEditor.image={};targetEditor.context={};',context);
const point=(x,y,pointerId=1)=>({clientX:21+x*800,clientY:31+y*450,button:0,pointerId});
handlers.pointerdown(point(.6,.8));handlers.pointermove(point(.1,.2));handlers.pointerup(point(.1,.2));
const box=vm.runInContext('targetEditor.boxes[0]',context);
for(const [key,value] of Object.entries({x_center:.35,y_center:.5,width:.5,height:.6}))assert(Math.abs(box[key]-value)<1e-12);
vm.runInContext('targetEditor.boxes=[];',context);
handlers.pointerdown(point(.2,.2));handlers.pointermove(point(.9,.9,2));handlers.pointerup(point(.9,.9,2));handlers.pointercancel(point(.9,.9,2));
assert.equal(vm.runInContext('targetEditor.boxes.length',context),0);
assert(vm.runInContext('targetEditor.drag!==null',context));
handlers.pointerup(point(.8,.8));assert.equal(vm.runInContext('targetEditor.boxes.length',context),1);
handlers.pointerdown(point(.2,.2));vm.runInContext('targetEditor.busy=true;',context);handlers.pointerup(point(.8,.8));
assert.equal(vm.runInContext('targetEditor.boxes.length',context),1);
vm.runInContext('targetEditor.busy=false;',context);
handlers.pointerdown(point(.2,.2));vm.runInContext('targetEditor.serial++;',context);handlers.pointerup(point(.8,.8));
assert.equal(vm.runInContext('targetEditor.boxes.length',context),1);
handlers.pointerdown(point(-.2,-.2));handlers.pointerup(point(1.2,1.2));
const full=vm.runInContext('targetEditor.boxes[1]',context);assert.equal(full.width,1);assert.equal(full.height,1);
handlers.pointerdown(point(.3,.3));handlers.pointercancel(point(.4,.4));assert.equal(vm.runInContext('targetEditor.drag',context),null);
"""
    source = Path(__file__).resolve().parents[1] / "scripts/audit_tool/static/app.js"
    subprocess.run([node,"-e",script,str(source)],check=True,capture_output=True,timeout=10)


def test_preview_managed_cancellation_reaps_process_and_releases_io(tmp_path, monkeypatch):
    import subprocess
    import sys
    import threading
    import time
    from scripts.audit_tool import service as module
    from src.hardware import scheduler
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    from src.core.visual_index import source_identity
    monkeypatch.setattr(module,"inspect_source_identity_owned",lambda fp,*args,**kwargs:tuple(source_identity(fp)))
    source = tmp_path / "source.mp4"
    source.write_bytes(b"controlled source identity")
    monkeypatch.setattr(module, "CACHE_DIR", tmp_path)
    audit = object.__new__(AuditService)
    monkeypatch.setattr(audit, "_resolve_local_timestamp", lambda fp,t: (t,30))
    disk = threading.BoundedSemaphore(1)
    monkeypatch.setattr(scheduler, "get_disk_semaphore", lambda: disk)
    original_popen = subprocess.Popen
    children, result, errors = [], [], []
    def blocked_process(cmd, **kwargs):
        process = original_popen([sys.executable,"-c","import time; time.sleep(30)"], **kwargs)
        children.append(process)
        return process
    monkeypatch.setattr(subprocess, "Popen", blocked_process)
    def preview():
        try:
            result.append(audit.extract_frame(str(source),1))
        except BaseException as error:
            errors.append(error)
    worker = threading.Thread(target=preview,daemon=True)
    FFmpegProcessRegistry.reset_interrupted()
    worker.start()
    try:
        deadline = time.monotonic()+3
        while not FFmpegProcessRegistry._processes and time.monotonic()<deadline:
            time.sleep(.01)
        assert FFmpegProcessRegistry._processes and children
        FFmpegProcessRegistry.kill_all()
        worker.join(timeout=3)
        assert not worker.is_alive() and not errors and result == [None]
        assert all(p.poll() is not None for p in children)
        assert not FFmpegProcessRegistry._processes
        assert disk.acquire(blocking=False)
        disk.release()
        assert not list(tmp_path.glob("preview_*"))
    finally:
        FFmpegProcessRegistry.kill_all()
        worker.join(timeout=3)
        FFmpegProcessRegistry.reset_interrupted()


def test_preview_busy_io_does_not_spawn_or_publish(tmp_path,monkeypatch):
    import threading
    from scripts.audit_tool import service as module
    from src.hardware import scheduler, ffmpeg
    from src.core.visual_index import source_identity
    monkeypatch.setattr(module,"inspect_source_identity_owned",lambda fp,*args,**kwargs:tuple(source_identity(fp)))
    disk = threading.BoundedSemaphore(1)
    disk.acquire()
    monkeypatch.setattr(scheduler,"get_disk_semaphore",lambda:disk)
    monkeypatch.setattr(module,"CACHE_DIR",tmp_path)
    monkeypatch.setattr(ffmpeg.subprocess,"Popen",lambda *args,**kwargs:pytest.fail("Busy I/O spawned preview"))
    source = tmp_path/"source.mp4"
    source.write_bytes(b"existing source")
    audit = object.__new__(AuditService)
    assert audit._cached_preview(source,("frame","1.000",640),".jpg",["ffmpeg","-version"],.02) is None
    assert not list(tmp_path.glob("preview_*"))
    assert not disk.acquire(blocking=False)
    disk.release()


def test_target_export_does_not_archive_or_read_another_database(tmp_path,monkeypatch):
    from pathlib import Path
    from scripts import export_dataset
    db=VlogDatabase(tmp_path/"isolated.db")
    service=AuditService(db=db)
    def forbidden():
        raise AssertionError("Export must not modify archive or launch source extraction")
    monkeypatch.setattr(service,"batch_archive_all",forbidden)
    monkeypatch.setattr(export_dataset,"load_archive_items",lambda path:[])
    seen=[]
    monkeypatch.setattr(export_dataset,"load_db_fallback_items",lambda path:(seen.append(Path(path)) or [object()]))
    monkeypatch.setattr(export_dataset,"build_yolo_dataset",lambda **kwargs:{"usable":False,"excluded":1,"train_images":0,"val_images":0})
    try:
        result=service.export_yolo_dataset(output_dir=tmp_path/"out")
        assert seen==[db.db_path]
        assert result["success"] is False and result["train_cmd"] is None
        assert "事件标签" in result["message"]
    finally:db.close()


def test_exclude_http_returns_conflict_for_owned_day(tmp_path, monkeypatch):
    import json
    import http.cookiejar
    import threading
    import urllib.error
    import urllib.request
    from http.server import HTTPServer
    from scripts.audit_tool import app
    from src.core.process_lock import database_day_lease
    db = VlogDatabase(tmp_path / 'http.db')
    fp = str(tmp_path / 'source.mp4')
    db.add_file_task(fp, 0, '20260901', '20260901100000', '20260901100100', 60)
    monkeypatch.setattr(app, 'service', AuditService(db=db))
    server = HTTPServer(('127.0.0.1', 0), app.AuditHandler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f'http://127.0.0.1:{server.server_address[1]}'
    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    try:
        opener.open(url + '/').read()
        request = urllib.request.Request(url + '/api/exclude_file', data=json.dumps({'filepath': fp}).encode(),
                                         headers={'Content-Type': 'application/json'})
        with database_day_lease(db.db_path, '20260901', 0):
            with pytest.raises(urllib.error.HTTPError) as error:
                opener.open(request)
            assert error.value.code == 409
            assert 'already owned' in json.loads(error.value.read())['error']
            assert db.get_all_file_tasks_for_date('20260901', 0)[0]['analysis_status'] != 'SKIPPED_EXCLUDED'
        assert json.loads(opener.open(request).read())['success']
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=3)
        db.close()
