"""Independent viewing editions over delivered frames; never analysis or training labels."""
from collections import defaultdict, deque
from copy import deepcopy
from datetime import datetime
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import threading
import uuid

from src.core.replay_journal import atomic_json, digest
from src.hardware.ffmpeg import run_ffmpeg, MediaTerminationUnconfirmed, MEDIA_IMPLEMENTATION_SHA256
from src.hardware.scheduler import VideoLease, get_nvenc_semaphore

# Captured at import, so later disk edits cannot silently claim to be the running exporter.
IMPLEMENTATION = {
    "scripts/audit_tool/watch_export.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "src/hardware/ffmpeg.py": MEDIA_IMPLEMENTATION_SHA256,
}
for _relative in ("src/hardware/scheduler.py", "src/core/process_lock.py", "src/core/replay_journal.py"):
    IMPLEMENTATION[_relative] = hashlib.sha256((Path(__file__).resolve().parents[2]/_relative).read_bytes()).hexdigest()

POLICY = "whole-delivery-segments-round-robin-v2"
POSITIVE = {"CONFIRMED_MOTION", "MISSED_MOTION"}
NEGATIVE = {"FALSE_ALARM", "CONFIRMED_STATIC"}


def canonical(path):
    return os.path.normcase(os.path.abspath(os.path.normpath(str(path))))


def file_identity(path):
    p = Path(path)
    stat = p.stat()
    if not p.is_file() or stat.st_size <= 0:
        raise ValueError(f"Media unavailable: {p}")
    return {"path": canonical(p), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def source_identity(path):
    try:
        return {**file_identity(path), "available": True}
    except (OSError, ValueError) as exc:
        return {"path": canonical(path), "available": False, "error": type(exc).__name__}


def overlap(a, b):
    return a["source_file"] == b["source_file"] and min(a["end"], b["end"]) > max(a["start"], b["start"]) + 1e-7


def read_inputs(db_path, date, camera):
    """Read transaction, no VlogDatabase constructor or migration."""
    with sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN")
        files = [dict(r) for r in conn.execute(
            "SELECT * FROM file_tasks WHERE date=? AND cam_index=? ORDER BY file_start_time,id", (date, camera))]
        segments = [dict(r) for r in conn.execute(
            "SELECT * FROM segments WHERE date=? AND cam_index=? ORDER BY filepath,start_time,id", (date, camera))]
        reviews = [dict(r) for r in conn.execute(
            "SELECT h.* FROM human_reviews h JOIN file_tasks f ON h.filepath=f.filepath "
            "WHERE f.date=? AND f.cam_index=? ORDER BY h.reviewed_at,h.id", (date, camera))]
    if not files:
        raise ValueError("No matching frozen analysis")
    analysis = []
    for f in files:
        for index, seg in enumerate(json.loads(f.get("analysis_segments") or "[]")):
            offset = float(seg.get("file_start_offset") or 0)
            analysis.append({**seg, "source_file": canonical(f["filepath"]),
                "start": float(seg["start_time"]) - offset, "end": float(seg["end_time"]) - offset,
                "provenance": "frozen_analysis", "analysis_index": index})
    # human_reviews are persistent file-relative intervals; segment projections use day offsets.
    human = [{**r, "source_file": canonical(r["filepath"]), "start": float(r["start_time"]),
              "end": float(r["end_time"]), "provenance": "human_reviews"} for r in reviews]
    return {"files": files, "segments": segments, "human": human, "analysis": analysis}


def analysis_disputes(evidence):
    reasons = []
    reason = (evidence.get("review_reason") or "").upper()
    if evidence.get("needs_review"):
        reasons.append("analysis_review_requested")
    if any(x in reason for x in ("FAIL", "INSUFFICIENT", "NEGATIVE", "UNAVAILABLE", "BORDERLINE")):
        reasons.append("detection_failed_or_insufficient")
    if "TARGET_PERSISTENCE" in reason:
        reasons.append("inferred_persistence_not_observation")
    if evidence.get("state") == "DYNAMIC_AUDIO" or "AUDIO" in reason:
        reasons.append("audio_not_listened")
    if evidence.get("state") != "STATIC" and not reason and float(evidence.get("avg_confidence") or 0) <= 0:
        reasons.append("no_observed_target_evidence")
    return reasons


def plan_selection(mapping, analysis, human, *, budget_s=1200, file_starts=None):
    """Pure conservation/selection: whole mapped segments, evidence never averaged."""
    fps = Fraction(str(mapping["fps"]))
    if fps <= 0 or not math.isfinite(float(budget_s)) or not 0 < budget_s <= 1200:
        raise ValueError("Viewing budget must be in (0, 1200] seconds")
    budget = int(Fraction(str(budget_s)) * fps)
    cursor, items, review = 0, [], []
    file_starts = file_starts or {}
    for source in mapping["segments"]:
        first, last = source["output_start_frame"], source["output_end_frame"]
        if not isinstance(first, int) or not isinstance(last, int) or first != cursor or last <= first:
            raise ValueError("Parent frame mapping is not a complete partition")
        cursor = last
        item = {**source, "source_file": canonical(source["source_file"]),
                "start": float(source["source_start_s"]), "end": float(source["source_end_s"])}
        if not 0 <= item["start"] < item["end"]:
            raise ValueError("Invalid source mapping interval")
        item["item_id"] = digest(item)
        ae = [deepcopy(e) for e in analysis if overlap(item, e)]
        he = [deepcopy(e) for e in human if overlap(item, e)]
        reasons = sorted({r for e in ae for r in analysis_disputes(e)})
        covered = item["start"]
        for e in sorted(ae, key=lambda e: e["start"]):
            if e["start"] > covered + .001:
                break
            covered = max(covered, e["end"])
        if covered < item["end"] - .001:
            reasons.append("analysis_association_unknown")
        negative = [e for e in he if e.get("manual_label") in NEGATIVE]
        positive = [e for e in he if e.get("manual_label") in POSITIVE]
        if any(e.get("manual_label") not in POSITIVE | NEGATIVE for e in he):
            reasons.append("unknown_human_label")
        if positive and negative:
            reasons.append("mixed_human_decisions")
        if negative:
            reasons.append("human_negative_overlap")
        priority = bool(positive) and not negative
        start = file_starts.get(item["source_file"], 0) + item["start"]
        item.update(evidence={"analysis": ae, "human": he},
            dispute_reasons=sorted(set(reasons)), priority=priority, wallclock_start_s=start,
            evidence_status="mixed_human_decisions" if positive and negative else ("human_decision" if he else
                ("disputed" if reasons else "algorithm_candidate")),
            includes_unreviewed_context=bool(he), frames=last-first,
            disposition="priority_pending" if priority else ("review" if reasons else
                ("background" if item["state"] == "STATIC" else "budget_unselected")))
        items.append(item)
    if cursor != mapping["planned_frames"]:
        raise ValueError("Parent frame mapping does not close")
    required = sum(i["frames"] for i in items if i["priority"])
    if required > budget:
        return {"status": "BUDGET_CONFLICT", "required_seconds": float(required/fps),
                "budget_s": budget_s, "items": items, "review": [], "selected_frames": 0}
    selected = {i["item_id"] for i in items if i["priority"]}
    remaining = budget - required
    groups = defaultdict(deque)
    for i in sorted(items, key=lambda i: (i["wallclock_start_s"], i["output_start_frame"])):
        if i["disposition"] == "budget_unselected":
            groups[int(i["wallclock_start_s"]//1800)].append(i)
    while groups:
        for key in sorted(list(groups)):
            item = groups[key].popleft()
            if item["frames"] <= remaining:
                selected.add(item["item_id"])
                remaining -= item["frames"]
            if not groups[key]:
                del groups[key]
    new_cursor = 0
    for i in sorted(items, key=lambda i: (i["wallclock_start_s"], i["output_start_frame"])):
        if i["item_id"] in selected:
            i.update(disposition="daily", daily_start_frame=new_cursor, daily_end_frame=new_cursor+i["frames"])
            new_cursor += i["frames"]
    for i in items:
        if i["dispute_reasons"]:
            review.append({"review_id": i["item_id"], "item_ids": [i["item_id"]], "source_file": i["source_file"],
                "start": i["start"], "end": i["end"], "reasons": i["dispute_reasons"], "evidence": i["evidence"],
                "daily_selected": i["disposition"] == "daily"})
    return {"status": "AVAILABLE" if new_cursor else "EMPTY", "policy": POLICY, "fps": float(fps),
        "budget_s": budget_s, "selected_frames": new_cursor, "duration_s": float(new_cursor/fps),
        "required_seconds": float(required/fps), "parent_frames": cursor, "items": items, "review": review,
        "counts": {key: sum(i["disposition"] == key for i in items)
                   for key in ("daily", "review", "budget_unselected", "background")},
        "review_total": len(review),
        "notice": "算法候选用于观看编排；人工审核与算法候选分别记录。完整映射段包含未复核上下文。预算未选不等于误检。"}


def inspect_parent(media, mapping):
    import av
    identity = file_identity(media)
    expected = mapping["media_identity"]
    if any(identity[k] != expected[k] for k in ("size", "mtime_ns")):
        raise ValueError("Parent media identity drift")
    with av.open(str(media)) as container:
        stream = container.streams.video[0]
        frame = next(container.decode(stream))
        origin = float(frame.pts * frame.time_base)
        fps, frames = float(stream.average_rate), stream.frames
        if abs(fps - mapping["fps"]) > .001 or frames != mapping["planned_frames"]:
            raise ValueError("Parent frame clock drift")
        if not container.streams.audio:
            raise ValueError("Parent audio unavailable")
    if file_identity(media) != identity:
        raise ValueError("Parent changed during inspection")
    return {**identity, "video_origin_s": origin, "fps": fps, "frames": frames}


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def capture(db_path, media, date, camera, budget_s=1200):
    metadata = Path(media).with_suffix(".meta.json")
    raw = metadata.read_bytes()
    meta = json.loads(raw)
    if meta.get("date") != date or meta.get("camera", {}).get("cam_index") != camera:
        raise ValueError("Parent date/camera mismatch")
    mapping = meta["delivery_mapping"]
    if mapping.get("status") != "AVAILABLE" or mapping.get("version") != 1:
        raise ValueError("Parent mapping unavailable")
    parent = inspect_parent(media, mapping)
    inputs = read_inputs(db_path, date, camera)
    starts = {}
    for f in inputs["files"]:
        stamp = str(f.get("file_start_time") or "")
        if len(stamp) < 14:
            raise ValueError("Missing source wall clock")
        starts[canonical(f["filepath"])] = int(stamp[8:10])*3600 + int(stamp[10:12])*60 + int(stamp[12:14])
    plan = plan_selection(mapping, inputs["analysis"], inputs["human"],
                          budget_s=budget_s, file_starts=starts)
    binding = {"parent": parent, "metadata_sha256": hashlib.sha256(raw).hexdigest(),
               "analysis_sha256": digest(inputs),
               "policy": POLICY, "implementation_sha256": IMPLEMENTATION, "date": date, "camera": camera, "budget_s": budget_s,
               "selection_sha256": digest(plan),
               "sources": {canonical(f["filepath"]): source_identity(f["filepath"]) for f in inputs["files"]}}
    return {"snapshot": digest(binding), "binding": binding, "plan": plan,
            "mapping": mapping, "inputs": inputs}


def _decode_daily(media, frames, fps, cancelled=None):
    """Decode all output video/audio, including every join; verify integer clocks."""
    import av
    count, samples, audio_end = 0, 0, 0.
    with av.open(str(media)) as container:
        for frame in container.decode(video=0):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("Export cancelled during validation")
            timestamp = float(frame.pts * frame.time_base)
            if abs(timestamp - count/fps) > float(frame.time_base) + 1e-7:
                raise ValueError(f"Daily video clock discontinuity at {count}: {timestamp}")
            count += 1
    with av.open(str(media)) as container:
        for frame in container.decode(audio=0):
            if cancelled is not None and cancelled.is_set():
                raise InterruptedError("Export cancelled during validation")
            timestamp = float(frame.pts * frame.time_base)
            if abs(timestamp - audio_end) > 1/frame.sample_rate + 1e-7:
                raise ValueError("Daily audio clock discontinuity")
            audio_end = timestamp + frame.samples/frame.sample_rate
            samples += frame.samples
    if count != frames or not samples or abs(audio_end - frames/fps) > 1024/48000 + 1e-6:
        raise ValueError(f"Daily AV closure failed: {count}, {audio_end}")
    return {"decoded_video_frames": count, "audio_samples": samples, "audio_end_s": audio_end,
            "duration_s": count/fps, "video_origin_s": 0, "all_joins_clock_checked": True}


def encode_daily(frozen, pending, cancel, progress):
    plan, parent = frozen["plan"], frozen["binding"]["parent"]
    fps = plan["fps"]
    selected = sorted((i for i in plan["items"] if i["disposition"] == "daily"),
                      key=lambda i: i["daily_start_frame"])
    chunks = []
    def execute(args, hardware=False):
        if cancel.is_set():
            raise InterruptedError("Export cancelled")
        result = run_ffmpeg(args, timeout=600, cancel_event=cancel, io_timeout=30,
                            io_lease=VideoLease(get_nvenc_semaphore()) if hardware else None)
        if cancel.is_set():
            raise InterruptedError("Export cancelled; media process exited")
        if result.returncode or result.timed_out:
            raise RuntimeError(result.stderr.decode("utf-8", errors="replace")[-3000:])
    for index, item in enumerate(selected):
        first, last = item["output_start_frame"], item["output_end_frame"]
        start = parent["video_origin_s"] + first/fps
        end = parent["video_origin_s"] + last/fps
        samples = round(item["frames"]*48000/fps)
        chunk = pending / f"part_{index:04d}.mkv"
        # Preserve original PTS across accurate input seeking; trim by the bound parent clock.
        graph = (f"[0:v]trim=start={start:.12f}:end={end:.12f},setpts=N/({fps}*TB)[v];"
                 f"[0:a]atrim=start={start:.12f}:end={end:.12f},asettb=1/48000,asetpts=N,"
                 f"aresample=48000,apad=whole_len={samples},atrim=end_sample={samples},asettb=1/48000,asetpts=N[a]")
        execute(["-ss", str(max(0, start-2)), "-copyts", "-i", parent["path"],
                 "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
                 "-c:v", "h264_nvenc", "-preset", "p4", "-qp", "18", "-bf", "0",
                 "-fps_mode", "passthrough", "-c:a", "pcm_s16le",
                 "-ar", "48000", "-ac", "2", str(chunk)], hardware=True)
        validate_chunk(chunk, item["frames"], fps)
        chunks.append(chunk)
        progress(completed_segments=index+1, total_segments=len(selected))
    listing = pending / "concat.txt"
    listing.write_text("".join("file '"+p.name+"'\n" for p in chunks), encoding="utf-8")
    samples = round(plan["selected_frames"]*48000/fps)
    progress(status="CONCATING")
    execute(["-f", "concat", "-safe", "0", "-i", str(listing), "-c:v", "copy", "-c:a", "aac",
             "-af", f"atrim=end_sample={samples},asettb=1/48000,asetpts=N", "-ar", "48000",
             "-movflags", "+faststart", str(pending/"daily.mp4")])
    progress(status="VALIDATING")
    validation = validate_daily(pending/"daily.mp4", plan["selected_frames"], fps, cancel)
    # Remove only owned temporary chunks after all readers have confirmed exit.
    for path in chunks + [listing]:
        path.unlink()
    return validation


class WatchExportManager:
    """One immutable package per snapshot. Failed attempts retain their own directory."""
    def __init__(self, db_path, output_root, workspace, parent_root=None):
        self.db_path, self.root = Path(db_path), Path(output_root)
        self.workspace = Path(workspace)
        self.parent_root = Path(parent_root) if parent_root else self.root
        self.lock = threading.RLock()
        self.tasks, self.workers, self.cancel_flags = {}, {}, {}

    def parent(self, date, camera):
        datetime.strptime(date, "%Y%m%d")
        if len(date) != 8 or int(camera) < 0:
            raise ValueError("Invalid date/camera")
        candidates = []
        for p in self.parent_root.glob("*.meta.json"):
            m = json.loads(p.read_text(encoding="utf-8"))
            if m.get("date") == date and m.get("camera", {}).get("cam_index") == int(camera):
                candidates.append(p.with_suffix("").with_suffix(".mp4"))
        if len(candidates) != 1:
            raise ValueError("Select one unambiguous parent delivery in the configured output directory")
        return candidates[0]

    def preview(self, date, camera, budget_s=1200):
        frozen = capture(self.db_path, self.parent(date, camera), date, camera, budget_s)
        p = frozen["plan"]
        return {"snapshot": frozen["snapshot"], **{k:v for k,v in p.items() if k not in ("items", "review")}}

    def directory(self, date, camera, export_id):
        if len(export_id) != 64 or any(c not in "0123456789abcdef" for c in export_id):
            raise ValueError("Invalid export identity")
        datetime.strptime(date, "%Y%m%d")
        if len(date) != 8 or int(camera) < 0:
            raise ValueError("Invalid date/camera")
        return self.root / "watch_exports" / f"{date}_cam{int(camera)}" / export_id

    def package(self, date, camera, export_id):
        directory = self.directory(date, camera, export_id)
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        if manifest["status"] != "COMPLETED" or manifest["snapshot"] != export_id or digest(manifest["binding"]) != export_id:
            raise ValueError("Incomplete export")
        for name in ("selection.json", "review.json"):
            if sha256(directory/name) != manifest["files"][name]["sha256"]:
                raise ValueError("Export manifest identity changed")
        if file_identity(directory/"daily.mp4") != manifest["files"]["daily.mp4"]["identity"]:
            raise ValueError("Daily media changed")
        return directory, manifest

    def start(self, date, camera, expected_snapshot, budget_s=1200):
        frozen = capture(self.db_path, self.parent(date, camera), date, camera, budget_s)
        if frozen["snapshot"] != expected_snapshot:
            raise ValueError("STALE_PREVIEW: inputs changed; preview again")
        if frozen["plan"]["status"] != "AVAILABLE":
            raise ValueError(frozen["plan"]["status"])
        key = frozen["snapshot"]
        directory = self.directory(date, camera, key)
        with self.lock:
            if directory.exists():
                _, manifest = self.package(date, camera, key)
                return {"status": "COMPLETED", "snapshot": key, "manifest": manifest}
            if key in self.workers and self.workers[key].is_alive():
                return deepcopy(self.tasks[key])
            cancel = self.cancel_flags[key] = threading.Event()
            self.tasks[key] = {"status": "QUEUED", "snapshot": key, "date": date, "camera": camera}
            worker = self.workers[key] = threading.Thread(target=self._run, args=(frozen, directory, cancel), daemon=True)
            worker.start()
            return deepcopy(self.tasks[key])

    def status(self, date, camera, export_id):
        directory = self.directory(date, camera, export_id)
        with self.lock:
            if export_id in self.tasks:
                return {**deepcopy(self.tasks[export_id]), "worker_alive": self.workers[export_id].is_alive()}
        if directory.exists():
            return {"status": "COMPLETED", "snapshot": export_id, "worker_alive": False}
        return {"status": "IDLE", "worker_alive": False}

    def cancel(self, export_id):
        with self.lock:
            worker = self.workers.get(export_id)
            event = self.cancel_flags.get(export_id)
            if not worker or not worker.is_alive():
                return False
            event.set()
            self.tasks[export_id]["status"] = "CANCELLING"
            return True

    def _run(self, frozen, directory, cancel):
        key = frozen["snapshot"]
        pending = directory.with_name(".pending_"+key+"_"+uuid.uuid4().hex)
        def update(**values):
            with self.lock:
                if cancel.is_set() and values.get("status") in ("RENDERING", "CONCATING", "VALIDATING"):
                    values["status"] = "CANCELLING"
                self.tasks[key].update(values)
        try:
            directory.parent.mkdir(parents=True, exist_ok=True)
            from src.core.process_lock import day_task_lease
            b = frozen["binding"]
            with day_task_lease(self.db_path, self.workspace, directory, b["date"], b["camera"]):
                if directory.exists():
                    self.package(b["date"], b["camera"], key)
                    update(status="COMPLETED")
                    return
                current = capture(self.db_path, self.parent(b["date"], b["camera"]), b["date"], b["camera"], b["budget_s"])
                if current["snapshot"] != key:
                    raise ValueError("STALE_PREVIEW: input drift before encoding")
                pending.mkdir()
                update(status="RENDERING", pending=str(pending))
                validation = encode_daily(frozen, pending, cancel, update)
                if capture(self.db_path, b["parent"]["path"], b["date"], b["camera"], b["budget_s"])["snapshot"] != key:
                    raise ValueError("Inputs changed during export")
                plan = frozen["plan"]
                atomic_json(pending/"selection.json", {k:v for k,v in plan.items() if k != "review"})
                atomic_json(pending/"review.json", {"total": len(plan["review"]), "items": plan["review"],
                    "media_dependency": "Original source availability checked on playback; not an offline review package"})
                files = {name: {"sha256": sha256(pending/name)} for name in ("daily.mp4", "selection.json", "review.json")}
                files["daily.mp4"]["identity"] = {**file_identity(pending/"daily.mp4"), "path": canonical(directory/"daily.mp4")}
                manifest = {"schema": 1, "status": "COMPLETED", "snapshot": key, "binding": b,
                            "files": files, "validation": validation,
                            "notice": "Historical cases are known regressions, not an independent holdout"}
                atomic_json(pending/"manifest.json", manifest)
                with self.lock:
                    if cancel.is_set():
                        raise InterruptedError("Export cancelled before publication")
                    pending.rename(directory)
                    update(status="COMPLETED", directory=str(directory), validation=validation)
        except MediaTerminationUnconfirmed as error:
            update(status="TERMINATION_UNCONFIRMED", error=str(error))
        except Exception as error:
            update(status="CANCELLED" if cancel.is_set() or isinstance(error, InterruptedError) else "FAILED", error=str(error))
        finally:
            if pending.exists():
                atomic_json(pending/"attempt.json", self.tasks[key])

    def listing(self, date, camera, export_id, kind="review", offset=0, limit=50):
        if kind not in ("review", "selection") or offset < 0 or not 1 <= limit <= 500:
            raise ValueError("Invalid list request")
        directory, manifest = self.package(date, camera, export_id)
        listing = json.loads((directory/(kind+".json")).read_text(encoding="utf-8"))
        return {"total": len(listing["items"]), "offset": offset, "items": listing["items"][offset:offset+limit],
                "manifest": manifest, "read_only": True}

    def playback(self, date, camera, export_id, kind="daily", item_id=None):
        directory, manifest = self.package(date, camera, export_id)
        if kind == "daily":
            return {"path": str(directory/"daily.mp4"), "identity": manifest["files"]["daily.mp4"]["identity"], "intervals": []}
        parent = manifest["binding"]["parent"]
        if kind == "full" and item_id is None:
            return {"path": parent["path"], "identity": parent, "intervals": []}
        selection = json.loads((directory/"selection.json").read_text(encoding="utf-8"))["items"]
        reviews = json.loads((directory/"review.json").read_text(encoding="utf-8"))["items"]
        item = next((i for i in selection+reviews if i.get("review_id", i.get("item_id")) == item_id), None)
        if item is None:
            raise ValueError("Unknown frozen interval")
        if kind == "source":
            identity = manifest["binding"]["sources"].get(item["source_file"], {"available": False})
            current = source_identity(item["source_file"])
            if not identity["available"] or current != identity:
                raise ValueError("Original media unavailable or identity changed; review incomplete")
            return {"path": item["source_file"], "identity": identity,
                    "intervals": [{"start": item["start"], "end": item["end"]}], "normal_speed": True}
        if kind != "full":
            raise ValueError("Invalid playback kind")
        intervals = [{"start": parent["video_origin_s"]+i["output_start_frame"]/parent["fps"],
                      "end": parent["video_origin_s"]+i["output_end_frame"]/parent["fps"]}
                     for i in selection if overlap(i, item)]
        if not intervals:
            raise ValueError("No full-version mapping for this source range; use original")
        return {"path": parent["path"], "identity": parent, "intervals": intervals, "includes_context": True}


def validate_chunk(path, frames, fps):
    import av
    # Packet clock checks include every extracted first/last boundary; final daily is fully decoded.
    with av.open(str(path)) as c:
        pts = sorted(float(p.pts*p.time_base) for p in c.demux(video=0) if p.pts is not None)
    if len(pts) != frames or any(abs(t-i/fps) > .0011 for i,t in enumerate(pts)):
        raise ValueError(f"Extracted segment frame boundary mismatch: {path.name}: {len(pts)} vs {frames}")


def validate_daily(media, frames, fps, cancelled=None):
    from src.hardware.scheduler import acquire_with_retry, get_disk_semaphore
    disk = get_disk_semaphore()
    if cancelled is not None and cancelled.is_set():
        raise InterruptedError("Export cancelled before validation admission")
    if not acquire_with_retry(disk, timeout=5, retries=1):
        raise RuntimeError("Daily validation I/O admission unavailable")
    try:
        return _decode_daily(media, frames, fps, cancelled)
    finally:
        disk.release()
