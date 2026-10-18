"""Measured production replays; no alternate detector or predicted heuristics."""
import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import subprocess

from src.core import config as core
from src.core.profiles import resolve_camera_config


class ReplayUnavailable(ValueError):
    """The requested measurement is missing, stale or incompletely covered."""


def content_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def receipt_sha256(receipt):
    payload = {key: value for key, value in receipt.items() if key != "payload_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def runtime_identity():
    packages = {}
    for name in ("av", "numpy", "opencv-python", "opencv-python-headless", "torch", "ultralytics"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5, check=True)
        packages["ffmpeg"] = result.stdout.decode(errors="replace").splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        packages["ffmpeg"] = None
    return packages


def production_code_sha256(root=None):
    root = Path(root or core.PROJECT_ROOT)
    files = sorted((root / "src").rglob("*.py")) + [root / "scripts/replay_benchmark.py"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_path(config):
    path = Path(config.get("yolo", {}).get("model_path", "models/yolo11m.pt"))
    return path if path.is_absolute() else core.PROJECT_ROOT / path


def sample_key(sample):
    return json.dumps([sample["filepath"], sample["date"], sample["cam_index"],
                       sample["start_time"], sample["end_time"]], separators=(",", ":"))


def config_sha256(config, samples):
    normalized = copy.deepcopy(config)
    normalized.setdefault("yolo", {})["model_path"] = str(model_path(config).resolve())
    effective = {}
    for sample in samples:
        key = (sample["filepath"], sample["cam_index"])
        effective[str(key)] = resolve_camera_config(normalized, filepath=key[0], cam_index=key[1])
    return hashlib.sha256(json.dumps(effective, sort_keys=True, default=str).encode()).hexdigest()


def project_predictions(samples, tasks, segments):
    """Project actual persisted source-time segments onto reference intervals."""
    task_map = {task["filepath"]: task for task in tasks}
    by_file = {}
    for segment in segments:
        by_file.setdefault(segment["filepath"], []).append(segment)
    projected = {}
    for sample in samples:
        task = task_map.get(sample["filepath"])
        if task and task.get("prescreen_status") == "STATIC" and not by_file.get(sample["filepath"]):
            from src.core.utils import ts_to_unix
            if task.get("file_start_time") and task.get("file_duration") is not None:
                offset = ts_to_unix(task["file_start_time"]) - ts_to_unix(sample["date"] + "000000")
                by_file[sample["filepath"]] = [{"start_time": offset, "end_time": offset + float(task["file_duration"]),
                                                "state": "STATIC"}]
        if not task or (task["analysis_status"] != "ANALYZED" and task.get("prescreen_status") != "STATIC"):
            projected[sample_key(sample)] = None
            continue
        start, end = float(sample["start_time"]), float(sample["end_time"])
        if end <= start:
            raise ValueError("Reference intervals must have positive duration")
        overlap = [(segment, max(0., min(end, segment["end_time"]) - max(start, segment["start_time"])))
                   for segment in by_file.get(sample["filepath"], [])]
        cursor = start
        continuous = True
        for lower, upper in sorted((max(start, segment["start_time"]), min(end, segment["end_time"]))
                                   for segment, duration in overlap if duration > 0):
            if abs(lower - cursor) > .01:
                continuous = False
                break
            cursor = upper
        if not continuous or abs(cursor - end) > .01:
            projected[sample_key(sample)] = None
        else:
            projected[sample_key(sample)] = any(duration > 0 and segment["state"] in
                                               ("DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE", "MICRO_MOTION", "NIGHT_STATIONARY")
                                               for segment, duration in overlap)
    return projected


class ProductionReplayStore:
    """Read immutable measurements produced by the isolated CLI, never launch jobs.

    Matching requires code, full effective configuration, model, source stats,
    reference identity and complete interval coverage. Source SHA is captured by
    the producer; stat checks here detect ordinary drift without re-reading GBs.
    """
    def __init__(self, directory=None):
        self.directory = Path(directory) if directory else core.TEMP_DIR / "benchmark_replays"

    def _receipts(self):
        for path in sorted(self.directory.glob("*/receipt.json"), reverse=True):
            try:
                receipt = json.loads(path.read_text(encoding="utf-8"))
                required = {"code_sha256", "model_sha256", "config_sha256", "references", "sources",
                            "predictions", "created_at", "payload_sha256", "runtime"}
                if (required.issubset(receipt) and receipt.get("schema") == 1
                        and receipt.get("basis") == "production_pipeline"
                        and receipt["payload_sha256"] == receipt_sha256(receipt)):
                    yield receipt
            except (OSError, ValueError):
                continue

    def references(self):
        receipts = list(self._receipts())
        if not receipts:
            return None
        return max(receipts, key=lambda item: item["created_at"])["references"]

    def matching_receipt(self, config, samples):
        code_digest = production_code_sha256()
        runtime = runtime_identity()
        model = model_path(config)
        model_digest = content_sha256(model) if model.exists() else None
        requested = {sample_key(sample) for sample in samples}
        for receipt in self._receipts():
            if (receipt["code_sha256"] != code_digest or receipt["model_sha256"] != model_digest
                    or receipt["runtime"] != runtime
                    or not requested.issubset(receipt["predictions"])):
                continue
            # Signatures include the complete source set of the measured job.
            if config_sha256(config, receipt["references"]) != receipt["config_sha256"]:
                continue
            sources_current = True
            try:
                for source in receipt["sources"]:
                    stat = Path(source["filepath"]).stat()
                    if (stat.st_size, stat.st_mtime_ns) != (source["size"], source["mtime_ns"]):
                        sources_current = False
                        break
            except OSError:
                continue
            if not sources_current:
                continue
            values = [receipt["predictions"][sample_key(sample)] for sample in samples]
            if all(type(value) is bool for value in values):
                return receipt
        raise ReplayUnavailable("没有匹配当前代码、完整配置、模型、源身份和区间覆盖的实测记录；请先运行隔离生产重放命令")

    def __call__(self, config, samples):
        return self.measure(config, samples)["predictions"]

    def details(self, config, samples):
        return self.measure(config, samples)["details"]

    def measure(self, config, samples):
        """Return retention and display data from one validated immutable receipt."""
        receipt = self.matching_receipt(config, samples)
        diagnostics = receipt.get("interval_diagnostics", {})
        return {"predictions": [receipt["predictions"][sample_key(sample)] for sample in samples],
                "details": [diagnostics.get(sample_key(sample), {"status": "UNAVAILABLE"}) for sample in samples],
                "receipt_sha256": receipt["payload_sha256"],
                "identity": {key: receipt[key] for key in ("code_sha256", "model_sha256", "runtime")}
                            | {"sources": sorted(receipt["sources"], key=lambda source: source["filepath"])}}


def interval_display_details(config, timeline, start, end, offset):
    """Shared proportional display contribution; not a fractional-frame oracle."""
    from src.stages.timeline import compute_display_plans, display_plan_options
    plans = compute_display_plans(timeline, **display_plan_options(config))
    portions = []
    for segment, (display, _) in zip(timeline, plans):
        lower, upper = offset + segment.start_in_file, offset + segment.end_in_file
        overlap = max(0., min(end, upper) - max(start, lower))
        if overlap:
            portions.append({"start_time": max(start, lower), "end_time": min(end, upper),
                             "state": segment.state, "source_s": overlap,
                             "display_s": display * overlap / segment.duration,
                             "speed": segment.duration / display})
    return {"normal_or_presence": any(p["state"] in ("DYNAMIC", "DYNAMIC_AUDIO", "PRESENCE") for p in portions),
            "portions": portions, "source_s": end-start,
            "display_s": sum(p["display_s"] for p in portions),
            "normal_source_s": sum(p["source_s"] for p in portions if p["state"] in ("DYNAMIC", "DYNAMIC_AUDIO")),
            "retained_source_s": sum(p["source_s"] for p in portions if p["state"] != "STATIC")}


def interval_diagnostics(config, samples, tasks, segments):
    """Measured source coverage and display-clock contributions, not human truth."""
    from src.stages.timeline import build_timeline_from_rows
    task_map = {task["filepath"]: dict(task) for task in tasks}
    segment_map = {}
    for segment in segments:
        segment_map.setdefault(segment["filepath"], []).append(dict(segment))
    diagnostics = {}
    for sample in samples:
        key = sample_key(sample)
        task = task_map.get(sample["filepath"])
        prediction = project_predictions([sample], tasks, segments)[key]
        if prediction is None or not task.get("file_start_time") or not task.get("file_end_time"):
            diagnostics[key] = {"status": "UNAVAILABLE"}
            continue
        effective = resolve_camera_config(config, filepath=sample["filepath"], cam_index=sample["cam_index"])
        row = dict(task, segments=segment_map.get(sample["filepath"], []), human_reviews=[])
        timeline = build_timeline_from_rows([row], sample["date"], config=effective, resolve_presence=False)
        from src.core.utils import ts_to_unix
        offset = ts_to_unix(task["file_start_time"]) - ts_to_unix(sample["date"] + "000000")
        diagnostics[key] = {"status": "MEASURED", **interval_display_details(
            effective, timeline, sample["start_time"], sample["end_time"], offset)}
    return diagnostics


def replay_receipt(config, samples, sources, tasks, segments, *, code_sha256, model_sha256):
    return {"schema": 1, "basis": "production_pipeline", "created_at": datetime.now(timezone.utc).isoformat(),
            "code_sha256": code_sha256, "model_sha256": model_sha256,
            "config_sha256": config_sha256(config, samples), "sources": sources,
            "runtime": runtime_identity(),
            "references": samples, "predictions": project_predictions(samples, tasks, segments),
            "interval_diagnostics": interval_diagnostics(config, samples, tasks, segments),
            "scope": "clip_nonstatic_retention", "source_revalidation": "captured_sha256_and_current_stat"}
