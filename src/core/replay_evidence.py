"""Shared identities and display measurements for delivery, review and isolated runs."""
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import subprocess

from src.core import config as core


def content_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    files = sorted((root / "src").rglob("*.py")) + [root / "main.py", root / "scripts/benchmark_day.py"]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model_path(config):
    path = Path(config.get("yolo", {}).get("model_path", "models/yolo11m.pt"))
    return path if path.is_absolute() else core.PROJECT_ROOT / path


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
