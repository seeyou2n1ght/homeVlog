"""Shared content, environment and model identities for delivery and isolated runs."""
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
