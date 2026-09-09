"""Content identity and container validation for atomic render artifacts."""
import hashlib
import json
from pathlib import Path


def file_identity(path):
    p = Path(path).resolve()
    try:
        stat = p.stat()
        return [str(p), stat.st_size, stat.st_mtime_ns]
    except OSError:
        return [str(p), None, None]


def render_fingerprint(inputs, filtergraph, encoder, fps, output, audio, config):
    from src.utils import PROJECT_ROOT
    model = Path(config.get("yolo", {}).get("model_path", "models/yolo11m.pt"))
    if not model.is_absolute():
        model = PROJECT_ROOT / model
    payload = [2, [file_identity(p) for p in inputs], filtergraph, encoder, fps,
               output, audio, config, file_identity(model)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def processing_fingerprint(filepath, config):
    selected = {key: config.get(key, {}) for key in ("detection", "yolo", "audio_vad", "segment")}
    return render_fingerprint([filepath], "analysis-v2", "analysis", 0, {}, {}, selected)


def valid_video(path, expected_duration=None):
    """Reject empty/corrupt output, not legitimate small videos."""
    import av
    try:
        if Path(path).stat().st_size <= 0:
            return False
        with av.open(str(path), timeout=10.0) as container:
            if not container.streams.video:
                return False
            stream = container.streams.video[0]
            duration = (float(stream.duration * stream.time_base) if stream.duration
                        else float(container.duration or 0) / av.time_base)
            if duration <= 0 or next(container.decode(stream), None) is None:
                return False
            return (expected_duration is None or
                    abs(duration - expected_duration) <= max(5.0, expected_duration * 0.05))
    except Exception:
        return False


def _manifest_path(path: Path | str) -> Path:
    p = Path(path)
    # 若在 output 目录下，统一将指纹清单收纳于 output/.manifest/ 隐藏目录中
    if p.parent.name == "output":
        return p.parent / ".manifest" / (p.name + ".json")
    return Path(str(p) + ".json")


def reusable(path, fingerprint, expected_duration=None):
    try:
        manifest_p = _manifest_path(path)
        if not manifest_p.exists():
            # 兼容读取旧版同目录平铺文件
            legacy_p = Path(str(path) + ".json")
            if legacy_p.exists():
                manifest_p = legacy_p
            else:
                return False
        manifest = json.loads(manifest_p.read_text(encoding="utf-8"))
        return (manifest["fingerprint"] == fingerprint and
                manifest["output"] == file_identity(path) and valid_video(path, expected_duration))
    except (OSError, ValueError, KeyError):
        return False


def save_manifest(path, fingerprint):
    manifest = _manifest_path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(manifest.name + ".tmp")
    temporary.write_text(json.dumps({"fingerprint": fingerprint, "output": file_identity(path)}), encoding="utf-8")
    temporary.replace(manifest)
