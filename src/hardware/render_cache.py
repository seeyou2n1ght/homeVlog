"""Content identity and container validation for atomic render artifacts."""
import hashlib
import json
import math
from pathlib import Path


def file_identity(path):
    p = Path(path).resolve()
    try:
        stat = p.stat()
        return [str(p), stat.st_size, stat.st_mtime_ns]
    except OSError:
        return [str(p), None, None]


def render_fingerprint(inputs, filtergraph, encoder, fps, output, audio, config):
    from src.core.config import PROJECT_ROOT
    model = Path(config.get("yolo", {}).get("model_path", "models/yolo11m.pt"))
    if not model.is_absolute():
        model = PROJECT_ROOT / model
    payload = [2 if encoder == "analysis" else 3, [file_identity(p) for p in inputs], filtergraph, encoder, fps,
               output, audio, config, file_identity(model)]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def processing_fingerprint(filepath, config):
    selected = {key: config.get(key, {}) for key in ("detection", "yolo", "audio_vad", "segment", "presence", "micro_motion")}
    return render_fingerprint([filepath], "analysis-v3-temporal-activity", "analysis", 0, {}, {}, selected)


def _window_is_continuous(container, stream, boundary: float, duration: float) -> bool:
    frame_step = 1.0 / float(stream.average_rate or 20)
    start = max(0.0, boundary - 0.5)
    stop = min(duration, boundary + 0.5)
    container.seek(int(start / stream.time_base), stream=stream, backward=True)
    seen = []
    for frame in container.decode(stream):
        if frame.is_corrupt or frame.pts is None:
            return False
        timestamp = float(frame.pts * frame.time_base)
        if timestamp >= start:
            seen.append(timestamp)
        if timestamp >= stop:
            break
    return bool(
        seen
        and seen[0] <= start + 2 * frame_step
        and seen[-1] >= stop - 2 * frame_step
        and all(0 < b - a <= 1.5 * frame_step for a, b in zip(seen, seen[1:]))
    )


def _validate_checkpoint_chunk(chunk: list[float], path: Path | str, duration: float) -> bool:
    import av
    import logging
    _log = logging.getLogger("homevlog")
    try:
        # 本地成片（尤其 4.5GB 大文件多线程解析尾部 moov atom 时）需要充足的 I/O 寻道容限，
        # 放宽至 60s 避免 Windows 磁盘刷盘争用时误触 PyAV AVERROR_EOF 超时中断
        with av.open(str(path), timeout=60.0) as container:
            if not container.streams.video:
                _log.warning("valid_video checkpoint chunk: file %s has no video stream", path)
                return False
            stream = container.streams.video[0]
            for boundary in chunk:
                if not _window_is_continuous(container, stream, boundary, duration):
                    _log.warning("valid_video: boundary %.2fs gap or frame discontinuity in %s", boundary, path)
                    return False
        return True
    except Exception as e:
        _log.warning("valid_video checkpoint chunk exception for %s: %s", path, e, exc_info=True)
        return False


def valid_video(path, expected_duration=None, checkpoints=(), require_audio=False):
    """Reject empty/corrupt output, not legitimate small videos."""
    import av
    import logging
    _log = logging.getLogger("homevlog")
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size <= 0:
            _log.warning("valid_video: file %s does not exist or is 0 bytes", path)
            return False
        with av.open(str(path), timeout=60.0) as container:
            if not container.streams.video:
                _log.warning("valid_video: file %s has no video stream", path)
                return False
            stream = container.streams.video[0]
            container_duration = float(container.duration or 0) / av.time_base
            duration = (float(stream.duration * stream.time_base) if stream.duration else container_duration)
            if duration <= 0:
                _log.warning("valid_video: file %s has invalid duration %.3fs", path, duration)
                return False
            try:
                first_frame = next(container.decode(stream), None)
                if first_frame is None:
                    _log.warning("valid_video: file %s could not decode first frame", path)
                    return False
            except Exception as dec_err:
                _log.warning("valid_video: file %s decode error: %s", path, dec_err)
                return False

            frame_step = 1 / float(stream.average_rate or 20)
            tolerance = min(0.5, max(0.05, 2 * frame_step))
            audio_duration = None
            if container.streams.audio:
                audio = container.streams.audio[0]
                audio_duration = (float(audio.duration * audio.time_base)
                                  if audio.duration else container_duration)
            elif require_audio:
                _log.warning("valid_video: file %s has no audio stream", path)
                return False
            if audio_duration is not None and abs(audio_duration - duration) > tolerance:
                _log.warning("valid_video: file %s A/V duration mismatch %.3fs", path, abs(audio_duration-duration))
                return False
            if expected_duration is not None:
                tol = tolerance
                diff = abs(duration - expected_duration)
                if diff > tol and not math.isclose(diff, tol, rel_tol=0, abs_tol=1e-9):
                    _log.warning(
                        "valid_video: file %s duration mismatch: actual=%.3fs, expected=%.3fs (diff=%.3fs > tol=%.3fs)",
                        path, duration, expected_duration, diff, tol,
                    )
                    return False
            if checkpoints:
                checkpoints_list = list(checkpoints)
                if len(checkpoints_list) <= 4:
                    for boundary in checkpoints_list:
                        if not _window_is_continuous(container, stream, boundary, duration):
                            return False
                else:
                    from concurrent.futures import ThreadPoolExecutor
                    num_workers = min(8, len(checkpoints_list))
                    chunks = [checkpoints_list[i::num_workers] for i in range(num_workers) if checkpoints_list[i::num_workers]]
                    with ThreadPoolExecutor(max_workers=num_workers) as pool:
                        results = list(pool.map(lambda c: _validate_checkpoint_chunk(c, path, duration), chunks))
                    if not all(results):
                        return False
            return _window_is_continuous(container, stream, max(0.5, duration - 0.5), duration)
    except Exception as e:
        _log.warning("valid_video: file %s validation exception: %s", path, e, exc_info=True)
        return False


def _manifest_path(path: Path | str) -> Path:
    p = Path(path)
    # 若在 output 目录下，统一将指纹清单收纳于 output/.manifest/ 隐藏目录中
    if p.parent.name == "output":
        return p.parent / ".manifest" / (p.name + ".json")
    return Path(str(p) + ".json")


def reusable(path, fingerprint, expected_duration=None, require_audio=False):
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
                manifest["output"] == file_identity(path) and
                valid_video(path, expected_duration, require_audio=require_audio))
    except (OSError, ValueError, KeyError):
        return False


def save_manifest(path, fingerprint):
    manifest = _manifest_path(path)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest.with_name(manifest.name + ".tmp")
    temporary.write_text(json.dumps({"fingerprint": fingerprint, "output": file_identity(path)}), encoding="utf-8")
    temporary.replace(manifest)
