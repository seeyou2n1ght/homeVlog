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


def _validate_checkpoint_chunk(chunk: list[float], path: Path | str, duration: float) -> bool:
    import av
    import logging
    _log = logging.getLogger("homevlog")
    try:
        with av.open(str(path), timeout=10.0) as container:
            if not container.streams.video:
                _log.warning("valid_video checkpoint chunk: file %s has no video stream", path)
                return False
            stream = container.streams.video[0]
            frame_step = 1.0 / float(stream.average_rate or 20)
            for boundary in chunk:
                start = max(0.0, boundary - 0.5)
                stop = min(duration, boundary + 0.5)
                container.seek(int(start / stream.time_base), stream=stream, backward=True)
                seen = []
                for frame in container.decode(stream):
                    if frame.is_corrupt or frame.pts is None:
                        _log.warning("valid_video: corrupt frame at boundary %.2fs in %s", boundary, path)
                        return False
                    timestamp = float(frame.pts * frame.time_base)
                    if timestamp >= start:
                        seen.append(timestamp)
                    if timestamp >= stop:
                        break
                if not seen or seen[0] > start + 2 * frame_step or seen[-1] < stop - 2 * frame_step:
                    _log.warning("valid_video: boundary %.2fs gap or frame discontinuity in %s", boundary, path)
                    return False
        return True
    except Exception as e:
        _log.warning("valid_video checkpoint chunk exception for %s: %s", path, e)
        return False


def valid_video(path, expected_duration=None, checkpoints=()):
    """Reject empty/corrupt output, not legitimate small videos."""
    import av
    import logging
    _log = logging.getLogger("homevlog")
    try:
        p = Path(path)
        if not p.exists() or p.stat().st_size <= 0:
            _log.warning("valid_video: file %s does not exist or is 0 bytes", path)
            return False
        with av.open(str(path), timeout=10.0) as container:
            if not container.streams.video:
                _log.warning("valid_video: file %s has no video stream", path)
                return False
            stream = container.streams.video[0]
            duration = (float(stream.duration * stream.time_base) if stream.duration
                        else float(container.duration or 0) / av.time_base)
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

            if expected_duration is not None:
                # 监控摄像头录像末尾常因物理丢包或分段截断提前数秒 EOF，且静态段变速平滑存在微小浮点累计误差。
                # 保持短视频 5.0s 下限拦截结构性坏片，将长视频自适应门限从 5% 适度放宽至 11%（允许约 10% 监控切片末尾波动）。
                tol = max(5.0, expected_duration * 0.11)
                diff = abs(duration - expected_duration)
                if diff > tol:
                    _log.warning(
                        "valid_video: file %s duration mismatch: actual=%.3fs, expected=%.3fs (diff=%.3fs > tol=%.3fs)",
                        path, duration, expected_duration, diff, tol,
                    )
                    return False
            if checkpoints:
                checkpoints_list = list(checkpoints)
                if len(checkpoints_list) <= 4:
                    frame_step = 1 / float(stream.average_rate or 20)
                    for boundary in checkpoints_list:
                        # Decode across each join, not just an independently seekable IDR.
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
                        if not seen or seen[0] > start + 2 * frame_step or seen[-1] < stop - 2 * frame_step:
                            return False
                else:
                    from concurrent.futures import ThreadPoolExecutor
                    num_workers = min(8, len(checkpoints_list))
                    chunks = [checkpoints_list[i::num_workers] for i in range(num_workers) if checkpoints_list[i::num_workers]]
                    with ThreadPoolExecutor(max_workers=num_workers) as pool:
                        results = list(pool.map(lambda c: _validate_checkpoint_chunk(c, path, duration), chunks))
                    if not all(results):
                        return False
            return True
    except Exception as e:
        _log.warning("valid_video: file %s validation exception: %s", path, e, exc_info=True)
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
