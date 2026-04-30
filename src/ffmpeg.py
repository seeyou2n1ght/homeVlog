import subprocess
import time as _time
from dataclasses import dataclass

from src.utils import load_config


@dataclass
class FFmpegResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    timed_out: bool = False
    duration: float = 0.0  # wall-clock seconds

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")


def run_ffmpeg(
    args: list[str],
    timeout: float | None = None,
    capture_output: bool = True,
    log_stderr: bool = False,
) -> FFmpegResult:
    cmd = ["ffmpeg", "-hide_banner", "-y"] + args
    kwargs = {}
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
    proc = subprocess.Popen(cmd, **kwargs)
    t0 = _time.monotonic()
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        returncode = proc.returncode
        timed_out = False
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate()
        returncode = -9
        timed_out = True
    elapsed = _time.monotonic() - t0

    return FFmpegResult(
        returncode=returncode,
        stdout=stdout or b"",
        stderr=stderr or b"",
        timed_out=timed_out,
        duration=elapsed,
    )


def run_ffprobe(filepath: str, timeout: float | None = None) -> dict | None:
    import json
    if timeout is None:
        timeout = load_config().get("detection", {}).get("prescreen_extract_timeout", 30.0)
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(filepath),
    ]
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        return json.loads(proc.stdout.decode("utf-8"))
    except json.JSONDecodeError:
        return None


def get_duration(filepath: str) -> float | None:
    info = run_ffprobe(str(filepath))
    if info is None:
        return None
    fmt = info.get("format", {})
    dur = fmt.get("duration")
    return float(dur) if dur else None


def build_qsv_decode_args(
    input_path: str,
    width: int = 320,
    height: int = 180,
    fps: int | None = None,
    start_time: float | None = None,
    duration: float | None = None,
    vframes: int | None = None,
) -> list[str]:
    """Build QSV hardware decode args for light prescreen pass."""
    args = [
        "-hwaccel", "qsv",
        "-hwaccel_output_format", "qsv",
    ]
    if start_time is not None:
        args += ["-ss", str(start_time)]
    args += ["-i", str(input_path)]
    if duration is not None:
        args += ["-t", str(duration)]
    vf_parts = [f"scale_qsv=w={width}:h={height}", "hwdownload", "format=nv12"]
    args += ["-vf", ",".join(vf_parts)]
    if fps:
        args += ["-r", str(fps)]
    if vframes is not None:
        args += ["-vframes", str(vframes)]
    args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return args


