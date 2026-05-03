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
        
    from src.utils import get_io_semaphore
    io_sem = get_io_semaphore()
    io_sem.acquire()
    try:
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
    finally:
        io_sem.release()
        
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
    
    from src.utils import get_io_semaphore
    io_sem = get_io_semaphore()
    io_sem.acquire()
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    finally:
        io_sem.release()
        
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


def build_hw_decode_args(
    input_path: str,
    width: int = 320,
    height: int = 180,
    fps: int | None = None,
    start_time: float | None = None,
    duration: float | None = None,
    vframes: int | None = None,
    gpu: str = "qsv",
) -> list[str]:
    """Build hardware decode args for light prescreen pass."""
    if gpu == "qsv":
        args = ["-hwaccel", "qsv", "-hwaccel_output_format", "qsv"]
        vf_parts = [f"scale_qsv=w={width}:h={height}", "hwdownload", "format=nv12"]
    else:
        args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        vf_parts = [f"scale_cuda={width}:{height}", "hwdownload", "format=nv12"]

    if start_time is not None:
        args += ["-ss", str(start_time)]
    args += ["-i", str(input_path)]
    if duration is not None:
        args += ["-t", str(duration)]
        
    args += ["-vf", ",".join(vf_parts)]
    if fps:
        args += ["-r", str(fps)]
    if vframes is not None:
        args += ["-vframes", str(vframes)]
    args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return args


def build_hw_fps_extract_args(
    input_path: str,
    fps_rate: float,
    width: int = 320,
    height: int = 180,
    gpu: str = "qsv",
) -> list[str]:
    """单流全速硬件解码并通过 fps 滤镜极速等距抽帧。
    相比 multi-seek 方案，该方案只需占用 1 个硬件 session，极大保护了带宽。
    """
    if gpu == "qsv":
        args = [
            "-hwaccel", "qsv",
            "-hwaccel_output_format", "qsv",
            "-i", str(input_path),
            "-vf", f"fps={fps_rate:.5f},scale_qsv=w={width}:h={height},hwdownload,format=nv12"
        ]
    else:
        args = [
            "-hwaccel", "cuda",
            "-hwaccel_output_format", "cuda",
            "-i", str(input_path),
            "-vf", f"fps={fps_rate:.5f},scale_cuda={width}:{height},hwdownload,format=nv12"
        ]
        
    args += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return args
