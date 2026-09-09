import logging
import subprocess
import time as _time
from dataclasses import dataclass

from src.scheduler import acquire_with_retry
from src.utils import load_config

logger = logging.getLogger("homevlog")


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
    stdout_consumer=None,
) -> FFmpegResult:
    if stdout_consumer is not None and not capture_output:
        raise ValueError("stdout_consumer requires capture_output=True")
    cmd = ["ffmpeg", "-hide_banner", "-y", "-nostdin"] + args
    kwargs = {"stdin": subprocess.DEVNULL}
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        
    from src.utils import get_disk_semaphore
    io_sem = get_disk_semaphore()
    # AGENTS.md 铁律：acquire 必须带 timeout 并重试，禁止无限阻塞
    if not acquire_with_retry(io_sem):
        from src.renderer import FFmpegProcessRegistry
        if FFmpegProcessRegistry.is_interrupted():
            return FFmpegResult(-1, b"", b"interrupted")
        logger.warning("run_ffmpeg: io semaphore acquire timeout")
        return FFmpegResult(
            returncode=-1,
            stdout=b"",
            stderr=b"io semaphore acquire timeout",
            timed_out=False,
            duration=0.0,
        )
    t0 = _time.monotonic()
    from src.renderer import FFmpegProcessRegistry
    proc = None
    key = f"ffmpeg:{id(args)}:{t0}"
    try:
        proc = subprocess.Popen(cmd, **kwargs)
        FFmpegProcessRegistry.register(key, proc)
        try:
            if stdout_consumer is None:
                stdout, stderr = proc.communicate(timeout=timeout)
            else:
                import threading
                from collections import deque
                errors = deque(maxlen=128)
                expired = threading.Event()
                def drain_errors():
                    while True:
                        block = proc.stderr.read(4096)
                        if not block:
                            break
                        errors.append(block)
                def expire():
                    expired.set()
                    proc.kill()
                reader = threading.Thread(target=drain_errors, daemon=True)
                timer = threading.Timer(timeout or 600, expire)
                timer.daemon = True
                reader.start()
                timer.start()
                try:
                    while True:
                        chunk = proc.stdout.read(65536)
                        if not chunk:
                            break
                        stdout_consumer(chunk)
                    proc.wait(timeout=5)
                    if expired.is_set():
                        raise subprocess.TimeoutExpired(cmd, timeout)
                finally:
                    timer.cancel()
                    if proc.poll() is None:
                        proc.kill()
                        proc.wait(timeout=5)
                    reader.join(timeout=5)
                    proc.stdout.close()
                    proc.stderr.close()
                stdout, stderr = b"", b"".join(errors)
            returncode = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            proc.kill()
            if stdout_consumer is None:
                stdout, stderr = proc.communicate()
            else:
                stdout, stderr = b"", b"stream timed out"
            returncode = -9
            timed_out = True
    except OSError as e:
        # ffmpeg 可执行文件缺失或 spawn 失败：返回结构化错误而非 UnboundLocalError
        return FFmpegResult(
            returncode=-1,
            stdout=b"",
            stderr=f"ffmpeg spawn failed: {e}".encode("utf-8", errors="replace"),
            timed_out=False,
            duration=_time.monotonic() - t0,
        )
    finally:
        try:
            if proc is not None and proc.poll() is None:
                proc.kill()
                proc.wait(timeout=5)
        finally:
            FFmpegProcessRegistry.deregister(key)
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
    
    from src.utils import get_disk_semaphore
    io_sem = get_disk_semaphore()
    # AGENTS.md 铁律：acquire 必须带 timeout 并重试，禁止无限阻塞
    if not acquire_with_retry(io_sem):
        logger.warning("run_ffprobe: io semaphore acquire timeout for %s", filepath)
        return None
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
