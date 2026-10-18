import logging
import os
import subprocess
import time as _time
import threading
import sys
from dataclasses import dataclass

from src.core.config import load_config
from src.core.process_lock import OwnedProcessJob

logger = logging.getLogger("homevlog")


class MediaTerminationUnconfirmed(RuntimeError):
    """An old media reader still owns resources; a second pipeline must stop."""


class MediaStartTerminationUnconfirmed(MediaTerminationUnconfirmed):
    """Hand a created but unconfirmed child back to its original lease owner."""

    def __init__(self, key, process):
        super().__init__(f"Media startup cleanup unconfirmed: {key}")
        self.process_key = key
        self.process = process

_BASE_POPEN = subprocess.Popen


class _OwnedFFmpegPopen(_BASE_POPEN):
    """Popen I/O and waits with creation-time Windows Job ownership.

    _execute_child is CPython's private Windows hook: its signature is a tested
    compatibility boundary. This media-only adapter requires shell=False and
    close_fds=True, as used by all managed FFmpeg entries.
    """
    def __init__(self, *args, owned_job, **kwargs):
        self.owned_job = owned_job
        super().__init__(*args, **kwargs)

    def _execute_child(self, args, executable, preexec_fn, close_fds,
                       pass_fds, cwd, env, startupinfo, creationflags, shell,
                       p2cread, p2cwrite, c2pread, c2pwrite, errread, errwrite,
                       unused_restore_signals, unused_gid, unused_gids,
                       unused_uid, unused_umask, unused_start_new_session,
                       unused_process_group):
        import _winapi
        if os.name != 'nt' or shell or not close_fds or pass_fds or preexec_fn:
            raise ValueError('Owned FFmpeg requires Windows, shell=False, close_fds=True')
        if not isinstance(args,str):
            args = subprocess.list2cmdline([args] if isinstance(args,(bytes,os.PathLike)) else args)
        startupinfo = startupinfo.copy() if startupinfo is not None else subprocess.STARTUPINFO()
        attributes = startupinfo.lpAttributeList or {}
        if set(attributes)-{'handle_list'} or attributes.get('handle_list'):
            raise ValueError('Custom startup attribute list is not supported')
        handles = (p2cread,c2pwrite,errwrite) if -1 not in (p2cread,c2pwrite,errwrite) else None
        executable = os.fsdecode(executable) if executable is not None else None
        cwd = os.fsdecode(cwd) if cwd is not None else None
        sys.audit('subprocess.Popen',executable,args,cwd,env)
        try:
            hp,ht,pid,_ = self.owned_job.create_process(args,executable=executable,
                cwd=cwd,env=env,creationflags=creationflags,startupinfo=startupinfo,std_handles=handles)
        finally:
            self._close_pipe_fds(p2cread,p2cwrite,c2pread,c2pwrite,errread,errwrite)
        self._child_created = True
        self._handle = subprocess.Handle(hp)
        self.pid = pid
        _winapi.CloseHandle(ht)


class FFmpegProcessRegistry:
    """Central lifecycle registry for managed FFmpeg subprocesses."""
    _processes = {}
    _jobs = {}
    _deferred = {}
    _starting = {}
    _lock = threading.Lock()
    _interrupted = False

    @staticmethod
    def has_exited(proc):
        """A cached exit code alone does not confirm Windows kernel termination."""
        if proc.poll() is None:
            return False
        handle = getattr(proc, '_handle', None)
        if os.name == 'nt' and isinstance(handle, int):
            import _winapi
            return _winapi.WaitForSingleObject(handle, 0) == _winapi.WAIT_OBJECT_0
        return True

    @staticmethod
    def wait_for_exit(proc, timeout=5):
        """Confirm the same native process within one shared exit-wait budget."""
        deadline = _time.monotonic() + timeout
        handle = getattr(proc, '_handle', None)
        if os.name == 'nt' and isinstance(handle, int):
            import _winapi
            result = _winapi.WaitForSingleObject(handle, round(timeout * 1000))
            if result != _winapi.WAIT_OBJECT_0:
                raise subprocess.TimeoutExpired(proc.args, timeout)
        return proc.wait(timeout=max(0, deadline - _time.monotonic()))

    @classmethod
    def start(cls, key, command, **kwargs):
        """Create then register; real Windows children already belong to a Job."""
        claim = {'proc': None}
        with cls._lock:
            if key in cls._processes or key in cls._starting:
                raise RuntimeError(f"FFmpeg registry key already owned: {key}")
            cls._starting[key] = claim
        job,proc = None,None
        try:
            # Preserve injected Popen test doubles and the non-Windows path.
            if os.name == 'nt' and subprocess.Popen is _BASE_POPEN:
                job = OwnedProcessJob()
                proc = _OwnedFFmpegPopen(command,owned_job=job,**kwargs)
            else:
                proc = subprocess.Popen(command,**kwargs)
            with cls._lock:
                claim['proc'] = proc
            cls.register(key,proc)
            return proc
        except BaseException:
            try:
                try:
                    if job is not None:
                        job.close()
                finally:
                    if proc is not None:
                        if not cls.has_exited(proc):
                            proc.kill()
                        cls.wait_for_exit(proc, timeout=5)
            except BaseException as cleanup_error:
                if proc is not None:
                    with cls._lock:
                        cls._processes[key] = proc
                        if job is not None:
                            cls._jobs[key] = job
                    raise MediaStartTerminationUnconfirmed(key, proc) from cleanup_error
                raise
            with cls._lock:
                if proc is not None and cls._processes.get(key) is proc:
                    cls._processes.pop(key)
                    cls._jobs.pop(key, None)
            raise
        finally:
            with cls._lock:
                if cls._starting.get(key) is claim:
                    cls._starting.pop(key)

    @classmethod
    def is_owned(cls, key):
        with cls._lock:
            return key in cls._starting or key in cls._processes or key in cls._deferred

    @classmethod
    def register(cls, key, proc):
        with cls._lock:
            claim = cls._starting.get(key)
            if claim is not None and claim['proc'] is not proc:
                raise RuntimeError(f"FFmpeg registry key is being started: {key}")
            if key in cls._processes:
                if cls._processes[key] is proc:
                    return
                proc.kill()
                proc.wait(timeout=5)
                raise RuntimeError(f"FFmpeg registry key already owned: {key}")
            if cls._interrupted:
                proc.kill()
            job = proc.owned_job if isinstance(proc,_OwnedFFmpegPopen) and proc.owned_job.handle else None
            handle = getattr(proc, '_handle', None)
            # Real Windows Popen handles are integer Handle objects. Test doubles
            # and non-Windows processes retain the existing explicit kill contract.
            if job is None and os.name == 'nt' and isinstance(handle, int) and not isinstance(handle, bool):
                import _winapi
                if _winapi.WaitForSingleObject(handle, 0) != _winapi.WAIT_OBJECT_0:
                    try:
                        job = OwnedProcessJob()
                        job.assign(proc)
                    except OSError as ownership_error:
                        if job is not None:
                            try:
                                job.close()
                            except OSError as error:
                                logger.error("FFmpeg ownership failure cleanup failed for %s: %s", key, error)
                        if _winapi.WaitForSingleObject(handle, 0) != _winapi.WAIT_OBJECT_0:
                            proc.kill()
                            proc.wait(timeout=5)
                            raise ownership_error
                        job = None  # A short command can finish before assignment.
            cls._processes[key] = proc
            if job is not None:
                cls._jobs[key] = job

    @classmethod
    def deregister(cls, key, proc=None):
        with cls._lock:
            if key in cls._starting:
                raise RuntimeError(f"FFmpeg startup still pending: {key}")
            current = cls._processes.get(key)
            if proc is not None and current is not None and current is not proc:
                raise RuntimeError(f"FFmpeg process identity mismatch: {key}")
            if key in cls._deferred:
                raise RuntimeError(f"FFmpeg cleanup still deferred: {key}")
            job = cls._jobs.get(key)
            if job is not None:
                job.detach()  # Preserve the contract: unregister does not kill.
                cls._jobs.pop(key)
            cls._processes.pop(key, None)

    @classmethod
    def defer_cleanup(cls, key, proc, io_sem, readers=()):
        """Retain the exact permit and readers for later nonblocking recovery."""
        with cls._lock:
            if key in cls._deferred:
                previous = cls._deferred[key]
                if previous['proc'] is not proc or previous['io_sem'] is not io_sem:
                    raise RuntimeError(f"Deferred cleanup identity conflict: {key}")
                return
            current = cls._processes.get(key)
            if current is not None and current is not proc:
                raise RuntimeError(f"Deferred process identity conflict: {key}")
            threads = list(readers)
            for name in ('stdout_thread', 'stderr_thread'):
                thread = getattr(proc, name, None)
                if isinstance(thread, threading.Thread) and thread not in threads:
                    threads.append(thread)
            cls._processes[key] = proc
            cls._deferred[key] = dict(proc=proc, io_sem=io_sem, readers=threads)

    @classmethod
    def reap_cleanups(cls):
        """Release deferred permits once exit, Job closure and readers agree.

        No wait, stop retry or cancellation reset is performed here. An open Job
        remains pending until its owner explicitly stops it successfully.
        """
        with cls._lock:
            return cls._reap_cleanups_locked()

    @classmethod
    def _reap_cleanups_locked(cls):
        recovered = []
        for key, item in list(cls._deferred.items()):
            proc, job = item['proc'], cls._jobs.get(key)
            if not cls.has_exited(proc) or (job is not None and job.handle):
                continue
            if any(reader.is_alive() for reader in item['readers']):
                continue
            try:
                for stream in (proc.stdout, proc.stderr):
                    if stream is not None:
                        stream.close()
                item['io_sem'].release()
            except Exception as error:
                logger.error("Deferred FFmpeg cleanup failed for %s: %s", key, error)
                continue
            cls._deferred.pop(key)
            cls._processes.pop(key, None)
            cls._jobs.pop(key, None)
            recovered.append(key)
            logger.info("Recovered deferred FFmpeg I/O permit for %s", key)
        return dict(recovered=recovered, pending=list(cls._deferred))

    @classmethod
    def _require_cleanups_locked(cls):
        report = cls._reap_cleanups_locked()
        if report['pending'] or cls._starting:
            cls._interrupted = True
            raise MediaTerminationUnconfirmed(
                f"Media cleanup/startup still pending: deferred={report['pending']}, "
                f"starting={list(cls._starting)}"
            )
        return report

    @classmethod
    def require_cleanups(cls):
        """Refuse new pipeline mutations until prior deferred owners are confirmed."""
        with cls._lock:
            return cls._require_cleanups_locked()

    @classmethod
    def mark_interrupted(cls):
        with cls._lock:
            cls._interrupted = True

    @classmethod
    def is_interrupted(cls) -> bool:
        with cls._lock:
            return cls._interrupted

    @classmethod
    def reset_interrupted(cls):
        with cls._lock:
            cls._require_cleanups_locked()
            cls._interrupted = False

    @classmethod
    def terminate(cls, key):
        """Stop this owned tree without cancelling unrelated registered work.

        Keep references until the caller confirms exit and deregisters. Closing
        a Windows Job also stops descendants holding inherited output pipes.
        """
        with cls._lock:
            proc, job = cls._processes.get(key), cls._jobs.get(key)
            if job is not None:
                job.close()
            if proc is not None and proc.poll() is None:
                try:
                    proc.kill()
                except OSError:
                    if proc.poll() is None:
                        raise

    @classmethod
    def kill_all(cls):
        with cls._lock:
            cls._interrupted = True
            failed = set()
            for k, p in cls._processes.items():
                try:
                    p.kill()
                except Exception as error:
                    logger.warning("FFmpeg termination failed for %s: %s", k, error)
                    failed.add(k)
            for key, job in list(cls._jobs.items()):
                try:
                    job.close()
                    cls._jobs.pop(key)
                    failed.discard(key)  # Closing the owned Job terminates its members.
                except OSError as error:
                    logger.error("FFmpeg owned Job close failed for %s: %s", key, error)
                    failed.add(key)
            for key in list(cls._processes):
                if key not in failed and key not in cls._deferred:
                    cls._processes.pop(key)


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
    io_timeout: float | None = None,
    io_lease=None,
) -> FFmpegResult:
    """Run managed FFmpeg; optional io_timeout bounds one admission attempt.

    timeout limits subprocess execution. Existing callers retain default I/O retries.
    io_lease supplies an unacquired scheduler lease whose admission, release or
    deferred ownership belongs to this call. Filesystem operations outside it
    are not bounded by these limits.
    """
    FFmpegProcessRegistry.reap_cleanups()
    if io_timeout is not None:
        from math import isfinite
        if not isfinite(io_timeout) or io_timeout <= 0:
            raise ValueError("io_timeout must be finite and positive")
    if stdout_consumer is not None and not capture_output:
        raise ValueError("stdout_consumer requires capture_output=True")
    cmd = ["ffmpeg", "-hide_banner", "-y", "-nostdin"] + args
    kwargs = {"stdin": subprocess.DEVNULL}
    if capture_output:
        kwargs["stdout"] = subprocess.PIPE
        kwargs["stderr"] = subprocess.PIPE
        
    from src.hardware.scheduler import acquire_with_retry, get_disk_semaphore
    io_sem = get_disk_semaphore() if io_lease is None else io_lease
    # AGENTS.md 铁律：acquire 必须带 timeout 并重试，禁止无限阻塞
    admission = {} if io_timeout is None else {"timeout": io_timeout, "retries": 1}
    if not acquire_with_retry(io_sem, **admission):
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
    proc = None
    key = f"ffmpeg:{id(args)}:{t0}"
    retain_reader = False
    cleanup_readers = []
    try:
        try:
            proc = FFmpegProcessRegistry.start(key,cmd,**kwargs)
        except MediaStartTerminationUnconfirmed as error:
            proc, key = error.process, error.process_key
            raise
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
                        try:
                            block = proc.stderr.read(4096)
                        except (OSError, ValueError):
                            break  # The stream may close after a consumer failure.
                        if not block:
                            break
                        errors.append(block)
                def expire():
                    expired.set()
                    FFmpegProcessRegistry.terminate(key)
                reader = threading.Thread(target=drain_errors, daemon=True)
                cleanup_readers.append(reader)
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
                        FFmpegProcessRegistry.terminate(key)
                        proc.wait(timeout=5)
                    reader.join(timeout=5)
                    if reader.is_alive():
                        retain_reader = True
                        raise RuntimeError("FFmpeg stderr reader termination unconfirmed")
                    proc.stdout.close()
                    proc.stderr.close()
                stdout, stderr = b"", b"".join(errors)
            returncode = proc.returncode
            timed_out = False
        except subprocess.TimeoutExpired:
            FFmpegProcessRegistry.terminate(key)
            if stdout_consumer is None:
                try:
                    stdout, stderr = proc.communicate(timeout=5)
                except subprocess.TimeoutExpired:
                    retain_reader = True
                    logger.error("FFmpeg pipe cleanup unconfirmed; retained reader and I/O permit for %s", key)
                    raise
            else:
                stdout, stderr = b"", b"stream timed out"
            returncode = -9
            timed_out = True
    except OSError as e:
        # ffmpeg 可执行文件缺失或 spawn 失败：返回结构化错误而非 UnboundLocalError
        phase = "spawn" if proc is None else "execution/cleanup"
        return FFmpegResult(
            returncode=-1,
            stdout=b"",
            stderr=f"ffmpeg {phase} failed: {e}".encode("utf-8", errors="replace"),
            timed_out=False,
            duration=_time.monotonic() - t0,
        )
    finally:
        try:
            if proc is not None:
                FFmpegProcessRegistry.terminate(key)
                FFmpegProcessRegistry.wait_for_exit(proc, timeout=5)
                # An earlier stop/wait error can bypass the streaming branch's
                # join. Native process exit does not prove its pipe reader ended.
                for reader in cleanup_readers:
                    reader.join(timeout=5)
                    if reader.is_alive():
                        raise RuntimeError("FFmpeg pipe reader termination unconfirmed")
        except BaseException as error:
            retain_reader = True
            logger.error("FFmpeg termination unconfirmed; retained reader and I/O permit for %s", key)
            FFmpegProcessRegistry.defer_cleanup(key, proc, io_sem, cleanup_readers)
            raise MediaTerminationUnconfirmed(f"FFmpeg cleanup unconfirmed: {key}") from error
        if retain_reader:
            FFmpegProcessRegistry.defer_cleanup(key, proc, io_sem, cleanup_readers)
            raise MediaTerminationUnconfirmed(f"FFmpeg pipe cleanup unconfirmed: {key}")
        else:
            if proc is not None:
                FFmpegProcessRegistry.deregister(key, proc)
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
    """Probe metadata under the shared process lifecycle and disk admission.

    TimeoutExpired and spawn errors retain their existing exception contract.
    Cleanup waits separately; unconfirmed termination retains ownership and the
    I/O permit rather than admitting another reader over a potentially live one.
    """
    import json
    FFmpegProcessRegistry.reap_cleanups()
    if timeout is None:
        timeout = load_config().get("detection", {}).get("prescreen_extract_timeout", 30.0)
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        str(filepath),
    ]
    
    from src.hardware.scheduler import acquire_with_retry, get_disk_semaphore
    io_sem = get_disk_semaphore()
    # AGENTS.md 铁律：acquire 必须带 timeout 并重试，禁止无限阻塞
    if not acquire_with_retry(io_sem):
        logger.warning("run_ffprobe: io semaphore acquire timeout for %s", filepath)
        return None
    proc = None
    key = f"ffprobe:{id(cmd)}:{_time.monotonic()}"
    completed = False
    try:
        try:
            proc = FFmpegProcessRegistry.start(
                key, cmd, stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
        except MediaStartTerminationUnconfirmed as error:
            proc, key = error.process, error.process_key
            raise
        stdout, _ = proc.communicate(timeout=timeout)
        completed = True
    finally:
        if proc is not None:
            cleanup_confirmed = False
            try:
                FFmpegProcessRegistry.terminate(key)
                FFmpegProcessRegistry.wait_for_exit(proc, timeout=5)
                if not completed:
                    proc.communicate(timeout=5)
                cleanup_confirmed = True
            except BaseException as error:
                logger.error("run_ffprobe: termination unconfirmed; retained registry and I/O permit for %s", key)
                FFmpegProcessRegistry.defer_cleanup(key, proc, io_sem)
                raise MediaTerminationUnconfirmed(f"ffprobe cleanup unconfirmed: {key}") from error
            finally:
                # Do not close pipes while an unconfirmed reader may still use them.
                if cleanup_confirmed:
                    for stream in (proc.stdout, proc.stderr):
                        if stream is not None:
                            stream.close()
        if proc is not None:
            FFmpegProcessRegistry.deregister(key, proc)
        io_sem.release()

    if proc.returncode != 0 or not stdout.strip():
        return None
    try:
        return json.loads(stdout.decode("utf-8"))
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
