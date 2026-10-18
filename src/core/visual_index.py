"""Rebuildable visual descriptors; event labels are never inferred from similarity."""
import bisect
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
from src.core.process_lock import OwnedProcessJob as _SourceJob


def source_identity(filepath):
    path = Path(filepath)
    stat = path.stat()
    if not path.is_file() or stat.st_size <= 0:
        raise OSError("Visual source is not a nonempty file")
    return [os.path.normcase(str(path.resolve())), stat.st_size, stat.st_mtime_ns]


def frame_signature(gray):
    import cv2
    import numpy as np
    if gray.ndim != 2 or gray.size == 0:
        raise ValueError("Visual index requires a nonempty grayscale frame")
    if gray.dtype != np.uint8:
        gray = np.clip(gray,0,255).astype(np.uint8)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    gradient = cv2.resize(small, (17, 16), interpolation=cv2.INTER_AREA)
    bits = np.packbits(gradient[:, 1:] > gradient[:, :-1]).tobytes().hex()
    hist = np.bincount((small.astype(np.uint16)//16).ravel(), minlength=16)
    histogram = np.rint(hist/hist.sum()*255).astype(np.uint8).tobytes().hex()
    return {"version":1,"dhash":bits,"histogram":histogram}


def signature_distance(first, second):
    if isinstance(first, str):
        first = json.loads(first)
    if isinstance(second, str):
        second = json.loads(second)
    if first["version"] != 1 or second["version"] != 1:
        raise ValueError("Unsupported visual signature")
    a, b = bytes.fromhex(first["dhash"]), bytes.fromhex(second["dhash"])
    ha, hb = bytes.fromhex(first["histogram"]), bytes.fromhex(second["histogram"])
    if len(a) != 32 or len(b) != 32 or len(ha) != 16 or len(hb) != 16:
        raise ValueError("Malformed visual signature")
    return .75 * (int.from_bytes(a)^int.from_bytes(b)).bit_count()/256 + .25 * sum(abs(x-y) for x,y in zip(ha,hb))/(2*255)


def store_samples(db, filepath, identity, samples, *, origin, resolved_source_path=None, cancelled=None):
    """Publish only samples inside each current source interval, transactionally."""
    # Historical readers resolve the path inside their bounded worker. Normal
    # analysis callers retain the existing direct source-binding validation.
    expected_path = resolved_source_path if resolved_source_path is not None else os.path.normcase(str(Path(filepath).resolve()))
    if not identity or identity[0] != expected_path:
        raise ValueError("Visual source identity mismatch")
    ordered = sorted(samples,key=lambda sample:sample[0])
    if any(not math.isfinite(float(t)) or t < 0 for t,_ in ordered):
        raise ValueError("Invalid visual sample timestamp")
    times = [item[0] for item in ordered]
    count = 0
    with db._lock:
        connection = db.conn
        owns = not connection.in_transaction
        if owns:
            connection.execute("BEGIN IMMEDIATE")
        savepoint = "visual_"+uuid.uuid4().hex
        connection.execute("SAVEPOINT "+savepoint)
        try:
            if cancelled and cancelled():
                raise InterruptedError('Visual publication cancelled')
            rows = connection.execute("SELECT * FROM segments WHERE filepath=? ORDER BY start_time",(filepath,)).fetchall()
            connection.execute("DELETE FROM visual_features WHERE filepath=?",(filepath,))
            for row in rows:
                start = row["start_time"]-row["file_start_offset"]
                end = row["end_time"]-row["file_start_offset"]
                low, high = bisect.bisect_left(times,start), bisect.bisect_left(times,end)
                if low >= high:
                    continue
                index = min(range(low,high), key=lambda i:abs(times[i]-(start+end)/2))
                timestamp, signature = ordered[index]
                signature_distance(signature,signature)  # Validate before any publication.
                connection.execute("""INSERT INTO visual_features
                    (segment_id,filepath,start_time,end_time,file_start_offset,source_identity,frame_time,signature,origin)
                    VALUES (?,?,?,?,?,?,?,?,?)""",(row["id"],filepath,row["start_time"],row["end_time"],row["file_start_offset"],json.dumps(identity),timestamp,json.dumps(signature),origin))
                count += 1
            if cancelled and cancelled():
                raise InterruptedError('Visual publication cancelled')
            connection.execute("RELEASE "+savepoint)
            if owns:
                connection.commit()
        except BaseException:
            connection.execute("ROLLBACK TO "+savepoint)
            connection.execute("RELEASE "+savepoint)
            if owns:
                connection.rollback()
            raise
    return count



def layout_signature(rows):
    return hashlib.sha256(json.dumps([[row[key] for key in ("id","filepath","start_time","end_time","file_start_offset")] for row in rows],sort_keys=True).encode()).hexdigest()


def _wait_reader_exit(process, timeout=5):
    """An exit code can precede completion of termination with pending native I/O."""
    if os.name=='nt':
        import _winapi
        result=_winapi.WaitForSingleObject(process._handle,round(timeout*1000))
        if result!=_winapi.WAIT_OBJECT_0:
            raise VisualReaderStopError(f'Owned visual reader kernel exit unconfirmed; worker={process.pid}; wait={result}')
    return process.wait(timeout=timeout)


def _stop_reader(process, job, connection):
    import subprocess
    try:
        try:
            if job:job.close()
            if process.poll() is None:process.kill()
            _wait_reader_exit(process)
        except (OSError,subprocess.TimeoutExpired) as error:
            raise VisualReaderStopError(f'Owned visual reader did not stop; worker={process.pid}') from error
    finally:
        if connection:connection.close()
        else:
            for stream in (process.stdin,process.stdout,process.stderr):stream.close()


def _collect_samples(filepath, targets):
    import av
    samples=[]
    with av.open(str(filepath),timeout=(10,10)) as container:
        stream=container.streams.video[0];stream.thread_count=1
        origin=float((stream.start_time or 0)*stream.time_base)
        for midpoint,end in targets:
            if midpoint < 0:
                continue
            container.seek(round((midpoint+origin)/stream.time_base),stream=stream,backward=True)
            start=2*midpoint-end
            candidate=None
            for frame in container.decode(stream):
                if frame.pts is None:
                    continue
                timestamp=float(frame.pts*frame.time_base)-origin
                if timestamp >= end:
                    break
                if timestamp < start or frame.is_corrupt:
                    continue
                if candidate is None or abs(timestamp-midpoint)<abs(candidate[0]-midpoint):
                    candidate=(timestamp,frame)
                if timestamp >= midpoint:
                    break
            if candidate is not None:
                timestamp,frame=candidate
                samples.append((timestamp,frame_signature(frame.to_ndarray(format='gray'))))
    return samples


def _source_reader_worker(connection):
    """No DB access. Wait for parent admission/job ownership before any source I/O."""
    try:
        filepath=connection.recv()
        identity=source_identity(filepath)
        connection.send(('identity',identity))
        action,targets=connection.recv()
        if action=='close':
            return
        if action!='sample':
            raise ValueError('Invalid visual reader action')
        samples=_collect_samples(filepath,targets)
        if source_identity(filepath)!=identity:
            raise ValueError('Visual source changed during backfill')
        connection.send(('samples',samples))
    except EOFError:
        pass  # Parent vanished before opening admission; no source was published.
    except BaseException as error:
        connection.send(('error',{'type':type(error).__name__,'message':str(error)}))
    finally:
        connection.close()


class _ReaderConnection:
    """Drain owned pipes in memory so native source stalls cannot block the parent."""
    def __init__(self, process):
        import queue
        import threading
        from collections import deque
        self.process=process;self.queue=queue.Queue();self.errors=deque(maxlen=32)
        self.closing=False;self.native_handles=[None,None];self.api=None
        if os.name=='nt':
            import ctypes
            from ctypes import wintypes
            self.api=ctypes.WinDLL('kernel32',use_last_error=True)
            self.api.OpenThread.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
            self.api.OpenThread.restype=wintypes.HANDLE
            self.api.GetCurrentThreadId.argtypes=[];self.api.GetCurrentThreadId.restype=wintypes.DWORD
            self.api.CancelSynchronousIo.argtypes=[wintypes.HANDLE];self.api.CancelSynchronousIo.restype=wintypes.BOOL
            self.api.CloseHandle.argtypes=[wintypes.HANDLE];self.api.CloseHandle.restype=wintypes.BOOL
        def read_output():
            try:
                while not self.closing:
                    line=process.stdout.readline()
                    if not line:break
                    self.queue.put(json.loads(line))
            except Exception as error:
                self.queue.put(('error',{'type':type(error).__name__,'message':str(error)}))
            finally:
                self.queue.put(('eof',None))
        def read_errors():
            try:
                while not self.closing:
                    chunk=process.stderr.read(4096)
                    if not chunk:break
                    self.errors.append(chunk)
            except OSError as error:
                if not self.closing:
                    self.queue.put(('error',{'type':type(error).__name__,'message':str(error)}))
        def run_reader(index,target):
            if self.api:
                # Capture the current thread object before any read. Holding the
                # handle prevents TID reuse from targeting an unrelated thread.
                handle=self.api.OpenThread(0x0001,False,self.api.GetCurrentThreadId())
                if not handle:
                    import ctypes
                    error=ctypes.WinError(ctypes.get_last_error())
                    self.queue.put(('error',{'type':type(error).__name__,'message':str(error)}))
                    return
                self.native_handles[index]=handle
            target()
        self.threads=[threading.Thread(target=run_reader,args=(index,target),daemon=True)
                      for index,target in enumerate((read_output,read_errors))]
        for thread in self.threads:thread.start()

    def send(self, value):
        self.process.stdin.write(json.dumps(value,ensure_ascii=True).encode('utf-8')+b'\n')
        self.process.stdin.flush()

    def close(self):
        import logging
        import time
        self.closing=True
        # Buffered close takes the reader's lock and can wait forever even after
        # child exit. Cancel our own synchronous pipe reads before closing them.
        deadline=time.monotonic()+2
        while any(thread.is_alive() for thread in self.threads) and time.monotonic()<deadline:
            for index,thread in enumerate(self.threads):
                if thread.is_alive() and self.api and self.native_handles[index]:
                    if not self.api.CancelSynchronousIo(self.native_handles[index]):
                        import ctypes
                        error=ctypes.get_last_error()
                        if error!=1168:  # ERROR_NOT_FOUND: read completed or has not started.
                            logging.getLogger('homevlog').warning('Visual pipe cancellation failed: %s',error)
                thread.join(timeout=.05)
        alive=any(thread.is_alive() for thread in self.threads)
        for index,handle in enumerate(self.native_handles):
            if handle:
                self.api.CloseHandle(handle);self.native_handles[index]=None
        if alive:
            # Never acquire a buffered stream lock still held by a reader.
            raise VisualReaderStopError(f'Visual pipe readers did not stop; worker={self.process.pid}')
        for stream in (self.process.stdin,self.process.stdout,self.process.stderr):
            try:stream.close()
            except OSError as error:logging.getLogger('homevlog').warning('Visual reader pipe close failed: %s',error)


def _receive_reader(connection, process, deadline, phase):
    import time
    import queue
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    while True:
        if FFmpegProcessRegistry.is_interrupted():
            raise InterruptedError('Visual source reader cancelled')
        remaining=deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'Visual source hard deadline exceeded during {phase}; worker={process.pid}')
        try:
            kind,value=connection.queue.get(timeout=min(.1,remaining))
            if kind=='error':
                raise RuntimeError(f"Visual reader {value['type']}: {value['message']}")
            if kind=='eof':
                raise RuntimeError(f'Visual reader exited without a complete {phase} response; worker={process.pid}; '+b''.join(connection.errors).decode('utf-8',errors='replace')[-4096:])
            return kind,value
        except queue.Empty:
            continue


def _start_reader():
    import subprocess
    import sys
    executable=sys.executable
    environment=os.environ.copy()
    # CPython's Windows multiprocessing uses this venv handoff to bypass the
    # redirect launcher while preserving the configured virtual environment.
    # A clean module avoids re-importing an app main that might open a DB.
    if os.name=='nt' and sys.prefix!=sys.base_prefix:
        executable=sys._base_executable;environment['__PYVENV_LAUNCHER__']=sys.executable
    return subprocess.Popen([executable,'-m','src.core.visual_index','--source-reader',str(os.getpid())],
            cwd=str(Path(__file__).resolve().parents[2]),env=environment,
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)


class VisualReaderStopError(RuntimeError):
    """Cannot continue admitting sources after owned reader termination failed."""


def reader_process_state(pid, *, handle=None):
    """Read only: identify a process instance and verify kernel termination."""
    if os.name!='nt':
        import psutil
        try:
            process=psutil.Process(pid)
            return {'pid':pid,'created':repr(process.create_time()),'platform':os.name,
                    'kernel_ended':process.status()==psutil.STATUS_ZOMBIE}
        except psutil.NoSuchProcess:
            return {'pid':pid,'missing':True}
    import ctypes
    import _winapi
    from ctypes import wintypes
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
    api.GetProcessTimes.argtypes=[wintypes.HANDLE]+[ctypes.POINTER(wintypes.FILETIME)]*4
    api.GetProcessTimes.restype=wintypes.BOOL
    api.CloseHandle.argtypes=[wintypes.HANDLE];api.CloseHandle.restype=wintypes.BOOL
    owns=handle is None
    if owns:
        handle=api.OpenProcess(0x100000|0x1000,False,pid)
        if not handle:
            error=ctypes.get_last_error()
            if error==87:return {'pid':pid,'missing':True}
            raise ctypes.WinError(error)
    try:
        times=[wintypes.FILETIME() for _ in range(4)]
        if not api.GetProcessTimes(int(handle),*(ctypes.byref(value) for value in times)):
            raise ctypes.WinError(ctypes.get_last_error())
        result=_winapi.WaitForSingleObject(int(handle),0)
        if result not in (_winapi.WAIT_OBJECT_0,_winapi.WAIT_TIMEOUT):
            raise RuntimeError(f'Cannot inspect reader kernel state: {result}')
        return {'pid':pid,'created':str(times[0].dwLowDateTime|(times[0].dwHighDateTime<<32)),
                'platform':os.name,'kernel_ended':result==_winapi.WAIT_OBJECT_0}
    finally:
        if owns and not api.CloseHandle(handle):raise ctypes.WinError(ctypes.get_last_error())


def confirm_reader_stopped(identity):
    """A missing or reused PID means this recorded instance ended; never kill it."""
    if identity.get('platform')!=os.name or not isinstance(identity.get('created'),str) or not identity['created']:
        raise VisualReaderStopError('Recorded reader identity cannot be verified on this platform')
    if isinstance(identity.get('pid'),bool) or not isinstance(identity.get('pid'),int) or identity['pid']<=0:
        raise VisualReaderStopError('Recorded reader PID is invalid')
    try:
        token=int(identity['created']) if os.name=='nt' else float(identity['created'])
        if token<=0 or not math.isfinite(token):raise ValueError('Invalid creation token')
    except (ValueError,TypeError,OverflowError) as error:
        raise VisualReaderStopError('Recorded reader creation identity is invalid') from error
    try:state=reader_process_state(identity['pid'])
    except (OSError,RuntimeError) as error:
        raise VisualReaderStopError(f'Cannot confirm previous visual reader termination; worker={identity["pid"]}') from error
    if (not state.get('missing') and state['created']==identity['created']
            and not state['kernel_ended']):
        raise VisualReaderStopError(f'Previous visual reader kernel exit unconfirmed; worker={identity["pid"]}')
    return state


_identity_held_io = {}


def inspect_source_identity_owned(filepath, directory, *, timeout_s=3):
    """Query identity in an owned worker; directory is local recovery state.

    The wait budget covers ownership, I/O admission and the worker response.
    Cleanup can take longer or fail explicitly. Unconfirmed readers prevent a
    subsequent query, and retain the current process's I/O permit until confirmed.
    """
    import socket
    import time
    from src.core.process_lock import exclusive_file_lock
    from src.core.replay_journal import atomic_json
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    from src.hardware.scheduler import acquire_with_retry, get_disk_semaphore
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('Metadata budget must be finite and positive')
    directory = Path(directory).resolve()
    directory_key = os.path.normcase(str(directory))
    deadline = time.monotonic() + timeout_s
    conflict = 'Metadata ownership is busy'
    claim = None
    while claim is None:
        if FFmpegProcessRegistry.is_interrupted():
            raise InterruptedError('Metadata query cancelled before admission')
        if time.monotonic() >= deadline:
            raise TimeoutError('Metadata ownership admission timed out')
        attempt = exclusive_file_lock(directory / 'owner.lock', conflict=conflict)
        try:
            attempt.__enter__()
            claim = attempt
        except RuntimeError as error:
            if str(error) != conflict:
                raise
            time.sleep(.01)
    disk, acquired = None, False
    process = job = connection = record = None
    try:
        disk = get_disk_semaphore()
        pointer = directory / 'current.json'
        host = {'hostname': socket.gethostname(), 'platform': os.name}
        held = _identity_held_io.get(directory_key)
        if held is not None:
            held_disk, held_identity = held
            confirm_reader_stopped(held_identity)
            held_disk.release()
            del _identity_held_io[directory_key]
        if pointer.exists():
            previous = json.loads(pointer.read_text(encoding='utf-8'))
            if previous['host'] != host:
                raise VisualReaderStopError('Metadata checkpoint belongs to another host')
            confirm_reader_stopped(previous['reader_identity'])
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not acquire_with_retry(disk, timeout=remaining, retries=1):
            if FFmpegProcessRegistry.is_interrupted():
                raise InterruptedError('Metadata query cancelled during I/O admission')
            raise TimeoutError('Metadata I/O admission timed out')
        acquired = True
        job = _SourceJob()
        process = _start_reader()
        job.assign(process)
        identity = reader_process_state(process.pid, handle=getattr(process, '_handle', None))
        folder = directory / uuid.uuid4().hex
        folder.mkdir()
        record = {'host': host, 'reader_identity': identity, 'status': 'RUNNING', 'attempt': str(folder)}
        atomic_json(folder / 'reader_identity.json', record)
        atomic_json(pointer, record)  # No source I/O until the reader is recorded.
        connection = _ReaderConnection(process)
        connection.send(str(filepath))
        kind, identity = _receive_reader(connection, process, deadline, 'metadata-only identity')
        if (kind != 'identity' or not isinstance(identity, list) or len(identity) != 3
                or not isinstance(identity[0], str) or not Path(identity[0]).is_absolute()
                or type(identity[1]) is not int or identity[1] <= 0 or type(identity[2]) is not int):
            raise ValueError('Invalid metadata identity response')
        return tuple(identity)
    finally:
        try:
            if process is not None:
                try:
                    _stop_reader(process, job, connection)
                    if record:
                        stopped = confirm_reader_stopped(record['reader_identity'])
                        atomic_json(Path(record['attempt']) / 'cleanup.json', {'status': 'STOPPED', 'observed': stopped})
                except BaseException:
                    if acquired and record:
                        try:
                            confirm_reader_stopped(record['reader_identity'])
                        except Exception:
                            _identity_held_io[directory_key] = (disk, record['reader_identity'])
                            acquired = False
                    raise
            elif job:
                job.close()
        finally:
            try:
                if acquired:
                    disk.release()
            finally:
                claim.__exit__(None, None, None)


def index_source(db, filepath, *, timeout_s=60, reuse_current=True, previous=None, retry_incomplete=False,on_reader_start=None):
    """Sparse historical backfill: one CPU reader, scheduler I/O admission, no image transport."""
    import time
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('Visual source budget must be finite and positive')
    from src.hardware.scheduler import get_disk_semaphore, acquire_with_retry
    rows = db.get_segments_for_file(filepath)
    disk = get_disk_semaphore()
    if not acquire_with_retry(disk, timeout=10, retries=2):
        raise TimeoutError("Visual index I/O admission timed out")
    started = time.monotonic()
    process=connection=job=None
    identity=None
    layout=layout_signature(rows)
    try:
        job=_SourceJob()
        process=_start_reader()
        job.assign(process)
        if on_reader_start:
            on_reader_start(reader_process_state(process.pid,handle=getattr(process,'_handle',None)))
        connection=_ReaderConnection(process)
        connection.send(str(filepath))
        kind,identity=_receive_reader(connection,process,started+timeout_s,'source metadata')
        if kind!='identity':
            raise ValueError('Invalid visual source metadata response')
        matches=previous and previous.get('source_identity')==identity and previous.get('segment_layout_sha256')==layout
        if matches and previous['status']!='COMPLETED' and not retry_incomplete and reuse_current:
            return {**previous,'deferred':True}
        if reuse_current and rows:
            with db._lock:
                existing = db.conn.execute("""SELECT v.* FROM visual_features v JOIN segments s ON s.id=v.segment_id
                    AND s.filepath=v.filepath AND s.start_time=v.start_time AND s.end_time=v.end_time
                    AND s.file_start_offset=v.file_start_offset WHERE s.filepath=?""",(filepath,)).fetchall()
            try:
                reusable = len(existing) == len(rows) and all(json.loads(item["source_identity"]) == identity
                    and item["start_time"]-item["file_start_offset"] <= item["frame_time"] < item["end_time"]-item["file_start_offset"]
                    and signature_distance(item["signature"],item["signature"]) == 0 for item in existing)
            except (ValueError,TypeError,KeyError):
                reusable = False
            if reusable:
                return {"source_identity":identity,"segment_layout_sha256":layout_signature(rows),"segments":len(rows),
                        "indexed":len(rows),"observed_frames":0,"cache_hit":True,"wall_s":round(time.monotonic()-started,6)}
        targets = sorted({((row["start_time"]+row["end_time"])/2-row["file_start_offset"],
                           row["end_time"]-row["file_start_offset"]) for row in rows if row["end_time"] > row["start_time"]})
        connection.send(('sample',targets))
        kind,samples=_receive_reader(connection,process,started+timeout_s,'source sampling')
        if kind!='samples':
            raise ValueError('Invalid visual source sample response')
        from src.hardware.ffmpeg import FFmpegProcessRegistry
        if FFmpegProcessRegistry.is_interrupted():
            raise InterruptedError('Visual source reader cancelled before publication')
        with db._lock:
            if layout_signature(db.get_segments_for_file(filepath)) != layout_signature(rows):
                raise ValueError("Source segment layout changed during backfill")
            count = store_samples(db,filepath,identity,samples,origin="sparse_source_backfill",resolved_source_path=identity[0],
                                  cancelled=FFmpegProcessRegistry.is_interrupted)
        return {"source_identity":identity,"segment_layout_sha256":layout_signature(rows),"segments":len(rows),"indexed":count,"observed_frames":len(samples),
                "cache_hit":False,"wall_s":round(time.monotonic()-started,6)}
    except Exception as error:
        error.source_identity=identity;error.segment_layout_sha256=layout
        raise
    finally:
        try:
            if process is not None:
                _stop_reader(process,job,connection)
            elif job:
                job.close()
        finally:
            disk.release()


def _reader_main(parent_pid):
    import sys
    if sys.platform=='linux':
        import ctypes
        import signal
        api=ctypes.CDLL(None,use_errno=True)
        if api.prctl(1,signal.SIGKILL,0,0,0)!=0:
            raise OSError(ctypes.get_errno(),'Cannot protect visual reader parent lifetime')
        if os.getppid()!=parent_pid:return
    elif os.name!='nt':
        raise RuntimeError('Bounded visual readers require Windows jobs or Linux parent-death signals')
    class Connection:
        def recv(self):
            line=sys.stdin.buffer.readline()
            if not line:raise EOFError()
            return json.loads(line)
        def send(self,value):
            sys.stdout.buffer.write(json.dumps(value,ensure_ascii=True).encode('utf-8')+b'\n');sys.stdout.buffer.flush()
        def close(self):pass
    _source_reader_worker(Connection())


if __name__=='__main__':
    import sys
    if len(sys.argv)!=3 or sys.argv[1]!='--source-reader':
        raise SystemExit('Internal visual source reader entry point')
    _reader_main(int(sys.argv[2]))
