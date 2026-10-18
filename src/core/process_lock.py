"""Kernel ownership for shared local work; leftover files are not live leases."""
from contextlib import contextmanager, ExitStack
import hashlib
import json
import os
from pathlib import Path


@contextmanager
def exclusive_file_lock(path, *, conflict):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    stream=path.open('a+b')
    locked=False
    try:
        if stream.seek(0,2)==0:
            stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
            locked=True
        except OSError as error:
            raise RuntimeError(conflict) from error
        yield
    finally:
        try:
            if locked:
                stream.seek(0)
                if os.name=='nt':
                    import msvcrt
                    msvcrt.locking(stream.fileno(),msvcrt.LK_UNLCK,1)
                else:
                    import fcntl
                    fcntl.flock(stream,fcntl.LOCK_UN)
        finally:
            stream.close()


def _lock_path(directory, identity):
    digest=hashlib.sha256(json.dumps(identity).encode()).hexdigest()
    return Path(directory)/'.task_locks'/(digest+'.lock')


def database_day_lease(database_path,date,cam_index):
    database=Path(database_path).resolve()
    path=_lock_path(database.parent,('database',os.path.normcase(str(database)),str(date),int(cam_index)))
    return exclusive_file_lock(path,conflict=f'Day task already owned: {date} cam{cam_index}; resource={path}')


def workspace_lease(temp_dir):
    path=_lock_path(Path(temp_dir).resolve(),('pipeline-workspace',))
    return exclusive_file_lock(path,conflict=f'Pipeline workspace already owned; resource={path}')


@contextmanager
def day_task_lease(database_path, temp_dir, output_path, date, cam_index):
    """Protect DB day, shared batch namespace and concrete delivery destination."""
    database=Path(database_path).resolve()
    temporary=Path(temp_dir).resolve()
    output=Path(output_path).resolve()
    resources=[(database.parent,('database',os.path.normcase(str(database)),str(date),int(cam_index))),
               (temporary,('pipeline-workspace',)),
               (output.parent,('delivery',os.path.normcase(str(output))))]
    paths=[]
    for directory,identity in resources:
        paths.append(_lock_path(directory,identity))
    with ExitStack() as stack:
        for path in sorted(set(paths),key=str):
            stack.enter_context(exclusive_file_lock(path,conflict=f'Day task already owned: {date} cam{cam_index}; resource={path}'))
        yield


class OwnedProcessJob:
    """Windows process ownership; default close or owning-parent death kills members."""
    def __init__(self):
        self.handle = None
        if os.name != 'nt':
            return
        import ctypes
        from ctypes import wintypes
        size = ctypes.c_size_t
        class Basic(ctypes.Structure):
            _fields_ = [('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),
                        ('flags',wintypes.DWORD),('minimum',size),('maximum',size),
                        ('active',wintypes.DWORD),('affinity',size),('priority',wintypes.DWORD),('scheduling',wintypes.DWORD)]
        class Counters(ctypes.Structure):
            _fields_ = [(name,ctypes.c_ulonglong) for name in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic',Basic),('io',Counters),('process_memory',size),('job_memory',size),('peak_process',size),('peak_job',size)]
        self._extended_type = Extended
        self.api = ctypes.WinDLL('kernel32',use_last_error=True)
        self.api.CreateJobObjectW.argtypes=[ctypes.c_void_p,wintypes.LPCWSTR]
        self.api.CreateJobObjectW.restype=wintypes.HANDLE
        self.api.SetInformationJobObject.argtypes=[wintypes.HANDLE,ctypes.c_int,ctypes.c_void_p,wintypes.DWORD]
        self.api.SetInformationJobObject.restype=wintypes.BOOL
        self.api.AssignProcessToJobObject.argtypes=[wintypes.HANDLE,wintypes.HANDLE]
        self.api.AssignProcessToJobObject.restype=wintypes.BOOL
        self.api.CloseHandle.argtypes=[wintypes.HANDLE];self.api.CloseHandle.restype=wintypes.BOOL
        self.handle=self.api.CreateJobObjectW(None,None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        info=Extended();info.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)):
            error=ctypes.WinError(ctypes.get_last_error());self.close();raise error

    def assign(self, process):
        if self.handle and not self.api.AssignProcessToJobObject(self.handle,int(process._handle)):
            import ctypes
            raise ctypes.WinError(ctypes.get_last_error())

    def create_process(self, command, *, executable=None, cwd=None, env=None,
                       creationflags=0, startupinfo=None, std_handles=None):
        """Windows atomic Job membership; return native process/thread handles.

        The caller owns the returned handles. Only explicit standard handles are
        inherited; the Job handle itself must remain private to its owner.
        """
        if os.name != 'nt' or not self.handle:
            raise RuntimeError('Creation-time Job ownership requires a live Windows Job')
        import ctypes
        from ctypes import wintypes

        class Startup(ctypes.Structure):
            _fields_ = [('cb',wintypes.DWORD),('reserved',wintypes.LPWSTR),
                        ('desktop',wintypes.LPWSTR),('title',wintypes.LPWSTR)] + [
                (name,wintypes.DWORD) for name in ('x','y','width','height','cx','cy','fill','flags')] + [
                ('show',wintypes.WORD),('reserved_size',wintypes.WORD),('reserved_bytes',ctypes.c_void_p),
                ('stdin',wintypes.HANDLE),('stdout',wintypes.HANDLE),('stderr',wintypes.HANDLE)]

        class ExtendedStartup(ctypes.Structure):
            _fields_ = [('startup',Startup),('attributes',ctypes.c_void_p)]

        class ProcessInfo(ctypes.Structure):
            _fields_ = [('process',wintypes.HANDLE),('thread',wintypes.HANDLE),
                        ('pid',wintypes.DWORD),('tid',wintypes.DWORD)]

        api = ctypes.WinDLL('kernel32',use_last_error=True)
        init = api.InitializeProcThreadAttributeList
        init.argtypes = [ctypes.c_void_p,wintypes.DWORD,wintypes.DWORD,ctypes.POINTER(ctypes.c_size_t)]
        init.restype = wintypes.BOOL
        update = api.UpdateProcThreadAttribute
        update.argtypes = [ctypes.c_void_p,wintypes.DWORD,ctypes.c_size_t,ctypes.c_void_p,
                           ctypes.c_size_t,ctypes.c_void_p,ctypes.c_void_p]
        update.restype = wintypes.BOOL
        delete = api.DeleteProcThreadAttributeList
        delete.argtypes = [ctypes.c_void_p]
        delete.restype = None
        create = api.CreateProcessW
        create.argtypes = [wintypes.LPCWSTR,wintypes.LPWSTR,ctypes.c_void_p,ctypes.c_void_p,
                           wintypes.BOOL,wintypes.DWORD,ctypes.c_void_p,wintypes.LPCWSTR,
                           ctypes.POINTER(ExtendedStartup),ctypes.POINTER(ProcessInfo)]
        create.restype = wintypes.BOOL
        size = ctypes.c_size_t()
        count = 2 if std_handles else 1
        init(None,count,0,ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not init(buffer,count,0,ctypes.byref(size)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            jobs = (wintypes.HANDLE * 1)(self.handle)
            if not update(buffer,0,0x2000d,jobs,ctypes.sizeof(jobs),None,None):
                raise ctypes.WinError(ctypes.get_last_error())
            info = ExtendedStartup()
            info.startup.cb = ctypes.sizeof(info)
            info.attributes = ctypes.addressof(buffer)
            if startupinfo is not None:
                info.startup.flags = startupinfo.dwFlags
                info.startup.show = startupinfo.wShowWindow
            if std_handles:
                unique = tuple(dict.fromkeys(map(int,std_handles)))
                handles = (wintypes.HANDLE * len(unique))(*unique)
                if not update(buffer,0,0x20002,handles,ctypes.sizeof(handles),None,None):
                    raise ctypes.WinError(ctypes.get_last_error())
                info.startup.flags |= 0x100  # STARTF_USESTDHANDLES
                info.startup.stdin,info.startup.stdout,info.startup.stderr = std_handles
            environment = None
            if env is not None:
                if any(not isinstance(k,str) or not isinstance(v,str) or '\0' in k+v or '=' in k for k,v in env.items()):
                    raise ValueError('Invalid environment mapping')
                environment = ctypes.create_unicode_buffer('\0'.join(f'{k}={v}' for k,v in sorted(env.items()))+'\0\0')
                creationflags |= 0x400  # CREATE_UNICODE_ENVIRONMENT
            result = ProcessInfo()
            command_buffer = ctypes.create_unicode_buffer(command)
            if not create(executable,command_buffer,None,None,bool(std_handles),creationflags | 0x80000,
                          environment,cwd,ctypes.byref(info),ctypes.byref(result)):
                raise ctypes.WinError(ctypes.get_last_error())
            return result.process,result.thread,result.pid,result.tid
        finally:
            delete(buffer)

    def close(self):
        if self.handle:
            if not self.api.CloseHandle(self.handle):
                import ctypes
                raise ctypes.WinError(ctypes.get_last_error())
            self.handle=None

    def detach(self):
        """Relinquish ownership without terminating a live member."""
        if self.handle:
            import ctypes
            info = self._extended_type()
            if not self.api.SetInformationJobObject(self.handle,9,ctypes.byref(info),ctypes.sizeof(info)):
                raise ctypes.WinError(ctypes.get_last_error())
            self.close()
