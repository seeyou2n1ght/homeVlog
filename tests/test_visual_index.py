"""Visual similarity and its independent, rebuildable persistence contract."""
import json
import numpy as np
import pytest
from src.database import VlogDatabase
from src.segment import Segment
from src.core.visual_index import frame_signature,signature_distance,source_identity,store_samples
from scripts.audit_tool.service import AuditService


def test_perceptual_distance_orders_duplicates_brightness_and_different_structure():
    gradient = np.tile(np.arange(64,dtype=np.uint8)*3,(64,1))
    signature = frame_signature(gradient)
    duplicate = frame_signature(gradient.copy())
    shifted = frame_signature(gradient+5)
    reversed_structure = frame_signature(np.fliplr(gradient))
    assert signature_distance(signature,duplicate) == 0
    assert 0 < signature_distance(signature,shifted) < signature_distance(signature,reversed_structure)
    assert signature_distance(frame_signature(np.zeros((64,64),np.uint8)),frame_signature(np.full((64,64),255,np.uint8))) > 0
    with pytest.raises(ValueError):
        frame_signature(np.zeros((32,32,3),np.uint8))


@pytest.fixture
def visual_inventory(tmp_path):
    db = VlogDatabase(db_path=tmp_path/"visual.sqlite")
    sources=[]
    for camera in (0,1):
        path=tmp_path/f"cam{camera}"/"same.mp4";path.parent.mkdir();path.write_bytes(b"source"+bytes([camera]))
        sources.append(str(path))
        db.add_file_task(str(path),camera,"20261012","20261012000000","20261012000006",6)
        db.set_analysis_result(str(path),"ANALYZED",[Segment(start_time=t,end_time=t+2,state="STATIC",source_file=str(path)) for t in (0,2,4)])
    yield db,sources
    db.close()


def test_visual_index_preserves_labels_boundaries_and_caller_transaction(visual_inventory):
    db,sources=visual_inventory
    signature=frame_signature(np.zeros((32,32),np.uint8))
    source=sources[0];identity=source_identity(source)
    labels=[tuple(row) for row in db.conn.execute("SELECT * FROM segments ORDER BY id")]
    assert store_samples(db,source,identity,[(1,signature),(3,signature),(5,signature)],origin="controlled") == 3
    original=[tuple(row) for row in db.conn.execute("SELECT * FROM visual_features ORDER BY segment_id")]
    db.conn.execute("BEGIN")
    db.conn.execute("UPDATE segments SET review_notes='caller pending' WHERE filepath=?",(source,))
    with pytest.raises(ValueError):
        store_samples(db,source,identity,[(1,signature),(3,{"version":2})],origin="controlled")
    assert db.conn.in_transaction
    assert original==[tuple(row) for row in db.conn.execute("SELECT * FROM visual_features ORDER BY segment_id")]
    assert store_samples(db,source,identity,[(1,signature)],origin="controlled") == 1
    assert db.conn.in_transaction
    db.conn.rollback()
    assert labels==[tuple(row) for row in db.conn.execute("SELECT * FROM segments ORDER BY id")]
    assert original==[tuple(row) for row in db.conn.execute("SELECT * FROM visual_features ORDER BY segment_id")]


def test_visual_queue_uses_image_distance_and_validates_source_identity(visual_inventory):
    db,sources=visual_inventory
    gradient=np.tile(np.arange(64,dtype=np.uint8)*3,(64,1))
    signature=frame_signature(gradient)
    store_samples(db,sources[0],source_identity(sources[0]),[(1,signature),(3,frame_signature(np.fliplr(gradient))),(5,signature)],origin="controlled")
    store_samples(db,sources[1],source_identity(sources[1]),[(1,signature),(3,signature)],origin="controlled")
    reference=db.get_segments_for_file(sources[0])[0]["id"]
    audit=AuditService(db=db)
    result=audit.get_review_queue("visual",reference_id=reference)
    assert result["counts"]["all"]==6 and result["indexed_items"]==5
    assert result["items"][0]["visual_distance"]==0
    assert result["items"][-1]["visual_distance"]>0.7
    assert all("cluster_ids" not in item and item["manual_label"] is None for item in result["items"])
    assert {item["filepath"] for item in result["items"]}==set(sources)
    assert len(audit.get_review_queue("visual",reference_id=reference,cam_index=0)["items"])==3
    assert [item["id"] for item in audit.get_review_queue("visual",reference_id=reference,limit=2,offset=2)["items"]] == [item["id"] for item in result["items"][2:4]]
    # Same basenames cannot make a changed source's features usable.
    from pathlib import Path
    Path(sources[1]).write_bytes(b"changed camera1 source")
    assert len(audit.get_review_queue("visual",reference_id=reference)["items"])==3
    Path(sources[0]).write_bytes(b"changed reference source")
    stale=audit.get_review_queue("visual",reference_id=reference)
    assert stale["status"]=="UNAVAILABLE" and stale["items"]==[]
    assert db.conn.execute("SELECT COUNT(*) FROM human_reviews").fetchone()[0]==0


def test_visual_samples_never_extend_a_short_interval(visual_inventory):
    db,sources=visual_inventory
    signature=frame_signature(np.zeros((32,32),np.uint8))
    assert store_samples(db,sources[0],source_identity(sources[0]),[(2,signature)],origin="controlled")==1
    item=db.conn.execute("SELECT frame_time,start_time,end_time FROM visual_features").fetchone()
    assert item["start_time"]==2 and item["end_time"]==4
    assert item["start_time"]<=item["frame_time"]<item["end_time"]


def test_reanalysis_discards_stale_derived_features(visual_inventory):
    db,sources=visual_inventory
    signature=frame_signature(np.zeros((32,32),np.uint8))
    store_samples(db,sources[0],source_identity(sources[0]),[(1,signature)],origin="controlled")
    db.set_analysis_result(sources[0],"ANALYZED",[Segment(start_time=0,end_time=6,state="DYNAMIC",source_file=sources[0])])
    assert db.conn.execute("SELECT COUNT(*) FROM visual_features").fetchone()[0]==0



def test_current_visual_index_reuse_never_opens_a_second_reader(visual_inventory,monkeypatch):
    import av
    from src.core.visual_index import index_source
    db,sources=visual_inventory
    signature=frame_signature(np.zeros((32,32),np.uint8))
    store_samples(db,sources[0],source_identity(sources[0]),[(1,signature),(3,signature),(5,signature)],origin="single_pass_analysis")
    monkeypatch.setattr(av,"open",lambda *args,**kwargs:pytest.fail("valid current features must not be decoded again"))
    result=index_source(db,sources[0])
    assert result["cache_hit"] is True and result["indexed"]==3 and result["observed_frames"]==0


def test_inventory_resume_rechecks_real_features_and_records_partial(visual_inventory,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts.index_review_visuals import run_inventory
    from src.core.replay_journal import atomic_json
    from src.core.visual_index import layout_signature
    db,sources=visual_inventory
    database_path=tmp_path/'visual.sqlite'
    identity=source_identity(sources[0]);layout=layout_signature(db.get_segments_for_file(sources[0]))
    report={'runs':[{'db':str(database_path.resolve()),'files':[{'filepath':sources[0],'status':'COMPLETED',
                     'source_identity':identity,'segment_layout_sha256':layout,'indexed':3,'segments':3}]}]}
    args=SimpleNamespace(db=database_path,report=tmp_path/'report.json',date=None,cam_index=0,
                         force=False,retry_incomplete=False,limit_files=1,source_file=None,source_budget_s=60)
    calls=[]
    def index(current,filepath,**kwargs):
        calls.append(filepath)
        assert kwargs['timeout_s']==args.source_budget_s
        if kwargs.get('previous',{}).get('status')=='PARTIAL' and not kwargs['retry_incomplete']:
            return {'deferred':True}
        return {'source_identity':identity,'segment_layout_sha256':layout,'indexed':2,'segments':3,'cache_hit':False}
    monkeypatch.setattr('src.core.visual_index.index_source',index)
    # Ledger claims complete but the database has no derived rows. It must
    # reach the current-index validator rather than silently skipping.
    run_inventory(args,report,VlogDatabase,atomic_json)
    assert calls==[sources[0]] and report['runs'][-1]['files'][0]['status']=='PARTIAL'
    assert report['runs'][-1]['status']=='INCOMPLETE'
    calls.clear()
    run_inventory(args,report,VlogDatabase,atomic_json)
    assert calls==[sources[0]]  # Metadata validation is bounded; decoding is deferred.
    calls.clear()
    args.retry_incomplete=True
    args.source_file=sources[0]
    args.source_budget_s=300
    run_inventory(args,report,VlogDatabase,atomic_json)
    assert calls==[sources[0]]


def test_inventory_new_sources_precede_prior_failures(visual_inventory,tmp_path,monkeypatch):
    from types import SimpleNamespace
    from scripts.index_review_visuals import run_inventory
    from src.core.replay_journal import atomic_json
    db,sources=visual_inventory
    report={'runs':[{'db':str((tmp_path/'visual.sqlite').resolve()),'files':[{'filepath':sources[0],'status':'FAILED','error':'prior unavailable'}]}]}
    args=SimpleNamespace(db=tmp_path/'visual.sqlite',report=tmp_path/'report.json',date=None,cam_index=None,
                         force=False,retry_incomplete=False,limit_files=1,source_file=None,source_budget_s=60)
    calls=[]
    def index(current,filepath,**kwargs):
        calls.append(filepath)
        return {'source_identity':source_identity(filepath),'indexed':3,'segments':3,'cache_hit':True}
    monkeypatch.setattr('src.core.visual_index.index_source',index)
    run_inventory(args,report,VlogDatabase,atomic_json)
    assert calls==[sources[1]] and report['runs'][-1]['recorded_failed_sources']==1
    assert report['runs'][-1]['status']=='INCOMPLETE'


@pytest.mark.parametrize('budget',['nan','inf','0','1801'])
def test_inventory_cli_rejects_invalid_budget_before_opening_db(tmp_path,monkeypatch,budget):
    from scripts.index_review_visuals import main
    database=tmp_path/'existing.sqlite';database.write_bytes(b'untouched sentinel')
    monkeypatch.setattr('sys.argv',['index_review_visuals.py','--db',str(database),'--report',str(tmp_path/'report.json'),
                                  '--source-budget-s',budget])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code==2 and database.read_bytes()==b'untouched sentinel'
    assert not (tmp_path/'report.json').exists()


@pytest.mark.parametrize('phase',['metadata','sampling'])
def test_native_source_stall_is_terminated_without_publication(visual_inventory,blocked_source_reader,monkeypatch,phase):
    import time
    from unittest.mock import MagicMock
    from src.core import visual_index
    db,sources=visual_inventory
    signature=frame_signature(np.zeros((32,32),np.uint8))
    store_samples(db,sources[0],source_identity(sources[0]),[(1,signature)],origin='preserved')
    before=[tuple(row) for row in db.conn.execute('SELECT * FROM visual_features')]
    segments=[tuple(row) for row in db.conn.execute('SELECT * FROM segments')]
    disk=MagicMock()
    monkeypatch.setattr('src.hardware.scheduler.get_disk_semaphore',lambda:disk)
    monkeypatch.setattr('src.hardware.scheduler.acquire_with_retry',lambda *args,**kwargs:True)
    monkeypatch.setattr(visual_index,'_start_reader',lambda:blocked_source_reader(phase))
    started=time.monotonic()
    with pytest.raises(TimeoutError,match='hard deadline'):
        visual_index.index_source(db,sources[0],timeout_s=2,reuse_current=False)
    assert time.monotonic()-started < 7
    assert blocked_source_reader.ready.exists()  # Native read was reached, not merely startup timed out.
    assert blocked_source_reader.owned[0].poll() is not None
    disk.release.assert_called_once_with()
    assert before==[tuple(row) for row in db.conn.execute('SELECT * FROM visual_features')]
    assert segments==[tuple(row) for row in db.conn.execute('SELECT * FROM segments')]


def test_active_cancel_terminates_native_reader(visual_inventory,blocked_source_reader,monkeypatch,tmp_path):
    import threading
    import time
    from src.core import visual_index
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    db,sources=visual_inventory
    ready=tmp_path/'ready'
    monkeypatch.setattr(visual_index,'_start_reader',lambda:blocked_source_reader('sampling',ready))
    def cancel():
        deadline=time.monotonic()+5
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
        FFmpegProcessRegistry.mark_interrupted()
    thread=threading.Thread(target=cancel);thread.start()
    try:
        with pytest.raises(InterruptedError):
            visual_index.index_source(db,sources[0],timeout_s=10)
        assert ready.exists() and blocked_source_reader.owned[0].poll() is not None
        assert db.conn.execute('SELECT COUNT(*) FROM visual_features').fetchone()[0]==0
    finally:
        thread.join(timeout=6);FFmpegProcessRegistry.reset_interrupted()


def test_cancel_during_transaction_rolls_back_old_index(visual_inventory):
    db,sources=visual_inventory
    signature=frame_signature(np.zeros((32,32),np.uint8));identity=source_identity(sources[0])
    store_samples(db,sources[0],identity,[(1,signature)],origin='preserved')
    before=[tuple(row) for row in db.conn.execute('SELECT * FROM visual_features')]
    checks=iter([False,True])
    with pytest.raises(InterruptedError):
        store_samples(db,sources[0],identity,[(3,signature)],origin='cancelled',cancelled=lambda:next(checks))
    assert not db.conn.in_transaction
    assert before==[tuple(row) for row in db.conn.execute('SELECT * FROM visual_features')]


@pytest.mark.parametrize('stop_type',['cancel','reader_stop'])
def test_inventory_terminal_stop_preserves_active_checkpoint(visual_inventory,tmp_path,monkeypatch,stop_type):
    from types import SimpleNamespace
    from scripts.index_review_visuals import run_inventory
    from src.core.replay_journal import atomic_json
    from src.core.visual_index import VisualReaderStopError
    db,sources=visual_inventory
    args=SimpleNamespace(db=tmp_path/'visual.sqlite',report=tmp_path/'report.json',date=None,cam_index=0,
                         force=False,retry_incomplete=False,limit_files=2,source_file=None,source_budget_s=60)
    calls=[]
    error=InterruptedError if stop_type=='cancel' else VisualReaderStopError
    def cancel(db,filepath,**kwargs):
        calls.append(filepath)
        raise error('controlled stop')
    monkeypatch.setattr('src.core.visual_index.index_source',cancel)
    report={'runs':[]}
    with pytest.raises(error):run_inventory(args,report,VlogDatabase,atomic_json)
    checkpoint=json.loads(args.report.read_text(encoding='utf-8'))['runs'][-1]
    assert checkpoint['status']==('INTERRUPTED' if stop_type=='cancel' else 'READER_STOP_FAILED') and checkpoint['active_source']==sources[0]
    assert checkpoint['files']==[]
    assert calls==[sources[0]]


def test_windows_parent_kill_stops_only_its_reader(visual_inventory,blocked_source_reader,tmp_path):
    import os
    import subprocess
    import sys
    import time
    import psutil
    from pathlib import Path
    if os.name!='nt':pytest.skip('Windows job lifetime verification')
    db,sources=visual_inventory
    ready=tmp_path/'parent_reader.ready'
    parent=tmp_path/'owner.py'
    parent.write_text('''import os, subprocess, sys
from src.database import VlogDatabase
from src.core import visual_index
def start():
    return subprocess.Popen([sys.executable,sys.argv[1],'sampling',sys.argv[2]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
visual_index._start_reader=start
db=VlogDatabase(db_path=sys.argv[3])
visual_index.index_source(db,sys.argv[4],timeout_s=60)
''',encoding='utf-8')
    environment=os.environ.copy();environment['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    executable=sys.executable
    if sys.prefix!=sys.base_prefix:
        executable=sys._base_executable;environment['__PYVENV_LAUNCHER__']=sys.executable
    # An independent owned sentinel proves the job cannot kill unrelated readers.
    sentinel=blocked_source_reader('metadata',tmp_path/'sentinel.ready')
    process=subprocess.Popen([executable,str(parent),str(blocked_source_reader.worker),str(ready),str(tmp_path/'visual.sqlite'),sources[0]],env=environment,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    child=None
    try:
        deadline=time.monotonic()+10
        while not ready.exists() and process.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists(),process.communicate(timeout=1) if process.poll() is not None else 'reader did not enter native read'
        child=psutil.Process(int(ready.read_text()))
        process.kill();process.wait(timeout=5)
        child.wait(timeout=5)
        assert sentinel.poll() is None
        assert db.conn.execute('SELECT COUNT(*) FROM visual_features').fetchone()[0]==0
    finally:
        if process.poll() is None:process.kill()
        process.wait(timeout=5)
        for stream in (process.stdout,process.stderr):stream.close()
        if child and child.is_running():child.kill()


def test_actual_pts_before_midpoint_captures_short_interval(short_local_video,tmp_path):
    from src.core.visual_index import index_source
    db=VlogDatabase(db_path=tmp_path/'short.sqlite')
    path=str(short_local_video)
    try:
        db.add_file_task(path,0,'20261012','20261012000000','20261012000006',6)
        db.set_analysis_result(path,'ANALYZED',[Segment(start_time=.95,end_time=1.,state='PRESENCE',source_file=path)])
        result=index_source(db,path,timeout_s=10)
        assert result['indexed']==1 and result['observed_frames']==1
        row=db.conn.execute('SELECT frame_time FROM visual_features').fetchone()
        assert .95<=row['frame_time']<1. and row['frame_time']<.975
        assert index_source(db,path,timeout_s=10)['cache_hit'] is True
    finally:db.close()


@pytest.mark.parametrize('cancel_available',[True,False])
def test_pipe_cleanup_has_deadline_when_writer_survives_child_exit(cancel_available):
    import io
    import os
    import time
    from types import SimpleNamespace
    from src.core.visual_index import _ReaderConnection,VisualReaderStopError
    if os.name!='nt' and cancel_available:pytest.skip('Windows synchronous pipe cancellation')
    out_read,out_write=os.pipe();err_read,err_write=os.pipe()
    process=SimpleNamespace(pid=98765,stdin=io.BytesIO(),stdout=os.fdopen(out_read,'rb'),stderr=os.fdopen(err_read,'rb'))
    connection=_ReaderConnection(process)
    time.sleep(.1)  # Both native reads enter their buffered stream locks.
    if not cancel_available:
        # Simulate an API/driver that cannot cancel; close must fail boundedly.
        if connection.api:
            import ctypes
            def cannot_cancel(handle):
                ctypes.set_last_error(1168)
                return 0
            connection.api.CancelSynchronousIo=cannot_cancel
    started=time.monotonic()
    try:
        if cancel_available:
            connection.close()
            assert all(not thread.is_alive() for thread in connection.threads)
        else:
            with pytest.raises(VisualReaderStopError,match='pipe readers did not stop'):
                connection.close()
        assert time.monotonic()-started<3
    finally:
        os.close(out_write);os.close(err_write)
        for thread in connection.threads:thread.join(timeout=3)
        for stream in (process.stdin,process.stdout,process.stderr):stream.close()


def test_reader_exit_requires_kernel_signal_even_with_cached_zero(monkeypatch):
    import os
    from types import SimpleNamespace
    from src.core.visual_index import _wait_reader_exit,VisualReaderStopError
    if os.name!='nt':pytest.skip('Windows native process termination')
    import _winapi
    waits=[]
    process=SimpleNamespace(pid=12345,_handle=456,returncode=0,wait=lambda **kwargs:waits.append(kwargs) or 0)
    monkeypatch.setattr(_winapi,'WaitForSingleObject',lambda handle,budget:_winapi.WAIT_TIMEOUT)
    with pytest.raises(VisualReaderStopError,match='kernel exit unconfirmed'):
        _wait_reader_exit(process,timeout=.1)
    assert waits==[]
    monkeypatch.setattr(_winapi,'WaitForSingleObject',lambda handle,budget:_winapi.WAIT_OBJECT_0)
    assert _wait_reader_exit(process,timeout=.1)==0 and len(waits)==1


@pytest.mark.parametrize('previous_status',['RUNNING','INTERRUPTED','READER_STOP_FAILED'])
def test_inventory_resume_refuses_previous_live_instance_before_database(tmp_path,external_day_owner,previous_status):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from scripts.index_review_visuals import run_inventory
    from src.core.replay_journal import atomic_json
    from src.core.visual_index import reader_process_state,VisualReaderStopError
    owner=external_day_owner(tmp_path/'lease.sqlite',tmp_path/'lease_temp',tmp_path/'lease.mp4')
    identity=reader_process_state(owner.pid,handle=getattr(owner,'_handle',None))
    assert not identity['kernel_ended']
    report={'runs':[{'status':previous_status,'active_reader':identity,'files':[]}]}
    args=SimpleNamespace(db=tmp_path/'visual.sqlite')
    factory=Mock(side_effect=AssertionError('Database must not open'))
    with pytest.raises(VisualReaderStopError,match='Previous visual reader kernel exit unconfirmed'):
        run_inventory(args,report,factory,atomic_json)
    factory.assert_not_called()
    assert len(report['runs'])==1 and owner.poll() is None


def test_inventory_legacy_stop_without_identity_refuses_before_database(tmp_path):
    from types import SimpleNamespace
    from unittest.mock import Mock
    from scripts.index_review_visuals import run_inventory
    from src.core.replay_journal import atomic_json
    from src.core.visual_index import VisualReaderStopError
    report={'runs':[{'status':'READER_STOP_FAILED','files':[]}]}
    factory=Mock(side_effect=AssertionError('Database must not open'))
    with pytest.raises(VisualReaderStopError,match='Legacy reader-stop failure lacks process identity'):
        run_inventory(SimpleNamespace(db=tmp_path/'visual.sqlite'),report,factory,atomic_json)
    factory.assert_not_called()
    assert len(report['runs'])==1


def test_recorded_reader_identity_handles_exit_and_pid_reuse_without_killing(tmp_path,external_day_owner):
    from src.core.visual_index import reader_process_state,confirm_reader_stopped
    owner=external_day_owner(tmp_path/'lease.sqlite',tmp_path/'lease_temp',tmp_path/'lease.mp4')
    identity=reader_process_state(owner.pid,handle=getattr(owner,'_handle',None))
    # Same PID but different creation identity represents an ended old instance.
    import os
    different=str(int(identity['created'])+1) if os.name=='nt' else repr(float(identity['created'])+1)
    observed=confirm_reader_stopped(dict(identity,created=different))
    assert observed['created']==identity['created'] and owner.poll() is None
    owner.kill();owner.wait(timeout=5)
    observed=confirm_reader_stopped(identity)
    assert observed.get('missing') or observed['kernel_ended']


def test_reader_checkpoint_failure_stops_before_source_io(visual_inventory,blocked_source_reader,monkeypatch):
    from src.core import visual_index
    db,sources=visual_inventory
    monkeypatch.setattr(visual_index,'_start_reader',lambda:blocked_source_reader('metadata'))
    def cannot_publish(identity):
        assert identity['pid'] and identity['created'] and not identity['kernel_ended']
        raise OSError('reader checkpoint unavailable')
    with pytest.raises(OSError,match='reader checkpoint unavailable'):
        visual_index.index_source(db,sources[0],on_reader_start=cannot_publish)
    assert not blocked_source_reader.ready.exists()
    assert blocked_source_reader.owned[0].poll() is not None
    assert db.conn.execute('SELECT COUNT(*) FROM visual_features').fetchone()[0]==0


def test_inventory_real_reader_checkpoint_and_ended_instance_resume(short_local_video,tmp_path):
    import copy
    from types import SimpleNamespace
    from scripts.index_review_visuals import run_inventory
    from src.core.replay_journal import atomic_json
    from src.core.visual_index import reader_process_state,confirm_reader_stopped
    database=tmp_path/'inventory.sqlite';path=str(short_local_video)
    db=VlogDatabase(db_path=database)
    db.add_file_task(path,0,'20261013','20261013000000','20261013000006',6)
    db.set_analysis_result(path,'ANALYZED',[Segment(start_time=t,end_time=t+2,state='STATIC',source_file=path) for t in (0,2,4)])
    before=[tuple(row) for row in db.conn.execute('SELECT * FROM segments')];db.close()
    args=SimpleNamespace(db=database,report=tmp_path/'report.json',date=None,cam_index=None,source_file=None,
                         limit_files=1,force=False,retry_incomplete=False,source_budget_s=10)
    identities=[]
    def persist(target,report):
        active=report['runs'][-1].get('active_reader')
        if active:
            observed=reader_process_state(active['pid'])
            assert observed['created']==active['created'] and not observed['kernel_ended']
            assert report['runs'][-1]['files']==[]
            identities.append(dict(active))
        atomic_json(target,report)
    report={'runs':[]}
    run_inventory(args,report,VlogDatabase,persist)
    assert identities and report['runs'][-1]['files'][0]['indexed']==3
    assert 'active_reader' not in report['runs'][-1]
    state=confirm_reader_stopped(identities[-1]);assert state.get('missing') or state['kernel_ended']
    previous={'status':'READER_STOP_FAILED','active_reader':identities[-1],'files':[]}
    preserved=copy.deepcopy(previous)
    resumed={'runs':[previous]}
    run_inventory(args,resumed,VlogDatabase,atomic_json)
    assert resumed['runs'][0]==preserved and len(resumed['runs'][-1]['reader_resolutions'])==1
    assert resumed['runs'][-1]['files'][0]['cache_hit']
    with __import__('sqlite3').connect(database) as connection:
        assert before==connection.execute('SELECT * FROM segments').fetchall()
def test_owned_metadata_normal_and_parallel_identity(tmp_path):
    import threading
    from src.core.visual_index import inspect_source_identity_owned
    from src.hardware.scheduler import get_disk_semaphore
    source=tmp_path/'source.bin';source.write_bytes(b'identity fixture')
    expected=tuple(source_identity(source));owner=tmp_path/'owner'
    disk=get_disk_semaphore();capacity=disk.available
    assert inspect_source_identity_owned(source,owner)==expected
    results=[];errors=[]
    def query():
        try:results.append(inspect_source_identity_owned(source,owner))
        except Exception as error:errors.append(error)
    threads=[threading.Thread(target=query) for _ in range(3)]
    for thread in threads:thread.start()
    for thread in threads:thread.join(timeout=5)
    assert all(not t.is_alive() for t in threads) and not errors and results==[expected]*3
    assert disk.available==capacity
    assert len(list(owner.glob('*/cleanup.json')))==4


@pytest.mark.parametrize('kind',['missing','empty','directory'])
def test_owned_metadata_invalid_source_releases_io(tmp_path,kind):
    from src.core.visual_index import inspect_source_identity_owned
    from src.hardware.scheduler import get_disk_semaphore
    source=tmp_path/'source.bin'
    if kind=='empty':source.write_bytes(b'')
    if kind=='directory':source=tmp_path
    disk=get_disk_semaphore();capacity=disk.available
    with pytest.raises(RuntimeError,match='Visual reader'):
        inspect_source_identity_owned(source,tmp_path/'owner')
    assert disk.available==capacity


def test_owned_metadata_deadline_and_cancel_are_process_backed(tmp_path,blocked_source_reader,monkeypatch):
    import threading
    import time
    from src.core import visual_index
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    from src.hardware.scheduler import get_disk_semaphore
    disk=get_disk_semaphore();capacity=disk.available
    monkeypatch.setattr(visual_index,'_start_reader',blocked_source_reader)
    with pytest.raises(TimeoutError):
        visual_index.inspect_source_identity_owned('unused',tmp_path/'deadline',timeout_s=.25)
    assert blocked_source_reader.owned[-1].poll() is not None and disk.available==capacity
    blocked_source_reader.ready.unlink(missing_ok=True)
    def cancel():
        deadline=time.monotonic()+2
        while not blocked_source_reader.ready.exists() and time.monotonic()<deadline:time.sleep(.01)
        if blocked_source_reader.ready.exists():FFmpegProcessRegistry.mark_interrupted()
    watcher=threading.Thread(target=cancel);watcher.start()
    try:
        with pytest.raises(InterruptedError):
            visual_index.inspect_source_identity_owned('unused',tmp_path/'cancelled')
    finally:
        watcher.join(timeout=3);FFmpegProcessRegistry.reset_interrupted()
    assert blocked_source_reader.owned[-1].poll() is not None and disk.available==capacity


@pytest.mark.parametrize('stop_report',['raises','returns'])
def test_owned_metadata_unconfirmed_reader_quarantines_permit_until_kernel_exit(tmp_path,blocked_source_reader,monkeypatch,stop_report):
    import os
    from src.core import visual_index
    from src.hardware.scheduler import get_disk_semaphore
    owner=tmp_path/'owner';source=tmp_path/'source.bin';source.write_bytes(b'source')
    original_stop=visual_index._stop_reader;held=[]
    disk=get_disk_semaphore();capacity=disk.available
    def not_stopped(process,job,connection):
        held.append((process,job,connection))
        if stop_report=='raises':
            raise visual_index.VisualReaderStopError('Controlled stop remains unconfirmed')
    try:
        with monkeypatch.context() as context:
            context.setattr(visual_index,'_start_reader',blocked_source_reader)
            context.setattr(visual_index,'_stop_reader',not_stopped)
            with pytest.raises(visual_index.VisualReaderStopError):
                visual_index.inspect_source_identity_owned(source,owner,timeout_s=.25)
        assert held[0][0].poll() is None and disk.available==capacity-1
        count=len(blocked_source_reader.owned)
        with pytest.raises(visual_index.VisualReaderStopError):
            visual_index.inspect_source_identity_owned(source,owner)
        assert len(blocked_source_reader.owned)==count and disk.available==capacity-1
        original_stop(*held[0])
        assert disk.available==capacity-1  # Confirmation in the next query releases it exactly once.
        assert visual_index.inspect_source_identity_owned(source,owner)==tuple(source_identity(source))
        assert disk.available==capacity
    finally:
        if held and held[0][0].poll() is None:original_stop(*held[0])
        pending=visual_index._identity_held_io.pop(os.path.normcase(str(owner.resolve())),None)
        if pending:pending[0].release()


def test_owned_metadata_checkpoint_failure_precedes_source_io(tmp_path,blocked_source_reader,monkeypatch):
    from src.core import visual_index,replay_journal
    from src.hardware.scheduler import get_disk_semaphore
    disk=get_disk_semaphore();capacity=disk.available
    monkeypatch.setattr(visual_index,'_start_reader',blocked_source_reader)
    def refused(*args,**kwargs):raise OSError('checkpoint unavailable')
    monkeypatch.setattr(replay_journal,'atomic_json',refused)
    with pytest.raises(OSError,match='checkpoint unavailable'):
        visual_index.inspect_source_identity_owned('unused',tmp_path/'owner')
    assert not blocked_source_reader.ready.exists()
    assert blocked_source_reader.owned[-1].poll() is not None and disk.available==capacity


@pytest.mark.parametrize('kind',['corrupt','foreign_host'])
def test_owned_metadata_bad_checkpoint_refuses_before_spawn(tmp_path,monkeypatch,kind):
    from src.core import visual_index
    owner=tmp_path/'owner';owner.mkdir();pointer=owner/'current.json'
    payload=b'invalid json' if kind=='corrupt' else json.dumps({'host':{'hostname':'foreign','platform':'foreign'}}).encode()
    pointer.write_bytes(payload)
    monkeypatch.setattr(visual_index,'_start_reader',lambda:pytest.fail('Bad checkpoint spawned reader'))
    with pytest.raises(ValueError if kind=='corrupt' else visual_index.VisualReaderStopError):
        visual_index.inspect_source_identity_owned('unused',owner)
    assert pointer.read_bytes()==payload

@pytest.mark.skipif(__import__('os').name!='nt',reason='Windows owned Job parent-death contract')
def test_owned_metadata_parent_death_checkpoint_allows_verified_restart(tmp_path,blocked_source_reader):
    import os
    import subprocess
    import sys
    import time
    from pathlib import Path
    from src.core.visual_index import confirm_reader_stopped,reader_process_state
    source=tmp_path/'source.bin';source.write_bytes(b'unchanged metadata fixture')
    owner=tmp_path/'owner';ready=tmp_path/'child.ready'
    environment=os.environ.copy();environment['PYTHONPATH']=str(Path(__file__).resolve().parents[1])
    program=r'''
import subprocess,sys
from src.core import visual_index
visual_index._start_reader=lambda:subprocess.Popen([sys.executable,sys.argv[1],'metadata',sys.argv[4]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
visual_index.inspect_source_identity_owned(sys.argv[3],sys.argv[2],timeout_s=30)
'''
    independent=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    parent=subprocess.Popen([sys.executable,'-c',program,str(blocked_source_reader.worker),str(owner),str(source),str(ready)],env=environment,
                            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        deadline=time.monotonic()+5
        while not ready.exists() and parent.poll() is None and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists()
        checkpoint=json.loads((owner/'current.json').read_text(encoding='utf-8'))
        identity=checkpoint['reader_identity']
        assert not reader_process_state(identity['pid'])['kernel_ended']
        original_record=Path(checkpoint['attempt'])/'reader_identity.json'
        original_bytes=original_record.read_bytes()
        parent.kill();parent.wait(timeout=5)
        deadline=time.monotonic()+5
        while True:
            try:
                observed=confirm_reader_stopped(identity)
                break
            except RuntimeError:
                if time.monotonic()>=deadline:raise
                time.sleep(.02)
        assert observed.get('missing') or observed['kernel_ended']
        restart='import json,sys;from src.core.visual_index import inspect_source_identity_owned;print(json.dumps(inspect_source_identity_owned(sys.argv[1],sys.argv[2])))'
        completed=subprocess.run([sys.executable,'-c',restart,str(source),str(owner)],env=environment,capture_output=True,timeout=8,check=True)
        assert json.loads(completed.stdout)==source_identity(source)
        assert independent.poll() is None and original_record.read_bytes()==original_bytes
    finally:
        if parent.poll() is None:parent.kill()
        parent.wait(timeout=5)
        parent.stdout.close();parent.stderr.close()
        independent.kill();independent.wait(timeout=5)
def test_owned_metadata_scheduler_failure_releases_directory_claim(tmp_path,monkeypatch):
    from src.core import visual_index
    from src.core.process_lock import exclusive_file_lock
    from src.hardware import scheduler
    def unavailable():raise RuntimeError('Controlled scheduler init failure')
    monkeypatch.setattr(scheduler,'get_disk_semaphore',unavailable)
    owner=tmp_path/'owner'
    with pytest.raises(RuntimeError,match='scheduler init failure') as captured:
        visual_index.inspect_source_identity_owned('unused',owner)
    assert captured.value  # Retain traceback so GC cannot hide a leaked claim.
    with exclusive_file_lock(owner/'owner.lock',conflict='Claim leaked'):pass


def test_owned_metadata_cancelled_admission_does_not_spawn(tmp_path,monkeypatch):
    from src.core import visual_index
    from src.hardware import scheduler
    from src.hardware.ffmpeg import FFmpegProcessRegistry
    def cancelled(*args,**kwargs):
        FFmpegProcessRegistry.mark_interrupted()
        return False
    monkeypatch.setattr(scheduler,'acquire_with_retry',cancelled)
    monkeypatch.setattr(visual_index,'_start_reader',lambda:pytest.fail('Cancelled admission spawned reader'))
    try:
        with pytest.raises(InterruptedError,match='during I/O admission'):
            visual_index.inspect_source_identity_owned('unused',tmp_path/'owner')
    finally:
        FFmpegProcessRegistry.reset_interrupted()


def test_owned_metadata_release_failure_does_not_leak_directory_claim(tmp_path,monkeypatch):
    from src.core import visual_index
    from src.core.process_lock import exclusive_file_lock
    from src.hardware import scheduler
    class Permit:
        def acquire(self,*args,**kwargs):return True
        def release(self):raise RuntimeError('Controlled permit release failure')
    monkeypatch.setattr(scheduler,'get_disk_semaphore',lambda:Permit())
    source=tmp_path/'source.bin';source.write_bytes(b'metadata')
    owner=tmp_path/'owner'
    with pytest.raises(RuntimeError,match='permit release failure') as captured:
        visual_index.inspect_source_identity_owned(source,owner)
    assert captured.value
    with exclusive_file_lock(owner/'owner.lock',conflict='Claim leaked'):pass
