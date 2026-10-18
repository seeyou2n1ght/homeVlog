"""测试解码流完整性与截断判定策略 (Stream Completion Policy) 纯函数边界"""

import pytest
from src.stages.detector import evaluate_stream_completion, DecodeStreamDecision


def test_evaluate_stream_completion_normal():
    # 60s 文件，5 fps，预期 300 帧。实际解码 300 帧，正常退出
    decision = evaluate_stream_completion(
        total_frames=300,
        file_duration=60.0,
        effective_fps=5.0,
        returncode=0,
    )
    assert decision.complete is True
    assert decision.truncated is False
    assert decision.corrupted_tail is False
    assert decision.reason == "NORMAL_COMPLETION"


def test_evaluate_stream_completion_truncated_clean_eof():
    # 600s 文件，5 fps，预期 3000 帧 (85% 门限为 2550 帧)。
    # 实际仅录制 58s (290 帧) 干净提前 EOF (returncode=0)
    decision = evaluate_stream_completion(
        total_frames=290,
        file_duration=600.0,
        effective_fps=5.0,
        returncode=0,
    )
    assert decision.complete is True
    assert decision.truncated is True
    assert decision.corrupted_tail is False
    assert decision.reason == "TRUNCATED_EOF"


def test_evaluate_stream_completion_corrupted_tail_recovery():
    # 录制到一半硬件断电导致尾部残卷，抛出解码异常但保留了 500 个有效帧
    decision = evaluate_stream_completion(
        total_frames=500,
        file_duration=600.0,
        effective_fps=5.0,
        returncode=-1,
        has_tail_error=True,
    )
    assert decision.complete is True
    assert decision.truncated is True
    assert decision.corrupted_tail is True
    assert decision.reason == "CORRUPTED_TAIL"


def test_evaluate_stream_completion_insufficient_frames():
    # 彻底损坏原片，仅读出 0 帧或 1 帧
    for frames in (0, 1):
        decision = evaluate_stream_completion(
            total_frames=frames,
            file_duration=60.0,
            effective_fps=5.0,
            returncode=0,
        )
        assert decision.complete is False
        assert decision.reason == "INSUFFICIENT_FRAMES"


def test_evaluate_stream_completion_interrupted():
    # 用户按 Ctrl+C 或管线提前中断
    decision = evaluate_stream_completion(
        total_frames=500,
        file_duration=60.0,
        effective_fps=5.0,
        is_interrupted=True,
    )
    assert decision.complete is False
    assert decision.reason == "INTERRUPTED"

    decision_aborted = evaluate_stream_completion(
        total_frames=500,
        file_duration=60.0,
        effective_fps=5.0,
        is_aborted=True,
    )
    assert decision_aborted.complete is False
    assert decision_aborted.reason == "INTERRUPTED"


def test_evaluate_stream_completion_process_error():
    # 解码进程非零退出且帧数严重不足，无尾部残卷容错保护
    decision = evaluate_stream_completion(
        total_frames=10,
        file_duration=60.0,
        effective_fps=5.0,
        returncode=1,
        has_tail_error=False,
    )
    assert decision.complete is False
    assert decision.reason == "PROCESS_ERROR"


@pytest.mark.parametrize("frames", [255, 299, 300, 320])
def test_high_frame_coverage_does_not_hide_process_failure(frames):
    decision = evaluate_stream_completion(frames, 60, 5, returncode=1)
    assert not decision.complete
    assert decision.reason == "PROCESS_ERROR"


def test_timeout_wins_over_frames_and_tail_recovery():
    decision = evaluate_stream_completion(300, 60, 5, returncode=0, has_tail_error=True, timed_out=True)
    assert not decision.complete
    assert decision.reason == "TIMEOUT"


def test_clean_eof_missing_ten_percent_is_still_disclosed():
    decision = evaluate_stream_completion(270, 60, 5, returncode=0)
    assert decision.complete and decision.truncated
    assert decision.reason == "TRUNCATED_EOF"


def test_persisted_decode_extent_survives_reopen(tmp_path):
    import json
    from src.core.database import VlogDatabase
    from src.stages.companion import unavailable_source_ranges
    path = tmp_path / "extent.sqlite"
    db = VlogDatabase(path)
    db.conn.execute("INSERT INTO file_tasks(filepath,cam_index,date,file_start_time,file_end_time,file_duration) "
                    "VALUES ('source.mp4',0,'20260610','20260610010131','20260610011046',555)")
    db.conn.commit()
    completion = {"complete": True, "reason": "TRUNCATED_EOF", "declared_duration_s": 555.4,
                  "available_duration_s": 58., "sample_period_s": .5}
    db.set_file_metadata("source.mp4", 1, 58., decode_completion=completion)
    db.close()
    db = VlogDatabase(path)
    row = dict(db.conn.execute("SELECT * FROM file_tasks").fetchone())
    assert row["file_duration"] == 58
    assert json.loads(row["decode_completion"]) == completion
    gap = unavailable_source_ranges([row])[0]
    assert gap["missing_duration_s"] == 497.4
    assert gap["missing_start_offset_s"] == 58
    assert gap["missing_end_offset_s"] == 555.4
    completion["available_duration_s"] = 555.
    row["decode_completion"] = completion
    assert unavailable_source_ranges([row]) == []
    db.close()
@pytest.mark.skipif(__import__('os').name != 'nt', reason='Real Windows process ownership')
@pytest.mark.parametrize('cached_exit', [False, True])
def test_detector_unknown_stop_retains_reader_and_blocks_fallback(tmp_path, monkeypatch, request, cached_exit):
    import json
    import subprocess
    import sys
    import threading
    import _winapi
    from unittest.mock import MagicMock
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry, MediaTerminationUnconfirmed
    from src.hardware.scheduler import FileBudget
    from src.core.visual_index import reader_process_state
    from src.stages.detector import MotionDetector
    from src.hardware import scheduler
    monkeypatch.setattr(registry,'_processes',{})
    monkeypatch.setattr(registry,'_jobs',{})
    monkeypatch.setattr(registry,'_deferred',{})
    monkeypatch.setattr(registry,'_interrupted',False)
    disk,hardware=FileBudget(1),threading.BoundedSemaphore(1)
    monkeypatch.setattr(scheduler,'get_disk_semaphore',lambda:disk)
    monkeypatch.setattr(scheduler,'get_qsv_semaphore',lambda:hardware)
    monkeypatch.setattr('src.core.utils.TEMP_DIR',tmp_path)
    real_start,real_stop=registry.start,registry.terminate
    children=[]
    def start(key,cmd,**kwargs):
        code='import os,time;os.write(1,bytes([12])*(64*48));os.close(1);time.sleep(30)'
        child=real_start(key,[sys.executable,'-c',code],**kwargs)
        children.append(child)
        if cached_exit:
            child.returncode=0  # Explicit cached-status injection, not a NAS event.
        else:
            child.wait=lambda timeout=None: (_ for _ in ()).throw(subprocess.TimeoutExpired(cmd,timeout))
        return child
    monkeypatch.setattr(registry,'start',start)
    monkeypatch.setattr(registry,'terminate',lambda key: (_ for _ in ()).throw(OSError('controlled stop failure')))
    detector=MotionDetector({'detection':{'analysis_resolution':'64x48','analysis_fps':2,'analysis_fps_adaptive':False},
        'yolo':{'enabled':False},'audio_vad':{'enabled':False}},decode_gpu='qsv')
    detector._stream_motion=True
    fallback=MagicMock(side_effect=AssertionError('Second decoder was started'))
    monkeypatch.setattr(detector,'_decode_file_pyav',fallback)
    key=None
    try:
        with pytest.raises(MediaTerminationUnconfirmed,match='Decode cleanup'):
            detector._decode_file('controlled.mp4',1.)
        fallback.assert_not_called()
        assert len(children)==1
        child=children[0];key=next(iter(registry._deferred))
        held=registry._deferred[key]
        assert held['proc'] is child and held['io_sem'].hardware is hardware and held['io_sem'].disk is disk
        assert _winapi.WaitForSingleObject(child._handle,0)==_winapi.WAIT_TIMEOUT
        assert not registry.has_exited(child) and disk.available==0
        assert not hardware.acquire(timeout=0)
        assert list(tmp_path.glob('.pipe_decode_*.log'))
        before=reader_process_state(child.pid,handle=child._handle)
        real_stop(key)
        assert _winapi.WaitForSingleObject(child._handle,5000)==_winapi.WAIT_OBJECT_0
        if not cached_exit:
            child.wait=subprocess.Popen.wait.__get__(child)
        child.poll()
        assert registry.reap_cleanups()['recovered']==[key]
        assert disk.available==1 and hardware.acquire(timeout=0)
        hardware.release()
        assert registry.reap_cleanups()=={'recovered':[],'pending':[]}
        request.node.user_properties.append(('native_reader',json.dumps(dict(before=before,after=reader_process_state(child.pid,handle=child._handle)),ensure_ascii=True)))
    finally:
        for child in children:
            if key is not None and key in registry._processes:
                real_stop(key)
            assert _winapi.WaitForSingleObject(child._handle,5000)==_winapi.WAIT_OBJECT_0
        registry.reap_cleanups()


def test_unknown_audio_stop_cannot_degrade_to_silent_video_or_second_decoder(monkeypatch):
    from unittest.mock import MagicMock
    from src.hardware.ffmpeg import MediaTerminationUnconfirmed
    from src.stages.detector import MotionDetector
    detector=MotionDetector({'yolo':{'enabled':False}})
    monkeypatch.setattr(detector,'_decode_file_pipe',lambda *a: ([object()],{},dict(complete=True,effective_fps=2)))
    monkeypatch.setattr(detector,'_decode_audio_pipe',MagicMock(side_effect=MediaTerminationUnconfirmed('retained audio reader')))
    fallback=MagicMock(side_effect=AssertionError('A second decoder was started'))
    monkeypatch.setattr(detector,'_decode_file_pyav',fallback)
    with pytest.raises(MediaTerminationUnconfirmed,match='retained audio reader'):
        detector._decode_file('controlled.mp4',2)
    fallback.assert_not_called()


@pytest.mark.skipif(__import__('os').name != 'nt', reason='Real Windows watchdog and process handle')
def test_detector_watchdog_stops_cached_zero_process_with_stdout_still_open(tmp_path, monkeypatch, request):
    import json
    import sys
    import threading
    import time
    import _winapi
    from src.core.visual_index import reader_process_state
    from src.hardware import scheduler
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    from src.stages import detector as detector_module

    monkeypatch.setattr(registry, '_processes', {})
    monkeypatch.setattr(registry, '_jobs', {})
    monkeypatch.setattr(registry, '_deferred', {})
    monkeypatch.setattr(registry, '_interrupted', False)
    releases, children, timers, calls, leases = [], [], [], [], []
    real_timer, real_start, real_stop = threading.Timer, registry.start, registry.terminate

    class Budget(scheduler.FileBudget):
        def release(self, weight=1):
            super().release(weight)
            releases.append(weight)

    disk, hardware = Budget(1), threading.BoundedSemaphore(1)
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: disk)
    monkeypatch.setattr(scheduler, 'get_qsv_semaphore', lambda: hardware)
    monkeypatch.setattr('src.core.utils.TEMP_DIR', tmp_path)

    def lease_factory(resource):
        lease = scheduler.VideoLease(resource)
        leases.append(lease)
        return lease

    def start(key, command, **kwargs):
        # Two actual raw frames, then keep the real stdout descriptor open.
        code = 'import os,time;os.write(1,bytes([12])*(64*48*2));time.sleep(30)'
        child = real_start(key, [sys.executable, '-c', code], **kwargs)
        before = reader_process_state(child.pid, handle=child._handle)
        child.returncode = 0  # Controlled stale-cache injection; kernel stays live.
        children.append((key, child, before))
        return child

    def early_timer(interval, callback):
        timer = real_timer(.15, callback)
        timers.append(timer)
        return timer

    def stop(key):
        child = registry._processes.get(key)
        calls.append(dict(from_watchdog=bool(timers and threading.current_thread() is timers[0]),
                          key=key, kernel_wait=None if child is None else
                          _winapi.WaitForSingleObject(child._handle, 0),
                          disk_available=disk.available, hardware_available=hardware._value,
                          original_lease_held=bool(leases and leases[0].disk is disk)))
        return real_stop(key)

    def safety_stop():
        # Bound a failing regression too: old poll-only watchdog skips the kill.
        for _, child, _ in children:
            if _winapi.WaitForSingleObject(child._handle, 0) != _winapi.WAIT_OBJECT_0:
                _winapi.TerminateProcess(child._handle, 1)

    monkeypatch.setattr(detector_module, 'VideoLease', lease_factory)
    monkeypatch.setattr(registry, 'start', start)
    monkeypatch.setattr(registry, 'terminate', stop)
    monkeypatch.setattr(detector_module.threading, 'Timer', early_timer)
    detector = detector_module.MotionDetector(
        {'detection': {'analysis_resolution': '64x48', 'analysis_fps': 2,
                       'analysis_fps_adaptive': False},
         'yolo': {'enabled': False}, 'audio_vad': {'enabled': False}}, decode_gpu='qsv')
    guard = real_timer(2, safety_stop)
    guard.daemon = True
    guard.start()
    report = {'scope': 'Actual local Python/stdout; cached exit injection and early production watchdog; no NAS/media/GPU'}
    began = time.monotonic()
    try:
        frames, _, metadata = detector._decode_file_pipe('local-controlled', 1., 2.)
        assert len(children) == 1
        key, child, before = children[0]
        report.update(before=before, calls=calls, metadata=metadata,
                      decoded_frames=len(frames), wall_s=time.monotonic()-began)
        watchdog_calls = [call for call in calls if call['from_watchdog']]
        assert len(watchdog_calls) == 1
        assert watchdog_calls[0]['key'] == key
        assert watchdog_calls[0]['kernel_wait'] == _winapi.WAIT_TIMEOUT
        assert watchdog_calls[0]['disk_available'] == 0
        assert watchdog_calls[0]['hardware_available'] == 0
        assert watchdog_calls[0]['original_lease_held']
        assert len(frames) == 2 and metadata['completion_reason'] == 'TIMEOUT'
        assert not metadata['complete']
        assert registry.has_exited(child)
        assert _winapi.WaitForSingleObject(child._handle, 0) == _winapi.WAIT_OBJECT_0
        assert disk.available == 1 and hardware._value == 1 and releases == [1]
        assert not registry._processes and not registry._jobs and not registry._deferred
        assert registry.reap_cleanups() == {'recovered': [], 'pending': []}
        assert releases == [1] and not registry.is_interrupted()
    finally:
        guard.cancel()
        guard.join(timeout=5)
        for key, child, before in children:
            if _winapi.WaitForSingleObject(child._handle, 0) != _winapi.WAIT_OBJECT_0:
                _winapi.TerminateProcess(child._handle, 1)
            assert _winapi.WaitForSingleObject(child._handle, 5000) == _winapi.WAIT_OBJECT_0
            if key in registry._processes:
                real_stop(key)
                registry.wait_for_exit(child, timeout=5)
            report['after'] = reader_process_state(child.pid, handle=child._handle)
            report['same_process_handle'] = int(child._handle)
        for timer in timers:
            timer.cancel()
            timer.join(timeout=5)
            assert not timer.is_alive()
        registry.reap_cleanups()
        request.node.user_properties.append(('native_cached_watchdog', json.dumps(report, sort_keys=True)))
