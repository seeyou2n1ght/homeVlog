"""tests/test_prescreen_adaptive.py

验证预筛选阶段自适应动态阈值、4x4网格空间集中度算子、时间邻域先验保护
以及 tune_thresholds 超参数寻优工具的逻辑准确性。
"""

import json
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from src.prescreen import (
    _calc_dynamic_threshold,
    _calc_spatial_concentration,
    _prescreen_keyframes,
    prescreen_file,
)
from scripts.tune_thresholds import (
    PrescreenSample,
    evaluate_grid,
    run_grid_search,
    simulate_prediction,
)


@pytest.mark.parametrize('failure', ['stop_timeout', 'cached_zero_live_stderr'])
def test_stream_fps_native_unconfirmed_stop_retains_original_lease(monkeypatch, tmp_path, failure):
    """Real Windows child, finite lifetime; injected stop failure is not kernel exit."""
    import os
    import subprocess
    import sys
    import threading
    import time
    from pathlib import Path
    from src.hardware import scheduler
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry, MediaTerminationUnconfirmed
    from src.stages import prescreen

    if os.name != 'nt':
        pytest.skip('native Windows process handle verification')
    import _winapi
    disk = scheduler.FileBudget(1)
    hardware = threading.BoundedSemaphore(1)
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: disk)
    monkeypatch.setattr(scheduler, 'get_qsv_semaphore', lambda: hardware)
    started = []
    real_start = registry.start
    report = {'scope': 'local Python/CPU only, injected stop no-op and wait timeout'}

    def start(key, command, **kwargs):
        # Close pipes before the injected failure, so the old close order cannot block.
        close_stderr = 'os.close(2);' if failure == 'stop_timeout' else ''
        child = f"import os,time; os.write(1,bytes([100]*12+[220]*12)); os.close(1); {close_stderr} time.sleep(15)"
        proc = real_start(key, [sys.executable, '-c', child], **kwargs)
        started.append((key, proc, proc.kill, proc.terminate, proc.wait, proc.owned_job.close))
        monkeypatch.setattr(proc, 'kill', lambda: None)
        monkeypatch.setattr(proc, 'terminate', lambda: None)
        monkeypatch.setattr(proc.owned_job, 'close', lambda: None)
        def wait(timeout=None):
            if failure == 'cached_zero_live_stderr':
                return 0
            raise subprocess.TimeoutExpired(proc.args, timeout)
        monkeypatch.setattr(proc, 'wait', wait)
        if failure == 'cached_zero_live_stderr':
            proc.returncode = 0  # Deliberately false cached status; same kernel stays live.
        return proc

    monkeypatch.setattr(registry, 'start', start)
    began = time.monotonic()
    try:
        error = None
        try:
            result = prescreen._prescreen_stream_fps('local-controlled', 2, 1, 2, 2, 8.5, 'qsv', 1)
        except BaseException as caught:
            error, result = caught, None
        key, proc, *_ = started[0]
        kernel = _winapi.WaitForSingleObject(proc._handle, 0)
        readmitted = hardware.acquire(blocking=False)
        if readmitted:
            hardware.release()
        report['observation'] = dict(pid=proc.pid, handle=int(proc._handle), kernel_wait=kernel,
            error=type(error).__name__ if error else None, result=result, wall_s=time.monotonic()-began,
            registry_present=registry._processes.get(key) is proc, deferred=key in registry._deferred,
            disk_available=disk.available, hardware_readmitted=readmitted, cached_returncode=proc.returncode,
            reader_alive=[reader.is_alive() for reader in registry._deferred.get(key, {}).get('readers', [])])
        assert kernel == _winapi.WAIT_TIMEOUT
        assert isinstance(error, MediaTerminationUnconfirmed)
        assert registry._processes[key] is proc
        assert registry._deferred[key]['io_sem'].hardware is hardware
        assert registry._deferred[key]['io_sem'].disk is disk
        assert registry._deferred[key]['readers']
        if failure == 'cached_zero_live_stderr':
            assert registry._deferred[key]['readers'][0].is_alive()
        assert disk.available == 0 and not readmitted
        assert registry.reap_cleanups()['pending'] == [key]
    finally:
        for key, proc, kill, terminate, wait, close in started:
            proc.kill, proc.terminate, proc.wait, proc.owned_job.close = kill, terminate, wait, close
            if _winapi.WaitForSingleObject(proc._handle, 0) == _winapi.WAIT_TIMEOUT:
                proc.returncode = None
            if not registry.has_exited(proc):
                kill()  # The vulnerable baseline may already have detached this key.
            registry.terminate(key)
            registry.wait_for_exit(proc, timeout=5)
            close()
            if key in registry._deferred:
                for reader in registry._deferred[key]['readers']:
                    reader.join(timeout=2)
                registry.reap_cleanups()
                assert registry.reap_cleanups()['recovered'] == []
            elif key in registry._processes:
                registry.deregister(key)
            for stream in (proc.stdout, proc.stderr):
                if stream:
                    stream.close()
            report['cleanup'] = dict(pid=proc.pid, kernel_wait=_winapi.WaitForSingleObject(proc._handle, 0),
                returncode=proc.returncode, disk_available=disk.available,
                registry_present=key in registry._processes, deferred=key in registry._deferred,
                job_handle=int(proc.owned_job.handle or 0))
            assert report['cleanup']['kernel_wait'] == _winapi.WAIT_OBJECT_0
            assert disk.available == 1 and hardware.acquire(blocking=False)
            hardware.release()
        destination = Path(os.environ.get('HOMEVLOG_PRESCREEN_EVIDENCE', tmp_path))
        destination.mkdir(parents=True, exist_ok=True)
        (destination/f'native_{failure}_receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


@pytest.mark.parametrize('case,expected', [
    ('early_positive', 'SUSPICIOUS'), ('negative', 'STATIC'),
    ('ordinary_failure', 'FAILED'), ('first_frame_failure', 'FAILED'), ('cancelled', 'FAILED'),
])
def test_stream_fps_local_python_confirmed_exit_contracts(monkeypatch, case, expected):
    import os
    import sys
    import threading
    from src.hardware import scheduler
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    from src.stages import prescreen

    disk, hardware = scheduler.FileBudget(1), threading.BoundedSemaphore(1)
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: disk)
    monkeypatch.setattr(scheduler, 'get_qsv_semaphore', lambda: hardware)
    started = []
    real_start = registry.start
    frames = [] if case in ('first_frame_failure', 'cancelled') else [100]*12 + [220 if case == 'early_positive' else 100]*12
    code = 2 if case in ('ordinary_failure', 'first_frame_failure') else 0
    pause = 'time.sleep(10);' if case in ('early_positive', 'cancelled') else ''

    def start(key, command, **kwargs):
        child = f"import os,time; os.write(1,bytes({frames!r})); os.write(2,b'controlled stderr\\n'); {pause} raise SystemExit({code})"
        proc = real_start(key, [sys.executable, '-c', child], **kwargs)
        started.append((key, proc))
        if case == 'cancelled':
            registry.kill_all()
        return proc

    monkeypatch.setattr(registry, 'start', start)
    try:
        result = prescreen._prescreen_stream_fps('local-controlled', 2, 1, 2, 2, 8.5, 'qsv', 1)
        assert result['status'] == expected
        key, proc = started[0]
        assert registry.has_exited(proc)
        assert key not in registry._processes and key not in registry._deferred
        assert disk.available == 1 and hardware.acquire(blocking=False)
        hardware.release()
        assert proc.stdout.closed and proc.stderr.closed
    finally:
        for key, proc in started:
            if not registry.has_exited(proc):
                proc.kill()
            registry.wait_for_exit(proc, timeout=5)
            if key in registry._processes:
                registry.terminate(key)
                registry.deregister(key)
            if os.name == 'nt':
                proc.owned_job.close()
        registry.reset_interrupted()


def test_stream_fps_spawn_failure_releases_original_lease(monkeypatch):
    import threading
    from src.hardware import scheduler
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    from src.stages import prescreen

    disk, hardware = scheduler.FileBudget(1), threading.BoundedSemaphore(1)
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: disk)
    monkeypatch.setattr(scheduler, 'get_qsv_semaphore', lambda: hardware)
    monkeypatch.setattr(registry, 'start', MagicMock(side_effect=OSError('controlled spawn failure')))
    with pytest.raises(OSError, match='controlled spawn failure'):
        prescreen._prescreen_stream_fps('local-controlled', 2, 1, 2, 2, 8.5, 'qsv', 1)
    assert disk.available == 1 and hardware.acquire(blocking=False)
    hardware.release()


def test_keyframe_unconfirmed_media_stop_propagates_without_seek_fallback(monkeypatch):
    from src.hardware.ffmpeg import MediaTerminationUnconfirmed
    from src.stages import prescreen
    import av

    opened = MagicMock(side_effect=MediaTerminationUnconfirmed('controlled unknown stop'))
    monkeypatch.setattr(av, 'open', opened)
    extract = MagicMock()
    monkeypatch.setattr(prescreen, '_extract_frame', extract)
    with pytest.raises(MediaTerminationUnconfirmed):
        prescreen.prescreen_file('local-controlled', 2, {})
    extract.assert_not_called()


def test_extract_frame_native_unknown_stop_retains_coupled_lease(monkeypatch, tmp_path):
    import os
    import sys
    import threading
    import time
    from pathlib import Path
    from src.hardware import scheduler
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry, MediaTerminationUnconfirmed
    from src.stages import prescreen

    if os.name != 'nt':
        pytest.skip('native Windows pipe readers and kernel handle verification')
    import _winapi
    disk, hardware = scheduler.FileBudget(1), threading.BoundedSemaphore(1)
    acquire, release = MagicMock(wraps=disk.acquire), MagicMock(wraps=disk.release)
    monkeypatch.setattr(disk, 'acquire', acquire)
    monkeypatch.setattr(disk, 'release', release)
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: disk)
    monkeypatch.setattr(scheduler, 'get_qsv_semaphore', lambda: hardware)
    started = []
    real_start = registry.start
    report = {'scope': '_extract_frame -> production run_ffmpeg; CPU finite local child, injected Job stop failure'}

    def start(key, command, **kwargs):
        child = "import os,time; os.write(1,bytes([100]*12)); os.write(2,b'controlled extract\\n'); time.sleep(15)"
        proc = real_start(key, [sys.executable, '-c', child], **kwargs)
        started.append((key, proc, proc.owned_job.close))
        monkeypatch.setattr(proc.owned_job, 'close', MagicMock(side_effect=OSError('controlled Job stop failure')))
        return proc

    monkeypatch.setattr(registry, 'start', start)
    began = time.monotonic()
    try:
        with pytest.raises(MediaTerminationUnconfirmed):
            prescreen._extract_frame('local-controlled', 0, 2, 2, timeout=.2)
        assert len(started) == 1  # An unknown stop cannot trigger the second seek attempt.
        key, proc, _ = started[0]
        item = registry._deferred[key]
        readers = item['readers']
        readmitted = hardware.acquire(blocking=False)
        if readmitted:
            hardware.release()
        report['observation'] = dict(pid=proc.pid, kernel_wait=_winapi.WaitForSingleObject(proc._handle, 0),
            wall_s=time.monotonic()-began, disk_available=disk.available, hardware_readmitted=readmitted,
            reader_alive=[reader.is_alive() for reader in readers], acquire_calls=acquire.call_count,
            release_calls=release.call_count, registry_present=registry._processes.get(key) is proc)
        assert report['observation']['kernel_wait'] == _winapi.WAIT_TIMEOUT
        assert isinstance(item['io_sem'], scheduler.VideoLease)
        assert item['io_sem'].hardware is hardware and item['io_sem'].disk is disk
        assert len(readers) == 2 and all(reader.is_alive() for reader in readers)
        assert disk.available == 0 and not readmitted
        acquire.assert_called_once()
        release.assert_not_called()
        assert key in registry.reap_cleanups()['pending']
    finally:
        for key, proc, close in started:
            proc.owned_job.close = close
            registry.terminate(key)
            registry.wait_for_exit(proc, timeout=5)
            if key in registry._deferred:
                for reader in registry._deferred[key]['readers']:
                    reader.join(timeout=2)
                assert registry.reap_cleanups()['recovered'] == [key]
                assert registry.reap_cleanups()['recovered'] == []
                release.assert_called_once_with(weight=1)
            else:
                registry.deregister(key)
            assert disk.available == 1 and hardware.acquire(blocking=False)
            hardware.release()
            report['cleanup'] = dict(pid=proc.pid, kernel_wait=_winapi.WaitForSingleObject(proc._handle, 0),
                job_handle=int(proc.owned_job.handle or 0), disk_available=disk.available,
                acquire_calls=acquire.call_count, release_calls=release.call_count,
                registry_present=key in registry._processes, deferred=key in registry._deferred)
        destination = Path(os.environ.get('HOMEVLOG_PRESCREEN_EVIDENCE', tmp_path))
        destination.mkdir(parents=True, exist_ok=True)
        (destination/'native_extract_receipt.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


@pytest.mark.parametrize('case,calls', [('success', 1), ('nonzero', 2), ('short_frame', 2)])
def test_extract_frame_local_confirmed_result_keeps_finite_retry(monkeypatch, case, calls):
    import os
    import sys
    import threading
    from src.hardware import scheduler
    from src.hardware.ffmpeg import FFmpegProcessRegistry as registry
    from src.stages import prescreen

    disk, hardware = scheduler.FileBudget(1), threading.BoundedSemaphore(1)
    monkeypatch.setattr(scheduler, 'get_disk_semaphore', lambda: disk)
    monkeypatch.setattr(scheduler, 'get_qsv_semaphore', lambda: hardware)
    real_start, started = registry.start, []
    def start(key, command, **kwargs):
        count, code = (1 if case == 'short_frame' else 12), (2 if case == 'nonzero' else 0)
        child = f'import os; os.write(1,bytes([100]*{count})); raise SystemExit({code})'
        proc = real_start(key, [sys.executable, '-c', child], **kwargs)
        started.append((key, proc))
        return proc
    monkeypatch.setattr(registry, 'start', start)
    try:
        result = prescreen._extract_frame('local-controlled', 0, 2, 2, timeout=2)
        assert len(started) == calls
        if case == 'success':
            assert np.array_equal(result, np.full((2, 2, 3), 100, dtype=np.uint8))
        else:
            assert result is None
        assert disk.available == 1 and hardware.acquire(blocking=False)
        hardware.release()
        assert all(registry.has_exited(proc) and key not in registry._processes for key, proc in started)
    finally:
        for key, proc in started:
            if not registry.has_exited(proc):
                proc.kill()
            registry.wait_for_exit(proc, timeout=5)
            if key in registry._processes:
                registry.terminate(key)
                registry.deregister(key)
            if os.name == 'nt':
                proc.owned_job.close()


def test_calc_dynamic_threshold():
    base_th = 8.0

    # 1. 正常白天光照 (mean_luma=100) -> 保持基准阈值
    th_day = _calc_dynamic_threshold(base_th, mean_luma=100.0, is_prior_active=False)
    assert th_day == 8.0

    # 2. 暗光/夜视微光 (mean_luma=30) -> 下调至 40% (3.20)
    th_night = _calc_dynamic_threshold(base_th, mean_luma=30.0, is_prior_active=False)
    assert th_night == 3.20

    # 3. 极暗光保底 (mean_luma=5.0, base_th=3.0) -> 下限保底 2.0
    th_min = _calc_dynamic_threshold(3.0, mean_luma=5.0, is_prior_active=False)
    assert th_min == 2.0

    # 4. 逆光/强光噪点密集 (mean_luma=200.0) -> 适度上浮 25% (10.0)
    th_bright = _calc_dynamic_threshold(base_th, mean_luma=200.0, is_prior_active=False)
    assert th_bright == 10.0

    # 5. 时间邻域先验保护生效 (is_prior_active=True) -> 下调 35%
    th_prior_day = _calc_dynamic_threshold(base_th, mean_luma=100.0, is_prior_active=True)
    assert th_prior_day == round(8.0 * 0.65, 2)  # 5.20

    th_prior_night = _calc_dynamic_threshold(base_th, mean_luma=30.0, is_prior_active=True)
    assert th_prior_night == round(3.20 * 0.65, 2)  # 2.08

    # 6. 先验保护下限保底 1.8
    th_prior_floor = _calc_dynamic_threshold(2.0, mean_luma=20.0, is_prior_active=True)
    assert th_prior_floor == 1.8


@pytest.mark.parametrize('threshold,change,prior', [
    (8.5, 13, True), (8.5, 13, False), (8.5, 20, True),
    (8.5, 0, True), (1.0, 3, True), (1.0, 3, False),
])
def test_deferred_keyframe_prior_matches_serial_without_second_decode(threshold, change, prior):
    import av
    from fractions import Fraction
    from src.stages import prescreen

    images = [np.full((16, 16), 100, dtype=np.uint8) for _ in range(3)]
    for image in images[1:]:
        image[:, :8] += change

    def container():
        frames = [av.VideoFrame.from_ndarray(image, format='gray') for image in images]
        for index, frame in enumerate(frames):
            frame.pts = index
        stream = MagicMock(duration=3, time_base=Fraction(1, 1))
        result = MagicMock(duration=3_000_000)
        result.streams.video = [stream]
        result.streams.audio = []
        result.decode.return_value = iter(frames)
        return result

    config = {'detection': {'prescreen_diff_threshold': threshold,
                            'prescreen_resolution': '16x16', 'prescreen_segments': 10}}
    semaphore = MagicMock()
    serial = container()
    deferred = container()
    with patch('av.open') as opened, \
         patch('src.hardware.scheduler.get_disk_semaphore', return_value=semaphore), \
         patch.object(prescreen, 'acquire_with_retry', return_value=True):
        opened.return_value.__enter__.return_value = serial
        expected = prescreen_file('controlled.mp4', 3, config, is_prior_active=prior)
        semaphore.reset_mock()
        opened.reset_mock()
        opened.return_value.__enter__.return_value = deferred

        def resolve():
            semaphore.release.assert_called_once()
            return prior

        actual = prescreen_file('controlled.mp4', 3, config, is_prior_active=not prior,
                               prior_active_resolver=resolve)
        opened.assert_called_once()
        deferred.decode.assert_called_once()
    assert actual['status'] == expected['status']
    a, b = [json.loads(r['result_json']) for r in (actual, expected)]
    for key in ('threshold', 'sample_ts', 'diffs', 'early_stop', 'checked_pairs'):
        assert a[key] == b[key]
    assert '_prior_active_result' not in actual


def test_spatial_concentration_diffuse_vs_localized():
    # 1. 全画幅弥散性光影跳变 (整图均匀变化)
    diffuse_map = np.full((180, 320), fill_value=10.0, dtype=np.float32)
    conc_diffuse = _calc_spatial_concentration(diffuse_map)
    # 所有 4x4 网格均值相同，集中度比值应为 1.0
    assert abs(conc_diffuse - 1.0) < 1e-3

    # 2. 局部高能前景运动 (仅左上角一个网格单元剧烈变化，其余全为 0)
    local_map = np.zeros((180, 320), dtype=np.float32)
    local_map[0:45, 0:80] = 50.0  # 局部能量密集
    conc_local = _calc_spatial_concentration(local_map)
    # 空间集中度应显著高于 1.35
    assert conc_local > 3.0


def test_prescreen_night_motion_not_dropped_as_static():
    """测试暗光下 max_diff 处于 (夜视阈值, 全局阈值) 区间的微弱运动不会在收尾时被误判为 STATIC"""
    # 模拟视频帧数据：两帧均为暗光（均值约 25），前后帧绝对差分均值 4.5
    # 若无动态阈值贯穿首尾判定，4.5 < 8.0 会被当场漏检为 STATIC
    fake_frame1 = np.full((180, 320), fill_value=25, dtype=np.uint8)
    fake_frame2 = np.full((180, 320), fill_value=25, dtype=np.uint8)
    # 在 1/4 画面制造高能前景微动 (均值差分 5.0, 介于夜视阈值 3.2 与全局阈值 8.0 之间)
    fake_frame2[0:90, 0:160] = 45

    # 构造 PyAV mock 解码
    mock_av_frame1 = MagicMock()
    mock_av_frame1.planes = [fake_frame1.tobytes()]
    mock_av_frame1.to_ndarray.return_value = fake_frame1
    mock_av_frame1.width = 320
    mock_av_frame1.height = 180
    mock_av_frame1.pts = 0

    mock_av_frame2 = MagicMock()
    mock_av_frame2.planes = [fake_frame2.tobytes()]
    mock_av_frame2.to_ndarray.return_value = fake_frame2
    mock_av_frame2.width = 320
    mock_av_frame2.height = 180
    mock_av_frame2.pts = 1000

    mock_stream = MagicMock()
    mock_stream.time_base = 0.001
    mock_stream.codec_context = MagicMock()

    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.streams.audio = []
    mock_container.decode.return_value = [mock_av_frame1, mock_av_frame2]

    with patch("av.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_container

        res = _prescreen_keyframes(
            filepath="dummy.mp4",
            duration=60.0,
            max_keyframes=5,
            threshold=8.0,
            is_prior_active=False,
        )

        assert res["status"] == "SUSPICIOUS"
        result_data = json.loads(res["result_json"])
        assert result_data["mean_luma"] < 50.0
        # 验证使用的判定阈值已自适应下调 (8.0 * 0.4 = 3.2)
        assert result_data["threshold"] == 3.2


def test_prescreen_diffuse_light_suppressed_as_static():
    """测试全画幅均匀弥散性光影跳变（如傍晚曝光变化）被空间集中度算子正确抑制，判定为 STATIC"""
    # 模拟两帧全图均匀变暗或变亮（全图差分均为 9.0，高于基础阈值 8.0，但无局部高能）
    fake_frame1 = np.full((180, 320), fill_value=120, dtype=np.uint8)
    fake_frame2 = np.full((180, 320), fill_value=129, dtype=np.uint8)  # 全图均差 9.0

    mock_av_frame1 = MagicMock()
    mock_av_frame1.planes = [fake_frame1.tobytes()]
    mock_av_frame1.to_ndarray.return_value = fake_frame1
    mock_av_frame1.width = 320
    mock_av_frame1.height = 180
    mock_av_frame1.pts = 0

    mock_av_frame2 = MagicMock()
    mock_av_frame2.planes = [fake_frame2.tobytes()]
    mock_av_frame2.to_ndarray.return_value = fake_frame2
    mock_av_frame2.width = 320
    mock_av_frame2.height = 180
    mock_av_frame2.pts = 1000

    mock_stream = MagicMock()
    mock_stream.time_base = 0.001
    mock_stream.codec_context = MagicMock()

    mock_container = MagicMock()
    mock_container.streams.video = [mock_stream]
    mock_container.streams.audio = []
    mock_container.decode.return_value = [mock_av_frame1, mock_av_frame2]

    with patch("av.open") as mock_open:
        mock_open.return_value.__enter__.return_value = mock_container

        res = _prescreen_keyframes(
            filepath="dusk_diffuse.mp4",
            duration=60.0,
            max_keyframes=5,
            threshold=8.0,
            is_prior_active=False,
        )

        assert res["status"] == "STATIC"
        result_data = json.loads(res["result_json"])
        assert result_data["early_stop"] is False
        assert result_data["max_diff"] == 9.0
        assert result_data["concentration"] < 1.2


def test_prescreen_checks_last_keyframe_skipped_by_stride():
    first = np.full((180, 320), 100, dtype=np.uint8)
    last = first.copy()
    last[135:180, 240:320] = 220
    frames = []
    for pts, pixels in ((0, first), (5000, first), (99000, last)):
        frame = MagicMock()
        frame.pts = pts
        frame.to_ndarray.return_value = pixels
        frames.append(frame)

    stream = MagicMock()
    stream.time_base = 0.001
    stream.duration = None
    container = MagicMock()
    container.duration = None
    container.streams.video = [stream]
    container.streams.audio = []
    container.decode.return_value = frames
    with patch("av.open") as mock_open:
        mock_open.return_value.__enter__.return_value = container
        result = _prescreen_keyframes(
            filepath="last_keyframe.mp4", duration=100.0,
            max_keyframes=10, threshold=8.5,
        )

    assert result["status"] == "SUSPICIOUS"
    assert json.loads(result["result_json"])["sample_ts"] == [0.0, 99.0]


def test_tune_thresholds_simulation_and_grid_search():
    """验证超参数寻优算法对暗光漏检和强光误报样本的正确打分与最优推荐"""
    samples = [
        # 样本1: 暗光真实人体移动 (luma 28, max_diff 4.0, 集中度高) -> 真值 MOTION
        PrescreenSample(
            file_id=1,
            filepath="night_motion.mp4",
            ground_truth="MOTION",
            max_diff=4.0,
            mean_luma=28.0,
            concentration=1.55,
            diffs=[1.0, 4.0],
        ),
        # 样本2: 傍晚光照渐变 (luma 120, max_diff 9.5, 但全图弥漫 concentration 1.05) -> 真值 STATIC
        PrescreenSample(
            file_id=2,
            filepath="dusk_light_shift.mp4",
            ground_truth="STATIC",
            max_diff=9.5,
            mean_luma=120.0,
            concentration=1.05,
            diffs=[2.0, 9.5],
        ),
        # 样本3: 白天正常人物穿行 (luma 130, max_diff 14.0, concentration 1.6) -> 真值 MOTION
        PrescreenSample(
            file_id=3,
            filepath="day_motion.mp4",
            ground_truth="MOTION",
            max_diff=14.0,
            mean_luma=130.0,
            concentration=1.6,
            diffs=[1.0, 14.0],
        ),
    ]

    # 1. 验证单一预测：
    # 当 base_threshold=8.0, night_factor=0.4 时：
    # 样本1 的有效阈值是 8.0*0.4=3.2，4.0 > 3.2 且 concentration=1.55 >= 1.35 -> 识别为 MOTION (TP)
    pred_sample1 = simulate_prediction(samples[0], base_threshold=8.0, night_factor=0.4, concentration_thresh=1.35)
    assert pred_sample1 == "MOTION"

    # 若未开启暗光自适应 (night_factor=1.0)：
    # 样本1 的阈值为 8.0，4.0 < 8.0 -> 误判为 STATIC (FN 漏检！)
    pred_sample1_rigid = simulate_prediction(samples[0], base_threshold=8.0, night_factor=1.0, concentration_thresh=1.35)
    assert pred_sample1_rigid == "STATIC"

    # 2. 验证网格评估打分：
    eval_adaptive = evaluate_grid(samples, base_threshold=8.0, night_factor=0.4, concentration_thresh=1.35)
    eval_rigid = evaluate_grid(samples, base_threshold=8.0, night_factor=1.0, concentration_thresh=1.35)

    # 开启自适应时 0 漏检 (fn=0, recall=1.0)
    assert eval_adaptive.fn == 0
    assert eval_adaptive.recall == 1.0

    # 未开启自适应时有 1 处严重漏检 (fn=1)
    assert eval_rigid.fn == 1

    # 成本打分对比：自适应方案显著优于僵化方案
    assert eval_adaptive.cost_score < eval_rigid.cost_score

    # 3. 验证全网格搜索能够推选出正确的最优解
    results = run_grid_search(samples, base_thresholds=[6.0, 8.0, 10.0], night_factors=[0.4, 0.9])
    best = results[0]
    assert best.night_factor == 0.4
    assert best.fn == 0


def test_calc_spatial_metrics_and_pure_evaluation_policy():
    """验证重构后的纯函数算子与多轨运动决策策略行为准确性。"""
    from src.stages.prescreen import _calc_spatial_metrics, _evaluate_prescreen_motion, MotionDecision

    # 1. 空间指标计算边界验证
    empty_map = np.zeros((0, 0), dtype=np.float32)
    conc, max_cell = _calc_spatial_metrics(empty_map)
    assert conc == 1.0 and max_cell == 0.0

    # 均匀扩散
    diffuse_map = np.full((180, 320), fill_value=10.0, dtype=np.float32)
    conc, max_cell = _calc_spatial_metrics(diffuse_map)
    assert abs(conc - 1.0) < 1e-3
    assert abs(max_cell - 10.0) < 1e-3

    # 局部高能单网格
    local_map = np.zeros((180, 320), dtype=np.float32)
    local_map[0:45, 0:80] = 60.0
    conc, max_cell = _calc_spatial_metrics(local_map)
    assert conc > 3.0
    assert abs(max_cell - 60.0) < 1e-3

    # 2. 纯决策策略：双轨爆发防漏检（全图均值被稀释但单格爆发）
    dec_burst = _evaluate_prescreen_motion(
        d=4.0, current_threshold=8.0, mean_luma=120.0,
        concentration=2.0, max_cell_energy=20.0, is_prior_active=False
    )
    assert dec_burst.is_motion is True
    assert dec_burst.reason == "dual_track_burst_motion"

    # 3. 纯决策策略：夜视全图白噪点抑制
    dec_night_noise = _evaluate_prescreen_motion(
        d=5.0, current_threshold=3.2, mean_luma=30.0,
        concentration=1.1, max_cell_energy=5.5, is_prior_active=False
    )
    assert dec_night_noise.is_motion is False
    assert dec_night_noise.suppressed is True
    assert dec_night_noise.reason == "night diffuse noise"

    # 4. 纯决策策略：夜视局部有效动作确认
    dec_night_valid = _evaluate_prescreen_motion(
        d=5.0, current_threshold=3.2, mean_luma=30.0,
        concentration=1.35, max_cell_energy=7.0, is_prior_active=False
    )
    assert dec_night_valid.is_motion is True
    assert dec_night_valid.reason == "night_motion_confirmed"

    # 5. 纯决策策略：白天弥漫运动抑制
    dec_day_diffuse = _evaluate_prescreen_motion(
        d=9.0, current_threshold=8.0, mean_luma=120.0,
        concentration=1.1, max_cell_energy=10.0, is_prior_active=False
    )
    assert dec_day_diffuse.is_motion is False
    assert dec_day_diffuse.suppressed is True
    assert dec_day_diffuse.reason == "diffuse motion"

    # 6. 纯决策策略：先验保护下放宽集中度
    dec_prior_relaxed = _evaluate_prescreen_motion(
        d=6.0, current_threshold=5.2, mean_luma=120.0,
        concentration=1.20, max_cell_energy=9.0, is_prior_active=True
    )
    assert dec_prior_relaxed.is_motion is True
    assert dec_prior_relaxed.reason == "normal_motion_confirmed"


def test_prescreen_legacy_seek_truncated_stream_routes_to_suspicious():
    """当视频物理流截断导致后续时间戳 seek 越界返回 None 时，安全降级为 SUSPICIOUS。"""
    config = {
        "detection": {
            "prescreen_mode": "legacy_seek",
            "prescreen_segments": 5,
            "prescreen_diff_threshold": 10.0,
        }
    }

    def mock_extract(filepath, timestamp, width, height, **kwargs):
        if timestamp == 0.0:
            return np.ones((height, width, 3), dtype=np.uint8) * 100
        return None  # 越界 EOF

    with patch("src.stages.prescreen._extract_frame", side_effect=mock_extract):
        res = prescreen_file("mock_truncated.mp4", duration=300.0, config=config)

    assert res["status"] == "SUSPICIOUS"
    result_meta = json.loads(res["result_json"])
    assert result_meta.get("truncated") is True
    assert "seek_past_eof" in result_meta.get("reason", "")


def test_prescreen_keyframes_error_after_samples_routes_to_suspicious():
    """当 PyAV 关键帧初筛在读出部分帧后遭遇数据损坏异常时，安全降级为 SUSPICIOUS。"""
    class MockStream:
        duration = 1000
        time_base = 1 / 1000
        codec_context = MagicMock()

    class MockContainer:
        def __init__(self):
            self.streams = MagicMock()
            self.streams.video = [MockStream()]
            self.streams.audio = []
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def decode(self, stream):
            for i in range(2):
                frame = MagicMock()
                frame.pts = i * 100
                frame.type = "video"
                yield frame
            import av
            raise av.error.InvalidDataError(1094995529, "Invalid data found")

    mock_io = MagicMock()
    with patch("av.open", return_value=MockContainer()), \
         patch("src.hardware.scheduler.get_disk_semaphore", return_value=mock_io), \
         patch("src.stages.prescreen.video_frame_to_gray", return_value=np.zeros((180, 320), dtype=np.uint8)), \
         patch("src.stages.prescreen.acquire_with_retry", return_value=True):
        res = _prescreen_keyframes("corrupted_tail.mp4", duration=100.0, max_keyframes=10, threshold=8.0)

    assert res["status"] == "SUSPICIOUS"
    meta = json.loads(res["result_json"])
    assert meta.get("truncated") is True
    assert "stream_error" in meta.get("reason", "")


def test_legacy_threshold_tool_rejects_application_before_io(tmp_path, monkeypatch):
    from scripts import tune_thresholds as tool
    config = tmp_path / "settings.yaml"
    config.write_text("detection: {prescreen_diff_threshold: 8.5}\n")
    original = config.read_bytes()
    with pytest.raises(RuntimeError, match="disabled"):
        tool.apply_best_config(config, None)
    monkeypatch.setattr("sys.argv", ["tune_thresholds.py", "--apply"])
    monkeypatch.setattr(tool, "print_dataset_summary", lambda *a: pytest.fail("database accessed"))
    with pytest.raises(SystemExit) as stopped:
        tool.main()
    assert stopped.value.code == 2
    assert config.read_bytes() == original
    assert not config.with_suffix(".yaml.bak").exists()


def test_legacy_threshold_partial_negative_is_not_file_truth(tmp_path):
    import sqlite3
    import json
    from scripts.tune_thresholds import load_labeled_samples
    db = tmp_path / "legacy.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("CREATE TABLE file_tasks(id INTEGER,filepath TEXT,prescreen_result TEXT); "
                           "CREATE TABLE segments(file_id INTEGER,manual_label TEXT);")
        conn.execute("INSERT INTO file_tasks VALUES(1,?,?)", ("source.mp4", json.dumps({"max_diff": 5})))
        conn.execute("INSERT INTO segments VALUES(1,'FALSE_ALARM')")
    before = db.read_bytes()
    assert load_labeled_samples(db) == []
    assert db.read_bytes() == before
