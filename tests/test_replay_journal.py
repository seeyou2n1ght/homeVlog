"""Isolated replay checkpoint identity, interruption and writer ownership."""
import json
import os
import signal
from pathlib import Path
import subprocess
import sys

import pytest

from src.core.replay_evidence import content_sha256
from src.core.replay_journal import ReplayJournal, atomic_json, run_lease


def test_interrupted_ranges_and_identity(tmp_path):
    identity = {"code": "frozen", "samples": [1, 2]}
    journal = ReplayJournal(tmp_path, identity)
    completed = {"tasks": [{"analysis_status": "ANALYZED"}], "segments": [1]}
    journal.record_range("first", completed, True, True)
    journal.record_range("second", {}, False, False)
    resumed = ReplayJournal(tmp_path, identity, resume=True)
    assert resumed.range_complete("first", completed, True)
    assert not resumed.range_complete("second", {}, False)
    with pytest.raises(ValueError, match="identity changed"):
        ReplayJournal(tmp_path, {"code": "changed"}, resume=True)
    with pytest.raises(ValueError, match="changed or lost"):
        resumed.range_complete("first", completed, False)
    with pytest.raises(ValueError, match="changed or lost"):
        resumed.range_complete("first", {"segments": []}, True)
    assert not (tmp_path / "receipt.json").exists()


def test_success_without_coverage_cannot_be_reused(tmp_path):
    journal = ReplayJournal(tmp_path, {})
    journal.record_range("range", {"prediction": None}, True, False)
    assert not journal.range_complete("range", {"prediction": None}, False)


def test_source_content_change_with_identical_stat(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"original")
    journal = ReplayJournal(tmp_path, {})
    original = journal.freeze_source(source, content_sha256)
    stat = source.stat()
    source.write_bytes(b"modified")
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    resumed = ReplayJournal(tmp_path, {}, resume=True)
    with pytest.raises(ValueError, match="source changed"):
        resumed.freeze_source(source, content_sha256)
    assert resumed.value["sources"][str(source)] == original


def test_hashing_source_change_does_not_publish(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"before")
    journal = ReplayJournal(tmp_path, {})
    def changing_hash(path):
        path.write_bytes(b"longer after")
        return "hash"
    with pytest.raises(ValueError, match="while freezing"):
        journal.freeze_source(source, changing_hash)
    assert json.loads(journal.path.read_text())["sources"] == {}


def test_kernel_lease_rejects_another_process_and_releases(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = "from src.core.replay_journal import run_lease; import sys;\nwith run_lease(sys.argv[1]): print('acquired')"
    with run_lease(tmp_path):
        blocked = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=project,
                                 capture_output=True, text=True, timeout=15)
        assert blocked.returncode != 0
        assert "active writer" in blocked.stderr
    available = subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=project,
                               capture_output=True, text=True, timeout=15)
    assert available.returncode == 0, available.stderr
    assert "acquired" in available.stdout


def test_atomic_json_failure_preserves_previous(tmp_path, monkeypatch):
    target = tmp_path / "checkpoint.json"
    atomic_json(target, {"complete": 1})
    def fail(*args):
        raise OSError("replace failed")
    monkeypatch.setattr(Path, "replace", fail)
    with pytest.raises(OSError, match="replace failed"):
        atomic_json(target, {"complete": 2})
    assert json.loads(target.read_text()) == {"complete": 1}
    assert not list(tmp_path.glob("*.tmp"))


def test_process_termination_releases_kernel_lease(tmp_path):
    project = Path(__file__).resolve().parents[1]
    script = "from src.core.replay_journal import run_lease; import sys, os;\nwith run_lease(sys.argv[1]): print(os.getpid(), flush=True); sys.stdin.read()"
    child = subprocess.Popen([sys.executable, "-c", script, str(tmp_path)], cwd=project,
                             stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        actual_pid = int(child.stdout.readline().strip())
        with pytest.raises(RuntimeError, match="active writer"):
            with run_lease(tmp_path):
                pass
        # A Windows venv launcher can own another Python process; terminate the
        # actual lease owner, rather than merely its launcher.
        os.kill(actual_pid, signal.SIGTERM)
        child.communicate(timeout=15)
        with run_lease(tmp_path):
            assert (tmp_path / "run.lock").exists()
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=15)
