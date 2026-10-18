"""Crash checkpoints for isolated analysis replays, never production task leases."""
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def run_lease(run):
    """Kernel-held byte lock releases on process death; a leftover file is harmless."""
    from src.core.process_lock import exclusive_file_lock
    with exclusive_file_lock(Path(run)/'run.lock',conflict='Replay directory already has an active writer'):
        yield


class ReplayJournal:
    def __init__(self, run, identity, *, resume=False):
        self.path = Path(run) / "checkpoint.json"
        if resume:
            self.value = json.loads(self.path.read_text(encoding="utf-8"))
            if self.value.get("schema") != 1 or self.value.get("identity") != identity:
                raise ValueError("Replay identity changed; create a new isolated run")
        else:
            self.value = {"schema": 1, "identity": identity, "sources": {}, "ranges": {}}
            self.save()

    def save(self):
        atomic_json(self.path, self.value)

    def freeze_source(self, path, content_hash):
        """Always reread full content on resume, including stat-preserving changes."""
        source = Path(path)
        before = source.stat()
        sha = content_hash(source)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Source changed while freezing identity")
        record = {"filepath": str(path), "size": after.st_size, "mtime_ns": after.st_mtime_ns, "sha256": sha}
        previous = self.value["sources"].get(str(path))
        if previous is not None and previous != record:
            raise ValueError("Frozen replay source changed; create a new run")
        self.value["sources"][str(path)] = record
        self.save()
        return record

    def range_complete(self, key, result, covered):
        previous = self.value["ranges"].get(key, {})
        if previous.get("success") and (not covered or previous.get("result_sha256") != digest(result)):
            raise ValueError("Completed replay range changed or lost reference coverage")
        return previous.get("success") is True

    def record_range(self, key, result, success, covered):
        self.value["ranges"][key] = {"success": bool(success and covered),
                                     "pipeline_success": bool(success), "covered": bool(covered),
                                     "result_sha256": digest(result)}
        self.save()
