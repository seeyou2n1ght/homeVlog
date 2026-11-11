"""Tests for VlogDatabase foreign key constraints, cascade deletion, and relational integrity."""

import sqlite3
import pytest

from src.core.database import VlogDatabase


def test_legacy_database_rejected_before_schema_changes(tmp_path):
    db_path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(db_path)
    try:
        with legacy:
            legacy.execute("CREATE TABLE file_tasks (analysis_segments TEXT)")
            legacy.execute("INSERT INTO file_tasks VALUES ('[]')")
    finally:
        legacy.close()
    original = db_path.read_bytes()
    original_entries = set(tmp_path.iterdir())

    with pytest.raises(RuntimeError, match="Back up and move the existing database"):
        VlogDatabase(db_path)

    assert db_path.read_bytes() == original
    assert set(tmp_path.iterdir()) == original_entries


def test_foreign_key_pragma_enabled(tmp_path):
    """Verify PRAGMA foreign_keys is enabled (1) upon database connection."""
    db_path = tmp_path / "test_fk_pragma.db"
    db = VlogDatabase(db_path=db_path)
    try:
        row = db.conn.execute("PRAGMA foreign_keys;").fetchone()
        assert row is not None
        assert row[0] == 1, f"Expected PRAGMA foreign_keys to be 1, got {row[0]}"
    finally:
        db.close()


def test_foreign_key_constraint_enforcement_on_insert(tmp_path):
    """Verify inserting a segment with a non-existent file_id raises sqlite3.IntegrityError."""
    db_path = tmp_path / "test_fk_enforce.db"
    db = VlogDatabase(db_path=db_path)
    try:
        non_existent_file_id = 999999
        with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY constraint failed"):
            with db._lock:
                db.conn.execute(
                    """INSERT INTO segments (file_id, start_time, end_time, state)
                       VALUES (?, ?, ?, ?)""",
                    (non_existent_file_id, 0.0, 10.0, "DYNAMIC"),
                )
                db.conn.commit()
    finally:
        db.close()


def test_foreign_key_on_delete_cascade_segments(tmp_path):
    """Verify deleting a file_tasks row automatically cascades and deletes all associated segments."""
    db_path = tmp_path / "test_cascade.db"
    db = VlogDatabase(db_path=db_path)
    try:
        filepath = "camera01_20261110_100000.mp4"
        db.add_file_task(
            filepath=filepath,
            cam_index=0,
            date="20261110",
            file_start_time="20261110100000",
            file_end_time="20261110100500",
            file_duration=300.0,
        )

        segments_data = [
            {"start_time": 0.0, "end_time": 10.0, "state": "DYNAMIC", "max_energy": 5.0, "avg_confidence": 0.9},
            {"start_time": 10.0, "end_time": 30.0, "state": "STATIC", "max_energy": 0.1, "avg_confidence": 0.0},
            {"start_time": 30.0, "end_time": 45.0, "state": "DYNAMIC", "max_energy": 7.2, "avg_confidence": 0.95},
        ]
        db.set_analysis_result(filepath, "ANALYZED", segments_data)

        # Confirm segments are present
        segments_before = db.get_segments_for_file(filepath)
        assert len(segments_before) == 3
        total_segs = db.conn.execute("SELECT count(*) FROM segments").fetchone()[0]
        assert total_segs == 3

        # Delete the parent file_tasks row
        with db._lock:
            db.conn.execute("DELETE FROM file_tasks WHERE filepath=?", (filepath,))
            db.conn.commit()

        # Confirm segments are cascade-deleted
        total_segs_after = db.conn.execute("SELECT count(*) FROM segments").fetchone()[0]
        assert total_segs_after == 0, f"Expected 0 segments after parent delete, found {total_segs_after}"

        segments_after = db.get_segments_for_file(filepath)
        assert len(segments_after) == 0
    finally:
        db.close()


def test_foreign_key_cascade_scoped_to_deleted_file(tmp_path):
    """Verify deleting one file_task only cascades its own segments, preserving other files' segments."""
    db_path = tmp_path / "test_cascade_scoped.db"
    db = VlogDatabase(db_path=db_path)
    try:
        f1 = "camera01_file1.mp4"
        f2 = "camera01_file2.mp4"

        db.add_file_task(f1, 0, "20261110", "20261110100000", "20261110100500", 300.0)
        db.add_file_task(f2, 0, "20261110", "20261110100500", "20261110101000", 300.0)

        db.set_analysis_result(f1, "ANALYZED", [
            {"start_time": 0.0, "end_time": 15.0, "state": "DYNAMIC"},
            {"start_time": 15.0, "end_time": 30.0, "state": "STATIC"},
        ])
        db.set_analysis_result(f2, "ANALYZED", [
            {"start_time": 0.0, "end_time": 20.0, "state": "DYNAMIC"},
        ])

        total_segs = db.conn.execute("SELECT count(*) FROM segments").fetchone()[0]
        assert total_segs == 3

        # Delete only f1
        with db._lock:
            db.conn.execute("DELETE FROM file_tasks WHERE filepath=?", (f1,))
            db.conn.commit()

        # f1 segments are gone, f2 segments remain
        assert len(db.get_segments_for_file(f1)) == 0
        f2_segs = db.get_segments_for_file(f2)
        assert len(f2_segs) == 1
        assert f2_segs[0]["duration"] == 20.0

        total_segs_remaining = db.conn.execute("SELECT count(*) FROM segments").fetchone()[0]
        assert total_segs_remaining == 1
    finally:
        db.close()


def test_foreign_key_persistence_across_reconnect(tmp_path):
    """Verify foreign key enforcement and cascade delete persist across database reconnection."""
    db_path = tmp_path / "test_reconnect.db"

    # Step 1: Create database, insert parent and child
    db1 = VlogDatabase(db_path=db_path)
    try:
        fp = "clip_persistent.mp4"
        db1.add_file_task(fp, 0, "20261110", "20261110100000", "20261110100500", 300.0)
        db1.set_analysis_result(fp, "ANALYZED", [{"start_time": 0.0, "end_time": 10.0, "state": "DYNAMIC"}])
        assert len(db1.get_segments_for_file(fp)) == 1
    finally:
        db1.close()

    # Step 2: Reopen database and verify PRAGMA foreign_keys is 1 and cascade delete works
    db2 = VlogDatabase(db_path=db_path)
    try:
        assert db2.conn.execute("PRAGMA foreign_keys;").fetchone()[0] == 1

        # Foreign key constraint violation still rejected
        with pytest.raises(sqlite3.IntegrityError):
            db2.conn.execute(
                "INSERT INTO segments (file_id, start_time, end_time, state) VALUES (8888, 0, 5, 'STATIC')"
            )

        # Deleting parent cascades to segments
        with db2._lock:
            db2.conn.execute("DELETE FROM file_tasks WHERE filepath=?", (fp,))
            db2.conn.commit()

        assert db2.conn.execute("SELECT count(*) FROM segments").fetchone()[0] == 0
    finally:
        db2.close()
