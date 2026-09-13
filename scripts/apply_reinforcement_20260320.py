import sys
import sqlite3
import json
from pathlib import Path

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os
sys.path.insert(0, os.getcwd())

from src.segment import (
    Segment,
    segments_from_json,
    segments_to_json,
    resolve_presence_segments,
    _merge_same_state,
    split_segments_at_file_boundaries,
)
from src.utils import ts_to_unix

db_path = Path("data/vlog.db")
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

print("=== 开始对 20260320 应用全局跨文件系统加固 (Presence & Micro-Motion) ===")

files = conn.execute("SELECT * FROM file_tasks WHERE date='20260320' AND analysis_status='ANALYZED'").fetchall()
print(f"找到 {len(files)} 个已分析文件")

day_start = ts_to_unix("20260320000000")
files_info = []
all_segs = []

for f in files:
    fid = f["id"]
    fp = f["filepath"]
    raw_json = f["analysis_segments"]
    if not raw_json:
        continue

    file_start_unix = ts_to_unix(f["file_start_time"])
    file_end_unix = ts_to_unix(f["file_end_time"])
    day_offset = max(file_start_unix - day_start, 0.0)
    file_dur = float(f["file_duration"] or (file_end_unix - file_start_unix))
    
    files_info.append({
        "filepath": fp,
        "start_offset": day_offset,
        "end_offset": day_offset + file_dur,
        "duration": file_dur,
        "file_id": fid,
    })

    segs = segments_from_json(raw_json)
    for s in segs:
        if s.state == "STATIC" and s.max_energy >= 2.5:
            s.state = "MICRO_MOTION"
            s.needs_review = True
            s.review_reason = f"YOLO_NEGATIVE_ENERGY_HIGH: 显著运动未识别目标(energy={s.max_energy:.1f})"
        s.source_file = fp
        s.file_start_offset = day_offset
        all_segs.append(s)

all_segs.sort(key=lambda s: s.start_time)
merged_segs = _merge_same_state(all_segs, gap_tolerance=1.5)
resolved = resolve_presence_segments(merged_segs, max_presence_gap_s=180.0, person_conf_threshold=0.25)
split_segs = split_segments_at_file_boundaries(resolved, files_info)

# Group split segments by file
file_to_segs = {}
for s in split_segs:
    fp = s.source_file
    if fp not in file_to_segs:
        file_to_segs[fp] = []
    file_to_segs[fp].append(s)

# Update database
updated_files = 0
total_presence = sum(1 for s in split_segs if s.state == "PRESENCE")
total_micro = sum(1 for s in split_segs if s.state == "MICRO_MOTION")

for f in files:
    fid = f["id"]
    fp = f["filepath"]
    segs = file_to_segs.get(fp, [])
    if not segs:
        continue

    new_json = segments_to_json(segs)
    conn.execute("UPDATE file_tasks SET analysis_segments=? WHERE id=?", (new_json, fid))

    conn.execute("DELETE FROM segments WHERE file_id=?", (fid,))
    for s in segs:
        conn.execute("""
            INSERT INTO segments (
                file_id, filepath, cam_index, date, start_time, end_time, duration,
                state, max_energy, avg_confidence, file_start_offset,
                needs_review, review_reason, manual_label, review_notes, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            fid, fp, 0, "20260320", s.start_time, s.end_time, s.duration,
            s.state, s.max_energy, s.avg_confidence, s.file_start_offset,
            1 if s.needs_review else 0, s.review_reason, None, None, None
        ))
    updated_files += 1

conn.commit()
conn.close()

print(f"全局跨文件加固升级完成！更新文件数: {updated_files}, 总 PRESENCE 切片: {total_presence}, 总 MICRO_MOTION 切片: {total_micro}")
