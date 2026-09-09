"""Export reproducible source-time audit windows, including prescreen negatives."""
import argparse
import csv
import random
import sqlite3
from pathlib import Path


def sample_windows(db_path, count=100, window_seconds=10.0, seed=42):
    if count < 1 or window_seconds <= 0:
        raise ValueError("Positive count and window length are required")
    with sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT filepath,date,cam_index,prescreen_status,file_duration FROM file_tasks WHERE file_duration>0 ORDER BY filepath").fetchall()
    groups = {}
    for row in rows:
        groups.setdefault((row["date"],row["cam_index"],row["prescreen_status"]), []).append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    samples = []
    for i in range(count if keys else 0):
        row = rng.choice(groups[keys[i % len(keys)]])
        duration = row["file_duration"]
        start = rng.uniform(0, max(0, duration-window_seconds))
        samples.append({"filepath":row["filepath"],"date":row["date"],"cam_index":row["cam_index"],
                        "prescreen_status":row["prescreen_status"],"start_seconds":round(start,3),
                        "end_seconds":round(min(duration,start+window_seconds),3),
                        "human_visual_event":"","human_audio_event":"","notes":""})
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=Path("data/vlog.db"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--window-seconds", type=float, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = sample_windows(args.db,args.count,args.window_seconds,args.seed)
    if not rows:
        parser.error("No source windows available")
    with args.output.open("x",encoding="utf-8-sig",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} source windows; labels intentionally await human review")


if __name__ == "__main__":
    main()
