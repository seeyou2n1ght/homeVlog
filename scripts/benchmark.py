"""HomeVlog 性能基准测试套件 (Benchmark Suite)

用于在代码重构与性能优化过程中量化评估核心算子、预筛选、YOLO 推理与时间轴构建的延迟、吞吐率与内存分配。

使用方法:
  uv run python scripts/benchmark.py --all
  uv run python scripts/benchmark.py --operator
  uv run python scripts/benchmark.py --prescreen
  uv run python scripts/benchmark.py --yolo
  uv run python scripts/benchmark.py --timeline
"""

import argparse
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


def benchmark_operator(num_iterations: int = 2000, width: int = 416, height: int = 234):
    """基准测试 1: 运动检测核心帧差算子 (cv2.norm L1 vs np.sum absdiff)"""
    import cv2

    print(f"\n=======================================================")
    print(f" 基准测试 1: 运动检测核心算子评估 (ROI: {width}x{height}, {num_iterations} 轮)")
    print(f"=======================================================")

    # Prepare random uint8 frame pairs
    np.random.seed(42)
    frame_a = np.random.randint(0, 256, (height, width), dtype=np.uint8)
    frame_b = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    # 1. Baseline: np.sum(cv2.absdiff())
    tracemalloc.start()
    t0 = time.perf_counter()
    baseline_sum = 0.0
    for _ in range(num_iterations):
        diff = cv2.absdiff(frame_a, frame_b)
        baseline_sum += float(np.sum(diff))
    baseline_time = time.perf_counter() - t0
    current_mem, peak_mem_baseline = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    baseline_fps = num_iterations / baseline_time
    baseline_us = (baseline_time / num_iterations) * 1e6

    # 2. Optimized: cv2.norm(..., cv2.NORM_L1)
    tracemalloc.start()
    t0 = time.perf_counter()
    opt_sum = 0.0
    for _ in range(num_iterations):
        opt_sum += cv2.norm(frame_a, frame_b, cv2.NORM_L1)
    opt_time = time.perf_counter() - t0
    current_mem, peak_mem_opt = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    opt_fps = num_iterations / opt_time
    opt_us = (opt_time / num_iterations) * 1e6

    speedup = baseline_time / opt_time
    mem_reduction = ((peak_mem_baseline - peak_mem_opt) / max(peak_mem_baseline, 1)) * 100

    print(f"{'算子实现':<30} | {'单帧耗时':<12} | {'吞吐率 (FPS)':<16} | {'堆内存峰值':<12}")
    print(f"{'-'*30}-|-{'-'*12}-|-{'-'*16}-|-{'-'*12}")
    print(f"{'Baseline: np.sum(absdiff)':<30} | {baseline_us:>8.2f} us | {baseline_fps:>12.1f} fps | {peak_mem_baseline/1024:>8.1f} KB")
    print(f"{'Optimized: cv2.norm(NORM_L1)':<30} | {opt_us:>8.2f} us | {opt_fps:>12.1f} fps | {peak_mem_opt/1024:>8.1f} KB")
    print(f"{'-'*76}")
    print(f"  --> 性能加速比: {speedup:.2f}x, 堆内存申请消除率: {mem_reduction:.1f}%\n")


def benchmark_prescreen(limit_files: int = 10):
    """基准测试 2: 预筛选关键帧 Seek 吞吐率与判定延迟"""
    from src.database import VlogDatabase
    from src.prescreen import prescreen_file
    from src.utils import load_config

    print(f"\n=======================================================")
    print(f" 基准测试 2: 快速预筛选 (Keyframe Seek) 实测采样 (最多 {limit_files} 文件)")
    print(f"=======================================================")

    db = VlogDatabase()
    try:
        rows = db.conn.execute(
            """SELECT filepath, file_duration FROM file_tasks 
               WHERE file_duration > 10.0 
               ORDER BY id DESC LIMIT ?""",
            (limit_files,),
        ).fetchall()
    finally:
        db.close()

    if not rows:
        print("未找到有效素材文件进行预筛选测试，请先执行 `python main.py --scan` 扫描数据源。")
        return

    config = load_config()
    latencies = []
    statuses = []

    for r in rows:
        fp = r["filepath"]
        dur = r["file_duration"] or 300.0
        t0 = time.perf_counter()
        try:
            res = prescreen_file(fp, dur, config, gpu="qsv")
            elapsed = time.perf_counter() - t0
            latencies.append(elapsed)
            statuses.append(res.get("status", "UNKNOWN"))
        except Exception as e:
            print(f"  [Error] {Path(fp).name}: {e}")

    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        p95 = sorted(latencies)[int(len(latencies) * 0.95)]
        fps_files = len(latencies) / sum(latencies)

        print(f"测试切片数: {len(latencies)} 文件")
        print(f"平均处理延迟: {avg_lat*1000:.1f} ms / 文件 (P50: {p50*1000:.1f} ms, P95: {p95*1000:.1f} ms)")
        print(f"预筛选吞吐率: {fps_files:.2f} files/s (即单日 288 切片仅需 {288 / fps_files:.1f} 秒)")
        print(f"状态判定分布: STATIC={statuses.count('STATIC')}, SUSPICIOUS={statuses.count('SUSPICIOUS')}, FAILED={statuses.count('FAILED')}\n")


def benchmark_yolo():
    """基准测试 3: YOLO 推理动态批处理 (Dynamic Batching) 延迟与显存"""
    import torch
    from src.utils import load_config

    print(f"\n=======================================================")
    print(f" 基准测试 3: YOLO 目标检测动态批处理评估 (yolo11n.pt)")
    print(f"=======================================================")

    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        print("当前环境未检测到 CUDA，跳过 GPU 批处理评估。")
        return

    from ultralytics import YOLO

    model_path = PROJECT_ROOT / "models" / "yolo11n.pt"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "yolo11n.pt"
    if not model_path.exists():
        print(f"模型文件 {model_path} 不存在，跳过 YOLO 基准测试。")
        return


    model = YOLO(str(model_path))
    model.to(device_str)

    batch_sizes = [1, 4, 16, 32]
    h, w = 234, 416

    print(f"{'Batch Size':<12} | {'总推理耗时':<14} | {'单帧摊销延迟':<14} | {'等效推断吞吐率':<16} | {'显存占用':<12}")
    print(f"{'-'*12}-|-{'-'*14}-|-{'-'*14}-|-{'-'*16}-|-{'-'*12}")

    with torch.inference_mode():
        for bs in batch_sizes:
            dummy_frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in range(bs)]
            
            # Warm up
            model(dummy_frames, verbose=False, device=device_str)
            torch.cuda.synchronize()

            # Benchmark 5 runs
            t0 = time.perf_counter()
            for _ in range(5):
                model(dummy_frames, verbose=False, device=device_str)
            torch.cuda.synchronize()
            total_time = (time.perf_counter() - t0) / 5

            ms_per_frame = (total_time / bs) * 1000
            fps = bs / total_time
            vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

            print(f"{bs:<12} | {total_time*1000:>10.2f} ms | {ms_per_frame:>10.2f} ms | {fps:>12.1f} FPS | {vram_mb:>8.1f} MB")

    print(f"{'-'*76}\n")


def benchmark_timeline():
    """基准测试 4: 时间轴跨文件边界分割与宏观折叠算力耗时"""
    from src.database import VlogDatabase
    from src.timeline import build_timeline

    print(f"\n=======================================================")
    print(f" 基准测试 4: 全天时间轴物理对齐与宏观折叠算法构建性能")
    print(f"=======================================================")

    db = VlogDatabase()
    try:
        cols = [c[1] for c in db.conn.execute("PRAGMA table_info(file_tasks)").fetchall()]
        cam_col = "cam_index" if "cam_index" in cols else "cam_id"
        row = db.conn.execute(
            f"""SELECT date, {cam_col} as cam_val, count(*) as cnt FROM file_tasks 
               GROUP BY date, {cam_col} HAVING cnt > 10 
               ORDER BY cnt DESC LIMIT 1"""
        ).fetchone()
        if not row:
            print("数据库中无包含充足切片的日期，跳过时间轴基准测试。")
            return

        date, cam_val = row["date"], row["cam_val"]
        rows = db.conn.execute(
            f"SELECT * FROM file_tasks WHERE date=? AND {cam_col}=? ORDER BY file_start_time",
            (date, cam_val)
        ).fetchall()
        rows = [dict(r) for r in rows]

        t0 = time.perf_counter()
        timeline = build_timeline(db, date, cam_val)
        build_time_ms = (time.perf_counter() - t0) * 1000

        # Verify physical file boundary invariants
        file_intervals = {}
        for r in rows:
            dur = max(0.0, r.get("file_duration") or 300.0)
            file_intervals[r["filepath"]] = dur

        violations = 0
        for seg in timeline:
            max_allowed = file_intervals.get(seg.filepath, 300.0)
            if seg.end_in_file > max_allowed + 0.1 or seg.start_in_file < 0:
                violations += 1

        print(f"测试日期与摄像头: {date} cam{cam_val} ({len(rows)} 个物理文件)")
        print(f"时间轴构建耗时: {build_time_ms:.2f} ms")
        print(f"输出时间轴片段数: {len(timeline)} 个 TimelineSegment")
        print(f"物理边界越界检测 (Boundary Invariant Check): {violations} 次违规 (目标值: 0)\n")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="HomeVlog Performance Benchmark Suite")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--operator", action="store_true", help="Benchmark OpenCV L1 norm operator")
    parser.add_argument("--prescreen", action="store_true", help="Benchmark Prescreen Keyframe Seek throughput")
    parser.add_argument("--yolo", action="store_true", help="Benchmark YOLO dynamic batching")
    parser.add_argument("--timeline", action="store_true", help="Benchmark Timeline physical boundary splitting")
    args = parser.parse_args()

    if not any([args.all, args.operator, args.prescreen, args.yolo, args.timeline]):
        parser.print_help()
        return

    if args.all or args.operator:
        benchmark_operator()

    if args.all or args.prescreen:
        benchmark_prescreen()

    if args.all or args.yolo:
        benchmark_yolo()

    if args.all or args.timeline:
        benchmark_timeline()


if __name__ == "__main__":
    main()
