# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

HomeVlog — 家庭监控素材智能浓缩系统。从 NAS 上的 H.265 监控录像中自动检测运动段，裁掉静止帧，输出精简的每日 vlog。

## Commands

```bash
# Full pipeline: scan -> prescreen -> analyze -> render
uv run python main.py

# Scan only (no processing)
uv run python main.py --scan

# Process specific date/cam, skip render
uv run python main.py --date 20260428 --cam 0 --no-render

# Single date+cam full pipeline
uv run python main.py --date 20260428 --cam 0

# Run tests
uv run pytest

# Run single test file
uv run pytest tests/test_segment.py
```

## Architecture: 3-Pass Pipeline

```
Pass 1 (prescreen):   QSV×4 parallel, sample N frames per file at 320×180,
                       single-process multi-seek (31 QSV inputs per ffmpeg call, 3x faster)
                       frame-diff classify -> STATIC (skip) or SUSPICIOUS
                       early-stop: first 10 pairs all < threshold*0.5 -> STATIC immediately

Pass 1.5 (detector):  NVDEC + QSV dual-GPU decode at 640×360 gray @5fps,
                       numpy frame-diff -> IQR adaptive threshold -> median filter
                       -> motion labels -> segment index (DYNAMIC/STATIC intervals)

Pass 2 (renderer):    NVDEC/QSV HW decode + GPU scale + HW encode, batched (≤15 files/batch)
                       work-stealing: NV/QSV workers independently pull from batch queue
                       (NV ~2.3x faster -> naturally takes more batches, no idle waiting)
                       DYNAMIC segments at original speed, STATIC segments fast-forward
                       (setpts=PTS/SPEED), concat via filter_complex_script
```

### Key modules

| File | Role |
|------|------|
| `pipeline.py` | Orchestrates Pass1->1.5->2 for each date-cam group. Crash recovery on startup. Robust try/except around render stage ensures DB always updated. |
| `scanner.py` | Scans `input_dir` for `{cam}_{YYYYMMDDHHMMSS}_{YYYYMMDDHHMMSS}.mp4` files, freeze detection (mtime+size), writes `file_tasks` |
| `prescreen.py` | Pass1: QSV-decoded frame sampling, numpy diff, classify STATIC/SUSPICIOUS |
| `detector.py` | Pass1.5: `MotionDetector` class, per-frame motion labels with median filter + IQR threshold. Dual-GPU worker split by duration interleave. |
| `segment.py` | `Segment` dataclass, build from frame labels, merge same-state, filter short segments (iterative absorption algorithm) |
| `timeline.py` | Builds `TimelineSegment[]` from DB, generates FFmpeg `concat` filter_complex string |
| `renderer.py` | Executes FFmpeg render. Batch NVDEC/QSV HW decode + NVENC/QSV encode. Parallel dual-GPU batches for multi-batch timelines. PerfRecord on all render paths. |
| `database.py` | SQLite WAL, `file_tasks` (per-file Pass1/1.5 status) + `render_tasks` (per date-cam Pass2 status). Thread-safe with `threading.Lock`. |
| `ffmpeg.py` | FFmpeg/ffprobe subprocess wrapper, QSV decode arg builder, `FFmpegResult` dataclass |
| `scheduler.py` | `choose_encoder(cam_index)`: cam0→"nv", else→"qsv". 11 lines, no state. |
| `utils.py` | Config loader (cached singleton), logging setup, `ts_to_unix()`, `parse_res()` |
| `monitor.py` | CPU/GPU sampling via psutil+pynvml, stage timing context manager, `PerfCollector` with `PerfRecord` |

### Data flow

`config/settings.yaml` -> `load_config()` (cached singleton) -> all modules

SQLite DB at `data/vlog.db` tracks all state. Pipeline is idempotent: re-run skips COMPLETED date-cam groups.

### Crash recovery

- On startup, `_recover_interrupted()` resets any RENDERING tasks to PENDING
- Prescreen/analysis are naturally idempotent (process only PENDING rows)
- `.completed_{date}_cam{index}` marker in output dir signals completion
- Render stage wrapped in try/except: DB always updated to COMPLETED or FAILED, even on exception

## Critical FFmpeg Patterns

- **Batch HW decode**: ≤15 files per batch (NVDEC VRAM limit on 8GB GPU). Per-batch `-hwaccel cuda` or `-hwaccel qsv` per input.
- **Parallel dual-GPU batches**: NVDEC+NVENC batch pairs with QSV decode+encode batch. Hardware-isolated (RTX 3060Ti vs UHD 770).
- **Shared CUDA device**: Use `-init_hw_device cuda=gpu:0` + `-filter_hw_device gpu` for NVENC encoding.
- **HW scale passthrough**: NVDEC CUDA frames → `scale_cuda` (no hwupload). QSV frames → `scale_qsv`. Both → `hwdownload,format=nv12` before concat.
- **CPU decode fallback**: Software decode + CPU/GPU scale for non-HW path. Uses `hwupload_cuda,scale_cuda=...` or `scale=...`.
- **filter_complex_script**: Write filter graph to temp file, pass via `-filter_complex_script` to avoid Windows 32K cmd-line limit.
- **concat pad order**: Strict `[v0][a0][v1][a1]...` interleaved, NOT `[v0][v1]...[a0][a1]...`.
- **GPU path limitations**: `fps` filter incompatible with hwdownload frames. Static segments use `setpts=PTS/SPEED` (fast-forward) instead of keyframe slideshow on GPU path.

## Hardware Setup

- RTX 3060Ti (8GB VRAM): NVDEC (H.265 decode, ~36 concurrent limit) + NVENC (H.265 encode, 3 session limit)
- UHD 770 (i5-12600K): QSV decode + QSV encode (system memory, no VRAM limit)
- Pass1: QSV×4 parallel prescreen (multi-seek: 31 inputs/ffmpeg call)
- Pass1.5: 40% NVDEC + 60% QSV split (NVDEC ~42% slower per-file, tuned from 43:57)
- Pass2: batch render (≤15 files/batch), work-stealing NV+QSV dual-worker queue

### Pass2 render performance (98 files, 389 segs, ~46min footage)

| Metric | Value |
|--------|-------|
| Total render time | ~56 min (3352s) pre-optimization |
| Batch count | 7 (15+15+15+15+15+15+8 files) |
| Scheduling | work-stealing (NV/QSV independent workers) |
| NVDEC batch avg | ~275s/batch |
| QSV batch avg | ~644s/batch |
| Output | 1757 MB, 1080p@20fps HEVC |
| GPU peak mem | 8052 MB |

## Config Tuning

Only `paths.input_dir` is required. Detection-critical params marked with "调优" in `config/settings.yaml`:

- `prescreen_diff_threshold`: 5 for IR, higher for daytime
- `motion_sensitivity`: IQR multiplier (2.0 ≈ 3.4σ)
- `median_filter_window`: 7 suppresses H.265 GOP boundary spikes (critical for IR)
- `min_motion_duration`: 1.0s for IR (compressed by median filter)

## Codebase Cleanup (2026-04-30)

Dead code and duplication removed across all modules:

- **Shared utilities**: `_ts_to_unix` (3 copies) and `_parse_res` (3 copies) consolidated into `utils.py:ts_to_unix()` and `utils.py:parse_res()`
- **scheduler.py**: Gutted from 68→11 lines. Removed `GPUPool` (never wired in), `choose_decode_gpu()`, `_pool` state
- **ffmpeg.py**: Removed `retry_ffmpeg()` (SMB retry, never called), `build_nvdec_decode_args()` (unused), `is_unc_error()` + patterns
- **renderer.py**: Removed `_split_render()` + `_find_split_index()` (superseded by batch render), `_parse_res()`, error log file writing (`_err_` temp files). Added PerfRecord to batch render path
- **database.py**: Removed `mark_file_failed()` (retry logic never integrated), `get_incomplete_render_tasks()`, `_FAIL_COLUMN_QUERIES`
- **detector.py**: Removed `analyze_file()` (standalone entry, unused), `_ts_to_unix()`, `_parse_res()`
- **monitor.py**: Removed `add_raw()`, `records` property, `get_gpu_names()`, `StageStats.__post_init__()`
- **segment.py**: `merge_cross_file` now delegates to `_merge_same_state` (was duplicate logic)
- **prescreen.py**: Removed `_parse_resolution()` (replaced by `parse_res`)
- **config/settings.yaml**: Removed `split_render`, `split_min_static_dur`, `smb_retry_wait`, `smb_retry_max`

## Input File Naming

```
{cam_index:02d}_{YYYYMMDDHHMMSS}_{YYYYMMDDHHMMSS}.mp4
Example: 00_20260428080000_20260428084600.mp4
```

Cross-midnight files belong to the start date. Files sorted by `file_start_time`.
