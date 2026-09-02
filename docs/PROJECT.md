# Project: HomeVlog Dual-GPU Pipeline Refactoring & Optimization

## Architecture
HomeVlog is a high-throughput, zero-idle, dual-GPU intelligent video condensation system combining an Intel UHD 770 iGPU (dual Gen12 VDBox QSV decoding) and an NVIDIA RTX 3060Ti dGPU (Tensor Core YOLO batch inference & NVENC hardware encoding).

```
[84 4K H.265 Files] -> scanner.py -> SQLite WAL DB
                            │
               ┌────────────┴────────────┐
               ▼                         ▼
   Intel UHD 770 (QSV)           NVIDIA RTX 3060Ti
   • Prescreen (8 workers)       • Cooperative Analysis Burst (when idle)
   • Primary Analysis Decode     • YOLOv11 Tensor Core Batch Inference
   • Static Batch Rendering      • Dynamic Batch Rendering (NVENC)
               │                         │
               └────────────┬────────────┘
                            ▼
           Timeline Assembly & Speed Ramping (PTS)
                            ▼
           Parallel Batch Render Array (1 NV + 1 QSV)
                            ▼
     Output DailyVlog_YYYYMMDD_cam0.mp4 (<=30min Wall-Clock)
```

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Dual-GPU Workload Decoupling | UHD 770 primary QSV decode + RTX 3060Ti tensor/NVENC render | M1 | ORIGINAL_REQUEST §R1 |
| 2 | Adaptive Work-Stealing | Load-aware device leasing with instant render preemption yield | M1 | ORIGINAL_REQUEST §R1 |
| 3 | Hardware Semaphore Isolation | Safe concurrency controls for NV (3), QSV (8), and Disk (8) | M1 | AGENTS.md |
| 4 | Single-Pass PyAV Multimodal Analysis | Synchronized video frame diff + audio energy VAD | M2 | AGENTS.md |
| 5 | Timeline Closure & Early-Term Sync | Fix audio timestamp sync and exact duration alignment | M2 | Explorer 3 Defect B |
| 6 | YOLO Multimodal Audio Exemption | DYNAMIC_AUDIO segments exempted from visual demotion | M2 | Codebase Survey |
| 7 | Dual-GPU Parallel Render Array | 1 NVENC + 1 QSV worker with complexity-based dispatch | M3 | ORIGINAL_REQUEST §R1, §R2 |
| 8 | VRAM Peak Protection (<=4.5GB) | `batch_max_files: 4`, zero-copy flow, eliminate bus congestion | M3 | ORIGINAL_REQUEST §R2 |
| 9 | Speed Ramping & Wall-Clock OSD | Smooth PTS curves (60x -> 1x), audio afade, timecode burn-in | M3 | Codebase Survey |
| 10| Test Suite Dataset Alignment | Fix 81->84 assertions in `test_tier4_acceptance.py` | M4 | Explorer 3 Defect A |
| 11| Dual-GPU Verification Gaps (GAP 01-06) | Concurrency, work-stealing, VRAM limit, and throughput tests | M4 | Explorer 3 Survey |
| 12| 24-Hour 84-Slice <= 30m Throughput | Benchmark simulation achieving >=42x real-time speed | M4 | ORIGINAL_REQUEST §R3 |
| 13| Full E2E All-Tier Verification | 100% green pass on all 2,388+ tests across Tiers 1-4 | M5 (Phase 1) | Acceptance Criteria |
| 14| Tier 5 Adversarial Coverage Hardening | White-box stress testing, race-condition and leak audit | M5 (Phase 2) | Project Pattern |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| 1 | M1: Work-Stealing & Concurrency | `src/utils.py`, `config/settings.yaml`, `src/pipeline.py` | none | PLANNED |
| 2 | M2: Multimodal Detector & Timeline Closure | `src/detector.py`, `src/yolo_verifier.py` | M1 | PLANNED |
| 3 | M3: Dual-GPU Rendering & VRAM Safety | `src/renderer.py`, `src/timeline.py` | M1, M2 | PLANNED |
| 4 | M4: E2E Test Suite & 24h Benchmark | `tests/test_tier4_acceptance.py`, `tests/test_benchmark_perf.py`, gap tests | M1, M2, M3 | PLANNED |
| 5 | M5: Final E2E Pass & Tier 5 Hardening | Full system verification, adversarial coverage audit, perf logs | M1, M2, M3, M4 | PLANNED |

## Interface Contracts
### `WorkStealingManager` (`src/utils.py`) ↔ `StreamingOrchestrator` (`src/pipeline.py`)
- `get_analysis_device(queue_size: int, is_render_active: bool) -> str`: Returns `"qsv"` or `"cuda"`.
- `lease_device(queue_size: int) -> ContextManager[str]`: Automatically acquires and releases NVDEC/QSV slots.
- `register_render_start()` / `register_render_end()`: Atomic flag updates to block NVDEC during render.

### `MotionDetector` (`src/detector.py`) ↔ `Timeline` (`src/timeline.py`)
- `analyze(video_path: Path, start_offset: float, file_duration: float) -> list[dict]`:
  - Returns frame records with `time_offset`, `state`, `energy`, `is_audio`.
  - Guaranteed closure: last record timestamp equals `start_offset + file_duration`.

### `Renderer` (`src/renderer.py`) ↔ `Timeline` (`src/timeline.py`)
- `partition_timeline_by_batches(segments, batch_max_files=4) -> list[TimelineBatch]`:
  - Batches partitioned with maximum 4 source files per batch to bound VRAM <= 4.5GB.
  - Workers dispatched: 1 NVENC worker + 1 QSV worker concurrently.

## Code Layout
- `main.py`: CLI entrypoint
- `config/settings.yaml`: Central configuration
- `src/scanner.py`: File discovery and task ingestion
- `src/prescreen.py`: Pass 1 fast I-frame prescreening
- `src/detector.py`: Pass 1.5 single-pass PyAV multimodal analysis
- `src/yolo_verifier.py`: Pass 1.8 Tensor Core YOLOv11 batch verification
- `src/segment.py`: Segment smoothing and cross-file merging
- `src/timeline.py`: Speed ramping PTS curves and FFmpeg filtergraph construction
- `src/renderer.py`: Pass 2 parallel batch rendering array
- `src/monitor.py`: System & GPU hardware telemetry collector
- `src/database.py`: SQLite WAL task and render state persistence
- `src/utils.py`: Hardware semaphores, work-stealing manager, config loading
- `tests/`: 2,388 automated test suite across 20 files
