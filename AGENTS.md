# AGENTS.md

This repository contains the HomeVlog video analysis and rendering pipeline.

This file defines repository-level constraints and routes coding agents to the project state, architecture, testing, and decision documentation. Do not treat it as a complete description of the implementation.

---

## 1. Project Goals

HomeVlog processes long-duration surveillance / home video into condensed Vlogs.

The system is optimized for:

- long-running batch processing;
- NAS and imperfect source media;
- hardware-accelerated decoding / analysis / rendering;
- resumable execution;
- deterministic timeline generation;
- graceful degradation instead of whole-pipeline failure.

When making architectural trade-offs, prioritize:

1. correctness of the final timeline;
2. recoverability and resumability;
3. bounded hardware resource usage;
4. throughput;
5. implementation simplicity.

---

## 2. Core Architecture Invariants

These constraints apply across the repository unless an explicit architecture decision replaces them.

### Single Decode

Decode source media once whenever practical and allow multiple consumers to reuse the decoded data.

Do not introduce independent full decode pipelines for YOLO, motion analysis, or other consumers without an explicit architectural justification.

### In-Memory Intermediate Frames

Candidate frames should remain in memory using the existing compressed-frame pipeline.

Avoid introducing large numbers of temporary image files or unnecessary disk I/O in intermediate processing stages.

### Fault-Tolerant Processing

NAS instability, damaged media, seek failures, and partial processing failures are expected operating conditions.

Prefer:

```text
failure
→ record diagnostics
→ safely skip or degrade
→ continue pipeline
```

over terminating an entire Vlog job when the failed unit can be isolated.

Do not silently ignore failures that can affect timeline correctness or final-output integrity.

### Deterministic Timeline

Rendering, subtitles, detection results, and wall-clock mapping must derive from a consistent timeline model.

`timeline.compute_display_plans()` is the authoritative source for display-duration planning.

Changes affecting timestamps, duration calculation, segment transformation, or subtitle mapping require review against the timeline contract documented in `docs/ARCHITECTURE.md`.

### Controlled Hardware Concurrency

GPU / QSV / FFmpeg concurrency must use the existing scheduler abstractions.

Do not bypass:

- `src.scheduler.get_nv_semaphore()`
- `src.scheduler.get_qsv_semaphore()`
- `WorkStealingManager`

or introduce independent hardware-concurrency control without updating the architecture decision.

### Managed FFmpeg Lifecycle

Long-running FFmpeg subprocesses must participate in the repository's process lifecycle management.

Processes created through `subprocess.Popen` must not become unmanaged orphan processes.

Use the existing `FFmpegProcessRegistry` and guarantee cleanup during exceptions and Ctrl+C shutdown.

### Resumable Rendering

Rendered batches are transactional units.

Incomplete output must not be indistinguishable from valid output.

Preserve the existing pattern:

```text
temporary batch
→ process completion
→ validate exit status / output
→ atomic promotion
→ final batch
```

Do not remove valid completed batches during normal cleanup unless explicitly performing a full reset.

---

## 3. Repository-Specific Constraints

### Metadata Loading

`src/scanner.py` must remain lightweight.

Do not introduce blocking media probing such as per-file `ffprobe` calls during scanning.

Media metadata should be lazily resolved during analysis and persisted through the existing metadata/database pipeline.

### Configuration Integrity

Every user-facing key in `config/settings.yaml` must correspond to real runtime behavior.

When changing configuration:

- new keys require implementation wiring;
- removed behavior requires removal of obsolete keys;
- preset / enum values require matching runtime branches.

Avoid configuration that exists only in YAML without executable semantics.

### Human Feedback Priority

For timeline reconstruction, explicit human labels override algorithmic status.

In particular, logic involving `human_label` and `algo_status` must preserve the repository's human-feedback precedence rules.

### Performance Metrics

Do not mix aggregate worker CPU/GPU time with elapsed wall-clock time.

Performance tooling and reports must distinguish at least:

- Worker Time
- Wall-clock Time

Performance baselines belong in `docs/PROGRESS.md` and `docs/ARCHITECTURE.md`.

---

## 4. Module-Level Invariants

Detailed implementation invariants are documented outside this file.

Before modifying the corresponding subsystem, read its relevant architecture section.

Important examples include:

- semaphore ownership and exactly-once release;
- FFmpeg process registration and deregistration;
- grayscale analysis pipe format;
- render-worker sentinel semantics;
- render completion / batch reconciliation;
- static-segment fast paths;
- timeline endpoint semantics;
- effective-FPS-based YOLO sampling;
- frame archival fallback behavior;
- active-learning dataset composition.

Primary reference:

```text
docs/ARCHITECTURE.md
```

Do not duplicate detailed implementation rules in this file unless they apply broadly across multiple subsystems.

---

## 5. Project Documentation

The repository documentation is authoritative for persistent project knowledge.

### Architecture

```text
docs/ARCHITECTURE.md
```

Contains:

- system architecture;
- module boundaries;
- pipeline and data flow;
- concurrency model;
- timeline contract;
- critical implementation invariants;
- architecture decisions.

Read the relevant section before making structural changes.

### Project State

```text
docs/PROGRESS.md
```

Contains the current project state:

- current focus;
- work in progress;
- known issues;
- recently completed work;
- next high-priority tasks.

For non-trivial work, read this file at task start.

### Decisions

```text
docs/DECISIONS.md
```

Contains architectural or engineering decisions whose rationale should survive across sessions.

Record a decision when a change introduces or materially modifies:

- architectural boundaries;
- data contracts;
- concurrency semantics;
- persistence semantics;
- pipeline behavior;
- hardware scheduling strategy;
- recovery behavior.

Do not record routine implementation details as decisions.

### Testing

```text
docs/TESTING.md
```

Contains:

- test strategy;
- test organization;
- environment requirements;
- verification commands;
- quality and accuracy acceptance baselines;
- known testing limitations.

Hardware baselines and production telemetry belong in `docs/ARCHITECTURE.md` and `docs/PROGRESS.md`.

---

## 6. Task Start Protocol

For non-trivial repository work:

1. Read this `AGENTS.md`.
2. Read `docs/PROGRESS.md`.
3. Identify the affected subsystem.
4. Read the relevant section of `docs/ARCHITECTURE.md`.
5. Read relevant entries in `docs/DECISIONS.md` when the task touches an existing architectural decision.
6. Inspect the smallest necessary code surface.
7. Identify the appropriate verification path before making changes.

Do not start by reading the entire repository.

Expand context only when dependencies or observed behavior require it.

---

## 7. Change Discipline

Work in independently understandable and verifiable units.

Avoid:

- unrelated refactors during feature or bug-fix work;
- opportunistic cleanup outside the affected subsystem;
- duplicating existing abstractions;
- bypassing scheduler, timeline, configuration, or lifecycle abstractions for local convenience.

Before changing repository structure or introducing a new abstraction, verify that the same responsibility is not already implemented elsewhere.

Preserve unrelated user modifications in the working tree.

---

## 8. Testing and Verification

The repository uses `uv` for Python environment and dependency management.

Do not replace the configured CUDA-enabled PyTorch dependency with a CPU-only build unless explicitly required.

### Standard test suite

```powershell
uv run python -m pytest tests/
```

### Syntax / import compilation check

```powershell
uv run python -m compileall main.py src scripts
```

Run the smallest relevant verification first during development.

Before considering a substantial change complete, run all relevant repository-level verification required by `docs/TESTING.md`.

A change is not considered correct merely because the code compiles.

For changes affecting concurrency, FFmpeg lifecycle, timeline behavior, resume behavior, or hardware scheduling, include targeted behavioral verification.

---

## 9. Test Organization

Keep tests organized around stable business or subsystem boundaries rather than individual bugs.

Prefer extending an existing suite for:

- database / scanning;
- rendering / FFmpeg;
- hardware scheduling;
- review / feedback services;
- timeline behavior;
- multimodal / detection logic.

Shared video-generation helpers, fixtures, and mock factories belong in `tests/conftest.py` or an established shared testing utility.

Do not create a new test file solely for one small regression if an appropriate existing suite exists.

---

## 10. Workspace Safety

Before modifying or reverting a file with existing changes, inspect the current diff.

Do not discard, overwrite, reset, stash, or revert unrelated user work.

If repository state is ambiguous, preserve the current state and diagnose before performing destructive Git operations.

Local agent transcripts or tool history may be used as recovery evidence, but they are not a substitute for normal Git-tracked project state.

---

## 11. Task Completion Protocol

Before declaring a non-trivial task complete:

1. Run relevant verification.
2. Confirm no known regression or incomplete intermediate state remains.
3. Review whether `docs/PROGRESS.md` must be updated.
4. Review whether the change modified architecture or an important invariant.
5. If so, update `docs/ARCHITECTURE.md` and/or `docs/DECISIONS.md`.
6. Review whether the task exposed a reusable project-level rule or failure pattern.
7. Update project guidance only when that knowledge is expected to matter in future tasks.

Do not update documentation merely to record that a routine task occurred.

Persist information that future agents would otherwise need to rediscover.

---

## 12. Documentation Discipline

Documentation should be concise, factual, and maintained as persistent engineering context.

Avoid:

- duplicate descriptions of the same architecture;
- temporary investigation reports in `docs/`;
- narrative development diaries;
- stale configuration documentation;
- implementation details duplicated across multiple documents.

Use:

```text
AGENTS.md
```

for repository-wide execution rules and documentation routing.

Use:

```text
docs/ARCHITECTURE.md
```

for architecture and subsystem invariants.

Use:

```text
docs/PROGRESS.md
```

for current project state.

Use:

```text
docs/DECISIONS.md
```

for durable engineering decisions.

Use:

```text
docs/TESTING.md
```

for verification strategy and quality baselines.