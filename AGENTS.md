# AGENTS.md

This file defines repository-level execution rules and routes coding agents to the authoritative project documentation.

It is not a complete description of the implementation.

---

## 1. Project Goals

HomeVlog processes long-duration surveillance and home video into condensed daily Vlogs.

When making engineering trade-offs, prioritize:

1. final timeline correctness;
2. recoverability and resumability;
3. bounded hardware and I/O usage;
4. end-to-end pipeline makespan;
5. implementation simplicity.

The system is expected to tolerate NAS instability, damaged media, seek failures, interrupted execution, and heterogeneous hardware.

---

## 2. Core Architecture Invariants

These constraints apply across the repository unless explicitly replaced by a recorded architecture decision.

### Single-Pass Analysis

Avoid independent full decode pipelines for consumers that can reuse existing decoded data.

### In-Memory Intermediate Frames

Intermediate candidate frames should remain in memory whenever practical.

Do not introduce large numbers of temporary image files as stage-to-stage transport.

### Fault-Tolerant Processing

Isolated media or processing failures should normally:

```text
diagnose → record → safely degrade/skip → continue
```

Do not silently continue when timeline correctness, batch integrity, or final output integrity may be affected.

### Deterministic Timeline

Rendering, subtitles, source-time mapping, and display duration must derive from the shared timeline model.

`timeline.compute_display_plans()` is the authoritative display-plan entry point.

### Controlled Hardware Concurrency

Hardware resources must be managed through the existing scheduler abstractions.

Do not introduce independent concurrency control that bypasses the scheduler.

### Managed FFmpeg Lifecycle

Long-running FFmpeg subprocesses must participate in the repository process lifecycle and be cleanly terminable.

### Resumable Rendering

Rendered batches are transactional work units.

Incomplete output must never be indistinguishable from valid completed output.

---

## 3. Repository-Specific Constraints

### Metadata

`src/scanner.py` must remain lightweight.

Do not introduce blocking per-file media probing during directory scanning.

Media metadata should be resolved lazily through the existing analysis pipeline and persisted.

### Configuration

Every user-facing key in `config/settings.yaml` must have real runtime semantics.

Do not add configuration without wiring its behavior.

Do not leave obsolete or unused configuration keys behind.

### Human Feedback

Explicit human review must take precedence over algorithmic classification when rebuilding the final timeline.

### Performance

Optimize for end-to-end pipeline makespan rather than maximizing utilization of an individual worker or device.

Distinguish wall-clock time from aggregate worker time when evaluating performance.

---

## 4. Module-Level Invariants

Detailed subsystem contracts are documented in:

```text
docs/ARCHITECTURE.md
```

Examples include:

- semaphore ownership and exactly-once release;
- NVENC / NVDEC / QSV scheduling semantics;
- FFmpeg registration and cleanup;
- grayscale analysis-frame contract;
- render-worker termination semantics;
- atomic render batches and completion reconciliation;
- timeline closure and source/display-time mapping;
- adaptive analysis FPS;
- active-learning and frame-archival behavior.

Do not duplicate detailed implementation rules here unless they apply broadly across multiple subsystems.

---

## 5. Project Documentation

Repository documentation is authoritative for persistent project knowledge.

### Architecture

```text
docs/ARCHITECTURE.md
```

Contains the current:

- system architecture;
- module boundaries;
- pipeline and data flow;
- data contracts;
- concurrency model;
- timeline contract;
- subsystem invariants.

Read the relevant section before structural or cross-module changes.

### Project State

```text
docs/PROGRESS.md
```

Contains:

- current goal;
- work in progress;
- known issues;
- blockers;
- recently completed work;
- next high-priority actions.

Read it at the start of non-trivial work.

### Decisions

```text
docs/DECISIONS.md
```

Contains durable architectural and engineering decisions and their rationale.

Create or update an ADR when changing:

- architectural boundaries;
- cross-stage data contracts;
- concurrency or resource semantics;
- persistence semantics;
- pipeline behavior;
- timeline semantics;
- hardware scheduling model;
- recovery or resumability semantics.

Do not use ADRs for routine implementation details or parameter tuning.

### Testing

```text
docs/TESTING.md
```

Contains:

- verification strategy;
- test organization;
- environment requirements;
- verification commands;
- known test limitations.

### Benchmarks

```text
docs/BENCHMARK.md
```

Contains reproducible:

- performance baselines;
- workload definitions;
- hardware measurements;
- optimization comparisons;
- benchmark results.

Do not create dated one-off benchmark documents unless explicitly required.

---

## 6. Task Start Protocol

For non-trivial repository work:

1. Read this `AGENTS.md`.
2. Read `docs/PROGRESS.md`.
3. Identify the affected subsystem and task boundary.
4. Read the relevant section of `docs/ARCHITECTURE.md`.
5. Read relevant ADRs when the task touches an existing architectural decision.
6. Inspect the smallest necessary code surface.
7. Identify the verification path before changing code.

Do not begin by reading the entire repository.

Expand context only when dependencies, failures, or observed behavior require it.

---

## 7. Change Discipline

Work in independently understandable and verifiable units.

Avoid:

- unrelated refactors during feature or bug-fix work;
- opportunistic cleanup outside the affected subsystem;
- duplicating existing abstractions;
- bypassing scheduler, timeline, configuration, persistence, or lifecycle abstractions for local convenience.

Before introducing a new abstraction or repository structure, verify that the same responsibility is not already implemented elsewhere.

Preserve unrelated user modifications.

---

## 8. Testing and Verification

Use `uv` for the project environment and dependency execution.

Do not replace the configured CUDA-enabled PyTorch environment with a CPU-only build unless explicitly required.

Run the smallest relevant verification first.

Repository-level verification:

```bash
uv run python -m pytest tests/
```

Compilation check:

```bash
uv run python -m compileall main.py src scripts
```

Compilation success is not sufficient evidence of behavioral correctness.

Changes involving concurrency, FFmpeg, timeline semantics, resumability, persistence, or hardware scheduling require targeted behavioral verification defined in `docs/TESTING.md`.

---

## 9. Test Organization

Organize tests around stable subsystems or business behavior rather than individual bugs.

Prefer extending existing suites for areas such as:

- scanning and persistence;
- analysis and detection;
- timeline behavior;
- rendering and FFmpeg;
- hardware scheduling;
- review and human feedback.

Reusable fixtures, mock factories, and synthetic media helpers belong in shared test utilities such as `tests/conftest.py`.

Do not create a dedicated test file for a single small regression when an appropriate subsystem suite already exists.

---

## 10. Workspace Safety

Before modifying or reverting a file with existing changes, inspect the current diff.

Do not discard, overwrite, reset, stash, or revert unrelated user work.

If repository state is ambiguous, preserve the current state and diagnose before destructive Git operations.

Agent transcripts or tool history may be used as recovery evidence, but they are not a substitute for Git-tracked project state.

---

## 11. Task Completion Protocol

Before declaring non-trivial work complete:

1. Run the relevant verification.
2. Confirm no known regression or incomplete intermediate state remains.
3. Review whether `docs/PROGRESS.md` requires an update.
4. Review whether current architecture or subsystem invariants changed.
5. If so, update `docs/ARCHITECTURE.md`.
6. If an important engineering decision was introduced or replaced, update `docs/DECISIONS.md`.
7. If meaningful performance evidence changed, update `docs/BENCHMARK.md`.
8. Evaluate whether the task exposed a reusable project-level rule or recurring failure pattern.

Do not update documentation merely to record that routine work occurred.

Persist information that a future agent would otherwise have to rediscover.

---

## 12. Documentation Discipline

Documentation should be concise, factual, and maintained as persistent engineering context.

Avoid:

- duplicate architecture descriptions;
- temporary investigation reports in `docs/`;
- narrative development diaries;
- stale configuration documentation;
- implementation details duplicated across multiple files;
- benchmark numbers copied into architecture documentation.

Document responsibilities:

```text
AGENTS.md
    Repository-wide execution rules and documentation routing

docs/ARCHITECTURE.md
    Current architecture and subsystem invariants

docs/PROGRESS.md
    Current project state

docs/DECISIONS.md
    Durable engineering decisions and rationale

docs/TESTING.md
    Verification strategy

docs/BENCHMARK.md
    Reproducible performance evidence
```