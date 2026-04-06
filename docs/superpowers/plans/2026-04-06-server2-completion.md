# Server2 Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish `server2` so it executes local-path dataset evaluation jobs over the 8B model while keeping the current HTTP API unchanged.

**Architecture:** Keep the existing FastAPI service, replace the in-memory placeholder job flow with background execution, and extract the reusable model inference logic from `temp/batch_run.py`. Use content-based dataset fingerprinting and configurable model lifecycle modes (`per_job` and `lazy_reuse`) so `client` can keep polling the same endpoints.

**Tech Stack:** Python, FastAPI, background threads, dataclasses, JSONL parsing, vLLM-compatible inference adapter

---

### Task 1: Define the Real Server2 Job Contract

**Files:**
- Modify: `server2/eval_models.py`
- Test: `tests/server2/test_eval_jobs.py`

- [ ] Add failing tests for path-based datasets and expanded job statuses.
- [ ] Update request/status models for string dataset paths and terminal/non-terminal states.
- [ ] Re-run `tests/server2/test_eval_jobs.py`.

### Task 2: Add Dataset Loading and Result Assembly

**Files:**
- Modify: `server2/eval_core.py`
- Test: `tests/server2/test_eval_core.py`

- [ ] Add failing tests for JSONL loading, dataset fingerprinting, binary label mapping, and failure sample extraction.
- [ ] Implement file loading and structured result assembly.
- [ ] Re-run `tests/server2/test_eval_core.py`.

### Task 3: Add Reusable 8B Runtime Wrapper

**Files:**
- Create: `server2/model_runtime.py`
- Test: `tests/server2/test_model_runtime.py`

- [ ] Add failing tests for `per_job` and `lazy_reuse` lifecycle behavior using a fake backend.
- [ ] Extract the `temp/batch_run.py` prompt/inference flow into a runtime wrapper with injectable backend.
- [ ] Re-run `tests/server2/test_model_runtime.py`.

### Task 4: Replace Placeholder Job Store with Real Execution

**Files:**
- Modify: `server2/eval_jobs.py`
- Modify: `server2/eval_service.py`
- Test: `tests/server2/test_eval_jobs.py`
- Test: `tests/server2/test_eval_service.py`

- [ ] Add failing tests for async job execution, status transitions, failure propagation, and content-based cache hits.
- [ ] Implement background execution, cache writes, and runtime integration.
- [ ] Re-run the focused `server2` tests.

### Task 5: Verify Controller Compatibility

**Files:**
- Modify: `tests/client/test_controller.py`

- [ ] Add or adjust tests so `client` continues to submit path strings and consume final `merged_metrics`.
- [ ] Re-run focused client tests plus `tests/server2`.
