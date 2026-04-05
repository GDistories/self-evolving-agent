# Self-Evolving Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a local controller that uses `server1` as the remote LLM brain and `server2` as the remote asynchronous evaluator to iteratively optimize prompt-based attack detection.

**Architecture:** The implementation keeps optimization policy local. `server1` remains a remote `/v1/chat/completions` endpoint accessed through the existing proxy pattern. `server2` is refactored from `batch_run.py` into an HTTP evaluation service with async jobs and cache, then the local controller orchestrates candidate generation, evaluation, judging, and experiment persistence.

**Tech Stack:** Python, FastAPI, Pydantic, httpx, websockets, vLLM, pytest

---

## File Structure

### Existing files to modify

- `client/client_ws_proxy.py`
  Keep as the reference implementation for the `server1` remote bridge. Do not overload it with controller logic.
- `server2/batch_run.py`
  Extract reusable evaluation logic out of this script into service-ready functions.

### New files to create

- `client/config.py`
  Runtime config loading for local controller execution.
- `client/openai_brain.py`
  Small wrapper for calling `server1` through the local OpenAI-compatible bridge.
- `client/evaluator_client.py`
  HTTP client for `server2` job submission and polling.
- `client/models.py`
  Shared Pydantic models for candidates, evaluation specs, judge decisions, and iteration records.
- `client/judge.py`
  Deterministic local promotion logic driven by runtime `metric_config`.
- `client/store.py`
  Local experiment persistence for candidates, runs, and summaries.
- `client/controller.py`
  Main optimization loop and CLI entrypoint.
- `server2/eval_models.py`
  Pydantic models for request, job status, cached result, and API payloads.
- `server2/eval_core.py`
  Reusable evaluation engine extracted from `batch_run.py`.
- `server2/eval_jobs.py`
  In-process job manager, cache key generation, and job state tracking.
- `server2/eval_service.py`
  FastAPI service exposing `POST /eval/jobs` and `GET /eval/jobs/{job_id}`.
- `tests/client/test_judge.py`
  Judge behavior tests.
- `tests/client/test_store.py`
  Local persistence tests.
- `tests/client/test_controller.py`
  Controller loop tests with mocked `server1` and `server2`.
- `tests/server2/test_eval_jobs.py`
  Cache and async job lifecycle tests.
- `tests/server2/test_eval_service.py`
  FastAPI endpoint tests.
- `tests/server2/test_eval_core.py`
  Evaluation engine tests for `tp/tn` aggregation and failure extraction.

### Reference files to inspect during implementation

- `server/server_ws_proxy.py`
  Proxy pattern for remote model-serving over the existing network path.
- `docs/superpowers/specs/2026-04-06-self-evolving-agent-design.zh-CN.md`
  Approved scope and boundary document.

## Task 1: Extract Server2 Evaluation Core

**Files:**
- Create: `server2/eval_core.py`
- Modify: `server2/batch_run.py`
- Test: `tests/server2/test_eval_core.py`

- [ ] **Step 1: Write the failing evaluation-core test**

```python
from server2.eval_core import summarize_binary_results


def test_summarize_binary_results_counts_tp_tn_fp_fn():
    rows = [
        {"dataset": "tp", "sample_id": "tp-1", "predicted_attack": True},
        {"dataset": "tp", "sample_id": "tp-2", "predicted_attack": False},
        {"dataset": "tn", "sample_id": "tn-1", "predicted_attack": False},
        {"dataset": "tn", "sample_id": "tn-2", "predicted_attack": True},
    ]

    result = summarize_binary_results(rows)

    assert result["tp_stats"]["tp"] == 1
    assert result["tp_stats"]["fn"] == 1
    assert result["tn_stats"]["tn"] == 1
    assert result["tn_stats"]["fp"] == 1
    assert result["merged_metrics"]["accuracy"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/server2/test_eval_core.py::test_summarize_binary_results_counts_tp_tn_fp_fn -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'server2.eval_core'`

- [ ] **Step 3: Write minimal reusable evaluation core**

```python
from __future__ import annotations

from typing import Iterable


def summarize_binary_results(rows: Iterable[dict]) -> dict:
    tp = fn = tn = fp = 0
    for row in rows:
        dataset = row["dataset"]
        predicted_attack = bool(row["predicted_attack"])
        if dataset == "tp":
            tp += int(predicted_attack)
            fn += int(not predicted_attack)
        elif dataset == "tn":
            tn += int(not predicted_attack)
            fp += int(predicted_attack)

    total = tp + fn + tn + fp
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "tp_stats": {"tp": tp, "fn": fn, "total": tp + fn},
        "tn_stats": {"tn": tn, "fp": fp, "total": tn + fp},
        "merged_metrics": {"accuracy": accuracy},
    }
```

- [ ] **Step 4: Refactor `server2/batch_run.py` to use the extracted core**

```python
from server2.eval_core import summarize_binary_results


def main():
    detail_rows = [
        {"dataset": "tp", "sample_id": "tp-1", "predicted_attack": True},
        {"dataset": "tn", "sample_id": "tn-1", "predicted_attack": False},
    ]
    summary = summarize_binary_results(detail_rows)
    print("[summary]", json.dumps(summary, ensure_ascii=False))
```

- [ ] **Step 5: Run tests to verify the extraction works**

Run: `pytest tests/server2/test_eval_core.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add server2/eval_core.py server2/batch_run.py tests/server2/test_eval_core.py
git commit -m "refactor: extract server2 evaluation core"
```

## Task 2: Add Server2 Job Models and In-Process Cache

**Files:**
- Create: `server2/eval_models.py`
- Create: `server2/eval_jobs.py`
- Test: `tests/server2/test_eval_jobs.py`

- [ ] **Step 1: Write the failing job-cache test**

```python
from server2.eval_jobs import build_cache_key


def test_build_cache_key_changes_when_prompt_changes():
    a = build_cache_key(
        system_prompt="prompt-a",
        user_template="tmpl",
        tp_hash="tp1",
        tn_hash="tn1",
        evaluator_version="qwen-8b",
        inference_config={"max_tokens": 4},
    )
    b = build_cache_key(
        system_prompt="prompt-b",
        user_template="tmpl",
        tp_hash="tp1",
        tn_hash="tn1",
        evaluator_version="qwen-8b",
        inference_config={"max_tokens": 4},
    )

    assert a != b
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/server2/test_eval_jobs.py::test_build_cache_key_changes_when_prompt_changes -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'server2.eval_jobs'`

- [ ] **Step 3: Define request and job-response models**

```python
from pydantic import BaseModel, Field


class EvalJobRequest(BaseModel):
    candidate_id: str
    system_prompt: str
    user_template: str
    tp_dataset: str
    tn_dataset: str
    metric_config: dict = Field(default_factory=dict)
    inference_config: dict = Field(default_factory=dict)


class EvalJobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    cache_hit: bool = False
    result: dict | None = None
    error: str | None = None
```

- [ ] **Step 4: Implement cache keying and basic job manager**

```python
import hashlib
import json
import uuid


def build_cache_key(**payload: object) -> str:
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class InMemoryEvalJobStore:
    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        self.cache: dict[str, dict] = {}

    def create_job(self) -> str:
        job_id = uuid.uuid4().hex
        self.jobs[job_id] = {"job_id": job_id, "status": "queued", "progress": 0.0}
        return job_id
```

- [ ] **Step 5: Add a job-lifecycle test**

```python
from server2.eval_jobs import InMemoryEvalJobStore


def test_job_store_returns_cached_result_without_new_run():
    store = InMemoryEvalJobStore()
    cache_key = "same-input"
    store.cache[cache_key] = {"merged_metrics": {"accuracy": 1.0}}

    cached = store.lookup_cache(cache_key)

    assert cached["merged_metrics"]["accuracy"] == 1.0
```

- [ ] **Step 6: Run tests to verify cache and models**

Run: `pytest tests/server2/test_eval_jobs.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add server2/eval_models.py server2/eval_jobs.py tests/server2/test_eval_jobs.py
git commit -m "feat: add server2 job models and cache"
```

## Task 3: Expose Server2 HTTP Evaluation Service

**Files:**
- Create: `server2/eval_service.py`
- Modify: `server2/eval_jobs.py`
- Test: `tests/server2/test_eval_service.py`

- [ ] **Step 1: Write the failing FastAPI service test**

```python
from fastapi.testclient import TestClient

from server2.eval_service import app


def test_post_eval_jobs_returns_job_id():
    client = TestClient(app)

    response = client.post(
        "/eval/jobs",
        json={
            "candidate_id": "cand-1",
            "system_prompt": "p",
            "user_template": "{text}",
            "tp_dataset": "/tmp/tp.jsonl",
            "tn_dataset": "/tmp/tn.jsonl",
            "metric_config": {},
        },
    )

    assert response.status_code == 200
    assert response.json()["job_id"]
    assert response.json()["status"] in {"queued", "cached"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/server2/test_eval_service.py::test_post_eval_jobs_returns_job_id -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'server2.eval_service'`

- [ ] **Step 3: Implement the HTTP service with in-process background jobs**

```python
from fastapi import FastAPI, HTTPException

from server2.eval_jobs import EvalServiceRuntime
from server2.eval_models import EvalJobRequest, EvalJobStatus

app = FastAPI(title="Server2 Evaluator")
runtime = EvalServiceRuntime()


@app.post("/eval/jobs", response_model=EvalJobStatus)
def create_eval_job(payload: EvalJobRequest) -> EvalJobStatus:
    return runtime.submit(payload)


@app.get("/eval/jobs/{job_id}", response_model=EvalJobStatus)
def get_eval_job(job_id: str) -> EvalJobStatus:
    job = runtime.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job
```

- [ ] **Step 4: Make remote-access assumptions explicit in module docs**

```python
"""
Server2 is deployed remotely and accessed through an HTTP proxy path similar
to server1. Unlike server1, this service is request/response oriented and does
not require WebSocket streaming for normal operation.
"""
```

- [ ] **Step 5: Add job-status polling test**

```python
def test_get_eval_job_returns_404_for_unknown_job():
    client = TestClient(app)
    response = client.get("/eval/jobs/missing")
    assert response.status_code == 404
```

- [ ] **Step 6: Run tests to verify the API**

Run: `pytest tests/server2/test_eval_service.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add server2/eval_service.py server2/eval_jobs.py tests/server2/test_eval_service.py
git commit -m "feat: add server2 evaluation service"
```

## Task 4: Add Local Candidate Models, Judge, and Experiment Store

**Files:**
- Create: `client/models.py`
- Create: `client/judge.py`
- Create: `client/store.py`
- Test: `tests/client/test_judge.py`
- Test: `tests/client/test_store.py`

- [ ] **Step 1: Write the failing judge test**

```python
from client.judge import Judge


def test_judge_promotes_candidate_that_meets_threshold_and_beats_best():
    judge = Judge()
    decision = judge.compare(
        best_metrics={"score": 0.70},
        challenger_metrics={"score": 0.81},
        metric_config={"primary_metric": "score", "min_value": 0.80},
    )

    assert decision.promote is True
    assert decision.reason == "challenger meets threshold and improves primary_metric"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/client/test_judge.py::test_judge_promotes_candidate_that_meets_threshold_and_beats_best -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'client.judge'`

- [ ] **Step 3: Define candidate and decision models**

```python
from pydantic import BaseModel, Field


class Candidate(BaseModel):
    candidate_id: str
    parent_candidate_id: str | None = None
    system_prompt: str
    user_template: str
    mutation_note: str = ""
    status: str = "draft"


class JudgeDecision(BaseModel):
    promote: bool
    reason: str
```

- [ ] **Step 4: Implement deterministic judge logic**

```python
class Judge:
    def compare(self, best_metrics: dict, challenger_metrics: dict, metric_config: dict) -> JudgeDecision:
        primary = metric_config["primary_metric"]
        min_value = metric_config.get("min_value")
        challenger_value = challenger_metrics.get(primary, 0.0)
        best_value = best_metrics.get(primary, 0.0)

        if min_value is not None and challenger_value < min_value:
            return JudgeDecision(promote=False, reason="challenger does not meet threshold")
        if challenger_value <= best_value:
            return JudgeDecision(promote=False, reason="challenger does not improve primary_metric")
        return JudgeDecision(promote=True, reason="challenger meets threshold and improves primary_metric")
```

- [ ] **Step 5: Implement local JSON persistence**

```python
import json
from pathlib import Path


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_candidate(self, candidate: Candidate) -> Path:
        path = self.root / "candidates" / f"{candidate.candidate_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(candidate.model_dump_json(indent=2), encoding="utf-8")
        return path
```

- [ ] **Step 6: Add persistence test**

```python
from pathlib import Path

from client.models import Candidate
from client.store import ExperimentStore


def test_store_saves_candidate_json(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    path = store.save_candidate(
        Candidate(candidate_id="cand-1", system_prompt="p", user_template="{text}")
    )

    assert path.exists()
    assert '"candidate_id": "cand-1"' in path.read_text(encoding="utf-8")
```

- [ ] **Step 7: Run tests to verify judge and store**

Run: `pytest tests/client/test_judge.py tests/client/test_store.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add client/models.py client/judge.py client/store.py tests/client/test_judge.py tests/client/test_store.py
git commit -m "feat: add local candidate judge and store"
```

## Task 5: Add Server1 and Server2 Clients for the Local Controller

**Files:**
- Create: `client/config.py`
- Create: `client/openai_brain.py`
- Create: `client/evaluator_client.py`
- Test: `tests/client/test_controller.py`

- [ ] **Step 1: Write the failing remote-client test**

```python
from client.evaluator_client import EvaluatorClient


def test_evaluator_client_builds_job_payload():
    client = EvaluatorClient(base_url="http://127.0.0.1:19000")
    payload = client.build_payload(
        candidate_id="cand-1",
        system_prompt="prompt",
        user_template="{text}",
        tp_dataset="/tmp/tp.jsonl",
        tn_dataset="/tmp/tn.jsonl",
        metric_config={"primary_metric": "score"},
    )

    assert payload["candidate_id"] == "cand-1"
    assert payload["metric_config"]["primary_metric"] == "score"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/client/test_controller.py::test_evaluator_client_builds_job_payload -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'client.evaluator_client'`

- [ ] **Step 3: Implement config and `server1` client**

```python
from pydantic import BaseModel
import httpx


class RuntimeConfig(BaseModel):
    server1_base_url: str
    server2_base_url: str
    poll_interval_seconds: float = 2.0


class OpenAIBrainClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def propose_candidate(self, messages: list[dict]) -> dict:
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
```

- [ ] **Step 4: Implement `server2` HTTP client**

```python
class EvaluatorClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def build_payload(self, **kwargs: object) -> dict:
        return dict(kwargs)

    def submit_job(self, payload: dict) -> dict:
        response = httpx.post(f"{self.base_url}/eval/jobs", json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()

    def get_job(self, job_id: str) -> dict:
        response = httpx.get(f"{self.base_url}/eval/jobs/{job_id}", timeout=30.0)
        response.raise_for_status()
        return response.json()
```

- [ ] **Step 5: Encode the access-pattern note for server2**

```python
"""
server2 is remote like server1 and should be accessed through the same network
exposure pattern, but it can use normal HTTP because evaluation is job-based
rather than token streaming.
"""
```

- [ ] **Step 6: Run tests to verify the client layer**

Run: `pytest tests/client/test_controller.py::test_evaluator_client_builds_job_payload -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add client/config.py client/openai_brain.py client/evaluator_client.py tests/client/test_controller.py
git commit -m "feat: add remote brain and evaluator clients"
```

## Task 6: Build the Local Optimization Controller

**Files:**
- Create: `client/controller.py`
- Modify: `client/models.py`
- Modify: `client/store.py`
- Test: `tests/client/test_controller.py`

- [ ] **Step 1: Write the failing controller-loop test**

```python
from client.controller import run_iteration
from client.models import Candidate


class FakeBrain:
    def propose_candidate(self, messages):
        return {
            "candidate": {
                "candidate_id": "cand-2",
                "parent_candidate_id": "cand-1",
                "system_prompt": "new prompt",
                "user_template": "{text}",
                "mutation_note": "tighten attack definition",
            }
        }


class FakeEvaluator:
    def submit_job(self, payload):
        return {"job_id": "job-1", "status": "completed"}

    def get_job(self, job_id):
        return {
            "job_id": job_id,
            "status": "completed",
            "result": {"merged_metrics": {"score": 0.91}},
        }


def test_run_iteration_promotes_better_candidate(tmp_path):
    best = Candidate(candidate_id="cand-1", system_prompt="old", user_template="{text}", status="best")

    outcome = run_iteration(
        best_candidate=best,
        best_metrics={"score": 0.80},
        metric_config={"primary_metric": "score", "min_value": 0.85},
        brain_client=FakeBrain(),
        evaluator_client=FakeEvaluator(),
        store_root=tmp_path,
    )

    assert outcome.best_candidate.candidate_id == "cand-2"
    assert outcome.decision.promote is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/client/test_controller.py::test_run_iteration_promotes_better_candidate -v`
Expected: FAIL with `ImportError` for `run_iteration`

- [ ] **Step 3: Implement one-iteration orchestration**

```python
def run_iteration(
    best_candidate,
    best_metrics,
    metric_config,
    brain_client,
    evaluator_client,
    store_root,
):
    store = ExperimentStore(store_root)
    proposal = brain_client.propose_candidate(messages=build_brain_messages(best_candidate, best_metrics))
    challenger = Candidate(**proposal["candidate"])
    store.save_candidate(challenger)

    payload = {
        "candidate_id": challenger.candidate_id,
        "system_prompt": challenger.system_prompt,
        "user_template": challenger.user_template,
        "tp_dataset": metric_config["tp_path"],
        "tn_dataset": metric_config["tn_path"],
        "metric_config": metric_config,
    }
    job = evaluator_client.submit_job(payload)
    result = evaluator_client.get_job(job["job_id"])
    decision = Judge().compare(best_metrics, result["result"]["merged_metrics"], metric_config)
    return IterationOutcome(best_candidate=challenger if decision.promote else best_candidate, decision=decision)
```

- [ ] **Step 4: Extend to bounded multi-round execution**

```python
def run_optimization_loop(config, baseline_candidate, baseline_metrics):
    best_candidate = baseline_candidate
    best_metrics = baseline_metrics
    for _ in range(config.max_rounds):
        outcome = run_iteration(
            best_candidate=best_candidate,
            best_metrics=best_metrics,
            metric_config={
                "primary_metric": config.primary_metric,
                "min_value": config.min_value,
                "tp_path": config.tp_path,
                "tn_path": config.tn_path,
            },
            brain_client=config.brain_client,
            evaluator_client=config.evaluator_client,
            store_root=config.store_root,
        )
        if outcome.decision.promote:
            best_candidate = outcome.best_candidate
            best_metrics = outcome.metrics
        if should_stop(
            iteration_index=outcome.iteration_index,
            max_rounds=config.max_rounds,
            no_improvement_rounds=outcome.no_improvement_rounds,
            max_no_improvement_rounds=config.max_no_improvement_rounds,
        ):
            break
    return best_candidate
```

- [ ] **Step 5: Add test for non-promotion path**

```python
def test_run_iteration_keeps_best_when_candidate_does_not_improve(tmp_path):
    class WeakBrain:
        def propose_candidate(self, messages):
            return {
                "candidate": {
                    "candidate_id": "cand-3",
                    "parent_candidate_id": "cand-1",
                    "system_prompt": "weaker prompt",
                    "user_template": "{text}",
                    "mutation_note": "overfit to negatives",
                }
            }

    class WeakEvaluator:
        def submit_job(self, payload):
            return {"job_id": "job-2", "status": "completed"}

        def get_job(self, job_id):
            return {
                "job_id": job_id,
                "status": "completed",
                "result": {"merged_metrics": {"score": 0.78}},
            }

    best = Candidate(candidate_id="cand-1", system_prompt="old", user_template="{text}", status="best")
    outcome = run_iteration(
        best_candidate=best,
        best_metrics={"score": 0.80},
        metric_config={"primary_metric": "score", "min_value": 0.75, "tp_path": "/tmp/tp.jsonl", "tn_path": "/tmp/tn.jsonl"},
        brain_client=WeakBrain(),
        evaluator_client=WeakEvaluator(),
        store_root=tmp_path,
    )

    assert outcome.best_candidate.candidate_id == "cand-1"
    assert outcome.decision.promote is False
```

- [ ] **Step 6: Run controller tests**

Run: `pytest tests/client/test_controller.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add client/controller.py client/models.py client/store.py tests/client/test_controller.py
git commit -m "feat: add self-evolving local controller"
```

## Task 7: Wire End-to-End Smoke Flow and Update Documentation

**Files:**
- Modify: `README.md`
- Modify: `server2/eval_service.py`
- Modify: `client/controller.py`
- Test: `tests/client/test_controller.py`
- Test: `tests/server2/test_eval_service.py`

- [ ] **Step 1: Write the failing smoke-style test**

```python
def test_controller_uses_completed_job_without_poll_loop_when_cached(tmp_path):
    class CachedBrain:
        def propose_candidate(self, messages):
            return {
                "candidate": {
                    "candidate_id": "cand-cache",
                    "parent_candidate_id": "cand-1",
                    "system_prompt": "cached prompt",
                    "user_template": "{text}",
                    "mutation_note": "reuse cached result",
                }
            }

    class CachedEvaluator:
        def submit_job(self, payload):
            return {
                "job_id": "job-cache",
                "status": "cached",
                "cache_hit": True,
                "result": {"merged_metrics": {"score": 0.93}},
            }

        def get_job(self, job_id):
            raise AssertionError("get_job should not be called for cached jobs")

    best = Candidate(candidate_id="cand-1", system_prompt="old", user_template="{text}", status="best")
    outcome = run_iteration(
        best_candidate=best,
        best_metrics={"score": 0.80},
        metric_config={"primary_metric": "score", "min_value": 0.85, "tp_path": "/tmp/tp.jsonl", "tn_path": "/tmp/tn.jsonl"},
        brain_client=CachedBrain(),
        evaluator_client=CachedEvaluator(),
        store_root=tmp_path,
    )

    assert outcome.cache_hit is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/client/test_controller.py::test_controller_uses_completed_job_without_poll_loop_when_cached -v`
Expected: FAIL because cache-hit behavior is not yet surfaced

- [ ] **Step 3: Add cache-hit handling and explicit polling behavior**

```python
if job["status"] == "cached":
    final_job = job
else:
    final_job = wait_for_job_completion(evaluator_client, job["job_id"], poll_interval_seconds)
```

- [ ] **Step 4: Document local startup and remote topology**

```markdown
## Topology

- Local machine runs the controller
- `server1` is the remote LLM brain exposed as `/v1/chat/completions`
- `server2` is the remote evaluator exposed over HTTP through the same proxy style

## Startup

1. Start the `server1` bridge
2. Start `server2/eval_service.py` remotely
3. Run `python -m client.controller --config path/to/run.json`
```

- [ ] **Step 5: Run targeted tests**

Run: `pytest tests/client/test_controller.py tests/server2/test_eval_service.py -v`
Expected: PASS

- [ ] **Step 6: Run full test suite**

Run: `pytest tests/client tests/server2 -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add README.md client/controller.py server2/eval_service.py tests/client/test_controller.py tests/server2/test_eval_service.py
git commit -m "docs: add self-evolving agent smoke flow"
```

## Spec Coverage Check

- 本地控制器作为唯一控制器
  Covered by Task 4, Task 5, Task 6, Task 7.
- `server1` 作为远端大模型大脑
  Covered by Task 5 and Task 6.
- `server2` 作为异步评测后端
  Covered by Task 1, Task 2, Task 3, Task 7.
- 动态指标输入
  Covered by Task 4, Task 5, Task 6.
- `tp.jsonl` 与 `tn.jsonl` 二元评测
  Covered by Task 1 and Task 3.
- 缓存
  Covered by Task 2 and Task 7.
- 单 best candidate 闭环
  Covered by Task 4 and Task 6.
- 错误边界与远端调用方式
  Covered by Task 3, Task 5, Task 7.

## Self-Review

- Placeholder scan complete: all previous `...` placeholders in Task 1, Task 6, and Task 7 were replaced with concrete code and assertions.
- Type consistency checked:
  `Candidate`, `JudgeDecision`, `EvalJobRequest`, and `EvalJobStatus` names are used consistently across tasks.
- Scope check:
  the plan stays within one integrated deliverable and does not introduce multi-branch search, reward models, or local LLM execution.
