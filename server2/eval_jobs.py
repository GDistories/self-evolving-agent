from __future__ import annotations

import hashlib
import json
from collections import deque
from threading import Lock
from typing import Any
from uuid import uuid4

from server2.eval_core import (
    build_dataset_fingerprint,
    load_dataset_records,
    summarize_binary_results,
)
from server2.eval_models import EvalJobRequest, EvalJobStatus
from server2.model_runtime import ModelRuntime


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    return value


def build_cache_key(**payload: Any) -> str:
    normalized = _canonicalize(payload)
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _json_snapshot(value: Any) -> Any:
    return json.loads(json.dumps(value))


class InMemoryEvalJobStore:
    def __init__(self, model_runtime: ModelRuntime | Any | None = None) -> None:
        self.jobs: dict[str, EvalJobStatus] = {}
        self.job_requests: dict[str, EvalJobRequest] = {}
        self.cache: dict[str, Any] = {}
        self.model_runtime = model_runtime or ModelRuntime()
        self._pending_job_ids: deque[str] = deque()
        self._lock = Lock()

    def create_job(self, request: EvalJobRequest) -> EvalJobStatus:
        payload = self._build_cache_payload(request)
        cache_hit, cache_result = self.lookup_cache(**payload)
        job = EvalJobStatus(
            job_id=uuid4().hex,
            status="cached" if cache_hit else "queued",
            progress=1.0 if cache_hit else 0.0,
            cache_hit=cache_hit,
            result=cache_result,
            error=None,
        )
        self.jobs[job.job_id] = job
        self.job_requests[job.job_id] = request
        if not cache_hit:
            with self._lock:
                self._pending_job_ids.append(job.job_id)
        return job

    def get_job(self, job_id: str) -> EvalJobStatus | None:
        return self.jobs.get(job_id)

    def lookup_cache(self, **payload: Any) -> tuple[bool, Any]:
        cache_key = build_cache_key(**payload)
        if cache_key not in self.cache:
            return False, None
        return True, _json_snapshot(self.cache[cache_key])

    def process_pending_jobs(self) -> None:
        while True:
            with self._lock:
                if not self._pending_job_ids:
                    return
                job_id = self._pending_job_ids.popleft()
            job = self.jobs[job_id]
            self._run_job(job_id, job)

    def _build_cache_payload(self, request: EvalJobRequest) -> dict[str, Any]:
        return {
            "candidate_id": request.candidate_id,
            "system_prompt": request.system_prompt,
            "user_template": request.user_template,
            "tp_fingerprint": build_dataset_fingerprint(request.tp_dataset),
            "tn_fingerprint": build_dataset_fingerprint(request.tn_dataset),
            "metric_config": request.metric_config,
            "inference_config": request.inference_config,
            "runtime_mode": getattr(self.model_runtime, "mode", "custom"),
            "runtime_backend": type(getattr(self.model_runtime, "backend_factory", self.model_runtime)).__name__,
            "tp_dataset": request.tp_dataset,
            "tn_dataset": request.tn_dataset,
        }

    def _run_job(self, job_id: str, job: EvalJobStatus) -> None:
        request = self.job_requests[job_id]
        try:
            job.status = "running"
            job.progress = 0.1
            tp_rows = load_dataset_records(request.tp_dataset, dataset="tp")
            tn_rows = load_dataset_records(request.tn_dataset, dataset="tn")
            dataset_records = tp_rows + tn_rows
            predictions = self.model_runtime.evaluate(
                system_prompt=request.system_prompt,
                user_template=request.user_template,
                dataset_records=dataset_records,
                inference_config=request.inference_config,
            )
            result = summarize_binary_results(predictions)
            result["job_id"] = job_id
            result["candidate_id"] = request.candidate_id
            result["dataset_fingerprint"] = {
                "tp": build_dataset_fingerprint(request.tp_dataset),
                "tn": build_dataset_fingerprint(request.tn_dataset),
            }
            result["cache_hit"] = False
            cache_payload = self._build_cache_payload(request)
            self.cache[build_cache_key(**cache_payload)] = result
            job.status = "completed"
            job.progress = 1.0
            job.result = _json_snapshot(result)
            job.error = None
        except Exception as exc:
            job.status = "failed"
            job.progress = 1.0
            job.error = str(exc)
