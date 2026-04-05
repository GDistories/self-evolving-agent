from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any
from uuid import uuid4

from server2.eval_models import EvalJobRequest, EvalJobStatus


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
    def __init__(self) -> None:
        self.jobs: dict[str, EvalJobStatus] = {}
        self.cache: dict[str, Any] = {}

    def create_job(self, request: EvalJobRequest) -> EvalJobStatus:
        payload = asdict(request)
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
        return job

    def get_job(self, job_id: str) -> EvalJobStatus | None:
        return self.jobs.get(job_id)

    def lookup_cache(self, **payload: Any) -> tuple[bool, Any]:
        cache_key = build_cache_key(**payload)
        if cache_key not in self.cache:
            return False, None
        return True, _json_snapshot(self.cache[cache_key])
