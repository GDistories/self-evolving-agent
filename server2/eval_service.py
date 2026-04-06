"""
Server2 is deployed remotely and accessed through the same kind of proxy/exposure
pattern as server1, but this endpoint is ordinary HTTP rather than WebSocket
streaming because evaluation is job-based.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from server2.eval_jobs import InMemoryEvalJobStore
from server2.eval_models import EvalJobRequest, EvalJobStatus


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


load_env_file(Path(__file__).with_name(".env"))


class EvalServiceRuntime:
    def __init__(self, store: InMemoryEvalJobStore | None = None) -> None:
        self.store = store or InMemoryEvalJobStore()

    def submit(self, request: EvalJobRequest) -> EvalJobStatus:
        return self.store.create_job(request)

    def get(self, job_id: str) -> EvalJobStatus | None:
        return self.store.get_job(job_id)


def create_app(runtime: EvalServiceRuntime | None = None) -> FastAPI:
    app = FastAPI(title="Server2 Evaluator")
    app.state.runtime = runtime or EvalServiceRuntime()

    @app.post("/eval/jobs")
    def create_eval_job(payload: EvalJobRequest, request: Request) -> EvalJobStatus:
        return request.app.state.runtime.submit(payload)

    @app.get("/eval/jobs/{job_id}")
    def get_eval_job(job_id: str, request: Request) -> EvalJobStatus:
        job = request.app.state.runtime.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="job not found")
        return job

    return app


app = create_app()
