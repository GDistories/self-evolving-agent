"""
Server2 is remote like server1 and should be accessed through a similar
exposure/proxy pattern, but with ordinary HTTP instead of WebSocket transport
because evaluation is job-based.
"""

from __future__ import annotations

from typing import Any

import httpx


class EvaluatorClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def build_payload(self, **kwargs: object) -> dict[str, object]:
        return dict(kwargs)

    def submit_job(self, payload: dict[str, Any]) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/eval/jobs",
            json=payload,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        response = httpx.get(
            f"{self.base_url}/eval/jobs/{job_id}",
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
