from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server2.eval_service import EvalServiceRuntime, create_app


def build_client() -> TestClient:
    return TestClient(create_app(EvalServiceRuntime()))


def test_post_eval_jobs_returns_job_id():
    client = build_client()

    response = client.post(
        "/eval/jobs",
        json={
            "candidate_id": "cand-1",
            "system_prompt": "prompt-a",
            "user_template": "user",
            "tp_dataset": ["tp-1"],
            "tn_dataset": ["tn-1"],
            "metric_config": {"accuracy": True},
            "inference_config": {"temperature": 0.0},
        },
    )

    assert response.status_code == 200
    assert response.json()["job_id"]
    assert response.json()["status"] == "queued"


def test_get_eval_job_returns_404_for_unknown_job():
    client = build_client()

    response = client.get("/eval/jobs/missing")

    assert response.status_code == 404


def test_get_eval_job_returns_created_job():
    client = build_client()

    payload = {
        "candidate_id": "cand-1",
        "system_prompt": "prompt-a",
        "user_template": "user",
        "tp_dataset": ["tp-1"],
        "tn_dataset": ["tn-1"],
        "metric_config": {"accuracy": True},
        "inference_config": {"temperature": 0.0},
    }

    post_response = client.post("/eval/jobs", json=payload)
    assert post_response.status_code == 200

    created_job = post_response.json()
    get_response = client.get(f"/eval/jobs/{created_job['job_id']}")

    assert get_response.status_code == 200
    assert get_response.json() == created_job
