from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server2.eval_service import EvalServiceRuntime, build_model_runtime_from_env, create_app


class FakeRuntime:
    def evaluate(self, *, system_prompt, user_template, dataset_records, inference_config):
        return [
            {
                "dataset": row["dataset"],
                "sample_id": row["id"],
                "predicted_attack": row["id"].endswith("attack"),
                "raw_response": "B" if row["id"].endswith("attack") else "A",
            }
            for row in dataset_records
        ]


def build_client(model_runtime=None) -> TestClient:
    return TestClient(create_app(EvalServiceRuntime(model_runtime=model_runtime)))


def test_post_eval_jobs_returns_job_id(tmp_path: Path):
    client = build_client(model_runtime=FakeRuntime())
    tp_path = tmp_path / "tp.jsonl"
    tn_path = tmp_path / "tn.jsonl"
    tp_path.write_text(json.dumps({"id": "tp-attack", "text": "alpha"}) + "\n", encoding="utf-8")
    tn_path.write_text(json.dumps({"id": "tn-normal", "text": "beta"}) + "\n", encoding="utf-8")

    response = client.post(
        "/eval/jobs",
        json={
            "candidate_id": "cand-1",
            "system_prompt": "prompt-a",
            "user_template": "user",
            "tp_dataset": str(tp_path),
            "tn_dataset": str(tn_path),
            "metric_config": {"accuracy": True},
            "inference_config": {"temperature": 0.0},
        },
    )

    assert response.status_code == 200
    assert response.json()["job_id"]
    assert response.json()["status"] in {"queued", "running", "completed"}


def test_get_eval_job_returns_404_for_unknown_job():
    client = build_client()

    response = client.get("/eval/jobs/missing")

    assert response.status_code == 404


def test_get_eval_job_returns_created_job(tmp_path: Path):
    client = build_client(model_runtime=FakeRuntime())
    tp_path = tmp_path / "tp.jsonl"
    tn_path = tmp_path / "tn.jsonl"
    tp_path.write_text(json.dumps({"id": "tp-attack", "text": "alpha"}) + "\n", encoding="utf-8")
    tn_path.write_text(json.dumps({"id": "tn-normal", "text": "beta"}) + "\n", encoding="utf-8")

    payload = {
        "candidate_id": "cand-1",
        "system_prompt": "prompt-a",
        "user_template": "user",
        "tp_dataset": str(tp_path),
        "tn_dataset": str(tn_path),
        "metric_config": {"accuracy": True},
        "inference_config": {"temperature": 0.0},
    }

    post_response = client.post("/eval/jobs", json=payload)
    assert post_response.status_code == 200

    created_job = post_response.json()
    get_response = client.get(f"/eval/jobs/{created_job['job_id']}")
    if get_response.json()["status"] in {"queued", "running"}:
        time.sleep(0.05)
        get_response = client.get(f"/eval/jobs/{created_job['job_id']}")

    assert get_response.status_code == 200
    body = get_response.json()
    assert body["job_id"] == created_job["job_id"]
    assert body["status"] in {"completed", "cached"}
    assert body["result"]["merged_metrics"]["accuracy"] == 1.0


def test_build_model_runtime_from_env_reads_lifecycle_mode(monkeypatch):
    monkeypatch.setenv("SERVER2_MODEL_PATH", "/models/qwen3-8b")
    monkeypatch.setenv("SERVER2_MODEL_LIFECYCLE_MODE", "per_job")
    monkeypatch.setenv("SERVER2_TENSOR_PARALLEL_SIZE", "2")

    runtime = build_model_runtime_from_env()

    assert runtime.mode == "per_job"
    assert runtime.backend_config["model_path"] == "/models/qwen3-8b"
    assert runtime.backend_config["tensor_parallel_size"] == 2
