from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx

from client.config import RuntimeConfig, load_runtime_config
from client.controller import run_iteration
from client.evaluator_client import EvaluatorClient
from client.models import Candidate
from client.openai_brain import OpenAIBrainClient


def test_evaluator_client_builds_job_payload() -> None:
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


def test_load_runtime_config_uses_local_bridge_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SERVER1_BASE_URL", raising=False)
    monkeypatch.delenv("SERVER2_BASE_URL", raising=False)
    monkeypatch.delenv("POLL_INTERVAL_SECONDS", raising=False)

    config = load_runtime_config()

    assert config == RuntimeConfig(
        server1_base_url="http://127.0.0.1:18000",
        server2_base_url="http://127.0.0.1:19000",
        poll_interval_seconds=2.0,
    )


def test_load_runtime_config_reads_env(monkeypatch) -> None:
    monkeypatch.setenv("SERVER1_BASE_URL", "http://server1.example")
    monkeypatch.setenv("SERVER2_BASE_URL", "http://server2.example")
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "3.5")

    config = load_runtime_config()

    assert config.server1_base_url == "http://server1.example"
    assert config.server2_base_url == "http://server2.example"
    assert config.poll_interval_seconds == 3.5


def test_load_runtime_config_rejects_invalid_poll_interval(monkeypatch) -> None:
    monkeypatch.setenv("POLL_INTERVAL_SECONDS", "nan")

    try:
        load_runtime_config()
    except ValueError as exc:
        assert "POLL_INTERVAL_SECONDS" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_openai_brain_client_propose_candidate_constructs_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"ok": True}

    def fake_post(url: str, **kwargs: object) -> DummyResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return DummyResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    client = OpenAIBrainClient(base_url="http://brain.example/", model="gpt-test")
    response = client.propose_candidate([{"role": "user", "content": "hello"}])

    assert response == {"ok": True}
    assert captured["url"] == "http://brain.example/v1/chat/completions"
    assert captured["kwargs"]["json"] == {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False,
    }
    assert captured["kwargs"]["timeout"] == 60.0


def test_evaluator_client_submit_job_constructs_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"job_id": "job-1"}

    def fake_post(url: str, **kwargs: object) -> DummyResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return DummyResponse()

    monkeypatch.setattr(httpx, "post", fake_post)

    client = EvaluatorClient(base_url="http://server2.example/")
    payload = {"candidate_id": "cand-1", "metric_config": {"primary_metric": "score"}}
    response = client.submit_job(payload)

    assert response == {"job_id": "job-1"}
    assert captured["url"] == "http://server2.example/eval/jobs"
    assert captured["kwargs"]["json"] == payload
    assert captured["kwargs"]["timeout"] == 30.0


def test_evaluator_client_get_job_constructs_request(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class DummyResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {"job_id": "job-1", "status": "queued"}

    def fake_get(url: str, **kwargs: object) -> DummyResponse:
        captured["url"] = url
        captured["kwargs"] = kwargs
        return DummyResponse()

    monkeypatch.setattr(httpx, "get", fake_get)

    client = EvaluatorClient(base_url="http://server2.example/")
    response = client.get_job("job-1")

    assert response == {"job_id": "job-1", "status": "queued"}
    assert captured["url"] == "http://server2.example/eval/jobs/job-1"
    assert captured["kwargs"]["timeout"] == 30.0


def test_run_iteration_promotes_better_candidate(tmp_path: Path) -> None:
    class FakeBrain:
        def __init__(self) -> None:
            self.messages: list[dict[str, object]] | None = None

        def propose_candidate(self, messages: list[dict[str, object]]) -> dict[str, object]:
            self.messages = messages
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
        def __init__(self) -> None:
            self.payload: dict[str, object] | None = None
            self.get_job_calls = 0

        def submit_job(self, payload: dict[str, object]) -> dict[str, object]:
            self.payload = payload
            return {"job_id": "job-1", "status": "completed"}

        def get_job(self, job_id: str) -> dict[str, object]:
            self.get_job_calls += 1
            return {
                "job_id": job_id,
                "status": "completed",
                "result": {"merged_metrics": {"score": 0.91}},
            }

    best = Candidate(
        candidate_id="cand-1",
        system_prompt="old prompt",
        user_template="{text}",
        status="best",
    )
    brain = FakeBrain()
    evaluator = FakeEvaluator()

    outcome = run_iteration(
        best_candidate=best,
        best_metrics={"score": 0.80},
        metric_config={
            "primary_metric": "score",
            "min_value": 0.85,
            "tp_path": "/tmp/tp.jsonl",
            "tn_path": "/tmp/tn.jsonl",
        },
        brain_client=brain,
        evaluator_client=evaluator,
        store_root=tmp_path,
    )

    assert brain.messages is not None
    assert evaluator.payload == {
        "candidate_id": "cand-2",
        "system_prompt": "new prompt",
        "user_template": "{text}",
        "tp_dataset": "/tmp/tp.jsonl",
        "tn_dataset": "/tmp/tn.jsonl",
        "metric_config": {
            "primary_metric": "score",
            "min_value": 0.85,
            "tp_path": "/tmp/tp.jsonl",
            "tn_path": "/tmp/tn.jsonl",
        },
    }
    assert outcome.best_candidate.candidate_id == "cand-2"
    assert outcome.decision.promote is True
    assert outcome.metrics == {"score": 0.91}
    assert (tmp_path / "candidates" / "cand-2.json").exists()
    iteration_log = tmp_path / "iterations" / "cand-2.json"
    assert iteration_log.exists()
    iteration_text = iteration_log.read_text(encoding="utf-8")
    assert '"decision"' in iteration_text
    assert '"submitted_job"' in iteration_text
    assert '"best_candidate_after"' in iteration_text


def test_run_iteration_keeps_best_when_candidate_does_not_improve(
    tmp_path: Path,
) -> None:
    class FakeBrain:
        def propose_candidate(self, messages: list[dict[str, object]]) -> dict[str, object]:
            return {
                "candidate": {
                    "candidate_id": "cand-3",
                    "parent_candidate_id": "cand-1",
                    "system_prompt": "weaker prompt",
                    "user_template": "{text}",
                    "mutation_note": "overfit to negatives",
                }
            }

    class FakeEvaluator:
        def submit_job(self, payload: dict[str, object]) -> dict[str, object]:
            return {"job_id": "job-2", "status": "completed"}

        def get_job(self, job_id: str) -> dict[str, object]:
            return {
                "job_id": job_id,
                "status": "completed",
                "result": {"merged_metrics": {"score": 0.78}},
            }

    best = Candidate(
        candidate_id="cand-1",
        system_prompt="old prompt",
        user_template="{text}",
        status="best",
    )

    outcome = run_iteration(
        best_candidate=best,
        best_metrics={"score": 0.80},
        metric_config={
            "primary_metric": "score",
            "min_value": 0.75,
            "tp_path": "/tmp/tp.jsonl",
            "tn_path": "/tmp/tn.jsonl",
        },
        brain_client=FakeBrain(),
        evaluator_client=FakeEvaluator(),
        store_root=tmp_path,
    )

    assert outcome.best_candidate.candidate_id == "cand-1"
    assert outcome.challenger_candidate.candidate_id == "cand-3"
    assert outcome.decision.promote is False
    assert outcome.metrics == {"score": 0.78}
    assert (tmp_path / "candidates" / "cand-3.json").exists()


def test_run_iteration_uses_inline_cached_result_without_fetching_job(
    tmp_path: Path,
) -> None:
    class FakeBrain:
        def propose_candidate(self, messages: list[dict[str, object]]) -> dict[str, object]:
            return {
                "candidate": {
                    "candidate_id": "cand-4",
                    "parent_candidate_id": "cand-1",
                    "system_prompt": "cached prompt",
                    "user_template": "{text}",
                    "mutation_note": "reuse cached eval",
                }
            }

    class FakeEvaluator:
        def submit_job(self, payload: dict[str, object]) -> dict[str, object]:
            return {
                "job_id": "job-4",
                "status": "cached",
                "result": {"merged_metrics": {"score": 0.92}},
            }

        def get_job(self, job_id: str) -> dict[str, object]:
            raise AssertionError("get_job must not be called for inline cached results")

    best = Candidate(
        candidate_id="cand-1",
        system_prompt="old prompt",
        user_template="{text}",
        status="best",
    )

    outcome = run_iteration(
        best_candidate=best,
        best_metrics={"score": 0.80},
        metric_config={
            "primary_metric": "score",
            "min_value": 0.85,
            "tp_path": "/tmp/tp.jsonl",
            "tn_path": "/tmp/tn.jsonl",
        },
        brain_client=FakeBrain(),
        evaluator_client=FakeEvaluator(),
        store_root=tmp_path,
    )

    assert outcome.best_candidate.candidate_id == "cand-4"
    assert outcome.decision.promote is True
    assert outcome.metrics == {"score": 0.92}
