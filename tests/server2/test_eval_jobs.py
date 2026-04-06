from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server2.eval_jobs import InMemoryEvalJobStore, build_cache_key
from server2.eval_models import EvalJobRequest, EvalJobStatus


def test_build_cache_key_changes_when_prompt_changes():
    base_payload = {
        "candidate_id": "cand-1",
        "system_prompt": "prompt-a",
        "user_template": "user",
        "tp_dataset": ["tp-1"],
        "tn_dataset": ["tn-1"],
        "metric_config": {"accuracy": True},
        "inference_config": {"temperature": 0.0},
    }

    changed_payload = dict(base_payload)
    changed_payload["system_prompt"] = "prompt-b"

    assert build_cache_key(**base_payload) != build_cache_key(**changed_payload)


def test_lookup_cache_returns_cached_result():
    store = InMemoryEvalJobStore()
    payload = {
        "candidate_id": "cand-1",
        "system_prompt": "prompt-a",
        "user_template": "user",
        "tp_dataset": ["tp-1"],
        "tn_dataset": ["tn-1"],
        "metric_config": {"accuracy": True},
        "inference_config": {"temperature": 0.0},
    }
    expected = {"score": 0.91, "status": "done"}
    store.cache[build_cache_key(**payload)] = expected

    cache_hit, cache_value = store.lookup_cache(**payload)

    assert cache_hit is True
    assert cache_value == expected


def test_build_cache_key_is_stable_for_nested_dict_key_order():
    payload_a = {
        "candidate_id": "cand-1",
        "system_prompt": "prompt",
        "user_template": "user",
        "tp_dataset": ["tp-1"],
        "tn_dataset": ["tn-1"],
        "metric_config": {
            "thresholds": {"high": 0.9, "low": 0.1},
            "accuracy": True,
        },
        "inference_config": {
            "temperature": 0.0,
            "options": {"top_p": 1.0, "top_k": -1},
        },
    }
    payload_b = {
        "candidate_id": "cand-1",
        "system_prompt": "prompt",
        "user_template": "user",
        "tp_dataset": ["tp-1"],
        "tn_dataset": ["tn-1"],
        "metric_config": {
            "accuracy": True,
            "thresholds": {"low": 0.1, "high": 0.9},
        },
        "inference_config": {
            "options": {"top_k": -1, "top_p": 1.0},
            "temperature": 0.0,
        },
    }

    assert build_cache_key(**payload_a) == build_cache_key(**payload_b)


def test_create_job_returns_queued_when_cache_misses(tmp_path: Path):
    _write_dataset_files(tmp_path)
    store = InMemoryEvalJobStore()
    request = _make_request(tmp_path)

    job = store.create_job(request)

    assert job.status == "queued"
    assert job.cache_hit is False
    assert job.progress == 0.0


def test_create_job_returns_cached_when_cache_hits_even_for_none_result(tmp_path: Path):
    _write_dataset_files(tmp_path)
    store = InMemoryEvalJobStore()
    request = _make_request(tmp_path)
    store.cache[build_cache_key(**store._build_cache_payload(request))] = None

    job = store.create_job(request)

    assert job.status == "cached"
    assert job.cache_hit is True
    assert job.progress == 1.0
    assert job.result is None


def test_lookup_cache_returns_explicit_miss_for_missing_entry():
    store = InMemoryEvalJobStore()

    cache_hit, cache_value = store.lookup_cache(**_make_payload())

    assert cache_hit is False
    assert cache_value is None


def test_lookup_cache_returns_explicit_hit_for_cached_none():
    store = InMemoryEvalJobStore()
    store.cache[build_cache_key(**_make_payload())] = None

    cache_hit, cache_value = store.lookup_cache(**_make_payload())

    assert cache_hit is True
    assert cache_value is None


def test_cached_mutable_results_are_returned_as_isolated_copies(tmp_path: Path):
    _write_dataset_files(tmp_path)
    store = InMemoryEvalJobStore()
    payload = _make_payload(tmp_path)
    cached_value = {"scores": [1, 2], "meta": {"labels": ["a"]}}
    store.cache[build_cache_key(**payload)] = cached_value

    cache_hit, lookup_result = store.lookup_cache(**payload)
    assert cache_hit is True
    lookup_result["scores"].append(3)
    lookup_result["meta"]["labels"].append("b")

    request = _make_request(tmp_path)
    store.cache[build_cache_key(**store._build_cache_payload(request))] = cached_value
    job = store.create_job(request)

    assert cached_value == {"scores": [1, 2], "meta": {"labels": ["a"]}}
    assert lookup_result == {"scores": [1, 2, 3], "meta": {"labels": ["a", "b"]}}
    assert job.result == {"scores": [1, 2], "meta": {"labels": ["a"]}}
    assert job.result is not lookup_result


def test_non_json_serializable_cached_value_is_rejected():
    store = InMemoryEvalJobStore()
    payload = _make_payload()
    store.cache[build_cache_key(**payload)] = {"unsupported": {1, 2}}

    with pytest.raises(TypeError):
        store.lookup_cache(**payload)


def test_eval_job_request_defaults_config_dicts():
    request = EvalJobRequest(
        candidate_id="cand-1",
        system_prompt="prompt-a",
        user_template="user",
        tp_dataset="/tmp/tp.jsonl",
        tn_dataset="/tmp/tn.jsonl",
    )

    assert request.metric_config == {}
    assert request.inference_config == {}


def test_eval_job_status_defaults_optional_fields():
    status = EvalJobStatus(job_id="job-1", status="queued")

    assert status.progress == 0.0
    assert status.cache_hit is False
    assert status.result is None
    assert status.error is None


def _make_request(tmp_path: Path | None = None):
    return EvalJobRequest(**_make_payload(tmp_path))


def _make_payload(tmp_path: Path | None = None):
    if tmp_path is None:
        tmp_path = Path("/tmp")
    return {
        "candidate_id": "cand-1",
        "system_prompt": "prompt-a",
        "user_template": "user",
        "tp_dataset": str(tmp_path / "tp.jsonl"),
        "tn_dataset": str(tmp_path / "tn.jsonl"),
        "metric_config": {"accuracy": True},
        "inference_config": {"temperature": 0.0},
    }


def _write_dataset_files(tmp_path: Path) -> None:
    (tmp_path / "tp.jsonl").write_text(
        json.dumps({"id": "tp-1", "text": "alpha"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "tn.jsonl").write_text(
        json.dumps({"id": "tn-1", "text": "beta"}) + "\n",
        encoding="utf-8",
    )


class FakeRuntime:
    def __init__(self) -> None:
        self.calls = []

    def evaluate(self, *, system_prompt, user_template, dataset_records, inference_config):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_template": user_template,
                "dataset_records": dataset_records,
                "inference_config": inference_config,
            }
        )
        results = []
        for row in dataset_records:
            predicted_attack = row["id"].endswith("attack")
            results.append(
                {
                    "dataset": row["dataset"],
                    "sample_id": row["id"],
                    "predicted_attack": predicted_attack,
                    "raw_response": "B" if predicted_attack else "A",
                }
            )
        return results


class FailingRuntime:
    def evaluate(self, *, system_prompt, user_template, dataset_records, inference_config):
        raise RuntimeError("backend exploded")


def test_process_pending_jobs_completes_and_caches_result(tmp_path: Path):
    tp_path = tmp_path / "tp.jsonl"
    tn_path = tmp_path / "tn.jsonl"
    tp_path.write_text(json.dumps({"id": "tp-attack", "text": "alpha"}) + "\n", encoding="utf-8")
    tn_path.write_text(json.dumps({"id": "tn-normal", "text": "beta"}) + "\n", encoding="utf-8")

    runtime = FakeRuntime()
    store = InMemoryEvalJobStore(model_runtime=runtime)
    job = store.create_job(
        EvalJobRequest(
            candidate_id="cand-1",
            system_prompt="prompt-a",
            user_template="{text}",
            tp_dataset=str(tp_path),
            tn_dataset=str(tn_path),
            metric_config={"primary_metric": "accuracy"},
        )
    )

    assert job.status == "queued"

    store.process_pending_jobs()
    final_job = store.get_job(job.job_id)

    assert final_job is not None
    assert final_job.status == "completed"
    assert final_job.progress == 1.0
    assert final_job.result["merged_metrics"]["accuracy"] == 1.0
    assert final_job.result["dataset_fingerprint"]
    assert final_job.result["failure_samples"] == {"missed_tp": [], "false_alarm_tn": []}
    assert len(runtime.calls) == 1

    cached_job = store.create_job(
        EvalJobRequest(
            candidate_id="cand-1",
            system_prompt="prompt-a",
            user_template="{text}",
            tp_dataset=str(tp_path),
            tn_dataset=str(tn_path),
            metric_config={"primary_metric": "accuracy"},
        )
    )
    assert cached_job.status == "cached"
    assert cached_job.result["merged_metrics"]["accuracy"] == 1.0


def test_process_pending_jobs_marks_failures(tmp_path: Path):
    tp_path = tmp_path / "tp.jsonl"
    tn_path = tmp_path / "tn.jsonl"
    tp_path.write_text(json.dumps({"id": "tp-attack", "text": "alpha"}) + "\n", encoding="utf-8")
    tn_path.write_text(json.dumps({"id": "tn-normal", "text": "beta"}) + "\n", encoding="utf-8")

    store = InMemoryEvalJobStore(model_runtime=FailingRuntime())
    job = store.create_job(
        EvalJobRequest(
            candidate_id="cand-1",
            system_prompt="prompt-a",
            user_template="{text}",
            tp_dataset=str(tp_path),
            tn_dataset=str(tn_path),
        )
    )

    store.process_pending_jobs()
    final_job = store.get_job(job.job_id)

    assert final_job is not None
    assert final_job.status == "failed"
    assert "backend exploded" in final_job.error
