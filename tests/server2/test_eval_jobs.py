from __future__ import annotations

import sys
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


def test_create_job_returns_queued_when_cache_misses():
    store = InMemoryEvalJobStore()
    request = _make_request()

    job = store.create_job(request)

    assert job.status == "queued"
    assert job.cache_hit is False
    assert job.progress == 0.0


def test_create_job_returns_cached_when_cache_hits_even_for_none_result():
    store = InMemoryEvalJobStore()
    request = _make_request()
    store.cache[build_cache_key(**_make_payload())] = None

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


def test_cached_mutable_results_are_returned_as_isolated_copies():
    store = InMemoryEvalJobStore()
    payload = _make_payload()
    cached_value = {"scores": [1, 2], "meta": {"labels": ["a"]}}
    store.cache[build_cache_key(**payload)] = cached_value

    cache_hit, lookup_result = store.lookup_cache(**payload)
    assert cache_hit is True
    lookup_result["scores"].append(3)
    lookup_result["meta"]["labels"].append("b")

    job = store.create_job(_make_request())

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
        tp_dataset=["tp-1"],
        tn_dataset=["tn-1"],
    )

    assert request.metric_config == {}
    assert request.inference_config == {}


def test_eval_job_status_defaults_optional_fields():
    status = EvalJobStatus(job_id="job-1", status="queued")

    assert status.progress == 0.0
    assert status.cache_hit is False
    assert status.result is None
    assert status.error is None


def _make_request():
    return EvalJobRequest(**_make_payload())


def _make_payload():
    return {
        "candidate_id": "cand-1",
        "system_prompt": "prompt-a",
        "user_template": "user",
        "tp_dataset": ["tp-1"],
        "tn_dataset": ["tn-1"],
        "metric_config": {"accuracy": True},
        "inference_config": {"temperature": 0.0},
    }
