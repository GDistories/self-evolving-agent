from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from client.models import Candidate


@dataclass(slots=True)
class RunnerConfig:
    brain_model: str
    store_root: str
    max_rounds: int


@dataclass(slots=True)
class ExperimentConfig:
    baseline_candidate: Candidate
    best_metrics: dict[str, Any]
    metric_config: dict[str, Any]
    runner: RunnerConfig


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object")

    baseline_candidate = _require_mapping(payload, "baseline_candidate")
    best_metrics = _require_mapping(payload, "best_metrics")
    metric_config = _require_mapping(payload, "metric_config")
    runner = _require_mapping(payload, "runner")

    candidate_id = _require_nonempty_string(baseline_candidate, "candidate_id")
    system_prompt = _require_nonempty_string(baseline_candidate, "system_prompt")
    user_template = _require_nonempty_string(baseline_candidate, "user_template")
    primary_metric = _require_nonempty_string(metric_config, "primary_metric")
    tp_path = _require_nonempty_string(metric_config, "tp_path")
    tn_path = _require_nonempty_string(metric_config, "tn_path")
    brain_model = _require_nonempty_string(runner, "brain_model")
    store_root = _require_nonempty_string(runner, "store_root")
    max_rounds = _require_positive_int(runner, "max_rounds")

    return ExperimentConfig(
        baseline_candidate=Candidate(
            candidate_id=candidate_id,
            system_prompt=system_prompt,
            user_template=user_template,
            status="best",
        ),
        best_metrics=best_metrics,
        metric_config=dict(metric_config),
        runner=RunnerConfig(
            brain_model=brain_model,
            store_root=store_root,
            max_rounds=max_rounds,
        ),
    )


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"{key} must be an object")
    return value


def _require_nonempty_string(payload: dict[str, Any], key: str) -> str:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _require_positive_int(payload: dict[str, Any], key: str) -> int:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive int")
    return value
