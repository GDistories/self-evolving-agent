from __future__ import annotations

import json
from dataclasses import dataclass
from json import JSONDecodeError
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
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise ValueError(f"invalid experiment config JSON: {config_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("experiment config must be a JSON object")

    baseline_candidate = _require_mapping(payload, "baseline_candidate")
    best_metrics = _require_mapping(payload, "best_metrics")
    metric_config = _require_mapping(payload, "metric_config")
    runner = _require_mapping(payload, "runner")

    candidate_id = _require_nonempty_string(baseline_candidate, "candidate_id")
    system_prompt = _require_prompt_source(
        baseline_candidate,
        config_path.parent,
        field="system_prompt",
        inline_key="system_prompt",
        file_key="system_prompt_file",
    )
    user_template = _require_prompt_source(
        baseline_candidate,
        config_path.parent,
        field="user_template",
        inline_key="user_template",
        file_key="user_template_file",
    )
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


def _require_prompt_source(
    payload: dict[str, Any],
    config_dir: Path,
    *,
    field: str,
    inline_key: str,
    file_key: str,
) -> str:
    has_inline = inline_key in payload
    has_file = file_key in payload
    if has_inline and has_file:
        raise ValueError(
            f"{field} cannot specify both {inline_key} and {file_key}"
        )
    if not has_inline and not has_file:
        raise ValueError(
            f"missing required field: {field} (provide {inline_key} or {file_key})"
        )
    if has_inline:
        return _require_nonempty_string(payload, inline_key)

    file_path = _require_nonempty_string(payload, file_key)
    prompt_path = config_dir / file_path
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"{field} file could not be read: {prompt_path}") from exc
    if not prompt:
        raise ValueError(f"{field} must be a non-empty string")
    return prompt


def _require_positive_int(payload: dict[str, Any], key: str) -> int:
    if key not in payload:
        raise ValueError(f"missing required field: {key}")
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{key} must be a positive int")
    return value
