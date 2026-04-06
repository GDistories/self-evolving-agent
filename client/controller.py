from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from client.judge import Judge
from client.models import Candidate, JudgeDecision
from client.store import ExperimentStore

TERMINAL_JOB_STATUSES = {"completed", "cached"}


@dataclass(slots=True)
class IterationOutcome:
    best_candidate: Candidate
    challenger_candidate: Candidate
    decision: JudgeDecision
    metrics: dict[str, Any]


def build_brain_messages(
    best_candidate: Candidate,
    best_metrics: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": "Propose one improved challenger candidate.",
        },
        {
            "role": "user",
            "content": {
                "best_candidate": best_candidate.model_dump(mode="json"),
                "best_metrics": best_metrics,
            },
        },
    ]


def _extract_metrics(job: dict[str, Any]) -> dict[str, Any]:
    if job.get("status") not in TERMINAL_JOB_STATUSES:
        raise RuntimeError(f"evaluation job not completed: {job.get('status')!r}")

    result = job.get("result")
    if not isinstance(result, dict):
        raise ValueError("evaluation job missing result")

    metrics = result.get("merged_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("evaluation job missing merged_metrics")

    return metrics


def run_iteration(
    best_candidate: Candidate,
    best_metrics: dict[str, Any],
    metric_config: dict[str, Any],
    brain_client: Any,
    evaluator_client: Any,
    store_root: Path,
    poll_interval_seconds: float = 2.0,
) -> IterationOutcome:
    store = ExperimentStore(store_root)
    brain_messages = build_brain_messages(best_candidate, best_metrics)
    proposal = brain_client.propose_candidate(messages=brain_messages)
    challenger = Candidate(**proposal["candidate"])
    store.save_candidate(challenger)

    payload = {
        "candidate_id": challenger.candidate_id,
        "system_prompt": challenger.system_prompt,
        "user_template": challenger.user_template,
        "tp_dataset": metric_config["tp_path"],
        "tn_dataset": metric_config["tn_path"],
        "metric_config": metric_config,
    }
    submitted_job = evaluator_client.submit_job(payload)
    if submitted_job.get("status") in TERMINAL_JOB_STATUSES and isinstance(
        submitted_job.get("result"), dict
    ):
        final_job = submitted_job
        metrics = _extract_metrics(submitted_job)
    else:
        job_id = submitted_job["job_id"]
        final_job = evaluator_client.get_job(job_id)
        while final_job.get("status") not in TERMINAL_JOB_STATUSES:
            time.sleep(poll_interval_seconds)
            final_job = evaluator_client.get_job(job_id)
        metrics = _extract_metrics(final_job)

    decision = Judge().compare(best_metrics, metrics, metric_config)
    outcome = IterationOutcome(
        best_candidate=challenger if decision.promote else best_candidate,
        challenger_candidate=challenger,
        decision=decision,
        metrics=metrics,
    )
    store.save_iteration_record(
        challenger.candidate_id,
        {
            "best_candidate_before": best_candidate.model_dump(mode="json"),
            "best_metrics_before": best_metrics,
            "brain_messages": brain_messages,
            "proposal": proposal,
            "challenger_candidate": challenger.model_dump(mode="json"),
            "evaluator_payload": payload,
            "submitted_job": submitted_job,
            "final_job": final_job,
            "decision": decision.model_dump(mode="json"),
            "metrics": metrics,
            "best_candidate_after": outcome.best_candidate.model_dump(mode="json"),
        },
    )
    return outcome
