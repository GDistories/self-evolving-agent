from __future__ import annotations

import sys
from pathlib import Path

from client.config import load_runtime_config
from client.controller import run_iteration
from client.evaluator_client import EvaluatorClient
from client.experiment_config import load_experiment_config
from client.openai_brain import OpenAIBrainClient


def run_loop(config_path: Path | None = None) -> None:
    runtime = load_runtime_config()
    experiment_path = config_path or Path(__file__).with_name("experiment.json")
    experiment = load_experiment_config(experiment_path)
    brain_client = OpenAIBrainClient(
        base_url=runtime.server1_base_url,
        model=experiment.runner.brain_model,
    )
    evaluator_client = EvaluatorClient(base_url=runtime.server2_base_url)

    best_candidate = experiment.baseline_candidate
    best_metrics = experiment.best_metrics

    for round_number in range(1, experiment.runner.max_rounds + 1):
        previous_best_candidate = best_candidate
        outcome = run_iteration(
            best_candidate=best_candidate,
            best_metrics=best_metrics,
            metric_config=experiment.metric_config,
            brain_client=brain_client,
            evaluator_client=evaluator_client,
            store_root=Path(experiment.runner.store_root),
        )

        if outcome.decision.promote:
            best_candidate = outcome.best_candidate
            best_metrics = outcome.metrics

        print(
            "round="
            f"{round_number} "
            f"previous_best_candidate_id={previous_best_candidate.candidate_id} "
            f"challenger_candidate_id={outcome.challenger_candidate.candidate_id} "
            f"promote={outcome.decision.promote} "
            f"current_best_candidate_id={best_candidate.candidate_id}"
        )

    print(f"final_best_candidate_id={best_candidate.candidate_id}")
    print(f"final_best_metrics={best_metrics}")


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_loop(config_arg)
