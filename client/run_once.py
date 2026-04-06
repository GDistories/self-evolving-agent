from __future__ import annotations

import sys
from pathlib import Path

from client.config import load_runtime_config
from client.controller import run_iteration
from client.evaluator_client import EvaluatorClient
from client.experiment_config import load_experiment_config
from client.openai_brain import OpenAIBrainClient


def run_once(config_path: Path | None = None) -> None:
    runtime = load_runtime_config()
    experiment_path = config_path or Path(__file__).with_name("experiment.json")
    experiment = load_experiment_config(experiment_path)
    outcome = run_iteration(
        best_candidate=experiment.baseline_candidate,
        best_metrics=experiment.best_metrics,
        metric_config=experiment.metric_config,
        brain_client=OpenAIBrainClient(
            base_url=runtime.server1_base_url,
            model=experiment.runner.brain_model,
        ),
        evaluator_client=EvaluatorClient(base_url=runtime.server2_base_url),
        store_root=Path(experiment.runner.store_root),
        poll_interval_seconds=runtime.poll_interval_seconds,
    )

    print(f"baseline_candidate_id={experiment.baseline_candidate.candidate_id}")
    print(f"challenger_candidate_id={outcome.challenger_candidate.candidate_id}")
    print(f"promote={outcome.decision.promote}")
    print(f"metrics={outcome.metrics}")


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_once(config_arg)
