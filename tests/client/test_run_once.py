from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.models import Candidate
import client.run_once as run_once_module
from client.run_once import run_once


def _make_fake_experiment(tmp_path: Path):
    class FakeExperiment:
        def __init__(self) -> None:
            self.baseline_candidate = Candidate(
                candidate_id="baseline-v1",
                system_prompt="prompt",
                user_template="{text}",
                status="best",
            )
            self.best_metrics = {"score": 0.6}
            self.metric_config = {
                "primary_metric": "score",
                "min_value": 0.8,
                "tp_path": "data/tp.jsonl",
                "tn_path": "data/tn.jsonl",
            }
            self.runner = type(
                "RunnerConfig",
                (),
                {
                    "brain_model": "gpt-test",
                    "store_root": str(tmp_path),
                    "max_rounds": 5,
                },
            )()

    return FakeExperiment()


def _install_fakes(monkeypatch, tmp_path: Path, load_experiment_capture: dict[str, Path]) -> None:
    class FakeBrainClient:
        def __init__(self, base_url: str, model: str) -> None:
            self.base_url = base_url
            self.model = model

    class FakeEvaluatorClient:
        def __init__(self, base_url: str) -> None:
            self.base_url = base_url

    def fake_load_runtime_config():
        class Runtime:
            server1_base_url = "http://127.0.0.1:18000"
            server2_base_url = "http://127.0.0.1:19000"

        return Runtime()

    def fake_load_experiment_config(path: Path):
        load_experiment_capture["path"] = path
        return _make_fake_experiment(tmp_path)

    def fake_run_iteration(**kwargs):
        assert kwargs["best_candidate"].candidate_id == "baseline-v1"
        assert kwargs["best_metrics"] == {"score": 0.6}
        assert kwargs["metric_config"]["primary_metric"] == "score"
        assert kwargs["brain_client"].base_url == "http://127.0.0.1:18000"
        assert kwargs["brain_client"].model == "gpt-test"
        assert kwargs["evaluator_client"].base_url == "http://127.0.0.1:19000"
        assert kwargs["store_root"] == tmp_path
        return type(
            "IterationOutcome",
            (),
            {
                "best_candidate": Candidate(
                    candidate_id="cand-2",
                    system_prompt="new prompt",
                    user_template="{text}",
                    status="best",
                ),
                "challenger_candidate": Candidate(
                    candidate_id="cand-2",
                    system_prompt="new prompt",
                    user_template="{text}",
                ),
                "decision": type(
                    "Decision",
                    (),
                    {"promote": True, "reason": "improved"},
                )(),
                "metrics": {"score": 0.91},
            },
        )()

    monkeypatch.setattr("client.run_once.load_runtime_config", fake_load_runtime_config)
    monkeypatch.setattr("client.run_once.load_experiment_config", fake_load_experiment_config)
    monkeypatch.setattr("client.run_once.OpenAIBrainClient", FakeBrainClient)
    monkeypatch.setattr("client.run_once.EvaluatorClient", FakeEvaluatorClient)
    monkeypatch.setattr("client.run_once.run_iteration", fake_run_iteration)

    return None


def test_run_once_executes_single_iteration(tmp_path: Path, monkeypatch, capsys) -> None:
    load_experiment_capture: dict[str, Path] = {}
    _install_fakes(monkeypatch, tmp_path, load_experiment_capture)

    config_path = tmp_path / "experiment.json"
    run_once(config_path)

    output = capsys.readouterr().out
    assert load_experiment_capture["path"] == config_path
    assert output.splitlines() == [
        "baseline_candidate_id=baseline-v1",
        "challenger_candidate_id=cand-2",
        "promote=True",
        "metrics={'score': 0.91}",
    ]


def test_run_once_uses_default_experiment_path(tmp_path: Path, monkeypatch, capsys) -> None:
    load_experiment_capture: dict[str, Path] = {}
    _install_fakes(monkeypatch, tmp_path, load_experiment_capture)

    run_once(None)

    output = capsys.readouterr().out
    assert load_experiment_capture["path"] == Path(run_once_module.__file__).with_name(
        "experiment.json"
    )
    assert output.splitlines() == [
        "baseline_candidate_id=baseline-v1",
        "challenger_candidate_id=cand-2",
        "promote=True",
        "metrics={'score': 0.91}",
    ]
