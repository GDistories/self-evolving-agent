from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.models import Candidate
import client.run_loop as run_loop_module
from client.run_loop import run_loop


def _make_fake_experiment(tmp_path: Path, max_rounds: int):
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
                    "max_rounds": max_rounds,
                },
            )()

    return FakeExperiment()


def _install_fakes(
    monkeypatch,
    tmp_path: Path,
    load_experiment_capture: dict[str, Path],
    run_iteration_calls: list[dict[str, object]],
    outcomes: list[object],
) -> None:
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
            poll_interval_seconds = 2.0

        return Runtime()

    def fake_load_experiment_config(path: Path):
        load_experiment_capture["path"] = path
        return _make_fake_experiment(tmp_path, max_rounds=len(outcomes))

    def fake_run_iteration(**kwargs):
        run_iteration_calls.append(kwargs)
        assert kwargs["poll_interval_seconds"] == 2.0
        return outcomes[len(run_iteration_calls) - 1]

    monkeypatch.setattr("client.run_loop.load_runtime_config", fake_load_runtime_config)
    monkeypatch.setattr("client.run_loop.load_experiment_config", fake_load_experiment_config)
    monkeypatch.setattr("client.run_loop.OpenAIBrainClient", FakeBrainClient)
    monkeypatch.setattr("client.run_loop.EvaluatorClient", FakeEvaluatorClient)
    monkeypatch.setattr("client.run_loop.run_iteration", fake_run_iteration)


def test_run_loop_stops_after_max_rounds(tmp_path: Path, monkeypatch, capsys) -> None:
    load_experiment_capture: dict[str, Path] = {}
    run_iteration_calls: list[dict[str, object]] = []
    outcomes = [
        type(
            "IterationOutcome",
            (),
            {
                "best_candidate": Candidate(
                    candidate_id="cand-1",
                    system_prompt="new prompt",
                    user_template="{text}",
                    status="best",
                ),
                "challenger_candidate": Candidate(
                    candidate_id="cand-1",
                    system_prompt="new prompt",
                    user_template="{text}",
                ),
                "decision": type("Decision", (), {"promote": True})(),
                "metrics": {"score": 0.81},
            },
        )(),
        type(
            "IterationOutcome",
            (),
            {
                "best_candidate": Candidate(
                    candidate_id="cand-2",
                    system_prompt="new prompt 2",
                    user_template="{text}",
                    status="best",
                ),
                "challenger_candidate": Candidate(
                    candidate_id="cand-2",
                    system_prompt="new prompt 2",
                    user_template="{text}",
                ),
                "decision": type("Decision", (), {"promote": True})(),
                "metrics": {"score": 0.83},
            },
        )(),
        type(
            "IterationOutcome",
            (),
            {
                "best_candidate": Candidate(
                    candidate_id="cand-3",
                    system_prompt="new prompt 3",
                    user_template="{text}",
                    status="best",
                ),
                "challenger_candidate": Candidate(
                    candidate_id="cand-3",
                    system_prompt="new prompt 3",
                    user_template="{text}",
                ),
                "decision": type("Decision", (), {"promote": True})(),
                "metrics": {"score": 0.85},
            },
        )(),
    ]
    _install_fakes(
        monkeypatch,
        tmp_path,
        load_experiment_capture,
        run_iteration_calls,
        outcomes,
    )

    config_path = tmp_path / "experiment.json"
    run_loop(config_path)

    output = capsys.readouterr().out
    assert load_experiment_capture["path"] == config_path
    assert len(run_iteration_calls) == 3
    assert [call["best_candidate"].candidate_id for call in run_iteration_calls] == [
        "baseline-v1",
        "cand-1",
        "cand-2",
    ]
    assert [call["best_metrics"] for call in run_iteration_calls] == [
        {"score": 0.6},
        {"score": 0.81},
        {"score": 0.83},
    ]
    assert output.splitlines() == [
        "round=1 previous_best_candidate_id=baseline-v1 challenger_candidate_id=cand-1 promote=True current_best_candidate_id=cand-1",
        "round=2 previous_best_candidate_id=cand-1 challenger_candidate_id=cand-2 promote=True current_best_candidate_id=cand-2",
        "round=3 previous_best_candidate_id=cand-2 challenger_candidate_id=cand-3 promote=True current_best_candidate_id=cand-3",
        "final_best_candidate_id=cand-3",
        "final_best_metrics={'score': 0.85}",
    ]


def test_run_loop_keeps_existing_best_when_decision_is_not_promote(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    load_experiment_capture: dict[str, Path] = {}
    run_iteration_calls: list[dict[str, object]] = []
    outcomes = [
        type(
            "IterationOutcome",
            (),
            {
                "best_candidate": Candidate(
                    candidate_id="cand-1",
                    system_prompt="new prompt",
                    user_template="{text}",
                    status="best",
                ),
                "challenger_candidate": Candidate(
                    candidate_id="cand-1",
                    system_prompt="new prompt",
                    user_template="{text}",
                ),
                "decision": type("Decision", (), {"promote": False})(),
                "metrics": {"score": 0.55},
            },
        )(),
        type(
            "IterationOutcome",
            (),
            {
                "best_candidate": Candidate(
                    candidate_id="cand-2",
                    system_prompt="new prompt 2",
                    user_template="{text}",
                    status="best",
                ),
                "challenger_candidate": Candidate(
                    candidate_id="cand-2",
                    system_prompt="new prompt 2",
                    user_template="{text}",
                ),
                "decision": type("Decision", (), {"promote": True})(),
                "metrics": {"score": 0.7},
            },
        )(),
    ]

    def fake_load_experiment_config(path: Path):
        load_experiment_capture["path"] = path
        return _make_fake_experiment(tmp_path, max_rounds=2)

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
            poll_interval_seconds = 2.0

        return Runtime()

    def fake_run_iteration(**kwargs):
        run_iteration_calls.append(kwargs)
        if len(run_iteration_calls) == 1:
            assert kwargs["best_candidate"].candidate_id == "baseline-v1"
            assert kwargs["best_metrics"] == {"score": 0.6}
        else:
            assert kwargs["best_candidate"].candidate_id == "baseline-v1"
            assert kwargs["best_metrics"] == {"score": 0.6}
        assert kwargs["poll_interval_seconds"] == 2.0
        return outcomes[len(run_iteration_calls) - 1]

    monkeypatch.setattr("client.run_loop.load_runtime_config", fake_load_runtime_config)
    monkeypatch.setattr("client.run_loop.load_experiment_config", fake_load_experiment_config)
    monkeypatch.setattr("client.run_loop.OpenAIBrainClient", FakeBrainClient)
    monkeypatch.setattr("client.run_loop.EvaluatorClient", FakeEvaluatorClient)
    monkeypatch.setattr("client.run_loop.run_iteration", fake_run_iteration)

    run_loop(None)

    output = capsys.readouterr().out
    assert load_experiment_capture["path"] == Path(run_loop_module.__file__).with_name(
        "experiment.json"
    )
    assert len(run_iteration_calls) == 2
    assert [call["best_candidate"].candidate_id for call in run_iteration_calls] == [
        "baseline-v1",
        "baseline-v1",
    ]
    assert [call["best_metrics"] for call in run_iteration_calls] == [
        {"score": 0.6},
        {"score": 0.6},
    ]
    assert output.splitlines() == [
        "round=1 previous_best_candidate_id=baseline-v1 challenger_candidate_id=cand-1 promote=False current_best_candidate_id=baseline-v1",
        "round=2 previous_best_candidate_id=baseline-v1 challenger_candidate_id=cand-2 promote=True current_best_candidate_id=cand-2",
        "final_best_candidate_id=cand-2",
        "final_best_metrics={'score': 0.7}",
    ]
