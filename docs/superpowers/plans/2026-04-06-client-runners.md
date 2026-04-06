# Client Runners Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `client/run_once.py` and `client/run_loop.py` with shared experiment-config loading that supports inline prompts and Markdown-backed prompt files.

**Architecture:** Keep the current runtime URL loading in `client/config.py`, add a separate experiment-config loader in the `client` package, and make both runners thin orchestration entrypoints over existing `run_iteration(...)`. Extend tests around config parsing and runner orchestration before adding implementation.

**Tech Stack:** Python, Pydantic, pytest, existing `client` library modules

---

## File Structure

- Create: `client/experiment_config.py`
  - Owns experiment JSON parsing, prompt-source resolution, and validation.
- Create: `client/run_once.py`
  - Thin smoke-test entrypoint that loads config, runs one iteration, and prints a concise summary.
- Create: `client/run_loop.py`
  - Thin multi-round entrypoint that loads config, runs repeated iterations, and prints per-round plus final summaries.
- Create: `client/experiment.json.example`
  - Documents the supported experiment config shape for inline and file-backed prompt sources.
- Modify: `README.md`
  - Document how to start the new client runners and where prompt/config values live.
- Modify: `tests/client/test_controller.py`
  - Keep existing controller/runtime tests and add focused tests for experiment config and runner helpers if practical.
- Create: `tests/client/test_experiment_config.py`
  - Covers parsing, relative-path resolution, and validation failures.
- Create: `tests/client/test_run_once.py`
  - Covers one-shot runner orchestration and output.
- Create: `tests/client/test_run_loop.py`
  - Covers looping behavior, promotion carry-forward, and stop at `max_rounds`.

### Task 1: Add experiment config tests

**Files:**
- Create: `tests/client/test_experiment_config.py`
- Modify: `client/experiment_config.py`

- [ ] **Step 1: Write the failing test for inline prompt loading**

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.experiment_config import load_experiment_config


def test_load_experiment_config_reads_inline_prompts(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        """
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "system_prompt": "classify attacks",
    "user_template": "{text}"
  },
  "best_metrics": {"score": 0.6},
  "metric_config": {
    "primary_metric": "score",
    "min_value": 0.8,
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": 5
  }
}
""".strip(),
        encoding="utf-8",
    )

    experiment = load_experiment_config(config_path)

    assert experiment.baseline_candidate.candidate_id == "baseline-v1"
    assert experiment.baseline_candidate.system_prompt == "classify attacks"
    assert experiment.baseline_candidate.user_template == "{text}"
    assert experiment.best_metrics == {"score": 0.6}
    assert experiment.metric_config["tp_path"] == "data/tp.jsonl"
    assert experiment.runner.brain_model == "gpt-test"
    assert experiment.runner.max_rounds == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/client/test_experiment_config.py::test_load_experiment_config_reads_inline_prompts -v`
Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `client.experiment_config`

- [ ] **Step 3: Write minimal implementation for inline loading**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from client.models import Candidate


@dataclass(frozen=True, slots=True)
class RunnerConfig:
    brain_model: str
    store_root: str
    max_rounds: int


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    baseline_candidate: Candidate
    best_metrics: dict[str, Any]
    metric_config: dict[str, Any]
    runner: RunnerConfig


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    baseline = payload["baseline_candidate"]
    runner = payload["runner"]
    return ExperimentConfig(
        baseline_candidate=Candidate(
            candidate_id=baseline["candidate_id"],
            system_prompt=baseline["system_prompt"],
            user_template=baseline["user_template"],
            status="best",
        ),
        best_metrics=payload["best_metrics"],
        metric_config=payload["metric_config"],
        runner=RunnerConfig(
            brain_model=runner["brain_model"],
            store_root=runner["store_root"],
            max_rounds=int(runner["max_rounds"]),
        ),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/client/test_experiment_config.py::test_load_experiment_config_reads_inline_prompts -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/client/test_experiment_config.py client/experiment_config.py
git commit -m "feat: add inline experiment config loading"
```

### Task 2: Add file-backed prompt loading and validation tests

**Files:**
- Modify: `tests/client/test_experiment_config.py`
- Modify: `client/experiment_config.py`

- [ ] **Step 1: Write the failing tests for file-backed prompts and invalid prompt source combinations**

```python
def test_load_experiment_config_reads_prompt_files_relative_to_config(
    tmp_path: Path,
) -> None:
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "system.md").write_text("# system\nclassify carefully", encoding="utf-8")
    (prompts_dir / "user.md").write_text("Input:\n{text}", encoding="utf-8")
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        """
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "system_prompt_file": "prompts/system.md",
    "user_template_file": "prompts/user.md"
  },
  "best_metrics": {"score": 0.6},
  "metric_config": {
    "primary_metric": "score",
    "min_value": 0.8,
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": 5
  }
}
""".strip(),
        encoding="utf-8",
    )

    experiment = load_experiment_config(config_path)

    assert experiment.baseline_candidate.system_prompt == "# system\nclassify carefully"
    assert experiment.baseline_candidate.user_template == "Input:\n{text}"


def test_load_experiment_config_rejects_both_inline_and_file_prompt_sources(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        """
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "system_prompt": "inline",
    "system_prompt_file": "prompts/system.md",
    "user_template": "{text}"
  },
  "best_metrics": {"score": 0.6},
  "metric_config": {
    "primary_metric": "score",
    "min_value": 0.8,
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": 5
  }
}
""".strip(),
        encoding="utf-8",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "system_prompt" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_missing_prompt_source(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        """
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "user_template": "{text}"
  },
  "best_metrics": {"score": 0.6},
  "metric_config": {
    "primary_metric": "score",
    "min_value": 0.8,
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": 5
  }
}
""".strip(),
        encoding="utf-8",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "system_prompt" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/client/test_experiment_config.py -v`
Expected: FAIL because file-backed prompt loading and validation are not implemented yet

- [ ] **Step 3: Extend experiment config loader with prompt-source resolution and validation**

```python
def _resolve_prompt_text(
    *,
    config_dir: Path,
    payload: dict[str, Any],
    inline_key: str,
    file_key: str,
) -> str:
    inline_value = payload.get(inline_key)
    file_value = payload.get(file_key)

    if inline_value and file_value:
        raise ValueError(f"{inline_key} and {file_key} are mutually exclusive")
    if not inline_value and not file_value:
        raise ValueError(f"missing prompt source for {inline_key}")
    if inline_value:
        return str(inline_value)

    resolved_path = (config_dir / str(file_value)).resolve()
    return resolved_path.read_text(encoding="utf-8")


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    baseline = payload["baseline_candidate"]
    runner = payload["runner"]
    config_dir = config_path.resolve().parent

    return ExperimentConfig(
        baseline_candidate=Candidate(
            candidate_id=baseline["candidate_id"],
            system_prompt=_resolve_prompt_text(
                config_dir=config_dir,
                payload=baseline,
                inline_key="system_prompt",
                file_key="system_prompt_file",
            ),
            user_template=_resolve_prompt_text(
                config_dir=config_dir,
                payload=baseline,
                inline_key="user_template",
                file_key="user_template_file",
            ),
            status="best",
        ),
        best_metrics=payload["best_metrics"],
        metric_config=payload["metric_config"],
        runner=RunnerConfig(
            brain_model=runner["brain_model"],
            store_root=runner["store_root"],
            max_rounds=int(runner["max_rounds"]),
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/client/test_experiment_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/client/test_experiment_config.py client/experiment_config.py
git commit -m "feat: support file-backed prompt config"
```

### Task 3: Validate required runner and metric fields

**Files:**
- Modify: `tests/client/test_experiment_config.py`
- Modify: `client/experiment_config.py`

- [ ] **Step 1: Write the failing tests for missing runner fields and invalid `max_rounds`**

```python
def test_load_experiment_config_rejects_missing_brain_model(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        """
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "system_prompt": "prompt",
    "user_template": "{text}"
  },
  "best_metrics": {"score": 0.6},
  "metric_config": {
    "primary_metric": "score",
    "min_value": 0.8,
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "store_root": "runs/demo",
    "max_rounds": 5
  }
}
""".strip(),
        encoding="utf-8",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "brain_model" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_non_positive_max_rounds(tmp_path: Path) -> None:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(
        """
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "system_prompt": "prompt",
    "user_template": "{text}"
  },
  "best_metrics": {"score": 0.6},
  "metric_config": {
    "primary_metric": "score",
    "min_value": 0.8,
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": 0
  }
}
""".strip(),
        encoding="utf-8",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "max_rounds" in str(exc)
    else:
        raise AssertionError("expected ValueError")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/client/test_experiment_config.py -v`
Expected: FAIL because loader does not validate the required runner fields yet

- [ ] **Step 3: Add explicit validation helpers for required fields**

```python
def _require_mapping_value(payload: dict[str, Any], key: str) -> Any:
    value = payload.get(key)
    if value in (None, ""):
        raise ValueError(f"missing required config field: {key}")
    return value


def _validate_max_rounds(raw_value: Any) -> int:
    max_rounds = int(raw_value)
    if max_rounds <= 0:
        raise ValueError(f"max_rounds must be positive, got {raw_value!r}")
    return max_rounds


def load_experiment_config(config_path: Path) -> ExperimentConfig:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    baseline = payload["baseline_candidate"]
    runner = payload["runner"]
    metric_config = payload["metric_config"]

    for key in ("primary_metric", "tp_path", "tn_path"):
        _require_mapping_value(metric_config, key)

    return ExperimentConfig(
        baseline_candidate=Candidate(
            candidate_id=str(_require_mapping_value(baseline, "candidate_id")),
            system_prompt=_resolve_prompt_text(
                config_dir=config_path.resolve().parent,
                payload=baseline,
                inline_key="system_prompt",
                file_key="system_prompt_file",
            ),
            user_template=_resolve_prompt_text(
                config_dir=config_path.resolve().parent,
                payload=baseline,
                inline_key="user_template",
                file_key="user_template_file",
            ),
            status="best",
        ),
        best_metrics=dict(payload["best_metrics"]),
        metric_config=dict(metric_config),
        runner=RunnerConfig(
            brain_model=str(_require_mapping_value(runner, "brain_model")),
            store_root=str(_require_mapping_value(runner, "store_root")),
            max_rounds=_validate_max_rounds(_require_mapping_value(runner, "max_rounds")),
        ),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/client/test_experiment_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/client/test_experiment_config.py client/experiment_config.py
git commit -m "feat: validate experiment runner config"
```

### Task 4: Add a smoke-test runner

**Files:**
- Create: `tests/client/test_run_once.py`
- Create: `client/run_once.py`

- [ ] **Step 1: Write the failing test for one-shot runner orchestration**

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.models import Candidate
from client.run_once import run_once


def test_run_once_executes_single_iteration(tmp_path: Path, monkeypatch, capsys) -> None:
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
                {"brain_model": "gpt-test", "store_root": str(tmp_path), "max_rounds": 5},
            )()

    def fake_run_iteration(**kwargs):
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
                "decision": type("Decision", (), {"promote": True, "reason": "improved"})(),
                "metrics": {"score": 0.91},
            },
        )()

    monkeypatch.setattr("client.run_once.load_runtime_config", fake_load_runtime_config)
    monkeypatch.setattr("client.run_once.load_experiment_config", lambda path: FakeExperiment())
    monkeypatch.setattr("client.run_once.OpenAIBrainClient", FakeBrainClient)
    monkeypatch.setattr("client.run_once.EvaluatorClient", FakeEvaluatorClient)
    monkeypatch.setattr("client.run_once.run_iteration", fake_run_iteration)

    run_once(tmp_path / "experiment.json")

    output = capsys.readouterr().out
    assert "baseline-v1" in output
    assert "cand-2" in output
    assert "score" in output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/client/test_run_once.py::test_run_once_executes_single_iteration -v`
Expected: FAIL with `ImportError` for `client.run_once`

- [ ] **Step 3: Implement `client/run_once.py` as a thin runner**

```python
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
    )
    print(f"baseline_candidate_id={experiment.baseline_candidate.candidate_id}")
    print(f"challenger_candidate_id={outcome.challenger_candidate.candidate_id}")
    print(f"promote={outcome.decision.promote}")
    print(f"metrics={outcome.metrics}")


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_once(config_arg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/client/test_run_once.py::test_run_once_executes_single_iteration -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/client/test_run_once.py client/run_once.py
git commit -m "feat: add one-shot client runner"
```

### Task 5: Add a multi-round runner

**Files:**
- Create: `tests/client/test_run_loop.py`
- Create: `client/run_loop.py`

- [ ] **Step 1: Write the failing tests for loop stop and promotion carry-forward**

```python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.models import Candidate
from client.run_loop import run_loop


def test_run_loop_stops_after_max_rounds(tmp_path: Path, monkeypatch, capsys) -> None:
    call_count = {"count": 0}

    def fake_load_runtime_config():
        class Runtime:
            server1_base_url = "http://127.0.0.1:18000"
            server2_base_url = "http://127.0.0.1:19000"
        return Runtime()

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
                {"brain_model": "gpt-test", "store_root": str(tmp_path), "max_rounds": 3},
            )()

    def fake_run_iteration(**kwargs):
        call_count["count"] += 1
        score = 0.7 + (0.1 * call_count["count"])
        candidate_id = f"cand-{call_count['count']}"
        return type(
            "IterationOutcome",
            (),
            {
                "best_candidate": Candidate(
                    candidate_id=candidate_id,
                    system_prompt=f"prompt-{call_count['count']}",
                    user_template="{text}",
                    status="best",
                ),
                "challenger_candidate": Candidate(
                    candidate_id=candidate_id,
                    system_prompt=f"prompt-{call_count['count']}",
                    user_template="{text}",
                ),
                "decision": type("Decision", (), {"promote": True, "reason": "improved"})(),
                "metrics": {"score": score},
            },
        )()

    monkeypatch.setattr("client.run_loop.load_runtime_config", fake_load_runtime_config)
    monkeypatch.setattr("client.run_loop.load_experiment_config", lambda path: FakeExperiment())
    monkeypatch.setattr("client.run_loop.OpenAIBrainClient", lambda **kwargs: object())
    monkeypatch.setattr("client.run_loop.EvaluatorClient", lambda **kwargs: object())
    monkeypatch.setattr("client.run_loop.run_iteration", fake_run_iteration)

    run_loop(tmp_path / "experiment.json")

    output = capsys.readouterr().out
    assert call_count["count"] == 3
    assert "round=3" in output
    assert "final_best_candidate_id=cand-3" in output


def test_run_loop_keeps_existing_best_when_decision_is_not_promote(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_best_ids: list[str] = []

    def fake_load_runtime_config():
        class Runtime:
            server1_base_url = "http://127.0.0.1:18000"
            server2_base_url = "http://127.0.0.1:19000"
        return Runtime()

    class FakeExperiment:
        def __init__(self) -> None:
            self.baseline_candidate = Candidate(
                candidate_id="baseline-v1",
                system_prompt="prompt",
                user_template="{text}",
                status="best",
            )
            self.best_metrics = {"score": 0.9}
            self.metric_config = {
                "primary_metric": "score",
                "min_value": 0.8,
                "tp_path": "data/tp.jsonl",
                "tn_path": "data/tn.jsonl",
            }
            self.runner = type(
                "RunnerConfig",
                (),
                {"brain_model": "gpt-test", "store_root": str(tmp_path), "max_rounds": 2},
            )()

    def fake_run_iteration(**kwargs):
        captured_best_ids.append(kwargs["best_candidate"].candidate_id)
        return type(
            "IterationOutcome",
            (),
            {
                "best_candidate": kwargs["best_candidate"],
                "challenger_candidate": Candidate(
                    candidate_id="cand-x",
                    system_prompt="weaker",
                    user_template="{text}",
                ),
                "decision": type("Decision", (), {"promote": False, "reason": "worse"})(),
                "metrics": {"score": 0.7},
            },
        )()

    monkeypatch.setattr("client.run_loop.load_runtime_config", fake_load_runtime_config)
    monkeypatch.setattr("client.run_loop.load_experiment_config", lambda path: FakeExperiment())
    monkeypatch.setattr("client.run_loop.OpenAIBrainClient", lambda **kwargs: object())
    monkeypatch.setattr("client.run_loop.EvaluatorClient", lambda **kwargs: object())
    monkeypatch.setattr("client.run_loop.run_iteration", fake_run_iteration)

    run_loop(tmp_path / "experiment.json")

    assert captured_best_ids == ["baseline-v1", "baseline-v1"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/client/test_run_loop.py -v`
Expected: FAIL with `ImportError` for `client.run_loop`

- [ ] **Step 3: Implement `client/run_loop.py` with `max_rounds` control**

```python
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

    best_candidate = experiment.baseline_candidate
    best_metrics = dict(experiment.best_metrics)
    brain_client = OpenAIBrainClient(
        base_url=runtime.server1_base_url,
        model=experiment.runner.brain_model,
    )
    evaluator_client = EvaluatorClient(base_url=runtime.server2_base_url)
    store_root = Path(experiment.runner.store_root)

    for round_number in range(1, experiment.runner.max_rounds + 1):
        outcome = run_iteration(
            best_candidate=best_candidate,
            best_metrics=best_metrics,
            metric_config=experiment.metric_config,
            brain_client=brain_client,
            evaluator_client=evaluator_client,
            store_root=store_root,
        )
        print(f"round={round_number}")
        print(f"previous_best_candidate_id={best_candidate.candidate_id}")
        print(f"challenger_candidate_id={outcome.challenger_candidate.candidate_id}")
        print(f"promote={outcome.decision.promote}")
        best_candidate = outcome.best_candidate
        best_metrics = outcome.metrics if outcome.decision.promote else best_metrics
        print(f"current_best_candidate_id={best_candidate.candidate_id}")

    print(f"final_best_candidate_id={best_candidate.candidate_id}")
    print(f"final_best_metrics={best_metrics}")


if __name__ == "__main__":
    config_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run_loop(config_arg)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/client/test_run_loop.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/client/test_run_loop.py client/run_loop.py
git commit -m "feat: add multi-round client runner"
```

### Task 6: Add example config and startup docs

**Files:**
- Create: `client/experiment.json.example`
- Modify: `README.md`

- [ ] **Step 1: Write the failing documentation checks by defining the exact examples to add**

```json
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "system_prompt_file": "prompts/system.md",
    "user_template": "{text}"
  },
  "best_metrics": {
    "score": 0.0
  },
  "metric_config": {
    "primary_metric": "score",
    "min_value": 0.8,
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "Qwen/Qwen3.5-35B-A3B",
    "store_root": "runs/default",
    "max_rounds": 20
  }
}
```

```markdown
6. Run the local client smoke test.

```bash
cp client/.env_example client/.env
cp client/experiment.json.example client/experiment.json
python -m client.run_once
```

7. Run the formal local optimization loop.

```bash
python -m client.run_loop
```

Prompt content can be supplied either inline in `client/experiment.json` or by using `system_prompt_file` / `user_template_file` with paths relative to `client/experiment.json`.
```

- [ ] **Step 2: Add the example config file**

```json
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "system_prompt_file": "prompts/system.md",
    "user_template": "{text}"
  },
  "best_metrics": {
    "score": 0.0
  },
  "metric_config": {
    "primary_metric": "score",
    "min_value": 0.8,
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "Qwen/Qwen3.5-35B-A3B",
    "store_root": "runs/default",
    "max_rounds": 20
  }
}
```

- [ ] **Step 3: Update `README.md` startup instructions**

```markdown
5. Configure the local client runtime and experiment definition.

```bash
cat > client/.env <<'EOF'
SERVER1_BASE_URL=http://127.0.0.1:18000
SERVER2_BASE_URL=http://127.0.0.1:19000
POLL_INTERVAL_SECONDS=2.0
EOF
cp client/experiment.json.example client/experiment.json
```

`client/.env` only controls local runtime URLs.
`client/experiment.json` controls the baseline candidate, datasets, metric policy, model name, store path, and loop count.

Prompt text can be configured either:

- inline with `system_prompt` and `user_template`
- from files with `system_prompt_file` and `user_template_file`

Paths for prompt files are resolved relative to `client/experiment.json`.

6. Run a one-shot smoke test.

```bash
python -m client.run_once
```

7. Run the formal multi-round loop.

```bash
python -m client.run_loop
```
```

- [ ] **Step 4: Verify the docs and examples read cleanly**

Run: `sed -n '1,260p' README.md`
Expected: README shows both runner commands and clearly separates `.env` from `experiment.json`

- [ ] **Step 5: Commit**

```bash
git add client/experiment.json.example README.md
git commit -m "docs: add client runner startup instructions"
```

### Task 7: Run the targeted verification suite

**Files:**
- Modify: `tests/client/test_controller.py`
- Test: `tests/client/test_experiment_config.py`
- Test: `tests/client/test_run_once.py`
- Test: `tests/client/test_run_loop.py`

- [ ] **Step 1: Add any small helper assertions needed to keep runner tests readable**

```python
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
```

- [ ] **Step 2: Run the focused client test suite**

Run: `pytest tests/client/test_controller.py tests/client/test_experiment_config.py tests/client/test_run_once.py tests/client/test_run_loop.py -v`
Expected: PASS

- [ ] **Step 3: Run the full existing client test directory**

Run: `pytest tests/client -v`
Expected: PASS

- [ ] **Step 4: Inspect the working tree**

Run: `git status --short`
Expected: only the intended client/docs/test files are modified or added

- [ ] **Step 5: Commit**

```bash
git add client tests README.md
git commit -m "feat: add configurable client runners"
```

## Self-Review

### Spec coverage

- Two runners: covered by Task 4 and Task 5.
- Separate runtime and experiment config layers: covered by Task 1 through Task 3 and Task 6.
- Prompt inline/file dual mode with Markdown support: covered by Task 2.
- Fail-fast validation: covered by Task 2 and Task 3.
- User-facing startup guidance: covered by Task 6.
- Testing expectations from the spec: covered by Task 1 through Task 5 and verified in Task 7.

### Placeholder scan

No `TBD`, `TODO`, or deferred “handle later” steps remain. Each task names exact files, commands, and code snippets.

### Type consistency

- Shared loader API is consistently named `load_experiment_config`.
- Shared config dataclasses are consistently named `ExperimentConfig` and `RunnerConfig`.
- Runner entrypoints are consistently named `run_once` and `run_loop`.
- `Candidate`, `run_iteration(...)`, `OpenAIBrainClient`, and `EvaluatorClient` names match the existing codebase.
