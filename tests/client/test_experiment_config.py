from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.experiment_config import load_experiment_config


def _write_config(tmp_path: Path, payload: str) -> Path:
    config_path = tmp_path / "experiment.json"
    config_path.write_text(payload.strip(), encoding="utf-8")
    return config_path


def _write_config_in_dir(config_dir: Path, payload: str) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    return _write_config(config_dir, payload)


def test_load_experiment_config_reads_inline_prompts(tmp_path: Path) -> None:
    config_path = _write_config(
        tmp_path,
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
""",
    )

    experiment = load_experiment_config(config_path)

    assert experiment.baseline_candidate.candidate_id == "baseline-v1"
    assert experiment.baseline_candidate.system_prompt == "classify attacks"
    assert experiment.baseline_candidate.user_template == "{text}"
    assert experiment.baseline_candidate.status == "best"
    assert experiment.best_metrics == {"score": 0.6}
    assert experiment.metric_config == {
        "primary_metric": "score",
        "min_value": 0.8,
        "tp_path": "data/tp.jsonl",
        "tn_path": "data/tn.jsonl",
    }
    assert experiment.runner.brain_model == "gpt-test"
    assert experiment.runner.store_root == "runs/demo"
    assert experiment.runner.max_rounds == 5


def test_load_experiment_config_reads_prompt_files_relative_to_config(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    (config_dir / "prompts").mkdir(parents=True)
    (config_dir / "prompts" / "system.md").write_text(
        "classify attacks from file",
        encoding="utf-8",
    )
    (config_dir / "prompts" / "user.md").write_text(
        "classify: {text}",
        encoding="utf-8",
    )
    config_path = _write_config_in_dir(
        config_dir,
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
""",
    )

    experiment = load_experiment_config(config_path)

    assert experiment.baseline_candidate.system_prompt == "classify attacks from file"
    assert experiment.baseline_candidate.user_template == "classify: {text}"


def test_load_experiment_config_rejects_empty_prompt_file_content(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    (config_dir / "prompts").mkdir(parents=True)
    (config_dir / "prompts" / "system.md").write_text("", encoding="utf-8")
    (config_dir / "prompts" / "user.md").write_text(
        "classify: {text}",
        encoding="utf-8",
    )
    config_path = _write_config_in_dir(
        config_dir,
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
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "system_prompt must be a non-empty string" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_missing_prompt_file(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    (config_dir / "prompts").mkdir(parents=True)
    (config_dir / "prompts" / "user.md").write_text(
        "classify: {text}",
        encoding="utf-8",
    )
    config_path = _write_config_in_dir(
        config_dir,
        """
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "system_prompt_file": "prompts/missing.md",
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
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "system_prompt file could not be read" in str(exc)
        assert "prompts/missing.md" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_both_inline_and_file_prompt_sources(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "system_prompt": "classify attacks",
    "system_prompt_file": "prompts/system.md",
    "user_template": "{text}"
  },
  "best_metrics": {"score": 0.6},
  "metric_config": {
    "primary_metric": "score",
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": 5
  }
}
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "system_prompt" in str(exc)
        assert "both" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_missing_prompt_source(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "user_template": "{text}"
  },
  "best_metrics": {"score": 0.6},
  "metric_config": {
    "primary_metric": "score",
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": 5
  }
}
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "system_prompt" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_invalid_max_rounds(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
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
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": 0
  }
}
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "max_rounds must be a positive int" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_malformed_section_type(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
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
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": []
}
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "runner must be an object" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_non_object_root(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path, "[1, 2, 3]")

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "experiment config must be a JSON object" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_missing_top_level_section(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
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
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  }
}
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "missing required field: runner" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_missing_nested_metric_field(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
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
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": 5
  }
}
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "missing required field: tp_path" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_missing_required_runner_field(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
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
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "store_root": "runs/demo",
    "max_rounds": 5
  }
}
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "missing required field: brain_model" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_empty_string_field(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
{
  "baseline_candidate": {
    "candidate_id": "",
    "system_prompt": "classify attacks",
    "user_template": "{text}"
  },
  "best_metrics": {"score": 0.6},
  "metric_config": {
    "primary_metric": "score",
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": 5
  }
}
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "candidate_id must be a non-empty string" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_wrong_typed_string_field(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
    "system_prompt": 123,
    "user_template": "{text}"
  },
  "best_metrics": {"score": 0.6},
  "metric_config": {
    "primary_metric": "score",
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": 5
  }
}
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "system_prompt must be a non-empty string" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_rejects_bool_max_rounds(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
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
    "tp_path": "data/tp.jsonl",
    "tn_path": "data/tn.jsonl"
  },
  "runner": {
    "brain_model": "gpt-test",
    "store_root": "runs/demo",
    "max_rounds": true
  }
}
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "max_rounds must be a positive int" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_experiment_config_wraps_invalid_json_as_value_error(
    tmp_path: Path,
) -> None:
    config_path = _write_config(
        tmp_path,
        """
{
  "baseline_candidate": {
    "candidate_id": "baseline-v1",
""",
    )

    try:
        load_experiment_config(config_path)
    except ValueError as exc:
        assert "invalid experiment config JSON" in str(exc)
    else:
        raise AssertionError("expected ValueError")
