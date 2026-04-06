from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server2.eval_core import (
    build_dataset_fingerprint,
    load_dataset_records,
    parse_prediction_label,
    summarize_binary_results,
)


def test_summarize_binary_results_counts_tp_tn_fp_fn():
    rows = [
        {"dataset": "tp", "sample_id": "tp-1", "predicted_attack": True},
        {"dataset": "tp", "sample_id": "tp-2", "predicted_attack": False},
        {"dataset": "tn", "sample_id": "tn-1", "predicted_attack": False},
        {"dataset": "tn", "sample_id": "tn-2", "predicted_attack": True},
    ]

    result = summarize_binary_results(rows)

    assert result["tp_stats"] == {"tp": 1, "fn": 1, "total": 2}
    assert result["tn_stats"] == {"tn": 1, "fp": 1, "total": 2}
    assert result["merged_metrics"] == {"accuracy": 0.5}
    assert result["failure_samples"] == {
        "missed_tp": [{"sample_id": "tp-2", "raw_response": None}],
        "false_alarm_tn": [{"sample_id": "tn-2", "raw_response": None}],
    }


def test_summarize_binary_results_requires_dataset():
    rows = [{"sample_id": "tp-1", "predicted_attack": True}]

    try:
        summarize_binary_results(rows)
    except ValueError as exc:
        assert "dataset" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_summarize_binary_results_rejects_invalid_dataset():
    rows = [{"dataset": "maybe", "sample_id": "x", "predicted_attack": True}]

    try:
        summarize_binary_results(rows)
    except ValueError as exc:
        assert "invalid 'dataset'" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_summarize_binary_results_requires_predicted_attack():
    rows = [{"dataset": "tp", "sample_id": "tp-1"}]

    try:
        summarize_binary_results(rows)
    except ValueError as exc:
        assert "predicted_attack" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_summarize_binary_results_rejects_blank_predicted_attack():
    rows = [{"dataset": "tp", "sample_id": "tp-1", "predicted_attack": ""}]

    try:
        summarize_binary_results(rows)
    except ValueError as exc:
        assert "predicted_attack" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_dataset_records_reads_jsonl_and_tags_dataset(tmp_path: Path):
    dataset_path = tmp_path / "tp.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "tp-1", "text": "alpha"}),
                json.dumps({"id": "tp-2", "text": "beta"}),
            ]
        ),
        encoding="utf-8",
    )

    records = load_dataset_records(dataset_path, dataset="tp")

    assert records == [
        {"dataset": "tp", "id": "tp-1", "text": "alpha"},
        {"dataset": "tp", "id": "tp-2", "text": "beta"},
    ]


def test_build_dataset_fingerprint_depends_on_content_not_only_path(tmp_path: Path):
    dataset_path = tmp_path / "tp.jsonl"
    dataset_path.write_text(json.dumps({"id": "tp-1", "text": "alpha"}) + "\n", encoding="utf-8")

    original = build_dataset_fingerprint(dataset_path)
    dataset_path.write_text(json.dumps({"id": "tp-1", "text": "beta"}) + "\n", encoding="utf-8")
    changed = build_dataset_fingerprint(dataset_path)

    assert original != changed


def test_parse_prediction_label_maps_a_to_normal_and_other_labels_to_attack():
    assert parse_prediction_label("A") is False
    assert parse_prediction_label("B") is True
    assert parse_prediction_label("J") is True
    assert parse_prediction_label("not-a-label") is False
