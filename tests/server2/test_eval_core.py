from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server2.eval_core import summarize_binary_results


def test_summarize_binary_results_counts_tp_tn_fp_fn():
    rows = [
        {"dataset": "tp", "sample_id": "tp-1", "predicted_attack": True},
        {"dataset": "tp", "sample_id": "tp-2", "predicted_attack": False},
        {"dataset": "tn", "sample_id": "tn-1", "predicted_attack": False},
        {"dataset": "tn", "sample_id": "tn-2", "predicted_attack": True},
    ]

    result = summarize_binary_results(rows)

    assert result == {
        "tp_stats": {"tp": 1, "fn": 1, "total": 2},
        "tn_stats": {"tn": 1, "fp": 1, "total": 2},
        "merged_metrics": {"accuracy": 0.5},
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
