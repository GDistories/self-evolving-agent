from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.judge import Judge


def test_judge_promotes_candidate_that_meets_threshold_and_beats_best():
    judge = Judge()

    decision = judge.compare(
        best_metrics={"score": 0.70},
        challenger_metrics={"score": 0.81},
        metric_config={"primary_metric": "score", "min_value": 0.80},
    )

    assert decision.promote is True
    assert decision.reason == "challenger meets threshold and improves primary_metric"


def test_judge_rejects_candidate_below_threshold():
    judge = Judge()

    decision = judge.compare(
        best_metrics={"score": 0.70},
        challenger_metrics={"score": 0.79},
        metric_config={"primary_metric": "score", "min_value": 0.80},
    )

    assert decision.promote is False
    assert decision.reason == "challenger does not meet threshold"


def test_judge_rejects_candidate_that_does_not_beat_best():
    judge = Judge()

    decision = judge.compare(
        best_metrics={"score": 0.81},
        challenger_metrics={"score": 0.81},
        metric_config={"primary_metric": "score", "min_value": 0.80},
    )

    assert decision.promote is False
    assert decision.reason == "challenger does not improve primary_metric"


def test_judge_rejects_missing_primary_metric():
    judge = Judge()

    try:
        judge.compare(
            best_metrics={},
            challenger_metrics={"score": 0.81},
            metric_config={"primary_metric": "score", "min_value": 0.80},
        )
    except ValueError as exc:
        assert "primary metric" in str(exc)
    else:
        raise AssertionError("expected ValueError")
