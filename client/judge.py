from __future__ import annotations

from typing import Any

from client.models import JudgeDecision


class Judge:
    def compare(
        self,
        best_metrics: dict[str, Any],
        challenger_metrics: dict[str, Any],
        metric_config: dict[str, Any],
    ) -> JudgeDecision:
        primary_metric = metric_config["primary_metric"]
        min_value = metric_config.get("min_value")
        if primary_metric not in best_metrics:
            raise ValueError(f"missing primary metric in best_metrics: {primary_metric}")
        if primary_metric not in challenger_metrics:
            raise ValueError(
                f"missing primary metric in challenger_metrics: {primary_metric}"
            )
        challenger_value = challenger_metrics[primary_metric]
        best_value = best_metrics[primary_metric]

        if min_value is not None and challenger_value < min_value:
            return JudgeDecision(
                promote=False,
                reason="challenger does not meet threshold",
            )

        if challenger_value <= best_value:
            return JudgeDecision(
                promote=False,
                reason="challenger does not improve primary_metric",
            )

        return JudgeDecision(
            promote=True,
            reason="challenger meets threshold and improves primary_metric",
        )
