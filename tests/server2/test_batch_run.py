from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server2.batch_run import run_batch_job


class FakeRuntime:
    def evaluate(self, *, system_prompt, user_template, dataset_records, inference_config):
        return [
            {
                "dataset": row["dataset"],
                "sample_id": row["id"],
                "predicted_attack": row["id"].endswith("attack"),
                "raw_response": "B" if row["id"].endswith("attack") else "A",
            }
            for row in dataset_records
        ]


def test_run_batch_job_returns_structured_result_and_writes_output(tmp_path: Path):
    tp_path = tmp_path / "tp.jsonl"
    tn_path = tmp_path / "tn.jsonl"
    output_path = tmp_path / "result.json"
    tp_path.write_text(json.dumps({"id": "tp-attack", "text": "alpha"}) + "\n", encoding="utf-8")
    tn_path.write_text(json.dumps({"id": "tn-normal", "text": "beta"}) + "\n", encoding="utf-8")

    result = run_batch_job(
        candidate_id="cand-1",
        system_prompt="system",
        user_template="{text}",
        tp_dataset=tp_path,
        tn_dataset=tn_path,
        model_runtime=FakeRuntime(),
        output_path=output_path,
    )

    assert result["candidate_id"] == "cand-1"
    assert result["merged_metrics"]["accuracy"] == 1.0
    assert result["failure_samples"] == {"missed_tp": [], "false_alarm_tn": []}
    assert json.loads(output_path.read_text(encoding="utf-8"))["merged_metrics"]["accuracy"] == 1.0
