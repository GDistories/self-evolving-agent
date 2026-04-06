from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from server2.model_runtime import ModelRuntime


class FakeBackend:
    init_calls = 0
    close_calls = 0

    def __init__(self, **config):
        type(self).init_calls += 1
        self.config = config

    def evaluate_dataset(self, *, system_prompt, user_template, dataset_records, inference_config):
        return [
            {
                "dataset": row["dataset"],
                "sample_id": row["id"],
                "predicted_attack": row["id"].endswith("attack"),
                "raw_response": "B" if row["id"].endswith("attack") else "A",
            }
            for row in dataset_records
        ]

    def close(self):
        type(self).close_calls += 1


def test_model_runtime_reuses_backend_in_lazy_reuse_mode():
    FakeBackend.init_calls = 0
    FakeBackend.close_calls = 0
    runtime = ModelRuntime(mode="lazy_reuse", backend_factory=FakeBackend, backend_config={"model_path": "fake"})

    first = runtime.evaluate(
        system_prompt="sys",
        user_template="{text}",
        dataset_records=[{"dataset": "tp", "id": "tp-attack", "text": "x"}],
        inference_config={},
    )
    second = runtime.evaluate(
        system_prompt="sys",
        user_template="{text}",
        dataset_records=[{"dataset": "tn", "id": "tn-normal", "text": "y"}],
        inference_config={},
    )

    assert len(first) == 1
    assert len(second) == 1
    assert FakeBackend.init_calls == 1
    assert FakeBackend.close_calls == 0


def test_model_runtime_creates_and_releases_backend_per_job():
    FakeBackend.init_calls = 0
    FakeBackend.close_calls = 0
    runtime = ModelRuntime(mode="per_job", backend_factory=FakeBackend, backend_config={"model_path": "fake"})

    runtime.evaluate(
        system_prompt="sys",
        user_template="{text}",
        dataset_records=[{"dataset": "tp", "id": "tp-attack", "text": "x"}],
        inference_config={},
    )
    runtime.evaluate(
        system_prompt="sys",
        user_template="{text}",
        dataset_records=[{"dataset": "tn", "id": "tn-normal", "text": "y"}],
        inference_config={},
    )

    assert FakeBackend.init_calls == 2
    assert FakeBackend.close_calls == 2
