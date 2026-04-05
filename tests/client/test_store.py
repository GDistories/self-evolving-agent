from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from client.models import Candidate
from client.store import ExperimentStore


def test_store_saves_candidate_json(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    path = store.save_candidate(
        Candidate(
            candidate_id="cand-1",
            system_prompt="prompt-a",
            user_template="{text}",
        )
    )

    assert path.exists()
    assert '"candidate_id": "cand-1"' in path.read_text(encoding="utf-8")
    assert '"parent_candidate_id": null' in path.read_text(encoding="utf-8")
    assert '"mutation_note": ""' in path.read_text(encoding="utf-8")
    assert '"status": "draft"' in path.read_text(encoding="utf-8")


def test_store_rejects_unsafe_candidate_id(tmp_path: Path):
    store = ExperimentStore(tmp_path)

    try:
        store.save_candidate(
            Candidate(
                candidate_id="../../outside",
                system_prompt="prompt-a",
                user_template="{text}",
            )
        )
    except ValueError as exc:
        assert "unsafe candidate_id" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_store_rejects_backslash_based_candidate_id(tmp_path: Path):
    store = ExperimentStore(tmp_path)

    try:
        store.save_candidate(
            Candidate(
                candidate_id="..\\..\\outside",
                system_prompt="prompt-a",
                user_template="{text}",
            )
        )
    except ValueError as exc:
        assert "unsafe candidate_id" in str(exc)
    else:
        raise AssertionError("expected ValueError")
