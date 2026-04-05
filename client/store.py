from __future__ import annotations

from pathlib import Path

from client.models import Candidate


class ExperimentStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_candidate(self, candidate: Candidate) -> Path:
        self._validate_candidate_id(candidate.candidate_id)
        path = self.root / "candidates" / f"{candidate.candidate_id}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(candidate.model_dump_json(indent=2), encoding="utf-8")
        return path

    def _validate_candidate_id(self, candidate_id: str) -> None:
        candidate_path = Path(candidate_id)
        if "\\" in candidate_id:
            raise ValueError(f"unsafe candidate_id: {candidate_id!r}")
        if candidate_id != candidate_path.name or candidate_path.name in {"", ".", ".."}:
            raise ValueError(f"unsafe candidate_id: {candidate_id!r}")
        if candidate_path.anchor or len(candidate_path.parts) != 1:
            raise ValueError(f"unsafe candidate_id: {candidate_id!r}")
