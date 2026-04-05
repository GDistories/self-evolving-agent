from __future__ import annotations

from pydantic import BaseModel


class Candidate(BaseModel):
    candidate_id: str
    system_prompt: str
    user_template: str
    parent_candidate_id: str | None = None
    mutation_note: str = ""
    status: str = "draft"


class JudgeDecision(BaseModel):
    promote: bool
    reason: str
