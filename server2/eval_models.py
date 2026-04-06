from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvalJobRequest:
    candidate_id: str
    system_prompt: str
    user_template: str
    tp_dataset: str
    tn_dataset: str
    metric_config: dict[str, Any] = field(default_factory=dict)
    inference_config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvalJobStatus:
    job_id: str
    status: str
    progress: float = 0.0
    cache_hit: bool = False
    result: Any = None
    error: str | None = None
