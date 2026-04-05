from __future__ import annotations

from typing import Any

import httpx


class OpenAIBrainClient:
    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def propose_candidate(self, messages: list[dict[str, Any]]) -> dict[str, Any]:
        response = httpx.post(
            f"{self.base_url}/v1/chat/completions",
            json={"model": self.model, "messages": messages, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()
