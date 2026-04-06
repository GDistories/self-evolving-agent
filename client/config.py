from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    server1_base_url: str
    server2_base_url: str
    poll_interval_seconds: float = 2.0


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def load_runtime_config() -> RuntimeConfig:
    load_env_file(Path(__file__).with_name(".env"))

    poll_interval_raw = os.getenv("POLL_INTERVAL_SECONDS", "2.0")
    poll_interval_seconds = float(poll_interval_raw)
    if not math.isfinite(poll_interval_seconds) or poll_interval_seconds <= 0:
        raise ValueError(
            "POLL_INTERVAL_SECONDS must be a finite positive number, "
            f"got {poll_interval_raw!r}"
        )

    return RuntimeConfig(
        server1_base_url=os.getenv("SERVER1_BASE_URL", "http://127.0.0.1:18000"),
        server2_base_url=os.getenv("SERVER2_BASE_URL", "http://127.0.0.1:19000"),
        poll_interval_seconds=poll_interval_seconds,
    )
