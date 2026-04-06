from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bridge.server1.bridge import load_env_file, resolve_cookie_file_path


def test_load_env_file_sets_missing_values_and_strips_quotes(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        """
        # comment
        REMOTE_WSS_URL=wss://example/ws
        MODEL_NAME='test-model'
        LOCAL_PORT="19000"
        MALFORMED_LINE
        """.strip(),
        encoding="utf-8",
    )
    monkeypatch.delenv("REMOTE_WSS_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    monkeypatch.delenv("LOCAL_PORT", raising=False)

    load_env_file(env_file)

    assert os.environ["REMOTE_WSS_URL"] == "wss://example/ws"
    assert os.environ["MODEL_NAME"] == "test-model"
    assert os.environ["LOCAL_PORT"] == "19000"


def test_load_env_file_does_not_override_existing_value(tmp_path: Path, monkeypatch) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_NAME=from-file\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_NAME", "from-env")

    load_env_file(env_file)

    assert os.environ["MODEL_NAME"] == "from-env"


def test_resolve_cookie_file_path_uses_bridge_directory_not_cwd(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("COOKIE_FILE", raising=False)

    resolved = resolve_cookie_file_path()

    assert resolved == Path(__file__).resolve().parents[2] / "bridge" / "server1" / "cookie.txt"
