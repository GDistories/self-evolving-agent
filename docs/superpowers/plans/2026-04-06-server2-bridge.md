# Server2 Bridge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated local `server2` bridge so both remote services are accessed through symmetric local bridge processes with separate cookies and proxy configuration.

**Architecture:** Split the bridge layer into `bridge/server1/` and `bridge/server2/`. `server1` keeps the existing WSS bridge semantics, while `server2` becomes a thin authenticated HTTP reverse proxy for `/eval/jobs` routes. The local client continues to call only local base URLs and does not learn any remote proxy or cookie details.

**Tech Stack:** Python, FastAPI, httpx, websockets, pytest

---

## File Structure

### Files to create

- `bridge/server1/bridge.py`
  Rehomes the current `server1` local WSS bridge.
- `bridge/server1/.env_example`
  Sample config for the `server1` bridge.
- `bridge/server2/bridge.py`
  New local HTTP reverse proxy for `server2`.
- `bridge/server2/.env_example`
  Sample config for the `server2` bridge.
- `tests/bridge/test_server1_bridge_config.py`
  Focused config/load tests for the `server1` bridge.
- `tests/bridge/test_server2_bridge.py`
  URL construction, header forwarding, and API proxy tests for the `server2` bridge.

### Files to modify

- `client/config.py`
  Point local client config defaults at local bridge ports rather than direct remote services.
- `README.md`
  Update topology and startup docs to use `bridge/server1/` and `bridge/server2/`.
- `.gitignore`
  Ensure only real `.env` files are ignored while `.env_example` files remain tracked.

### Files to delete or stop using

- `bridge/client_ws_proxy.py`
  Replaced by `bridge/server1/bridge.py`.

## Task 1: Move the Existing Server1 Bridge into `bridge/server1/`

**Files:**
- Create: `bridge/server1/bridge.py`
- Create: `tests/bridge/test_server1_bridge_config.py`
- Delete: `bridge/client_ws_proxy.py`

- [ ] **Step 1: Write the failing bridge-config test**

```python
from pathlib import Path

from bridge.server1.bridge import load_env_file


def test_load_env_file_sets_missing_server1_values(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("REMOTE_WSS_URL=wss://example/ws\nMODEL_NAME=test-model\n", encoding="utf-8")
    monkeypatch.delenv("REMOTE_WSS_URL", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)

    load_env_file(env_file)

    assert "REMOTE_WSS_URL" in __import__("os").environ
    assert __import__("os").environ["MODEL_NAME"] == "test-model"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/bridge/test_server1_bridge_config.py::test_load_env_file_sets_missing_server1_values -v`
Expected: FAIL with `ModuleNotFoundError` for `bridge.server1.bridge`

- [ ] **Step 3: Create `bridge/server1/bridge.py` by moving the current bridge logic**

```python
from pathlib import Path
import os


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env_file(Path(__file__).with_name(".env"))
```

Move the existing `bridge/client_ws_proxy.py` implementation into this new file without changing its bridge behavior.

- [ ] **Step 4: Add a minimal `server1` config test**

```python
def test_load_env_file_does_not_override_existing_value(tmp_path: Path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("MODEL_NAME=from-file\n", encoding="utf-8")
    monkeypatch.setenv("MODEL_NAME", "from-env")

    load_env_file(env_file)

    assert __import__("os").environ["MODEL_NAME"] == "from-env"
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/bridge/test_server1_bridge_config.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add bridge/server1/bridge.py tests/bridge/test_server1_bridge_config.py bridge/client_ws_proxy.py
git commit -m "refactor: move server1 bridge into subdirectory"
```

## Task 2: Add the New `server2` HTTP Bridge

**Files:**
- Create: `bridge/server2/bridge.py`
- Create: `tests/bridge/test_server2_bridge.py`

- [ ] **Step 1: Write the failing proxy test**

```python
from fastapi.testclient import TestClient

from bridge.server2.bridge import create_app


def test_post_eval_jobs_forwards_to_remote(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 200

        def json(self):
            return {"job_id": "job-1", "status": "queued"}

        @property
        def text(self):
            return '{"job_id":"job-1","status":"queued"}'

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return DummyResponse()

    monkeypatch.setattr("httpx.post", fake_post)
    client = TestClient(create_app())

    response = client.post("/eval/jobs", json={"candidate_id": "cand-1"})

    assert response.status_code == 200
    assert response.json()["job_id"] == "job-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/bridge/test_server2_bridge.py::test_post_eval_jobs_forwards_to_remote -v`
Expected: FAIL with `ModuleNotFoundError` for `bridge.server2.bridge`

- [ ] **Step 3: Implement the `server2` bridge shell**

```python
import httpx
from fastapi import FastAPI, Request, Response


def create_app() -> FastAPI:
    app = FastAPI(title="Local Server2 Bridge")

    @app.post("/eval/jobs")
    async def create_job(request: Request) -> Response:
        payload = await request.body()
        upstream = httpx.post(
            f"{REMOTE_BASE_URL.rstrip('/')}/eval/jobs",
            content=payload,
            headers=build_upstream_headers(request.headers.get("content-type")),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return Response(content=upstream.text, status_code=upstream.status_code, media_type="application/json")

    @app.get("/eval/jobs/{job_id}")
    async def get_job(job_id: str) -> Response:
        upstream = httpx.get(
            f"{REMOTE_BASE_URL.rstrip('/')}/eval/jobs/{job_id}",
            headers=build_upstream_headers(),
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        return Response(content=upstream.text, status_code=upstream.status_code, media_type="application/json")

    return app
```

It must:

- load `bridge/server2/.env`
- read a dedicated cookie file
- add `Cookie`, `Origin`, and `Referer`
- forward to `REMOTE_BASE_URL + /eval/jobs` and `REMOTE_BASE_URL + /eval/jobs/{job_id}`
- preserve upstream status code and body

- [ ] **Step 4: Add focused tests for headers and GET path**

```python
def test_get_eval_job_forwards_cookie_and_headers(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 200
        text = '{"job_id":"job-1","status":"running"}'

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return DummyResponse()

    monkeypatch.setattr("httpx.get", fake_get)
    monkeypatch.setattr("bridge.server2.bridge.read_cookie", lambda: "cookie=value")
    monkeypatch.setattr("bridge.server2.bridge.REMOTE_BASE_URL", "https://proxy.example/proxy/19000")
    monkeypatch.setattr("bridge.server2.bridge.REMOTE_ORIGIN", "https://origin.example")
    monkeypatch.setattr("bridge.server2.bridge.REMOTE_REFERER", "https://referer.example")

    assert captured["kwargs"]["headers"]["Cookie"] == "cookie=value"
    assert captured["kwargs"]["headers"]["Origin"] == "https://origin.example"
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/bridge/test_server2_bridge.py -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add bridge/server2/bridge.py tests/bridge/test_server2_bridge.py
git commit -m "feat: add local server2 bridge"
```

## Task 3: Point Local Client Config at Local Bridges

**Files:**
- Modify: `client/config.py`
- Modify: `tests/client/test_controller.py`

- [ ] **Step 1: Write the failing config-defaults test**

```python
from client.config import RuntimeConfig, load_runtime_config


def test_load_runtime_config_defaults_to_local_bridge_ports(monkeypatch):
    monkeypatch.delenv("SERVER1_BASE_URL", raising=False)
    monkeypatch.delenv("SERVER2_BASE_URL", raising=False)
    monkeypatch.delenv("POLL_INTERVAL_SECONDS", raising=False)

    config = load_runtime_config()

    assert config == RuntimeConfig(
        server1_base_url="http://127.0.0.1:18000",
        server2_base_url="http://127.0.0.1:19000",
        poll_interval_seconds=2.0,
    )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/client/test_controller.py::test_load_runtime_config_defaults_to_local_bridge_ports -v`
Expected: FAIL because current defaults still point elsewhere

- [ ] **Step 3: Update local config defaults**

```python
return RuntimeConfig(
    server1_base_url=os.getenv("SERVER1_BASE_URL", "http://127.0.0.1:18000"),
    server2_base_url=os.getenv("SERVER2_BASE_URL", "http://127.0.0.1:19000"),
    poll_interval_seconds=poll_interval_seconds,
)
```

- [ ] **Step 4: Run targeted tests**

Run: `pytest tests/client/test_controller.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add client/config.py tests/client/test_controller.py
git commit -m "refactor: point client at local bridge defaults"
```

## Task 4: Track `.env_example` Files and Ignore Only `.env`

**Files:**
- Modify: `.gitignore`
- Create: `bridge/server1/.env_example`
- Create: `bridge/server2/.env_example`

- [ ] **Step 1: Write the failing ignore-rule check**

```bash
git check-ignore -v bridge/server1/.env_example
```

Expected: no output; `.env_example` must not be ignored

- [ ] **Step 2: Update ignore rules and create example files**

```text
.env
```

Example `bridge/server2/.env_example`:

```text
REMOTE_BASE_URL=https://your-host/proxy/19000
REMOTE_ORIGIN=https://your-origin
REMOTE_REFERER=https://your-referer
LOCAL_HOST=127.0.0.1
LOCAL_PORT=19000
COOKIE_FILE=cookie.txt
REQUEST_TIMEOUT_SECONDS=30
```

- [ ] **Step 3: Verify ignore behavior**

Run: `git check-ignore -v bridge/server1/.env bridge/server1/.env_example`
Expected:
- `.env` is ignored
- `.env_example` is not ignored

- [ ] **Step 4: Commit**

```bash
git add .gitignore bridge/server1/.env_example bridge/server2/.env_example
git commit -m "chore: add bridge env examples"
```

## Task 5: Update README and Validate the End-to-End Bridge Contract

**Files:**
- Modify: `README.md`
- Modify: `tests/bridge/test_server2_bridge.py`

- [ ] **Step 1: Write the failing docs-alignment test**

```python
def test_post_eval_jobs_returns_upstream_status_code(monkeypatch):
    class DummyResponse:
        status_code = 502
        text = '{"detail":"upstream timeout"}'

    monkeypatch.setattr("httpx.post", lambda *args, **kwargs: DummyResponse())
    client = TestClient(create_app())
    response = client.post("/eval/jobs", json={"candidate_id": "cand-1"})

    assert response.status_code == 502
```

This ensures the `server2` bridge docs do not overpromise hidden behavior and the proxy preserves upstream responses.

- [ ] **Step 2: Update README**

Add:

- `bridge/server1/bridge.py`
- `bridge/server2/bridge.py`
- separate `cookie.txt` files
- startup order:
  - start `server1`
  - start `server2`
  - start local `bridge/server1`
  - start local `bridge/server2`
  - run local controller

- [ ] **Step 3: Run targeted bridge tests**

Run: `pytest tests/bridge tests/client/test_controller.py -v`
Expected: PASS

- [ ] **Step 4: Run broader suite**

Run: `pytest tests/client tests/server2 tests/bridge -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md tests/bridge/test_server2_bridge.py
git commit -m "docs: describe dual bridge topology"
```

## Spec Coverage Check

- `bridge/server1/` and `bridge/server2/` split
  Covered by Task 1 and Task 2.
- `server2` accessed through local HTTP bridge with separate cookie
  Covered by Task 2.
- local client only knows local base URLs
  Covered by Task 3.
- separate `.env`, `.env_example`, and `cookie.txt` layout
  Covered by Task 1, Task 2, and Task 4.
- remote proxy URL, `Cookie`, `Origin`, `Referer` forwarding
  Covered by Task 2.
- bridge-focused testing and regression protection
  Covered by Task 1, Task 2, and Task 5.

## Self-Review

- Placeholder scan complete after editing; rerun the placeholder regex check on this plan file and expect no matches.
- Type consistency checked:
  `RuntimeConfig`, `create_app`, and `REMOTE_BASE_URL` naming is consistent across tasks.
- Scope check:
  the plan stays focused on the bridge and transport layer and does not expand into evaluator logic or prompt evolution behavior.
