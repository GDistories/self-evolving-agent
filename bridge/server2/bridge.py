from __future__ import annotations

import os
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request, Response

DEFAULT_REMOTE_BASE_URL = "https://your-host/proxy/19000"
DEFAULT_REMOTE_ORIGIN = "https://your-origin"
DEFAULT_REMOTE_REFERER = "https://your-referer"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 30.0
DEFAULT_COOKIE_FILE = "cookie.txt"


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


load_env_file(Path(__file__).with_name(".env"))

LOCAL_HOST = os.getenv("LOCAL_HOST", "127.0.0.1")
LOCAL_PORT = int(os.getenv("LOCAL_PORT", "19000"))


def get_remote_base_url() -> str:
    return os.getenv("REMOTE_BASE_URL", DEFAULT_REMOTE_BASE_URL)


def get_remote_origin() -> str:
    return os.getenv("REMOTE_ORIGIN", DEFAULT_REMOTE_ORIGIN)


def get_remote_referer() -> str:
    return os.getenv("REMOTE_REFERER", DEFAULT_REMOTE_REFERER)


def get_request_timeout_seconds() -> float:
    return float(os.getenv("REQUEST_TIMEOUT_SECONDS", str(DEFAULT_REQUEST_TIMEOUT_SECONDS)))


def get_cookie_file_name() -> str:
    return os.getenv("COOKIE_FILE", DEFAULT_COOKIE_FILE)


def resolve_cookie_file_path() -> Path:
    cookie_file = Path(get_cookie_file_name())
    if cookie_file.is_absolute():
        return cookie_file
    return Path(__file__).resolve().parent / cookie_file


def read_cookie_value(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"cookie txt does not exist: {path}")

    cookie = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not cookie:
        raise ValueError(f"cookie txt is empty: {path}")
    return cookie


def build_upstream_headers(content_type: str | None = None) -> dict[str, str]:
    headers = {
        "Cookie": read_cookie_value(resolve_cookie_file_path()),
        "Origin": get_remote_origin(),
        "Referer": get_remote_referer(),
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def forward_upstream(method: str, path: str, content: bytes | None = None, content_type: str | None = None) -> Response:
    url = f"{get_remote_base_url().rstrip('/')}{path}"
    try:
        headers = build_upstream_headers(content_type)
        upstream = httpx.request(
            method,
            url,
            content=content,
            headers=headers,
            timeout=get_request_timeout_seconds(),
        )
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="upstream timeout") from exc
    except httpx.TransportError as exc:
        raise HTTPException(status_code=502, detail=f"upstream transport error: {exc}") from exc
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    media_type = upstream.headers.get("content-type")
    return Response(content=upstream.content, status_code=upstream.status_code, media_type=media_type)


def create_app() -> FastAPI:
    app = FastAPI(title="Local Server2 Bridge")

    @app.post("/eval/jobs")
    async def create_eval_job(request: Request) -> Response:
        return forward_upstream(
            "POST",
            "/eval/jobs",
            content=await request.body(),
            content_type=request.headers.get("content-type"),
        )

    @app.get("/eval/jobs/{job_id}")
    async def get_eval_job(job_id: str) -> Response:
        return forward_upstream("GET", f"/eval/jobs/{job_id}")

    return app


app = create_app()
