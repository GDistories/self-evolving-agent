from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient
import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bridge.server2.bridge import create_app


def test_post_eval_jobs_forwards_body_and_headers(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = b'{"job_id":"job-1","status":"queued"}'

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["kwargs"] = kwargs
        return DummyResponse()

    def fake_read_cookie_value(path):
        captured["cookie_path"] = path
        return "cookie=value"

    monkeypatch.setattr("bridge.server2.bridge.httpx.request", fake_request)
    monkeypatch.setattr("bridge.server2.bridge.read_cookie_value", fake_read_cookie_value)
    monkeypatch.setenv("REMOTE_BASE_URL", "https://proxy.example/proxy/19000")
    monkeypatch.setenv("REMOTE_ORIGIN", "https://origin.example")
    monkeypatch.setenv("REMOTE_REFERER", "https://referer.example")
    monkeypatch.setenv("REQUEST_TIMEOUT_SECONDS", "12.5")
    monkeypatch.setenv("COOKIE_FILE", "nested/cookie.txt")

    client = TestClient(create_app())
    response = client.post("/eval/jobs", json={"candidate_id": "cand-1"})

    assert response.status_code == 200
    assert response.json() == {"job_id": "job-1", "status": "queued"}
    assert captured["method"] == "POST"
    assert captured["url"] == "https://proxy.example/proxy/19000/eval/jobs"
    assert captured["cookie_path"] == Path(__file__).resolve().parents[2] / "bridge" / "server2" / "nested" / "cookie.txt"
    assert json.loads(captured["kwargs"]["content"].decode("utf-8")) == {"candidate_id": "cand-1"}
    assert captured["kwargs"]["headers"] == {
        "Cookie": "cookie=value",
        "Origin": "https://origin.example",
        "Referer": "https://referer.example",
        "Content-Type": "application/json",
    }
    assert captured["kwargs"]["timeout"] == 12.5


def test_get_eval_job_forwards_path_and_headers(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 200
        headers = {"content-type": "application/json"}
        content = b'{"job_id":"job-9","status":"running"}'

    def fake_request(method, url, **kwargs):
        captured["method"] = method
        captured["url"] = url
        captured["kwargs"] = kwargs
        return DummyResponse()

    def fake_read_cookie_value(path):
        captured["cookie_path"] = path
        return "cookie=value"

    monkeypatch.setattr("bridge.server2.bridge.httpx.request", fake_request)
    monkeypatch.setattr("bridge.server2.bridge.read_cookie_value", fake_read_cookie_value)
    monkeypatch.setenv("REMOTE_BASE_URL", "https://proxy.example/proxy/19000")
    monkeypatch.setenv("REMOTE_ORIGIN", "https://origin.example")
    monkeypatch.setenv("REMOTE_REFERER", "https://referer.example")
    monkeypatch.setenv("COOKIE_FILE", "cookie.txt")

    client = TestClient(create_app())
    response = client.get("/eval/jobs/job-9")

    assert response.status_code == 200
    assert response.json() == {"job_id": "job-9", "status": "running"}
    assert captured["method"] == "GET"
    assert captured["url"] == "https://proxy.example/proxy/19000/eval/jobs/job-9"
    assert captured["cookie_path"] == Path(__file__).resolve().parents[2] / "bridge" / "server2" / "cookie.txt"
    assert captured["kwargs"]["content"] is None
    assert captured["kwargs"]["headers"] == {
        "Cookie": "cookie=value",
        "Origin": "https://origin.example",
        "Referer": "https://referer.example",
    }


def test_upstream_status_and_body_are_preserved(monkeypatch):
    class DummyResponse:
        status_code = 502
        headers = {"content-type": "text/plain; charset=utf-8"}
        content = b"upstream exploded"

    monkeypatch.setattr("bridge.server2.bridge.httpx.request", lambda method, url, **kwargs: DummyResponse())
    monkeypatch.setattr("bridge.server2.bridge.read_cookie_value", lambda path: "cookie=value")
    monkeypatch.setenv("REMOTE_BASE_URL", "https://proxy.example/proxy/19000")

    client = TestClient(create_app())
    response = client.get("/eval/jobs/job-1")

    assert response.status_code == 502
    assert response.content == b"upstream exploded"


def test_transport_errors_are_mapped_to_bad_gateway(monkeypatch):
    def fake_request(method, url, **kwargs):
        raise httpx.ConnectError("connection refused", request=httpx.Request(method, url))

    monkeypatch.setattr("bridge.server2.bridge.httpx.request", fake_request)
    monkeypatch.setattr("bridge.server2.bridge.read_cookie_value", lambda path: "cookie=value")
    monkeypatch.setenv("REMOTE_BASE_URL", "https://proxy.example/proxy/19000")

    client = TestClient(create_app())
    response = client.get("/eval/jobs/job-1")

    assert response.status_code == 502
    assert response.json() == {"detail": "upstream transport error: connection refused"}
