"""Tests for Coordinator FastAPI endpoints (OpenAI standard API)."""

import pytest
import asyncio
from fastapi.testclient import TestClient
from situs.coordinator.app import app, registry, relay
from situs.protocol.messages import WorkerRegister


@pytest.fixture
def client():
    registry.clear()
    return TestClient(app)


def test_cluster_status_initially_empty(client):
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["active_workers_count"] == 0
    assert data["status"] == "no_workers"


def test_list_models_openai(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    assert any(m["id"] == "gemma3:4b" for m in data["data"])
    assert any("llama" in m["id"] for m in data["data"])


def test_chat_completions_requires_minimum_workers(client):
    # When 0 workers are connected, completions should return 503
    payload = {
        "model": "llama-3.1:8b-instruct-q8_0",
        "messages": [{"role": "user", "content": "Hello!"}],
        "stream": False,
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 503
    assert "not ready" in response.json()["detail"].lower()


def test_progress_bar_and_stages():
    from situs.inference_backend import make_progress_bar, format_stage_name
    bar_0 = make_progress_bar(0.0, width=10)
    assert len(bar_0) == 12  # [ + 10 chars + ]
    assert "░" in bar_0

    bar_100 = make_progress_bar(100.0, width=10)
    assert len(bar_100) == 12
    assert "█" in bar_100

    assert "Manifest" in format_stage_name("pulling manifest")
    assert "Streaming" in format_stage_name("downloading layer 1")
    assert "Verifying" in format_stage_name("verifying sha256")
    assert "Allocating" in format_stage_name("writing manifest")
    assert "Ready" in format_stage_name("success")


def test_admin_pull_sse_endpoint(client):
    # Test that /admin/pull streams SSE events with valid headers
    with client.stream("POST", "/admin/pull?model_name=gemma3:4b") as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]
        first_line = next(resp.iter_lines())
        assert first_line in (": connected", "data: [DONE]") or "data: {" in first_line

