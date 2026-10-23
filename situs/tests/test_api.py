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
    assert data["data"][0]["id"] == "llama-3.1:8b-instruct-q8_0"


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
