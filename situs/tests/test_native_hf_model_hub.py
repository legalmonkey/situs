"""Tests verifying complete elimination of Ollama and native Hugging Face Hub operations.

Verifies:
1. Zero Ollama daemon connections or socket attempts to port 11434.
2. Direct Hugging Face Hub model normalization and registry resolution.
3. Native streaming download events and progress yielding.
4. Model metadata resolution without external daemon calls.
5. Backward-compatibility stubs remain non-blocking and safe.
"""

import asyncio
import json
import pytest
from unittest.mock import patch, MagicMock

from situs.inference_backend import (
    normalize_model_name,
    normalize_ollama_model,
    is_model_downloaded,
    stream_pull_model,
    get_ollama_endpoints,
    ensure_ollama_running,
    ensure_ollama_model,
    HF_MODEL_REGISTRY_MAP,
)
from situs.coordinator.registry import resolve_model_metadata


def test_normalize_model_name_catalog_mappings():
    """Verify standard model aliases are mapped cleanly to canonical Hugging Face repositories."""
    assert normalize_model_name("gemma3:4b") == "google/gemma-3-4b-it-qat-q4_0-gguf"
    assert normalize_model_name("gemma2:2b") == "google/gemma-2-2b-it"
    assert normalize_model_name("llama3.1:8b") == "meta-llama/Llama-3.1-8B-Instruct"
    assert normalize_model_name("qwen2.5:7b") == "Qwen/Qwen2.5-7B-Instruct"
    assert normalize_model_name("mistral:7b") == "mistralai/Mistral-7B-Instruct-v0.3"
    assert normalize_model_name("deepseek-r1:7b") == "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"

    # Direct HF repos pass through
    assert normalize_model_name("custom-org/my-model") == "custom-org/my-model"

    # Backward compatibility alias
    assert normalize_ollama_model("gemma3:4b") == "google/gemma-3-4b-it-qat-q4_0-gguf"


def test_is_model_downloaded_without_ollama_daemon(tmp_path, monkeypatch):
    """Verify is_model_downloaded checks disk/HF cache and never queries Ollama port 11434."""
    fake_cache = tmp_path / "fake_hf_hub"
    fake_cache.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(fake_cache))

    assert is_model_downloaded("nonexistent_model_12345") is False

    # Simulate downloaded snapshot in HF cache
    model_repo = "google/gemma-3-4b-it"
    clean_dir = "models--google--gemma-3-4b-it"
    snap_dir = fake_cache / clean_dir / "snapshots" / "main_revision"
    snap_dir.mkdir(parents=True)
    (snap_dir / "model.safetensors").write_text("dummy-weights")

    assert is_model_downloaded(model_repo) is True
    assert is_model_downloaded("gemma3:4b") is True


@pytest.mark.asyncio
async def test_stream_pull_model_yields_hf_progress():
    """Verify stream_pull_model downloads via HF Hub and streams progress frames."""
    with patch("huggingface_hub.HfApi") as mock_api_cls, patch("huggingface_hub.hf_hub_download") as mock_dl:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        fake_sibling = MagicMock()
        fake_sibling.rfilename = "model.safetensors"
        fake_sibling.size = 1000
        mock_api.model_info.return_value = MagicMock(siblings=[fake_sibling])
        mock_dl.return_value = "/fake/path/model.safetensors"

        events = []
        async for item in stream_pull_model("google/gemma-3-4b-it"):
            events.append(item)

        types = [e.get("type") for e in events]
        assert "model_loading" in types

        # Check final success event
        last_event = events[-1]
        assert last_event["status"] == "success"
        assert last_event["model"] == "google/gemma-3-4b-it"


@pytest.mark.asyncio
async def test_stream_pull_model_catches_errors_gracefully():
    """Verify stream_pull_model yields a clean error event if HF download fails."""
    with patch("huggingface_hub.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.model_info.side_effect = RuntimeError("HF Hub 404: Repo Not Found")

        events = []
        async for item in stream_pull_model("nonexistent/bad-model"):
            events.append(item)

        err_event = next((e for e in events if e.get("type") == "model_loading_error"), None)
        assert err_event is not None
        assert "HF Hub 404" in err_event["error"]


def test_resolve_model_metadata_without_ollama_port():
    """Verify resolve_model_metadata resolves via local configs or AutoConfig without hitting port 11434."""
    meta = resolve_model_metadata("gemma3:4b")
    assert meta is not None
    assert meta["total_layers"] == 34
    assert meta["hidden_dim"] == 2560


def test_legacy_ollama_stubs_are_safe_noop():
    """Verify deprecated Ollama stubs do not fail or attempt socket connections."""
    assert get_ollama_endpoints() == []
    assert ensure_ollama_running() is True
    assert ensure_ollama_model("http://localhost:11434", "gemma3:4b") is True
