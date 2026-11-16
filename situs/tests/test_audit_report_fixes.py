import pytest
import os
import threading
import asyncio
import time
from unittest.mock import patch, MagicMock

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from situs.worker.loader import load_hf_stage_model, DistributedStageModel
from situs.worker.client import WorkerDaemon
from situs.config import WorkerConfig
from situs.tokenizer import get_or_load_tokenizer, _tokenizers_cache
from situs.hardware import query_nvidia_smi
from situs.network.p2p import get_safe_sctp_max_bytes, P2PDataChannelNode


def test_loader_non_gemma_spec_defined():
    """Verify Issue 1.1: non-Gemma architecture does not raise NameError: name 'spec' is not defined."""
    if not HAS_TORCH:
        pytest.skip("PyTorch required")

    # Mock resolve_model_metadata returning a non-Gemma architecture
    mock_meta = {
        "architecture": "llama",
        "num_layers": 4,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 32,
        "vocab_size": 1000,
        "intermediate_size": 256,
    }

    with patch("situs.worker.loader.resolve_model_metadata", return_value=mock_meta), \
         patch("situs.worker.loader.find_local_safetensors_files", return_value=[]), \
         patch("situs.worker.loader.find_local_ollama_gguf_file", return_value=None):
        stage = load_hf_stage_model(
            "test-llama-model",
            layer_start=0,
            layer_end=2,
            total_layers=4,
            is_first_stage=True,
            is_last_stage=False,
            device="cpu",
        )
        assert stage is not None
        assert stage.architecture == "llama"


def test_tui_slash_model_no_name_error():
    """Verify Issue 1.2: execute_slash_command handles model pull without history_entries NameError."""
    from situs.tui import execute_slash_command

    session_state = {
        "model": "gemma3:4b",
        "coordinator_url": "http://127.0.0.1:8000",
        "is_pulling": False,
    }
    messages = []
    logged_messages = []

    def log_fn(txt, title):
        logged_messages.append((txt, title))

    # Test /model command
    ret = execute_slash_command(
        cmd="/model gemma3:4b",
        coordinator_url="http://127.0.0.1:8000",
        model="gemma3:4b",
        messages=messages,
        session_state=session_state,
        log_output_fn=log_fn,
    )
    assert ret is True
    assert len(logged_messages) > 0


def test_worker_daemon_running_property_and_heartbeat_loop():
    """Verify Issue 1.3: WorkerDaemon defines running property and heartbeat_loop survives transient exceptions."""
    cfg = WorkerConfig(worker_id="test-worker", device="cpu")
    daemon = WorkerDaemon(cfg)

    # Verify running property reflects is_running
    assert daemon.running is True
    daemon.running = False
    assert daemon.is_running is False
    daemon.running = True

    # Test heartbeat loop termination without AttributeError
    async def run_hb():
        daemon.ws = MagicMock()
        # Mock send_heartbeat_once raising an error
        daemon.send_heartbeat_once = MagicMock(side_effect=RuntimeError("transient socket error"))
        # Run 1 iteration of heartbeat loop
        task = asyncio.create_task(daemon.heartbeat_loop())
        await asyncio.sleep(0.05)
        daemon.is_running = False
        await task

    asyncio.run(run_hb())


def test_tokenizer_cache_thread_safety():
    """Verify Issue 4.2: concurrent get_or_load_tokenizer accesses do not race on _tokenizers_cache."""
    results = []
    errors = []

    def worker():
        try:
            tok = get_or_load_tokenizer("default")
            results.append(tok)
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    assert len(results) == 8


def test_hardware_smi_lock_concurrency():
    """Verify Issue 4.2: concurrent query_nvidia_smi calls execute safely with thread lock."""
    results = []

    def worker():
        res = query_nvidia_smi(force_refresh=False)
        results.append(res)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 8


def test_p2p_configurable_mtu_threshold():
    """Verify Issue 3.3: get_safe_sctp_max_bytes respects SITUS_P2P_MAX_BYTES."""
    default_bytes = get_safe_sctp_max_bytes()
    assert default_bytes == 128 * 1024

    with patch.dict(os.environ, {"SITUS_P2P_MAX_BYTES": "524288"}):
        assert get_safe_sctp_max_bytes() == 512 * 1024


@pytest.mark.asyncio
async def test_strict_distributed_rejection():
    """Verify Issue 3.1: SITUS_STRICT_DISTRIBUTED=1 rejects unsupported models with 400."""
    from fastapi.testclient import TestClient
    from situs.coordinator.app import app

    client = TestClient(app)
    unsupported_meta = {
        "architecture": "unsupported_exotic_arch",
        "num_layers": 4,
        "hidden_dim": 128,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 32,
        "vocab_size": 1000,
        "intermediate_size": 256,
    }

    dummy_worker = MagicMock()
    dummy_worker.worker_id = "w1"
    dummy_worker.is_local = True

    mock_tok = MagicMock()
    mock_tok.encode.return_value = [1, 2, 3]
    mock_tok.apply_chat_template.return_value = "hello"

    with patch.dict(os.environ, {"SITUS_STRICT_DISTRIBUTED": "1"}), \
         patch("situs.coordinator.app.registry.get_alive_workers", return_value=[dummy_worker]), \
         patch("situs.coordinator.app.registry.get_pipeline_order", return_value=[dummy_worker]), \
         patch("situs.coordinator.app.get_or_load_tokenizer", return_value=mock_tok), \
         patch("situs.coordinator.app.resolve_model_metadata", return_value=unsupported_meta):
        payload = {
            "model": "unsupported_exotic_arch",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": False,
        }
        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 400
        assert "not supported" in resp.json()["detail"].lower()
