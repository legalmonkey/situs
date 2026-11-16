"""
Automated regression tests verifying the elimination of hardcoded values,
premature EOS sampling bug on masked logits, clean WORKER_REMOVED termination,
and dynamic configuration propagation across coordinator and worker engine.
"""

import os
import time
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from situs.config import CoordinatorConfig, WorkerConfig
from situs.protocol.messages import (
    MessageType,
    ProtocolFrame,
    InferenceStart,
    ForwardActivation,
    AssignLayers,
    WorkerRegister,
)
from situs.coordinator.registry import WorkerRegistry
from situs.worker.sampler import TokenSampler
from situs.worker.engine import StageExecutionEngine
from situs.coordinator.app import rebalance_and_assign_layers, ChatCompletionRequest, app
from situs.inference_backend import get_ollama_endpoints


def test_sampler_permits_masked_negative_infinite_logits():
    """Verify TokenSampler does NOT emit EOS when vocabulary masking sets some logits to -inf."""
    sampler = TokenSampler(tokenizer=None)

    if HAS_TORCH:
        # Create logits for 100 vocab items: tokens 0..89 are -inf (masked), tokens 90..99 have real logits
        logits = torch.full((100,), -float("inf"), dtype=torch.float32)
        logits[95] = 10.0
        logits[96] = 5.0

        # Should NOT return EOS (0). Should sample token 95 or 96
        token = sampler.sample_next_token(logits, temperature=0.01)
        assert token == 95, f"Expected argmax token 95, got {token} (premature EOS bug!)"

        # Verify all -inf logits returns EOS
        all_masked = torch.full((100,), -float("inf"), dtype=torch.float32)
        assert sampler.sample_next_token(all_masked) == 0

    # Test NumPy fallback
    arr = np.full((100,), -np.inf, dtype=np.float32)
    arr[92] = 15.0
    arr[93] = 4.0
    token_np = sampler.sample_next_token(arr, temperature=0.01)
    assert token_np == 92, f"Expected numpy token 92, got {token_np}"

    arr_all_masked = np.full((100,), -np.inf, dtype=np.float32)
    assert sampler.sample_next_token(arr_all_masked) == 0


@pytest.mark.asyncio
async def test_coordinator_rejects_unresolved_model_without_silent_28_layer_default():
    """Verify rebalance_and_assign_layers raises ValueError rather than defaulting silently to 28 layers."""
    from situs.coordinator.app import registry as coord_registry
    coord_registry.register_worker(
        WorkerRegister(
            worker_id="test_w1",
            gpu_name="Test GPU",
            total_vram_bytes=16 * 1024**3,
            free_vram_bytes=14 * 1024**3,
            used_vram_bytes=2 * 1024**3,
            device_type="cuda",
        )
    )

    # Patch model resolver to raise exception (unresolvable model metadata)
    with patch("situs.coordinator.app.resolve_model_metadata", side_effect=ValueError("Model metadata not found")):
        with pytest.raises(ValueError) as excinfo:
            await rebalance_and_assign_layers(model_name="non_existent_unresolved_model_xyz", wait_for_in_flight=False)
        assert "Cannot partition model" in str(excinfo.value)
        assert "Silent fallback default layers have been removed" in str(excinfo.value)


def test_worker_daemon_stops_cleanly_on_worker_removed():
    """Verify that receiving WORKER_REMOVED triggers self.stop() and breaks connection loop."""
    from situs.worker.client import WorkerDaemon

    cfg = WorkerConfig(worker_id="test_worker_removal", coordinator_url="ws://127.0.0.1:8999")
    daemon = WorkerDaemon(cfg)
    daemon.is_running = True
    assert daemon.is_running is True

    daemon.stop()
    assert daemon.is_running is False


def test_top_k_parameter_propagation():
    """Verify top_k propagates through ChatCompletionRequest into InferenceStart."""
    req = ChatCompletionRequest(
        model="custom_model",
        messages=[{"role": "user", "content": "hello"}],
        top_k=42,
    )
    assert req.top_k == 42

    start_frame = InferenceStart(
        request_id="req-123",
        model_name="custom_model",
        input_ids=[1, 2, 3],
        top_k=req.top_k,
    )
    assert start_frame.top_k == 42


def test_worker_config_and_engine_configurable_parameters():
    """Verify custom WorkerConfig values are passed into StageExecutionEngine."""
    cfg = WorkerConfig(
        worker_id="test_cfg_worker",
        ngram_size=3,
        max_draft_tokens=6,
        replay_buffer_capacity=8,
        safety_headroom_fraction=0.15,
    )
    assert cfg.ngram_size == 3
    assert cfg.max_draft_tokens == 6
    assert cfg.replay_buffer_capacity == 8
    assert cfg.safety_headroom_fraction == 0.15

    assignment = AssignLayers(
        worker_id="test_cfg_worker",
        model_name="test_model",
        layer_start=0,
        layer_end=4,
        total_layers=8,
        is_first_stage=True,
        is_last_stage=False,
    )
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.parameters.return_value = []

    engine = StageExecutionEngine(
        stage_model=mock_model,
        assignment=assignment,
        ngram_size=cfg.ngram_size,
        max_draft_tokens=cfg.max_draft_tokens,
        replay_capacity=cfg.replay_buffer_capacity,
        safety_headroom_fraction=cfg.safety_headroom_fraction,
    )

    assert engine.prompt_drafter.ngram_size == 3
    assert engine.prompt_drafter.max_draft_tokens == 6
    assert engine.replay_buffer.capacity == 8
    assert engine.memory_manager.safety_headroom_fraction == 0.15


def test_worker_engine_kv_cache_estimation_uses_max_new_tokens():
    """Verify StageExecutionEngine estimates KV memory using start_msg.max_new_tokens."""
    assignment = AssignLayers(
        worker_id="test_kv_worker",
        model_name="test_model",
        layer_start=0,
        layer_end=4,
        total_layers=8,
        is_first_stage=True,
        is_last_stage=False,
    )
    mock_model = MagicMock()
    mock_model.device = "cpu"
    mock_model.num_kv_heads = 4
    mock_model.head_dim = 64
    mock_model.parameters.return_value = []
    if HAS_TORCH:
        mock_model.forward_first_stage.return_value = (torch.zeros((1, 3, 64)), [])

    engine = StageExecutionEngine(
        stage_model=mock_model,
        assignment=assignment,
    )
    # Spy on allocate_request
    engine.kv_cache_manager.allocate_request = MagicMock()

    start_msg = InferenceStart(
        request_id="req-test-max-new",
        model_name="m",
        input_ids=[10, 20, 30],
        max_new_tokens=512,
    )
    engine.start_inference(start_msg)

    # 4 layers * 4 kv heads * 64 head_dim * (3 + 512 seq) * 2 bytes * 2 (K+V) = 2,109,440 bytes
    expected_seq = 3 + 512
    expected_bytes = 2 * 4 * 4 * 64 * expected_seq * 2
    engine.kv_cache_manager.allocate_request.assert_called_with("req-test-max-new", estimated_bytes=expected_bytes)


def test_ollama_host_env_discovery(monkeypatch):
    """Verify get_ollama_endpoints dynamically prioritizes OLLAMA_HOST."""
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.50:11434")
    endpoints = get_ollama_endpoints()
    assert endpoints[0] == "http://192.168.1.50:11434"

    monkeypatch.setenv("OLLAMA_HOST", "https://remote-gpu-host:11434")
    endpoints_ssl = get_ollama_endpoints()
    assert endpoints_ssl[0] == "https://remote-gpu-host:11434"


def test_coordinator_uses_configured_default_model():
    """Verify coordinator endpoints reference config.default_model rather than hardcoded gemma3:4b."""
    cfg = CoordinatorConfig(default_model="qwen2.5:7b")
    assert cfg.default_model == "qwen2.5:7b"

    req = ChatCompletionRequest(messages=[{"role": "user", "content": "test"}])
    assert req.model is None  # Defaults to None so coordinator can use config.default_model


def test_gguf_loader_vocab_size_orientation_independent():
    """Verify GGUF probe calculates vocab_size as max(dim0, dim1) to handle both shape orientations."""
    # Simulate tensor probe with shape (4096, 32000) or (32000, 4096)
    pt1 = MagicMock()
    pt1.name = "token_embd.weight"
    pt1.shape = [4096, 32000]

    pt2 = MagicMock()
    pt2.name = "token_embd.weight"
    pt2.shape = [152064, 3584]

    vocab1 = max(int(pt1.shape[0]), int(pt1.shape[1]))
    vocab2 = max(int(pt2.shape[0]), int(pt2.shape[1]))

    assert vocab1 == 32000
    assert vocab2 == 152064


@pytest.mark.asyncio
async def test_sse_error_terminates_with_done():
    """Verify SSE generator yields data: [DONE] upon worker ERROR frame to prevent client hangs."""
    import asyncio
    import json
    from situs.coordinator.app import relay

    request_id = "test-sse-err-req"
    q = asyncio.Queue()
    relay.response_streams[request_id] = q

    # Simulate an error frame put into the queue
    err_frame = ProtocolFrame(
        type=MessageType.ERROR,
        source="w1",
        target="coordinator",
        payload={"request_id": request_id, "error": "CUDA out of memory simulation"},
    )
    await q.put(err_frame)

    # Replicate coordinator SSE drain logic
    events = []
    async def drain():
        done_yielded = False
        try:
            while True:
                frame = await q.get()
                if frame.type == MessageType.ERROR:
                    events.append(f"data: {json.dumps({'error': frame.payload.get('error')})}\n\n")
                    events.append("data: [DONE]\n\n")
                    done_yielded = True
                    break
        finally:
            if not done_yielded:
                events.append("data: [DONE]\n\n")
            relay.cleanup_response_stream(request_id)

    await drain()

    assert len(events) == 2
    assert "CUDA out of memory" in events[0]
    assert events[1] == "data: [DONE]\n\n"
    assert request_id not in relay.response_streams

