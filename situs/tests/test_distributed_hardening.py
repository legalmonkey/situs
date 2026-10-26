"""Unit and integration tests for distributed inference resilience, progressive weight loading, and security."""

import pytest
import asyncio
import torch
from unittest.mock import patch
from fastapi.testclient import TestClient

from situs.coordinator.app import app, registry, relay, config
from situs.protocol.messages import (
    MessageType,
    ProtocolFrame,
    AssignLayers,
    InferenceStart,
    ForwardActivation,
    NextToken,
    StreamLayerChunk,
    InferenceAbort,
    WorkerRegister,
)
from situs.worker.loader import (
    DistributedStageModel,
    RealTransformerBlock,
    deserialize_and_validate_layer_weights,
)
from situs.worker.engine import StageExecutionEngine
from situs.worker.jit_streamer import generate_mock_layer_bytes
from situs.tokenizer import EnglishFallbackTokenizer


def test_distributed_inference_succeeds_when_coordinator_local_backend_unavailable():
    """
    P0 Verification: Proves distributed inference is genuine and workers execute
    forward passes and sample logits locally, completely succeeding even when the
    coordinator's local inference backend raises an exception / is unavailable.
    """
    total_layers = 4
    hidden_dim = 64
    vocab_size = 200

    model_a = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        head_dim=16,
        architecture="llama",
        device="cpu",
    )
    model_b = DistributedStageModel(
        layer_start=2,
        layer_end=4,
        total_layers=total_layers,
        is_first_stage=False,
        is_last_stage=True,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        head_dim=16,
        architecture="llama",
        device="cpu",
    )

    assign_a = AssignLayers(
        worker_id="w-1",
        model_name="test-model",
        layer_start=0,
        layer_end=2,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        next_worker_id="w-2",
        first_worker_id="w-1",
    )
    assign_b = AssignLayers(
        worker_id="w-2",
        model_name="test-model",
        layer_start=2,
        layer_end=4,
        total_layers=total_layers,
        is_first_stage=False,
        is_last_stage=True,
        prev_worker_id="w-1",
        first_worker_id="w-1",
    )

    engine_a = StageExecutionEngine(model_a, assign_a)
    engine_b = StageExecutionEngine(model_b, assign_b)

    req_id = "req-distributed-resilience-001"
    start_msg = InferenceStart(
        request_id=req_id,
        model_name="test-model",
        input_ids=[1, 5, 10],
        prompt="Tell me about distributed compute",
        max_new_tokens=4,
        temperature=0.0,
    )
    engine_b.session_configs[req_id] = start_msg

    # Mock the coordinator's local LLM backend to fail completely
    with patch("situs.inference_backend.generate_llm_answer", side_effect=RuntimeError("Coordinator local inference offline")):
        # 1. First stage forward pass on worker A
        frame_a = engine_a.start_inference(start_msg)
        assert frame_a.type == MessageType.FORWARD_ACTIVATION
        assert frame_a.target == "w-2"

        # 2. Last stage forward pass on worker B (samples from real distributed logits)
        act_a = ForwardActivation.model_validate(frame_a.payload)
        frame_b = engine_b.process_forward_activation(act_a)
        assert frame_b.type == MessageType.NEXT_TOKEN
        assert frame_b.target == "w-1"

        token_b = NextToken.model_validate(frame_b.payload)
        assert isinstance(token_b.token_id, int)
        assert token_b.token_id >= 0


def test_progressive_weight_deserialization_and_validation():
    """
    P0 Verification: Proves progressive weights are deserialized from real byte formats,
    validated for shapes and finite values, and hot-swapped into RealTransformerBlock.
    """
    hidden_dim = 64
    intermediate_size = 128
    num_heads = 4
    num_kv_heads = 2

    # 1. Generate valid serialized weights
    valid_bytes = generate_mock_layer_bytes(
        layer_idx=1,
        hidden_dim=hidden_dim,
        intermediate_size=intermediate_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        use_safetensors=True,
    )
    assert len(valid_bytes) > 0

    block = deserialize_and_validate_layer_weights(
        weights_data=valid_bytes,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        layer_idx=1,
        device="cpu",
    )
    assert isinstance(block, RealTransformerBlock)

    # 2. Test hot swap into model
    model = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=2,
        is_first_stage=True,
        is_last_stage=True,
        hidden_dim=hidden_dim,
        vocab_size=200,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        head_dim=hidden_dim // num_heads,
        architecture="llama",
        device="cpu",
        progressive=True,
    )
    assert model.is_layer_ready(1) is False

    model.hot_swap_layer(layer_index=1, weights=valid_bytes)
    assert model.is_layer_ready(1) is True
    assert isinstance(model.layers[1], RealTransformerBlock)

    # 3. Test rejection of corrupted weights (contains NaN)
    corrupted_dict = {k: v.clone() for k, v in block.state_dict().items()}
    corrupted_dict["q_proj.weight"][0, 0] = float("nan")

    with pytest.raises(ValueError, match="NaN or Inf"):
        deserialize_and_validate_layer_weights(
            weights_data=corrupted_dict,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
            layer_idx=1,
            device="cpu",
        )


def test_security_api_and_admin_key_protection():
    """
    P0 Verification: Verifies API key enforcement on /v1/* and admin key on /admin/*.
    """
    registry.clear()
    client = TestClient(app)

    # Set keys in configuration
    orig_api = config.api_key
    orig_admin = config.admin_key
    try:
        config.api_key = "secret-client-key"
        config.admin_key = "secret-admin-key"

        # 1. /v1/models without key -> 401
        res = client.get("/v1/models")
        assert res.status_code == 401

        # 2. /v1/models with wrong key -> 401
        res = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
        assert res.status_code == 401

        # 3. /v1/models with correct key -> 200
        res = client.get("/v1/models", headers={"Authorization": "Bearer secret-client-key"})
        assert res.status_code == 200

        # 4. /admin/partition without admin key -> 403
        res = client.post("/admin/partition?model_name=gemma3:4b")
        assert res.status_code == 403

        # 5. /admin/partition with correct admin key
        res = client.post(
            "/admin/partition?model_name=gemma3:4b",
            headers={"X-Admin-Key": "secret-admin-key"},
        )
        # Status code 400 because no workers alive, but not 403 (passed auth)
        assert res.status_code == 400
        assert "no alive workers" in res.json()["detail"].lower()
    finally:
        config.api_key = orig_api
        config.admin_key = orig_admin


@pytest.mark.asyncio
async def test_security_worker_websocket_token_and_duplicate_prevention():
    """
    P0 Verification: Verifies worker token authentication and duplicate worker rejection.
    """
    from unittest.mock import AsyncMock, MagicMock
    from situs.coordinator.app import worker_websocket

    registry.clear()
    relay.active_connections.clear()

    orig_token = config.worker_auth_token
    try:
        config.worker_auth_token = "worker-secret-pass"

        # 1. Worker connects without token -> closed with 1008
        mock_ws = AsyncMock()
        mock_ws.query_params = {}
        mock_ws.headers = {}
        await worker_websocket(mock_ws, worker_id="w-bad")
        mock_ws.close.assert_awaited_with(code=1008, reason="Authentication failed: invalid worker token")

        # 2. Worker connects with wrong token -> closed with 1008
        mock_ws_wrong = AsyncMock()
        mock_ws_wrong.query_params = {"token": "wrong-secret"}
        mock_ws_wrong.headers = {}
        await worker_websocket(mock_ws_wrong, worker_id="w-bad")
        mock_ws_wrong.close.assert_awaited_with(code=1008, reason="Authentication failed: invalid worker token")

        # 3. Duplicate worker ID rejection: worker w-dup is already connected
        relay.active_connections["w-dup"] = MagicMock()
        mock_ws_dup = AsyncMock()
        mock_ws_dup.query_params = {"token": "worker-secret-pass"}
        mock_ws_dup.headers = {}
        await worker_websocket(mock_ws_dup, worker_id="w-dup")
        mock_ws_dup.close.assert_awaited_with(
            code=1008, reason="Worker ID 'w-dup' is already actively connected"
        )
    finally:
        config.worker_auth_token = orig_token
        relay.active_connections.clear()


@pytest.mark.asyncio
async def test_security_frame_spoofing_prevention():
    """
    P0 Verification: Verifies that frames with forged source IDs are dropped.
    """
    from unittest.mock import AsyncMock, patch
    from situs.coordinator.app import worker_websocket
    from fastapi import WebSocketDisconnect

    registry.clear()
    relay.active_connections.clear()

    spoofed_frame = ProtocolFrame(
        type=MessageType.FORWARD_ACTIVATION,
        source="victim-worker",
        target="coordinator",
        payload={"dummy": "data"},
    )

    mock_ws = AsyncMock()
    mock_ws.query_params = {}
    mock_ws.headers = {}
    # First message is spoofed, second raises disconnect to exit loop
    mock_ws.receive_text.side_effect = [
        spoofed_frame.model_dump_json(),
        WebSocketDisconnect(1000),
    ]

    with patch.object(relay, "route_frame", new_callable=AsyncMock) as mock_route:
        await worker_websocket(mock_ws, worker_id="attacker-worker")
        # Ensure the spoofed frame was dropped and NOT routed
        mock_route.assert_not_awaited()


def test_inference_abort_cleanup():
    """
    Verifies that INFERENCE_ABORT properly aborts generation and cleans up KV cache.
    """
    model = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=2,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=64,
        vocab_size=200,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        head_dim=16,
        architecture="llama",
        device="cpu",
    )
    assign = AssignLayers(
        worker_id="w-abort",
        model_name="test-model",
        layer_start=0,
        layer_end=2,
        total_layers=2,
        is_first_stage=True,
        is_last_stage=False,
    )
    engine = StageExecutionEngine(model, assign)

    req_id = "req-to-abort"
    start_msg = InferenceStart(
        request_id=req_id,
        model_name="test-model",
        input_ids=[1, 2, 3],
        max_new_tokens=10,
    )
    engine.start_inference(start_msg)
    assert req_id in engine.session_kv_caches

    # Send abort signal
    engine.abort_inference(req_id)
    assert req_id not in engine.session_kv_caches
    assert req_id in engine.finished_sessions


def test_network_latency_and_speed_aware_partitioning():
    """
    Verifies that the DP makespan partitioning penalizes high-latency or slow workers
    and allocates more layers to low-latency, high-speed workers.
    """
    from situs.coordinator.registry import WorkerNode, optimal_heterogeneous_partition

    w_fast = WorkerNode(
        worker_id="w-fast",
        gpu_name="NVIDIA GPU Fast",
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=14 * 1024**3,
        device_type="cuda",
        last_heartbeat=100.0,
        compute_speed=20.0,
        network_latency_ms=2.0,  # Ultra-low 2ms LAN latency
    )
    w_slow = WorkerNode(
        worker_id="w-slow",
        gpu_name="NVIDIA GPU Slow",
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=14 * 1024**3,
        device_type="cuda",
        last_heartbeat=100.0,
        compute_speed=20.0,
        network_latency_ms=150.0,  # High 150ms WAN latency
    )

    meta = {
        "hidden_dim": 2048,
        "intermediate_size": 8192,
        "num_heads": 16,
        "num_kv_heads": 8,
        "vocab_size": 32000,
        "context_length": 2048,
        "torch_dtype": "bfloat16",
    }
    intervals = optimal_heterogeneous_partition([w_fast, w_slow], total_layers=32, model_metadata=meta)
    fast_count = intervals[0][1] - intervals[0][0]
    slow_count = intervals[1][1] - intervals[1][0]
    assert fast_count + slow_count == 32
    assert fast_count > slow_count, "Fast low-latency node must receive more layers than high-latency node"


def test_coordinator_request_limit_concurrency():
    """
    Verifies that when coordinator concurrency limit is reached, incoming requests
    receive HTTP 429 Too Many Requests.
    """
    from situs.coordinator.app import get_request_semaphore
    client = TestClient(app)

    sem = get_request_semaphore()
    orig_value = sem._value
    sem._value = 0  # Saturate semaphore

    try:
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": False},
        )
        assert resp.status_code == 429
        assert "Too Many Requests" in resp.json()["detail"]
    finally:
        sem._value = orig_value


def test_stage_model_loads_cached_weights_on_startup(tmp_path):
    """
    Verifies that load_hf_stage_model detects and hot-swaps cached layer weights
    from disk during initialization.
    """
    from situs.worker.loader import load_hf_stage_model
    from situs.worker.jit_streamer import save_layer_to_cache

    model_name = "test-cached-startup-model"
    # Create fake cached layer for layer 0
    weights = generate_mock_layer_bytes(layer_idx=0, hidden_dim=64)
    save_layer_to_cache(model_name, layer_index=0, data=weights)

    stage = load_hf_stage_model(
        model_name=model_name,
        layer_start=0,
        layer_end=2,
        total_layers=2,
        is_first_stage=True,
        is_last_stage=False,
        device="cpu",
        progressive=False,
        fast_boot=False,
    )
    # Layer 0 was found in cache and loaded
    assert stage.is_layer_ready(0)

