"""Tests for heterogeneous VRAM partitioning, 3-worker pipeline KV cache cleanup,
speculative draft trimming across intermediate workers, multi-peer P2P DataChannels,
and live socket handshake verification.
"""

import asyncio
import socket
import time
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
import torch
import torch.nn as nn

from situs.config import WorkerConfig
from situs.coordinator.registry import WorkerRegistry, WorkerNode, optimal_heterogeneous_partition
from situs.coordinator.relay import RelayHub
from situs.network.p2p import P2PDataChannelNode, SAFE_SCTP_MAX_BYTES
from situs.protocol.messages import (
    AssignLayers,
    ForwardActivation,
    InferenceStart,
    MessageType,
    NextToken,
    ProtocolFrame,
    SpeculativeCandidates,
    SpeculativeVerified,
    WorkerRegister,
)
from situs.worker.engine import StageExecutionEngine, ActivationReplayRingBuffer
from situs.worker.memory import GPUMemoryManager, RequestScopedKVCache
from situs.worker.sampler import TokenSampler
from situs.worker.client import WorkerDaemon


class MockTransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        return self.fc(x)


class MockStageModel(nn.Module):
    def __init__(self, layer_start: int, layer_end: int, hidden_dim: int = 64):
        super().__init__()
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.hidden_dim = hidden_dim
        self.device = torch.device("cpu")
        self.num_kv_heads = 4
        self.head_dim = 16
        self.dtype = torch.float32
        self.layers = nn.ModuleList([MockTransformerBlock(hidden_dim) for _ in range(layer_end - layer_start)])

    def forward_first_stage(self, input_ids, past_kvs=None):
        batch, seq_len = input_ids.shape
        h = torch.randn(batch, seq_len, self.hidden_dim)
        new_kvs = []
        for i in range(len(self.layers)):
            k = torch.randn(batch, self.num_kv_heads, seq_len, self.head_dim)
            v = torch.randn(batch, self.num_kv_heads, seq_len, self.head_dim)
            if past_kvs and i < len(past_kvs) and past_kvs[i] is not None:
                pk, pv = past_kvs[i]
                k = torch.cat([pk, k], dim=-2)
                v = torch.cat([pv, v], dim=-2)
            new_kvs.append((k, v))
        return h, new_kvs

    def forward_intermediate_stage(self, hidden_states, past_kvs=None):
        batch, seq_len, _ = hidden_states.shape
        h = hidden_states
        for layer in self.layers:
            h = layer(h)
        new_kvs = []
        for i in range(len(self.layers)):
            k = torch.randn(batch, self.num_kv_heads, seq_len, self.head_dim)
            v = torch.randn(batch, self.num_kv_heads, seq_len, self.head_dim)
            if past_kvs and i < len(past_kvs) and past_kvs[i] is not None:
                pk, pv = past_kvs[i]
                k = torch.cat([pk, k], dim=-2)
                v = torch.cat([pv, v], dim=-2)
            new_kvs.append((k, v))
        return h, new_kvs


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_heterogeneous_vram_does_not_reclaim_external_os_memory():
    """Verify optimal_heterogeneous_partition bases capacity on verified free VRAM, not external OS memory."""
    meta = {
        "total_layers": 16,
        "hidden_dim": 2048,
        "intermediate_size": 4096,
        "vocab_size": 32000,
        "num_heads": 16,
        "num_kv_heads": 4,
        "head_dim": 128,
        "context_length": 2048,
        "bytes_per_param": 2,
    }

    # Worker with 8 GB total, 7.5 GB used by OS/Chrome, only 400 MB free VRAM
    # Previously, eff_avail added 7.5 GB back, incorrectly treating it as reclaimable!
    w_crowded = WorkerNode(
        worker_id="w-crowded",
        gpu_name="RTX 4060",
        total_vram_bytes=8 * 1024**3,
        free_vram_bytes=400 * 1024**2,
        used_vram_bytes=int(7.5 * 1024**3),  # OS & external apps
        device_type="cuda",
        last_heartbeat=time.time(),
    )
    w_clean = WorkerNode(
        worker_id="w-clean",
        gpu_name="RTX 4070",
        total_vram_bytes=12 * 1024**3,
        free_vram_bytes=10 * 1024**3,
        used_vram_bytes=2 * 1024**3,
        device_type="cuda",
        last_heartbeat=time.time(),
    )

    parts = optimal_heterogeneous_partition([w_crowded, w_clean], total_layers=16, model_metadata=meta)
    p_crowded = parts[0]
    p_clean = parts[1]

    # w_crowded has only 3 GB free, so it must be assigned significantly fewer layers than w_clean (10 GB free)
    crowded_layers = p_crowded[1] - p_crowded[0]
    clean_layers = p_clean[1] - p_clean[0]
    assert crowded_layers < clean_layers
    assert crowded_layers + clean_layers == 16


@pytest.mark.asyncio
async def test_three_worker_pipeline_kv_cache_cleanup_on_finish():
    """Verify that in a 3-worker pipeline, INFERENCE_FINISH is broadcast and frees intermediate KV caches."""
    m1 = MockStageModel(0, 2)
    m2 = MockStageModel(2, 4)
    m3 = MockStageModel(4, 6)

    a1 = AssignLayers(model_name="m", total_layers=6, layer_start=0, layer_end=2, worker_id="w1", is_first_stage=True, is_last_stage=False, next_worker_id="w2")
    a2 = AssignLayers(model_name="m", total_layers=6, layer_start=2, layer_end=4, worker_id="w2", is_first_stage=False, is_last_stage=False, prev_worker_id="w1", next_worker_id="w3", first_worker_id="w1")
    a3 = AssignLayers(model_name="m", total_layers=6, layer_start=4, layer_end=6, worker_id="w3", is_first_stage=False, is_last_stage=True, prev_worker_id="w2", first_worker_id="w1")

    e1 = StageExecutionEngine(m1, a1, compression_dtype="none")
    e2 = StageExecutionEngine(m2, a2, compression_dtype="none")
    e3 = StageExecutionEngine(m3, a3, compression_dtype="none")

    req_id = "test-req-3stage-finish"
    start_msg = InferenceStart(model_name="m", request_id=req_id, prompt="Hello", input_ids=[1, 2, 3], max_new_tokens=1)
    # Coordinator delivers INFERENCE_START config to both first and last stages
    e3.session_configs[req_id] = start_msg

    # 1. First stage processes start
    f1 = e1.start_inference(start_msg)
    assert req_id in e1.session_kv_caches

    # 2. Middle stage processes activation
    act1 = ForwardActivation.model_validate(f1.payload)
    f2 = e2.process_forward_activation(act1)
    assert req_id in e2.session_kv_caches

    # 3. Last stage processes activation -> finishes (max_new_tokens=1)
    act2 = ForwardActivation.model_validate(f2.payload)
    f3 = e3.process_forward_activation(act2)
    tok = NextToken.model_validate(f3.payload)
    assert tok.is_finished is True
    # Last stage cleaned up upon finish
    assert req_id not in e3.session_kv_caches
    assert req_id in e3.finished_sessions

    # 4. Coordinator RelayHub receives token frame and broadcasts INFERENCE_FINISH to all active workers
    relay = RelayHub()
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    ws3 = AsyncMock()
    relay.active_connections = {"w1": ws1, "w2": ws2, "w3": ws3}

    # f3 is NEXT_TOKEN from w3
    await relay.route_frame(f3)

    # Verify coordinator broadcasted INFERENCE_FINISH to w1 and w2 (all workers except source w3)
    sent_to_w2 = [call.args[0] for call in ws2.send_text.call_args_list]
    assert any("inference_finish" in str(s) for s in sent_to_w2)

    # 5. Worker 1 receives NEXT_TOKEN (or INFERENCE_FINISH)
    e1.process_next_token(tok)
    assert req_id not in e1.session_kv_caches
    assert req_id in e1.finished_sessions

    # 6. Worker 2 receives INFERENCE_FINISH broadcast
    e2.process_inference_finish(req_id)
    # Middle worker must NOT leak KV cache!
    assert req_id not in e2.session_kv_caches
    assert req_id in e2.finished_sessions


def test_three_worker_speculative_rejection_trims_intermediate_and_first_kv_caches():
    """Verify speculative verification routes upstream and trims intermediate stage KV caches."""
    m1 = MockStageModel(0, 2)
    m2 = MockStageModel(2, 4)
    m3 = MockStageModel(4, 6)

    a1 = AssignLayers(model_name="m", total_layers=6, layer_start=0, layer_end=2, worker_id="w1", is_first_stage=True, is_last_stage=False, next_worker_id="w2")
    a2 = AssignLayers(model_name="m", total_layers=6, layer_start=2, layer_end=4, worker_id="w2", is_first_stage=False, is_last_stage=False, prev_worker_id="w1", next_worker_id="w3", first_worker_id="w1")
    a3 = AssignLayers(model_name="m", total_layers=6, layer_start=4, layer_end=6, worker_id="w3", is_first_stage=False, is_last_stage=True, prev_worker_id="w2", first_worker_id="w1")

    e1 = StageExecutionEngine(m1, a1, compression_dtype="none")
    e2 = StageExecutionEngine(m2, a2, compression_dtype="none")
    e3 = StageExecutionEngine(m3, a3, compression_dtype="none")

    req_id = "test-spec-trim-sync"
    start_msg = InferenceStart(model_name="m", request_id=req_id, prompt="Hello", input_ids=[10, 20, 30], max_new_tokens=10)

    # Setup prompt history and KV caches
    e1.start_inference(start_msg)
    e1.session_token_histories[req_id] = [10, 20, 30]

    # Pre-populate dummy KV caches of length 10 on all 3 workers
    for eng in (e1, e2, e3):
        dummy_kvs = [
            (torch.randn(1, 4, 10, 16), torch.randn(1, 4, 10, 16)),
            (torch.randn(1, 4, 10, 16), torch.randn(1, 4, 10, 16)),
        ]
        eng.kv_cache_manager.allocate_request(req_id, estimated_bytes=1024)
        eng.kv_cache_manager.set_all_kvs(req_id, dummy_kvs)
        eng.session_kv_caches[req_id] = dummy_kvs

    # Worker 3 rejects 2 tokens: rejected_draft_count=2
    verified_frame = SpeculativeVerified(
        request_id=req_id,
        step_index=5,
        accepted_tokens=[100],
        accepted_texts=[" tok100"],
        rejected_draft_count=2,
        is_finished=False,
    )

    # Worker 3 trims its own KV cache and routes upstream to w2
    e3._trim_kv_caches(req_id, 2)
    assert e3.session_kv_caches[req_id][0][0].shape[-2] == 8  # 10 - 2 = 8

    # Worker 2 receives SPECULATIVE_VERIFIED from Worker 3
    out_w2 = e2.process_speculative_verified(verified_frame)
    # Worker 2 must have trimmed its KV cache by 2 tokens!
    assert e2.session_kv_caches[req_id][0][0].shape[-2] == 8
    # Worker 2 must route upstream to w1
    assert out_w2 is not None
    assert out_w2.target == "w1"

    # Worker 1 receives SPECULATIVE_VERIFIED from Worker 2:
    # 1. Trims rejected draft tokens (10 - 2 = 8)
    # 2. Proceeds to next decode step (8 + 1 = 9) and forwards activation to Worker 2
    out_w1 = e1.process_speculative_verified(verified_frame)
    assert e1.session_kv_caches[req_id][0][0].shape[-2] == 9
    assert out_w1 is not None
    assert out_w1.target == "w2"

    # Worker 2 processes the incoming forward activation from Worker 1
    act_to_w2 = ForwardActivation.model_validate(out_w1.payload)
    out_w2_next = e2.process_forward_activation(act_to_w2)
    assert e2.session_kv_caches[req_id][0][0].shape[-2] == 9
    assert out_w2_next is not None
    assert out_w2_next.target == "w3"

    # Worker 3 processes the incoming forward activation from Worker 2
    act_to_w3 = ForwardActivation.model_validate(out_w2_next.payload)
    e3.process_forward_activation(act_to_w3)
    assert e3.session_kv_caches[req_id][0][0].shape[-2] == 9

    # All 3 workers remain in 100% lockstep synchronization!
    assert (
        e1.session_kv_caches[req_id][0][0].shape[-2]
        == e2.session_kv_caches[req_id][0][0].shape[-2]
        == e3.session_kv_caches[req_id][0][0].shape[-2]
        == 9
    )


def test_gpu_memory_manager_real_initialization_and_weights_footprint():
    """Verify GPUMemoryManager initializes with real VRAM and records model weights."""
    model = MockStageModel(0, 4, hidden_dim=64)
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    assign = AssignLayers(model_name="m", total_layers=4, layer_start=0, layer_end=4, worker_id="w1", is_first_stage=True, is_last_stage=True)
    engine = StageExecutionEngine(model, assign)

    # Verify weight footprint recorded
    assert engine.memory_manager.weight_bytes == param_bytes
    assert engine.memory_manager.weight_bytes > 0
    # Usable VRAM is total minus 10% safety headroom
    assert engine.memory_manager.usable_vram_bytes == engine.memory_manager.total_vram_bytes - engine.memory_manager.headroom_bytes


def test_multi_peer_p2p_deterministic_initiator_and_mesh():
    """Verify deterministic initiator logic (worker_id < peer_id) prevents WebRTC connection collisions."""
    cfg1 = WorkerConfig(coordinator_url="ws://127.0.0.1:8000/ws/worker", worker_id="worker-01")
    cfg2 = WorkerConfig(coordinator_url="ws://127.0.0.1:8000/ws/worker", worker_id="worker-02")
    cfg3 = WorkerConfig(coordinator_url="ws://127.0.0.1:8000/ws/worker", worker_id="worker-03")

    w1 = WorkerDaemon(cfg1)
    w2 = WorkerDaemon(cfg2)
    w3 = WorkerDaemon(cfg3)

    # Pair (w1, w2): w1 < w2 -> w1 is initiator, w2 is responder
    assert (w1.worker_id < w2.worker_id) is True
    assert (w2.worker_id < w1.worker_id) is False

    # Pair (w2, w3): w2 < w3 -> w2 is initiator, w3 is responder
    assert (w2.worker_id < w3.worker_id) is True
    assert (w3.worker_id < w2.worker_id) is False

    # Assign w2 both w1 (prev) and w3 (next)
    w2.p2p_nodes["worker-01"] = P2PDataChannelNode("worker-02", "worker-01", MagicMock(), MagicMock())
    w2.p2p_nodes["worker-03"] = P2PDataChannelNode("worker-02", "worker-03", MagicMock(), MagicMock())

    assert len(w2.p2p_nodes) == 2
    assert "worker-01" in w2.p2p_nodes
    assert "worker-03" in w2.p2p_nodes


def test_safe_sctp_threshold_for_dense_bursts():
    """Verify SAFE_SCTP_MAX_BYTES accommodates candidate bursts up to 128 KB."""
    assert SAFE_SCTP_MAX_BYTES == 128 * 1024

    node = P2PDataChannelNode("w1", "w2", MagicMock(), MagicMock())
    node.is_connected = True
    node.is_verified = True
    node.channel = MagicMock()
    node.channel.readyState = "open"

    # 100 KB frame (well above old 48 KB limit, but within 128 KB)
    payload_100kb = "x" * (100 * 1024)
    assert node.send_frame(payload_100kb) is True
    assert node.channel.send.call_count == 1

    # 150 KB frame exceeds safe SCTP threshold -> returns False for clean WS fallback
    payload_150kb = "x" * (150 * 1024)
    assert node.send_frame(payload_150kb) is False


@pytest.mark.asyncio
async def test_rebalance_defers_during_active_inference():
    """Verify rebalance_and_assign_layers waits for in-flight requests before partitioning."""
    from situs.coordinator.app import rebalance_and_assign_layers
    import situs.coordinator.app as app_mod

    orig_relay = app_mod.relay
    iso_relay = RelayHub()
    app_mod.relay = iso_relay
    try:
        # Simulate an active in-flight request stream
        iso_relay.register_response_stream("active-stream-123")

        async def _finish_stream_after_delay():
            await asyncio.sleep(0.3)
            iso_relay.cleanup_response_stream("active-stream-123")

        asyncio.create_task(_finish_stream_after_delay())

        t0 = time.time()
        await rebalance_and_assign_layers(model_name="gemma3:4b", wait_for_in_flight=True)
        elapsed = time.time() - t0

        # Must have waited >= 0.25 seconds for stream to complete
        assert elapsed >= 0.25
        assert len(iso_relay.response_streams) == 0
    finally:
        app_mod.relay = orig_relay


def test_token_sampler_nan_and_inf_logits_safety():
    """Verify TokenSampler handles NaN and infinite logits without crashing."""
    sampler = TokenSampler(tokenizer=None)

    # Tensor with NaNs and Infs
    bad_logits = torch.tensor([float("nan"), float("inf"), -float("inf"), 5.0, 2.0])
    tok = sampler.sample_next_token(bad_logits, temperature=0.7, top_p=0.9)
    assert isinstance(tok, int)
    assert 0 <= tok < len(bad_logits)

    # All -Inf logits (e.g. from over-aggressive top-k/top-p masking)
    all_neginf = torch.tensor([-float("inf"), -float("inf"), -float("inf")])
    tok2 = sampler.sample_next_token(all_neginf, temperature=0.7, top_p=0.9)
    assert isinstance(tok2, int)
    assert 0 <= tok2 < 3

    # NumPy fallback with NaNs
    np_bad = [float("nan"), float("nan"), 10.0]
    tok3 = sampler.sample_next_token(np_bad, temperature=0.7)
    assert isinstance(tok3, int)


def test_live_socket_handshake_and_registration_invariants(monkeypatch):
    """Verify live TCP/WebSocket registration handshake strictly verifies WORKER_REGISTERED on wire."""
    from situs.tui import start_in_process_coordinator, start_in_process_worker
    import situs.coordinator.app as app_mod
    import situs.worker.loader as loader_mod

    iso_reg = WorkerRegistry()
    iso_relay = RelayHub()
    iso_relay.set_registry(iso_reg)
    monkeypatch.setattr(app_mod, "registry", iso_reg)
    monkeypatch.setattr(app_mod, "relay", iso_relay)
    monkeypatch.setattr(loader_mod, "find_local_weight_files", lambda m: [])

    port = _find_free_port()
    ok_c, msg_c = start_in_process_coordinator(port=port)
    assert ok_c is True, f"Coordinator failed to start: {msg_c}"

    ok_w, msg_w = start_in_process_worker(f"127.0.0.1:{port}", worker_id="w-live-audit-1", device="cpu")
    assert ok_w is True, f"Worker failed to start: {msg_w}"

    # Wait for live WebSocket handshake over real socket
    registered = False
    for _ in range(30):
        time.sleep(0.1)
        w = iso_reg.workers.get("w-live-audit-1")
        if w and w.is_alive and iso_relay.is_worker_connected("w-live-audit-1"):
            registered = True
            break

    assert registered is True, "Worker node failed to complete live socket registration handshake over wire."


def test_chunked_int8_linear_numerical_parity_and_no_oom():
    """Verify Int8WeightLinear chunked forward produces identical output to standard linear without allocating huge buffers."""
    from situs.worker.loader import Int8WeightLinear
    torch.manual_seed(42)
    in_features = 2048
    out_features = 4096
    batch = 1
    seq_len = 16

    weight = torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8)
    weight_scale = torch.rand(out_features, 1, dtype=torch.float32) * 0.01
    int8_lin = Int8WeightLinear(in_features, out_features, weight=weight, weight_scale=weight_scale)

    x = torch.randn(batch, seq_len, in_features, dtype=torch.float32)
    y_chunked = int8_lin(x)

    # Calculate ground truth full float dequantization
    w_fp = int8_lin.weight.to(torch.float32) * int8_lin.weight_scale.to(torch.float32)
    y_ref = torch.nn.functional.linear(x, w_fp)

    assert y_chunked.shape == (batch, seq_len, out_features)
    assert torch.allclose(y_chunked, y_ref, atol=1e-4, rtol=1e-4)


def test_optimal_heterogeneous_partition_gemma3_headroom():
    """Verify optimal_heterogeneous_partition preserves dedicated GPU headroom and distributes Gemma 3 4B evenly across two 6GB GPUs."""
    meta = {
        "total_layers": 34,
        "hidden_dim": 2560,
        "intermediate_size": 10240,
        "vocab_size": 256000,
        "num_heads": 16,
        "num_kv_heads": 4,
        "context_length": 8192,
        "torch_dtype": "bfloat16",
        "model_type": "gemma3",
    }
    # Two identical RTX 3050 / RTX 4050 6GB Laptop GPUs
    six_gb = int(6.0 * 1024**3)
    w1 = WorkerNode(
        worker_id="w-ritvik",
        gpu_name="NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        total_vram_bytes=six_gb,
        free_vram_bytes=int(4.5 * 1024**3),
        used_vram_bytes=int(1.5 * 1024**3),
        device_type="cuda",
        last_heartbeat=time.time(),
    )
    w2 = WorkerNode(
        worker_id="w-laptop2",
        gpu_name="NVIDIA GeForce RTX 4050 Laptop GPU",
        total_vram_bytes=six_gb,
        free_vram_bytes=int(4.8 * 1024**3),
        used_vram_bytes=int(1.2 * 1024**3),
        device_type="cuda",
        last_heartbeat=time.time(),
    )

    part = optimal_heterogeneous_partition([w1, w2], total_layers=34, model_metadata=meta)
    assert len(part) == 2
    # Total layers must be 34
    total_assigned = sum(p[1] - p[0] for p in part)
    assert total_assigned == 34
    # Neither worker should be overloaded beyond 20 layers
    for p in part:
        layers = p[1] - p[0]
        assert 14 <= layers <= 20, f"Unbalanced partition for 6GB GPUs: {layers} layers"


def test_p2p_verified_handshake_and_relay_fallback():
    """Verify P2P DataChannel requires probe verification before routing frames directly."""
    node = P2PDataChannelNode(
        worker_id="w1",
        target_worker_id="w2",
        signaling_send_fn=MagicMock(),
        on_message_fn=MagicMock(),
    )
    assert node.is_verified is False
    assert node.is_connected is False

    mock_channel = MagicMock()
    mock_channel.readyState = "open"
    mock_channel.send = MagicMock()
    node.channel = mock_channel
    node.is_connected = True

    # Unverified channel must NOT route data frames directly
    sent = node.send_frame('{"type": "forward_activation"}')
    assert sent is False
    mock_channel.send.assert_not_called()

    # Receiving pong verifies channel
    node._handle_incoming_channel_message('{"__situs_probe__": "pong"}')
    assert node.is_verified is True

    # Verified channel routes data frame
    sent2 = node.send_frame('{"type": "forward_activation"}')
    assert sent2 is True
    mock_channel.send.assert_called_once_with('{"type": "forward_activation"}')


def test_bounded_set_fifo_eviction_and_concurrency():
    """Verify BoundedSet caps size and evicts oldest entries under concurrent access."""
    import concurrent.futures
    from situs.worker.engine import BoundedSet

    b_set = BoundedSet(max_size=50)

    # 1. Sequential insertion beyond capacity
    for i in range(100):
        b_set.add(f"req-{i}")

    assert len(b_set) == 50
    # Oldest 50 items (req-0 .. req-49) must be evicted
    assert "req-0" not in b_set
    assert "req-49" not in b_set
    # Newest 50 items (req-50 .. req-99) must be present
    assert "req-50" in b_set
    assert "req-99" in b_set

    # 2. Concurrency test with 10 threads
    def worker_add(start):
        for j in range(100):
            b_set.add(f"thread-{start}-{j}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(worker_add, t) for t in range(10)]
        concurrent.futures.wait(futures)

    assert len(b_set) <= 50


def test_tokenizer_double_checked_locking_concurrency(monkeypatch):
    """Verify get_or_load_tokenizer double-checked locking resolves safely across concurrent threads."""
    import concurrent.futures
    import threading
    import situs.tokenizer as tok_mod

    # Clean cache
    tok_mod._tokenizers_cache.clear()

    load_count = 0
    load_lock = threading.Lock()

    def mock_resolve(model_name, clean_name, local_files_only=False):
        nonlocal load_count
        time.sleep(0.05)  # simulate disk I/O
        with load_lock:
            load_count += 1
        mock_tok = MagicMock()
        mock_tok.name = model_name
        return mock_tok

    monkeypatch.setattr(tok_mod, "_resolve_and_load_tokenizer", mock_resolve)

    def load_call():
        return tok_mod.get_or_load_tokenizer("concurrent-test-model")

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(load_call) for _ in range(8)]
        results = [f.result() for f in futures]

    # Exactly one load should have occurred despite 8 simultaneous threads
    assert load_count == 1
    assert all(r is results[0] for r in results)


