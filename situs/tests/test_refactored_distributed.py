"""Comprehensive tests for refactored distributed execution, memory-aware GPU partitioning,
error propagation, cancellation lifecycle, and sampler stabilization."""

import pytest
import torch
import numpy as np
import asyncio
from situs.protocol.messages import (
    WorkerRegister,
    MessageType,
    ProtocolFrame,
    InferenceStart,
    AssignLayers,
)
from situs.coordinator.registry import (
    WorkerRegistry,
    WorkerNode,
    optimal_heterogeneous_partition,
)
from situs.coordinator.relay import RelayHub
from situs.worker.sampler import TokenSampler
from situs.worker.loader import DistributedStageModel, RealTransformerBlock
from situs.worker.engine import StageExecutionEngine


def test_heterogeneous_gpu_partitioning():
    """Verify that a worker with more VRAM receives proportionally more layers."""
    reg = WorkerRegistry()
    # Worker 1: 4 GB VRAM
    reg.register_worker(WorkerRegister(
        worker_id="w-small",
        gpu_name="RTX 3050 Laptop (4GB)",
        device_type="cuda",
        total_vram_bytes=4 * 1024**3,
        free_vram_bytes=3 * 1024**3,
        compute_speed=1.0,
    ))
    # Worker 2: 12 GB VRAM
    reg.register_worker(WorkerRegister(
        worker_id="w-large",
        gpu_name="RTX 3060 Desktop (12GB)",
        device_type="cuda",
        total_vram_bytes=12 * 1024**3,
        free_vram_bytes=10 * 1024**3,
        compute_speed=2.5,
    ))

    assignments = reg.partition_model("gemma3:4b", total_layers=28)
    assert len(assignments) == 2

    w_small_assign = next(a for a in assignments if a.worker_id == "w-small")
    w_large_assign = next(a for a in assignments if a.worker_id == "w-large")

    small_layers = w_small_assign.layer_end - w_small_assign.layer_start
    large_layers = w_large_assign.layer_end - w_large_assign.layer_start

    assert small_layers + large_layers == 28
    assert large_layers >= small_layers
    assert w_small_assign.is_first_stage
    assert w_large_assign.is_last_stage


def test_telemetry_weighted_partition():
    """Verify that compute speed and network latency influence partitioning."""
    w1 = WorkerNode(
        worker_id="w1",
        gpu_name="GPU 1",
        total_vram_bytes=8 * 1024**3,
        free_vram_bytes=8 * 1024**3,
        device_type="cuda",
        last_heartbeat=0.0,
        compute_speed=1.0,
        network_latency_ms=50.0,
    )
    w2 = WorkerNode(
        worker_id="w2",
        gpu_name="GPU 2",
        total_vram_bytes=8 * 1024**3,
        free_vram_bytes=8 * 1024**3,
        device_type="cuda",
        last_heartbeat=0.0,
        compute_speed=4.0,
        network_latency_ms=5.0,
    )
    parts = optimal_heterogeneous_partition(
        [w1, w2],
        total_layers=24,
        model_metadata={
            "hidden_dim": 2048,
            "intermediate_size": 8192,
            "num_heads": 16,
            "num_kv_heads": 8,
            "vocab_size": 32000,
            "context_length": 2048,
        },
    )
    assert len(parts) == 2
    # w2 is 4x faster and lower latency, so it should receive at least as many or more layers
    w1_count = parts[0][1] - parts[0][0]
    w2_count = parts[1][1] - parts[1][0]
    assert w2_count >= w1_count
    assert w1_count + w2_count == 24


def test_sampler_handles_all_zero_and_nan_logits():
    """Verify sampler prevents gibberish fallback when logits are invalid."""
    sampler = TokenSampler()

    # 1. All-zero logits
    zero_logits = torch.zeros(1000)
    tok = sampler.sample_next_token(zero_logits)
    assert tok == 0  # Cleanly returns EOS (token 0) instead of uniform random sampling

    # 2. NaN logits
    nan_logits = torch.full((1000,), float("nan"))
    tok_nan = sampler.sample_next_token(nan_logits)
    assert tok_nan == 0

    # 3. Valid logits with strong peak
    peak_logits = torch.zeros(1000)
    peak_logits[42] = 100.0
    tok_peak = sampler.sample_next_token(peak_logits, temperature=0.1)
    assert tok_peak == 42


def test_engine_cancellation_and_kv_cleanup():
    """Verify that abort_inference completely flushes state and marks session as finished."""
    assignment = AssignLayers(
        worker_id="w-test",
        model_name="gemma3:4b",
        layer_start=0,
        layer_end=4,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=True,
    )
    model = DistributedStageModel(
        layer_start=0,
        layer_end=4,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=True,
        hidden_dim=64,
        intermediate_size=128,
        num_heads=4,
        num_kv_heads=2,
        vocab_size=100,
        device="cpu",
    )
    engine = StageExecutionEngine(stage_model=model, assignment=assignment)

    req_id = "req-cancel-123"
    engine.session_kv_caches[req_id] = [torch.zeros((1, 2, 5, 16))]
    engine.session_configs[req_id] = InferenceStart(request_id=req_id, model_name="gemma3:4b", input_ids=[1, 2, 3])
    engine.session_token_histories[req_id] = [1, 2, 3]
    engine.session_fast_boot_sequences[req_id] = [1, 2]

    assert req_id in engine.session_kv_caches

    # Abort
    engine.abort_inference(req_id)

    assert req_id not in engine.session_kv_caches
    assert req_id not in engine.session_configs
    assert req_id not in engine.session_token_histories
    assert req_id not in engine.session_fast_boot_sequences
    assert req_id in engine.finished_sessions


@pytest.mark.asyncio
async def test_relay_worker_disconnect_fails_inflight_streams():
    """Verify that when a worker disconnects, in-flight response streams receive an error."""
    relay = RelayHub()
    q = relay.register_response_stream("req-inflight-456")

    # Connect dummy worker
    class DummyWS:
        async def accept(self):
            pass
        async def send_text(self, text):
            pass

    await relay.connect_worker("worker-victim", DummyWS())
    assert relay.is_worker_connected("worker-victim")

    # Disconnect
    relay.disconnect_worker("worker-victim")
    assert not relay.is_worker_connected("worker-victim")

    # The queue must receive an ERROR frame immediately
    frame = await asyncio.wait_for(q.get(), timeout=1.0)
    assert frame.type == MessageType.ERROR
    assert "disconnected" in frame.payload["error"]

    # Test cleanup drains queue
    relay.cleanup_response_stream("req-inflight-456")
    assert "req-inflight-456" not in relay.response_streams


def test_real_transformer_block_custom_head_dim():
    """Verify that RealTransformerBlock correctly supports explicit head_dim (Gemma architecture)."""
    block = RealTransformerBlock(
        hidden_dim=2560,
        num_heads=8,
        num_kv_heads=4,
        intermediate_size=10240,
        layer_idx=0,
        head_dim=256,
    )
    assert block.head_dim == 256
    assert block.q_proj.out_features == 2048
    assert block.k_proj.out_features == 1024
    assert block.v_proj.out_features == 1024
    assert block.o_proj.in_features == 2048

    x = torch.randn(1, 2, 2560)
    out, new_kv = block(x)
    assert out.shape == torch.Size([1, 2, 2560])
    assert not torch.isnan(out).any()


def test_resolve_model_metadata_includes_head_dim():
    """Verify that resolve_model_metadata correctly returns head_dim."""
    from situs.coordinator.registry import resolve_model_metadata
    meta = resolve_model_metadata("gemma3:4b")
    assert "head_dim" in meta
    assert meta["head_dim"] == 256
    assert meta["num_heads"] == 8
    assert meta["num_kv_heads"] == 4
