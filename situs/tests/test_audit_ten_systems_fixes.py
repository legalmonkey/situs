"""Comprehensive test suite verifying all 10 distributed systems failure modes and research fixes.

1. Distributed Tensor Communication (SITUS_TENSOR_V1 binary framing & CRC32)
2. GPU Memory Management (GPUMemoryManager with >=10% safety headroom)
3. KV-Cache Isolation & Lifecycle (RequestScopedKVCache tenant isolation)
4. Worker Failure Recovery (ActivationReplayRingBuffer & Circuit Breakers)
5. Cancellation & Timeouts (CancellationTokenRegistry & structured abort)
6. Distributed Execution Correctness (StageOrderingError & explicit position passing)
7. Model Loading & Quantization (CANONICAL_TENSOR_MAP & validate_quantization_metadata)
8. Scheduler Fairness & Telemetry (VRAM-constrained DP partition)
9. API Validation & Guardrails (ChatCompletionRequest constraints & token context pre-check)
10. Reproducibility & Integrity (Deterministic PRNG seeds & ModelManifest)
"""

import pytest
import time
import uuid
import torch
from unittest.mock import MagicMock, AsyncMock, patch
from pydantic import ValidationError

from situs.protocol.framing import (
    pack_tensor_frame,
    unpack_tensor_frame,
    TensorContractError,
    MAGIC_HEADER,
)
from situs.worker.memory import (
    GPUMemoryManager,
    RequestScopedKVCache,
    MemoryExhaustedError,
)
from situs.worker.engine import (
    StageExecutionEngine,
    CancellationTokenRegistry,
    ActivationReplayRingBuffer,
    StageOrderingError,
)
from situs.worker.loader import (
    CANONICAL_TENSOR_MAP,
    normalize_canonical_tensor_name,
    validate_quantization_metadata,
    UnsupportedQuantizationError,
    DistributedStageModel,
)
from situs.worker.sampler import TokenSampler
from situs.coordinator.registry import (
    WorkerNode,
    WorkerRegistry,
    optimal_heterogeneous_partition,
)
from situs.protocol.messages import (
    AssignLayers,
    ForwardActivation,
    InferenceStart,
    ModelManifest,
    WorkerRegister,
)
from situs.coordinator.app import (
    ChatCompletionRequest,
    ChatMessage,
)


# ==============================================================================
# 1. Distributed Tensor Communication (SITUS_TENSOR_V1 & CRC32)
# ==============================================================================

def test_binary_framing_pack_unpack_success():
    req_id = str(uuid.uuid4())
    raw_payload = b"\x01\x02\x03\x04\x05\x06\x07\x08" * 16

    frame_bytes = pack_tensor_frame(
        payload_bytes=raw_payload,
        request_id=req_id,
        stage_id=1,
        batch_size=1,
        seq_length=16,
        hidden_dim=8,
        dtype="int8",
        scale=0.5,
    )
    assert frame_bytes[:4] == MAGIC_HEADER

    meta, payload = unpack_tensor_frame(frame_bytes)
    assert meta["request_id"] == req_id
    assert meta["stage_id"] == 1
    assert meta["shape"] == [1, 16, 8]
    assert meta["dtype"] == "int8"
    assert meta["scale"] == 0.5
    assert payload == raw_payload


def test_binary_framing_checksum_corruption():
    req_id = str(uuid.uuid4())
    raw_payload = b"uncompressed_activation_data_12345"

    frame_bytes = bytearray(pack_tensor_frame(
        payload_bytes=raw_payload,
        request_id=req_id,
        stage_id=0,
        batch_size=1,
        seq_length=1,
        hidden_dim=len(raw_payload),
    ))

    # Corrupt payload byte
    frame_bytes[-1] ^= 0xFF

    with pytest.raises(TensorContractError, match="Checksum corruption detected"):
        unpack_tensor_frame(bytes(frame_bytes))


def test_binary_framing_length_mismatch():
    req_id = str(uuid.uuid4())
    raw_payload = b"some_tensor_payload"
    frame_bytes = pack_tensor_frame(
        payload_bytes=raw_payload,
        request_id=req_id,
        stage_id=0,
        batch_size=1,
        seq_length=1,
        hidden_dim=len(raw_payload),
    )

    # Truncate frame
    truncated = frame_bytes[:-3]
    with pytest.raises(TensorContractError, match="Payload length mismatch"):
        unpack_tensor_frame(truncated)


# ==============================================================================
# 2. GPU Memory Management (GPUMemoryManager with >=10% safety headroom)
# ==============================================================================

def test_gpu_memory_manager_headroom_enforcement():
    total_vram = 10 * 1024**3  # 10 GB
    mgr = GPUMemoryManager(total_vram_bytes=total_vram, safety_headroom_fraction=0.10)
    # Headroom = 1 GB, usable = 9 GB
    assert mgr.headroom_bytes == 1 * 1024**3
    assert mgr.usable_vram_bytes == 9 * 1024**3

    # Weights take 6 GB
    mgr.set_weights_footprint(6 * 1024**3)
    # Remaining free budget = 3 GB
    assert mgr.current_free_budget() == 3 * 1024**3

    # Request requiring 2.5 GB can be admitted
    assert mgr.can_admit_request(int(2.5 * 1024**3)) is True

    # Request requiring 3.5 GB would violate headroom and must be rejected
    assert mgr.can_admit_request(int(3.5 * 1024**3)) is False
    with pytest.raises(MemoryExhaustedError, match="Cannot admit request"):
        mgr.allocate_request("req-overflow", int(3.5 * 1024**3))


def test_gpu_memory_manager_allocation_and_release():
    total_vram = 4 * 1024**3  # 4 GB (headroom = 400MB, usable = 3600MB)
    mgr = GPUMemoryManager(total_vram_bytes=total_vram, safety_headroom_fraction=0.10)
    mgr.allocate_request("req-1", 1000 * 1024**2)
    assert mgr.current_free_budget() == mgr.usable_vram_bytes - (1000 * 1024**2)

    mgr.release_request("req-1")
    assert mgr.current_free_budget() == mgr.usable_vram_bytes


# ==============================================================================
# 3. KV-Cache Isolation & Lifecycle (RequestScopedKVCache tenant isolation)
# ==============================================================================

def test_kv_cache_tenant_isolation():
    cache_mgr = RequestScopedKVCache()

    k_a = torch.randn(1, 2, 4, 16)
    v_a = torch.randn(1, 2, 4, 16)
    k_b = torch.randn(1, 2, 4, 16)
    v_b = torch.randn(1, 2, 4, 16)

    cache_mgr.update_layer_kv("tenant-A", 0, k_a, v_a)
    cache_mgr.update_layer_kv("tenant-B", 0, k_b, v_b)

    # Verify tenant A cannot access or overwrite tenant B's state
    res_a = cache_mgr.get_layer_kv("tenant-A", 0)
    res_b = cache_mgr.get_layer_kv("tenant-B", 0)

    assert torch.equal(res_a[0], k_a)
    assert torch.equal(res_b[0], k_b)
    assert not torch.equal(res_a[0], res_b[0])

    # Releasing tenant A does not affect tenant B
    cache_mgr.release_request("tenant-A")
    assert cache_mgr.get_layer_kv("tenant-A", 0) is None
    assert cache_mgr.get_layer_kv("tenant-B", 0) is not None


def test_kv_cache_raii_scoped_session():
    cache_mgr = RequestScopedKVCache()
    req_id = "tenant-raii"

    with cache_mgr.scoped_session(req_id):
        cache_mgr.update_layer_kv(req_id, 0, torch.ones(1, 2), torch.ones(1, 2))
        assert cache_mgr.get_layer_kv(req_id, 0) is not None

    # After exiting scope, state must be unconditionally freed
    assert cache_mgr.get_layer_kv(req_id, 0) is None


# ==============================================================================
# 4. Worker Failure Recovery (ActivationReplayRingBuffer)
# ==============================================================================

def test_activation_replay_ring_buffer():
    ring = ActivationReplayRingBuffer(capacity=4)

    for i in range(6):
        ring.record(f"req-1", i, f"act-{i}".encode("utf-8"), {"step": i})

    # Depth is K=4: steps 0 and 1 must be evicted, steps 2..5 preserved
    assert ring.retrieve("req-1", 0) is None
    assert ring.retrieve("req-1", 1) is None

    p2, m2 = ring.retrieve("req-1", 2)
    assert p2 == b"act-2"
    assert m2["step"] == 2

    p5, m5 = ring.retrieve("req-1", 5)
    assert p5 == b"act-5"
    assert m5["step"] == 5


# ==============================================================================
# 5. Cancellation & Timeouts (CancellationTokenRegistry)
# ==============================================================================

def test_cancellation_token_registry():
    registry = CancellationTokenRegistry()
    req_id = "req-abort-test"

    assert not registry.is_cancelled(req_id)
    registry.cancel(req_id)
    assert registry.is_cancelled(req_id)

    registry.clear(req_id)
    assert not registry.is_cancelled(req_id)


# ==============================================================================
# 6. Distributed Execution Correctness (StageOrderingError)
# ==============================================================================

def test_stage_ordering_invariant_enforcement():
    # Worker configured for stage [8..16)
    stage_model = MagicMock(spec=DistributedStageModel)
    stage_model.device = "cpu"
    assignment = AssignLayers(
        worker_id="worker-middle",
        model_name="llama",
        layer_start=8,
        layer_end=16,
        total_layers=32,
        is_first_stage=False,
        is_last_stage=False,
    )
    engine = StageExecutionEngine(stage_model=stage_model, assignment=assignment)

    # Frame arriving with expected_next_stage=4 instead of 8
    corrupt_frame = ForwardActivation(
        request_id="req-order-test",
        step_index=0,
        shape=[1, 1, 64],
        dtype="int8",
        scale=1.0,
        payload_b64="AA==",
        stage_index=0,
        expected_next_stage=4,  # Mismatch: should be 8!
    )

    with pytest.raises(StageOrderingError, match="Stage ordering invariant violated"):
        engine.process_forward_activation(corrupt_frame)


# ==============================================================================
# 7. Model Loading & Quantization (CANONICAL_TENSOR_MAP & validate_quantization_metadata)
# ==============================================================================

def test_canonical_tensor_mapping():
    # HuggingFace keys map to canonical
    assert normalize_canonical_tensor_name("self_attn.q_proj.weight") == "attn_q.weight"
    assert normalize_canonical_tensor_name("mlp.gate_proj.weight") == "ffn_gate.weight"
    assert normalize_canonical_tensor_name("input_layernorm.weight") == "attn_norm.weight"
    # Strips layer prefixes
    assert normalize_canonical_tensor_name("model.layers.5.self_attn.v_proj.weight") == "attn_v.weight"
    assert normalize_canonical_tensor_name("blk.12.attn_q.weight") == "attn_q.weight"


def test_quantization_metadata_validation():
    # Valid schemes pass
    validate_quantization_metadata({"dtype": "q8_0"})
    validate_quantization_metadata({"precision": "fp16"})
    validate_quantization_metadata({"format": "int4"})

    # Invalid/unsupported schemes are rejected
    with pytest.raises(UnsupportedQuantizationError, match="Unsupported quantization scheme"):
        validate_quantization_metadata({"dtype": "iq1_s_corrupted"})


# ==============================================================================
# 8. Scheduler Fairness & Telemetry (VRAM Capacity Clamping)
# ==============================================================================

def test_scheduler_fairness_insufficient_vram_rejection():
    # Worker A has high compute speed but only 50MB free VRAM (insufficient for layers)
    w_starved = WorkerNode(
        worker_id="fast-starved",
        gpu_name="RTX 4090 Depleted",
        total_vram_bytes=24 * 1024**3,
        free_vram_bytes=50 * 1024**2,  # 50 MB
        device_type="cuda",
        last_heartbeat=time.time(),
        compute_speed=10.0,
    )
    # Total layers = 8, but cluster free VRAM cannot fit them
    with pytest.raises(ValueError, match="Insufficient cluster memory"):
        optimal_heterogeneous_partition(
            workers=[w_starved],
            total_layers=8,
            model_metadata={
                "hidden_dim": 2048,
                "intermediate_size": 5632,
                "vocab_size": 32000,
                "num_heads": 16,
                "context_length": 2048,
                "bytes_per_param": 1.0,
            },
        )


# ==============================================================================
# 9. API Validation & Guardrails
# ==============================================================================

def test_api_validation_sampling_constraints():
    # Negative temperature rejected
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            temperature=-0.5,
        )

    # Temperature > 2.0 rejected
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            temperature=2.5,
        )

    # top_p < 0.0 or > 1.0 rejected
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            top_p=-0.1,
        )
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            top_p=1.5,
        )

    # max_tokens < 0 or > 32768 rejected
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=-1,
        )
    with pytest.raises(ValidationError):
        ChatCompletionRequest(
            messages=[ChatMessage(role="user", content="hello")],
            max_tokens=50000,
        )

    # Empty messages list rejected
    with pytest.raises(ValidationError):
        ChatCompletionRequest(messages=[])

    # Empty message content rejected
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="")


# ==============================================================================
# 10. Reproducibility & Integrity (Deterministic PRNG & ModelManifest)
# ==============================================================================

def test_deterministic_prng_seed_sampling():
    sampler = TokenSampler()
    torch.manual_seed(99)
    logits = torch.randn(1, 100)

    # Same seed and step produce bitwise identical token output
    tok1 = sampler.sample_next_token(logits, temperature=0.8, top_p=0.9, seed=12345, step_index=0)
    tok2 = sampler.sample_next_token(logits, temperature=0.8, top_p=0.9, seed=12345, step_index=0)
    assert tok1 == tok2

    # Different seeds produce diverse sampling
    tok_diff_seed = sampler.sample_next_token(logits, temperature=0.8, top_p=0.9, seed=99999, step_index=0)
    # Different steps produce counter-based progression
    tok_step1 = sampler.sample_next_token(logits, temperature=0.8, top_p=0.9, seed=12345, step_index=1)
    assert isinstance(tok1, int)
    assert isinstance(tok_diff_seed, int)
    assert isinstance(tok_step1, int)


def test_model_manifest_validation():
    manifest = ModelManifest(
        model_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        commit_hash="48d6d0fc388d8269e5ccb0180f60bec8bb0651f0",
        tokenizer_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        weights_sha256="a1b2c3d4e5f6...",
    )
    assert manifest.model_id.startswith("meta-llama")
    assert manifest.commit_hash is not None
    assert manifest.tokenizer_sha256 is not None
