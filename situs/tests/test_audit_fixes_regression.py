"""
Regression tests covering all 5 audit issues:
1. Hardcoded worker secret removed from config defaults.
2. Rejection of incompatible embedding and head weight tensors (no partial copying, strict type checking).
3. Explicit validation and rejection of unsupported model architectures.
4. Scheduler requires verified model metadata, verified context length, precision, and avoids synthetic GPU speed fabrication.
5. Out-of-range token IDs rejected with ValueError instead of silent clamping.
"""

import os
import pytest
import torch
from unittest.mock import patch

from situs.config import CoordinatorConfig, WorkerConfig, INSECURE_SECRETS
from situs.worker.loader import (
    DistributedStageModel,
    SUPPORTED_ARCHITECTURES,
    is_architecture_supported,
)
from situs.coordinator.registry import (
    WorkerNode,
    optimal_heterogeneous_partition,
)


# =====================================================================
# Issue 1: Hardcoded worker secret default removed
# =====================================================================

def test_coordinator_config_no_default_hardcoded_secret():
    """Ensure CoordinatorConfig worker_auth_token defaults to None, never 'situs-cluster-secret'."""
    with patch.dict(os.environ, {}, clear=True):
        cfg = CoordinatorConfig()
        assert cfg.worker_auth_token is None
        assert cfg.worker_auth_token != "situs-cluster-secret"


def test_worker_config_no_default_hardcoded_secret():
    """Ensure WorkerConfig auth_token and worker_auth_token default to None, never 'situs-cluster-secret'."""
    with patch.dict(os.environ, {}, clear=True):
        cfg = WorkerConfig(worker_id="worker-test-1")
        assert cfg.auth_token is None
        assert cfg.worker_auth_token is None
        assert cfg.auth_token != "situs-cluster-secret"


def test_worker_config_synchronizes_auth_token_and_worker_auth_token():
    """Ensure passing either worker_auth_token or auth_token populates both fields."""
    cfg1 = WorkerConfig(worker_id="w-1", worker_auth_token="custom-token-123")
    assert cfg1.auth_token == "custom-token-123"
    assert cfg1.worker_auth_token == "custom-token-123"

    cfg2 = WorkerConfig(worker_id="w-2", auth_token="custom-token-456")
    assert cfg2.auth_token == "custom-token-456"
    assert cfg2.worker_auth_token == "custom-token-456"


def test_production_mode_rejects_insecure_secrets():
    """Ensure both CoordinatorConfig and WorkerConfig reject default/insecure secrets in production mode."""
    for bad_secret in INSECURE_SECRETS:
        with pytest.raises(ValueError, match="Insecure worker_auth_token"):
            CoordinatorConfig(
                environment="production",
                worker_auth_token=bad_secret,
                admin_key="secure-admin-12345",
                api_key="secure-api-12345",
            )
        with pytest.raises(ValueError, match="Insecure auth_token"):
            WorkerConfig(
                worker_id="w-prod",
                environment="production",
                auth_token=bad_secret,
            )
        with pytest.raises(ValueError, match="Insecure worker_auth_token"):
            WorkerConfig(
                worker_id="w-prod-2",
                environment="production",
                auth_token="valid-secret-token",
                worker_auth_token=bad_secret,
            )


# =====================================================================
# Issue 2: Embedding and output-head weights partial copy rejected
# =====================================================================

def test_load_embedding_weights_rejects_shape_mismatch():
    """Ensure load_embedding_weights raises ValueError on shape mismatch rather than copying partially."""
    model = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=4,
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
    # Target shape is (200, 64)
    # Shape with mismatched rows
    incompatible_rows = torch.randn(150, 64)
    with pytest.raises(ValueError, match="Embedding weight shape mismatch"):
        model.load_embedding_weights(incompatible_rows)

    # Shape with mismatched columns
    incompatible_cols = torch.randn(200, 32)
    with pytest.raises(ValueError, match="Embedding weight shape mismatch"):
        model.load_embedding_weights(incompatible_cols)

    # Rejects non-tensor input
    with pytest.raises(ValueError, match="Expected torch.Tensor"):
        model.load_embedding_weights("not-a-tensor")

    # Matching shape succeeds
    matching = torch.randn(200, 64)
    model.load_embedding_weights(matching)
    assert torch.allclose(model.embed_tokens.weight, matching)


def test_load_head_weights_rejects_shape_mismatch():
    """Ensure load_head_weights raises ValueError on shape mismatch for LM head or RMSNorm."""
    model = DistributedStageModel(
        layer_start=2,
        layer_end=4,
        total_layers=4,
        is_first_stage=False,
        is_last_stage=True,
        hidden_dim=64,
        vocab_size=200,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        head_dim=16,
        architecture="llama",
        device="cpu",
    )
    # Target lm_head weight shape is (200, 64), target norm shape is (64,)
    incompatible_head = torch.randn(100, 64)
    with pytest.raises(ValueError, match="LM Head weight shape mismatch"):
        model.load_head_weights(lm_head_weight=incompatible_head)

    incompatible_norm = torch.randn(32)
    with pytest.raises(ValueError, match="RMSNorm weight shape mismatch"):
        model.load_head_weights(lm_head_weight=None, norm_weight=incompatible_norm)

    # Rejects non-tensor inputs
    with pytest.raises(ValueError, match="Expected torch.Tensor"):
        model.load_head_weights(lm_head_weight="invalid-head-weight")

    with pytest.raises(ValueError, match="Expected torch.Tensor"):
        model.load_head_weights(lm_head_weight=None, norm_weight="invalid-norm-weight")

    # Valid shapes succeed
    valid_head = torch.randn(200, 64)
    valid_norm = torch.ones(64)
    model.load_head_weights(lm_head_weight=valid_head, norm_weight=valid_norm)
    assert torch.allclose(model.lm_head.weight, valid_head)
    assert torch.allclose(model.norm.weight, valid_norm)


# =====================================================================
# Issue 3: Explicit architecture compatibility validation
# =====================================================================

def test_architecture_compatibility_check():
    """Verify is_architecture_supported accepts compatible types and rejects incompatible ones."""
    # Supported verified architectures (LLaMA, Mistral, Gemma families)
    for arch in ("llama", "LlamaForCausalLM", "llama2", "llama3", "gemma", "gemma2", "gemma3", "Gemma3ForCausalLM", "mistral"):
        assert is_architecture_supported(arch), f"Expected {arch} to be supported"

    # Unsupported architectures (unverified/non-parity: qwen, transformer, gpt2, mpt, moe, etc.)
    for arch in ("qwen", "qwen2", "transformer", "gpt2", "mpt", "qwen2_moe", "mixtral", "falcon", "chatglm", "bloom", "opt", "bert", "t5", "vision_transformer", "albert_transformer"):
        assert not is_architecture_supported(arch), f"Expected {arch} to be rejected"

    # None and empty
    assert not is_architecture_supported(None)
    assert not is_architecture_supported("")


def test_distributed_stage_model_rejects_unsupported_architecture():
    """Ensure DistributedStageModel raises ValueError when initialized with unsupported architecture."""
    # Incompatible architecture names
    with pytest.raises(ValueError, match="Unsupported model architecture"):
        DistributedStageModel(
            layer_start=0,
            layer_end=2,
            total_layers=4,
            is_first_stage=True,
            is_last_stage=False,
            hidden_dim=64,
            vocab_size=200,
            num_heads=4,
            num_kv_heads=4,
            intermediate_size=128,
            head_dim=16,
            architecture="gpt2",
            device="cpu",
        )

    with pytest.raises(ValueError, match="Unsupported model architecture"):
        DistributedStageModel(
            layer_start=0,
            layer_end=2,
            total_layers=4,
            is_first_stage=True,
            is_last_stage=False,
            hidden_dim=64,
            vocab_size=200,
            num_heads=4,
            num_kv_heads=4,
            intermediate_size=128,
            head_dim=16,
            architecture="qwen2_moe",
            device="cpu",
        )

    # architecture=None must also be rejected
    with pytest.raises(ValueError, match="Unsupported model architecture"):
        DistributedStageModel(
            layer_start=0,
            layer_end=2,
            total_layers=4,
            is_first_stage=True,
            is_last_stage=False,
            hidden_dim=64,
            vocab_size=200,
            num_heads=4,
            num_kv_heads=4,
            intermediate_size=128,
            head_dim=16,
            architecture=None,
            device="cpu",
        )

    # Supported architecture works
    model = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=4,
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
    assert model.architecture == "llama"


# =====================================================================
# Issue 4: Scheduler requires verified metadata and avoids synthetic assumptions
# =====================================================================

def test_optimal_heterogeneous_partition_requires_verified_metadata():
    """Ensure optimal_heterogeneous_partition rejects None or incomplete model metadata even with K=1."""
    w1 = WorkerNode(
        worker_id="w-1",
        gpu_name="GPU A",
        total_vram_bytes=8 * 1024**3,
        free_vram_bytes=6 * 1024**3,
        device_type="cuda",
        last_heartbeat=0.0,
    )
    w2 = WorkerNode(
        worker_id="w-2",
        gpu_name="GPU B",
        total_vram_bytes=8 * 1024**3,
        free_vram_bytes=6 * 1024**3,
        device_type="cuda",
        last_heartbeat=0.0,
    )

    # Rejects None (both multi-worker and single worker)
    with pytest.raises(ValueError, match="requires verified model_metadata"):
        optimal_heterogeneous_partition([w1, w2], total_layers=8, model_metadata=None)

    with pytest.raises(ValueError, match="requires verified model_metadata"):
        optimal_heterogeneous_partition([w1], total_layers=8, model_metadata=None)

    # Rejects empty dict
    with pytest.raises(ValueError, match="requires verified model_metadata"):
        optimal_heterogeneous_partition([w1, w2], total_layers=8, model_metadata={})

    # Rejects missing dimensions (e.g. missing intermediate_size)
    incomplete_meta = {
        "hidden_dim": 2048,
        "num_heads": 16,
        "vocab_size": 32000,
    }
    with pytest.raises(ValueError, match="missing or invalid"):
        optimal_heterogeneous_partition([w1, w2], total_layers=8, model_metadata=incomplete_meta)

    # Rejects missing context_length (no assumed fixed sequence length)
    no_ctx_meta = {
        "hidden_dim": 2048,
        "intermediate_size": 8192,
        "num_heads": 16,
        "vocab_size": 32000,
    }
    with pytest.raises(ValueError, match="requires verified 'context_length'"):
        optimal_heterogeneous_partition([w1, w2], total_layers=8, model_metadata=no_ctx_meta)


def test_scheduler_telemetry_does_not_fabricate_synthetic_speeds():
    """Ensure missing worker telemetry is handled explicitly as unmeasured without synthetic speed fabrication."""
    w1 = WorkerNode(
        worker_id="w-cuda-big",
        gpu_name="NVIDIA RTX 4090",
        total_vram_bytes=24 * 1024**3,
        free_vram_bytes=20 * 1024**3,
        device_type="cuda",
        last_heartbeat=0.0,
        compute_speed=None,  # UNKNOWN
        network_latency_ms=None,  # UNKNOWN
    )
    w2 = WorkerNode(
        worker_id="w-cuda-small",
        gpu_name="NVIDIA RTX 3060",
        total_vram_bytes=12 * 1024**3,
        free_vram_bytes=20 * 1024**3,
        device_type="cuda",
        last_heartbeat=0.0,
        compute_speed=None,  # UNKNOWN
        network_latency_ms=None,  # UNKNOWN
    )

    meta = {
        "hidden_dim": 1024,
        "intermediate_size": 4096,
        "num_heads": 8,
        "vocab_size": 32000,
        "context_length": 2048,
        "torch_dtype": "bfloat16",
    }

    # With unknown compute speeds, neither node is fabricated to be faster than the other.
    parts = optimal_heterogeneous_partition([w1, w2], total_layers=8, model_metadata=meta)
    assert w1.compute_speed_status == "unknown"
    assert w2.compute_speed_status == "unknown"
    assert w1.network_latency_status == "unknown"
    assert w2.network_latency_status == "unknown"
    w1_count = parts[0][1] - parts[0][0]
    w2_count = parts[1][1] - parts[1][0]
    assert w1_count == 4
    assert w2_count == 4


def test_scheduler_accounts_for_context_length_and_precision():
    """Ensure context length and precision change layer memory calculations and placement in heterogeneous cluster."""
    # Worker 1 has constrained VRAM (1.2 GB free)
    w1 = WorkerNode(
        worker_id="w-small",
        gpu_name="GPU Constrained",
        total_vram_bytes=4 * 1024**3,
        free_vram_bytes=int(1.2 * 1024**3),
        device_type="cuda",
        last_heartbeat=0.0,
        compute_speed=5.0,
    )
    # Worker 2 has abundant VRAM (20 GB free)
    w2 = WorkerNode(
        worker_id="w-large",
        gpu_name="GPU Abundant",
        total_vram_bytes=24 * 1024**3,
        free_vram_bytes=20 * 1024**3,
        device_type="cuda",
        last_heartbeat=0.0,
        compute_speed=5.0,
    )

    # Standard context (2048 tokens)
    meta_standard = {
        "hidden_dim": 2048,
        "intermediate_size": 8192,
        "num_heads": 16,
        "num_kv_heads": 16,
        "vocab_size": 32000,
        "context_length": 2048,
        "bytes_per_param": 2,
    }
    # Large context (32k tokens) drastically expands KV-cache footprint per layer
    meta_large_ctx = {
        "hidden_dim": 2048,
        "intermediate_size": 8192,
        "num_heads": 16,
        "num_kv_heads": 16,
        "vocab_size": 32000,
        "context_length": 32768,
        "bytes_per_param": 2,
    }

    parts_standard = optimal_heterogeneous_partition([w1, w2], total_layers=12, model_metadata=meta_standard)
    parts_large_ctx = optimal_heterogeneous_partition([w1, w2], total_layers=12, model_metadata=meta_large_ctx)

    w1_std_layers = parts_standard[0][1] - parts_standard[0][0]
    w1_large_layers = parts_large_ctx[0][1] - parts_large_ctx[0][0]

    # Under large context, KV-cache footprint forces fewer layers onto the memory-constrained worker
    assert w1_large_layers < w1_std_layers

    # Test precision effect: fp16 (2 bytes) vs int8 (1 byte) on a very tight worker (0.6 GB free)
    w_tight = WorkerNode(
        worker_id="w-tight",
        gpu_name="GPU Tight",
        total_vram_bytes=4 * 1024**3,
        free_vram_bytes=int(0.6 * 1024**3),
        device_type="cuda",
        last_heartbeat=0.0,
        compute_speed=10.0,
    )
    meta_fp16 = {
        "hidden_dim": 2048,
        "intermediate_size": 8192,
        "num_heads": 16,
        "num_kv_heads": 16,
        "vocab_size": 32000,
        "context_length": 2048,
        "bytes_per_param": 2,
    }
    meta_int8 = {
        "hidden_dim": 2048,
        "intermediate_size": 8192,
        "num_heads": 16,
        "num_kv_heads": 16,
        "vocab_size": 32000,
        "context_length": 2048,
        "bytes_per_param": 1,
    }
    parts_fp16 = optimal_heterogeneous_partition([w_tight, w2], total_layers=12, model_metadata=meta_fp16)
    parts_int8 = optimal_heterogeneous_partition([w_tight, w2], total_layers=12, model_metadata=meta_int8)

    w_tight_fp16 = parts_fp16[0][1] - parts_fp16[0][0]
    w_tight_int8 = parts_int8[0][1] - parts_int8[0][0]
    assert w_tight_int8 > w_tight_fp16


# =====================================================================
# Issue 5: Invalid token IDs are rejected rather than clamped
# =====================================================================

def test_forward_first_stage_rejects_out_of_range_tokens():
    """Ensure forward_first_stage raises ValueError on out-of-range token IDs instead of silent clamping."""
    vocab_size = 500
    model = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=64,
        vocab_size=vocab_size,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        head_dim=16,
        architecture="llama",
        device="cpu",
    )

    # 1. Negative token ID
    neg_tokens = torch.tensor([[10, 20, -1, 30]], dtype=torch.long)
    with pytest.raises(ValueError, match="Invalid token ID detected in input_ids"):
        model.forward_first_stage(neg_tokens)

    # 2. Token ID equal to or greater than vocab_size
    exceed_tokens = torch.tensor([[10, 20, vocab_size, 30]], dtype=torch.long)
    with pytest.raises(ValueError, match="Invalid token ID detected in input_ids"):
        model.forward_first_stage(exceed_tokens)

    far_exceed_tokens = torch.tensor([[10, 20, vocab_size + 9999]], dtype=torch.long)
    with pytest.raises(ValueError, match="Invalid token ID detected in input_ids"):
        model.forward_first_stage(far_exceed_tokens)

    # 3. Invalid types (e.g. string or dict)
    with pytest.raises(ValueError, match="input_ids must be a torch.Tensor"):
        model.forward_first_stage("invalid-tokens")

    # 4. Valid token IDs in range [0, vocab_size - 1] succeed without error
    valid_tokens = torch.tensor([[0, 100, vocab_size - 1]], dtype=torch.long)
    h, kvs = model.forward_first_stage(valid_tokens)
    assert h.shape == (1, 3, 64)
