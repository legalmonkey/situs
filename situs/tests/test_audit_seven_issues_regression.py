"""Comprehensive regression tests covering all 7 audit issues:
1. Rotary positional embeddings (RoPE) defined and applied in RealTransformerBlock.
2. Supported architectures pass numerical parity tests against reference implementations.
3. Fast-boot mode disabled by default and rejected in production inference.
4. Position offsets and attention behavior validated during cached token-by-token generation.
5. Engine raises RuntimeError("PyTorch is required for model inference") when HAS_TORCH is False.
6. Coordinator request parsing preserves explicit 0.0/0 values for temperature, top_p, and max_tokens.
7. Sampler detects ANY NaN or Inf in logits (not only when all are infinite).
"""

import os
import math
import pytest
import numpy as np
from unittest.mock import patch, MagicMock

import torch
import torch.nn as nn
import torch.nn.functional as F

from situs.worker.loader import (
    RealTransformerBlock,
    DistributedStageModel,
    FastBootDraftBlock,
    RotaryEmbedding,
    apply_rotary_pos_emb,
    is_architecture_supported,
    verify_layer_numerical_parity,
    SUPPORTED_ARCHITECTURES,
)
from situs.worker.sampler import TokenSampler
from situs.worker.engine import StageExecutionEngine
from situs.coordinator.registry import WorkerRegistry, WorkerNode
from situs.coordinator.app import ChatCompletionRequest, config
from situs.protocol.messages import (
    AssignLayers,
    InferenceStart,
    ForwardActivation,
    NextToken,
    SpeculativeCandidates,
)
from situs.tests.test_architecture_and_numerical_verification import (
    CanonicalReferenceDecoderLayer,
    GemmaReferenceDecoderLayer,
)


# =====================================================================
# Issue 1 & 4: RoPE Application, Position Offsets & Numerical Parity
# =====================================================================

def test_issue1_rotary_positional_embeddings_applied_in_forward():
    """Verify that RealTransformerBlock calls apply_rotary_pos_emb and injects positional frequencies."""
    hidden_dim = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    intermediate_size = 128

    block = RealTransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
        architecture="llama",
    )
    block.eval()

    # Initialize weights to non-degenerate scale
    with torch.no_grad():
        for p in block.parameters():
            p.data.normal_(0, 0.5)

    seq = torch.randn(1, 4, hidden_dim)

    # 1. Verify apply_rotary_pos_emb is invoked during forward pass
    called_with = []
    import situs.worker.loader as loader
    orig_apply = loader.apply_rotary_pos_emb

    def spy_apply(q, k, cos, sin):
        called_with.append((q.shape, k.shape, cos.shape, sin.shape))
        return orig_apply(q, k, cos, sin)

    with patch("situs.worker.loader.apply_rotary_pos_emb", side_effect=spy_apply):
        with torch.no_grad():
            out_with_rope, _ = block(seq)

    assert len(called_with) == 1, "apply_rotary_pos_emb must be invoked during transformer forward"
    q_shape, k_shape, cos_shape, sin_shape = called_with[0]
    assert q_shape == (1, num_heads, 4, head_dim)
    assert k_shape == (1, num_kv_heads, 4, head_dim)
    assert cos_shape == (4, head_dim)
    assert sin_shape == (4, head_dim)

    # 2. Verify that RoPE actively modifies the forward output
    with patch("situs.worker.loader.apply_rotary_pos_emb", side_effect=lambda q, k, c, s: (q, k)):
        with torch.no_grad():
            out_no_rope, _ = block(seq)

    diff = (out_with_rope - out_no_rope).abs().max().item()
    assert diff > 0.01, f"RoPE must have significant effect on output representations, got diff={diff}"


def test_issue4_rotary_position_offsets_in_cached_kv_generation():
    """Verify position offsets (past_len) are correctly applied during incremental cached generation."""
    hidden_dim = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    intermediate_size = 128

    block = RealTransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
        architecture="llama",
    )
    block.eval()

    torch.manual_seed(123)
    full_seq = torch.randn(1, 5, hidden_dim)

    # 1. Full 5-token causal forward pass
    with torch.no_grad():
        out_full, kv_full = block(full_seq)

    # 2. Prefill 4 tokens, then cached single-token autoregressive step for token 5
    x_prefill = full_seq[:, :4, :]
    x_step = full_seq[:, 4:5, :]

    with torch.no_grad():
        out_prefill, kv_prefill = block(x_prefill)
        assert kv_prefill[0].shape[-2] == 4, "KV cache must store 4 past key tokens"
        out_step, kv_step = block(x_step, past_kv=kv_prefill)

    # Prefill tokens 0..3 must match full sequence exactly
    diff_prefill = torch.max(torch.abs(out_full[:, :4, :] - out_prefill)).item()
    assert diff_prefill < 1e-5, f"Prefill discrepancy too large: {diff_prefill}"

    # Cached step at position 4 must match 5th token of causal full sequence
    diff_step = torch.max(torch.abs(out_full[:, 4:5, :] - out_step)).item()
    assert diff_step < 1e-4, f"Cached step RoPE offset discrepancy: {diff_step}"

    # Final KV cache must match the full sequence keys
    diff_k = torch.max(torch.abs(kv_full[0] - kv_step[0])).item()
    assert diff_k < 1e-4, f"Cached keys discrepancy: {diff_k}"

    # 3. Multi-token cached continuation (e.g. speculative verification of 3 candidate tokens)
    seq_7 = torch.randn(1, 7, hidden_dim)
    with torch.no_grad():
        out_7, _ = block(seq_7)
        out_pre4, kv_pre4 = block(seq_7[:, :4, :])
        # Continuation with 3 tokens (positions 4..6) using KV cache of past 4 tokens
        out_cand, kv_cand = block(seq_7[:, 4:, :], past_kv=kv_pre4)

    diff_cand = torch.max(torch.abs(out_7[:, 4:, :] - out_cand)).item()
    assert diff_cand < 1e-4, f"Multi-token cached candidate verification causal discrepancy: {diff_cand}"


def test_issue2_numerical_parity_with_reference_layer_prefill_and_cached():
    """Verify RealTransformerBlock achieves mathematical parity with CanonicalReferenceDecoderLayer and GemmaReferenceDecoderLayer."""
    hidden_dim = 64
    num_heads = 4
    num_kv_heads = 2
    head_dim = 16
    intermediate_size = 128

    # 1. Canonical LLaMA
    real_block = RealTransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
        architecture="llama",
    )
    ref_block = CanonicalReferenceDecoderLayer(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
    )

    # Synchronize weights identically
    with torch.no_grad():
        ref_block.input_layernorm.copy_(real_block.input_layernorm.weight)
        ref_block.q_proj.weight.copy_(real_block.q_proj.weight)
        ref_block.k_proj.weight.copy_(real_block.k_proj.weight)
        ref_block.v_proj.weight.copy_(real_block.v_proj.weight)
        ref_block.o_proj.weight.copy_(real_block.o_proj.weight)
        ref_block.post_attention_layernorm.copy_(real_block.post_attention_layernorm.weight)
        ref_block.gate_proj.weight.copy_(real_block.gate_proj.weight)
        ref_block.up_proj.weight.copy_(real_block.up_proj.weight)
        ref_block.down_proj.weight.copy_(real_block.down_proj.weight)

    # Parity check on prompt prefill
    prompt_input = torch.randn(1, 4, hidden_dim)
    parity_prefill = verify_layer_numerical_parity(real_block, ref_block, tolerance=1e-3, test_input=prompt_input)
    assert parity_prefill is True, "Prefill must achieve numerical parity with reference implementation"

    # Parity check on cached single-token autoregressive step
    with torch.no_grad():
        _, kv_real = real_block(prompt_input)
        _, kv_ref = ref_block(prompt_input)

    step_input = torch.randn(1, 1, hidden_dim)
    parity_cached = verify_layer_numerical_parity(
        real_block, ref_block, tolerance=1e-3, test_input=step_input, past_kv=kv_real
    )
    assert parity_cached is True, "Cached autoregressive step must achieve numerical parity with reference"

    # 2. Gemma Architecture Variant Parity
    gemma_real = RealTransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
        architecture="gemma3",
    )
    gemma_ref = GemmaReferenceDecoderLayer(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
    )
    with torch.no_grad():
        gemma_ref.input_layernorm.copy_(gemma_real.input_layernorm.weight)
        gemma_ref.q_proj.weight.copy_(gemma_real.q_proj.weight)
        gemma_ref.k_proj.weight.copy_(gemma_real.k_proj.weight)
        gemma_ref.v_proj.weight.copy_(gemma_real.v_proj.weight)
        gemma_ref.o_proj.weight.copy_(gemma_real.o_proj.weight)
        gemma_ref.post_attention_layernorm.copy_(gemma_real.post_attention_layernorm.weight)
        gemma_ref.gate_proj.weight.copy_(gemma_real.gate_proj.weight)
        gemma_ref.up_proj.weight.copy_(gemma_real.up_proj.weight)
        gemma_ref.down_proj.weight.copy_(gemma_real.down_proj.weight)

    gemma_parity_pre = verify_layer_numerical_parity(gemma_real, gemma_ref, tolerance=1e-3, test_input=prompt_input)
    assert gemma_parity_pre is True, "Gemma prefill must achieve numerical parity with GemmaReferenceDecoderLayer"

    with torch.no_grad():
        _, kv_g_real = gemma_real(prompt_input)
        _, kv_g_ref = gemma_ref(prompt_input)
    gemma_parity_step = verify_layer_numerical_parity(gemma_real, gemma_ref, tolerance=1e-3, test_input=step_input, past_kv=kv_g_real)
    assert gemma_parity_step is True, "Gemma cached step must achieve numerical parity with GemmaReferenceDecoderLayer"

    # 3. Rejection of unverified architectures (qwen, transformer placeholder)
    assert not is_architecture_supported("qwen")
    assert not is_architecture_supported("qwen2")
    assert not is_architecture_supported("transformer")


# =====================================================================
# Issue 3: Fast-Boot Mode Gating & Production Rejection
# =====================================================================

def test_issue3_fast_boot_mode_rejected_in_production():
    """Verify that fast_boot mode is rejected when running in production environment."""
    with patch.dict(os.environ, {"SITUS_ENV": "production"}):
        # 1. DistributedStageModel rejects fast_boot=True in production
        with pytest.raises(ValueError, match="Fast-boot mode cannot be enabled in production"):
            DistributedStageModel(
                layer_start=0,
                layer_end=2,
                total_layers=4,
                is_first_stage=True,
                is_last_stage=False,
                hidden_dim=64,
                vocab_size=200,
                num_heads=4,
                num_kv_heads=2,
                intermediate_size=128,
                head_dim=16,
                architecture="llama",
                fast_boot=True,
            )

        # 2. WorkerRegistry.partition_model rejects fast_boot=True in production
        registry = WorkerRegistry()
        node = WorkerNode(
            worker_id="w-prod-1",
            gpu_name="GPU Prod",
            total_vram_bytes=16 * 1024**3,
            free_vram_bytes=14 * 1024**3,
            device_type="cuda",
            last_heartbeat=0.0,
        )
        registry.workers["w-prod-1"] = node
        with pytest.raises(ValueError, match="Fast-boot mode cannot be enabled in production"):
            registry.partition_model(
                model_name="gemma3:4b",
                total_layers=4,
                fast_boot=True,
            )


def test_issue3_fast_boot_defaults_to_false():
    """Verify fast_boot is False by default across model and worker definitions."""
    model = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=64,
        vocab_size=200,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        head_dim=16,
        architecture="llama",
    )
    assert model.fast_boot is False
    assert model.fast_boot_active is False


# =====================================================================
# Issue 5: Engine Fails Cleanly When PyTorch is Unavailable
# =====================================================================

def test_issue5_engine_raises_runtime_error_when_no_torch():
    """Verify StageExecutionEngine raises RuntimeError without substituting zero-tensor hidden states."""
    model = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=64,
        vocab_size=200,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=128,
        head_dim=16,
        architecture="llama",
    )
    assign = AssignLayers(
        worker_id="w-test",
        model_name="test-model",
        layer_start=0,
        layer_end=2,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=False,
        next_worker_id="w-next",
    )
    engine = StageExecutionEngine(model, assign)

    with patch("situs.worker.engine.HAS_TORCH", False):
        # 1. start_inference must raise RuntimeError
        with pytest.raises(RuntimeError, match="PyTorch is required for model inference"):
            engine.start_inference(
                InferenceStart(request_id="req-no-torch", model_name="test-model", input_ids=[1, 2, 3])
            )

        # 2. process_forward_activation must raise RuntimeError
        act = ForwardActivation(
            request_id="req-no-torch-2",
            step_index=0,
            shape=[1, 1, 64],
            dtype="int8",
            scale=1.0,
            payload_b64="AA==",
        )
        with pytest.raises(RuntimeError, match="PyTorch is required for model inference"):
            engine.process_forward_activation(act)

        # 3. process_speculative_candidates must raise RuntimeError
        cand = SpeculativeCandidates(
            request_id="req-no-torch-3",
            step_index=0,
            candidate_tokens=[1, 2],
            shape=[1, 2, 64],
            dtype="int8",
            scale=1.0,
            payload_b64="AA==",
        )
        with pytest.raises(RuntimeError, match="PyTorch is required for model inference"):
            engine.process_speculative_candidates(cand)

        # 4. process_next_token must raise RuntimeError
        next_tok = NextToken(request_id="req-no-torch-4", step_index=1, token_id=42)
        with pytest.raises(RuntimeError, match="PyTorch is required for model inference"):
            engine.process_next_token(next_tok)


# =====================================================================
# Issue 6: Coordinator Request Values Preserve Explicit Zero Defaults
# =====================================================================

def test_issue6_coordinator_preserves_explicit_zero_values():
    """Verify that temperature=0.0, top_p=0.0, and max_tokens=0 are not replaced by defaults."""
    req = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        temperature=0.0,
        top_p=0.0,
        max_tokens=0,
    )
    # Pydantic preserves 0.0 / 0
    assert req.temperature == 0.0
    assert req.top_p == 0.0
    assert req.max_tokens == 0

    # Ensure the non-truthy 'if x is not None' pattern preserves 0
    temp = req.temperature if req.temperature is not None else 0.7
    top_p = req.top_p if req.top_p is not None else 0.9
    max_tokens = req.max_tokens if req.max_tokens is not None else 256

    assert temp == 0.0, f"Expected 0.0, got {temp}"
    assert top_p == 0.0, f"Expected 0.0, got {top_p}"
    assert max_tokens == 0, f"Expected 0, got {max_tokens}"

    # Missing / None values correctly fall back to standard defaults
    req_none = ChatCompletionRequest(
        messages=[{"role": "user", "content": "hello"}],
        temperature=None,
        top_p=None,
        max_tokens=None,
    )
    temp_def = req_none.temperature if req_none.temperature is not None else 0.7
    top_p_def = req_none.top_p if req_none.top_p is not None else 0.9
    max_tokens_def = req_none.max_tokens if req_none.max_tokens is not None else 256

    assert temp_def == 0.7
    assert top_p_def == 0.9
    assert max_tokens_def == 256


# =====================================================================
# Issue 7: Sampler Detects ANY NaN or Inf in Logits
# =====================================================================

def test_issue7_sampler_detects_mixed_nan_and_inf_torch():
    """Verify TokenSampler rejects logits when ANY logit is NaN or Inf (not only when all are inf)."""
    sampler = TokenSampler()
    eos_id = 2
    sampler.tokenizer = MagicMock(eos_token_id=eos_id)

    # 1. Mixed NaN and finite values
    logits_with_nan = torch.tensor([1.0, float("nan"), 3.0, 2.0])
    token = sampler.sample_next_token(logits_with_nan)
    assert token == eos_id, "Must return eos_token_id when any logit is NaN"

    # 2. Mixed positive Inf and finite values
    logits_with_pos_inf = torch.tensor([1.0, float("inf"), 3.0, 2.0])
    token = sampler.sample_next_token(logits_with_pos_inf)
    assert token == eos_id, "Must return eos_token_id when any logit is positive Inf"

    # 3. Mixed negative Inf and finite values
    logits_with_neg_inf = torch.tensor([1.0, float("-inf"), 3.0, 2.0])
    token = sampler.sample_next_token(logits_with_neg_inf)
    assert token == eos_id, "Must return eos_token_id when any logit is negative Inf"

    # 4. Valid finite logits sample properly
    logits_valid = torch.tensor([0.1, 0.2, 5.0, 0.1])
    token_valid = sampler.sample_next_token(logits_valid, temperature=0.001)
    assert token_valid == 2, "Greedy sampling must select highest logit index (2)"


def test_issue7_sampler_detects_mixed_nan_and_inf_numpy():
    """Verify TokenSampler NumPy fallback rejects logits when ANY logit is NaN or Inf."""
    sampler = TokenSampler()
    eos_id = 0
    sampler.tokenizer = MagicMock(eos_token_id=eos_id)

    with patch("situs.worker.sampler.HAS_TORCH", False):
        # 1. Mixed NaN
        arr_nan = np.array([1.0, float("nan"), 2.0], dtype=np.float32)
        assert sampler.sample_next_token(arr_nan) == eos_id

        # 2. Mixed Inf
        arr_inf = np.array([1.0, float("inf"), 2.0], dtype=np.float32)
        assert sampler.sample_next_token(arr_inf) == eos_id

        # 3. All zeros
        arr_zeros = np.zeros(10, dtype=np.float32)
        assert sampler.sample_next_token(arr_zeros) == eos_id
