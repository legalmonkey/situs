"""Comprehensive regression tests for architecture generalization, zero-default dimensions,
grounded scheduler telemetry & memory feasibility, fast-boot gating, and numerical parity verification.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from situs.worker.loader import (
    RealTransformerBlock,
    DistributedStageModel,
    FastBootDraftBlock,
    is_architecture_supported,
    verify_layer_numerical_parity,
    RotaryEmbedding,
    apply_rotary_pos_emb,
)
from situs.coordinator.registry import (
    WorkerNode,
    optimal_heterogeneous_partition,
)
from situs.protocol.messages import (
    AssignLayers,
    InferenceStart,
    ForwardActivation,
    NextToken,
    SpeculativeVerified,
    MessageType,
)
from situs.worker.engine import StageExecutionEngine


# =====================================================================
# 1. Architecture Validation & Zero-Default Dimensions
# =====================================================================

def test_real_transformer_block_requires_explicit_dimensions_and_architecture():
    """Verify that RealTransformerBlock raises TypeError if any parameter is omitted."""
    # Omission of required arguments raises TypeError
    with pytest.raises(TypeError):
        RealTransformerBlock()

    with pytest.raises(TypeError):
        RealTransformerBlock(hidden_dim=128, num_heads=4)

    # Valid instantiation requires all parameters
    block = RealTransformerBlock(
        hidden_dim=128,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="llama",
    )
    assert block.hidden_dim == 128
    assert block.num_heads == 4
    assert block.num_kv_heads == 2
    assert block.head_dim == 32
    assert block.architecture == "llama"


def test_real_transformer_block_rejects_invalid_or_unsupported_parameters():
    """Verify that RealTransformerBlock rejects negative/zero dimensions and unsupported architectures."""
    # Non-positive dimensions
    with pytest.raises(ValueError, match="verified positive integer"):
        RealTransformerBlock(
            hidden_dim=0,
            num_heads=4,
            num_kv_heads=2,
            intermediate_size=256,
            head_dim=32,
            architecture="llama",
        )

    with pytest.raises(ValueError, match="verified positive integer"):
        RealTransformerBlock(
            hidden_dim=128,
            num_heads=-1,
            num_kv_heads=2,
            intermediate_size=256,
            head_dim=32,
            architecture="llama",
        )

    # Unsupported architectures
    for bad_arch in ("gpt2", "mpt", "qwen2_moe", "mixtral", "falcon", "bert"):
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            RealTransformerBlock(
                hidden_dim=128,
                num_heads=4,
                num_kv_heads=2,
                intermediate_size=256,
                head_dim=32,
                architecture=bad_arch,
            )


def test_distributed_stage_model_requires_explicit_parameters():
    """Verify that DistributedStageModel rejects omitted dimensions or unsupported architectures."""
    with pytest.raises(TypeError):
        DistributedStageModel(
            layer_start=0,
            layer_end=2,
            total_layers=4,
            is_first_stage=True,
            is_last_stage=False,
        )

    # Valid instantiation succeeds
    stage = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=128,
        vocab_size=1000,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="gemma3",
    )
    assert stage.architecture == "gemma3"
    assert stage.hidden_dim == 128
    assert stage.head_dim == 32


# =====================================================================
# 2. Reference Layer Numerical Parity Verification
# =====================================================================

class CanonicalReferenceDecoderLayer(nn.Module):
    """
    Independent reference implementation of a Transformer Decoder Block
    (RMSNorm + RoPE Attention with KV Cache + SwiGLU MLP).
    """
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int, intermediate_size: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps

        self.input_layernorm = nn.Parameter(torch.ones(hidden_dim))
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        self.post_attention_layernorm = nn.Parameter(torch.ones(hidden_dim))
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_dim, bias=False)
        self.rotary = RotaryEmbedding(head_dim)

    def _rmsnorm(self, x, weight):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * weight

    def forward(self, hidden_states, past_kv=None):
        residual = hidden_states
        norm_x = self._rmsnorm(hidden_states, self.input_layernorm)
        b, s, _ = norm_x.shape

        q = self.q_proj(norm_x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is not None and isinstance(past_kv, (tuple, list)) and len(past_kv) == 2:
            past_k, past_v = past_kv
            if isinstance(past_k, torch.Tensor):
                past_len = past_k.shape[-2]

        cos, sin = self.rotary(norm_x, seq_len=s, offset=past_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None and isinstance(past_kv, (tuple, list)) and len(past_kv) == 2:
            past_k, past_v = past_kv
            if isinstance(past_k, torch.Tensor) and isinstance(past_v, torch.Tensor):
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)

        new_kv = (k, v)

        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k_exp = k.repeat_interleave(repeats, dim=1)
            v_exp = v.repeat_interleave(repeats, dim=1)
        else:
            k_exp = k
            v_exp = v

        if s > 1:
            if past_len > 0:
                q_idx = torch.arange(past_len, past_len + s, device=q.device).unsqueeze(1)
                k_idx = torch.arange(past_len + s, device=q.device).unsqueeze(0)
                attn_mask = k_idx <= q_idx
                attn = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask)
            else:
                attn = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        else:
            attn = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

        attn = attn.transpose(1, 2).contiguous().view(b, s, -1)
        attn_out = self.o_proj(attn)
        h = residual + attn_out

        residual = h
        norm_mlp = self._rmsnorm(h, self.post_attention_layernorm)
        gate = F.silu(self.gate_proj(norm_mlp))
        up = self.up_proj(norm_mlp)
        mlp_out = self.down_proj(gate * up)
        return residual + mlp_out, new_kv


class GemmaReferenceDecoderLayer(nn.Module):
    """
    Independent reference implementation of a Gemma Transformer Decoder Block
    (GemmaRMSNorm with unit-offset + RoPE Attention with KV Cache + GeGLU MLP).
    """
    def __init__(self, hidden_dim: int, num_heads: int, num_kv_heads: int, intermediate_size: int, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.eps = eps

        self.input_layernorm = nn.Parameter(torch.zeros(hidden_dim))
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim, bias=False)

        self.post_attention_layernorm = nn.Parameter(torch.zeros(hidden_dim))
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_dim, bias=False)
        self.rotary = RotaryEmbedding(head_dim)
        self.act_fn = nn.GELU(approximate="tanh")

    def _rmsnorm(self, x, weight):
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * (1.0 + weight)

    def forward(self, hidden_states, past_kv=None):
        residual = hidden_states
        norm_x = self._rmsnorm(hidden_states, self.input_layernorm)
        b, s, _ = norm_x.shape

        q = self.q_proj(norm_x).view(b, s, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(b, s, self.num_kv_heads, self.head_dim).transpose(1, 2)

        past_len = 0
        if past_kv is not None and isinstance(past_kv, (tuple, list)) and len(past_kv) == 2:
            past_k, past_v = past_kv
            if isinstance(past_k, torch.Tensor):
                past_len = past_k.shape[-2]

        cos, sin = self.rotary(norm_x, seq_len=s, offset=past_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if past_kv is not None and isinstance(past_kv, (tuple, list)) and len(past_kv) == 2:
            past_k, past_v = past_kv
            if isinstance(past_k, torch.Tensor) and isinstance(past_v, torch.Tensor):
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)

        new_kv = (k, v)

        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k_exp = k.repeat_interleave(repeats, dim=1)
            v_exp = v.repeat_interleave(repeats, dim=1)
        else:
            k_exp = k
            v_exp = v

        if s > 1:
            if past_len > 0:
                q_idx = torch.arange(past_len, past_len + s, device=q.device).unsqueeze(1)
                k_idx = torch.arange(past_len + s, device=q.device).unsqueeze(0)
                attn_mask = k_idx <= q_idx
                attn = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=attn_mask)
            else:
                attn = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True)
        else:
            attn = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False)

        attn = attn.transpose(1, 2).contiguous().view(b, s, -1)
        attn_out = self.o_proj(attn)
        h = residual + attn_out

        residual = h
        norm_mlp = self._rmsnorm(h, self.post_attention_layernorm)
        gate = self.act_fn(self.gate_proj(norm_mlp))
        up = self.up_proj(norm_mlp)
        mlp_out = self.down_proj(gate * up)
        return residual + mlp_out, new_kv


def test_verify_layer_numerical_parity_matching_weights():
    """Verify that RealTransformerBlock achieves numerical parity with reference decoder layer on identical weights."""
    hidden_dim = 128
    num_heads = 4
    num_kv_heads = 2
    intermediate_size = 256
    head_dim = 32

    # Instantiate both blocks
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

    # Parity check with test input
    test_input = torch.randn(1, 4, hidden_dim)
    parity = verify_layer_numerical_parity(real_block, ref_block, tolerance=1e-3, test_input=test_input)
    assert parity is True, "RealTransformerBlock must achieve mathematical parity with reference transformer block"


def test_verify_layer_numerical_parity_detects_mismatch():
    """Verify that verify_layer_numerical_parity detects weight corruption and returns False."""
    hidden_dim = 128
    real_block = RealTransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="llama",
    )
    ref_block = CanonicalReferenceDecoderLayer(
        hidden_dim=hidden_dim,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
    )

    # Random dissimilar weights
    parity = verify_layer_numerical_parity(real_block, ref_block, tolerance=1e-3)
    assert parity is False, "Parity verification must fail when weights differ"


def test_gemma_layer_numerical_parity_matching_weights():
    """Verify that RealTransformerBlock achieves numerical parity with GemmaReferenceDecoderLayer."""
    hidden_dim = 128
    num_heads = 4
    num_kv_heads = 2
    intermediate_size = 256
    head_dim = 32

    real_block = RealTransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
        architecture="gemma3",
    )
    ref_block = GemmaReferenceDecoderLayer(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        head_dim=head_dim,
    )

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

    test_input = torch.randn(1, 4, hidden_dim)
    parity_prefill = verify_layer_numerical_parity(real_block, ref_block, tolerance=1e-3, test_input=test_input)
    assert parity_prefill is True, "Gemma block must achieve numerical parity with Gemma reference layer on prefill"

    # Cached single-token step
    with torch.no_grad():
        _, kv_real = real_block(test_input)
        _, kv_ref = ref_block(test_input)
    step_input = torch.randn(1, 1, hidden_dim)
    parity_step = verify_layer_numerical_parity(real_block, ref_block, tolerance=1e-3, test_input=step_input, past_kv=kv_real)
    assert parity_step is True, "Gemma block must achieve numerical parity on cached autoregressive step"


def test_unverified_architectures_rejected():
    """Verify that unverified architectures (qwen, transformer placeholder, gpt2, etc.) are strictly rejected."""
    for bad_arch in ("qwen", "qwen2", "qwen2.5", "transformer", "gpt2", "mpt", "bert"):
        assert not is_architecture_supported(bad_arch)
        with pytest.raises(ValueError, match="Unsupported model architecture"):
            RealTransformerBlock(
                hidden_dim=128,
                num_heads=4,
                num_kv_heads=2,
                intermediate_size=256,
                head_dim=32,
                architecture=bad_arch,
            )


# =====================================================================
# 3. Grounded Scheduler Telemetry & Memory Feasibility
# =====================================================================

def test_worker_node_explicit_telemetry_status():
    """Verify WorkerNode explicitly distinguishes measured vs unknown telemetry."""
    # Unknown telemetry
    node_unknown = WorkerNode(
        worker_id="w-unknown",
        gpu_name="GPU A",
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=14 * 1024**3,
        device_type="cuda",
        last_heartbeat=0.0,
        compute_speed=None,
        network_latency_ms=None,
    )
    assert node_unknown.compute_speed_status == "unknown"
    assert node_unknown.network_latency_status == "unknown"

    # Measured telemetry
    node_measured = WorkerNode(
        worker_id="w-measured",
        gpu_name="GPU B",
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=14 * 1024**3,
        device_type="cuda",
        last_heartbeat=0.0,
        compute_speed=12.5,
        network_latency_ms=4.2,
    )
    assert node_measured.compute_speed_status == "measured"
    assert node_measured.network_latency_status == "measured"


def test_scheduler_strictly_requires_precision():
    """Verify optimal_heterogeneous_partition rejects missing precision without defaulting to 2 bytes."""
    w1 = WorkerNode(
        worker_id="w-1",
        gpu_name="GPU 1",
        total_vram_bytes=16 * 1024**3,
        free_vram_bytes=14 * 1024**3,
        device_type="cuda",
        last_heartbeat=0.0,
    )
    meta_without_precision = {
        "hidden_dim": 1024,
        "intermediate_size": 4096,
        "num_heads": 8,
        "vocab_size": 32000,
        "context_length": 2048,
    }
    with pytest.raises(ValueError, match="requires verified precision"):
        optimal_heterogeneous_partition([w1], total_layers=8, model_metadata=meta_without_precision)

    # Supplying precision in runtime_config satisfies requirement
    parts = optimal_heterogeneous_partition(
        [w1],
        total_layers=8,
        model_metadata=meta_without_precision,
        runtime_config={"bytes_per_param": 2.0},
    )
    assert len(parts) == 1
    assert parts[0] == (0, 8)


def test_scheduler_rejects_insufficient_cluster_memory():
    """Verify optimal_heterogeneous_partition rejects placement when cluster memory cannot accommodate model."""
    # Worker with only 100 MB free VRAM
    w_tiny = WorkerNode(
        worker_id="w-tiny",
        gpu_name="Tiny GPU",
        total_vram_bytes=128 * 1024**2,
        free_vram_bytes=100 * 1024**2,
        device_type="cuda",
        last_heartbeat=0.0,
    )
    meta_large = {
        "hidden_dim": 4096,
        "intermediate_size": 14336,
        "num_heads": 32,
        "num_kv_heads": 8,
        "vocab_size": 128256,
        "context_length": 4096,
        "torch_dtype": "bfloat16",
    }
    # Model requires several gigabytes, far exceeding 100 MB
    with pytest.raises(ValueError, match="Insufficient cluster memory"):
        optimal_heterogeneous_partition([w_tiny], total_layers=32, model_metadata=meta_large)


# =====================================================================
# 4. Fast-Boot Gating & Quality Isolation
# =====================================================================

def test_fast_boot_draft_block_metadata():
    """Verify FastBootDraftBlock is explicitly flagged as unverified experimental draft block."""
    draft = FastBootDraftBlock(hidden_dim=128)
    assert draft.is_verified_inference is False
    assert draft.is_experimental is True
    assert FastBootDraftBlock.is_verified_inference is False
    assert FastBootDraftBlock.is_experimental is True


def test_fast_boot_skeleton_propagates_unverified_flags():
    """Verify that StageExecutionEngine propagates is_verified=False and is_experimental=True during fast-boot."""
    total_layers = 4
    hidden_dim = 128
    vocab_size = 500

    # Model in fast-boot mode
    model_fast = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="llama",
        fast_boot=True,
    )
    assign_fast = AssignLayers(
        worker_id="w-fast",
        model_name="fast-model",
        layer_start=0,
        layer_end=2,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        next_worker_id="w-last",
        fast_boot=True,
    )
    engine_fast = StageExecutionEngine(model_fast, assign_fast)

    start_frame = engine_fast.start_inference(
        InferenceStart(
            request_id="req-fast-draft",
            model_name="fast-model",
            input_ids=[10, 20],
        )
    )
    assert start_frame.type == MessageType.FORWARD_ACTIVATION
    act = ForwardActivation.model_validate(start_frame.payload)
    assert act.is_verified is False, "ForwardActivation on fast-boot skeleton must be flagged is_verified=False"
    assert act.is_experimental is True, "ForwardActivation on fast-boot skeleton must be flagged is_experimental=True"

    # Last stage in fast-boot mode emits unverified NextToken
    model_last_fast = DistributedStageModel(
        layer_start=2,
        layer_end=4,
        total_layers=total_layers,
        is_first_stage=False,
        is_last_stage=True,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="llama",
        fast_boot=True,
    )
    assign_last_fast = AssignLayers(
        worker_id="w-last",
        model_name="fast-model",
        layer_start=2,
        layer_end=4,
        total_layers=total_layers,
        is_first_stage=False,
        is_last_stage=True,
        first_worker_id="w-fast",
        fast_boot=True,
    )
    engine_last_fast = StageExecutionEngine(model_last_fast, assign_last_fast)
    token_frame = engine_last_fast.process_forward_activation(act)
    assert token_frame is not None
    assert token_frame.type == MessageType.NEXT_TOKEN
    tok = NextToken.model_validate(token_frame.payload)
    assert tok.is_verified is False, "NextToken emitted from fast-boot skeleton must be flagged is_verified=False"
    assert tok.is_experimental is True, "NextToken emitted from fast-boot skeleton must be flagged is_experimental=True"
