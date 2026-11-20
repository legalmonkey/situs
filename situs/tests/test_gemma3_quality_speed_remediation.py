"""Comprehensive unit and integration tests verifying all Gemma 3 quality, speed, and timeout fixes."""

import pytest
import time
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch

from situs.config import LayerAllocation, WorkerConfig
from situs.protocol.messages import (
    MessageType,
    AssignLayers,
    InferenceStart,
    NextToken,
    SpeculativeCandidates,
    SpeculativeVerified,
)
from situs.worker.loader import (
    DistributedStageModel,
    RealTransformerBlock,
    RotaryEmbedding,
    Int4WeightLinear,
    Int8WeightLinear,
    _resolve_layer_sliding_and_rope,
    _get_arch_spec,
)
from situs.worker.engine import StageExecutionEngine
from situs.worker.speculative import PromptLookupDrafter


# =========================================================================
# Fix 1: Gemma 3 RoPE linear scaling factor (8.0 for full-attention layers)
# =========================================================================

def test_gemma3_rope_linear_scaling_factor():
    """Verify that full-attention layers in Gemma 3 use scaling_factor=8.0 and divide inv_freq by 8.0."""
    spec = _get_arch_spec("gemma3")
    dim = 256
    base = 1000000.0

    # Layer 0..4 are sliding layers; Layer 5 ((5 + 1) % 6 == 0) is a full-attention layer
    theta_sliding, sw_sliding, scale_sliding = _resolve_layer_sliding_and_rope(spec, 0)
    assert sw_sliding == 1024
    assert scale_sliding == 1.0

    theta_full, sw_full, scale_full = _resolve_layer_sliding_and_rope(spec, 5)
    assert sw_full is None
    assert scale_full == 8.0

    # Verify RotaryEmbedding applies the 8.0 scaling factor to inv_freq
    rope_unscaled = RotaryEmbedding(dim=dim, base=base, scaling_factor=1.0)
    rope_scaled = RotaryEmbedding(dim=dim, base=base, scaling_factor=8.0)

    expected_scaled_inv_freq = rope_unscaled.inv_freq / 8.0
    assert torch.allclose(rope_scaled.inv_freq, expected_scaled_inv_freq)

    # Verify RealTransformerBlock for layer 5 instantiates RotaryEmbedding with 8.0 scaling
    block_full = RealTransformerBlock(
        hidden_dim=256,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        head_dim=64,
        architecture="gemma3",
        layer_idx=5,
        device="cpu",
    )
    assert block_full.rope_scaling_factor == 8.0
    assert block_full.sliding_window is None
    assert block_full.rotary.scaling_factor == 8.0

    # Layer 0 must use scaling_factor 1.0
    block_sliding = RealTransformerBlock(
        hidden_dim=256,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        head_dim=64,
        architecture="gemma3",
        layer_idx=0,
        device="cpu",
    )
    assert block_sliding.rope_scaling_factor == 1.0
    assert block_sliding.sliding_window == 1024
    assert block_sliding.rotary.scaling_factor == 1.0


# =========================================================================
# Fix 2: Single-worker KV cache trimming deduplication
# =========================================================================

def test_single_worker_speculative_kv_trim_deduplication():
    """Verify that on a single worker, KV trimming is not executed twice upon candidate rejection."""
    assign_single = AssignLayers(
        worker_id="single-worker",
        model_name="test-gemma",
        layer_start=0,
        layer_end=2,
        total_layers=2,
        is_first_stage=True,
        is_last_stage=True,
    )
    model = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=2,
        is_first_stage=True,
        is_last_stage=True,
        hidden_dim=64,
        vocab_size=100,
        num_heads=2,
        num_kv_heads=2,
        intermediate_size=128,
        head_dim=32,
        architecture="gemma3",
        device="cpu",
    )
    engine = StageExecutionEngine(stage_model=model, assignment=assign_single, enable_speculative=True)
    req_id = "req-dedup-test"

    # Mock KV cache with 10 tokens
    k = torch.randn(1, 2, 10, 32)
    v = torch.randn(1, 2, 10, 32)
    engine.session_kv_caches[req_id] = [(k, v), (k, v)]
    engine.kv_cache_manager.set_all_kvs(req_id, [(k, v), (k, v)])

    start_msg = InferenceStart(request_id=req_id, model_name="test-gemma", input_ids=[1, 2, 3])
    engine.session_configs[req_id] = start_msg
    engine.session_token_histories[req_id] = [1, 2, 3]

    cand_frame = SpeculativeCandidates(
        request_id=req_id,
        step_index=0,
        candidate_tokens=[10, 20, 30],
        position_offset=3,
        shape=[1, 4, 64],
        dtype="fp16",
        scale=1.0,
        payload_b64="",
    )

    # Mock logits where candidate tokens are rejected: 3 rejected drafts
    logits = torch.randn(1, 4, 100)
    with patch.object(engine.verifier, "verify_candidates", return_value=([], 99)):
        # In _verify_and_create_speculative_frame, trimming MUST be skipped on single-worker
        frame = engine._verify_and_create_speculative_frame(cand_frame, logits)
        assert frame.type == MessageType.SPECULATIVE_VERIFIED
        verified_payload = SpeculativeVerified.model_validate(frame.payload)
        assert verified_payload.rejected_draft_count == 3

        # Cache must still contain all 10 tokens (not trimmed prematurely in _verify_and_create_speculative_frame)
        kvs = engine.session_kv_caches[req_id]
        assert kvs[0][0].shape[-2] == 10

        # Now when process_speculative_verified runs, it trims rejected drafts exactly once (10 - 3 = 7),
        # followed by process_next_token forwarding the last accepted token (+1), ending at shape 8.
        # If double-trimming were present, it would be (10 - 3 - 3 + 1 = 5).
        engine.process_speculative_verified(verified_payload)
        kvs_trimmed = engine.session_kv_caches[req_id]
        assert kvs_trimmed[0][0].shape[-2] == 8  # 10 - 3 (trimmed once) + 1 (next token forward) = 8


# =========================================================================
# Fix 3: Monotonic cumulative sequence position tracking
# =========================================================================

def test_monotonic_sequence_position_tracking():
    """Verify that process_speculative_verified calculates true cumulative sequence offset."""
    assign = AssignLayers(
        worker_id="w-0",
        model_name="test-m",
        layer_start=0,
        layer_end=2,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=False,
    )
    model = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=64,
        vocab_size=100,
        num_heads=2,
        num_kv_heads=2,
        intermediate_size=128,
        head_dim=32,
        architecture="llama",
        device="cpu",
    )
    engine = StageExecutionEngine(stage_model=model, assignment=assign)
    req_id = "req-pos-test"

    # Prompt of 25 tokens
    prompt_ids = list(range(100, 125))
    start_msg = InferenceStart(request_id=req_id, model_name="test-m", input_ids=prompt_ids)
    engine.session_configs[req_id] = start_msg
    # History contains prompt + first generated token (token 50)
    engine.session_token_histories[req_id] = prompt_ids + [50]
    engine.session_step_counts[req_id] = 26

    # Downstream stage accepts 3 tokens: [60, 70, 80] with step_index=3
    verified = SpeculativeVerified(
        request_id=req_id,
        step_index=3,  # Output step index (NOT sequence position)
        accepted_tokens=[60, 70, 80],
        accepted_texts=[" 60", " 70", " 80"],
        rejected_draft_count=1,
    )

    # Process verified frame
    engine.process_speculative_verified(verified)

    # Sequence position MUST account for prompt length (25) + generated history (1) + accepted prefix (2) = 28,
    # plus the immediate process_next_token step for the last accepted token (+1) = 29.
    # It must NOT be overwritten with verified.step_index (which is 3 or 4)!
    assert engine.session_step_counts[req_id] == 29
    assert engine.session_step_counts[req_id] != verified.step_index


# =========================================================================
# Fix 4: Drafter 16-token cycle breaking and system prompt filtering
# =========================================================================

def test_prompt_lookup_drafter_16_token_cycle_breaking():
    """Verify repetition cycle breaking detects and rejects loops up to 16 tokens."""
    drafter = PromptLookupDrafter(ngram_size=2, max_draft_tokens=4)

    # 8-token loop (e.g. "running on a distributed GPU cluster") repeated twice
    phrase_8 = [101, 102, 103, 104, 105, 106, 107, 108]
    looping_history = [1, 2] + phrase_8 + phrase_8
    candidates = drafter.draft_candidates(looping_history)
    assert candidates == [], "Drafter must break 8-token repetition cycles"

    # 12-token loop repeated twice
    phrase_12 = list(range(200, 212))
    looping_12 = phrase_12 + phrase_12
    candidates_12 = drafter.draft_candidates(looping_12)
    assert candidates_12 == [], "Drafter must break 12-token repetition cycles"


def test_prompt_lookup_drafter_system_prompt_filter():
    """Verify that phrases originating exclusively from the system prompt are not drafted."""
    drafter = PromptLookupDrafter(ngram_size=2, max_draft_tokens=4)

    # System prompt: [10, 20, 30, 40, 50, 60] (length 6)
    # User / generation history: [99, 10, 20]
    # The pattern [10, 20] matches in the system prompt followed by [30, 40, 50, 60]
    sys_prompt_len = 6
    history = [10, 20, 30, 40, 50, 60, 99, 10, 20]

    # Without system prompt length, it would draft [30, 40, 50, 60]
    cand_unfiltered = drafter.draft_candidates(history, system_prompt_len=None)
    assert cand_unfiltered == [30, 40, 50, 60]

    # With system prompt length specified, phrases exclusively in system prompt are filtered
    cand_filtered = drafter.draft_candidates(history, system_prompt_len=sys_prompt_len)
    assert cand_filtered == [], "Phrases exclusively from system prompt must be filtered"

    # But if the phrase appeared in the conversation after the system prompt, it should be draftable
    conversation_history = [10, 20, 30, 40, 50, 60, 10, 20, 30, 40, 88, 10, 20]
    cand_conversation = drafter.draft_candidates(conversation_history, system_prompt_len=sys_prompt_len)
    assert cand_conversation != []


# =========================================================================
# Fix 5: Static Linear Execution (Zero dynamic allocations)
# =========================================================================

def test_int4_weight_linear_dynamic_execution():
    """Verify Int4WeightLinear does not cache unpacked weight to prevent VRAM overflow across 238 layers."""
    torch.manual_seed(42)
    in_dim, out_dim = 64, 128
    w = torch.randn(out_dim, in_dim, dtype=torch.float32)
    linear = Int4WeightLinear.from_float(w)

    x = torch.randn(1, 1, in_dim, dtype=torch.float32)

    # Dynamic forward pass unpacks temporarily and frees unpacked tensor immediately after F.linear
    assert linear._cached_weight is None
    out1 = linear(x)
    assert linear._cached_weight is None

    out2 = linear(x)
    assert torch.allclose(out1, out2)
    assert linear._cached_weight is None


def test_int8_weight_linear_dynamic_execution():
    """Verify Int8WeightLinear does not cache unpacked weight to prevent VRAM overflow across 238 layers."""
    torch.manual_seed(42)
    in_dim, out_dim = 64, 128
    w = torch.randn(out_dim, in_dim, dtype=torch.float32)
    linear = Int8WeightLinear.from_float(w)

    x = torch.randn(1, 1, in_dim, dtype=torch.float32)

    assert linear._cached_weight is None
    out1 = linear(x)
    assert linear._cached_weight is None

    out2 = linear(x)
    assert torch.allclose(out1, out2)
    assert linear._cached_weight is None


# =========================================================================
# Fix 6: Single-Worker Speculative Default Setting
# =========================================================================

def test_enable_speculative_defaults():
    """Verify enable_speculative defaults to False on single-worker and True on multi-worker."""
    assign_single = AssignLayers(
        worker_id="w-single",
        model_name="test-m",
        layer_start=0,
        layer_end=4,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=True,
    )
    assign_multi = AssignLayers(
        worker_id="w-multi-0",
        model_name="test-m",
        layer_start=0,
        layer_end=2,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=False,
    )
    model = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=32,
        vocab_size=50,
        num_heads=2,
        num_kv_heads=2,
        intermediate_size=64,
        head_dim=16,
        architecture="llama",
        device="cpu",
    )

    # Single-worker default: False
    e_single = StageExecutionEngine(stage_model=model, assignment=assign_single)
    assert e_single.enable_speculative is False

    # Multi-worker default: True
    e_multi = StageExecutionEngine(stage_model=model, assignment=assign_multi)
    assert e_multi.enable_speculative is True

    # Explicit override honored
    e_single_explicit = StageExecutionEngine(stage_model=model, assignment=assign_single, enable_speculative=True)
    assert e_single_explicit.enable_speculative is True
