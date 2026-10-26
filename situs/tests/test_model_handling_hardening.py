"""
Regression tests for model weights validation, metadata parsing, and tokenizer safety.
Ensures zero mock/random blocks, dynamic runtime architecture derivation, and strict safety checks.
"""

import os
import json
import pytest
import torch
from safetensors.torch import save

from situs.worker.loader import (
    deserialize_and_validate_layer_weights,
    DistributedStageModel,
    RealTransformerBlock,
)
from situs.coordinator.registry import (
    resolve_model_metadata,
    _parse_verified_config_dict,
)
from situs.tokenizer import (
    get_or_load_tokenizer,
    EnglishFallbackTokenizer,
)


# =====================================================================
# 1. Model Weights Validation Tests
# =====================================================================

def _make_valid_block_tensors(hidden_dim=64, num_heads=4, num_kv_heads=4, intermediate_dim=128):
    head_dim = hidden_dim // num_heads
    return {
        "input_layernorm.weight": torch.ones(hidden_dim, dtype=torch.float32),
        "q_proj.weight": torch.randn(num_heads * head_dim, hidden_dim, dtype=torch.float32),
        "k_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_dim, dtype=torch.float32),
        "v_proj.weight": torch.randn(num_kv_heads * head_dim, hidden_dim, dtype=torch.float32),
        "o_proj.weight": torch.randn(hidden_dim, num_heads * head_dim, dtype=torch.float32),
        "post_attention_layernorm.weight": torch.ones(hidden_dim, dtype=torch.float32),
        "gate_proj.weight": torch.randn(intermediate_dim, hidden_dim, dtype=torch.float32),
        "up_proj.weight": torch.randn(intermediate_dim, hidden_dim, dtype=torch.float32),
        "down_proj.weight": torch.randn(hidden_dim, intermediate_dim, dtype=torch.float32),
    }


def test_weight_validation_all_required_tensors():
    """Verify that all 9 required transformer block tensors must be present."""
    tensors = _make_valid_block_tensors(hidden_dim=64, intermediate_dim=128)
    # Remove one required tensor
    del tensors["gate_proj.weight"]

    block = RealTransformerBlock(hidden_dim=64, num_heads=4, num_kv_heads=4, intermediate_size=128, head_dim=16, architecture="llama")
    with pytest.raises(ValueError, match="required parameter 'gate_proj.weight' is missing"):
        block.load_layer_weights(tensors)


def test_weight_validation_shape_mismatch():
    """Verify that tensors with mismatched shapes are rejected."""
    tensors = _make_valid_block_tensors(hidden_dim=64, intermediate_dim=128)
    # Corrupt shape of k_proj
    tensors["k_proj.weight"] = torch.randn(128, 128)

    block = RealTransformerBlock(hidden_dim=64, num_heads=4, num_kv_heads=4, intermediate_size=128, head_dim=16, architecture="llama")
    with pytest.raises(ValueError, match="Shape mismatch for parameter 'k_proj.weight'"):
        block.load_layer_weights(tensors)


def test_weight_validation_rejects_nan_and_inf():
    """Verify that weights containing NaN or Inf are strictly rejected."""
    tensors_nan = _make_valid_block_tensors(hidden_dim=64, intermediate_dim=128)
    tensors_nan["q_proj.weight"][0, 0] = float("nan")

    block = RealTransformerBlock(hidden_dim=64, num_heads=4, num_kv_heads=4, intermediate_size=128, head_dim=16, architecture="llama")
    with pytest.raises(ValueError, match="contains NaN or Inf values"):
        block.load_layer_weights(tensors_nan)

    tensors_inf = _make_valid_block_tensors(hidden_dim=64, intermediate_dim=128)
    tensors_inf["o_proj.weight"][0, 0] = float("inf")
    with pytest.raises(ValueError, match="contains NaN or Inf values"):
        block.load_layer_weights(tensors_inf)


def test_deserialize_garbage_bytes_rejected():
    """Verify that arbitrary or corrupted byte payloads cannot produce mock blocks."""
    garbage = b"not-a-safetensor-or-torch-payload-123456789"
    with pytest.raises((ValueError, RuntimeError), match="Failed to deserialize layer weights"):
        deserialize_and_validate_layer_weights(garbage, layer_idx=0, hidden_dim=64, num_heads=4)


def test_deserialize_valid_safetensors_derives_architecture():
    """Verify deserialization correctly infers dimensions from genuine tensor shapes."""
    tensors = _make_valid_block_tensors(hidden_dim=64, num_heads=4, num_kv_heads=2, intermediate_dim=128)
    raw_bytes = save(tensors)

    block = deserialize_and_validate_layer_weights(raw_bytes, layer_idx=1, hidden_dim=64, num_heads=4)
    assert isinstance(block, RealTransformerBlock)
    assert block.hidden_dim == 64
    assert block.num_heads == 4
    assert block.num_kv_heads == 2
    assert block.intermediate_size == 128


def test_mark_layer_ready_rejects_none():
    """Verify that mark_layer_ready strictly refuses None weights."""
    stage = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=2,
        is_first_stage=True,
        is_last_stage=True,
        hidden_dim=64,
        vocab_size=1000,
        num_heads=4,
        num_kv_heads=4,
        intermediate_size=128,
        head_dim=16,
        architecture="llama",
        progressive=True,
    )
    with pytest.raises(ValueError, match="Cannot mark layer 0 ready without valid weights"):
        stage.mark_layer_ready(layer_index=0, weights=None)


# =====================================================================
# 2. Model Metadata Parsing & Resolution Tests
# =====================================================================

def test_metadata_dynamic_known_models():
    """Verify exact vocabulary sizes and dimensions are derived from verified configs."""
    llama_meta = resolve_model_metadata("llama-3.1:8b-instruct-q8_0")
    assert llama_meta["vocab_size"] == 128256
    assert llama_meta["total_layers"] == 32
    assert llama_meta["hidden_dim"] == 4096
    assert llama_meta["num_heads"] == 32
    assert llama_meta["num_kv_heads"] == 8

    gemma_meta = resolve_model_metadata("gemma-2-9b")
    assert gemma_meta["vocab_size"] == 256000
    assert gemma_meta["total_layers"] == 42
    assert gemma_meta["hidden_dim"] == 3584

    qwen_meta = resolve_model_metadata("qwen2.5:7b")
    assert qwen_meta["vocab_size"] == 152064
    assert qwen_meta["total_layers"] == 28
    assert qwen_meta["hidden_dim"] == 3584


def test_metadata_rejects_unknown_model_without_arbitrary_fallback():
    """Verify that completely unknown models raise RuntimeError rather than using arbitrary fallbacks."""
    with pytest.raises(RuntimeError, match="Cannot resolve model metadata for 'completely-unknown-fantasy-model'"):
        resolve_model_metadata("completely-unknown-fantasy-model")


def test_metadata_rejects_corrupted_config():
    """Verify that config dicts missing mandatory fields are rejected."""
    corrupted_dict = {
        "hidden_size": 2048,
        # missing num_hidden_layers and vocab_size
    }
    with pytest.raises(ValueError, match="is missing required attributes"):
        _parse_verified_config_dict(corrupted_dict)


# =====================================================================
# 3. Tokenizer Safety Tests
# =====================================================================

def test_english_fallback_tokenizer_disabled():
    """Verify that EnglishFallbackTokenizer cannot be instantiated."""
    with pytest.raises(RuntimeError, match="EnglishFallbackTokenizer is disabled"):
        EnglishFallbackTokenizer()


def test_tokenizer_refuses_arbitrary_gpt2_fallback():
    """Verify that querying a non-GPT-2 model does not silently return GPT-2 tokenizer."""
    with pytest.raises(RuntimeError, match="Could not load valid tokenizer for model 'non-existent-llama-variant'"):
        get_or_load_tokenizer("non-existent-llama-variant")


def test_weight_validation_rejects_indivisible_gqa_heads():
    """Verify that non-divisible GQA query-to-KV head ratios are rejected."""
    # 5 query heads with 2 KV heads (5 is not divisible by 2)
    head_dim = 16
    hidden_dim = 5 * head_dim  # 80
    tensors = {
        "input_layernorm.weight": torch.ones(hidden_dim, dtype=torch.float32),
        "q_proj.weight": torch.randn(5 * head_dim, hidden_dim, dtype=torch.float32),
        "k_proj.weight": torch.randn(2 * head_dim, hidden_dim, dtype=torch.float32),
        "v_proj.weight": torch.randn(2 * head_dim, hidden_dim, dtype=torch.float32),
        "o_proj.weight": torch.randn(hidden_dim, 5 * head_dim, dtype=torch.float32),
        "post_attention_layernorm.weight": torch.ones(hidden_dim, dtype=torch.float32),
        "gate_proj.weight": torch.randn(128, hidden_dim, dtype=torch.float32),
        "up_proj.weight": torch.randn(128, hidden_dim, dtype=torch.float32),
        "down_proj.weight": torch.randn(hidden_dim, 128, dtype=torch.float32),
    }
    raw_bytes = save(tensors)
    with pytest.raises(ValueError, match="Invalid GQA configuration"):
        deserialize_and_validate_layer_weights(raw_bytes, layer_idx=0, hidden_dim=hidden_dim, num_heads=5, num_kv_heads=2)


def test_sampler_decode_token_no_ascii_chr_fallback():
    """Verify that TokenSampler.decode_token returns empty string without tokenizer and never emits ASCII chr fallback."""
    from situs.worker.sampler import TokenSampler
    sampler = TokenSampler(tokenizer=None)
    # Token 65 ('A' in ASCII) must NOT be decoded to 'A'
    decoded = sampler.decode_token(65)
    assert decoded == "", f"Expected empty string, got '{decoded}' via unsafe ASCII fallback"


def test_cache_metadata_requires_verified_config(tmp_path):
    """Verify that cached model dirs without config.json do not fabricate arbitrary 32k metadata."""
    from situs.worker.jit_streamer import save_layer_to_cache
    model_name = "cached-layer-only-model-no-config"
    fake_weights = save(_make_valid_block_tensors(hidden_dim=64, intermediate_dim=128))
    save_layer_to_cache(model_name, layer_index=0, data=fake_weights)

    # Must raise RuntimeError rather than assuming 32000 vocab or arbitrary layer counts
    with pytest.raises(RuntimeError, match="Cannot resolve model metadata"):
        resolve_model_metadata(model_name)

