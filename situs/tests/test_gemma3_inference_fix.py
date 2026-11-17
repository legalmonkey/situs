"""Automated unit tests for the Gemma 3 inference and snapshot completeness fixes."""

import json
import pathlib
import pytest
import torch
from unittest.mock import MagicMock

from situs.inference_backend import is_model_downloaded, _is_snapshot_complete
from situs.worker.loader import (
    DistributedStageModel,
    find_local_safetensors_files,
    resolve_model_metadata,
)
from situs.worker.sampler import TokenSampler


def test_gemma3_config_vocab_size_is_262208():
    """Verify that gemma-3-4b config specifies the verified 262,208 vocabulary size."""
    cfg_path = pathlib.Path(__file__).resolve().parent.parent / "configs" / "gemma-3-4b.json"
    assert cfg_path.is_file()
    with open(cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data.get("vocab_size") == 262208

    meta = resolve_model_metadata("google/gemma-3-4b-it")
    assert meta["vocab_size"] == 262208


def test_is_snapshot_complete_detects_missing_shards(tmp_path):
    """Verify that a snapshot missing index shards or tokenizer is strictly marked incomplete."""
    snap = tmp_path / "mock_snapshot"
    snap.mkdir()

    # Empty directory -> incomplete
    assert not _is_snapshot_complete(snap)

    # config.json present but no tokenizer or weights -> incomplete
    (snap / "config.json").write_text("{}", encoding="utf-8")
    assert not _is_snapshot_complete(snap)

    # tokenizer present but no weights -> incomplete
    (snap / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert not _is_snapshot_complete(snap)

    # Index says 2 shards are required
    index_data = {
        "weight_map": {
            "layer_0.weight": "model-00001-of-00002.safetensors",
            "layer_1.weight": "model-00002-of-00002.safetensors",
        }
    }
    (snap / "model.safetensors.index.json").write_text(json.dumps(index_data), encoding="utf-8")

    # Only shard 1 exists (> 1MB) -> incomplete
    (snap / "model-00001-of-00002.safetensors").write_bytes(b"\x00" * (1024 * 1024 + 10))
    assert not _is_snapshot_complete(snap)

    # Shard 2 is also created -> complete!
    (snap / "model-00002-of-00002.safetensors").write_bytes(b"\x00" * (1024 * 1024 + 10))
    assert _is_snapshot_complete(snap)


def test_find_local_safetensors_files_normalizes_aliases():
    """Verify that short model aliases like gemma3:4b correctly resolve to HF cache paths."""
    files = find_local_safetensors_files("gemma3:4b")
    # If the user's HF cache contains google/gemma-3-4b-it, it must be discovered
    if files:
        assert any("gemma-3-4b-it" in f for f in files)


def test_load_embedding_weights_dynamic_adaptation():
    """Verify that load_embedding_weights dynamically adapts to checkpoint tensor shape."""
    stage = DistributedStageModel(
        layer_start=0,
        layer_end=1,
        total_layers=2,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=64,
        vocab_size=256000,  # Initially set to smaller estimate
        num_heads=2,
        num_kv_heads=2,
        intermediate_size=128,
        head_dim=32,
        architecture="gemma3",
        device="cpu",
    )

    # Checkpoint has 262208 embeddings
    real_weights = torch.randn(262208, 64)
    # Should adapt seamlessly without throwing ValueError
    stage.load_embedding_weights(real_weights)
    assert stage.vocab_size == 262208
    assert stage.embed_tokens.num_embeddings == 262208


def test_load_head_weights_dynamic_adaptation():
    """Verify that load_head_weights dynamically adapts lm_head to checkpoint tensor shape."""
    stage = DistributedStageModel(
        layer_start=1,
        layer_end=2,
        total_layers=2,
        is_first_stage=False,
        is_last_stage=True,
        hidden_dim=64,
        vocab_size=256000,
        num_heads=2,
        num_kv_heads=2,
        intermediate_size=128,
        head_dim=32,
        architecture="gemma3",
        device="cpu",
    )

    real_lm_head = torch.randn(262208, 64)
    real_norm = torch.randn(64)
    stage.load_head_weights(lm_head_weight=real_lm_head, norm_weight=real_norm)
    assert stage.lm_head.out_features == 262208


def test_token_sampler_decode_token_filters_unused_tokens():
    """Verify that TokenSampler.decode_token suppresses special IDs and <unused...> tags."""
    mock_tok = MagicMock()
    mock_tok.all_special_ids = [0, 1, 2, 1018]
    mock_tok.decode.side_effect = lambda ids, **kwargs: "<unused846>" if 1018 in ids else "Hello"

    sampler = TokenSampler(tokenizer=mock_tok)

    # Special token 1018 should decode to empty string
    assert sampler.decode_token(1018) == ""

    # Normal token 9259 should decode to "Hello"
    assert sampler.decode_token(9259) == "Hello"
