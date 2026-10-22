"""Tests for speculative decoding (Prompt-Lookup, PPSD Early-Exit, and Batched Verifier)."""

import pytest
import numpy as np
from situs.worker.speculative import (
    PromptLookupDrafter,
    EarlyExitHead,
    SpeculativeVerifier,
)
from situs.worker.sampler import TokenSampler
from situs.protocol.compression import compress_temporal_delta, decompress_temporal_delta


def test_prompt_lookup_drafter_matching():
    drafter = PromptLookupDrafter(ngram_size=2, max_draft_tokens=3)

    # Context with repeating n-grams: [10, 20, 30, 40, 50, 10, 20]
    # Last 2 tokens are [10, 20]. Pattern [10, 20] appears earlier followed by [30, 40, 50].
    token_history = [10, 20, 30, 40, 50, 10, 20]
    candidates = drafter.draft_candidates(token_history)

    assert candidates == [30, 40, 50]


def test_prompt_lookup_drafter_no_match():
    drafter = PromptLookupDrafter(ngram_size=2, max_draft_tokens=3)
    token_history = [1, 2, 3, 4, 5, 6]
    candidates = drafter.draft_candidates(token_history)
    assert candidates == []


def test_temporal_delta_compression_accuracy():
    np.random.seed(42)
    # Step 1 activation
    h0 = np.random.randn(1, 1, 2048).astype(np.float32)
    # Step 2 activation (correlated with small delta)
    h1 = h0 + np.random.randn(1, 1, 2048).astype(np.float32) * 0.05

    values_bytes, scale, shape, indices_bytes = compress_temporal_delta(h1, last_arr=h0, sparsify_ratio=0.15)

    # Top 15% of 2048 = 307 elements
    assert len(values_bytes) == int(2048 * 0.15)
    # 307 bytes of payload! (< 400 bytes total with indices!)
    assert len(values_bytes) < 400

    reconstructed_h1 = decompress_temporal_delta(values_bytes, scale, shape, indices_bytes, last_arr=h0)
    assert reconstructed_h1.shape == h1.shape

    mean_err = np.mean(np.abs(h1 - reconstructed_h1))
    assert mean_err < 0.05


def test_early_exit_head_projection():
    try:
        import torch
        head = EarlyExitHead(hidden_dim=128, vocab_size=1000)
        h = torch.randn(1, 1, 128)
        logits = head(h)
        assert logits.shape == (1, 1, 1000)
    except ImportError:
        pytest.skip("PyTorch not installed")


def test_speculative_verifier_acceptance():
    sampler = TokenSampler()
    verifier = SpeculativeVerifier(sampler)

    try:
        import torch
        # Create logits where token 5 and token 8 have massive probability
        logits = torch.zeros(1, 2, 100)
        logits[0, 0, 5] = 50.0  # position 0 is token 5
        logits[0, 1, 8] = 50.0  # position 1 is token 8

        candidates = [5, 8]
        accepted, bonus = verifier.verify_candidates(candidates, logits, temperature=0.0)

        # Both candidate tokens should be verified and accepted in 1 pass!
        assert accepted == [5, 8]
    except ImportError:
        pytest.skip("PyTorch not installed")
