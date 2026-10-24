"""Speculative decoding implementations: Prompt-Lookup, PPSD Early-Exit, and Batched Verifier."""

import logging
from typing import List, Tuple, Optional, Dict, Any
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object

logger = logging.getLogger("situs.worker.speculative")


class PromptLookupDrafter:
    """
    Drafterless candidate generation using n-gram prompt & history lookup.
    Requires ZERO additional weights and ZERO additional VRAM.
    """
    def __init__(self, ngram_size: int = 3, max_draft_tokens: int = 4):
        self.ngram_size = ngram_size
        self.max_draft_tokens = max_draft_tokens

    def draft_candidates(self, token_history: List[int]) -> List[int]:
        """
        Scans past token history for matches of the last `ngram_size` tokens
        and extracts subsequent tokens as speculative candidates.
        Includes cycle breaking to prevent infinite n-gram repetition loops.
        """
        if len(token_history) < self.ngram_size + 1:
            return []

        # Anti-repetition check: break runaway loops immediately
        if len(token_history) >= 3 and token_history[-1] == token_history[-2] == token_history[-3]:
            return []
        for period in (2, 3, 4):
            if len(token_history) >= period * 2 and token_history[-period:] == token_history[-period * 2 : -period]:
                return []

        pattern = token_history[-self.ngram_size:]
        search_window = token_history[:-1]

        # Search backward from the recent tokens for the pattern
        for i in range(len(search_window) - self.ngram_size, -1, -1):
            if search_window[i : i + self.ngram_size] == pattern:
                start_idx = i + self.ngram_size
                candidates = search_window[start_idx : start_idx + self.max_draft_tokens]
                # Reject candidates that immediately recreate the repeating loop
                if candidates and candidates[:self.ngram_size] == pattern:
                    continue
                if candidates:
                    return candidates

        return []


class EarlyExitHead(nn.Module if HAS_TORCH else object):
    """
    PPSD (Pipeline-Parallel Self-Speculation) early exit projection head.
    Projects intermediate stage hidden states directly to vocabulary logits.
    """
    def __init__(self, hidden_dim: int = 2048, vocab_size: int = 32000):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        if HAS_TORCH:
            self.norm = nn.LayerNorm(hidden_dim)
            self.proj = nn.Linear.__new__(nn.Linear)
            super(nn.Linear, self.proj).__init__()
            self.proj.in_features = hidden_dim
            self.proj.out_features = vocab_size
            self.proj.weight = nn.Parameter(torch.zeros(vocab_size, hidden_dim, dtype=torch.float32))
            self.proj.bias = None

    def forward(self, h: "torch.Tensor") -> "torch.Tensor":
        if not HAS_TORCH:
            return h
        return self.proj(self.norm(h))


class SpeculativeVerifier:
    """
    Verifies multi-token candidate sequences in a single parallel forward pass
    on the terminal pipeline stage (Worker B).
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def verify_candidates(
        self,
        candidate_tokens: List[int],
        logits_sequence: "torch.Tensor",
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[List[int], Optional[int]]:
        """
        Evaluates candidate tokens against full-model logits.
        Returns:
            (accepted_tokens, optional_resampled_correction_token)
        """
        accepted = []
        num_candidates = len(candidate_tokens)

        if not HAS_TORCH or not isinstance(logits_sequence, torch.Tensor):
            # No valid torch logits available: safely reject candidates to prevent runaway loops
            return [], None

        # logits_sequence shape: [1, num_candidates, vocab_size]
        for idx in range(num_candidates):
            cand_token = candidate_tokens[idx]
            token_logits = logits_sequence[0, idx, :]

            # Sample true token from target model
            target_token = self.sampler.sample_next_token(token_logits, temperature=temperature, top_p=top_p)

            if cand_token == target_token or temperature < 0.1:
                accepted.append(cand_token)
            else:
                # Rejection: accept tokens up to this point and append the target model's correction
                return accepted, target_token

        # If all candidates accepted, sample one bonus token from the last position
        bonus_token = self.sampler.sample_next_token(logits_sequence[0, -1, :], temperature=temperature, top_p=top_p)
        return accepted, bonus_token
