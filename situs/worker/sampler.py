"""Token sampling and decoding utilities for the last pipeline stage."""

import numpy as np
from typing import Optional, List, Tuple, Union, Any

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TokenSampler:
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer

    def sample_next_token(
        self,
        logits: Union["torch.Tensor", np.ndarray],
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
    ) -> int:
        """
        Samples the next token from logits using temperature, top-k, and nucleus (top-p) sampling.
        """
        if HAS_TORCH and isinstance(logits, torch.Tensor):
            # Take logits of the last token in sequence: shape [vocab_size]
            if logits.dim() == 3:
                logits = logits[0, -1, :]
            elif logits.dim() == 2:
                logits = logits[-1, :]

            if temperature <= 0.01:
                return int(torch.argmax(logits).item())

            logits = logits / max(temperature, 1e-5)

            # Top-K
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1, None]
                logits[indices_to_remove] = -float("Inf")

            # Top-P (Nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift right to keep at least one token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[indices_to_remove] = -float("Inf")

            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
            return int(token.item())
        else:
            # NumPy fallback
            arr = np.asarray(logits, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0, -1, :]
            elif arr.ndim == 2:
                arr = arr[-1, :]

            if temperature <= 0.01:
                return int(np.argmax(arr))

            arr = arr / max(temperature, 1e-5)
            # Subtract max for numerical stability
            arr = arr - np.max(arr)
            exp_logits = np.exp(arr)
            probs = exp_logits / np.sum(exp_logits)
            return int(np.random.choice(len(probs), p=probs))

    def decode_token(self, token_id: int) -> str:
        """Decodes token ID to text string."""
        if not self.tokenizer:
            try:
                from situs.tokenizer import get_or_load_tokenizer
                self.tokenizer = get_or_load_tokenizer()
            except Exception:
                pass

        if self.tokenizer:
            try:
                text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                if text:
                    return text
            except Exception:
                pass

        # Safe fallback: ASCII only, never produce CJK unicode glyphs from arbitrary integers
        if 32 <= token_id <= 126 or token_id in (10, 13, 9):
            return chr(token_id)
        elif 0 <= token_id < 256:
            try:
                return bytes([token_id]).decode("utf-8", errors="replace")
            except Exception:
                pass
        return " "
