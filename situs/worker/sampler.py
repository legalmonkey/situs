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
        seed: Optional[int] = None,
        step_index: int = 0,
    ) -> int:
        """
        Samples the next token from logits using temperature, top-k, and nucleus (top-p) sampling.
        Supports deterministic Philox counter-based PRNG seeding (seed + step_index).
        """
        if HAS_TORCH and isinstance(logits, torch.Tensor):
            # Take logits of the last token in sequence: shape [vocab_size]
            if logits.dim() == 3:
                logits = logits[0, -1, :]
            elif logits.dim() == 2:
                logits = logits[-1, :]

            if torch.isnan(logits).any() or torch.isinf(logits).any():
                return int(getattr(self.tokenizer, "eos_token_id", 0) or 0)

            # Prevent uniform random sampling across 256k vocab if weights are uninitialized (all zeros)
            if torch.all(logits == 0):
                return int(getattr(self.tokenizer, "eos_token_id", 0) or 0)

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
            gen = None
            if seed is not None:
                device_str = logits.device.type if (hasattr(logits, "device") and logits.device.type in ("cpu", "cuda")) else "cpu"
                gen = torch.Generator(device=device_str)
                gen.manual_seed(int(seed) + int(step_index))
            token = torch.multinomial(probs, num_samples=1, generator=gen)
            return int(token.item())
        else:
            # NumPy fallback
            arr = np.asarray(logits, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr[0, -1, :]
            elif arr.ndim == 2:
                arr = arr[-1, :]

            if np.isnan(arr).any() or np.isinf(arr).any():
                return int(getattr(self.tokenizer, "eos_token_id", 0) or 0)

            if np.all(arr == 0):
                return int(getattr(self.tokenizer, "eos_token_id", 0) or 0)

            if temperature <= 0.01:
                return int(np.argmax(arr))

            arr = arr / max(temperature, 1e-5)
            # Subtract max for numerical stability
            arr = arr - np.max(arr)
            exp_logits = np.exp(arr)
            probs = exp_logits / np.sum(exp_logits)
            rng = np.random.RandomState(int(seed) + int(step_index)) if seed is not None else np.random
            return int(rng.choice(len(probs), p=probs))

    def decode_token(self, token_id: int) -> str:
        """Decodes token ID to text string using the verified tokenizer."""

        if self.tokenizer:
            try:
                text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                if text is not None:
                    return text
            except Exception:
                pass

        return ""
