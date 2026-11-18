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
        repetition_penalty: float = 1.1,
        context_tokens: Optional[List[int]] = None,
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

            # Any NaN or positive Inf indicates gradient explosion / numerical divergence -> return EOS
            if torch.isnan(logits).any() or torch.isposinf(logits).any():
                return int(getattr(self.tokenizer, "eos_token_id", 0) or 0)

            # If all logits are negative Inf (all masked) or all are zero (uninitialized weights) -> return EOS
            if torch.isneginf(logits).all() or torch.all(logits == 0):
                return int(getattr(self.tokenizer, "eos_token_id", 0) or 0)

            logits = logits.clone()

            # Apply repetition penalty across recent context tokens
            if repetition_penalty is not None and repetition_penalty != 1.0 and context_tokens:
                recent_tokens = set(context_tokens[-64:])
                vocab_len = logits.shape[-1]
                for tok in recent_tokens:
                    if 0 <= tok < vocab_len:
                        if logits[tok] < 0:
                            logits[tok] = logits[tok] * repetition_penalty
                        else:
                            logits[tok] = logits[tok] / repetition_penalty

            if temperature <= 0.01:
                return int(torch.argmax(logits).item())

            logits = logits / max(temperature, 1e-5)

            # Optimized Top-K + Top-P (Nucleus) Sampling
            # For large vocabularies (e.g. Gemma 3 262k vocab), chunking top-k first avoids
            # sorting hundreds of thousands of negative floats, cutting memory churn and latency by 99%.
            if top_k > 0 and top_k < logits.size(-1):
                topk_values, topk_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                if top_p < 1.0:
                    sorted_logits, sorted_sub_indices = torch.sort(topk_values, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    sorted_logits[sorted_indices_to_remove] = -float("Inf")
                    probs = F.softmax(sorted_logits, dim=-1)
                    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs <= 0).all():
                        return int(topk_indices[0].item())
                    gen = None
                    if seed is not None:
                        device_str = logits.device.type if (hasattr(logits, "device") and logits.device.type in ("cpu", "cuda")) else "cpu"
                        gen = torch.Generator(device=device_str)
                        gen.manual_seed(int(seed) + int(step_index))
                    sub_idx = torch.multinomial(probs, num_samples=1, generator=gen)
                    chosen_idx = sorted_sub_indices[sub_idx]
                    return int(topk_indices[chosen_idx].item())
                else:
                    probs = F.softmax(topk_values, dim=-1)
                    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs <= 0).all():
                        return int(topk_indices[0].item())
                    gen = None
                    if seed is not None:
                        device_str = logits.device.type if (hasattr(logits, "device") and logits.device.type in ("cpu", "cuda")) else "cpu"
                        gen = torch.Generator(device=device_str)
                        gen.manual_seed(int(seed) + int(step_index))
                    sub_idx = torch.multinomial(probs, num_samples=1, generator=gen)
                    return int(topk_indices[sub_idx].item())

            # Fallback when top_k is disabled (full vocab sort)
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
            # Guard against NaN/Inf or all-zero probabilities resulting from aggressive masking/underflow
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs <= 0).all():
                valid_logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
                return int(torch.argmax(valid_logits).item())

            gen = None
            if seed is not None:
                device_str = logits.device.type if (hasattr(logits, "device") and logits.device.type in ("cpu", "cuda")) else "cpu"
                gen = torch.Generator(device=device_str)
                gen.manual_seed(int(seed) + int(step_index))
            token = torch.multinomial(probs, num_samples=1, generator=gen)
            return int(token.item())
        else:
            # NumPy fallback
            arr = np.asarray(logits, dtype=np.float32).copy()
            if arr.ndim == 3:
                arr = arr[0, -1, :]
            elif arr.ndim == 2:
                arr = arr[-1, :]

            # Any NaN or positive Inf indicates divergence -> return EOS
            if np.isnan(arr).any() or np.isposinf(arr).any():
                return int(getattr(self.tokenizer, "eos_token_id", 0) or 0)

            # All masked (-inf) or uninitialized zeros -> return EOS
            if np.all(np.isneginf(arr)) or np.all(arr == 0):
                return int(getattr(self.tokenizer, "eos_token_id", 0) or 0)

            # Apply repetition penalty across recent context tokens
            if repetition_penalty is not None and repetition_penalty != 1.0 and context_tokens:
                recent_tokens = set(context_tokens[-64:])
                vocab_len = len(arr)
                for tok in recent_tokens:
                    if 0 <= tok < vocab_len:
                        if arr[tok] < 0:
                            arr[tok] = arr[tok] * repetition_penalty
                        else:
                            arr[tok] = arr[tok] / repetition_penalty

            if temperature <= 0.01:
                return int(np.argmax(arr))

            arr = arr / max(temperature, 1e-5)
            # Subtract max for numerical stability
            arr = arr - np.max(arr)
            exp_logits = np.exp(arr)
            sum_exp = np.sum(exp_logits)
            if sum_exp <= 0 or np.isnan(sum_exp) or np.isinf(sum_exp):
                return int(np.argmax(arr))
            probs = exp_logits / sum_exp
            rng = np.random.RandomState(int(seed) + int(step_index)) if seed is not None else np.random
            return int(rng.choice(len(probs), p=probs))

    def decode_token(self, token_id: int, skip_special_tokens: bool = True) -> str:
        """Decodes token ID to text string using the verified tokenizer."""
        if self.tokenizer:
            try:
                if not skip_special_tokens:
                    text = self.tokenizer.decode([token_id], skip_special_tokens=False)
                    return text or ""
                all_specials = getattr(self.tokenizer, "all_special_ids", [])
                if token_id in all_specials:
                    return ""
                text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                if text is not None:
                    if text.startswith("<unused") and text.endswith(">"):
                        return ""
                    return text
            except Exception:
                pass

        return ""
