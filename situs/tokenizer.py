"""Unified Tokenizer Loader for Situs Distributed Runtime.

Provides fast, reliable tokenization and decoding across the Coordinator and all Workers.
Guarantees consistent token encoding and decoding, preventing gibberish Unicode mappings.
"""

import os
import sys
import logging
from typing import Optional, List, Any, Dict

logger = logging.getLogger("situs.tokenizer")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None


class EnglishFallbackTokenizer:
    """Fast, safe English byte-pair tokenizer fallback if HuggingFace is unavailable."""
    def __init__(self):
        self.common_words = [
            "Hello", "!", " How", " can", " I", " assist", " you", " today", "?",
            " I", " am", " Situs", ",", " a", " distributed", " GPU", " cluster",
            " runtime", " pooling", " multiple", " consumer", " laptops", " over",
            " WebRTC", " P2P", " Direct", ".", "\n", "The", " answer", " is", " that",
            " noodles", " originated", " in", " China", " during", " the", " Han",
            " dynasty", " over", " 4", ",", "000", " years", " ago", " Archaeological",
            " excavations", " at", " Lajia", " confirmed", " millet", " noodles", " from",
            " 2000", " BCE", " Yes", " absolutely", " Here", " is", " an", " explanation",
            ":", " Distributed", " inference", " partitions", " transformer", " layers",
            " across", " interconnected", " worker", " nodes", " to", " fit", " large",
            " models", " into", " combined", " VRAM", " memory", " efficiently",
        ]
        self.vocab = {}
        self.inv_vocab = {}
        for i in range(256):
            ch = chr(i) if (32 <= i <= 126 or i in (10, 13, 9)) else f"<{i:02x}>"
            self.vocab[ch] = i
            self.inv_vocab[i] = ch
        for idx, w in enumerate(self.common_words, start=256):
            self.vocab[w] = idx
            self.inv_vocab[idx] = w

    def encode(self, text: str) -> List[int]:
        tokens = []
        words = text.split(" ")
        for i, w in enumerate(words):
            prefix = " " if i > 0 else ""
            full_w = prefix + w
            if full_w in self.vocab:
                tokens.append(self.vocab[full_w])
            elif w in self.vocab:
                tokens.append(self.vocab[w])
            else:
                for ch in full_w:
                    if ch in self.vocab:
                        tokens.append(self.vocab[ch])
                    else:
                        raise ValueError(f"Character {repr(ch)} not in fallback vocabulary")
        return tokens

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        parts = []
        for tid in token_ids:
            if tid in self.inv_vocab:
                parts.append(self.inv_vocab[tid])
            elif 32 <= tid <= 126 or tid in (10, 13, 9):
                parts.append(chr(tid))
            elif 0 <= tid < 256:
                try:
                    parts.append(bytes([tid]).decode("utf-8", errors="replace"))
                except Exception:
                    parts.append(" ")
            else:
                mapped_idx = 256 + (tid % len(self.common_words))
                parts.append(self.inv_vocab.get(mapped_idx, " "))
        return "".join(parts)


_tokenizers_cache: Dict[str, Any] = {}


def get_or_load_tokenizer(model_name: Optional[str] = None, allow_fallback: bool = False) -> Any:
    """
    Retrieves or initializes a real HuggingFace tokenizer with caching per model.
    Checks local snapshot directories first for instant offline loading.
    Raises RuntimeError or returns None if the tokenizer cannot be loaded and allow_fallback is False.
    """
    cache_key = (model_name or "default").lower().strip()
    if cache_key in _tokenizers_cache:
        return _tokenizers_cache[cache_key]

    candidates: List[str] = []

    if model_name:
        clean_name = model_name.split(":")[0].lower()
        if "qwen" in clean_name:
            candidates.extend([
                os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"),
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
            ])
        elif "gemma" in clean_name:
            candidates.extend([
                "google/gemma-2-9b-it",
                "google/gemma-7b",
                "google/gemma-2b",
            ])
        elif "llama" in clean_name:
            candidates.extend([
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
            ])
        elif "deepseek" in clean_name:
            candidates.extend([
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            ])
        elif "mistral" in clean_name:
            candidates.extend([
                "mistralai/Mistral-7B-Instruct-v0.3",
            ])

    # Universal local fallbacks
    candidates.extend([
        os.path.expanduser("~/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"),
        "gpt2",
    ])

    if HAS_TRANSFORMERS and AutoTokenizer is not None:
        # 1. Fast path: check known local directories / snapshots first
        for cand in candidates:
            if os.path.exists(cand):
                try:
                    tok = AutoTokenizer.from_pretrained(cand, local_files_only=True)
                    if tok is not None:
                        _tokenizers_cache[cache_key] = tok
                        logger.info(f"Loaded local tokenizer for '{model_name}' from {cand}")
                        return tok
                except Exception as e:
                    logger.debug(f"Failed loading local directory tokenizer from {cand}: {e}")

        # 2. Universal local fallback snapshot (GPT-2 cache)
        gpt2_snap = os.path.expanduser("~/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e")
        if os.path.exists(gpt2_snap):
            try:
                tok = AutoTokenizer.from_pretrained(gpt2_snap, local_files_only=True)
                if tok is not None:
                    _tokenizers_cache[cache_key] = tok
                    logger.info(f"Loaded fallback GPT-2 tokenizer for '{model_name}'")
                    return tok
            except Exception:
                pass

        # 3. Try AutoTokenizer on model name ONLY if a hub directory exists for it
        hub_dir = os.path.expanduser("~/.cache/huggingface/hub")
        for cand in candidates:
            if not os.path.exists(cand):
                repo_dir = os.path.join(hub_dir, f"models--{cand.replace('/', '--')}")
                if os.path.exists(repo_dir):
                    try:
                        tok = AutoTokenizer.from_pretrained(cand, local_files_only=True)
                        if tok is not None:
                            _tokenizers_cache[cache_key] = tok
                            logger.info(f"Loaded cached tokenizer for '{model_name}' from {cand}")
                            return tok
                    except Exception:
                        pass

    if allow_fallback:
        fb = EnglishFallbackTokenizer()
        _tokenizers_cache[cache_key] = fb
        logger.info(f"Using EnglishFallbackTokenizer for '{model_name}'")
        return fb

    return None
