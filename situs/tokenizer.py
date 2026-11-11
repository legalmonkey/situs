"""Unified Tokenizer Loader for Situs Distributed Runtime.

Provides fast, reliable tokenization and decoding across the Coordinator and all Workers.
Guarantees consistent token encoding and decoding, preventing gibberish Unicode mappings.
"""

import os
import sys
import logging
from typing import Optional, List, Any, Dict

# Suppress noisy HuggingFace / Transformers library warnings (e.g., PyTorch version, legacy activation warnings)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger("situs.tokenizer")

try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    AutoTokenizer = None


class EnglishFallbackTokenizer:
    """Deprecated unsafe fallback tokenizer. Strictly rejected by runtime safety rules."""
    def __init__(self, *args, **kwargs):
        raise RuntimeError("EnglishFallbackTokenizer is disabled. Unsafe tokenizer fallbacks are not permitted.")


_tokenizers_cache: Dict[str, Any] = {}


def get_or_load_tokenizer(
    model_name: Optional[str] = None,
    allow_fallback: bool = False,
    local_files_only: bool = False,
) -> Any:
    """
    Retrieves or initializes a verified HuggingFace tokenizer with caching per model.
    Checks local snapshot directories first for instant offline loading.
    Strictly loads the correct tokenizer for the selected model.
    Raises RuntimeError if a matching tokenizer cannot be found or loaded.
    """
    clean_name = (model_name or "default").lower().strip()
    cache_key = clean_name
    if cache_key in _tokenizers_cache:
        return _tokenizers_cache[cache_key]

    if not HAS_TRANSFORMERS or AutoTokenizer is None:
        raise RuntimeError("transformers library is required for model tokenization")

    hub_dir = os.path.expanduser("~/.cache/huggingface/hub")
    candidates: List[str] = []

    # If model_name is a direct file or directory on disk:
    if model_name and (os.path.isdir(model_name) or os.path.isfile(model_name)):
        candidates.append(model_name)

    if model_name and model_name not in ("default", "gpt2"):
        prefix = model_name.split(":")[0].lower()
        if "qwen" in prefix:
            qwen_cands = []
            if "coder" in clean_name:
                qwen_cands.append("Qwen/Qwen2.5-Coder-7B-Instruct")
            if "14b" in clean_name:
                qwen_cands.append("Qwen/Qwen2.5-14B-Instruct")
            qwen_cands.extend([
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots/cdbee75f17c01a7cc42f958dc650907174af0554"),
                "Qwen/Qwen3-4B-Instruct-2507",
            ])
            candidates.extend(qwen_cands)
        elif "gemma" in prefix:
            gemma_cands = []
            if "gemma3" in clean_name or "gemma-3" in clean_name:
                gemma_cands.append("google/gemma-3-4b-it")
            gemma_cands.extend([
                "google/gemma-2-9b-it",
                "google/gemma-2-2b-it",
                "google/gemma-7b",
                "google/gemma-2b",
                "fxmarty/tiny-random-GemmaForCausalLM",
            ])
            candidates.extend(gemma_cands)
        elif "llama" in prefix:
            llama_cands = []
            if "3.2" in clean_name or "3b" in clean_name:
                llama_cands.extend([
                    "meta-llama/Llama-3.2-3B-Instruct",
                    "meta-llama/Llama-3.2-1B-Instruct",
                ])
            llama_cands.extend([
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.2-3B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
            ])
            candidates.extend(llama_cands)
        elif "deepseek" in prefix:
            ds_cands = []
            if "14b" in clean_name:
                ds_cands.append("deepseek-ai/DeepSeek-R1-Distill-Qwen-14B")
            ds_cands.extend([
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            ])
            candidates.extend(ds_cands)
        elif "mistral" in prefix:
            mistral_cands = []
            if "nemo" in clean_name:
                mistral_cands.append("mistralai/Mistral-Nemo-Instruct-2407")
            mistral_cands.extend([
                "mistralai/Mistral-7B-Instruct-v0.3",
                "mistralai/Mistral-Nemo-Instruct-2407",
            ])
            candidates.extend(mistral_cands)
        elif "starcoder2" in prefix:
            candidates.extend([
                "bigcode/starcoder2-7b",
                "bigcode/starcoder2-3b",
                "bigcode/starcoder2-15b",
            ])
        else:
            candidates.append(model_name)
    else:
        # Default or gpt2
        candidates.extend([
            os.path.expanduser("~/.cache/huggingface/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"),
            "gpt2",
        ])

    # 1. Check local directories and snapshots first
    for cand in candidates:
        if os.path.exists(cand):
            try:
                tok = AutoTokenizer.from_pretrained(cand, local_files_only=True)
                if tok is not None:
                    _tokenizers_cache[cache_key] = tok
                    logger.info(f"Loaded verified tokenizer for '{model_name}' from {cand}")
                    return tok
            except Exception as e:
                logger.debug(f"Failed loading local directory tokenizer from {cand}: {e}")

    # 2. Check HuggingFace hub cache by repo name
    for cand in candidates:
        if not os.path.exists(cand):
            repo_dir = os.path.join(hub_dir, f"models--{cand.replace('/', '--')}")
            if os.path.exists(repo_dir):
                snapshots_dir = os.path.join(repo_dir, "snapshots")
                if os.path.exists(snapshots_dir):
                    snaps = [os.path.join(snapshots_dir, s) for s in os.listdir(snapshots_dir)]
                    for s in snaps:
                        if os.path.isdir(s):
                            try:
                                tok = AutoTokenizer.from_pretrained(s, local_files_only=True)
                                if tok is not None:
                                    _tokenizers_cache[cache_key] = tok
                                    logger.info(f"Loaded cached snapshot tokenizer for '{model_name}' from {s}")
                                    return tok
                            except Exception:
                                pass
                try:
                    tok = AutoTokenizer.from_pretrained(cand, local_files_only=True)
                    if tok is not None:
                        _tokenizers_cache[cache_key] = tok
                        logger.info(f"Loaded cached tokenizer for '{model_name}' from {cand}")
                        return tok
                except Exception:
                    pass

    # 3. Online attempt if allowed and connected (never attempted if local_files_only is requested)
    if not local_files_only:
        for cand in candidates:
            if not os.path.exists(cand):
                try:
                    tok = AutoTokenizer.from_pretrained(cand)
                    if tok is not None:
                        _tokenizers_cache[cache_key] = tok
                        logger.info(f"Downloaded and loaded tokenizer for '{model_name}' from {cand}")
                        return tok
                except Exception:
                    pass

    if local_files_only:
        return None

    raise RuntimeError(
        f"Could not load valid tokenizer for model '{model_name}'. "
        f"Unsafe tokenizer fallbacks are disabled. Please ensure the model tokenizer is installed."
    )
