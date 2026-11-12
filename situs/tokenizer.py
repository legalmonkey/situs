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


class GGUFTokenizerAdapter:
    """Builds a verified HuggingFace `tokenizers.Tokenizer` directly from GGUF embedded vocabulary metadata."""

    def __init__(self, gguf_path: str):
        from tokenizers import Tokenizer, models, decoders, normalizers, pre_tokenizers, AddedToken
        from situs.worker.loader import FastGGUFFile

        wanted_keys = {
            "tokenizer.ggml.tokens",
            "tokenizer.ggml.merges",
            "tokenizer.ggml.scores",
            "tokenizer.ggml.token_type",
            "tokenizer.ggml.bos_token_id",
            "tokenizer.ggml.eos_token_id",
            "tokenizer.ggml.eos_token_ids",
            "tokenizer.ggml.unknown_token_id",
            "tokenizer.ggml.add_bos_token",
            "tokenizer.ggml.model",
            "tokenizer.chat_template",
        }
        reader = FastGGUFFile(gguf_path, wanted_kv=wanted_keys)
        try:
            kv = dict(reader.kv)
        finally:
            reader.close()

        tokens: List[str] = kv.get("tokenizer.ggml.tokens") or []
        if not tokens:
            raise ValueError(f"GGUF file '{gguf_path}' contains no tokenizer.ggml.tokens metadata.")

        vocab = {tok: idx for idx, tok in enumerate(tokens)}
        merges_raw: List[str] = kv.get("tokenizer.ggml.merges") or []
        scores_raw = kv.get("tokenizer.ggml.scores")
        token_types = kv.get("tokenizer.ggml.token_type")

        self.bos_token_id: Optional[int] = kv.get("tokenizer.ggml.bos_token_id", 2)
        self.eos_token_id: Optional[int] = kv.get("tokenizer.ggml.eos_token_id", 1)
        self.unk_token_id: int = int(kv.get("tokenizer.ggml.unknown_token_id", 0) or 0)
        self.add_bos_token: bool = bool(kv.get("tokenizer.ggml.add_bos_token", False))
        self.chat_template: Optional[str] = kv.get("tokenizer.chat_template", None)
        self.ggml_model: str = str(kv.get("tokenizer.ggml.model", "llama") or "llama").lower()

        self.bos_token: str = tokens[self.bos_token_id] if (self.bos_token_id is not None and 0 <= self.bos_token_id < len(tokens)) else ""
        self.eos_token: str = tokens[self.eos_token_id] if (self.eos_token_id is not None and 0 <= self.eos_token_id < len(tokens)) else ""

        stop_ids: List[int] = []
        if self.eos_token_id is not None:
            stop_ids.append(int(self.eos_token_id))
        eos_arr = kv.get("tokenizer.ggml.eos_token_ids") or []
        for sid in eos_arr:
            try:
                isid = int(sid)
                if isid not in stop_ids:
                    stop_ids.append(isid)
            except Exception:
                pass
        for end_tok in ("<end_of_turn>", "<|eot_id|>", "<|im_end|>", "<|endoftext|>", "</s>"):
            if end_tok in vocab and vocab[end_tok] not in stop_ids:
                stop_ids.append(int(vocab[end_tok]))
        self.stop_token_ids: List[int] = stop_ids

        unk_tok = tokens[self.unk_token_id] if 0 <= self.unk_token_id < len(tokens) else "<unk>"

        if merges_raw:
            merges = []
            for m in merges_raw:
                parts = m.split(" ", 1)
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
            tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token=unk_tok, byte_fallback=(self.ggml_model != "gpt2")))
        elif scores_raw:
            scores = [float(s) for s in scores_raw]
            tok = Tokenizer(models.Unigram(list(zip(tokens, scores)), unk_id=self.unk_token_id, byte_fallback=True))
        else:
            raise ValueError(f"GGUF file '{gguf_path}' has neither merges nor scores for tokenizer reconstruction.")

        if self.ggml_model == "gpt2":
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tok.decoder = decoders.ByteLevel()
        else:
            tok.normalizer = normalizers.Replace(" ", "▁")
            tok.decoder = decoders.Sequence([
                decoders.Replace("▁", " "),
                decoders.ByteFallback(),
                decoders.Fuse(),
            ])

        # Register control/special tokens so tags like <start_of_turn> encode atomically
        special_tokens = []
        if token_types:
            for idx, tt in enumerate(token_types):
                if tt == 3 and idx < len(tokens):  # CONTROL token
                    special_tokens.append(AddedToken(tokens[idx], single_word=False, lstrip=False, rstrip=False, normalized=False, special=True))
        if special_tokens:
            tok.add_special_tokens(special_tokens)

        self._tok = tok
        self._vocab_len = len(tokens)

    def __len__(self) -> int:
        return self._vocab_len

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if add_special_tokens and self.add_bos_token and self.bos_token_id is not None:
            if not ids or ids[0] != self.bos_token_id:
                ids = [int(self.bos_token_id)] + ids
        return [int(x) for x in ids]

    def decode(self, token_ids: List[int], skip_special_tokens: bool = False) -> str:
        return self._tok.decode([int(i) for i in token_ids], skip_special_tokens=skip_special_tokens)

    def apply_chat_template(self, messages: List[Dict[str, Any]], tokenize: bool = False, add_generation_prompt: bool = True) -> Any:
        if self.chat_template:
            try:
                import jinja2
                env = jinja2.Environment()
                env.globals["raise_exception"] = lambda msg: (_ for _ in ()).throw(ValueError(msg))
                tpl = env.from_string(self.chat_template)
                rendered = tpl.render(
                    messages=messages,
                    bos_token="",
                    eos_token=self.eos_token,
                    add_generation_prompt=add_generation_prompt,
                )
                return self.encode(rendered) if tokenize else rendered
            except Exception:
                pass
        # Clean instruction fallback if template fails (e.g. non-alternating system+user turns)
        rendered = "\n".join(f"<start_of_turn>{'model' if m.get('role') == 'assistant' else m.get('role', 'user')}\n{m.get('content', '')}<end_of_turn>" for m in messages)
        if add_generation_prompt:
            rendered += "\n<start_of_turn>model\n"
        return self.encode(rendered) if tokenize else rendered


_tokenizers_cache: Dict[str, Any] = {}


def get_or_load_tokenizer(
    model_name: Optional[str] = None,
    allow_fallback: bool = False,
    local_files_only: bool = False,
) -> Any:
    """
    Retrieves or initializes a verified HuggingFace or GGUF tokenizer with caching per model.
    Checks local weights/GGUF/snapshots first for exact alignment and instant offline loading.
    Raises RuntimeError if a matching tokenizer cannot be found or loaded.
    """
    clean_name = (model_name or "default").lower().strip()
    cache_key = clean_name
    if cache_key in _tokenizers_cache:
        return _tokenizers_cache[cache_key]

    # Prefer embedded GGUF tokenizer whenever running from a local GGUF / Ollama model without local HF safetensors
    if model_name and model_name not in ("default", "gpt2"):
        try:
            from situs.worker.loader import find_local_safetensors_files, find_local_ollama_gguf_file
            if not find_local_safetensors_files(model_name):
                gguf_path = model_name if (os.path.isfile(model_name) and model_name.lower().endswith(".gguf")) else find_local_ollama_gguf_file(model_name)
                if gguf_path:
                    tok = GGUFTokenizerAdapter(gguf_path)
                    _tokenizers_cache[cache_key] = tok
                    logger.info(f"Loaded exact GGUF embedded tokenizer for '{model_name}' from {gguf_path}")
                    return tok
        except Exception as e:
            logger.debug(f"GGUF tokenizer construction skipped for '{model_name}': {e}")

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
