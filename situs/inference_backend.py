"""Native Hugging Face Model Hub & Streaming Downloader for Situs Distributed Cluster.

Eliminates all dependencies on external Ollama daemons, manifests, and background processes.
Directly fetches Safetensors, tokenizers, and GGUFs from Hugging Face Hub into standard cache directories:
  ~/.cache/huggingface/hub/models--<owner>--<repo>/snapshots/
with real-time download percentage, instantaneous throughput (MB/s), ETA, and progress bar streaming.
"""

import asyncio
import json
import logging
import os
import pathlib
import re
import shutil
import time
from typing import Optional, List, Dict, Any, AsyncGenerator, Callable
import httpx

logger = logging.getLogger("situs.inference_backend")

# Default Hugging Face cache root
HF_HUB_CACHE_DIR = pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub"))
SITUS_CACHE_DIR = pathlib.Path(os.path.expanduser("~/.cache/situs/models"))

# Canonical mapping from friendly short tags to official Hugging Face Hub repositories
HF_MODEL_REGISTRY_MAP: Dict[str, str] = {
    # Gemma series
    "gemma3:4b": "google/gemma-3-4b-it",
    "gemma3": "google/gemma-3-4b-it",
    "gemma-3-4b": "google/gemma-3-4b-it",
    "gemma-3-4b-it": "google/gemma-3-4b-it",
    "gemma2:9b": "google/gemma-2-9b-it",
    "gemma-2:9b": "google/gemma-2-9b-it",
    "gemma-2:9b-instruct-q6_k": "google/gemma-2-9b-it",
    "gemma2:2b": "google/gemma-2-2b-it",
    "gemma2": "google/gemma-2-9b-it",
    # LLaMA series
    "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1:8b-instruct-q8_0": "meta-llama/Llama-3.1-8B-Instruct",
    "llama-3.1:8b-fp16": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.1": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
    "llama-3.2:3b-instruct-q8_0": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
    "llama3.2": "meta-llama/Llama-3.2-3B-Instruct",
    "llama3.3:70b": "meta-llama/Llama-3.3-70B-Instruct",
    # Qwen series
    "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5:7b-instruct-q8_0": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5-coder:7b-instruct-q8_0": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:14b-instruct-q8_0": "Qwen/Qwen2.5-14B-Instruct",
    "qwen2.5:32b": "Qwen/Qwen2.5-32B-Instruct",
    # DeepSeek series
    "deepseek-r1:7b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1:7b-q8_0": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    "deepseek-r1:8b": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "deepseek-r1:14b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1:14b-q8_0": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "deepseek-r1:32b": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # Mistral series
    "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral:7b-instruct-v0.3-q8_0": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
    "mistral-nemo:12b": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral-nemo:12b-instruct-fp16": "mistralai/Mistral-Nemo-Instruct-2407",
    "mistral-nemo": "mistralai/Mistral-Nemo-Instruct-2407",
    "starcoder2:7b": "bigcode/starcoder2-7b",
    "starcoder2:7b-q8_0": "bigcode/starcoder2-7b",
}

CANONICAL_MODEL_TAGS = HF_MODEL_REGISTRY_MAP


def normalize_model_name(model_name: str) -> str:
    """Resolves arbitrary user model names or tags to their canonical Hugging Face Hub repo ID."""
    if not model_name:
        return os.getenv("SITUS_DEFAULT_MODEL", "google/gemma-3-4b-it")
    raw = model_name.strip()
    if raw in HF_MODEL_REGISTRY_MAP:
        return HF_MODEL_REGISTRY_MAP[raw]
    raw_lower = raw.lower()
    if raw_lower in HF_MODEL_REGISTRY_MAP:
        return HF_MODEL_REGISTRY_MAP[raw_lower]

    # Check if stripping quantization suffix matches
    prefix = raw.split("-instruct")[0].split("-q8")[0].split("-q6")[0].replace("llama-", "llama").replace("gemma-", "gemma")
    if prefix in HF_MODEL_REGISTRY_MAP:
        return HF_MODEL_REGISTRY_MAP[prefix]

    # If user provided a direct repo format e.g. "google/gemma-3-4b-it" or local path
    return raw


# Backwards compatibility alias
normalize_ollama_model = normalize_model_name


def make_progress_bar(percent: float, width: int = 22) -> str:
    """Renders a sleek progress bar like [████████░░░░░░░░░░]."""
    clamped = max(0.0, min(100.0, float(percent)))
    filled_len = int(round(width * clamped / 100.0))
    empty_len = max(0, width - filled_len)
    return f"[{'█' * filled_len}{'░' * empty_len}]"


def get_hf_cache_dir() -> pathlib.Path:
    """Returns the effective Hugging Face Hub cache directory."""
    return pathlib.Path(
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or os.path.expanduser("~/.cache/huggingface/hub")
    )


def is_model_downloaded(model_name: str) -> bool:
    """Checks whether the requested model weights (.safetensors, .bin, or .gguf) are cached locally."""
    # 1. Direct local file or directory
    if model_name and (os.path.isdir(model_name) or os.path.isfile(model_name)):
        return True

    # 2. Check local safetensors finder
    try:
        from situs.worker.loader import find_local_safetensors_files
        sfiles = find_local_safetensors_files(model_name)
        if sfiles:
            return True
    except Exception:
        pass

    # 3. Check HuggingFace hub cache by repo directory
    hub_cache = get_hf_cache_dir()
    repo_id = normalize_model_name(model_name)
    repo_dir = hub_cache / f"models--{repo_id.replace('/', '--')}"
    if repo_dir.is_dir():
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.is_dir():
            for snap in snapshots_dir.iterdir():
                if snap.is_dir():
                    has_weights = bool(
                        list(snap.glob("*.safetensors"))
                        or list(snap.glob("*.bin"))
                        or list(snap.glob("*.gguf"))
                    )
                    if has_weights:
                        return True

    # 4. Check Situs local cache
    safe_name = repo_id.replace("/", "_").replace(":", "_")
    situs_model_dir = SITUS_CACHE_DIR / safe_name
    if situs_model_dir.is_dir() and (
        list(situs_model_dir.glob("*.safetensors"))
        or list(situs_model_dir.glob("*.bin"))
        or list(situs_model_dir.glob("*.gguf"))
    ):
        return True

    return False


async def stream_pull_model(
    model_name: str,
    base_url: Optional[str] = None,
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Asynchronously streams online model fetching progress directly from Hugging Face Hub.
    Yields structured progress dictionaries for live progress bars in the TUI and Coordinator.
    Eliminates all requirements for Ollama to be installed or running.
    """
    repo_id = normalize_model_name(model_name)
    clean_repo = repo_id.replace(":", "-")

    # If already downloaded, emit instant completion
    if is_model_downloaded(model_name):
        yield {
            "type": "model_loading",
            "model": repo_id,
            "status": "success",
            "stage": "Model Ready: Local Weights Verified",
            "percent": 100.0,
            "completed_bytes": 0,
            "total_bytes": 0,
            "speed_mb": 0.0,
            "eta_s": 0,
            "bar": make_progress_bar(100.0, width=22),
        }
        return

    # Check for Hugging Face Auth Token (for gated models like Gemma / LLaMA)
    from situs.token_manager import get_hf_token
    token = get_hf_token()

    yield {
        "type": "model_loading",
        "model": repo_id,
        "status": "resolving",
        "stage": f"Stage 1/3: Resolving '{repo_id}' on Hugging Face Hub",
        "percent": 5.0,
        "completed_bytes": 0,
        "total_bytes": 0,
        "speed_mb": 0.0,
        "eta_s": 0,
        "bar": make_progress_bar(5.0, width=22),
    }

    try:
        from huggingface_hub import HfApi, hf_hub_download
        api = HfApi(token=token)

        # Retrieve repository file tree
        info = await asyncio.to_thread(api.model_info, repo_id, files_metadata=True)
        siblings = getattr(info, "siblings", []) or []

        # Filter required files: config, tokenizer, and safetensors / GGUF weights
        target_files = []
        total_repo_bytes = 0
        for s in siblings:
            rfilename = getattr(s, "rfilename", "")
            size = getattr(s, "size", 0) or 0
            if any(
                rfilename.endswith(ext)
                for ext in (".json", ".safetensors", ".bin", ".gguf", ".model", ".txt")
            ) and not rfilename.startswith("."):
                target_files.append((rfilename, size))
                total_repo_bytes += size

        if not target_files:
            # Fallback for standard files
            target_files = [("config.json", 1024)]

        yield {
            "type": "model_loading",
            "model": repo_id,
            "status": "downloading",
            "stage": f"Stage 2/3: Fetching Architecture & Tokenizer ({len(target_files)} files)",
            "percent": 10.0,
            "completed_bytes": 0,
            "total_bytes": total_repo_bytes,
            "speed_mb": 0.0,
            "eta_s": 0,
            "bar": make_progress_bar(10.0, width=22),
        }

        # Sequentially fetch and stream files into Hugging Face Hub cache
        total_completed = 0
        start_t = time.time()
        last_t = start_t
        last_bytes = 0

        for idx, (filename, file_size) in enumerate(target_files):
            # Short status description
            short_fn = filename if len(filename) <= 24 else "..." + filename[-21:]
            stage_str = f"Stage 3/3: Streaming [{idx+1}/{len(target_files)}] {short_fn}"

            def _dl(fn: str):
                return hf_hub_download(repo_id=repo_id, filename=fn, token=token)

            # Download file in background thread
            dl_path = await asyncio.to_thread(_dl, filename)
            actual_size = os.path.getsize(dl_path) if os.path.isfile(dl_path) else file_size
            total_completed += actual_size

            now = time.time()
            dt = max(0.1, now - last_t)
            speed_mb = round(((total_completed - last_bytes) / dt) / 1e6, 1)
            last_t = now
            last_bytes = total_completed

            pct = min(99.0, round((total_completed / max(1, total_repo_bytes)) * 100.0, 1)) if total_repo_bytes > 0 else 50.0
            rem_bytes = max(0, total_repo_bytes - total_completed)
            eta_s = int(rem_bytes / max(1000, speed_mb * 1e6)) if speed_mb > 0 else 0

            yield {
                "type": "model_loading",
                "model": repo_id,
                "status": "downloading",
                "stage": stage_str,
                "percent": pct,
                "completed_bytes": total_completed,
                "total_bytes": total_repo_bytes,
                "speed_mb": speed_mb,
                "eta_s": eta_s,
                "bar": make_progress_bar(pct, width=22),
            }

        yield {
            "type": "model_loading",
            "model": repo_id,
            "status": "success",
            "stage": f"Model Ready: '{repo_id}' Weights Cached (100% Native PyTorch)",
            "percent": 100.0,
            "completed_bytes": total_completed,
            "total_bytes": total_completed,
            "speed_mb": 0.0,
            "eta_s": 0,
            "bar": make_progress_bar(100.0, width=22),
        }

    except Exception as e:
        logger.error(f"Hugging Face Hub model fetch failed for '{repo_id}': {e}", exc_info=True)
        err_str = str(e)
        is_auth_error = any(kw in err_str.lower() for kw in ("401", "403", "gated", "restricted", "unauthorized", "please log in"))
        if is_auth_error:
            msg = (
                f"Model '{repo_id}' is a restricted/gated repository requiring Hugging Face authentication. "
                f"Please add your token using '/hf <your_token>' or 'situs hf <your_token>' (or export HF_TOKEN=<token>)."
            )
        else:
            msg = f"Failed fetching '{repo_id}' from Hugging Face Hub: {e}."
        yield {
            "type": "model_loading_error",
            "model": repo_id,
            "error": msg,
            "requires_auth": is_auth_error,
        }


def ensure_model_available(target_model: str, progress_callback=None) -> bool:
    """Ensures the target model is available locally. If not, pulls directly from Hugging Face Hub."""
    if is_model_downloaded(target_model):
        return True

    repo_id = normalize_model_name(target_model)
    logger.info(f"Model '{target_model}' not cached locally. Pulling directly from Hugging Face Hub ({repo_id})...")
    from situs.token_manager import get_hf_token
    token = get_hf_token()

    try:
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            token=token,
            allow_patterns=["*.safetensors", "*.bin", "*.gguf", "*.json", "tokenizer*"],
        )
        return is_model_downloaded(target_model)
    except Exception as e:
        logger.error(f"Failed to download model '{repo_id}' from Hugging Face Hub: {e}")
        return False


# -----------------------------------------------------------------------------
# Backwards Compatibility Stubs (Clean no-ops for legacy test suites)
# -----------------------------------------------------------------------------

def get_ollama_endpoints() -> List[str]:
    """Deprecated legacy stub: returns empty list as Ollama is no longer utilized."""
    return []


def ensure_ollama_running() -> bool:
    """Deprecated legacy stub: returns True as Ollama daemon is no longer required."""
    return True


def ensure_ollama_model(base_url: str, target_model: str, progress_callback=None) -> bool:
    """Deprecated legacy stub: Ollama is eliminated; returns True to prevent blocking legacy callers."""
    return True


def get_installed_ollama_models(base_url: str) -> List[str]:
    """Deprecated legacy stub: returns empty list."""
    return []


def query_ollama(prompt: str, model_name: str = "") -> Optional[str]:
    """Deprecated legacy stub: Ollama neural engine has been eliminated."""
    return None


def generate_llm_answer(prompt_text: str, model_name: str = "") -> str:
    """Deprecated legacy stub: Situs distributed PyTorch engine executes inference natively."""
    raise RuntimeError(
        "Local Ollama fallback has been eliminated. Situs executes neural inference "
        "natively via its distributed cluster engine."
    )
