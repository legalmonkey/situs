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
from typing import Optional, List, Dict, Any, AsyncGenerator, Callable, Tuple
import httpx

logger = logging.getLogger("situs.inference_backend")

# Default Hugging Face cache root
HF_HUB_CACHE_DIR = pathlib.Path(os.path.expanduser("~/.cache/huggingface/hub"))
SITUS_CACHE_DIR = pathlib.Path(os.path.expanduser("~/.cache/situs/models"))

# Force disable Xet client dynamically to guarantee standard streaming HTTP downloads
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ["HF_HUB_DISABLE_XET"] = "1"
try:
    import huggingface_hub.constants as _hfc
    _hfc.HF_HUB_DISABLE_XET = True
except Exception:
    pass

# Canonical mapping from friendly short tags to official Hugging Face Hub repositories
HF_MODEL_REGISTRY_MAP: Dict[str, str] = {
    # Gemma series (mapped to official Google Gemma 3 4-bit Quantization-Aware Trained Q4_0 GGUF for native 6GB VRAM fit)
    "gemma3:4b": "google/gemma-3-4b-it-qat-q4_0-gguf",
    "gemma3": "google/gemma-3-4b-it-qat-q4_0-gguf",
    "gemma-3-4b": "google/gemma-3-4b-it-qat-q4_0-gguf",
    "gemma-3-4b-it": "google/gemma-3-4b-it-qat-q4_0-gguf",
    "gemma3:4b-qat": "google/gemma-3-4b-it-qat-q4_0-gguf",
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


def _prune_stale_hf_locks(repo_id: Optional[str] = None) -> None:
    """Removes stale .lock files created by interrupted previous download processes."""
    try:
        hub_cache = get_hf_cache_dir()
        locks_dir = hub_cache / ".locks"
        if not locks_dir.is_dir():
            return
        now = time.time()
        target_dirs = [locks_dir]
        if repo_id:
            safe_repo = f"models--{repo_id.replace('/', '--')}"
            repo_lock_dir = locks_dir / safe_repo
            if repo_lock_dir.is_dir():
                target_dirs.append(repo_lock_dir)

        for d in target_dirs:
            for lf in d.glob("*.lock"):
                try:
                    # If lock file hasn't been touched in > 30 seconds, prune it
                    if (now - lf.stat().st_mtime) > 30:
                        lf.unlink(missing_ok=True)
                except Exception:
                    pass
    except Exception:
        pass


def _is_snapshot_complete(snap_path: pathlib.Path) -> bool:
    """Verifies that all weight shards, tokenizer files, and config files exist and have non-zero size."""
    if not snap_path.is_dir():
        return False

    # 1. Check for single GGUF file (> 10MB, excluding mmproj vision projector)
    gguf_files = [f for f in snap_path.glob("*.gguf") if f.is_file() and f.stat().st_size > 10 * 1024 * 1024]
    llm_ggufs = [f for f in gguf_files if not f.name.lower().startswith("mmproj")]
    if llm_ggufs:
        return True

    # 2. Check for indexed multi-shard model (e.g. Gemma, LLaMA, Qwen multi-shard safetensors)
    index_file = snap_path / "model.safetensors.index.json"
    if not index_file.is_file():
        index_file = snap_path / "pytorch_model.bin.index.json"

    if index_file.is_file() and index_file.stat().st_size > 0:
        # Indexed models strictly require config and tokenizer files
        cfg_file = snap_path / "config.json"
        if not (cfg_file.is_file() and cfg_file.stat().st_size > 0):
            return False

        has_tokenizer = any(
            (snap_path / tok_name).is_file() and (snap_path / tok_name).stat().st_size > 0
            for tok_name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
        )
        if not has_tokenizer:
            return False

        try:
            with open(index_file, "r", encoding="utf-8") as f:
                idx_data = json.load(f)
            weight_map = idx_data.get("weight_map", {})
            if not weight_map:
                return False
            required_shards = set(weight_map.values())
            for shard in required_shards:
                shard_path = snap_path / shard
                if not (shard_path.is_file() and shard_path.stat().st_size > 1024 * 1024):
                    return False
            return True
        except Exception:
            return False

    # 3. If no index file, check for at least one valid safetensors or bin weight file
    weights = [
        f for f in (list(snap_path.glob("*.safetensors")) + list(snap_path.glob("*.bin")))
        if f.is_file() and f.stat().st_size > 0
    ]
    return len(weights) > 0


def is_model_downloaded(model_name: str) -> bool:
    """Checks whether the requested model weights and tokenizer are completely cached locally."""
    if not model_name:
        return False
    # 1. Direct local file or directory
    if os.path.isdir(model_name) or os.path.isfile(model_name):
        return True

    repo_id = normalize_model_name(model_name)
    hub_cache = get_hf_cache_dir()

    # 2. Check HuggingFace hub cache by repo directory
    repo_dir = hub_cache / f"models--{repo_id.replace('/', '--')}"
    if repo_dir.is_dir():
        snapshots_dir = repo_dir / "snapshots"
        if snapshots_dir.is_dir():
            for snap in snapshots_dir.iterdir():
                if snap.is_dir() and _is_snapshot_complete(snap):
                    return True

    # 3. Check local weight files (GGUF, safetensors, bin)
    try:
        from situs.worker.loader import find_local_weight_files
        wfiles = find_local_weight_files(model_name)
        def _is_valid_wf(wf: str) -> bool:
            if not os.path.isfile(wf):
                return False
            if _is_snapshot_complete(pathlib.Path(wf).parent):
                return True
            if os.path.getsize(wf) > 10 * 1024 * 1024:
                if wf.lower().endswith(".gguf"):
                    return True
                try:
                    with open(wf, "rb") as f:
                        return f.read(4) == b"GGUF"
                except Exception:
                    pass
            return False

        if wfiles and any(_is_valid_wf(wf) for wf in wfiles):
            return True
    except Exception:
        pass

    # 4. Check Situs local cache
    safe_name = repo_id.replace("/", "_").replace(":", "_")
    situs_model_dir = SITUS_CACHE_DIR / safe_name
    if situs_model_dir.is_dir() and _is_snapshot_complete(situs_model_dir):
        return True

    return False


def _make_progress_tqdm(progress_cb: Callable[[int], None]):
    """Creates a custom tqdm class that feeds chunk-level byte progress from hf_hub_download."""
    from tqdm import tqdm

    class _FileProgressTqdm(tqdm):
        def update(self, n=1):
            super().update(n)
            try:
                # Pass absolute byte progress (self.n) from tqdm
                cur_pos = getattr(self, "n", n)
                progress_cb(cur_pos)
            except Exception:
                pass

    return _FileProgressTqdm


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

    # Enforce standard streaming HTTP download without Xet chunk retry inflation
    try:
        import huggingface_hub.constants as _hfc
        _hfc.HF_HUB_DISABLE_XET = True
    except Exception:
        pass

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

    _prune_stale_hf_locks(repo_id)
    os.environ["HF_HUB_DISABLE_XET"] = "1"

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

        # Check if repository provides GGUF quantizations
        gguf_siblings = [
            s for s in siblings
            if getattr(s, "rfilename", "").endswith(".gguf") and not getattr(s, "rfilename", "").startswith(".")
        ]
        if gguf_siblings:
            # Exclude mmproj (vision projector) unless specifically requested
            llm_ggufs = [s for s in gguf_siblings if not getattr(s, "rfilename", "").lower().startswith("mmproj")]
            if not llm_ggufs:
                llm_ggufs = gguf_siblings

            # Determine best quantization file
            clean_tag = model_name.split(":")[-1].lower() if ":" in model_name else ""
            known_quant_tags = {"q4_k_m", "q4_k_s", "q4_0", "q4_1", "q5_k_m", "q5_0", "q6_k", "q8_0", "bf16", "fp16", "q4", "q8"}
            selected_s = None
            if clean_tag in known_quant_tags:
                for s in llm_ggufs:
                    if clean_tag in getattr(s, "rfilename", "").lower():
                        selected_s = s
                        break

            # Prioritize 4-bit standard: Q4_K_M -> Q4_0 -> Q4_K_S -> first under 4GB -> first
            if selected_s is None:
                priority_tags = ("q4_k_m", "q4_0", "q4_k_s", "q4_1", "q5_k_m", "q8_0")
                for ptag in priority_tags:
                    for s in llm_ggufs:
                        if ptag in getattr(s, "rfilename", "").lower():
                            selected_s = s
                            break
                    if selected_s is not None:
                        break

            if selected_s is None:
                selected_s = llm_ggufs[0]

            # In a GGUF repo, ONLY download the selected GGUF file (plus any accompanying config if available)
            target_files = [(getattr(selected_s, "rfilename", ""), getattr(selected_s, "size", 0) or 0)]
            total_repo_bytes = target_files[0][1]
            for s in siblings:
                rfn = getattr(s, "rfilename", "")
                if rfn in ("config.json", "generation_config.json") or (rfn.endswith((".json", ".txt")) and not rfn.startswith(".")):
                    sz = getattr(s, "size", 0) or 0
                    target_files.append((rfn, sz))
                    total_repo_bytes += sz
        else:
            # Filter required files: config, tokenizer, and safetensors / bin weights
            target_files = []
            total_repo_bytes = 0
            for s in siblings:
                rfilename = getattr(s, "rfilename", "")
                size = getattr(s, "size", 0) or 0
                if any(
                    rfilename.endswith(ext)
                    for ext in (".json", ".safetensors", ".bin", ".model", ".txt")
                ) and not rfilename.startswith("."):
                    target_files.append((rfilename, size))
                    total_repo_bytes += size

            if not target_files:
                target_files = [("config.json", 1024)]

        # Sort files so configs and tokenizers are downloaded FIRST, followed by large weights
        def _file_priority_key(item: Tuple[str, int]) -> Tuple[int, str]:
            fn = item[0].lower()
            if fn in ("config.json", "generation_config.json") or "index.json" in fn:
                return (0, fn)
            if "tokenizer" in fn or fn.endswith(".model") or fn in ("special_tokens_map.json", "chat_template.json", "added_tokens.json"):
                return (1, fn)
            if fn.endswith((".json", ".txt")):
                return (2, fn)
            return (3, fn)

        target_files.sort(key=_file_priority_key)

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

        # Sequentially fetch and stream files into Hugging Face Hub cache with real-time byte telemetry
        total_completed = 0
        start_t = time.time()
        last_speed_t = start_t
        last_speed_bytes = 0
        speed_mb = 0.0

        for idx, (filename, file_size) in enumerate(target_files):
            # Short status description
            short_fn = filename if len(filename) <= 24 else "..." + filename[-21:]
            stage_str = f"Stage 3/3: Streaming [{idx+1}/{len(target_files)}] {short_fn}"

            file_bytes = [0]
            def _on_chunk(cur_file_pos: int):
                # Clamp to file_size so chunk repetitions/retries never exceed declared file size
                file_bytes[0] = min(file_size, max(0, cur_file_pos))

            tracker_cls = _make_progress_tqdm(_on_chunk)

            # Immediately yield start of this file so UI reflects new file name right away
            cur_total = min(total_repo_bytes, total_completed)
            pct = min(99.0, round((cur_total / max(1, total_repo_bytes)) * 100.0, 1)) if total_repo_bytes > 0 else 50.0
            rem_bytes = max(0, total_repo_bytes - cur_total)
            eta_s = int(rem_bytes / max(1000, speed_mb * 1e6)) if speed_mb > 0 else 0

            yield {
                "type": "model_loading",
                "model": repo_id,
                "status": "downloading",
                "stage": stage_str,
                "percent": pct,
                "completed_bytes": cur_total,
                "total_bytes": total_repo_bytes,
                "speed_mb": speed_mb,
                "eta_s": eta_s,
                "bar": make_progress_bar(pct, width=22),
            }

            def _dl():
                _prune_stale_hf_locks(repo_id)
                return hf_hub_download(repo_id=repo_id, filename=filename, token=token, tqdm_class=tracker_cls)

            loop = asyncio.get_running_loop()
            dl_future = loop.run_in_executor(None, _dl)

            # Stream real-time progress while file is downloading
            while not dl_future.done():
                await asyncio.sleep(0.25)
                cur_total = min(total_repo_bytes, total_completed + file_bytes[0])
                now = time.time()
                dt = now - last_speed_t
                if dt >= 0.8:
                    speed_mb = round(((cur_total - last_speed_bytes) / dt) / 1e6, 1)
                    last_speed_t = now
                    last_speed_bytes = cur_total

                pct = min(99.0, round((cur_total / max(1, total_repo_bytes)) * 100.0, 1)) if total_repo_bytes > 0 else 50.0
                rem_bytes = max(0, total_repo_bytes - cur_total)
                eta_s = int(rem_bytes / max(1000, speed_mb * 1e6)) if speed_mb > 0 else 0

                yield {
                    "type": "model_loading",
                    "model": repo_id,
                    "status": "downloading",
                    "stage": stage_str,
                    "percent": pct,
                    "completed_bytes": cur_total,
                    "total_bytes": total_repo_bytes,
                    "speed_mb": speed_mb,
                    "eta_s": eta_s,
                    "bar": make_progress_bar(pct, width=22),
                }

            # Retrieve result (or raise exception if download failed)
            dl_path = await dl_future
            actual_size = os.path.getsize(dl_path) if os.path.isfile(dl_path) else file_size
            total_completed += min(file_size, actual_size)

            cur_total = min(total_repo_bytes, total_completed)
            pct = min(99.0, round((cur_total / max(1, total_repo_bytes)) * 100.0, 1)) if total_repo_bytes > 0 else 50.0
            rem_bytes = max(0, total_repo_bytes - cur_total)
            eta_s = int(rem_bytes / max(1000, speed_mb * 1e6)) if speed_mb > 0 else 0

            yield {
                "type": "model_loading",
                "model": repo_id,
                "status": "downloading",
                "stage": stage_str,
                "percent": pct,
                "completed_bytes": cur_total,
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


def format_stage_name(stage_name: str) -> str:
    """Format download or pipeline stage status for UI display."""
    s = (stage_name or "").lower()
    if "writing" in s or "allocat" in s:
        return "Allocating Local Storage"
    if "manifest" in s:
        return "Resolving Model Manifest"
    if "download" in s or "stream" in s:
        return "Streaming Model Weights"
    if "verify" in s:
        return "Verifying Checksum Integrity"
    if "success" in s or "ready" in s:
        return "Ready for Inference"
    return stage_name.capitalize() if stage_name else "Processing"


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


def extract_latest_user_query(prompt_str: str) -> str:
    """Extracts the latest user query from a multi-turn chat prompt."""
    lines = prompt_str.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line.lower().startswith("user:"):
            return line[5:].strip()
    return prompt_str.strip()

