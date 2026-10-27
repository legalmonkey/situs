"""Worker registry, health tracking, and dynamic programming layer partition scheduler."""

import time
import math
import logging
import functools
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, model_validator
from situs.protocol.messages import WorkerRegister, AssignLayers

logger = logging.getLogger("situs.coordinator.registry")


class WorkerNode(BaseModel):
    worker_id: str
    gpu_name: str
    total_vram_bytes: int
    free_vram_bytes: int
    used_vram_bytes: int = 0
    device_type: str
    last_heartbeat: float
    is_alive: bool = True
    assigned_layer_start: Optional[int] = None
    assigned_layer_end: Optional[int] = None
    is_first_stage: bool = False
    is_last_stage: bool = False
    next_worker_id: Optional[str] = None
    prev_worker_id: Optional[str] = None
    first_worker_id: Optional[str] = None
    compute_speed: Optional[float] = None
    network_latency_ms: Optional[float] = None
    compute_speed_status: str = "unknown"
    network_latency_status: str = "unknown"

    @model_validator(mode="after")
    def _validate_telemetry_status(self):
        if self.compute_speed is not None and self.compute_speed > 0:
            self.compute_speed_status = "measured"
        else:
            self.compute_speed_status = "unknown"

        if self.network_latency_ms is not None and self.network_latency_ms >= 0:
            self.network_latency_status = "measured"
        else:
            self.network_latency_status = "unknown"
        return self


def _parse_verified_config_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Dynamically parses and validates a model config dict without arbitrary fallback defaults."""
    total_layers = data.get("num_hidden_layers") or data.get("n_layer") or data.get("num_layers") or data.get("total_layers")
    hidden_dim = data.get("hidden_size") or data.get("n_embd") or data.get("hidden_dim")
    vocab_size = data.get("vocab_size")
    num_heads = data.get("num_attention_heads") or data.get("n_head") or data.get("num_heads")
    num_kv_heads = data.get("num_key_value_heads") or data.get("num_kv_heads") or num_heads
    intermediate_size = data.get("intermediate_size") or data.get("n_inner")
    arch = data.get("model_type") or data.get("architecture")
    if isinstance(data.get("architectures"), list) and data["architectures"]:
        arch = data["architectures"][0]

    def _to_int(val):
        try:
            return int(val) if val is not None else None
        except (ValueError, TypeError):
            return None

    total_layers = _to_int(total_layers)
    hidden_dim = _to_int(hidden_dim)
    vocab_size = _to_int(vocab_size)
    num_heads = _to_int(num_heads)
    num_kv_heads = _to_int(num_kv_heads) or num_heads
    intermediate_size = _to_int(intermediate_size)

    missing = []
    if total_layers is None or total_layers <= 0:
        missing.append("total_layers")
    if hidden_dim is None or hidden_dim <= 0:
        missing.append("hidden_dim")
    if vocab_size is None or vocab_size <= 0:
        missing.append("vocab_size")
    if num_heads is None or num_heads <= 0:
        missing.append("num_heads")
    if intermediate_size is None or intermediate_size <= 0:
        missing.append("intermediate_size")

    if missing:
        raise ValueError(f"Verified model config is missing required attributes: {missing}")

    head_dim = _to_int(data.get("head_dim") or data.get("attention_head_dim") or data.get("key_length"))
    if head_dim is None and hidden_dim and num_heads:
        head_dim = max(1, hidden_dim // max(1, num_heads))

    context_length = _to_int(
        data.get("max_position_embeddings")
        or data.get("context_length")
        or data.get("seq_length")
        or data.get("max_sequence_length")
        or data.get("n_positions")
    )
    torch_dtype = data.get("torch_dtype") or data.get("precision") or data.get("dtype")

    return {
        "total_layers": total_layers,
        "hidden_dim": hidden_dim,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads if num_kv_heads else num_heads,
        "head_dim": head_dim,
        "architecture": str(arch) if arch is not None else None,
        "context_length": context_length,
        "max_position_embeddings": context_length,
        "torch_dtype": torch_dtype,
    }


@functools.lru_cache(maxsize=64)
def resolve_model_metadata(model_name: str) -> Dict[str, Any]:
    """
    Dynamically discovers model configuration (total layers, hidden dimensions, heads, vocab).
    Derives values directly from verified model configs (AutoConfig or local config files),
    cached weights, or Ollama daemon.
    Explicitly rejects unsupported models and disallows arbitrary fallback defaults.
    """
    import os
    import json
    import pathlib

    def _apply_inferred_precision(res: Dict[str, Any]) -> Dict[str, Any]:
        model_name_lower = model_name.lower()
        tag = model_name_lower.split(":", 1)[1] if ":" in model_name_lower else model_name_lower
        inferred = None
        if any(q in tag for q in ("q8", "int8", "fp8", "8bit")):
            inferred = "int8"
        elif any(q in tag for q in ("q6", "int6", "6bit")):
            inferred = "int6"
        elif any(q in tag for q in ("q5", "int5", "5bit")):
            inferred = "int5"
        elif any(q in tag for q in ("q4", "int4", "4bit")):
            inferred = "int4"
        elif any(q in tag for q in ("fp16", "bf16", "16bit", "half")):
            inferred = "bfloat16"
        elif any(q in tag for q in ("fp32", "32bit")):
            inferred = "float32"

        if inferred:
            res["torch_dtype"] = inferred
            res["precision"] = inferred
        return res

    # 1. Direct filesystem path to config file or model directory
    if os.path.isfile(model_name):
        try:
            with open(model_name, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _apply_inferred_precision(_parse_verified_config_dict(data))
        except Exception as e:
            logger.debug(f"Failed parsing config file at '{model_name}': {e}")
    elif os.path.isdir(model_name):
        cfg_path = os.path.join(model_name, "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return _apply_inferred_precision(_parse_verified_config_dict(data))
            except Exception as e:
                logger.debug(f"Failed parsing config file at '{cfg_path}': {e}")

    # 2. Local verified configs directory (situs/configs/)
    configs_dir = pathlib.Path(__file__).resolve().parent.parent / "configs"
    if configs_dir.is_dir():
        clean = model_name.lower().split(":")[0].replace("-", "").replace("_", "")
        # Find matching config files
        best_match = None
        for cfg_file in configs_dir.glob("*.json"):
            stem_clean = cfg_file.stem.lower().replace("-", "").replace("_", "")
            if stem_clean == clean:
                best_match = cfg_file
                break
            if stem_clean in clean or clean in stem_clean:
                best_match = cfg_file

        if best_match is not None:
            try:
                data = json.loads(best_match.read_text(encoding="utf-8"))
                return _apply_inferred_precision(_parse_verified_config_dict(data))
            except Exception as e:
                logger.debug(f"Failed loading verified config from {best_match}: {e}")

    # 3. Check local model directory or cache config file
    try:
        from situs.worker.jit_streamer import DEFAULT_CACHE_DIR
        safe_name = model_name.replace("/", "_").replace(":", "_")
        model_dir = DEFAULT_CACHE_DIR / safe_name
        if model_dir.is_dir():
            cached_cfg = model_dir / "config.json"
            if cached_cfg.is_file():
                data = json.loads(cached_cfg.read_text(encoding="utf-8"))
                return _apply_inferred_precision(_parse_verified_config_dict(data))

            # Check if layer safetensors header has verified embedded metadata
            layer_files = sorted(model_dir.glob("layer_*.bin"))
            if layer_files:
                cached_data = layer_files[0].read_bytes()
                if len(cached_data) > 8:
                    h_len = int.from_bytes(cached_data[:8], "little")
                    if 0 < h_len < len(cached_data):
                        header = json.loads(cached_data[8:8 + h_len].decode("utf-8"))
                        meta = header.get("__metadata__")
                        if meta and isinstance(meta, dict):
                            return _apply_inferred_precision(_parse_verified_config_dict(meta))
    except Exception as e:
        logger.debug(f"Cache config check failed for '{model_name}': {e}")

    # 4. Check HuggingFace AutoConfig
    try:
        from transformers import AutoConfig
        config = None
        try:
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
        except Exception:
            hub_dir = os.path.expanduser("~/.cache/huggingface/hub")
            repo_name = f"models--{model_name.replace('/', '--')}"
            repo_dir = os.path.join(hub_dir, repo_name)
            if os.path.isdir(repo_dir):
                snapshots_dir = os.path.join(repo_dir, "snapshots")
                if os.path.isdir(snapshots_dir):
                    for s in os.listdir(snapshots_dir):
                        s_path = os.path.join(snapshots_dir, s)
                        if os.path.isdir(s_path):
                            try:
                                config = AutoConfig.from_pretrained(s_path, local_files_only=True)
                                if config:
                                    break
                            except Exception:
                                pass

        if config is not None:
            return _apply_inferred_precision(_parse_verified_config_dict(config.to_dict() if hasattr(config, "to_dict") else vars(config)))
    except Exception as e:
        logger.debug(f"AutoConfig metadata resolution failed for '{model_name}': {e}")

    # 5. Check local Ollama daemon /api/show
    try:
        import httpx
        with httpx.Client(timeout=1.0) as client:
            resp = client.post("http://127.0.0.1:11434/api/show", json={"name": model_name})
            if resp.status_code == 200:
                data = resp.json()
                model_info = data.get("model_info", {})
                block_count = None
                hidden_dim = None
                vocab_size = None
                intermediate = None
                heads = None
                kv_heads = None
                head_dim = None
                context_length = None
                torch_dtype = None
                for k, v in model_info.items():
                    if k.endswith(".block_count") and isinstance(v, int):
                        block_count = v
                    elif (k.endswith(".embedding_length") or k.endswith(".hidden_size")) and isinstance(v, int):
                        hidden_dim = v
                    elif k.endswith(".vocab_size") and isinstance(v, int):
                        vocab_size = v
                    elif k.endswith(".feed_forward_length") and isinstance(v, int):
                        intermediate = v
                    elif k.endswith(".head_count") and not k.endswith("_kv") and isinstance(v, int):
                        heads = v
                    elif k.endswith(".head_count_kv") and isinstance(v, int):
                        kv_heads = v
                    elif (k.endswith(".key_length") or k.endswith(".head_dim")) and isinstance(v, int):
                        head_dim = v
                    elif (k.endswith(".context_length") or k.endswith(".max_position_embeddings")) and isinstance(v, int):
                        context_length = v
                    elif k.endswith(".file_type") or k.endswith(".precision"):
                        torch_dtype = str(v)
                arch = model_info.get("general.architecture")
                if head_dim is None and hidden_dim and heads:
                    head_dim = max(1, hidden_dim // max(1, heads))

                if block_count and hidden_dim and vocab_size and intermediate and heads:
                    return _apply_inferred_precision({
                        "total_layers": block_count,
                        "hidden_dim": hidden_dim,
                        "intermediate_size": intermediate,
                        "vocab_size": vocab_size,
                        "num_heads": heads,
                        "num_kv_heads": kv_heads if kv_heads else heads,
                        "head_dim": head_dim,
                        "architecture": arch,
                        "context_length": context_length,
                        "max_position_embeddings": context_length,
                        "torch_dtype": torch_dtype,
                    })
                else:
                    logger.warning(
                        f"Ollama returned incomplete model_info for '{model_name}'; rejecting due to missing attributes."
                    )
    except Exception as e:
        logger.debug(f"Ollama metadata query failed: {e}")

    raise RuntimeError(
        f"Cannot resolve model metadata for '{model_name}'. "
        f"Model is unsupported: verified configuration could not be resolved from "
        f"local config files, Hugging Face AutoConfig, or Ollama daemon."
    )


def optimal_heterogeneous_partition(
    workers: List[WorkerNode],
    total_layers: int,
    model_metadata: Optional[Dict[str, Any]] = None,
    runtime_config: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, int]]:
    """
    Computes optimal contiguous layer slices [start, end) for each worker using Dynamic Programming.
    Discovers hardware capabilities, VRAM limits, model layer footprints (weights + KV cache + overhead),
    and performance telemetry at runtime. Strictly prevents naive equal division across heterogeneous GPUs.
    """
    if not workers:
        return []

    if not model_metadata or not isinstance(model_metadata, dict):
        raise ValueError(
            "optimal_heterogeneous_partition requires verified model_metadata. "
            "Arbitrary fallback model dimensions have been removed."
        )

    # Validate required model geometry - no fallback dimensions allowed!
    required_attrs = ("hidden_dim", "intermediate_size", "vocab_size", "num_heads")
    missing_attrs = [
        attr for attr in required_attrs
        if model_metadata.get(attr) is None or model_metadata.get(attr) <= 0
    ]
    if missing_attrs:
        raise ValueError(
            f"optimal_heterogeneous_partition requires verified model metadata attributes: missing or invalid {missing_attrs}"
        )

    meta = model_metadata
    h_dim = meta["hidden_dim"]
    inter_dim = meta["intermediate_size"]
    vocab_size = meta["vocab_size"]
    num_heads = meta["num_heads"]
    num_kv_heads = meta.get("num_kv_heads") or num_heads
    head_dim = meta.get("head_dim") or max(1, h_dim // max(1, num_heads))

    # Actual verified context length for KV-cache footprint (reject assumed fixed length)
    context_length = (
        (runtime_config.get("context_length") or runtime_config.get("max_seq_len"))
        if runtime_config else None
    ) or (
        meta.get("context_length")
        or meta.get("max_position_embeddings")
        or meta.get("seq_length")
        or meta.get("max_seq_len")
    )
    if context_length is None or context_length <= 0:
        raise ValueError(
            "optimal_heterogeneous_partition requires verified 'context_length' (or 'max_position_embeddings') "
            "in model_metadata or runtime_config. Assumed fixed sequence lengths have been removed."
        )

    # Precision: dtype bytes per param (fp32=4, fp16/bf16=2, int8/fp8=1, int4=0.5)
    precision_val = (
        (runtime_config.get("bytes_per_param") or runtime_config.get("precision") or runtime_config.get("compression_dtype") or runtime_config.get("torch_dtype") or runtime_config.get("dtype"))
        if runtime_config else None
    ) or (
        meta.get("bytes_per_param")
        or meta.get("precision")
        or meta.get("torch_dtype")
        or meta.get("compression_dtype")
        or meta.get("dtype")
    )
    if precision_val is None:
        raise ValueError(
            "optimal_heterogeneous_partition requires verified precision ('bytes_per_param', 'torch_dtype', or 'precision') "
            "in model_metadata or runtime_config. Arbitrary default precision has been removed."
        )

    if isinstance(precision_val, (int, float)) and precision_val > 0:
        bytes_per_param = float(precision_val)
    elif isinstance(precision_val, str):
        p_str = precision_val.strip().lower()
        if "32" in p_str:
            bytes_per_param = 4.0
        elif "16" in p_str or "bf" in p_str or "half" in p_str:
            bytes_per_param = 2.0
        elif "8" in p_str:
            bytes_per_param = 1.0
        elif "6" in p_str:
            bytes_per_param = 0.75
        elif "5" in p_str:
            bytes_per_param = 0.625
        elif "4" in p_str:
            bytes_per_param = 0.5
        else:
            raise ValueError(f"Unsupported or unrecognized precision / torch_dtype: '{precision_val}'")
    else:
        raise ValueError(f"Invalid precision type: {type(precision_val)}")

    batch_size = int((runtime_config.get("batch_size") if runtime_config else None) or 1)

    # Layer weights: Attention Q, K, V, O projections + SwiGLU Gate, Up, Down projections + RMSNorms
    params_per_layer = (
        (h_dim * num_heads * head_dim) +
        (h_dim * num_kv_heads * head_dim * 2) +
        (num_heads * head_dim * h_dim) +
        (h_dim * inter_dim * 3) +
        (h_dim * 2)
    )
    layer_weight_bytes = int(params_per_layer * bytes_per_param)
    # Dynamic KV cache footprint per layer using actual context length, batch size, and precision
    kv_cache_bytes = int(2 * num_kv_heads * head_dim * context_length * batch_size * bytes_per_param)
    bytes_per_layer = max(10 * 1024**2, layer_weight_bytes + kv_cache_bytes)

    # Stage-specific memory overhead:
    # First stage holds embedding table
    first_overhead = int(vocab_size * h_dim * bytes_per_param)
    # Last stage holds final RMSNorm + lm_head projection
    last_overhead = int((h_dim + h_dim * vocab_size) * bytes_per_param)

    # Total model memory footprint
    total_model_memory = (total_layers * bytes_per_layer) + first_overhead + last_overhead
    total_usable_cluster_vram = sum(max(50 * 1024**2, int(w.free_vram_bytes * 0.9)) for w in workers)
    if total_usable_cluster_vram < total_model_memory:
        raise ValueError(
            f"Insufficient cluster memory: total cluster usable memory ({total_usable_cluster_vram / (1024**3):.2f} GB) "
            f"cannot accommodate model memory requirements ({total_model_memory / (1024**3):.2f} GB)."
        )

    K = len(workers)
    if K == 1:
        return [(0, total_layers)]

    speeds = []
    capacities_layers = []

    for idx, w in enumerate(workers):
        # 1. Performance telemetry: use runtime-measured compute speed if available
        # Explicitly mark unmeasured telemetry as unknown without fabricating synthetic speeds
        measured_speed = getattr(w, "compute_speed", None)
        if measured_speed is not None and measured_speed > 0:
            base_speed = float(measured_speed)
            w.compute_speed_status = "measured"
        else:
            base_speed = 1.0
            w.compute_speed_status = "unknown"
            logger.debug(f"Worker {w.worker_id}: compute_speed is unmeasured/unknown. Using baseline.")

        # 2. Network telemetry: degrade effective throughput with measured network latency
        measured_lat = getattr(w, "network_latency_ms", None)
        if measured_lat is not None and measured_lat >= 0:
            lat_penalty = 100.0 / (100.0 + max(0.0, float(measured_lat)))
            w.network_latency_status = "measured"
        else:
            lat_penalty = 1.0
            w.network_latency_status = "unknown"
            logger.debug(f"Worker {w.worker_id}: network_latency_ms is unmeasured/unknown. Using baseline.")

        speed = base_speed * lat_penalty
        speeds.append(speed)

        # 3. Memory telemetry: available free VRAM with 10% safety margin for runtime buffers
        avail_vram = max(0, int(w.free_vram_bytes * 0.90))
        stage_overhead = (first_overhead if idx == 0 else 0) + (last_overhead if idx == (K - 1) else 0)
        usable_for_layers = avail_vram - stage_overhead
        if usable_for_layers < bytes_per_layer:
            max_fit = 0
            logger.warning(
                f"Worker {w.worker_id} has insufficient free VRAM ({w.free_vram_bytes / 1024**2:.1f} MB) "
                f"for stage overhead ({stage_overhead / 1024**2:.1f} MB) and layer ({bytes_per_layer / 1024**2:.1f} MB). Capacity set to 0."
            )
        else:
            max_fit = int(usable_for_layers // bytes_per_layer)
        capacities_layers.append(max_fit)

    if sum(capacities_layers) < total_layers:
        raise ValueError(
            f"Insufficient cluster memory: total worker layer capacities sum to {sum(capacities_layers)}, "
            f"which cannot accommodate {total_layers} required layers."
        )

    # 4. Dynamic Programming: minimize makespan max_k (l_k / speed_k) under capacity limits
    dp = [[math.inf] * (total_layers + 1) for _ in range(K + 1)]
    parent = [[0] * (total_layers + 1) for _ in range(K + 1)]
    dp[0][0] = 0.0

    for k in range(1, K + 1):
        w_idx = k - 1
        speed = speeds[w_idx]
        cap = capacities_layers[w_idx]
        is_last = (k == K)

        for i in range(total_layers + 1):
            min_l = 1 if (total_layers >= K and not is_last and cap > 0) else 0
            # Search strictly within actual capacity bound (preventing memory-infeasible allocations)
            max_search = min(i, cap)
            for l in range(min_l, max_search + 1):
                prev_i = i - l
                if dp[k - 1][prev_i] == math.inf:
                    continue
                cost_l = (l / speed) if speed > 0 else l
                stage_cost = max(dp[k - 1][prev_i], cost_l)
                if stage_cost < dp[k][i]:
                    dp[k][i] = stage_cost
                    parent[k][i] = l

    # 5. Reject if strict DP bounds cannot cover all layers (no unsafe memory overflow fallback)
    if dp[K][total_layers] == math.inf:
        raise ValueError(
            f"Insufficient cluster memory: cannot find a feasible layer partition across {K} workers "
            f"within physical memory bounds (total required: {total_model_memory / (1024**3):.2f} GB)."
        )

    alloc = [0] * K
    curr = total_layers
    for k in range(K, 0, -1):
        l = parent[k][curr]
        alloc[k - 1] = l
        curr -= l

    # Ensure every worker has at least 1 layer if total_layers >= K
    if total_layers >= K:
        for idx in range(K):
            if alloc[idx] == 0:
                max_idx = max(range(K), key=lambda x: alloc[x])
                if alloc[max_idx] > 1:
                    alloc[max_idx] -= 1
                    alloc[idx] += 1

    diff = total_layers - sum(alloc)
    if diff != 0:
        alloc[-1] += diff

    intervals = []
    curr_s = 0
    for count in alloc:
        intervals.append((curr_s, curr_s + count))
        curr_s += count

    return intervals


class WorkerRegistry:
    def __init__(self, heartbeat_timeout_s: float = 120.0):
        self.heartbeat_timeout_s = heartbeat_timeout_s
        self.workers: Dict[str, WorkerNode] = {}
        self.active_model: Optional[str] = None
        self.total_layers: int = 32

    def clear(self):
        """Clears all registered workers and resets state."""
        self.workers.clear()
        self.active_model = None

    def register_worker(self, reg: WorkerRegister) -> WorkerNode:
        used_b = getattr(reg, "used_vram_bytes", None)
        if used_b is None:
            used_b = max(0, reg.total_vram_bytes - reg.free_vram_bytes)
        node = WorkerNode(
            worker_id=reg.worker_id,
            gpu_name=reg.gpu_name,
            total_vram_bytes=reg.total_vram_bytes,
            free_vram_bytes=reg.free_vram_bytes,
            used_vram_bytes=used_b,
            device_type=reg.device_type,
            last_heartbeat=time.time(),
            is_alive=True,
        )
        self.workers[reg.worker_id] = node
        logger.info(f"Registered worker {reg.worker_id} ({reg.gpu_name}, {reg.free_vram_bytes / 1e9:.2f} GB free)")
        return node

    def update_heartbeat(
        self,
        worker_id: str,
        free_vram: int,
        gpu_name: Optional[str] = None,
        total_vram: Optional[int] = None,
        device_type: Optional[str] = None,
        used_vram: Optional[int] = None,
    ) -> bool:
        if worker_id in self.workers:
            w = self.workers[worker_id]
            w.last_heartbeat = time.time()
            w.free_vram_bytes = free_vram
            if used_vram is not None:
                w.used_vram_bytes = used_vram
            elif w.total_vram_bytes > free_vram:
                w.used_vram_bytes = w.total_vram_bytes - free_vram
            w.is_alive = True
            if gpu_name and ("Family" not in gpu_name or "Family" in w.gpu_name):
                w.gpu_name = gpu_name
            if total_vram and total_vram > 0:
                w.total_vram_bytes = total_vram
            if device_type:
                w.device_type = device_type
            return True
        return False

    def remove_worker(self, worker_id: str):
        if worker_id in self.workers:
            logger.warning(f"Removing worker {worker_id}")
            del self.workers[worker_id]

    def get_alive_workers(self) -> List[WorkerNode]:
        now = time.time()
        alive = []
        for w in self.workers.values():
            if (now - w.last_heartbeat) <= self.heartbeat_timeout_s:
                w.is_alive = True
                alive.append(w)
            else:
                w.is_alive = False
        return alive

    def partition_model(
        self,
        model_name: str,
        total_layers: Optional[int] = None,
        progressive: bool = False,
        codec: str = "zstd",
        runtime_config: Optional[Dict[str, Any]] = None,
    ) -> List[AssignLayers]:
        """
        Partitions transformer layers across currently connected alive workers using Dynamic Programming.
        Respects heterogeneous device capabilities (VRAM, compute speed) without hardcoding 32 layers.
        Assigns embed_tokens to the first worker and norm+lm_head to the last worker.
        """
        alive = self.get_alive_workers()
        if not alive:
            raise RuntimeError("No alive workers available to partition model.")

        meta = resolve_model_metadata(model_name)
        resolved_layers = total_layers if (total_layers is not None and total_layers > 0) else meta["total_layers"]

        self.active_model = model_name
        self.total_layers = resolved_layers
        num_workers = len(alive)

        # Compute optimal heterogeneous intervals via DP
        intervals = optimal_heterogeneous_partition(alive, resolved_layers, model_metadata=meta, runtime_config=runtime_config)

        assignments: List[AssignLayers] = []
        first_id = alive[0].worker_id if alive else None

        for idx, worker in enumerate(alive):
            layer_start, layer_end = intervals[idx]
            is_first = (idx == 0)
            is_last = (idx == num_workers - 1)
            next_id = alive[idx + 1].worker_id if idx + 1 < num_workers else None
            prev_id = alive[idx - 1].worker_id if idx > 0 else None

            # Update worker node record
            worker.assigned_layer_start = layer_start
            worker.assigned_layer_end = layer_end
            worker.is_first_stage = is_first
            worker.is_last_stage = is_last
            worker.next_worker_id = next_id
            worker.prev_worker_id = prev_id
            worker.first_worker_id = first_id

            assignments.append(
                AssignLayers(
                    worker_id=worker.worker_id,
                    model_name=model_name,
                    layer_start=layer_start,
                    layer_end=layer_end,
                    total_layers=resolved_layers,
                    is_first_stage=is_first,
                    is_last_stage=is_last,
                    next_worker_id=next_id,
                    prev_worker_id=prev_id,
                    first_worker_id=first_id,
                    progressive=progressive,
                    codec=codec,
                )
            )

        logger.info(
            f"Optimally partitioned {model_name} ({resolved_layers} layers) across {num_workers} workers via DP: "
            f"{[f'{w.worker_id}: [{a.layer_start}..{a.layer_end})' for w, a in zip(alive, assignments)]}"
        )
        return assignments

    def get_pipeline_order(self) -> List[WorkerNode]:
        """Returns alive workers in pipeline execution order (first stage -> last stage)."""
        alive = self.get_alive_workers()
        assigned = [w for w in alive if w.assigned_layer_start is not None]
        return sorted(assigned, key=lambda w: w.assigned_layer_start)
