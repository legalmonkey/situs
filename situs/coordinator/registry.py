"""Worker registry, health tracking, and dynamic programming layer partition scheduler."""

import time
import math
import logging
import functools
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel
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
    fast_boot: bool = False
    compute_speed: Optional[float] = None
    network_latency_ms: Optional[float] = None


def _parse_verified_config_dict(data: Dict[str, Any]) -> Dict[str, Any]:
    """Dynamically parses and validates a model config dict without arbitrary fallback defaults."""
    total_layers = data.get("num_hidden_layers") or data.get("n_layer") or data.get("num_layers") or data.get("total_layers")
    hidden_dim = data.get("hidden_size") or data.get("n_embd") or data.get("hidden_dim")
    vocab_size = data.get("vocab_size")
    num_heads = data.get("num_attention_heads") or data.get("n_head") or data.get("num_heads")
    num_kv_heads = data.get("num_key_value_heads") or data.get("num_kv_heads") or num_heads
    intermediate_size = data.get("intermediate_size") or data.get("n_inner")
    arch = data.get("model_type") or data.get("architecture") or "transformer"
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

    return {
        "total_layers": total_layers,
        "hidden_dim": hidden_dim,
        "intermediate_size": intermediate_size,
        "vocab_size": vocab_size,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads if num_kv_heads else num_heads,
        "head_dim": head_dim,
        "architecture": str(arch),
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

    # 1. Direct filesystem path to config file or model directory
    if os.path.isfile(model_name):
        try:
            with open(model_name, "r", encoding="utf-8") as f:
                data = json.load(f)
            return _parse_verified_config_dict(data)
        except Exception as e:
            logger.debug(f"Failed parsing config file at '{model_name}': {e}")
    elif os.path.isdir(model_name):
        cfg_path = os.path.join(model_name, "config.json")
        if os.path.isfile(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return _parse_verified_config_dict(data)
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
                return _parse_verified_config_dict(data)
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
                return _parse_verified_config_dict(data)

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
                            return _parse_verified_config_dict(meta)
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
            return _parse_verified_config_dict(config.to_dict() if hasattr(config, "to_dict") else vars(config))
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
                head_dim = None
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
                arch = model_info.get("general.architecture", "transformer")
                if head_dim is None and hidden_dim and heads:
                    head_dim = max(1, hidden_dim // max(1, heads))

                if block_count and hidden_dim and vocab_size and intermediate and heads:
                    return {
                        "total_layers": block_count,
                        "hidden_dim": hidden_dim,
                        "intermediate_size": intermediate,
                        "vocab_size": vocab_size,
                        "num_heads": heads,
                        "num_kv_heads": kv_heads if kv_heads else heads,
                        "head_dim": head_dim,
                        "architecture": arch,
                    }
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
) -> List[Tuple[int, int]]:
    """
    Computes optimal contiguous layer slices [start, end) for each worker using Dynamic Programming.
    Discovers hardware capabilities, VRAM limits, model layer footprints (weights + KV cache + overhead),
    and performance telemetry at runtime. Strictly prevents naive equal division across heterogeneous GPUs.
    """
    K = len(workers)
    if K == 0:
        return []
    if K == 1:
        return [(0, total_layers)]

    meta = model_metadata or {}
    h_dim = meta.get("hidden_dim", 2048)
    inter_dim = meta.get("intermediate_size", h_dim * 4)
    vocab_size = meta.get("vocab_size", 32000)
    num_heads = meta.get("num_heads", 16)
    num_kv_heads = meta.get("num_kv_heads", num_heads)
    head_dim = meta.get("head_dim") or max(1, h_dim // max(1, num_heads))

    # Bytes per element (fp16/bf16 = 2, int8 = 1)
    bytes_per_param = 2
    # Layer weights: Attention Q, K, V, O projections + SwiGLU Gate, Up, Down projections + RMSNorms
    params_per_layer = (
        (h_dim * num_heads * head_dim) +
        (h_dim * num_kv_heads * head_dim * 2) +
        (num_heads * head_dim * h_dim) +
        (h_dim * inter_dim * 3) +
        (h_dim * 2)
    )
    layer_weight_bytes = params_per_layer * bytes_per_param
    # Dynamic KV cache footprint per layer (seq_len ~2048, batch_size=1)
    kv_cache_bytes = 2 * num_kv_heads * head_dim * 2048 * bytes_per_param
    bytes_per_layer = max(10 * 1024**2, layer_weight_bytes + kv_cache_bytes)

    # Stage-specific memory overhead:
    # First stage holds embedding table
    first_overhead = vocab_size * h_dim * bytes_per_param
    # Last stage holds final RMSNorm + lm_head projection
    last_overhead = (h_dim + h_dim * vocab_size) * bytes_per_param

    speeds = []
    capacities_layers = []

    for idx, w in enumerate(workers):
        is_cuda = (w.device_type == "cuda")
        is_mps = (w.device_type == "mps")
        vram_gb = max(1.0, w.total_vram_bytes / 1e9)

        # 1. Performance telemetry: use runtime-measured compute speed if available
        base_speed = getattr(w, "compute_speed", None)
        if not base_speed or base_speed <= 0:
            if is_cuda:
                base_speed = 10.0 + (vram_gb * 1.5)
            elif is_mps:
                base_speed = 8.0
            else:
                base_speed = 1.0  # CPU is order of magnitude slower

        # 2. Network telemetry: degrade effective throughput with network latency
        lat_ms = getattr(w, "network_latency_ms", None) or 0.0
        lat_penalty = 100.0 / (100.0 + max(0.0, lat_ms))
        speed = base_speed * lat_penalty
        speeds.append(speed)

        # 3. Memory telemetry: available free VRAM with 10% safety margin for runtime buffers
        avail_vram = max(50 * 1024**2, int(w.free_vram_bytes * 0.9))
        stage_overhead = (first_overhead if idx == 0 else 0) + (last_overhead if idx == (K - 1) else 0)
        usable_for_layers = max(bytes_per_layer, avail_vram - min(avail_vram // 2, stage_overhead))
        max_fit = max(1, int(usable_for_layers // bytes_per_layer))
        capacities_layers.append(max_fit)

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
            min_l = 1 if (total_layers >= K and not is_last) else 0
            # Search within actual capacity bound (with 50% burst margin before hard fallback)
            max_search = min(i, max(cap, int(cap * 1.5)))
            for l in range(min_l, max_search + 1):
                prev_i = i - l
                if dp[k - 1][prev_i] == math.inf:
                    continue
                cost_l = (l / speed) if speed > 0 else l
                stage_cost = max(dp[k - 1][prev_i], cost_l)
                if stage_cost < dp[k][i]:
                    dp[k][i] = stage_cost
                    parent[k][i] = l

    # 5. Proportional memory/speed fallback if strict DP bounds cannot cover all layers
    if dp[K][total_layers] == math.inf:
        logger.warning("DP partition constraint fallback to memory- and telemetry-weighted proportional partitioning.")
        # Weight by available memory and compute speed
        weights = [speeds[j] * max(1, capacities_layers[j]) for j in range(K)]
        total_w = sum(weights)
        alloc = []
        rem = total_layers
        for j in range(K - 1):
            share = weights[j] / max(1e-6, total_w)
            n = max(1, min(rem - (K - 1 - j), int(round(total_layers * share))))
            alloc.append(n)
            rem -= n
        alloc.append(max(1, rem))
    else:
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
        fast_boot: bool = False,
        codec: str = "zstd",
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
        intervals = optimal_heterogeneous_partition(alive, resolved_layers, model_metadata=meta)

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
            worker.fast_boot = fast_boot

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
                    fast_boot=fast_boot,
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
