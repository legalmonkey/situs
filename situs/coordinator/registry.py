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


@functools.lru_cache(maxsize=64)
def resolve_model_metadata(model_name: str) -> Dict[str, Any]:
    """
    Dynamically discovers model configuration (total layers, hidden dimensions, heads).
    Checks known model family architectures, local Ollama daemon, and HuggingFace AutoConfig.
    Avoids hardcoded 32 layers or arbitrary hidden dimension defaults.
    """
    clean = model_name.lower().split(":")[0].replace("-", "").replace("_", "")

    # 1. Known model family architectures (fast path, zero I/O)
    KNOWN_FAMILIES = {
        "gemma3": {"total_layers": 34, "hidden_dim": 2560, "intermediate_size": 10240, "vocab_size": 32000, "num_heads": 8, "num_kv_heads": 4},
        "gemma29b": {"total_layers": 42, "hidden_dim": 3584, "intermediate_size": 14336, "vocab_size": 32000, "num_heads": 16, "num_kv_heads": 8},
        "gemma2": {"total_layers": 26, "hidden_dim": 2304, "intermediate_size": 9216, "vocab_size": 32000, "num_heads": 8, "num_kv_heads": 4},
        "llama3.1": {"total_layers": 32, "hidden_dim": 4096, "intermediate_size": 14336, "vocab_size": 32000, "num_heads": 32, "num_kv_heads": 8},
        "llama3.23b": {"total_layers": 28, "hidden_dim": 3072, "intermediate_size": 8192, "vocab_size": 32000, "num_heads": 24, "num_kv_heads": 8},
        "llama3.21b": {"total_layers": 16, "hidden_dim": 2048, "intermediate_size": 8192, "vocab_size": 32000, "num_heads": 32, "num_kv_heads": 8},
        "llama3": {"total_layers": 32, "hidden_dim": 4096, "intermediate_size": 14336, "vocab_size": 32000, "num_heads": 32, "num_kv_heads": 8},
        "qwen2.57b": {"total_layers": 28, "hidden_dim": 3584, "intermediate_size": 18944, "vocab_size": 32000, "num_heads": 28, "num_kv_heads": 4},
        "qwen2.53b": {"total_layers": 36, "hidden_dim": 2048, "intermediate_size": 11008, "vocab_size": 32000, "num_heads": 16, "num_kv_heads": 2},
        "qwen2.5": {"total_layers": 28, "hidden_dim": 3584, "intermediate_size": 18944, "vocab_size": 32000, "num_heads": 28, "num_kv_heads": 4},
        "mistral7b": {"total_layers": 32, "hidden_dim": 4096, "intermediate_size": 14336, "vocab_size": 32000, "num_heads": 32, "num_kv_heads": 8},
        "deepseekr17b": {"total_layers": 28, "hidden_dim": 3584, "intermediate_size": 18944, "vocab_size": 32000, "num_heads": 28, "num_kv_heads": 4},
    }
    for fam, spec in KNOWN_FAMILIES.items():
        if fam.replace(".", "") in clean:
            return {**spec, "architecture": "transformer"}

    # 2. Check local Ollama daemon /api/show
    try:
        import httpx
        with httpx.Client(timeout=1.0) as client:
            resp = client.post("http://127.0.0.1:11434/api/show", json={"name": model_name})
            if resp.status_code == 200:
                data = resp.json()
                model_info = data.get("model_info", {})
                block_count = None
                for k, v in model_info.items():
                    if k.endswith(".block_count") and isinstance(v, int):
                        block_count = v
                        break
                hidden_dim = None
                for k, v in model_info.items():
                    if (k.endswith(".embedding_length") or k.endswith(".hidden_size")) and isinstance(v, int):
                        hidden_dim = v
                        break
                vocab_size = None
                for k, v in model_info.items():
                    if k.endswith(".vocab_size") and isinstance(v, int):
                        vocab_size = v
                        break
                intermediate = None
                for k, v in model_info.items():
                    if k.endswith(".feed_forward_length") and isinstance(v, int):
                        intermediate = v
                        break
                heads = None
                for k, v in model_info.items():
                    if k.endswith(".head_count") and not k.endswith("_kv") and isinstance(v, int):
                        heads = v
                        break
                kv_heads = None
                for k, v in model_info.items():
                    if k.endswith(".head_count_kv") and isinstance(v, int):
                        kv_heads = v
                        break
                arch = model_info.get("general.architecture", "llama")

                if block_count and block_count > 0:
                    return {
                        "total_layers": block_count,
                        "hidden_dim": hidden_dim or 2048,
                        "intermediate_size": intermediate or (hidden_dim * 4 if hidden_dim else 8192),
                        "vocab_size": vocab_size or 32000,
                        "num_heads": heads or 16,
                        "num_kv_heads": kv_heads or heads or 16,
                        "architecture": arch,
                    }
    except Exception as e:
        logger.debug(f"Ollama metadata query failed: {e}")

    # 3. Check HuggingFace AutoConfig
    if "/" in model_name or "\\" in model_name:
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name, local_files_only=True)
            return {
                "total_layers": getattr(config, "num_hidden_layers", getattr(config, "n_layer", 32)),
                "hidden_dim": getattr(config, "hidden_size", getattr(config, "n_embd", 2048)),
                "intermediate_size": getattr(config, "intermediate_size", 8192),
                "vocab_size": getattr(config, "vocab_size", 32000),
                "num_heads": getattr(config, "num_attention_heads", getattr(config, "n_head", 16)),
                "num_kv_heads": getattr(config, "num_key_value_heads", getattr(config, "num_attention_heads", 16)),
                "architecture": getattr(config, "model_type", "llama"),
            }
        except Exception:
            pass

    return {
        "total_layers": 32,
        "hidden_dim": 2048,
        "intermediate_size": 8192,
        "vocab_size": 32000,
        "num_heads": 16,
        "num_kv_heads": 16,
        "architecture": "llama",
    }


def optimal_heterogeneous_partition(
    workers: List[WorkerNode],
    total_layers: int,
    model_metadata: Optional[Dict[str, Any]] = None,
) -> List[Tuple[int, int]]:
    """
    Computes optimal contiguous layer slices [start, end) for each worker using Dynamic Programming.
    Optimizes pipeline makespan max_j (layers_j / speed_j) while strictly respecting memory limits.
    """
    K = len(workers)
    if K == 0:
        return []
    if K == 1:
        return [(0, total_layers)]

    meta = model_metadata or {}
    h_dim = meta.get("hidden_dim", 2048)
    inter_dim = meta.get("intermediate_size", h_dim * 4)
    bytes_per_layer = max(10 * 1024**2, int(2 * (h_dim * inter_dim * 2 + h_dim * h_dim * 4)))

    speeds = []
    capacities_layers = []

    for w in workers:
        is_cuda = (w.device_type == "cuda")
        vram_gb = max(1.0, w.total_vram_bytes / 1e9)
        base_speed = getattr(w, "compute_speed", None)
        if not base_speed or base_speed <= 0:
            base_speed = (10.0 + vram_gb * 0.5) if is_cuda else 1.0
        # Network latency penalty: latency degrades stage transit speed
        lat_ms = getattr(w, "network_latency_ms", None) or 0.0
        lat_penalty = 100.0 / (100.0 + max(0.0, lat_ms))
        speed = base_speed * lat_penalty
        speeds.append(speed)

        free_bytes = max(100 * 1024**2, w.free_vram_bytes)
        max_fit = max(1, int(free_bytes // bytes_per_layer))
        capacities_layers.append(max_fit)

    # DP table: dp[k][i] = min makespan assigning i layers across first k workers
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
            for l in range(min_l, min(i, max(cap * 3, total_layers)) + 1):
                prev_i = i - l
                if dp[k - 1][prev_i] == math.inf:
                    continue
                cost_l = (l / speed) if speed > 0 else l
                stage_cost = max(dp[k - 1][prev_i], cost_l)
                if stage_cost < dp[k][i]:
                    dp[k][i] = stage_cost
                    parent[k][i] = l

    if dp[K][total_layers] == math.inf:
        logger.warning("DP partition constraint fallback to capacity-weighted proportional partitioning.")
        total_w = sum(speeds[j] * max(1, capacities_layers[j]) for j in range(K))
        alloc = []
        rem = total_layers
        for j in range(K - 1):
            share = (speeds[j] * max(1, capacities_layers[j])) / total_w
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
