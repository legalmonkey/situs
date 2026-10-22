"""Worker registry, health tracking, and model layer partition scheduler."""

import time
import logging
from typing import Dict, List, Optional
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


class WorkerRegistry:
    def __init__(self, heartbeat_timeout_s: float = 45.0):
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
        total_layers: int = 32,
        progressive: bool = False,
        fast_boot: bool = False,
        codec: str = "zstd",
    ) -> List[AssignLayers]:
        """
        Partitions total_layers across currently connected alive workers.
        Assigns embed_tokens to the first worker and norm+lm_head to the last worker.
        Supports Strategy 3 fast-boot skeleton initialization and Strategy 4 wire compression.
        """
        alive = self.get_alive_workers()
        if not alive:
            raise RuntimeError("No alive workers available to partition model.")

        self.active_model = model_name
        self.total_layers = total_layers
        num_workers = len(alive)

        # Calculate contiguous slice per worker
        layers_per_worker = total_layers // num_workers
        remainder = total_layers % num_workers

        assignments: List[AssignLayers] = []
        current_layer = 0

        first_id = alive[0].worker_id if alive else None

        for idx, worker in enumerate(alive):
            extra = 1 if idx < remainder else 0
            layer_start = current_layer
            layer_end = current_layer + layers_per_worker + extra
            current_layer = layer_end

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
                    total_layers=total_layers,
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

        logger.info(f"Partitioned {model_name} ({total_layers} layers) across {num_workers} workers (FastBoot={fast_boot}, Codec={codec}).")
        return assignments

    def get_pipeline_order(self) -> List[WorkerNode]:
        """Returns alive workers in pipeline execution order (first stage -> last stage)."""
        alive = self.get_alive_workers()
        # Sort by assigned_layer_start
        assigned = [w for w in alive if w.assigned_layer_start is not None]
        return sorted(assigned, key=lambda w: w.assigned_layer_start)
