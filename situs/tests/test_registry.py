"""Tests for WorkerRegistry layer partitioning and health tracking."""

import pytest
import time
from situs.coordinator.registry import WorkerRegistry
from situs.protocol.messages import WorkerRegister


def test_registry_registration_and_heterogeneous_partitioning():
    registry = WorkerRegistry(heartbeat_timeout_s=5.0)

    # Register Worker A (6GB RTX 3050)
    w_a = registry.register_worker(
        WorkerRegister(
            worker_id="worker-a",
            gpu_name="NVIDIA RTX 3050",
            total_vram_bytes=6 * 1024**3,
            free_vram_bytes=5 * 1024**3,
            device_type="cuda",
        )
    )

    # Register Worker B (24GB RTX 4090)
    w_b = registry.register_worker(
        WorkerRegister(
            worker_id="worker-b",
            gpu_name="NVIDIA RTX 4090",
            total_vram_bytes=24 * 1024**3,
            free_vram_bytes=20 * 1024**3,
            device_type="cuda",
        )
    )

    assert len(registry.get_alive_workers()) == 2

    # Partition 32 layers for llama-3.1 across the 2 heterogeneous workers
    assignments = registry.partition_model("llama-3.1:8b-instruct-q8_0", total_layers=32)

    assert len(assignments) == 2
    # First stage (Worker A - smaller GPU gets fewer layers)
    assign_a = assignments[0]
    assert assign_a.worker_id == "worker-a"
    assert assign_a.layer_start == 0
    assert assign_a.is_first_stage is True
    assert assign_a.is_last_stage is False
    assert assign_a.next_worker_id == "worker-b"

    # Last stage (Worker B - larger GPU gets more layers)
    assign_b = assignments[1]
    assert assign_b.worker_id == "worker-b"
    assert assign_b.layer_start == assign_a.layer_end
    assert assign_b.layer_end == 32
    assert assign_b.is_first_stage is False
    assert assign_b.is_last_stage is True
    assert assign_b.prev_worker_id == "worker-a"

    # DP optimization allocates more layers to the 24GB card than the 6GB card
    assert (assign_a.layer_end - assign_a.layer_start) < (assign_b.layer_end - assign_b.layer_start)
    assert (assign_a.layer_end - assign_a.layer_start) + (assign_b.layer_end - assign_b.layer_start) == 32


def test_registry_homogeneous_partitioning():
    registry = WorkerRegistry(heartbeat_timeout_s=5.0)

    # Two identical workers get equal splits
    registry.register_worker(
        WorkerRegister(
            worker_id="worker-1",
            gpu_name="NVIDIA RTX 4090",
            total_vram_bytes=24 * 1024**3,
            free_vram_bytes=20 * 1024**3,
            device_type="cuda",
        )
    )
    registry.register_worker(
        WorkerRegister(
            worker_id="worker-2",
            gpu_name="NVIDIA RTX 4090",
            total_vram_bytes=24 * 1024**3,
            free_vram_bytes=20 * 1024**3,
            device_type="cuda",
        )
    )

    assignments = registry.partition_model("llama-3.1:8b-instruct-q8_0", total_layers=32)
    assert len(assignments) == 2
    assert assignments[0].layer_start == 0
    assert assignments[0].layer_end == 16
    assert assignments[1].layer_start == 16
    assert assignments[1].layer_end == 32


def test_registry_heartbeat_timeout():
    registry = WorkerRegistry(heartbeat_timeout_s=0.5)
    registry.register_worker(
        WorkerRegister(
            worker_id="worker-temp",
            gpu_name="GPU",
            total_vram_bytes=8 * 1024**3,
            free_vram_bytes=8 * 1024**3,
            device_type="cuda",
        )
    )
    assert len(registry.get_alive_workers()) == 1

    # Sleep past timeout
    time.sleep(0.6)
    assert len(registry.get_alive_workers()) == 0
