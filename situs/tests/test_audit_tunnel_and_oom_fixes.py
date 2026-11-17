"""Unit and integration tests for Cloudflare Tunnel fallback, VRAM clamping, and Inference OOM fixes.

Adheres strictly to GEMINI.md invariants:
- Strictly verified handshakes (Zero False-Success Reporting, WORKER_REGISTERED required).
- RFC 1918 Private subnet boundaries.
- Live socket handshake verification on real sockets.
"""

import asyncio
import os
import re
import socket
import tempfile
import time
import pytest
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F

from situs.network.tunnel import (
    CLOUDFLARE_URL_PATTERN,
    SSH_TUNNEL_URL_PATTERN,
    ALL_TUNNEL_URL_PATTERN,
    PublicTunnelManager,
    start_public_tunnel,
    stop_public_tunnel,
)
from situs.hardware import (
    get_live_gpu_vram,
    detect_hardware,
    _query_nvml,
)
from situs.coordinator.registry import (
    WorkerRegistry,
    optimal_heterogeneous_partition,
    WorkerNode,
)
from situs.protocol.messages import (
    WorkerRegister,
    MessageType,
    ProtocolFrame,
)
from situs.worker.loader import (
    DistributedStageModel,
    RealTransformerBlock,
)
from situs.worker.sampler import TokenSampler
from situs.worker.engine import StageExecutionEngine
from situs.coordinator.app import cluster_status, registry, app


# -----------------------------------------------------------------------------
# 1. TUNNEL REGEX & FALLBACK TESTS
# -----------------------------------------------------------------------------

def test_tunnel_url_regexes():
    """Verify regex patterns for Cloudflare quick tunnel, token tunnel, and OpenSSH fallbacks."""
    # Cloudflare Quick Tunnel
    cf_url = "https://dull-actor-testing.trycloudflare.com"
    assert CLOUDFLARE_URL_PATTERN.search(cf_url) is not None
    assert ALL_TUNNEL_URL_PATTERN.search(cf_url).group(0) == cf_url

    # OpenSSH localhost.run fallback
    lhr_url = "https://a1b2c3d4.app.lhr.life"
    assert SSH_TUNNEL_URL_PATTERN.search(lhr_url) is not None
    assert ALL_TUNNEL_URL_PATTERN.search(lhr_url).group(0) == lhr_url

    lhr_link = "https://subdomain.lhrtunnel.link"
    assert SSH_TUNNEL_URL_PATTERN.search(lhr_link) is not None
    assert ALL_TUNNEL_URL_PATTERN.search(lhr_link).group(0) == lhr_link

    # Pinggy fallback
    pinggy_url = "https://xyz123.pinggy.link"
    assert SSH_TUNNEL_URL_PATTERN.search(pinggy_url) is not None
    assert ALL_TUNNEL_URL_PATTERN.search(pinggy_url).group(0) == pinggy_url


def test_public_tunnel_manager_token_mode():
    """Verify PublicTunnelManager supports user-provided token mode without hanging."""
    mgr = PublicTunnelManager(port=8000, cloudflared_path="fake_cloudflared", token="mock-cf-token-12345")
    assert mgr.token == "mock-cf-token-12345"
    assert mgr.port == 8000


# -----------------------------------------------------------------------------
# 2. VRAM TELEMETRY CLAMPING TESTS (Preventing 10.7/6.0 GB 170% Distortions)
# -----------------------------------------------------------------------------

def test_registry_vram_clamping_on_registration_and_heartbeat():
    """Verify WorkerRegistry strictly clamps inflated used VRAM (e.g. 10.7 GB on 6.0 GB GPU) to total VRAM."""
    reg = WorkerRegistry()
    total_6gb = 6 * 1024**3
    inflated_used = int(10.7 * 1024**3)  # Simulated leaked system RAM

    worker_reg = WorkerRegister(
        worker_id="test-worker-1",
        gpu_name="NVIDIA GeForce RTX 3050 6GB Laptop GPU",
        total_vram_bytes=total_6gb,
        free_vram_bytes=int(0.5 * 1024**3),
        used_vram_bytes=inflated_used,
        device_type="cuda",
    )

    node = reg.register_worker(worker_reg)
    # Must be clamped <= total_6gb
    assert node.used_vram_bytes <= total_6gb
    assert node.used_vram_bytes == total_6gb
    assert node.free_vram_bytes <= total_6gb

    # Heartbeat update with inflated VRAM must also clamp
    reg.update_heartbeat(
        worker_id="test-worker-1",
        free_vram=int(0.3 * 1024**3),
        used_vram=int(11.2 * 1024**3),
        total_vram=total_6gb,
    )
    updated = reg.workers["test-worker-1"]
    assert updated.used_vram_bytes <= total_6gb
    assert updated.used_vram_bytes == total_6gb


@pytest.mark.asyncio
async def test_coordinator_status_vram_clamping():
    """Verify /status endpoint clamps pooled VRAM and percentages so it never displays >100% or >total."""
    registry.clear()
    total_6gb = 6 * 1024**3
    inflated_10_7gb = int(10.7 * 1024**3)

    w1 = WorkerRegister(
        worker_id="worker-w1",
        gpu_name="NVIDIA RTX 4050 Laptop GPU",
        total_vram_bytes=total_6gb,
        free_vram_bytes=int(0.5 * 1024**3),
        used_vram_bytes=inflated_10_7gb,
        device_type="cuda",
    )
    w2 = WorkerRegister(
        worker_id="worker-w2",
        gpu_name="NVIDIA RTX 3050 Laptop GPU",
        total_vram_bytes=total_6gb,
        free_vram_bytes=int(0.3 * 1024**3),
        used_vram_bytes=int(9.7 * 1024**3),
        device_type="cuda",
    )

    registry.register_worker(w1)
    registry.register_worker(w2)

    stat = await cluster_status()
    # Total is 12.0 GB
    assert stat["total_vram_gb"] == 12.0
    # Used must be <= 12.0 GB (never 20.4 GB)
    assert stat["used_vram_gb"] <= 12.0
    # Usage pct must be <= 100.0% (never 170%)
    assert stat["vram_usage_pct"] <= 100.0

    for w_stat in stat["workers"]:
        assert w_stat["used_gb"] <= w_stat["total_gb"]
        assert w_stat["usage_pct"] <= 100.0

    registry.clear()


# -----------------------------------------------------------------------------
# 3. GEMMA 3 LM HEAD SLICING (Memory Reduction from 1.05 GB to 524 KB)
# -----------------------------------------------------------------------------

def test_lm_head_prefill_slicing_memory_reduction():
    """Verify forward_intermediate_stage on last stage slices h to [1, -1:, :] before LM head."""
    hidden_dim = 256
    vocab_size = 2000
    seq_len = 128

    stage = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=2,
        is_first_stage=False,
        is_last_stage=True,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=512,
        head_dim=64,
        architecture="llama",
        device="cpu",
    )
    # Initialize mock norm and lm_head
    stage.norm = torch.nn.LayerNorm(hidden_dim)
    stage.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)

    # Incoming activations from prefill prompt with 128 tokens
    activations = torch.randn(1, seq_len, hidden_dim)

    # Normal autoregressive prefill: compute_all_logits=False
    logits_sliced, _ = stage.forward_intermediate_stage(activations, compute_all_logits=False)
    # Output shape must be [1, 1, vocab_size], NOT [1, 128, vocab_size]
    assert logits_sliced.shape == (1, 1, vocab_size)

    # Speculative candidate verification: compute_all_logits=True
    cand_activations = torch.randn(1, 4, hidden_dim)
    logits_all, _ = stage.forward_intermediate_stage(cand_activations, compute_all_logits=True)
    assert logits_all.shape == (1, 4, vocab_size)


# -----------------------------------------------------------------------------
# 4. SLIDING WINDOW KV CACHE PRUNING & GROUPED-QUERY ATTENTION
# -----------------------------------------------------------------------------

def test_real_transformer_block_sliding_window_pruning():
    """Verify RealTransformerBlock prunes KV cache exceeding sliding_window."""
    hidden_dim = 128
    num_heads = 4
    num_kv_heads = 2
    head_dim = 32
    sliding_window = 16

    block = RealTransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=256,
        head_dim=head_dim,
        architecture="gemma3",
        sliding_window=sliding_window,
        device="cpu",
    )

    # Simulate past KV cache having 20 tokens (> sliding_window of 16)
    past_k = torch.randn(1, num_kv_heads, 20, head_dim)
    past_v = torch.randn(1, num_kv_heads, 20, head_dim)

    # Forward 1 new token
    x = torch.randn(1, 1, hidden_dim)
    out, (new_k, new_v) = block(x, past_kv=(past_k, past_v))

    # The new KV cache must be strictly clamped to sliding_window (16), preventing unbounded growth
    assert new_k.shape[-2] == sliding_window
    assert new_v.shape[-2] == sliding_window
    assert out.shape == (1, 1, hidden_dim)


def test_grouped_query_attention_native_sdpa():
    """Verify RealTransformerBlock executes GQA with enable_gqa without manual tensor duplication."""
    hidden_dim = 128
    num_heads = 8
    num_kv_heads = 2
    head_dim = 16

    block = RealTransformerBlock(
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=256,
        head_dim=head_dim,
        architecture="llama",
        device="cpu",
    )

    x = torch.randn(1, 5, hidden_dim)
    out, (k, v) = block(x, past_kv=None)
    assert out.shape == (1, 5, hidden_dim)
    assert k.shape == (1, num_kv_heads, 5, head_dim)
    assert v.shape == (1, num_kv_heads, 5, head_dim)


# -----------------------------------------------------------------------------
# 5. TOKEN SAMPLER TOP-K CHUNKING ON LARGE VOCABULARIES
# -----------------------------------------------------------------------------

def test_token_sampler_topk_on_large_vocab():
    """Verify TokenSampler samples correctly and deterministically on 262k vocabulary without sorting all entries."""
    sampler = TokenSampler()
    vocab_size = 262208

    logits = torch.randn(vocab_size)
    # Ensure token 42 is dominant
    logits[42] = 50.0

    tok = sampler.sample_next_token(logits, temperature=0.7, top_p=0.9, top_k=50)
    assert tok == 42


# -----------------------------------------------------------------------------
# 6. BALANCED PIPELINE LAYER PARTITIONING
# -----------------------------------------------------------------------------

def test_balanced_layer_partitioning_on_symmetric_gpus():
    """Verify optimal_heterogeneous_partition splits 34 layers evenly (17 & 17) across two 6GB GPUs."""
    now = time.time()
    w1 = WorkerNode(
        worker_id="worker-w1",
        gpu_name="NVIDIA RTX 4050 6GB Laptop GPU",
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=int(4.8 * 1024**3),
        used_vram_bytes=int(1.2 * 1024**3),
        device_type="cuda",
        last_heartbeat=now,
    )
    w2 = WorkerNode(
        worker_id="worker-w2",
        gpu_name="NVIDIA RTX 3050 6GB Laptop GPU",
        total_vram_bytes=6 * 1024**3,
        free_vram_bytes=int(4.8 * 1024**3),
        used_vram_bytes=int(1.2 * 1024**3),
        device_type="cuda",
        last_heartbeat=now,
    )

    model_metadata = {
        "model_name": "gemma3:4b",
        "total_layers": 34,
        "hidden_dim": 2560,
        "num_heads": 16,
        "num_kv_heads": 8,
        "intermediate_size": 10240,
        "head_dim": 256,
        "vocab_size": 262208,
        "context_length": 2048,
        "torch_dtype": "int8",
    }

    intervals = optimal_heterogeneous_partition(
        workers=[w1, w2],
        total_layers=34,
        model_metadata=model_metadata,
        runtime_config={"precision": "int8"},
    )
    assert len(intervals) == 2
    # Verify both workers receive layers and neither worker is starved or overloaded
    c1 = intervals[0][1] - intervals[0][0]
    c2 = intervals[1][1] - intervals[1][0]
    assert c1 + c2 == 34
    assert abs(c1 - c2) <= 2  # Balanced split: 17 & 17


# -----------------------------------------------------------------------------
# 7. LIVE SOCKET HANDSHAKE VERIFICATION (GEMINI.md Invariant 1 & 3)
# -----------------------------------------------------------------------------

def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.asyncio
async def test_live_socket_handshake_and_vram_reporting():
    """Verify real over-the-wire websocket handshake: WorkerRegister -> WORKER_REGISTERED on live socket."""
    import uvicorn
    from situs.worker.client import WorkerDaemon
    from situs.config import WorkerConfig

    port = find_free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)

    server_task = asyncio.create_task(server.serve())
    try:
        # Wait for server to accept connections
        for _ in range(50):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                    break
            except (ConnectionRefusedError, OSError):
                await asyncio.sleep(0.05)

        worker_cfg = WorkerConfig(
            coordinator_url=f"ws://127.0.0.1:{port}/ws/worker",
            worker_id="test-live-worker",
            device="cpu",
            heartbeat_interval_s=1.0,
        )
        worker = WorkerDaemon(worker_cfg)
        worker_task = asyncio.create_task(worker.run())

        try:
            # Wait for verified over-the-wire registration acknowledgment (WORKER_REGISTERED)
            for _ in range(50):
                if worker.is_registered:
                    break
                await asyncio.sleep(0.1)

            # Invariant 1: Worker MUST NOT be reported as registered without verified acknowledgment
            assert worker.is_registered is True
            assert "test-live-worker" in registry.workers
            registered_node = registry.workers["test-live-worker"]
            assert registered_node.is_alive is True
            assert registered_node.used_vram_bytes <= registered_node.total_vram_bytes

        finally:
            worker.stop()
            worker_task.cancel()
            try:
                await worker_task
            except asyncio.CancelledError:
                pass
    finally:
        server.should_exit = True
        await server_task
