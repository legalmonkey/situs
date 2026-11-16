import pytest
import unittest.mock as mock
try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from situs.worker.loader import Int8WeightLinear, DistributedStageModel
from situs.worker.client import WorkerDaemon
from situs.config import WorkerConfig


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for loader tests")
def test_int8_weight_linear_chunked_forward_parity():
    """Verify Int8WeightLinear chunks large vocabulary matrices without memory spikes or numerical drift."""
    hidden_dim = 64
    vocab_size = 40000  # > 32768 threshold to trigger chunking
    torch.manual_seed(42)
    weight = torch.randn(vocab_size, hidden_dim, dtype=torch.float32)
    bias = torch.randn(vocab_size, dtype=torch.float32)

    linear = Int8WeightLinear.from_float(weight, bias)
    assert linear.out_features == vocab_size

    x = torch.randn(2, 3, hidden_dim, dtype=torch.float32)

    # Output from chunked forward
    out_chunked = linear(x)
    assert out_chunked.shape == (2, 3, vocab_size)

    # Reference unchunked computation
    w_dequant = linear.weight.float() * linear.weight_scale.float()
    b_dequant = linear.bias.float()
    out_ref = F.linear(x, w_dequant, b_dequant)

    assert torch.allclose(out_chunked, out_ref, atol=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for stage model tests")
def test_gemma_stage_model_dtype_selection():
    """Verify that Gemma architecture models select bfloat16 when supported on CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    can_bf16 = hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()

    stage = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=256,
        vocab_size=1000,
        num_heads=4,
        num_kv_heads=4,
        head_dim=64,
        intermediate_size=512,
        device="cuda:0",
        progressive=False,
        architecture="gemma3",
        _lazy_alloc=True,
    )

    if can_bf16:
        assert stage.dtype == torch.bfloat16
    else:
        assert stage.dtype == torch.float32


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for worker telemetry tests")
def test_worker_caching_allocator_vram_telemetry():
    """Verify worker client telemetry accounts for PyTorch caching allocator reserved memory."""
    cfg = WorkerConfig(worker_id="test-worker", device="cuda:0")
    daemon = WorkerDaemon(cfg)

    # Mock PyTorch memory functions
    total_mem = 6 * (1024 ** 3)  # 6 GB
    # OS reports only 50 MB free because PyTorch reserved 5.9 GB
    os_free = 50 * (1024 ** 2)
    torch_alloc = 2 * (1024 ** 3)  # Actually only 2 GB allocated
    torch_reserved = 5800 * (1024 ** 2)  # 5.8 GB reserved by caching allocator

    with mock.patch("torch.cuda.is_available", return_value=True), \
         mock.patch("torch.cuda.mem_get_info", return_value=(os_free, total_mem)), \
         mock.patch("torch.cuda.memory_allocated", return_value=torch_alloc), \
         mock.patch("torch.cuda.memory_reserved", return_value=torch_reserved):

        daemon._cached_hw = ("NVIDIA GeForce RTX 3050", total_mem, os_free, "cuda", 0)
        dev_name, tot, free, dev_type, used = daemon.query_gpu_info()

        # Effective free should be os_free + (torch_reserved - torch_alloc)
        # = 50MB + 3800MB = 3850MB free, NOT 50MB!
        expected_free = os_free + (torch_reserved - torch_alloc)
        assert free == expected_free
        assert free > 3 * (1024 ** 3)  # Should report >3 GB free, preventing false cluster exhaustion
        assert used == max(torch_alloc, total_mem - expected_free)


def test_tui_in_process_coordinator_ws_ping_settings():
    """Verify start_in_process_coordinator configures ws_ping_interval=None and ws_ping_timeout=None."""
    import situs.tui as tui_mod
    with mock.patch("situs.tui.is_port_open", return_value=False), \
         mock.patch("uvicorn.Config") as mock_cfg, \
         mock.patch("uvicorn.Server"), \
         mock.patch("threading.Thread"):
        tui_mod.start_in_process_coordinator(host="127.0.0.1", port=8999)
        assert mock_cfg.called
        kwargs = mock_cfg.call_args[1]
        assert kwargs.get("ws_ping_interval") is None
        assert kwargs.get("ws_ping_timeout") is None
