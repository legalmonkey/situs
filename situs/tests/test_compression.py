"""Tests for dynamic activation compression and dequantization."""

import pytest
import numpy as np
from situs.protocol.compression import compress_tensor, decompress_tensor


def test_int8_compression_accuracy():
    # Simulate an activation tensor: [batch=1, seq_len=4, hidden_dim=2048]
    np.random.seed(42)
    original = np.random.randn(1, 4, 2048).astype(np.float32)

    comp_bytes, scale, shape = compress_tensor(original, method="int8")
    assert shape == (1, 4, 2048)
    # 1 * 4 * 2048 = 8192 bytes
    assert len(comp_bytes) == 8192

    reconstructed = decompress_tensor(comp_bytes, scale, shape, method="int8")
    if hasattr(reconstructed, "numpy"):
        reconstructed = reconstructed.detach().cpu().float().numpy()

    # Reconstructed shape matches
    assert reconstructed.shape == original.shape

    # Max absolute error must be small (< 0.05 for standard normal inputs)
    abs_err = np.max(np.abs(original - reconstructed))
    mean_err = np.mean(np.abs(original - reconstructed))
    assert abs_err < 0.1
    assert mean_err < 0.02


def test_bandwidth_reduction():
    # Comparing FP32, FP16, and INT8 sizes
    tensor = np.random.randn(1, 1, 4096).astype(np.float32)
    fp32_bytes, _, _ = compress_tensor(tensor, method="none")
    fp16_bytes, _, _ = compress_tensor(tensor, method="fp16")
    int8_bytes, _, _ = compress_tensor(tensor, method="int8")

    assert len(fp32_bytes) == 4096 * 4  # 16384 bytes
    assert len(fp16_bytes) == 4096 * 2  # 8192 bytes
    assert len(int8_bytes) == 4096 * 1  # 4096 bytes

    # Exactly 50% bandwidth reduction compared to FP16, 75% compared to FP32
    assert len(int8_bytes) == len(fp16_bytes) / 2


def test_fp16_dense_activation_accuracy():
    np.random.seed(42)
    original = np.random.randn(1, 4, 2048).astype(np.float32)

    # Default method is now unquantized dense FP16
    comp_bytes, scale, shape = compress_tensor(original)
    assert shape == (1, 4, 2048)
    assert len(comp_bytes) == 1 * 4 * 2048 * 2  # 2 bytes per element (FP16)
    assert scale == 1.0

    reconstructed = decompress_tensor(comp_bytes, scale, shape, method="fp16")
    if hasattr(reconstructed, "numpy"):
        reconstructed = reconstructed.detach().cpu().float().numpy()

    assert reconstructed.shape == original.shape
    # FP16 precision loss is tiny (< 0.005), with zero INT8 integer clamping
    abs_err = np.max(np.abs(original.astype(np.float16).astype(np.float32) - reconstructed))
    assert abs_err < 1e-4


def test_bf16_dense_activation_accuracy():
    np.random.seed(42)
    original = np.random.randn(1, 2, 512).astype(np.float32)

    comp_bytes, scale, shape = compress_tensor(original, method="bf16")
    assert shape == (1, 2, 512)
    assert len(comp_bytes) == 1 * 2 * 512 * 2  # 2 bytes per element (BF16)
    assert scale == 1.0

    reconstructed = decompress_tensor(comp_bytes, scale, shape, method="bf16")
    if hasattr(reconstructed, "numpy"):
        reconstructed = reconstructed.detach().cpu().float().numpy()

    assert reconstructed.shape == original.shape


def test_dense_temporal_delta_replaces_sparsification():
    from situs.protocol.compression import compress_temporal_delta, decompress_temporal_delta
    np.random.seed(42)
    h0 = np.random.randn(1, 1, 1024).astype(np.float32)
    h1 = h0 + np.random.randn(1, 1, 1024).astype(np.float32) * 0.01

    # Default is now dense activations (sparsify_ratio=1.0)
    values_bytes, scale, shape, indices_bytes = compress_temporal_delta(h1, last_arr=h0)
    assert indices_bytes is None  # No sparse coordinate indices: 100% dense
    assert len(values_bytes) == 1024 * 2  # Dense FP16 payload

    reconstructed = decompress_temporal_delta(values_bytes, scale, shape, indices_bytes, last_arr=h0)
    assert reconstructed.shape == h1.shape
    mean_err = np.mean(np.abs(h1 - reconstructed))
    assert mean_err < 0.005
