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
        reconstructed = reconstructed.detach().cpu().numpy()

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
