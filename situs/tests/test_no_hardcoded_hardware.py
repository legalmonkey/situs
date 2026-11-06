"""Tests verifying that no hardware names or device types are hardcoded in model catalogs or TUI displays."""

import pytest
from unittest.mock import patch
from situs.tui import DISTRIBUTED_MODELS_6GB_10GB, render_workers_table
from situs.hardware import get_live_gpu_vram


def test_no_hardcoded_devices_in_distributed_models():
    """Ensure no model in DISTRIBUTED_MODELS_6GB_10GB has hardcoded hardware names or states."""
    forbidden = ["RTX 3050", "RTX 4050", "RTX 3060", "GeForce", "Local CUDA GPU", "Ready on"]
    for m in DISTRIBUTED_MODELS_6GB_10GB:
        for key in ("name", "recommended_split", "desc"):
            val = m.get(key, "")
            for f in forbidden:
                assert f.lower() not in val.lower(), (
                    f"Model '{m['id']}' has hardcoded '{f}' in '{key}': '{val}'"
                )

    # Verify gemma3:4b entry specifically
    gemma = next(m for m in DISTRIBUTED_MODELS_6GB_10GB if m["id"] == "gemma3:4b")
    assert gemma["name"] == "Google Gemma 3 4B"
    assert "1.7 GB" in gemma["recommended_split"]


def test_render_workers_table_no_spoofing():
    """Verify CPU workers are not spoofed as RTX 3050 / RTX 4050."""
    workers = [
        {
            "id": "worker-1",
            "gpu": "Intel64 Family 6 Model 165 Stepping 2, GenuineIntel",
            "free_gb": 1.5,
            "total_gb": 16.0,
            "device": "cpu",
        },
        {
            "id": "worker-2",
            "gpu": "AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD",
            "free_gb": 2.0,
            "total_gb": 32.0,
            "device": "cpu",
        },
    ]
    table = render_workers_table(workers)
    # Ensure no rows contain RTX
    for row in table.columns[2]._cells:
        assert "RTX" not in str(row)
        assert "3050" not in str(row)
        assert "4050" not in str(row)


def test_get_live_gpu_vram_no_fake_fallback():
    """Ensure get_live_gpu_vram falls back to real system RAM instead of hardcoded 6GB when no GPU."""
    with patch("situs.hardware.HAS_TORCH", False), \
         patch("situs.hardware.query_nvidia_smi", return_value=None), \
         patch("situs.hardware.query_windows_cim_gpus", return_value=None), \
         patch("situs.hardware.get_system_ram_and_cpu", return_value=(16 * 1024**3, 10 * 1024**3, "Test CPU")):
        tot, used, free = get_live_gpu_vram()
        assert tot == 16 * 1024**3
        assert free == 10 * 1024**3
        assert tot != 6 * 1024**3
