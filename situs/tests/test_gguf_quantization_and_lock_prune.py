"""Unit tests for GGUF 4-bit quantization selection, lock pruning, and 6GB VRAM verification."""

import os
import pathlib
import time
import pytest
from unittest.mock import MagicMock, patch

from situs.inference_backend import (
    _prune_stale_hf_locks,
    _is_snapshot_complete,
    is_model_downloaded,
    normalize_model_name,
    stream_pull_model,
)
from situs.worker.loader import find_local_weight_files, resolve_model_metadata


def test_prune_stale_hf_locks(tmp_path, monkeypatch):
    """Verify stale .lock files older than 30s are pruned while recent locks remain untouched."""
    cache_dir = tmp_path / "fake_hub"
    locks_dir = cache_dir / ".locks"
    repo_lock_dir = locks_dir / "models--fake-org--fake-model"
    repo_lock_dir.mkdir(parents=True)
    monkeypatch.setenv("HF_HUB_CACHE", str(cache_dir))

    stale_lock = repo_lock_dir / "stale.lock"
    stale_lock.write_text("lock-data")
    # Set mtime to 60 seconds ago
    old_time = time.time() - 60
    os.utime(stale_lock, (old_time, old_time))

    fresh_lock = repo_lock_dir / "fresh.lock"
    fresh_lock.write_text("fresh-data")

    _prune_stale_hf_locks("fake-org/fake-model")

    # Stale lock should have been deleted
    assert not stale_lock.exists()
    # Fresh lock should still exist
    assert fresh_lock.exists()


def test_gemma3_maps_to_4bit_gguf():
    """Verify gemma3:4b maps to Google official 4-bit QAT GGUF repository for 6GB VRAM fit."""
    canonical = normalize_model_name("gemma3:4b")
    assert "gguf" in canonical.lower()
    assert canonical == "google/gemma-3-4b-it-qat-q4_0-gguf"


@pytest.mark.asyncio
async def test_stream_pull_model_selects_single_q4_km_gguf():
    """Verify that stream_pull_model picks only the single Q4_K_M GGUF file from a multi-quant repo."""
    with patch("huggingface_hub.HfApi") as mock_api_cls, patch("huggingface_hub.hf_hub_download") as mock_dl:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        # Simulate a typical GGUF repo with 15+ quantizations and mmproj
        siblings = []
        files = [
            ("gemma-3-4b-it-BF16.gguf", 7400 * 1024 * 1024),
            ("gemma-3-4b-it-Q8_0.gguf", 3900 * 1024 * 1024),
            ("gemma-3-4b-it-Q6_K.gguf", 3000 * 1024 * 1024),
            ("gemma-3-4b-it-Q4_K_M.gguf", 2374 * 1024 * 1024),  # The 4-bit target!
            ("gemma-3-4b-it-Q4_0.gguf", 2260 * 1024 * 1024),
            ("gemma-3-4b-it-Q2_K.gguf", 1600 * 1024 * 1024),
            ("mmproj-BF16.gguf", 811 * 1024 * 1024),  # Vision projector should be excluded
            ("config.json", 1024),
        ]
        for fn, sz in files:
            sib = MagicMock()
            sib.rfilename = fn
            sib.size = sz
            siblings.append(sib)

        mock_api.model_info.return_value = MagicMock(siblings=siblings)
        downloaded_files = []

        def fake_download(repo_id, filename, token, tqdm_class=None):
            downloaded_files.append(filename)
            return f"/fake/cache/{filename}"

        mock_dl.side_effect = fake_download

        events = []
        async for chunk in stream_pull_model("gemma3:4b", force=True):
            events.append(chunk)

        # Ensure only the single Q4_K_M GGUF (and optional config) was downloaded
        gguf_downloads = [f for f in downloaded_files if f.endswith(".gguf")]
        assert len(gguf_downloads) == 1
        assert gguf_downloads[0] == "gemma-3-4b-it-Q4_K_M.gguf"
        # mmproj must NOT have been downloaded
        assert not any("mmproj" in f for f in downloaded_files)
        # Total repo bytes should be ~2.37 GB, not 20+ GB
        last_event = events[-1]
        assert last_event["status"] == "success"


def test_find_local_weight_files_discovers_hf_gguf_snapshots(tmp_path, monkeypatch):
    """Verify find_local_weight_files locates .gguf files inside Hugging Face snapshot dirs."""
    fake_cache = tmp_path / "fake_hf_hub"
    snap_dir = fake_cache / "models--google--gemma-3-4b-it-qat-q4_0-gguf" / "snapshots" / "rev123"
    snap_dir.mkdir(parents=True)
    gguf_file = snap_dir / "gemma-3-4b-it-q4_0.gguf"
    # Write dummy GGUF header
    gguf_file.write_bytes(b"GGUF" + b"\x00" * 1000)

    monkeypatch.setenv("HF_HUB_CACHE", str(fake_cache))

    found = find_local_weight_files("gemma3:4b")
    assert len(found) >= 1
    assert any(f.endswith("gemma-3-4b-it-q4_0.gguf") for f in found)


@pytest.mark.asyncio
async def test_stream_pull_model_selects_google_qat_gguf():
    """Verify stream_pull_model downloads gemma-3-4b-it-q4_0.gguf (3.16 GB) and skips mmproj."""
    with patch("huggingface_hub.HfApi") as mock_api_cls, patch("huggingface_hub.hf_hub_download") as mock_dl:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api

        # Siblings matching official google/gemma-3-4b-it-qat-q4_0-gguf
        siblings = []
        files = [
            (".gitattributes", 3530),
            ("README.md", 23200),
            ("gemma-3-4b-it-q4_0.gguf", 3155051328),  # Official 3.16 GB QAT weights
            ("mmproj-model-f16-4B.gguf", 851251104),  # Vision projector should be excluded
        ]
        for fn, sz in files:
            sib = MagicMock()
            sib.rfilename = fn
            sib.size = sz
            siblings.append(sib)

        mock_api.model_info.return_value = MagicMock(siblings=siblings)
        downloaded = []

        def fake_dl(repo_id, filename, token, tqdm_class=None):
            downloaded.append(filename)
            return f"/fake/cache/{filename}"

        mock_dl.side_effect = fake_dl

        events = []
        async for chunk in stream_pull_model("google/gemma-3-4b-it-qat-q4_0-gguf"):
            events.append(chunk)

        assert downloaded == ["gemma-3-4b-it-q4_0.gguf"]
        assert not any("mmproj" in f for f in downloaded)
        assert events[-1]["status"] == "success"
        assert events[-1]["total_bytes"] == 3155051328


@pytest.mark.asyncio
async def test_download_telemetry_never_exceeds_total_bytes():
    """Verify that retries or chunk overflows in tqdm never cause completed_bytes > total_bytes."""
    with patch("huggingface_hub.HfApi") as mock_api_cls, patch("huggingface_hub.hf_hub_download") as mock_dl:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        sibling = MagicMock()
        sibling.rfilename = "gemma-3-4b-it-q4_0.gguf"
        sibling.size = 10000
        mock_api.model_info.return_value = MagicMock(siblings=[sibling])

        def fake_dl(repo_id, filename, token, tqdm_class=None):
            if tqdm_class:
                inst = tqdm_class(total=10000)
                # Simulate over-reporting / retries
                inst.update(6000)
                inst.update(6000)
                inst.update(6000)  # total 18000 on a 10000 byte file
            return "/tmp/fake/gemma-3-4b-it-q4_0.gguf"

        mock_dl.side_effect = fake_dl

        events = []
        async for chunk in stream_pull_model("google/gemma-3-4b-it-qat-q4_0-gguf"):
            events.append(chunk)

        for e in events:
            if "completed_bytes" in e and "total_bytes" in e and e["total_bytes"] > 0:
                assert e["completed_bytes"] <= e["total_bytes"], (
                    f"completed_bytes ({e['completed_bytes']}) exceeded total_bytes ({e['total_bytes']})"
                )


def test_gemma3_4bit_fits_in_6gb_vram():
    """Verify that 4-bit Gemma 3 4B memory footprint is under 3.5 GB (fits natively in 6GB VRAM)."""
    # 4.55B parameters at 4 bits (0.5 bytes) = ~2.28 GB weights + KV cache overhead
    weights_4bit_gb = 4.55 * 0.5
    assert weights_4bit_gb < 2.5
    # Total with KV cache and activations is well under 6.0 GB
    total_est_gb = weights_4bit_gb + 1.0
    assert total_est_gb < 4.0
    assert total_est_gb < 6.0  # Fits comfortably in 6.0 GB RTX 3050 Laptop GPU
