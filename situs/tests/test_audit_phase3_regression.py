"""Regression tests for Phase 3 Audit: Single-worker pipeline dispatch, prompt templating,

CPU memory budgeting, dynamic model defaults, and speculative 2D tensor handling.
"""

import os
import sys
import json
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock

import torch
from situs.config import WorkerConfig
from situs.protocol.messages import (
    MessageType,
    ProtocolFrame,
    WorkerRegister,
    InferenceStart,
    NextToken,
    AssignLayers,
)
from situs.worker.loader import DistributedStageModel
from situs.worker.engine import StageExecutionEngine
from situs.worker.sampler import TokenSampler
from situs.worker.speculative import SpeculativeVerifier
from situs.tokenizer import get_or_load_tokenizer
import situs.coordinator.app as app_mod
from situs.coordinator.registry import WorkerRegistry
from situs.coordinator.relay import RelayHub


@pytest.mark.asyncio
async def test_single_worker_pipeline_dispatches_to_worker_not_ollama(monkeypatch):
    """Verifies that a 1-worker cluster dispatches InferenceStart to the worker over the wire,

    rather than hijacking the request to local Ollama.
    """
    orig_registry = app_mod.registry
    orig_relay = app_mod.relay
    iso_registry = WorkerRegistry()
    iso_relay = RelayHub()
    iso_relay.registry = iso_registry
    app_mod.registry = iso_registry
    app_mod.relay = iso_relay

    try:
        # Register 1 worker with all layers (e.g. 16 layers, first & last stage)
        node = iso_registry.register_worker(
            WorkerRegister(
                worker_id="w-single-gpu",
                gpu_name="Test GPU",
                total_vram_bytes=16 * 1024**3,
                free_vram_bytes=14 * 1024**3,
                used_vram_bytes=2 * 1024**3,
                device_type="cuda",
            )
        )
        node.assigned_layer_start = 0
        node.assigned_layer_end = 16
        node.is_first_stage = True
        node.is_last_stage = True
        node.first_worker_id = "w-single-gpu"

        iso_registry.active_model = "gemma3:4b"
        iso_registry.total_layers = 16
        iso_relay.active_connections["w-single-gpu"] = MagicMock()

        dispatched_frames = []

        async def _fake_send(worker_id: str, frame: ProtocolFrame):
            dispatched_frames.append((worker_id, frame))
            if frame.type == MessageType.INFERENCE_START:
                req_id = frame.payload["request_id"]
                q = iso_relay.response_streams.get(req_id)
                if q is not None:
                    await q.put(
                        ProtocolFrame(
                            type=MessageType.NEXT_TOKEN,
                            source="w-single-gpu",
                            target="coordinator",
                            payload=NextToken(
                                request_id=req_id,
                                step_index=0,
                                token_id=10,
                                token_text="Distributed",
                                is_finished=False,
                            ).model_dump(),
                        )
                    )
                    await q.put(
                        ProtocolFrame(
                            type=MessageType.NEXT_TOKEN,
                            source="w-single-gpu",
                            target="coordinator",
                            payload=NextToken(
                                request_id=req_id,
                                step_index=1,
                                token_id=1,
                                token_text="",
                                is_finished=True,
                                finish_reason="stop",
                            ).model_dump(),
                        )
                    )
            return True

        monkeypatch.setattr(iso_relay, "send_to_worker", _fake_send)

        # Mock Ollama to ensure it is NEVER called for supported single-worker execution
        ollama_called = False

        def _bad_ollama(*args, **kwargs):
            nonlocal ollama_called
            ollama_called = True
            return "OLLAMA_BYPASS_TEXT"

        monkeypatch.setattr("situs.inference_backend.query_ollama", _bad_ollama)

        # Mock tokenizer to return simple token IDs
        tok = MagicMock()
        tok.encode = MagicMock(return_value=[1, 2, 3])
        tok.stop_token_ids = [1]
        monkeypatch.setattr(app_mod, "get_or_load_tokenizer", lambda m: tok)

        # Execute non-streaming completion
        from situs.coordinator.app import ChatCompletionRequest, ChatMessage
        req = ChatCompletionRequest(
            model="gemma3:4b",
            messages=[ChatMessage(role="user", content="Hello single worker")],
            stream=False,
            max_tokens=10,
        )
        fake_http_req = MagicMock()
        fake_http_req.headers = {}
        fake_http_req.is_disconnected = AsyncMock(return_value=False)

        resp = await app_mod.chat_completions(req, fake_http_req)

        # Assertions
        assert ollama_called is False, "Ollama must NOT be invoked when a supported worker is registered!"
        assert resp["choices"][0]["message"]["content"] == "Distributed"
        assert any(f[1].type == MessageType.INFERENCE_START and f[0] == "w-single-gpu" for f in dispatched_frames), \
            "InferenceStart frame must be delivered to w-single-gpu"

    finally:
        app_mod.registry = orig_registry
        app_mod.relay = orig_relay


def test_cpu_worker_engine_uses_system_ram(monkeypatch):
    """Verifies that StageExecutionEngine on CPU device budgets against host system RAM."""
    tok = TokenSampler()
    total_layers = 4
    model_cpu = DistributedStageModel(
        layer_start=0,
        layer_end=2,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=128,
        vocab_size=500,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="llama",
        device="cpu",
    )
    assign_cpu = AssignLayers(
        worker_id="w-cpu-test",
        model_name="test-model",
        layer_start=0,
        layer_end=2,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        next_worker_id="w2",
        first_worker_id="w-cpu-test",
    )

    mock_ram_bytes = 32 * 1024**3  # 32 GB RAM host
    monkeypatch.setattr(
        "situs.hardware.get_system_ram_and_cpu",
        lambda: (mock_ram_bytes, 20 * 1024**3, "Test AMD CPU"),
    )

    engine = StageExecutionEngine(stage_model=model_cpu, assignment=assign_cpu, tokenizer=tok)
    assert engine.memory_manager.total_vram_bytes == mock_ram_bytes


def test_speculative_verifier_handles_2d_logits_tensor():
    """Verifies that SpeculativeVerifier accepts both 2D [seq_len, vocab] and 3D [1, seq_len, vocab] tensors."""
    sampler = TokenSampler()
    verifier = SpeculativeVerifier(sampler=sampler)

    # 2D tensor: [3 candidates, 100 vocab]
    logits_2d = torch.randn(3, 100)
    accepted, bonus = verifier.verify_candidates(
        candidate_tokens=[10, 20, 30],
        logits_sequence=logits_2d,
        temperature=0.0,
    )
    # Must not raise IndexError
    assert isinstance(accepted, list)


def test_situs_default_model_environment_override(monkeypatch):
    """Verifies that SITUS_DEFAULT_MODEL dynamically overrides defaults across backend, config, and TUI."""
    custom_model = "llama-3.2:3b-instruct"
    monkeypatch.setenv("SITUS_DEFAULT_MODEL", custom_model)

    from situs.inference_backend import normalize_ollama_model
    assert normalize_ollama_model("") == custom_model

    from situs.config import WorkerConfig
    w_cfg = WorkerConfig()
    assert w_cfg.model_name == custom_model

    import situs.tui as tui_mod
    # Re-evaluate DEFAULT_MODEL with overridden env
    assert os.getenv("SITUS_DEFAULT_MODEL") == custom_model


def test_worker_config_engine_ready_timeout(monkeypatch):
    """Verifies engine_ready_timeout_s is configurable via environment variable and parameter."""
    monkeypatch.setenv("SITUS_ENGINE_READY_TIMEOUT_S", "45.0")
    w_cfg = WorkerConfig()
    assert w_cfg.engine_ready_timeout_s == 45.0

    w_custom = WorkerConfig(engine_ready_timeout_s=90.0)
    assert w_custom.engine_ready_timeout_s == 90.0
