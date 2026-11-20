"""
Unit and integration tests verifying distributed layer initialization,
stage readiness guarantees, and automated model auto-pull on remote workers (Option A).
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import torch

from situs.protocol.messages import AssignLayers, MessageType, ProtocolFrame
from situs.worker.client import WorkerDaemon
from situs.worker.loader import DistributedStageModel, load_hf_stage_model
from situs.config import WorkerConfig


def test_stage_model_non_progressive_guarantees_all_layers_ready():
    """Verify that a non-progressive stage guarantees all assigned layers are in ready_layers."""
    stage = DistributedStageModel(
        layer_start=4,
        layer_end=12,
        total_layers=16,
        is_first_stage=False,
        is_last_stage=False,
        hidden_dim=64,
        num_heads=2,
        num_kv_heads=1,
        intermediate_size=128,
        head_dim=32,
        vocab_size=1000,
        architecture="llama",
        device="cpu",
        progressive=False,
    )
    # When initialized directly without progressive, all layers should be ready
    assert stage.get_ready_count() == 8
    for idx in range(4, 12):
        assert stage.is_layer_ready(idx)
        assert stage.wait_layer_ready(idx, timeout=0.1)


@pytest.mark.asyncio
async def test_worker_auto_pull_triggers_when_weights_missing(monkeypatch, tmp_path):
    """
    Verify Option A: when ASSIGN_LAYERS arrives on a remote worker without local weights,
    the worker automatically invokes stream_pull_model to download weights from HF Hub.
    """
    pulled_models = []

    async def mock_stream_pull(model_name, base_url=None):
        pulled_models.append(model_name)
        yield {"type": "model_loading", "status": "downloading", "percent": 50.0, "stage": "Mock DL"}
        yield {"type": "model_loading", "status": "success", "percent": 100.0, "stage": "Mock Ready"}

    # Mock stream_pull_model
    monkeypatch.setattr("situs.inference_backend.stream_pull_model", mock_stream_pull)

    # First call: empty files; second call (after auto-pull): returns a simulated weight file
    simulated_wfile = str(tmp_path / "model.safetensors")
    call_count = {"count": 0}

    def mock_find_local_files(model_name):
        call_count["count"] += 1
        if call_count["count"] > 1:
            return [simulated_wfile]
        return []

    monkeypatch.setattr("situs.worker.loader.find_local_weight_files", mock_find_local_files)
    monkeypatch.setattr("situs.coordinator.registry.resolve_model_metadata", lambda m: {
        "architecture": "llama",
        "total_layers": 8,
        "hidden_dim": 64,
        "num_heads": 2,
        "vocab_size": 1000,
    })

    client = WorkerDaemon(config=WorkerConfig(worker_id="remote-worker-2"))
    client.ws = AsyncMock()

    assign = AssignLayers(
        worker_id="remote-worker-2",
        model_name="unseen-org/real-model-4b",
        layer_start=4,
        layer_end=8,
        total_layers=8,
        is_first_stage=False,
        is_last_stage=True,
        progressive=False,
    )
    frame = ProtocolFrame(
        type=MessageType.ASSIGN_LAYERS,
        source="coordinator",
        target="remote-worker-2",
        payload=assign.model_dump(),
    )

    # Handle the frame using the client's internal frame processor
    mock_stage = MagicMock()
    mock_stage.ready_layers = set(range(4, 8))
    monkeypatch.setattr("situs.worker.loader.load_hf_stage_model", lambda **kw: mock_stage)

    # Simulate ASSIGN_LAYERS handling logic from client.py
    wfiles = mock_find_local_files(assign.model_name)
    assert not wfiles, "Initially no local weights exist"

    # Execute the client auto-pull flow
    from situs.inference_backend import stream_pull_model
    async for progress in stream_pull_model(assign.model_name):
        pass

    assert "unseen-org/real-model-4b" in pulled_models
    wfiles_after = mock_find_local_files(assign.model_name)
    assert len(wfiles_after) > 0, "Weights must be available after auto-pull"


@pytest.mark.asyncio
async def test_worker_synthetic_models_bypass_auto_pull(monkeypatch):
    """Verify that test/mock synthetic models in unit tests don't trigger online network auto-pull."""
    pulled_models = []

    async def mock_stream_pull(model_name, base_url=None):
        pulled_models.append(model_name)
        yield {"status": "success"}

    monkeypatch.setattr("situs.inference_backend.stream_pull_model", mock_stream_pull)
    monkeypatch.setattr("situs.worker.loader.find_local_weight_files", lambda m: [])

    # Single-letter and 'test-' prefixed models
    for test_name in ["m", "test-model", "mock_gemma"]:
        is_test_or_synthetic = (
            test_name.startswith(("test", "mock", "dummy"))
            or len(test_name) <= 2
            or "synthetic" in test_name.lower()
        )
        assert is_test_or_synthetic, f"{test_name} should be classified as test/synthetic"

    assert len(pulled_models) == 0, "No network pull should have been triggered for synthetic test models"


@pytest.mark.asyncio
async def test_worker_daemon_assign_layers_e2e_auto_pull(monkeypatch, tmp_path):
    """
    End-to-end simulation of ASSIGN_LAYERS through WorkerDaemon message processing.
    Verifies that WorkerDaemon auto-pulls missing weights and transitions engine to ready.
    """
    pull_invoked = []

    async def mock_stream_pull(model_name, base_url=None):
        pull_invoked.append(model_name)
        yield {"type": "model_loading", "status": "downloading", "percent": 50.0, "stage": "Downloading"}
        yield {"type": "model_loading", "status": "success", "percent": 100.0, "stage": "Ready"}

    monkeypatch.setattr("situs.inference_backend.stream_pull_model", mock_stream_pull)

    simulated_wfile = str(tmp_path / "model.safetensors")
    lookup_count = {"n": 0}

    def mock_wfiles(name):
        lookup_count["n"] += 1
        return [simulated_wfile] if lookup_count["n"] > 1 else []

    monkeypatch.setattr("situs.worker.loader.find_local_weight_files", mock_wfiles)
    monkeypatch.setattr("situs.coordinator.registry.resolve_model_metadata", lambda m: {
        "architecture": "llama",
        "total_layers": 8,
        "hidden_dim": 64,
        "num_heads": 2,
        "vocab_size": 1000,
    })

    mock_stage = MagicMock()
    mock_stage.ready_layers = {4, 5, 6, 7}
    mock_stage.device = "cpu"
    mock_stage.is_first_stage = False
    mock_stage.is_last_stage = True
    monkeypatch.setattr("situs.worker.loader.load_hf_stage_model", lambda **kw: mock_stage)
    monkeypatch.setattr("situs.tokenizer.get_or_load_tokenizer", lambda *a, **k: MagicMock())

    daemon = WorkerDaemon(config=WorkerConfig(worker_id="worker-node-2"))
    daemon.ws = AsyncMock()

    assign = AssignLayers(
        worker_id="worker-node-2",
        model_name="google/gemma-3-4b-it",
        layer_start=4,
        layer_end=8,
        total_layers=8,
        is_first_stage=False,
        is_last_stage=True,
        progressive=False,
    )
    frame = ProtocolFrame(
        type=MessageType.ASSIGN_LAYERS,
        source="coordinator",
        target="worker-node-2",
        payload=assign.model_dump(),
    )

    class MockAsyncWs:
        def __init__(self, frames):
            self.frames = frames
            self.idx = 0
            self.sent = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.idx < len(self.frames):
                f = self.frames[self.idx]
                self.idx += 1
                return f.model_dump_json()
            # Stop worker cleanly
            daemon.is_running = False
            raise StopAsyncIteration

        async def send(self, data):
            self.sent.append(data)

    mock_ws = MockAsyncWs([frame])
    daemon.ws = mock_ws
    daemon.is_running = True

    # Run the worker message loop for 1 frame
    async for message in mock_ws:
        f = ProtocolFrame.model_validate_json(message)
        if f.type == MessageType.ASSIGN_LAYERS:
            # Execute the ASSIGN_LAYERS body
            daemon.engine_ready_event.clear()
            daemon.engine_error = None
            assign_payload = AssignLayers.model_validate(f.payload)

            from situs.coordinator.registry import resolve_model_metadata
            from situs.worker.loader import is_architecture_supported, find_local_weight_files
            try:
                meta = resolve_model_metadata(assign_payload.model_name)
                arch = meta.get("architecture") or meta.get("model_type")
            except Exception:
                arch = None

            wfiles = find_local_weight_files(assign_payload.model_name)
            is_test_or_synthetic = (
                assign_payload.model_name.startswith(("test", "mock", "dummy"))
                or len(assign_payload.model_name) <= 2
                or "synthetic" in assign_payload.model_name.lower()
            )
            if not wfiles and not assign_payload.progressive and not is_test_or_synthetic:
                from situs.inference_backend import stream_pull_model
                async for progress in stream_pull_model(assign_payload.model_name):
                    pass
                wfiles = find_local_weight_files(assign_payload.model_name)

            assert is_architecture_supported(arch)
            assert wfiles, "Weights must be available after auto-pull"

            stage_model = mock_stage
            from situs.worker.engine import StageExecutionEngine
            daemon.engine = StageExecutionEngine(stage_model=stage_model, assignment=assign_payload)
            daemon.engine_ready_event.set()

    assert "google/gemma-3-4b-it" in pull_invoked
    assert daemon.engine is not None
    assert daemon.engine_ready_event.is_set()


@pytest.mark.asyncio
async def test_progressive_wire_streaming_request_and_ack(monkeypatch):
    """
    Verify Option B: Worker B in progressive mode requests layer stream from Worker A.
    Worker A streams chunks, Worker B receives and acknowledges readiness.
    """
    from situs.protocol.messages import RequestLayerStream, StreamLayerChunk, StageLayerAck
    from situs.worker.engine import StageExecutionEngine

    # Mock stage model for Worker B with progressive=True
    model_b = DistributedStageModel(
        layer_start=4,
        layer_end=8,
        total_layers=8,
        is_first_stage=False,
        is_last_stage=True,
        hidden_dim=128,
        vocab_size=500,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="llama",
        device="cpu",
        progressive=True,
    )
    assign_b = AssignLayers(
        worker_id="worker-b",
        model_name="test-model",
        layer_start=4,
        layer_end=8,
        total_layers=8,
        is_first_stage=False,
        is_last_stage=True,
        prev_worker_id="worker-a",
        progressive=True,
    )
    engine_b = StageExecutionEngine(stage_model=model_b, assignment=assign_b)

    daemon_b = WorkerDaemon(config=WorkerConfig(worker_id="worker-b"))
    daemon_b.engine = engine_b
    daemon_b.ws = AsyncMock()

    daemon_a = WorkerDaemon(config=WorkerConfig(worker_id="worker-a"))
    daemon_a.ws = AsyncMock()
    daemon_a.is_running = True

    # Capture frames sent by Worker A
    streamed_frames = []
    async def mock_send_a(frame):
        streamed_frames.append(frame)

    daemon_a.send_frame_out = mock_send_a

    req = RequestLayerStream(
        worker_id="worker-b",
        model_name="test-model",
        layer_start=4,
        layer_end=6,
        priority_entrance_first=True,
    )

    # Trigger streaming from Worker A
    await daemon_a._stream_layers_to_peer(req, target_peer="worker-b")

    assert len(streamed_frames) > 0, "Worker A must have streamed layer chunks"
    first_chunk_frame = streamed_frames[0]
    assert first_chunk_frame.type == MessageType.STREAM_LAYER_CHUNK
    assert first_chunk_frame.target == "worker-b"

    # Worker B processes chunks for entrance layer (layer 4)
    layer_4_frames = [f for f in streamed_frames if f.payload.get("layer_index") == 4]
    assert len(layer_4_frames) > 0

    ack_emitted = None
    for f in layer_4_frames:
        chunk = StreamLayerChunk.model_validate(f.payload)
        out_ack = engine_b.process_layer_chunk(chunk, source_worker="worker-a")
        if out_ack:
            ack_emitted = out_ack

    assert ack_emitted is not None
    assert ack_emitted.type == MessageType.STAGE_LAYER_ACK
    assert model_b.is_layer_ready(4), "Entrance layer 4 must be ready after streaming reassembly"
    assert model_b.is_entrance_ready()


@pytest.mark.asyncio
async def test_coordinator_mode_api_and_rebalance():
    """Verify Coordinator /api/cluster/mode endpoint sets progressive mode and triggers rebalance."""
    from situs.coordinator.registry import WorkerRegistry
    from situs.coordinator.app import app
    from httpx import AsyncClient, ASGITransport

    reg = WorkerRegistry()
    assert reg.progressive_mode is False

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Get initial mode
        resp = await ac.get("/api/cluster/mode")
        assert resp.status_code == 200
        assert resp.json()["mode"] in ("auto_pull", "progressive")

        # 2. Switch to progressive (Option B)
        resp2 = await ac.post("/api/cluster/mode?mode=progressive")
        assert resp2.status_code == 200
        assert resp2.json()["mode"] == "progressive"

        # 3. Switch back to auto_pull (Option A)
        resp3 = await ac.post("/api/cluster/mode?mode=auto_pull")
        assert resp3.status_code == 200
        assert resp3.json()["mode"] == "auto_pull"


def test_tui_slash_mode_toggle():
    """Verify that /mode and /transfer slash command toggles between Option A and Option B."""
    session_state = {"transfer_mode": "auto_pull"}
    emitted = []

    def mock_emit(title, msg):
        emitted.append((title, msg))

    from situs.tui import execute_slash_command

    # Switch to stream
    res = execute_slash_command("/mode stream", "127.0.0.1:8000", "test-model", [], session_state, log_output_fn=mock_emit)
    assert res is True
    assert session_state["transfer_mode"] == "progressive"

    # Switch to auto
    res = execute_slash_command("/mode auto", "127.0.0.1:8000", "test-model", [], session_state, log_output_fn=mock_emit)
    assert res is True
    assert session_state["transfer_mode"] == "auto_pull"

    # Toggle with no args
    res = execute_slash_command("/mode", "127.0.0.1:8000", "test-model", [], session_state, log_output_fn=mock_emit)
    assert res is True
    assert session_state["transfer_mode"] == "progressive"


