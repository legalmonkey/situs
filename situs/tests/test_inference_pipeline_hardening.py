"""Unit and integration regression tests verifying inference pipeline hardening fixes:
1. P2P speculative verification across 3 workers ensures only the last stage sends token frames to coordinator (0 duplicates).
2. Relay reverse routing preserves explicit intermediate targets.
3. Native server restart accesses self.assignment without AttributeError.
4. Missing engine emits explicit ERROR frame on NEXT_TOKEN, SPECULATIVE_CANDIDATES, and SPECULATIVE_VERIFIED.
5. Intermediate worker session token history synchronization.
"""

import json
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import pytest

from situs.config import WorkerConfig
from situs.protocol.messages import (
    MessageType,
    ProtocolFrame,
    AssignLayers,
    InferenceStart,
    NextToken,
    SpeculativeCandidates,
    SpeculativeVerified,
)
from situs.coordinator.relay import RelayHub
from situs.coordinator.registry import WorkerRegistry
from situs.worker.client import WorkerDaemon
from situs.worker.engine import StageExecutionEngine


class MockAsyncWs:
    """Mock websocket yielding pre-configured message frames."""
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent_messages = []

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.messages:
            return self.messages.pop(0)
        raise StopAsyncIteration

    async def send(self, data):
        self.sent_messages.append(data)


class MockConnectManager:
    """Wraps MockAsyncWs to emulate websockets.connect async iterator."""
    def __init__(self, ws):
        self.ws = ws
        self.yielded = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.yielded:
            self.yielded = True
            return self.ws
        raise StopAsyncIteration


@pytest.mark.asyncio
async def test_speculative_verification_p2p_three_workers_no_coordinator_duplicates():
    """Verify that in a 3-worker pipeline, intermediate workers forwarding SPECULATIVE_VERIFIED
    over P2P NEVER send duplicate copies to the coordinator. Only the last stage sends a copy.
    """
    cfg0 = WorkerConfig(worker_id="w-0")
    cfg1 = WorkerConfig(worker_id="w-1")
    cfg2 = WorkerConfig(worker_id="w-2")

    worker0 = WorkerDaemon(cfg0)
    worker1 = WorkerDaemon(cfg1)
    worker2 = WorkerDaemon(cfg2)

    # Mock engines
    mock_engine0 = MagicMock()
    mock_engine0.is_first_stage.return_value = True
    mock_engine0.is_last_stage.return_value = False
    worker0.engine = mock_engine0

    mock_engine1 = MagicMock()
    mock_engine1.is_first_stage.return_value = False
    mock_engine1.is_last_stage.return_value = False
    worker1.engine = mock_engine1

    mock_engine2 = MagicMock()
    mock_engine2.is_first_stage.return_value = False
    mock_engine2.is_last_stage.return_value = True
    worker2.engine = mock_engine2

    # Mock WebSockets
    worker0.ws = AsyncMock()
    worker1.ws = AsyncMock()
    worker2.ws = AsyncMock()

    # Mock P2P nodes
    p2p_node_2_to_1 = MagicMock()
    p2p_node_2_to_1.is_connected = True
    p2p_node_2_to_1.send_frame.return_value = True
    worker2.p2p_nodes["w-1"] = p2p_node_2_to_1

    p2p_node_1_to_0 = MagicMock()
    p2p_node_1_to_0.is_connected = True
    p2p_node_1_to_0.send_frame.return_value = True
    worker1.p2p_nodes["w-0"] = p2p_node_1_to_0

    # Step 1: Worker 2 (last stage) emits SPECULATIVE_VERIFIED towards w-1
    ver_payload = SpeculativeVerified(
        request_id="req-spec-p2p",
        step_index=1,
        accepted_tokens=[101, 102],
        accepted_texts=[" tok1", " tok2"],
        is_finished=False,
    )
    frame_w2 = ProtocolFrame(
        type=MessageType.SPECULATIVE_VERIFIED,
        source="w-2",
        target="w-1",
        payload=ver_payload.model_dump(),
    )
    await worker2.send_frame_out(frame_w2)

    # Worker 2 sent P2P to w-1 AND sent exactly 1 copy to coordinator (because is_last_stage == True)
    assert p2p_node_2_to_1.send_frame.call_count == 1
    assert worker2.ws.send.call_count == 1
    sent_to_coord = json.loads(worker2.ws.send.call_args[0][0])
    assert sent_to_coord["target"] == "coordinator"
    assert sent_to_coord["type"] == MessageType.SPECULATIVE_VERIFIED

    # Step 2: Worker 1 (intermediate stage) forwards SPECULATIVE_VERIFIED towards w-0
    frame_w1 = ProtocolFrame(
        type=MessageType.SPECULATIVE_VERIFIED,
        source="w-1",
        target="w-0",
        payload=ver_payload.model_dump(),
    )
    await worker1.send_frame_out(frame_w1)

    # Worker 1 sent P2P to w-0, but MUST NOT send any copy to coordinator!
    assert p2p_node_1_to_0.send_frame.call_count == 1
    assert worker1.ws.send.call_count == 0, (
        f"Intermediate worker sent duplicate frame to coordinator! Calls: {worker1.ws.send.call_args_list}"
    )


@pytest.mark.asyncio
async def test_relay_reverse_routing_preserves_explicit_intermediate_targets():
    """Verify that RelayHub does not overwrite target_worker with pipeline[0]
    when frame.target is already specified.
    """
    mock_registry = MagicMock(spec=WorkerRegistry)
    mock_registry.get_pipeline_order.return_value = [
        MagicMock(worker_id="worker-0"),
        MagicMock(worker_id="worker-1"),
        MagicMock(worker_id="worker-2"),
    ]

    relay = RelayHub(registry=mock_registry)
    relay.send_to_worker = AsyncMock()

    # Case 1: Frame explicitly targeted at intermediate worker-1
    frame_explicit = ProtocolFrame(
        type=MessageType.SPECULATIVE_VERIFIED,
        source="worker-2",
        target="worker-1",
        payload={"request_id": "test-req-rev"},
    )
    await relay.route_frame(frame_explicit)
    relay.send_to_worker.assert_awaited_once_with("worker-1", frame_explicit)

    relay.send_to_worker.reset_mock()

    # Case 2: Frame target is empty/None -> defaults to pipeline[0] (worker-0)
    frame_no_target = ProtocolFrame(
        type=MessageType.SPECULATIVE_VERIFIED,
        source="worker-2",
        target="",
        payload={"request_id": "test-req-rev"},
    )
    await relay.route_frame(frame_no_target)
    relay.send_to_worker.assert_awaited_once_with("worker-0", frame_no_target)

    relay.send_to_worker.reset_mock()

    # Case 3: Frame target is 'coordinator' -> early return, never routed to any worker
    frame_coord = ProtocolFrame(
        type=MessageType.SPECULATIVE_VERIFIED,
        source="worker-2",
        target="coordinator",
        payload={"request_id": "test-req-rev"},
    )
    await relay.route_frame(frame_coord)
    assert relay.send_to_worker.call_count == 0


@pytest.mark.asyncio
async def test_native_server_restart_accesses_assignment_without_error():
    """Verify that WorkerDaemon initializes self.assignment, records it on ASSIGN_LAYERS,
    and native server auto-restart does not raise AttributeError.
    """
    cfg = WorkerConfig(worker_id="w-single", coordinator_url="ws://127.0.0.1:9999/ws")
    worker = WorkerDaemon(cfg)

    # Initial state: self.assignment is None (no AttributeError on access)
    assert hasattr(worker, "assignment")
    assert worker.assignment is None

    # Assign layers message
    assign = AssignLayers(
        worker_id="w-single",
        model_name="gemma-2-2b",
        layer_start=0,
        layer_end=26,
        total_layers=26,
        is_first_stage=True,
        is_last_stage=True,
    )
    assign_frame = ProtocolFrame(
        type=MessageType.ASSIGN_LAYERS,
        source="coordinator",
        target="w-single",
        payload=assign.model_dump(),
    )
    stop_frame = ProtocolFrame(
        type=MessageType.WORKER_REMOVED,
        source="coordinator",
        target="w-single",
        payload={},
    )

    mock_ws = MockAsyncWs([assign_frame.model_dump_json(), stop_frame.model_dump_json()])

    with patch("websockets.connect", side_effect=lambda *args, **kwargs: MockConnectManager(mock_ws)), \
         patch("situs.coordinator.registry.resolve_model_metadata", return_value={"total_layers": 26}), \
         patch("situs.worker.loader.find_local_weight_files", return_value=["dummy.safetensors"]), \
         patch("situs.worker.loader.load_hf_stage_model", return_value=MagicMock()):
        await worker.run()

    assert worker.assignment is not None
    assert worker.assignment.model_name == "gemma-2-2b"
    assert worker.assignment.is_first_stage is True
    assert worker.assignment.is_last_stage is True

    # Now verify native server auto-restart accessing self.assignment
    mock_native = MagicMock()
    mock_native.is_ready.return_value = False
    mock_native.start = MagicMock()
    async def mock_stream(msg):
        if False:
            yield None
    mock_native.stream_inference = mock_stream
    worker.native_server = mock_native

    start_msg = InferenceStart(
        request_id="req-restart-test",
        model_name="gemma-2-2b",
        input_ids=[1, 2, 3],
        max_new_tokens=5,
    )
    # Must not raise AttributeError: 'WorkerDaemon' object has no attribute 'assignment'
    await worker._run_single_worker_loop(start_msg)
    mock_native.start.assert_called_once_with("gemma-2-2b")


@pytest.mark.asyncio
async def test_missing_engine_emits_explicit_error_frame():
    """Verify that when engine is None, NEXT_TOKEN, SPECULATIVE_CANDIDATES, and
    SPECULATIVE_VERIFIED frames emit an explicit ERROR frame instead of being dropped.
    """
    cfg = WorkerConfig(worker_id="w-no-engine", engine_ready_timeout_s=0.01)
    worker = WorkerDaemon(cfg)
    worker.engine = None
    worker.engine_ready_event.set()  # immediate ready check
    worker.ws = AsyncMock()

    # 1. P2P NEXT_TOKEN via handle_incoming_frame
    tok_frame = ProtocolFrame(
        type=MessageType.NEXT_TOKEN,
        source="peer",
        target="w-no-engine",
        payload={"request_id": "req-tok-1", "token_id": 42, "step_index": 1},
    )
    await worker.handle_incoming_frame(tok_frame.model_dump_json())
    assert worker.ws.send.call_count == 1
    err_p2p = json.loads(worker.ws.send.call_args[0][0])
    assert err_p2p["type"] == MessageType.ERROR
    assert err_p2p["payload"]["request_id"] == "req-tok-1"

    worker.ws.send.reset_mock()

    # 2. P2P SPECULATIVE_CANDIDATES via handle_incoming_frame
    cand_frame = ProtocolFrame(
        type=MessageType.SPECULATIVE_CANDIDATES,
        source="peer",
        target="w-no-engine",
        payload={"request_id": "req-cand-1", "candidate_tokens": [10, 20]},
    )
    await worker.handle_incoming_frame(cand_frame.model_dump_json())
    assert worker.ws.send.call_count == 1
    err_cand_p2p = json.loads(worker.ws.send.call_args[0][0])
    assert err_cand_p2p["type"] == MessageType.ERROR
    assert err_cand_p2p["payload"]["request_id"] == "req-cand-1"

    worker.ws.send.reset_mock()

    # 3. P2P SPECULATIVE_VERIFIED via handle_incoming_frame
    ver_frame = ProtocolFrame(
        type=MessageType.SPECULATIVE_VERIFIED,
        source="peer",
        target="w-no-engine",
        payload={"request_id": "req-ver-1", "accepted_tokens": [10, 20], "accepted_texts": ["a", "b"]},
    )
    await worker.handle_incoming_frame(ver_frame.model_dump_json())
    assert worker.ws.send.call_count == 1
    err_ver_p2p = json.loads(worker.ws.send.call_args[0][0])
    assert err_ver_p2p["type"] == MessageType.ERROR
    assert err_ver_p2p["payload"]["request_id"] == "req-ver-1"

    worker.ws.send.reset_mock()

    # 4. WebSocket message loop via mock websocket into worker.run()
    stop_frame = ProtocolFrame(
        type=MessageType.WORKER_REMOVED,
        source="coordinator",
        target="w-no-engine",
        payload={},
    )
    ws_frames = [
        tok_frame.model_dump_json(),
        cand_frame.model_dump_json(),
        ver_frame.model_dump_json(),
        stop_frame.model_dump_json(),
    ]
    mock_ws = MockAsyncWs(ws_frames)

    with patch("websockets.connect", side_effect=lambda *args, **kwargs: MockConnectManager(mock_ws)):
        await worker.run()

    # In worker.run(), initial registration frame is sent, followed by 3 ERROR frames for the 3 missing-engine frames
    error_frames = [
        json.loads(msg) for msg in mock_ws.sent_messages
        if json.loads(msg).get("type") == MessageType.ERROR
    ]
    assert len(error_frames) == 3
    assert error_frames[0]["payload"]["request_id"] == "req-tok-1"
    assert error_frames[1]["payload"]["request_id"] == "req-cand-1"
    assert error_frames[2]["payload"]["request_id"] == "req-ver-1"


def test_intermediate_worker_session_token_history_synchronization():
    """Verify that when an intermediate worker processes SPECULATIVE_VERIFIED,
    it appends accepted_tokens to session_token_histories so cumulative position
    tracking stays synchronized with the full sequence length.
    """
    assign = AssignLayers(
        worker_id="w-mid",
        model_name="test-spec",
        layer_start=4,
        layer_end=8,
        total_layers=12,
        is_first_stage=False,
        is_last_stage=False,
        prev_worker_id="w-first",
        next_worker_id="w-last",
    )
    mock_model = MagicMock()
    engine = StageExecutionEngine(stage_model=mock_model, assignment=assign)

    req_id = "req-sync-hist"
    start_cfg = InferenceStart(
        request_id=req_id,
        model_name="test-spec",
        input_ids=[1, 2, 3, 4, 5],
        max_new_tokens=20,
    )
    engine.session_configs[req_id] = start_cfg
    engine.session_token_histories[req_id] = list(start_cfg.input_ids)

    # First burst of speculative verified tokens
    ver1 = SpeculativeVerified(
        request_id=req_id,
        step_index=1,
        accepted_tokens=[10, 20, 30],
        accepted_texts=[" 10", " 20", " 30"],
        is_finished=False,
    )
    out1 = engine.process_speculative_verified(ver1)
    assert out1 is not None
    assert out1.type == MessageType.SPECULATIVE_VERIFIED
    assert out1.target == "w-first"

    # Verify history contains prompt tokens + first burst
    assert engine.session_token_histories[req_id] == [1, 2, 3, 4, 5, 10, 20, 30]
    # At ver1, hist_len was 5, added_prefix was max(0, 3-1)=2, so step_counts was 5 + 2 = 7
    assert engine.session_step_counts[req_id] == 7

    # Second burst of speculative verified tokens
    ver2 = SpeculativeVerified(
        request_id=req_id,
        step_index=4,
        accepted_tokens=[40, 50],
        accepted_texts=[" 40", " 50"],
        is_finished=False,
    )
    out2 = engine.process_speculative_verified(ver2)
    assert out2 is not None

    # Verify history now contains all accumulated tokens
    assert engine.session_token_histories[req_id] == [1, 2, 3, 4, 5, 10, 20, 30, 40, 50]
    # At ver2, hist_len was 8 (because ver1 was appended!), added_prefix is max(0, 2-1)=1,
    # so step_counts is 8 + 1 = 9. (Without the fix, hist_len would have been 5 and step_counts 6!)
    assert engine.session_step_counts[req_id] == 9


@pytest.mark.asyncio
async def test_speculative_verification_relay_fallback_three_workers_no_coordinator_duplicates():
    """Verify that in fallback WebSocket relay routing (P2P inactive), intermediate workers
    forwarding SPECULATIVE_VERIFIED to upstream stages do NOT cause duplicate tokens in the
    coordinator's client response stream.
    """
    mock_registry = MagicMock(spec=WorkerRegistry)
    mock_registry.get_pipeline_order.return_value = [
        MagicMock(worker_id="worker-0"),
        MagicMock(worker_id="worker-1"),
        MagicMock(worker_id="worker-2"),
    ]

    relay = RelayHub(registry=mock_registry)
    relay.send_to_worker = AsyncMock()

    req_id = "req-relay-dedup"
    stream_q = relay.register_response_stream(req_id)

    # Step 1: Worker 2 (last stage) emits SPECULATIVE_VERIFIED towards Worker 1 via relay
    ver_payload = SpeculativeVerified(
        request_id=req_id,
        step_index=1,
        accepted_tokens=[101, 102],
        accepted_texts=[" a", " b"],
        is_finished=False,
    )
    frame_w2 = ProtocolFrame(
        type=MessageType.SPECULATIVE_VERIFIED,
        source="worker-2",
        target="worker-1",
        payload=ver_payload.model_dump(),
    )
    await relay.route_frame(frame_w2)

    # Coordinator routed frame to worker-1 AND pushed to client stream
    relay.send_to_worker.assert_awaited_once_with("worker-1", frame_w2)
    assert stream_q.qsize() == 1
    stream_item1 = await stream_q.get()
    assert stream_item1.source == "worker-2"

    relay.send_to_worker.reset_mock()

    # Step 2: Intermediate worker-1 forwards SPECULATIVE_VERIFIED to worker-0 via relay
    frame_w1 = ProtocolFrame(
        type=MessageType.SPECULATIVE_VERIFIED,
        source="worker-1",
        target="worker-0",
        payload=ver_payload.model_dump(),
    )
    await relay.route_frame(frame_w1)

    # Coordinator routed frame to worker-0, but MUST NOT push duplicate to client stream!
    relay.send_to_worker.assert_awaited_once_with("worker-0", frame_w1)
    assert stream_q.qsize() == 0, "Intermediate forwarded frame caused duplicate in client response stream!"


def test_session_step_counts_cleaned_up_on_finish_and_abort():
    """Verify that session_step_counts is cleaned up when an inference request finishes or aborts."""
    assign = AssignLayers(
        worker_id="w-stage",
        model_name="test-model",
        layer_start=0,
        layer_end=4,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=True,
    )
    engine = StageExecutionEngine(stage_model=MagicMock(), assignment=assign)

    req1 = "req-finish-test"
    engine.session_step_counts[req1] = 42
    engine.process_inference_finish(req1)
    assert req1 not in engine.session_step_counts

    req2 = "req-abort-test"
    engine.session_step_counts[req2] = 84
    engine.abort_inference(req2)
    assert req2 not in engine.session_step_counts

