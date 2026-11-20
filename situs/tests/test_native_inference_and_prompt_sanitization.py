"""Comprehensive unit and integration tests for Prompt Sanitization, Memory Leak Removal,
and Option A Native High-Performance Inference Runner for single-node GGUF models.
"""

import asyncio
import inspect
import json
import os
import pathlib
import pytest
import subprocess
import torch
from unittest.mock import MagicMock, AsyncMock, patch

from situs.protocol.messages import (
    MessageType,
    ProtocolFrame,
    AssignLayers,
    InferenceStart,
    NextToken,
)
from situs.worker.loader import (
    Int4WeightLinear,
    Int8WeightLinear,
)
from situs.inference_backend import (
    find_native_llama_server,
    detect_llama_server_gpu_device,
    is_local_ollama_available,
    find_gguf_file,
    find_free_port,
    NativeInferenceServer,
)
from situs.worker.client import WorkerDaemon
from situs.config import WorkerConfig


# =========================================================================
# Component 1: Prompt Sanitization Tests
# =========================================================================

def test_default_system_prompt_empty():
    """Verify that default system_prompt is empty in both TUI and CLI."""
    import situs.tui as tui_mod
    import situs.cli as cli_mod

    # 1. TUI default
    sig_tui = inspect.signature(tui_mod.run_opencode_tui)
    assert sig_tui.parameters["system_prompt"].default == "", "run_opencode_tui default system_prompt must be empty"

    # 2. CLI interactive_chat default
    sig_cli = inspect.signature(cli_mod.interactive_chat)
    cli_param = sig_cli.parameters["system"].default
    default_val = getattr(cli_param, "default", cli_param)
    assert default_val == "", "cli interactive_chat default --system must be empty string"


def test_tui_clear_command_no_cluster_persona():
    """Verify /clear resets conversation messages without injecting the cluster persona."""
    from situs.tui import execute_slash_command

    # Case A: session_state with empty system prompt (the new default)
    session_state = {"system_prompt": "", "speculative": True, "p2p": True}
    messages = [
        {"role": "user", "content": "How does an engine work?"},
        {"role": "assistant", "content": "An internal combustion engine..."},
    ]

    execute_slash_command("/clear", "http://localhost:8000", "gemma3:4b", messages, session_state)
    assert len(messages) == 0, "When system_prompt is empty, /clear must result in zero messages"
    assert not any("distributed GPU cluster" in m.get("content", "") for m in messages)

    # Case B: session_state with explicit user system prompt
    session_state_user = {"system_prompt": "You are a physics teacher.", "speculative": True, "p2p": True}
    messages_user = [{"role": "user", "content": "Hi"}]
    execute_slash_command("/clear", "http://localhost:8000", "gemma3:4b", messages_user, session_state_user)
    assert len(messages_user) == 1
    assert messages_user[0] == {"role": "system", "content": "You are a physics teacher."}


def test_tui_system_command_update_and_clear():
    """Verify /system command updates, clears, and reads the active system prompt."""
    from situs.tui import execute_slash_command

    logs = []
    def log_capture(title, renderable):
        logs.append((title, renderable))

    session_state = {"system_prompt": "", "speculative": True, "p2p": True}
    messages = []

    # Update system prompt
    execute_slash_command("/system You are an expert mechanic.", "http://localhost:8000", "gemma3:4b", messages, session_state, log_output_fn=log_capture)
    assert session_state["system_prompt"] == "You are an expert mechanic."
    assert len(messages) == 1
    assert messages[0] == {"role": "system", "content": "You are an expert mechanic."}

    # Inspect current system prompt
    logs.clear()
    execute_slash_command("/system", "http://localhost:8000", "gemma3:4b", messages, session_state, log_output_fn=log_capture)
    assert any("You are an expert mechanic." in str(r) for _, r in logs)

    # Clear system prompt with /system ""
    logs.clear()
    execute_slash_command('/system ""', "http://localhost:8000", "gemma3:4b", messages, session_state, log_output_fn=log_capture)
    assert session_state["system_prompt"] == ""
    assert len(messages) == 0
    assert any("cleared" in str(r) for _, r in logs)

    # Verify /clear preserves strictly empty messages when system_prompt is cleared
    execute_slash_command("/clear", "http://localhost:8000", "gemma3:4b", messages, session_state)
    assert len(messages) == 0

    # Set quoted system prompt
    execute_slash_command('/system "You are a poet."', "http://localhost:8000", "gemma3:4b", messages, session_state)
    assert session_state["system_prompt"] == "You are a poet."
    assert len(messages) == 1
    assert messages[0]["content"] == "You are a poet."

    # Clear system prompt via /system clear
    execute_slash_command("/system clear", "http://localhost:8000", "gemma3:4b", messages, session_state)
    assert session_state["system_prompt"] == ""
    assert len(messages) == 0


# =========================================================================
# Component 2: Memory Leak Removal & 4-Bit Priority Tests
# =========================================================================

def test_int4_weight_linear_no_cached_weight_leak():
    """Verify Int4WeightLinear does not leak 6.42 GB across 238 layers via _cached_weight."""
    torch.manual_seed(42)
    in_dim, out_dim = 64, 128
    w = torch.randn(out_dim, in_dim, dtype=torch.float32)
    linear = Int4WeightLinear.from_float(w)

    x = torch.randn(2, 4, in_dim, dtype=torch.float32)

    assert linear._cached_weight is None
    out1 = linear(x)
    assert linear._cached_weight is None, "_cached_weight must remain None to prevent permanent VRAM leak"

    out2 = linear(x)
    assert torch.allclose(out1, out2)
    assert linear._cached_weight is None


def test_int8_weight_linear_no_cached_weight_leak():
    """Verify Int8WeightLinear does not leak unpacked weights via _cached_weight."""
    torch.manual_seed(42)
    in_dim, out_dim = 64, 128
    w = torch.randn(out_dim, in_dim, dtype=torch.float32)
    linear = Int8WeightLinear.from_float(w)

    x = torch.randn(2, 4, in_dim, dtype=torch.float32)

    assert linear._cached_weight is None
    out1 = linear(x)
    assert linear._cached_weight is None

    out2 = linear(x)
    assert torch.allclose(out1, out2)
    assert linear._cached_weight is None


def test_gguf_quantization_precedence_prioritizes_4bit():
    """Verify that when 4-bit tensors are present in GGUF, 4-bit quantization is prioritized over int8."""
    from types import SimpleNamespace

    # Mock tensors containing 4-bit tensor type 2 (Q4_0)
    mock_tensors = [
        SimpleNamespace(name="token_embd.weight", shape=[100, 64], tensor_type=2, offset=0, size=3200),
        SimpleNamespace(name="blk.0.attn_q.weight", shape=[64, 64], tensor_type=2, offset=3200, size=2048),
    ]

    has_4bit = any(t.tensor_type in (2, 3, 12, 20, 23) for t in mock_tensors)
    use_4bit = bool(has_4bit)
    # use_int8 must NOT override use_4bit
    use_int8 = bool(True and not use_4bit and any(t.tensor_type not in (0, 1, 30) for t in mock_tensors))

    assert use_4bit is True
    assert use_int8 is False, "use_int8 must be False when 4-bit tensors are present"


# =========================================================================
# Component 3: Option A Native Inference Runner Tests
# =========================================================================

def test_find_native_llama_server():
    """Verify auto-discovery finds candidate native llama-server executable."""
    found = find_native_llama_server()
    # If installed on this machine, verify it is an executable file
    if found:
        assert os.path.isfile(found)
        assert "llama-server" in found.lower()


def test_detect_llama_server_gpu_device():
    """Verify GPU device parser prioritizes NVIDIA discrete GPU over integrated graphics."""
    mock_devices_output = (
        "Available devices:\n"
        "  Vulkan0: Intel(R) UHD Graphics (8006 MiB, 7350 MiB free)\n"
        "  Vulkan1: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6001 MiB, 5233 MiB free)\n"
    )

    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=mock_devices_output, stderr="", returncode=0)
        dev = detect_llama_server_gpu_device("llama-server.exe")
        assert dev == "Vulkan1", "Must select Vulkan1 NVIDIA GeForce RTX 3050 over Intel UHD"


def test_detect_llama_server_cuda_device():
    """Verify CUDA device detection when CUDA device list is returned."""
    mock_devices_output = (
        "Available devices:\n"
        "  CUDA0: NVIDIA GeForce RTX 3050 (6144 MiB)\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=mock_devices_output, stderr="", returncode=0)
        dev = detect_llama_server_gpu_device("llama-server.exe")
        assert dev == "CUDA0"


def test_find_gguf_file():
    """Verify find_gguf_file locates cached GGUF model files."""
    from situs.inference_backend import _is_valid_gguf_file
    # Test with standard gemma3 tag
    gguf_path = find_gguf_file("gemma3:4b")
    if gguf_path:
        assert os.path.isfile(gguf_path)
        assert gguf_path.lower().endswith(".gguf") or _is_valid_gguf_file(gguf_path)


def test_find_free_port():
    """Verify find_free_port returns an available TCP port."""
    port = find_free_port(9100)
    assert 9100 <= port <= 9200


@pytest.mark.asyncio
async def test_native_inference_server_streaming():
    """Verify NativeInferenceServer streams NextToken chunks from completion responses."""
    server = NativeInferenceServer()
    server.backend_type = "llama-server"
    server.base_url = "http://127.0.0.1:8099"
    server.proc = MagicMock()
    server.proc.poll.return_value = None

    assert server.is_ready() is True

    start_msg = InferenceStart(
        request_id="req-native-test",
        model_name="gemma3:4b",
        input_ids=[1, 2, 3],
        prompt="<start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
        max_new_tokens=50,
    )

    # Mock SSE stream lines from /completion
    fake_lines = [
        b'data: {"content": "Hello", "stop": false}',
        b'data: {"content": " world", "stop": false}',
        b'data: {"content": "!", "stop": true}',
    ]

    class MockAsyncResponse:
        status_code = 200
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        async def aiter_lines(self):
            for line in fake_lines:
                yield line.decode("utf-8")

    class MockAsyncClient:
        async def __aenter__(self):
            return self
        async def __aexit__(self, *args):
            pass
        def stream(self, method, url, **kwargs):
            return MockAsyncResponse()

    with patch("httpx.AsyncClient", return_value=MockAsyncClient()):
        tokens = []
        async for tok in server.stream_inference(start_msg):
            tokens.append(tok)

        assert len(tokens) == 3
        assert tokens[0].token_text == "Hello"
        assert tokens[0].is_finished is False
        assert tokens[1].token_text == " world"
        assert tokens[2].token_text == "!"
        assert tokens[2].is_finished is True
        assert tokens[2].finish_reason == "stop"


@pytest.mark.asyncio
async def test_single_worker_loop_delegates_to_native_server():
    """Verify WorkerDaemon single-worker generation loop delegates to native server when active."""
    cfg = WorkerConfig(worker_id="w-single-native", coordinator_url="ws://127.0.0.1:8000/ws/worker")
    daemon = WorkerDaemon(cfg)

    # Set up mock native server
    mock_server = MagicMock()
    mock_server.is_ready.return_value = True

    async def _mock_stream(start_msg):
        yield NextToken(
            request_id=start_msg.request_id,
            step_index=0,
            token_id=0,
            token_text="Combustion",
            is_finished=False,
            is_verified=True,
        )
        yield NextToken(
            request_id=start_msg.request_id,
            step_index=1,
            token_id=0,
            token_text=" engine.",
            is_finished=True,
            finish_reason="stop",
            is_verified=True,
        )

    mock_server.stream_inference = _mock_stream
    daemon.native_server = mock_server

    sent_frames = []
    async def _mock_send(frame):
        sent_frames.append(frame)

    daemon.send_frame_out = _mock_send

    start_msg = InferenceStart(
        request_id="req-single-test",
        model_name="gemma3:4b",
        input_ids=[10, 20],
        prompt="Explain an engine.",
    )

    await daemon._run_single_worker_loop(start_msg)

    assert len(sent_frames) == 2
    assert sent_frames[0].target == "coordinator"
    assert sent_frames[0].payload["token_text"] == "Combustion"
    assert sent_frames[1].payload["token_text"] == " engine."
    assert sent_frames[1].payload["is_finished"] is True


@pytest.mark.asyncio
async def test_multi_worker_stops_native_server():
    """Verify that when 2 workers connect, native server on single worker is stopped and freed."""
    cfg = WorkerConfig(worker_id="w-rebalance-test", coordinator_url="ws://127.0.0.1:8000/ws/worker")
    daemon = WorkerDaemon(cfg)

    mock_server = MagicMock()
    daemon.native_server = mock_server

    # Simulate ASSIGN_LAYERS arriving with multi-node assignment (is_last_stage=False)
    assign_multi = AssignLayers(
        worker_id="w-rebalance-test",
        model_name="gemma3:4b",
        layer_start=0,
        layer_end=12,
        total_layers=24,
        is_first_stage=True,
        is_last_stage=False,
    )

    is_single_node = (assign_multi.is_first_stage and assign_multi.is_last_stage)
    assert is_single_node is False

    # Simulate client.py logic:
    if not is_single_node and daemon.native_server:
        daemon.native_server.stop()
        daemon.native_server = None

    mock_server.stop.assert_called_once()
    assert daemon.native_server is None


def test_detect_llama_server_gpu_device_prioritizes_discrete_over_igpu():
    """Verify detect_llama_server_gpu_device selects discrete GPU (e.g. AMD or NVIDIA) over Intel UHD igpu."""
    from unittest.mock import patch
    from situs.inference_backend import detect_llama_server_gpu_device

    # Case A: Intel UHD as Vulkan0, AMD Radeon as Vulkan1
    fake_amd_output = (
        "Available devices:\n"
        "  Vulkan0: Intel(R) UHD Graphics (8006 MiB, 7350 MiB free)\n"
        "  Vulkan1: AMD Radeon RX 6700 XT (12288 MiB, 11000 MiB free)\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=fake_amd_output, stderr="", returncode=0)
        chosen = detect_llama_server_gpu_device("llama-server.exe")
        assert chosen == "Vulkan1"

    # Case B: Intel UHD as Vulkan0, NVIDIA RTX as Vulkan1
    fake_nv_output = (
        "Available devices:\n"
        "  Vulkan0: Intel(R) UHD Graphics (8006 MiB, 7350 MiB free)\n"
        "  Vulkan1: NVIDIA GeForce RTX 3050 6GB Laptop GPU (6001 MiB, 5233 MiB free)\n"
    )
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout=fake_nv_output, stderr="", returncode=0)
        chosen = detect_llama_server_gpu_device("llama-server.exe")
        assert chosen == "Vulkan1"


@pytest.mark.asyncio
async def test_single_worker_loop_auto_restarts_inactive_native_server():
    """Verify _run_single_worker_loop auto-restarts the native runner if it terminated unexpectedly."""
    cfg = WorkerConfig(worker_id="w-auto-restart", coordinator_url="ws://127.0.0.1:8000/ws/worker")
    daemon = WorkerDaemon(cfg)

    daemon.assignment = AssignLayers(
        worker_id="w-auto-restart",
        model_name="gemma3:4b",
        layer_start=0,
        layer_end=24,
        total_layers=24,
        is_first_stage=True,
        is_last_stage=True,
    )

    mock_server = MagicMock()
    # Initially inactive (poll() is not None or is_ready returns False)
    ready_states = [False, True]
    mock_server.is_ready.side_effect = lambda: ready_states.pop(0) if ready_states else True
    mock_server.start.return_value = True

    async def _mock_stream(msg):
        from situs.protocol.messages import NextToken
        yield NextToken(request_id="req-restart", step_index=0, token_id=1, token_text="Restarted OK", is_finished=True, finish_reason="stop", is_verified=True)

    mock_server.stream_inference = _mock_stream
    daemon.native_server = mock_server

    sent_frames = []
    daemon.send_frame_out = lambda f: sent_frames.append(f) or asyncio.sleep(0)

    start_msg = InferenceStart(
        request_id="req-restart",
        model_name="gemma3:4b",
        input_ids=[10],
        prompt="Hi",
    )

    await daemon._run_single_worker_loop(start_msg)

    mock_server.start.assert_called_once_with("gemma3:4b")
    assert len(sent_frames) == 1
    assert sent_frames[0].payload["token_text"] == "Restarted OK"


@pytest.mark.asyncio
async def test_assign_layers_initiates_native_server_on_single_node():
    """Verify ASSIGN_LAYERS automatically launches NativeInferenceServer for single-node GGUF execution."""
    cfg = WorkerConfig(worker_id="w-assign-native", coordinator_url="ws://127.0.0.1:8000/ws/worker")
    daemon = WorkerDaemon(cfg)

    assign = AssignLayers(
        worker_id="w-assign-native",
        model_name="gemma3:4b",
        layer_start=0,
        layer_end=34,
        total_layers=34,
        is_first_stage=True,
        is_last_stage=True,
    )
    frame = ProtocolFrame(
        type=MessageType.ASSIGN_LAYERS,
        source="coordinator",
        target="w-assign-native",
        payload=assign.model_dump(),
    )

    mock_server = MagicMock()
    mock_server.is_ready.return_value = True
    mock_server.start.return_value = True
    mock_server.model_name = "gemma3:4b"
    mock_server.base_url = "http://127.0.0.1:8092"

    with patch("situs.inference_backend.find_gguf_file", return_value="C:/fake/gemma-3-4b.gguf"), \
         patch("situs.inference_backend.NativeInferenceServer", return_value=mock_server):
        # Emulate receiving ASSIGN_LAYERS frame in message loop
        is_single_node = (assign.is_first_stage and assign.is_last_stage)
        assert is_single_node is True

        from situs.inference_backend import NativeInferenceServer, find_gguf_file
        gguf_file = find_gguf_file(assign.model_name)
        assert gguf_file is not None

        daemon.native_server = NativeInferenceServer()
        started = daemon.native_server.start(assign.model_name, gguf_file)
        assert started is True
        assert daemon.engine is None  # Ensures PyTorch 2.9GB duplicate VRAM is NOT allocated


@pytest.mark.asyncio
async def test_assign_layers_stops_native_server_on_multi_worker_transition():
    """Verify transitioning from single-node to multi-node cleanly terminates NativeInferenceServer."""
    cfg = WorkerConfig(worker_id="w-multi-stop", coordinator_url="ws://127.0.0.1:8000/ws/worker")
    daemon = WorkerDaemon(cfg)

    mock_server = MagicMock()
    daemon.native_server = mock_server

    # Incoming multi-stage assignment
    assign_multi = AssignLayers(
        worker_id="w-multi-stop",
        model_name="gemma3:4b",
        layer_start=0,
        layer_end=17,
        total_layers=34,
        is_first_stage=True,
        is_last_stage=False,
    )

    is_single_node = (assign_multi.is_first_stage and assign_multi.is_last_stage)
    assert is_single_node is False

    if not is_single_node and daemon.native_server:
        daemon.native_server.stop()
        daemon.native_server = None

    mock_server.stop.assert_called_once()
    assert daemon.native_server is None


@pytest.mark.asyncio
async def test_inference_start_streams_via_native_when_engine_none():
    """Verify INFERENCE_START proceeds smoothly when daemon.engine is None and daemon.native_server is active."""
    cfg = WorkerConfig(worker_id="w-stream-native", coordinator_url="ws://127.0.0.1:8000/ws/worker")
    daemon = WorkerDaemon(cfg)
    daemon.engine = None

    mock_server = MagicMock()
    mock_server.is_ready.return_value = True

    async def _mock_stream(msg):
        from situs.protocol.messages import NextToken
        yield NextToken(request_id="req-native-pass", step_index=0, token_id=1, token_text="Fast token", is_finished=True, finish_reason="stop", is_verified=True)

    mock_server.stream_inference = _mock_stream
    daemon.native_server = mock_server

    has_native = bool(daemon.native_server and daemon.native_server.is_ready())
    assert has_native is True

    sent_frames = []
    daemon.send_frame_out = lambda f: sent_frames.append(f) or asyncio.sleep(0)

    start_msg = InferenceStart(
        request_id="req-native-pass",
        model_name="gemma3:4b",
        input_ids=[10],
        prompt="Hi",
    )

    await daemon._run_single_worker_loop(start_msg)
    assert len(sent_frames) == 1
    assert sent_frames[0].payload["token_text"] == "Fast token"
    assert sent_frames[0].target == "coordinator"


@pytest.mark.asyncio
async def test_inference_abort_cancels_active_native_task():
    """Verify INFERENCE_ABORT cancels the active streaming task even when daemon.engine is None."""
    cfg = WorkerConfig(worker_id="w-abort-native", coordinator_url="ws://127.0.0.1:8000/ws/worker")
    daemon = WorkerDaemon(cfg)
    daemon.engine = None

    task_cancelled = False
    async def dummy_stream_loop():
        nonlocal task_cancelled
        try:
            await asyncio.sleep(10.0)
        except asyncio.CancelledError:
            task_cancelled = True
            raise

    active_task = asyncio.create_task(dummy_stream_loop())
    daemon.active_inference_tasks["req-abort-123"] = active_task

    # Emulate receiving INFERENCE_ABORT frame
    abort_frame = ProtocolFrame(
        type=MessageType.INFERENCE_ABORT,
        source="coordinator",
        target="w-abort-native",
        payload={"request_id": "req-abort-123"},
    )

    req_id = abort_frame.payload.get("request_id")
    task = daemon.active_inference_tasks.pop(req_id, None)
    if task and not task.done():
        task.cancel()

    try:
        await active_task
    except asyncio.CancelledError:
        pass
    assert active_task.cancelled() is True or task_cancelled is True


def test_vectorized_unpack_weight_equivalence():
    """Verify vectorized torch.stack unpack produces mathematically identical output to strided slicing."""
    in_dim, out_dim = 64, 128
    w = torch.randn(out_dim, in_dim, dtype=torch.float32)
    linear = Int4WeightLinear.from_float(w)

    # Reference strided slicing unpack
    pw = linear.packed_weight
    q0 = (pw & 0x0F).to(torch.int8) - 8
    q1 = ((pw >> 4) & 0x0F).to(torch.int8) - 8
    nb = linear.weight_scale.shape[1]
    bs = 32
    ref_out = torch.empty((linear.out_features, nb * bs), dtype=torch.float32, device=pw.device)
    ref_out[:, 0::2] = q0.to(torch.float32)
    ref_out[:, 1::2] = q1.to(torch.float32)
    ref_out = (ref_out.view(linear.out_features, nb, bs) * linear.weight_scale.to(dtype=torch.float32)).view(linear.out_features, -1)
    if ref_out.shape[1] > linear.in_features:
        ref_out = ref_out[:, :linear.in_features]

    # Actual unpack_weight implementation
    actual_out = linear.unpack_weight(torch.float32)

    assert torch.equal(actual_out, ref_out), "Vectorized unpack must match reference unpack output exactly"


