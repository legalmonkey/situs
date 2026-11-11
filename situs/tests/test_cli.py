"""Unit tests for the Situs CLI commands and URL normalizers."""

import pytest
from typer.testing import CliRunner
from situs.cli import app, normalize_ws_url, normalize_http_url, get_local_ip

runner = CliRunner()


def test_url_normalization():
    # WebSocket normalizer
    assert normalize_ws_url("192.168.1.10:8000") == "ws://192.168.1.10:8000/ws/worker"
    assert normalize_ws_url("http://192.168.1.10:8000") == "ws://192.168.1.10:8000/ws/worker"
    assert normalize_ws_url("ws://192.168.1.10:8000/ws/worker") == "ws://192.168.1.10:8000/ws/worker"
    assert normalize_ws_url("https://relay.situs.io") == "wss://relay.situs.io/ws/worker"

    # HTTP normalizer
    assert normalize_http_url("192.168.1.10:8000") == "http://192.168.1.10:8000"
    assert normalize_http_url("ws://192.168.1.10:8000/ws/worker") == "http://192.168.1.10:8000"
    assert normalize_http_url("http://localhost:8000/") == "http://localhost:8000"


def test_local_ip_detection():
    ip = get_local_ip()
    assert isinstance(ip, str)
    assert len(ip.split(".")) == 4


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Situs: Distributed GPU Pooling Across the Internet" in result.output
    assert "setup" in result.output
    assert "coordinator" in result.output
    assert "join" in result.output
    assert "status" in result.output
    assert "chat" in result.output


def test_cli_setup():
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "Situs Device Diagnostics" in result.output
    assert "Operating System" in result.output
    assert "Python Runtime" in result.output


def test_tui_slash_completer():
    from situs.tui import OpenCodeSlashCompleter
    from prompt_toolkit.document import Document

    completer = OpenCodeSlashCompleter()
    
    # 1. Test typing '/' gives all slash commands
    doc_slash = Document(text="/", cursor_position=1)
    all_completions = list(completer.get_completions(doc_slash, None))
    all_cmds = [c.text for c in all_completions]
    assert "/status" in all_cmds
    assert "/vram" in all_cmds
    assert "/gpu" in all_cmds
    assert "/partition" in all_cmds
    assert "/nodes" in all_cmds
    assert "/doctor" in all_cmds
    assert "/model" in all_cmds
    assert "/models" in all_cmds
    assert "/p2p" in all_cmds
    assert "/speculative" in all_cmds
    assert "/clear" in all_cmds
    assert "/help" in all_cmds
    assert "/exit" in all_cmds

    # 2. Test typing '/st' filters to /status
    doc_st = Document(text="/st", cursor_position=3)
    st_completions = [c.text for c in completer.get_completions(doc_st, None)]
    assert st_completions == ["/status"]

    # 3. Test typing '/p' filters to /partition and /p2p
    doc_p = Document(text="/p", cursor_position=2)
    p_completions = [c.text for c in completer.get_completions(doc_p, None)]
    assert "/partition" in p_completions
    assert "/p2p" in p_completions

    # 4. Test typing '/model ' returns curated distributed model choices (including 15GB models)
    doc_m = Document(text="/model ", cursor_position=7)
    m_completions = list(completer.get_completions(doc_m, None))
    assert len(m_completions) >= 8
    assert m_completions[0].text == "/model 1"
    assert "Llama 3.1 8B" in m_completions[0].display_text
    assert any("14B" in c.display_text for c in m_completions)

    # 5. Test typing '/join ' suggests custom worker IDs
    doc_j = Document(text="/join ", cursor_position=6)
    j_completions = list(completer.get_completions(doc_j, None))
    assert any("worker-2" in c.text for c in j_completions)


def test_tui_execute_slash_commands():
    from situs.tui import execute_slash_command, get_git_branch, fetch_cluster_metadata

    session_state = {"speculative": True, "p2p": True, "system_prompt": "Initial"}
    messages = [{"role": "system", "content": "Initial"}, {"role": "user", "content": "Hi"}]

    # Test /help with log output capture
    logs = []
    def log_capture(title, renderable):
        logs.append((title, renderable))

    assert execute_slash_command("/help", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    titles = [t for t, _ in logs]
    assert "Help Commands" in titles
    assert "Worker ID Guide" in titles
    assert "Model Selection Guide" in titles

    # Test /models
    logs.clear()
    assert execute_slash_command("/models", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    titles = [t for t, _ in logs]
    assert any("Curated" in t for t in titles)
    assert "Select Model" in titles

    # Test /vram
    logs.clear()
    assert execute_slash_command("/vram", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    titles = [t for t, _ in logs]
    assert any("Live GPU VRAM Monitor" in t for t in titles)

    # Test /model 1 selects Llama 3.1 8B
    logs.clear()
    assert execute_slash_command("/model 1", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    assert session_state["model"] == "llama-3.1:8b-instruct-q8_0"

    # Test /model 2 selects Qwen 2.5 7B
    assert execute_slash_command("/model 2", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    assert session_state["model"] == "qwen2.5:7b-instruct-q8_0"

    # Test /model with no args inspects the active model
    logs.clear()
    assert execute_slash_command("/model", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    assert any("Active Model" in t for t, _ in logs)

    # Test /join parsing with custom worker-id and device
    assert execute_slash_command("/join 127.0.0.1:8000 worker-2 cpu", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    assert execute_slash_command("/join 127.0.0.1:8000 --worker-id worker-test --device cpu", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True

    # Test /speculative toggle
    assert execute_slash_command("/speculative", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state) is True
    assert session_state["speculative"] is False
    assert execute_slash_command("/speculative", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state) is True
    assert session_state["speculative"] is True

    # Test /p2p toggle
    assert execute_slash_command("/p2p", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state) is True
    assert session_state["p2p"] is False

    # Test /clear
    assert execute_slash_command("/clear", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state) is True
    assert len(messages) == 1

    # Test in-interface coordinator and join commands
    assert execute_slash_command("situs coordinator", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state) is True
    assert execute_slash_command("/coordinator", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state) is True
    assert execute_slash_command("situs status", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state) is True
    assert execute_slash_command("help", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state) is True

    # Test /tokens
    assert execute_slash_command("/tokens", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    assert execute_slash_command("/tokens 4096", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    assert session_state["max_tokens"] == 4096
    assert execute_slash_command("/tokens 0", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    assert execute_slash_command("/tokens 50000", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    assert execute_slash_command("/tokens invalid", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True

    # Test /exit
    assert execute_slash_command("/exit", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state) is False

    # Test metadata and git branch
    meta = fetch_cluster_metadata("http://localhost:9999")
    assert "status" in meta
    assert meta["status"] == "offline"

    branch = get_git_branch()
    assert branch.startswith("~:")


def test_placeholder_processor():
    from situs.tui import PlaceholderProcessor
    from prompt_toolkit.document import Document
    from prompt_toolkit.layout.processors import TransformationInput

    proc = PlaceholderProcessor("Ask anything...")

    # Empty document -> shows placeholder
    doc_empty = Document(text="", cursor_position=0)
    ti_empty = TransformationInput(None, doc_empty, 0, lambda i: i, [], 80, 1)
    t_empty = proc.apply_transformation(ti_empty)
    assert ("class:placeholder", "Ask anything...") in t_empty.fragments

    # Non-empty document -> preserves text
    doc_typed = Document(text="hello", cursor_position=5)
    ti_typed = TransformationInput(None, doc_typed, 0, lambda i: i, [("", "hello")], 80, 1)
    t_typed = proc.apply_transformation(ti_typed)
    assert t_typed.fragments == [("", "hello")]


def test_prompt_in_centered_box_pipe():
    from situs.tui import prompt_in_centered_box
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.data_structures import Size
    import io

    # Use simulated pipe input and virtual vt100 output
    with create_pipe_input() as pipe:
        pipe.send_text("Explain distributed inference\r")
        fake_out = Vt100_Output(io.StringIO(), lambda: Size(24, 80))
        result = prompt_in_centered_box(
            "http://localhost:8000",
            {"active_workers_count": 2, "model": "llama-3.1:8b-instruct-q8_0"},
            app_input=pipe,
            app_output=fake_out,
        )
        assert result == "Explain distributed inference"


def test_prompt_in_centered_box_tab_shows_nodes():
    from situs.tui import prompt_in_centered_box
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.data_structures import Size
    import io

    # When Tab is pressed on empty input, it returns '/nodes'
    with create_pipe_input() as pipe:
        pipe.send_text("\t")
        fake_out = Vt100_Output(io.StringIO(), lambda: Size(24, 80))
        result = prompt_in_centered_box(
            "http://localhost:8000",
            {"active_workers_count": 0, "model": "llama-3.1:8b-instruct-q8_0"},
            app_input=pipe,
            app_output=fake_out,
        )
        assert result == "/nodes"


def test_opencode_persistent_tui_pipe():
    from situs.tui import run_opencode_tui
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output
    from prompt_toolkit.data_structures import Size
    import io

    with create_pipe_input() as pipe:
        pipe.send_text("/exit\r")
        fake_out = Vt100_Output(io.StringIO(), lambda: Size(30, 100))
        res = run_opencode_tui(
            coordinator_url="http://127.0.0.1:8000",
            model="llama-3.1:8b-instruct-q8_0",
            app_input=pipe,
            app_output=fake_out,
            full_screen=False,
        )
        assert res == "/exit"


def test_hardware_detection_and_safe_join():
    from situs.hardware import detect_hardware
    from situs.worker.loader import load_hf_stage_model
    from situs.tui import execute_slash_command

    # 1. Test CPU detection
    hw_cpu = detect_hardware(requested_device="cpu", bypass_cache=True)
    assert hw_cpu.gpu_available is False
    assert hw_cpu.device == "cpu"
    assert hw_cpu.device_type == "cpu"
    assert hw_cpu.total_memory_bytes > 0
    assert hw_cpu.free_memory_bytes > 0

    # 2. Test auto detection
    hw_auto = detect_hardware(requested_device="auto", bypass_cache=True)
    assert hw_auto.device in ("cuda:0", "mps", "cpu")
    assert hw_auto.device_type in ("cuda", "mps", "cpu")
    assert hw_auto.device_name != ""
    assert hw_auto.status_message != ""
    if hw_auto.physical_gpu_detected:
        if hw_auto.cuda_runtime_working:
            assert hw_auto.gpu_available is True
            assert hw_auto.memory_type == "VRAM"
            assert hw_auto.total_memory_gb <= 12.0
        else:
            assert hw_auto.gpu_available is False
            assert hw_auto.device == "cpu"
            assert "unavailable" in hw_auto.status_message.lower()

    # 3. Test DistributedStageModel safe fallback: requesting cuda:0 on cpu-only torch should not crash
    stage = load_hf_stage_model(
        model_name="llama-3.1:8b-instruct-q8_0",
        layer_start=0,
        layer_end=2,
        total_layers=4,
        is_first_stage=True,
        is_last_stage=False,
        device="cuda:0",
    )
    assert stage.device in ("cuda:0", "cpu")

    # 4. Test /join with auto device in TUI
    messages = []
    session_state = {"model": "llama-3.1:8b-instruct-q8_0", "is_coordinator": True}
    captured = []
    def log_capture(title, renderable):
        captured.append((title, str(renderable)))

    assert execute_slash_command("/join 127.0.0.1:8000 worker-hw-test", "http://localhost:8000", "llama-3.1:8b-instruct-q8_0", messages, session_state, log_output_fn=log_capture) is True
    assert any("Worker Joined" in t or "Worker" in t for t, _ in captured)


def test_cli_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "situs" in result.output
    assert "0.1.0" in result.output


def test_cli_setup_json():
    import json
    result = runner.invoke(app, ["setup", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "os" in data
    assert "python_version" in data
    assert "lan_ip" in data


def test_cli_status_json_offline():
    import json
    result = runner.invoke(app, ["status", "--coordinator", "http://127.0.0.1:9999", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert data["status"] == "offline"
    assert "error" in data


def test_cli_examples_in_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Examples:" in result.output
    assert "situs setup" in result.output
    assert "situs coordinator" in result.output
    assert "situs join" in result.output


def test_coordinator_port_collision_handled():
    from unittest.mock import patch
    with patch("uvicorn.run", side_effect=OSError(10048, "Port in use")):
        result = runner.invoke(app, ["coordinator", "--port", "8000"])
        assert result.exit_code == 1
        assert "Port 8000 cannot be bound" in result.output


def test_partition_no_workers_error():
    import json
    result = runner.invoke(app, ["partition", "--coordinator", "http://127.0.0.1:9999", "--json"])
    assert result.exit_code == 1
    data = json.loads(result.output)
    assert "error" in data


def test_sanitize_coordinator_address_spaced_ip():
    from situs.config import sanitize_coordinator_address

    assert sanitize_coordinator_address("192. 168 . 29.189:8000") == "192.168.29.189:8000"
    assert sanitize_coordinator_address("192 . 168 . 29 . 189 : 8000") == "192.168.29.189:8000"
    assert sanitize_coordinator_address("192.168.29.189") == "192.168.29.189:8000"
    assert sanitize_coordinator_address("http : // 192. 168 . 29 . 189 : 8000") == "http://192.168.29.189:8000"
    assert sanitize_coordinator_address("ws : // 127.0.0.1 : 8000 / ws / worker") == "ws://127.0.0.1:8000/ws/worker"

    with pytest.raises(ValueError, match="Incomplete IPv4 address"):
        sanitize_coordinator_address("192.168")

    with pytest.raises(ValueError, match="Incomplete IPv4 address"):
        sanitize_coordinator_address("192.168.29")


def test_parse_join_command_spaced_ip():
    from situs.tui import parse_join_command

    addr, w_id, dev = parse_join_command("/join 192. 168 . 29.189:8000")
    assert addr == "192.168.29.189:8000"
    assert w_id is None
    assert dev is None

    addr2, w_id2, dev2 = parse_join_command("/join 192. 168 . 29.189:8000 worker-2 cuda:0")
    assert addr2 == "192.168.29.189:8000"
    assert w_id2 == "worker-2"
    assert dev2 == "cuda:0"

    addr3, w_id3, dev3 = parse_join_command("/join 192.168.29.189:8000 --id node-1 --device cpu")
    assert addr3 == "192.168.29.189:8000"
    assert w_id3 == "node-1"
    assert dev3 == "cpu"


def test_cli_cuda_command():
    result = runner.invoke(app, ["cuda"])
    assert result.exit_code == 0
    assert "CUDA Diagnostics" in result.output


def test_tui_cuda_command():
    from situs.tui import execute_slash_command
    messages = []
    session_state = {}
    captured = []
    def log_capture(title, renderable):
        captured.append((title, str(renderable)))

    assert execute_slash_command("/cuda", "http://localhost:8000", "gemma3:4b", messages, session_state, log_output_fn=log_capture) is True
    assert any("CUDA" in t for t, _ in captured)


def test_sanitize_coordinator_address_trailing_slash_and_tunnel():
    from situs.config import sanitize_coordinator_address
    from situs.cli import normalize_http_url, normalize_ws_url

    sanitized = sanitize_coordinator_address("https://idol-detroit-educated.trycloudflare.com/")
    assert sanitized == "https://idol-detroit-educated.trycloudflare.com"

    http_url = normalize_http_url("https://idol-detroit-educated.trycloudflare.com/")
    ws_url = normalize_ws_url("https://idol-detroit-educated.trycloudflare.com/")
    assert http_url == "https://idol-detroit-educated.trycloudflare.com"
    assert ws_url == "wss://idol-detroit-educated.trycloudflare.com/ws/worker"
    assert "//ws" not in ws_url
    assert "//status" not in f"{http_url}/status"


def test_cli_connect_alias():
    result = runner.invoke(app, ["connect", "--help"])
    assert result.exit_code == 0
    assert "Join a Situs coordinator cluster" in result.output or "Connect to" in result.output or "join" in result.output.lower()


def test_cli_main_tunnel_url_rerouting():
    import sys
    from unittest.mock import patch
    from situs.cli import main

    test_args = ["situs", "https://idol-detroit-educated.trycloudflare.com"]
    with patch.object(sys, "argv", test_args.copy()), patch("situs.cli.app") as mock_app:
        main()
        mock_app.assert_called_once()
        assert sys.argv == ["situs", "join", "https://idol-detroit-educated.trycloudflare.com"]


def test_cors_allows_trycloudflare():
    from situs.coordinator.app import StrictCORSMiddleware
    from starlette.datastructures import Headers

    # Mock app for middleware instance
    dummy_app = lambda scope, receive, send: None
    middleware = StrictCORSMiddleware(dummy_app, allow_origins=["http://localhost:3000"])

    # StrictCORSMiddleware should allow trycloudflare origins
    assert middleware.is_allowed_origin("https://idol-detroit-educated.trycloudflare.com") is True
    assert middleware.is_allowed_origin("https://any-subdomain.trycloudflare.com") is True
    assert middleware.is_allowed_origin("http://localhost:3000") is True
    assert middleware.is_allowed_origin("https://untrusted-malicious-site.com") is False






