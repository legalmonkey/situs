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

    # 4. Test typing '/model ' returns curated 6GB-10GB model choices
    doc_m = Document(text="/model ", cursor_position=7)
    m_completions = list(completer.get_completions(doc_m, None))
    assert len(m_completions) == 8
    assert m_completions[0].text == "/model 1"
    assert "Qwen 2.5 7B" in m_completions[0].display_text

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

    assert execute_slash_command("/help", "http://localhost:8000", "nemotron", messages, session_state, log_output_fn=log_capture) is True
    titles = [t for t, _ in logs]
    assert "Help Commands" in titles
    assert "Worker ID Guide" in titles
    assert "Model Selection Guide" in titles

    # Test /models
    logs.clear()
    assert execute_slash_command("/models", "http://localhost:8000", "nemotron", messages, session_state, log_output_fn=log_capture) is True
    titles = [t for t, _ in logs]
    assert any("Curated 6GB - 10GB" in t for t in titles)
    assert "Select Model" in titles

    # Test /model 1 selects Qwen 2.5 7B
    logs.clear()
    assert execute_slash_command("/model 1", "http://localhost:8000", "nemotron", messages, session_state, log_output_fn=log_capture) is True
    assert session_state["model"] == "qwen2.5:7b-instruct-q8_0"

    # Test /model 2 selects Llama 3.1 8B
    assert execute_slash_command("/model 2", "http://localhost:8000", "nemotron", messages, session_state, log_output_fn=log_capture) is True
    assert session_state["model"] == "llama-3.1:8b-instruct-q8_0"

    # Test /model with no args inspects the active model
    logs.clear()
    assert execute_slash_command("/model", "http://localhost:8000", "nemotron", messages, session_state, log_output_fn=log_capture) is True
    assert any("Active Model" in t for t, _ in logs)

    # Test /join parsing with custom worker-id and device
    assert execute_slash_command("/join 127.0.0.1:8000 worker-2 cpu", "http://localhost:8000", "nemotron", messages, session_state, log_output_fn=log_capture) is True
    assert execute_slash_command("/join 127.0.0.1:8000 --worker-id worker-test --device cpu", "http://localhost:8000", "nemotron", messages, session_state, log_output_fn=log_capture) is True

    # Test /speculative toggle
    assert execute_slash_command("/speculative", "http://localhost:8000", "nemotron", messages, session_state) is True
    assert session_state["speculative"] is False
    assert execute_slash_command("/speculative", "http://localhost:8000", "nemotron", messages, session_state) is True
    assert session_state["speculative"] is True

    # Test /p2p toggle
    assert execute_slash_command("/p2p", "http://localhost:8000", "nemotron", messages, session_state) is True
    assert session_state["p2p"] is False

    # Test /clear
    assert execute_slash_command("/clear", "http://localhost:8000", "nemotron", messages, session_state) is True
    assert len(messages) == 1

    # Test in-interface coordinator and join commands
    assert execute_slash_command("situs coordinator", "http://localhost:8000", "nemotron", messages, session_state) is True
    assert execute_slash_command("/coordinator", "http://localhost:8000", "nemotron", messages, session_state) is True
    assert execute_slash_command("situs status", "http://localhost:8000", "nemotron", messages, session_state) is True
    assert execute_slash_command("help", "http://localhost:8000", "nemotron", messages, session_state) is True

    # Test /exit
    assert execute_slash_command("/exit", "http://localhost:8000", "nemotron", messages, session_state) is False

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
            {"active_workers_count": 2, "model": "nemotron-3-nano:4b-q8_0"},
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
            {"active_workers_count": 0, "model": "nemotron"},
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
            model="nemotron-3-nano:4b-q8_0",
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

    # 3. Test DistributedStageModel safe fallback: requesting cuda:0 on cpu-only torch should not crash
    stage = load_hf_stage_model(
        model_name="nemotron",
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
    session_state = {"model": "nemotron", "is_coordinator": True}
    captured = []
    def log_capture(title, renderable):
        captured.append((title, str(renderable)))

    assert execute_slash_command("/join 127.0.0.1:8000 worker-hw-test", "http://localhost:8000", "nemotron", messages, session_state, log_output_fn=log_capture) is True
    assert any("Worker Joined" in t or "Worker" in t for t, _ in captured)


