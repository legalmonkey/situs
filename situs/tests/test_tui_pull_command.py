"""Unit tests for the /pull slash command, autocompleter, and streaming progress."""

import asyncio
import pytest
from unittest.mock import patch, MagicMock
from prompt_toolkit.document import Document

from situs.tui import (
    OpenCodeSlashCompleter,
    execute_slash_command,
    DISTRIBUTED_MODELS_6GB_10GB,
    _handle_pull_chunk,
)
from situs.inference_backend import stream_pull_model


def _renderable_to_str(obj):
    if hasattr(obj, "renderable"):
        return str(obj.renderable)
    return str(obj)


def test_pull_command_in_commands_list():
    """Verify /pull is registered in OpenCodeSlashCompleter.COMMANDS."""
    cmd_names = [cmd for cmd, _ in OpenCodeSlashCompleter.COMMANDS]
    assert "/pull" in cmd_names


def test_pull_completer_all_models():
    """Verify /pull with trailing space yields exactly 1 clean autocompletion per curated model."""
    completer = OpenCodeSlashCompleter()
    doc = Document("/pull ")
    completions = list(completer.get_completions(doc, None))

    assert len(completions) == len(DISTRIBUTED_MODELS_6GB_10GB)
    comp_texts = [c.text for c in completions]

    # Check that each numerical index is present once
    for idx, m in enumerate(DISTRIBUTED_MODELS_6GB_10GB, start=1):
        assert f"/pull {idx}" in comp_texts
        c = next(c for c in completions if c.text == f"/pull {idx}")
        # Verify display format and metadata
        assert f"/pull {idx} ({m['name']})" in str(c.display)
        assert m["hf_repo"] in str(c.display_meta)
        assert f"{m['vram_gb']} GB" in str(c.display_meta)
        # Ensure no legacy GGUF tags appear in display name
        assert "(Q8_0)" not in str(c.display)
        assert "(Q6_K)" not in str(c.display)


def test_pull_completer_numeric_index_filter():
    """Verify typing /pull 3 matches only model 3 and does not match models with '3' in name."""
    completer = OpenCodeSlashCompleter()
    doc = Document("/pull 3")
    completions = list(completer.get_completions(doc, None))

    comp_texts = [c.text for c in completions]
    assert "/pull 3" in comp_texts
    # Llama 3.1 and Gemma 3 have '3' in their text, but their index is not 3, so must be excluded
    assert "/pull 1" not in comp_texts
    assert "/pull 5" not in comp_texts
    assert len(completions) == 1
    assert "DeepSeek R1 Distill Qwen 7B" in str(completions[0].display)


def test_pull_completer_filter():
    """Verify typing /pull with partial text filters the model list accurately."""
    completer = OpenCodeSlashCompleter()
    doc = Document("/pull gem")
    completions = list(completer.get_completions(doc, None))
    comp_texts = [c.text for c in completions]

    assert "/pull 4" in comp_texts  # Google Gemma 2 9B
    assert "/pull 5" in comp_texts  # Google Gemma 3 4B
    assert "/pull 1" not in comp_texts  # Llama 3.1


def test_pull_completer_repo_slash_filter():
    """Verify typing /pull with repo namespace yields the canonical repo path."""
    completer = OpenCodeSlashCompleter()
    doc = Document("/pull meta-llama/")
    completions = list(completer.get_completions(doc, None))
    comp_texts = [c.text for c in completions]

    assert "/pull meta-llama/Llama-3.1-8B-Instruct" in comp_texts
    assert "/pull meta-llama/Llama-3.2-3B-Instruct" in comp_texts


def test_execute_slash_command_pull_no_args():
    """Verify running /pull without arguments prints the curated models table."""
    captured = []
    def mock_emit(title, renderable):
        captured.append((title, renderable))

    session_state = {}
    res = execute_slash_command(
        cmd="/pull",
        coordinator_url="http://127.0.0.1:8000",
        model="gemma3:4b",
        messages=[],
        session_state=session_state,
        log_output_fn=mock_emit,
    )

    assert res is True
    assert len(captured) >= 2
    titles = [t for t, _ in captured]
    assert "Pull Models" in titles
    assert "Pull Model Guide" in titles


def test_execute_slash_command_pull_already_cached():
    """Verify /pull reports already cached when model weights exist locally."""
    captured = []
    def mock_emit(title, renderable):
        captured.append((title, renderable))

    session_state = {}
    with patch("situs.inference_backend.is_model_downloaded", return_value=True):
        res = execute_slash_command(
            cmd="/pull 5",
            coordinator_url="http://127.0.0.1:8000",
            model="gemma3:4b",
            messages=[],
            session_state=session_state,
            log_output_fn=mock_emit,
        )

    assert res is True
    assert any("already cached" in _renderable_to_str(r) for _, r in captured)


def test_execute_slash_command_pull_triggers_download():
    """Verify /pull resolves index/id and invokes background pull streaming when not cached."""
    captured = []
    def mock_emit(title, renderable):
        captured.append((title, renderable))

    session_state = {}
    with patch("situs.inference_backend.is_model_downloaded", return_value=False), \
         patch("situs.tui._trigger_model_pull_streaming") as mock_trigger:
        res = execute_slash_command(
            cmd="/pull 5",
            coordinator_url="http://127.0.0.1:8000",
            model="gemma3:4b",
            messages=[],
            session_state=session_state,
            log_output_fn=mock_emit,
        )

    assert res is True
    assert any("Starting Hugging Face Hub streaming download" in _renderable_to_str(r) for _, r in captured)
    mock_trigger.assert_called_once()
    args = mock_trigger.call_args[0]
    assert args[0] == "gemma3:4b"


def test_handle_pull_chunk_updates_session_state():
    """Verify _handle_pull_chunk correctly updates session_state and calls log_output_fn with LOADING title."""
    session_state = {}
    logs = []
    def mock_log(title, renderable):
        logs.append((title, renderable))

    chunk = {
        "type": "model_loading",
        "stage": "Stage 3/3: Streaming [5/13] model.safetensors",
        "percent": 45.5,
        "completed_bytes": 4550000000,
        "total_bytes": 10000000000,
        "speed_mb": 25.5,
        "eta_s": 213,
    }
    _handle_pull_chunk(chunk, session_state, mock_log)

    assert session_state.get("is_pulling") is True
    assert session_state.get("pull_percent") == 45.5
    assert session_state.get("pull_speed") == "25.5 MB/s"
    assert len(logs) == 1
    assert logs[0][0] == "LOADING"
    assert "45.5%" in logs[0][1]


def test_handle_pull_chunk_clamps_completed_bytes():
    """Verify _handle_pull_chunk clamps completed_bytes so it never renders exceeding total_bytes."""
    session_state = {}
    logs = []
    def mock_log(title, renderable):
        logs.append((title, renderable))

    chunk = {
        "type": "model_loading",
        "stage": "Stage 3/3: Streaming [2/2] gemma-3-4b-it-q4_0.gguf",
        "percent": 99.0,
        "completed_bytes": 3400000000,  # 3.40 GB
        "total_bytes": 2490000000,      # 2.49 GB (overflow)
        "speed_mb": 0.8,
        "eta_s": 0,
    }
    _handle_pull_chunk(chunk, session_state, mock_log)
    assert len(logs) == 1
    rendered_text = logs[0][1]
    # Completed bytes must be clamped to 2.49 / 2.49 GB, NEVER 3.40 / 2.49 GB
    assert "3.40" not in rendered_text
    assert "2.49 / 2.49 GB" in rendered_text


@pytest.mark.asyncio
async def test_stream_pull_model_chunked_progress():
    """Verify stream_pull_model yields chunked progress frames during file download."""
    with patch("huggingface_hub.HfApi") as mock_api_cls, \
         patch("huggingface_hub.hf_hub_download") as mock_dl:

        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        sibling = MagicMock()
        sibling.rfilename = "weights.safetensors"
        sibling.size = 10000
        mock_api.model_info.return_value = MagicMock(siblings=[sibling])

        def fake_download(repo_id, filename, token, tqdm_class=None):
            if tqdm_class:
                # Simulate chunk updates
                inst = tqdm_class(total=10000)
                inst.update(3000)
                inst.update(3000)
                inst.update(4000)
            return "/tmp/fake/weights.safetensors"

        mock_dl.side_effect = fake_download

        events = []
        async for chunk in stream_pull_model("test-model"):
            events.append(chunk)

        # Verify we got model_loading events with progress
        loading_events = [e for e in events if e.get("type") == "model_loading"]
        assert len(loading_events) >= 2
        last_event = events[-1]
        assert last_event["status"] == "success"
