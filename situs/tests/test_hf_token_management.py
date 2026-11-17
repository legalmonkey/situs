"""Unit tests verifying Hugging Face token resolution, persistence, and CLI/TUI integration."""

import json
import os
import pytest
from unittest.mock import patch, MagicMock
from typer.testing import CliRunner

from situs.token_manager import (
    get_hf_token,
    get_hf_token_source,
    mask_token,
    validate_hf_token,
    save_hf_token,
    clear_hf_token,
    is_hf_token_configured,
)
from situs.cli import app
from situs.tui import OpenCodeSlashCompleter, execute_slash_command
from prompt_toolkit.document import Document

runner = CliRunner()


def test_token_resolution_hierarchy(tmp_path, monkeypatch):
    """Verify strict resolution order: Environment -> HF Cache -> Situs Config -> None."""
    fake_home = tmp_path / "user_home"
    fake_home.mkdir()
    monkeypatch.setattr("os.path.expanduser", lambda p: str(fake_home / p.replace("~/", "")) if p.startswith("~") else p)
    monkeypatch.setenv("SITUS_CONFIG_DIR", str(fake_home / ".situs"))

    # 1. Initially nothing is configured
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)
    assert get_hf_token() is None
    assert is_hf_token_configured() is False
    assert get_hf_token_source() is None

    # 2. Add to ~/.situs/config.json
    situs_cfg = fake_home / ".situs" / "config.json"
    situs_cfg.parent.mkdir(parents=True, exist_ok=True)
    situs_cfg.write_text(json.dumps({"hf_token": "hf_from_situs_cfg"}), encoding="utf-8")
    assert get_hf_token() == "hf_from_situs_cfg"
    assert "Situs Configuration" in get_hf_token_source()

    # 3. Add to Hugging Face Hub cache (~/.cache/huggingface/token) -> overrides Situs config
    hf_cache = fake_home / ".cache" / "huggingface" / "token"
    hf_cache.parent.mkdir(parents=True, exist_ok=True)
    hf_cache.write_text("hf_from_cache_file", encoding="utf-8")
    monkeypatch.setattr("huggingface_hub.get_token", lambda: "hf_from_cache_file")
    assert get_hf_token() == "hf_from_cache_file"
    assert "Hugging Face Cache" in get_hf_token_source()

    # 4. Add HUGGING_FACE_HUB_TOKEN environment variable -> overrides Cache
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_from_hub_env")
    assert get_hf_token() == "hf_from_hub_env"
    assert "HUGGING_FACE_HUB_TOKEN" in get_hf_token_source()

    # 5. Add HF_TOKEN environment variable -> overrides HUGGING_FACE_HUB_TOKEN
    monkeypatch.setenv("HF_TOKEN", "hf_from_hf_token_env")
    assert get_hf_token() == "hf_from_hf_token_env"
    assert "HF_TOKEN" in get_hf_token_source()


def test_token_masking():
    """Verify mask_token securely hides secret characters and formats cleanly."""
    assert mask_token(None) == "Not configured"
    assert mask_token("") == "Not configured"
    assert mask_token("short") == "••••••••"
    masked = mask_token("hf_abcdef1234567890xyz")
    assert masked.startswith("hf_")
    assert masked.endswith("0xyz")
    assert "abcdef" not in masked
    assert "•" in masked


def test_validate_and_save_hf_token(tmp_path, monkeypatch):
    """Verify save_hf_token validates and writes to both HF cache and Situs config."""
    fake_home = tmp_path / "home2"
    fake_home.mkdir()
    monkeypatch.setattr("os.path.expanduser", lambda p: str(fake_home / p.replace("~/", "")) if p.startswith("~") else p)
    monkeypatch.setenv("SITUS_CONFIG_DIR", str(fake_home / ".situs"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    with patch("huggingface_hub.whoami", return_value={"name": "test_user_ritvik"}), \
         patch("huggingface_hub.login") as mock_login:
        succ, msg, user = save_hf_token("hf_valid_test_token_12345", validate=True)
        assert succ is True
        assert user == "test_user_ritvik"
        mock_login.assert_called_once_with(token="hf_valid_test_token_12345", add_to_git_credential=False)

        # Check os.environ updated
        assert os.environ.get("HF_TOKEN") == "hf_valid_test_token_12345"

        # Check Situs config updated
        situs_cfg = fake_home / ".situs" / "config.json"
        assert situs_cfg.is_file()
        data = json.loads(situs_cfg.read_text(encoding="utf-8"))
        assert data.get("hf_token") == "hf_valid_test_token_12345"

    # Test clear_hf_token
    clear_hf_token()
    assert os.environ.get("HF_TOKEN") is None
    cfg_after = json.loads(situs_cfg.read_text(encoding="utf-8"))
    assert "hf_token" not in cfg_after


def test_cli_hf_commands(tmp_path, monkeypatch):
    """Verify CLI `situs hf` inspect, save, and clear commands."""
    fake_home = tmp_path / "home3"
    fake_home.mkdir()
    monkeypatch.setattr("os.path.expanduser", lambda p: str(fake_home / p.replace("~/", "")) if p.startswith("~") else p)
    monkeypatch.setenv("SITUS_CONFIG_DIR", str(fake_home / ".situs"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr("huggingface_hub.get_token", lambda: None)

    # 1. Unconfigured status
    res = runner.invoke(app, ["hf"])
    assert res.exit_code == 0
    assert "Not Configured" in res.output or "Hugging Face Authentication Status" in res.output

    # 2. Configure token
    with patch("huggingface_hub.whoami", return_value={"name": "ritvik"}), patch("huggingface_hub.login"):
        res_save = runner.invoke(app, ["hf", "hf_sample_cli_token_9999"])
        assert res_save.exit_code == 0
        assert "successfully configured" in res_save.output.lower()

    # 3. Clear token
    res_clear = runner.invoke(app, ["hf", "--clear"])
    assert res_clear.exit_code == 0
    assert "cleared" in res_clear.output.lower()


def test_tui_slash_hf_completions_and_execution(tmp_path, monkeypatch):
    """Verify /hf completion in TUI and execution of /hf command."""
    fake_home = tmp_path / "home4"
    fake_home.mkdir()
    monkeypatch.setattr("os.path.expanduser", lambda p: str(fake_home / p.replace("~/", "")) if p.startswith("~") else p)
    monkeypatch.setenv("SITUS_CONFIG_DIR", str(fake_home / ".situs"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)

    completer = OpenCodeSlashCompleter()
    doc = Document(text="/hf ", cursor_position=4)
    completions = [c.text for c in completer.get_completions(doc, None)]
    assert "/hf status" in completions
    assert "/hf clear" in completions

    # Execute /hf status in TUI
    outputs = []
    def capture_log(title, renderable):
        content = str(getattr(renderable, "renderable", renderable))
        panel_title = str(getattr(renderable, "title", ""))
        outputs.append((title, content, panel_title))

    session = {}
    execute_slash_command("/hf", "http://127.0.0.1:8000", "gemma3:4b", [], session, log_output_fn=capture_log)
    assert len(outputs) > 0
    assert "Hugging Face" in outputs[-1][0] or "Authentication" in outputs[-1][2]

    # Execute /hf <token> in TUI
    with patch("huggingface_hub.whoami", return_value={"name": "test_tui_user"}), patch("huggingface_hub.login"):
        execute_slash_command("/hf hf_tui_test_1234", "http://127.0.0.1:8000", "gemma3:4b", [], session, log_output_fn=capture_log)
        assert any("Saved" in out[1] or "Configured" in out[2] for out in outputs)


@pytest.mark.asyncio
async def test_stream_pull_model_guides_user_on_auth_error():
    """Verify stream_pull_model catches gated repo 401 errors and yields requires_auth."""
    from situs.inference_backend import stream_pull_model

    with patch("huggingface_hub.HfApi") as mock_api_cls:
        mock_api = MagicMock()
        mock_api_cls.return_value = mock_api
        mock_api.model_info.side_effect = RuntimeError("401 Client Error: Access to model google/gemma-3-4b-it is restricted. Please log in.")

        events = []
        async for item in stream_pull_model("google/gemma-3-4b-it"):
            events.append(item)

        err_event = next((e for e in events if e.get("type") == "model_loading_error"), None)
        assert err_event is not None
        assert err_event.get("requires_auth") is True
        assert "/hf" in err_event.get("error", "")
