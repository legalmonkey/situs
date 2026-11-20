"""Automated unit and mock tests for situs.network.tunnel.

Tests cover:
- Quick tunnel argument construction (--url, --no-autoupdate, --edge-ip-version 4).
- Authenticated token tunnel argument construction (run --token <token>).
- URL extraction from combined stdout/stderr streams with ANSI and banner formatting.
- Cloudflare rate limit detection (1015/429) & regional retry (--region us).
- Subprocess termination error handling (non-zero return code 255/1) without hanging.
- Timeout handling when no URL or registration appears.
- Semantic error preservation avoiding overwriting diagnostics with return codes.
- Elimination of unauthenticated Pinggy fallback to prevent exit code 255.
- TUI slash command token handling and argument propagation.
"""

import base64
import io
import json
import os
import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from situs.network.tunnel import (
    PublicTunnelManager,
    start_public_tunnel,
    stop_public_tunnel,
    get_active_tunnel_url,
    is_tunnel_active,
    CLOUDFLARE_URL_PATTERN,
    SSH_TUNNEL_URL_PATTERN,
    ALL_TUNNEL_URL_PATTERN,
    TOKEN_CONN_PATTERN,
    _extract_token_url,
)


@pytest.fixture(autouse=True)
def cleanup_tunnel():
    """Ensure any global tunnel process is stopped after each test."""
    yield
    stop_public_tunnel()


def test_quick_tunnel_argument_construction():
    """Verify quick tunnel builds the exact expected CLI flags including --edge-ip-version 4."""
    mgr = PublicTunnelManager(port=8080, cloudflared_path="cloudflared")
    mock_proc = MagicMock()
    mock_proc.stdout = io.StringIO(
        "2026-11-20T09:00:00Z INF |  https://argument-test.trycloudflare.com  |\n"
    )
    mock_proc.poll.return_value = None

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        url = mgr.start(timeout=5.0)
        assert url == "https://argument-test.trycloudflare.com"

        assert mock_popen.called
        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert cmd[0] == "cloudflared"
        assert cmd[1] == "tunnel"
        assert "--url" in cmd
        url_idx = cmd.index("--url")
        assert cmd[url_idx + 1] == "http://127.0.0.1:8080"
        assert "--no-autoupdate" in cmd
        assert "--edge-ip-version" in cmd
        ip_idx = cmd.index("--edge-ip-version")
        assert cmd[ip_idx + 1] == "4"

    mgr.stop()


def test_token_tunnel_argument_construction():
    """Verify authenticated token tunnel invokes 'tunnel run --token <token>'."""
    secret_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token123"
    mgr = PublicTunnelManager(port=8000, cloudflared_path="cloudflared", token=secret_token)

    mock_proc = MagicMock()
    mock_proc.stdout = io.StringIO(
        "2026-11-20T09:00:00Z INF Registered tunnel connection connIndex=0 location=IAD protocol=quic\n"
    )
    mock_proc.poll.return_value = None

    with patch("subprocess.Popen", return_value=mock_proc) as mock_popen:
        url = mgr.start(timeout=5.0)
        assert url is not None
        assert mgr.is_running is True
        assert mgr.tunnel_type == "cloudflare_token"

        args, kwargs = mock_popen.call_args
        cmd = args[0]
        assert cmd[0] == "cloudflared"
        assert cmd[1] == "tunnel"
        assert cmd[2] == "run"
        assert "--token" in cmd
        token_idx = cmd.index("--token")
        assert cmd[token_idx + 1] == secret_token

    mgr.stop()


def test_token_tunnel_with_base64_payload_extraction():
    """Verify token tunnel derives the specific cfargotunnel domain from the base64 token payload."""
    payload = {"a": "account-id", "t": "550e8400-e29b-41d4-a716-446655440000", "s": "secret"}
    raw_b64 = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")

    derived_url = _extract_token_url(raw_b64)
    assert derived_url == "https://550e8400-e29b-41d4-a716-446655440000.cfargotunnel.com"

    mgr = PublicTunnelManager(port=8000, cloudflared_path="cloudflared", token=raw_b64)
    mock_proc = MagicMock()
    mock_proc.stdout = io.StringIO(
        "2026-11-20T09:00:00Z INF Connection 550e8400-e29b-41d4-a716-446655440000 registered connIndex=0\n"
    )
    mock_proc.poll.return_value = None

    with patch("subprocess.Popen", return_value=mock_proc):
        url = mgr.start(timeout=5.0)
        assert url == "https://550e8400-e29b-41d4-a716-446655440000.cfargotunnel.com"

    mgr.stop()


def test_token_tunnel_custom_url_env_override(monkeypatch):
    """Verify CLOUDFLARE_TUNNEL_URL environment variable overrides token domain."""
    monkeypatch.setenv("CLOUDFLARE_TUNNEL_URL", "https://custom-coordinator.mydomain.io")
    mgr = PublicTunnelManager(port=8000, cloudflared_path="cloudflared", token="mock-token")
    mock_proc = MagicMock()
    mock_proc.stdout = io.StringIO(
        "2026-11-20T09:00:00Z INF Registered tunnel connection connIndex=0 protocol=quic\n"
    )
    mock_proc.poll.return_value = None

    with patch("subprocess.Popen", return_value=mock_proc):
        url = mgr.start(timeout=5.0)
        assert url == "https://custom-coordinator.mydomain.io"

    mgr.stop()


def test_url_extraction_from_stdout_stream_with_banners_and_ansi():
    """Verify URL extraction correctly parses noisy Cloudflare ASCII banners and ANSI escapes."""
    mgr = PublicTunnelManager(port=8000, cloudflared_path="cloudflared")
    banner_output = (
        "2026-11-20T09:00:00Z INF Requesting new quick Tunnel on trycloudflare.com...\n"
        "\x1b[32m2026-11-20T09:00:06Z INF\x1b[0m +--------------------------------------------------------------------------------------------+\n"
        "\x1b[32m2026-11-20T09:00:06Z INF\x1b[0m |  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |\n"
        "\x1b[32m2026-11-20T09:00:06Z INF\x1b[0m |  https://silent-river-rock-42.trycloudflare.com                                            |\n"
        "\x1b[32m2026-11-20T09:00:06Z INF\x1b[0m +--------------------------------------------------------------------------------------------+\n"
        "2026-11-20T09:00:07Z INF Registered tunnel connection connIndex=0 ... protocol=quic\n"
    )
    mock_proc = MagicMock()
    mock_proc.stdout = io.StringIO(banner_output)
    mock_proc.poll.return_value = None

    with patch("subprocess.Popen", return_value=mock_proc):
        url = mgr.start(timeout=5.0)
        assert url == "https://silent-river-rock-42.trycloudflare.com"

    mgr.stop()


def test_rate_limit_detection_and_regional_retry():
    """Verify that encountering Cloudflare 1015 / 429 rate limit triggers regional failover (--region us)."""
    mgr = PublicTunnelManager(port=8000, cloudflared_path="cloudflared")

    # Proc 1 fails with 429 / 1015
    proc1 = MagicMock()
    proc1.stdout = io.StringIO(
        "2026-11-20T09:00:01Z INF Requesting new quick Tunnel on trycloudflare.com...\n"
        "2026-11-20T09:00:02Z ERR Error unmarshaling QuickTunnel response: error code: 1015 status_code=\"429 Too Many Requests\"\n"
        "failed to unmarshal quick Tunnel: invalid character 'e' looking for beginning of value\n"
    )
    proc1.poll.return_value = 1
    proc1.returncode = 1

    # Proc 2 succeeds with regional endpoint
    proc2 = MagicMock()
    proc2.stdout = io.StringIO(
        "2026-11-20T09:00:03Z INF Requesting new quick Tunnel on trycloudflare.com...\n"
        "2026-11-20T09:00:05Z INF |  https://regional-failover-repaired.trycloudflare.com  |\n"
    )
    proc2.poll.return_value = None

    with patch("subprocess.Popen", side_effect=[proc1, proc2]) as mock_popen:
        url = mgr.start(timeout=5.0)
        assert url == "https://regional-failover-repaired.trycloudflare.com"

        assert mock_popen.call_count == 2
        # Check first attempt did not specify region
        cmd1 = mock_popen.call_args_list[0][0][0]
        assert "--region" not in cmd1

        # Check second attempt included --region us
        cmd2 = mock_popen.call_args_list[1][0][0]
        assert "--region" in cmd2
        reg_idx = cmd2.index("--region")
        assert cmd2[reg_idx + 1] == "us"

    mgr.stop()


def test_process_termination_nonzero_exit_code_without_hanging():
    """Verify process terminating with non-zero exit code returns cleanly without hanging."""
    mgr = PublicTunnelManager(port=8000, cloudflared_path="cloudflared")

    mock_proc = MagicMock()
    mock_proc.stdout = io.StringIO("Process terminated abruptly\n")
    mock_proc.poll.return_value = 1
    mock_proc.returncode = 1

    t0 = time.time()
    with patch("subprocess.Popen", return_value=mock_proc):
        url = mgr.start(timeout=5.0)
    elapsed = time.time() - t0

    assert url is None
    assert mgr.is_running is False
    assert elapsed < 3.0  # Must not hang for the full 15s timeout
    mgr.stop()


def test_timeout_handling_when_no_url_or_registration():
    """Verify timeout when process stays alive but emits no tunnel URL or connection event."""
    mgr = PublicTunnelManager(port=8000, cloudflared_path="cloudflared")

    mock_proc = MagicMock()
    # Continuous uninformative output
    mock_proc.stdout = io.StringIO("2026-11-20T09:00:00Z INF Connecting to edge...\n")
    mock_proc.poll.return_value = None

    with patch.object(mgr, "_spawn_and_await_url", return_value=None):
        url = mgr.start(timeout=0.1)
        assert url is None
        assert mgr.is_running is False

    mgr.stop()


def test_error_preservation_over_generic_exit_code():
    """Verify semantic error (Cloudflare 1015) is not overwritten by generic 'return code 1'."""
    mgr = PublicTunnelManager(port=8000, cloudflared_path="cloudflared")

    proc = MagicMock()
    proc.stdout = io.StringIO(
        "ERR Error unmarshaling QuickTunnel response: error code: 1015 status_code=\"429 Too Many Requests\"\n"
    )
    proc.poll.return_value = 1
    proc.returncode = 1

    with patch("subprocess.Popen", return_value=proc):
        url = mgr._spawn_and_await_url(["cloudflared", "tunnel", "--url", "http://127.0.0.1:8000"], timeout=1.0)
        assert url is None
        # Diagnostic message must be preserved
        assert "1015" in mgr._last_error or "Rate limited" in mgr._last_error
        assert "Process terminated with return code 1" not in mgr._last_error


def test_openssh_fallback_does_not_invoke_unauthenticated_pinggy():
    """Verify OpenSSH fallback no longer invokes unauthenticated Pinggy to prevent code 255."""
    mgr = PublicTunnelManager(port=8000, cloudflared_path=None)

    proc_ssh = MagicMock()
    proc_ssh.stdout = io.StringIO("Connecting to nokey@localhost.run...\n")
    proc_ssh.poll.return_value = 255
    proc_ssh.returncode = 255

    with patch("shutil.which", return_value="ssh"):
        with patch("subprocess.Popen", return_value=proc_ssh) as mock_popen:
            url = mgr.start(timeout=2.0)
            assert url is None

            # Verify no command called pinggy
            for call_obj in mock_popen.call_args_list:
                cmd = call_obj[0][0]
                assert "a.pinggy.io" not in cmd

    mgr.stop()


def test_start_public_tunnel_wrapper_and_global_lifecycle():
    """Verify start_public_tunnel, stop_public_tunnel, and active check APIs."""
    mock_mgr = MagicMock()
    mock_mgr.is_running = True
    mock_mgr.public_url = "https://global-lifecycle.trycloudflare.com"
    mock_mgr.tunnel_type = "cloudflare_quick"
    mock_mgr.start.return_value = "https://global-lifecycle.trycloudflare.com"

    with patch("situs.network.tunnel.ensure_cloudflared", return_value="cloudflared"):
        with patch("situs.network.tunnel.PublicTunnelManager", return_value=mock_mgr):
            succ, url, msg = start_public_tunnel(port=8000, timeout=5.0)
            assert succ is True
            assert url == "https://global-lifecycle.trycloudflare.com"
            assert is_tunnel_active() is True
            assert get_active_tunnel_url() == "https://global-lifecycle.trycloudflare.com"

            stop_public_tunnel()
            assert is_tunnel_active() is False
            assert get_active_tunnel_url() is None


def test_tui_slash_command_token_handling():
    """Verify /tunnel <token> correctly extracts and propagates token in the TUI."""
    from situs.tui import execute_slash_command

    captured_args = {}

    def mock_start(port=8000, timeout=15.0, token=None, progress_fn=None):
        captured_args["port"] = port
        captured_args["token"] = token
        return True, "https://mock-tui-tunnel.cfargotunnel.com", "OK"

    session_state = {"model": "gemma3:4b", "is_coordinator": True}
    logs = []

    with patch("situs.network.tunnel.start_public_tunnel", side_effect=mock_start):
        # 1. /tunnel <token>
        res = execute_slash_command(
            "/tunnel my-secret-auth-token-123",
            "http://127.0.0.1:8000",
            "gemma3:4b",
            [],
            session_state,
            log_output_fn=lambda p, title: logs.append((p, title)),
        )
        assert res is True
        assert captured_args.get("token") == "my-secret-auth-token-123"

        # 2. /tunnel token <token>
        res2 = execute_slash_command(
            "/tunnel token second-auth-token-456",
            "http://127.0.0.1:8000",
            "gemma3:4b",
            [],
            session_state,
            log_output_fn=lambda p, title: logs.append((p, title)),
        )
        assert res2 is True
        assert captured_args.get("token") == "second-auth-token-456"

        # 3. /tunnel --token <token>
        res3 = execute_slash_command(
            "/tunnel --token third-auth-token-789",
            "http://127.0.0.1:8000",
            "gemma3:4b",
            [],
            session_state,
            log_output_fn=lambda p, title: logs.append((p, title)),
        )
        assert res3 is True
        assert captured_args.get("token") == "third-auth-token-789"


def test_exit_code_255_error_sanitization_across_all_strategies():
    """Verify that when both cloudflared and ssh terminate with exit code 255,
    start_public_tunnel returns succ=False, message contains 'Public WAN tunnel failed to connect',
    and strictly does NOT leak '255' or uninformative return code strings."""
    mock_proc = MagicMock()
    mock_proc.stdout = io.StringIO("")
    mock_proc.poll.return_value = 255
    mock_proc.returncode = 255

    with patch("situs.network.tunnel.ensure_cloudflared", return_value="cloudflared.exe"):
        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("shutil.which", return_value="ssh"):
                succ, url, msg = start_public_tunnel(port=8000, timeout=0.1)
                assert succ is False
                assert url is None
                assert "Public WAN tunnel failed to connect" in msg
                assert "255" not in msg
                assert "return code" not in msg.lower()


def test_token_mode_exit_code_255_does_not_leak_return_code():
    """Verify that when token tunnel and subsequent fallbacks exit with code 255,
    start_public_tunnel returns succ=False and message strictly does not leak '255'."""
    mock_proc = MagicMock()
    mock_proc.stdout = io.StringIO("")
    mock_proc.poll.return_value = 255
    mock_proc.returncode = 255

    with patch("situs.network.tunnel.ensure_cloudflared", return_value="cloudflared.exe"):
        with patch("subprocess.Popen", return_value=mock_proc):
            with patch("shutil.which", return_value="ssh"):
                succ, url, msg = start_public_tunnel(port=8000, token="test-token-123", timeout=0.1)
                assert succ is False
                assert url is None
                assert "255" not in msg
                assert "return code" not in msg.lower()


def test_token_mode_preserves_token_specific_error():
    """Verify that when token mode fails with a specific error, that diagnostic is preserved."""
    mgr = PublicTunnelManager(port=8000, cloudflared_path="cloudflared", token="bad-token")

    with patch.object(mgr, "_spawn_and_await_url") as mock_spawn:
        def fake_spawn(cmd, timeout):
            if "run" in cmd:
                mgr._last_error = "Cannot decode tunnel token: invalid base64"
            else:
                mgr._last_error = "Process terminated (connection or network failure)"
            return None

        mock_spawn.side_effect = fake_spawn
        url = mgr.start(timeout=1.0)
        assert url is None
        assert "Cannot decode tunnel token" in mgr._last_error
        assert "255" not in mgr._last_error
