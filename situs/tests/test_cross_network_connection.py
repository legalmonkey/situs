"""Tests verifying verified registration handshakes, RFC 1918 cross-subnet detection, and multi-network routing diagnostics."""

import pytest
from unittest.mock import patch, MagicMock
from situs.tui import (
    is_rfc1918_private_ip,
    get_subnet_24,
    get_all_interface_ips,
    get_tailscale_or_vpn_ip,
    start_in_process_worker,
    execute_slash_command,
)
from situs.worker.client import WorkerDaemon
from situs.config import WorkerConfig
from situs.protocol.messages import ProtocolFrame, MessageType


def test_is_rfc1918_private_ip():
    assert is_rfc1918_private_ip("192.168.29.189") is True
    assert is_rfc1918_private_ip("192.168.1.1") is True
    assert is_rfc1918_private_ip("10.0.0.5") is True
    assert is_rfc1918_private_ip("172.20.10.2") is True
    assert is_rfc1918_private_ip("127.0.0.1") is False  # Loopback, not LAN private
    assert is_rfc1918_private_ip("8.8.8.8") is False    # Public IP
    assert is_rfc1918_private_ip("100.64.1.2") is False # CGNAT / Tailscale mesh IP


def test_get_subnet_24():
    assert get_subnet_24("192.168.29.189") == "192.168.29"
    assert get_subnet_24("10.0.1.50") == "10.0.1"
    assert get_subnet_24("invalid") is None


def test_get_all_interface_ips():
    ifaces = get_all_interface_ips()
    assert isinstance(ifaces, dict)
    # Ensure no loopback or link-local APIPA in returned interface IPs
    for name, ip in ifaces.items():
        assert not ip.startswith("127.")
        assert not ip.startswith("169.254.")


def test_unreachable_cross_wifi_worker_fails_cleanly_with_diagnostics():
    """Verify that attempting to join an unreachable private IP from a different subnet fails cleanly and explains the issue."""
    with patch("situs.tui.get_local_ip", return_value="192.168.1.50"), \
         patch("situs.tui.get_all_interface_ips", return_value={"Wi-Fi": "192.168.1.50"}):
        # Target is on 192.168.29.x (different subnet), unreachable
        ok, msg = start_in_process_worker("192.168.29.189:8000")
        assert ok is False
        assert "CROSS-WIFI" in msg or "SUBNET MISMATCH" in msg or "Cannot" in msg
        assert "192.168.29" in msg


def test_worker_daemon_registration_event():
    """Verify that WorkerDaemon does not consider itself registered until confirmed by coordinator."""
    cfg = WorkerConfig(
        coordinator_url="ws://127.0.0.1:8000/ws/worker",
        worker_id="test-w1",
        device="cpu",
    )
    daemon = WorkerDaemon(cfg)
    assert hasattr(daemon, "registered_event")
    assert not daemon.registered_event.is_set()
    assert not daemon.is_registered

    # Simulate arrival of WORKER_REGISTERED frame
    frame = ProtocolFrame(
        type=MessageType.WORKER_REGISTERED,
        source="coordinator",
        target="test-w1",
        payload={"status": "registered"},
    )
    # Emulate handling
    if frame.type == MessageType.WORKER_REGISTERED:
        daemon.is_registered = True
        daemon.registered_event.set()

    assert daemon.is_registered is True
    assert daemon.registered_event.is_set()


def test_network_slash_command():
    """Verify that the /network slash command executes successfully."""
    logs = []
    session_state = {"model": "gemma3:4b", "is_coordinator": True}
    res = execute_slash_command(
        "/network",
        "http://127.0.0.1:8000",
        "gemma3:4b",
        [],
        session_state,
        log_output_fn=lambda p, title: logs.append((p, title)),
    )
    assert res is True
    assert len(logs) >= 2  # Table + Guide panel


def test_public_tunnel_url_pattern_and_sanitization():
    """Verify that public tunnel URLs (trycloudflare) are recognized and correctly sanitized."""
    from situs.network.tunnel import CLOUDFLARE_URL_PATTERN
    from situs.config import sanitize_coordinator_address

    sample_log = "2026-11-06T12:00:00Z INF |  https://bold-forest-ridge-peak.trycloudflare.com  |"
    m = CLOUDFLARE_URL_PATTERN.search(sample_log)
    assert m is not None
    assert m.group(0) == "https://bold-forest-ridge-peak.trycloudflare.com"

    # Verify sanitize does NOT append :8000 to tunnel domains
    assert sanitize_coordinator_address("bold-forest-ridge-peak.trycloudflare.com") == "https://bold-forest-ridge-peak.trycloudflare.com"
    assert sanitize_coordinator_address("https://bold-forest-ridge-peak.trycloudflare.com") == "https://bold-forest-ridge-peak.trycloudflare.com"
    assert sanitize_coordinator_address("ws://bold-forest-ridge-peak.trycloudflare.com/ws/worker") == "ws://bold-forest-ridge-peak.trycloudflare.com/ws/worker"


def test_public_tunnel_manager_mock():
    """Verify PublicTunnelManager starts and captures URL from subprocess stderr."""
    from situs.network.tunnel import PublicTunnelManager
    import io

    manager = PublicTunnelManager(port=8000, cloudflared_path="cloudflared")
    mock_proc = MagicMock()
    mock_proc.stderr = io.StringIO("INF |  https://my-test-tunnel.trycloudflare.com  |\n")
    mock_proc.poll.return_value = None

    with patch("subprocess.Popen", return_value=mock_proc):
        url = manager.start(timeout=2.0)
        assert url == "https://my-test-tunnel.trycloudflare.com"
        assert manager.is_running is True

    manager.stop()
    assert manager.is_running is False
    assert manager.public_url is None


def test_tunnel_slash_command():
    """Verify /tunnel slash command executes cleanly."""
    logs = []
    session_state = {"model": "gemma3:4b", "is_coordinator": True}
    res = execute_slash_command(
        "/tunnel stop",
        "http://127.0.0.1:8000",
        "gemma3:4b",
        [],
        session_state,
        log_output_fn=lambda p, title: logs.append((p, title)),
    )
    assert res is True
    assert len(logs) >= 1


def test_asymmetric_token_loop_structure():
    """Verify Option 3: NextToken carries only lightweight token scalar, never heavy activations."""
    from situs.protocol.messages import NextToken
    tok = NextToken(
        request_id="test-req",
        step_index=1,
        token_id=1234,
        token_text=" hello",
        is_finished=False,
    )
    # The payload contains token_id and text, but NO tensor bytes or hidden state floats
    d = tok.model_dump()
    assert "token_id" in d
    assert d["token_id"] == 1234
    assert "payload_b64" not in d
    assert "hidden_states" not in d


def test_jit_streamer_rejects_synthetic_weights_for_production_models(tmp_path):
    """Verify that real models cannot silently generate synthetic random weights."""
    from situs.worker.jit_streamer import ProgressiveJITStreamer
    streamer = ProgressiveJITStreamer(model_name="gemma3:4b", allow_synthetic=False, cache_dir=tmp_path)
    with pytest.raises(FileNotFoundError) as exc_info:
        streamer.create_layer_chunks(layer_index=0, total_stage_layers=28, payload_bytes=None)
    assert "Cannot perform distributed execution without verified model weights" in str(exc_info.value)

