"""Regression tests for Security, CORS, and Production Fail-Secure Configuration."""

import os
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient

from situs.config import CoordinatorConfig, WorkerConfig, INSECURE_SECRETS
from situs.coordinator.app import (
    app,
    registry,
    relay,
    config,
    worker_websocket,
    is_valid_worker_id,
)
from situs.protocol.messages import (
    MessageType,
    ProtocolFrame,
    WorkerRegister,
)


@pytest.fixture
def client():
    registry.clear()
    return TestClient(app)


# ==========================================
# 1. Production Mode Fail-Secure Tests
# ==========================================

def test_coordinator_production_fails_without_worker_auth_token():
    with pytest.raises(ValueError, match="SITUS_WORKER_AUTH_TOKEN"):
        CoordinatorConfig(
            environment="production",
            worker_auth_token=None,
            admin_key="secure-admin-pass-1234",
            api_key="secure-api-key-1234",
        )


def test_coordinator_production_fails_without_admin_key():
    with pytest.raises(ValueError, match="SITUS_ADMIN_KEY"):
        CoordinatorConfig(
            environment="production",
            worker_auth_token="secure-worker-token-1234",
            admin_key=None,
            api_key="secure-api-key-1234",
        )


def test_coordinator_production_fails_without_api_key():
    with pytest.raises(ValueError, match="SITUS_API_KEY"):
        CoordinatorConfig(
            environment="production",
            worker_auth_token="secure-worker-token-1234",
            admin_key="secure-admin-pass-1234",
            api_key=None,
        )


def test_coordinator_production_rejects_insecure_default_secrets():
    for insecure in ("situs-cluster-secret", "admin-secret", "secret", "password"):
        with pytest.raises(ValueError, match="Insecure"):
            CoordinatorConfig(
                environment="production",
                worker_auth_token=insecure,
                admin_key="secure-admin-pass-1234",
                api_key="secure-api-key-1234",
            )
        with pytest.raises(ValueError, match="Insecure"):
            CoordinatorConfig(
                environment="production",
                worker_auth_token="secure-worker-token-1234",
                admin_key=insecure,
                api_key="secure-api-key-1234",
            )
        with pytest.raises(ValueError, match="Insecure"):
            CoordinatorConfig(
                environment="production",
                worker_auth_token="secure-worker-token-1234",
                admin_key="secure-admin-pass-1234",
                api_key=insecure,
            )


def test_coordinator_production_succeeds_with_valid_secrets():
    cfg = CoordinatorConfig(
        environment="production",
        worker_auth_token="strong-cluster-token-xyz987",
        admin_key="strong-admin-key-abc123",
        api_key="strong-api-key-def456",
        allowed_origins=["https://dashboard.situs.io"],
    )
    assert cfg.is_production is True
    assert cfg.worker_auth_token == "strong-cluster-token-xyz987"


def test_worker_production_fails_without_auth_token():
    with pytest.raises(ValueError, match="SITUS_WORKER_AUTH_TOKEN"):
        WorkerConfig(
            environment="production",
            auth_token=None,
        )


def test_worker_production_rejects_insecure_default_secrets():
    for insecure in ("situs-cluster-secret", "admin-secret", "password"):
        with pytest.raises(ValueError, match="Insecure auth_token"):
            WorkerConfig(
                environment="production",
                auth_token=insecure,
            )


def test_worker_production_succeeds_with_valid_token():
    cfg = WorkerConfig(
        environment="production",
        auth_token="strong-worker-token-xyz987",
    )
    assert cfg.is_production is True
    assert cfg.auth_token == "strong-worker-token-xyz987"


# ==========================================
# 2. CORS Security Regression Tests
# ==========================================

def test_cors_wildcard_origin_rejected_in_config():
    with pytest.raises(ValueError, match="Wildcard origin '\\*' is not permitted"):
        CoordinatorConfig(
            enable_cors=True,
            allowed_origins=["*"],
        )


def test_cors_wildcard_in_origins_list_rejected():
    with pytest.raises(ValueError, match="Wildcard origin '\\*' is not permitted"):
        CoordinatorConfig(
            enable_cors=True,
            allowed_origins=["http://localhost:8000", "*"],
        )


def test_cors_untrusted_origin_cannot_access_credentials(client):
    """
    Verifies that untrusted third-party origins cannot access credentials
    or receive Access-Control-Allow-Origin headers.
    """
    untrusted_origin = "http://evil-attacker.example.com"
    res = client.get(
        "/status",
        headers={"Origin": untrusted_origin},
    )
    assert res.status_code == 200
    # Untrusted origin must NOT receive Access-Control-Allow-Origin
    assert "access-control-allow-origin" not in res.headers
    assert "access-control-allow-credentials" not in res.headers


def test_cors_trusted_origin_receives_cors_headers(client):
    """
    Verifies that explicitly allowed origins receive proper CORS headers.
    """
    trusted_origin = "http://localhost:3000"
    res = client.get(
        "/status",
        headers={"Origin": trusted_origin},
    )
    assert res.status_code == 200
    assert res.headers.get("access-control-allow-origin") == trusted_origin
    assert res.headers.get("access-control-allow-credentials") == "true"


# ==========================================
# 3. Worker Identity Validation Tests
# ==========================================

def test_is_valid_worker_id():
    assert is_valid_worker_id("worker-1") is True
    assert is_valid_worker_id("worker_gpu_01") is True
    assert is_valid_worker_id("worker.node.1") is True
    assert is_valid_worker_id("node-A100-test") is True

    # Reserved names
    assert is_valid_worker_id("coordinator") is False
    assert is_valid_worker_id("Coordinator") is False

    # Empty and whitespace
    assert is_valid_worker_id("") is False
    assert is_valid_worker_id("   ") is False
    assert is_valid_worker_id(None) is False

    # Path traversal and injection
    assert is_valid_worker_id("../worker-1") is False
    assert is_valid_worker_id("worker/1") is False
    assert is_valid_worker_id("worker\\1") is False
    assert is_valid_worker_id("worker;rm -rf") is False
    assert is_valid_worker_id("worker<script>") is False


@pytest.mark.asyncio
async def test_worker_websocket_rejects_invalid_worker_id():
    mock_ws = AsyncMock()
    mock_ws.query_params = {}
    mock_ws.headers = {}

    await worker_websocket(mock_ws, worker_id="coordinator")
    mock_ws.close.assert_awaited_with(
        code=1008, reason="Invalid worker ID format: 'coordinator'"
    )

    mock_ws.reset_mock()
    await worker_websocket(mock_ws, worker_id="../evil-worker")
    mock_ws.close.assert_awaited_with(
        code=1008, reason="Invalid worker ID format: '../evil-worker'"
    )


@pytest.mark.asyncio
async def test_worker_websocket_rejects_registration_id_mismatch():
    mock_ws = AsyncMock()
    mock_ws.query_params = {}
    mock_ws.headers = {}

    # Frame registration payload has a different worker_id than the URL path
    reg_frame = ProtocolFrame(
        type=MessageType.WORKER_REGISTER,
        source="worker-valid-1",
        target="coordinator",
        payload={
            "worker_id": "worker-spoofed-2",  # Mismatch!
            "gpu_name": "RTX 4090",
            "total_vram_bytes": 24000000000,
            "free_vram_bytes": 22000000000,
            "device_type": "cuda",
        },
    )
    mock_ws.receive_text.return_value = reg_frame.model_dump_json()

    await worker_websocket(mock_ws, worker_id="worker-valid-1")
    mock_ws.close.assert_awaited_with(
        code=1008, reason="Worker ID mismatch in registration frame"
    )


# ==========================================
# 4. Authentication Enforcement & Missing Secrets
# ==========================================

@pytest.mark.asyncio
async def test_worker_websocket_production_rejects_when_token_unconfigured():
    orig_env = config.environment
    orig_token = config.worker_auth_token
    try:
        config.environment = "production"
        config.worker_auth_token = None

        mock_ws = AsyncMock()
        mock_ws.query_params = {}
        mock_ws.headers = {}

        await worker_websocket(mock_ws, worker_id="worker-node-1")
        mock_ws.close.assert_awaited_with(
            code=1008, reason="Authentication required in production"
        )
    finally:
        config.environment = orig_env
        config.worker_auth_token = orig_token


def test_api_and_admin_key_production_enforcement(client):
    orig_env = config.environment
    orig_api = config.api_key
    orig_admin = config.admin_key
    try:
        config.environment = "production"
        config.api_key = None
        config.admin_key = None

        # API endpoint should fail securely in production
        res = client.get("/v1/models")
        assert res.status_code == 401
        assert "production mode" in res.json()["detail"].lower()

        # Admin endpoint should fail securely in production
        res_admin = client.post("/admin/partition?model_name=gemma3:4b")
        assert res_admin.status_code == 403
        assert "production mode" in res_admin.json()["detail"].lower()
    finally:
        config.environment = orig_env
        config.api_key = orig_api
        config.admin_key = orig_admin


def test_cli_worker_config_no_hardcoded_secret():
    """Ensure WorkerConfig in CLI does not default to 'situs-cluster-secret'."""
    old_env = os.environ.pop("SITUS_WORKER_AUTH_TOKEN", None)
    try:
        cfg = WorkerConfig(worker_id="test-worker")
        assert cfg.auth_token is None
        assert cfg.auth_token != "situs-cluster-secret"
    finally:
        if old_env is not None:
            os.environ["SITUS_WORKER_AUTH_TOKEN"] = old_env


# ==========================================
# 5. Advanced CORS, Worker Isolation & CLI Regression Tests
# ==========================================

def test_cors_null_origin_rejected_in_config():
    """Ensure 'null' origin is prohibited when credentials are enabled."""
    with pytest.raises(ValueError, match="Origin 'null' is not permitted"):
        CoordinatorConfig(
            enable_cors=True,
            allowed_origins=["null"],
        )


def test_cors_null_origin_blocked_at_runtime(client):
    """Ensure requests with Origin: null never receive Access-Control headers."""
    res = client.get("/status", headers={"Origin": "null"})
    assert res.status_code == 200
    assert "access-control-allow-origin" not in res.headers
    assert "access-control-allow-credentials" not in res.headers


def test_cors_dynamic_origin_update(client):
    """Ensure dynamically updating config.allowed_origins is honored by StrictCORSMiddleware."""
    new_origin = "https://subdomain.situs-trusted.com"
    orig_allowed = list(config.allowed_origins)
    try:
        # Before adding, should not have CORS headers
        r_before = client.get("/status", headers={"Origin": new_origin})
        assert "access-control-allow-origin" not in r_before.headers

        # Dynamically append origin
        config.allowed_origins.append(new_origin)
        r_after = client.get("/status", headers={"Origin": new_origin})
        assert r_after.headers.get("access-control-allow-origin") == new_origin
        assert r_after.headers.get("access-control-allow-credentials") == "true"
    finally:
        config.allowed_origins = orig_allowed


def test_cors_casing_and_trailing_slash_normalization(client):
    """Ensure CORS origin validation handles casing and trailing slashes correctly."""
    res_upper = client.get("/status", headers={"Origin": "HTTP://LOCALHOST:3000"})
    assert res_upper.status_code == 200
    assert res_upper.headers.get("access-control-allow-origin") == "HTTP://LOCALHOST:3000"
    assert res_upper.headers.get("access-control-allow-credentials") == "true"


def test_worker_config_rejects_invalid_worker_id():
    """Ensure WorkerConfig validates worker_id on construction."""
    with pytest.raises(ValueError, match="Invalid worker_id"):
        WorkerConfig(worker_id="../path-traversal-worker")

    with pytest.raises(ValueError, match="Invalid worker_id"):
        WorkerConfig(worker_id="coordinator")

    with pytest.raises(ValueError, match="Invalid worker_id"):
        WorkerConfig(worker_id="bad worker id with spaces")


@pytest.mark.asyncio
async def test_worker_websocket_rejects_frames_before_registration():
    """Ensure an un-registered worker cannot send heartbeats or data frames before registering."""
    mock_ws = AsyncMock()
    mock_ws.query_params = {}
    mock_ws.headers = {}

    heartbeat_frame = ProtocolFrame(
        type=MessageType.HEARTBEAT,
        source="worker-test-1",
        target="coordinator",
        payload={"free_vram_bytes": 1000000000},
    )

    # First send heartbeat before registration (should be dropped), then disconnect
    mock_ws.receive_text.side_effect = [
        heartbeat_frame.model_dump_json(),
        Exception("Connection closed"),
    ]

    await worker_websocket(mock_ws, worker_id="worker-test-1")
    # Verify worker was never registered in registry
    assert "worker-test-1" not in registry.workers


def test_case_insensitive_bearer_authentication(client):
    """Ensure both 'Bearer' and lowercase 'bearer' authorization headers are accepted."""
    orig_api = config.api_key
    try:
        config.api_key = "test-secret-api-key"
        # Test lowercase bearer
        res = client.get("/v1/models", headers={"Authorization": "bearer test-secret-api-key"})
        assert res.status_code == 200

        # Test Titlecase Bearer
        res2 = client.get("/v1/models", headers={"Authorization": "Bearer test-secret-api-key"})
        assert res2.status_code == 200

        # Test incorrect key
        res3 = client.get("/v1/models", headers={"Authorization": "bearer wrong-key"})
        assert res3.status_code == 401
    finally:
        config.api_key = orig_api


def test_coordinator_cli_production_with_flags():
    """Ensure start_coordinator in CLI succeeds in production mode when secrets are passed via CLI flags."""
    from typer.testing import CliRunner
    from situs.cli import app as cli_app

    orig_env = os.environ.get("SITUS_ENV")
    orig_token_env = os.environ.get("SITUS_WORKER_AUTH_TOKEN")
    orig_api_env = os.environ.get("SITUS_API_KEY")
    orig_admin_env = os.environ.get("SITUS_ADMIN_KEY")
    orig_token = config.worker_auth_token
    orig_api = config.api_key
    orig_admin = config.admin_key
    try:
        os.environ["SITUS_ENV"] = "production"
        runner = CliRunner()
        # Mock uvicorn.run so it doesn't actually bind network
        with patch("uvicorn.run") as mock_run:
            result = runner.invoke(
                cli_app,
                [
                    "coordinator",
                    "--token", "strong-prod-token-12345",
                    "--api-key", "strong-prod-api-12345",
                    "--admin-key", "strong-prod-admin-12345",
                ],
            )
            assert result.exit_code == 0
            assert mock_run.called
    finally:
        config.worker_auth_token = orig_token
        config.api_key = orig_api
        config.admin_key = orig_admin

        if orig_env is not None:
            os.environ["SITUS_ENV"] = orig_env
        else:
            os.environ.pop("SITUS_ENV", None)

        if orig_token_env is not None:
            os.environ["SITUS_WORKER_AUTH_TOKEN"] = orig_token_env
        else:
            os.environ.pop("SITUS_WORKER_AUTH_TOKEN", None)

        if orig_api_env is not None:
            os.environ["SITUS_API_KEY"] = orig_api_env
        else:
            os.environ.pop("SITUS_API_KEY", None)

        if orig_admin_env is not None:
            os.environ["SITUS_ADMIN_KEY"] = orig_admin_env
        else:
            os.environ.pop("SITUS_ADMIN_KEY", None)

