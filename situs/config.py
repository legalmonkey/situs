import os
import re
from dataclasses import dataclass, field
from typing import Optional, List


WORKER_ID_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]{1,128}$")


def is_valid_worker_id(worker_id: str) -> bool:
    """Validates worker identity format, rejecting empty names, path traversals, and reserved IDs."""
    if not worker_id or not isinstance(worker_id, str):
        return False
    worker_id = worker_id.strip()
    if not worker_id or worker_id.lower() == "coordinator":
        return False
    if ".." in worker_id or "/" in worker_id or "\\" in worker_id:
        return False
    return bool(WORKER_ID_REGEX.match(worker_id))


INSECURE_SECRETS = {
    "situs-cluster-secret",
    "admin-secret",
    "secret",
    "admin",
    "password",
    "default",
    "123456",
    "12345678",
}


@dataclass
class CoordinatorConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    heartbeat_timeout_s: float = 45.0
    default_model: str = "gemma3:4b"
    relay_buffer_size: int = 64 * 1024 * 1024  # 64 MB
    enable_cors: bool = True
    worker_auth_token: Optional[str] = field(
        default_factory=lambda: os.getenv("SITUS_WORKER_AUTH_TOKEN")
    )
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("SITUS_API_KEY", None))
    admin_key: Optional[str] = field(default_factory=lambda: os.getenv("SITUS_ADMIN_KEY", None))
    allowed_origins: List[str] = field(
        default_factory=lambda: [
            o.strip().rstrip("/")
            for o in os.getenv(
                "SITUS_ALLOWED_ORIGINS",
                "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000,http://127.0.0.1:3000",
            ).split(",")
            if o.strip()
        ]
    )
    request_timeout_s: float = 60.0
    max_concurrent_requests: int = 16
    ssl_certfile: Optional[str] = None
    ssl_keyfile: Optional[str] = None
    environment: str = field(
        default_factory=lambda: os.getenv("SITUS_ENV", os.getenv("ENV", "development")).strip().lower()
    )

    @property
    def is_production(self) -> bool:
        return self.environment in ("production", "prod")

    def validate(self) -> None:
        if self.enable_cors:
            for origin in self.allowed_origins:
                norm = origin.strip().lower()
                if norm == "*" or "*" in norm:
                    raise ValueError(
                        "Insecure CORS configuration: Wildcard origin '*' is not permitted when credentials are enabled."
                    )
                if norm == "null":
                    raise ValueError(
                        f"Insecure CORS configuration: Origin '{origin}' is not permitted when credentials are enabled."
                    )
        if self.is_production:
            if not self.worker_auth_token or not self.worker_auth_token.strip():
                raise ValueError("Production mode requires SITUS_WORKER_AUTH_TOKEN to be configured.")
            if self.worker_auth_token.strip().lower() in INSECURE_SECRETS:
                raise ValueError(
                    f"Insecure worker_auth_token '{self.worker_auth_token}' is not allowed in production mode."
                )

            if not self.admin_key or not self.admin_key.strip():
                raise ValueError("Production mode requires SITUS_ADMIN_KEY to be configured.")
            if self.admin_key.strip().lower() in INSECURE_SECRETS:
                raise ValueError(
                    f"Insecure admin_key '{self.admin_key}' is not allowed in production mode."
                )

            if not self.api_key or not self.api_key.strip():
                raise ValueError("Production mode requires SITUS_API_KEY to be configured.")
            if self.api_key.strip().lower() in INSECURE_SECRETS:
                raise ValueError(
                    f"Insecure api_key '{self.api_key}' is not allowed in production mode."
                )

            if self.enable_cors and not self.allowed_origins:
                raise ValueError(
                    "Production mode requires at least one explicit trusted origin in allowed_origins when CORS is enabled."
                )

    def __post_init__(self):
        if self.allowed_origins:
            self.allowed_origins = [o.strip().rstrip("/") for o in self.allowed_origins if o.strip()]
        self.validate()


@dataclass
class WorkerConfig:
    coordinator_url: str = "ws://localhost:8000/ws/worker"
    worker_id: Optional[str] = None
    device: str = "cuda:0"  # or 'cpu'
    model_name: str = "gemma3:4b"
    vram_budget_gb: Optional[float] = None
    assigned_layer_start: Optional[int] = None
    assigned_layer_end: Optional[int] = None
    compression_dtype: str = "int8"  # 'none', 'fp8', 'int8'
    heartbeat_interval_s: float = 3.0
    enable_speculative: bool = True
    p2p_direct_enabled: bool = True
    auth_token: Optional[str] = field(
        default_factory=lambda: os.getenv("SITUS_WORKER_AUTH_TOKEN")
    )
    environment: str = field(
        default_factory=lambda: os.getenv("SITUS_ENV", os.getenv("ENV", "development")).strip().lower()
    )

    @property
    def is_production(self) -> bool:
        return self.environment in ("production", "prod")

    def validate(self) -> None:
        if self.worker_id is not None and not is_valid_worker_id(self.worker_id):
            raise ValueError(
                f"Invalid worker_id '{self.worker_id}': ID must be 1-128 alphanumeric/hyphen/underscore/dot characters and cannot be reserved or contain path traversal."
            )
        if self.is_production:
            if not self.auth_token or not self.auth_token.strip():
                raise ValueError("Production mode requires SITUS_WORKER_AUTH_TOKEN (auth_token) to be configured.")
            if self.auth_token.strip().lower() in INSECURE_SECRETS:
                raise ValueError(
                    f"Insecure auth_token '{self.auth_token}' is not allowed in production mode."
                )

    def __post_init__(self):
        self.validate()


@dataclass
class LayerAllocation:
    worker_id: str
    layer_start: int
    layer_end: int
    is_first_stage: bool  # Holds embed_tokens
    is_last_stage: bool   # Holds norm + lm_head
