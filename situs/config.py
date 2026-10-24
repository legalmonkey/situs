"""Configuration settings and dataclasses for Situs Coordinator and Workers."""

import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class CoordinatorConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    heartbeat_timeout_s: float = 45.0
    default_model: str = "gemma3:4b"
    relay_buffer_size: int = 64 * 1024 * 1024  # 64 MB
    enable_cors: bool = True
    worker_auth_token: Optional[str] = field(
        default_factory=lambda: os.getenv("SITUS_WORKER_AUTH_TOKEN", "situs-cluster-secret")
    )
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("SITUS_API_KEY", None))
    admin_key: Optional[str] = field(default_factory=lambda: os.getenv("SITUS_ADMIN_KEY", None))
    allowed_origins: List[str] = field(
        default_factory=lambda: [
            o.strip()
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
        default_factory=lambda: os.getenv("SITUS_WORKER_AUTH_TOKEN", "situs-cluster-secret")
    )


@dataclass
class LayerAllocation:
    worker_id: str
    layer_start: int
    layer_end: int
    is_first_stage: bool  # Holds embed_tokens
    is_last_stage: bool   # Holds norm + lm_head
