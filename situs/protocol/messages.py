"""Typed protocol messages and framing for Situs distributed communication."""

from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class MessageType(str, Enum):
    # Registration & Lifecycle
    WORKER_REGISTER = "worker_register"
    WORKER_REGISTERED = "worker_registered"
    ASSIGN_LAYERS = "assign_layers"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    
    # WebRTC P2P Signaling
    ICE_OFFER = "ice_offer"
    ICE_ANSWER = "ice_answer"
    ICE_CANDIDATE = "ice_candidate"
    P2P_READY = "p2p_ready"

    # Inference Control (Standard & Speculative)
    INFERENCE_START = "inference_start"
    FORWARD_ACTIVATION = "forward_activation"
    NEXT_TOKEN = "next_token"
    SPECULATIVE_CANDIDATES = "speculative_candidates"
    SPECULATIVE_VERIFIED = "speculative_verified"
    INFERENCE_FINISH = "inference_finish"
    INFERENCE_ABORT = "inference_abort"
    
    # Progressive JIT Layer Streaming
    STREAM_LAYER_CHUNK = "stream_layer_chunk"
    STAGE_LAYER_ACK = "stage_layer_ack"

    # Error
    ERROR = "error"


class WorkerRegister(BaseModel):
    worker_id: str
    gpu_name: str
    total_vram_bytes: int
    free_vram_bytes: int
    used_vram_bytes: Optional[int] = None
    device_type: str = "cuda"  # 'cuda' or 'cpu'
    supported_models: List[str] = Field(default_factory=list)


class AssignLayers(BaseModel):
    worker_id: str
    model_name: str
    layer_start: int
    layer_end: int
    total_layers: int
    is_first_stage: bool
    is_last_stage: bool
    next_worker_id: Optional[str] = None
    prev_worker_id: Optional[str] = None
    first_worker_id: Optional[str] = None
    progressive: bool = False
    fast_boot: bool = False
    codec: str = "zstd"


class Heartbeat(BaseModel):
    worker_id: str
    timestamp: float
    free_vram_bytes: int
    used_vram_bytes: Optional[int] = None
    gpu_name: Optional[str] = None
    total_vram_bytes: Optional[int] = None
    device_type: Optional[str] = None
    active_sessions: int = 0


class InferenceStart(BaseModel):
    request_id: str
    model_name: str
    input_ids: List[int]
    prompt: Optional[str] = None
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    stop_token_ids: List[int] = Field(default_factory=list)


class ForwardActivation(BaseModel):
    request_id: str
    step_index: int
    shape: List[int]
    dtype: str = "int8"
    scale: float = 1.0
    payload_b64: Optional[str] = None  # Base64 string for JSON or raw bytes in binary mode


class NextToken(BaseModel):
    request_id: str
    step_index: int
    token_id: int
    token_text: str = ""
    is_finished: bool = False
    finish_reason: Optional[str] = None


class SpeculativeCandidates(BaseModel):
    request_id: str
    step_index: int
    candidate_tokens: List[int]
    shape: List[int]
    dtype: str = "int8"
    scale: float = 1.0
    payload_b64: Optional[str] = None


class SpeculativeVerified(BaseModel):
    request_id: str
    step_index: int
    accepted_tokens: List[int]
    accepted_texts: List[str]
    is_finished: bool = False
    finish_reason: Optional[str] = None


class WebRTCSignal(BaseModel):
    source_worker: str
    target_worker: str
    sdp: Optional[str] = None
    type: Optional[str] = None  # 'offer' or 'answer'
    candidate: Optional[Dict[str, Any]] = None


class StreamLayerChunk(BaseModel):
    model_name: str
    layer_index: int
    total_stage_layers: int
    chunk_index: int
    total_chunks: int
    payload_b64: Optional[str] = None
    raw_bytes: Optional[bytes] = None
    scale: float = 1.0
    shape: List[int] = Field(default_factory=list)
    dtype: str = "int8"
    codec: str = "zstd"
    is_last_chunk: bool = False

    def get_payload_bytes(self) -> bytes:
        """Extract raw bytes from either direct binary or Base64 payload."""
        if self.raw_bytes is not None:
            return self.raw_bytes
        if self.payload_b64:
            import base64
            return base64.b64decode(self.payload_b64)
        return b""


class StageLayerAck(BaseModel):
    worker_id: str
    model_name: str
    layer_index: int
    status: str = "ready"  # 'ready', 'loading', 'error'
    ready_layer_count: int = 1
    total_assigned: int = 16
    is_fast_boot_active: bool = False


class ProtocolFrame(BaseModel):
    type: MessageType
    source: str
    target: Optional[str] = None  # If None, directed to Coordinator
    payload: Dict[str, Any] = Field(default_factory=dict)
