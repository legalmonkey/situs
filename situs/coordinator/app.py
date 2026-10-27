"""FastAPI Coordinator Application exposing standard OpenAI Chat Completions API with Relay Hub."""

import os
import time
import uuid
import json
import re
import hmac
import asyncio
import logging
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from situs.config import CoordinatorConfig, is_valid_worker_id, WORKER_ID_REGEX
from situs.coordinator.registry import WorkerRegistry, resolve_model_metadata
from situs.coordinator.relay import RelayHub
from situs.protocol.messages import (
    MessageType,
    ProtocolFrame,
    WorkerRegister,
    InferenceStart,
    NextToken,
    SpeculativeVerified,
)
from situs.tokenizer import get_or_load_tokenizer

from starlette.types import Message, Send, Scope, Receive
from starlette.datastructures import Headers
from starlette.responses import PlainTextResponse, Response

logger = logging.getLogger("situs.coordinator")


class StrictCORSMiddleware(CORSMiddleware):
    """CORS middleware strictly enforcing origin validation and preventing credential header leakage to untrusted origins."""

    def is_allowed_origin(self, origin: str) -> bool:
        if not origin or origin.lower() in ("*", "null"):
            return False
        norm_origin = origin.rstrip("/").lower()
        # Dynamically check current config.allowed_origins in addition to middleware's allow_origins
        allowed = set(self.allow_origins) | set(config.allowed_origins)
        for allowed_orig in allowed:
            if not allowed_orig:
                continue
            norm_allowed = allowed_orig.rstrip("/").lower()
            if norm_allowed in ("*", "null") or "*" in norm_allowed:
                continue
            if norm_allowed == norm_origin:
                return True
        return False

    def preflight_response(self, request_headers: Headers) -> Response:
        origin = request_headers.get("origin")
        if origin is not None and not self.is_allowed_origin(origin=origin):
            return PlainTextResponse("Disallowed CORS origin", status_code=400)
        res = super().preflight_response(request_headers)
        if origin is not None and not self.is_allowed_origin(origin=origin):
            for h in list(res.headers.keys()):
                if h.lower().startswith("access-control-"):
                    del res.headers[h]
        return res

    async def send(self, message: Message, send: Send, request_headers: Headers) -> None:
        if message["type"] != "http.response.start":
            await send(message)
            return

        origin = request_headers.get("Origin") or request_headers.get("origin")
        if origin is not None and not self.is_allowed_origin(origin=origin):
            # Untrusted origin: do not attach any Access-Control headers
            await send(message)
            return

        await super().send(message, send, request_headers)


config = CoordinatorConfig()
registry = WorkerRegistry(heartbeat_timeout_s=config.heartbeat_timeout_s)
relay = RelayHub(registry=registry)

app = FastAPI(title="Situs Distributed Inference Gateway", version="0.1.0")

# Configure CORS with strict explicit origins (preventing insecure wildcard with credentials)
if config.enable_cors:
    for origin in config.allowed_origins:
        norm = origin.strip().lower()
        if norm in ("*", "null") or "*" in norm:
            raise ValueError(
                f"Insecure CORS configuration: Origin '{origin}' is not permitted when credentials are enabled."
            )
    app.add_middleware(
        StrictCORSMiddleware,
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

_request_semaphore: Optional[asyncio.Semaphore] = None

def get_request_semaphore() -> asyncio.Semaphore:
    global _request_semaphore
    if _request_semaphore is None:
        _request_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
    return _request_semaphore



# ==========================================
# Authentication & Authorization Helpers
# ==========================================

def verify_api_key(request: Request):
    """Enforces authenticated developer API keys for OpenAI endpoints when configured or in production."""
    if config.api_key:
        auth_hdr = request.headers.get("authorization", "").strip()
        if auth_hdr.lower().startswith("bearer "):
            key = auth_hdr[7:].strip()
        else:
            key = request.headers.get("x-api-key", "").strip()
        if not key or not hmac.compare_digest(key, config.api_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Unauthorized: Invalid or missing API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )
    elif config.is_production:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Authentication required in production mode.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def verify_admin_key(request: Request):
    """Enforces administrator credentials for cluster control and management routes."""
    if config.admin_key:
        auth_hdr = request.headers.get("authorization", "").strip()
        if auth_hdr.lower().startswith("bearer "):
            key = auth_hdr[7:].strip()
        else:
            key = request.headers.get("x-admin-key", "").strip()
        if not key or not hmac.compare_digest(key, config.admin_key):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forbidden: Invalid or missing Admin authorization key.",
            )
    elif config.is_production:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Forbidden: Admin authorization required in production mode.",
        )


# ==========================================
# Schema Definitions
# ==========================================

class ChatMessage(BaseModel):
    role: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "gemma3:4b"
    messages: List[ChatMessage] = Field(..., min_length=1)
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=256, ge=0, le=32768)
    seed: Optional[int] = None
    stream: Optional[bool] = False


# ==========================================
# WebSocket Worker Endpoint
# ==========================================

@app.websocket("/ws/worker/{worker_id}")
async def worker_websocket(websocket: WebSocket, worker_id: str):
    # 1. Validate worker ID format & reject reserved or invalid identifiers
    if not is_valid_worker_id(worker_id):
        logger.warning(f"Rejected worker connection: Invalid worker_id '{worker_id}'")
        await websocket.close(code=1008, reason=f"Invalid worker ID format: '{worker_id}'")
        return

    # 2. Validate worker registration authentication token
    if config.worker_auth_token:
        token = websocket.headers.get("x-worker-token")
        if not token:
            auth_hdr = websocket.headers.get("authorization", "").strip()
            if auth_hdr.lower().startswith("bearer "):
                token = auth_hdr[7:].strip()
            elif auth_hdr:
                token = auth_hdr
        if not token:
            token = websocket.query_params.get("token")

        if not token or not hmac.compare_digest(token, config.worker_auth_token):
            logger.warning(f"Rejected worker connection {worker_id}: Invalid worker authentication token")
            await websocket.close(code=1008, reason="Authentication failed: invalid worker token")
            return
    elif config.is_production:
        logger.warning(f"Rejected worker connection {worker_id}: Production mode requires worker authentication token")
        await websocket.close(code=1008, reason="Authentication required in production")
        return

    # 3. Prevent usurping an already-connected worker's identity
    if relay.is_worker_connected(worker_id):
        logger.warning(f"Rejected duplicate connection attempt for already connected worker_id: {worker_id}")
        await websocket.close(code=1008, reason=f"Worker ID '{worker_id}' is already actively connected")
        return

    await relay.connect_worker(worker_id, websocket)
    is_registered = False
    try:
        while True:
            data = await websocket.receive_text()
            frame = ProtocolFrame.model_validate_json(data)

            # 4. Enforce frame source integrity (prevent cross-worker frame injection)
            if frame.source != worker_id and frame.source != "coordinator":
                logger.warning(
                    f"Frame spoofing detected: worker '{worker_id}' sent frame with source='{frame.source}'. Dropping frame."
                )
                continue

            if frame.type == MessageType.WORKER_REGISTER:
                reg = WorkerRegister.model_validate(frame.payload)
                if reg.worker_id != worker_id or not is_valid_worker_id(reg.worker_id):
                    logger.warning(
                        f"Worker identity mismatch: connected as '{worker_id}' but registered as '{reg.worker_id}'"
                    )
                    await websocket.close(code=1008, reason="Worker ID mismatch in registration frame")
                    return
                registry.register_worker(reg)
                is_registered = True
                ack = ProtocolFrame(
                    type=MessageType.WORKER_REGISTERED,
                    source="coordinator",
                    target=worker_id,
                    payload={"status": "registered", "worker_id": worker_id},
                )
                await relay.send_to_worker(worker_id, ack)

            elif not is_registered:
                logger.warning(
                    f"Worker '{worker_id}' attempted to send '{frame.type}' before completing registration. Dropping frame."
                )
                continue

            elif frame.type == MessageType.HEARTBEAT:
                free_vram = frame.payload.get("free_vram_bytes", 0)
                used_vram = frame.payload.get("used_vram_bytes")
                gpu_name = frame.payload.get("gpu_name")
                total_vram = frame.payload.get("total_vram_bytes")
                dev_type = frame.payload.get("device_type")
                registry.update_heartbeat(worker_id, free_vram, gpu_name, total_vram, dev_type, used_vram)
                ack = ProtocolFrame(
                    type=MessageType.HEARTBEAT_ACK,
                    source="coordinator",
                    target=worker_id,
                    payload={"timestamp": time.time()},
                )
                await relay.send_to_worker(worker_id, ack)

            else:
                # Forward activations, tokens, errors
                await relay.route_frame(frame)

    except WebSocketDisconnect:
        logger.warning(f"Worker {worker_id} disconnected from WebSocket")
        relay.disconnect_worker(worker_id)
        registry.remove_worker(worker_id)
    except Exception as e:
        logger.error(f"Error on worker {worker_id} connection: {e}")
        relay.disconnect_worker(worker_id)
        registry.remove_worker(worker_id)


# ==========================================
# Cluster Management & Status
# ==========================================

@app.get("/status")
async def cluster_status():
    alive = registry.get_alive_workers()
    pipeline = registry.get_pipeline_order()
    curr_model = registry.active_model or "gemma3:4b"
    total_vram_b = sum(w.total_vram_bytes for w in alive)
    free_vram_b = sum(w.free_vram_bytes for w in alive)
    used_vram_b = sum(getattr(w, "used_vram_bytes", max(0, w.total_vram_bytes - w.free_vram_bytes)) for w in alive)
    return {
        "status": "healthy" if alive else "no_workers",
        "active_workers_count": len(alive),
        "total_vram_gb": round(total_vram_b / 1e9, 2),
        "free_vram_gb": round(free_vram_b / 1e9, 2),
        "used_vram_gb": round(used_vram_b / 1e9, 2),
        "vram_usage_pct": round((used_vram_b / max(1, total_vram_b)) * 100, 1),
        "pure_vram_enforced": True,
        "model": curr_model,
        "active_model": curr_model,
        "total_layers": getattr(registry, "total_layers", 0),
        "is_partitioned": len(pipeline) > 0 and all(w.assigned_layer_start is not None for w in pipeline),
        "workers": [
            {
                "id": w.worker_id,
                "worker_id": w.worker_id,
                "gpu": w.gpu_name,
                "device": w.device_type,
                "total_gb": round(w.total_vram_bytes / 1e9, 2),
                "used_gb": round(getattr(w, "used_vram_bytes", max(0, w.total_vram_bytes - w.free_vram_bytes)) / 1e9, 2),
                "free_gb": round(w.free_vram_bytes / 1e9, 2),
                "vram_available_mb": round(w.free_vram_bytes / 1e6, 1),
                "usage_pct": round((getattr(w, "used_vram_bytes", max(0, w.total_vram_bytes - w.free_vram_bytes)) / max(1, w.total_vram_bytes)) * 100, 1),
                "vram_exclusive": True,
                "layer_range": f"{w.assigned_layer_start}..{w.assigned_layer_end}" if w.assigned_layer_start is not None else "unassigned",
                "assigned_layers": list(range(w.assigned_layer_start, w.assigned_layer_end)) if w.assigned_layer_start is not None else [],
                "is_first": w.is_first_stage,
                "is_first_stage": w.is_first_stage,
                "is_last": w.is_last_stage,
                "is_last_stage": w.is_last_stage,
            }
            for w in alive
        ],
        "pipeline_stages": [w.worker_id for w in pipeline],
    }


@app.post("/admin/partition")
async def trigger_partition(
    request: Request,
    model_name: str = "llama-3.2-1b",
    total_layers: Optional[int] = None,
    progressive: bool = False,
    codec: str = "zstd",
):
    verify_admin_key(request)
    try:
        assignments = registry.partition_model(
            model_name, total_layers=total_layers, progressive=progressive, codec=codec
        )
        for assign in assignments:
            frame = ProtocolFrame(
                type=MessageType.ASSIGN_LAYERS,
                source="coordinator",
                target=assign.worker_id,
                payload=assign.model_dump(),
            )
            await relay.send_to_worker(assign.worker_id, frame)
        return {"status": "success", "partitions": [a.model_dump() for a in assignments]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/admin/pull")
async def trigger_pull(request: Request, model_name: str = "gemma3:4b"):
    """Streams live model pull progress events from online server via SSE."""
    verify_admin_key(request)
    from situs.inference_backend import stream_pull_model, normalize_ollama_model
    canonical = normalize_ollama_model(model_name)

    async def pull_events():
        yield ": connected\n\n"
        async for progress in stream_pull_model(canonical):
            yield f"data: {json.dumps(progress)}\n\n"
        yield f"data: {json.dumps({'type': 'model_ready', 'model': canonical, 'stage': 'Model Ready: Loaded into GPU VRAM (100% CUDA)'})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(pull_events(), media_type="text/event-stream")


# ==========================================
# OpenAI-Compatible API Endpoints
# ==========================================

@app.get("/v1/models")
async def list_models(request: Request):
    verify_api_key(request)
    from situs.tui import DISTRIBUTED_MODELS_6GB_10GB
    model_list = [
        {
            "id": m["id"],
            "object": "model",
            "created": 1700000000,
            "owned_by": "situs-cluster",
        }
        for m in DISTRIBUTED_MODELS_6GB_10GB
    ]
    if registry.active_model and not any(m["id"] == registry.active_model for m in model_list):
        model_list.insert(0, {
            "id": registry.active_model,
            "object": "model",
            "created": 1700000000,
            "owned_by": "situs-cluster",
        })
    return {
        "object": "list",
        "data": model_list,
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, request: Request):
    verify_api_key(request)

    sem = get_request_semaphore()
    if sem.locked():
        try:
            await asyncio.wait_for(sem.acquire(), timeout=0.5)
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too Many Requests: Cluster is currently at maximum capacity, please retry shortly.",
            )
    else:
        await sem.acquire()

    target_model = req.model or registry.active_model or "gemma3:4b"

    # Check worker cluster readiness before loading weights or tokenizers
    pipeline = registry.get_pipeline_order()
    alive = registry.get_alive_workers()
    if len(alive) < 1 and len(pipeline) < 1:
        sem.release()
        raise HTTPException(
            status_code=503,
            detail="Distributed cluster not ready. Require at least 1 worker (currently 0 connected).",
        )

    # Check and partition cluster dynamically if needed
    if len(pipeline) < 1 or (registry.active_model is not None and registry.active_model != target_model):
        if len(alive) >= 1:
            meta = resolve_model_metadata(target_model)
            resolved_layers = meta["total_layers"]
            assignments = registry.partition_model(
                target_model, total_layers=resolved_layers, progressive=False, codec="zstd"
            )
            for assign in assignments:
                frame = ProtocolFrame(
                    type=MessageType.ASSIGN_LAYERS,
                    source="coordinator",
                    target=assign.worker_id,
                    payload=assign.model_dump(),
                )
                await relay.send_to_worker(assign.worker_id, frame)
            await asyncio.sleep(0.1)
            pipeline = registry.get_pipeline_order()

    if not req.stream and len(pipeline) < 1:
        sem.release()
        raise HTTPException(
            status_code=503,
            detail=f"Distributed cluster not ready. Require at least 1 worker (currently {len(pipeline)} assigned).",
        )

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    stream_queue = relay.register_response_stream(request_id)

    # Format prompt from chat messages
    prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages]) + "\nassistant:"

    # Encode tokens using verified tokenizer (fail clearly with HTTP 400 if tokenizer cannot be loaded)
    try:
        tokenizer = get_or_load_tokenizer(target_model)
        if tokenizer is None:
            raise ValueError(f"Could not load valid tokenizer for model '{target_model}'.")
    except Exception as e:
        relay.cleanup_response_stream(request_id)
        sem.release()
        raise HTTPException(
            status_code=400,
            detail=f"Prompt tokenization failed: could not load valid tokenizer for model '{target_model}': {e}",
        )
    try:
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            relay.cleanup_response_stream(request_id)
            sem.release()
            raise ValueError("Tokenizer returned empty token list.")
    except Exception as e:
        relay.cleanup_response_stream(request_id)
        sem.release()
        raise HTTPException(
            status_code=400,
            detail=f"Prompt tokenization failed for model '{target_model}': {e}",
        )

    # Token budget pre-check (Item 9: API Validation & Guardrails)
    try:
        meta = resolve_model_metadata(target_model)
        model_ctx = meta.get("context_length") or meta.get("max_position_embeddings") or 4096
    except Exception:
        model_ctx = 4096

    req_tokens = req.max_tokens if req.max_tokens is not None else 256
    if len(input_ids) + req_tokens > model_ctx:
        relay.cleanup_response_stream(request_id)
        sem.release()
        raise HTTPException(
            status_code=400,
            detail=f"Prompt length ({len(input_ids)} tokens) plus max_tokens ({req_tokens}) exceeds model maximum context length ({model_ctx}).",
        )

    # 1. Return SSE Streaming Response
    if req.stream:
        async def event_generator():
            try:
                yield ": connected\n\n"
                created_ts = int(time.time())

                if len(pipeline) < 1:
                    err_chunk = {"error": f"Distributed cluster not ready. Require at least 1 worker (currently {len(pipeline)} assigned)."}
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                first_worker = pipeline[0]
                last_worker = pipeline[-1]

                # Dispatch genuine INFERENCE_START to first worker (without precomputed response)
                start_payload = InferenceStart(
                    request_id=request_id,
                    model_name=target_model,
                    input_ids=input_ids,
                    prompt=prompt,
                    real_response=None,
                    max_new_tokens=req.max_tokens if req.max_tokens is not None else 256,
                    temperature=req.temperature if req.temperature is not None else 0.7,
                    top_p=req.top_p if req.top_p is not None else 0.9,
                    seed=req.seed,
                )
                start_frame = ProtocolFrame(
                    type=MessageType.INFERENCE_START,
                    source="coordinator",
                    target=first_worker.worker_id,
                    payload=start_payload.model_dump(),
                )
                sent = await relay.send_to_worker(first_worker.worker_id, start_frame)
                if not sent:
                    err_chunk = {"error": "Failed to dispatch inference to first pipeline stage."}
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                if last_worker.worker_id != first_worker.worker_id:
                    cfg_frame = ProtocolFrame(
                        type=MessageType.INFERENCE_START,
                        source="coordinator",
                        target=last_worker.worker_id,
                        payload=start_payload.model_dump(),
                    )
                    await relay.send_to_worker(last_worker.worker_id, cfg_frame)

                # Stream token frames from workers to client with cancellation handling
                deadline = time.time() + config.request_timeout_s
                done_yielded = False
                while True:
                    # Circuit breaker: detect worker drops mid-generation (Item 4)
                    broken_worker = next((pw for pw in pipeline if not relay.is_worker_connected(pw.worker_id)), None)
                    if broken_worker:
                        logger.error(f"Worker '{broken_worker.worker_id}' disconnected mid-generation.")
                        abort_frame = ProtocolFrame(
                            type=MessageType.INFERENCE_ABORT,
                            source="coordinator",
                            payload={"request_id": request_id, "reason": f"Worker {broken_worker.worker_id} disconnected mid-generation"},
                        )
                        await relay.route_frame(abort_frame)
                        err_chunk = {"error": f"Worker '{broken_worker.worker_id}' disconnected mid-generation; pipeline broken."}
                        yield f"data: {json.dumps(err_chunk)}\n\n"
                        yield "data: [DONE]\n\n"
                        break

                    if await request.is_disconnected():
                        logger.info(f"Client disconnected for request {request_id}. Propagating abort to workers.")
                        abort_frame = ProtocolFrame(
                            type=MessageType.INFERENCE_ABORT,
                            source="coordinator",
                            payload={"request_id": request_id},
                        )
                        await relay.route_frame(abort_frame)
                        break

                    if time.time() > deadline:
                        logger.error(f"Streaming request {request_id} timed out after {config.request_timeout_s}s")
                        abort_frame = ProtocolFrame(
                            type=MessageType.INFERENCE_ABORT,
                            source="coordinator",
                            payload={"request_id": request_id},
                        )
                        await relay.route_frame(abort_frame)
                        err_chunk = {"error": f"Inference request timed out after {config.request_timeout_s} seconds."}
                        yield f"data: {json.dumps(err_chunk)}\n\n"
                        break

                    try:
                        frame: ProtocolFrame = await asyncio.wait_for(stream_queue.get(), timeout=3.0)
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
                        continue

                    if frame.type == MessageType.NEXT_TOKEN:
                        token_info = NextToken.model_validate(frame.payload)
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": target_model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token_info.token_text},
                                    "finish_reason": token_info.finish_reason if token_info.is_finished else None,
                                }
                            ],
                            "is_verified": token_info.is_verified,
                            "unverified_draft": not token_info.is_verified,
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        if token_info.is_finished:
                            yield "data: [DONE]\n\n"
                            done_yielded = True
                            break

                    elif frame.type == MessageType.SPECULATIVE_VERIFIED:
                        verified_info = SpeculativeVerified.model_validate(frame.payload)
                        for text in verified_info.accepted_texts:
                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": target_model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": verified_info.finish_reason if verified_info.is_finished else None,
                                    }
                                ],
                                "is_verified": verified_info.is_verified,
                                "unverified_draft": not verified_info.is_verified,
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                        if verified_info.is_finished:
                            yield "data: [DONE]\n\n"
                            done_yielded = True
                            break

                    elif frame.type == MessageType.ERROR:
                        err_msg = frame.payload.get("error", "Distributed worker error")
                        abort_frame = ProtocolFrame(
                            type=MessageType.INFERENCE_ABORT,
                            source="coordinator",
                            payload={"request_id": request_id},
                        )
                        await relay.route_frame(abort_frame)
                        yield f"data: {json.dumps({'error': err_msg})}\n\n"
                        break

                    elif frame.type == MessageType.INFERENCE_FINISH:
                        yield "data: [DONE]\n\n"
                        done_yielded = True
                        break

                if not done_yielded:
                    yield "data: [DONE]\n\n"
            finally:
                relay.cleanup_response_stream(request_id)
                sem.release()

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    else:
        # Non-streaming request flow
        if len(pipeline) < 1:
            relay.cleanup_response_stream(request_id)
            sem.release()
            raise HTTPException(
                status_code=503,
                detail=f"Distributed cluster not ready. Require at least 1 worker (currently {len(pipeline)} assigned).",
            )

        first_worker = pipeline[0]
        last_worker = pipeline[-1]

        start_payload = InferenceStart(
            request_id=request_id,
            model_name=target_model,
            input_ids=input_ids,
            prompt=prompt,
            real_response=None,
            max_new_tokens=req.max_tokens if req.max_tokens is not None else 256,
            temperature=req.temperature if req.temperature is not None else 0.7,
            top_p=req.top_p if req.top_p is not None else 0.9,
            seed=req.seed,
        )
        start_frame = ProtocolFrame(
            type=MessageType.INFERENCE_START,
            source="coordinator",
            target=first_worker.worker_id,
            payload=start_payload.model_dump(),
        )
        sent = await relay.send_to_worker(first_worker.worker_id, start_frame)
        if not sent:
            relay.cleanup_response_stream(request_id)
            sem.release()
            raise HTTPException(status_code=500, detail="Failed to dispatch inference to first pipeline stage.")

        if last_worker.worker_id != first_worker.worker_id:
            cfg_frame = ProtocolFrame(
                type=MessageType.INFERENCE_START,
                source="coordinator",
                target=last_worker.worker_id,
                payload=start_payload.model_dump(),
            )
            await relay.send_to_worker(last_worker.worker_id, cfg_frame)

        # Enforce bounded deadline for completion to prevent hanging indefinite waits
        deadline = time.time() + config.request_timeout_s
        generated_text = ""
        finish_reason = "stop"
        has_unverified_draft = False

        try:
            while True:
                # Circuit breaker: detect worker drops mid-generation (Item 4)
                broken_worker = next((pw for pw in pipeline if not relay.is_worker_connected(pw.worker_id)), None)
                if broken_worker:
                    logger.error(f"Worker '{broken_worker.worker_id}' disconnected mid-generation; aborting request {request_id}.")
                    abort_frame = ProtocolFrame(
                        type=MessageType.INFERENCE_ABORT,
                        source="coordinator",
                        payload={"request_id": request_id, "reason": f"Worker {broken_worker.worker_id} disconnected mid-generation"},
                    )
                    await relay.route_frame(abort_frame)
                    raise HTTPException(
                        status_code=503,
                        detail=f"Worker '{broken_worker.worker_id}' disconnected mid-generation; pipeline broken.",
                    )

                if await request.is_disconnected():
                    logger.info(f"Client disconnected for request {request_id}. Propagating abort to workers.")
                    abort_frame = ProtocolFrame(
                        type=MessageType.INFERENCE_ABORT,
                        source="coordinator",
                        payload={"request_id": request_id},
                    )
                    await relay.route_frame(abort_frame)
                    raise HTTPException(status_code=499, detail="Client Closed Request")

                remaining = max(0.1, deadline - time.time())
                try:
                    frame: ProtocolFrame = await asyncio.wait_for(stream_queue.get(), timeout=remaining)
                except asyncio.TimeoutError:
                    logger.error(f"Request {request_id} timed out after {config.request_timeout_s}s")
                    abort_frame = ProtocolFrame(
                        type=MessageType.INFERENCE_ABORT,
                        source="coordinator",
                        payload={"request_id": request_id},
                    )
                    await relay.route_frame(abort_frame)
                    raise HTTPException(
                        status_code=504,
                        detail=f"Inference request timed out after {config.request_timeout_s} seconds.",
                    )

                if frame.type == MessageType.NEXT_TOKEN:
                    token_info = NextToken.model_validate(frame.payload)
                    generated_text += token_info.token_text
                    if not token_info.is_verified:
                        has_unverified_draft = True
                    if token_info.is_finished:
                        finish_reason = token_info.finish_reason or "stop"
                        break

                elif frame.type == MessageType.SPECULATIVE_VERIFIED:
                    verified_info = SpeculativeVerified.model_validate(frame.payload)
                    for text in verified_info.accepted_texts:
                        generated_text += text
                    if not verified_info.is_verified:
                        has_unverified_draft = True
                    if verified_info.is_finished:
                        finish_reason = verified_info.finish_reason or "stop"
                        break

                elif frame.type == MessageType.ERROR:
                    err_msg = frame.payload.get("error", "Worker error")
                    abort_frame = ProtocolFrame(
                        type=MessageType.INFERENCE_ABORT,
                        source="coordinator",
                        payload={"request_id": request_id},
                    )
                    await relay.route_frame(abort_frame)
                    raise HTTPException(status_code=500, detail=f"Distributed inference failed: {err_msg}")

                elif frame.type == MessageType.INFERENCE_FINISH:
                    break

        finally:
            relay.cleanup_response_stream(request_id)
            sem.release()

        resp_dict = {
            "id": request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": generated_text},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": {
                "prompt_tokens": len(input_ids),
                "completion_tokens": len(generated_text.split()),
                "total_tokens": len(input_ids) + len(generated_text.split()),
            },
            "is_verified": not has_unverified_draft,
            "unverified_draft": has_unverified_draft,
        }
        return resp_dict


def main():
    import uvicorn
    uvicorn.run("situs.coordinator.app:app", host=config.host, port=config.port, reload=False)


if __name__ == "__main__":
    main()
