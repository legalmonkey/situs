"""FastAPI Coordinator Application exposing standard OpenAI Chat Completions API with Relay Hub."""

import time
import uuid
import json
import asyncio
import logging
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from situs.config import CoordinatorConfig
from situs.coordinator.registry import WorkerRegistry
from situs.coordinator.relay import RelayHub
from situs.protocol.messages import (
    MessageType,
    ProtocolFrame,
    WorkerRegister,
    InferenceStart,
    NextToken,
    SpeculativeVerified,
)

logger = logging.getLogger("situs.coordinator")

app = FastAPI(title="Situs Distributed Inference Gateway", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

config = CoordinatorConfig()
registry = WorkerRegistry(heartbeat_timeout_s=config.heartbeat_timeout_s)
relay = RelayHub()


# ==========================================
# Schema Definitions
# ==========================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "nemotron-3-nano:4b-q8_0"
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 256
    stream: Optional[bool] = False



# ==========================================
# WebSocket Worker Endpoint
# ==========================================

@app.websocket("/ws/worker/{worker_id}")
async def worker_websocket(websocket: WebSocket, worker_id: str):
    await relay.connect_worker(worker_id, websocket)
    try:
        while True:
            data = await websocket.receive_text()
            frame = ProtocolFrame.model_validate_json(data)

            if frame.type == MessageType.WORKER_REGISTER:
                reg = WorkerRegister.model_validate(frame.payload)
                registry.register_worker(reg)
                ack = ProtocolFrame(
                    type=MessageType.WORKER_REGISTERED,
                    source="coordinator",
                    target=worker_id,
                    payload={"status": "registered", "worker_id": worker_id},
                )
                await relay.send_to_worker(worker_id, ack)

            elif frame.type == MessageType.HEARTBEAT:
                free_vram = frame.payload.get("free_vram_bytes", 0)
                registry.update_heartbeat(worker_id, free_vram)
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
    return {
        "status": "healthy" if alive else "no_workers",
        "active_workers_count": len(alive),
        "total_vram_gb": sum(w.total_vram_bytes for w in alive) / 1e9,
        "free_vram_gb": sum(w.free_vram_bytes for w in alive) / 1e9,
        "workers": [
            {
                "id": w.worker_id,
                "gpu": w.gpu_name,
                "device": w.device_type,
                "free_gb": round(w.free_vram_bytes / 1e9, 2),
                "layer_range": f"{w.assigned_layer_start}..{w.assigned_layer_end}" if w.assigned_layer_start is not None else "unassigned",
                "is_first": w.is_first_stage,
                "is_last": w.is_last_stage,
                "fast_boot": getattr(w, "fast_boot", False),
            }
            for w in alive
        ],
        "pipeline_stages": [w.worker_id for w in pipeline],
    }


@app.post("/admin/partition")
async def trigger_partition(
    model_name: str = "nemotron-3-nano:4b-q8_0",
    total_layers: int = 32,
    progressive: bool = False,
    fast_boot: bool = False,
    codec: str = "zstd",
):
    try:
        assignments = registry.partition_model(
            model_name, total_layers, progressive=progressive, fast_boot=fast_boot, codec=codec
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


# ==========================================
# OpenAI-Compatible API Endpoints
# ==========================================

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": "nemotron-3-nano:4b-q8_0",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "situs-cluster",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    pipeline = registry.get_pipeline_order()
    if len(pipeline) < 2:
        alive = registry.get_alive_workers()
        if len(alive) >= 2:
            # Auto-partition layers across available workers with fast-boot & wire compression
            model_target = req.model or "nemotron-3-nano:4b-q8_0"
            assignments = registry.partition_model(
                model_target, total_layers=32, fast_boot=True, progressive=True, codec="zstd"
            )
            for assign in assignments:
                frame = ProtocolFrame(
                    type=MessageType.ASSIGN_LAYERS,
                    source="coordinator",
                    target=assign.worker_id,
                    payload=assign.model_dump(),
                )
                await relay.send_to_worker(assign.worker_id, frame)
            # Give workers a brief moment to instantiate stage layers
            await asyncio.sleep(0.5)
            pipeline = registry.get_pipeline_order()

    if len(pipeline) < 2:
        raise HTTPException(
            status_code=503,
            detail=f"Distributed cluster not ready. Require at least 2 workers (currently {len(pipeline)} assigned).",
        )

    first_worker = pipeline[0]
    last_worker = pipeline[-1]
    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    stream_queue = relay.register_response_stream(request_id)

    # Format prompt from chat messages
    prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages]) + "\nassistant:"

    # Encode tokens - fast tokenizer with immediate byte fallback for local/mock/offline tags
    input_ids = None
    if req.model and "/" in req.model:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(req.model, local_files_only=True)
            input_ids = tokenizer.encode(prompt)
        except Exception:
            pass

    if input_ids is None:
        # Fast byte-level encoding for local tests and offline mode
        input_ids = [int(b) for b in prompt.encode("utf-8")]

    # 1. Send INFERENCE_START to Worker A (first stage)
    start_payload = InferenceStart(
        request_id=request_id,
        model_name=req.model or "nemotron-3-nano:4b-q8_0",
        input_ids=input_ids,
        max_new_tokens=req.max_tokens or 256,
        temperature=req.temperature or 0.7,
        top_p=req.top_p or 0.9,
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
        raise HTTPException(status_code=500, detail="Failed to dispatch inference to first pipeline stage.")

    # 2. Return SSE Streaming Response or Full JSON
    if req.stream:
        async def event_generator():
            try:
                created_ts = int(time.time())
                while True:
                    frame: ProtocolFrame = await stream_queue.get()
                    if frame.type == MessageType.NEXT_TOKEN:
                        token_info = NextToken.model_validate(frame.payload)
                        chunk = {
                            "id": request_id,
                            "object": "chat.completion.chunk",
                            "created": created_ts,
                            "model": req.model,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": token_info.token_text},
                                    "finish_reason": token_info.finish_reason if token_info.is_finished else None,
                                }
                            ],
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        if token_info.is_finished:
                            break
                    elif frame.type == MessageType.SPECULATIVE_VERIFIED:
                        verified_info = SpeculativeVerified.model_validate(frame.payload)
                        for text in verified_info.accepted_texts:
                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": req.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": text},
                                        "finish_reason": verified_info.finish_reason if verified_info.is_finished else None,
                                    }
                                ],
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                        if verified_info.is_finished:
                            break
                    elif frame.type in (MessageType.INFERENCE_FINISH, MessageType.ERROR):
                        break

                yield "data: [DONE]\n\n"
            finally:
                relay.cleanup_response_stream(request_id)

        return StreamingResponse(event_generator(), media_type="text/event-stream")
    else:
        # Accumulate all tokens
        generated_text = ""
        finish_reason = "stop"
        try:
            while True:
                frame: ProtocolFrame = await stream_queue.get()
                if frame.type == MessageType.NEXT_TOKEN:
                    token_info = NextToken.model_validate(frame.payload)
                    generated_text += token_info.token_text
                    if token_info.is_finished:
                        finish_reason = token_info.finish_reason or "stop"
                        break
                elif frame.type == MessageType.SPECULATIVE_VERIFIED:
                    verified_info = SpeculativeVerified.model_validate(frame.payload)
                    for text in verified_info.accepted_texts:
                        generated_text += text
                    if verified_info.is_finished:
                        finish_reason = verified_info.finish_reason or "stop"
                        break
                elif frame.type in (MessageType.INFERENCE_FINISH, MessageType.ERROR):
                    break
        finally:
            relay.cleanup_response_stream(request_id)

        return {
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
        }




def main():
    import uvicorn
    uvicorn.run("situs.coordinator.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
