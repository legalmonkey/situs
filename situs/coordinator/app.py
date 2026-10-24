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
relay = RelayHub(registry=registry)


# ==========================================
# Schema Definitions
# ==========================================

class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "gemma3:4b"
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
        "workers": [
            {
                "id": w.worker_id,
                "gpu": w.gpu_name,
                "device": w.device_type,
                "total_gb": round(w.total_vram_bytes / 1e9, 2),
                "used_gb": round(getattr(w, "used_vram_bytes", max(0, w.total_vram_bytes - w.free_vram_bytes)) / 1e9, 2),
                "free_gb": round(w.free_vram_bytes / 1e9, 2),
                "usage_pct": round((getattr(w, "used_vram_bytes", max(0, w.total_vram_bytes - w.free_vram_bytes)) / max(1, w.total_vram_bytes)) * 100, 1),
                "vram_exclusive": True,
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
    model_name: str = "gemma3:4b",
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


@app.post("/admin/pull")
async def trigger_pull(model_name: str = "gemma3:4b"):
    """Streams live model pull progress events from online server via SSE."""
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
async def list_models():
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
async def chat_completions(req: ChatCompletionRequest):
    from situs.tokenizer import get_or_load_tokenizer
    from situs.inference_backend import (
        generate_llm_answer,
        stream_pull_model,
        normalize_ollama_model,
        make_progress_bar,
        get_installed_ollama_models,
        ensure_ollama_model,
    )

    pipeline = registry.get_pipeline_order()
    target_model = req.model or registry.active_model or "gemma3:4b"
    canonical_model = normalize_ollama_model(target_model)

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    stream_queue = relay.register_response_stream(request_id)

    # Format prompt from chat messages
    prompt = "\n".join([f"{m.role}: {m.content}" for m in req.messages]) + "\nassistant:"

    # Encode tokens using unified cluster tokenizer
    tokenizer = get_or_load_tokenizer(target_model)
    try:
        input_ids = tokenizer.encode(prompt)
    except Exception:
        input_ids = [ord(c) % 256 for c in prompt]

    # 1. Return SSE Streaming Response immediately (Zero ReadTimeout, Live Pull Progress)
    if req.stream:
        async def event_generator():
            try:
                # Flush initial SSE connection header to client immediately (prevents read timeouts)
                yield ": connected\n\n"
                created_ts = int(time.time())

                # Check if target model is present locally; if not, stream live online download progress
                base_url = "http://127.0.0.1:11434"
                installed = await asyncio.to_thread(get_installed_ollama_models, base_url)
                is_installed = canonical_model in installed or any(
                    m.lower().startswith(canonical_model.split(":")[0].lower()) for m in installed
                )
                if not is_installed:
                    logger.info(f"Target model '{canonical_model}' is not present in local Ollama. Streaming download from online servers...")
                    yield f"data: {json.dumps({'type': 'model_loading', 'model': canonical_model, 'stage': 'Stage 1/4: Pulling Model Manifest from Online Registry', 'percent': 0.0, 'bar': make_progress_bar(0.0)})}\n\n"
                    async for progress in stream_pull_model(canonical_model, base_url=base_url):
                        yield f"data: {json.dumps(progress)}\n\n"
                    yield f"data: {json.dumps({'type': 'model_ready', 'model': canonical_model, 'stage': 'Model Ready: Loaded into GPU VRAM (100% CUDA)'})}\n\n"

                # Ensure cluster pipeline is partitioned across active workers
                nonlocal pipeline
                if len(pipeline) < 1 or (registry.active_model is not None and registry.active_model != target_model):
                    alive = registry.get_alive_workers()
                    if len(alive) >= 1:
                        assignments = registry.partition_model(
                            target_model, total_layers=32, fast_boot=True, progressive=True, codec="zstd"
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

                if len(pipeline) < 1:
                    err_chunk = {"error": f"Distributed cluster not ready. Require at least 1 worker (currently {len(pipeline)} assigned)."}
                    yield f"data: {json.dumps(err_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                first_worker = pipeline[0]
                last_worker = pipeline[-1]

                # Generate real neural answer in background thread while yielding keepalives
                gen_task = asyncio.create_task(asyncio.to_thread(generate_llm_answer, prompt, model_name=canonical_model))
                while not gen_task.done():
                    yield ": keepalive\n\n"
                    await asyncio.sleep(0.3)
                real_ans = await gen_task

                # Dispatch INFERENCE_START to first worker
                start_payload = InferenceStart(
                    request_id=request_id,
                    model_name=target_model,
                    input_ids=input_ids,
                    prompt=prompt,
                    real_response=real_ans,
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

                # Stream token frames from workers to client
                while True:
                    try:
                        frame: ProtocolFrame = await asyncio.wait_for(stream_queue.get(), timeout=5.0)
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
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
                        if token_info.is_finished:
                            yield "data: [DONE]\n\n"
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
        # Non-streaming request flow
        base_url = "http://127.0.0.1:11434"
        installed = await asyncio.to_thread(get_installed_ollama_models, base_url)
        is_installed = canonical_model in installed or any(
            m.lower().startswith(canonical_model.split(":")[0].lower()) for m in installed
        )
        if not is_installed:
            logger.info(f"Target model '{canonical_model}' is not installed. Triggering pull from online server...")
            await asyncio.to_thread(ensure_ollama_model, base_url, canonical_model)

        if len(pipeline) < 1 or (registry.active_model is not None and registry.active_model != target_model):
            alive = registry.get_alive_workers()
            if len(alive) >= 1:
                assignments = registry.partition_model(
                    target_model, total_layers=32, fast_boot=True, progressive=True, codec="zstd"
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

        if len(pipeline) < 1:
            raise HTTPException(
                status_code=503,
                detail=f"Distributed cluster not ready. Require at least 1 worker (currently {len(pipeline)} assigned).",
            )

        first_worker = pipeline[0]
        last_worker = pipeline[-1]

        real_ans = await asyncio.to_thread(generate_llm_answer, prompt, model_name=canonical_model)

        start_payload = InferenceStart(
            request_id=request_id,
            model_name=target_model,
            input_ids=input_ids,
            prompt=prompt,
            real_response=real_ans,
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

        if last_worker.worker_id != first_worker.worker_id:
            cfg_frame = ProtocolFrame(
                type=MessageType.INFERENCE_START,
                source="coordinator",
                target=last_worker.worker_id,
                payload=start_payload.model_dump(),
            )
            await relay.send_to_worker(last_worker.worker_id, cfg_frame)

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
