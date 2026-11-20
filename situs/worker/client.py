"""Worker client daemon connecting outbound to Coordinator Relay."""

import os
import sys
import time
import uuid
import json
import shutil
import subprocess
import asyncio
import threading
import logging
import argparse
from typing import Optional, Dict, List, Any
import websockets

from situs.config import WorkerConfig
from situs.protocol.messages import (
    MessageType,
    ProtocolFrame,
    WorkerRegister,
    AssignLayers,
    InferenceStart,
    ForwardActivation,
    NextToken,
    SpeculativeCandidates,
    SpeculativeVerified,
    StreamLayerChunk,
    StageLayerAck,
    RequestLayerStream,
    Heartbeat,
)
from situs.worker.loader import load_hf_stage_model
from situs.worker.engine import StageExecutionEngine
from situs.network.p2p import P2PDataChannelNode

logger = logging.getLogger("situs.worker")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class WorkerDaemon:
    def __init__(self, config: WorkerConfig):
        self.config = config
        self.worker_id = config.worker_id or f"worker-{uuid.uuid4().hex[:8]}"
        self.engine: Optional[StageExecutionEngine] = None
        self.assignment: Optional[AssignLayers] = None
        self.engine_ready_event = asyncio.Event()
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.p2p_nodes: Dict[str, P2PDataChannelNode] = {}
        self.is_running = True
        self.is_connected = False
        self.is_registered = False
        self.registered_event = threading.Event()
        self.connection_error: Optional[str] = None
        self.last_error: Optional[str] = None
        self.engine_error: Optional[str] = None
        self._cached_hw = None
        self.active_inference_tasks: Dict[str, asyncio.Task] = {}
        self.native_server: Optional[Any] = None

    @property
    def running(self) -> bool:
        return self.is_running

    @running.setter
    def running(self, val: bool):
        self.is_running = val

    @property
    def p2p_node(self) -> Optional[P2PDataChannelNode]:
        if not self.p2p_nodes:
            return None
        return next(iter(self.p2p_nodes.values()))

    @p2p_node.setter
    def p2p_node(self, val: Optional[P2PDataChannelNode]):
        if val is None:
            self.p2p_nodes.clear()
        else:
            self.p2p_nodes[val.target_worker_id] = val

    def stop(self):
        """Signals the daemon to shut down and disconnects WebSocket and P2P direct channel."""
        self.is_running = False
        if self.native_server is not None:
            try:
                self.native_server.stop()
            except Exception:
                pass
            self.native_server = None
        for task in list(self.active_inference_tasks.values()):
            if not task.done():
                task.cancel()
        self.active_inference_tasks.clear()
        loop = getattr(self, "_loop", None)
        if self.p2p_nodes and loop and loop.is_running():
            for node in list(self.p2p_nodes.values()):
                try:
                    asyncio.run_coroutine_threadsafe(node.close(), loop)
                except Exception:
                    pass
            self.p2p_nodes.clear()
        if self.ws and loop and loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(self.ws.close(), loop)
            except Exception:
                pass

    async def send_signaling_frame(self, msg_dict: dict):
        if self.ws:
            frame = ProtocolFrame(
                type=MessageType(msg_dict.get("type", "ice_offer")),
                source=self.worker_id,
                target=msg_dict.get("target_worker"),
                payload=msg_dict,
            )
            await self.ws.send(frame.model_dump_json())

    def on_p2p_message(self, message: str):
        # Schedule incoming direct P2P frame handling
        asyncio.create_task(self.handle_incoming_frame(message))

    async def handle_incoming_frame(self, message: str):
        """Processes an incoming ProtocolFrame from either WebSocket or P2P DataChannel."""
        try:
            frame = ProtocolFrame.model_validate_json(message)
            if frame.type in (
                MessageType.INFERENCE_START,
                MessageType.FORWARD_ACTIVATION,
                MessageType.NEXT_TOKEN,
                MessageType.SPECULATIVE_CANDIDATES,
                MessageType.SPECULATIVE_VERIFIED,
            ):
                try:
                    eng_timeout = getattr(self.config, "engine_ready_timeout_s", 120.0)
                    await asyncio.wait_for(self.engine_ready_event.wait(), timeout=eng_timeout)
                except asyncio.TimeoutError:
                    pass

            if frame.type == MessageType.FORWARD_ACTIVATION:
                if not self.engine:
                    act_req = frame.payload.get("request_id") if isinstance(frame.payload, dict) else None
                    await self._emit_inference_error(
                        act_req,
                        self.engine_error or f"Worker '{self.worker_id}' has no active pipeline engine loaded.",
                    )
                    return
                act_frame = ForwardActivation.model_validate(frame.payload)
                out_frame = await asyncio.to_thread(self.engine.process_forward_activation, act_frame)
                if out_frame:
                    await self.send_frame_out(out_frame)
            elif frame.type == MessageType.SPECULATIVE_CANDIDATES:
                if not self.engine:
                    cand_req = frame.payload.get("request_id") if isinstance(frame.payload, dict) else None
                    await self._emit_inference_error(
                        cand_req,
                        self.engine_error or f"Worker '{self.worker_id}' has no active pipeline engine loaded.",
                    )
                    return
                cand_frame = SpeculativeCandidates.model_validate(frame.payload)
                out_frame = await asyncio.to_thread(self.engine.process_speculative_candidates, cand_frame)
                if out_frame:
                    await self.send_frame_out(out_frame)
            elif frame.type == MessageType.SPECULATIVE_VERIFIED:
                if not self.engine:
                    ver_req = frame.payload.get("request_id") if isinstance(frame.payload, dict) else None
                    await self._emit_inference_error(
                        ver_req,
                        self.engine_error or f"Worker '{self.worker_id}' has no active pipeline engine loaded.",
                    )
                    return
                ver_frame = SpeculativeVerified.model_validate(frame.payload)
                out_frame = await asyncio.to_thread(self.engine.process_speculative_verified, ver_frame)
                if out_frame:
                    await self.send_frame_out(out_frame)
            elif frame.type == MessageType.NEXT_TOKEN:
                if not self.engine:
                    tok_req = frame.payload.get("request_id") if isinstance(frame.payload, dict) else None
                    await self._emit_inference_error(
                        tok_req,
                        self.engine_error or f"Worker '{self.worker_id}' has no active pipeline engine loaded.",
                    )
                    return
                tok_frame = NextToken.model_validate(frame.payload)
                out_frame = await asyncio.to_thread(self.engine.process_next_token, tok_frame)
                if out_frame:
                    await self.send_frame_out(out_frame)
            elif frame.type == MessageType.INFERENCE_START:
                if not self.engine:
                    start_req = frame.payload.get("request_id") if isinstance(frame.payload, dict) else None
                    await self._emit_inference_error(
                        start_req,
                        self.engine_error or f"Worker '{self.worker_id}' has no active pipeline engine loaded.",
                    )
                    return
                start_msg = InferenceStart.model_validate(frame.payload)
                if self.engine.is_first_stage() and self.engine.is_last_stage():
                    t = asyncio.create_task(self._run_single_worker_loop(start_msg))
                    self.active_inference_tasks[start_msg.request_id] = t
                elif self.engine.is_first_stage():
                    out_frame = await asyncio.to_thread(self.engine.start_inference, start_msg)
                    if out_frame:
                        await self.send_frame_out(out_frame)
                else:
                    self.engine.session_configs[start_msg.request_id] = start_msg
                    self.engine.session_token_histories[start_msg.request_id] = list(start_msg.input_ids)
                    self.engine.session_step_counts[start_msg.request_id] = len(start_msg.input_ids)
            elif frame.type == MessageType.STREAM_LAYER_CHUNK and self.engine:
                chunk = StreamLayerChunk.model_validate(frame.payload)
                out_frame = await asyncio.to_thread(self.engine.process_layer_chunk, chunk, frame.source)
                if out_frame:
                    await self.send_frame_out(out_frame)
            elif frame.type == MessageType.REQUEST_LAYER_STREAM:
                req = RequestLayerStream.model_validate(frame.payload)
                asyncio.create_task(self._stream_layers_to_peer(req, frame.source))
            elif frame.type == MessageType.STAGE_LAYER_ACK:
                ack = StageLayerAck.model_validate(frame.payload)
                logger.info(
                    f"Stream ACK from '{frame.source}': Layer {ack.layer_index} {ack.status} "
                    f"({ack.ready_layer_count}/{ack.total_assigned} ready)"
                )
            elif frame.type == MessageType.INFERENCE_ABORT and self.engine:
                req_id = frame.payload.get("request_id")
                if req_id:
                    self.engine.abort_inference(req_id)
                    task = self.active_inference_tasks.pop(req_id, None)
                    if task and not task.done():
                        task.cancel()
            elif frame.type == MessageType.INFERENCE_FINISH and self.engine:
                req_id = frame.payload.get("request_id")
                if req_id:
                    self.engine.process_inference_finish(req_id)
        except Exception as e:
            logger.error(f"Error handling frame: {e}", exc_info=True)

    async def send_frame_out(self, out_frame: ProtocolFrame):
        """Sends frame via direct WebRTC UDP DataChannel if open, otherwise falls back to WebSocket relay."""
        frame_json = out_frame.model_dump_json()
        sent_p2p = False
        target_node = self.p2p_nodes.get(out_frame.target)
        if target_node and target_node.is_connected:
            sent_p2p = target_node.send_frame(frame_json)

        # Token emissions MUST reach Coordinator WebSocket for HTTP SSE streaming,
        # but if already delivered via direct P2P, mark target="coordinator" so the
        # Coordinator RelayHub does NOT re-route or reflect the frame back to the worker!
        if sent_p2p:
            should_send_to_coordinator = False
            if out_frame.type == MessageType.SPECULATIVE_VERIFIED:
                should_send_to_coordinator = bool(self.engine and self.engine.is_last_stage())
            elif out_frame.type in (MessageType.NEXT_TOKEN, MessageType.INFERENCE_FINISH, MessageType.ERROR):
                should_send_to_coordinator = True

            if should_send_to_coordinator:
                coord_frame = out_frame.model_copy()
                coord_frame.target = "coordinator"
                if self.ws:
                    try:
                        await self.ws.send(coord_frame.model_dump_json())
                    except Exception as e:
                        logger.debug(f"Failed to send coordinator frame: {e}")
        else:
            if self.ws:
                try:
                    await self.ws.send(frame_json)
                except Exception as e:
                    logger.debug(f"Failed to send frame: {e}")

    async def _stream_layers_to_peer(self, req: RequestLayerStream, target_peer: str):
        """Streams assigned layer chunks asynchronously across P2P / Relay to a downstream worker."""
        try:
            from situs.worker.jit_streamer import ProgressiveJITStreamer
            streamer = ProgressiveJITStreamer(
                model_name=req.model_name,
                allow_synthetic=False,
            )
            logger.info(
                f"Starting wire streaming to '{target_peer}' for '{req.model_name}' "
                f"stage [{req.layer_start}..{req.layer_end}) (entrance first={req.priority_entrance_first})..."
            )
            async for chunk in streamer.stream_stage_layers_async(
                layer_start=req.layer_start,
                layer_end=req.layer_end,
                priority_entrance_first=req.priority_entrance_first,
                extract_from_disk=True,
            ):
                if not self.is_running:
                    break
                chunk_frame = ProtocolFrame(
                    type=MessageType.STREAM_LAYER_CHUNK,
                    source=self.worker_id,
                    target=target_peer,
                    payload=chunk.model_dump(),
                )
                await self.send_frame_out(chunk_frame)
            logger.info(
                f"Completed wire streaming to '{target_peer}' for stage [{req.layer_start}..{req.layer_end})."
            )
        except Exception as e:
            logger.error(f"Error streaming layers to peer '{target_peer}': {e}", exc_info=True)

    def query_gpu_info(self):
        try:
            from situs.hardware import detect_hardware, get_live_gpu_vram, get_system_ram_and_cpu
            if str(self.config.device).strip().lower() == "cpu":
                tot_ram, free_ram, cpu_name = get_system_ram_and_cpu()
                used_ram = max(0, tot_ram - free_ram)
                return f"{cpu_name} (CPU)", tot_ram, free_ram, "cpu", used_ram

            dev_idx = 0
            if ":" in str(self.config.device):
                try:
                    dev_idx = int(str(self.config.device).split(":")[1])
                except Exception:
                    dev_idx = 0

            if self._cached_hw is None or (self._cached_hw[3] != "cuda" and HAS_TORCH and torch.cuda.is_available() and str(self.config.device).startswith("cuda")):
                hw = detect_hardware(requested_device=self.config.device, bypass_cache=True)
                if hw.device != self.config.device and str(self.config.device).startswith("cuda"):
                    self.config.device = hw.device
                self._cached_hw = (hw.device_name, hw.total_memory_bytes, hw.free_memory_bytes, hw.device_type, hw.used_memory_bytes)
            dev_name, tot_b, free_b, dev_type, used_b = self._cached_hw

            if dev_type == "cuda":
                live_tot, live_used, live_free = get_live_gpu_vram(dev_idx)
                if live_tot > 0:
                    if HAS_TORCH and torch.cuda.is_available():
                        try:
                            t_free, t_tot = torch.cuda.mem_get_info(dev_idx)
                            t_used = max(0, t_tot - t_free)
                            if t_used > live_used:
                                live_used = t_used
                                live_free = max(0, live_tot - live_used)
                        except Exception:
                            pass
                        try:
                            alloc_b = torch.cuda.memory_allocated(dev_idx)
                            if alloc_b > live_used:
                                live_used = alloc_b
                                live_free = max(0, live_tot - live_used)
                        except Exception:
                            pass
                    live_used = min(live_tot, max(0, live_used))
                    live_free = min(live_tot, max(0, live_free))
                    return dev_name, live_tot, live_free, "cuda", live_used

                # Fallback if live NVML / nvidia-smi is unavailable
                if HAS_TORCH and torch.cuda.is_available():
                    try:
                        free_b, total_b = torch.cuda.mem_get_info(dev_idx)
                        alloc_b = torch.cuda.memory_allocated(dev_idx)
                        res_b = torch.cuda.memory_reserved(dev_idx)
                        effective_free_b = min(total_b, max(0, free_b + max(0, res_b - alloc_b)))
                        used_b = min(total_b, max(alloc_b, max(0, total_b - effective_free_b)))
                        return dev_name, total_b, effective_free_b, "cuda", used_b
                    except Exception:
                        pass

            used_b = min(tot_b, max(0, used_b))
            free_b = min(tot_b, max(0, free_b))
            return dev_name, tot_b, free_b, dev_type, used_b
        except Exception as e:
            logger.warning(f"Hardware query failed, falling back to real CPU/RAM: {e}")
            from situs.hardware import get_system_ram_and_cpu
            tot_ram, free_ram, cpu_name = get_system_ram_and_cpu()
            return f"{cpu_name} (CPU)", tot_ram, free_ram, "cpu", 0

    async def _emit_inference_error(self, request_id: Optional[str], detail: str):
        """Send an explicit ERROR frame back to the coordinator so inference never stalls silently."""
        err_frame = ProtocolFrame(
            type=MessageType.ERROR,
            source=self.worker_id,
            target="coordinator",
            payload={"request_id": request_id, "error": detail},
        )
        if self.ws:
            try:
                await self.ws.send(err_frame.model_dump_json())
            except Exception as send_err:
                logger.debug(f"Failed sending ERROR frame to coordinator: {send_err}")

    async def _run_single_worker_loop(self, start_msg: InferenceStart):
        """Short-circuits the single-worker token generation loop in-memory without WebSocket roundtrips."""
        req_id = start_msg.request_id
        try:
            if self.assignment and self.assignment.is_first_stage and self.assignment.is_last_stage:
                if self.native_server is None:
                    from situs.inference_backend import NativeInferenceServer, find_gguf_file
                    gguf_file = find_gguf_file(self.assignment.model_name)
                    if gguf_file:
                        self.native_server = NativeInferenceServer()
                        await asyncio.to_thread(self.native_server.start, self.assignment.model_name, gguf_file)
                elif not self.native_server.is_ready():
                    logger.info(f"Native server inactive; auto-restarting for model '{self.assignment.model_name}'...")
                    await asyncio.to_thread(self.native_server.start, self.assignment.model_name)

            if self.native_server and self.native_server.is_ready():
                step_idx = 0
                async for tok in self.native_server.stream_inference(start_msg):
                    if not self.is_running or (self.engine and self.engine.cancellation_tokens.is_cancelled(req_id)):
                        break
                    tok.step_index = step_idx
                    step_idx += 1
                    coord_frame = ProtocolFrame(
                        type=MessageType.NEXT_TOKEN,
                        source=self.worker_id,
                        target="coordinator",
                        payload=tok.model_dump(),
                    )
                    await self.send_frame_out(coord_frame)
                    if tok.is_finished:
                        break
                return

            if not self.engine:
                await self._emit_inference_error(
                    req_id,
                    self.engine_error or f"Worker '{self.worker_id}' has no active pipeline engine loaded.",
                )
                return

            out_frame = await asyncio.to_thread(self.engine.start_inference, start_msg)
            curr_frame = out_frame

            while self.is_running and curr_frame is not None:
                if not self.engine:
                    break
                if self.engine.cancellation_tokens.is_cancelled(req_id) or req_id in self.engine.finished_sessions:
                    break

                if curr_frame.type == MessageType.ERROR:
                    await self.send_frame_out(curr_frame)
                    break

                # Route frame to coordinator for SSE streaming with target="coordinator"
                # so the coordinator relay streams to client without reflecting back into the pipeline ring.
                coord_frame = curr_frame.model_copy()
                coord_frame.target = "coordinator"
                await self.send_frame_out(coord_frame)

                payload = curr_frame.payload
                is_finished = payload.get("is_finished", False) if isinstance(payload, dict) else False
                if is_finished:
                    break

                if curr_frame.type == MessageType.NEXT_TOKEN:
                    tok = NextToken.model_validate(payload)
                    curr_frame = await asyncio.to_thread(self.engine.process_next_token, tok)
                elif curr_frame.type == MessageType.SPECULATIVE_VERIFIED:
                    ver = SpeculativeVerified.model_validate(payload)
                    curr_frame = await asyncio.to_thread(self.engine.process_speculative_verified, ver)
                else:
                    break
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in single-worker token loop for request {req_id}: {e}", exc_info=True)
            await self._emit_inference_error(req_id, str(e))
        finally:
            self.active_inference_tasks.pop(req_id, None)

    async def send_heartbeat_once(self):
        """Send a single immediate telemetry heartbeat (e.g. right after loading model weights)."""
        if not self.ws or not self.is_running:
            return
        gpu_name, total_b, free_b, dev_type, used_b = await asyncio.to_thread(self.query_gpu_info)
        p2p_v = bool(any(n.is_connected and getattr(n, "is_verified", False) for n in self.p2p_nodes.values()))
        hb = Heartbeat(
            worker_id=self.worker_id,
            timestamp=time.time(),
            free_vram_bytes=free_b,
            used_vram_bytes=used_b,
            gpu_name=gpu_name,
            total_vram_bytes=total_b,
            device_type=dev_type,
            active_sessions=len(self.engine.session_kv_caches) if self.engine else 0,
            p2p_verified=p2p_v,
        )
        frame = ProtocolFrame(
            type=MessageType.HEARTBEAT,
            source=self.worker_id,
            target="coordinator",
            payload=hb.model_dump(),
        )
        await self.ws.send(frame.model_dump_json())

    async def heartbeat_loop(self):
        while self.is_running and self.ws:
            try:
                await self.send_heartbeat_once()
                await asyncio.sleep(self.config.heartbeat_interval_s)
            except websockets.ConnectionClosed:
                logger.debug("Heartbeat loop: connection closed")
                break
            except Exception as e:
                if (not self.is_running) or ("cannot schedule new futures after shutdown" in str(e)):
                    break
                logger.warning(f"Heartbeat transient error: {e}")
                await asyncio.sleep(2.0)

    async def run(self):
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        ws_base = self.config.coordinator_url.rstrip("/")
        if not ws_base.endswith(f"/{self.worker_id}"):
            ws_url = f"{ws_base}/{self.worker_id}"
        else:
            ws_url = ws_base
        if getattr(self.config, "auth_token", None):
            sep = "&" if "?" in ws_url else "?"
            ws_url = f"{ws_url}{sep}token={self.config.auth_token}"
        logger.info(f"Connecting outbound to Coordinator Relay: {ws_url}")

        gpu_name, total_vram, free_vram, dev_type, used_vram = self.query_gpu_info()
        logger.info(f"Hardware: {gpu_name} (Used: {used_vram / (1024**3):.2f} GB | Free: {free_vram / (1024**3):.2f} GB | Total: {total_vram / (1024**3):.2f} GB VRAM)")

        import inspect
        sig_params = inspect.signature(websockets.connect).parameters
        # Disable low-level RFC 6455 transport pings because reverse proxies & Cloudflare tunnels
        # drop or do not return WebSocket pong frames, causing false ping timeouts.
        # Situs relies on high-frequency application-level HEARTBEAT/HEARTBEAT_ACK frames (1.0s interval).
        connect_kwargs = {"ping_interval": None, "ping_timeout": None}
        if "max_size" in sig_params:
            connect_kwargs["max_size"] = 64 * 1024 * 1024
        if getattr(self.config, "auth_token", None):
            auth_headers = [
                ("X-Worker-Token", self.config.auth_token),
                ("Authorization", f"Bearer {self.config.auth_token}"),
            ]
            if "additional_headers" in sig_params:
                connect_kwargs["additional_headers"] = auth_headers
            elif "extra_headers" in sig_params:
                connect_kwargs["extra_headers"] = dict(auth_headers)

        retry_count = 0
        has_connected_once = False
        while self.is_running:
            try:
                curr_ws_url = ws_url
                if retry_count > 0 or has_connected_once:
                    sep = "&" if "?" in curr_ws_url else "?"
                    curr_ws_url = f"{curr_ws_url}{sep}reconnect=true"
                async for websocket in websockets.connect(curr_ws_url, **connect_kwargs):
                    if not self.is_running:
                        break
                    retry_count = 0
                    has_connected_once = True
                    self.ws = websocket
                    self.is_connected = True
                    self.is_registered = False
                    self.registered_event.clear()
                    try:
                        # 1. Register with Coordinator (query live hardware state)
                        gpu_name, total_vram, free_vram, dev_type, used_vram = self.query_gpu_info()
                        reg = WorkerRegister(
                            worker_id=self.worker_id,
                            gpu_name=gpu_name,
                            total_vram_bytes=total_vram,
                            free_vram_bytes=free_vram,
                            used_vram_bytes=used_vram,
                            device_type=dev_type,
                            supported_models=[self.config.model_name],
                        )
                        reg_frame = ProtocolFrame(
                            type=MessageType.WORKER_REGISTER,
                            source=self.worker_id,
                            target="coordinator",
                            payload=reg.model_dump(),
                        )
                        await websocket.send(reg_frame.model_dump_json())
                        logger.info(f"Registered with coordinator as {self.worker_id}")

                        # 2. Start Heartbeat background task
                        hb_task = asyncio.create_task(self.heartbeat_loop())

                        # 3. Main message listening loop
                        async for message in websocket:
                            frame = ProtocolFrame.model_validate_json(message)

                            if frame.type == MessageType.WORKER_REGISTERED:
                                self.is_registered = True
                                self.registered_event.set()
                                logger.info(f"Coordinator confirmed registration for worker {self.worker_id}")

                            elif frame.type == MessageType.WORKER_REMOVED:
                                logger.info(f"Worker '{self.worker_id}' was removed by coordinator. Stopping daemon cleanly.")
                                self.is_running = False
                                break
    
                            elif frame.type == MessageType.ASSIGN_LAYERS:
                                try:
                                    self.engine_ready_event.clear()
                                    self.engine_error = None
                                    if self.engine is not None:
                                        self.engine = None
                                        import gc
                                        gc.collect()
                                        if HAS_TORCH and torch.cuda.is_available():
                                            torch.cuda.empty_cache()
                                    assign = AssignLayers.model_validate(frame.payload)
                                    self.assignment = assign
                                    logger.info(
                                        f"Assigned layers [{assign.layer_start}..{assign.layer_end}) of {assign.model_name}. "
                                        f"First={assign.is_first_stage}, Last={assign.is_last_stage}"
                                    )

                                    # 1. Native Fused Server Acceleration for Single-Node Full Allocation
                                    is_single_node = (assign.is_first_stage and assign.is_last_stage)
                                    if not is_single_node and self.native_server:
                                        logger.info(f"Worker '{self.worker_id}' transitioned to multi-node pipeline; stopping native server.")
                                        try:
                                            self.native_server.stop()
                                        except Exception:
                                            pass
                                        self.native_server = None

                                    if is_single_node:
                                        from situs.inference_backend import NativeInferenceServer, find_gguf_file
                                        gguf_file = find_gguf_file(assign.model_name)
                                        if gguf_file:
                                            if self.native_server is None:
                                                self.native_server = NativeInferenceServer()
                                            if not self.native_server.is_ready() or self.native_server.model_name != assign.model_name:
                                                started = await asyncio.to_thread(self.native_server.start, assign.model_name, gguf_file)
                                                if started:
                                                    logger.info(
                                                        f"Worker '{self.worker_id}' single-node accelerated by native server on "
                                                        f"{self.native_server.base_url} (fused GPU kernels, ~30-40 t/s)."
                                                    )
                                                    self.engine = None
                                                    self.engine_ready_event.set()
                                                    try:
                                                        await self.send_heartbeat_once()
                                                    except Exception:
                                                        pass
                                                    continue
                                                else:
                                                    logger.warning(
                                                        f"Could not start native server for '{assign.model_name}'; "
                                                        f"falling back to PyTorch stage model execution."
                                                    )
                                                    try:
                                                        self.native_server.stop()
                                                    except Exception:
                                                        pass
                                                    self.native_server = None
                                    from situs.coordinator.registry import resolve_model_metadata
                                    from situs.worker.loader import is_architecture_supported, find_local_weight_files
                                    try:
                                        meta = resolve_model_metadata(assign.model_name)
                                        arch = meta.get("architecture") or meta.get("model_type")
                                    except Exception:
                                        arch = None

                                    wfiles = find_local_weight_files(assign.model_name)
                                    if not is_architecture_supported(arch):
                                        self.engine_error = (
                                            f"Worker '{self.worker_id}': architecture '{arch}' for '{assign.model_name}' "
                                            f"is not supported for multi-node PyTorch pipeline execution."
                                        )
                                        logger.info(
                                            f"Model '{assign.model_name}' (architecture '{arch}') is not supported for multi-node PyTorch pipeline execution. "
                                            f"Pipeline assignment [{assign.layer_start}..{assign.layer_end}) recorded for cluster topology."
                                        )
                                        self.engine = None
                                        continue

                                    if not wfiles and not assign.progressive:
                                        self.engine_error = (
                                            f"Worker '{self.worker_id}' could not find local weights "
                                            f"for '{assign.model_name}' (stage [{assign.layer_start}..{assign.layer_end}))."
                                        )
                                        logger.warning(self.engine_error)
                                        self.engine = None
                                        continue

                                    stage_model = await asyncio.to_thread(
                                        load_hf_stage_model,
                                        model_name=assign.model_name,
                                        layer_start=assign.layer_start,
                                        layer_end=assign.layer_end,
                                        total_layers=assign.total_layers,
                                        is_first_stage=assign.is_first_stage,
                                        is_last_stage=assign.is_last_stage,
                                        device=self.config.device,
                                        progressive=assign.progressive,
                                    )
                                    from situs.tokenizer import get_or_load_tokenizer
                                    tok = await asyncio.to_thread(
                                        get_or_load_tokenizer,
                                        assign.model_name,
                                        local_files_only=True,
                                    )
                                    engine_comp_dtype = self.config.compression_dtype
                                    if getattr(stage_model, "dtype", None) == torch.bfloat16 and engine_comp_dtype == "fp16":
                                        engine_comp_dtype = "bf16"
                                    self.engine = StageExecutionEngine(
                                        stage_model=stage_model,
                                        assignment=assign,
                                        compression_dtype=engine_comp_dtype,
                                        tokenizer=tok,
                                        ngram_size=self.config.ngram_size,
                                        max_draft_tokens=self.config.max_draft_tokens,
                                        replay_capacity=self.config.replay_buffer_capacity,
                                        safety_headroom_fraction=self.config.safety_headroom_fraction,
                                    )
                                    logger.info(
                                        f"Pipeline stage [{assign.layer_start}..{assign.layer_end}) of {assign.model_name} "
                                        f"pre-warmed and allocated in GPU VRAM (instant zero-latency readiness)."
                                    )

                                    try:
                                        await self.send_heartbeat_once()
                                    except Exception:
                                        pass

                                    # Reset any stale P2P peer connections before establishing new pipeline channels
                                    for peer_id, node in list(self.p2p_nodes.items()):
                                        try:
                                            await node.close()
                                        except Exception:
                                            pass
                                    self.p2p_nodes.clear()

                                    # Identify all partner workers in the pipeline topology
                                    peers_to_connect = set()
                                    if assign.next_worker_id and assign.next_worker_id != self.worker_id:
                                        peers_to_connect.add(assign.next_worker_id)
                                    if assign.prev_worker_id and assign.prev_worker_id != self.worker_id:
                                        peers_to_connect.add(assign.prev_worker_id)
                                    if getattr(assign, "first_worker_id", None) and assign.first_worker_id != self.worker_id:
                                        peers_to_connect.add(assign.first_worker_id)

                                    for peer_id in peers_to_connect:
                                        p_node = P2PDataChannelNode(
                                            worker_id=self.worker_id,
                                            target_worker_id=peer_id,
                                            signaling_send_fn=self.send_signaling_frame,
                                            on_message_fn=self.on_p2p_message,
                                        )
                                        self.p2p_nodes[peer_id] = p_node
                                        # Deterministic initiator: lower worker_id initiates WebRTC offer
                                        is_initiator = self.worker_id < peer_id
                                        asyncio.create_task(p_node.initialize(is_initiator=is_initiator))
                                except Exception as e:
                                    if (not self.is_running) or ("cannot schedule new futures after shutdown" in str(e)):
                                        pass
                                    else:
                                        self.engine = None
                                        self.engine_error = f"Worker '{self.worker_id}' stage load failed: {e}"
                                        logger.error(f"Error handling ASSIGN_LAYERS: {e}", exc_info=True)
                                finally:
                                    self.engine_ready_event.set()
    
                            elif frame.type == MessageType.UNLOAD_MODEL:
                                logger.info(f"Worker '{self.worker_id}': Received UNLOAD_MODEL signal. Freeing GPU memory...")
                                self.engine_ready_event.clear()
                                self.engine_error = None
                                if self.native_server is not None:
                                    try:
                                        self.native_server.stop()
                                    except Exception:
                                        pass
                                    self.native_server = None
                                if self.engine is not None:
                                    self.engine = None
                                import gc
                                gc.collect()
                                if HAS_TORCH and torch.cuda.is_available():
                                    try:
                                        torch.cuda.empty_cache()
                                    except Exception:
                                        pass
                                self.engine_ready_event.set()
                                try:
                                    await self.send_heartbeat_once()
                                except Exception:
                                    pass
                                logger.info(f"Worker '{self.worker_id}': Model unloaded. GPU VRAM returned to idle.")

                            elif frame.type in (MessageType.ICE_OFFER, MessageType.ICE_ANSWER):
                                src_peer = frame.payload.get("source_worker") if isinstance(frame.payload, dict) else None
                                if src_peer:
                                    if src_peer not in self.p2p_nodes:
                                        p_node = P2PDataChannelNode(
                                            worker_id=self.worker_id,
                                            target_worker_id=src_peer,
                                            signaling_send_fn=self.send_signaling_frame,
                                            on_message_fn=self.on_p2p_message,
                                        )
                                        self.p2p_nodes[src_peer] = p_node
                                        asyncio.create_task(p_node.initialize(is_initiator=False))
                                    await self.p2p_nodes[src_peer].handle_signaling_message(frame.payload)
                                elif self.p2p_nodes:
                                    first_node = next(iter(self.p2p_nodes.values()))
                                    await first_node.handle_signaling_message(frame.payload)
    
                            elif frame.type == MessageType.INFERENCE_START:
                                eng_timeout = getattr(self.config, "engine_ready_timeout_s", 120.0)
                                try:
                                    await asyncio.wait_for(self.engine_ready_event.wait(), timeout=eng_timeout)
                                except asyncio.TimeoutError:
                                    logger.error(f"Worker {self.worker_id}: Engine not ready for INFERENCE_START within {eng_timeout}s")
                                start_msg = InferenceStart.model_validate(frame.payload)
                                has_native = bool(self.native_server and self.native_server.is_ready())
                                if not self.engine and not has_native:
                                    await self._emit_inference_error(
                                        start_msg.request_id,
                                        self.engine_error or f"Worker '{self.worker_id}' has no loaded model engine for '{start_msg.model_name}'.",
                                    )
                                    continue
                                if has_native or (self.engine and self.engine.is_first_stage() and self.engine.is_last_stage()):
                                    t = asyncio.create_task(self._run_single_worker_loop(start_msg))
                                    self.active_inference_tasks[start_msg.request_id] = t
                                elif self.engine and self.engine.is_first_stage():
                                    out_frame = await asyncio.to_thread(self.engine.start_inference, start_msg)
                                    if out_frame:
                                        await self.send_frame_out(out_frame)
                                elif self.engine:
                                    self.engine.session_configs[start_msg.request_id] = start_msg
                                    self.engine.session_token_histories[start_msg.request_id] = list(start_msg.input_ids)
                                    self.engine.session_step_counts[start_msg.request_id] = len(start_msg.input_ids)
    
                            elif frame.type == MessageType.FORWARD_ACTIVATION:
                                eng_timeout = getattr(self.config, "engine_ready_timeout_s", 120.0)
                                try:
                                    await asyncio.wait_for(self.engine_ready_event.wait(), timeout=eng_timeout)
                                except asyncio.TimeoutError:
                                    logger.error(f"Worker {self.worker_id}: Engine not ready for FORWARD_ACTIVATION within {eng_timeout}s")
                                act_frame = ForwardActivation.model_validate(frame.payload)
                                if not self.engine:
                                    await self._emit_inference_error(
                                        act_frame.request_id,
                                        self.engine_error or f"Worker '{self.worker_id}' has no loaded pipeline stage to process activations.",
                                    )
                                    continue
                                out_frame = await asyncio.to_thread(self.engine.process_forward_activation, act_frame)
                                if out_frame:
                                    await self.send_frame_out(out_frame)
    
                            elif frame.type == MessageType.SPECULATIVE_CANDIDATES:
                                eng_timeout = getattr(self.config, "engine_ready_timeout_s", 120.0)
                                try:
                                    await asyncio.wait_for(self.engine_ready_event.wait(), timeout=eng_timeout)
                                except asyncio.TimeoutError:
                                    logger.error(f"Worker {self.worker_id}: Engine not ready for SPECULATIVE_CANDIDATES within {eng_timeout}s")
                                cand_req = frame.payload.get("request_id") if isinstance(frame.payload, dict) else None
                                if not self.engine:
                                    await self._emit_inference_error(
                                        cand_req,
                                        self.engine_error or f"Worker '{self.worker_id}' has no loaded pipeline stage to process speculative candidates.",
                                    )
                                    continue
                                cand_frame = SpeculativeCandidates.model_validate(frame.payload)
                                out_frame = await asyncio.to_thread(self.engine.process_speculative_candidates, cand_frame)
                                if out_frame:
                                    await self.send_frame_out(out_frame)
    
                            elif frame.type == MessageType.SPECULATIVE_VERIFIED:
                                eng_timeout = getattr(self.config, "engine_ready_timeout_s", 120.0)
                                try:
                                    await asyncio.wait_for(self.engine_ready_event.wait(), timeout=eng_timeout)
                                except asyncio.TimeoutError:
                                    logger.error(f"Worker {self.worker_id}: Engine not ready for SPECULATIVE_VERIFIED within {eng_timeout}s")
                                ver_req = frame.payload.get("request_id") if isinstance(frame.payload, dict) else None
                                if not self.engine:
                                    await self._emit_inference_error(
                                        ver_req,
                                        self.engine_error or f"Worker '{self.worker_id}' has no loaded pipeline stage to process speculative verified.",
                                    )
                                    continue
                                ver_frame = SpeculativeVerified.model_validate(frame.payload)
                                out_frame = await asyncio.to_thread(self.engine.process_speculative_verified, ver_frame)
                                if out_frame:
                                    await self.send_frame_out(out_frame)
    
                            elif frame.type == MessageType.NEXT_TOKEN:
                                eng_timeout = getattr(self.config, "engine_ready_timeout_s", 120.0)
                                try:
                                    await asyncio.wait_for(self.engine_ready_event.wait(), timeout=eng_timeout)
                                except asyncio.TimeoutError:
                                    logger.error(f"Worker {self.worker_id}: Engine not ready for NEXT_TOKEN within {eng_timeout}s")
                                tok_req = frame.payload.get("request_id") if isinstance(frame.payload, dict) else None
                                if not self.engine:
                                    await self._emit_inference_error(
                                        tok_req,
                                        self.engine_error or f"Worker '{self.worker_id}' has no loaded pipeline stage to process next token.",
                                    )
                                    continue
                                tok_frame = NextToken.model_validate(frame.payload)
                                out_frame = await asyncio.to_thread(self.engine.process_next_token, tok_frame)
                                if out_frame:
                                    await self.send_frame_out(out_frame)
    
                            elif frame.type == MessageType.STREAM_LAYER_CHUNK:
                                if self.engine:
                                    chunk = StreamLayerChunk.model_validate(frame.payload)
                                    out_frame = await asyncio.to_thread(self.engine.process_layer_chunk, chunk, frame.source)
                                    if out_frame:
                                        await self.send_frame_out(out_frame)

                            elif frame.type == MessageType.REQUEST_LAYER_STREAM:
                                req = RequestLayerStream.model_validate(frame.payload)
                                asyncio.create_task(self._stream_layers_to_peer(req, frame.source))

                            elif frame.type == MessageType.STAGE_LAYER_ACK:
                                ack = StageLayerAck.model_validate(frame.payload)
                                logger.info(
                                    f"Stream ACK from '{frame.source}': Layer {ack.layer_index} {ack.status} "
                                    f"({ack.ready_layer_count}/{ack.total_assigned} ready)"
                                )
    
                            elif frame.type == MessageType.INFERENCE_ABORT:
                                req_id = frame.payload.get("request_id") if isinstance(frame.payload, dict) else None
                                if req_id:
                                    if self.engine:
                                        self.engine.abort_inference(req_id)
                                    task = self.active_inference_tasks.pop(req_id, None)
                                    if task and not task.done():
                                        task.cancel()

                            elif frame.type == MessageType.INFERENCE_FINISH:
                                req_id = frame.payload.get("request_id") if isinstance(frame.payload, dict) else None
                                if req_id and self.engine:
                                    self.engine.process_inference_finish(req_id)

                            elif frame.type == MessageType.WORKER_REMOVED:
                                logger.info(f"Worker '{self.worker_id}' received WORKER_REMOVED from coordinator. Terminating daemon cleanly.")
                                self.stop()
                                break
    
                    except websockets.ConnectionClosed as e:
                        self.is_connected = False
                        reason_str = getattr(e, 'reason', str(e))
                        self.last_error = reason_str
                        if getattr(e, "code", None) == 1008:
                            logger.error(f"Registration rejected by coordinator policy (1008): {reason_str}")
                            self.connection_error = f"Policy rejection (1008): {reason_str}"
                            self.is_running = False
                            break
                        else:
                            logger.warning(f"Connection closed ({e}). Reconnecting in 1.5 seconds...")
                            await asyncio.sleep(1.5)
                    except Exception as e:
                        self.is_connected = False
                        self.last_error = str(e)
                        logger.error(f"Worker runtime exception: {e}", exc_info=True)
                        await asyncio.sleep(2.0)
                    finally:
                        self.is_connected = False
                        self.is_registered = False
                        self.registered_event.clear()
                        if 'hb_task' in locals() and not hb_task.done():
                            hb_task.cancel()
                    if not self.is_running:
                        break
            except Exception as conn_err:
                self.connection_error = str(conn_err)
                self.last_error = str(conn_err)
                if not self.is_running:
                    break
                retry_count += 1
                wait_time = min(5.0, 1.0 + (retry_count * 0.5))
                logger.info(f"Connection attempt {retry_count} to {ws_url} failed ({conn_err}). Retrying in {wait_time:.1f}s...")
                await asyncio.sleep(wait_time)
        self.is_running = False
        self.is_connected = False
        self.is_registered = False
        self.registered_event.clear()
        for peer_id, node in list(self.p2p_nodes.items()):
            try:
                await node.close()
            except Exception:
                pass
        self.p2p_nodes.clear()


def main():
    parser = argparse.ArgumentParser(description="Situs Distributed GPU Worker Daemon")
    parser.add_argument("--coordinator", default="ws://127.0.0.1:8000/ws/worker", help="Coordinator WebSocket URL")
    parser.add_argument("--id", default=None, help="Worker ID (auto-generated if omitted)")
    parser.add_argument("--device", default="cuda:0" if HAS_TORCH and torch.cuda.is_available() else "cpu", help="PyTorch compute device")
    parser.add_argument("--model", default="gemma3:4b", help="Model identifier")
    parser.add_argument("--compression", default="bf16", choices=["none", "fp16", "bf16", "int8"], help="Activation compression dtype")
    parser.add_argument("--token", "--auth-token", default=os.getenv("SITUS_WORKER_AUTH_TOKEN"), help="Worker authentication token")
    args = parser.parse_args()

    try:
        cfg = WorkerConfig(
            coordinator_url=args.coordinator,
            worker_id=args.id,
            device=args.device,
            model_name=args.model,
            compression_dtype=args.compression,
            auth_token=args.token,
        )
    except ValueError as e:
        sys.stderr.write(f"\n[Situs Security Configuration Error] {e}\n\n")
        sys.exit(1)

    daemon = WorkerDaemon(cfg)
    asyncio.run(daemon.run())


if __name__ == "__main__":
    main()
