"""Worker client daemon connecting outbound to Coordinator Relay."""

import os
import sys
import time
import uuid
import json
import shutil
import subprocess
import asyncio
import logging
import argparse
from typing import Optional
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
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self.p2p_node: Optional[P2PDataChannelNode] = None
        self.is_running = True
        self._cached_hw = None

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
            if frame.type == MessageType.FORWARD_ACTIVATION and self.engine:
                act_frame = ForwardActivation.model_validate(frame.payload)
                out_frame = await asyncio.to_thread(self.engine.process_forward_activation, act_frame)
                if out_frame:
                    await self.send_frame_out(out_frame)
            elif frame.type == MessageType.SPECULATIVE_CANDIDATES and self.engine:
                cand_frame = SpeculativeCandidates.model_validate(frame.payload)
                out_frame = await asyncio.to_thread(self.engine.process_speculative_candidates, cand_frame)
                if out_frame:
                    await self.send_frame_out(out_frame)
            elif frame.type == MessageType.SPECULATIVE_VERIFIED and self.engine:
                ver_frame = SpeculativeVerified.model_validate(frame.payload)
                out_frame = await asyncio.to_thread(self.engine.process_speculative_verified, ver_frame)
                if out_frame:
                    await self.send_frame_out(out_frame)
            elif frame.type == MessageType.NEXT_TOKEN and self.engine:
                tok_frame = NextToken.model_validate(frame.payload)
                out_frame = await asyncio.to_thread(self.engine.process_next_token, tok_frame)
                if out_frame:
                    await self.send_frame_out(out_frame)
            elif frame.type == MessageType.STREAM_LAYER_CHUNK and self.engine:
                chunk = StreamLayerChunk.model_validate(frame.payload)
                out_frame = await asyncio.to_thread(self.engine.process_layer_chunk, chunk, frame.source)
                if out_frame:
                    await self.send_frame_out(out_frame)
            elif frame.type == MessageType.INFERENCE_ABORT and self.engine:
                req_id = frame.payload.get("request_id")
                if req_id:
                    self.engine.abort_inference(req_id)
        except Exception as e:
            logger.error(f"Error handling frame: {e}", exc_info=True)

    async def send_frame_out(self, out_frame: ProtocolFrame):
        """Sends frame via direct WebRTC UDP DataChannel if open, otherwise falls back to WebSocket relay."""
        frame_json = out_frame.model_dump_json()
        sent_p2p = False
        if self.p2p_node and self.p2p_node.is_connected and out_frame.target == self.p2p_node.target_worker_id:
            sent_p2p = self.p2p_node.send_frame(frame_json)

        # Token emissions MUST reach Coordinator WebSocket for HTTP SSE streaming,
        # but if already delivered via direct P2P, mark target="coordinator" so the
        # Coordinator RelayHub does NOT re-route or reflect the frame back to the worker!
        if sent_p2p:
            if out_frame.type in (MessageType.NEXT_TOKEN, MessageType.SPECULATIVE_VERIFIED, MessageType.INFERENCE_FINISH, MessageType.ERROR):
                coord_frame = out_frame.model_copy()
                coord_frame.target = "coordinator"
                if self.ws:
                    try:
                        await self.ws.send(coord_frame.model_dump_json())
                    except Exception:
                        pass
        else:
            if self.ws:
                try:
                    await self.ws.send(frame_json)
                except Exception:
                    pass

    def query_gpu_info(self):
        try:
            from situs.hardware import detect_hardware, get_live_gpu_vram
            if self._cached_hw is None:
                hw = detect_hardware(requested_device=self.config.device)
                if hw.device != self.config.device and self.config.device.startswith("cuda"):
                    self.config.device = hw.device
                self._cached_hw = (hw.device_name, hw.total_memory_bytes, hw.free_memory_bytes, hw.device_type, hw.used_memory_bytes)
            dev_name, tot_b, free_b, dev_type, used_b = self._cached_hw
            live_tot, live_used, live_free = get_live_gpu_vram()
            if live_tot > 0:
                return dev_name, live_tot, live_free, dev_type, live_used
            return dev_name, tot_b, free_b, dev_type, used_b
        except Exception as e:
            logger.warning(f"Hardware query failed, falling back to real CPU/RAM: {e}")
            from situs.hardware import get_system_ram_and_cpu
            tot_ram, free_ram, cpu_name = get_system_ram_and_cpu()
            return f"{cpu_name} (CPU)", tot_ram, free_ram, "cpu", max(0, tot_ram - free_ram)

    async def heartbeat_loop(self):
        while self.is_running and self.ws:
            try:
                gpu_name, total_b, free_b, dev_type, used_b = await asyncio.to_thread(self.query_gpu_info)
                hb = Heartbeat(
                    worker_id=self.worker_id,
                    timestamp=time.time(),
                    free_vram_bytes=free_b,
                    used_vram_bytes=used_b,
                    gpu_name=gpu_name,
                    total_vram_bytes=total_b,
                    device_type=dev_type,
                    active_sessions=len(self.engine.session_kv_caches) if self.engine else 0,
                )
                frame = ProtocolFrame(
                    type=MessageType.HEARTBEAT,
                    source=self.worker_id,
                    target="coordinator",
                    payload=hb.model_dump(),
                )
                await self.ws.send(frame.model_dump_json())
                await asyncio.sleep(self.config.heartbeat_interval_s)
            except websockets.ConnectionClosed:
                logger.debug("Heartbeat loop: connection closed")
                break
            except Exception as e:
                logger.warning(f"Heartbeat transient error: {e}")
                await asyncio.sleep(2.0)

    async def run(self):
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
        logger.info(f"Hardware: {gpu_name} (Used: {used_vram / 1e9:.2f} GB | Free: {free_vram / 1e9:.2f} GB | Total: {total_vram / 1e9:.2f} GB VRAM)")

        # Pre-warm tokenizer in background so registration is instant without network delays
        async def _prewarm_tok():
            try:
                from situs.tokenizer import get_or_load_tokenizer
                await asyncio.to_thread(get_or_load_tokenizer, self.config.model_name)
            except Exception as e:
                logger.debug(f"Tokenizer pre-warm notice: {e}")

        asyncio.create_task(_prewarm_tok())

        connect_kwargs = {"ping_interval": 20, "ping_timeout": 120}
        if getattr(self.config, "auth_token", None):
            import inspect
            sig_params = inspect.signature(websockets.connect).parameters
            auth_headers = [
                ("X-Worker-Token", self.config.auth_token),
                ("Authorization", f"Bearer {self.config.auth_token}"),
            ]
            if "additional_headers" in sig_params:
                connect_kwargs["additional_headers"] = auth_headers
            elif "extra_headers" in sig_params:
                connect_kwargs["extra_headers"] = dict(auth_headers)

        async for websocket in websockets.connect(ws_url, **connect_kwargs):
            self.ws = websocket
            try:
                # 1. Register with Coordinator
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

                    if frame.type == MessageType.ASSIGN_LAYERS:
                        try:
                            assign = AssignLayers.model_validate(frame.payload)
                            logger.info(
                                f"Assigned layers [{assign.layer_start}..{assign.layer_end}) of {assign.model_name}. "
                                f"First={assign.is_first_stage}, Last={assign.is_last_stage}"
                            )
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
                                fast_boot=assign.fast_boot,
                            )
                            from situs.tokenizer import get_or_load_tokenizer
                            tok = await asyncio.to_thread(
                                get_or_load_tokenizer,
                                assign.model_name,
                            )
                            self.engine = StageExecutionEngine(
                                stage_model=stage_model,
                                assignment=assign,
                                compression_dtype=self.config.compression_dtype,
                                tokenizer=tok,
                            )

                            # Initialize WebRTC direct P2P connection to partner worker
                            target_id = assign.next_worker_id or assign.prev_worker_id
                            if target_id:
                                self.p2p_node = P2PDataChannelNode(
                                    worker_id=self.worker_id,
                                    target_worker_id=target_id,
                                    signaling_send_fn=self.send_signaling_frame,
                                    on_message_fn=self.on_p2p_message,
                                    )
                                # First stage initiates the WebRTC offer
                                asyncio.create_task(self.p2p_node.initialize(is_initiator=assign.is_first_stage))
                        except Exception as e:
                            logger.error(f"Error handling ASSIGN_LAYERS: {e}", exc_info=True)

                    elif frame.type in (MessageType.ICE_OFFER, MessageType.ICE_ANSWER):
                        if self.p2p_node:
                            await self.p2p_node.handle_signaling_message(frame.payload)

                    elif frame.type == MessageType.INFERENCE_START:
                        if self.engine:
                            start_msg = InferenceStart.model_validate(frame.payload)
                            if self.engine.is_first_stage():
                                out_frame = await asyncio.to_thread(self.engine.start_inference, start_msg)
                                if out_frame:
                                    await self.send_frame_out(out_frame)
                            else:
                                self.engine.session_configs[start_msg.request_id] = start_msg
                                self.engine.session_token_histories[start_msg.request_id] = list(start_msg.input_ids)
                                if hasattr(self.engine, "init_fast_boot_sequence"):
                                    prompt_str = getattr(start_msg, "prompt", "") or ""
                                    if not prompt_str and self.engine.sampler.tokenizer:
                                        try:
                                            prompt_str = self.engine.sampler.tokenizer.decode(start_msg.input_ids)
                                        except Exception:
                                            prompt_str = ""
                                    self.engine.init_fast_boot_sequence(start_msg.request_id, prompt_str, start_msg.model_name)

                    elif frame.type == MessageType.FORWARD_ACTIVATION:
                        if self.engine:
                            act_frame = ForwardActivation.model_validate(frame.payload)
                            out_frame = await asyncio.to_thread(self.engine.process_forward_activation, act_frame)
                            if out_frame:
                                await self.send_frame_out(out_frame)

                    elif frame.type == MessageType.SPECULATIVE_CANDIDATES:
                        if self.engine:
                            cand_frame = SpeculativeCandidates.model_validate(frame.payload)
                            out_frame = await asyncio.to_thread(self.engine.process_speculative_candidates, cand_frame)
                            if out_frame:
                                await self.send_frame_out(out_frame)

                    elif frame.type == MessageType.SPECULATIVE_VERIFIED:
                        if self.engine:
                            ver_frame = SpeculativeVerified.model_validate(frame.payload)
                            out_frame = await asyncio.to_thread(self.engine.process_speculative_verified, ver_frame)
                            if out_frame:
                                await self.send_frame_out(out_frame)

                    elif frame.type == MessageType.NEXT_TOKEN:
                        if self.engine:
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

                    elif frame.type == MessageType.INFERENCE_ABORT:
                        if self.engine:
                            req_id = frame.payload.get("request_id")
                            if req_id:
                                self.engine.abort_inference(req_id)

            except websockets.ConnectionClosed as e:
                logger.warning(f"Connection closed ({e}). Reconnecting in 3 seconds...")
                await asyncio.sleep(3.0)
            except Exception as e:
                logger.error(f"Worker runtime exception: {e}", exc_info=True)
                await asyncio.sleep(3.0)


def main():
    parser = argparse.ArgumentParser(description="Situs Distributed GPU Worker Daemon")
    parser.add_argument("--coordinator", default="ws://127.0.0.1:8000/ws/worker", help="Coordinator WebSocket URL")
    parser.add_argument("--id", default=None, help="Worker ID (auto-generated if omitted)")
    parser.add_argument("--device", default="cuda:0" if HAS_TORCH and torch.cuda.is_available() else "cpu", help="PyTorch compute device")
    parser.add_argument("--model", default="gemma3:4b", help="Model identifier")
    parser.add_argument("--compression", default="int8", choices=["none", "fp16", "int8"], help="Activation compression dtype")
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
