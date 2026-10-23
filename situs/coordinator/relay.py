"""Duplex Outbound Relay Hub for routing activations and tokens between NAT-isolated workers."""

import asyncio
import json
import logging
from typing import Dict, Optional, AsyncGenerator
from fastapi import WebSocket, WebSocketDisconnect
from situs.protocol.messages import ProtocolFrame, MessageType

logger = logging.getLogger("situs.coordinator.relay")


class RelayHub:
    def __init__(self, registry=None):
        self.registry = registry
        # Maps worker_id -> WebSocket connection
        self.active_connections: Dict[str, WebSocket] = {}
        # Maps request_id -> asyncio.Queue for streaming generated tokens back to HTTP client
        self.response_streams: Dict[str, asyncio.Queue] = {}

    def set_registry(self, registry):
        self.registry = registry

    async def connect_worker(self, worker_id: str, websocket: WebSocket):
        await websocket.accept()
        self.active_connections[worker_id] = websocket
        logger.info(f"Worker connected to relay: {worker_id}")

    def disconnect_worker(self, worker_id: str):
        if worker_id in self.active_connections:
            del self.active_connections[worker_id]
            logger.info(f"Worker disconnected from relay: {worker_id}")

    def is_worker_connected(self, worker_id: str) -> bool:
        return worker_id in self.active_connections

    async def send_to_worker(self, worker_id: str, frame: ProtocolFrame) -> bool:
        ws = self.active_connections.get(worker_id)
        if not ws:
            logger.error(f"Cannot route message: Worker {worker_id} not connected")
            return False
        try:
            await ws.send_text(frame.model_dump_json())
            return True
        except Exception as e:
            logger.error(f"Failed to send to worker {worker_id}: {e}")
            self.disconnect_worker(worker_id)
            return False

    async def route_frame(self, frame: ProtocolFrame):
        """
        Routes an incoming frame:
        - If target is specified, forward to target worker.
        - If type is NEXT_TOKEN or SPECULATIVE_VERIFIED, push to HTTP response stream
          AND ensure it is routed directly to the first pipeline stage to continue token generation.
        """
        request_id = frame.payload.get("request_id")
        
        # If this frame carries a token or completion, push to the waiting HTTP response stream
        if frame.type in (MessageType.NEXT_TOKEN, MessageType.SPECULATIVE_VERIFIED, MessageType.INFERENCE_FINISH, MessageType.ERROR):
            if request_id and request_id in self.response_streams:
                await self.response_streams[request_id].put(frame)

        # For tokens/speculative verification, route to worker ONLY if not already delivered via direct P2P
        if frame.type in (MessageType.NEXT_TOKEN, MessageType.SPECULATIVE_VERIFIED):
            if frame.target == "coordinator":
                return  # Delivered for HTTP SSE client streaming only; do not reflect back into pipeline ring
            target_worker = frame.target
            if self.registry:
                try:
                    pipeline = self.registry.get_pipeline_order()
                    if pipeline:
                        target_worker = pipeline[0].worker_id
                except Exception:
                    pass
            if target_worker:
                await self.send_to_worker(target_worker, frame)
            return

        # If it has a target worker (e.g. forward activation A -> B)
        if frame.target and frame.target != "coordinator":
            await self.send_to_worker(frame.target, frame)

    def register_response_stream(self, request_id: str) -> asyncio.Queue:
        q = asyncio.Queue()
        self.response_streams[request_id] = q
        return q

    def cleanup_response_stream(self, request_id: str):
        if request_id in self.response_streams:
            del self.response_streams[request_id]
