"""Direct Peer-to-Peer WebRTC DataChannel transport with STUN NAT hole-punching."""

import os
import json
import logging
import asyncio
from typing import Optional, Callable, Dict, Any, Union

try:
    from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceCandidate, RTCConfiguration, RTCIceServer
    HAS_AIORTC = True
except ImportError:
    HAS_AIORTC = False

logger = logging.getLogger("situs.network.p2p")

# Default public Google STUN servers for residential UDP hole-punching
DEFAULT_STUN_SERVERS = [
    "stun:stun.l.google.com:19302",
    "stun:stun1.l.google.com:19302",
]

# Conservative SCTP message limit for direct DataChannel transfer.
# Sized at 128 KB by default so dense speculative candidate bursts (~55 KB) transmit
# directly over P2P UDP, configurable via SITUS_P2P_MAX_BYTES for higher bandwidth environments.
def get_safe_sctp_max_bytes() -> int:
    try:
        return int(os.environ.get("SITUS_P2P_MAX_BYTES", 128 * 1024))
    except (ValueError, TypeError):
        return 128 * 1024

SAFE_SCTP_MAX_BYTES = 128 * 1024


class P2PDataChannelNode:
    """
    Manages a direct WebRTC SCTP-over-DTLS/UDP DataChannel between two workers.
    Bypasses the Coordinator Relay for direct fiber-to-fiber ping.
    """
    def __init__(
        self,
        worker_id: str,
        target_worker_id: str,
        signaling_send_fn: Callable[[Dict[str, Any]], asyncio.Future],
        on_message_fn: Callable[[str], None],
    ):
        self.worker_id = worker_id
        self.target_worker_id = target_worker_id
        self.signaling_send_fn = signaling_send_fn
        self.on_message_fn = on_message_fn
        self.pc: Optional[RTCPeerConnection] = None
        self.channel = None
        self.is_connected = False
        self.is_verified = False

    async def initialize(self, is_initiator: bool = True):
        if not HAS_AIORTC:
            logger.warning("aiortc not installed; P2P direct data channels disabled, using WebSocket relay.")
            return

        try:
            config = RTCConfiguration(iceServers=[RTCIceServer(urls=DEFAULT_STUN_SERVERS)])
            self.pc = RTCPeerConnection(configuration=config)

            @self.pc.on("iceconnectionstatechange")
            def on_ice_state_change():
                state = getattr(self.pc, "iceConnectionState", "")
                if state in ("failed", "closed", "disconnected"):
                    self.is_connected = False
                    self.is_verified = False
                    logger.warning(f"P2P ICE state changed to '{state}' for {self.target_worker_id}; falling back to WebSocket relay.")

            if is_initiator:
                # Worker A initiates the DataChannel
                self.channel = self.pc.createDataChannel("situs-tensors", ordered=True)
                self._setup_channel(self.channel)

                offer = await self.pc.createOffer()
                await self.pc.setLocalDescription(offer)
                # Allow STUN queries to complete so public NAT candidates are embedded in SDP
                for _ in range(16):
                    if getattr(self.pc, "iceGatheringState", None) == "complete":
                        break
                    await asyncio.sleep(0.05)
                # Send offer through Coordinator WebSocket signaling
                await self.signaling_send_fn({
                    "type": "ice_offer",
                    "source_worker": self.worker_id,
                    "target_worker": self.target_worker_id,
                    "sdp": self.pc.localDescription.sdp,
                })
                logger.info(f"Worker {self.worker_id} sent WebRTC P2P offer to {self.target_worker_id}")
            else:
                # Worker B awaits incoming DataChannel
                @self.pc.on("datachannel")
                def on_datachannel(channel):
                    logger.info(f"Worker {self.worker_id} received incoming P2P DataChannel from {self.target_worker_id}")
                    self.channel = channel
                    self._setup_channel(channel)
        except Exception as e:
            logger.warning(f"Failed to initialize P2P direct channel to {self.target_worker_id}: {e}")

    def _setup_channel(self, channel):
        @channel.on("open")
        def on_open():
            logger.info(f"Direct P2P WebRTC DataChannel OPEN between {self.worker_id} and {self.target_worker_id}! Probing bidirectional link...")
            self.is_connected = True
            # Send initial probe to verify bidirectional packet delivery over UDP
            try:
                channel.send(json.dumps({"__situs_probe__": "ping", "src": self.worker_id}))
            except Exception:
                self.is_connected = False
                self.is_verified = False

        @channel.on("close")
        def on_close():
            logger.warning(f"P2P WebRTC DataChannel CLOSED between {self.worker_id} and {self.target_worker_id}")
            self.is_connected = False
            self.is_verified = False

        @channel.on("message")
        def on_message(message):
            self._handle_incoming_channel_message(message)

    def _handle_incoming_channel_message(self, message: Any):
        if isinstance(message, str) and "__situs_probe__" in message:
            try:
                p = json.loads(message)
                if p.get("__situs_probe__") == "ping":
                    if self.channel:
                        self.channel.send(json.dumps({"__situs_probe__": "pong", "src": self.worker_id}))
                    self.is_verified = True
                    logger.info(f"P2P DataChannel verified bidirectional UDP delivery with {self.target_worker_id}")
                elif p.get("__situs_probe__") == "pong":
                    self.is_verified = True
                    logger.info(f"P2P DataChannel confirmed bidirectional UDP delivery from {self.target_worker_id}")
                return
            except Exception:
                pass
        self.on_message_fn(message)

    async def handle_signaling_message(self, msg: Dict[str, Any]):
        """Handles incoming SDP offers/answers and ICE candidates from peer."""
        if not self.pc:
            return

        msg_type = msg.get("type")
        sdp = msg.get("sdp")

        if msg_type == "ice_offer" and sdp:
            desc = RTCSessionDescription(sdp=sdp, type="offer")
            await self.pc.setRemoteDescription(desc)
            answer = await self.pc.createAnswer()
            await self.pc.setLocalDescription(answer)
            # Allow STUN queries to complete so public NAT candidates are embedded in SDP
            for _ in range(16):
                if getattr(self.pc, "iceGatheringState", None) == "complete":
                    break
                await asyncio.sleep(0.05)
            await self.signaling_send_fn({
                "type": "ice_answer",
                "source_worker": self.worker_id,
                "target_worker": self.target_worker_id,
                "sdp": self.pc.localDescription.sdp,
            })
            logger.info(f"Worker {self.worker_id} answered WebRTC offer from {self.target_worker_id}")

        elif msg_type == "ice_answer" and sdp:
            if self.pc.signalingState == "have-local-offer":
                desc = RTCSessionDescription(sdp=sdp, type="answer")
                await self.pc.setRemoteDescription(desc)
                logger.info(f"Worker {self.worker_id} established remote description with {self.target_worker_id}")
            else:
                logger.debug(f"Ignoring ice_answer in signaling state {self.pc.signalingState}")

    def send_frame(self, data_str: Union[str, bytes]) -> bool:
        """Sends data over direct P2P UDP DataChannel if open, verified, and within safe SCTP bounds."""
        if not (self.is_connected and self.is_verified and self.channel and self.channel.readyState == "open"):
            return False
        byte_len = len(data_str) if isinstance(data_str, bytes) else len(data_str.encode("utf-8"))
        max_allowed = get_safe_sctp_max_bytes()
        if byte_len > max_allowed:
            return False
        try:
            self.channel.send(data_str)
            return True
        except Exception as e:
            logger.debug(f"P2P DataChannel send failed ({e}); falling back to WebSocket relay.")
            self.is_connected = False
            self.is_verified = False
            return False

    async def close(self):
        if self.pc:
            await self.pc.close()
            self.is_connected = False
            self.is_verified = False
