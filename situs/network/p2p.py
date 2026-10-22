"""Direct Peer-to-Peer WebRTC DataChannel transport with STUN NAT hole-punching."""

import json
import logging
import asyncio
from typing import Optional, Callable, Dict, Any

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

    async def initialize(self, is_initiator: bool = True):
        if not HAS_AIORTC:
            logger.warning("aiortc not installed; P2P direct data channels disabled, using WebSocket relay.")
            return

        config = RTCConfiguration(iceServers=[RTCIceServer(urls=DEFAULT_STUN_SERVERS)])
        self.pc = RTCPeerConnection(configuration=config)

        if is_initiator:
            # Worker A initiates the DataChannel
            self.channel = self.pc.createDataChannel("situs-tensors", ordered=True)
            self._setup_channel(self.channel)

            offer = await self.pc.createOffer()
            await self.pc.setLocalDescription(offer)
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

    def _setup_channel(self, channel):
        @channel.on("open")
        def on_open():
            logger.info(f"Direct P2P WebRTC DataChannel OPEN between {self.worker_id} and {self.target_worker_id}! (Direct UDP mode active)")
            self.is_connected = True

        @channel.on("close")
        def on_close():
            logger.warning(f"P2P WebRTC DataChannel CLOSED between {self.worker_id} and {self.target_worker_id}")
            self.is_connected = False

        @channel.on("message")
        def on_message(message):
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

    def send_frame(self, data_str: str) -> bool:
        """Sends data over direct P2P UDP DataChannel if open."""
        if self.is_connected and self.channel and self.channel.readyState == "open":
            self.channel.send(data_str)
            return True
        return False

    async def close(self):
        if self.pc:
            await self.pc.close()
            self.is_connected = False
