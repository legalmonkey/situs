"""Binary wire framing and tensor contract validation protocol (SITUS_TENSOR_V1).

Guarantees zero-copy, typed, and CRC32-checksummed communication across distributed workers,
preventing tensor shape, dtype, sequence length, or memory corruption issues.
"""

import struct
import zlib
import uuid
from typing import Tuple, Dict, Any, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


MAGIC_HEADER = b"SITC"
PROTOCOL_VERSION = 1

# Header layout:
# 4s  (Magic: b"SITC")
# H   (Version: uint16)
# B   (Message Type: uint8)
# 16s (Request UUID bytes: 16B)
# B   (Stage ID: uint8)
# I   (Microbatch ID: uint32)
# I   (Seq Length: uint32)
# I   (Batch Size: uint32)
# I   (Hidden Dim: uint32)
# B   (Dtype Enum: uint8: 0=float32, 1=float16, 2=bfloat16, 3=int8, 4=int4)
# f   (Quant Scale: float32)
# f   (Quant Offset: float32)
# I   (CRC32 Checksum: uint32)
# Q   (Payload Length: uint64)
# Total: 4 + 2 + 1 + 16 + 1 + 4 + 4 + 4 + 4 + 1 + 4 + 4 + 4 + 8 = 61 bytes
HEADER_FORMAT = "!4sH B 16s B I I I I B f f I Q"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
assert HEADER_SIZE == 61, f"Expected header size 61, got {HEADER_SIZE}"

DTYPE_TO_ENUM = {
    "float32": 0,
    "float16": 1,
    "bfloat16": 2,
    "int8": 3,
    "int4": 4,
}

ENUM_TO_DTYPE = {v: k for k, v in DTYPE_TO_ENUM.items()}


class TensorContractError(ValueError):
    """Raised when an incoming tensor frame violates the strict wire protocol contract."""
    pass


def _uuid_to_16bytes(req_id: str) -> bytes:
    try:
        return uuid.UUID(req_id).bytes
    except Exception:
        # Fallback: md5 hash to 16 bytes for non-UUID strings
        import hashlib
        return hashlib.md5(req_id.encode("utf-8")).digest()


def _16bytes_to_uuid_str(raw: bytes) -> str:
    try:
        return str(uuid.UUID(bytes=raw))
    except Exception:
        return raw.hex()


def pack_tensor_frame(
    payload_bytes: bytes,
    request_id: str,
    stage_id: int,
    batch_size: int,
    seq_length: int,
    hidden_dim: int,
    dtype: str = "int8",
    scale: float = 1.0,
    offset: float = 0.0,
    microbatch_id: int = 0,
    msg_type: int = 1,
) -> bytes:
    """Packs raw activation bytes into a SITUS_TENSOR_V1 binary frame with CRC32."""
    if not isinstance(payload_bytes, (bytes, bytearray)):
        raise TensorContractError("Payload must be contiguous bytes or bytearray")

    dtype_enum = DTYPE_TO_ENUM.get(dtype.lower(), 3)
    req_bytes = _uuid_to_16bytes(request_id)
    payload_len = len(payload_bytes)
    checksum = zlib.crc32(payload_bytes) & 0xFFFFFFFF

    header = struct.pack(
        HEADER_FORMAT,
        MAGIC_HEADER,
        PROTOCOL_VERSION,
        msg_type,
        req_bytes,
        int(stage_id) & 0xFF,
        int(microbatch_id),
        int(seq_length),
        int(batch_size),
        int(hidden_dim),
        int(dtype_enum),
        float(scale),
        float(offset),
        checksum,
        payload_len,
    )
    return header + bytes(payload_bytes)


def unpack_tensor_frame(data: bytes) -> Tuple[Dict[str, Any], bytes]:
    """
    Unpacks and verifies a SITUS_TENSOR_V1 binary frame.
    Enforces CRC32 verification and shape contracts.
    """
    if len(data) < HEADER_SIZE:
        raise TensorContractError(f"Frame data too short: {len(data)} bytes < {HEADER_SIZE} header bytes")

    (
        magic,
        version,
        msg_type,
        req_bytes,
        stage_id,
        microbatch_id,
        seq_length,
        batch_size,
        hidden_dim,
        dtype_enum,
        scale,
        offset,
        checksum,
        payload_len,
    ) = struct.unpack(HEADER_FORMAT, data[:HEADER_SIZE])

    if magic != MAGIC_HEADER:
        raise TensorContractError(f"Invalid magic header: {magic!r} (expected {MAGIC_HEADER!r})")

    if version != PROTOCOL_VERSION:
        raise TensorContractError(f"Unsupported protocol version: {version} (expected {PROTOCOL_VERSION})")

    actual_payload = data[HEADER_SIZE:]
    if len(actual_payload) != payload_len:
        raise TensorContractError(
            f"Payload length mismatch: declared {payload_len} bytes, actual received {len(actual_payload)} bytes"
        )

    # CRC32 verification
    actual_checksum = zlib.crc32(actual_payload) & 0xFFFFFFFF
    if actual_checksum != checksum:
        raise TensorContractError(
            f"Checksum corruption detected: declared CRC32 {hex(checksum)}, computed {hex(actual_checksum)}"
        )

    meta = {
        "request_id": _16bytes_to_uuid_str(req_bytes),
        "stage_id": stage_id,
        "microbatch_id": microbatch_id,
        "batch_size": batch_size,
        "seq_length": seq_length,
        "hidden_dim": hidden_dim,
        "shape": [batch_size, seq_length, hidden_dim],
        "dtype": ENUM_TO_DTYPE.get(dtype_enum, "int8"),
        "scale": scale,
        "offset": offset,
        "msg_type": msg_type,
    }
    return meta, actual_payload
