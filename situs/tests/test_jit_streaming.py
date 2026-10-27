"""Unit and integration tests for Progressive Just-In-Time (JIT) Layer Streaming."""

import asyncio
import base64
import pytest
import tempfile
import pathlib

from situs.protocol.messages import (
    MessageType,
    AssignLayers,
    InferenceStart,
    ForwardActivation,
    NextToken,
    StreamLayerChunk,
    StageLayerAck,
)
from situs.worker.loader import DistributedStageModel
from situs.worker.engine import StageExecutionEngine
from situs.worker.jit_streamer import (
    ProgressiveJITStreamer,
    save_layer_to_cache,
    load_layer_from_cache,
    is_layer_cached,
    generate_mock_layer_bytes,
)


def test_jit_streamer_chunking_and_priority():
    streamer = ProgressiveJITStreamer(model_name="llama-test", chunk_size=32 * 1024)

    # Test chunk creation for layer 16 (mock layer default 128 KB -> 4 chunks)
    chunks = streamer.create_layer_chunks(layer_index=16, total_stage_layers=8)
    assert len(chunks) == 4
    assert chunks[0].chunk_index == 0
    assert not chunks[0].is_last_chunk
    assert chunks[-1].chunk_index == 3
    assert chunks[-1].is_last_chunk
    assert chunks[0].layer_index == 16

    # Test stream_stage_layers priority ordering: entrance layer (16) must be first!
    streamed = list(streamer.stream_stage_layers(layer_start=16, layer_end=20))
    # 4 layers * 4 chunks = 16 chunks
    assert len(streamed) == 16
    first_4 = streamed[:4]
    for c in first_4:
        assert c.layer_index == 16, "Entrance layer must be streamed first"

    second_4 = streamed[4:8]
    for c in second_4:
        assert c.layer_index == 17, "Subsequent layers stream sequentially after entrance"


@pytest.mark.asyncio
async def test_jit_async_streamer():
    streamer = ProgressiveJITStreamer(model_name="llama-test", chunk_size=64 * 1024)
    chunks = []
    async for chunk in streamer.stream_stage_layers_async(layer_start=8, layer_end=10):
        chunks.append(chunk)

    assert len(chunks) > 0
    assert chunks[0].layer_index == 8


def test_progressive_layer_reassembly_and_ack():
    # Setup worker B with progressive=True (layers 4 to 8)
    total_layers = 8
    model_b = DistributedStageModel(
        layer_start=4,
        layer_end=8,
        total_layers=total_layers,
        is_first_stage=False,
        is_last_stage=True,
        hidden_dim=128,
        vocab_size=500,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="llama",
        device="cpu",
        progressive=True,
    )
    assign_b = AssignLayers(
        worker_id="worker-b",
        model_name="test-model",
        layer_start=4,
        layer_end=8,
        total_layers=total_layers,
        is_first_stage=False,
        is_last_stage=True,
        progressive=True,
    )
    engine_b = StageExecutionEngine(stage_model=model_b, assignment=assign_b)

    # Initially, no layers are ready
    assert model_b.get_ready_count() == 0
    assert not model_b.is_entrance_ready()

    # Stream chunks for Layer 4 (entrance)
    streamer = ProgressiveJITStreamer(model_name="test-model", chunk_size=32 * 1024)
    layer_4_chunks = streamer.create_layer_chunks(layer_index=4, total_stage_layers=4)

    # Process all intermediate chunks (should return None)
    for c in layer_4_chunks[:-1]:
        ack_frame = engine_b.process_layer_chunk(c, source_worker="worker-a")
        assert ack_frame is None
        assert not model_b.is_entrance_ready()

    # Process final chunk
    ack_frame = engine_b.process_layer_chunk(layer_4_chunks[-1], source_worker="worker-a")
    assert ack_frame is not None
    assert ack_frame.type == MessageType.STAGE_LAYER_ACK
    assert ack_frame.target == "worker-a"

    ack_payload = StageLayerAck.model_validate(ack_frame.payload)
    assert ack_payload.layer_index == 4
    assert ack_payload.status == "ready"
    assert ack_payload.ready_layer_count == 1
    assert model_b.is_entrance_ready()
    assert model_b.wait_layer_ready(4, timeout=0.1)


def test_progressive_pipelined_execution():
    """
    Simulates real-world Wi-Fi JIT: Worker B begins execution as soon as Layer 4 arrives,
    while layers 5, 6, 7 arrive concurrently during execution.
    """
    total_layers = 8
    hidden_dim = 128
    vocab_size = 500

    model_a = DistributedStageModel(
        layer_start=0,
        layer_end=4,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="llama",
        device="cpu",
    )
    model_b = DistributedStageModel(
        layer_start=4,
        layer_end=8,
        total_layers=total_layers,
        is_first_stage=False,
        is_last_stage=True,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="llama",
        device="cpu",
        progressive=True,
    )

    assign_a = AssignLayers(
        worker_id="worker-a",
        model_name="test-model",
        layer_start=0,
        layer_end=4,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        next_worker_id="worker-b",
    )
    assign_b = AssignLayers(
        worker_id="worker-b",
        model_name="test-model",
        layer_start=4,
        layer_end=8,
        total_layers=total_layers,
        is_first_stage=False,
        is_last_stage=True,
        prev_worker_id="worker-a",
        progressive=True,
    )

    engine_a = StageExecutionEngine(stage_model=model_a, assignment=assign_a)
    engine_b = StageExecutionEngine(stage_model=model_b, assignment=assign_b)

    # Streamer creates chunks for layers 4..8
    streamer = ProgressiveJITStreamer(model_name="test-model", chunk_size=64 * 1024)

    # 1. Entrance layer (4) arrives at Worker B
    for c in streamer.create_layer_chunks(layer_index=4, total_stage_layers=4):
        engine_b.process_layer_chunk(c, source_worker="worker-a")

    assert model_b.is_entrance_ready()
    assert model_b.get_ready_count() == 1

    # 2. Worker A computes prompt forward pass
    request_id = "jit-test-01"
    start_msg = InferenceStart(
        request_id=request_id,
        model_name="test-model",
        input_ids=[10, 20, 30],
        max_new_tokens=3,
        temperature=0.0,
    )
    frame_a = engine_a.start_inference(start_msg)
    engine_b.session_configs[request_id] = start_msg

    # 3. Simulate background network streaming: remaining layers (5, 6, 7) stream in
    import threading
    def deliver_remaining_layers():
        import time
        for l_idx in [5, 6, 7]:
            time.sleep(0.01)  # small network delay
            for c in streamer.create_layer_chunks(layer_index=l_idx, total_stage_layers=4):
                engine_b.process_layer_chunk(c, source_worker="worker-a")

    bg_thread = threading.Thread(target=deliver_remaining_layers)
    bg_thread.start()

    # 4. Worker B receives forward activation and executes
    # Because Layer 4 is ready, it starts immediately; as layers 5, 6, 7 arrive,
    # wait_layer_ready unblocks without stalling the whole model load upfront.
    act_payload = ForwardActivation.model_validate(frame_a.payload)
    out_frame = engine_b.process_forward_activation(act_payload)

    bg_thread.join()

    assert out_frame is not None
    assert out_frame.type == MessageType.NEXT_TOKEN
    assert model_b.get_ready_count() == 4


def test_layer_cache_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = pathlib.Path(tmpdir)
        model = "cache-test-model"
        layer = 3
        data = b"serialized_weights_data_block_test"

        assert not is_layer_cached(model, layer, cache_dir=cache_path)
        saved_file = save_layer_to_cache(model, layer, data, cache_dir=cache_path)
        assert saved_file.exists()
        assert is_layer_cached(model, layer, cache_dir=cache_path)

        loaded = load_layer_from_cache(model, layer, cache_dir=cache_path)
        assert loaded == data


def test_binary_chunk_packet_framing():
    """Verifies Strategy 4 zero-overhead binary packet serialization."""
    from situs.protocol.compression import (
        encode_binary_chunk_packet,
        decode_binary_chunk_packet,
        BINARY_CHUNK_MAGIC,
    )

    raw_payload = b"weight_bytes_block_sample_12345"
    packet = encode_binary_chunk_packet(
        model_name="qwen2.5:7b",
        layer_index=16,
        chunk_index=2,
        total_chunks=8,
        total_stage_layers=16,
        scale=1.25,
        is_last=False,
        codec="zstd",
        raw_payload=raw_payload,
    )

    assert packet.startswith(BINARY_CHUNK_MAGIC)
    meta, extracted_payload = decode_binary_chunk_packet(packet)
    assert meta["model_name"] == "qwen2.5:7b"
    assert meta["layer_index"] == 16
    assert meta["chunk_index"] == 2
    assert meta["total_chunks"] == 8
    assert meta["codec"] == "zstd"
    assert extracted_payload == raw_payload


def test_layer_wire_compression_zstd_and_zlib():
    """Verifies Strategy 4 lossless wire compression on layer chunks."""
    from situs.protocol.compression import compress_layer_bytes, decompress_layer_bytes

    original = b"A" * 1024 + b"B" * 2048 + b"C" * 4096
    # Test zstd (or fallback to zlib)
    compressed = compress_layer_bytes(original, codec="zstd")
    assert len(compressed) < len(original)
    decompressed = decompress_layer_bytes(compressed, codec="zstd")
    assert decompressed == original

    # Test zlib
    compressed_zlib = compress_layer_bytes(original, codec="zlib")
    assert len(compressed_zlib) < len(original)
    decompressed_zlib = decompress_layer_bytes(compressed_zlib, codec="zlib")
    assert decompressed_zlib == original


def test_ternary_packing_and_unpacking():
    """Verifies Strategy 4 BitNet 1.6 bpw ternary weight packing."""
    import numpy as np
    from situs.protocol.compression import pack_ternary_weights, unpack_ternary_weights

    # Generate test ternary weights in {-1, 0, 1}
    weights = np.array([-1.0, 0.0, 1.0, 1.0, -1.0, 0.0, 0.0, -1.0, 1.0, 0.0], dtype=np.float32)
    packed_bytes, scale, shape = pack_ternary_weights(weights)
    # 10 weights pack into 2 bytes!
    assert len(packed_bytes) == 2
    unpacked = unpack_ternary_weights(packed_bytes, scale, shape)
    np.testing.assert_allclose(unpacked, weights, atol=1e-5)


def test_progressive_streaming_and_hotswap_without_fast_boot():
    """
    Verifies Progressive Streaming: Workers initialize with genuine RealTransformerBlock layers.
    Before streamed weights arrive, unready layers block/wait. As layer chunks stream in
    and pass validation, layers become ready and execute genuine forward inference.
    """
    import threading
    import time
    from situs.worker.loader import RealTransformerBlock

    total_layers = 8
    hidden_dim = 128
    vocab_size = 500

    # Progressive workers without fast boot
    model_a = DistributedStageModel(
        layer_start=0,
        layer_end=4,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="llama",
        device="cpu",
        progressive=True,
    )
    model_b = DistributedStageModel(
        layer_start=4,
        layer_end=8,
        total_layers=total_layers,
        is_first_stage=False,
        is_last_stage=True,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_heads=4,
        num_kv_heads=2,
        intermediate_size=256,
        head_dim=32,
        architecture="llama",
        device="cpu",
        progressive=True,
    )

    assign_a = AssignLayers(
        worker_id="worker-a",
        model_name="prog-model",
        layer_start=0,
        layer_end=4,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        next_worker_id="worker-b",
        progressive=True,
    )
    assign_b = AssignLayers(
        worker_id="worker-b",
        model_name="prog-model",
        layer_start=4,
        layer_end=8,
        total_layers=total_layers,
        is_first_stage=False,
        is_last_stage=True,
        prev_worker_id="worker-a",
        progressive=True,
    )

    engine_a = StageExecutionEngine(stage_model=model_a, assignment=assign_a)
    engine_b = StageExecutionEngine(stage_model=model_b, assignment=assign_b)

    # 1. Verify all layers are genuine RealTransformerBlock and NOT unverified draft blocks
    assert isinstance(model_a.layers[0], RealTransformerBlock)
    assert isinstance(model_b.layers[0], RealTransformerBlock)
    assert not hasattr(model_a, "fast_boot_active")
    assert not hasattr(model_b, "fast_boot_active")
    assert not model_a.is_entrance_ready()
    assert not model_b.is_entrance_ready()

    # 2. Mark stage A layers ready with mock weights
    for l_idx in range(0, 4):
        raw = generate_mock_layer_bytes(layer_idx=l_idx, hidden_dim=hidden_dim)
        model_a.mark_layer_ready(l_idx, weights=raw)

    assert model_a.is_entrance_ready()
    assert model_a.get_ready_count() == 4

    # 3. Simulate background network streaming for Stage B
    streamer_b = ProgressiveJITStreamer(model_name="prog-model", chunk_size=32 * 1024, codec="zstd", enable_wire_compression=True)
    for l_idx in range(4, 8):
        raw = generate_mock_layer_bytes(layer_idx=l_idx, hidden_dim=hidden_dim)
        save_layer_to_cache("prog-model", l_idx, raw)
        chunks = streamer_b.create_layer_chunks(layer_index=l_idx, total_stage_layers=4)
        for c in chunks:
            engine_b.process_layer_chunk(c, source_worker="worker-a")

    assert model_b.is_entrance_ready()
    assert model_b.get_ready_count() == 4

    # 4. Execute forward inference across both stages with genuine weights
    req_id = "prog-req-1"
    start_msg = InferenceStart(
        request_id=req_id,
        model_name="prog-model",
        input_ids=[10, 20, 30],
        max_new_tokens=2,
        temperature=0.0,
    )
    engine_b.session_configs[req_id] = start_msg
    frame_a = engine_a.start_inference(start_msg)
    assert frame_a.type == MessageType.FORWARD_ACTIVATION

    act_payload = ForwardActivation.model_validate(frame_a.payload)
    assert act_payload.is_verified is True
    assert act_payload.is_experimental is False

    out_frame = engine_b.process_forward_activation(act_payload)
    assert out_frame is not None
    assert out_frame.type == MessageType.NEXT_TOKEN
    tok_payload = NextToken.model_validate(out_frame.payload)
    assert tok_payload.is_verified is True
    assert tok_payload.is_experimental is False
