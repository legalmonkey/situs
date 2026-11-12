"""Regression tests for multi-node wireless/WAN pipeline execution, Ollama GGUF loading, and WebRTC fallback."""

import asyncio
import json
import socket
import struct
import time
from unittest.mock import MagicMock

import httpx
import numpy as np
import pytest
import torch

from situs.network.p2p import P2PDataChannelNode, SAFE_SCTP_MAX_BYTES
from situs.protocol.compression import compress_tensor
from situs.protocol.messages import AssignLayers, ForwardActivation, InferenceStart, MessageType
from situs.worker.engine import StageExecutionEngine
from situs.worker.loader import (
    DistributedStageModel,
    FastGGUFFile,
    Int8WeightEmbedding,
    Int8WeightLinear,
    RealTransformerBlock,
    _dequantize_gguf_raw_chunk,
    find_local_ollama_gguf_file,
    find_local_safetensors_files,
    find_local_weight_files,
    load_hf_stage_model,
)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_p2p_send_frame_guards_oversized_sctp_and_exceptions():
    """Prefill frames exceeding safe SCTP limits or failing DataChannel send must return False for WS fallback."""
    node = P2PDataChannelNode("w1", "w2", MagicMock(), MagicMock())
    node.is_connected = True
    node.channel = MagicMock()
    node.channel.readyState = "open"

    # Small frame uses direct DataChannel
    assert node.send_frame('{"ok": true}') is True
    assert node.channel.send.call_count == 1

    # Oversized prefill frame triggers clean WS relay fallback without touching channel.send
    oversized = "x" * (SAFE_SCTP_MAX_BYTES + 100)
    assert node.send_frame(oversized) is False
    assert node.channel.send.call_count == 1

    # Transport exception triggers clean WS relay fallback and marks channel disconnected
    node.channel.send.side_effect = RuntimeError("SCTP buffer full")
    assert node.send_frame('{"small": 1}') is False
    assert node.is_connected is False


def test_stage_engine_decompresses_activations_to_stage_dtype():
    """Stage 2 running in float16 must decompress incoming int8 activations to float16 (not hardcoded float32)."""
    torch.manual_seed(7)
    h, heads, kv, inter, hd, vocab = 32, 4, 2, 64, 8, 64
    stage2 = DistributedStageModel(
        layer_start=1,
        layer_end=2,
        total_layers=2,
        is_first_stage=False,
        is_last_stage=True,
        hidden_dim=h,
        vocab_size=vocab,
        num_heads=heads,
        num_kv_heads=kv,
        intermediate_size=inter,
        head_dim=hd,
        architecture="llama",
        device="cpu",
        dtype=torch.float16,
    )
    assign2 = AssignLayers(
        worker_id="w2",
        model_name="test-fp16",
        layer_start=1,
        layer_end=2,
        total_layers=2,
        is_first_stage=False,
        is_last_stage=True,
        prev_worker_id="w1",
        first_worker_id="w1",
    )
    engine2 = StageExecutionEngine(stage2, assign2, compression_dtype="int8", tokenizer=None)
    engine2.session_configs["req-fp16"] = InferenceStart(
        request_id="req-fp16",
        model_name="test-fp16",
        input_ids=[1, 2],
        prompt="hi",
        max_new_tokens=4,
    )

    import base64
    hidden = torch.randn(1, 2, h, dtype=torch.float16)
    comp, scale, shape = compress_tensor(hidden, method="int8")
    act = ForwardActivation(
        request_id="req-fp16",
        step_index=0,
        shape=list(shape),
        dtype="int8",
        scale=scale,
        payload_b64=base64.b64encode(comp).decode("ascii"),
        stage_index=0,
        expected_next_stage=1,
    )
    out_frame = engine2.process_forward_activation(act)
    assert out_frame is not None
    assert out_frame.type == MessageType.NEXT_TOKEN


def test_ollama_gguf_discovery_and_fast_dequant_and_int8_parity(tmp_path, monkeypatch):
    """Verify Ollama manifest -> GGUF blob discovery, vectorized Q8_0/Q4_K/Q6_K dequant parity, and Int8 layers."""
    import gguf

    # 1. Mock ~/.ollama/models layout
    ollama_dir = tmp_path / "ollama_models"
    manifest = ollama_dir / "manifests" / "registry.ollama.ai" / "library" / "gemma3" / "4b"
    manifest.parent.mkdir(parents=True)
    blob = ollama_dir / "blobs" / "sha256-deadbeef1234"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"GGUF\x03\x00\x00\x00" + b"\x00" * 64)
    manifest.write_text(
        json.dumps({
            "layers": [{"mediaType": "application/vnd.ollama.image.model", "digest": "sha256:deadbeef1234"}]
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("OLLAMA_MODELS", str(ollama_dir))

    assert find_local_safetensors_files("gemma3:4b") == []
    assert find_local_ollama_gguf_file("gemma3:4b") == str(blob)
    assert find_local_weight_files("gemma3:4b") == [str(blob)]

    # 2. Vectorized Q8_0, Q4_K, Q6_K parity vs reference gguf.quants
    rng = np.random.default_rng(123)
    w_src = rng.standard_normal((16, 256)).astype(np.float32)
    q8_blocks = gguf.quants.quantize(w_src, gguf.GGMLQuantizationType.Q8_0)
    ref_q8 = torch.from_numpy(gguf.quants.dequantize(q8_blocks, gguf.GGMLQuantizationType.Q8_0).copy())
    got_q8 = _dequantize_gguf_raw_chunk(q8_blocks, int(gguf.GGMLQuantizationType.Q8_0), 16, "cpu", torch.float32)
    assert (ref_q8 - got_q8).abs().max().item() < 1e-6

    # Synthesize realistic Q4_K (144B/block) and Q6_K (210B/block) byte blocks with finite fp16 scales
    q4k_blocks = rng.integers(0, 256, size=(16, 144), dtype=np.uint8)
    q4k_blocks[:, 0:4] = np.array([0.05, -0.02], dtype=np.float16).view(np.uint8)
    ref_q4k = torch.from_numpy(gguf.quants.dequantize(q4k_blocks, gguf.GGMLQuantizationType.Q4_K).copy())
    got_q4k = _dequantize_gguf_raw_chunk(q4k_blocks, int(gguf.GGMLQuantizationType.Q4_K), 16, "cpu", torch.float32)
    assert (ref_q4k - got_q4k).abs().max().item() < 1e-6

    q6k_blocks = rng.integers(0, 256, size=(16, 210), dtype=np.uint8)
    q6k_blocks[:, 208:210] = np.array([0.03], dtype=np.float16).view(np.uint8)
    ref_q6k = torch.from_numpy(gguf.quants.dequantize(q6k_blocks, gguf.GGMLQuantizationType.Q6_K).copy())
    got_q6k = _dequantize_gguf_raw_chunk(q6k_blocks, int(gguf.GGMLQuantizationType.Q6_K), 16, "cpu", torch.float32)
    assert (ref_q6k - got_q6k).abs().max().item() < 1e-6

    # 3. Real FastGGUFFile roundtrip using gguf.GGUFWriter
    gguf_file = tmp_path / "tiny_test.gguf"
    writer = gguf.GGUFWriter(str(gguf_file), "llama")
    writer.add_uint32("tokenizer.ggml.bos_token_id", 2)
    writer.add_tensor("test_w", q8_blocks, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    fast = FastGGUFFile(str(gguf_file), wanted_kv={"tokenizer.ggml.bos_token_id"})
    try:
        assert fast.kv["tokenizer.ggml.bos_token_id"] == 2
        assert len(fast.tensors) == 1
        t_entry = fast.tensors[0]
        assert t_entry.name == "test_w"
        loaded = _dequantize_gguf_raw_chunk(t_entry.data, t_entry.tensor_type, 16, "cpu", torch.float32)
        assert (loaded - ref_q8).abs().max().item() < 1e-6
    finally:
        fast.close()

    # 4. Int8WeightLinear / Int8WeightEmbedding close numerical parity
    t_w = torch.randn(32, 64) * 0.1
    q_lin = Int8WeightLinear.from_float(t_w)
    x = torch.randn(1, 3, 64)
    ref_out = torch.nn.functional.linear(x, t_w)
    got_out = q_lin(x)
    assert torch.allclose(got_out, ref_out, atol=3e-2)
    assert torch.nn.functional.cosine_similarity(got_out.flatten(), ref_out.flatten(), dim=0) > 0.9999

    q_emb = Int8WeightEmbedding.from_float_chunked([t_w[:16], t_w[16:]], 32, 64, "cpu", torch.float32)
    assert torch.allclose(q_emb(torch.tensor([[1, 20]])), t_w[[1, 20]], atol=1e-2)


def test_live_socket_error_fast_fail_and_two_worker_pipeline(tmp_path, monkeypatch):
    """
    Rule 3 live socket test:
    1) Worker with unloaded engine fails fast via MessageType.ERROR (no silent hang).
    2) Two live workers connected over real WebSockets run 2-stage distributed inference end-to-end.
    """
    import safetensors.torch
    import situs.coordinator.app as app_mod
    import situs.coordinator.registry as reg_mod
    from situs.coordinator.registry import WorkerRegistry
    from situs.coordinator.relay import RelayHub
    from situs.tui import start_in_process_coordinator, start_in_process_worker

    # Isolate coordinator state cleanly without mutating other tests' port-8000 workers
    iso_reg = WorkerRegistry()
    iso_relay = RelayHub()
    iso_relay.set_registry(iso_reg)
    monkeypatch.setattr(app_mod, "registry", iso_reg)
    monkeypatch.setattr(app_mod, "relay", iso_relay)

    # Tiny verified 2-layer Llama safetensors model + fake fast tokenizer
    h, heads, kv, inter, hd, vocab = 32, 4, 2, 64, 8, 64
    model_dir = tmp_path / "tiny_2layer_llama"
    model_dir.mkdir()

    def _make_layer(seed):
        torch.manual_seed(seed)
        blk = RealTransformerBlock(h, heads, kv, inter, hd, architecture="llama")
        return {k: v.clone() for k, v in blk.state_dict().items()}

    tensors = {
        "model.embed_tokens.weight": torch.randn(vocab, h),
        "model.norm.weight": torch.ones(h),
        "lm_head.weight": torch.randn(vocab, h),
    }
    for l_idx in range(2):
        for k, v in _make_layer(100 + l_idx).items():
            tensors[f"model.layers.{l_idx}.{k}"] = v
    safetensors.torch.save_file(tensors, str(model_dir / "model.safetensors"))

    class _FakeTok:
        eos_token_id = 2
        bos_token_id = 1
        stop_token_ids = [2]

        def encode(self, text, add_special_tokens=True):
            return [1, 5, 6]

        def decode(self, ids, skip_special_tokens=False):
            return " hi" if ids and ids[0] != 2 else ""

    fake_tok = _FakeTok()
    tiny_meta = {
        "total_layers": 2,
        "hidden_dim": h,
        "intermediate_size": inter,
        "vocab_size": vocab,
        "num_heads": heads,
        "num_kv_heads": kv,
        "head_dim": hd,
        "architecture": "llama",
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "context_length": 2048,
        "max_position_embeddings": 2048,
        "torch_dtype": "float32",
    }

    import situs.tokenizer as tok_mod
    import situs.worker.loader as loader_mod
    import situs.worker.client as client_mod
    import situs.inference_backend as ib_mod

    monkeypatch.setattr(tok_mod, "get_or_load_tokenizer", lambda *a, **kw: fake_tok)
    monkeypatch.setattr(app_mod, "get_or_load_tokenizer", lambda *a, **kw: fake_tok)
    monkeypatch.setattr(reg_mod, "resolve_model_metadata", lambda m: tiny_meta if "tiny_2layer" in str(m) else tiny_meta)
    monkeypatch.setattr(app_mod, "resolve_model_metadata", lambda m: tiny_meta)
    monkeypatch.setattr(loader_mod, "resolve_model_metadata", lambda m: tiny_meta)
    monkeypatch.setattr(ib_mod, "is_model_downloaded", lambda m: True)

    port = _find_free_port()
    ok_c, msg_c = start_in_process_coordinator(port=port)
    assert ok_c is True, f"Coordinator failed to start: {msg_c}"

    # Part A: Two live workers with NO local weights for 'ghost-unloaded-model' must return explicit ERROR immediately
    app_mod.registry.active_model = "ghost-unloaded-model"
    monkeypatch.setattr(loader_mod, "find_local_weight_files", lambda m: [])

    ok_g1, _ = start_in_process_worker(f"127.0.0.1:{port}", worker_id="w-ghost-1", device="cpu")
    ok_g2, _ = start_in_process_worker(f"127.0.0.1:{port}", worker_id="w-ghost-2", device="cpu")
    assert ok_g1 and ok_g2
    time.sleep(0.3)

    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": "ghost-unloaded-model", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3, "stream": True},
        ) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
            assert '"error"' in body, f"Expected immediate worker error frame, got: {body}"

    # Disconnect ghost workers cleanly via admin removal
    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=5.0) as client:
        for w_id in ("w-ghost-1", "w-ghost-2"):
            client.delete(f"/admin/workers/{w_id}")
    time.sleep(0.2)

    # Part B: Two live workers WITH real weights over live WebSockets produce real multi-stage streamed tokens
    model_str = str(model_dir)
    app_mod.registry.active_model = model_str
    monkeypatch.setattr(loader_mod, "find_local_weight_files", lambda m: [str(model_dir / "model.safetensors")])

    ok_w1, msg1 = start_in_process_worker(f"127.0.0.1:{port}", worker_id="w-stage-1", device="cpu")
    ok_w2, msg2 = start_in_process_worker(f"127.0.0.1:{port}", worker_id="w-stage-2", device="cpu")
    assert ok_w1 and ok_w2, f"Workers failed: {msg1}, {msg2}"

    # Wait briefly for async ASSIGN_LAYERS engine init on both workers
    time.sleep(0.8)
    assert app_mod.registry.workers["w-stage-1"].assigned_layer_start is not None
    assert app_mod.registry.workers["w-stage-2"].assigned_layer_start is not None

    with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=20.0) as client:
        with client.stream(
            "POST",
            "/v1/chat/completions",
            json={"model": model_str, "messages": [{"role": "user", "content": "hello"}], "max_tokens": 3, "temperature": 0.0, "stream": True},
        ) as resp:
            assert resp.status_code == 200
            lines = list(resp.iter_lines())
            content_chunks = [l for l in lines if '"content"' in l]
            assert len(content_chunks) >= 1, f"Expected streamed tokens from 2-stage pipeline, got lines: {lines}"
            assert any("[DONE]" in l for l in lines)
        for w_id in ("w-stage-1", "w-stage-2"):
            client.delete(f"/admin/workers/{w_id}")


def test_speculative_rejection_rolls_back_stage_kv_caches_and_truncates_stop():
    """Partial draft rejections must roll back KV caches on both stages; mid-sequence stop tokens must truncate."""
    from situs.protocol.messages import SpeculativeCandidates, SpeculativeVerified

    def _tiny_model(layer_start: int, layer_end: int, is_first: bool, is_last: bool) -> DistributedStageModel:
        return DistributedStageModel(
            layer_start=layer_start,
            layer_end=layer_end,
            total_layers=2,
            is_first_stage=is_first,
            is_last_stage=is_last,
            hidden_dim=16,
            vocab_size=128,
            num_heads=2,
            num_kv_heads=2,
            intermediate_size=32,
            head_dim=8,
            architecture="llama",
            device="cpu",
            dtype=torch.float32,
        )

    class _StubTokenizer:
        eos_token_id = 106
        stop_token_ids = [106]

        def decode(self, ids, skip_special_tokens=False):
            return "".join("<end_of_turn>" if i == 106 else f"t{i}" for i in ids)

    assign_a = AssignLayers(
        worker_id="w1",
        model_name="tiny",
        layer_start=0,
        layer_end=1,
        total_layers=2,
        is_first_stage=True,
        is_last_stage=False,
        next_worker_id="w2",
        first_worker_id="w1",
        codec="none",
    )
    assign_b = AssignLayers(
        worker_id="w2",
        model_name="tiny",
        layer_start=1,
        layer_end=2,
        total_layers=2,
        is_first_stage=False,
        is_last_stage=True,
        prev_worker_id="w1",
        first_worker_id="w1",
        codec="none",
    )
    eng_a = StageExecutionEngine(_tiny_model(0, 1, True, False), assign_a, compression_dtype="none", tokenizer=_StubTokenizer())
    eng_b = StageExecutionEngine(_tiny_model(1, 2, False, True), assign_b, compression_dtype="none", tokenizer=_StubTokenizer())

    req_id = "req-spec-rollback"
    start_cfg = InferenceStart(request_id=req_id, model_name="tiny", input_ids=[1, 2, 3], max_new_tokens=32, temperature=0.7, stop_token_ids=[106])
    eng_a.session_configs[req_id] = start_cfg
    eng_b.session_configs[req_id] = start_cfg
    eng_a.session_token_histories[req_id] = [1, 2, 3]

    # Seed both stages with 3 prompt tokens + 3 drafted candidate positions (seq_len == 6)
    seed_kvs_a = [(torch.zeros(1, 2, 6, 8), torch.zeros(1, 2, 6, 8))]
    eng_a.kv_cache_manager.set_all_kvs(req_id, seed_kvs_a)
    eng_a.session_kv_caches[req_id] = seed_kvs_a

    seed_kvs_b = [(torch.zeros(1, 2, 3, 8), torch.zeros(1, 2, 3, 8))]
    eng_b.kv_cache_manager.set_all_kvs(req_id, seed_kvs_b)
    eng_b.session_kv_caches[req_id] = seed_kvs_b

    # Force Stage 2 verifier to accept only 1 of 3 drafts + 1 correction -> rejected_draft_count == 2
    eng_b.verifier.verify_candidates = MagicMock(return_value=([10], 25))
    hidden = torch.randn(1, 3, 16, dtype=torch.float32)
    comp_bytes, scale, shape = compress_tensor(hidden, method="none")
    import base64
    cand_frame = SpeculativeCandidates(
        request_id=req_id,
        step_index=0,
        candidate_tokens=[10, 11, 12],
        shape=list(shape),
        dtype="none",
        scale=scale,
        payload_b64=base64.b64encode(comp_bytes).decode("ascii"),
    )
    verified_proto = eng_b.process_speculative_candidates(cand_frame)
    assert verified_proto is not None
    verified = SpeculativeVerified(**verified_proto.payload)
    assert verified.rejected_draft_count == 2
    assert verified.accepted_tokens == [10, 25]
    # Stage 2 forward added 3 tokens (3 -> 6) then trimmed 2 rejected drafts back to 4
    assert eng_b.session_kv_caches[req_id][0][0].shape[-2] == 4
    assert eng_b.kv_cache_manager.get_all_kvs(req_id)[0][0].shape[-2] == 4

    # Stage 1 processes SpeculativeVerified: trims its 6-token cache by 2 (-> 4), then forwards 1 token (+1 -> 5)
    next_proto = eng_a.process_speculative_verified(verified)
    assert next_proto is not None
    assert eng_a.session_kv_caches[req_id][0][0].shape[-2] == 5
    assert eng_a.kv_cache_manager.get_all_kvs(req_id)[0][0].shape[-2] == 5

    # Now verify mid-sequence stop token truncation & empty-string suppression in process_speculative_candidates
    req_eos = "req-spec-eos"
    eng_b.session_configs[req_eos] = InferenceStart(request_id=req_eos, model_name="tiny", input_ids=[1, 2], max_new_tokens=32, temperature=0.7, stop_token_ids=[106])
    eng_b.kv_cache_manager.set_all_kvs(req_eos, [(torch.zeros(1, 2, 2, 8), torch.zeros(1, 2, 2, 8))])
    eng_b.verifier.verify_candidates = MagicMock(return_value=([10, 106], 12))
    cand_eos = SpeculativeCandidates(
        request_id=req_eos,
        step_index=0,
        candidate_tokens=[10, 106, 11],
        shape=list(shape),
        dtype="none",
        scale=scale,
        payload_b64=base64.b64encode(comp_bytes).decode("ascii"),
    )
    eos_proto = eng_b.process_speculative_candidates(cand_eos)
    eos_verified = SpeculativeVerified(**eos_proto.payload)
    assert eos_verified.accepted_tokens == [10, 106]
    assert eos_verified.accepted_texts == ["t10", ""]
    assert eos_verified.is_finished is True
    assert eos_verified.finish_reason == "stop"
    assert req_eos in eng_b.finished_sessions


def test_coordinator_sse_suppresses_redecoding_empty_stop_token_text(monkeypatch):
    """Coordinator SSE stream must not re-decode a suppressed stop token when token_text is empty."""
    import threading
    import uvicorn
    import situs.coordinator.app as app_mod
    from situs.coordinator.registry import WorkerRegistry
    from situs.coordinator.relay import RelayHub
    from situs.protocol.messages import NextToken, ProtocolFrame

    fresh_registry = WorkerRegistry()
    fresh_relay = RelayHub()
    fresh_relay.set_registry(fresh_registry)
    monkeypatch.setattr(app_mod, "registry", fresh_registry)
    monkeypatch.setattr(app_mod, "relay", fresh_relay)

    class _MockStopTokenizer:
        eos_token_id = 106
        stop_token_ids = [106]

        def encode(self, text, add_special_tokens=False):
            return [1, 2]

        def decode(self, ids, skip_special_tokens=False):
            return "".join("<end_of_turn>" if i == 106 else "Hello" for i in ids)

    monkeypatch.setattr("situs.tokenizer.get_or_load_tokenizer", lambda *a, **k: _MockStopTokenizer())
    monkeypatch.setattr(app_mod, "get_or_load_tokenizer", lambda *a, **k: _MockStopTokenizer())
    monkeypatch.setattr("situs.inference_backend.is_model_downloaded", lambda *a, **k: True)

    from situs.protocol.messages import WorkerRegister
    n1 = fresh_registry.register_worker(
        WorkerRegister(
            worker_id="w-sse-1",
            gpu_name="Mock GPU 1",
            total_vram_bytes=8 * 1024**3,
            free_vram_bytes=7 * 1024**3,
            used_vram_bytes=1 * 1024**3,
            device_type="cuda",
        )
    )
    n1.assigned_layer_start = 0
    n1.assigned_layer_end = 17
    n1.is_first_stage = True
    n1.is_last_stage = False
    n1.next_worker_id = "w-sse-2"
    n1.first_worker_id = "w-sse-1"

    n2 = fresh_registry.register_worker(
        WorkerRegister(
            worker_id="w-sse-2",
            gpu_name="Mock GPU 2",
            total_vram_bytes=8 * 1024**3,
            free_vram_bytes=7 * 1024**3,
            used_vram_bytes=1 * 1024**3,
            device_type="cuda",
        )
    )
    n2.assigned_layer_start = 17
    n2.assigned_layer_end = 34
    n2.is_first_stage = False
    n2.is_last_stage = True
    n2.prev_worker_id = "w-sse-1"
    n2.first_worker_id = "w-sse-1"

    fresh_registry.active_model = "gemma3:4b"
    fresh_registry.total_layers = 34
    fresh_relay.active_connections["w-sse-1"] = MagicMock()
    fresh_relay.active_connections["w-sse-2"] = MagicMock()

    async def _fake_send_to_worker(worker_id: str, frame: ProtocolFrame):
        if frame.type == MessageType.INFERENCE_START:
            req_id = frame.payload["request_id"]
            q = fresh_relay.response_streams.get(req_id)
            if q is not None:
                await q.put(
                    ProtocolFrame(
                        type=MessageType.NEXT_TOKEN,
                        source="w-sse-2",
                        target="coordinator",
                        payload=NextToken(request_id=req_id, step_index=0, token_id=5, token_text="Hello", is_finished=False).model_dump(),
                    )
                )
                await q.put(
                    ProtocolFrame(
                        type=MessageType.NEXT_TOKEN,
                        source="w-sse-2",
                        target="coordinator",
                        payload=NextToken(request_id=req_id, step_index=1, token_id=106, token_text="", is_finished=True, finish_reason="stop").model_dump(),
                    )
                )
        return True

    monkeypatch.setattr(fresh_relay, "send_to_worker", _fake_send_to_worker)

    port = _find_free_port()
    server = uvicorn.Server(uvicorn.Config(app_mod.app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    try:
        deadline = time.time() + 5.0
        while not server.started and time.time() < deadline:
            time.sleep(0.05)

        with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=10.0) as client:
            with client.stream(
                "POST",
                "/v1/chat/completions",
                json={"model": "gemma3:4b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "stream": True},
            ) as resp:
                assert resp.status_code == 200
                body = "\n".join(resp.iter_lines())
                assert "Hello" in body
                assert "<end_of_turn>" not in body
                assert '"finish_reason": "stop"' in body or '"finish_reason":"stop"' in body
                assert "[DONE]" in body

            non_stream = client.post(
                "/v1/chat/completions",
                json={"model": "gemma3:4b", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "stream": False},
            )
            assert non_stream.status_code == 200
            data = non_stream.json()
            assert data["choices"][0]["message"]["content"] == "Hello"
            assert data["choices"][0]["finish_reason"] == "stop"
    finally:
        server.should_exit = True
        t.join(timeout=3.0)


def test_ws_max_size_64mib_configured():
    """Worker and coordinator entrypoints must raise WebSocket max frame size to 64 MiB for large prefill activations."""
    import inspect
    import situs.cli as cli_mod
    import situs.coordinator.app as app_mod
    import situs.tui as tui_mod
    import situs.worker.client as client_mod

    assert "64 * 1024 * 1024" in inspect.getsource(client_mod)
    assert "64 * 1024 * 1024" in inspect.getsource(app_mod)
    assert "64 * 1024 * 1024" in inspect.getsource(tui_mod)
    assert "64 * 1024 * 1024" in inspect.getsource(cli_mod)

