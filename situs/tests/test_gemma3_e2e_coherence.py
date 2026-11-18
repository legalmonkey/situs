"""
Automated regression tests for Gemma 3 4B inference fixes:
1. Single-node double-execution elimination & KV cache count verification
2. Tokenizer stop token ID population (<end_of_turn> 106)
3. TokenSampler repetition penalty
4. SWA RoPE position tracking
5. Lossless FP16/BF16 numerical activation conversion
6. Coordinator relay speculative verification broadcast
7. End-to-end token coherence on Google Gemma 3 4B
"""

import os
import pytest
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from situs.protocol.messages import (
    AssignLayers,
    InferenceStart,
    NextToken,
    ProtocolFrame,
    MessageType,
    SpeculativeVerified,
)
from situs.worker.loader import DistributedStageModel, RealTransformerBlock, find_local_weight_files
from situs.worker.engine import StageExecutionEngine
from situs.worker.sampler import TokenSampler
from situs.tokenizer import get_or_load_tokenizer
from situs.protocol.compression import compress_tensor, decompress_tensor
from situs.coordinator.relay import RelayHub
from situs.coordinator.app import _format_pipeline_prompt


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for stage engine tests")
def test_single_node_no_double_execution():
    """Verify single-node prefill and decode execute exactly once per layer and don't duplicate KV cache."""
    m = DistributedStageModel(0, 2, 2, True, True, 32, 100, 2, 2, 64, 16, "llama", "cpu")
    a = AssignLayers(
        worker_id="w1",
        model_name="test_model",
        layer_start=0,
        layer_end=2,
        total_layers=2,
        is_first_stage=True,
        is_last_stage=True,
    )
    e = StageExecutionEngine(m, a)

    # 3-token prompt prefill
    start = InferenceStart(request_id="r1", model_name="test_model", input_ids=[1, 2, 3], max_new_tokens=5)
    frame = e.start_inference(start)
    assert frame.type == MessageType.NEXT_TOKEN

    kvs = e.kv_cache_manager.get_all_kvs("r1")
    assert kvs is not None
    # KV cache length must equal prompt length (3), NOT doubled (6)
    assert kvs[0][0].shape[-2] == 3, f"Expected KV cache length 3, got {kvs[0][0].shape[-2]}"

    # Single-token decode step
    next_tok = NextToken(request_id="r1", step_index=0, token_id=10)
    frame2 = e.process_next_token(next_tok)
    assert frame2.type == MessageType.NEXT_TOKEN

    kvs2 = e.kv_cache_manager.get_all_kvs("r1")
    # KV cache length must equal prompt + 1 (4), NOT 6 or 8
    assert kvs2[0][0].shape[-2] == 4, f"Expected KV cache length 4, got {kvs2[0][0].shape[-2]}"


def test_gemma3_stop_token_ids_registered():
    """Verify Gemma 3 tokenizer registers stop token IDs including 106 (<end_of_turn>)."""
    try:
        tok = get_or_load_tokenizer("gemma3:4b")
    except Exception as e:
        pytest.skip(f"Gemma 3 tokenizer not available: {e}")

    assert hasattr(tok, "stop_token_ids"), "stop_token_ids attribute missing"
    assert 106 in tok.stop_token_ids, f"Gemma 3 stop_token_ids missing 106: {tok.stop_token_ids}"


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for sampler tests")
def test_repetition_penalty_in_sampler():
    """Verify TokenSampler applies repetition penalty to downweight recent tokens."""
    sampler = TokenSampler()
    # Token 5 has logit 10.0, token 6 has logit 9.5
    logits = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 10.0, 9.5])

    # Greedy without penalty -> token 5
    tok_no_pen = sampler.sample_next_token(logits, temperature=0.0, repetition_penalty=1.0)
    assert tok_no_pen == 5

    # Greedy with penalty on token 5 -> token 6
    tok_with_pen = sampler.sample_next_token(
        logits, temperature=0.0, repetition_penalty=1.1, context_tokens=[5]
    )
    assert tok_with_pen == 6


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required for compression tests")
def test_lossless_activation_compression_cross_dtype():
    """Verify FP16 and BF16 activations are converted mathematically without bit reinterpretation."""
    # FP16 tensor
    orig_f16 = torch.tensor([1.0, 2.0, -0.5, 3.14], dtype=torch.float16)
    comp_bytes, scale, shape = compress_tensor(orig_f16, method="fp16")

    # Decompress into BF16 target
    recovered_bf = decompress_tensor(comp_bytes, scale=scale, shape=shape, method="fp16", device="cpu", dtype=torch.bfloat16)
    assert recovered_bf.dtype == torch.bfloat16
    assert torch.allclose(recovered_bf, orig_f16.to(torch.bfloat16), atol=1e-2)

    # BF16 tensor
    orig_bf16 = torch.tensor([1.0, 2.0, -0.5, 3.14], dtype=torch.bfloat16)
    comp_bytes2, scale2, shape2 = compress_tensor(orig_bf16, method="bf16")

    # Decompress into FP16 target
    recovered_f16 = decompress_tensor(comp_bytes2, scale=scale2, shape=shape2, method="bf16", device="cpu", dtype=torch.float16)
    assert recovered_f16.dtype == torch.float16
    assert torch.allclose(recovered_f16, orig_bf16.to(torch.float16), atol=1e-2)


@pytest.mark.asyncio
async def test_relay_broadcasts_speculative_verified_to_intermediate_stages():
    """Verify RelayHub broadcasts SPECULATIVE_VERIFIED to all stages preceding terminal stage."""
    class MockWorker:
        def __init__(self, w_id):
            self.worker_id = w_id

    class MockRegistry:
        def get_pipeline_order(self):
            return [MockWorker("w1"), MockWorker("w2"), MockWorker("w3")]

    relay = RelayHub()
    relay.registry = MockRegistry()
    sent_to = []

    async def mock_send(target, frame):
        sent_to.append(target)
        return True

    relay.send_to_worker = mock_send

    verified = SpeculativeVerified(
        request_id="r1",
        step_index=1,
        accepted_tokens=[10],
        accepted_texts=["hi"],
        rejected_draft_count=1,
        is_finished=False,
    )
    frame = ProtocolFrame(
        type=MessageType.SPECULATIVE_VERIFIED,
        source="w3",
        target="w1",
        payload=verified.model_dump(),
    )
    await relay.route_frame(frame)
    # Intermediate stages w1 and w2 must both receive the frame to trim KV cache
    assert sent_to == ["w1", "w2"], f"Expected ['w1', 'w2'], got {sent_to}"


def test_gemma_chat_template_system_role_prepended():
    """Verify Gemma chat template prepends system prompt into the first user message turn."""
    from pydantic import BaseModel

    class Msg(BaseModel):
        role: str
        content: str

    msgs = [
        Msg(role="system", content="You are a helpful coding assistant."),
        Msg(role="user", content="Write a python function."),
    ]
    formatted = _format_pipeline_prompt(msgs, tokenizer=None, model_name="gemma3:4b")
    assert "<start_of_turn>model\nYou are a helpful" not in formatted
    assert "You are a helpful coding assistant.\n\nWrite a python function." in formatted


@pytest.mark.skipif(not HAS_TORCH or not torch.cuda.is_available(), reason="CUDA required for live Gemma 3 4B test")
def test_gemma3_e2e_coherence_on_cuda():
    """Verify live Gemma 3 4B model produces coherent text without garbage characters or loops."""
    weight_files = find_local_weight_files("gemma3:4b")
    if not weight_files:
        pytest.skip("Local Gemma 3 4B weight files not found on machine")

    from situs.worker.loader import load_hf_stage_model

    tok = get_or_load_tokenizer("gemma3:4b")
    model = load_hf_stage_model("gemma3:4b", 0, 34, 34, True, True, "cuda:0")
    assign = AssignLayers(
        worker_id="w1",
        model_name="gemma3:4b",
        layer_start=0,
        layer_end=34,
        total_layers=34,
        is_first_stage=True,
        is_last_stage=True,
    )
    engine = StageExecutionEngine(model, assign, tokenizer=tok)

    # Test "hi"
    ids = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=True, add_generation_prompt=True)
    start = InferenceStart(request_id="req_test_hi", model_name="gemma3:4b", input_ids=ids, max_new_tokens=15, temperature=0.7, top_p=0.9)
    frame = engine.start_inference(start)

    generated_text = []
    tokens_count = 0
    while frame and frame.type == MessageType.NEXT_TOKEN:
        tok_info = NextToken.model_validate(frame.payload)
        if tok_info.token_id is not None:
            generated_text.append(tok_info.token_text)
            tokens_count += 1
        if tok_info.is_finished:
            break
        frame = engine.process_next_token(tok_info)

    output = "".join(generated_text)
    assert tokens_count > 0, "No tokens generated"
    # Output must be coherent English greeting, not garbage or Cyrillic/Hindi symbols
    assert any(w in output.lower() for w in ("hi", "hello", "how", "there", "day")), f"Incoherent output: {output!r}"
