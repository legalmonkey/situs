"""End-to-end integration test verifying that speculative candidate bursts accelerate pipeline steps."""

import pytest
import torch
from situs.protocol.messages import (
    MessageType,
    AssignLayers,
    InferenceStart,
    ForwardActivation,
    NextToken,
    SpeculativeCandidates,
    SpeculativeVerified,
)
from situs.worker.loader import DistributedStageModel
from situs.worker.engine import StageExecutionEngine


def test_speculative_pipeline_burst_acceleration():
    total_layers = 8
    hidden_dim = 128
    vocab_size = 200

    model_a = DistributedStageModel(
        layer_start=0,
        layer_end=4,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
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
        device="cpu",
    )

    assign_a = AssignLayers(
        worker_id="worker-a",
        model_name="test-spec",
        layer_start=0,
        layer_end=4,
        total_layers=total_layers,
        is_first_stage=True,
        is_last_stage=False,
        next_worker_id="worker-b",
    )
    assign_b = AssignLayers(
        worker_id="worker-b",
        model_name="test-spec",
        layer_start=4,
        layer_end=8,
        total_layers=total_layers,
        is_first_stage=False,
        is_last_stage=True,
        prev_worker_id="worker-a",
    )

    engine_a = StageExecutionEngine(stage_model=model_a, assignment=assign_a, compression_dtype="int8")
    engine_b = StageExecutionEngine(stage_model=model_b, assignment=assign_b, compression_dtype="int8")

    request_id = "spec-req-001"
    # Prompt contains recurring sequence [10, 20] followed by [30, 40]
    prompt_tokens = [10, 20, 30, 40, 99, 10]
    start_msg = InferenceStart(
        request_id=request_id,
        model_name="test-spec",
        input_ids=prompt_tokens,
        max_new_tokens=10,
        temperature=0.0,
    )

    # 1. Prefill
    frame_a1 = engine_a.start_inference(start_msg)
    engine_b.session_configs[request_id] = start_msg

    # 2. Worker B processes prefill
    act_payload_b = ForwardActivation.model_validate(frame_a1.payload)
    frame_b1 = engine_b.process_forward_activation(act_payload_b)
    assert frame_b1.type == MessageType.NEXT_TOKEN

    # 3. Simulate Worker B sending token 20 (which triggers the [10, 20] pattern match!)
    tok_frame = NextToken(
        request_id=request_id,
        step_index=1,
        token_id=20,
        token_text=" 20",
    )

    # Worker A receives token 20. Since history ends with [10, 20], it drafts [30, 40]!
    frame_a2 = engine_a.process_next_token(tok_frame)
    assert frame_a2.type == MessageType.SPECULATIVE_CANDIDATES

    cand_payload = SpeculativeCandidates.model_validate(frame_a2.payload)
    assert cand_payload.candidate_tokens == [30, 40, 99, 10]

    # 4. Worker B receives the multi-token candidates and verifies them in 1 pass!
    frame_b2 = engine_b.process_speculative_candidates(cand_payload)
    assert frame_b2.type == MessageType.SPECULATIVE_VERIFIED

    verified_payload = SpeculativeVerified.model_validate(frame_b2.payload)
    # At temperature=0, verification accepts tokens in 1 pass!
    assert len(verified_payload.accepted_tokens) >= 3
    assert 30 in verified_payload.accepted_tokens
    assert 40 in verified_payload.accepted_tokens
