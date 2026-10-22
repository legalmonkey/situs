"""End-to-end integration test for two-stage pipeline parallelism and KV cache cycling."""

import pytest
from situs.protocol.messages import (
    MessageType,
    AssignLayers,
    InferenceStart,
    ForwardActivation,
    NextToken,
    SpeculativeCandidates,
)
from situs.worker.loader import DistributedStageModel
from situs.worker.engine import StageExecutionEngine


def test_two_worker_pipeline_autoregressive_loop():
    # Configure 2-stage pipeline (total 8 layers, 4 layers per worker)
    total_layers = 8
    hidden_dim = 128
    vocab_size = 500

    # 1. Instantiate Stage Models
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
    )

    engine_a = StageExecutionEngine(stage_model=model_a, assignment=assign_a, compression_dtype="int8")
    engine_b = StageExecutionEngine(stage_model=model_b, assignment=assign_b, compression_dtype="int8")

    # 2. Start inference with a 3-token prompt
    request_id = "test-req-001"
    prompt_tokens = [12, 45, 99]
    max_new_tokens = 5

    start_msg = InferenceStart(
        request_id=request_id,
        model_name="test-model",
        input_ids=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=0.0,  # greedy deterministic for test
    )

    # Worker A executes prefill
    frame_a1 = engine_a.start_inference(start_msg)
    assert frame_a1.type == MessageType.FORWARD_ACTIVATION
    assert frame_a1.target == "worker-b"
    assert request_id in engine_a.session_kv_caches

    # Send activation to Worker B
    act_payload_b = ForwardActivation.model_validate(frame_a1.payload)
    # Store session config on worker B as coordinator would
    engine_b.session_configs[request_id] = start_msg

    # Autoregressive generation loop
    generated_tokens = []
    current_act_frame = act_payload_b

    for step in range(max_new_tokens):
        # Worker B processes intermediate activation -> produces NextToken
        frame_b = engine_b.process_forward_activation(current_act_frame)
        assert frame_b.type == MessageType.NEXT_TOKEN
        token_data = NextToken.model_validate(frame_b.payload)
        generated_tokens.append(token_data.token_id)

        if token_data.is_finished:
            # Notify Worker A of finish so it cleans up local session cache
            engine_a.process_next_token(token_data)
            break

        # Worker B sends NextToken back to Worker A
        frame_a = engine_a.process_next_token(token_data)
        assert frame_a is not None
        assert frame_a.type in (MessageType.FORWARD_ACTIVATION, MessageType.SPECULATIVE_CANDIDATES)
        if frame_a.type == MessageType.FORWARD_ACTIVATION:
            current_act_frame = ForwardActivation.model_validate(frame_a.payload)
        else:
            cand = SpeculativeCandidates.model_validate(frame_a.payload)
            current_act_frame = ForwardActivation(
                request_id=cand.request_id,
                step_index=cand.step_index,
                payload_b64=cand.payload_b64,
                scale=cand.scale,
                shape=cand.shape,
                dtype=cand.dtype,
            )

    # Verify that generation completed
    assert len(generated_tokens) > 0
    assert len(generated_tokens) <= max_new_tokens
    # Session KV caches are cleaned up on finish across both workers
    assert request_id not in engine_a.session_kv_caches
    assert request_id not in engine_b.session_kv_caches
