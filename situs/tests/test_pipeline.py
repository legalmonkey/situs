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


def test_three_worker_pipeline_autoregressive_loop():
    # Configure 3-stage pipeline (total 12 layers: 0..4, 4..8, 8..12)
    total_layers = 12
    hidden_dim = 128
    vocab_size = 500

    model_a = DistributedStageModel(
        layer_start=0, layer_end=4, total_layers=total_layers,
        is_first_stage=True, is_last_stage=False,
        hidden_dim=hidden_dim, vocab_size=vocab_size, device="cpu",
    )
    model_b = DistributedStageModel(
        layer_start=4, layer_end=8, total_layers=total_layers,
        is_first_stage=False, is_last_stage=False,
        hidden_dim=hidden_dim, vocab_size=vocab_size, device="cpu",
    )
    model_c = DistributedStageModel(
        layer_start=8, layer_end=12, total_layers=total_layers,
        is_first_stage=False, is_last_stage=True,
        hidden_dim=hidden_dim, vocab_size=vocab_size, device="cpu",
    )

    assign_a = AssignLayers(
        worker_id="worker-a", model_name="test-model",
        layer_start=0, layer_end=4, total_layers=total_layers,
        is_first_stage=True, is_last_stage=False,
        next_worker_id="worker-b", prev_worker_id=None, first_worker_id="worker-a",
    )
    assign_b = AssignLayers(
        worker_id="worker-b", model_name="test-model",
        layer_start=4, layer_end=8, total_layers=total_layers,
        is_first_stage=False, is_last_stage=False,
        next_worker_id="worker-c", prev_worker_id="worker-a", first_worker_id="worker-a",
    )
    assign_c = AssignLayers(
        worker_id="worker-c", model_name="test-model",
        layer_start=8, layer_end=12, total_layers=total_layers,
        is_first_stage=False, is_last_stage=True,
        next_worker_id=None, prev_worker_id="worker-b", first_worker_id="worker-a",
    )

    engine_a = StageExecutionEngine(stage_model=model_a, assignment=assign_a, compression_dtype="int8")
    engine_b = StageExecutionEngine(stage_model=model_b, assignment=assign_b, compression_dtype="int8")
    engine_c = StageExecutionEngine(stage_model=model_c, assignment=assign_c, compression_dtype="int8")

    # Verify middle stage bounce: if worker-b accidentally gets a NextToken, it routes to first_worker_id
    dummy_tok = NextToken(request_id="test-bounce", step_index=1, token_id=42)
    b_bounce = engine_b.process_next_token(dummy_tok)
    assert b_bounce is not None
    assert b_bounce.target == "worker-a"
    assert b_bounce.type == MessageType.NEXT_TOKEN

    # Start 3-worker inference
    request_id = "test-req-3workers"
    prompt_tokens = [10, 20, 30]
    max_new_tokens = 4

    start_msg = InferenceStart(
        request_id=request_id,
        model_name="test-model",
        input_ids=prompt_tokens,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )

    # Worker C needs session config
    engine_c.session_configs[request_id] = start_msg

    # Step 1: Worker A prefill
    frame_a1 = engine_a.start_inference(start_msg)
    assert frame_a1.type == MessageType.FORWARD_ACTIVATION
    assert frame_a1.target == "worker-b"

    # Worker B intermediate stage
    act_b = ForwardActivation.model_validate(frame_a1.payload)
    frame_b1 = engine_b.process_forward_activation(act_b)
    assert frame_b1.type == MessageType.FORWARD_ACTIVATION
    assert frame_b1.target == "worker-c"

    # Worker C last stage -> samples token 1
    act_c = ForwardActivation.model_validate(frame_b1.payload)
    frame_c1 = engine_c.process_forward_activation(act_c)
    assert frame_c1.type == MessageType.NEXT_TOKEN
    # CRITICAL: NextToken must target worker-a (first stage), NOT worker-b (prev worker)
    assert frame_c1.target == "worker-a"

    # Autoregressive multi-token loop across 3 workers
    generated = []
    current_token_frame = frame_c1

    for step in range(max_new_tokens):
        tok_data = NextToken.model_validate(current_token_frame.payload)
        generated.append(tok_data.token_id)
        if tok_data.is_finished:
            engine_a.process_next_token(tok_data)
            break

        # Send NextToken to Worker A (first stage)
        frame_a_next = engine_a.process_next_token(tok_data)
        assert frame_a_next is not None
        assert frame_a_next.target == "worker-b"

        # Worker A activation -> Worker B
        act_b_step = ForwardActivation.model_validate(frame_a_next.payload)
        frame_b_step = engine_b.process_forward_activation(act_b_step)
        assert frame_b_step.type == MessageType.FORWARD_ACTIVATION
        assert frame_b_step.target == "worker-c"

        # Worker B activation -> Worker C
        act_c_step = ForwardActivation.model_validate(frame_b_step.payload)
        frame_c_step = engine_c.process_forward_activation(act_c_step)
        assert frame_c_step.type == MessageType.NEXT_TOKEN
        assert frame_c_step.target == "worker-a"
        current_token_frame = frame_c_step

    assert len(generated) > 0
    assert len(generated) <= max_new_tokens

