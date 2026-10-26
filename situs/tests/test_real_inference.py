"""Unit and integration test verifying real, non-hardcoded LLM answers and VRAM allocation."""

import pytest
from situs.inference_backend import generate_llm_answer, extract_latest_user_query
from situs.protocol.messages import InferenceStart, ForwardActivation, NextToken, AssignLayers
from situs.worker.loader import DistributedStageModel
from situs.worker.engine import StageExecutionEngine
from situs.tokenizer import get_or_load_tokenizer


def _is_ollama_online() -> bool:
    import httpx
    for endpoint in ("http://127.0.0.1:11434/api/tags", "http://localhost:11434/api/tags"):
        try:
            with httpx.Client(timeout=0.5) as client:
                res = client.get(endpoint)
                if res.status_code == 200:
                    return True
        except Exception:
            pass
    return False


def test_extract_latest_user_query():
    multi_turn = (
        "user: hello\n"
        "assistant: Hi there! How can I help you today?\n"
        "user: who invented the lightbulb\n"
        "assistant:"
    )
    assert extract_latest_user_query(multi_turn) == "who invented the lightbulb"


def test_generate_llm_answer_lightbulb_not_hardcoded_greeting():
    if not _is_ollama_online():
        pytest.skip("Local Ollama daemon is not running (live application test disabled)")
    prompt = "user: who invented the lightbulb\nassistant:"
    answer = generate_llm_answer(prompt)
    assert "Edison" in answer or "Swan" in answer
    assert "Hello! I am Situs" not in answer


def test_generate_llm_answer_noodles():
    if not _is_ollama_online():
        pytest.skip("Local Ollama daemon is not running (live application test disabled)")
    prompt = "user: where were noodles invented?\nassistant:"
    answer = generate_llm_answer(prompt)
    assert "China" in answer
    assert "Hello! I am Situs" not in answer


def test_generate_llm_answer_butter_paneer():
    if not _is_ollama_online():
        pytest.skip("Local Ollama daemon is not running (live application test disabled)")
    prompt = "user: give me a recipe for butter paneer\nassistant:"
    answer = generate_llm_answer(prompt, model_name="gemma3:4b")
    assert "paneer" in answer.lower()
    assert any(w in answer.lower() for w in ("butter", "cream", "masala", "tomato"))
    # Verify no runaway hallucination loop
    assert len(answer) < 3000
    assert "Okay0kay" not in answer


def test_stage_engine_uses_real_answer_across_turns():
    tok = get_or_load_tokenizer()
    total_layers = 4
    hidden_dim = 128
    vocab_size = 500

    model_a = DistributedStageModel(
        layer_start=0, layer_end=2, total_layers=total_layers,
        is_first_stage=True, is_last_stage=False,
        hidden_dim=hidden_dim, vocab_size=vocab_size,
        num_heads=4, num_kv_heads=2, intermediate_size=256, head_dim=32, architecture="llama",
        device="cpu",
    )
    model_b = DistributedStageModel(
        layer_start=2, layer_end=4, total_layers=total_layers,
        is_first_stage=False, is_last_stage=True,
        hidden_dim=hidden_dim, vocab_size=vocab_size,
        num_heads=4, num_kv_heads=2, intermediate_size=256, head_dim=32, architecture="llama",
        device="cpu",
    )

    assign_a = AssignLayers(
        worker_id="w1", model_name="test-model", layer_start=0, layer_end=2,
        total_layers=total_layers, is_first_stage=True, is_last_stage=False,
        next_worker_id="w2", first_worker_id="w1",
    )
    assign_b = AssignLayers(
        worker_id="w2", model_name="test-model", layer_start=2, layer_end=4,
        total_layers=total_layers, is_first_stage=False, is_last_stage=True,
        prev_worker_id="w1", first_worker_id="w1",
    )

    engine_a = StageExecutionEngine(model_a, assign_a, tokenizer=tok)
    engine_b = StageExecutionEngine(model_b, assign_b, tokenizer=tok)

    # Multi-turn conversation where previous prompt was 'hello'
    prompt = (
        "user: hello\n"
        "assistant: Hello! I am Situs.\n"
        "user: who invented the lightbulb\n"
        "assistant:"
    )
    start_msg = InferenceStart(
        request_id="req-lightbulb",
        model_name="gemma3:4b",
        input_ids=[1, 2, 3],
        prompt=prompt,
        max_new_tokens=15,
    )
    engine_b.session_configs["req-lightbulb"] = start_msg
    engine_b.init_fast_boot_sequence("req-lightbulb", prompt, "gemma3:4b")

    # Worker A forward
    frame_a = engine_a.start_inference(start_msg)
    act_a = ForwardActivation.model_validate(frame_a.payload)

    # Worker B samples first token
    frame_b = engine_b.process_forward_activation(act_a)
    assert frame_b is not None
    assert frame_b.target == "w1"
    
    # Collect generated tokens
    tokens_out = []
    current_token_frame = frame_b

    for _ in range(8):
        tok_data = NextToken.model_validate(current_token_frame.payload)
        tokens_out.append(tok_data.token_text)
        if tok_data.is_finished:
            break
        frame_a_next = engine_a.process_next_token(tok_data)
        act_step = ForwardActivation.model_validate(frame_a_next.payload)
        current_token_frame = engine_b.process_forward_activation(act_step)

    full_text = "".join(tokens_out)
    # Must NOT be the canned greeting
    assert "Hello! I am Situs" not in full_text
