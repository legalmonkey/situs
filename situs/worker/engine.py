"""Stage execution engine managing local KV cache and pipelined forward passes."""

import base64
import logging
from typing import Optional, Dict, List, Any, Tuple
from situs.protocol.compression import compress_tensor, decompress_tensor, decompress_layer_bytes
from situs.protocol.messages import (
    MessageType,
    ProtocolFrame,
    InferenceStart,
    ForwardActivation,
    NextToken,
    AssignLayers,
    SpeculativeCandidates,
    SpeculativeVerified,
    StreamLayerChunk,
    StageLayerAck,
)
from situs.worker.loader import DistributedStageModel
from situs.worker.sampler import TokenSampler
from situs.worker.speculative import PromptLookupDrafter, SpeculativeVerifier
from situs.worker.jit_streamer import save_layer_to_cache

logger = logging.getLogger("situs.worker.engine")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class StageExecutionEngine:
    def __init__(
        self,
        stage_model: DistributedStageModel,
        assignment: AssignLayers,
        compression_dtype: str = "int8",
        tokenizer=None,
    ):
        self.stage_model = stage_model
        self.assignment = assignment
        self.compression_dtype = compression_dtype
        self.sampler = TokenSampler(tokenizer=tokenizer)
        self.prompt_drafter = PromptLookupDrafter(ngram_size=2, max_draft_tokens=4)
        self.verifier = SpeculativeVerifier(self.sampler)
        # Session KV cache store: request_id -> list of layer KV states
        self.session_kv_caches: Dict[str, List[Any]] = {}
        # Generation configuration per active session: request_id -> InferenceStart
        self.session_configs: Dict[str, InferenceStart] = {}
        # Token history per session for prompt-lookup drafting
        self.session_token_histories: Dict[str, List[int]] = {}
        # Buffer for incoming progressive JIT layer chunks: layer_index -> {chunk_index: b64_payload}
        self.layer_chunk_buffers: Dict[int, Dict[int, str]] = {}

    def is_first_stage(self) -> bool:
        return self.assignment.is_first_stage

    def is_last_stage(self) -> bool:
        return self.assignment.is_last_stage

    def process_layer_chunk(
        self, chunk: StreamLayerChunk, source_worker: Optional[str] = None
    ) -> Optional[ProtocolFrame]:
        """
        Processes an incoming chunk of a streamed transformer layer.
        Reassembles chunks, decompresses wire format (Strategy 4), caches payload to disk,
        and hot-swaps the layer in memory (Strategy 3).
        Returns STAGE_LAYER_ACK when a layer is completely assembled and ready.
        """
        layer_idx = chunk.layer_index
        if layer_idx not in self.layer_chunk_buffers:
            self.layer_chunk_buffers[layer_idx] = {}

        # Store chunk payload (either raw binary bytes or Base64 string)
        if chunk.raw_bytes is not None:
            self.layer_chunk_buffers[layer_idx][chunk.chunk_index] = (chunk.raw_bytes, chunk.codec)
        elif chunk.payload_b64:
            self.layer_chunk_buffers[layer_idx][chunk.chunk_index] = (chunk.payload_b64, chunk.codec)
        else:
            self.layer_chunk_buffers[layer_idx][chunk.chunk_index] = (b"", chunk.codec)

        buf = self.layer_chunk_buffers[layer_idx]

        # Check if all chunks have arrived
        if len(buf) == chunk.total_chunks:
            # Reassemble ordered chunks
            try:
                raw_parts = []
                codec = chunk.codec
                for i in range(chunk.total_chunks):
                    item = buf[i]
                    if isinstance(item, tuple):
                        p, c = item
                        codec = c or codec
                    else:
                        p = item
                    if isinstance(p, bytes):
                        raw_parts.append(p)
                    elif isinstance(p, str):
                        raw_parts.append(base64.b64decode(p))
                    else:
                        raw_parts.append(b"")

                wire_bytes = b"".join(raw_parts)
                reassembled = decompress_layer_bytes(wire_bytes, codec=codec)
                try:
                    save_layer_to_cache(chunk.model_name, layer_idx, reassembled)
                except Exception as cache_err:
                    logger.debug(f"Cache write skipped for layer {layer_idx}: {cache_err}")
            except Exception as e:
                logger.error(f"Failed to reassemble layer {layer_idx}: {e}")
                reassembled = None

            # Mark layer operational in DistributedStageModel (triggers hot-swap if fast_boot_active)
            self.stage_model.mark_layer_ready(layer_idx, weights=reassembled)
            del self.layer_chunk_buffers[layer_idx]

            total_assigned = self.assignment.layer_end - self.assignment.layer_start
            ack = StageLayerAck(
                worker_id=self.assignment.worker_id,
                model_name=chunk.model_name,
                layer_index=layer_idx,
                status="ready",
                ready_layer_count=self.stage_model.get_ready_count(),
                total_assigned=total_assigned,
                is_fast_boot_active=self.stage_model.fast_boot_active,
            )
            target = source_worker or self.assignment.prev_worker_id or "coordinator"
            return ProtocolFrame(
                type=MessageType.STAGE_LAYER_ACK,
                source=self.assignment.worker_id,
                target=target,
                payload=ack.model_dump(),
            )

        return None

    def start_inference(self, start_msg: InferenceStart) -> ProtocolFrame:
        """
        Invoked on the first pipeline stage when Coordinator dispatches a prompt.
        """
        if not self.is_first_stage():
            raise RuntimeError("start_inference called on a non-first stage worker")

        request_id = start_msg.request_id
        self.session_configs[request_id] = start_msg
        self.session_token_histories[request_id] = list(start_msg.input_ids)

        if HAS_TORCH:
            input_ids = torch.tensor([start_msg.input_ids], dtype=torch.long, device=self.stage_model.device)
            h, new_kvs = self.stage_model.forward_first_stage(input_ids, past_kvs=None)
            self.session_kv_caches[request_id] = new_kvs
        else:
            h = [[0.0] * self.stage_model.hidden_dim]
            self.session_kv_caches[request_id] = []

        # Compress boundary activation
        comp_bytes, scale, shape = compress_tensor(h, method=self.compression_dtype)
        b64_payload = base64.b64encode(comp_bytes).decode("ascii")

        forward_act = ForwardActivation(
            request_id=request_id,
            step_index=0,
            shape=list(shape),
            dtype=self.compression_dtype,
            scale=scale,
            payload_b64=b64_payload,
        )

        return ProtocolFrame(
            type=MessageType.FORWARD_ACTIVATION,
            source=self.assignment.worker_id,
            target=self.assignment.next_worker_id,
            payload=forward_act.model_dump(),
        )

    def process_forward_activation(self, act_frame: ForwardActivation) -> Optional[ProtocolFrame]:
        """
        Invoked on middle and last stages when activations arrive from previous stage.
        """
        request_id = act_frame.request_id
        comp_bytes = base64.b64decode(act_frame.payload_b64)

        if HAS_TORCH:
            hidden_states = decompress_tensor(
                comp_bytes,
                scale=act_frame.scale,
                shape=tuple(act_frame.shape),
                method=act_frame.dtype,
                device=self.stage_model.device,
                dtype=torch.float32,
            )
            past_kvs = self.session_kv_caches.get(request_id)
            out, new_kvs = self.stage_model.forward_intermediate_stage(hidden_states, past_kvs=past_kvs)
            self.session_kv_caches[request_id] = new_kvs
        else:
            out = [[0.0] * self.stage_model.vocab_size] if self.is_last_stage() else [[0.0] * self.stage_model.hidden_dim]

        if not self.is_last_stage():
            # Middle stage: forward activations down the chain
            comp_bytes, scale, shape = compress_tensor(out, method=self.compression_dtype)
            b64_payload = base64.b64encode(comp_bytes).decode("ascii")
            forward_act = ForwardActivation(
                request_id=request_id,
                step_index=act_frame.step_index,
                shape=list(shape),
                dtype=self.compression_dtype,
                scale=scale,
                payload_b64=b64_payload,
            )
            return ProtocolFrame(
                type=MessageType.FORWARD_ACTIVATION,
                source=self.assignment.worker_id,
                target=self.assignment.next_worker_id,
                payload=forward_act.model_dump(),
            )
        else:
            # Last stage: compute next token from logits!
            cfg = self.session_configs.get(request_id)
            temperature = cfg.temperature if cfg else 0.7
            top_p = cfg.top_p if cfg else 0.9
            max_new_tokens = cfg.max_new_tokens if cfg else 256
            stop_token_ids = cfg.stop_token_ids if cfg else []

            token_id = self.sampler.sample_next_token(out, temperature=temperature, top_p=top_p)
            token_text = self.sampler.decode_token(token_id)
            step_idx = act_frame.step_index + 1

            is_eos = (token_id in stop_token_ids) or (token_id == 2) or (token_id == 0)
            is_finished = is_eos or (step_idx >= max_new_tokens)
            finish_reason = "stop" if is_eos else ("length" if is_finished else None)

            if is_finished:
                # Clean up session KV cache
                self.session_kv_caches.pop(request_id, None)
                self.session_configs.pop(request_id, None)

            next_token = NextToken(
                request_id=request_id,
                step_index=step_idx,
                token_id=token_id,
                token_text=token_text,
                is_finished=is_finished,
                finish_reason=finish_reason,
            )

            # Target the first worker (to embed next token) AND Coordinator
            return ProtocolFrame(
                type=MessageType.NEXT_TOKEN,
                source=self.assignment.worker_id,
                target=self.assignment.prev_worker_id,  # In a 2-worker ring, prev is Worker A
                payload=next_token.model_dump(),
            )

    def process_speculative_candidates(self, cand_frame: SpeculativeCandidates) -> Optional[ProtocolFrame]:
        """
        Invoked on Worker B to verify a drafted multi-token candidate batch in a single forward pass.
        """
        request_id = cand_frame.request_id
        comp_bytes = base64.b64decode(cand_frame.payload_b64)

        if HAS_TORCH:
            hidden_states = decompress_tensor(
                comp_bytes,
                scale=cand_frame.scale,
                shape=tuple(cand_frame.shape),
                method=cand_frame.dtype,
                device=self.stage_model.device,
                dtype=torch.float32,
            )
            past_kvs = self.session_kv_caches.get(request_id)
            logits, new_kvs = self.stage_model.forward_intermediate_stage(hidden_states, past_kvs=past_kvs)
            self.session_kv_caches[request_id] = new_kvs
        else:
            logits = [[0.0] * self.stage_model.vocab_size]

        cfg = self.session_configs.get(request_id)
        temp = cfg.temperature if cfg else 0.7
        top_p = cfg.top_p if cfg else 0.9

        accepted_tokens, correction = self.verifier.verify_candidates(
            cand_frame.candidate_tokens,
            logits,
            temperature=temp,
            top_p=top_p,
        )
        if correction is not None:
            accepted_tokens.append(correction)

        accepted_texts = [self.sampler.decode_token(t) for t in accepted_tokens]
        step_idx = cand_frame.step_index + len(accepted_tokens)

        max_tokens = cfg.max_new_tokens if cfg else 256
        is_finished = (step_idx >= max_tokens)

        if is_finished:
            self.session_kv_caches.pop(request_id, None)
            self.session_configs.pop(request_id, None)

        verified = SpeculativeVerified(
            request_id=request_id,
            step_index=step_idx,
            accepted_tokens=accepted_tokens,
            accepted_texts=accepted_texts,
            is_finished=is_finished,
            finish_reason="length" if is_finished else None,
        )

        return ProtocolFrame(
            type=MessageType.SPECULATIVE_VERIFIED,
            source=self.assignment.worker_id,
            target=self.assignment.prev_worker_id,
            payload=verified.model_dump(),
        )

    def process_speculative_verified(self, verified_frame: SpeculativeVerified) -> Optional[ProtocolFrame]:
        """
        Invoked on Worker A when downstream stage accepts speculative tokens.
        Appends verified tokens to history and triggers the next draft/decode step.
        """
        if not self.is_first_stage():
            return None

        request_id = verified_frame.request_id
        if verified_frame.is_finished:
            self.session_kv_caches.pop(request_id, None)
            self.session_configs.pop(request_id, None)
            self.session_token_histories.pop(request_id, None)
            return None

        # Append accepted tokens to history
        if request_id in self.session_token_histories:
            self.session_token_histories[request_id].extend(verified_frame.accepted_tokens)

        # Take last accepted token to continue decoding
        last_token = verified_frame.accepted_tokens[-1]
        dummy_next = NextToken(
            request_id=request_id,
            step_index=verified_frame.step_index,
            token_id=last_token,
        )
        return self.process_next_token(dummy_next)

    def process_next_token(self, token_frame: NextToken) -> Optional[ProtocolFrame]:
        """
        Invoked on the first pipeline stage when the last stage emits the next token ID.
        Worker A embeds the single token and sends intermediate activation to continue the cycle!
        """
        if not self.is_first_stage():
            return None

        request_id = token_frame.request_id
        if token_frame.is_finished:
            # Generation finished, clean up local KV cache
            self.session_kv_caches.pop(request_id, None)
            self.session_configs.pop(request_id, None)
            self.session_token_histories.pop(request_id, None)
            return None

        # Track history
        if request_id not in self.session_token_histories:
            self.session_token_histories[request_id] = []
        self.session_token_histories[request_id].append(token_frame.token_id)

        # 1. Speculative Path: check for candidate n-grams
        candidates = self.prompt_drafter.draft_candidates(self.session_token_histories[request_id])
        if candidates and HAS_TORCH:
            input_tensor = torch.tensor([[token_frame.token_id] + candidates], dtype=torch.long, device=self.stage_model.device)
            past_kvs = self.session_kv_caches.get(request_id)
            h, new_kvs = self.stage_model.forward_first_stage(input_tensor, past_kvs=past_kvs)
            self.session_kv_caches[request_id] = new_kvs

            comp_bytes, scale, shape = compress_tensor(h, method=self.compression_dtype)
            b64_payload = base64.b64encode(comp_bytes).decode("ascii")

            spec_cand = SpeculativeCandidates(
                request_id=request_id,
                step_index=token_frame.step_index,
                candidate_tokens=candidates,
                shape=list(shape),
                dtype=self.compression_dtype,
                scale=scale,
                payload_b64=b64_payload,
            )
            return ProtocolFrame(
                type=MessageType.SPECULATIVE_CANDIDATES,
                source=self.assignment.worker_id,
                target=self.assignment.next_worker_id,
                payload=spec_cand.model_dump(),
            )

        # 2. Standard Single-Token Path
        if HAS_TORCH:
            input_tensor = torch.tensor([[token_frame.token_id]], dtype=torch.long, device=self.stage_model.device)
            past_kvs = self.session_kv_caches.get(request_id)
            h, new_kvs = self.stage_model.forward_first_stage(input_tensor, past_kvs=past_kvs)
            self.session_kv_caches[request_id] = new_kvs
        else:
            h = [[0.0] * self.stage_model.hidden_dim]

        comp_bytes, scale, shape = compress_tensor(h, method=self.compression_dtype)
        b64_payload = base64.b64encode(comp_bytes).decode("ascii")

        forward_act = ForwardActivation(
            request_id=request_id,
            step_index=token_frame.step_index,
            shape=list(shape),
            dtype=self.compression_dtype,
            scale=scale,
            payload_b64=b64_payload,
        )

        return ProtocolFrame(
            type=MessageType.FORWARD_ACTIVATION,
            source=self.assignment.worker_id,
            target=self.assignment.next_worker_id,
            payload=forward_act.model_dump(),
        )
