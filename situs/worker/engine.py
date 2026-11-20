"""Stage execution engine managing local KV cache and pipelined forward passes."""

import base64
import contextlib
import logging
import os
import threading
from collections import OrderedDict
from typing import Optional, Dict, List, Any, Tuple

import sys
import warnings

def configure_pytorch_allocator():
    """Configures PYTORCH_CUDA_ALLOC_CONF safely based on platform capabilities.
    On Windows (WDDM), expandable_segments is NOT supported by PyTorch CUDAAllocatorConfig.h
    and triggers noisy UserWarnings that corrupt full-screen terminal UIs.
    On Linux/POSIX, expandable_segments:True prevents virtual memory fragmentation."""
    if sys.platform == "win32":
        alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        parts = [p.strip() for p in alloc.split(",") if p.strip() and not p.strip().startswith("expandable_segments")]
        if not any(p.startswith("max_split_size_mb") for p in parts):
            parts.append("max_split_size_mb:128")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)
    else:
        alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        parts = [p.strip() for p in alloc.split(",") if p.strip()]
        if not any(p.startswith("expandable_segments") for p in parts):
            parts.append("expandable_segments:True")
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = ",".join(parts)

configure_pytorch_allocator()

# Suppress PyTorch internal C++ allocator warnings that can disrupt full-screen terminal layouts
warnings.filterwarnings("ignore", message=".*expandable_segments not supported on this platform.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.*")

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
from situs.protocol.framing import pack_tensor_frame, unpack_tensor_frame, TensorContractError
from situs.worker.memory import GPUMemoryManager, RequestScopedKVCache, MemoryExhaustedError

logger = logging.getLogger("situs.worker.engine")

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@contextlib.contextmanager
def _inference_context():
    """Wrap forward passes in torch.inference_mode to reduce autograd tracking overhead and memory churn."""
    if HAS_TORCH and hasattr(torch, "inference_mode"):
        with torch.inference_mode():
            yield
    elif HAS_TORCH and hasattr(torch, "no_grad"):
        with torch.no_grad():
            yield
    else:
        yield


class StageOrderingError(ValueError):
    """Raised when an activation frame violates strict pipeline stage ordering invariants."""
    pass


class BoundedSet:
    """Thread-safe bounded set keeping at most max_size most recent elements (FIFO eviction)."""
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self._items: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, item: str) -> None:
        with self._lock:
            self._items[item] = None
            self._items.move_to_end(item)
            if len(self._items) > self.max_size:
                self._items.popitem(last=False)

    def discard(self, item: str) -> None:
        with self._lock:
            self._items.pop(item, None)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __contains__(self, item: str) -> bool:
        with self._lock:
            return item in self._items

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def __iter__(self):
        with self._lock:
            return iter(list(self._items.keys()))


class CancellationTokenRegistry:
    """Out-of-band cancellation registry for cooperative request abortion."""
    def __init__(self, max_size: int = 10000):
        self._cancelled = BoundedSet(max_size=max_size)

    def cancel(self, request_id: str) -> None:
        self._cancelled.add(request_id)

    def is_cancelled(self, request_id: str) -> bool:
        return request_id in self._cancelled

    def clear(self, request_id: str) -> None:
        self._cancelled.discard(request_id)


class ActivationReplayRingBuffer:
    """Ring buffer caching recent boundary activations (depth K=4) for fault-tolerant replay."""
    def __init__(self, capacity: int = 4):
        self.capacity = capacity
        self.buffer: List[Tuple[str, int, bytes, Dict[str, Any]]] = []

    def record(self, request_id: str, step_index: int, payload_bytes: bytes, metadata: Dict[str, Any]) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((request_id, step_index, payload_bytes, metadata))

    def retrieve(self, request_id: str, step_index: int) -> Optional[Tuple[bytes, Dict[str, Any]]]:
        for req_id, step, p_bytes, meta in reversed(self.buffer):
            if req_id == request_id and step == step_index:
                return p_bytes, meta
        return None

    def clear(self, request_id: Optional[str] = None) -> None:
        if request_id is None:
            self.buffer.clear()
        else:
            self.buffer = [item for item in self.buffer if item[0] != request_id]


class StageExecutionEngine:
    def __init__(
        self,
        stage_model: DistributedStageModel,
        assignment: AssignLayers,
        compression_dtype: str = "fp16",
        tokenizer=None,
        ngram_size: int = 2,
        max_draft_tokens: int = 4,
        replay_capacity: int = 4,
        safety_headroom_fraction: float = 0.10,
        enable_speculative: Optional[bool] = None,
    ):
        self.stage_model = stage_model
        self.assignment = assignment
        self.compression_dtype = compression_dtype
        self.sampler = TokenSampler(tokenizer=tokenizer)
        self.prompt_drafter = PromptLookupDrafter(ngram_size=ngram_size, max_draft_tokens=max_draft_tokens)
        self.verifier = SpeculativeVerifier(self.sampler)
        if enable_speculative is not None:
            self.enable_speculative = bool(enable_speculative)
        else:
            # Default to False for single-worker runs to avoid speculative overhead on a single GPU
            is_single = bool(assignment.is_first_stage and assignment.is_last_stage)
            self.enable_speculative = not is_single
        # Session KV cache store: request_id -> list of layer KV states
        self.session_kv_caches: Dict[str, List[Any]] = {}
        # Generation configuration per active session: request_id -> InferenceStart
        self.session_configs: Dict[str, InferenceStart] = {}
        # Monotonically increasing cumulative step positions per session (prevents sliding-window RoPE offset freeze)
        self.session_step_counts: Dict[str, int] = {}
        # Token history per session for prompt-lookup drafting
        self.session_token_histories: Dict[str, List[int]] = {}
        # Buffer for incoming progressive JIT layer chunks: layer_index -> {chunk_index: b64_payload}
        self.layer_chunk_buffers: Dict[int, Dict[int, str]] = {}
        # Completed sessions to prevent stale in-flight duplicate frames from reviving finished generations (bounded to prevent unbounded growth)
        self.finished_sessions = BoundedSet(max_size=10000)

        # State-of-the-art memory budgeting, cache isolation, and fault recovery
        dev = getattr(stage_model, "device", "cpu")
        try:
            if str(dev).strip().lower() == "cpu":
                from situs.hardware import get_system_ram_and_cpu
                tot_ram, _, _ = get_system_ram_and_cpu()
                total_vram_b = tot_ram if (tot_ram and tot_ram > 0) else (16 * 1024**3)
            else:
                from situs.hardware import get_live_gpu_vram
                tot_vram, _, free_vram = get_live_gpu_vram()
                total_vram_b = tot_vram if (tot_vram and tot_vram > 0) else (16 * 1024**3)
        except Exception:
            total_vram_b = 16 * 1024**3
        self.memory_manager = GPUMemoryManager(
            total_vram_bytes=total_vram_b,
            safety_headroom_fraction=safety_headroom_fraction,
            device=dev,
        )
        if stage_model is not None and hasattr(stage_model, "parameters"):
            try:
                weight_bytes = sum(p.numel() * p.element_size() for p in stage_model.parameters())
                self.memory_manager.set_weights_footprint(weight_bytes)
            except Exception:
                pass
        self.kv_cache_manager = RequestScopedKVCache(memory_manager=self.memory_manager)
        self.cancellation_tokens = CancellationTokenRegistry()
        self.replay_buffer = ActivationReplayRingBuffer(capacity=replay_capacity)

    def is_first_stage(self) -> bool:
        return self.assignment.is_first_stage

    def is_last_stage(self) -> bool:
        return self.assignment.is_last_stage

    def _call_forward_first_stage(self, input_ids: Any, past_kvs: Any = None, position_offset: Optional[int] = None):
        try:
            return self.stage_model.forward_first_stage(input_ids, past_kvs=past_kvs, position_offset=position_offset)
        except TypeError:
            try:
                return self.stage_model.forward_first_stage(input_ids, past_kvs=past_kvs)
            except TypeError:
                return self.stage_model.forward_first_stage(input_ids)

    def _call_forward_intermediate_stage(self, hidden_states: Any, past_kvs: Any = None, position_offset: Optional[int] = None, compute_all_logits: bool = False):
        try:
            if compute_all_logits:
                return self.stage_model.forward_intermediate_stage(hidden_states, past_kvs=past_kvs, position_offset=position_offset, compute_all_logits=True)
            return self.stage_model.forward_intermediate_stage(hidden_states, past_kvs=past_kvs, position_offset=position_offset)
        except TypeError:
            try:
                if compute_all_logits:
                    return self.stage_model.forward_intermediate_stage(hidden_states, past_kvs=past_kvs, compute_all_logits=True)
                return self.stage_model.forward_intermediate_stage(hidden_states, past_kvs=past_kvs)
            except TypeError:
                try:
                    return self.stage_model.forward_intermediate_stage(hidden_states, past_kvs=past_kvs)
                except TypeError:
                    return self.stage_model.forward_intermediate_stage(hidden_states)

    def _call_compute_logits(self, h: Any, compute_all_logits: Optional[bool] = None) -> Any:
        fn = getattr(self.stage_model, "compute_logits", None)
        if callable(fn):
            try:
                return fn(h, compute_all_logits=compute_all_logits)
            except TypeError:
                return fn(h)
        norm = getattr(self.stage_model, "norm", None)
        head = getattr(self.stage_model, "lm_head", None)
        out = h
        if norm is not None and callable(norm):
            out = norm(out)
        if head is not None and callable(head):
            out = head(out)
        return out

    def _sample_next_token_from_logits(
        self,
        request_id: str,
        step_idx: int,
        logits: Any,
    ) -> ProtocolFrame:
        cfg = self.session_configs.get(request_id)
        temperature = cfg.temperature if (cfg and cfg.temperature is not None) else 0.7
        top_p = cfg.top_p if (cfg and cfg.top_p is not None) else 0.9
        seed = getattr(cfg, "seed", None)
        max_new_tokens = cfg.max_new_tokens if (cfg and cfg.max_new_tokens is not None) else 2048
        stop_token_ids = list(cfg.stop_token_ids or []) if cfg else []
        tok_stops = getattr(self.sampler.tokenizer, "stop_token_ids", None)
        if tok_stops:
            for sid in tok_stops:
                if sid not in stop_token_ids:
                    stop_token_ids.append(int(sid))

        if HAS_TORCH and isinstance(logits, torch.Tensor):
            if logits.dim() == 3:
                last_logits = logits[0, -1, :]
            elif logits.dim() == 2:
                last_logits = logits[-1, :]
            else:
                last_logits = logits
        else:
            last_logits = logits

        top_k = getattr(cfg, "top_k", 50) if cfg else 50
        token_id = self.sampler.sample_next_token(
            last_logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            step_index=step_idx,
        )
        token_text = self.sampler.decode_token(token_id)
        eos_id = getattr(self.sampler.tokenizer, "eos_token_id", None)
        if stop_token_ids:
            is_eos = token_id in stop_token_ids
        elif eos_id is not None:
            is_eos = (token_id == eos_id)
        else:
            is_eos = False

        if not is_eos and token_text:
            is_eos = token_text.strip() in ("<eos>", "<end_of_turn>", "</s>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>")
        if is_eos and token_text.strip() in ("<eos>", "<end_of_turn>", "</s>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>"):
            token_text = ""

        is_finished = is_eos or (step_idx + 1 >= max_new_tokens)
        finish_reason = "stop" if is_eos else ("length" if is_finished else None)

        if is_finished:
            self.process_inference_finish(request_id)

        next_token = NextToken(
            request_id=request_id,
            step_index=step_idx + 1,
            token_id=token_id,
            token_text=token_text,
            is_finished=is_finished,
            finish_reason=finish_reason,
            is_verified=True,
            is_experimental=False,
        )

        target_dest = (
            "coordinator"
            if (self.is_first_stage() and self.is_last_stage())
            else (getattr(self.assignment, "first_worker_id", None) or self.assignment.prev_worker_id)
        )
        return ProtocolFrame(
            type=MessageType.NEXT_TOKEN,
            source=self.assignment.worker_id,
            target=target_dest,
            payload=next_token.model_dump(),
        )

    def _verify_and_create_speculative_frame(self, cand_frame: SpeculativeCandidates, logits: Any) -> ProtocolFrame:
        request_id = cand_frame.request_id
        cfg = self.session_configs.get(request_id)
        temp = cfg.temperature if (cfg and cfg.temperature is not None) else 0.7
        top_p = cfg.top_p if (cfg and cfg.top_p is not None) else 0.9
        max_tokens = cfg.max_new_tokens if (cfg and cfg.max_new_tokens is not None) else 2048

        accepted_drafts, correction = self.verifier.verify_candidates(
            cand_frame.candidate_tokens,
            logits,
            temperature=temp,
            top_p=top_p,
        )
        rejected_draft_count = max(0, len(cand_frame.candidate_tokens) - len(accepted_drafts))
        accepted_tokens = list(accepted_drafts)
        if correction is not None:
            accepted_tokens.append(correction)
        elif not accepted_tokens and HAS_TORCH and isinstance(logits, torch.Tensor):
            target_logits = logits[0, -1, :] if logits.dim() == 3 else (logits[-1, :] if logits.dim() == 2 else logits)
            top_k = getattr(cfg, "top_k", 50) if cfg else 50
            sampled_tok = self.sampler.sample_next_token(target_logits, temperature=temp, top_p=top_p, top_k=top_k)
            accepted_tokens.append(sampled_tok)

        eos_id = getattr(self.sampler.tokenizer, "eos_token_id", None)
        stop_token_ids = list(cfg.stop_token_ids or []) if cfg else []
        tok_stops = getattr(self.sampler.tokenizer, "stop_token_ids", None)
        if tok_stops:
            for sid in tok_stops:
                if sid not in stop_token_ids:
                    stop_token_ids.append(int(sid))
        stop_strs = ("<eos>", "<end_of_turn>", "</s>", "<|endoftext|>", "<|eot_id|>", "<|im_end|>")
        is_eos = False
        for idx, tok in enumerate(accepted_tokens):
            tok_txt = self.sampler.decode_token(tok)
            if (stop_token_ids and tok in stop_token_ids) or (not stop_token_ids and eos_id is not None and tok == eos_id) or (tok_txt and tok_txt.strip() in stop_strs):
                accepted_tokens = accepted_tokens[: idx + 1]
                is_eos = True
                break

        new_step_idx = cand_frame.step_index + len(accepted_tokens)
        accepted_texts = [
            "" if (is_eos and i == len(accepted_tokens) - 1 and self.sampler.decode_token(t).strip() in stop_strs)
            else self.sampler.decode_token(t)
            for i, t in enumerate(accepted_tokens)
        ]
        is_finished = is_eos or (new_step_idx >= max_tokens)

        if is_finished:
            self.process_inference_finish(request_id)
        elif rejected_draft_count > 0:
            # On single-worker runs (where node is both first and last stage), defer KV cache trimming
            # to process_speculative_verified to prevent trimming valid context tokens twice.
            if not (self.is_first_stage() and self.is_last_stage()):
                self._trim_kv_caches(request_id, rejected_draft_count)

        verified = SpeculativeVerified(
            request_id=request_id,
            step_index=new_step_idx,
            accepted_tokens=accepted_tokens,
            accepted_texts=accepted_texts,
            rejected_draft_count=rejected_draft_count,
            is_finished=is_finished,
            finish_reason="stop" if is_eos else ("length" if is_finished else None),
            is_verified=True,
            is_experimental=False,
        )

        target_upstream = (
            "coordinator"
            if (self.is_first_stage() and self.is_last_stage())
            else (self.assignment.prev_worker_id or getattr(self.assignment, "first_worker_id", None))
        )
        return ProtocolFrame(
            type=MessageType.SPECULATIVE_VERIFIED,
            source=self.assignment.worker_id,
            target=target_upstream,
            payload=verified.model_dump(),
        )

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

                # Deserialize, map architecture, validate tensors, and install into model layer:
                self.stage_model.mark_layer_ready(layer_idx, weights=reassembled)
                status = "ready"
            except Exception as e:
                logger.error(f"Failed to validate and install layer {layer_idx}: {e}")
                reassembled = None
                status = "error"

            del self.layer_chunk_buffers[layer_idx]

            total_assigned = self.assignment.layer_end - self.assignment.layer_start
            ack = StageLayerAck(
                worker_id=self.assignment.worker_id,
                model_name=chunk.model_name,
                layer_index=layer_idx,
                status=status,
                ready_layer_count=self.stage_model.get_ready_count(),
                total_assigned=total_assigned,
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
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for model inference")
        if not self.is_first_stage():
            raise RuntimeError("start_inference called on a non-first stage worker")

        request_id = start_msg.request_id
        if self.cancellation_tokens.is_cancelled(request_id):
            return ProtocolFrame(
                type=MessageType.ERROR,
                source=self.assignment.worker_id,
                target="coordinator",
                payload={"request_id": request_id, "error": "Request cancelled prior to start"},
            )

        self.finished_sessions.discard(request_id)
        self.session_configs[request_id] = start_msg
        self.session_step_counts[request_id] = len(start_msg.input_ids)
        self.session_token_histories[request_id] = list(start_msg.input_ids)

        try:
            with _inference_context():
                # Memory check & reservation based on model KV architecture
                n_layers = max(1, self.assignment.layer_end - self.assignment.layer_start)
                n_kv_heads = getattr(self.stage_model, "num_kv_heads", getattr(self.stage_model, "n_local_kv_heads", 4))
                h_dim = getattr(self.stage_model, "head_dim", 64)
                max_new = min(getattr(start_msg, "max_new_tokens", 2048) or 2048, 1024) if start_msg else 1024
                expected_total_seq = len(start_msg.input_ids) + max_new
                est_bytes = 2 * n_layers * n_kv_heads * h_dim * expected_total_seq * 2
                try:
                    self.kv_cache_manager.allocate_request(request_id, estimated_bytes=est_bytes)
                except Exception as alloc_err:
                    logger.debug(f"kv_cache_manager allocation note for {request_id}: {alloc_err}")

                input_ids = torch.tensor([start_msg.input_ids], dtype=torch.long, device=self.stage_model.device)
                h, new_kvs = self._call_forward_first_stage(input_ids, past_kvs=None, position_offset=0)
                self.kv_cache_manager.set_all_kvs(request_id, new_kvs)
                self.session_kv_caches[request_id] = new_kvs

                if self.is_last_stage():
                    # Single worker holds all stages: h is already post-transformer layers.
                    # Compute logits directly without re-running layers in process_forward_activation!
                    logits = self._call_compute_logits(h, compute_all_logits=False)
                    return self._sample_next_token_from_logits(request_id, 0, logits)

                # Compress boundary activation
                comp_bytes, scale, shape = compress_tensor(h, method=self.compression_dtype)
                b64_payload = base64.b64encode(comp_bytes).decode("ascii")

                # Record in activation replay ring buffer for downstream fault tolerance
                self.replay_buffer.record(
                    request_id=request_id,
                    step_index=0,
                    payload_bytes=comp_bytes,
                    metadata={"shape": list(shape), "scale": scale, "dtype": self.compression_dtype},
                )

                forward_act = ForwardActivation(
                    request_id=request_id,
                    step_index=0,
                    shape=list(shape),
                    dtype=self.compression_dtype,
                    scale=scale,
                    payload_b64=b64_payload,
                    stage_index=self.assignment.layer_start,
                    expected_next_stage=self.assignment.layer_end,
                    position_offset=0,
                    is_verified=True,
                    is_experimental=False,
                )

                return ProtocolFrame(
                    type=MessageType.FORWARD_ACTIVATION,
                    source=self.assignment.worker_id,
                    target=self.assignment.next_worker_id,
                    payload=forward_act.model_dump(),
                )
        except Exception as e:
            logger.error(f"Error in start_inference for {request_id}: {e}", exc_info=True)
            self.abort_inference(request_id)
            return ProtocolFrame(
                type=MessageType.ERROR,
                source=self.assignment.worker_id,
                target="coordinator",
                payload={"request_id": request_id, "error": str(e)},
            )

    def process_forward_activation(self, act_frame: ForwardActivation) -> Optional[ProtocolFrame]:
        """
        Invoked on middle and last stages when activations arrive from previous stage.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for model inference")
        request_id = act_frame.request_id
        if self.cancellation_tokens.is_cancelled(request_id) or request_id in self.finished_sessions:
            return None

        # Verify distributed stage ordering invariants
        if not self.is_first_stage() and act_frame.expected_next_stage is not None:
            if act_frame.expected_next_stage != self.assignment.layer_start:
                raise StageOrderingError(
                    f"Stage ordering invariant violated: incoming activation targeted stage {act_frame.expected_next_stage}, "
                    f"but current worker executes stage [{self.assignment.layer_start}..{self.assignment.layer_end})"
                )

        try:
            with _inference_context():
                comp_bytes = base64.b64decode(act_frame.payload_b64)

                act_dtype = getattr(self.stage_model, "dtype", None) or torch.float32
                hidden_states = decompress_tensor(
                    comp_bytes,
                    scale=act_frame.scale,
                    shape=tuple(act_frame.shape),
                    method=act_frame.dtype,
                    device=self.stage_model.device,
                    dtype=act_dtype,
                )
                del comp_bytes

                if request_id not in self.session_kv_caches and request_id not in self.finished_sessions:
                    n_layers = max(1, self.assignment.layer_end - self.assignment.layer_start)
                    n_kv_heads = getattr(self.stage_model, "num_kv_heads", getattr(self.stage_model, "n_local_kv_heads", 4))
                    h_dim = getattr(self.stage_model, "head_dim", 64)
                    seq_len = act_frame.shape[1] if (len(act_frame.shape) > 1) else 1
                    cfg_session = self.session_configs.get(request_id)
                    max_new = min(cfg_session.max_new_tokens if (cfg_session and cfg_session.max_new_tokens is not None) else 2048, 1024)
                    est_bytes = 2 * n_layers * n_kv_heads * h_dim * (seq_len + max_new) * 2
                    try:
                        self.kv_cache_manager.allocate_request(request_id, estimated_bytes=est_bytes)
                    except Exception as alloc_err:
                        logger.debug(f"kv_cache_manager allocation note for {request_id}: {alloc_err}")

                past_kvs = self.kv_cache_manager.get_all_kvs(request_id) or self.session_kv_caches.get(request_id)
                pos_offset = act_frame.position_offset if act_frame.position_offset is not None else self.session_step_counts.get(request_id, 0)
                out, new_kvs = self._call_forward_intermediate_stage(hidden_states, past_kvs=past_kvs, position_offset=pos_offset)
                del hidden_states
                self.kv_cache_manager.set_all_kvs(request_id, new_kvs)
                self.session_kv_caches[request_id] = new_kvs

                if not self.is_last_stage():
                    # Middle stage: forward activations down the chain
                    comp_bytes, scale, shape = compress_tensor(out, method=self.compression_dtype)
                    b64_payload = base64.b64encode(comp_bytes).decode("ascii")

                    # Record in activation replay buffer
                    self.replay_buffer.record(
                        request_id=request_id,
                        step_index=act_frame.step_index,
                        payload_bytes=comp_bytes,
                        metadata={"shape": list(shape), "scale": scale, "dtype": self.compression_dtype},
                    )

                    forward_act = ForwardActivation(
                        request_id=request_id,
                        step_index=act_frame.step_index,
                        shape=list(shape),
                        dtype=self.compression_dtype,
                        scale=scale,
                        payload_b64=b64_payload,
                        stage_index=self.assignment.layer_start,
                        expected_next_stage=self.assignment.layer_end,
                        position_offset=act_frame.position_offset,
                        is_verified=True,
                        is_experimental=False,
                    )
                    return ProtocolFrame(
                        type=MessageType.FORWARD_ACTIVATION,
                        source=self.assignment.worker_id,
                        target=self.assignment.next_worker_id,
                        payload=forward_act.model_dump(),
                    )
                else:
                    # Last stage: compute next token from logits!
                    return self._sample_next_token_from_logits(request_id, act_frame.step_index, out)
        except Exception as e:
            logger.error(f"Error in process_forward_activation for {request_id}: {e}", exc_info=True)
            self.abort_inference(request_id)
            return ProtocolFrame(
                type=MessageType.ERROR,
                source=self.assignment.worker_id,
                target="coordinator",
                payload={"request_id": request_id, "error": str(e)},
            )

    def _trim_kv_caches(self, request_id: str, trim_tail_tokens: int) -> None:
        """Discard rejected speculative draft positions from all layer KV caches."""
        if trim_tail_tokens <= 0:
            return
        kvs = self.kv_cache_manager.get_all_kvs(request_id) or self.session_kv_caches.get(request_id)
        if not kvs:
            return
        trimmed = []
        for kv in kvs:
            if isinstance(kv, (tuple, list)) and len(kv) == 2 and HAS_TORCH and isinstance(kv[0], torch.Tensor):
                keep_len = max(0, kv[0].shape[-2] - trim_tail_tokens)
                trimmed.append((kv[0][..., :keep_len, :], kv[1][..., :keep_len, :]))
            else:
                trimmed.append(kv)
        self.kv_cache_manager.set_all_kvs(request_id, trimmed)
        self.session_kv_caches[request_id] = trimmed

    def process_speculative_candidates(self, cand_frame: SpeculativeCandidates) -> Optional[ProtocolFrame]:
        """
        Invoked on Worker B to verify a drafted multi-token candidate batch in a single forward pass.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for model inference")
        request_id = cand_frame.request_id
        if request_id in self.finished_sessions:
            return None

        with _inference_context():
            comp_bytes = base64.b64decode(cand_frame.payload_b64)

            act_dtype = getattr(self.stage_model, "dtype", None) or torch.float32
            hidden_states = decompress_tensor(
                comp_bytes,
                scale=cand_frame.scale,
                shape=tuple(cand_frame.shape),
                method=cand_frame.dtype,
                device=self.stage_model.device,
                dtype=act_dtype,
            )
            del comp_bytes
            past_kvs = self.kv_cache_manager.get_all_kvs(request_id) or self.session_kv_caches.get(request_id)
            pos_offset = getattr(cand_frame, "position_offset", None)
            if pos_offset is None:
                pos_offset = self.session_step_counts.get(request_id, 0)
            out, new_kvs = self._call_forward_intermediate_stage(
                hidden_states, past_kvs=past_kvs, position_offset=pos_offset, compute_all_logits=True
            )
            del hidden_states
            self.kv_cache_manager.set_all_kvs(request_id, new_kvs)
            self.session_kv_caches[request_id] = new_kvs

        if not self.is_last_stage():
            # Middle stage: forward intermediate candidate activations down the chain
            comp_bytes, scale, shape = compress_tensor(out, method=self.compression_dtype)
            b64_payload = base64.b64encode(comp_bytes).decode("ascii")
            cand_frame.payload_b64 = b64_payload
            cand_frame.scale = scale
            cand_frame.shape = list(shape)
            cand_frame.dtype = self.compression_dtype
            return ProtocolFrame(
                type=MessageType.SPECULATIVE_CANDIDATES,
                source=self.assignment.worker_id,
                target=self.assignment.next_worker_id,
                payload=cand_frame.model_dump(),
            )

        return self._verify_and_create_speculative_frame(cand_frame, out)

    def process_speculative_verified(self, verified_frame: SpeculativeVerified) -> Optional[ProtocolFrame]:
        """
        Invoked on Worker A (and intermediate stages) when downstream stage accepts speculative tokens.
        Appends verified tokens to history and triggers the next draft/decode step.
        """
        request_id = verified_frame.request_id
        if request_id in self.finished_sessions:
            return None

        # Discard rejected speculative draft tokens from all layer KV caches on this stage
        rej_cnt = getattr(verified_frame, "rejected_draft_count", 0) or 0
        if rej_cnt > 0:
            self._trim_kv_caches(request_id, rej_cnt)

        # Monotonic sequence position tracking: calculate true cumulative position offset
        # rather than overwriting with verified_frame.step_index (which tracks generated output steps)
        cfg = self.session_configs.get(request_id)
        prompt_len = len(cfg.input_ids) if (cfg and getattr(cfg, "input_ids", None)) else 0
        hist = self.session_token_histories.get(request_id, [])
        if hist and prompt_len > 0 and len(hist) >= prompt_len and hist[:prompt_len] == list(cfg.input_ids):
            hist_len = len(hist)
        else:
            hist_len = prompt_len + len(hist)
        acc_tokens = getattr(verified_frame, "accepted_tokens", None) or []
        added_prefix = max(0, len(acc_tokens) - 1)
        self.session_step_counts[request_id] = hist_len + added_prefix

        if not self.is_first_stage():
            if acc_tokens:
                if request_id not in self.session_token_histories:
                    self.session_token_histories[request_id] = []
                self.session_token_histories[request_id].extend(acc_tokens)
            if verified_frame.is_finished:
                self.process_inference_finish(request_id)
            target_upstream = self.assignment.prev_worker_id or getattr(self.assignment, "first_worker_id", None)
            if target_upstream and target_upstream != self.assignment.worker_id:
                return ProtocolFrame(
                    type=MessageType.SPECULATIVE_VERIFIED,
                    source=self.assignment.worker_id,
                    target=target_upstream,
                    payload=verified_frame.model_dump(),
                )
            return None

        if verified_frame.is_finished:
            self.process_inference_finish(request_id)
            return None

        if not verified_frame.accepted_tokens:
            return None

        # Append accepted prefix tokens to history (last accepted token will be appended by process_next_token)
        if request_id in self.session_token_histories:
            self.session_token_histories[request_id].extend(verified_frame.accepted_tokens[:-1])

        # Take last accepted token to continue decoding
        last_token = verified_frame.accepted_tokens[-1]
        next_token_frame = NextToken(
            request_id=request_id,
            step_index=verified_frame.step_index,
            token_id=last_token,
            is_verified=verified_frame.is_verified,
            is_experimental=verified_frame.is_experimental,
        )
        return self.process_next_token(next_token_frame)

    def process_next_token(self, token_frame: NextToken) -> Optional[ProtocolFrame]:
        """
        Invoked on the first pipeline stage when the last stage emits the next token ID.
        Worker A embeds the single token and sends intermediate activation to continue the cycle!
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for model inference")
        if not self.is_first_stage():
            first_target = getattr(self.assignment, "first_worker_id", None)
            if first_target and first_target != self.assignment.worker_id:
                return ProtocolFrame(
                    type=MessageType.NEXT_TOKEN,
                    source=self.assignment.worker_id,
                    target=first_target,
                    payload=token_frame.model_dump(),
                )
            return None

        request_id = token_frame.request_id
        if request_id in self.finished_sessions:
            return None

        if token_frame.is_finished:
            # Generation finished, clean up local KV cache
            self.kv_cache_manager.release_request(request_id)
            self.session_kv_caches.pop(request_id, None)
            self.session_configs.pop(request_id, None)
            self.session_token_histories.pop(request_id, None)
            self.session_step_counts.pop(request_id, None)
            self.finished_sessions.add(request_id)
            return None

        # Track history
        if request_id not in self.session_token_histories:
            self.session_token_histories[request_id] = []
        self.session_token_histories[request_id].append(token_frame.token_id)

        try:
            with _inference_context():
                curr_pos = self.session_step_counts.get(request_id)
                if curr_pos is None:
                    past_kvs = self.kv_cache_manager.get_all_kvs(request_id) or self.session_kv_caches.get(request_id)
                    curr_pos = past_kvs[0][0].shape[-2] if (past_kvs and isinstance(past_kvs[0], (tuple, list)) and len(past_kvs[0]) == 2 and isinstance(past_kvs[0][0], torch.Tensor)) else 0

                # 1. Speculative Path: check for candidate n-grams
                sys_len = getattr(self.session_configs.get(request_id), "system_prompt_len", None)
                candidates = (
                    self.prompt_drafter.draft_candidates(self.session_token_histories[request_id], system_prompt_len=sys_len)
                    if self.enable_speculative
                    else []
                )
                if candidates:
                    input_tensor = torch.tensor([[token_frame.token_id] + candidates], dtype=torch.long, device=self.stage_model.device)
                    past_kvs = self.kv_cache_manager.get_all_kvs(request_id) or self.session_kv_caches.get(request_id)
                    h, new_kvs = self._call_forward_first_stage(input_tensor, past_kvs=past_kvs, position_offset=curr_pos)
                    self.session_step_counts[request_id] = curr_pos + input_tensor.shape[1]
                    self.kv_cache_manager.set_all_kvs(request_id, new_kvs)
                    self.session_kv_caches[request_id] = new_kvs

                    if self.is_last_stage():
                        logits = self._call_compute_logits(h, compute_all_logits=True)
                        cand_frame = SpeculativeCandidates(
                            request_id=request_id,
                            step_index=token_frame.step_index,
                            candidate_tokens=candidates,
                            position_offset=curr_pos,
                            shape=list(h.shape),
                            dtype=self.compression_dtype,
                            scale=1.0,
                            payload_b64="",
                            is_verified=True,
                            is_experimental=False,
                        )
                        return self._verify_and_create_speculative_frame(cand_frame, logits)

                    comp_bytes, scale, shape = compress_tensor(h, method=self.compression_dtype)
                    b64_payload = base64.b64encode(comp_bytes).decode("ascii")
                    del h, input_tensor

                    spec_cand = SpeculativeCandidates(
                        request_id=request_id,
                        step_index=token_frame.step_index,
                        candidate_tokens=candidates,
                        position_offset=curr_pos,
                        shape=list(shape),
                        dtype=self.compression_dtype,
                        scale=scale,
                        payload_b64=b64_payload,
                        is_verified=True,
                        is_experimental=False,
                    )
                    return ProtocolFrame(
                        type=MessageType.SPECULATIVE_CANDIDATES,
                        source=self.assignment.worker_id,
                        target=self.assignment.next_worker_id,
                        payload=spec_cand.model_dump(),
                    )

                # 2. Standard Single-Token Path
                input_tensor = torch.tensor([[token_frame.token_id]], dtype=torch.long, device=self.stage_model.device)
                past_kvs = self.kv_cache_manager.get_all_kvs(request_id) or self.session_kv_caches.get(request_id)
                h, new_kvs = self._call_forward_first_stage(input_tensor, past_kvs=past_kvs, position_offset=curr_pos)
                self.session_step_counts[request_id] = curr_pos + input_tensor.shape[1]
                self.kv_cache_manager.set_all_kvs(request_id, new_kvs)
                self.session_kv_caches[request_id] = new_kvs

                if self.is_last_stage():
                    logits = self._call_compute_logits(h, compute_all_logits=False)
                    return self._sample_next_token_from_logits(request_id, token_frame.step_index, logits)

                comp_bytes, scale, shape = compress_tensor(h, method=self.compression_dtype)
                b64_payload = base64.b64encode(comp_bytes).decode("ascii")
                del h, input_tensor

                forward_act = ForwardActivation(
                    request_id=request_id,
                    step_index=token_frame.step_index,
                    position_offset=curr_pos,
                    shape=list(shape),
                    dtype=self.compression_dtype,
                    scale=scale,
                    payload_b64=b64_payload,
                    stage_index=self.assignment.layer_start,
                    expected_next_stage=self.assignment.layer_end,
                    is_verified=True,
                    is_experimental=False,
                )

                return ProtocolFrame(
                    type=MessageType.FORWARD_ACTIVATION,
                    source=self.assignment.worker_id,
                    target=self.assignment.next_worker_id,
                    payload=forward_act.model_dump(),
                )
        except Exception as e:
            logger.error(f"Error in process_next_token for {request_id}: {e}", exc_info=True)
            self.abort_inference(request_id)
            return ProtocolFrame(
                type=MessageType.ERROR,
                source=self.assignment.worker_id,
                target="coordinator",
                payload={"request_id": request_id, "error": str(e)},
            )

    def process_inference_finish(self, request_id: str) -> None:
        """Frees KV cache and session state when an inference request completes or aborts."""
        self.kv_cache_manager.release_request(request_id)
        self.session_kv_caches.pop(request_id, None)
        self.session_configs.pop(request_id, None)
        self.session_token_histories.pop(request_id, None)
        self.session_step_counts.pop(request_id, None)
        self.replay_buffer.clear(request_id)
        self.finished_sessions.add(request_id)
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.debug(f"Freed session {request_id} KV cache and state.")

    def abort_inference(self, request_id: str):
        """Cancels an active session and cleans up session KV cache and history."""
        self.cancellation_tokens.cancel(request_id)
        self.process_inference_finish(request_id)
        logger.info(f"Aborted session {request_id} and cleared session state.")
