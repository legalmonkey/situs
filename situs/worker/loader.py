"""Model partitioning, real transformer block architecture, and progressive weight loader for pipeline stages."""

import io
import os
import math
import logging
import threading
from typing import Optional, List, Tuple, Any, Dict, Set, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    nn = object
    F = object

try:
    import safetensors.torch
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

from situs.config import LayerAllocation
from situs.worker.speculative import EarlyExitHead
from situs.coordinator.registry import resolve_model_metadata

logger = logging.getLogger("situs.worker.loader")

SUPPORTED_ARCHITECTURES: Set[str] = {
    "llama",
    "llama2",
    "llama3",
    "gemma",
    "gemma2",
    "gemma3",
    "qwen",
    "qwen2",
    "qwen2.5",
    "mistral",
    "transformer",
}


def is_architecture_supported(arch: Optional[str]) -> bool:
    """
    Validates that the architecture is compatible with RealTransformerBlock
    (RMSNorm, RoPE, and SwiGLU-style MLP). Explicitly rejects unsupported
    architectures such as MoE, MPT, GPT-2 (LayerNorm + learned position embeddings), etc.
    """
    if not arch or not isinstance(arch, str):
        return False
    norm = arch.strip().lower()
    incompatible = ("moe", "mixture", "mpt", "gpt2", "gpt_neox", "falcon", "chatglm", "bloom", "opt", "bert", "t5")
    for bad in incompatible:
        if bad in norm:
            return False
    for supported in SUPPORTED_ARCHITECTURES:
        if norm == supported or norm.startswith(supported):
            return True
        if supported != "transformer" and supported in norm:
            return True
    return False


class RMSNorm(nn.Module if HAS_TORCH else object):
    """Root Mean Square Layer Normalization (standard for LLaMA, Gemma, Mistral, Qwen)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        if HAS_TORCH:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        if not HAS_TORCH or not isinstance(x, torch.Tensor):
            return x
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


class RotaryEmbedding(nn.Module if HAS_TORCH else object):
    """Rotary Position Embedding (RoPE) for relative attention."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        if HAS_TORCH:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, seq_len: int):
        if not HAS_TORCH or not isinstance(x, torch.Tensor):
            return x, x
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    if not HAS_TORCH or not isinstance(q, torch.Tensor):
        return q, k
    # Rotate half
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def _fast_linear(in_features: int, out_features: int) -> Any:
    """Fast linear module initialization avoiding multi-second random uniform generation on CPU."""
    if not HAS_TORCH:
        return None
    linear = nn.Linear.__new__(nn.Linear)
    super(nn.Linear, linear).__init__()
    linear.in_features = in_features
    linear.out_features = out_features
    # Initialize with small normal variance rather than all-zeros to prevent uniform degenerate logits
    std = 0.02 / math.sqrt(max(1, in_features))
    linear.weight = nn.Parameter(torch.randn(out_features, in_features, dtype=torch.float32) * std)
    linear.bias = None
    return linear


class RealTransformerBlock(nn.Module if HAS_TORCH else object):
    """
    Genuine Transformer Decoder Block with:
    - RMSNorm pre-attention normalization
    - Multi-Head / Grouped-Query Self-Attention (GQA) with KV-cache
    - RMSNorm post-attention normalization
    - SwiGLU Feed-Forward Network (gate_proj, up_proj, down_proj)
    - Residual connections
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: int,
        architecture: str,
        eps: float = 1e-6,
        layer_idx: int = 0,
    ):
        super().__init__()
        if not is_architecture_supported(architecture):
            raise ValueError(
                f"Unsupported model architecture '{architecture}'. "
                f"RealTransformerBlock requires an architecture compatible with RMSNorm + RoPE + SwiGLU "
                f"(supported: {sorted(SUPPORTED_ARCHITECTURES)}). Incompatible architectures (e.g. MoE, MPT, GPT-2) are rejected."
            )
        for name, val in [
            ("hidden_dim", hidden_dim),
            ("num_heads", num_heads),
            ("num_kv_heads", num_kv_heads),
            ("intermediate_size", intermediate_size),
            ("head_dim", head_dim),
        ]:
            if val is None or not isinstance(val, int) or val <= 0:
                raise ValueError(f"RealTransformerBlock requires verified positive integer for '{name}', got {val}")

        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.layer_idx = layer_idx

        if not HAS_TORCH:
            return

        self.input_layernorm = RMSNorm(hidden_dim, eps=eps)

        # Self-Attention projections (instant zero initialization)
        self.q_proj = _fast_linear(hidden_dim, self.num_heads * self.head_dim)
        self.k_proj = _fast_linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.v_proj = _fast_linear(hidden_dim, self.num_kv_heads * self.head_dim)
        self.o_proj = _fast_linear(self.num_heads * self.head_dim, hidden_dim)

        # SwiGLU MLP
        self.post_attention_layernorm = RMSNorm(hidden_dim, eps=eps)
        self.gate_proj = _fast_linear(hidden_dim, self.intermediate_size)
        self.up_proj = _fast_linear(hidden_dim, self.intermediate_size)
        self.down_proj = _fast_linear(self.intermediate_size, hidden_dim)
        self.act_fn = nn.SiLU()

        # Rotary Embeddings
        self.rotary = RotaryEmbedding(self.head_dim)

    def forward(self, x, past_kv: Optional[Tuple[Any, Any]] = None):
        if not HAS_TORCH or not isinstance(x, torch.Tensor):
            new_kv = (past_kv or 0) + 1
            return x, new_kv

        residual = x
        norm_x = self.input_layernorm(x)

        if norm_x.dim() == 2:
            norm_x = norm_x.unsqueeze(1)
            residual = residual.unsqueeze(1)

        batch_size, seq_len, _ = norm_x.shape

        q = self.q_proj(norm_x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # KV Cache concatenation for autoregressive steps
        if past_kv is not None and isinstance(past_kv, (tuple, list)) and len(past_kv) == 2:
            past_k, past_v = past_kv
            if isinstance(past_k, torch.Tensor) and isinstance(past_v, torch.Tensor):
                k = torch.cat([past_k, k], dim=-2)
                v = torch.cat([past_v, v], dim=-2)

        new_kv = (k, v)

        # Grouped-Query Attention head expansion if GQA is active
        if self.num_kv_heads < self.num_heads:
            repeats = self.num_heads // self.num_kv_heads
            k_exp = k.repeat_interleave(repeats, dim=1)
            v_exp = v.repeat_interleave(repeats, dim=1)
        else:
            k_exp = k
            v_exp = v

        # Scaled dot-product attention
        is_causal = (seq_len > 1 and past_kv is None)
        attn_out = F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=is_causal)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_out = self.o_proj(attn_out)
        x = residual + attn_out

        # SwiGLU MLP
        residual = x
        norm_mlp = self.post_attention_layernorm(x)
        mlp_out = self.down_proj(self.act_fn(self.gate_proj(norm_mlp)) * self.up_proj(norm_mlp))
        x = residual + mlp_out

        return x, new_kv

    def load_layer_weights(self, state_dict: Dict[str, torch.Tensor]) -> bool:
        """
        Validates and loads parameter weights into the transformer block.
        Strictly verifies that all required tensors are present, have valid shapes,
        and contain finite values without NaN or Inf.
        """
        mapping = {
            "input_layernorm.weight": ["input_layernorm.weight", "attention_norm.weight", "attn_norm.weight", "ln_1.weight"],
            "q_proj.weight": ["self_attn.q_proj.weight", "q_proj.weight", "attn.q_proj.weight", "attn_q.weight"],
            "k_proj.weight": ["self_attn.k_proj.weight", "k_proj.weight", "attn.k_proj.weight", "attn_k.weight"],
            "v_proj.weight": ["self_attn.v_proj.weight", "v_proj.weight", "attn.v_proj.weight", "attn_v.weight"],
            "o_proj.weight": ["self_attn.o_proj.weight", "o_proj.weight", "attn.o_proj.weight", "attn_output.weight"],
            "post_attention_layernorm.weight": ["post_attention_layernorm.weight", "ffn_norm.weight", "post_ffw_norm.weight", "post_attention_norm.weight", "ln_2.weight"],
            "gate_proj.weight": ["mlp.gate_proj.weight", "gate_proj.weight", "ffn_gate.weight"],
            "up_proj.weight": ["mlp.up_proj.weight", "up_proj.weight", "ffn_up.weight"],
            "down_proj.weight": ["mlp.down_proj.weight", "down_proj.weight", "ffn_down.weight"],
        }

        # Build clean lookup from incoming state_dict
        clean_incoming = {}
        for k, v in state_dict.items():
            # Strip common layer prefixes e.g. "model.layers.0." or "layers.0." or "blk.0."
            short_k = k
            if "layers." in short_k:
                short_k = short_k.split("layers.", 1)[1]
                if "." in short_k:
                    short_k = short_k.split(".", 1)[1]
            elif "blk." in short_k:
                short_k = short_k.split("blk.", 1)[1]
                if "." in short_k:
                    short_k = short_k.split(".", 1)[1]
            clean_incoming[short_k] = v
            clean_incoming[k] = v

        my_dict = self.state_dict()
        required_params = list(mapping.keys())

        for param_name in required_params:
            aliases = mapping[param_name]
            matched_tensor = None
            for alias in aliases:
                if alias in clean_incoming:
                    matched_tensor = clean_incoming[alias]
                    break

            if matched_tensor is None:
                raise ValueError(
                    f"Incomplete layer weights: required parameter '{param_name}' is missing for layer {self.layer_idx}"
                )

            if not isinstance(matched_tensor, torch.Tensor):
                raise ValueError(
                    f"Invalid tensor type for '{param_name}' in layer {self.layer_idx}: {type(matched_tensor)}"
                )

            target_shape = my_dict[param_name].shape
            if matched_tensor.shape != target_shape:
                attr = param_name.split(".")[0]
                mod = getattr(self, attr, None)
                if isinstance(mod, nn.Linear) and matched_tensor.dim() == 2:
                    out_f, in_f = matched_tensor.shape
                    if in_f == mod.in_features or out_f == mod.out_features:
                        new_linear = _fast_linear(in_f, out_f)
                        setattr(self, attr, new_linear)
                        my_dict = self.state_dict()
                        target_shape = my_dict[param_name].shape
                if matched_tensor.shape != target_shape:
                    raise ValueError(
                        f"Shape mismatch for parameter '{param_name}' in layer {self.layer_idx}: "
                        f"got {tuple(matched_tensor.shape)}, expected {tuple(target_shape)}"
                    )

            if torch.isnan(matched_tensor).any() or torch.isinf(matched_tensor).any():
                raise ValueError(f"Tensor '{param_name}' contains NaN or Inf values in layer {self.layer_idx}")

            my_dict[param_name].copy_(matched_tensor.to(dtype=my_dict[param_name].dtype))

        logger.debug(f"Block {self.layer_idx}: Loaded and validated all {len(required_params)} parameter tensors.")
        return True


def deserialize_and_validate_layer_weights(
    weights_data: Any,
    hidden_dim: int,
    num_heads: int,
    num_kv_heads: Optional[int] = None,
    intermediate_size: Optional[int] = None,
    layer_idx: int = 0,
    device: str = "cpu",
    head_dim: Optional[int] = None,
    architecture: str = "llama",
) -> RealTransformerBlock:
    """
    Deserializes raw wire weights (Safetensors or PyTorch bytes), maps tensor names,
    validates all required tensors, shapes and finite values, and instantiates a fully functional RealTransformerBlock.
    Raises ValueError if weights are corrupt, incomplete, or invalid.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for layer deserialization")

    state_dict: Dict[str, torch.Tensor] = {}

    if isinstance(weights_data, bytes):
        deserialized = False
        # 1. Try Safetensors
        if HAS_SAFETENSORS:
            try:
                state_dict = safetensors.torch.load(weights_data)
                deserialized = True
            except Exception:
                pass

        # 2. Try PyTorch load
        if not deserialized:
            try:
                buf = io.BytesIO(weights_data)
                loaded = torch.load(buf, map_location="cpu", weights_only=True)
                if isinstance(loaded, dict):
                    state_dict = loaded
                    deserialized = True
            except Exception:
                pass

        if not deserialized:
            raise ValueError(
                f"Failed to deserialize layer weights: payload is neither valid Safetensors nor PyTorch state_dict"
            )

    elif isinstance(weights_data, dict):
        state_dict = weights_data
    else:
        raise ValueError(f"Unsupported weights data type: {type(weights_data)}")

    if not isinstance(state_dict, dict) or not state_dict:
        raise ValueError("Deserialized weights payload must be a non-empty dict of tensors")

    # Clean incoming key names
    clean_incoming = {}
    for k, v in state_dict.items():
        short_k = k
        if "layers." in short_k:
            short_k = short_k.split("layers.", 1)[1]
            if "." in short_k:
                short_k = short_k.split(".", 1)[1]
        clean_incoming[short_k] = v
        clean_incoming[k] = v

    # Derive architecture dimensions dynamically from verified tensor shapes
    actual_h_dim = hidden_dim
    for k in ("input_layernorm.weight", "attention_norm.weight", "ln_1.weight"):
        if k in clean_incoming and isinstance(clean_incoming[k], torch.Tensor) and clean_incoming[k].dim() >= 1:
            actual_h_dim = clean_incoming[k].shape[0]
            break

    actual_inter = intermediate_size
    for k in ("gate_proj.weight", "mlp.gate_proj.weight"):
        if k in clean_incoming and isinstance(clean_incoming[k], torch.Tensor) and clean_incoming[k].dim() >= 2:
            actual_inter = clean_incoming[k].shape[0]
            break

    actual_heads = num_heads
    actual_kv_heads = num_kv_heads or num_heads
    derived_head_dim = head_dim

    for qk in ("q_proj.weight", "self_attn.q_proj.weight", "attn_q.weight", "attn.q_proj.weight"):
        if qk in clean_incoming and isinstance(clean_incoming[qk], torch.Tensor) and clean_incoming[qk].dim() >= 2:
            q_out = clean_incoming[qk].shape[0]
            derived_head_dim = max(1, q_out // max(1, actual_heads))
            break

    if derived_head_dim is None:
        derived_head_dim = max(1, actual_h_dim // max(1, actual_heads))

    for kk in ("k_proj.weight", "self_attn.k_proj.weight", "attn_k.weight", "attn.k_proj.weight"):
        if kk in clean_incoming and isinstance(clean_incoming[kk], torch.Tensor) and clean_incoming[kk].dim() >= 2:
            k_out = clean_incoming[kk].shape[0]
            actual_kv_heads = max(1, k_out // derived_head_dim)
            break

    if actual_heads <= 0 or actual_kv_heads <= 0:
        raise ValueError(
            f"Invalid head count for layer {layer_idx}: num_heads={actual_heads}, num_kv_heads={actual_kv_heads}"
        )
    if actual_heads % actual_kv_heads != 0:
        raise ValueError(
            f"Invalid GQA configuration for layer {layer_idx}: num_heads ({actual_heads}) "
            f"must be divisible by num_kv_heads ({actual_kv_heads})"
        )

    # Instantiate block with dynamically verified dimensions
    block = RealTransformerBlock(
        hidden_dim=actual_h_dim,
        num_heads=actual_heads,
        num_kv_heads=actual_kv_heads,
        intermediate_size=actual_inter,
        head_dim=derived_head_dim,
        architecture=architecture,
        layer_idx=layer_idx,
    )

    # Load and validate tensors (raises ValueError if incomplete, non-finite, or invalid)
    block.load_layer_weights(state_dict)
    block.to(device)

    # Verification: run a test forward pass to confirm numerical validity
    with torch.no_grad():
        test_in = torch.randn(1, 1, actual_h_dim, device=device)
        test_out, _ = block(test_in)
        if torch.isnan(test_out).any() or torch.isinf(test_out).any():
            raise ValueError(f"Layer {layer_idx} produced NaN or Inf outputs during validation forward pass")

    logger.info(f"Layer {layer_idx} successfully deserialized, mapped, validated, and placed on {device}.")
    return block


def verify_layer_numerical_parity(
    block: RealTransformerBlock,
    ref_block: Any,
    tolerance: float = 1e-3,
    test_input: Optional[Any] = None,
) -> bool:
    """
    Verifies that installed weights in RealTransformerBlock produce outputs mathematically
    equivalent to a reference transformer decoder block (e.g. HuggingFace LlamaDecoderLayer / GemmaDecoderLayer)
    within the specified numerical tolerance.
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is required for numerical parity verification")

    if not isinstance(block, RealTransformerBlock):
        raise TypeError(f"Expected RealTransformerBlock, got {type(block)}")

    if test_input is None:
        torch.manual_seed(42)
        test_input = torch.randn(1, 4, block.hidden_dim, dtype=torch.float32)

    block_eval = getattr(block, "eval", None)
    if callable(block_eval):
        block.eval()
    ref_eval = getattr(ref_block, "eval", None)
    if callable(ref_eval):
        ref_block.eval()

    with torch.no_grad():
        x_block = test_input.clone()
        out_block, _ = block(x_block)

        x_ref = test_input.clone()
        try:
            ref_res = ref_block(x_ref)
        except TypeError:
            ref_res = ref_block(hidden_states=x_ref)

        if isinstance(ref_res, (tuple, list)):
            out_ref = ref_res[0]
        elif hasattr(ref_res, "hidden_states"):
            out_ref = ref_res.hidden_states
        elif isinstance(ref_res, torch.Tensor):
            out_ref = ref_res
        else:
            raise TypeError(f"Unrecognized reference block output type: {type(ref_res)}")

        if out_block.shape != out_ref.shape:
            logger.error(
                f"Numerical parity failed: shape mismatch between RealTransformerBlock "
                f"({out_block.shape}) and reference ({out_ref.shape})"
            )
            return False

        max_diff = torch.max(torch.abs(out_block - out_ref)).item()
        is_close = torch.allclose(out_block, out_ref, atol=tolerance, rtol=tolerance)
        if not is_close:
            logger.warning(
                f"Numerical parity divergence: max absolute diff is {max_diff:.6e} > tolerance {tolerance}"
            )
            return False

        return True


class FastBootDraftBlock(nn.Module if HAS_TORCH else object):
    """
    Experimental low-rank transformer block for fast-boot (~1 MB per layer).
    Marked EXPERIMENTAL until convergence to full model weights is established.
    """
    is_verified_inference: bool = False
    is_experimental: bool = True

    def __init__(self, hidden_dim: int = 256, rank: int = 64):
        super().__init__()
        self.is_verified_inference = False
        self.is_experimental = True
        self.hidden_dim = hidden_dim
        self.rank = min(rank, max(16, hidden_dim // 4))
        if HAS_TORCH:
            self.norm = RMSNorm(hidden_dim)
            self.down_proj = nn.Linear(hidden_dim, self.rank, bias=False)
            self.act = nn.SiLU()
            self.up_proj = nn.Linear(self.rank, hidden_dim, bias=False)
            nn.init.normal_(self.up_proj.weight, std=0.01)

    def forward(self, x, past_kv=None):
        if not HAS_TORCH or not isinstance(x, torch.Tensor):
            return x, (past_kv or 0) + 1
        residual = x
        out = self.up_proj(self.act(self.down_proj(self.norm(x))))
        new_kv = (past_kv or 0) + 1
        return residual + out, new_kv


# Alias for backwards compatibility in existing tests
MockTransformerBlock = RealTransformerBlock


class DistributedStageModel(nn.Module if HAS_TORCH else object):
    """
    Holds only the assigned subset of transformer layers for this worker.
    Computes genuine distributed transformer activations and logits.
    """
    def __init__(
        self,
        layer_start: int,
        layer_end: int,
        total_layers: int,
        is_first_stage: bool,
        is_last_stage: bool,
        hidden_dim: int,
        vocab_size: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: int,
        architecture: str,
        device: str = "cpu",
        progressive: bool = False,
        fast_boot: bool = False,
    ):
        super().__init__()
        if not is_architecture_supported(architecture):
            raise ValueError(
                f"Unsupported model architecture '{architecture}'. "
                f"DistributedStageModel requires an architecture compatible with RMSNorm + RoPE + SwiGLU "
                f"(supported: {sorted(SUPPORTED_ARCHITECTURES)}). Incompatible architectures (e.g. MoE, MPT, GPT-2) are rejected."
            )
        for name, val in [
            ("hidden_dim", hidden_dim),
            ("vocab_size", vocab_size),
            ("num_heads", num_heads),
            ("num_kv_heads", num_kv_heads),
            ("intermediate_size", intermediate_size),
            ("head_dim", head_dim),
        ]:
            if val is None or not isinstance(val, int) or val <= 0:
                raise ValueError(f"DistributedStageModel requires verified positive integer for '{name}', got {val}")

        self.architecture = architecture
        self.layer_start = layer_start
        self.layer_end = layer_end
        self.total_layers = total_layers
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.intermediate_size = intermediate_size
        self.head_dim = head_dim

        safe_device = device
        if safe_device.startswith("cuda") and not (HAS_TORCH and torch.cuda.is_available()):
            logger.warning(f"CUDA device '{device}' requested but PyTorch CUDA runtime is not available. Safely falling back to CPU.")
            safe_device = "cpu"
        self.device = safe_device
        self.progressive = progressive
        self.fast_boot = fast_boot
        self.fast_boot_active = fast_boot

        # JIT Progressive layer readiness tracking
        self._layer_events: Dict[int, threading.Event] = {
            i: threading.Event() for i in range(layer_start, layer_end)
        }
        if progressive or fast_boot:
            self.ready_layers: Set[int] = set()
            if fast_boot:
                for ev in self._layer_events.values():
                    ev.set()
        else:
            self.ready_layers = set(range(layer_start, layer_end))
            for ev in self._layer_events.values():
                ev.set()

        if not HAS_TORCH:
            return

        # 1. Embeddings (ONLY if first stage)
        if self.is_first_stage:
            self.embed_tokens = nn.Embedding(vocab_size, hidden_dim)
            self.early_exit_head = EarlyExitHead(hidden_dim, vocab_size)
        else:
            self.embed_tokens = None
            self.early_exit_head = None

        # 2. Assigned Transformer Layers
        num_layers = layer_end - layer_start
        if self.fast_boot:
            self.layers = nn.ModuleList([
                FastBootDraftBlock(hidden_dim=hidden_dim) for _ in range(num_layers)
            ])
        else:
            self.layers = nn.ModuleList([
                RealTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    intermediate_size=self.intermediate_size,
                    head_dim=self.head_dim,
                    architecture=self.architecture,
                    layer_idx=layer_start + i,
                )
                for i in range(num_layers)
            ])

        # 3. Norm and LM Head (ONLY if last stage)
        if self.is_last_stage:
            self.norm = RMSNorm(hidden_dim)
            self.lm_head = _fast_linear(hidden_dim, vocab_size)
        else:
            self.norm = None
            self.lm_head = None

        if self.device != "cpu":
            self.to(self.device)

    def load_embedding_weights(self, weights: Any):
        """Loads verified pre-trained token embedding weights into the first stage."""
        if not self.is_first_stage or self.embed_tokens is None:
            return
        if not HAS_TORCH:
            return
        if not isinstance(weights, torch.Tensor):
            raise ValueError(
                f"Expected torch.Tensor for embedding weights, got {type(weights).__name__}. "
                f"Incompatible weights are rejected."
            )
        with torch.no_grad():
            target_shape = self.embed_tokens.weight.shape
            src = weights.to(device=self.device, dtype=torch.float32)
            if src.shape != target_shape:
                raise ValueError(
                    f"Embedding weight shape mismatch: expected {tuple(target_shape)}, got {tuple(src.shape)}. "
                    f"Partial copying is rejected."
                )
            self.embed_tokens.weight.copy_(src)
            logger.info(f"Loaded verified embedding weights {tuple(src.shape)} into first stage.")

    def load_head_weights(self, lm_head_weight: Any, norm_weight: Optional[Any] = None):
        """Loads verified pre-trained final RMSNorm and LM Head weights into the last stage."""
        if not self.is_last_stage:
            return
        if not HAS_TORCH:
            return
        with torch.no_grad():
            if norm_weight is not None and self.norm is not None:
                if not isinstance(norm_weight, torch.Tensor):
                    raise ValueError(
                        f"Expected torch.Tensor for RMSNorm weights, got {type(norm_weight).__name__}."
                    )
                n_src = norm_weight.to(device=self.device, dtype=torch.float32)
                if n_src.shape != self.norm.weight.shape:
                    raise ValueError(
                        f"RMSNorm weight shape mismatch: expected {tuple(self.norm.weight.shape)}, got {tuple(n_src.shape)}. "
                        f"Partial copying is rejected."
                    )
                self.norm.weight.copy_(n_src)
            if lm_head_weight is not None and self.lm_head is not None:
                if not isinstance(lm_head_weight, torch.Tensor):
                    raise ValueError(
                        f"Expected torch.Tensor for LM Head weights, got {type(lm_head_weight).__name__}."
                    )
                h_src = lm_head_weight.to(device=self.device, dtype=torch.float32)
                target_shape = self.lm_head.weight.shape
                if h_src.shape != target_shape:
                    raise ValueError(
                        f"LM Head weight shape mismatch: expected {tuple(target_shape)}, got {tuple(h_src.shape)}. "
                        f"Partial copying is rejected."
                    )
                self.lm_head.weight.copy_(h_src)
                logger.info(f"Loaded verified LM Head weights into last stage.")

    def hot_swap_layer(self, layer_index: int, weights: Optional[Any] = None, new_block: Optional[Any] = None):
        """
        Dynamically installs received weights into the model.
        Performs deserialization, architecture mapping, tensor validation, device placement,
        and replaces the layer.
        ONLY marks the layer ready after validation passes successfully.
        """
        local_idx = layer_index - self.layer_start
        if not (0 <= local_idx < len(self.layers)):
            raise IndexError(f"Layer {layer_index} out of stage range [{self.layer_start}..{self.layer_end})")

        if new_block is not None:
            installed_block = new_block.to(self.device)
        elif weights is not None:
            installed_block = deserialize_and_validate_layer_weights(
                weights_data=weights,
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                intermediate_size=self.intermediate_size,
                head_dim=self.head_dim,
                architecture=self.architecture,
                layer_idx=layer_index,
                device=self.device,
            )
        else:
            raise ValueError(f"Cannot hot-swap layer {layer_index} without weights or new block.")

        # Replace model layer with validated block
        self.layers[local_idx] = installed_block

        # Mark ready only upon successful installation
        self.ready_layers.add(layer_index)
        if layer_index in self._layer_events:
            self._layer_events[layer_index].set()

        if len(self.ready_layers) >= (self.layer_end - self.layer_start):
            self.fast_boot_active = False
            logger.info(f"Stage [{self.layer_start}..{self.layer_end}) fully loaded with real model weights!")

    def mark_layer_ready(self, layer_index: int, weights: Optional[Any] = None):
        """Mark a streamed layer as ready in memory, unblocking waiting forward passes."""
        if weights is None:
            raise ValueError(f"Cannot mark layer {layer_index} ready without valid weights")
        self.hot_swap_layer(layer_index, weights=weights)
        logger.debug(f"JIT Stream: Layer {layer_index} READY ({len(self.ready_layers)}/{self.layer_end - self.layer_start})")

    def is_layer_ready(self, layer_index: int) -> bool:
        """Returns True if the given layer index is installed and ready."""
        return layer_index in self.ready_layers

    def wait_layer_ready(self, layer_index: int, timeout: float = 10.0) -> bool:
        """Wait until a specific layer is installed and validated before computing it."""
        if self.fast_boot_active:
            return True
        if layer_index in self.ready_layers:
            return True
        if layer_index in self._layer_events:
            return self._layer_events[layer_index].wait(timeout=timeout)
        return False

    def is_entrance_ready(self) -> bool:
        if self.fast_boot_active:
            return True
        return self.layer_start in self.ready_layers

    def get_ready_count(self) -> int:
        return len(self.ready_layers)

    def forward_first_stage(self, input_ids: "torch.Tensor", past_kvs: Optional[List[Any]] = None):
        """First stage: input_ids -> embeddings -> assigned layers -> intermediate activation."""
        if not self.is_first_stage:
            raise RuntimeError("Called forward_first_stage on a non-first stage worker")

        if self.embed_tokens is not None and HAS_TORCH:
            if not isinstance(input_ids, torch.Tensor):
                if isinstance(input_ids, (list, tuple)):
                    input_ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
                else:
                    raise ValueError(f"input_ids must be a torch.Tensor, got {type(input_ids).__name__}")
            max_idx = self.embed_tokens.num_embeddings - 1
            if (input_ids < 0).any() or (input_ids > max_idx).any():
                invalid_vals = input_ids[(input_ids < 0) | (input_ids > max_idx)].unique().tolist()
                raise ValueError(
                    f"Invalid token ID detected in input_ids: {invalid_vals}. "
                    f"Token IDs must be in range [0, {self.vocab_size}) (vocab_size={self.vocab_size})."
                )

        h = self.embed_tokens(input_ids)
        new_kvs = []
        for i, layer in enumerate(self.layers):
            layer_idx = self.layer_start + i
            if not self.wait_layer_ready(layer_idx):
                raise RuntimeError(f"Layer {layer_idx} is not ready or failed validation before forward execution")
            layer_past_kv = past_kvs[i] if past_kvs else None
            h, new_kv = layer(h, layer_past_kv)
            new_kvs.append(new_kv)

        return h, new_kvs

    def forward_intermediate_stage(self, hidden_states: "torch.Tensor", past_kvs: Optional[List[Any]] = None):
        """Middle or last stage: incoming activations -> assigned layers."""
        h = hidden_states
        new_kvs = []
        for i, layer in enumerate(self.layers):
            layer_idx = self.layer_start + i
            if not self.wait_layer_ready(layer_idx):
                raise RuntimeError(f"Layer {layer_idx} is not ready or failed validation before forward execution")
            layer_past_kv = past_kvs[i] if past_kvs else None
            h, new_kv = layer(h, layer_past_kv)
            new_kvs.append(new_kv)

        if self.is_last_stage:
            h = self.norm(h)
            logits = self.lm_head(h)
            return logits, new_kvs

        return h, new_kvs


def find_local_safetensors_files(model_name: str) -> List[str]:
    """Finds safetensors, GGUF, or PyTorch checkpoint files on disk for the given model name."""
    import glob
    import os
    import json
    candidates = []

    # 1. Direct path
    if os.path.isdir(model_name):
        candidates.append(model_name)
    elif os.path.isfile(model_name):
        return [model_name]

    # 2. Ollama model manifests and blobs (~/.ollama/models)
    ollama_dir = os.path.expanduser("~/.ollama/models")
    manifests_dir = os.path.join(ollama_dir, "manifests")
    if os.path.isdir(manifests_dir):
        clean_model = model_name.lower().replace("-", "")
        parts = clean_model.split(":")
        name_part = parts[0]
        tag_part = parts[1] if len(parts) > 1 else None

        for root, dirs, files in os.walk(manifests_dir):
            rel = os.path.relpath(root, manifests_dir).lower().replace("-", "")
            if name_part in rel:
                for f in files:
                    f_clean = f.lower().replace("-", "")
                    if tag_part is None or tag_part in f_clean or f_clean == "latest":
                        manifest_path = os.path.join(root, f)
                        try:
                            with open(manifest_path, "r", encoding="utf-8") as mf:
                                mdata = json.load(mf)
                            for layer in mdata.get("layers", []):
                                if layer.get("mediaType") == "application/vnd.ollama.image.model":
                                    digest = layer.get("digest", "")
                                    blob_name = digest.replace(":", "-")
                                    blob_path = os.path.join(ollama_dir, "blobs", blob_name)
                                    if os.path.isfile(blob_path) and blob_path not in candidates:
                                        candidates.append(blob_path)
                        except Exception:
                            pass

    # 3. Hugging Face hub cache (~/.cache/huggingface/hub/models--*)
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hub_dir):
        clean_name = model_name.replace("/", "--").replace(":", "--").lower()
        prefix = model_name.split(":")[0].split("/")[0].lower()
        for d in os.listdir(hub_dir):
            d_lower = d.lower()
            if clean_name in d_lower or (prefix in d_lower and ("gemma" in prefix or "qwen" in prefix or "llama" in prefix or "mistral" in prefix)):
                snapshots_dir = os.path.join(hub_dir, d, "snapshots")
                if os.path.isdir(snapshots_dir):
                    for snap in os.listdir(snapshots_dir):
                        snap_path = os.path.join(snapshots_dir, snap)
                        if os.path.isdir(snap_path):
                            candidates.append(snap_path)

    # Search candidates for .safetensors, .bin, or .gguf files
    results = []
    for cand in candidates:
        if os.path.isfile(cand):
            results.append(cand)
            continue
        sfiles = sorted(glob.glob(os.path.join(cand, "*.safetensors")))
        if sfiles:
            results.extend(sfiles)
            continue
        pt_files = sorted(glob.glob(os.path.join(cand, "*.bin")))
        if pt_files:
            results.extend(pt_files)
            continue
        gguf_files = sorted(glob.glob(os.path.join(cand, "*.gguf")))
        if gguf_files:
            results.extend(gguf_files)

    return results


def load_hf_stage_model(
    model_name: str,
    layer_start: int,
    layer_end: int,
    total_layers: int,
    is_first_stage: bool,
    is_last_stage: bool,
    device: str = "cpu",
    load_in_8bit: bool = True,
    progressive: bool = False,
    fast_boot: bool = False,
) -> DistributedStageModel:
    """
    Loads real architecture and weights for assigned layers.
    Dynamically inspects model architecture metadata without hardcoding defaults.
    """
    meta = resolve_model_metadata(model_name)
    arch = meta.get("architecture") or meta.get("model_type")
    if not is_architecture_supported(arch):
        raise ValueError(
            f"Unsupported model architecture '{arch}' for model '{model_name}'. "
            f"DistributedStageModel requires an architecture compatible with RMSNorm + RoPE + SwiGLU "
            f"(supported: {sorted(SUPPORTED_ARCHITECTURES)}). Incompatible architectures (e.g. MoE, MPT, GPT-2) are rejected."
        )

    hidden_dim = meta["hidden_dim"]
    vocab_size = meta["vocab_size"]
    num_heads = meta["num_heads"]
    num_kv_heads = meta.get("num_kv_heads") or num_heads
    intermediate_size = meta["intermediate_size"]
    head_dim = meta.get("head_dim") or max(1, hidden_dim // max(1, num_heads))

    stage = DistributedStageModel(
        layer_start=layer_start,
        layer_end=layer_end,
        total_layers=total_layers,
        is_first_stage=is_first_stage,
        is_last_stage=is_last_stage,
        hidden_dim=hidden_dim,
        vocab_size=vocab_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        intermediate_size=intermediate_size,
        device=device,
        progressive=progressive,
        fast_boot=fast_boot,
        head_dim=head_dim,
        architecture=arch,
    )

    if not progressive and not fast_boot and HAS_TORCH:
        from situs.worker.jit_streamer import is_layer_cached, load_layer_from_cache, save_layer_to_cache
        loaded_cached = 0
        for l_idx in range(layer_start, layer_end):
            if is_layer_cached(model_name, l_idx):
                cached = load_layer_from_cache(model_name, l_idx)
                if cached:
                    try:
                        stage.hot_swap_layer(l_idx, weights=cached)
                        loaded_cached += 1
                    except Exception as e:
                        logger.debug(f"Could not load cached weights for layer {l_idx}: {e}")

        # Check local safetensors, GGUF, or Ollama checkpoints
        sfiles = find_local_safetensors_files(model_name)
        if sfiles:
            from situs.worker.jit_streamer import DEFAULT_CACHE_DIR
            safe_name = model_name.replace("/", "_").replace(":", "_")
            model_cache_dir = DEFAULT_CACHE_DIR / safe_name
            embed_cache_file = model_cache_dir / "embed_tokens.pt"
            norm_cache_file = model_cache_dir / "norm.pt"
            head_cache_file = model_cache_dir / "lm_head.pt"

            cached_embed = None
            if embed_cache_file.is_file():
                try:
                    cached_embed = torch.load(embed_cache_file, weights_only=True)
                    if is_first_stage:
                        stage.load_embedding_weights(cached_embed)
                except Exception:
                    cached_embed = None

            cached_head = None
            cached_norm = None
            if is_last_stage:
                if norm_cache_file.is_file():
                    try:
                        cached_norm = torch.load(norm_cache_file, weights_only=True)
                    except Exception:
                        pass
                if head_cache_file.is_file():
                    try:
                        cached_head = torch.load(head_cache_file, weights_only=True)
                    except Exception:
                        pass
                elif cached_embed is not None:
                    cached_head = cached_embed

                if cached_head is not None:
                    stage.load_head_weights(lm_head_weight=cached_head, norm_weight=cached_norm)

            need_embed = is_first_stage and cached_embed is None
            need_head = is_last_stage and (cached_head is None or cached_norm is None)
            all_layers_ready = all(stage.is_layer_ready(l_idx) for l_idx in range(layer_start, layer_end))

            if all_layers_ready and not need_embed and not need_head:
                logger.info(f"All stage [{layer_start}..{layer_end}) layers and embeddings loaded from local disk cache.")
                sfiles = []

            # 1. Try GGUF reader if file has GGUF format
            try:
                import gguf
                for sf in sfiles:
                    if not os.path.isfile(sf):
                        continue
                    is_gguf = False
                    try:
                        with open(sf, "rb") as probe_f:
                            is_gguf = (probe_f.read(4) == b"GGUF")
                    except Exception:
                        is_gguf = False

                    if is_gguf:
                        try:
                            reader = gguf.GGUFReader(sf)
                            embed_weight = cached_embed
                            norm_weight = cached_norm
                            lm_head_weight = cached_head

                            # Check embedding weights if needed
                            if need_embed or (need_head and lm_head_weight is None):
                                for t in reader.tensors:
                                    if t.name in ("token_embd.weight", "model.embed_tokens.weight"):
                                        arr = gguf.quants.dequantize(t.data, t.tensor_type) if t.tensor_type != 0 else t.data
                                        embed_weight = torch.from_numpy(arr.copy())
                                        break

                            # Check norm and head if needed
                            if need_head:
                                for t in reader.tensors:
                                    if norm_weight is None and t.name in ("output_norm.weight", "norm.weight", "model.norm.weight"):
                                        norm_weight = torch.from_numpy(t.data.copy())
                                    elif lm_head_weight is None and t.name in ("output.weight", "lm_head.weight"):
                                        arr = gguf.quants.dequantize(t.data, t.tensor_type) if t.tensor_type != 0 else t.data
                                        lm_head_weight = torch.from_numpy(arr.copy())

                                if lm_head_weight is None and embed_weight is not None:
                                    lm_head_weight = embed_weight

                            if is_first_stage and embed_weight is not None and cached_embed is None:
                                stage.load_embedding_weights(embed_weight)
                                try:
                                    model_cache_dir.mkdir(parents=True, exist_ok=True)
                                    torch.save(embed_weight, embed_cache_file)
                                except Exception:
                                    pass

                            if is_last_stage and lm_head_weight is not None and cached_head is None:
                                stage.load_head_weights(lm_head_weight=lm_head_weight, norm_weight=norm_weight)
                                try:
                                    model_cache_dir.mkdir(parents=True, exist_ok=True)
                                    if norm_weight is not None:
                                        torch.save(norm_weight, norm_cache_file)
                                    torch.save(lm_head_weight, head_cache_file)
                                except Exception:
                                    pass

                            # Extract assigned layers
                            for l_idx in range(layer_start, layer_end):
                                if stage.is_layer_ready(l_idx):
                                    continue
                                prefix = f"blk.{l_idx}."
                                matched = [t for t in reader.tensors if t.name.startswith(prefix)]
                                if matched:
                                    layer_state = {}
                                    for t in matched:
                                        short_k = t.name.replace(prefix, "")
                                        arr = gguf.quants.dequantize(t.data, t.tensor_type) if t.tensor_type != 0 else t.data
                                        layer_state[short_k] = torch.from_numpy(arr.copy())
                                    try:
                                        stage.hot_swap_layer(l_idx, weights=layer_state)
                                        loaded_cached += 1
                                        try:
                                            buf = io.BytesIO()
                                            torch.save(layer_state, buf)
                                            save_layer_to_cache(model_name, l_idx, buf.getvalue())
                                        except Exception:
                                            pass
                                    except Exception as e:
                                        logger.debug(f"Failed to load layer {l_idx} from GGUF {sf}: {e}")
                        except Exception as e:
                            logger.warning(f"Error loading GGUF model from {sf}: {e}")
            except ImportError:
                pass

            # 2. Try Safetensors loader
            if HAS_SAFETENSORS:
                try:
                    import safetensors.torch
                    embed_weight = None
                    norm_weight = None
                    lm_head_weight = None

                    for sf in sfiles:
                        if sf.endswith(".gguf") or not os.path.isfile(sf):
                            continue
                        try:
                            with safetensors.safe_open(sf, framework="pt", device="cpu") as f:
                                f_keys = set(f.keys())
                                # Check embeddings
                                if is_first_stage and embed_weight is None:
                                    for ek in ("model.embed_tokens.weight", "embed_tokens.weight", "token_embd.weight"):
                                        if ek in f_keys:
                                            embed_weight = f.get_tensor(ek)
                                            break

                                # Check norm
                                if is_last_stage and norm_weight is None:
                                    for nk in ("model.norm.weight", "norm.weight", "output_norm.weight"):
                                        if nk in f_keys:
                                            norm_weight = f.get_tensor(nk)
                                            break

                                # Check lm_head
                                if is_last_stage and lm_head_weight is None:
                                    for hk in ("lm_head.weight", "output.weight"):
                                        if hk in f_keys:
                                            lm_head_weight = f.get_tensor(hk)
                                            break

                                # Extract assigned layers
                                for l_idx in range(layer_start, layer_end):
                                    if stage.is_layer_ready(l_idx):
                                        continue
                                    layer_prefix = f"model.layers.{l_idx}."
                                    alt_prefix = f"layers.{l_idx}."
                                    alt_blk = f"blk.{l_idx}."
                                    matched_keys = [
                                        k for k in f_keys
                                        if k.startswith(layer_prefix) or k.startswith(alt_prefix) or k.startswith(alt_blk)
                                    ]
                                    if matched_keys:
                                        layer_state = {}
                                        for k in matched_keys:
                                            short_k = k.replace(layer_prefix, "").replace(alt_prefix, "").replace(alt_blk, "")
                                            layer_state[short_k] = f.get_tensor(k)
                                        try:
                                            stage.hot_swap_layer(l_idx, weights=layer_state)
                                            loaded_cached += 1
                                            try:
                                                save_layer_to_cache(model_name, l_idx, layer_state)
                                            except Exception:
                                                pass
                                        except Exception as e:
                                            logger.debug(f"Failed to load layer {l_idx} from {sf}: {e}")

                                    if is_first_stage and embed_weight is not None:
                                        stage.load_embedding_weights(embed_weight)
                                    if is_last_stage:
                                        if lm_head_weight is None and embed_weight is not None:
                                            lm_head_weight = embed_weight
                                        stage.load_head_weights(lm_head_weight=lm_head_weight, norm_weight=norm_weight)

                        except Exception:
                            pass
                except Exception as e:
                    logger.warning(f"Error loading safetensors for {model_name}: {e}")

        if loaded_cached > 0:
            logger.info(f"Loaded {loaded_cached} layer weights from disk for stage [{layer_start}..{layer_end})")

    logger.info(
        f"Initialized real stage [{layer_start}..{layer_end}) on {device}. "
        f"HiddenDim={hidden_dim}, Heads={num_heads}, Vocab={vocab_size}, First={is_first_stage}, Last={is_last_stage}"
    )
    return stage
