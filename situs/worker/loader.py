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
    "gemma3forcausallm",
    "gemma3forconditionalgeneration",
    "mistral",
    "mistralnemo",
    "qwen",
    "qwen2",
    "qwen2.5",
    "qwen2forcausallm",
    "starcoder2",
    "starcoder2forcausallm",
}

CANONICAL_TENSOR_MAP: Dict[str, str] = {
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_proj.bias": "attn_q.bias",
    "self_attn.k_proj.bias": "attn_k.bias",
    "self_attn.v_proj.bias": "attn_v.bias",
    "self_attn.o_proj.bias": "attn_output.bias",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.c_fc.weight": "ffn_up.weight",
    "mlp.c_fc.bias": "ffn_up.bias",
    "mlp.c_proj.weight": "ffn_down.weight",
    "mlp.c_proj.bias": "ffn_down.bias",
    "input_layernorm.weight": "attn_norm.weight",
    "input_layernorm.bias": "attn_norm.bias",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "post_attention_layernorm.bias": "ffn_norm.bias",
    "pre_feedforward_layernorm.weight": "pre_ffn_norm.weight",
    "post_feedforward_layernorm.weight": "post_ffn_norm.weight",
    # GGUF standardized keys
    "attn_q.weight": "attn_q.weight",
    "attn_k.weight": "attn_k.weight",
    "attn_v.weight": "attn_v.weight",
    "attn_output.weight": "attn_output.weight",
    "attn_q.bias": "attn_q.bias",
    "attn_k.bias": "attn_k.bias",
    "attn_v.bias": "attn_v.bias",
    "attn_output.bias": "attn_output.bias",
    "attn_q_norm.weight": "attn_q_norm.weight",
    "attn_k_norm.weight": "attn_k_norm.weight",
    "ffn_gate.weight": "ffn_gate.weight",
    "ffn_up.weight": "ffn_up.weight",
    "ffn_up.bias": "ffn_up.bias",
    "ffn_down.weight": "ffn_down.weight",
    "ffn_down.bias": "ffn_down.bias",
    "attn_norm.weight": "attn_norm.weight",
    "attn_norm.bias": "attn_norm.bias",
    "attn_post_norm.weight": "post_attn_norm.weight",
    "ffn_norm.weight": "ffn_norm.weight",
    "ffn_norm.bias": "ffn_norm.bias",
    "ffn_pre_norm.weight": "pre_ffn_norm.weight",
    "ffn_post_norm.weight": "post_ffn_norm.weight",
    "post_ffw_norm.weight": "post_ffn_norm.weight",
}

SUPPORTED_QUANTIZATION_SCHEMES: Set[str] = {
    "fp32", "float32",
    "fp16", "float16",
    "bf16", "bfloat16",
    "int8", "q8_0", "8bit",
    "int4", "q4_0", "q4_1", "q4_k", "4bit",
}


class UnsupportedQuantizationError(ValueError):
    """Raised when an incompatible or corrupted quantization scheme is detected."""
    pass


def normalize_canonical_tensor_name(raw_name: str) -> str:
    """Normalizes arbitrary HuggingFace or GGUF tensor names into unified canonical keys."""
    clean = raw_name.strip()
    # Strip leading layer prefixes e.g. "model.layers.0.", "language_model.model.layers.0.", or "blk.0."
    import re
    clean = re.sub(r"^(?:(?:language_model\.)?model\.layers\.\d+\.|layers\.\d+\.|blk\.\d+\.)", "", clean)
    return CANONICAL_TENSOR_MAP.get(clean, clean)


def validate_quantization_metadata(quant_meta: Dict[str, Any]) -> None:
    """Strictly validates tensor quantization metadata format before instantiation."""
    if not isinstance(quant_meta, dict):
        raise UnsupportedQuantizationError(f"Quantization metadata must be a dictionary, got {type(quant_meta)}")
    scheme = quant_meta.get("dtype") or quant_meta.get("scheme") or quant_meta.get("format") or quant_meta.get("precision")
    if not scheme:
        return
    norm_scheme = str(scheme).strip().lower()
    if norm_scheme not in SUPPORTED_QUANTIZATION_SCHEMES:
        raise UnsupportedQuantizationError(
            f"Unsupported quantization scheme '{norm_scheme}'. Supported schemes: {sorted(SUPPORTED_QUANTIZATION_SCHEMES)}"
        )


def is_architecture_supported(arch: Optional[str]) -> bool:
    """
    Validates that the architecture is supported by RealTransformerBlock
    (LLaMA, Mistral/NeMo, Qwen2, Gemma 1/2/3, StarCoder2 with verified numerical parity).
    """
    if not arch or not isinstance(arch, str):
        return False
    norm = arch.strip().lower()
    incompatible = (
        "moe", "mixture", "mpt", "gpt2", "gpt_neox", "falcon",
        "chatglm", "bloom", "opt", "bert", "t5", "transformer"
    )
    for bad in incompatible:
        if bad in norm:
            return False
    for supported in SUPPORTED_ARCHITECTURES:
        if norm == supported or norm.startswith(supported):
            return True
        if supported in norm:
            return True
    return False


def _get_arch_spec(architecture: Optional[str]) -> Dict[str, Any]:
    """Return normalized architecture-specific defaults for norms, biases, RoPE, MLP, and sliding windows."""
    norm = str(architecture or "").strip().lower()
    if "gemma3" in norm:
        return {
            "family": "gemma3",
            "norm_type": "rmsnorm",
            "norm_unit_offset": True,
            "has_4_norms": True,
            "has_qk_norm": True,
            "mlp_type": "geglu",
            "hidden_act": "gelu_pytorch_tanh",
            "qkv_bias": False,
            "o_bias": False,
            "mlp_bias": False,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": 10000.0,
            "norm_eps": 1e-6,
            "query_pre_attn_scalar": 256.0,
            "attn_logit_softcapping": None,
            "final_logit_softcapping": None,
            "sliding_window": 1024,
            "sliding_window_pattern": 6,
            "sliding_rule": "pattern",
            "embed_scale_sqrt": True,
        }
    if "gemma2" in norm:
        return {
            "family": "gemma2",
            "norm_type": "rmsnorm",
            "norm_unit_offset": True,
            "has_4_norms": True,
            "has_qk_norm": False,
            "mlp_type": "geglu",
            "hidden_act": "gelu_pytorch_tanh",
            "qkv_bias": False,
            "o_bias": False,
            "mlp_bias": False,
            "rope_theta": 10000.0,
            "rope_local_base_freq": None,
            "norm_eps": 1e-6,
            "query_pre_attn_scalar": 256.0,
            "attn_logit_softcapping": 50.0,
            "final_logit_softcapping": 30.0,
            "sliding_window": 4096,
            "sliding_window_pattern": 2,
            "sliding_rule": "even",
            "embed_scale_sqrt": True,
        }
    if "gemma" in norm:
        return {
            "family": "gemma",
            "norm_type": "rmsnorm",
            "norm_unit_offset": True,
            "has_4_norms": False,
            "has_qk_norm": False,
            "mlp_type": "geglu",
            "hidden_act": "gelu_pytorch_tanh",
            "qkv_bias": False,
            "o_bias": False,
            "mlp_bias": False,
            "rope_theta": 10000.0,
            "rope_local_base_freq": None,
            "norm_eps": 1e-6,
            "query_pre_attn_scalar": None,
            "attn_logit_softcapping": None,
            "final_logit_softcapping": None,
            "sliding_window": None,
            "sliding_window_pattern": None,
            "sliding_rule": "none",
            "embed_scale_sqrt": True,
        }
    if "starcoder2" in norm:
        return {
            "family": "starcoder2",
            "norm_type": "layernorm",
            "norm_unit_offset": False,
            "has_4_norms": False,
            "has_qk_norm": False,
            "mlp_type": "gelu_2layer",
            "hidden_act": "gelu_pytorch_tanh",
            "qkv_bias": True,
            "o_bias": True,
            "mlp_bias": True,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": None,
            "norm_eps": 1e-5,
            "query_pre_attn_scalar": None,
            "attn_logit_softcapping": None,
            "final_logit_softcapping": None,
            "sliding_window": 4096,
            "sliding_window_pattern": None,
            "sliding_rule": "all",
            "embed_scale_sqrt": False,
        }
    if "qwen" in norm:
        return {
            "family": "qwen2",
            "norm_type": "rmsnorm",
            "norm_unit_offset": False,
            "has_4_norms": False,
            "has_qk_norm": False,
            "mlp_type": "swiglu",
            "hidden_act": "silu",
            "qkv_bias": True,
            "o_bias": False,
            "mlp_bias": False,
            "rope_theta": 1000000.0,
            "rope_local_base_freq": None,
            "norm_eps": 1e-6,
            "query_pre_attn_scalar": None,
            "attn_logit_softcapping": None,
            "final_logit_softcapping": None,
            "sliding_window": None,
            "sliding_window_pattern": None,
            "sliding_rule": "none",
            "embed_scale_sqrt": False,
        }
    return {
        "family": "llama",
        "norm_type": "rmsnorm",
        "norm_unit_offset": False,
        "has_4_norms": False,
        "has_qk_norm": False,
        "mlp_type": "swiglu",
        "hidden_act": "silu",
        "qkv_bias": False,
        "o_bias": False,
        "mlp_bias": False,
        "rope_theta": 10000.0,
        "rope_local_base_freq": None,
        "norm_eps": 1e-6,
        "query_pre_attn_scalar": None,
        "attn_logit_softcapping": None,
        "final_logit_softcapping": None,
        "sliding_window": None,
        "sliding_window_pattern": None,
        "sliding_rule": "none",
        "embed_scale_sqrt": False,
    }


def _resolve_layer_sliding_and_rope(
    spec: Dict[str, Any],
    layer_idx: int,
    rope_theta: Optional[float] = None,
    rope_local_base_freq: Optional[float] = None,
    sliding_window: Optional[int] = None,
    sliding_window_pattern: Optional[int] = None,
) -> Tuple[float, Optional[int]]:
    """Compute per-layer RoPE base frequency and active sliding window size."""
    base_theta = float(rope_theta if rope_theta is not None else spec["rope_theta"])
    local_theta = float(
        rope_local_base_freq
        if rope_local_base_freq is not None
        else (spec["rope_local_base_freq"] if spec["rope_local_base_freq"] is not None else base_theta)
    )
    sw = sliding_window if sliding_window is not None else spec["sliding_window"]
    rule = spec.get("sliding_rule", "none")
    if rule == "even":
        # Gemma2 alternates sliding attention on even layers and full attention on odd layers.
        return base_theta, (sw if (layer_idx % 2 == 0) else None)
    if rule == "pattern":
        # Gemma3 uses local sliding attention except on every N-th (typically 6th) full layer.
        pattern = sliding_window_pattern if sliding_window_pattern is not None else (spec["sliding_window_pattern"] or 6)
        is_sliding = ((layer_idx + 1) % pattern) != 0
        return (local_theta if is_sliding else base_theta), (sw if is_sliding else None)
    if rule == "all":
        return base_theta, sw
    return base_theta, sw


class RMSNorm(nn.Module if HAS_TORCH else object):
    """Root Mean Square Layer Normalization (standard for LLaMA, Mistral, Qwen2, with Gemma unit-offset support)."""
    def __init__(self, dim: int, eps: float = 1e-6, is_gemma: bool = False):
        super().__init__()
        self.eps = eps
        self.is_gemma = is_gemma
        if HAS_TORCH:
            self.weight = nn.Parameter(torch.zeros(dim) if is_gemma else torch.ones(dim))

    def forward(self, x):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for model inference")
        if not isinstance(x, torch.Tensor):
            return x
        orig_dtype = x.dtype
        x_f = x.float()
        variance = x_f.pow(2).mean(-1, keepdim=True)
        x_norm = x_f * torch.rsqrt(variance + self.eps)
        if self.is_gemma:
            return (x_norm * (1.0 + self.weight.float())).to(orig_dtype)
        return (x_norm * self.weight.float()).to(orig_dtype)


class RotaryEmbedding(nn.Module if HAS_TORCH else object):
    """Rotary Position Embedding (RoPE) for relative attention."""
    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        if HAS_TORCH:
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        x,
        seq_len: Optional[int] = None,
        offset: int = 0,
        position_ids: Optional[torch.Tensor] = None,
    ):
        if not HAS_TORCH or not isinstance(x, torch.Tensor):
            return x, x
        if position_ids is not None:
            t = position_ids.to(device=x.device, dtype=self.inv_freq.dtype)
            while t.dim() > 2 and t.shape[1] == 1:
                t = t.squeeze(1)
            if t.dim() == 2:
                freqs = t.unsqueeze(-1) * self.inv_freq.to(t.device)
                emb = torch.cat((freqs, freqs), dim=-1)
                return emb.cos(), emb.sin()
            t = t.view(-1)
        else:
            if seq_len is None:
                seq_len = x.shape[1] if x.dim() >= 2 else 1
            t = torch.arange(offset, offset + seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq.to(t.device))
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

    cos = cos.to(device=q.device, dtype=q.dtype)
    sin = sin.to(device=k.device, dtype=k.dtype)

    if cos.dim() == 2:
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, dim]
        sin = sin.unsqueeze(0).unsqueeze(0)
    elif cos.dim() == 3:
        cos = cos.unsqueeze(1)  # [batch, 1, seq_len, dim]
        sin = sin.unsqueeze(1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


_UNSET = object()


def _fast_linear(
    in_features: int,
    out_features: int,
    device: str = "cpu",
    dtype: Any = None,
    bias: bool = False,
) -> Any:
    """Safe linear module initialization."""
    if not HAS_TORCH:
        return None
    dt = dtype if dtype is not None else (
        torch.float16 if (device != "cpu" and torch.cuda.is_available() and "cuda" in str(device)) else torch.float32
    )
    dev = device if (device != "cpu" and torch.cuda.is_available() and "cuda" in str(device)) else "cpu"
    try:
        return nn.Linear(in_features, out_features, bias=bias, device=dev, dtype=dt)
    except Exception:
        linear = nn.Linear(in_features, out_features, bias=bias)
        linear.to(dtype=dt)
        if dev != "cpu":
            linear.to(dev)
        return linear


def _build_norm(
    dim: int,
    eps: float,
    norm_type: str = "rmsnorm",
    is_gemma: bool = False,
    device: str = "cpu",
    dtype: Any = None,
) -> Any:
    """Initialize RMSNorm or LayerNorm matching the model architecture."""
    if not HAS_TORCH:
        return None
    dev = device if (device != "cpu" and torch.cuda.is_available() and "cuda" in str(device)) else "cpu"
    if norm_type == "layernorm":
        dt = dtype if dtype is not None else (
            torch.float16 if dev != "cpu" else torch.float32
        )
        try:
            return nn.LayerNorm(dim, eps=eps, elementwise_affine=True, device=dev, dtype=dt)
        except Exception:
            ln = nn.LayerNorm(dim, eps=eps, elementwise_affine=True)
            ln.to(dtype=dt)
            if dev != "cpu":
                ln.to(dev)
            return ln
    norm_mod = RMSNorm(dim, eps=eps, is_gemma=is_gemma)
    if dev != "cpu":
        norm_mod.to(dev)
    return norm_mod


class Int8WeightLinear(nn.Module if HAS_TORCH else object):
    """Compact symmetric per-row INT8 linear layer cutting GPU VRAM 2x vs float16."""

    def __init__(self, in_features: int, out_features: int, weight: Any, weight_scale: Any, bias: Optional[Any] = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        if HAS_TORCH:
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)
            self.bias = nn.Parameter(bias, requires_grad=False) if bias is not None else None

    @classmethod
    def from_float(cls, weight_tensor: "torch.Tensor", bias_tensor: Optional["torch.Tensor"] = None) -> "Int8WeightLinear":
        scale_dtype = weight_tensor.dtype if weight_tensor.dtype in (torch.float16, torch.bfloat16) else torch.float32
        w_work = weight_tensor.float()
        abs_max = w_work.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
        scale = (abs_max / 127.0).to(dtype=scale_dtype)
        q = torch.round(w_work / scale.float()).clamp(-127, 127).to(torch.int8)
        b = bias_tensor.to(dtype=scale_dtype, device=weight_tensor.device) if bias_tensor is not None else None
        return cls(weight_tensor.shape[1], weight_tensor.shape[0], q, scale, b)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        w = self.weight.to(dtype=x.dtype) * self.weight_scale.to(dtype=x.dtype)
        b = self.bias.to(dtype=x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)


class Int8WeightEmbedding(nn.Module if HAS_TORCH else object):
    """Compact symmetric per-row INT8 embedding table cutting embedding VRAM 2x vs float16."""

    def __init__(self, num_embeddings: int, embedding_dim: int, weight: Any, weight_scale: Any):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if HAS_TORCH:
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.weight_scale = nn.Parameter(weight_scale, requires_grad=False)

    @classmethod
    def from_float_chunked(
        cls,
        tensor_iter,
        num_embeddings: int,
        embedding_dim: int,
        device: str,
        scale_dtype: Any,
    ) -> "Int8WeightEmbedding":
        q_out = torch.empty((num_embeddings, embedding_dim), dtype=torch.int8, device=device)
        s_out = torch.empty((num_embeddings, 1), dtype=scale_dtype, device=device)
        offset = 0
        for chunk in tensor_iter:
            rows = chunk.shape[0]
            c_work = chunk.to(device=device, dtype=torch.float32)
            abs_max = c_work.abs().amax(dim=1, keepdim=True).clamp(min=1e-6)
            scale = (abs_max / 127.0).to(dtype=scale_dtype)
            q_out[offset : offset + rows].copy_(torch.round(c_work / scale.float()).clamp(-127, 127).to(torch.int8))
            s_out[offset : offset + rows].copy_(scale)
            offset += rows
        return cls(num_embeddings, embedding_dim, q_out, s_out)

    def forward(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        idx_w = F.embedding(input_ids, self.weight)
        idx_s = F.embedding(input_ids, self.weight_scale)
        return idx_w.to(dtype=self.weight_scale.dtype) * idx_s


def set_norm_unit_offset(module: Any, is_gemma_offset: bool) -> None:
    """Recursively configure RMSNorm Gemma +1.0 offset mode (GGUF already pre-adds +1.0)."""
    if module is None or not HAS_TORCH or not isinstance(module, nn.Module):
        return
    if isinstance(module, RMSNorm):
        module.is_gemma = is_gemma_offset
    for child in module.modules():
        if isinstance(child, RMSNorm):
            child.is_gemma = is_gemma_offset


def quantize_block_to_int8(block: "RealTransformerBlock") -> None:
    """Converts all 2D linear projections in a loaded RealTransformerBlock to Int8WeightLinear."""
    if not HAS_TORCH:
        return
    for attr in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "c_fc", "c_proj"):
        mod = getattr(block, attr, None)
        if isinstance(mod, nn.Linear):
            setattr(block, attr, Int8WeightLinear.from_float(mod.weight.data, mod.bias.data if mod.bias is not None else None))


def _dequantize_gguf_raw_chunk(raw_arr: Any, qtype_int: int, n_rows: int, target_device: str, out_dtype: Any) -> "torch.Tensor":
    """Vectorized PyTorch dequantization for common GGUF block formats with gguf.quants fallback."""
    import numpy as np
    if qtype_int == 0:  # F32
        t = torch.from_numpy(np.array(raw_arr, dtype=np.float32, copy=True)).to(device=target_device)
        return t.to(dtype=out_dtype)
    if qtype_int == 1:  # F16
        t = torch.from_numpy(np.array(raw_arr, dtype=np.float16, copy=True)).to(device=target_device)
        return t.to(dtype=out_dtype)
    if qtype_int == 30:  # BF16
        u16 = torch.from_numpy(np.array(raw_arr, dtype=np.uint16, copy=True)).to(device=target_device)
        return u16.view(torch.bfloat16).to(dtype=out_dtype)

    if qtype_int in (8, 12, 14):
        u8 = torch.from_numpy(np.array(raw_arr, dtype=np.uint8, copy=True)).to(device=target_device)
        if qtype_int == 8:  # Q8_0: 34 bytes per 32 weights
            blocks = u8.view(-1, 34)
            d = blocks[:, :2].contiguous().view(torch.float16).float()
            x = blocks[:, 2:].contiguous().view(torch.int8).float()
            return (x * d).reshape(n_rows, -1).to(dtype=out_dtype)
        if qtype_int == 12:  # Q4_K: 144 bytes per 256 weights
            blocks = u8.view(-1, 144)
            nb = blocks.shape[0]
            d = blocks[:, :2].contiguous().view(torch.float16).float()
            dmin = blocks[:, 2:4].contiguous().view(torch.float16).float()
            sc_raw = blocks[:, 4:16].reshape(nb, 3, 4)
            d_p, m_p, md_p = sc_raw[:, 0], sc_raw[:, 1], sc_raw[:, 2]
            sc = torch.cat([d_p & 0x3F, (md_p & 0x0F) | ((d_p >> 2) & 0x30)], dim=-1).float().reshape(nb, 8, 1)
            m = torch.cat([m_p & 0x3F, (md_p >> 4) | ((m_p >> 2) & 0x30)], dim=-1).float().reshape(nb, 8, 1)
            d_eff = d.reshape(nb, 1, 1) * sc
            dm_eff = dmin.reshape(nb, 1, 1) * m
            qs = blocks[:, 16:].reshape(nb, 4, 1, 32)
            shifts = torch.tensor([0, 4], device=target_device, dtype=torch.uint8).reshape(1, 1, 2, 1)
            qs = ((qs >> shifts) & 0x0F).reshape(nb, 8, 32).float()
            return (d_eff * qs - dm_eff).reshape(n_rows, -1).to(dtype=out_dtype)
        if qtype_int == 14:  # Q6_K: 210 bytes per 256 weights
            blocks = u8.view(-1, 210)
            nb = blocks.shape[0]
            ql = blocks[:, :128].reshape(nb, -1, 1, 64)
            qh = blocks[:, 128:192].reshape(nb, -1, 1, 32)
            scales = blocks[:, 192:208].contiguous().view(torch.int8).float().reshape(nb, 16, 1)
            d = blocks[:, 208:210].contiguous().view(torch.float16).float().reshape(nb, 1, 1)
            d_eff = d * scales
            ql = ((ql >> torch.tensor([0, 4], device=target_device, dtype=torch.uint8).reshape(1, 1, 2, 1)) & 0x0F).reshape(nb, -1, 32)
            qh = ((qh >> torch.tensor([0, 2, 4, 6], device=target_device, dtype=torch.uint8).reshape(1, 1, 4, 1)) & 0x03).reshape(nb, -1, 32)
            q = ((ql | (qh << 4)).to(torch.int8) - 32).reshape(nb, 16, 16).float()
            return (d_eff * q).reshape(n_rows, -1).to(dtype=out_dtype)

    import gguf
    arr = gguf.quants.dequantize(raw_arr, qtype_int)
    return torch.from_numpy(arr.copy()).to(device=target_device, dtype=out_dtype)


def _iter_dequantized_gguf_chunks(gguf_tensor: Any, out_dtype: Any, target_device: str, chunk_rows: int = 2048):
    """Yield row-chunked dequantized tensors from a GGUFReader tensor entry cleanly bounding RAM/VRAM."""
    data = gguf_tensor.data
    qt = int(gguf_tensor.tensor_type)
    if data.ndim <= 1 or data.shape[0] <= chunk_rows:
        yield _dequantize_gguf_raw_chunk(data, qt, data.shape[0] if data.ndim > 1 else 1, target_device, out_dtype)
        return
    total_rows = data.shape[0]
    for start in range(0, total_rows, chunk_rows):
        end = min(total_rows, start + chunk_rows)
        yield _dequantize_gguf_raw_chunk(data[start:end], qt, end - start, target_device, out_dtype)


def _dequantize_gguf_tensor_fast(gguf_tensor: Any, out_dtype: Any, target_device: str) -> "torch.Tensor":
    """Dequantize a GGUFReader tensor entry into a PyTorch tensor."""
    chunks = list(_iter_dequantized_gguf_chunks(gguf_tensor, out_dtype=out_dtype, target_device=target_device))
    return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)


class _FastGGUFTensor:
    __slots__ = ("name", "shape", "tensor_type", "data")

    def __init__(self, name: str, shape: Tuple[int, ...], tensor_type: int, data: Any):
        self.name = name
        self.shape = shape
        self.tensor_type = tensor_type
        self.data = data


class FastGGUFFile:
    """
    Lightweight binary GGUF reader (~50ms vs ~30s in pure-Python gguf.GGUFReader)
    that skips unrequested KV string arrays in O(1) mmap scans and exposes zero-copy
    memmap views matching GGUFReader.tensors.
    """

    _SCALAR_FMTS = {
        0: ("B", 1),   # UINT8
        1: ("b", 1),   # INT8
        2: ("H", 2),   # UINT16
        3: ("h", 2),   # INT16
        4: ("I", 4),   # UINT32
        5: ("i", 4),   # INT32
        6: ("f", 4),   # FLOAT32
        7: ("?", 1),   # BOOL
        10: ("Q", 8),  # UINT64
        11: ("q", 8),  # INT64
        12: ("d", 8),  # FLOAT64
    }
    _QUANT_BLOCKS = {
        0: (1, 4),     # F32
        1: (1, 2),     # F16
        2: (32, 18),   # Q4_0
        3: (32, 20),   # Q4_1
        6: (32, 22),   # Q5_0
        7: (32, 24),   # Q5_1
        8: (32, 34),   # Q8_0
        9: (32, 36),   # Q8_1
        10: (256, 84), # Q2_K
        11: (256, 110),# Q3_K
        12: (256, 144),# Q4_K
        13: (256, 176),# Q5_K
        14: (256, 210),# Q6_K
        30: (1, 2),    # BF16
    }

    def __init__(self, path: str, wanted_kv: Optional[Set[str]] = None):
        import mmap
        import struct
        import numpy as np

        self.path = path
        self.kv: Dict[str, Any] = {}
        self.tensors: List[_FastGGUFTensor] = []
        want = set(wanted_kv) if wanted_kv else set()
        want.add("general.alignment")

        f = open(path, "rb")
        self._file = f
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        self._mm = mm
        try:
            pos = 0
            magic = mm[0:4]
            if magic != b"GGUF":
                raise ValueError(f"Not a valid GGUF file: {path}")
            version, tensor_count, kv_count = struct.unpack_from("<IQQ", mm, 4)
            if version not in (2, 3):
                raise ValueError(f"Unsupported GGUF version {version} in {path}")
            pos = 24

            def _read_str(p: int) -> Tuple[str, int]:
                slen = struct.unpack_from("<Q", mm, p)[0]
                p += 8
                s = mm[p : p + slen].decode("utf-8", errors="replace")
                return s, p + slen

            def _skip_str(p: int) -> int:
                slen = struct.unpack_from("<Q", mm, p)[0]
                return p + 8 + slen

            def _read_val(vtype: int, p: int, keep: bool) -> Tuple[Any, int]:
                if vtype == 8:  # STRING
                    if keep:
                        return _read_str(p)
                    return None, _skip_str(p)
                if vtype in self._SCALAR_FMTS:
                    fmt, width = self._SCALAR_FMTS[vtype]
                    val = struct.unpack_from(f"<{fmt}", mm, p)[0] if keep else None
                    return val, p + width
                if vtype == 9:  # ARRAY
                    etype, count = struct.unpack_from("<IQ", mm, p)
                    p += 12
                    if etype == 8:
                        if keep:
                            arr = []
                            for _ in range(count):
                                s, p = _read_str(p)
                                arr.append(s)
                            return arr, p
                        for _ in range(count):
                            p = _skip_str(p)
                        return None, p
                    if etype in self._SCALAR_FMTS:
                        fmt, width = self._SCALAR_FMTS[etype]
                        total_b = count * width
                        if keep:
                            vals = list(struct.unpack_from(f"<{count}{fmt}", mm, p))
                            return vals, p + total_b
                        return None, p + total_b
                    # Nested array fallback
                    out = [] if keep else None
                    for _ in range(count):
                        v, p = _read_val(etype, p, keep)
                        if keep:
                            out.append(v)
                    return out, p
                raise ValueError(f"Unknown GGUF metadata type {vtype}")

            for _ in range(kv_count):
                klen = struct.unpack_from("<Q", mm, pos)[0]
                pos += 8
                key_bytes = mm[pos : pos + klen]
                pos += klen
                vtype = struct.unpack_from("<I", mm, pos)[0]
                pos += 4
                key_str = key_bytes.decode("utf-8", errors="replace")
                keep = key_str in want
                val, pos = _read_val(vtype, pos, keep)
                if keep:
                    self.kv[key_str] = val

            tensor_infos = []
            for _ in range(tensor_count):
                tname, pos = _read_str(pos)
                ndims = struct.unpack_from("<I", mm, pos)[0]
                pos += 4
                dims = struct.unpack_from(f"<{ndims}Q", mm, pos)
                pos += 8 * ndims
                qtype, rel_off = struct.unpack_from("<IQ", mm, pos)
                pos += 12
                tensor_infos.append((tname, tuple(int(d) for d in dims), int(qtype), int(rel_off)))

            alignment = int(self.kv.get("general.alignment") or 32)
            data_offset = (pos + alignment - 1) & ~(alignment - 1)

            for tname, dims, qtype, rel_off in tensor_infos:
                if qtype in self._QUANT_BLOCKS:
                    bs, ts = self._QUANT_BLOCKS[qtype]
                else:
                    import gguf
                    bs, ts = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(qtype)]

                ne0 = dims[0] if dims else 1
                n_rows = 1
                for d in dims[1:]:
                    n_rows *= d
                row_bytes = (ne0 // bs) * ts
                byte_start = data_offset + rel_off
                byte_len = n_rows * row_bytes

                if qtype == 0:
                    np_view = np.frombuffer(mm, dtype=np.float32, count=ne0 * n_rows, offset=byte_start)
                    arr = np_view.reshape(dims[::-1]) if len(dims) > 1 else np_view
                elif qtype == 1:
                    np_view = np.frombuffer(mm, dtype=np.float16, count=ne0 * n_rows, offset=byte_start)
                    arr = np_view.reshape(dims[::-1]) if len(dims) > 1 else np_view
                else:
                    np_view = np.frombuffer(mm, dtype=np.uint8, count=byte_len, offset=byte_start)
                    arr = np_view.reshape((n_rows, row_bytes)) if len(dims) > 1 else np_view
                self.tensors.append(_FastGGUFTensor(tname, dims, qtype, arr))
        except Exception:
            self.close()
            raise

    def close(self):
        self.tensors.clear()
        mm = getattr(self, "_mm", None)
        if mm is not None:
            try:
                mm.close()
            except Exception:
                pass
            self._mm = None
        f = getattr(self, "_file", None)
        if f is not None:
            try:
                f.close()
            except Exception:
                pass
            self._file = None


def _fast_embedding(num_embeddings: int, embedding_dim: int, device: str = "cpu", dtype: Any = None) -> Any:
    """Safe embedding module initialization avoiding multi-GB host CPU memory allocation."""
    if not HAS_TORCH:
        return None
    dt = dtype if dtype is not None else (
        torch.float16 if (device != "cpu" and torch.cuda.is_available() and "cuda" in str(device)) else torch.float32
    )
    dev = device if (device != "cpu" and torch.cuda.is_available() and "cuda" in str(device)) else "cpu"
    emb = nn.Embedding.__new__(nn.Embedding)
    nn.Module.__init__(emb)
    emb.num_embeddings = num_embeddings
    emb.embedding_dim = embedding_dim
    emb.padding_idx = None
    emb.max_norm = None
    emb.norm_type = 2.0
    emb.scale_grad_by_freq = False
    emb.sparse = False
    emb.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, dtype=dt, device=dev))
    return emb


class RealTransformerBlock(nn.Module if HAS_TORCH else object):
    """
    Genuine Transformer Decoder Block supporting all Situs catalog architectures:
    - LLaMA / Mistral / Mistral-NeMo (RMSNorm, RoPE, SwiGLU)
    - Qwen2 / DeepSeek-R1-Distill-Qwen (RMSNorm, RoPE, QKV bias, SwiGLU)
    - Gemma / Gemma2 / Gemma3 (unit-offset RMSNorm, 4-norm flow, Q/K norm, logit softcapping, GeGLU)
    - StarCoder2 (LayerNorm with bias, QKV+O+MLP biases, RoPE, 2-layer GELU-tanh MLP)
    """
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_kv_heads: int,
        intermediate_size: int,
        head_dim: int,
        architecture: str,
        eps: Optional[float] = None,
        layer_idx: int = 0,
        device: str = "cpu",
        dtype: Optional[Any] = None,
        rope_theta: Optional[float] = None,
        sliding_window: Any = _UNSET,
        query_pre_attn_scalar: Optional[float] = None,
        attn_logit_softcapping: Any = _UNSET,
        qkv_bias: Optional[bool] = None,
        o_bias: Optional[bool] = None,
        mlp_bias: Optional[bool] = None,
        hidden_act: Optional[str] = None,
    ):
        super().__init__()
        if not is_architecture_supported(architecture):
            raise ValueError(
                f"Unsupported model architecture '{architecture}'. "
                f"RealTransformerBlock requires a supported decoder architecture "
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

        spec = _get_arch_spec(architecture)
        self.architecture = architecture
        self.spec = spec
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.layer_idx = layer_idx
        self.device = device
        self.dtype = dtype if dtype is not None else (
            torch.float16 if (device != "cpu" and HAS_TORCH and torch.cuda.is_available() and "cuda" in str(device)) else (torch.float32 if HAS_TORCH else None)
        )

        self.norm_type = spec["norm_type"]
        self.is_gemma = spec["norm_unit_offset"]
        self.has_4_norms = spec["has_4_norms"]
        self.has_qk_norm = spec["has_qk_norm"]
        self.mlp_type = spec["mlp_type"]
        self.qkv_bias = spec["qkv_bias"] if qkv_bias is None else bool(qkv_bias)
        self.o_bias = spec["o_bias"] if o_bias is None else bool(o_bias)
        self.mlp_bias = spec["mlp_bias"] if mlp_bias is None else bool(mlp_bias)
        norm_eps = float(eps if eps is not None else spec["norm_eps"])

        if sliding_window is _UNSET:
            layer_rope_theta, layer_sw = _resolve_layer_sliding_and_rope(
                spec, layer_idx, rope_theta=rope_theta
            )
        else:
            layer_rope_theta = float(rope_theta if rope_theta is not None else spec["rope_theta"])
            layer_sw = sliding_window
        self.rope_theta = layer_rope_theta
        self.sliding_window = layer_sw

        scalar = query_pre_attn_scalar if query_pre_attn_scalar is not None else spec["query_pre_attn_scalar"]
        self.query_pre_attn_scalar = scalar
        self.attn_scale = (float(scalar) ** -0.5) if (scalar is not None and scalar > 0) else (float(head_dim) ** -0.5)
        self.attn_logit_softcapping = (
            spec["attn_logit_softcapping"] if attn_logit_softcapping is _UNSET else attn_logit_softcapping
        )

        if not HAS_TORCH:
            return

        self.input_layernorm = _build_norm(
            hidden_dim, eps=norm_eps, norm_type=self.norm_type, is_gemma=self.is_gemma, device=self.device, dtype=self.dtype
        )

        # Self-Attention projections
        self.q_proj = _fast_linear(hidden_dim, self.num_heads * self.head_dim, device=self.device, dtype=self.dtype, bias=self.qkv_bias)
        self.k_proj = _fast_linear(hidden_dim, self.num_kv_heads * self.head_dim, device=self.device, dtype=self.dtype, bias=self.qkv_bias)
        self.v_proj = _fast_linear(hidden_dim, self.num_kv_heads * self.head_dim, device=self.device, dtype=self.dtype, bias=self.qkv_bias)
        self.o_proj = _fast_linear(self.num_heads * self.head_dim, hidden_dim, device=self.device, dtype=self.dtype, bias=self.o_bias)

        if self.has_qk_norm:
            self.q_norm = _build_norm(self.head_dim, eps=norm_eps, norm_type="rmsnorm", is_gemma=self.is_gemma, device=self.device)
            self.k_norm = _build_norm(self.head_dim, eps=norm_eps, norm_type="rmsnorm", is_gemma=self.is_gemma, device=self.device)

        self.post_attention_layernorm = _build_norm(
            hidden_dim, eps=norm_eps, norm_type=self.norm_type, is_gemma=self.is_gemma, device=self.device, dtype=self.dtype
        )
        if self.has_4_norms:
            self.pre_feedforward_layernorm = _build_norm(
                hidden_dim, eps=norm_eps, norm_type=self.norm_type, is_gemma=self.is_gemma, device=self.device, dtype=self.dtype
            )
            self.post_feedforward_layernorm = _build_norm(
                hidden_dim, eps=norm_eps, norm_type=self.norm_type, is_gemma=self.is_gemma, device=self.device, dtype=self.dtype
            )

        act_name = str(hidden_act or spec["hidden_act"]).lower()
        if self.mlp_type == "gelu_2layer":
            self.c_fc = _fast_linear(hidden_dim, self.intermediate_size, device=self.device, dtype=self.dtype, bias=self.mlp_bias)
            self.c_proj = _fast_linear(self.intermediate_size, hidden_dim, device=self.device, dtype=self.dtype, bias=self.mlp_bias)
            self.act_fn = nn.GELU(approximate="none" if act_name == "gelu" else "tanh")
        else:
            self.gate_proj = _fast_linear(hidden_dim, self.intermediate_size, device=self.device, dtype=self.dtype, bias=self.mlp_bias)
            self.up_proj = _fast_linear(hidden_dim, self.intermediate_size, device=self.device, dtype=self.dtype, bias=self.mlp_bias)
            self.down_proj = _fast_linear(self.intermediate_size, hidden_dim, device=self.device, dtype=self.dtype, bias=self.mlp_bias)
            if "gelu" in act_name or self.mlp_type == "geglu":
                self.act_fn = nn.GELU(approximate="tanh")
            else:
                self.act_fn = nn.SiLU()

        # Rotary Embeddings
        self.rotary = RotaryEmbedding(self.head_dim, base=layer_rope_theta)

    def _compute_mlp(self, norm_mlp: torch.Tensor) -> torch.Tensor:
        if self.mlp_type == "gelu_2layer":
            return self.c_proj(self.act_fn(self.c_fc(norm_mlp)))
        return self.down_proj(self.act_fn(self.gate_proj(norm_mlp)) * self.up_proj(norm_mlp))

    def _compute_attention(
        self,
        q: torch.Tensor,
        k_exp: torch.Tensor,
        v_exp: torch.Tensor,
        seq_len: int,
        past_len: int,
    ) -> torch.Tensor:
        total_kv_len = k_exp.shape[-2]
        use_sliding = self.sliding_window is not None and self.sliding_window > 0 and (
            seq_len > 1 or total_kv_len > self.sliding_window
        )

        def _build_mask():
            q_idx = torch.arange(past_len, past_len + seq_len, device=q.device).unsqueeze(1)
            k_idx = torch.arange(total_kv_len, device=q.device).unsqueeze(0)
            mask = k_idx <= q_idx
            if use_sliding:
                mask = mask & (k_idx > (q_idx - self.sliding_window))
            return mask

        if self.attn_logit_softcapping is not None and self.attn_logit_softcapping > 0:
            cap = float(self.attn_logit_softcapping)
            attn_weights = torch.matmul(q.float(), k_exp.float().transpose(-2, -1)) * self.attn_scale
            attn_weights = torch.tanh(attn_weights / cap) * cap
            if seq_len > 1 or use_sliding:
                mask = _build_mask()
                attn_weights = attn_weights.masked_fill(~mask, torch.finfo(attn_weights.dtype).min)
            attn_probs = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
            return torch.matmul(attn_probs, v_exp)

        sdpa_scale = self.attn_scale if self.query_pre_attn_scalar is not None else None
        if use_sliding:
            return F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=_build_mask(), scale=sdpa_scale)
        if seq_len > 1:
            if past_len > 0:
                return F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=_build_mask(), scale=sdpa_scale)
            return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=True, scale=sdpa_scale)
        return F.scaled_dot_product_attention(q, k_exp, v_exp, is_causal=False, scale=sdpa_scale)

    def forward(
        self,
        x,
        past_kv: Optional[Tuple[Any, Any]] = None,
        position_ids: Optional[Any] = None,
    ):
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for model inference")
        if not isinstance(x, torch.Tensor):
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

        if self.has_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # Determine past KV cache length for RoPE offset
        past_len = 0
        if past_kv is not None and isinstance(past_kv, (tuple, list)) and len(past_kv) == 2:
            past_k, past_v = past_kv
            if isinstance(past_k, torch.Tensor):
                past_len = past_k.shape[-2]

        # Apply Rotary Position Embeddings (RoPE)
        if hasattr(self, "rotary") and self.rotary is not None:
            cos, sin = self.rotary(norm_x, seq_len=seq_len, offset=past_len, position_ids=position_ids)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

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

        attn_out = self._compute_attention(q, k_exp, v_exp, seq_len=seq_len, past_len=past_len)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_out = self.o_proj(attn_out)

        if self.has_4_norms:
            attn_out = self.post_attention_layernorm(attn_out)
            x = residual + attn_out
            residual = x
            norm_mlp = self.pre_feedforward_layernorm(x)
            mlp_out = self._compute_mlp(norm_mlp)
            mlp_out = self.post_feedforward_layernorm(mlp_out)
            x = residual + mlp_out
        else:
            x = residual + attn_out
            residual = x
            norm_mlp = self.post_attention_layernorm(x)
            mlp_out = self._compute_mlp(norm_mlp)
            x = residual + mlp_out

        return x, new_kv

    def _copy_validated_param(self, param_name: str, tensor: Any, my_dict: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(
                f"Invalid tensor type for '{param_name}' in layer {self.layer_idx}: {type(tensor)}"
            )
        if param_name not in my_dict:
            attr, field = param_name.split(".", 1)
            mod = getattr(self, attr, None)
            if isinstance(mod, nn.Linear) and field == "bias" and tensor.dim() == 1:
                mod.bias = nn.Parameter(torch.empty_like(tensor, device=mod.weight.device, dtype=mod.weight.dtype))
                my_dict = self.state_dict()
            else:
                return my_dict

        target_shape = my_dict[param_name].shape
        if tensor.shape != target_shape:
            attr = param_name.split(".")[0]
            mod = getattr(self, attr, None)
            if isinstance(mod, nn.Linear) and tensor.dim() == 2:
                out_f, in_f = tensor.shape
                if in_f == mod.in_features or out_f == mod.out_features:
                    new_linear = _fast_linear(
                        in_f,
                        out_f,
                        device=str(mod.weight.device),
                        dtype=mod.weight.dtype,
                        bias=(mod.bias is not None),
                    )
                    setattr(self, attr, new_linear)
                    my_dict = self.state_dict()
                    target_shape = my_dict[param_name].shape
            if tensor.shape != target_shape:
                raise ValueError(
                    f"Shape mismatch for parameter '{param_name}' in layer {self.layer_idx}: "
                    f"got {tuple(tensor.shape)}, expected {tuple(target_shape)}"
                )

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValueError(f"Tensor '{param_name}' contains NaN or Inf values in layer {self.layer_idx}")

        my_dict[param_name].copy_(tensor.to(dtype=my_dict[param_name].dtype))
        return my_dict

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
        }
        if self.has_4_norms:
            mapping["post_attention_layernorm.weight"] = [
                "post_attention_layernorm.weight",
                "attn_post_norm.weight",
                "post_attention_norm.weight",
                "ffn_norm.weight",
            ]
        else:
            mapping["post_attention_layernorm.weight"] = [
                "post_attention_layernorm.weight",
                "ffn_norm.weight",
                "post_ffw_norm.weight",
                "post_attention_norm.weight",
                "ln_2.weight",
            ]

        if self.mlp_type == "gelu_2layer":
            mapping["c_fc.weight"] = ["mlp.c_fc.weight", "c_fc.weight", "ffn_up.weight"]
            mapping["c_proj.weight"] = ["mlp.c_proj.weight", "c_proj.weight", "ffn_down.weight"]
        else:
            mapping["gate_proj.weight"] = ["mlp.gate_proj.weight", "gate_proj.weight", "ffn_gate.weight"]
            mapping["up_proj.weight"] = ["mlp.up_proj.weight", "up_proj.weight", "ffn_up.weight"]
            mapping["down_proj.weight"] = ["mlp.down_proj.weight", "down_proj.weight", "ffn_down.weight"]

        optional_mapping: Dict[str, List[str]] = {
            "q_proj.bias": ["self_attn.q_proj.bias", "q_proj.bias", "attn.q_proj.bias", "attn_q.bias"],
            "k_proj.bias": ["self_attn.k_proj.bias", "k_proj.bias", "attn.k_proj.bias", "attn_k.bias"],
            "v_proj.bias": ["self_attn.v_proj.bias", "v_proj.bias", "attn.v_proj.bias", "attn_v.bias"],
            "o_proj.bias": ["self_attn.o_proj.bias", "o_proj.bias", "attn.o_proj.bias", "attn_output.bias"],
            "input_layernorm.bias": ["input_layernorm.bias", "attention_norm.bias", "attn_norm.bias", "ln_1.bias"],
            "post_attention_layernorm.bias": ["post_attention_layernorm.bias", "ffn_norm.bias", "post_attention_norm.bias", "ln_2.bias"],
        }
        if self.mlp_type == "gelu_2layer":
            optional_mapping["c_fc.bias"] = ["mlp.c_fc.bias", "c_fc.bias", "ffn_up.bias"]
            optional_mapping["c_proj.bias"] = ["mlp.c_proj.bias", "c_proj.bias", "ffn_down.bias"]
        if self.has_qk_norm:
            optional_mapping["q_norm.weight"] = ["self_attn.q_norm.weight", "q_norm.weight", "attn_q_norm.weight"]
            optional_mapping["k_norm.weight"] = ["self_attn.k_norm.weight", "k_norm.weight", "attn_k_norm.weight"]
        if self.has_4_norms:
            optional_mapping["pre_feedforward_layernorm.weight"] = [
                "pre_feedforward_layernorm.weight",
                "ffn_pre_norm.weight",
                "ffn_norm.weight",
            ]
            optional_mapping["post_feedforward_layernorm.weight"] = [
                "post_feedforward_layernorm.weight",
                "ffn_post_norm.weight",
                "post_ffw_norm.weight",
            ]

        # Build clean lookup from incoming state_dict
        clean_incoming = {}
        for k, v in state_dict.items():
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
            matched_alias = None
            for alias in aliases:
                if alias in clean_incoming:
                    matched_tensor = clean_incoming[alias]
                    matched_alias = alias
                    break

            if matched_tensor is None:
                raise ValueError(
                    f"Incomplete layer weights: required parameter '{param_name}' is missing for layer {self.layer_idx}"
                )

            # Avoid reusing generic fallback ffn_norm.weight into pre_feedforward_layernorm when it filled post_attn fallback
            if (
                param_name == "post_attention_layernorm.weight"
                and self.has_4_norms
                and matched_alias == "ffn_norm.weight"
                and "pre_feedforward_layernorm.weight" in optional_mapping
            ):
                optional_mapping["pre_feedforward_layernorm.weight"] = [
                    a for a in optional_mapping["pre_feedforward_layernorm.weight"] if a != "ffn_norm.weight"
                ]

            my_dict = self._copy_validated_param(param_name, matched_tensor, my_dict)

        for param_name, aliases in optional_mapping.items():
            for alias in aliases:
                if alias in clean_incoming:
                    my_dict = self._copy_validated_param(param_name, clean_incoming[alias], my_dict)
                    break

        logger.debug(f"Block {self.layer_idx}: Loaded and validated all {len(required_params)} required parameter tensors.")
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
    eps: Optional[float] = None,
    rope_theta: Optional[float] = None,
    sliding_window: Any = _UNSET,
    query_pre_attn_scalar: Optional[float] = None,
    attn_logit_softcapping: Any = _UNSET,
    qkv_bias: Optional[bool] = None,
    o_bias: Optional[bool] = None,
    mlp_bias: Optional[bool] = None,
    hidden_act: Optional[str] = None,
    dtype: Optional[Any] = None,
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

    # Clean incoming key names and map to canonical names
    clean_incoming = {}
    for k, v in state_dict.items():
        short_k = k
        if "layers." in short_k:
            short_k = short_k.split("layers.", 1)[1]
            if "." in short_k:
                short_k = short_k.split(".", 1)[1]
        elif "blk." in short_k:
            short_k = short_k.split("blk.", 1)[1]
            if "." in short_k:
                short_k = short_k.split(".", 1)[1]
        canonical_k = normalize_canonical_tensor_name(short_k)
        clean_incoming[short_k] = v
        clean_incoming[k] = v
        clean_incoming[canonical_k] = v

    # Derive architecture dimensions dynamically from verified tensor shapes
    actual_h_dim = hidden_dim
    for k in ("input_layernorm.weight", "attention_norm.weight", "attn_norm.weight", "ln_1.weight"):
        if k in clean_incoming and isinstance(clean_incoming[k], torch.Tensor) and clean_incoming[k].dim() >= 1:
            actual_h_dim = clean_incoming[k].shape[0]
            break

    actual_inter = intermediate_size
    for k in ("gate_proj.weight", "mlp.gate_proj.weight", "mlp.c_fc.weight", "c_fc.weight", "ffn_up.weight"):
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

    detected_qkv_bias = qkv_bias
    if detected_qkv_bias is None and any(bk in clean_incoming for bk in ("attn_q.bias", "q_proj.bias", "self_attn.q_proj.bias")):
        detected_qkv_bias = True
    detected_o_bias = o_bias
    if detected_o_bias is None and any(bk in clean_incoming for bk in ("attn_output.bias", "o_proj.bias", "self_attn.o_proj.bias")):
        detected_o_bias = True
    detected_mlp_bias = mlp_bias
    if detected_mlp_bias is None and any(bk in clean_incoming for bk in ("c_fc.bias", "mlp.c_fc.bias", "c_proj.bias", "mlp.c_proj.bias", "ffn_up.bias", "ffn_down.bias")):
        detected_mlp_bias = True

    # Instantiate block with dynamically verified dimensions
    block = RealTransformerBlock(
        hidden_dim=actual_h_dim,
        num_heads=actual_heads,
        num_kv_heads=actual_kv_heads,
        intermediate_size=actual_inter,
        head_dim=derived_head_dim,
        architecture=architecture,
        eps=eps,
        layer_idx=layer_idx,
        dtype=dtype,
        rope_theta=rope_theta,
        sliding_window=sliding_window,
        query_pre_attn_scalar=query_pre_attn_scalar,
        attn_logit_softcapping=attn_logit_softcapping,
        qkv_bias=detected_qkv_bias,
        o_bias=detected_o_bias,
        mlp_bias=detected_mlp_bias,
        hidden_act=hidden_act,
    )

    # Load and validate tensors (raises ValueError if incomplete, non-finite, or invalid)
    block.load_layer_weights(state_dict)
    block.to(device)

    # Verification: run a test forward pass to confirm numerical validity
    with torch.no_grad():
        first_param = next(block.parameters(), None)
        test_dtype = first_param.dtype if first_param is not None else torch.float32
        test_in = torch.randn(1, 1, actual_h_dim, device=device, dtype=test_dtype)
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
    past_kv: Optional[Any] = None,
) -> bool:
    """
    Verifies that installed weights in RealTransformerBlock produce outputs mathematically
    equivalent to a reference transformer decoder block (e.g. HuggingFace LlamaDecoderLayer / GemmaDecoderLayer)
    within the specified numerical tolerance for both prompt prefill and cached single-token autoregressive generation.
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
        out_block, block_new_kv = block(x_block, past_kv=past_kv)

        x_ref = test_input.clone()
        try:
            ref_res = ref_block(x_ref, past_kv=past_kv)
        except TypeError:
            try:
                ref_res = ref_block(x_ref)
            except TypeError:
                ref_res = ref_block(hidden_states=x_ref)

        ref_new_kv = None
        if isinstance(ref_res, (tuple, list)):
            out_ref = ref_res[0]
            if len(ref_res) > 1:
                ref_new_kv = ref_res[1]
        elif hasattr(ref_res, "hidden_states"):
            out_ref = ref_res.hidden_states
            ref_new_kv = getattr(ref_res, "past_key_values", None)
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

        if past_kv is not None and block_new_kv is not None and ref_new_kv is not None:
            if isinstance(block_new_kv, (tuple, list)) and isinstance(ref_new_kv, (tuple, list)):
                if len(block_new_kv) >= 2 and len(ref_new_kv) >= 2:
                    k_match = torch.allclose(block_new_kv[0], ref_new_kv[0], atol=tolerance, rtol=tolerance)
                    v_match = torch.allclose(block_new_kv[1], ref_new_kv[1], atol=tolerance, rtol=tolerance)
                    if not (k_match and v_match):
                        logger.warning("Numerical parity divergence in cached KV states.")
                        return False

        return True


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
        dtype: Optional[Any] = None,
        rms_norm_eps: Optional[float] = None,
        rope_theta: Optional[float] = None,
        rope_local_base_freq: Optional[float] = None,
        query_pre_attn_scalar: Optional[float] = None,
        attn_logit_softcapping: Any = _UNSET,
        final_logit_softcapping: Any = _UNSET,
        sliding_window: Any = _UNSET,
        sliding_window_pattern: Optional[int] = None,
        qkv_bias: Optional[bool] = None,
        o_bias: Optional[bool] = None,
        mlp_bias: Optional[bool] = None,
        hidden_act: Optional[str] = None,
        _lazy_alloc: bool = False,
    ):
        super().__init__()
        if not is_architecture_supported(architecture):
            raise ValueError(
                f"Unsupported model architecture '{architecture}'. "
                f"DistributedStageModel requires a supported decoder architecture "
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

        spec = _get_arch_spec(architecture)
        self.spec = spec
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

        self.rms_norm_eps = float(rms_norm_eps if rms_norm_eps is not None else spec["norm_eps"])
        self.rope_theta = float(rope_theta if rope_theta is not None else spec["rope_theta"])
        self.rope_local_base_freq = (
            rope_local_base_freq if rope_local_base_freq is not None else spec["rope_local_base_freq"]
        )
        self.query_pre_attn_scalar = (
            query_pre_attn_scalar if query_pre_attn_scalar is not None else spec["query_pre_attn_scalar"]
        )
        self.attn_logit_softcapping = (
            spec["attn_logit_softcapping"] if attn_logit_softcapping is _UNSET else attn_logit_softcapping
        )
        self.final_logit_softcapping = (
            spec["final_logit_softcapping"] if final_logit_softcapping is _UNSET else final_logit_softcapping
        )
        self.sliding_window_cfg = spec["sliding_window"] if sliding_window is _UNSET else sliding_window
        self.sliding_window_pattern = (
            sliding_window_pattern if sliding_window_pattern is not None else spec["sliding_window_pattern"]
        )
        self.qkv_bias = spec["qkv_bias"] if qkv_bias is None else bool(qkv_bias)
        self.o_bias = spec["o_bias"] if o_bias is None else bool(o_bias)
        self.mlp_bias = spec["mlp_bias"] if mlp_bias is None else bool(mlp_bias)
        self.hidden_act = str(hidden_act or spec["hidden_act"])
        self.embed_scale = math.sqrt(float(hidden_dim)) if spec["embed_scale_sqrt"] else 1.0

        safe_device = device
        if safe_device.startswith("cuda") and not (HAS_TORCH and torch.cuda.is_available()):
            logger.warning(f"CUDA device '{device}' requested but PyTorch CUDA runtime is not available. Safely falling back to CPU.")
            safe_device = "cpu"
        self.device = safe_device
        self.dtype = dtype if dtype is not None else (
            torch.float16 if (self.device != "cpu" and HAS_TORCH and torch.cuda.is_available() and "cuda" in str(self.device)) else (torch.float32 if HAS_TORCH else None)
        )
        self.progressive = progressive

        # JIT Progressive layer readiness tracking
        self._layer_events: Dict[int, threading.Event] = {
            i: threading.Event() for i in range(layer_start, layer_end)
        }
        if progressive:
            self.ready_layers: Set[int] = set()
        else:
            self.ready_layers = set(range(layer_start, layer_end))
            for ev in self._layer_events.values():
                ev.set()

        if not HAS_TORCH:
            return

        if _lazy_alloc:
            # Weights & layer blocks will be populated sequentially onto target device without pre-allocating empty buffers
            self.embed_tokens = None
            self.early_exit_head = None
            self.layers = nn.ModuleList([nn.Identity() for _ in range(layer_end - layer_start)])
            self.norm = (
                _build_norm(
                    hidden_dim,
                    eps=self.rms_norm_eps,
                    norm_type=self.spec["norm_type"],
                    is_gemma=self.spec["norm_unit_offset"],
                    device=self.device,
                    dtype=self.dtype,
                )
                if self.is_last_stage
                else None
            )
            self.lm_head = None
            if self.norm is not None and self.device != "cpu":
                self.norm.to(self.device)
            return

        # 1. Embeddings (ONLY if first stage)
        if self.is_first_stage:
            embed_dev = "cpu" if self.progressive else self.device
            self.embed_tokens = _fast_embedding(vocab_size, hidden_dim, device=embed_dev, dtype=self.dtype)
            self.early_exit_head = None
        else:
            self.embed_tokens = None
            self.early_exit_head = None

        # 2. Assigned Transformer Layers (always genuine RealTransformerBlock)
        # When progressive=True, layers initialize on CPU and are hot-swapped onto target device Just-In-Time as wire chunks arrive
        num_layers = layer_end - layer_start
        layer_dev = "cpu" if self.progressive else self.device
        layers_list = []
        for i in range(num_layers):
            layer_idx = layer_start + i
            layer_rope, layer_sw = _resolve_layer_sliding_and_rope(
                self.spec,
                layer_idx,
                rope_theta=self.rope_theta,
                rope_local_base_freq=self.rope_local_base_freq,
                sliding_window=self.sliding_window_cfg,
                sliding_window_pattern=self.sliding_window_pattern,
            )
            layers_list.append(
                RealTransformerBlock(
                    hidden_dim=hidden_dim,
                    num_heads=self.num_heads,
                    num_kv_heads=self.num_kv_heads,
                    intermediate_size=self.intermediate_size,
                    head_dim=self.head_dim,
                    architecture=self.architecture,
                    eps=self.rms_norm_eps,
                    layer_idx=layer_idx,
                    device=layer_dev,
                    dtype=self.dtype,
                    rope_theta=layer_rope,
                    sliding_window=layer_sw,
                    query_pre_attn_scalar=self.query_pre_attn_scalar,
                    attn_logit_softcapping=self.attn_logit_softcapping,
                    qkv_bias=self.qkv_bias,
                    o_bias=self.o_bias,
                    mlp_bias=self.mlp_bias,
                    hidden_act=self.hidden_act,
                )
            )
        self.layers = nn.ModuleList(layers_list)

        # 3. Norm and LM Head (ONLY if last stage)
        if self.is_last_stage:
            head_dev = "cpu" if self.progressive else self.device
            self.norm = _build_norm(
                hidden_dim,
                eps=self.rms_norm_eps,
                norm_type=self.spec["norm_type"],
                is_gemma=self.spec["norm_unit_offset"],
                device=head_dev,
                dtype=self.dtype,
            )
            self.lm_head = _fast_linear(hidden_dim, vocab_size, device=head_dev, dtype=self.dtype, bias=False)
        else:
            self.norm = None
            self.lm_head = None

        if not self.progressive and self.device != "cpu":
            try:
                self.to(self.device)
            except Exception as e:
                logger.warning(f"Could not load stage model onto {self.device} ({e}). Falling back safely to CPU.")
                self.device = "cpu"
                self.to("cpu")

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
            src = weights.to(device=self.device)
            if src.shape != target_shape:
                if src.dim() == 2 and src.T.shape == target_shape:
                    src = src.T
                else:
                    raise ValueError(
                        f"Embedding weight shape mismatch: expected {tuple(target_shape)}, got {tuple(src.shape)}. "
                        f"Partial copying is rejected."
                    )
            if getattr(self.embed_tokens.weight, "is_meta", False):
                self.embed_tokens = nn.Embedding.from_pretrained(src.to(dtype=torch.float16), freeze=False)
            else:
                self.embed_tokens.weight.copy_(src.to(dtype=self.embed_tokens.weight.dtype))
            logger.info(f"Loaded verified embedding weights {tuple(src.shape)} into first stage.")

    def load_head_weights(
        self,
        lm_head_weight: Any,
        norm_weight: Optional[Any] = None,
        norm_bias: Optional[Any] = None,
    ):
        """Loads verified pre-trained final norm and LM Head weights into the last stage."""
        if not self.is_last_stage:
            return
        if not HAS_TORCH:
            return
        with torch.no_grad():
            if norm_weight is not None and self.norm is not None:
                norm_label = "LayerNorm" if isinstance(self.norm, nn.LayerNorm) else "RMSNorm"
                if not isinstance(norm_weight, torch.Tensor):
                    raise ValueError(
                        f"Expected torch.Tensor for {norm_label} weights, got {type(norm_weight).__name__}."
                    )
                n_src = norm_weight.to(device=self.device)
                if n_src.shape != self.norm.weight.shape:
                    raise ValueError(
                        f"{norm_label} weight shape mismatch: expected {tuple(self.norm.weight.shape)}, got {tuple(n_src.shape)}. "
                        f"Partial copying is rejected."
                    )
                self.norm.weight.copy_(n_src.to(dtype=self.norm.weight.dtype))
            if norm_bias is not None and self.norm is not None and getattr(self.norm, "bias", None) is not None:
                if not isinstance(norm_bias, torch.Tensor):
                    raise ValueError(
                        f"Expected torch.Tensor for LayerNorm bias, got {type(norm_bias).__name__}."
                    )
                b_src = norm_bias.to(device=self.device)
                if b_src.shape != self.norm.bias.shape:
                    raise ValueError(
                        f"LayerNorm bias shape mismatch: expected {tuple(self.norm.bias.shape)}, got {tuple(b_src.shape)}. "
                        f"Partial copying is rejected."
                    )
                self.norm.bias.copy_(b_src.to(dtype=self.norm.bias.dtype))
            if lm_head_weight is not None and self.lm_head is not None:
                if not isinstance(lm_head_weight, torch.Tensor):
                    raise ValueError(
                        f"Expected torch.Tensor for LM Head weights, got {type(lm_head_weight).__name__}."
                    )
                h_src = lm_head_weight.to(device=self.device)
                target_shape = self.lm_head.weight.shape
                if h_src.shape != target_shape:
                    if h_src.dim() == 2 and h_src.T.shape == target_shape:
                        h_src = h_src.T
                    else:
                        raise ValueError(
                            f"LM Head weight shape mismatch: expected {tuple(target_shape)}, got {tuple(h_src.shape)}. "
                            f"Partial copying is rejected."
                        )
                self.lm_head.weight.copy_(h_src.to(dtype=self.lm_head.weight.dtype))
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
            layer_rope, layer_sw = _resolve_layer_sliding_and_rope(
                self.spec,
                layer_index,
                rope_theta=self.rope_theta,
                rope_local_base_freq=self.rope_local_base_freq,
                sliding_window=self.sliding_window_cfg,
                sliding_window_pattern=self.sliding_window_pattern,
            )
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
                eps=self.rms_norm_eps,
                rope_theta=layer_rope,
                sliding_window=layer_sw,
                query_pre_attn_scalar=self.query_pre_attn_scalar,
                attn_logit_softcapping=self.attn_logit_softcapping,
                qkv_bias=self.qkv_bias,
                o_bias=self.o_bias,
                mlp_bias=self.mlp_bias,
                hidden_act=self.hidden_act,
                dtype=self.dtype,
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
        if layer_index in self.ready_layers:
            return True
        if layer_index in self._layer_events:
            return self._layer_events[layer_index].wait(timeout=timeout)
        return False

    def is_entrance_ready(self) -> bool:
        return self.layer_start in self.ready_layers

    def get_ready_count(self) -> int:
        return len(self.ready_layers)

    def forward_first_stage(
        self,
        input_ids: "torch.Tensor",
        past_kvs: Optional[List[Any]] = None,
        position_ids: Optional[Any] = None,
    ):
        """First stage: input_ids -> embeddings -> assigned layers -> intermediate activation."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for model inference")
        if not self.is_first_stage:
            raise RuntimeError("Called forward_first_stage on a non-first stage worker")

        if self.embed_tokens is not None:
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
        if getattr(self, "embed_scale", 1.0) != 1.0:
            scale = torch.tensor(self.embed_scale, device=h.device, dtype=h.dtype)
            h = h * scale
        new_kvs = []
        for i, layer in enumerate(self.layers):
            layer_idx = self.layer_start + i
            if not self.wait_layer_ready(layer_idx):
                raise RuntimeError(f"Layer {layer_idx} is not ready or failed validation before forward execution")
            layer_past_kv = past_kvs[i] if past_kvs else None
            h, new_kv = layer(h, layer_past_kv, position_ids=position_ids)
            new_kvs.append(new_kv)

        return h, new_kvs

    def forward_intermediate_stage(
        self,
        hidden_states: "torch.Tensor",
        past_kvs: Optional[List[Any]] = None,
        position_ids: Optional[Any] = None,
    ):
        """Middle or last stage: incoming activations -> assigned layers."""
        if not HAS_TORCH:
            raise RuntimeError("PyTorch is required for model inference")
        h = hidden_states
        new_kvs = []
        for i, layer in enumerate(self.layers):
            layer_idx = self.layer_start + i
            if not self.wait_layer_ready(layer_idx):
                raise RuntimeError(f"Layer {layer_idx} is not ready or failed validation before forward execution")
            layer_past_kv = past_kvs[i] if past_kvs else None
            h, new_kv = layer(h, layer_past_kv, position_ids=position_ids)
            new_kvs.append(new_kv)

        if self.is_last_stage:
            h = self.norm(h)
            logits = self.lm_head(h)
            cap = getattr(self, "final_logit_softcapping", None)
            if cap is not None and cap > 0:
                logits = (torch.tanh(logits.float() / float(cap)) * float(cap)).to(logits.dtype)
            return logits, new_kvs

        return h, new_kvs


def find_local_safetensors_files(model_name: str) -> List[str]:
    """Finds genuine safetensors or PyTorch checkpoint files on disk for the given model name."""
    import glob
    import os
    candidates = []

    # 1. Direct path
    if os.path.isdir(model_name):
        candidates.append(model_name)
    elif os.path.isfile(model_name):
        if not model_name.endswith(".gguf"):
            return [model_name]

    # 2. Hugging Face hub cache (~/.cache/huggingface/hub/models--*)
    hub_dir = os.path.expanduser("~/.cache/huggingface/hub")
    if os.path.isdir(hub_dir):
        clean_name = model_name.replace("/", "--").replace(":", "--").lower()
        prefix = model_name.split(":")[0].split("/")[0].lower()
        known_prefixes = ("gemma", "qwen", "llama", "mistral", "deepseek", "starcoder2")
        for d in os.listdir(hub_dir):
            d_lower = d.lower()
            if clean_name in d_lower or (prefix in d_lower and any(kp in prefix for kp in known_prefixes)):
                snapshots_dir = os.path.join(hub_dir, d, "snapshots")
                if os.path.isdir(snapshots_dir):
                    for snap in os.listdir(snapshots_dir):
                        snap_path = os.path.join(snapshots_dir, snap)
                        if os.path.isdir(snap_path):
                            candidates.append(snap_path)

    # Search candidates for .safetensors or .bin files (genuine unquantized PyTorch weights)
    results = []
    for cand in candidates:
        if os.path.isfile(cand):
            if not cand.endswith(".gguf"):
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

    return results


def find_local_ollama_gguf_file(model_name: str) -> Optional[str]:
    """Resolves a locally installed Ollama model tag to its on-disk GGUF blob path, if present."""
    import json
    import os
    import pathlib

    if not model_name:
        return None

    ollama_root = pathlib.Path(os.environ.get("OLLAMA_MODELS") or os.path.expanduser("~/.ollama/models"))
    manifests_root = ollama_root / "manifests" / "registry.ollama.ai"
    blobs_dir = ollama_root / "blobs"
    if not manifests_root.is_dir() or not blobs_dir.is_dir():
        return None

    candidates_names = [model_name.strip()]
    try:
        from situs.inference_backend import normalize_ollama_model
        norm_name = normalize_ollama_model(model_name)
        if norm_name and norm_name not in candidates_names:
            candidates_names.append(norm_name)
    except Exception:
        pass

    manifest_candidates: List[pathlib.Path] = []
    for cand in candidates_names:
        if "/" in cand:
            ns_repo, _, tag = cand.partition(":")
            repo_parts = ns_repo.split("/")
            base = manifests_root.joinpath(*repo_parts) if len(repo_parts) >= 2 else manifests_root / "library" / repo_parts[0]
        else:
            repo, _, tag = cand.partition(":")
            base = manifests_root / "library" / repo
        tag = tag or "latest"
        manifest_candidates.append(base / tag)
        manifest_candidates.append(base / "latest")
        if base.is_dir():
            # Fallback to first matching tag file in repo directory
            for f in sorted(base.iterdir()):
                if f.is_file() and f not in manifest_candidates:
                    manifest_candidates.append(f)

    seen = set()
    for mf in manifest_candidates:
        if mf in seen or not mf.is_file():
            continue
        seen.add(mf)
        try:
            data = json.loads(mf.read_text(encoding="utf-8"))
            for layer in data.get("layers", []):
                if layer.get("mediaType") == "application/vnd.ollama.image.model":
                    digest = str(layer.get("digest", ""))
                    if not digest:
                        continue
                    blob = blobs_dir / digest.replace(":", "-")
                    if blob.is_file():
                        with open(blob, "rb") as probe:
                            if probe.read(4) == b"GGUF":
                                return str(blob)
        except Exception:
            continue
    return None


def find_local_weight_files(model_name: str) -> List[str]:
    """Finds local weights from HuggingFace safetensors/bin, direct GGUF paths, or Ollama GGUF blobs."""
    sfiles = find_local_safetensors_files(model_name)
    if sfiles:
        return sfiles
    if model_name and os.path.isfile(model_name) and model_name.lower().endswith(".gguf"):
        return [model_name]
    gguf_blob = find_local_ollama_gguf_file(model_name)
    if gguf_blob:
        return [gguf_blob]
    return []


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
            f"DistributedStageModel requires a supported decoder architecture "
            f"(supported: {sorted(SUPPORTED_ARCHITECTURES)}). Incompatible architectures (e.g. MoE, MPT, GPT-2) are rejected."
        )

    hidden_dim = meta["hidden_dim"]
    vocab_size = meta["vocab_size"]
    num_heads = meta["num_heads"]
    num_kv_heads = meta.get("num_kv_heads") or num_heads
    intermediate_size = meta["intermediate_size"]
    head_dim = meta.get("head_dim") or max(1, hidden_dim // max(1, num_heads))

    meta_dtype_str = meta.get("torch_dtype") or meta.get("dtype")
    stage_dtype = None
    if HAS_TORCH and meta_dtype_str and isinstance(meta_dtype_str, str):
        if "bfloat16" in meta_dtype_str.lower() and hasattr(torch, "bfloat16"):
            stage_dtype = torch.bfloat16
        elif "float16" in meta_dtype_str.lower():
            stage_dtype = torch.float16
        elif "float32" in meta_dtype_str.lower():
            stage_dtype = torch.float32

    use_bias_cfg = meta.get("use_bias")
    qkv_bias_cfg = meta.get("attention_bias") if meta.get("attention_bias") is not None else use_bias_cfg
    o_bias_cfg = use_bias_cfg
    mlp_bias_cfg = meta.get("mlp_bias") if meta.get("mlp_bias") is not None else use_bias_cfg

    sfiles: List[str] = []
    has_gguf_source = False
    fast_gguf_files: Dict[str, FastGGUFFile] = {}
    if not progressive and HAS_TORCH:
        sfiles = find_local_weight_files(model_name)
        for sf in sfiles:
            if os.path.isfile(sf):
                try:
                    with open(sf, "rb") as probe_f:
                        if probe_f.read(4) == b"GGUF":
                            has_gguf_source = True
                            try:
                                probe_reader = FastGGUFFile(sf)
                                fast_gguf_files[sf] = probe_reader
                                for pt in probe_reader.tensors:
                                    if pt.name in ("token_embd.weight", "model.embed_tokens.weight") and len(pt.shape) == 2:
                                        vocab_size = int(pt.shape[1])
                                        break
                            except Exception:
                                pass
                            break
                except Exception:
                    pass

    if has_gguf_source and HAS_TORCH:
        if "cuda" in str(device) and torch.cuda.is_available():
            stage_dtype = torch.float16
        else:
            stage_dtype = torch.float32

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
        head_dim=head_dim,
        architecture=arch,
        dtype=stage_dtype,
        rms_norm_eps=meta.get("rms_norm_eps"),
        rope_theta=meta.get("rope_theta"),
        rope_local_base_freq=meta.get("rope_local_base_freq"),
        query_pre_attn_scalar=meta.get("query_pre_attn_scalar"),
        attn_logit_softcapping=meta["attn_logit_softcapping"] if meta.get("attn_logit_softcapping") is not None else _UNSET,
        final_logit_softcapping=meta["final_logit_softcapping"] if meta.get("final_logit_softcapping") is not None else _UNSET,
        sliding_window=meta["sliding_window"] if meta.get("sliding_window") is not None else _UNSET,
        sliding_window_pattern=meta.get("sliding_window_pattern"),
        qkv_bias=qkv_bias_cfg,
        o_bias=o_bias_cfg,
        mlp_bias=mlp_bias_cfg,
        hidden_act=meta.get("hidden_act"),
        _lazy_alloc=(has_gguf_source and not progressive),
    )

    if not progressive and HAS_TORCH:
        from situs.worker.jit_streamer import is_layer_cached, load_layer_from_cache, save_layer_to_cache
        loaded_cached = 0
        if not has_gguf_source:
            for l_idx in range(layer_start, layer_end):
                if is_layer_cached(model_name, l_idx):
                    cached = load_layer_from_cache(model_name, l_idx)
                    if cached:
                        try:
                            stage.hot_swap_layer(l_idx, weights=cached)
                            loaded_cached += 1
                        except Exception as e:
                            logger.debug(f"Could not load cached weights for layer {l_idx}: {e}")

        if sfiles:
            from situs.worker.jit_streamer import DEFAULT_CACHE_DIR
            safe_name = model_name.replace("/", "_").replace(":", "_")
            model_cache_dir = DEFAULT_CACHE_DIR / safe_name
            embed_cache_file = model_cache_dir / "embed_tokens.pt"
            norm_cache_file = model_cache_dir / "norm.pt"
            norm_bias_cache_file = model_cache_dir / "norm_bias.pt"
            head_cache_file = model_cache_dir / "lm_head.pt"

            cached_embed = None
            if not has_gguf_source and embed_cache_file.is_file():
                try:
                    cached_embed = torch.load(embed_cache_file, weights_only=True)
                    if is_first_stage:
                        stage.load_embedding_weights(cached_embed)
                except Exception:
                    cached_embed = None

            cached_head = None
            cached_norm = None
            cached_norm_bias = None
            if not has_gguf_source and is_last_stage:
                if norm_cache_file.is_file():
                    try:
                        cached_norm = torch.load(norm_cache_file, weights_only=True)
                    except Exception:
                        pass
                if norm_bias_cache_file.is_file():
                    try:
                        cached_norm_bias = torch.load(norm_bias_cache_file, weights_only=True)
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
                    stage.load_head_weights(
                        lm_head_weight=cached_head,
                        norm_weight=cached_norm,
                        norm_bias=cached_norm_bias,
                    )

            need_embed = is_first_stage and cached_embed is None
            need_head = is_last_stage and (cached_head is None or cached_norm is None)
            all_layers_ready = (not has_gguf_source) and all(
                stage.is_layer_ready(l_idx) for l_idx in range(layer_start, layer_end)
            )

            if all_layers_ready and not need_embed and not need_head:
                logger.info(f"All stage [{layer_start}..{layer_end}) layers and embeddings loaded from local disk cache.")
                sfiles = []

            # 1. Try FastGGUFFile reader if file has GGUF format (handles quantized Ollama blobs & unquantized GGUF)
            gguf_loaded_files = set()
            for sf in sfiles:
                if not os.path.isfile(sf):
                    continue
                is_gguf = sf in fast_gguf_files
                if not is_gguf:
                    try:
                        with open(sf, "rb") as probe_f:
                            is_gguf = (probe_f.read(4) == b"GGUF")
                    except Exception:
                        is_gguf = False

                if is_gguf:
                    gguf_loaded_files.add(sf)
                    reader = fast_gguf_files.get(sf) or FastGGUFFile(sf)
                    try:
                        tensor_by_name = {t.name: t for t in reader.tensors}
                        use_int8 = bool(load_in_8bit and any(t.tensor_type not in (0, 1, 30) for t in reader.tensors))

                        # Stage 1: Token embeddings
                        emb_entry = tensor_by_name.get("token_embd.weight") or tensor_by_name.get("model.embed_tokens.weight")
                        if is_first_stage and emb_entry is not None:
                            n_vocab = int(emb_entry.shape[1]) if len(emb_entry.shape) == 2 else stage.vocab_size
                            stage.vocab_size = n_vocab
                            chunks = _iter_dequantized_gguf_chunks(emb_entry, out_dtype=stage.dtype, target_device=stage.device)
                            if use_int8:
                                stage.embed_tokens = Int8WeightEmbedding.from_float_chunked(
                                    chunks, num_embeddings=n_vocab, embedding_dim=hidden_dim, device=stage.device, scale_dtype=stage.dtype
                                )
                            else:
                                emb_mod = _fast_embedding(n_vocab, hidden_dim, device=stage.device, dtype=stage.dtype)
                                with torch.no_grad():
                                    offset = 0
                                    for c in chunks:
                                        emb_mod.weight[offset : offset + c.shape[0]].copy_(c)
                                        offset += c.shape[0]
                                stage.embed_tokens = emb_mod
                            logger.info(f"Loaded GGUF embedding table {(n_vocab, hidden_dim)} ({'int8' if use_int8 else stage.dtype}) onto {stage.device}.")

                        # Stage N: Final norm & LM head
                        if is_last_stage:
                            set_norm_unit_offset(stage.norm, False)
                            norm_entry = (
                                tensor_by_name.get("output_norm.weight")
                                or tensor_by_name.get("norm.weight")
                                or tensor_by_name.get("model.norm.weight")
                            )
                            norm_bias_entry = (
                                tensor_by_name.get("output_norm.bias")
                                or tensor_by_name.get("norm.bias")
                                or tensor_by_name.get("model.norm.bias")
                            )
                            if norm_entry is not None and stage.norm is not None:
                                nw = _dequantize_gguf_tensor_fast(norm_entry, out_dtype=stage.norm.weight.dtype, target_device=stage.device)
                                with torch.no_grad():
                                    stage.norm.weight.copy_(nw.to(dtype=stage.norm.weight.dtype))
                            if norm_bias_entry is not None and stage.norm is not None and getattr(stage.norm, "bias", None) is not None:
                                nb = _dequantize_gguf_tensor_fast(norm_bias_entry, out_dtype=stage.norm.bias.dtype, target_device=stage.device)
                                with torch.no_grad():
                                    stage.norm.bias.copy_(nb.to(dtype=stage.norm.bias.dtype))

                            head_entry = tensor_by_name.get("output.weight") or tensor_by_name.get("lm_head.weight") or emb_entry
                            if head_entry is not None:
                                n_vocab = int(head_entry.shape[1]) if len(head_entry.shape) == 2 else stage.vocab_size
                                if use_int8 and head_entry is emb_entry and isinstance(stage.embed_tokens, Int8WeightEmbedding):
                                    stage.lm_head = Int8WeightLinear(
                                        hidden_dim, n_vocab, stage.embed_tokens.weight, stage.embed_tokens.weight_scale, None
                                    )
                                elif use_int8:
                                    h_chunks = _iter_dequantized_gguf_chunks(head_entry, out_dtype=stage.dtype, target_device=stage.device)
                                    tmp_emb = Int8WeightEmbedding.from_float_chunked(
                                        h_chunks, num_embeddings=n_vocab, embedding_dim=hidden_dim, device=stage.device, scale_dtype=stage.dtype
                                    )
                                    stage.lm_head = Int8WeightLinear(hidden_dim, n_vocab, tmp_emb.weight, tmp_emb.weight_scale, None)
                                else:
                                    lm_mod = _fast_linear(hidden_dim, n_vocab, device=stage.device, dtype=stage.dtype, bias=False)
                                    with torch.no_grad():
                                        offset = 0
                                        for c in _iter_dequantized_gguf_chunks(head_entry, out_dtype=stage.dtype, target_device=stage.device):
                                            lm_mod.weight[offset : offset + c.shape[0]].copy_(c)
                                            offset += c.shape[0]
                                    stage.lm_head = lm_mod
                                logger.info(f"Loaded GGUF LM head {(n_vocab, hidden_dim)} ({'int8' if use_int8 else stage.dtype}) onto {stage.device}.")

                        # Transformer layers [layer_start..layer_end)
                        for l_idx in range(layer_start, layer_end):
                            prefix = f"blk.{l_idx}."
                            matched = [t for t in reader.tensors if t.name.startswith(prefix)]
                            if not matched:
                                raise RuntimeError(f"GGUF file '{sf}' is missing assigned transformer layer {l_idx}.")

                            layer_rope, layer_sw = _resolve_layer_sliding_and_rope(
                                stage.spec,
                                l_idx,
                                rope_theta=stage.rope_theta,
                                rope_local_base_freq=stage.rope_local_base_freq,
                                sliding_window=stage.sliding_window_cfg,
                                sliding_window_pattern=stage.sliding_window_pattern,
                            )
                            layer_state = {}
                            for t in matched:
                                short_k = t.name[len(prefix):]
                                layer_state[short_k] = _dequantize_gguf_tensor_fast(
                                    t, out_dtype=stage.dtype, target_device=stage.device
                                )

                            block = RealTransformerBlock(
                                hidden_dim=hidden_dim,
                                num_heads=num_heads,
                                num_kv_heads=num_kv_heads,
                                intermediate_size=intermediate_size,
                                head_dim=head_dim,
                                architecture=arch,
                                eps=stage.rms_norm_eps,
                                layer_idx=l_idx,
                                device=stage.device,
                                dtype=stage.dtype,
                                rope_theta=layer_rope,
                                sliding_window=layer_sw,
                                query_pre_attn_scalar=stage.query_pre_attn_scalar,
                                attn_logit_softcapping=stage.attn_logit_softcapping,
                                qkv_bias=stage.qkv_bias,
                                o_bias=stage.o_bias,
                                mlp_bias=stage.mlp_bias,
                                hidden_act=stage.hidden_act,
                            )
                            set_norm_unit_offset(block, False)
                            block.load_layer_weights(layer_state)
                            del layer_state
                            if use_int8:
                                quantize_block_to_int8(block)

                            local_idx = l_idx - layer_start
                            stage.layers[local_idx] = block
                            stage.ready_layers.add(l_idx)
                            if l_idx in stage._layer_events:
                                stage._layer_events[l_idx].set()
                            loaded_cached += 1

                        if stage.device != "cpu" and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    finally:
                        reader.close()

            # 2. Try Safetensors loader
            if HAS_SAFETENSORS:
                try:
                    import safetensors.torch
                    embed_weight = None
                    norm_weight = None
                    norm_bias = None
                    lm_head_weight = None

                    for sf in sfiles:
                        if sf in gguf_loaded_files or sf.endswith(".gguf") or not os.path.isfile(sf):
                            continue
                        try:
                            with safetensors.safe_open(sf, framework="pt", device="cpu") as f:
                                f_keys = set(f.keys())
                                # Load embeddings for first stage OR tied-lm_head fallback on last stage
                                if embed_weight is None and (is_first_stage or (is_last_stage and lm_head_weight is None)):
                                    for ek in (
                                        "model.embed_tokens.weight",
                                        "language_model.model.embed_tokens.weight",
                                        "embed_tokens.weight",
                                        "token_embd.weight",
                                    ):
                                        if ek in f_keys:
                                            embed_weight = f.get_tensor(ek)
                                            break

                                # Check norm weight & optional bias (e.g. StarCoder2 LayerNorm)
                                if is_last_stage and norm_weight is None:
                                    for nk in (
                                        "model.norm.weight",
                                        "language_model.model.norm.weight",
                                        "norm.weight",
                                        "output_norm.weight",
                                    ):
                                        if nk in f_keys:
                                            norm_weight = f.get_tensor(nk)
                                            break
                                if is_last_stage and norm_bias is None:
                                    for nbk in (
                                        "model.norm.bias",
                                        "language_model.model.norm.bias",
                                        "norm.bias",
                                        "output_norm.bias",
                                    ):
                                        if nbk in f_keys:
                                            norm_bias = f.get_tensor(nbk)
                                            break

                                # Check lm_head
                                if is_last_stage and lm_head_weight is None:
                                    for hk in ("lm_head.weight", "language_model.lm_head.weight", "output.weight"):
                                        if hk in f_keys:
                                            lm_head_weight = f.get_tensor(hk)
                                            break

                                # Extract assigned layers
                                for l_idx in range(layer_start, layer_end):
                                    if stage.is_layer_ready(l_idx):
                                        continue
                                    prefixes = (
                                        f"model.layers.{l_idx}.",
                                        f"language_model.model.layers.{l_idx}.",
                                        f"layers.{l_idx}.",
                                        f"blk.{l_idx}.",
                                    )
                                    matched_keys = [k for k in f_keys if k.startswith(prefixes)]
                                    if matched_keys:
                                        layer_state = {}
                                        for k in matched_keys:
                                            short_k = k
                                            for p in prefixes:
                                                if short_k.startswith(p):
                                                    short_k = short_k[len(p):]
                                                    break
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

                        except Exception:
                            pass

                    if is_first_stage and embed_weight is not None:
                        stage.load_embedding_weights(embed_weight)
                    if is_last_stage:
                        if lm_head_weight is None and embed_weight is not None:
                            lm_head_weight = embed_weight
                        if lm_head_weight is not None or norm_weight is not None:
                            stage.load_head_weights(
                                lm_head_weight=lm_head_weight,
                                norm_weight=norm_weight,
                                norm_bias=norm_bias,
                            )
                except Exception as e:
                    logger.warning(f"Error loading safetensors for {model_name}: {e}")

        if loaded_cached > 0:
            logger.info(f"Loaded {loaded_cached} layer weights from disk for stage [{layer_start}..{layer_end})")

    logger.info(
        f"Initialized real stage [{layer_start}..{layer_end}) on {device}. "
        f"HiddenDim={hidden_dim}, Heads={num_heads}, Vocab={vocab_size}, First={is_first_stage}, Last={is_last_stage}"
    )
    return stage
