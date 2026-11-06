"""A small Qwen2-shaped decoder, plus an adapter for real Qwen2/2.5 weights.

The default random model validates inference mechanics; it is not a trained
language model. Pretrained loading is optional and uses standard HF modules as
weight containers. All KV management and decode attention run through our code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .selection import Selection, quest_selection, select_pages


@dataclass
class ModelConfig:
    vocab_size: int = 256
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0

    @property
    def head_dim(self):
        return self.hidden_size // self.num_attention_heads

    def __post_init__(self):
        if (
            self.hidden_size % self.num_attention_heads
            or self.num_attention_heads % self.num_key_value_heads
            or self.head_dim % 2
        ):
            raise ValueError("model dimensions must support grouped-query attention and RoPE")


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class Projections(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.q_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=True)
        self.k_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        self.v_proj = nn.Linear(cfg.hidden_size, cfg.num_key_value_heads * cfg.head_dim, bias=True)
        self.o_proj = nn.Linear(cfg.hidden_size, cfg.hidden_size, bias=False)


class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Projections(cfg)
        self.mlp = MLP(cfg)


class Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.num_hidden_layers))
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)


class Model(nn.Module):
    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        self.model = Decoder(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, name: str, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = AutoConfig.from_pretrained(name)
        # Transformers 4 stores rope_theta directly; 5 normalizes even standard
        # RoPE into rope_parameters (and may expose it through rope_scaling).
        rope = getattr(cfg, "rope_parameters", None) or getattr(cfg, "rope_scaling", None)
        rope = rope or {}
        if (
            cfg.model_type != "qwen2"
            or rope.get("rope_type", rope.get("type", "default")) != "default"
        ):
            raise ValueError("only Qwen2/Qwen2.5 with standard RoPE are supported")
        if getattr(cfg, "use_sliding_window", False) or cfg.hidden_act != "silu":
            raise ValueError("sliding window / non-SiLU models are unsupported")
        hf = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype).eval().to(device)
        result = cls.__new__(cls)
        nn.Module.__init__(result)
        fields = {
            name: getattr(cfg, name)
            for name in ModelConfig.__dataclass_fields__
            if name != "rope_theta"
        }
        fields["rope_theta"] = rope.get("rope_theta", getattr(cfg, "rope_theta", 10000.0))
        result.config = ModelConfig(**fields)
        result.model, result.lm_head = hf.model, hf.lm_head
        return result.eval()

    @property
    def device(self):
        return self.model.embed_tokens.weight.device

    @property
    def dtype(self):
        return self.model.embed_tokens.weight.dtype

    def _project(self, layer, x, start):
        cfg, length = self.config, len(x)
        x = layer.input_layernorm(x)
        q = layer.self_attn.q_proj(x).view(length, cfg.num_attention_heads, cfg.head_dim)
        k = layer.self_attn.k_proj(x).view(length, cfg.num_key_value_heads, cfg.head_dim)
        v = layer.self_attn.v_proj(x).view(length, cfg.num_key_value_heads, cfg.head_dim)
        inv = cfg.rope_theta ** (
            -torch.arange(0, cfg.head_dim, 2, device=x.device).float() / cfg.head_dim
        )
        positions = torch.arange(start, start + length, device=x.device).float()
        angles = positions[:, None] * inv[None, :]
        angles = torch.cat((angles, angles), -1)[:, None, :]
        cosine, sine = angles.cos().to(x.dtype), angles.sin().to(x.dtype)

        def rotate(tensor):
            a, b = tensor.chunk(2, -1)
            return tensor * cosine + torch.cat((-b, a), -1) * sine

        return rotate(q), rotate(k), v

    @staticmethod
    def _finish(layer, residual, attention):
        x = residual + layer.self_attn.o_proj(attention.reshape(len(residual), -1))
        return x + layer.mlp(layer.post_attention_layernorm(x))

    @torch.inference_mode()
    def prefill(self, tokens: list[int], cache=None, request_id=None):
        if not tokens:
            raise ValueError("prompt cannot be empty")
        if cache is not None and any(cache.storage.requests[request_id].lengths):
            raise ValueError("prefill is allowed exactly once per request")
        x = self.model.embed_tokens(torch.tensor(tokens, device=self.device))
        for index, layer in enumerate(self.model.layers):
            q, k, v = self._project(layer, x, 0)
            if cache is not None:
                cache.write(request_id, index, k, v)
            groups = self.config.num_attention_heads // self.config.num_key_value_heads
            attention = F.scaled_dot_product_attention(
                q.transpose(0, 1),
                k.repeat_interleave(groups, 1).transpose(0, 1),
                v.repeat_interleave(groups, 1).transpose(0, 1),
                is_causal=True,
            ).transpose(0, 1)
            x = self._finish(layer, x, attention)
        return self.lm_head(self.model.norm(x[-1])).float()

    @torch.inference_mode()
    def decode(
        self,
        token: int,
        request_id,
        cache,
        top_k=None,
        backend="auto",
        selector="quest",
    ):
        lengths = cache.storage.requests[request_id].lengths
        if len(set(lengths)) != 1 or lengths[0] == 0:
            raise ValueError("all layers must have the same nonempty history")
        position = lengths[0]
        x = self.model.embed_tokens(torch.tensor([token], device=self.device))
        for index, layer in enumerate(self.model.layers):
            q, k, v = self._project(layer, x, position)
            cache.write(request_id, index, k, v)
            if selector == "quest" and top_k is not None:
                request = cache.storage.requests[request_id]
                selection = quest_selection(
                    q[0],
                    request.minimum[index],
                    request.maximum[index],
                    request.lengths[index],
                    cache.config.page_size,
                    top_k,
                )
            elif selector in ("quest", "mean"):
                pages = select_pages(cache, request_id, index, q[0], top_k)
                selection = Selection.from_pages(
                    pages,
                    cache.storage.requests[request_id].lengths[index],
                    cache.config.page_size,
                    q.shape[1],
                )
            else:
                raise ValueError("selector must be quest or mean")
            attention = cache.attention(q[0], request_id, index, selection, backend)
            x = self._finish(layer, x, attention.unsqueeze(0))
        return self.lm_head(self.model.norm(x[0])).float()
