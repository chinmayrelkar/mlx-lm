# Copyright © 2026 Apple Inc.

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu
from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .rope_utils import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    rms_norm_eps: float
    vocab_size: int
    head_dim: Optional[int] = None
    max_position_embeddings: Optional[int] = None
    attention_bias: bool = False
    mlp_bias: bool = False
    tie_word_embeddings: bool = True
    rope_theta: Optional[float] = None
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    rope_parameters: Optional[dict] = None
    factorized_embedding: bool = False
    embedding_rank: int = 768
    shared_kv: bool = False
    qk_norm: bool = False
    q_norm: bool = False
    layer_sharing: bool = False
    layer_sharing_repeats: int = 1

    def __post_init__(self):
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.rope_parameters:
            if self.rope_theta is None:
                self.rope_theta = self.rope_parameters.get("rope_theta")
            if self.rope_scaling is None:
                self.rope_scaling = self.rope_parameters
        if self.rope_theta is None:
            self.rope_theta = 1000000.0
        if self.layer_sharing and self.layer_sharing_repeats > 1:
            raise ValueError("Nandi layer sharing is not supported.")


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5
        bias = args.attention_bias

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.shared_kv = args.shared_kv
        if not self.shared_kv:
            self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=bias)

        self.use_q_norm = args.qk_norm or args.q_norm
        self.use_k_norm = args.qk_norm
        if self.use_q_norm:
            self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        if self.use_k_norm:
            self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x).reshape(B, L, self.n_heads, -1)
        k_raw = self.k_proj(x).reshape(B, L, self.n_kv_heads, -1)
        if self.use_q_norm:
            queries = self.q_norm(queries)
        keys = self.k_norm(k_raw) if self.use_k_norm else k_raw

        # With shared_kv, the values are the keys before norm and RoPE.
        if self.shared_kv:
            values = k_raw
        else:
            values = self.v_proj(x).reshape(B, L, self.n_kv_heads, -1)

        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim, hidden_dim, bias = args.hidden_size, args.intermediate_size, args.mlp_bias
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)

    def __call__(self, x) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.self_attn = Attention(args)
        self.mlp = MLP(args)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = x + self.self_attn(self.input_layernorm(x), mask, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class NandiModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        embed_dim = (
            args.embedding_rank if args.factorized_embedding else args.hidden_size
        )
        self.embed_tokens = nn.Embedding(args.vocab_size, embed_dim)
        if args.factorized_embedding:
            self.embedding_proj = nn.Linear(
                args.embedding_rank, args.hidden_size, bias=False
            )
        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)
        if self.args.factorized_embedding:
            h = self.embedding_proj(h)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])
        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, c)

        return self.norm(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = NandiModel(args)
        if args.factorized_embedding:
            self.lm_head_proj = nn.Linear(
                args.hidden_size, args.embedding_rank, bias=False
            )
        if not args.tie_word_embeddings:
            out_dim = (
                args.embedding_rank if args.factorized_embedding else args.hidden_size
            )
            self.lm_head = nn.Linear(out_dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
        input_embeddings: Optional[mx.array] = None,
    ):
        out = self.model(inputs, cache, input_embeddings)
        if self.args.factorized_embedding:
            out = self.lm_head_proj(out)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    def sanitize(self, weights):
        weights = {
            k: v for k, v in weights.items() if "self_attn.rotary_emb.inv_freq" not in k
        }
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers
