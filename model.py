"""A tiny decoder-only transformer in plain JAX.

Parameters are a nested dict (a pytree). That is the main reason this repo uses
plain JAX instead of Flax: for GRPO you need three versions of the same model
(the policy you train, the "old" policy that sampled the batch, and a frozen
reference), and with a dict each of those is just another variable.
"""

from dataclasses import dataclass

import jax
import jax.numpy as jnp


@dataclass(frozen=True)  # frozen => hashable => usable as a static jit argument
class ModelConfig:
    vocab_size: int
    max_len: int = 16
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_ff: int = 512


def init_params(key, cfg: ModelConfig):
    def dense(k, fan_in, fan_out):
        return jax.random.normal(k, (fan_in, fan_out)) * (fan_in ** -0.5)

    keys = iter(jax.random.split(key, 4 + 4 * cfg.n_layers))
    d = cfg.d_model
    params = {
        "tok_emb": jax.random.normal(next(keys), (cfg.vocab_size, d)) * 0.02,
        "pos_emb": jax.random.normal(next(keys), (cfg.max_len, d)) * 0.02,
        "ln_f": {"g": jnp.ones(d), "b": jnp.zeros(d)},
        "unembed": dense(next(keys), d, cfg.vocab_size) * 0.1,
        "layers": [],
    }
    for _ in range(cfg.n_layers):
        params["layers"].append({
            "ln1": {"g": jnp.ones(d), "b": jnp.zeros(d)},
            "qkv": dense(next(keys), d, 3 * d),
            "proj": dense(next(keys), d, d),
            "ln2": {"g": jnp.ones(d), "b": jnp.zeros(d)},
            "ff1": dense(next(keys), d, cfg.d_ff),
            "ff2": dense(next(keys), cfg.d_ff, d),
        })
    return params


def layer_norm(p, x, eps=1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return p["g"] * (x - mu) / jnp.sqrt(var + eps) + p["b"]


def attention(p, x, n_heads):
    B, T, D = x.shape
    q, k, v = jnp.split(x @ p["qkv"], 3, axis=-1)
    # [B, T, D] -> [B, H, T, D/H]
    split = lambda t: t.reshape(B, T, n_heads, D // n_heads).transpose(0, 2, 1, 3)
    q, k, v = split(q), split(k), split(v)
    scores = (q @ k.transpose(0, 1, 3, 2)) / jnp.sqrt(D // n_heads)
    causal = jnp.tril(jnp.ones((T, T), dtype=bool))
    scores = jnp.where(causal, scores, -1e9)
    out = jax.nn.softmax(scores, axis=-1) @ v
    return out.transpose(0, 2, 1, 3).reshape(B, T, D) @ p["proj"]


def forward(params, tokens, cfg: ModelConfig):
    """tokens: int [B, T] -> logits: float [B, T, vocab]. Position t predicts token t+1."""
    T = tokens.shape[1]
    x = params["tok_emb"][tokens] + params["pos_emb"][:T]
    for layer in params["layers"]:
        x = x + attention(layer, layer_norm(layer["ln1"], x), cfg.n_heads)
        h = layer_norm(layer["ln2"], x)
        x = x + jax.nn.gelu(h @ layer["ff1"]) @ layer["ff2"]
    return layer_norm(params["ln_f"], x) @ params["unembed"]
