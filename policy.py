r"""Turning the transformer into a policy: sampling, log-probs and masks.

Sequence layout used everywhere (B rows, T = PROMPT_LEN + MAX_NEW columns):

    position:  0 1 2 3 4 5 | 6 7 8 9 10
    token:     0 7 + 4 5 = | 5 2 <eos> PAD PAD
               \_ prompt _/  \__ completion __/
    mask:                    1 1  1     0   0     <- completion_mask

Only completion tokens up to and including the first EOS count as actions.
"""

from functools import partial

import jax
import jax.numpy as jnp

from model import forward
from task import EOS, MAX_NEW, PAD, PROMPT_LEN


def policy_logits(logits, temperature):
    """The ONE place logits become the policy's distribution.

    Sampling and log-prob computation must both go through this, otherwise the
    log-probs you train on are not the log-probs of the policy that sampled.
    (A classic silent bug: sample at temperature 0.7, compute log-probs at 1.0.)
    """
    logits = logits.at[..., PAD].set(-1e9)  # the policy may never emit PAD
    return logits / temperature


@partial(jax.jit, static_argnames=("cfg", "temperature", "greedy"))
def generate(params, prompts, key, cfg, temperature=1.0, greedy=False):
    """prompts int [B, PROMPT_LEN] -> full sequences int [B, PROMPT_LEN + MAX_NEW].

    Re-runs the full forward pass for every new token (no KV cache). That is
    O(T^2) per token but T = 11 here, and it keeps the code obvious.
    """
    B = prompts.shape[0]
    seq = jnp.concatenate([prompts, jnp.full((B, MAX_NEW), PAD, jnp.int32)], axis=1)

    def step(i, carry):
        seq, done, key = carry
        pos = PROMPT_LEN + i
        # Causal attention means the PAD placeholders to the right of pos don't
        # affect the logits at pos - 1, so a fixed-size buffer is fine.
        logits = forward(params, seq, cfg)[:, pos - 1]
        logits = policy_logits(logits, temperature)
        key, sub = jax.random.split(key)
        tok = jnp.argmax(logits, -1) if greedy else jax.random.categorical(sub, logits)
        tok = jnp.where(done, PAD, tok)          # after EOS, just pad
        seq = seq.at[:, pos].set(tok)
        return seq, done | (tok == EOS), key

    done = jnp.zeros(B, dtype=bool)
    seq, _, _ = jax.lax.fori_loop(0, MAX_NEW, step, (seq, done, key))
    return seq


def completion_mask(completions):
    """1.0 for completion tokens up to and including the first EOS, 0.0 after.
    A completion with no EOS (ran out of room) keeps all MAX_NEW tokens."""
    is_eos = (completions == EOS).astype(jnp.int32)
    eos_strictly_before = jnp.cumsum(is_eos, axis=1) - is_eos
    return (eos_strictly_before == 0).astype(jnp.float32)


@partial(jax.jit, static_argnames=("cfg", "temperature"))
def completion_logprobs(params, seq, cfg, temperature=1.0):
    """log pi(token_t | tokens_<t) for each completion token: float [B, MAX_NEW].

    Values at padded positions are junk; always multiply by completion_mask.
    """
    logits = forward(params, seq[:, :-1], cfg)          # position t predicts t+1
    logp = jax.nn.log_softmax(policy_logits(logits, temperature), axis=-1)
    lp = jnp.take_along_axis(logp, seq[:, 1:, None], axis=-1)[..., 0]  # [B, T-1]
    return lp[:, PROMPT_LEN - 1:]                       # keep completion targets only
