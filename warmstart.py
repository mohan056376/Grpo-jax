"""Stage 0: supervised warm start (SFT) so the policy is sometimes right.

RL on a randomly initialised model gets reward 0 on every sample, so every
advantage is 0 and nothing learns. LLM RL starts from a pretrained model for the
same reason. Here we fake "pretrained but unreliable" by training on labels that
are wrong p_noise of the time.

    python warmstart.py                  # writes ckpt/warmstart.pkl
"""

import argparse
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

import task
from model import ModelConfig, forward, init_params
from policy import completion_mask, generate


def evaluate(params, cfg, pairs, key, temperature=1.0):
    """Return (sampled accuracy, greedy accuracy) on the given (a, b) pairs."""
    prompts = jnp.array([task.make_prompt(a, b) for a, b in pairs], dtype=jnp.int32)
    answers = pairs.sum(1)
    out = {}
    for name, greedy in [("sampled", False), ("greedy", True)]:
        seq = generate(params, prompts, key, cfg, temperature=temperature, greedy=greedy)
        out[name] = float(task.reward(np.asarray(seq[:, task.PROMPT_LEN:]), answers).mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--p_noise", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="ckpt/warmstart.pkl")
    args = ap.parse_args()

    cfg = ModelConfig(vocab_size=task.VOCAB_SIZE)
    train_pairs, eval_pairs = task.split_pairs()
    rng = np.random.default_rng(args.seed)
    key = jax.random.PRNGKey(args.seed)
    params = init_params(key, cfg)
    opt = optax.adamw(optax.warmup_cosine_decay_schedule(0, args.lr, 100, args.steps))
    opt_state = opt.init(params)

    @jax.jit
    def train_step(params, opt_state, toks, mask):
        def loss_fn(p):
            logits = forward(p, toks[:, :-1], cfg)
            nll = optax.softmax_cross_entropy_with_integer_labels(logits, toks[:, 1:])
            m = mask[:, 1:]  # mask[t] marks target token t, which logits[t-1] predict
            return (nll * m).sum() / m.sum()
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = opt.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    t0 = time.time()
    for step in range(1, args.steps + 1):
        toks, mask = task.sft_batch(rng, train_pairs, args.batch, args.p_noise)
        params, opt_state, loss = train_step(params, opt_state, toks, mask)
        if step % 50 == 0 or step == args.steps:
            acc = evaluate(params, cfg, eval_pairs, jax.random.PRNGKey(step))
            print(f"sft step {step:5d}  loss {float(loss):.3f}  held-out acc: "
                  f"sampled {acc['sampled']:.3f}  greedy {acc['greedy']:.3f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump({"params": jax.device_get(params), "cfg": cfg}, f)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
