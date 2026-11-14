"""Small sanity checks for the pieces that are easy to get silently wrong.

    python checks.py      # needs ckpt/warmstart.pkl
"""

import pickle

import jax
import jax.numpy as jnp
import numpy as np

import task
from grpo import STAGES, GRPOConfig, compute_advantages, grpo_loss
from policy import completion_logprobs, completion_mask, generate

E, P = task.EOS, task.PAD


def check_mask():
    comp = jnp.array([[5, 2, E, P, P],     # normal answer
                      [E, P, P, P, P],     # empty answer
                      [1, 2, 3, 4, 5]])    # never stopped: keep everything
    want = jnp.array([[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=jnp.float32)
    assert (completion_mask(comp) == want).all()
    print("ok  completion_mask keeps tokens up to and including the first EOS")


def check_advantages():
    r = jnp.array([1., 0., 0., 0.,   1., 1., 1., 1.])
    cfg = GRPOConfig(group_size=4, **STAGES[3])
    a = compute_advantages(r, cfg).reshape(2, 4)
    assert jnp.allclose(a[0].mean(), 0, atol=1e-5) and jnp.allclose(a[0].std(), 1, atol=1e-3)
    assert (a[1] == 0).all()
    print(f"ok  advantages: group [1,0,0,0] -> {np.round(np.asarray(a[0]), 2)}; "
          f"all-correct group -> {np.asarray(a[1])} (no signal)")


def check_first_inner_step(params, mcfg):
    """On the first inner step ρ = 1, so clipping can't bite: stage 4's gradient
    equals stage 3's. Clipping only matters from the second inner step on."""
    train_pairs, _ = task.split_pairs()
    prompts, answers = task.batch_prompts(np.random.default_rng(0), train_pairs, 8)
    prompts, answers = np.repeat(prompts, 8, 0), np.repeat(answers, 8)
    seq = generate(params, jnp.asarray(prompts), jax.random.PRNGKey(0), mcfg)
    mask = completion_mask(seq[:, task.PROMPT_LEN:])
    lp = jnp.where(mask > 0, completion_logprobs(params, seq, mcfg), 0.0)
    rewards = jnp.asarray(task.reward(np.asarray(seq[:, task.PROMPT_LEN:]), answers))
    grads = {}
    for s in (3, 4):
        cfg = GRPOConfig(**STAGES[s])
        adv = compute_advantages(rewards, cfg)
        grads[s] = jax.grad(lambda p: grpo_loss(p, seq, mask, adv, lp, lp, cfg, mcfg)[0])(params)
    diff = max(float(jnp.abs(a - b).max()) for a, b in
               zip(jax.tree.leaves(grads[3]), jax.tree.leaves(grads[4])))
    assert diff < 1e-6, diff
    _, m = grpo_loss(params, seq, mask, adv, lp, lp, GRPOConfig(**STAGES[5]), mcfg)
    assert abs(float(m["kl"])) < 1e-6
    print("ok  first inner step: clipped gradient == unclipped gradient; KL to itself == 0")


if __name__ == "__main__":
    check_mask()
    check_advantages()
    saved = pickle.load(open("ckpt/warmstart.pkl", "rb"))
    check_first_inner_step(jax.tree.map(jnp.asarray, saved["params"]), saved["cfg"])
