"""GRPO from scratch in JAX, built up in five stages you switch on with --stage.

The full objective (DeepSeekMath, Shao et al. 2024). For a prompt q, sample a
group of G completions o_1..o_G from the current policy, score each with a
verifiable reward r_i, and maximise

  J(θ) = 1/G Σ_i 1/|o_i| Σ_t [ min( ρ_it Â_i , clip(ρ_it, 1-ε, 1+ε) Â_i )  -  β KL_it ]

  ρ_it  = π_θ(o_it | q, o_i,<t) / π_old(o_it | q, o_i,<t)       (stage 4)
  Â_i   = (r_i - mean(r_1..r_G)) / std(r_1..r_G)                 (stages 2 and 3)
  KL_it = π_ref/π_θ - log(π_ref/π_θ) - 1   ("k3" estimator)      (stage 5)

The five stages switch these pieces on one at a time:

  --stage 1  sample a group, score it, plain REINFORCE with Â_i = r_i
  --stage 2  + group baseline:        Â_i = r_i - mean(r)        <- GRPO's key idea: no value net
  --stage 3  + group normalisation:   Â_i = (r_i - mean(r)) / std(r)
  --stage 4  + PPO clipping, reusing each batch for several inner steps
  --stage 5  + KL penalty to a frozen copy of the warm-start model (= full GRPO)

    python warmstart.py          # once: writes ckpt/warmstart.pkl
    python grpo.py --stage 1     # ... through --stage 5; logs to runs/
    python plot.py               # compares every run in runs/
"""

import argparse
import dataclasses
import json
import os
import pickle
import time
from dataclasses import dataclass
from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax

import task
from policy import completion_logprobs, completion_mask, generate
from warmstart import evaluate


@dataclass(frozen=True)
class GRPOConfig:
    steps: int = 250
    prompts_per_step: int = 32
    group_size: int = 8           # G: completions sampled per prompt
    lr: float = 1e-4
    temperature: float = 1.0
    # ---- stage switches ----
    baseline: bool = False        # stage 2: subtract the group mean
    std_norm: bool = False        # stage 3: divide by the group std
    clip_eps: Optional[float] = None  # stage 4: PPO clip range (None = no clipping)
    inner_steps: int = 1          # stage 4: gradient steps per sampled batch (μ in the paper)
    beta: float = 0.0             # stage 5: KL penalty weight


STAGES = {
    1: dict(),
    2: dict(baseline=True),
    3: dict(baseline=True, std_norm=True),
    4: dict(baseline=True, std_norm=True, clip_eps=0.2, inner_steps=4),
    5: dict(baseline=True, std_norm=True, clip_eps=0.2, inner_steps=4, beta=0.04),
}


# --------------------------------------------------------------------------
# Stages 1-3: advantages. This is all of GRPO's "critic".
# --------------------------------------------------------------------------
def compute_advantages(rewards, cfg: GRPOConfig):
    """rewards: [n_prompts * G], laid out group by group -> advantages, same shape.

    Every token of completion i gets the same advantage Â_i: GRPO does no
    per-token credit assignment at all.
    """
    r = rewards.reshape(-1, cfg.group_size)          # [n_prompts, G]
    adv = r                                          # stage 1: raw reward
    if cfg.baseline:                                 # stage 2: "better than my siblings?"
        adv = adv - r.mean(axis=1, keepdims=True)
    if cfg.std_norm:                                 # stage 3: same scale for easy and hard prompts
        adv = adv / (r.std(axis=1, keepdims=True) + 1e-4)
    # A group where every answer got the same reward has adv == 0 from stage 2
    # on: it contributes no gradient. Watch "dead_groups" in the logs.
    return adv.reshape(-1)


# --------------------------------------------------------------------------
# Stages 4-5: the per-token loss
# --------------------------------------------------------------------------
def grpo_loss(params, seq, mask, adv, old_lp, ref_lp, cfg: GRPOConfig, model_cfg):
    lp = completion_logprobs(params, seq, model_cfg, cfg.temperature)  # [B, MAX_NEW]
    lp = jnp.where(mask > 0, lp, 0.0)  # padded positions hold junk; zero them before exp()

    ratio = jnp.exp(lp - old_lp)       # ρ = π_θ / π_old, exactly 1 on the first inner step
    A = adv[:, None]
    if cfg.clip_eps is None:
        # Stages 1-3. At ρ = 1, ∇(ρ·A) = A·∇log π_θ: this IS plain policy gradient.
        pg = -ratio * A
    else:
        # Stage 4. Pessimistic bound: once ρ leaves [1-ε, 1+ε] in the direction
        # the advantage wants, the clipped branch is flat and the gradient stops.
        pg = -jnp.minimum(ratio * A, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * A)

    # Stage 5. k3 estimate of KL(π_θ || π_ref) at each sampled token. Unbiased
    # because tokens were sampled from (approximately) π_θ, and never negative.
    d = ref_lp - lp
    kl = jnp.exp(d) - d - 1
    per_token = pg + cfg.beta * kl

    # 1/|o_i| Σ_t ... then mean over all completions (= 1/G Σ_i, averaged over prompts)
    n_tok = mask.sum(axis=1)
    loss = ((per_token * mask).sum(axis=1) / n_tok).mean()

    clip_frac = 0.0
    if cfg.clip_eps is not None:
        clip_frac = ((jnp.abs(ratio - 1) > cfg.clip_eps) * mask).sum() / mask.sum()
    metrics = {"kl": (kl * mask).sum() / mask.sum(), "clip_frac": clip_frac}
    return loss, metrics


def make_update(opt, cfg: GRPOConfig, model_cfg):
    @jax.jit
    def update(params, opt_state, seq, mask, adv, old_lp, ref_lp):
        (loss, metrics), grads = jax.value_and_grad(grpo_loss, has_aux=True)(
            params, seq, mask, adv, old_lp, ref_lp, cfg, model_cfg)
        updates, opt_state = opt.update(grads, opt_state, params)
        metrics["loss"] = loss
        metrics["grad_norm"] = optax.global_norm(grads)
        return optax.apply_updates(params, updates), opt_state, metrics
    return update


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
def train(cfg: GRPOConfig, ckpt, out_path, seed=0, eval_every=10):
    with open(ckpt, "rb") as f:
        saved = pickle.load(f)
    model_cfg = saved["cfg"]
    params = jax.tree.map(jnp.asarray, saved["params"])
    ref_params = params          # stage 5's frozen reference. JAX arrays are immutable,
                                 # so "freezing" is just never updating this variable.
    opt = optax.adam(cfg.lr)
    opt_state = opt.init(params)
    update = make_update(opt, cfg, model_cfg)
    logprobs = partial(completion_logprobs, cfg=model_cfg, temperature=cfg.temperature)

    train_pairs, eval_pairs = task.split_pairs()
    eval_pairs = eval_pairs[:500]
    rng = np.random.default_rng(seed)
    key = jax.random.PRNGKey(seed)
    G = cfg.group_size
    log = open(out_path, "w")
    t0 = time.time()

    for step in range(0, cfg.steps + 1):
        # held-out eval of the policy BEFORE this step's update
        ev = evaluate(params, model_cfg, eval_pairs, jax.random.PRNGKey(10_000 + step),
                      cfg.temperature) if step % eval_every == 0 else None
        if step == cfg.steps:
            print(f"final held-out acc: sampled {ev['sampled']:.3f}  greedy {ev['greedy']:.3f}")
            log.write(json.dumps({"step": step, "eval_sampled": ev["sampled"],
                                  "eval_greedy": ev["greedy"]}) + "\n")
            break

        # 1. Sample a group of G completions for each prompt.
        prompts, answers = task.batch_prompts(rng, train_pairs, cfg.prompts_per_step)
        prompts = np.repeat(prompts, G, axis=0)      # rows [i*G, (i+1)*G) share prompt i
        answers = np.repeat(answers, G)
        key, sub = jax.random.split(key)
        seq = generate(params, jnp.asarray(prompts), sub, model_cfg, temperature=cfg.temperature)
        completions = seq[:, task.PROMPT_LEN:]
        mask = completion_mask(completions)

        # 2. Score them with the verifier, turn scores into advantages.
        rewards = jnp.asarray(task.reward(np.asarray(completions), answers))
        adv = compute_advantages(rewards, cfg)

        # 3. Freeze the log-probs of the policy that sampled (π_old) and of π_ref.
        old_lp = jnp.where(mask > 0, logprobs(params, seq), 0.0)
        ref_lp = jnp.where(mask > 0, logprobs(ref_params, seq), 0.0)

        # 4. A few gradient steps on this same batch (1 unless stage >= 4).
        for _ in range(cfg.inner_steps):
            params, opt_state, m = update(params, opt_state, seq, mask, adv, old_lp, ref_lp)

        r_groups = np.asarray(rewards).reshape(-1, G)
        row = {
            "step": step,
            "reward": float(rewards.mean()),
            "length": float(mask.sum(1).mean()),     # tokens incl. EOS
            "kl": float(m["kl"]),                     # logged even when beta = 0
            "clip_frac": float(m["clip_frac"]),
            "dead_groups": float((r_groups.std(1) == 0).mean()),
            "grad_norm": float(m["grad_norm"]),
        }
        if ev is not None:
            row.update(eval_sampled=ev["sampled"], eval_greedy=ev["greedy"])
            print(f"step {step:4d}  reward {row['reward']:.3f}  len {row['length']:.2f}  "
                  f"kl {row['kl']:.4f}  clip {row['clip_frac']:.3f}  "
                  f"dead {row['dead_groups']:.2f}  | held-out sampled {ev['sampled']:.3f} "
                  f"greedy {ev['greedy']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        log.write(json.dumps(row) + "\n")
    log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", type=int, default=5, choices=sorted(STAGES))
    ap.add_argument("--ckpt", default="ckpt/warmstart.pkl")
    ap.add_argument("--name", default=None, help="run name (default: stageN)")
    ap.add_argument("--seed", type=int, default=0)
    # any GRPOConfig field can be overridden, e.g. --beta 0.1 --group_size 4
    str2bool = lambda s: s.lower() in ("1", "true", "yes")
    for f in dataclasses.fields(GRPOConfig):
        t = {bool: str2bool, int: int}.get(f.type, float)  # Optional[float] -> float
        ap.add_argument(f"--{f.name}", type=t, default=None)
    args = ap.parse_args()

    overrides = {f.name: getattr(args, f.name) for f in dataclasses.fields(GRPOConfig)
                 if getattr(args, f.name) is not None}
    cfg = GRPOConfig(**{**STAGES[args.stage], **overrides})
    name = args.name or f"stage{args.stage}"
    os.makedirs("runs", exist_ok=True)
    print(f"{name}: {cfg}")
    train(cfg, args.ckpt, f"runs/{name}.jsonl", seed=args.seed)


if __name__ == "__main__":
    main()
