# GRPO from scratch in JAX

A small, readable GRPO you build up one piece at a time. A tiny transformer
learns 2-digit addition ("07+45=" → "52"), first by supervised warm start,
then by GRPO with a reward that is just a Python checker. It runs on a laptop
CPU: the warm start takes under a minute and each GRPO stage a few minutes.

## Files

| File | What's in it |
|---|---|
| `model.py` | Decoder-only transformer in plain JAX, params are a dict |
| `task.py` | Tokenizer, prompts, the verifiable reward, noisy SFT data |
| `policy.py` | Sampling, per-token log-probs, the completion mask |
| `warmstart.py` | Stage 0: short SFT so the policy is right some of the time |
| `grpo.py` | **The part to read.** Advantages, the loss, the loop, `--stage 1..5` |
| `checks.py` | Sanity checks for the easy-to-get-wrong parts |
| `plot.py` | Puts every run in `runs/` on one figure |

**Why plain JAX + Optax instead of Flax NNX?** GRPO needs three copies of the
same model: the policy you're training (π_θ), the policy that sampled the batch
(π_old) and a frozen reference (π_ref). With params as a dict, π_old is "the
log-probs I computed before the update" and π_ref is "a variable I never
update". Nothing is hidden in module state. Flax NNX is the better choice once
you load a real pretrained checkpoint (see the end).

## Run it

```bash
pip install "jax[cpu]" optax numpy matplotlib   # or jax[cuda12] on a GPU
python warmstart.py            # ~1 min, writes ckpt/warmstart.pkl
python checks.py               # 3 sanity checks
for s in 1 2 3 4 5; do python grpo.py --stage $s; done
python plot.py                 # writes runs/compare.png
```

Every `GRPOConfig` field is also a flag, so you can try one change at a time:
`python grpo.py --stage 5 --beta 0.2 --name beta0.2`.

## The objective

For each prompt q, sample G completions o_1..o_G from the current policy,
reward each one with r_i ∈ {0, 1}, then maximise

```
J(θ) = 1/G Σ_i 1/|o_i| Σ_t [ min(ρ_it Â_i, clip(ρ_it, 1-ε, 1+ε) Â_i) - β KL_it ]

ρ_it  = π_θ(o_it | q, o_i,<t) / π_old(o_it | q, o_i,<t)
Â_i   = (r_i - mean(r)) / std(r)          over the G completions of the same prompt
KL_it = π_ref/π_θ - log(π_ref/π_θ) - 1    k3 estimator, per sampled token
```

That's all of GRPO: PPO's clipped objective, with the value network replaced
by "how did this answer do compared with its siblings".

## The five stages and what to watch

| Stage | Adds | Code | Watch for |
|---|---|---|---|
| 1 | Sample G per prompt, Â = r (REINFORCE) | `compute_advantages` | Learns, but only correct answers get a push; wrong ones get no signal |
| 2 | Â = r − group mean | 1 line | Wrong answers now get pushed *down*. All-same groups give exactly zero gradient (`dead_groups`) |
| 3 | ÷ group std | 1 line | A 1-of-8 group gives the lone right answer Â ≈ +2.6; easy and hard prompts get equal weight |
| 4 | Clipped ratio, 4 inner steps per batch | `grpo_loss` | `clip_frac` > 0 only from the 2nd inner step; more learning per sample |
| 5 | KL to the frozen warm start | `grpo_loss` | `kl` stays well below stage 4 for about the same reward |

### What one run of all five stages looked like

CPU, seed 0, 250 steps × 32 prompts × G = 8, all starting from the same warm
start (held-out sampled accuracy 40%, greedy 96%). Figure: `runs/compare.png`.

| Stage | Held-out sampled acc | Greedy | KL (last 25 steps) | Dead groups |
|---|---|---|---|---|
| 1 REINFORCE | 76% | 99% | 0.13 | 17% |
| 2 + baseline | 71% | 95% | 0.11 | 13% |
| 3 + std | 71% | 97% | 0.11 | 11% |
| 4 + clip, 4 inner steps | 86% | 96% | 0.65 (one spike to ~7) | 52% |
| 5 + KL (β = 0.04) | 85% | 97% | 0.31 | 50% |

What's worth noticing:

- **Inner steps are the big win here**: 4 gradient steps per sampled batch
  learn about twice as fast per sample. Clipping is what makes reusing the
  batch safe (`clip_frac` ≈ 5%).
- **The KL penalty earns its keep at stage 4 → 5.** Without it, KL drifts up
  and one step spikes to ~7: k3 has a heavy tail when π_θ has nearly ruled out
  a token the warm start liked. With β = 0.04, same reward, half the KL, no spike.
- **Stages 1–3 are within noise of each other on one seed.** With Adam the
  overall advantage scale barely matters, and with 0/1 rewards REINFORCE
  without a baseline is still sensible. The baseline should matter more with
  smaller G, harder prompts or rewards that aren't 0/1. Run a few seeds before
  believing any gap (exercise 1).
- **Greedy dips before it recovers** in stages 4 and 5: RL is reshaping the
  whole distribution, not just raising the right answer's probability.
- **Length is flat** because the right answer's length is fixed by the
  question. It becomes interesting with partial-credit rewards or real text.

Things to keep an eye on in every run:

- **Don't watch the loss.** With a baseline the loss is ≈ 0 by construction.
  Watch reward, held-out accuracy and KL.
- **`dead_groups`** is the fraction of prompts where all G answers scored the
  same. Those prompts teach nothing. It goes up as the model gets good, which
  is why GRPO learning slows down near the top (and why DAPO filters them out).
- **`kl`** is logged in every stage, even with β = 0, so you can see how far
  each variant drifts from the warm start.
- **Sampled vs greedy accuracy.** The warm start is trained on labels that are
  wrong half the time, so its greedy answer is usually right but samples often
  aren't. Much of what GRPO does here is *sharpen* the distribution so
  sampling at temperature 1 lands on the right answer. That's also a large
  part of what RL does to pretrained LLMs.

## Details that are easy to get wrong (and where they live)

1. **Mask the prompt and the padding.** Only completion tokens up to and
   including the first EOS are actions (`completion_mask` in `policy.py`).
   The SFT loss is masked the same way.
2. **Same distribution for sampling and log-probs.** Temperature and the
   "never emit PAD" rule are applied in one function, `policy_logits`, used by
   both. Sample at 0.7 and compute log-probs at 1.0 and you're silently off-policy.
3. **Zero the junk before `exp`.** Log-probs at padded positions are huge
   negative numbers; their differences can overflow `exp` into `inf`, and
   `inf × 0 = NaN`. `grpo_loss` zeroes them first.
4. **π_old is not π_ref.** π_old is the policy that sampled *this* batch (it
   moves every step); π_ref is the warm start (it never moves).
5. **Clipping does nothing with one inner step.** On the first step ρ = 1
   exactly, so the clipped gradient equals plain policy gradient. `checks.py`
   asserts this.
6. **Prompts here are all the same length** (operands are zero-padded), so
   there's no left-padding. With real text prompts you left-pad, pass an
   attention mask, and compute position ids from the mask.

## Exercises

1. **Group size.** Run stage 3 with `--group_size 2`, `4`, `16`. How do
   `dead_groups` and the learning curve change?
2. **Length bias.** The 1/|o_i| term means a wrong answer's per-token penalty
   is smaller when it's longer. Change the normalisation to divide by the total
   number of tokens in the batch (Dr. GRPO / DAPO style) and compare lengths.
3. **Std bias.** Drop `std_norm` but keep everything else (Dr. GRPO argues the
   std division over-weights very easy and very hard prompts).
4. **k1 vs k3.** Replace the KL estimator with `lp - ref_lp` (k1). It can go
   negative per token; what does that do to training at β = 0.1?
5. **Too many inner steps.** Stage 4 with `--inner_steps 16`, with and without
   clipping (`--clip_eps 1e9`). Watch `clip_frac` and KL.
6. **Harder start.** `python warmstart.py --steps 150 --out ckpt/weak.pkl`, then
   `python grpo.py --ckpt ckpt/weak.pkl`. When the base model is rarely right,
   most groups are dead. How much does a bigger G help?
7. **Partial credit.** Give 0.5 for the right number of digits. Does it learn
   faster, or does it learn to game the partial reward?
8. **KV cache.** `generate` reruns the whole forward pass per token. Add a
   cache and measure the speed-up.

## Next step: a real pretrained model

Swap the task for GSM8K-style problems with an answer checker, and the model
for a small instruct model. The pieces that change are `model.forward`
(load a real architecture and checkpoint, likely in Flax NNX), the tokenizer,
variable-length prompts (left-padding and an attention mask) and a KV cache in
`generate`. The pieces that don't change are `compute_advantages` and
`grpo_loss`: those are the algorithm.

To check your numbers against a library, use Google's
[Tunix](https://tunix.readthedocs.io), a JAX-native post-training library with
a GRPO trainer (the JAX counterpart of TRL's `GRPOTrainer`). Check its docs
for which model families and sizes it ships.
