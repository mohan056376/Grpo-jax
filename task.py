"""Toy verifiable task: 2-digit addition.

Prompt     "07+45="  (operands zero-padded, so every prompt is exactly 6 tokens)
Completion "52<eos>" (no zero padding, so answers are 1 to 3 digits + EOS)
Reward     1.0 if the completion is exactly the right number followed by EOS, else 0.0

The reward is a plain Python function of the text. That is the whole point of
"verifiable rewards": no reward model, just a checker.
"""

import numpy as np

CHARS = "0123456789+="
EOS = len(CHARS)       # 12
PAD = len(CHARS) + 1   # 13
VOCAB_SIZE = len(CHARS) + 2
PROMPT_LEN = 6
MAX_NEW = 5            # longest real answer is "198" + EOS = 4 tokens; 1 spare


def encode(s):
    return [CHARS.index(c) for c in s]


def decode(tokens):
    out = []
    for t in tokens:
        if t == EOS:
            out.append("<eos>")
            break
        out.append(CHARS[t] if t < len(CHARS) else "_")
    return "".join(out)


def make_prompt(a, b):
    return encode(f"{a:02d}+{b:02d}=")


def split_pairs(seed=0, n_eval=1000):
    """All (a, b) pairs with a, b in [0, 99], split into train and held-out eval."""
    pairs = np.array([(a, b) for a in range(100) for b in range(100)])
    np.random.default_rng(seed).shuffle(pairs)
    return pairs[n_eval:], pairs[:n_eval]


def batch_prompts(rng, pairs, n):
    idx = rng.integers(0, len(pairs), size=n)
    ab = pairs[idx]
    prompts = np.array([make_prompt(a, b) for a, b in ab], dtype=np.int32)
    return prompts, ab.sum(1)


def reward(completions, answers):
    """completions: int [N, MAX_NEW]; answers: int [N] -> float [N] in {0, 1}."""
    target = [encode(str(int(x))) + [EOS] for x in answers]
    out = np.zeros(len(completions), dtype=np.float32)
    for i, (c, t) in enumerate(zip(completions.tolist(), target)):
        out[i] = float(c[: len(t)] == t)
    return out


def noisy_answer(rng, true, p_noise):
    """Warm-start data: with probability p_noise, return a plausible WRONG answer.

    The SFT model learns this noisy distribution, so when you sample from it at
    temperature 1 it is right only some of the time. That gives GRPO something
    to fix, the same way a pretrained LLM "knows" an answer but doesn't reliably
    produce it.
    """
    if rng.random() >= p_noise:
        return true
    kind = rng.integers(0, 3)
    if kind == 0:   # off by one
        wrong = true + rng.choice([-1, 1])
    elif kind == 1:  # dropped carry / off by ten
        wrong = true + rng.choice([-10, 10])
    else:            # one digit changed
        s = list(str(true))
        j = rng.integers(0, len(s))
        s[j] = str((int(s[j]) + rng.integers(1, 10)) % 10)
        wrong = int("".join(s))
    return wrong if wrong >= 0 and wrong != true else true + 1


def sft_batch(rng, pairs, n, p_noise):
    """Teacher-forcing batch: tokens [n, PROMPT_LEN+MAX_NEW] and a loss mask that is
    1 only on completion tokens (we never train the model to predict the prompt)."""
    T = PROMPT_LEN + MAX_NEW
    toks = np.full((n, T), PAD, dtype=np.int32)
    mask = np.zeros((n, T), dtype=np.float32)
    idx = rng.integers(0, len(pairs), size=n)
    for i, (a, b) in enumerate(pairs[idx]):
        comp = encode(str(noisy_answer(rng, a + b, p_noise))) + [EOS]
        seq = make_prompt(a, b) + comp
        toks[i, : len(seq)] = seq
        mask[i, PROMPT_LEN : len(seq)] = 1.0
    return toks, mask
