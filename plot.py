"""Plot every run in runs/ on the same axes: python plot.py [run names...]"""

import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def smooth(y, k=10):
    y = np.asarray(y, dtype=float)
    return np.convolve(y, np.ones(k) / k, mode="valid") if len(y) > k else y


def main():
    names = sys.argv[1:] or sorted(os.path.basename(p)[:-6] for p in glob.glob("runs/*.jsonl"))
    panels = [("reward", "train reward (sampled, smoothed)", True),
              ("eval_sampled", "held-out accuracy, sampled", False),
              ("eval_greedy", "held-out accuracy, greedy", False),
              ("length", "completion length (tokens, smoothed)", True),
              ("kl", "KL to warm start (k3, smoothed)", True),
              ("dead_groups", "groups with zero reward spread", True)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for name in names:
        rows = [json.loads(l) for l in open(f"runs/{name}.jsonl")]
        for ax, (key, title, sm) in zip(axes.flat, panels):
            pts = [(r["step"], r[key]) for r in rows if key in r]
            if not pts:
                continue
            x, y = map(np.array, zip(*pts))
            if sm:
                y = smooth(y)
                x = x[len(x) - len(y):]
            ax.plot(x, y, label=name)
            ax.set_title(title)
            ax.set_xlabel("step")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig("runs/compare.png", dpi=110)
    print("wrote runs/compare.png")


if __name__ == "__main__":
    main()
