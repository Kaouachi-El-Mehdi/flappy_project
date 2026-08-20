"""
Plots the evaluation score over training timesteps (from EvalCallback logs),
overlaid with the random-agent baseline score, for one or more training runs
(seeds). Multiple seeds on one plot make it easy to see whether repeated
training runs converge to similar curves.

Usage:
    python training/plot_progress.py --seeds 0 1 2 --baseline-score 1.3
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np


def plot(seeds, baseline_score, out_path):
    fig, ax = plt.subplots(figsize=(9, 5))

    for seed in seeds:
        data = np.load(f"logs/eval_seed_{seed}/evaluations.npz")
        timesteps = data["timesteps"]
        mean_scores = data["results"].mean(axis=1)
        ax.plot(timesteps, mean_scores, label=f"seed {seed}")

    ax.axhline(baseline_score, color="gray", linestyle="--", label="random agent baseline")
    ax.set_xlabel("training timesteps")
    ax.set_ylabel("mean evaluation score")
    ax.set_title("DQN training progress vs random baseline (FlappyBird-v0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--baseline-score", type=float, required=True)
    parser.add_argument("--out", type=str, default="logs/training_curve.png")
    args = parser.parse_args()
    plot(args.seeds, args.baseline_score, args.out)
