"""
Plots training progression for the different DQN experiments.

V1:
    seeds 0, 1, 2, 3, 4

V2:
    seed 5 with reward shaping and improved hyperparameters.

Usage:
    python training/plot_progress.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np


def load_run(path):
    """Load Stable-Baselines3 EvalCallback results."""
    data = np.load(path)

    timesteps = data["timesteps"]
    mean_rewards = data["results"].mean(axis=1)
    std_rewards = data["results"].std(axis=1)

    return timesteps, mean_rewards, std_rewards


def plot():
    fig, ax = plt.subplots(figsize=(10, 6))

    # --------------------------------------------------
    # V1 experiments
    # --------------------------------------------------

    v1_runs = {
        0: "DQN V1 - seed 0 (200k)",
        1: "DQN V1 - seed 1 (200k)",
        2: "DQN V1 - seed 2 (200k)",
        3: "DQN V1 - seed 3 (1M)",
        4: "DQN V1 - seed 4 (2M)",
    }

    for seed, label in v1_runs.items():
        path = f"logs/eval_seed_{seed}/evaluations.npz"

        if not os.path.exists(path):
            print(f"Skipping missing file: {path}")
            continue

        timesteps, means, stds = load_run(path)

        ax.plot(
            timesteps,
            means,
            linewidth=1.5,
            alpha=0.65,
            label=label,
        )

    # --------------------------------------------------
    # V2 - improved agent
    # --------------------------------------------------

    v2_path = "logs/eval_v2_seed_5/evaluations.npz"

    if os.path.exists(v2_path):
        timesteps, means, stds = load_run(v2_path)

        ax.plot(
            timesteps,
            means,
            linewidth=3,
            label="DQN V2 - seed 5 (2M)",
        )

        ax.fill_between(
            timesteps,
            means - stds,
            means + stds,
            alpha=0.15,
        )

    else:
        print(f"Missing V2 log: {v2_path}")

    # --------------------------------------------------
    # Graph formatting
    # --------------------------------------------------

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean evaluation reward")

    ax.set_title(
        "DQN training progression - FlappyBird-v0"
    )

    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    fig.tight_layout()

    output = "logs/training_curve.png"
    fig.savefig(output, dpi=180)

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    plot()