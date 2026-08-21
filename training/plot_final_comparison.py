import matplotlib.pyplot as plt


labels = [
    "Random",
    "DQN V1\n200k",
    "DQN V1\n1M",
    "DQN V1\n2M",
    "DQN V2\n2M",
]

mean_scores = [
    0.00,
    0.47,
    4.03,
    3.27,
    33.43,
]

max_scores = [
    0,
    1,
    11,
    10,
    181,
]


def main():
    fig, ax = plt.subplots(figsize=(9, 5))

    bars = ax.bar(labels, mean_scores)

    ax.set_ylabel("Mean game score (pipes passed)")
    ax.set_title("Final agent performance comparison")

    ax.grid(axis="y", alpha=0.25)

    for bar, mean, max_score in zip(bars, mean_scores, max_scores):
        height = bar.get_height()

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.5,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height / 2 if height > 2 else height + 2,
            f"max: {max_score}",
            ha="center",
            va="center",
            fontsize=9,
        )

    ax.set_ylim(0, 40)

    fig.tight_layout()

    output = "logs/final_comparison.png"
    fig.savefig(output, dpi=180)

    print(f"Saved: {output}")


if __name__ == "__main__":
    main()