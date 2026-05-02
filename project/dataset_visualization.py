"""
Generate dataset visualization screenshots for Module 0.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt

from minitorch.datasets import simple, split, xor


DATASETS = {
    "simple": simple,
    "split": split,
    "xor": xor,
}


def save_dataset_visualizations(
    output_dir: str = "results/module0_streamlit",
    num_points: int = 200,
    seed: int = 0,
) -> None:
    """Save one PNG per dataset: simple, split, and xor."""
    random.seed(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    for name, dataset_fn in DATASETS.items():
        points, labels = dataset_fn(num_points)
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        fig, ax = plt.subplots(figsize=(6, 6))
        scatter = ax.scatter(xs, ys, c=labels, cmap="coolwarm", edgecolor="black")
        ax.set_title(f"{name.title()} Dataset")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
        ax.legend(*scatter.legend_elements(), title="Class")
        fig.tight_layout()
        fig.savefig(output / f"{name}_dataset.png")
        plt.close(fig)


if __name__ == "__main__":
    save_dataset_visualizations()
