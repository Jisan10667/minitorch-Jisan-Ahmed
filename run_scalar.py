"""
Train the scalar MiniTorch example and save a loss curve.
"""

from __future__ import annotations

import random
from pathlib import Path

import matplotlib.pyplot as plt

from minitorch.datasets import simple
from minitorch.training_example import SimpleModel, binary_cross_entropy


def train_scalar(
    epochs: int = 100,
    learning_rate: float = 0.5,
    seed: int = 0,
    output_path: str = "results/module1_scalar/loss_curve.png",
) -> float:
    """Train a scalar classifier on the simple dataset and save loss history."""
    random.seed(seed)
    x_values, y_values = simple(100)
    model = SimpleModel()
    losses = []
    accuracies = []

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        for (x_coord, _), label in zip(x_values, y_values):
            model.zero_grad()
            pred = model.forward(x_coord)
            loss = binary_cross_entropy(pred, label)
            total_loss += loss.data
            loss.backward()

            for param in model.parameters():
                if param.derivative is not None:
                    param.data = param.data - learning_rate * param.derivative

            predicted_label = 1 if pred.data > 0.5 else 0
            if predicted_label == label:
                correct += 1

        losses.append(total_loss / len(x_values))
        accuracies.append(correct / len(x_values))

        if epoch % 10 == 0:
            print(
                f"Epoch {epoch}: loss={losses[-1]:.4f}, "
                f"accuracy={accuracies[-1]:.2%}"
            )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(epochs), losses, linewidth=2)
    ax.set_title("Scalar Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)

    final_accuracy = accuracies[-1]
    print(f"Final accuracy: {final_accuracy:.2%}")
    print(f"Saved loss curve: {output}")
    return final_accuracy


if __name__ == "__main__":
    train_scalar()
