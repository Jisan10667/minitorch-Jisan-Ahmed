"""
Train a tiny tensor logistic classifier on the simple dataset.
"""

from __future__ import annotations

import random
from pathlib import Path

import minitorch
from minitorch.datasets import simple


def _param(value: float) -> minitorch.Tensor:
    out = minitorch.tensor([value], requires_grad=True)
    return out


def _zero_grad(params: list[minitorch.Tensor]) -> None:
    for param in params:
        param.zero_grad_()


def _step(params: list[minitorch.Tensor], learning_rate: float) -> None:
    for param in params:
        if param.grad is not None:
            param._tensor._storage[0] -= learning_rate * param.grad[0]


def _bce(pred: minitorch.Tensor, target: float) -> minitorch.Tensor:
    y = minitorch.tensor([target], backend=pred.backend)
    one = minitorch.tensor([1.0], backend=pred.backend)
    return -(y * pred.log() + (one - y) * (one - pred).log())


def train_tensor(
    epochs: int = 120,
    learning_rate: float = 0.5,
    n_samples: int = 100,
    seed: int = 0,
    output_path: str = "results/module2_tensor/run_tensor_output.txt",
) -> float:
    """Train on the simple dataset and write a small run log."""
    random.seed(seed)
    points, labels = simple(n_samples)

    w_x = _param(random.uniform(-1.0, 1.0))
    w_y = _param(random.uniform(-1.0, 1.0))
    bias = _param(random.uniform(-1.0, 1.0))
    params = [w_x, w_y, bias]
    log_lines = []

    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0

        for (x_coord, y_coord), label in zip(points, labels):
            _zero_grad(params)
            x = minitorch.tensor([x_coord])
            y = minitorch.tensor([y_coord])
            pred = (w_x * x + w_y * y + bias).sigmoid()
            loss = _bce(pred, float(label))
            total_loss += loss[0]
            loss.backward()
            _step(params, learning_rate)

            predicted = 1 if pred[0] >= 0.5 else 0
            correct += int(predicted == label)

        accuracy = correct / len(points)
        avg_loss = total_loss / len(points)
        if epoch % 10 == 0 or epoch == epochs - 1:
            line = f"epoch={epoch:03d} loss={avg_loss:.6f} accuracy={accuracy:.2%}"
            print(line)
            log_lines.append(line)

    final_accuracy = accuracy
    summary = (
        f"final_accuracy={final_accuracy:.2%}\n"
        f"w_x={w_x[0]:.6f} w_y={w_y[0]:.6f} bias={bias[0]:.6f}\n"
    )
    print(summary, end="")
    log_lines.append(summary.rstrip())

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(log_lines) + "\n")

    assert final_accuracy > 0.85, f"Expected >85% accuracy, got {final_accuracy:.2%}"
    return final_accuracy


if __name__ == "__main__":
    train_tensor()
