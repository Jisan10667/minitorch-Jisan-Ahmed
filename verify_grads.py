"""
Cross-check MiniTorch tensor gradients against PyTorch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

import minitorch

try:
    import torch
except ImportError as exc:  # pragma: no cover - dependency check path
    raise SystemExit(
        "PyTorch is required for verify_grads.py. Install requirements first."
    ) from exc


Array = list[list[float]]


def _mt_tensor(values: Array) -> minitorch.Tensor:
    return minitorch.tensor(values, requires_grad=True)


def _torch_tensor(values: Array) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def _mt_grad(tensor: minitorch.Tensor) -> np.ndarray:
    assert tensor.grad is not None
    return np.array(tensor.grad.to_numpy(), dtype=np.float64)


def _compare(
    name: str,
    mt_fn: Callable[[minitorch.Tensor, minitorch.Tensor], minitorch.Tensor],
    torch_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> str:
    a_values = [[0.2, -0.4], [0.7, 1.1]]
    b_values = [[1.3, -0.6], [0.5, 0.9]]

    mt_a = _mt_tensor(a_values)
    mt_b = _mt_tensor(b_values)
    torch_a = _torch_tensor(a_values)
    torch_b = _torch_tensor(b_values)

    mt_out = mt_fn(mt_a, mt_b).sum()
    torch_out = torch_fn(torch_a, torch_b).sum()

    mt_out.backward()
    torch_out.backward()

    mt_a_grad = _mt_grad(mt_a)
    mt_b_grad = _mt_grad(mt_b)
    torch_a_grad = torch_a.grad.detach().numpy()
    torch_b_grad = torch_b.grad.detach().numpy()

    np.testing.assert_allclose(mt_a_grad, torch_a_grad, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(mt_b_grad, torch_b_grad, rtol=1e-4, atol=1e-4)
    return (
        f"{name}: ok "
        f"max_abs_a={np.max(np.abs(mt_a_grad - torch_a_grad)):.2e} "
        f"max_abs_b={np.max(np.abs(mt_b_grad - torch_b_grad)):.2e}"
    )


def main(output_path: str = "results/module2_tensor/verify_grads_output.txt") -> None:
    checks = [
        (
            "add_mul",
            lambda a, b: a * b + a,
            lambda a, b: a * b + a,
        ),
        (
            "sigmoid_exp",
            lambda a, b: (a * 0.1).exp() + b.sigmoid(),
            lambda a, b: torch.exp(a * 0.1) + torch.sigmoid(b),
        ),
        (
            "log_product",
            lambda a, b: (a * a + 1e-6).log() * b,
            lambda a, b: torch.log((a * a + 1e-6) + 1e-6) * b,
        ),
    ]

    lines = [_compare(name, mt_fn, torch_fn) for name, mt_fn, torch_fn in checks]
    lines.append("all gradient checks passed within 1e-4")

    for line in lines:
        print(line)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
