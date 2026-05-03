"""Optimization helpers for MiniTorch parameters."""

from typing import Iterable, List

from .module import Parameter
from .scalar import Scalar
from .tensor import Tensor


class SGD:
    """Stochastic gradient descent optimizer."""

    def __init__(self, parameters: Iterable[Parameter], lr: float):
        self.parameters: List[Parameter] = list(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        """Clear gradients for all parameters."""
        for parameter in self.parameters:
            value = parameter.value
            if hasattr(value, "zero_grad_"):
                value.zero_grad_()

    def step(self) -> None:
        """Apply one SGD update to all parameters."""
        for parameter in self.parameters:
            value = parameter.value

            if isinstance(value, Scalar):
                if value.derivative is not None:
                    value.data = value.data - self.lr * value.derivative
                continue

            if isinstance(value, Tensor):
                if value.grad is not None:
                    updated = (value - self.lr * value.grad).detach()
                    updated.requires_grad_(True)
                    parameter.value = updated
                continue

            raise TypeError(f"Unsupported parameter type: {type(value)!r}")
