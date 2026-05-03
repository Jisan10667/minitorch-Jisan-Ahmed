"""Testing utilities for MiniTorch."""

import math

from . import operators


def assert_close(a: float, b: float, eps: float = 1e-2):
    """Assert two floats are close within tolerance."""
    assert abs(a - b) < eps, f"Values not close: {a} vs {b}"


class MathTestVariable:
    """Small catalog of scalar/tensor operations used by later task tests."""

    @staticmethod
    def _comp_testing():
        one_arg = [
            ("neg", lambda x: -x, lambda x: -x),
            ("sigmoid", lambda x: 1.0 / (1.0 + math.exp(-x)), lambda x: x.sigmoid()),
            ("relu", lambda x: x * x + 1e-6, lambda x: (x * x + 1e-6).relu()),
            ("log", lambda x: operators.log(x * x + 1e-6), lambda x: (x * x + 1e-6).log()),
            ("exp", lambda x: math.exp(0.1 * x), lambda x: (x * 0.1).exp()),
        ]
        two_arg = [
            ("add", lambda x, y: x + y, lambda x, y: x + y),
            ("mul", lambda x, y: x * y, lambda x, y: x * y),
        ]
        red_arg = [
            ("sum", lambda x: x, lambda x: x.sum()),
        ]
        return one_arg, two_arg, red_arg


def grad_check(*args, **kwargs):
    from .tensor_functions import grad_check as tensor_grad_check

    return tensor_grad_check(*args, **kwargs)
