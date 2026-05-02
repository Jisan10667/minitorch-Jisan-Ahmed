"""Tests for autodiff module."""

import pytest
import math
from hypothesis import given
from hypothesis.strategies import floats


small_floats = floats(min_value=-100, max_value=100, allow_nan=False)


@pytest.mark.task1_1
def test_central_difference_square():
    """Test derivative of x^2."""
    from minitorch.autodiff import central_difference

    def square(x):
        return x * x

    # Derivative of x^2 is 2x
    assert abs(central_difference(square, 3.0) - 6.0) < 0.01
    assert abs(central_difference(square, 0.0) - 0.0) < 0.01
    assert abs(central_difference(square, -2.0) - (-4.0)) < 0.01


@pytest.mark.task1_1
def test_central_difference_mul():
    """Test partial derivatives of multiplication."""
    from minitorch.autodiff import central_difference

    def mul(x, y):
        return x * y

    # df/dx = y, df/dy = x
    assert abs(central_difference(mul, 3.0, 4.0, arg=0) - 4.0) < 0.01
    assert abs(central_difference(mul, 3.0, 4.0, arg=1) - 3.0) < 0.01


@pytest.mark.task1_1
@given(small_floats)
def test_central_difference_exp(x):
    """Test derivative of e^x."""
    from minitorch.autodiff import central_difference

    if math.isfinite(x) and abs(x) < 50:  # Avoid overflow
        def exp_fn(a):
            return math.exp(a)

        approx = central_difference(exp_fn, x)
        expected = math.exp(x)
        assert abs(approx - expected) < 0.01 * max(1, abs(expected))


@pytest.mark.task1_4
def test_topological_sort_simple():
    """Test topological sorting of computation graph."""
    from minitorch.scalar import Scalar
    from minitorch.autodiff import topological_sort

    a = Scalar(1.0)
    a.requires_grad_(True)
    b = Scalar(2.0)
    b.requires_grad_(True)
    c = a + b

    sorted_vars = topological_sort(c)
    # c should come before a and b in the sorted order
    assert c in sorted_vars
    assert a in sorted_vars or b in sorted_vars


@pytest.mark.task1_4
def test_backpropagate_simple():
    """Test backpropagation on simple computation."""
    from minitorch.scalar import Scalar
    from minitorch.autodiff import backpropagate

    a = Scalar(2.0)
    a.requires_grad_(True)
    b = Scalar(3.0)
    b.requires_grad_(True)

    c = a * b  # c = 6
    backpropagate(c, 1.0)

    # dc/da = b = 3, dc/db = a = 2
    assert abs(a.derivative - 3.0) < 0.01
    assert abs(b.derivative - 2.0) < 0.01
