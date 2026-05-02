"""Tests for Scalar class and scalar functions."""

import pytest
import math
from hypothesis import given
from hypothesis.strategies import floats


small_floats = floats(min_value=-100, max_value=100, allow_nan=False)


@pytest.mark.task1_2
def test_scalar_creation():
    """Test basic Scalar creation."""
    from minitorch.scalar import Scalar

    s = Scalar(5.0)
    assert s.data == 5.0
    assert s.is_leaf()
    assert s.derivative is None


@pytest.mark.task1_3
def test_scalar_arithmetic():
    """Test Scalar arithmetic operations (requires ScalarFunctions)."""
    from minitorch.scalar import Scalar

    a = Scalar(2.0)
    b = Scalar(3.0)

    c = a + b
    assert c.data == 5.0

    d = a * b
    assert d.data == 6.0

    e = -a
    assert e.data == -2.0


@pytest.mark.task1_3
def test_scalar_requires_grad():
    """Test gradient tracking (requires ScalarFunctions)."""
    from minitorch.scalar import Scalar

    a = Scalar(2.0)
    a.requires_grad_(True)
    b = Scalar(3.0)
    b.requires_grad_(True)

    c = a * b
    assert not c.is_leaf()
    assert c.history is not None
    assert c.history.last_fn is not None


@pytest.mark.task1_3
def test_scalar_sigmoid():
    """Test sigmoid function."""
    from minitorch.scalar import Scalar

    a = Scalar(0.0)
    a.requires_grad_(True)
    b = a.sigmoid()

    assert abs(b.data - 0.5) < 0.01


@pytest.mark.task1_3
def test_scalar_relu():
    """Test ReLU function."""
    from minitorch.scalar import Scalar

    a = Scalar(-2.0)
    a.requires_grad_(True)
    b = a.relu()
    assert b.data == 0.0

    c = Scalar(3.0)
    c.requires_grad_(True)
    d = c.relu()
    assert d.data == 3.0


@pytest.mark.task1_3
@given(small_floats, small_floats)
def test_scalar_mul_backward(x, y):
    """Test multiplication backward pass."""
    from minitorch.scalar import Scalar
    from minitorch.autodiff import backpropagate

    if math.isfinite(x) and math.isfinite(y):
        a = Scalar(x)
        a.requires_grad_(True)
        b = Scalar(y)
        b.requires_grad_(True)

        c = a * b
        backpropagate(c, 1.0)

        # dc/da = b, dc/db = a
        assert a.derivative is not None
        assert b.derivative is not None
        assert abs(a.derivative - y) < 0.01
        assert abs(b.derivative - x) < 0.01
