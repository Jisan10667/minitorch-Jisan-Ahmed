"""Testing utilities for MiniTorch."""

def assert_close(a: float, b: float, eps: float = 1e-2):
    """Assert two floats are close within tolerance."""
    assert abs(a - b) < eps, f"Values not close: {a} vs {b}"
