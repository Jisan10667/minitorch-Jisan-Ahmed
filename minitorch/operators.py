"""
Mathematical operators for MiniTorch.

These form the foundation of all neural network operations.
You'll implement each function to understand how deep learning
frameworks handle basic mathematics.
"""

import math
from typing import Callable, Iterable


# TODO: Implement these functions in Task 0.1
def mul(x: float, y: float) -> float:
    """Multiply two numbers."""
    raise NotImplementedError("Implement in Task 0.1")


def id(x: float) -> float:
    """Identity function."""
    raise NotImplementedError("Implement in Task 0.1")


def add(x: float, y: float) -> float:
    """Add two numbers."""
    raise NotImplementedError("Implement in Task 0.1")


def neg(x: float) -> float:
    """Negate a number."""
    raise NotImplementedError("Implement in Task 0.1")


def lt(x: float, y: float) -> float:
    """Less than comparison."""
    raise NotImplementedError("Implement in Task 0.1")


def eq(x: float, y: float) -> float:
    """Equality comparison."""
    raise NotImplementedError("Implement in Task 0.1")


def max(x: float, y: float) -> float:
    """Maximum of two numbers."""
    raise NotImplementedError("Implement in Task 0.1")


def is_close(x: float, y: float) -> float:
    """Check if numbers are close."""
    raise NotImplementedError("Implement in Task 0.1")


def sigmoid(x: float) -> float:
    """Sigmoid activation function."""
    raise NotImplementedError("Implement in Task 0.1")


def relu(x: float) -> float:
    """ReLU activation function."""
    raise NotImplementedError("Implement in Task 0.1")


def log(x: float) -> float:
    """Natural logarithm."""
    raise NotImplementedError("Implement in Task 0.1")


def exp(x: float) -> float:
    """Exponential function."""
    raise NotImplementedError("Implement in Task 0.1")


def inv(x: float) -> float:
    """Reciprocal function."""
    raise NotImplementedError("Implement in Task 0.1")


def log_back(x: float, grad: float) -> float:
    """Gradient of log."""
    raise NotImplementedError("Implement in Task 0.1")


def inv_back(x: float, grad: float) -> float:
    """Gradient of inv."""
    raise NotImplementedError("Implement in Task 0.1")


def relu_back(x: float, grad: float) -> float:
    """Gradient of ReLU."""
    raise NotImplementedError("Implement in Task 0.1")


# TODO: Implement these in Task 0.3
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Higher-order map function."""
    raise NotImplementedError("Implement in Task 0.3")


def zipWith(fn: Callable[[float, float], float]) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order zipWith function."""
    raise NotImplementedError("Implement in Task 0.3")


def reduce(fn: Callable[[float, float], float], init: float) -> Callable[[Iterable[float]], float]:
    """Higher-order reduce function."""
    raise NotImplementedError("Implement in Task 0.3")


def sum(ls: Iterable[float]) -> float:
    """Sum using reduce."""
    raise NotImplementedError("Implement in Task 0.3")


def prod(ls: Iterable[float]) -> float:
    """Product using reduce."""
    raise NotImplementedError("Implement in Task 0.3")


def negList(ls: Iterable[float]) -> Iterable[float]:
    """Negate list using map."""
    raise NotImplementedError("Implement in Task 0.3")


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Add lists using zipWith."""
    raise NotImplementedError("Implement in Task 0.3")
