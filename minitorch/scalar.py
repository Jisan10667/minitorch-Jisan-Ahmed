"""
Scalar class for autodifferentiation on single values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Type, Union

from .autodiff import Variable


@dataclass
class ScalarHistory:
    """
    Records the operation that created a Scalar.
    """

    last_fn: Optional[Type["ScalarFunction"]] = None
    ctx: Optional[Context] = None
    inputs: Sequence["Scalar"] = ()


class Scalar(Variable):
    """
    A scalar value that tracks computation history for autodiff.
    """

    def __init__(
        self,
        data: float,
        history: Optional[ScalarHistory] = None,
        name: Optional[str] = None,
    ):
        super().__init__()
        self.data = float(data)
        self.history = history
        self.name = name
        self.derivative = None

    def __repr__(self) -> str:
        return f"Scalar({self.data})"

    def __float__(self) -> float:
        return float(self.data)

    def __lt__(self, other: Union[Scalar, float]) -> bool:
        other_val = other.data if isinstance(other, Scalar) else other
        return self.data < other_val

    def __gt__(self, other: Union[Scalar, float]) -> bool:
        other_val = other.data if isinstance(other, Scalar) else other
        return self.data > other_val

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Scalar):
            return self.data == other.data
        return self.data == other

    def is_leaf(self) -> bool:
        """Leaf scalars have no operation history."""
        return self.history is None or self.history.last_fn is None

    def requires_grad_(self, requires_grad: bool = True) -> Scalar:
        """Enable or disable gradient tracking."""
        if requires_grad:
            if self.history is None:
                self.history = ScalarHistory()
        else:
            self.history = None
        return self

    def accumulate_derivative(self, deriv: float) -> None:
        """Add to the accumulated derivative."""
        if self.derivative is None:
            self.derivative = deriv
        else:
            self.derivative = self.derivative + deriv

    def zero_grad_(self) -> None:
        """Reset gradient to None."""
        self.derivative = None

    def backward(self, deriv: float = 1.0) -> None:
        """
        Compute gradients for all variables in this scalar's computation graph.
        """
        from .autodiff import backpropagate

        backpropagate(self, deriv)

    def __add__(self, other: Union[Scalar, float]) -> Scalar:
        return Add.apply(self, other)

    def __radd__(self, other: Union[Scalar, float]) -> Scalar:
        return Add.apply(other, self)

    def __mul__(self, other: Union[Scalar, float]) -> Scalar:
        return Mul.apply(self, other)

    def __rmul__(self, other: Union[Scalar, float]) -> Scalar:
        return Mul.apply(other, self)

    def __neg__(self) -> Scalar:
        return Neg.apply(self)

    def __sub__(self, other: Union[Scalar, float]) -> Scalar:
        return Add.apply(self, Neg.apply(other))

    def __rsub__(self, other: Union[Scalar, float]) -> Scalar:
        return Add.apply(other, Neg.apply(self))

    def __truediv__(self, other: Union[Scalar, float]) -> Scalar:
        return Mul.apply(self, Inv.apply(other))

    def __rtruediv__(self, other: Union[Scalar, float]) -> Scalar:
        return Mul.apply(other, Inv.apply(self))

    def log(self) -> Scalar:
        return Log.apply(self)

    def exp(self) -> Scalar:
        return Exp.apply(self)

    def sigmoid(self) -> Scalar:
        return Sigmoid.apply(self)

    def relu(self) -> Scalar:
        return ReLU.apply(self)

    def lt(self, other: Union[Scalar, float]) -> Scalar:
        return Lt.apply(self, other)

    def eq(self, other: Union[Scalar, float]) -> Scalar:
        return Eq.apply(self, other)


class Context:
    """
    Storage for values needed during the backward pass.
    """

    def __init__(self):
        self._saved_values: Tuple[float, ...] = ()

    def save_for_backward(self, *values: float) -> None:
        """Store values needed for computing gradients."""
        self._saved_values = values

    @property
    def saved_values(self) -> Tuple[float, ...]:
        return self._saved_values


from .scalar_functions import Add, Eq, Exp, Inv, Log, Lt, Mul, Neg, ReLU, Sigmoid
