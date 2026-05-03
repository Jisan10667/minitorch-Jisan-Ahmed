"""
Automatic differentiation utilities for MiniTorch.
"""

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Any


def central_difference(f: Callable[..., float], *vals: float, arg: int = 0, epsilon: float = 1e-6) -> float:
    """
    Compute numerical derivative of f with respect to argument `arg`.

    Uses central difference: (f(x+h) - f(x-h)) / (2h)

    Args:
        f: Function to differentiate
        *vals: Input values to f
        arg: Which argument to differentiate with respect to (0-indexed)
        epsilon: Step size for numerical differentiation

    Returns:
        Approximate derivative

    Example:
        >>> def mul(x, y): return x * y
        >>> central_difference(mul, 3, 4, arg=0)  # df/dx at (3,4)
        4.0
        >>> central_difference(mul, 3, 4, arg=1)  # df/dy at (3,4)
        3.0
    """
    vals_list = list(vals)

    # Create vals with arg incremented by epsilon
    vals_plus = vals_list.copy()
    vals_plus[arg] = vals_plus[arg] + epsilon  

    # Create vals with arg decremented by epsilon
    vals_minus = vals_list.copy()
    vals_minus[arg] = vals_minus[arg] - epsilon  

    # Compute central difference
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    return (f_plus - f_minus) / (2 * epsilon)  


    
@dataclass
class Variable:
    """
    A node in the computation graph.

    Attributes:
        history: Record of the operation that created this variable
        derivative: Accumulated gradient (set during backward pass)
        name: Optional name for debugging
    """
    history: Optional["History"] = None
    derivative: Optional[float] = None
    name: Optional[str] = None

    def is_leaf(self) -> bool:
        """A leaf variable has no history (was not created by an operation)."""
        return self.history is None

    def is_constant(self) -> bool:
        """A constant has no history and will not receive gradients."""
        return self.history is None

    def requires_grad_(self, requires_grad: bool = True) -> "Variable":
        """Set whether this variable should track gradients."""
        if requires_grad:
            self.history = History()
        else:
            self.history = None
        return self


@dataclass
class History:
    """
    Records the operation that created a variable.

    Attributes:
        last_fn: The function class that created this variable
        ctx: Context object storing values needed for backward
        inputs: The input variables to the operation
    """
    last_fn: Optional[type] = None
    ctx: Optional["Context"] = None
    inputs: Sequence["Variable"] = ()

def backpropagate(final_var: Variable, deriv: float = 1.0) -> None:
    """
    Run reverse-mode autodiff starting from final_var.
    """

    stack = [(final_var, deriv)]

    while stack:
        var, d = stack.pop()

        # accumulate gradient
        if hasattr(var, "accumulate_derivative"):
            var.accumulate_derivative(d)

        # stop if leaf
        if var.history is None or var.history.last_fn is None:
            continue

        h = var.history

        grads = h.last_fn.backward(h.ctx, d)

        for inp, g in zip(h.inputs, grads):
            stack.append((inp, g))

from typing import List, Set


def topological_sort(variable: Variable) -> List[Variable]:
    """
    Return variables in topological order (children before parents).

    For backpropagation, a variable must be processed AFTER all
    variables that depend on it have been processed.

    Args:
        variable: The output variable (e.g., loss)

    Returns:
        List of variables in topological order
    """
    order: List[Variable] = []
    visited: Set[int] = set()

    def parents(var: Variable) -> Iterable[Variable]:
        if hasattr(var, "parents"):
            return var.parents
        if var.history is not None:
            return var.history.inputs
        return ()

    def visit(var: Variable) -> None:
        # Use id() to handle variables that might compare equal
        var_id = id(var)

        if var_id in visited:
            return
        visited.add(var_id)

        if var.is_constant():
            return

        for input_var in parents(var):
            visit(input_var)

        # Add this variable AFTER its children
        order.append(var)  # Q2: Append what?

    visit(variable)
    return order


def backpropagate(variable: Variable, deriv: float = 1.0) -> None:
    """
    Run backpropagation starting from variable.

    Computes gradients for all leaf variables in the computation graph.

    Args:
        variable: Output variable to differentiate (e.g., loss)
        deriv: Gradient of variable (default 1.0 for scalar loss)
    """
    derivatives = {id(variable): deriv}

    for var in reversed(topological_sort(variable)):
        var_deriv = derivatives.get(id(var))
        if var_deriv is None:
            continue

        if var.is_leaf():
            var.accumulate_derivative(var_deriv)
            continue

        if hasattr(var, "chain_rule"):
            input_grads = var.chain_rule(var_deriv)
        else:
            history = var.history
            assert history is not None
            assert history.last_fn is not None
            grads = history.last_fn.backward(history.ctx, var_deriv)
            input_grads = zip(history.inputs, grads)

        for input_var, grad in input_grads:
            if input_var.is_constant():
                continue
            key = id(input_var)
            if key in derivatives:
                derivatives[key] = derivatives[key] + grad
            else:
                derivatives[key] = grad

@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
