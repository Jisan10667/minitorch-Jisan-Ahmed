"""
Implementation of the autodifferentiation Functions for Tensor.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

from .fast_conv import tensor_conv1d, tensor_conv2d

if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape

import numpy as np
from .autodiff import Context, History

import minitorch

from . import operators
from .tensor_ops import SimpleBackend, TensorBackend


def wrap_tuple(x):  # type: ignore
    "Turn a possible value into a tuple"
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Tensor) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Tensor) -> Tensor:
        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        # assert isinstance(c, Tensor), "Expected return type Tensor got %s" % (
        #     type(c)
        # )

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        return (grad_output.f.neg_map(grad_output),)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        (t1,) = ctx.saved_values
        return (grad_output.f.inv_back_zip(t1, grad_output),)


def grad_reduce_broadcast(grad: Tensor, original_shape: UserShape) -> Tensor:
    """Reduce a gradient tensor to match the original shape by summing over broadcasted dimensions."""
    if grad.shape == original_shape:
        return grad
    # Pad original_shape with leading 1s to match grad's number of dims
    ndims_added = len(grad.shape) - len(original_shape)
    padded_shape = (1,) * ndims_added + original_shape
    # Sum over any dimension where broadcasting expanded a size-1 dim
    for i in range(len(grad.shape)):
        if padded_shape[i] == 1 and grad.shape[i] > 1:
            grad = grad.sum(i)
    # Reshape to match original (removes extra leading dims of size 1)
    return grad.contiguous().view(*original_shape)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1.shape, t2.shape)
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1_shape, t2_shape = ctx.saved_values
        return (
            grad_reduce_broadcast(grad_output, t1_shape),
            grad_reduce_broadcast(grad_output, t2_shape),
        )

class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        return a.f.mul_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement for Task 2.4
        # d(a*b)/da = b, d(a*b)/db = a
        a, b = ctx.saved_values
        return (
            grad_output.f.mul_zip(grad_output, b),  # Q1: Gradient for a
            grad_output.f.mul_zip(a, grad_output)   # Q2: Gradient for b
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        out = t1.f.sigmoid_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4
        # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
        (out,) = ctx.saved_values  # We saved the output
        # Compute sigmoid * (1 - sigmoid) * grad_output
        one = out.zeros(out.shape)
        one._tensor._storage[:] = 1.0

        # sig_deriv = out * (1 - out)
        one_minus_out = one.f.add_zip(one, out.f.neg_map(out))  # Q3: 1 - out
        sig_deriv = out.f.mul_zip(out, one_minus_out)  # Q4: out * (1 - out)

        return grad_output.f.mul_zip(grad_output, sig_deriv)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.relu_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4
        # d(relu)/dx = 1 if x > 0 else 0
        (t1,) = ctx.saved_values
        return grad_output.f.relu_back_zip(t1, grad_output)  # Q5: Use relu_back


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.log_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4
        # d(log)/dx = 1/x
        (t1,) = ctx.saved_values
        return grad_output.f.log_back_zip(t1, grad_output)  # Q6: Use log_back


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        out = t1.f.exp_map(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        # TODO: Implement for Task 2.4
        # d(exp)/dx = exp(x) = out
        (out,) = ctx.saved_values
        return grad_output.f.mul_zip(grad_output, out)  # Q7: grad * exp(x)


class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return a.f.lt_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        # TODO: Implement for Task 2.4
        # Comparison has no gradient (return zeros)
        zeros_a = grad_output.zeros(grad_output.shape)
        zeros_b = grad_output.zeros(grad_output.shape)
        return zeros_a, zeros_b


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3
        return a.f.eq_zip(a, b)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        zeros_a = grad_output.zeros(grad_output.shape)
        zeros_b = grad_output.zeros(grad_output.shape)
        return zeros_a, zeros_b

class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3
        return a.f.is_close_zip(a, b)

class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tensor) -> Tensor:
        # TODO: Implement for Task 2.3
        ctx.save_for_backward(order)
        order_list = [int(order[i]) for i in range(order.size)]
        return a._new(a._tensor.permute(*order_list))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (order,) = ctx.saved_values
        order_list = [int(order[i]) for i in range(order.size)]
        inv_order = [0] * len(order_list)
        for i, o in enumerate(order_list):
            inv_order[o] = i
        return grad_output._new(grad_output._tensor.permute(*inv_order)), 0.0

class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape, dim)
        return a.f.add_reduce(a, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        a_shape, dim = ctx.saved_values
        return grad_output + grad_output.zeros(a_shape), 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Tensor) -> Tensor:
        if dim is not None:
            return a.f.mul_reduce(a, int(dim.item()))
        else:
            return a.f.mul_reduce(a.contiguous().view(int(operators.prod(a.shape))), 0)

class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Tensor) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        shape2 = [int(shape[i]) for i in range(shape.size)]
        return minitorch.Tensor.make(
            a._tensor._storage, tuple(shape2), backend=a.backend
        )

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, float]:
        (original,) = ctx.saved_values
        return (
            minitorch.Tensor.make(
                grad_output.contiguous()._tensor._storage, original, backend=grad_output.backend
            ),
            0.0,
        )


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor]:
        return (grad_output,)


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        t1, t2 = ctx.saved_values
        
        # Matrix Calc: dL/dA = grad_out @ B.T | dL/dB = A.T @ grad_out
        dims = list(range(len(t1.shape)))
        dims[-2], dims[-1] = dims[-1], dims[-2]
        t1_T = t1.permute(*dims)

        dims = list(range(len(t2.shape)))
        dims[-2], dims[-1] = dims[-1], dims[-2]
        t2_T = t2.permute(*dims)

        grad_t1 = grad_output.f.matrix_multiply(grad_output, t2_T)
        grad_t2 = grad_output.f.matrix_multiply(t1_T, grad_output)

        # Handle Batch Broadcasting
        if t1.shape[0] == 1 and grad_t1.shape[0] > 1:
            grad_t1 = grad_t1.sum(0)
        if t2.shape[0] == 1 and grad_t2.shape[0] > 1:
            grad_t2 = grad_t2.sum(0)

        return grad_t1, grad_t2


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """
    Produce a zero tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend

    Returns:
        new tensor
    """
    return minitorch.Tensor.make(
        [0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a random tensor of size `shape`.

    Args:
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """
    Produce a tensor with data ls and shape `shape`.

    Args:
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
        new tensor
    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """
    Produce a tensor with data and shape from ls

    Args:
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
        :class:`Tensor` : new tensor
    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )

    @classmethod
    def backward(cls, ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, height, width = input.shape
        out_channels, in_channels2, kh, kw = weight.shape

        # grad_input: convolve grad_output with weight (reversed)
        grad_input = input.zeros((batch, in_channels, height, width))
        # Transpose weight: (out_channels, in_channels, kh, kw) -> (in_channels, out_channels, kh, kw)
        weight_t = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            grad_input._tensor._storage, grad_input.shape, grad_input._tensor.strides, grad_input.size,
            grad_output._tensor._storage, grad_output.shape, grad_output._tensor.strides,
            weight_t.contiguous()._tensor._storage, weight_t.shape, weight_t.contiguous()._tensor.strides,
            True
        )

        # grad_weight: convolve input with grad_output
        # Transpose input: (batch, in_channels, h, w) -> (in_channels, batch, h, w)
        input_t = input.permute(1, 0, 2, 3).contiguous()
        # Transpose grad_output: (batch, out_channels, h, w) -> (out_channels, batch, h, w)
        grad_output_t = grad_output.permute(1, 0, 2, 3).contiguous()
        # Output shape (in_channels, out_channels, kh, kw) so batch loop aligns with input_t dim 0
        grad_weight_t = weight.zeros((in_channels, out_channels, kh, kw))
        tensor_conv2d(
            grad_weight_t._tensor._storage, grad_weight_t.shape, grad_weight_t._tensor.strides, grad_weight_t.size,
            input_t._tensor._storage, input_t.shape, input_t._tensor.strides,
            grad_output_t._tensor._storage, grad_output_t.shape, grad_output_t._tensor.strides,
            False
        )
        # Permute back to (out_channels, in_channels, kh, kw)
        grad_weight = grad_weight_t.permute(1, 0, 2, 3).contiguous()

        return grad_input, grad_weight
"""
from __future__ import annotations

from typing import Any, List, Sequence, Tuple

import numpy as np

from . import operators
from .autodiff import Context
from .tensor_data import TensorData, UserShape
from .tensor_ops import SimpleBackend


class Function:
    @classmethod
    def apply(cls, *vals):
        from .tensor import History, Tensor

        raw_vals = []
        need_grad = False
        for v in vals:
            if v.requires_grad():
                need_grad = True
            raw_vals.append(v.detach())

        ctx = Context(not need_grad)
        c = cls.forward(ctx, *raw_vals)
        history = History(cls, ctx, vals) if need_grad else None
        return Tensor(c._tensor, history, backend=c.backend)

    @classmethod
    def _backward(cls, ctx: Context, grad_out):
        return cls.backward(ctx, grad_out)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1):
        return t1.backend.map(operators.neg)(t1)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return (grad_output.backend.map(operators.neg)(grad_output),)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1, t2):
        return t1.backend.zip(operators.add)(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output, grad_output


class Mul(Function):
    @staticmethod
    def forward(ctx: Context, t1, t2):
        ctx.save_for_backward(t1, t2)
        return t1.backend.zip(operators.mul)(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output):
        t1, t2 = ctx.saved_values
        return grad_output * t2, grad_output * t1


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1):
        ctx.save_for_backward(t1)
        return t1.backend.map(operators.inv)(t1)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (t1,) = ctx.saved_values
        return grad_output.backend.zip(operators.mul)(
            grad_output, grad_output.backend.map(lambda x: operators.inv_back(x, 1.0))(t1)
        )


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, t1):
        out = t1.backend.map(operators.sigmoid)(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (out,) = ctx.saved_values
        return out * (1.0 - out) * grad_output


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, t1):
        ctx.save_for_backward(t1)
        return t1.backend.map(operators.relu)(t1)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (t1,) = ctx.saved_values
        return grad_output.backend.zip(operators.relu_back)(t1, grad_output)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, t1):
        ctx.save_for_backward(t1)
        return t1.backend.map(operators.log)(t1)

    @staticmethod
    def backward(ctx: Context, grad_output):
        (t1,) = ctx.saved_values
        return grad_output.backend.zip(operators.log_back)(t1, grad_output)


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, t1):
        out = t1.backend.map(operators.exp)(t1)
        ctx.save_for_backward(out)
        return out

    @staticmethod
    def backward(ctx: Context, grad_output):
        (out,) = ctx.saved_values
        return out * grad_output


class LT(Function):
    @staticmethod
    def forward(ctx: Context, t1, t2):
        return t1.backend.zip(operators.lt)(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output.zeros(), grad_output.zeros()


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, t1, t2):
        return t1.backend.zip(operators.eq)(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output.zeros(), grad_output.zeros()


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, t1, t2):
        return t1.backend.zip(operators.is_close)(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output.zeros(), grad_output.zeros()


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, t1, dim):
        return t1.backend.reduce(operators.add, 0.0)(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output, 0.0


class All(Function):
    @staticmethod
    def forward(ctx: Context, t1, dim):
        return t1.backend.reduce(operators.mul, 1.0)(t1, int(dim.item()))

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output.zeros(), 0.0


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, t1, order):
        order_tuple = tuple(int(order[i]) for i in range(order.size))
        return t1._new(t1._tensor.permute(*order_tuple))

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output, 0.0


class View(Function):
    @staticmethod
    def forward(ctx: Context, t1, shape):
        shape_tuple = tuple(int(shape[i]) for i in range(shape.size))
        assert t1._tensor.is_contiguous(), "Must be contiguous to view"
        return t1._new(TensorData(t1._tensor._storage, shape_tuple))

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output, 0.0


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, t1):
        return t1.backend.map(operators.id)(t1)

    @staticmethod
    def backward(ctx: Context, grad_output):
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1, t2):
        raise NotImplementedError("Matrix multiplication is not implemented until Module 3.")

    @staticmethod
    def backward(ctx: Context, grad_output):
        raise NotImplementedError("Matrix multiplication is not implemented until Module 3.")


def _shape(ls: Any) -> UserShape:
    if isinstance(ls, (list, tuple)):
        if len(ls) == 0:
            return (0,)
        return (len(ls),) + tuple(_shape(ls[0]))
    return ()


def _flatten(ls: Any) -> List[float]:
    if isinstance(ls, (list, tuple)):
        out: List[float] = []
        for x in ls:
            out.extend(_flatten(x))
        return out
    return [float(ls)]


def tensor(ls: Any, backend=SimpleBackend, requires_grad: bool = False):
    from .tensor import Tensor

    if isinstance(ls, np.ndarray):
        ls = ls.tolist()
    shape = _shape(ls)
    storage = _flatten(ls)
    out = Tensor.make(storage, shape, backend=backend)
    if requires_grad:
        out.requires_grad_(True)
    return out
"""

class Conv1dFun(Function):
    @classmethod
    def forward(cls, ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        1D Convolution Forward
        Args:
            input: tensor of shape (batch, in_channels, width)
            weight: tensor of shape (out_channels, in_channels, kernel_width)
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, width = input.shape
        out_channels, in_channels2, kw = weight.shape

        # Output width matches input width (kernel slides with implicit zero-padding)
        out_width = width

        # Allocate output tensor
        output = input.zeros((batch, out_channels, out_width))

        tensor_conv1d(
            output._tensor._storage, output.shape, output._tensor.strides, output.size,
            input._tensor._storage, input.shape, input._tensor.strides,
            weight._tensor._storage, weight.shape, weight._tensor.strides,
            False # reverse
        )

        return output

    @classmethod
    def backward(cls, ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, width = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        grad_input = input.zeros((batch, in_channels, width))
        weight_t = weight.permute(1, 0, 2).contiguous()
        tensor_conv1d(
            grad_input._tensor._storage, grad_input.shape, grad_input._tensor.strides, grad_input.size,
            grad_output._tensor._storage, grad_output.shape, grad_output._tensor.strides,
            weight_t._tensor._storage, weight_t.shape, weight_t._tensor.strides,
            True
        )

        input_t = input.permute(1, 0, 2).contiguous()
        grad_output_t = grad_output.permute(1, 0, 2).contiguous()
        grad_weight_t = weight.zeros((in_channels, out_channels, kw))
        tensor_conv1d(
            grad_weight_t._tensor._storage, grad_weight_t.shape, grad_weight_t._tensor.strides, grad_weight_t.size,
            input_t._tensor._storage, input_t.shape, input_t._tensor.strides,
            grad_output_t._tensor._storage, grad_output_t.shape, grad_output_t._tensor.strides,
            False
        )
        grad_weight = grad_weight_t.permute(1, 0, 2).contiguous()

        return grad_input, grad_weight


class Conv2dFun(Function):
    @classmethod
    def forward(cls, ctx: Context, input: Tensor, weight: Tensor) -> Tensor:
        """
        2D Convolution Forward
        Args:
            input: tensor of shape (batch, in_channels, height, width)
            weight: tensor of shape (out_channels, in_channels, kernel_height, kernel_width)
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, height, width = input.shape
        out_channels, in_channels2, kh, kw = weight.shape

        out_height = height
        out_width = width

        output = input.zeros((batch, out_channels, out_height, out_width))

        tensor_conv2d(
            output._tensor._storage, output.shape, output._tensor.strides, output.size,
            input._tensor._storage, input.shape, input._tensor.strides,
            weight._tensor._storage, weight.shape, weight._tensor.strides,
            False
        )

        return output

    @classmethod
    def backward(cls, ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        input, weight = ctx.saved_values
        batch, in_channels, height, width = input.shape
        out_channels, in_channels2, kh, kw = weight.shape

        # grad_input: convolve grad_output with weight (reversed)
        grad_input = input.zeros((batch, in_channels, height, width))
        # Transpose weight: (out_channels, in_channels, kh, kw) -> (in_channels, out_channels, kh, kw)
        weight_t = weight.permute(1, 0, 2, 3)
        tensor_conv2d(
            grad_input._tensor._storage, grad_input.shape, grad_input._tensor.strides, grad_input.size,
            grad_output._tensor._storage, grad_output.shape, grad_output._tensor.strides,
            weight_t.contiguous()._tensor._storage, weight_t.shape, weight_t.contiguous()._tensor.strides,
            True
        )

        # grad_weight: convolve input with grad_output
        # Transpose input: (batch, in_channels, h, w) -> (in_channels, batch, h, w)
        input_t = input.permute(1, 0, 2, 3).contiguous()
        # Transpose grad_output: (batch, out_channels, h, w) -> (out_channels, batch, h, w)
        grad_output_t = grad_output.permute(1, 0, 2, 3).contiguous()
        # Output shape (in_channels, out_channels, kh, kw) so batch loop aligns with input_t dim 0
        grad_weight_t = weight.zeros((in_channels, out_channels, kh, kw))
        tensor_conv2d(
            grad_weight_t._tensor._storage, grad_weight_t.shape, grad_weight_t._tensor.strides, grad_weight_t.size,
            input_t._tensor._storage, input_t.shape, input_t._tensor.strides,
            grad_output_t._tensor._storage, grad_output_t.shape, grad_output_t._tensor.strides,
            False
        )
        # Permute back to (out_channels, in_channels, kh, kw)
        grad_weight = grad_weight_t.permute(1, 0, 2, 3).contiguous()

        return grad_input, grad_weight
