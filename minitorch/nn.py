from typing import Tuple
from . import operators
from .tensor import Tensor
from .tensor_functions import Function, rand, zeros


def tile(input: Tensor, kernel: Tuple[int, int]) -> Tuple[Tensor, int, int]:
    """
    Reshape an image tensor for 2D pooling.

    Args:
        input: (batch, channel, height, width)
        kernel: (kernel_height, kernel_width)

    Returns:
        Reshaped tensor: (batch, channel, new_h, new_w, kernel_h * kernel_w)
        new_height
        new_width
    """
    batch, channel, height, width = input.shape
    kh, kw = kernel

    assert height % kh == 0, f"Height {height} must be divisible by kernel {kh}"
    assert width % kw == 0, f"Width {width} must be divisible by kernel {kw}"

    # TODO: Implement for Task 4.3
    # Reshape to group pixels into pooling windows

    new_h = height // kh
    new_w = width // kw

    # Reshape: (batch, channel, height, width)
    #       -> (batch, channel, new_h, kh, new_w, kw)
    #       -> (batch, channel, new_h, new_w, kh * kw)

    # Step 1: view to split height and width dimensions
    x = input.contiguous().view(batch, channel, new_h, kh, new_w, kw)  # Q1, Q2

    # Step 2: permute to group spatial dimensions
    x = x.permute(0, 1, 2, 4, 3, 5)  # Q3, Q4: Move kh, kw to end

    # Step 3: combine kh * kw into single dimension
    x = x.contiguous().view(batch, channel, new_h, new_w, kh * kw)  # Q5

    return x, new_h, new_w


def avgpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """
    Tiled average pooling 2D.

    Args:
        input: (batch, channel, height, width)
        kernel: (kernel_height, kernel_width)

    Returns:
        Pooled tensor: (batch, channel, new_height, new_width)
    """
    # TODO: Implement for Task 4.3

    batch, channel = input.shape[:2]
    tiled, new_h, new_w = tile(input, kernel)

    # Average over the last dimension (kh * kw pixels)
    # Use contiguous().view() to ensure proper output shape
    return tiled.mean(dim=4).contiguous().view(batch, channel, new_h, new_w)  # Q6

from .fast_ops import FastOps

max_reduce = FastOps.reduce(operators.max, -1e9)


def argmax(input: Tensor, dim: int) -> Tensor:
    """Return 1-hot tensor indicating maximum positions."""
    out = max_reduce(input, dim)
    return out == input


class Max(Function):
    @staticmethod
    def forward(ctx, input: Tensor, dim: Tensor) -> Tensor:
        """Max reduction along dimension."""
        # TODO: Implement for Task 4.4
        ctx.save_for_backward(input, int(dim.item()))
        return max_reduce(input, int(dim.item()))  # Q1

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Tensor, float]:
        """Gradient flows only to max positions."""
        # TODO: Implement for Task 4.4
        input, dim = ctx.saved_values
        return grad_output * argmax(input, dim), 0.0  # Q2, Q3


def max(input: Tensor, dim: int) -> Tensor:
    return Max.apply(input, input._ensure_tensor(dim))


def softmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute softmax: exp(x) / sum(exp(x))

    Uses the log-sum-exp trick for numerical stability.
    """
    # TODO: Implement for Task 4.4

    # Step 1: Subtract max for numerical stability
    x_max = max(input, dim)
    x_stable = input - x_max

    # Step 2: Compute exp
    exp_x = x_stable.exp()

    # Step 3: Normalize
    sum_exp = exp_x.sum(dim=dim)  # Q4
    return exp_x / sum_exp


def logsoftmax(input: Tensor, dim: int) -> Tensor:
    """
    Compute log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))

    More numerically stable than log(softmax(x)).
    """
    # TODO: Implement for Task 4.4

    x_max = max(input, dim)
    x_stable = input - x_max

    log_sum_exp = x_stable.exp().sum(dim=dim).log()

    return x_stable - log_sum_exp  # Q5


def maxpool2d(input: Tensor, kernel: Tuple[int, int]) -> Tensor:
    """Max pooling using tile and reduce."""
    # TODO: Implement for Task 4.4

    batch, channel = input.shape[:2]
    tiled, new_h, new_w = tile(input, kernel)
    return max(tiled, dim=4).contiguous().view(batch, channel, new_h, new_w)  # Q6, Q7


def dropout(input: Tensor, rate: float, ignore: bool = False) -> Tensor:
    """
    Dropout: randomly zero positions with probability rate.

    Args:
        input: input tensor
        rate: probability of dropping (0 to 1)
        ignore: if True, skip dropout (inference mode)

    Returns:
        Tensor with random positions zeroed
    """
    # TODO: Implement for Task 4.4

    if ignore or rate == 0.0:
        return input

    if rate >= 1.0:
        return input * 0.0

    # Generate random mask
    mask = rand(input.shape, backend=input.backend) > rate  # Q8

    # Apply mask (and scale by 1/(1-rate) for inverted dropout)
    return input * mask / (1.0 - rate)

import math
from .module import Module, Parameter
from .tensor_functions import Conv1dFun, Conv2dFun
from .tensor_ops import TensorBackend
from .fast_ops import FastOps

FastBackend = TensorBackend(FastOps)


class Linear(Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        k = 1.0 / math.sqrt(in_features)
        w = (rand((in_features, out_features), backend=FastBackend) * (2 * k) - k).detach()
        b = (rand((out_features,), backend=FastBackend) * (2 * k) - k).detach()
        w.requires_grad_(True)
        b.requires_grad_(True)
        self.weight = Parameter(w)
        self.bias = Parameter(b)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.value + self.bias.value


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: Tuple[int, int]):
        super().__init__()
        kh, kw = kernel
        k = 1.0 / math.sqrt(in_channels * kh * kw)
        w = (rand((out_channels, in_channels, kh, kw), backend=FastBackend) * (2 * k) - k).detach()
        w.requires_grad_(True)
        self.weight = Parameter(w)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return Conv2dFun.apply(x, self.weight.value)


class Conv1d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_width: int):
        super().__init__()
        k = 1.0 / math.sqrt(in_channels * kernel_width)
        w = (rand((out_channels, in_channels, kernel_width), backend=FastBackend) * (2 * k) - k).detach()
        w.requires_grad_(True)
        self.weight = Parameter(w)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return Conv1dFun.apply(x, self.weight.value)

def one_hot(labels, num_classes):
    """Convert integer labels to one-hot encoded tensor."""
    n = labels.size
    out = zeros((n, num_classes), backend=labels.backend)
    for i in range(n):
        label_idx = int(float(labels._tensor._storage[i]))
        out._tensor._storage[i * num_classes + label_idx] = 1.0
    return out


class no_grad:
    """Context manager to disable gradient tracking."""
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass
