from typing import Tuple
import numpy as np
from numba import njit, prange

from .tensor_functions import Function
from .tensor_data import (
    MAX_DIMS,
    Index,
    Shape,
    Strides,
    broadcast_index,
    index_to_position,
    to_index,
)

# JIT compile helper functions
# .py_func extracts the original Python function if already JIT-compiled
to_index = njit(inline="always")(to_index.py_func)
index_to_position = njit(inline="always")(index_to_position.py_func)
broadcast_index = njit(inline="always")(broadcast_index.py_func)


def _tensor_conv1d(
    out, out_shape, out_strides, out_size,
    input, input_shape, input_strides,
    weight, weight_shape, weight_strides,
    reverse
):
    """
    1D Convolution implementation.

    Input:  (batch, in_channels, width)
    Weight: (out_channels, in_channels, kernel_width)
    Output: (batch, out_channels, width)

    Args:
        reverse: If True, anchor kernel at right (for backward pass)
    """
    batch, out_channels, out_width = out_shape
    batch_, in_channels, width = input_shape
    out_channels_, in_channels_, kw = weight_shape

    # Extract strides
    s1_batch, s1_chan, s1_w = input_strides[0], input_strides[1], input_strides[2]
    s2_out, s2_in, s2_k = weight_strides[0], weight_strides[1], weight_strides[2]
    so_batch, so_chan, so_w = out_strides[0], out_strides[1], out_strides[2]

    # TODO: Implement for Task 4.1
    # Parallel over output positions

    for b in prange(batch):
        for oc in range(out_channels):
            for w in range(out_width):
                # Output position
                out_pos = b * so_batch + oc * so_chan + w * so_w

                # Accumulate convolution
                acc = 0.0
                for ic in range(in_channels):
                    for k in range(kw):
                        # Compute input position
                        if reverse:
                            w_in = w - k  # Anchor right
                        else:
                            w_in = w + k  # Anchor left

                        # Check bounds (implicit zero padding)
                        if 0 <= w_in < width:
                            in_pos = (
                                b * s1_batch +
                                ic * s1_chan +   # Q1: Channel stride
                                w_in * s1_w   # Q2: Width stride
                            )
                            weight_pos = (
                                oc * s2_out +
                                ic * s2_in +   # Q3: Input channel stride
                                k * s2_k      # Q4: Kernel position stride
                            )
                            acc += input[in_pos] * weight[weight_pos]

                out[out_pos] = acc


tensor_conv1d = njit(parallel=True)(_tensor_conv1d)


class Conv1dFun(Function):
    @staticmethod
    def forward(ctx, input, weight):
        """
        1D convolution.

        Args:
            input: batch x in_channels x width
            weight: out_channels x in_channels x kernel_width
        """
        ctx.save_for_backward(input, weight)
        batch, in_channels, width = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        out = input.zeros((batch, out_channels, width))
        tensor_conv1d(
            out._tensor._storage,
            out.shape,
            out._tensor.strides,
            out.size,
            input._tensor._storage,
            input.shape,
            input._tensor.strides,
            weight._tensor._storage,
            weight.shape,
            weight._tensor.strides,
            False,
        )
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_values
        batch, in_channels, width = input.shape
        out_channels, in_channels2, kw = weight.shape
        assert in_channels == in_channels2

        grad_input = input.zeros(input.shape)
        weight_t = weight.permute(1, 0, 2).contiguous()
        tensor_conv1d(
            grad_input._tensor._storage,
            grad_input.shape,
            grad_input._tensor.strides,
            grad_input.size,
            grad_output._tensor._storage,
            grad_output.shape,
            grad_output._tensor.strides,
            weight_t._tensor._storage,
            weight_t.shape,
            weight_t._tensor.strides,
            True,
        )

        input_t = input.permute(1, 0, 2).contiguous()
        grad_output_t = grad_output.permute(1, 0, 2).contiguous()
        grad_weight_t = weight.zeros((in_channels, out_channels, kw))
        tensor_conv1d(
            grad_weight_t._tensor._storage,
            grad_weight_t.shape,
            grad_weight_t._tensor.strides,
            grad_weight_t.size,
            input_t._tensor._storage,
            input_t.shape,
            input_t._tensor.strides,
            grad_output_t._tensor._storage,
            grad_output_t.shape,
            grad_output_t._tensor.strides,
            False,
        )
        grad_weight = grad_weight_t.permute(1, 0, 2).contiguous()

        return grad_input, grad_weight
    
