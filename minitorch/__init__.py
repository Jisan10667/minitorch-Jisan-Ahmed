"""
MiniTorch: A minimal deep learning library for educational purposes.

Module 0: ML Programming Foundations
"""

__version__ = "0.1.0"

# We'll import our implementations as we build them
# from .operators import *  # Module 0.1
# from .module import *     # Module 0.4

from .autodiff import central_difference, History
from .scalar import Scalar
from . import operators
from .tensor_data import TensorData, IndexingError, shape_broadcast
from .operators import prod
from .tensor import Tensor
from .tensor_functions import tensor, zeros, rand
from .optim import SGD
from .testing import MathTestVariable, grad_check, sum_practice, mm_practice
from .tensor_ops import SimpleBackend
from .module import Module, Parameter
from .tensor_functions import Conv1dFun, Conv2dFun      # Added in Chapter 2
from .nn import (                                        # NEW: nn.py exports
    avgpool2d, maxpool2d, softmax, logsoftmax, dropout,
    max, Linear, Conv1d, Conv2d, one_hot, no_grad, argmax,
)
from .fast_ops import *  # noqa: F401,F403
from .tensor_ops import *  # noqa: F401,F403
from .cuda_ops import CudaOps





try:
    from .tensor import Tensor
    from .tensor_functions import tensor
    from .testing import MathTestVariable, grad_check
    from .tensor_ops import SimpleBackend
except ImportError:
    class Tensor:  # type: ignore[no-redef]
        pass

    class SimpleBackend:  # type: ignore[no-redef]
        pass

    def tensor(*args, **kwargs):  # type: ignore[no-redef]
        raise NotImplementedError("Tensor functions are not implemented yet.")

    from .testing import MathTestVariable, grad_check
