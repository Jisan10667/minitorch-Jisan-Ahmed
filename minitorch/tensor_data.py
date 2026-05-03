from __future__ import annotations
import random
from typing import Iterable, Optional, Sequence, Tuple, Union
import numpy as np
import numpy.typing as npt
from numpy import array, float64
from typing_extensions import TypeAlias
from .operators import prod

MAX_DIMS = 32


class IndexingError(RuntimeError):
    "Exception raised for indexing errors."
    pass


Storage: TypeAlias = npt.NDArray[np.float64]
OutIndex: TypeAlias = npt.NDArray[np.int32]
Index: TypeAlias = npt.NDArray[np.int32]
Shape: TypeAlias = npt.NDArray[np.int32]
Strides: TypeAlias = npt.NDArray[np.int32]

UserIndex: TypeAlias = Sequence[int]
UserShape: TypeAlias = Sequence[int]
UserStrides: TypeAlias = Sequence[int]


def index_to_position(index: Index, strides: Strides) -> int:
    """
    Convert a multidimensional tensor index into a single-dimensional
    position in storage based on strides.

    Args:
        index: index tuple of ints (as numpy array)
        strides: tensor strides (as numpy array)

    Returns:
        Position in storage
    """
    # TODO: Implement for Task 2.1
    # Hint: The position is the dot product of index and strides
    # Example: index=[1, 2], strides=[4, 1] -> position = 1*4 + 2*1 = 6
    position = np.dot(index, strides)  # Q1: What operation combines index and strides?
    return int(position)      # Q2: Return type should be int


def to_index(ordinal: int, shape: Shape, out_index: OutIndex) -> None:
    """
    Convert an ordinal (flat position 0...size-1) to a multi-dimensional
    index in the given shape.

    This is the inverse mapping: given position in enumeration order,
    produce the corresponding index.

    Args:
        ordinal: ordinal position to convert
        shape: tensor shape
        out_index: output array to fill with index values

    Returns:
        None (modifies out_index in place)
    """
    # TODO: Implement for Task 2.1
    # Hint: Work from the last dimension to the first
    # Use modulo and integer division
    # Example: ordinal=5, shape=(2,3) -> out_index=[1, 2]
    #   5 % 3 = 2 (last index)
    #   5 // 3 = 1 (remaining ordinal for next dimension)

    cur_ord = ordinal
    for i in range(len(shape) - 1, -1, -1):
        out_index[i] = cur_ord % shape[i]  # Q3: What operation gives the index for dimension i?
        cur_ord = cur_ord // shape[i]       # Q4: How do you get the remaining ordinal?


def strides_from_shape(shape: UserShape) -> UserStrides:
    """Return row-major contiguous strides for a shape."""
    layout = [1] * len(shape)
    stride = 1
    for i in range(len(shape) - 1, -1, -1):
        layout[i] = stride
        stride *= shape[i]
    return tuple(layout)


def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
    """Broadcast two shapes following NumPy broadcasting rules."""
    out = []
    for a, b in zip(reversed(shape_a), reversed(shape_b)):
        if a == b:
            out.append(a)
        elif a == 1:
            out.append(b)
        elif b == 1:
            out.append(a)
        else:
            raise IndexingError(f"Cannot broadcast shapes {shape_a} and {shape_b}.")

    longer = shape_a if len(shape_a) > len(shape_b) else shape_b
    remaining = longer[: abs(len(shape_a) - len(shape_b))]
    return tuple(remaining) + tuple(reversed(out))


def broadcast_index(
    big_index: Index, big_shape: Shape, shape: Shape, out_index: OutIndex
) -> None:
    """Convert a broadcasted index into an index for a smaller shape."""
    offset = len(big_shape) - len(shape)
    for i in range(len(shape)):
        if shape[i] == 1:
            out_index[i] = 0
        else:
            out_index[i] = big_index[i + offset]

class TensorData:
    _storage: Storage
    _strides: Strides
    _shape: Shape
    strides: UserStrides
    shape: UserShape
    dims: int

    def __init__(
        self,
        storage: Union[Sequence[float], Storage],
        shape: UserShape,
        strides: Optional[UserStrides] = None,
    ):
        if isinstance(storage, np.ndarray):
            self._storage = storage
        else:
            self._storage = array(storage, dtype=float64)

        if strides is None:
            strides = strides_from_shape(shape)

        assert isinstance(strides, tuple), "Strides must be tuple"
        assert isinstance(shape, tuple), "Shape must be tuple"
        if len(strides) != len(shape):
            raise IndexingError(f"Len of strides {strides} must match {shape}.")

        self._strides = array(strides)
        self._shape = array(shape)
        self.strides = strides
        self.dims = len(strides)
        self.size = int(prod(shape))
        self.shape = shape
        assert len(self._storage) == self.size

    def to_cuda_(self) -> None:  # pragma: no cover
        if not numba.cuda.is_cuda_array(self._storage):
            self._storage = numba.cuda.to_device(self._storage)

    def is_contiguous(self) -> bool:
        """
        Check that the layout is contiguous, i.e. outer dimensions have bigger strides than inner dimensions.

        Returns:
            bool : True if contiguous
        """
        last = 1e9
        for stride in self._strides:
            if stride > last:
                return False
            last = stride
        return True

    @staticmethod
    def shape_broadcast(shape_a: UserShape, shape_b: UserShape) -> UserShape:
        return shape_broadcast(shape_a, shape_b)

    def index(self, index: Union[int, UserIndex]) -> int:
        if isinstance(index, int):
            aindex: Index = array([index])
        if isinstance(index, tuple):
            aindex = array(index)

        # Pretend 0-dim shape is 1-dim shape of singleton
        shape = self.shape
        if len(shape) == 0 and len(aindex) != 0:
            shape = (1,)

        # Check for errors
        if aindex.shape[0] != len(self.shape):
            raise IndexingError(f"Index {aindex} must be size of {self.shape}.")
        for i, ind in enumerate(aindex):
            if ind >= self.shape[i]:
                raise IndexingError(f"Index {aindex} out of range {self.shape}.")
            if ind < 0:
                raise IndexingError(f"Negative indexing for {aindex} not supported.")

        # Call fast indexing.
        return index_to_position(array(index), self._strides)

    def indices(self) -> Iterable[UserIndex]:
        lshape: Shape = array(self.shape)
        out_index: Index = array(self.shape)
        for i in range(self.size):
            to_index(i, lshape, out_index)
            yield tuple(out_index)

    def sample(self) -> UserIndex:
        return tuple((random.randint(0, s - 1) for s in self.shape))

    def get(self, key: UserIndex) -> float:
        x: float = self._storage[self.index(key)]
        return x

    def set(self, key: UserIndex, val: float) -> None:
        self._storage[self.index(key)] = val

    def tuple(self) -> Tuple[Storage, Shape, Strides]:
        return (self._storage, self._shape, self._strides)

    def permute(self, *order: int) -> TensorData:
        """
        Permute the dimensions of the tensor.

        Args:
            *order: a permutation of the dimensions

        Returns:
            New TensorData with the same storage and a new dimension order.
        """
        assert list(sorted(order)) == list(range(len(self.shape))), \
            f"Must give a position to each dimension. Shape: {self.shape} Order: {order}"

        # TODO: Implement for Task 2.1
        # Hint: Reorder both shape and strides according to order
        # The storage stays the same - only the view changes

        new_shape = tuple(self.shape[i] for i in order)   # Q5: Reorder shape by order
        new_strides = tuple(self.strides[i] for i in order) # Q6: Reorder strides by order

        return TensorData(self._storage, new_shape, new_strides)
    
    def to_string(self) -> str:
        s = ""
        for index in self.indices():
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == 0:
                    l = "\n%s[" % ("\t" * i) + l
                else:
                    break
            s += l
            v = self.get(index)
            s += f"{v:3.2f}"
            l = ""
            for i in range(len(index) - 1, -1, -1):
                if index[i] == self.shape[i] - 1:
                    l += "]"
                else:
                    break
            if l:
                s += l
            else:
                s += " "
        return s
