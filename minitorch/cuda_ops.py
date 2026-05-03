from numba import cuda
import numba
import numpy as np
from .tensor_data import (
    MAX_DIMS,
    to_index,
    index_to_position,
    broadcast_index,
)

THREADS_PER_BLOCK = 32

def tensor_map(fn):
    f = cuda.jit(device=True)(fn)
    @cuda.jit()
    def _map(out, out_shape, out_strides, out_size, in_storage, in_shape, in_strides):
        i = cuda.grid(1)  # Q1: Compute global thread index
        if i < out_size:  # Q2: Bounds check
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            in_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, in_shape, in_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(in_index, in_strides)
            out[o] = f(in_storage[j])  # Q3: Apply function
    return _map

def tensor_zip(fn):
    f = cuda.jit(device=True)(fn)
    @cuda.jit()
    def _zip(out, out_shape, out_strides, out_size, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides):
        i = cuda.grid(1)
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            a_index = cuda.local.array(MAX_DIMS, numba.int32)
            b_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            broadcast_index(out_index, out_shape, a_shape, a_index)
            broadcast_index(out_index, out_shape, b_shape, b_index)
            o = index_to_position(out_index, out_strides)
            j = index_to_position(a_index, a_strides)
            k = index_to_position(b_index, b_strides)
            out[o] = f(a_storage[j], b_storage[k])  # Q4: Apply function to both inputs
    return _zip

def tensor_reduce(fn, reduce_value):
    f = cuda.jit(device=True)(fn)
    @cuda.jit()
    def _reduce(out, out_shape, out_strides, out_size, a_storage, a_shape, a_strides, reduce_dim, reduce_val):
        i = cuda.grid(1)
        if i < out_size:
            out_index = cuda.local.array(MAX_DIMS, numba.int32)
            to_index(i, out_shape, out_index)
            o = index_to_position(out_index, out_strides)
            acc = reduce_val  # Q5: Initialize accumulator
            for s in range(a_shape[reduce_dim]):
                out_index[reduce_dim] = s
                j = index_to_position(out_index, a_strides)
                acc = f(acc, a_storage[j])  # Q6: Accumulate
            out[o] = acc
    return _reduce

def _broadcast_shape(shape1, shape2):
    res = []
    for d1, d2 in zip(reversed(shape1), reversed(shape2)):
        if d1 != 1 and d2 != 1 and d1 != d2:
            raise Exception(f"Shapes {shape1} and {shape2} are not broadcastable")
        res.append(max(d1, d2))
    while len(shape1) < len(shape2):
        res.append(shape2[len(shape2) - len(shape1) - 1])
        shape1 = (1,) + shape1
    while len(shape2) < len(shape1):
        res.append(shape1[len(shape1) - len(shape2) - 1])
        shape2 = (1,) + shape2
    return tuple(reversed(res))


@cuda.jit()
def _sum_practice(out, in_storage, size):
    BLOCK_SIZE = 32
    cache = cuda.shared.array(BLOCK_SIZE, numba.float64)
    i = cuda.grid(1)
    local_i = cuda.threadIdx.x
    block_id = cuda.blockIdx.x

    # Step 1: Load data into shared memory
    if i < size:
        cache[local_i] = in_storage[i]  # Q7: Load from global memory
    else:
        cache[local_i] = 0.0
    cuda.syncthreads()

    # Step 2: Tree-based parallel reduction within block
    s = BLOCK_SIZE // 2  # Q8: Starting stride (half of BLOCK_SIZE)
    while s > 0:
        if local_i < s:
            cache[local_i] += cache[local_i + s]
        cuda.syncthreads()
        s //= 2

    # Step 3: Thread 0 writes block result to global memory
    if local_i == 0:
        out[block_id] = cache[0]


def sum_practice(a):
    (size,) = a.shape
    out = a.zeros((2,))
    threadsperblock = 32
    blockspergrid = 2
    _sum_practice[blockspergrid, threadsperblock](
        out._tensor._storage, a._tensor._storage, size
    )
    return out

@cuda.jit()
def _mm_kernel(out, out_shape, out_strides, a_storage, a_shape, a_strides, b_storage, b_shape, b_strides):
    TILE = 32

    # Shared memory tiles
    a_shared = cuda.shared.array((TILE, TILE), numba.float64)
    b_shared = cuda.shared.array((TILE, TILE), numba.float64)

    # Thread and block indices
    batch = cuda.blockIdx.z
    ti, tj = cuda.threadIdx.x, cuda.threadIdx.y
    i, j = cuda.blockIdx.x * TILE + ti, cuda.blockIdx.y * TILE + tj

    M, K, L = a_shape[1], a_shape[2], b_shape[2]

    acc = 0.0
    for k_start in range(0, K, TILE):
        # 1. Zero shared memory and sync
        a_shared[ti, tj] = 0.0
        b_shared[ti, tj] = 0.0
        cuda.syncthreads()

        # 2. Load tiles with boundary checks
        if i < M and (k_start + tj) < K:
            a_shared[ti, tj] = a_storage[
                (batch % a_shape[0]) * a_strides[0] +
                i * a_strides[1] +                    # Q1: Row stride
                (k_start + tj) * a_strides[2]         # Q2: Column stride
            ]
        if (k_start + ti) < K and j < L:
            b_shared[ti, tj] = b_storage[
                (batch % b_shape[0]) * b_strides[0] +
                (k_start + ti) * b_strides[1] +       # Q3: Row stride
                j * b_strides[2]                      # Q4: Column stride
            ]
        cuda.syncthreads()

        # 3. Compute partial products from shared memory
        for k_off in range(TILE):
            if (k_start + k_off) < K:
                acc += a_shared[ti, k_off] * b_shared[k_off, tj]
        cuda.syncthreads()

    # Write result
    if i < M and j < L:
        idx = batch * out_strides[0] + i * out_strides[1] + j * out_strides[2]
        out[idx] = acc

def matrix_multiply_launcher(out, a, b):
    def get_meta(t):
        s = list(t.shape)
        st = list(t.strides)
        while len(s) < 3:
            s, st = [1] + s, [0] + st
        return np.array(s, dtype=np.int32), np.array(st, dtype=np.int32)

    s_out, str_out = get_meta(out._tensor)
    s_a, str_a = get_meta(a._tensor)
    s_b, str_b = get_meta(b._tensor)

    TILE = 32
    blockspergrid = (
        (s_out[1] + TILE - 1) // TILE,
        (s_out[2] + TILE - 1) // TILE,
        s_out[0]
    )
    _mm_kernel[blockspergrid, (TILE, TILE, 1)](
        out._tensor._storage, s_out, str_out,
        a._tensor._storage, s_a, str_a,
        b._tensor._storage, s_b, str_b
    )
class CudaOps:
    cuda = True

    @staticmethod
    def map(fn):
        f = tensor_map(fn)
        def ret(a):
            out = a.zeros(a.shape)
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, THREADS_PER_BLOCK](
                out._tensor._storage, out._tensor._shape, out._tensor._strides, out.size,
                a._tensor._storage, a._tensor._shape, a._tensor._strides
            )
            return out
        return ret

    @staticmethod
    def zip(fn):
        f = tensor_zip(fn)
        def ret(a, b):
            out_shape = _broadcast_shape(a.shape, b.shape)
            out = a.zeros(out_shape)
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, THREADS_PER_BLOCK](
                out._tensor._storage, out._tensor._shape, out._tensor._strides, out.size,
                a._tensor._storage, a._tensor._shape, a._tensor._strides,
                b._tensor._storage, b._tensor._shape, b._tensor._strides
            )
            return out
        return ret

    @staticmethod
    def reduce(fn, start=0.0):
        f = tensor_reduce(fn, start)
        def ret(a, dim):
            out_shape = list(a.shape)
            out_shape[dim] = 1
            out = a.zeros(tuple(out_shape))
            blockspergrid = (out.size + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
            f[blockspergrid, THREADS_PER_BLOCK](
                out._tensor._storage, out._tensor._shape, out._tensor._strides, out.size,
                a._tensor._storage, a._tensor._shape, a._tensor._strides,
                int(dim), start
            )
            return out
        return ret

    @staticmethod
    def matrix_multiply(a, b):
        s1, s2 = a.shape, b.shape
        res_shape = (s1[0], s1[1], s2[2]) if len(s1) == 3 else (s1[0], s2[1])
        out = a.zeros(res_shape)
        matrix_multiply_launcher(out, a, b)
        return out

    cmap = map
