# MiniTorch Submission Notes

This repository contains a from-scratch MiniTorch implementation through the tensor and CPU fast-ops modules. The main graded Module 3 work is in `minitorch/fast_ops.py`, with CPU parallel tensor operations and matrix multiplication implemented using Numba.

## Module 2.5 Gradient Cross-Verification

The `verify_grads.py` script cross-checks MiniTorch tensor gradients against PyTorch on representative composed tensor expressions. The committed output is in `results/module2_tensor/verify_grads_output.txt`.

```text
add_mul: ok max_abs_a=0.00e+00 max_abs_b=0.00e+00
sigmoid_exp: ok max_abs_a=0.00e+00 max_abs_b=0.00e+00
log_product: ok max_abs_a=0.00e+00 max_abs_b=0.00e+00
all gradient checks passed within 1e-4
```

## Module 3 CPU Numba Benchmark

The benchmark below compares the Numba fast matrix multiply against a plain Python nested-loop baseline on `128x128` matrices. The Numba timing is measured after one warmup call so compilation time is not included.

| Operation | Size | Time |
| --- | ---: | ---: |
| FastOps Numba matmul | 128x128 | 0.004578s |
| Python nested-loop matmul | 128x128 | 0.328331s |
| Speedup | 128x128 | 71.72x |

This exceeds the required 3x speedup target.

## Module 3 Training Curve

`project/run_fast_tensor.py` was run with:

```bash
venv/bin/python project/run_fast_tensor.py --BACKEND cpu --HIDDEN 100 --DATASET split
```

Key points from the training curve:

| Epoch | Loss | Correct | Time |
| ---: | ---: | ---: | ---: |
| 10 | 43.0044 | 31/50 | 0.1474s |
| 20 | 221.8573 | 24/50 | 0.1407s |
| 30 | 199.5607 | 26/50 | 0.1409s |
| 40 | 69.2398 | 27/50 | 0.1418s |
| 50 | 34.0376 | 35/50 | 0.1410s |
| 60 | 19.5842 | 40/50 | 0.1716s |
| 70 | 8.8654 | 47/50 | 0.1715s |
| 80 | 7.2322 | 49/50 | 0.1802s |
| 90 | 6.6742 | 50/50 | 0.1845s |
| 100 | 6.2607 | 50/50 | 0.1710s |
| 110 | 5.9744 | 50/50 | 0.1800s |
| 120 | 5.7398 | 50/50 | 0.1677s |
| 130 | 5.5393 | 50/50 | 0.1839s |
| 140 | 5.3793 | 50/50 | 0.2166s |
| 150 | 5.2360 | 50/50 | 0.1885s |
| 160 | 5.1084 | 50/50 | 0.2301s |
| 170 | 4.9880 | 50/50 | 0.3042s |
| 180 | 4.8781 | 50/50 | 0.2267s |
| 190 | 4.7757 | 50/50 | 0.2283s |
| 200 | 4.6809 | 50/50 | 0.1815s |
| 210 | 4.5901 | 50/50 | 0.1801s |
| 220 | 4.5037 | 50/50 | 0.1805s |
| 230 | 4.4197 | 50/50 | 0.1645s |
| 240 | 4.3407 | 50/50 | 0.1768s |
| 250 | 4.2707 | 50/50 | 0.1705s |
| 260 | 4.2007 | 50/50 | 0.2041s |
| 270 | 4.1380 | 50/50 | 0.1750s |
| 280 | 4.0765 | 50/50 | 0.1636s |
| 290 | 4.0151 | 50/50 | 0.1327s |
| 300 | 3.9540 | 50/50 | 0.1344s |
| 310 | 3.8993 | 50/50 | 0.1345s |
| 320 | 3.8463 | 50/50 | 0.1362s |
| 330 | 3.7959 | 50/50 | 0.1327s |
| 340 | 3.7479 | 50/50 | 0.1399s |
| 350 | 3.7001 | 50/50 | 0.1449s |
| 360 | 3.6539 | 50/50 | 0.1417s |
| 370 | 3.6090 | 50/50 | 0.1632s |
| 380 | 3.5675 | 50/50 | 0.1581s |
| 390 | 3.5263 | 50/50 | 0.1612s |
| 400 | 3.4856 | 50/50 | 0.1770s |
| 410 | 3.4471 | 50/50 | 0.1905s |
| 420 | 3.4107 | 50/50 | 0.1820s |
| 430 | 3.3748 | 50/50 | 0.1942s |
| 440 | 3.3387 | 50/50 | 0.1328s |
| 450 | 3.3060 | 50/50 | 0.1333s |
| 460 | 3.2732 | 50/50 | 0.1331s |
| 470 | 3.2417 | 50/50 | 0.1374s |
| 480 | 3.2103 | 50/50 | 0.1343s |
| 490 | 3.1802 | 50/50 | 0.1372s |
| 500 | 3.1512 | 50/50 | 0.1337s |

The run reached `50/50` correct by epoch 90 and finished epoch 500 with loss `3.1512`.

MNIST note: this repository does not currently include an MNIST training script, MNIST dataset loader, or committed MNIST curve artifact. The available Module 3 training artifact is the CPU fast tensor split-dataset run above.

## CUDA Reading Response

A CUDA thread block is a group of GPU threads that execute together on one streaming multiprocessor and can coordinate through synchronization barriers. In tiled matrix multiplication, shared memory is used to cache small tiles of the input matrices so many threads can reuse the same values without repeatedly reading slower global memory. This improves arithmetic intensity because each loaded tile contributes to many multiply-add operations. The Numba CPU implementation in `fast_ops.py` uses `@njit(parallel=True)` and `prange` to distribute loop iterations across CPU threads, while the CUDA approach maps work to thousands of lightweight GPU threads organized into blocks and explicitly manages shared memory.

## CUDA Bonus Verification

The CUDA bonus tests were verified in a Kaggle Notebook with GPU enabled.

```text
platform linux -- Python 3.12.12
tests/test_tensor_general_.py ................................ [100%]
32 passed, 21 deselected, 74 warnings in 63.14s
```

The warnings were Numba performance warnings about small test grids and host/device copy overhead in the test cases; all selected CUDA tests passed.

## Module 3 Rubric Checklist

| Requirement | Status | Evidence |
| --- | --- | --- |
| 3.1 Numba `tensor_map` and `tensor_zip` | Done | `task3_1: 19 passed, 2 deselected` |
| 3.2 Fast matrix multiply | Done | `task3_2: 2 passed, 19 deselected` |
| 3.3 CUDA reading response | Done | See CUDA Reading Response above |
| 3.4 CPU benchmark | Done | 128x128 benchmark table shows `71.72x` speedup |
| CUDA bonus | Done | Kaggle GPU run: `32 passed, 21 deselected` |

## Module Design Decisions

| Module | Key design decision |
| --- | --- |
| Module 0 | Implemented scalar operators as small pure functions first, making later scalar and tensor backprop rules easy to test independently. |
| Module 1 | Built scalar autodiff around computation histories and topological backpropagation, so each operation only needs to define its local chain rule. |
| Module 2 | Represented tensors with shared storage, shape, and strides, allowing views, permutes, broadcasting, and reductions without unnecessary copies. |
| Module 2.5 | Added PyTorch gradient cross-checking to validate MiniTorch tensor gradients against a trusted reference implementation. |
| Module 3 | Used Numba `njit(parallel=True)` with explicit stride/index math for CPU fast ops, avoiding Python overhead while preserving MiniTorch broadcasting semantics. |

## Verification Commands

```bash
venv/bin/python -m pytest tests/test_tensor_general_.py -m task3_1
venv/bin/python -m pytest tests/test_tensor_general_.py -m task3_2
python -m pytest tests/test_tensor_general_.py -m "task3_3 or task3_4"
venv/bin/python verify_grads.py
```

Recent results:

```text
task3_1: 19 passed, 2 deselected
task3_2: 2 passed, 19 deselected
task3_3/task3_4 on Kaggle GPU: 32 passed, 21 deselected
verify_grads.py: all gradient checks passed within 1e-4
```
