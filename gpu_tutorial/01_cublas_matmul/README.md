# Module 01: cuBLAS matmul—the biggest porting trap <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

## Concept: row-major meets column-major

The CPU `matmul(xout, x, w)` computes
`xout[i] = Σ_j w[i*n + j] * x[j]`, or `y = Wx`, where W is a row-major d×n
matrix.

cuBLAS assumes column-major storage. The same buffer that represents a row-major d×n
matrix represents **Wᵀ**, an n×d matrix, when interpreted as column-major. Therefore,
compute `y = Wx = (Wᵀ)ᵀx` by applying **OP_T** to that column-major view:

```cpp
// y = W x; w is row-major d×n, exactly matching the CPU memory layout.
void matmul_gpu(cublasHandle_t h, float* y, const float* x, const float* w, int n, int d) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemv(h, CUBLAS_OP_T,
                             n, d,          // Wᵀ is n×d in the column-major view.
                             &alpha,
                             w, n,          // lda = n, the row length in row-major.
                             x, 1,
                             &beta,
                             y, 1));
}
```

Three important details:

- `lda` is the stride along a column of the column-major array, which equals the
  number of elements `n` in each original row-major row; a wrong value produces
  nonsensical output and is the most common cuBLAS porting bug
- `alpha` and `beta` are **host pointers**, not device pointers
- Decode processes one token at a time, so GEMV is appropriate; batched prefill
  would use `cublasSgemm`

## Task

Implement `matmul_gpu`. Replace the seven matmuls in forward (q/k/v/wo/w1/w3/w2)
and the final classifier-head matmul with it. The `n` and `d` arguments differ at
each call site; copy the dimensions from the matching CPU code.

## Acceptance criteria

Generate random `w` and `x` on the host, run CPU `matmul` and `matmul_gpu`,
copy the GPU result back, and check that the maximum absolute error is below 1e-4.

The outputs are not bitwise identical because GPU reductions use a different order.
Differences around 1e-5 are acceptable; differences above 1e-2 indicate a bug.

## Files

- `main.cu` — exercise template with case harness, upload/download helpers, and
  error macros; `matmul_gpu` (task 1) is the TODO
- `solution.cu` — reference implementation
- `cases.h` — toy 4×3 case and 288→768 LCG case with the CPU `matmul_cpu`
  reference, shared by template, solution, and data generator

```bash
# Build:  nvcc -O2 -arch=sm_89 -o main main.cu -lcublas
# Run:    ./main
# Verify: python3 ../../cpu_tutorial/tools/compare.py out_toy.txt  data/expected_toy.txt
#         python3 ../../cpu_tutorial/tools/compare.py out_real.txt data/expected_real.txt
```
