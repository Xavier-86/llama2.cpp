# Module 05: Small elementwise kernels <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

## Concept

The remaining operators follow a simple one-thread-per-element pattern: compute
`i = blockIdx.x * blockDim.x + threadIdx.x`, return when out of bounds, and copy
the CPU loop body. Launch all of them with `<<<(n + 255) / 256, 256>>>`.

## Task

```cpp
// SwiGLU: hb[i] = silu(hb[i]) * hb2[i], where silu(x) = x / (1 + e^-x).
__global__ void swiglu_kernel(float* hb, const float* hb2, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = hb[i];
    hb[i] = v / (1.0f + expf(-v)) * hb2[i];
}

// Residual connection: x[i] += y[i].
__global__ void add_kernel(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// Embedding lookup: copy row token into x.
__global__ void embed_kernel(float* x, const float* table, int token, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) x[i] = table[(size_t)token * dim + i];
}
```

CPU counterparts:

- `swiglu_kernel` matches the SwiGLU loop at `cpu/run.cpp:322`; preserve the
  operation order, applying SiLU before multiplying by `hb2`
- `add_kernel` implements the two residual additions after the attention and FFN
  output projections
- `embed_kernel` implements the embedding-row copy at the start of forward

## Acceptance criteria

Each element matches the CPU implementation for random inputs. These kernels contain
no reduction, so results should be bitwise equal or differ by only one ULP due to
device `expf` versus host `std::exp`.

## Files

- `main.cu` — complete upload/launch/download harness with three TODO stubs
- `solution.cu` — the same harness with all three reference kernels
- `cases.h` — toy SwiGLU, add, and embedding cases with CPU references

```bash
# Build (cuBLAS is not required)
nvcc -O2 -arch=sm_89 -o main main.cu
# Run
./main
# Verify
python3 ../../cpu_tutorial/tools/compare.py out_swiglu.txt data/expected_swiglu.txt
python3 ../../cpu_tutorial/tools/compare.py out_add.txt    data/expected_add.txt
python3 ../../cpu_tutorial/tools/compare.py out_embed.txt  data/expected_embed.txt
```
