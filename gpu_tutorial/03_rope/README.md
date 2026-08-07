# Module 03: RoPE kernel—pairwise rotation <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

## Concept

The CPU implementation (`cpu/run.cpp:262`) has two nested loops. The outer loop
visits each dimension pair with `i += 2`, computes a rotation frequency from
`head_dim = i % head_size`, and rotates `(v0, v1)` as a complex number by
`pos * freq` radians. For pairs where `i < kvd`, the inner `rotn` logic rotates
both q and k.

The GPU mapping is direct: map the CPU loop index to a CUDA thread and keep the loop
body unchanged. One thread handles one pair.

## Task

```cpp
// q has dim elements; k has kv_dim elements; head_size = dim / n_heads.
__global__ void rope_kernel(float* q, float* k, int pos,
                            int dim, int kvd, int head_size) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= dim) return;

    const int head_dim = i % head_size;
    const float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    const float val = pos * freq;
    const float fcr = cosf(val), fci = sinf(val);

    // Rotate q pair i.
    const float q0 = q[i], q1 = q[i + 1];
    q[i]     = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    // Rotate the matching k pair when i < kvd, matching the CPU rotn logic.
    if (i < kvd) {
        const float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}
```

Launch `dim / 2` threads:
`rope_kernel<<<(dim/2 + 255) / 256, 256>>>(...)`.

**Common mistake:** the `i < kvd` boundary controls whether k is rotated. In MHA
models such as stories15M, `kvd == dim`, so all of k is rotated. In GQA models,
`kvd < dim`, so only the first `kvd` dimensions of k are rotated. Follow the CPU
implementation exactly.

## Acceptance criteria

For fixed `pos` (for example 7) and random q/k, the maximum error after GPU and CPU
RoPE is below 1e-5. Small differences between device `cosf`/`powf` and host
`std::cos`/`std::pow` are expected.

## Files

- `main.cu` — exercise template with a complete upload → kernel → download
  harness and a TODO kernel stub
- `solution.cu` — the same harness with the complete reference kernel
- `cases.h` — CPU reference and two cases: explicit dim=8 toy data and
  LCG-generated dim=288 data; rotation is in place

```bash
# Build
nvcc -O2 -arch=sm_89 -o main main.cu
# Run
./main
# Verify
python3 ../../cpu_tutorial/tools/compare.py out_q.txt      data/expected_q.txt
python3 ../../cpu_tutorial/tools/compare.py out_k.txt      data/expected_k.txt
python3 ../../cpu_tutorial/tools/compare.py out_q_real.txt data/expected_q_real.txt
python3 ../../cpu_tutorial/tools/compare.py out_k_real.txt data/expected_k_real.txt
```
