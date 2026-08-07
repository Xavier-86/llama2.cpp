# Module 02: RMSNorm kernel—an introduction to reductions <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

## Concept

The CPU implementation (`cpu/run.cpp:198`) computes `ss = mean(x²)`, followed by
`o[i] = w[i] * x[i] / sqrt(ss + eps)`. This operator first reduces a sum and then
scales each element. Reduction is a fundamental CUDA pattern, making this a useful
first kernel.

Use one block for one vector row and perform a tree reduction in shared memory.

## Task

```cpp
__global__ void rmsnorm_kernel(float* o, const float* x, const float* weight, int n) {
    // One block handles the entire vector; x and o both have n elements.
    __shared__ float sdata[256];
    const int tid = threadIdx.x;

    // 1) Accumulate chunks of x² into shared memory.
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) acc += x[i] * x[i];
    sdata[tid] = acc;
    __syncthreads();

    // 2) Tree reduction.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // 3) Broadcast the scale and write each output element.
    const float scale = rsqrtf(sdata[0] / n + 1e-5f);   // Match the CPU epsilon.
    for (int i = tid; i < n; i += blockDim.x) o[i] = weight[i] * x[i] * scale;
}
```

Launch with `rmsnorm_kernel<<<1, 256>>>(xb, x, rms_att_weight_l, dim);`. There is
one block per row, so the grid size is always one. A 15M model has `dim=288`, but
the strided loops also handle larger dimensions.

**Check epsilon:** the kernel value must match the CPU `rmsnorm` near
`cpu/run.cpp:198`, or exact end-to-end alignment will fail here.

Forward uses RMSNorm three times: before attention, before the FFN, and at the end.
Their weight slices are `rms_att_weight + l*dim`,
`rms_ffn_weight + l*dim`, and `rms_final_weight`.

## Acceptance criteria

For random input, maximum RMSNorm error between GPU and CPU is below 1e-5.

## Files

- `main.cu` — exercise template with a complete harness; replace the TODO
  `rmsnorm_kernel` stub
- `solution.cu` — the same harness with the complete reference kernel
- `cases.h` — test cases and CPU reference: an explicit 8-element toy vector and
  an LCG-generated 288-element case

```bash
# Build (cuBLAS is not required)
nvcc -O2 -arch=sm_89 -o main main.cu
# Run
./main
# Verify
python3 ../../cpu_tutorial/tools/compare.py out.txt      data/expected.txt
python3 ../../cpu_tutorial/tools/compare.py out_real.txt data/expected_real.txt
```
