# Module 00: Project skeleton <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

Copy `cpu/run.cpp` to `gpu/rungpu.cu`, remove the bodies of `rmsnorm`,
`softmax`, `matmul`, and `forward`, and keep everything else (`MappedFile`,
`Tokenizer`, `Sampler`, `generate`, and `main`) unchanged. Then add the
following three pieces.

## 1. Error-checking macros

The first rule of CUDA debugging is to check every API call. Many failures otherwise
appear asynchronous or silent.

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err_ = (call);                                           \
    if (err_ != cudaSuccess)                                             \
        throw std::runtime_error(std::string("CUDA error: ") +           \
            cudaGetErrorString(err_) + " at " + __FILE__ +               \
            ":" + std::to_string(__LINE__));                             \
} while (0)

#define CUBLAS_CHECK(call) do {                                          \
    cublasStatus_t st_ = (call);                                         \
    if (st_ != CUBLAS_STATUS_SUCCESS)                                    \
        throw std::runtime_error("cuBLAS error " + std::to_string(st_) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
} while (0)
```

Kernel launches (`<<<...>>>`) do not return an error code. During debugging, call
`CUDA_CHECK(cudaGetLastError());` immediately after each launch.

## 2. Weight upload: a device mirror of `TransformerWeights`

Each member of the CPU weights structure is a `std::span` into the memory-mapped
checkpoint. The device version is also a pointer table, but every pointer refers to
device memory:

```cpp
struct DeviceWeights {
    float *token_embedding_table, *rms_att_weight, *wq, *wk, *wv, *wo,
          *rms_ffn_weight, *w1, *w2, *w3, *rms_final_weight, *wcls;
};

// Upload one tensor with cudaMalloc + cudaMemcpy and return its device pointer.
float* upload(std::span<const float> src) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, src.size_bytes()));
    CUDA_CHECK(cudaMemcpy(d, src.data(), src.size_bytes(), cudaMemcpyHostToDevice));
    return d;
}
```

Call `upload` for every member in the `Transformer` constructor (or a separate
`to_gpu()`). When the classifier is shared, `wcls` aliases
`token_embedding_table`; preserve that alias on the device instead of uploading it
twice.

## 3. cuBLAS handle and activation buffers

```cpp
cublasHandle_t cublas;
CUBLAS_CHECK(cublasCreate(&cublas));          // Call cublasDestroy(cublas) at shutdown.
```

Replace each `std::vector<float>` in `RunState` with an equally sized device
allocation. Zero the KV cache with `cudaMemset`. This is not required for correct
execution, but makes accidental out-of-bounds reads easier to diagnose.

## Acceptance criteria

The program builds, uploads every weight, and prints tensor byte counts whose total
matches the checkpoint file.

## Files

- `main.cu` — exercise template; header/weight loading, size-table output, and
  round-trip checks are provided; `upload()` (task 1) and per-tensor
  `DeviceWeights` uploads (task 2) are TODOs
- `solution.cu` — reference implementation
- `cases.h` — checkpoint tensor-size table shared by the template, solution, and
  data generator

```bash
# Build:  nvcc -O2 -arch=sm_89 -o main main.cu
# Run:    ./main
# Verify: python3 ../../cpu_tutorial/tools/compare.py out_upload.txt data/expected_upload.txt --exact
```
