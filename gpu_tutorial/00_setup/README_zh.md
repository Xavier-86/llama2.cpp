# 模块 00：工程骨架 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

从 `cpu/run.cpp` 复制出 `gpu/rungpu.cu`，**删掉** `rmsnorm`/`softmax`/`matmul`/`forward` 的函数体，其余（`MappedFile`、`Tokenizer`、`Sampler`、`generate`、`main`）原样保留。然后加三样东西。

## 1. 错误检查宏

CUDA 调试的第一课：所有 API 错误都是异步/静默的，不检查就等于裸奔。

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

kernel 启动（`<<<...>>>`）不返回错误码，调试期在启动后加一句 `CUDA_CHECK(cudaGetLastError());`。

## 2. 权重上传：镜像 `TransformerWeights` 的设备版

CPU 版 weights 的每个成员是指向 mmap 区域的 `std::span`。设备版同样是一张"指针表"，只是指针指向显存：

```cpp
struct DeviceWeights {
    float *token_embedding_table, *rms_att_weight, *wq, *wk, *wv, *wo,
          *rms_ffn_weight, *w1, *w2, *w3, *rms_final_weight, *wcls;
};

// 上传一个张量：cudaMalloc + cudaMemcpy，返回设备指针
float* upload(std::span<const float> src) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, src.size_bytes()));
    CUDA_CHECK(cudaMemcpy(d, src.data(), src.size_bytes(), cudaMemcpyHostToDevice));
    return d;
}
```

在 `Transformer` 构造（或单独的 `to_gpu()`）里对每个成员调一次 `upload`。注意 `wcls`：共享分类头时它指向 `token_embedding_table`，设备端也让两个指针相等即可，不要重复上传。

## 3. cuBLAS 句柄与激活缓冲区

```cpp
cublasHandle_t cublas;
CUBLAS_CHECK(cublasCreate(&cublas));          // 程序结束 cublasDestroy(cublas)
```

`RunState` 的每个 `std::vector<float>` 换成 `cudaMalloc` 出来的设备指针，大小不变。KV cache 用 `cudaMemset` 清零（其实不必，但清零后越界读不会读到随机大数，调试友好）。

## 验收

编译通过，程序能跑完权重上传并打印每个张量的字节数，与 checkpoint 文件大小对账一致。

## 代码文件

- `main.cu` —— 练习模板：读 header/权重区、尺寸表输出、roundtrip 对拍都已给定；`upload()`（task 1）和 `DeviceWeights` 逐张量上传（task 2）是 TODO
- `solution.cu` —— 参考实现
- `cases.h` —— checkpoint 权重张量尺寸表（纯 C++，模板/答案/数据生成器共用）

```bash
# Build:  nvcc -O2 -arch=sm_89 -o main main.cu
# Run:    ./main
# Verify: python3 ../../cpu_tutorial/tools/compare.py out_upload.txt data/expected_upload.txt --exact
```
