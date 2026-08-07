# 模块 05：逐元素小 kernel 合集 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 概念

剩下的都是"一个线程一个元素"的简单 kernel，一个模板打天下：线程下标 `i = blockIdx.x * blockDim.x + threadIdx.x`，越界返回，循环体照抄 CPU 版。启动方式统一：`<<<(n + 255) / 256, 256>>>`。

## 任务

```cpp
// SwiGLU：hb[i] = silu(hb[i]) * hb2[i]，silu(x) = x / (1 + e^-x)
__global__ void swiglu_kernel(float* hb, const float* hb2, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = hb[i];
    hb[i] = v / (1.0f + expf(-v)) * hb2[i];
}

// 残差连接：x[i] += y[i]
__global__ void add_kernel(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// embedding 查表：把第 token 行拷进 x（也可以直接用 cudaMemcpyDeviceToDevice）
__global__ void embed_kernel(float* x, const float* table, int token, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) x[i] = table[(size_t)token * dim + i];
}
```

各自对应 CPU 版的位置：

- `swiglu_kernel` ← `cpu/run.cpp:322` 的 SwiGLU 循环（注意 CPU 版先算 silu 再乘 `hb2`，顺序一致）
- `add_kernel` ← attention 输出投影后（`x += xb2`）和 FFN 输出投影后（`x += xb`）的两处残差
- `embed_kernel` ← forward 开头的 embedding 行拷贝

## 验收

随机输入下与 CPU 版逐元素一致（这些 kernel 没有归约，浮点结果应当逐 bit 相同或只差 1 ulp——`expf` 与 host `std::exp` 的精度差异）。

## 代码文件

- `main.cu` — 学生模板：harness 完整（上传、启动、下载、写 `out_*.txt`），三个 kernel 是 TODO stub
- `solution.cu` — 参考答案：相同 harness + 三个 kernel 的完整实现
- `cases.h` — 玩具用例（swiglu / add / embed）与 CPU 参考实现，纯 C++ 头文件，三处共用

```bash
# Build（本模块不需要 cuBLAS）
nvcc -O2 -arch=sm_89 -o main main.cu

# Run
./main

# Verify
python3 ../../cpu_tutorial/tools/compare.py out_swiglu.txt data/expected_swiglu.txt
python3 ../../cpu_tutorial/tools/compare.py out_add.txt    data/expected_add.txt
python3 ../../cpu_tutorial/tools/compare.py out_embed.txt  data/expected_embed.txt
```
