# 模块 02：RMSNorm kernel —— 归约范式入门 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 概念

CPU 版（`cpu/run.cpp:198`）：`ss = mean(x²)`，然后 `o[i] = w[i] * x[i] / sqrt(ss + eps)`。这是一个"先归约求和、再逐元素缩放"的算子——归约（reduction）是 CUDA 的经典范式，这个 kernel 是你练手的第一个。

思路：一个 block 处理一行向量，shared memory 做树形归约。

## 任务

```cpp
__global__ void rmsnorm_kernel(float* o, const float* x, const float* weight, int n) {
    // 一个 block 处理整个向量；x/o 都是 n 维
    __shared__ float sdata[256];
    const int tid = threadIdx.x;

    // 1) 分块累加 x² 到 shared memory
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) acc += x[i] * x[i];
    sdata[tid] = acc;
    __syncthreads();

    // 2) 树形归约
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // 3) 广播 scale 并逐元素写回
    const float scale = rsqrtf(sdata[0] / n + 1e-5f);   // 与 CPU 版 eps 保持一致
    for (int i = tid; i < n; i += blockDim.x) o[i] = weight[i] * x[i] * scale;
}
```

启动：`rmsnorm_kernel<<<1, 256>>>(xb, x, rms_att_weight_l, dim);`——一行一个 block，所以 grid 恒为 1。15M 模型 dim=288，一个 block 就够；dim 更大时这个写法仍然成立（stride 循环）。

**自查 eps**：CPU 版 `rmsnorm` 里的 eps 是多少（`cpu/run.cpp:198` 附近），kernel 里必须一致，否则逐字对齐会在这里挂掉。

forward 中 rmsnorm 出现三次（attention 前、FFN 前、final），权重切片分别是 `rms_att_weight + l*dim`、`rms_ffn_weight + l*dim`、`rms_final_weight`。

## 验收

随机输入，GPU/CPU rmsnorm 最大误差 < 1e-5。

## 代码文件

- `main.cu` —— 学生模板：harness 完整（用例执行、upload/download、写 `out*.txt`），`rmsnorm_kernel` 是 TODO 注释 + stub，替换成你的实现即可。
- `solution.cu` —— 参考答案：与模板相同的 harness + 完整 kernel 实现。
- `cases.h` —— 测试用例与 CPU 参考实现（玩具 8 维显式数组 + LCG 伪随机 288 维真实用例），纯 C++，三处共用。

```bash
# Build（不需要 cuBLAS）
nvcc -O2 -arch=sm_89 -o main main.cu
# Run
./main
# Verify
python3 ../../cpu_tutorial/tools/compare.py out.txt      data/expected.txt
python3 ../../cpu_tutorial/tools/compare.py out_real.txt data/expected_real.txt
```
