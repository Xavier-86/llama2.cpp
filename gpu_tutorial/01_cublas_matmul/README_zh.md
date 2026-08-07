# 模块 01：cuBLAS matmul —— 全移植最大的坑 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 概念：row-major 遇上 column-major

CPU 版 `matmul(xout, x, w)` 计算 `xout[i] = Σ_j w[i*n + j] * x[j]`，即 `y = Wx`，W 是 **row-major** 的 d×n（d 行 n 列）。

cuBLAS 假定 **column-major**。同一个内存缓冲区，row-major 的 d×n 矩阵，按 column-major 解读就是 n×d 的 **Wᵀ**。所以要算 `y = Wx`，等价于算 `y = (Wᵀ)ᵀx`，即对 Wᵀ 做 **OP_T**：

```cpp
// y = W x；w 是 row-major 的 d×n（和 CPU 版完全相同的内存布局）
void matmul_gpu(cublasHandle_t h, float* y, const float* x, const float* w, int n, int d) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemv(h, CUBLAS_OP_T,
                             n, d,          // column-major 视角下 Wᵀ 是 n×d
                             &alpha,
                             w, n,          // lda = n（= row-major 的行长度）
                             x, 1,
                             &beta,
                             y, 1));
}
```

三个要点：

- `lda` 永远是 **column-major 数组的列方向跨度**，也就是原 row-major 矩阵每行的元素数 `n`，填错结果就是乱码（这是 cuBLAS 移植最常见的 bug）
- `alpha`/`beta` 是 **host 指针**，不是设备指针
- decode 阶段每次只算一个 token，GEMV（矩阵×向量）就对了；以后做 prefill 批量才需要 `cublasSgemm`

## 任务

实现 `matmul_gpu`，把 forward 里 7 处 matmul（q/k/v/wo/w1/w3/w2）和最后的分类头 logits 都换成它。注意每处的 `n`/`d` 参数不同（照抄 CPU 版对应位置的维度）。

## 验收

写一个小测试：host 上生成随机 `w`/`x`，分别跑 CPU `matmul` 和 `matmul_gpu`，结果拷回对比，最大绝对误差 < 1e-4。

误差不会逐 bit 相等（GPU 归约顺序不同），这是正常的——回忆 cpu_tutorial 模块 12 里 FMA 收缩导致文本分叉的教训，1e-5 量级差异可接受，1e-2 以上就是 bug。

## 代码文件

- `main.cu` —— 练习模板：用例 harness、upload/download、错误宏都已给定；`matmul_gpu`（task 1）是 TODO
- `solution.cu` —— 参考实现
- `cases.h` —— 玩具 4×3 用例 + 288→768 LCG 用例与 CPU 参考 `matmul_cpu`（纯 C++，模板/答案/数据生成器共用）

```bash
# Build:  nvcc -O2 -arch=sm_89 -o main main.cu -lcublas
# Run:    ./main
# Verify: python3 ../../cpu_tutorial/tools/compare.py out_toy.txt  data/expected_toy.txt
#         python3 ../../cpu_tutorial/tools/compare.py out_real.txt data/expected_real.txt
```
