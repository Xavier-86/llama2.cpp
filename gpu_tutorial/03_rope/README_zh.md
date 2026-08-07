# 模块 03：RoPE kernel —— 逐对旋转 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 概念

CPU 版（`cpu/run.cpp:262`）是两层循环：外层按 `i += 2` 遍历所有维度对，每对按 `head_dim = i % head_size` 算出旋转频率，把 `(v0, v1)` 当作复数旋转 `pos * freq` 弧度；内层对 `i < kvd` 的对同时旋转 q 和 k（`rotn` 逻辑）。

GPU 思路：**把 CPU 的循环下标映射成线程下标，循环体原样保留**——一个线程处理一对。

## 任务

```cpp
// q: dim 维；k: kv_dim 维；head_size = dim / n_heads
__global__ void rope_kernel(float* q, float* k, int pos,
                            int dim, int kvd, int head_size) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;  // 偶数下标
    if (i >= dim) return;

    const int head_dim = i % head_size;
    const float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    const float val = pos * freq;
    const float fcr = cosf(val), fci = sinf(val);

    // 旋转 q 的第 i 对
    const float q0 = q[i], q1 = q[i + 1];
    q[i]     = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    // i < kvd 时同步旋转 k（与 CPU 版 rotn 逻辑对应）
    if (i < kvd) {
        const float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}
```

启动线程数 `dim / 2`：`rope_kernel<<<(dim/2 + 255) / 256, 256>>>(...)`。

**易错点**：`i < kvd` 这个边界决定 k 是否被旋转。MHA 模型（stories15M）`kvd == dim`，k 全部被旋转；GQA 模型 `kvd < dim`，只有前 `kvd` 维对应的 k 对被旋转。照抄 CPU 版，别想当然。

## 验收

固定 `pos`（比如 7），随机 q/k，GPU/CPU RoPE 后最大误差 < 1e-5（`cosf`/`powf` 与 host 端 `std::cos`/`std::pow` 精度略有差异，属正常）。

## 代码文件

- `main.cu`：学生模板。harness 完整（upload → `rope_kernel` → download → 写 `out_*.txt`），kernel 是 `// TODO(task 1)` stub，由你实现。
- `solution.cu`：参考答案。harness 与 `main.cu` 相同，kernel 完整实现。
- `cases.h`：测试用例与 CPU 参考（纯 C++，三处共用）。玩具用例 dim=8/kvd=8/head_size=4/pos=7，q/k 为显式数组；真实用例 dim=288/kvd=288/head_size=48/pos=42，q/k 用 LCG 生成。`rope_cpu` 逐行翻译 `cpu/run.cpp:262-276`。旋转原地进行，输出的是旋转后的 q 和 k。

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
