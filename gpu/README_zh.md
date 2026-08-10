# GPU 推理 <span style="float: right;"><a href="README.md">English</a></span>

使用 cuBLAS 和手写 CUDA kernel 实现 GPU 推理。FP32 矩阵乘法使用 cuBLAS，量化推理使用手写 int8 kernel；其余算子（RMSNorm、RoPE、softmax、attention 和 SwiGLU）全部手写。

| 文件 | 说明 |
| --- | --- |
| `rungpu.cu` | FP32 推理，与 `cpu/run.cpp` 一一对应 |
| `runqgpu.cu` | int8 量化推理，与 `cpu/runq.cpp` 一一对应 |

两个文件都通过 `#include` 复用对应 CPU 版本的 host 代码（checkpoint 加载、tokenizer、sampler 和 CLI），仅在 GPU 上实现 forward。

```bash
# 从仓库根目录编译（host 编译器需支持 C++20；CUDA 12.8 下默认的 g++-9 太旧，
# 这里显式指定 g++-13）
nvcc -O3 -std=c++20 -ccbin g++-13 -arch=sm_89 -o gpu/rungpu gpu/rungpu.cu -lcublas
nvcc -O3 -std=c++20 -ccbin g++-13 -arch=sm_89 -o gpu/runqgpu gpu/runqgpu.cu

# 运行
./gpu/rungpu models/stories15M.bin -t 0.0 -n 128 -s 42 -i "Once upon a time"
./gpu/runqgpu models/stories15M-q32.bin -t 0.0 -n 128 -s 42 -i "Once upon a time"
```

**分步教程：[`gpu_tutorial/`](../gpu_tutorial/README_zh.md)** —— `rungpu.cu` 就是模块 06 的最终成品。八个模块涵盖从环境准备到与 CPU 版本输出逐字对齐的完整过程。

已在 RTX 4080 SUPER（sm_89）+ CUDA 12.8 上通过编译验证。注意：CUDA 12.0 的 nvcc 无法编译这份代码（系统 g++ 的 C++20 头文件不兼容），因此使用了 CUDA 12.8 + g++-13。

实测（RTX 4080 SUPER，`-t 0.0 -n 256 -i "Once upon a time"`），FP32 对比 int8（GS=32）：

| 模型 | 体积 | tok/s FP32 (cuBLAS) | tok/s int8 naive kernel | tok/s int8 融合 kernel |
| --- | --- | --- | --- | --- |
| stories15M | 58 MB → 17 MB | ~2660 | ~2200 | ~3060 |
| stories42M | 160 MB → 45 MB | ~1490 | ~1510 | ~2100 |

"naive kernel" 指第一版 int8 实现（一个 block 算一行、逐字节加载、每次 matmul 前单独跑 `quantize_kernel`）；"融合 kernel" 指下面介绍的当前 `qmatmul_kernel`。

CPU 上 int8 decode 提速约 10 倍，因为 decode 是带宽受限的（见 [`docs/quantization_zh.md`](../docs/quantization_zh.md)）。GPU 上这个结论成立的前提是 int8 kernel 真的利用到权重读取量减少 4 倍的优势：naive 版本比 cuBLAS FP32 还慢，因为 launch 开销和空转线程抵消了带宽收益。现在的 `qmatmul_kernel` 是融合的 warp-per-row GEMV（float4 / int8x4 向量化加载、warp 分段归约在线量化激活、`__dp4a` 点积、不再单独 launch 量化 kernel），让 int8 反超 FP32——而且模型越大优势越明显，符合带宽受限负载的预期。量化同时还能把显存占用降为 1/4。
