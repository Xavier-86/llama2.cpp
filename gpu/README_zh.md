# GPU 推理 <span style="float: right;"><a href="README.md">English</a></span>

使用 cuBLAS 和手写 CUDA kernel 实现 GPU 推理。FP32 矩阵乘法使用 cuBLAS，量化推理使用手写 int8 kernel；其余算子（RMSNorm、RoPE、softmax、attention 和 SwiGLU）全部手写。

| 文件 | 说明 |
| --- | --- |
| `rungpu.cu` | FP32 推理，与 `cpu/run.cpp` 一一对应 |
| `runqgpu.cu` | int8 量化推理，与 `cpu/runq.cpp` 一一对应 |

两个文件都通过 `#include` 复用对应 CPU 版本的 host 代码（checkpoint 加载、tokenizer、sampler 和 CLI），仅在 GPU 上实现 forward。

```bash
# 从仓库根目录编译
nvcc -O3 -std=c++20 -arch=sm_89 -o gpu/rungpu gpu/rungpu.cu -lcublas
nvcc -O3 -std=c++20 -arch=sm_89 -o gpu/runqgpu gpu/runqgpu.cu

# 运行
./gpu/rungpu models/stories15M.bin -t 0.0 -n 128 -s 42 -i "Once upon a time"
./gpu/runqgpu models/stories15M-q32.bin -t 0.0 -n 128 -s 42 -i "Once upon a time"
```

**分步教程：[`gpu_tutorial/`](../gpu_tutorial/README_zh.md)** —— `rungpu.cu` 就是模块 06 的最终成品。八个模块涵盖从环境准备到与 CPU 版本输出逐字对齐的完整过程。

注意：编写这些文件的机器没有 CUDA 工具包，因此尚未通过编译验证；kernel 逻辑与 `gpu_tutorial/` 各模块的参考答案一一对应。
