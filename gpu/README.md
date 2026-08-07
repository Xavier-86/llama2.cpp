# GPU inference <span style="float: right;"><a href="README_zh.md">中文</a></span>

GPU inference with cuBLAS and hand-written CUDA kernels. Matrix multiplication uses
cuBLAS for FP32 or a hand-written int8 kernel for quantized inference; all other
operations (RMSNorm, RoPE, softmax, attention, and SwiGLU) are hand-written.

| File | Description |
| --- | --- |
| `rungpu.cu` | FP32 inference; mirrors `cpu/run.cpp` |
| `runqgpu.cu` | int8 quantized inference; mirrors `cpu/runq.cpp` |

Both reuse the host code of their CPU counterpart via `#include` (checkpoint loading,
tokenizer, sampler, and CLI) and implement only the forward pass on the GPU.

```bash
# Build from the repository root
nvcc -O3 -std=c++20 -arch=sm_89 -o gpu/rungpu gpu/rungpu.cu -lcublas
nvcc -O3 -std=c++20 -arch=sm_89 -o gpu/runqgpu gpu/runqgpu.cu

# Run
./gpu/rungpu models/stories15M.bin -t 0.0 -n 128 -s 42 -i "Once upon a time"
./gpu/runqgpu models/stories15M-q32.bin -t 0.0 -n 128 -s 42 -i "Once upon a time"
```

**Step-by-step tutorial: [`gpu_tutorial/`](../gpu_tutorial/README.md)** —
`rungpu.cu` is the finished result of module 06. The eight modules cover everything
from environment setup to exact output alignment with the CPU implementation.

Note: the machine this was written on had no CUDA toolkit, so these files are not yet
compile-verified; the kernel logic mirrors `gpu_tutorial/` module solutions 1:1.
