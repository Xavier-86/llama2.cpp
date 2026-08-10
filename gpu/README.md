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
# Build from the repository root (host compiler must support C++20; with CUDA 12.8
# the default g++-9 was too old, so g++-13 was selected explicitly)
nvcc -O3 -std=c++20 -ccbin g++-13 -arch=sm_89 -o gpu/rungpu gpu/rungpu.cu -lcublas
nvcc -O3 -std=c++20 -ccbin g++-13 -arch=sm_89 -o gpu/runqgpu gpu/runqgpu.cu

# Run
./gpu/rungpu models/stories15M.bin -t 0.0 -n 128 -s 42 -i "Once upon a time"
./gpu/runqgpu models/stories15M-q32.bin -t 0.0 -n 128 -s 42 -i "Once upon a time"
```

**Step-by-step tutorial: [`gpu_tutorial/`](../gpu_tutorial/README.md)** —
`rungpu.cu` is the finished result of module 06. The eight modules cover everything
from environment setup to exact output alignment with the CPU implementation.

Compile-verified on an RTX 4080 SUPER (sm_89) with CUDA 12.8. Note: CUDA 12.0's
nvcc rejected this code (host-side C++20 headers with the system g++), so CUDA
12.8 + g++-13 was used.

Measured on the RTX 4080 SUPER (`-t 0.0 -n 256 -i "Once upon a time"`), FP32 vs
int8 (GS=32):

| Model | Size | tok/s FP32 (cuBLAS) | tok/s int8 naive kernel | tok/s int8 fused kernel |
| --- | --- | --- | --- | --- |
| stories15M | 58 MB → 17 MB | ~2660 | ~2200 | ~3060 |
| stories42M | 160 MB → 45 MB | ~1490 | ~1510 | ~2100 |

"naive kernel" is the first int8 version (one block per row, byte-wise loads, a
separate `quantize_kernel` before every matmul); "fused kernel" is the current
`qmatmul_kernel` described below.

On the CPU, int8 decode is ~10× faster because decode is memory-bandwidth bound
(see [`docs/quantization.md`](../docs/quantization.md)). On the GPU the same
holds only if the int8 kernel actually exploits the 4× smaller weight reads: the
naive version was *slower* than cuBLAS FP32, because launch overhead and idle
threads cancelled the bandwidth win. The current `qmatmul_kernel` is a fused
warp-per-row GEMV (float4 / int8x4 vectorized loads, on-the-fly activation
quantization via segmented warp reductions, `__dp4a` products, no separate
quantize launch), which lifts int8 past FP32 — the gap grows with model size, as
expected for a bandwidth-bound workload. Quantization also cuts VRAM usage by 4×.
