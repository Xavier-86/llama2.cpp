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
./gpu/runqgpu models/stories15M-q32.bin -k naive -t 0.0 -n 128 -s 42 -i "..."  # -k naive|fused (default fused)
```

**Step-by-step tutorial: [`gpu_tutorial/`](../gpu_tutorial/README.md)** —
`rungpu.cu` is the finished result of module 06. The eight modules cover everything
from environment setup to exact output alignment with the CPU implementation.

Compile-verified on an RTX 4080 SUPER (sm_89) with CUDA 12.8. Note: CUDA 12.0's
nvcc rejected this code (host-side C++20 headers with the system g++), so CUDA
12.8 + g++-13 was used.

## Alibaba PPU (PPU-ZW810E)

The PPU SDK (v2.0.0) ships a CUDA 12.8-compatible toolchain (nvcc + cuBLAS), so
both files build and run unmodified — the default host g++ is new enough, and
`-ccbin`/`-arch` must be left out:

```bash
nvcc -O3 -std=c++20 -o rungpu gpu/rungpu.cu -lcublas
nvcc -O3 -std=c++20 -o runqgpu gpu/runqgpu.cu
```

Two PPU-specific gotchas:

- **Do not pass `-arch=sm_89`**: it compiles, but the binary silently produces
  garbage output (`<unk><unk>...`). Omitting `-arch` lets nvcc target the PPU
  natively and gives correct results.
- If the repo sits on a network filesystem (ossfs), link the binary to a local
  disk (`-o /tmp/rungpu`); `ld` fails with `final link failed: file truncated`
  when writing the executable to ossfs.

Correctness (`-t 0.0 -n 256 -s 42 -i "Once upon a time"`): FP32 output matches
`cpu/runcpp` exactly for both stories models. The int8 outputs are coherent
stories but may diverge from `cpu/runqcpp` at a near-tie argmax (expected, see
[`docs/quantization.md`](../docs/quantization.md)); the fused kernel matched the
CPU text exactly on stories42M.

Measured on the PPU-ZW810E (same options as above), FP32 vs int8 (GS=32):

| Model | Size | tok/s FP32 (cuBLAS) | tok/s int8 naive kernel | tok/s int8 fused kernel |
| --- | --- | --- | --- | --- |
| stories15M | 58 MB → 17 MB | ~1660 | ~1640 | ~1350 |
| stories42M | 160 MB → 45 MB | ~1136 | ~944 | ~972 |

Unlike on the RTX 4080 SUPER, cuBLAS FP32 stays fastest on the PPU: the fused
kernel's bandwidth win does not materialize there. The naive kernel beats the
fused one on the tiny 15M matrices (fewer launches win over per-warp reduction
overhead), while the fused kernel pulls ahead on 42M. Quantization still cuts
the model's device memory 4×, which can matter more than speed on this device.

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
