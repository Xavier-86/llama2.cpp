# GPU Inference <span style="float: right;"><a href="README_zh.md">中文</a></span>

The GPU implementation is split into three hardware targets. FP32 uses cuBLAS, while int8 uses hand-written CUDA kernels. Checkpoint loading, tokenization, sampling, and the CLI are shared with the CPU implementation.

| Directory | Contents | Purpose |
| --- | --- | --- |
| `default/` | `rungpu.cu`, `runqgpu.cu` | Default FP32/int8 entry points and shared implementation |
| `4080s/` | `runqgpu.cu`, `bench.sh` | RTX 4080 SUPER fused int8 GEMV and benchmark |
| `ppu/` | `runqgpu.cu`, `test_qgemv.cu`, `bench.sh` | Zhenwu 810E PPU int8 GEMV and tests |

`default/runqgpu_impl.cuh` is the internal implementation shared by the three int8 entry points. Each `runqgpu.cu` selects its kernel at compile time; there is no `-k` option.

## RTX 4080 SUPER

```bash
# CUDA 12.8; compile-verified for sm_89
nvcc -O3 -std=c++20 -ccbin g++-13 -arch=sm_89 \
  -o /tmp/rungpu gpu/default/rungpu.cu -lcublas
nvcc -O3 -std=c++20 -ccbin g++-13 -arch=sm_89 \
  -o /tmp/runqgpu-default gpu/default/runqgpu.cu
nvcc -O3 -std=c++20 -ccbin g++-13 -arch=sm_89 \
  -o /tmp/runqgpu-4080s gpu/4080s/runqgpu.cu

/tmp/runqgpu-4080s models/stories15M-q32.bin \
  -t 0.0 -n 256 -s 42 -i "Once upon a time"
```

Measured with `-t 0.0 -n 256 -s 42 -i "Once upon a time"` and quantization group size 32.

| Model | FP32 | int8 | RTX 4080s optimized int8 |
| --- | ---: | ---: | ---: |
| stories15M | ~2660 tok/s | ~2200 tok/s | ~3060 tok/s |
| stories42M | ~1490 tok/s | ~1510 tok/s | ~2100 tok/s |

The 4080s implementation uses a fused warp-per-row GEMV with vectorized `float4`/`int8x4` loads, segmented warp reductions, on-the-fly activation quantization, and `__dp4a`, avoiding a separate quantization launch.

Run each of the three variants three times with `./gpu/4080s/bench.sh`.

## Zhenwu 810E PPU

PPU SDK v2.0 provides a CUDA-compatible toolchain. Do not pass `-arch=sm_89`; let the compiler emit a native PPU target. When the repository is on ossfs, write executables to `/tmp`.

```bash
nvcc -O3 -std=c++20 -o /tmp/rungpu gpu/default/rungpu.cu -lcublas
nvcc -O3 -std=c++20 -o /tmp/runqgpu-default gpu/default/runqgpu.cu
nvcc -O3 -std=c++20 -o /tmp/runqgpu-ppu gpu/ppu/runqgpu.cu

/tmp/runqgpu-ppu models/stories15M-q32.bin \
  -t 0.0 -n 256 -s 42 -i "Once upon a time"
```

| Model | FP32 | int8 | PPU optimized int8 |
| --- | ---: | ---: | ---: |
| stories15M | ~1660 tok/s | ~1640 tok/s | ~2317 tok/s |
| stories42M | ~1136 tok/s | ~944 tok/s | ~1078 tok/s |

The PPU implementation addresses the original fused kernel's weak points on the 810E: each warp handles four 32-element quantization groups, activations are quantized once and reused, Q/K/V share one launch, W1/W3 share one launch, and GEMV uses `int8x4 + __dp4a`. It currently requires a checkpoint with `GS=32` and rejects other group sizes explicitly.

The complete test first compares results element-by-element against a CPU reference, runs a QKV microbenchmark, then performs three end-to-end runs of FP32, ordinary int8, and PPU-optimized int8 on both models:

```bash
./gpu/ppu/bench.sh
```

Measured on a Zhenwu 810E (driver 1.3.2-d7f5a2), the three PPU-optimized int8 runs achieved `2316.67 / 2316.67 / 2316.67 tok/s` for stories15M and `1079.30 / 1047.01 / 1108.60 tok/s` for stories42M; the table reports their means. The kernel correctness test passed, with a maximum absolute error of `4.76837e-06` against the CPU reference for the Q/K/V/W1/W3 projections. The QKV microbenchmark dropped from `18.53 us` to `3.95 us`, a `4.69x` speedup.

## Tutorial

[`gpu_tutorial/`](../gpu_tutorial/README.md) explains the CUDA port in eight modules. Its cuBLAS FP32 path corresponds to `default/rungpu.cu` here.
