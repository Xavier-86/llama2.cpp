# gpu_tutorial: Step-by-step GPU inference with cuBLAS and hand-written kernels <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← Project README](../README.md)

> Goal: port the FP32 inference path in `cpu/run.cpp` to the GPU. Use cuBLAS for
> matrix multiplication and hand-written CUDA kernels for RMSNorm, RoPE, softmax,
> attention, and SwiGLU. The final result is the single-file `gpu/rungpu.cu`.
>
> Acceptance criterion: with the same greedy-decoding options `-t 0.0 -s 42`,
> the output must match `./cpu/runcpp` exactly.

## Relationship to cpu_tutorial

[`cpu_tutorial/`](../cpu_tutorial/README.md) explains inference itself—what each
operator computes. This tutorial assumes that you understand
`Transformer::forward` in `cpu/run.cpp` (or have completed cpu_tutorial module
09) and focuses on moving those operators to the GPU correctly. Operator semantics
are not repeated here; `cpu/run.cpp` is the source of truth.

## Prerequisites

```bash
sudo apt install nvidia-cuda-toolkit   # Provides nvcc and cuBLAS
nvcc --version                         # Confirm the installation
```

The same build command is used throughout. Adjust `-arch` for your GPU (RTX 40
series uses `sm_89`); if omitted, nvcc detects the target automatically.

```bash
nvcc -O3 -std=c++20 -arch=sm_89 -o gpu/rungpu gpu/rungpu.cu -lcublas
```

## Overall design

### Host/device responsibilities

| Keep on the CPU (copied from `cpu/run.cpp`) | Move to the GPU (this tutorial) |
| --- | --- |
| Checkpoint loading (`MappedFile`) | Weight uploads with `cudaMemcpy` |
| BPE tokenizer | Every operator in forward |
| Sampler (argmax / top-p sampling) | KV cache (resident in device memory) |
| Generate/chat loops | |

### Memory strategy

- **Weights:** allocate and upload once with `cudaMalloc` + `cudaMemcpy`, then
  leave them on the device (stories15M is only 60 MB and 42M is about 210 MB)
- **Activation buffers** (x / xb / xb2 / q / hb / hb2 / att): one persistent
  device allocation for each `RunState` member
- **KV cache:** resident in device memory
  (`n_layers × seq_len × kv_dim × 2 × 4` bytes, about 6 MB for the 15M model)
- **Logits:** copy 32,000 floats (128 KB) back to the CPU for sampling each step;
  this is the only device-to-host transfer per step and its cost is negligible

### Data flow for decoding one token (per layer)

```
x ──rmsnorm──▶ xb ──GEMV──▶ q ──┐
                   ├──GEMV──▶ k ──rope──▶ write to KV cache
                   └──GEMV──▶ v ────────▶ write to KV cache
q + KV cache ──attention kernel──▶ xb ──GEMV(wo)──▶ xb2 ──residual add──▶ x
x ──rmsnorm──▶ xb ──GEMV(w1)──▶ hb ──┐
                   └──GEMV(w3)──▶ hb2 ──swiglu kernel──▶ hb ──GEMV(w2)──▶ xb ──residual add──▶ x
```

After the layer loop: final RMSNorm → GEMV (classifier head) → copy logits to CPU.

## Modules

| Module | Topic | Acceptance criterion |
| --- | --- | --- |
| [00_setup](00_setup/README.md) | Project skeleton: error-checking macros, weight upload, cuBLAS handle | Builds successfully; uploaded byte counts match the checkpoint |
| [01_cublas_matmul](01_cublas_matmul/README.md) | cuBLAS GEMV and the row-major transpose trick | Error against CPU matmul < 1e-4 |
| [02_rmsnorm](02_rmsnorm/README.md) | Block-reduction kernel | Error against CPU < 1e-5 |
| [03_rope](03_rope/README.md) | Pairwise rotation kernel | Error against CPU < 1e-5 |
| [04_attention](04_attention/README.md) | Attention kernel + KV cache (including GQA boundaries) | Error against CPU < 1e-4 |
| [05_elementwise](05_elementwise/README.md) | SwiGLU / residual add / embedding lookup | Bitwise agreement or error around 1e-6 |
| [06_forward](06_forward/README.md) | Assemble forward and align with CPU | `diff` produces no output |
| [07_benchmark](07_benchmark/README.md) | Benchmarking, bandwidth ceiling, optimization roadmap | Report tok/s and explain the gap from the theoretical ceiling |

Use the same method as cpu_tutorial: **compare every kernel's intermediate output
with the CPU implementation as soon as you write it; do not wait until the full
forward pass is assembled.** Each module has `main.cu` (exercise template with a
complete harness and TODO kernel stub), `solution.cu` (reference implementation),
and `cases.h` (test cases and CPU reference shared by both). Golden data lives in
`data/expected_*.txt`; compare it with `cpu_tutorial/tools/compare.py`.

The CPU is the source of truth for golden data. `tools/gen_data.cpp` is a pure-CPU
program that includes every module's `cases.h` and generates expected files for
modules 00–05. Regenerate them after changing a test case:

```bash
c++ -O2 -std=c++20 -o gpu_tutorial/tools/gen_data gpu_tutorial/tools/gen_data.cpp   # Use g++-12 if c++ is too old
./gpu_tutorial/tools/gen_data gpu_tutorial models/stories15M.bin                    # Run from repository root
```

Module 06's `expected_gen.txt` is greedy-decoding output from `./cpu/runcpp`;
see that module's README. Module 07 contains documentation only.

## Common pitfalls

| Symptom | Likely cause |
| --- | --- |
| Matmul output is globally shifted or nonsensical | Incorrect `lda`, or OP_T/OP_N reversed (module 01) |
| `cublasSgemv` returns INVALID_VALUE | alpha/beta are device pointers, or the handle was not created |
| Results are intermittently correct | Forgot that kernels are asynchronous and did not synchronize before a host read |
| Text diverges at a fixed position | Incorrect RoPE `kvd` boundary, attention `kv_mul`, or softmax range `[0, pos]` |
| Numerical error exceeds 1e-2 | This is a bug, not noise; compare per-operator dumps (module 06) |
| Out-of-bounds access / illegal address | Incorrect weight-slice offset; copy every offset such as `l*dim*dim` from the CPU code |

## References

- `cpu/run.cpp` — the source of truth for all operator logic
- `cpu_tutorial/` — step-by-step CPU tutorial and reusable golden data
- [karpathy/llm.c](https://github.com/karpathy/llm.c) — CUDA kernel examples
- [cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/) — parameter details
  for `cublasSgemv` and `cublasSgemm`
