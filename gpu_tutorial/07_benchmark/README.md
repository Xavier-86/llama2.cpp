# Module 07: Benchmarking and optimization roadmap <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

## Task

First estimate the theoretical ceiling. Decoding reads all weights once per step:

```
theoretical tok/s ≈ device-memory bandwidth / weight bytes
RTX 4080: ≈ 717 GB/s / 60 MB (15M FP32) ≈ 12,000 tok/s
```

Compare this with the program's `achieved tok/s` on stderr and put it in the same
table as CPU performance (about 155 tok/s for FP32 and 460 tok/s for int8).

Then profile where time is spent:

```bash
nsys profile -o /tmp/rungpu_prof ./gpu/rungpu models/stories15M.bin -n 256 -i "Once upon a time"
nsys stats /tmp/rungpu_prof.nsys-rep   # Kernel time share and launch counts
ncu --set basic ./gpu/rungpu ...        # Per-kernel bandwidth utilization; may require permission
```

Measured throughput will be far below the ceiling: each decode step launches dozens
of kernels, while each GEMV is too small to saturate bandwidth. This module is
complete when you can explain the gap.

## Optimization menu, ordered by return on effort

1. **Reduce kernel launches:** fuse RMSNorm + matmul, combine q/k/v GEMVs by
   concatenating their weight matrices, and fuse SwiGLU with the preceding GEMV
2. **Use GEMM for prefill:** process prompt tokens in a batch with `cublasSgemm`
   instead of token-by-token GEMV to reduce time to first generated token
3. **FP16/BF16 weights:** halve weight traffic and roughly double decode throughput;
   this is the same bandwidth-bound result discussed in cpu_tutorial module 12
4. **cuBLASLt epilogue:** perform matmul + SiLU + elementwise multiplication in one
   call
5. **FlashAttention-style fusion:** combine module 04's three stages in one kernel,
   eliminating global-memory traffic for the attention score buffer

## Acceptance criteria

Write a short benchmark report containing the theoretical ceiling, measured tok/s
for GPU, CPU FP32, and CPU int8, the three most expensive kernels in nsys, and the
first optimization you would attempt with a reason.
