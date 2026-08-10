# Quantized Models <span style="float: right;"><a href="quantization_zh.md">中文</a></span>

Convert an FP32 checkpoint to int8 (the group size must divide every weight dimension; 32 works for all stories models):

```bash
./cpu/quantize models/stories15M.bin models/stories15M-q32.bin 32
./cpu/runqcpp models/stories15M-q32.bin -t 0.0 -n 256 -i "Once upon a time"
```

Measured on Apple Silicon: 58 MB → 16 MB, tok/s roughly 138 → 1343. Cutting weight reads to 1/4 is the main reason decode gets faster — a direct demonstration that decode is memory-bandwidth bound.

The same FP32 vs int8 comparison on GPU (RTX 4080 SUPER) is recorded in [`gpu/README.md`](../gpu/README.md): with a fused warp-per-row int8 kernel, decode is faster than cuBLAS FP32 there too — but only after the kernel was written to actually exploit the 4× smaller weight reads (a naive int8 kernel was slower than FP32).

Note: the int8 quantized path is very sensitive to the compiler's optimization-level floating-point codegen (FMA contraction, vectorized reduction order) — a 1-ulp difference in an activation can flip an int8 rounding, which in turn can send greedy decoding down a different (equally valid) text at some near-tie argmax. So the exact text runq generates may differ across compile options; this is expected. FP32 run is not affected.
