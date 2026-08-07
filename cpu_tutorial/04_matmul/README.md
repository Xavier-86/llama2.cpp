# 04 matmul: matrix-vector multiply <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project home](../../README.md)

## Overall task

Fill in the single TODO in `main.cpp` — the `matmul(xout, x, w)` kernel — implementing `W (d,n) @ x (n,) -> xout (d,)`. The model spends 99% of its time inside this one function.

```
xout_i = sum_j  W[i*n + j] * x[j]        i = 0..d-1
```

Standalone module: no model weights needed, no files to parse.

**Inputs**: const variables in main.cpp:

| Variable | Location | Shape | Layout | Meaning | Where it comes from in the model |
| --- | --- | --- | --- | --- | --- |
| `kW` | main.cpp | (3, 4), 12 values | row-major: `kW[i*4 + j]` is the weight connecting input j to output i | weight matrix | toy example; in the real model this is a weight tensor like `wq`/`wk`/`wv`/`wo`/`w1`/`w2`/`w3`/`wcls` (values identical to `data/input_w.txt`) |
| `kX` | main.cpp | (4,) | contiguous 1-D | input activation vector | toy example; in the real model this is a normalized activation like `xb` (values identical to `data/input_x.txt`) |

Constants: `kD = 3` (number of output rows), `kN = 4` (input size). This module has no data.h — 16 numbers fit comfortably inline.

**Outputs**: `main()` is already written — it calls your kernel on kW/kX and writes `out.txt`. `data/expected_out.txt` is golden data — do not modify it.

## Subtask 1: `matmul(xout, x, w)`

Apply the formula above, writing into `xout`. Here `n = x.size()`, `d = xout.size()`, and `w` holds d×n elements. Accumulate in `float` (not `double`) or the golden data won't match.

Sanity check by hand: row 0 = `0.5*0.5 + (-1.25)*(-1.0) + 2.0*2.0 + 0.75*0.25 = 5.6875`.

Background you need — where this kernel is used and why the signature looks like this. Every linear projection in a forward pass reduces to the same matrix-vector multiply:

- attention: `wq`, `wk`, `wv`, `wo`
- FFN: `w1`, `w2`, `w3`
- output layer: the classifier `wcls` (weights shared with the embedding)

llama2.c fixes the signature as `matmul(out, x, w)`: **output first, then the input vector, weight matrix last**. Each call projects the current activation `x` (length n) through the weight matrix `w` (shape (d,n), row-major) into the output `out` (length d). Why do weight rows correspond to output dims and not the other way around? Every weight tensor in the checkpoint is stored as (out_dim, in_dim): row i holds all the weights that produce output component i — and during the dot product that row is contiguous in memory, which is optimal for caches and bandwidth.

Background you need — the memory layout of W. W is stored row-major: d rows of n floats each. Row i of W dotted with x gives `xout_i`. In this module's toy example d=3, n=4; in the real model the shapes are (288, 288), (768, 288), (288, 768), (32000, 288) and so on.

## Build / run / verify

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out.txt data/expected_out.txt
```

## Common pitfalls

- Outputs in the wrong order → you swapped rows and columns: `w[i*n + j]` is row i, column j, and row i produces `xout[i]`.
- Systematic deviation from the expected data → the accumulator used `double`; use `float` to match the reference.
- Correctness is only the entry ticket — **performance** is the point: every weight element participates in exactly one multiply-add, so the work is proportional to bytes read — textbook memory-bandwidth-bound. Compare `-O2` vs `-O0`, or try multithreading the outer loop; module 12's int8 version cuts bytes read to 1/4 — that is the entire secret of faster decode.

## Done when

The `compare.py` comparison PASSes. `solution.cpp` is the reference answer — peek if stuck, then close it and write your own.
