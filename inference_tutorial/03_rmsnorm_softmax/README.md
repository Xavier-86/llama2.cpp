# 03 rmsnorm / softmax: two small kernels <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project home](../../README.md)

> Goal: implement the two normalization functions used all over the Transformer. Each is under 10 lines.

## Background

Both functions are called repeatedly in a single forward pass:

```
token embedding → [ RMSNorm → attention (softmax inside) → RMSNorm → FFN ] × 6 layers → RMSNorm → logits → softmax (when sampling)
```

RMSNorm normalizes vector magnitudes for numerical stability; softmax turns arbitrary scores into a probability distribution (attention weights, sampling probabilities).

## The math

### RMSNorm

For a vector x of length n and a learned weight w:

```
ss = (1/n) * sum(x_i^2)      # mean square
ss = 1 / sqrt(ss + 1e-5)     # eps guards against division by zero
out_i = w_i * ss * x_i
```

### Softmax (numerically stable)

```
m = max(x)
out_i = exp(x_i - m) / sum_j exp(x_j - m)
```

Subtracting the max does not change the math, but prevents `exp(1000)` from overflowing to inf — essential in practice.

## Input data

All inputs are const variables in the code — no files to parse:

| Variable | Location | Shape | Meaning |
| --- | --- | --- | --- |
| `kRmsXToy` / `kRmsWToy` | main.cpp | (8,) | toy example: input vector and weight |
| `kRmsNormXReal` | data.h | (288,) | real input: the embedding row of the first prompt token (`<s>`, id 1) |
| `kRmsNormWReal` | data.h | (288,) | layer-0 slice of `rms_att_weight` (the first dim floats of that tensor in the checkpoint) |
| `kSoftmaxIn` | main.cpp | (8,) | ordinary softmax input |
| `kSoftmaxBig` | main.cpp | (4,) | values around 1000, exercises the stability trick |

Model constant: `dim = 288` (`kDim` in the code). data.h is generated from `data/*.txt` by `../tools/embed_data.py`; the values are identical.

## Tasks

Fill in the two functions in `main.cpp` (each marked with `TODO`):

1. **task 1 — `rmsnorm(o, x, weight)`**: apply the formula above, writing into `o`. Accumulate in `float` (including the running sum) or the golden data won't match.
2. **task 2 — `softmax(x)`**: modify `x` **in place** (attention calls it per head on slices of a shared buffer); subtract the max before exp.

`main()` is already written: it calls your kernels on the 4 input sets and writes 4 output files.

## Build / run / verify

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out.txt      data/expected_rmsnorm.txt
python3 ../tools/compare.py out_real.txt data/expected_rmsnorm_real.txt
python3 ../tools/compare.py out_sm.txt   data/expected_softmax.txt
python3 ../tools/compare.py out_big.txt  data/expected_softmax_big.txt
```

## Common pitfalls

- `out_big.txt` comes out all zeros or NaN → you forgot to subtract the max before exp.
- Systematic deviation in `out_real.txt` → the running sum used `double`, or eps was applied in the wrong place (add eps, then take the reciprocal square root).
- softmax not in place → this module still passes, but module 06's attention relies on that convention.

## Done when

All four comparisons PASS. `solution.cpp` is the reference answer — peek if stuck, then close it and write your own.
