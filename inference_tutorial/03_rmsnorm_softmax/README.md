# 03 rmsnorm / softmax: two small kernels <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md)

> Goal: implement the two normalization functions that appear everywhere in the
> Transformer. Each is under 10 lines.

## RMSNorm

For a vector x (length n) and a learned weight w:

```
ss = (1/n) * sum(x_i^2)      # mean square
ss = 1 / sqrt(ss + 1e-5)     # eps against division by zero
out_i = w_i * ss * x_i
```

It normalizes the vector's magnitude for numerical stability. Used before
attention, before the FFN, and once at the very end.

## Softmax (numerically stable)

```
m = max(x)
out_i = exp(x_i - m) / sum_j exp(x_j - m)
```

Subtracting the max prevents `exp(1000)` from overflowing to inf —
mathematically a no-op, numerically essential.

## Data files

| File | Content |
| --- | --- |
| `input_rmsnorm_x.txt` / `input_rmsnorm_w.txt` | 8-dim toy case: input and weight |
| `expected_rmsnorm.txt` | expected output |
| `input_rmsnorm_x_real.txt` | real 288-dim input (embedding row of the first prompt token) |
| `expected_rmsnorm_real.txt` | normalized with layer-0 rms_att_weight |
| `input_softmax.txt` / `expected_softmax.txt` | ordinary 8-dim case |
| `input_softmax_big.txt` / `expected_softmax_big.txt` | values around 1000 — targets the stability trick |

## Tasks and verification

For the real-sized case, fetch the layer-0 slice of `rms_att_weight` (offset 0,
length dim) from the checkpoint using your module-01 loader.

```bash
python3 ../tools/compare.py out.txt data/expected_rmsnorm.txt
python3 ../tools/compare.py out_real.txt data/expected_rmsnorm_real.txt
python3 ../tools/compare.py out_sm.txt data/expected_softmax.txt
python3 ../tools/compare.py out_big.txt data/expected_softmax_big.txt
```

If the `big` case comes out all zeros or NaN, you forgot to subtract the max.

## Hints

- Compute everything in `float` (including the running sum) to stay consistent
  with the golden data.
- Softmax modifies its input **in place** — attention calls it per head on
  slices of a shared buffer.
