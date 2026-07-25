# 07 FFN: SwiGLU feed-forward network <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md)

> Goal: implement the layer's second big block. Attention mixes information
> **across positions**; the FFN processes **each position independently** — and
> holds most of the model's parameters.

## Algorithm

```
h1 = W1 @ x          # (hidden_dim, dim) @ (dim,) -> (hidden_dim,), project up
h3 = W3 @ x          # same, second branch
h  = silu(h1) * h3   # elementwise; silu(v) = v * sigmoid(v) = v / (1 + exp(-v))
out = W2 @ h         # (dim, hidden_dim) @ (hidden_dim,) -> (dim,), project down
```

Two up-projections, one gated by the SiLU activation and multiplied
elementwise, then a down-projection: this structure is called **SwiGLU**.

Model: dim=288, hidden_dim=768. Get layer 0's W1/W2/W3 slices with your
module-01 loader (layer offset 0).

## Data files (real forward, layer 0, last position pos=4)

| File | Content |
| --- | --- |
| `input_x.txt` | FFN input (after rms_ffn normalization), 288 |
| `expected_hidden.txt` | hidden state after SwiGLU, 768 |
| `expected_out.txt` | output after W2 (before the residual add), 288 |

## Tasks and verification

```bash
python3 ../tools/compare.py out_h.txt data/expected_hidden.txt
python3 ../tools/compare.py out.txt data/expected_out.txt
```

## Hints

- Check hidden before out: if hidden is wrong, suspect the W1/W3 slices or
  silu; if hidden is right but out is wrong, it's W2.
- Use the float version of `exp`.
- Don't flip the matrix orientation: W1 is (hidden_dim, dim) — hidden_dim rows
  of dim elements. If your matmul derives dimensions from the buffer sizes
  (`xout.size()` = d, `x.size()` = n), it can't go wrong.
