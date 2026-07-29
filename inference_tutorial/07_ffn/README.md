# 07 FFN: SwiGLU feed-forward network <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project home](../../README.md)

> Goal: implement the second big block of each Transformer layer. Attention mixes information **across positions**; the FFN processes **each position independently** — and holds most of the model's parameters.

## Background

Inside every Transformer layer, the FFN follows attention:

```
x → RMSNorm(rms_ffn_weight) → FFN(SwiGLU) → residual add
```

This module covers only the FFN part: the input `kX` is the real forward-pass vector at layer 0's last prompt position (pos=4; the prompt is "Once upon a time", P=5 tokens) after normalization by `rms_ffn_weight`, and the output is the value before the residual add.

## The math

**SwiGLU** is two up-projections, a gated activation, and a down-projection:

```
h1  = W1 @ x          # (hidden_dim, dim) @ (dim,) -> (hidden_dim,), project up
h3  = W3 @ x          # same shape, second branch
h   = silu(h1) ⊙ h3   # elementwise; silu(v) = v * sigmoid(v) = v / (1 + e^(-v))
out = W2 @ h          # (dim, hidden_dim) @ (hidden_dim,) -> (dim,), project down
```

One up-projection is passed through the SiLU activation and multiplied elementwise with the other (the gate), then projected back down.

## Input data

There are only two kinds of input: the const array `kX` in `data.h`, and weights sliced from the checkpoint. No file parsing happens in this module.

| Variable | Location | Shape | Layout | Meaning | Where it comes from |
| --- | --- | --- | --- | --- | --- |
| `kX` | data.h | (288,) | 1-D float array | FFN input: x at the last prompt position after `rms_ffn_weight` normalization | layer 0 FFN entry (mirror of `data/input_x.txt`) |
| `ckpt.weights.w1` | `tut::load_checkpoint("../../stories15M.bin")` | (6, 768, 288) = (n_layers, hidden_dim, dim) | row-major, hidden_dim rows of dim per layer | up-projection weights (gate branch) | `w1` tensor in the checkpoint weight region; layer-l slice starts at `l * hidden_dim * dim` |
| `ckpt.weights.w3` | same | (6, 768, 288) | row-major, same as w1 | up-projection weights (linear branch) | `w3` tensor in the checkpoint; layer-l slice starts at `l * hidden_dim * dim` |
| `ckpt.weights.w2` | same | (6, 288, 768) = (n_layers, dim, hidden_dim) | row-major, dim rows of hidden_dim per layer | down-projection weights | `w2` tensor in the checkpoint; layer-l slice starts at `l * dim * hidden_dim` |

Only layer 0 is used (l = 0, offset 0). The binary parsing lives in `../common/checkpoint.h`: `tut::load_checkpoint()` returns `Weights` in which every tensor is a `std::span` view into one buffer, layers packed back to back — take a layer with `subspan`. The normalization weight `rms_ffn_weight` (layer l starts at `l * dim`) is not used directly here — `kX` is already normalized.

Model constants (stories15M): `dim = 288`, `hidden_dim = 768`, `n_layers = 6`, `n_heads = 6`, `n_kv_heads = 6`, `vocab_size = 32000`, `seq_len = 256`, `head_size = 48`, `kv_dim = 288`. In the code, `dim`/`hidden` are read from the checkpoint header, not hardcoded.

`data.h` is generated from `data/*.txt` by `../tools/embed_data.py`, with identical values; **do not edit it by hand** — regenerate with `python3 ../tools/embed_data.py ..` after re-running the dump tools.

## Tasks

Fill in the 4 `TODO`s in `main.cpp` (matmul is given — module 04 builds it from scratch):

1. **task 1 — slice the weights**: take layer 0's W1/W2/W3 out of `ckpt.weights.w1/w2/w3` (`subspan`, offsets in the table above).
2. **task 2 — up-projections**: compute `h1 = W1 @ x` and `h3 = W3 @ x` with `matmul()`; `x` is simply `kX`.
3. **task 3 — SwiGLU gate**: elementwise `h = silu(h1) * h3`, leaving the result in `h1` (use the float overload of `exp`); write `out_h.txt`.
4. **task 4 — down-projection**: `out = W2 @ h` (the value before the residual add); write `out.txt`.

The skeleton of `main()` is already written: it loads the checkpoint, prepares the buffers, and writes the output files.

## Build / run / verify

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_h.txt data/expected_hidden.txt
python3 ../tools/compare.py out.txt data/expected_out.txt
```

## Common pitfalls

- Check `out_h.txt` before `out.txt`: if hidden is wrong, suspect the W1/W3 slices or silu; if hidden is right but out is wrong, it's the W2 slice.
- Flipped matrix orientation: W1 is (hidden_dim, dim) — hidden_dim rows of dim elements. `matmul` derives dimensions from the buffer sizes (`xout.size()` = rows, `x.size()` = columns), so with correct slice lengths it cannot be flipped.
- Wrong slice offsets: `w1`/`w3` hold `hidden_dim * dim` elements per layer, `w2` holds `dim * hidden_dim`; `rms_ffn_weight` has `dim` per layer — don't mix them up.
- `silu` computed with the `double` overload of `exp`, or accumulation in `double` → the golden data won't match; stay in float throughout.

## Done when

Both comparisons PASS. `solution.cpp` is the reference answer — peek if stuck, then close it and write your own.
