# 05 RoPE: rotary position embedding <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

## Overall task

Fill in the four TODOs of `rope(q, k, pos)` in `main.cpp`: rotate the q/k slices of **one position**, in place.

The q / k projections from module 04 are pure "content" vectors with no position information — shuffle the sentence and the attention scores wouldn't change. RoPE (Rotary Position Embedding) applies a **position-dependent rotation** to q and k right before attention, so the q·k dot product of two tokens depends only on their **relative distance** — exactly the position signal attention needs.

```
embedding → RMSNorm → q/k/v projection (module 04) → [RoPE (this module)] → k into the key cache → attention (module 06)
```

**Inputs**: const arrays in data.h — no files to parse. The data comes from a real forward pass: stories15M, layer 0, prompt `"Once upon a time"` → token ids `[1, 9038, 2501, 263, 931]`, i.e. P = 5 positions.

| Variable | Location | Shape | Layout | Meaning | Where it comes from |
| --- | --- | --- | --- | --- | --- |
| `kQ` | data.h | (P, dim) = (5, 288) | pos-major | q projections before RoPE | layer-0 `x @ wq`: the queries of the 5 prompt positions |
| `kK` | data.h | (P, kv_dim) = (5, 288) | pos-major | k projections before RoPE | layer-0 `x @ wk`: the keys of the 5 prompt positions |

Model constants: `dim = 288`, `kv_dim = 288`, `n_heads = n_kv_heads = 6`, `head_size = 48`, `P = 5` (`kDim`, `kKvDim`, `kHeadSize`, `kPositions` in the code). data.h is generated from `data/*.txt` by `../tools/embed_data.py` — the values are identical; do not edit it by hand.

**Outputs**: `main()` is already written — it copies the inputs out of data.h, calls `rope` for each position pos = 0..4, and writes `out_q.txt` / `out_k.txt`. `data/expected_*` is golden data — do not modify it.

## Subtask 1: loop over the adjacent pairs

Loop over the adjacent pairs with stride 2: `i = 0, 2, 4, ... dim-2`.

Background you need — the memory layout. Both arrays are **position-major** — position pos occupies `[pos*dim, (pos+1)*dim)`; within a position the layout is **head-major** — head h occupies `[h*head_size, (h+1)*head_size)` (head_size = 48, 6 heads = 288 dims). So index i belongs to head `i / head_size`, with in-head pair index `i % head_size`.

## Subtask 2: compute `head_dim`, `freq` and `angle`

For each pair, compute `head_dim = i % head_size`, `freq = 1 / 10000^(head_dim / head_size)`, `angle = pos * freq`, using the float versions of `std::cos` / `std::sin`.

Background you need — the math. Treat each adjacent pair `(v0, v1)` within a head as a 2D vector (equivalently, a complex number) and rotate it by `pos * freq`:

```
for i = 0, 2, 4, ... dim-2:
    head_dim = i % head_size                     # pair index within its own head
    freq     = 1 / 10000^(head_dim / head_size)  # each dim pair gets its own frequency
    angle    = pos * freq                        # rotation grows linearly with position
    (v0', v1') = (v0*cos(angle) - v1*sin(angle),  v0*sin(angle) + v1*cos(angle))
```

**Frequencies are computed from the in-head index**, not the global index: the first 24 pairs of every head (head_size = 48) share one set of frequencies from 1 down to ~1e-4. Low frequencies rotate slowly (long range), high ones fast (local).

## Subtask 3: rotate the q pair in place

Rotate the q pair in place: `(v0', v1') = (v0*cos - v1*sin, v0*sin + v1*cos)`.

Background you need — **rotate in place**: the rotated k is written back and later stored into the key cache as-is — **the cache holds post-rotation k**.

## Subtask 4: rotate the k pair, guarded by `i < kv_dim`

Rotate the same k pair the same way, but only while `i < kv_dim`.

Background you need — **rotate all `dim` values of q, but only the first `kv_dim` values of k** (the second rotation is guarded by `if (i < kv_dim)`). Why: q has n_heads × head_size = dim values, while k only has n_kv_heads × head_size = kv_dim values; in GQA models n_kv_heads < n_heads so kv_dim < dim and going past it reads out of bounds. In this model kv_dim == dim == 288, so k is fully rotated too, but the bound check must stay.

## Build / run / verify

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_q.txt data/expected_q.txt
python3 ../tools/compare.py out_k.txt data/expected_k.txt
```

## Common pitfalls

- All frequencies come out wrong → you forgot `head_dim = i % head_size` and used the global index i directly.
- Systematic small deviation → `std::cos` / `std::sin` took the double overload (keep angle, freq and the trig calls in float end to end).
- pos=0 is a free self-check: every angle is 0 there, so q/k must pass through unchanged; if your pos=0 data changed, you're writing out of bounds or copied the inputs wrong.
- Looping k up to dim without the `i < kv_dim` guard → passes here (kv_dim == dim) but reads out of bounds on a GQA model.

## Done when

Both comparisons PASS. Hand-off to module 06: this module's `out_k.txt` (post-rotation k) is exactly what gets stored in the key cache, and `out_q.txt` is dotted against the cached k. `solution.cpp` is the reference answer — peek if stuck, then close it and write your own.
