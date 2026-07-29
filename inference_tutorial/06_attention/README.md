# 06 attention: multi-head causal attention <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

> Goal: implement the heart of the Transformer — multi-head causal attention with a KV cache. The easiest module to get subtly wrong; budget extra time.

## Background

In every layer's forward pass, attention lets the current position "look back" at all previous positions: the query is scored against the key of every past position, the scores are normalized into weights, and the values are combined with those weights. Generation is autoregressive — position pos may only see 0..pos (the **causal mask**) — so inference stores each step's K/V in a **KV cache** and later positions reuse it instead of recomputing the whole prefix.

This module gives you real data from layer 0 for the first 5 positions of the prompt "Once upon a time" (token ids `[1, 9038, 2501, 263, 931]`, P=5): q and k are already RoPE-rotated (module 05's output); v is not rotated. This module is attention only — the wo projection and residual connection come in module 08.

## The math

Split the dim=288 q into `n_heads` heads of `head_size` each. For a single position pos and a single head h:

```
qh      = q[h*head_size : (h+1)*head_size]
score_t = (qh · k_t) / sqrt(head_size)     # t = 0..pos; iterating only 0..pos IS the causal mask (t > pos is invisible)
att     = softmax(score_0..pos)            # softmax over just these pos+1 scores
out_h   = sum_t att_t * v_t                # attention-weighted sum of the value rows
```

Concatenate the 6 heads' `out_h` back into a 288-dim vector: the layer's attention output (before wo).

- Dividing by `sqrt(head_size)` keeps the dot products from growing with the dimension and saturating the softmax.
- **GQA**: query head h maps to KV head `kv_h = h / kv_mul`, where `kv_mul = n_heads / n_kv_heads`. In stories15M n_heads = n_kv_heads = 6, so kv_mul = 1 and kv_h = h (every head has its own K/V); in GQA models (n_kv_heads < n_heads) several query heads share one K/V group. Write the code for a general kv_mul — do not hardcode kv_h = h.

### KV cache layout

In real inference the cache has shape (n_layers, seq_len, kv_dim): each generated position writes its new k and v into row pos, and all later positions read rows 0..pos. This module hands you a slice of layer 0's cache — its first P=5 rows — flattened position-major:

```
kKCache[t * kv_dim + kv_h * head_size + i]   # position t, KV head kv_h, component i
```

i.e. the logical 3-level index `kKCache[pos][head][head_size]`; same for the v cache.

## Input data

All inputs are const arrays in data.h (generated from `data/*.txt` by `../tools/embed_data.py`, values identical — do not edit data.h by hand). No files to parse:

| Variable | Location | Shape | Layout | Meaning | Where it comes from in the model |
| --- | --- | --- | --- | --- | --- |
| `kQ` | data.h | (5, 288) | pos-major: position pos occupies `[pos*288, (pos+1)*288)`; within a position, head h sits at `h*48` for 48 floats | the 5 positions' queries, post-RoPE | layer 0: RMSNorm-ed xb × `wq`, then RoPE (module 05's output) |
| `kKCache` | data.h | (5, 288) | same; logically `cache[pos][kv_head][head_size]` | layer-0 key cache, first 5 rows, post-RoPE | layer 0: xb × `wk`, RoPE-ed, written into the key cache |
| `kVCache` | data.h | (5, 288) | same | layer-0 value cache, first 5 rows (V is not rotated) | layer 0: xb × `wv`, written into the value cache |

Model constants (stories15M): dim=288, hidden_dim=768, n_layers=6, n_heads=6, n_kv_heads=6, vocab_size=32000, seq_len=256, head_size=48, kv_dim=288. In the code: `kDim` / `kNumHeads` / `kHeadSize` / `kKvMul` (=1) / `kKvDim` / `kPositions` (=5).

## Tasks

Fill in the 5 `TODO`s in `main.cpp`:

1. **task 1**: loop over heads h = 0..5, compute `kv_h = h / kv_mul`, and slice this head's `qh` out of q (48 floats starting at `h * head_size`).
2. **task 2**: for t = 0..pos (history + current only — that range is the causal mask), slice head kv_h's key out of cache row t (starting at `t * kv_dim + kv_h * head_size`), compute the dot product `qh · key`, scale by `1 / sqrt(head_size)`, and store it in this head's att slice at index t.
3. **task 3**: softmax only the 0..pos segment of this head's att slice — reuse the given `softmax`, which works **in place** on the shared scratch (module 03's convention); future positions' slots must not be exp'ed in.
4. **task 4**: zero this head's slice of out, then accumulate `att_t * v_t` over the v_cache rows 0..pos, using the same row/head slicing as for K.
5. **task 5** (in `main()`): after the loop, att holds the last position's (pos=4) weights; collect them head-major (head 0's 5 weights, then head 1's, ... — 30 values total) and write them to out_att.txt.

The rest of `main()` is already written: it calls `attention` for pos = 0..4 and writes out.txt (5×288, all positions concatenated).

## Build / run / verify

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out.txt     data/expected_out.txt
python3 ../tools/compare.py out_att.txt data/expected_att_weights_lastpos.txt
```

## Common pitfalls

- **Match att_weights before the full output**: weights right but output wrong → the bug is in the weighted sum or head slicing; weights wrong → check the dot product or softmax range.
- softmax over the whole scratch (or seq_len=256) instead of the 0..pos segment → the zeros from future positions get exp'ed in.
- Forgetting `/ sqrt(head_size)`.
- Wrong in-row offset: row t, head kv_h starts at `t * kv_dim + kv_h * head_size`.
- Hardcoding kv_h = h instead of going through `kv_mul` → passes here, breaks on GQA models.

## Done when

Both comparisons PASS. `solution.cpp` is the reference answer — peek if stuck, then close it and write your own.
