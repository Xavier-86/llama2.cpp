# 08 transformer layer: assemble ONE layer (FP32) <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project home](../../README.md)

## Overall task

Fill in the TODOs of `main.cpp` to assemble the kernels from modules 03-07 into **one** transformer layer, using only the layer-0 weights — the repeating block that makes up the whole model. One `forward_layer0` call processes **one** token at **one** position pos through the embedding lookup and layer 0; a prompt of P tokens means P sequential calls (the prefill phase), and every position reads the k/v of all previous positions from the KV cache — that is exactly why the cache exists:

```
"Once upon a time" -> tokens {1, 9038, 2501, 263, 931} -> forward_layer0 x 5 (pos 0..4)
   per position: embedding lookup -> rmsnorm -> qkv -> cache k/v + rope -> attention
                 -> wo + residual -> rmsnorm -> ffn + residual
```

This module is the first half of what used to be a single "forward" module — the second half (all 6 layers + final norm + classifier) is module 09_forward. Splitting it in two keeps the debugging surface small: if your single layer matches the golden activations here, module 09 is just repetition (6 layers) plus a head (final rmsnorm + classifier).

Checkpoint parsing is already done by `tut::load_checkpoint` in `../common/checkpoint.h` (it is the packaged answer of module 01), and output files are written with `../common/io.h`. Only the algorithms remain in this module's code.

**Inputs**: the prompt is a const array in main.cpp; the model weights are loaded by `tut::load_checkpoint("../../models/stories15M.bin")`, which returns `tut::Checkpoint{config, weights, buffer}` — the 11 weight tensors are `std::span` views into the buffer, zero-copy. This module uses only the embedding table and the **layer-0 slice** of each per-layer tensor (all layer-major, so layer 0 is the first block, at offset 0 of each tensor):

| Variable | Location | Shape | Layout | Meaning | Where in the model |
| --- | --- | --- | --- | --- | --- |
| `kTokens` | main.cpp | (5,) | int array | token ids of the prompt "Once upon a time": `{1, 9038, 2501, 263, 931}`; 1 is the `<s>` start token | tokenizer output (same as data/input_tokens.txt) |
| `ckpt.config` | checkpoint.h | 7 × int32 | see below | model hyperparameters | stories15M.bin header |
| `w.token_embedding_table` row `token` | checkpoint.h | (288,) | row `token` of (32000, 288), at offset `token * 288` | embedding lookup: the layer input x | 1st tensor in the weight region |
| `w.rms_att_weight[0]` | checkpoint.h | (288,) | offset 0 of (6, 288) | layer-0 RMSNorm weight before attention | 2nd tensor, layer 0 |
| `w.wq[0]` / `w.wo[0]` | checkpoint.h | (288, 288) | offset 0 of (6, 288, 288), row-major inside | layer-0 q projection / attention output projection | 3rd / 6th tensor, layer 0 |
| `w.wk[0]` / `w.wv[0]` | checkpoint.h | (288, 288) | offset 0 of (6, 288, 288) | layer-0 k / v projection (kv_dim = dim for this model) | 4th / 5th tensor, layer 0 |
| `w.rms_ffn_weight[0]` | checkpoint.h | (288,) | offset 0 of (6, 288) | layer-0 RMSNorm weight before the FFN | 7th tensor, layer 0 |
| `w.w1[0]` / `w.w3[0]` | checkpoint.h | (768, 288) | offset 0 of (6, 768, 288) | layer-0 FFN up-projections (gate / linear branch) | 8th / 10th tensor, layer 0 |
| `w.w2[0]` | checkpoint.h | (288, 768) | offset 0 of (6, 288, 768) | layer-0 FFN down-projection | 9th tensor, layer 0 |

Model constants (stories15M): `dim=288, hidden_dim=768, n_layers=6, n_heads=6, n_kv_heads=6, vocab_size=32000, seq_len=256, head_size=48, kv_dim=288`. `rms_final_weight` and `wcls` exist in the checkpoint but are **not used** in this module — they belong to module 09's head.

**Outputs**: `main()` is already written — it calls `forward_layer0` for each of the 5 prompt tokens (pos = 0..4), collects both recordings of every position, and writes two output files (into this module folder), both **position-major** (pos 0's 288 values, then pos 1's, ... — 5×288 lines each):

| File | Content |
| --- | --- |
| `out_att_residual.txt` | the activation stream x of layer 0 right **after the attention residual add** (`x += Wo[0] @ attention(...)`), for all 5 positions |
| `out_layer_out.txt` | the activation stream x of layer 0 **after the FFN residual add** — i.e. the output of the whole layer, for all 5 positions |

`data/expected_*` is golden data — do not modify it. Two checkpoints instead of one lets you tell the attention half and the FFN half apart when something is off: if `out_att_residual.txt` already mismatches, the bug is in steps b)-f); if only `out_layer_out.txt` mismatches, look at steps g)-h).

## Subtask 1: copy your kernels `TODO(module 03..07)`

Copy your kernels from modules 03-07 verbatim into their stubs (each marked `TODO(module 0N)`): `rmsnorm`, `softmax`, `matmul`, `rope`, `attention`, `ffn` — no new work, just keep the signatures.

## Subtask 2: task 1 — embedding lookup

`TODO(task 1)` in `forward_layer0()`: copy row `token` of `w.token_embedding_table` (vocab_size × dim, row-major) into `s.x`. Row `token` sits at offset `token * dim` of the table — see the inputs table above.

## Subtask 3: task 2 — one transformer layer

`TODO(task 2)` in `forward_layer0()`: follow steps a)-h) of the data flow below. Only layer 0 runs, so every weight slice starts at offset 0 of its tensor (`wq`: `dim*dim` floats; `wk`/`wv`: `dim*kv_dim`; `w1`/`w2`/`w3`: `dim*hidden`; rmsnorm weights: `dim`), and the cache spans are the whole `key_cache`/`value_cache`, with position pos's rows at `pos * kv_dim`. Write k/v straight into the cache rows — **cache first, then rope**. Both residuals are `x[i] += ...`.

The two `std::ranges::copy` lines interleaved with the task are **given**: they just record `s.x` for the golden-data files — the first must run after your step f) (attention residual add), the second after step h) (ffn residual add).

Background you need — the data flow of one layer:

```
x = embedding[token]                                  # table lookup (task 1)
xb = rmsnorm(x, rms_att_weight[0])                    # (03) pre-norm
q = Wq[0] @ xb;  k = Wk[0] @ xb;  v = Wv[0] @ xb      # (04) qkv projections
k_cache[pos] = k;  v_cache[pos] = v                   # write cache first, then rope
rope(q, k_cache[pos], pos)                            # rotate in place (05)
xb = attention(q, k_cache, v_cache, pos)              # (06)
x += Wo[0] @ xb                                       # wo projection + residual add
                                                      #   -> recorded as att_residual
xb = rmsnorm(x, rms_ffn_weight[0])                    # (03) pre-norm
x += ffn(xb, W1[0], W2[0], W3[0])                     # (07) + residual add
                                                      #   -> recorded as layer_out
```

Key points:

- **pre-norm + residual**: the normalized output feeds the branch; the original x is kept for the residual add — two blocks per layer, two adds, both `x += branch`, never assignment.
- **KV cache**: shape (seq_len, kv_dim) for this single layer, one row per position; attention at position pos reads only cache rows 0..pos (that is the causal mask).
- **cache first, then rope**: the k projection is written straight into the cache row and rotated in place — the cache stores the rotated k.

Background you need — `RunState`, the activation and cache buffers (given in main.cpp — do not modify). It is allocated once from the config and reused across all positions. It is trimmed versus module 09: no `logits` (no classifier head here), and the caches hold one layer only:

| Member | Size | Role |
| --- | --- | --- |
| `x` | (288,) | current activation stream: copy of the embedding row; both residual adds land on it |
| `xb` | (288,) | branch buffer: rmsnorm output → qkv input; then reused as attention output and ffn output |
| `xb2` | (288,) | second branch buffer: wo projection result, then added into x |
| `q` | (288,) | query of the current position (after rope), 6 heads concatenated |
| `hb` / `hb2` | (768,) | the two FFN up-projection results (gate / linear branch); after SwiGLU, hb feeds the down-projection |
| `att` | (6×256,) | per-head attention scores; head h uses the slice `[h*seq_len, h*seq_len+pos]` |
| `key_cache` / `value_cache` | (256, 288) | k/v rows per position for this single layer; position pos at offset `pos * kv_dim` |

## Build / run / verify

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_att_residual.txt data/expected_att_residual.txt
python3 ../tools/compare.py out_layer_out.txt data/expected_layer_out.txt
```

## Common pitfalls

- All-zero outputs → some kernel is still a stub.
- `out_att_residual.txt` matches but `out_layer_out.txt` does not → the bug is in the FFN half: wrong `w1`/`w2`/`w3` slices, silu applied to the wrong buffer, or the second residual written as `x = branch` instead of `x += branch`.
- Errors only at later positions (worse as pos grows) → wrong KV cache read/write offsets, or attention reading beyond rows 0..pos.
- Position 0 already wrong in `out_att_residual.txt` → check the embedding row index (`token * dim`), the rmsnorm weight slice, and the q/k/v projections.
- Accidentally normalizing in place (`rmsnorm(s.x, s.x, ...)`) before the qkv projections destroys the residual stream — the pre-norm output must go to `s.xb`.
- Stepwise debugging: go back to the module 05/06/07 golden data and re-verify each block.

## Done when

Both comparisons PASS (`atol=0.001, rtol=0.001`). `solution.cpp` is the reference answer — peek if stuck, then close it and write your own. Module 09_forward then repeats this layer 6 times and adds the final norm + classifier head.
