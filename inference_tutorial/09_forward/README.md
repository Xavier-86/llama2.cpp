# 09 forward: the full single-step pass (FP32) <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project home](../../README.md)

> Goal: stack the single transformer layer of module 08_transformer_layer 6 times and add the output head, completing `forward(token, pos) -> logits`. This is the FP32 grand assembly and the first milestone acceptance — once it passes, you have rebuilt `run.cpp`'s `Transformer::forward`. This module is the second half of what used to be a single "forward" module; the first half (one layer) is module 08_transformer_layer.

## Background

Module 08 assembled one transformer layer and verified its activations. This module answers "what does the whole model do in one inference step": the same layer block repeated `n_layers` times, followed by a final RMSNorm and the classifier matmul that produces the 32000-dim logits for the next token. A prompt of P tokens means P sequential calls (the prefill phase): every position reads the k/v of all previous positions from the KV cache — that is exactly why the cache exists.

Because the single layer is already verified, this module gives you all of it: the six kernels (modules 03-07), the single-layer assembly `transformer_layer(l, ...)` (module 08), and the embedding lookup are **given code** in main.cpp. The new work is only the stacking loop and the output head.

Checkpoint parsing is already done by `tut::load_checkpoint` in `../common/checkpoint.h` (it is the packaged answer of module 01), and output files are written with `../common/io.h`. Only the algorithms remain in this module's code.

## Data flow

```
x = embedding[token]                                  # table lookup (given)
for l in 0..n_layers-1:                               # 6 layers (task 1)
    transformer_layer(l, ...):                        # given, from module 08
        xb = rmsnorm(x, rms_att_weight[l])            # (03)
        q = Wq[l] @ xb;  k = Wk[l] @ xb;  v = Wv[l] @ xb  # (04)
        k_cache[l][pos] = k;  v_cache[l][pos] = v     # write cache first, then rope
        rope(q, k_cache[l][pos], pos)                 # rotate in place (05)
        xb = attention(q, k_cache[l], v_cache[l], pos)  # (06)
        x += Wo[l] @ xb                               # wo projection + residual add
        xb = rmsnorm(x, rms_ffn_weight[l])            # (03)
        x += ffn(xb, W1[l], W2[l], W3[l])             # (07) + residual add
x = rmsnorm(x, rms_final_weight)                      # (task 2)
logits = Wcls @ x                                     # (task 3) shared embedding table
```

Key points:

- **pre-norm + residual**: inside each layer the normalized output feeds the branch; the original x is kept for the residual add — two blocks per layer, two adds, both `x += branch`, never assignment.
- **per-layer weights and cache**: every layer has its own slices — weights at `l * per-layer size`, KV cache rows at `l * seq_len * kv_dim`. Layer l never touches another layer's weights or cache.
- **the head**: after the last layer, a final RMSNorm (with its own weight, not a per-layer one) and one more matmul against `wcls` produce the logits. For this model `wcls` **is** the embedding table (weight sharing).

## Input data

The prompt is a const array in main.cpp; the model weights are loaded by `tut::load_checkpoint("../../stories15M.bin")`, which returns `tut::Checkpoint{config, weights, buffer}` — the 11 weight tensors are `std::span` views into the buffer, zero-copy:

| Variable | Location | Shape | Layout | Meaning | Where in the model |
| --- | --- | --- | --- | --- | --- |
| `kTokens` | main.cpp | (5,) | int array | token ids of the prompt "Once upon a time": `{1, 9038, 2501, 263, 931}`; 1 is the `<s>` start token | tokenizer output (same as data/input_tokens.txt) |
| `ckpt.config` | checkpoint.h | 7 × int32 | see below | model hyperparameters | stories15M.bin header |
| `w.token_embedding_table` | checkpoint.h | (32000, 288) | row-major, one token per row | embedding lookup table; also serves as the `wcls` classifier head (weight sharing) | 1st tensor in the weight region |
| `w.rms_att_weight` | checkpoint.h | (6, 288) | layer-major | per-layer RMSNorm weight before attention | 2nd tensor |
| `w.wq` / `w.wo` | checkpoint.h | (6, 288, 288) | layer-major, row-major inside | q projection / attention output projection | 3rd / 6th tensor |
| `w.wk` / `w.wv` | checkpoint.h | (6, 288, 288) | layer-major | k / v projection (kv_dim = dim for this model) | 4th / 5th tensor |
| `w.rms_ffn_weight` | checkpoint.h | (6, 288) | layer-major | per-layer RMSNorm weight before the FFN | 7th tensor |
| `w.w1` / `w.w3` | checkpoint.h | (6, 768, 288) | layer-major | FFN up-projections (gate / linear branch) | 8th / 10th tensor |
| `w.w2` | checkpoint.h | (6, 288, 768) | layer-major | FFN down-projection | 9th tensor |
| `w.rms_final_weight` | checkpoint.h | (288,) | — | final RMSNorm weight (after the last layer) | 11th tensor |
| `w.wcls` | checkpoint.h | (32000, 288) | row-major | classifier head; shared with the embedding table for this model (the same span) | shared from the 1st tensor |

Model constants (stories15M): `dim=288, hidden_dim=768, n_layers=6, n_heads=6, n_kv_heads=6, vocab_size=32000, seq_len=256, head_size=48, kv_dim=288`. After `rms_final_weight` the weight region holds two legacy precomputed RoPE tables; `tut::load_checkpoint` already skips them — no need to care.

## RunState: activation and cache buffers

`RunState` is allocated once from the config and reused across all positions (given in main.cpp — do not modify):

| Member | Size | Role |
| --- | --- | --- |
| `x` | (288,) | current activation stream: copy of the embedding row; both residual adds of every layer land on it |
| `xb` | (288,) | branch buffer: rmsnorm output → qkv input; then reused as attention output and ffn output |
| `xb2` | (288,) | second branch buffer: wo projection result, then added into x |
| `q` | (288,) | query of the current position (after rope), 6 heads concatenated |
| `hb` / `hb2` | (768,) | the two FFN up-projection results (gate / linear branch); after SwiGLU, hb feeds the down-projection |
| `att` | (6×256,) | per-head attention scores; head h uses the slice `[h*seq_len, h*seq_len+pos]` |
| `key_cache` / `value_cache` | (6, 256, 288) | k/v rows per layer per position; layer l at offset `l * seq_len * kv_dim`, position pos at `pos * kv_dim` inside it |
| `logits` | (32000,) | the return value of `forward`: scores for the next token at this position |

## Tasks

Everything except the stacking and the head is given in main.cpp: the six kernels `rmsnorm`, `softmax`, `matmul`, `rope`, `attention`, `ffn` (you implemented them in modules 03-07), the single-layer assembly `transformer_layer(l, pos, p, w, s)` (you assembled it in module 08_transformer_layer — here it is parameterized by the layer index l, slicing weights at `l * per-layer size` and the cache at `l * seq_len * kv_dim`), the embedding lookup, and all of `main()`.

The new work of this module is the 3 tasks inside `forward()`:

1. **task 1 — stack the layers**: call `transformer_layer(l, pos, p, w, s)` for every `l` in `0 .. p.n_layers-1`, in order. The given function already handles all per-layer offsets; your job is only the loop.
2. **task 2 — final rmsnorm**: in place on `s.x`, with `w.rms_final_weight` — note this is a single (288,) weight, not a per-layer slice.
3. **task 3 — classifier head**: `matmul(s.logits, s.x, w.wcls)`; for this model `wcls` is the shared embedding table.

`main()` is already written: it calls `forward` for each of the 5 prompt tokens (pos = 0..4), collects every position's logits, and takes their argmax.

## Build / run / verify

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_argmax.txt data/expected_argmax.txt --exact
python3 ../tools/compare.py out_logits.txt data/expected_logits.txt
```

Output files (written into this module folder):

| File | Content |
| --- | --- |
| `out_logits.txt` | full logits of all 5 positions, concatenated **position-major**: 32000 values of pos 0, then pos 1, ... — 5×32000 lines in total |
| `out_argmax.txt` | argmax of each position's logits, 5 integers (the greedily decoded next tokens) |

## Common pitfalls

- **Compare argmax before logits**: matching argmax means the information flow is basically right; a ~1e-3 relative logit diff is normal (floating-point summation order), while anything beyond 1e-2 usually means a real bug.
- All-zero logits or identical argmax everywhere → a task is still a stub (most often the loop of task 1), or the classifier matmul of task 3 is missing.
- Applying the final rmsnorm with a per-layer weight (e.g. `rms_att_weight`) instead of `rms_final_weight` — the head has its own (288,) weight.
- Looping only over layer 0, or over `1..n_layers` — the offsets inside `transformer_layer` already depend on l, so the loop bounds are the only place to get this wrong.
- Errors only at later positions (worse as pos grows) → almost certainly a cache bug carried over from your module-08 code; the given `transformer_layer` here matches the module-08 reference, so diff your module 08 against it.
- Stepwise debugging: if this module fails, go back to module 08_transformer_layer's golden data and re-verify the single layer first.

## Done when

Both comparisons PASS: argmax exactly equal, logits within tolerance (`atol=0.001, rtol=0.001`). `solution.cpp` is the reference answer — peek if stuck, then close it and write your own. Next up: module 10_sampler turns these logits into sampled tokens.
