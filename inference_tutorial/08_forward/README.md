# 08 forward: assemble the full single-step pass (FP32) <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md)

> Goal: assemble modules 01-07 into a complete `forward(token, pos) -> logits`.
> This is the FP32 grand assembly — the first milestone acceptance.

## Full pipeline (map each line to one of your earlier modules)

```
x = embedding[token]                          # table lookup (01)
for l in 0..n_layers-1:
    xb = rmsnorm(x, rms_att_weight[l])        # (03)
    q = Wq[l] @ xb;  k = Wk[l] @ xb;  v = Wv[l] @ xb   # (04)
    k_cache[l][pos] = k;  v_cache[l][pos] = v # write cache first
    rope(q, k_cache[l][pos], pos)             # rotate in place (05)
    xb = attention(q, k_cache[l], v_cache[l], pos)     # (06)
    x += Wo[l] @ xb                           # projection + residual
    xb = rmsnorm(x, rms_ffn_weight[l])
    x += ffn(xb, W1[l], W2[l], W3[l])         # (07) + residual
x = rmsnorm(x, rms_final_weight)
logits = Wcls @ x                             # shared embedding table
```

Key points:

- **pre-norm + residual**: the normalized output feeds the branch; the original
  x is kept for the residual add — once per block, two adds per layer
- KV cache shape (n_layers, seq_len, kv_dim): one row per layer per position
- One call processes one token; a prompt of N tokens means N sequential calls
  (that is the prefill phase)

## Data files

| File | Content |
| --- | --- |
| `input_tokens.txt` | the prompt's 5 token ids |
| `expected_logits.txt` | full logits per position, 5 x 32000, position-major |
| `expected_argmax.txt` | argmax of each position's logits, 5 integers |

## Tasks and verification

Call your forward for each pos in order, collect the logits:

```bash
python3 ../tools/compare.py out_argmax.txt data/expected_argmax.txt --exact
python3 ../tools/compare.py out_logits.txt data/expected_logits.txt
```

**Compare argmax first**: if it matches, the information flow is basically
right. Logit diffs up to ~1e-3 are normal (summation order); beyond 1e-2 means
a real bug.

## Debugging route (when logits don't match, work top-down)

1. Temporarily run only 1 layer — quickly tells whether some layer indexing is off
2. Go back to the module 05/06/07 golden data and re-verify each block
3. Check layer indexing: wq layer l at offset `l * dim * dim`; cache layer l at
   `l * seq_len * kv_dim`
4. Check residuals: `x += branch_output`, not `x = branch_output`

Once this passes, you have rebuilt `run.cpp`'s `Transformer::forward`.
