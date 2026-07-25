# 06 attention: multi-head causal attention

[← All modules](../README.md)

> Goal: implement the heart of the Transformer — multi-head attention with a KV
> cache. The easiest module to get subtly wrong; budget extra time.

## Algorithm (single position pos, one layer)

Inputs: rotated Q (dim,), and rows 0..pos of K and V from the KV cache (each
row kv_dim,).

Split dim into `n_heads` heads of `head_size` each. For each head h:

```
qh = Q[h*head_size : (h+1)*head_size]
for t = 0..pos:                              # history + current only = causal mask
    kh_t = K_cache[t][kv_h*head_size : ...]    # kv_h = h / kv_mul (GQA sharing; here kv_mul=1 so kv_h=h)
    score_t = (qh . kh_t) / sqrt(head_size)
att = softmax(score)                         # over the 0..pos segment only
out_h = sum_t att_t * V_cache[t][kv_h slice]
```

Concatenate the 6 heads' `out_h` back into a dim vector: that is the layer's
attention output (it still needs the wo projection + residual — module 08's job).

Model parameters: n_heads=6, head_size=48, n_kv_heads=6, kv_mul=1, kv_dim=288.

## Data files (position-major, P=5, 288 values each)

| File | Content |
| --- | --- |
| `input_q.txt` | rotated Q per position |
| `input_k_cache.txt` | K per position (post-rotation, as stored in cache) |
| `input_v_cache.txt` | V per position |
| `expected_out.txt` | attention output per position (heads concatenated, before wo) |
| `expected_att_weights_lastpos.txt` | softmax weights at the last position (pos=4): 6 heads x 5 = 30 values, head-major |

## Tasks and verification

For each pos = 0..4, compute the output from its Q and cache rows 0..pos:

```bash
python3 ../tools/compare.py out.txt data/expected_out.txt
python3 ../tools/compare.py out_att.txt data/expected_att_weights_lastpos.txt
```

## Hints

- **Match att_weights before the full output**: if weights are right but the
  output is wrong, the bug is in the weighted sum or head slicing; if weights
  are wrong, it's the dot product or softmax range.
- Classic bug 1: softmax over the whole seq_len (256) instead of 0..pos — the
  zeros from future positions get exp'ed in.
- Classic bug 2: forgetting `/ sqrt(head_size)`.
- Classic bug 3: wrong in-row offset — row t, head h lives at
  `t * kv_dim + kv_h * head_size`.
- Notice the `t = 0..pos` loop grows with pos: attention cost scales linearly
  with sequence length, and this is exactly why decode must cache K/V instead
  of recomputing the whole prefix every step.
