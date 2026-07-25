# 05 RoPE: rotary position embedding <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md)

> Goal: inject position information into Q and K. Attention alone cannot tell
> word order — this step is what makes it position-aware.

## Algorithm

Treat each adjacent pair `(v0, v1)` of Q (and K) as a 2D vector and rotate it
by `pos * freq`:

```
for i = 0, 2, 4, ... dim-2:
    head_dim = i % head_size                     # pair index within its own head
    freq     = 1 / 10000^(head_dim / head_size)  # each dim pair gets its own frequency
    angle    = pos * freq
    (v0', v1') = (v0*cos(angle) - v1*sin(angle),  v0*sin(angle) + v1*cos(angle))
```

Rules:

- Rotate all `dim` pairs of Q
- Rotate only the first `kv_dim` pairs of K (here kv_dim = dim = 288, so K is
  fully rotated too; the difference only shows in GQA models where kv_dim < dim)
- Rotation happens **in place**: the rotated K stays in the KV cache (so the
  cache holds post-rotation K)
- Different frequencies encode different distance scales: low frequencies
  rotate slowly (long range), high ones fast (local)

The net effect: the Q·K dot product of two tokens depends only on their
**relative distance** — exactly the position signal attention needs.

## Data files (position-major, P=5 positions x 288 values)

| File | Content |
| --- | --- |
| `input_q.txt` | Q after projection, before rotation (5 x 288) |
| `input_k.txt` | K before rotation (5 x 288) |
| `expected_q.txt` | Q after rotation |
| `expected_k.txt` | K after rotation (= what the KV cache stores) |

The data comes from a real forward pass: layer 0, prompt "Once upon a time".

## Tasks and verification

Load Q/K, rotate for each position pos = 0..4, compare:

```bash
python3 ../tools/compare.py out_q.txt data/expected_q.txt
python3 ../tools/compare.py out_k.txt data/expected_k.txt
```

## Hints

- Easiest mistake: forgetting `head_dim = i % head_size` and using i directly —
  all frequencies come out wrong.
- At pos=0 every angle is 0, so Q/K must pass through unchanged — a free
  self-check.
- Use the float versions of `std::cos` / `std::sin`, not double-and-cast.
