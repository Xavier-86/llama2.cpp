# 11 quantize: int8 quantized inference <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md)

> Goal: upgrade the FP32 implementation to int8 quantization and rebuild
> `runq.cpp`. The changes are concentrated in three places: weight storage
> format, matmul, and quantizing activations before every matmul. Everything
> else (tokenizer / sampler / generation loop) is reused unchanged.

## Why quantization is faster

Every decode step reads all weights once; speed ≈ bandwidth ÷ bytes. int8 cuts
weight reads to 1/4 (plus one float scale per 32 values), so throughput goes up
nearly 4x — that is the entire motivation of this module.

## 11.1 Quantized checkpoint format (`stories15M-q32.bin`)

**256-byte header**:

```
u32 magic = 0x616b3432 ("ak42")
i32 version = 2
Config (7 x i32, same as FP32)
u8  shared_classifier (1 = wcls shares the embedding)
i32 group_size (GS, = 32 in this file)
```

**Weight region** (starting at offset 256, in this order):

| # | Content |
| --- | --- |
| 1-3 | rms_att_weight, rms_ffn_weight (n_layers x dim each), rms_final_weight (dim) — **still FP32** |
| 4 | q_tokens: int8 (vocab x dim) + float scales (vocab x dim / GS) |
| 5-7 | wq, wk, wv, wo: layer by layer, each int8 (dim x dim) + scales (dim x dim / GS) |
| 8-10 | w1, w2, w3: layer by layer, each int8 (dim x hidden_dim) + scales |
| 11 | wcls: absent when shared=1 (reuse q_tokens) |

Layout per quantized tensor: **all int8 values first, then all its scales**;
per-layer tensors are `[L0.q][L0.s][L1.q][L1.s]...`.

At load time, dequantize q_tokens once into an FP32 copy for embedding lookup
(avoids dequantizing every step).

Verification: print config + GS (8 integers, `--exact` against
`expected_config.txt`); then a weight summary in module-01 style
(`expected_weight_summary.txt`: first the 3 FP32 tensors x 3 lines
size/first/sum, then the 8 quantized tensors x 5 lines
q_total_size/q_first/q_sum/s_total_size/s_first):

```bash
python3 ../tools/compare.py out_config.txt data/expected_config.txt --exact
python3 ../tools/compare.py out_summary.txt data/expected_weight_summary.txt
```

## 11.2 quantize / dequantize

Symmetric quantization, one scale per group of GS values:

```
scale = max(|group values|) / 127
q[i]  = round(x[i] / scale)     # int8 in [-127, 127]
s[g]  = scale
dequantize: x[i] = q[i] * s[i / GS]
```

Data: `input_x.txt` (64 values) -> `expected_q.txt` (64 int8, `--exact`),
`expected_s.txt` (2 scales), `expected_deq.txt` (round-trip). GS=32.

Beware: `round` here is round-half-**away-from-zero** (`std::round`), not
banker's rounding — watch out especially if you cross-check in Python (Python's
`round` is banker's).

## 11.3 int8 matmul

```
per row:
    process GS-sized groups: int8 x int8 -> int32 integer multiply-accumulate
    (cannot overflow: 127*127*32 ~ 5*10^5)
    at each group boundary: val += (float)group_int_sum * w.s[group] * x.s[group]
```

The integer part is exact; floats appear only at group boundaries — that is
what makes the quantized version both fast and stable.

Data (n=8, d=3, GS=4): `input_matmul_wq.txt` (24 int8, row-major),
`input_matmul_ws.txt` (3x2 scales, row-major), `input_matmul_xq.txt` (8 int8),
`input_matmul_xs.txt` (2 scales) -> `expected_matmul_out.txt` (3 values).

## 11.4 Quantized forward + generation

Take module 08's forward and, **before every matmul**, quantize the activation
to int8 (then multiply with the int8 weights via 11.3). RMSNorm / RoPE /
attention / SwiGLU / softmax all stay FP32, untouched.

Data: `input_tokens.txt` (same 5 tokens) -> `expected_argmax.txt` (5 integers,
`--exact`) + `expected_logits_lastpos.txt` (full logits at the last position).

Finally run 64 greedy steps -> `expected_greedy_ids.txt` /
`expected_greedy_text.txt`:

```bash
python3 ../tools/compare.py out_ids.txt data/expected_greedy_ids.txt --exact
python3 ../tools/compare.py out_text.txt data/expected_greedy_text.txt --text
```

## Tasks and full verification

`main.cpp` keeps the completed tokenizer, sampler, attention, and generation
code from the earlier modules. Fill in the four quantization-specific tasks:

1. map the 256-byte checkpoint header and mixed FP32/int8 weight region
2. implement group-wise quantize and dequantize
3. implement grouped int8 matrix-vector multiplication
4. replace module 08's FP32 projections with quantize + int8 matmul

The program writes all intermediate results needed for staged debugging:

| Output | Golden data |
| --- | --- |
| `out_config.txt` | `expected_config.txt` |
| `out_summary.txt` | `expected_weight_summary.txt` |
| `out_q.txt`, `out_s.txt`, `out_deq.txt` | quantize/dequantize outputs |
| `out_matmul.txt` | standalone int8 matmul output |
| `out_argmax.txt`, `out_logits.txt` | five-position quantized forward |
| `out_ids.txt`, `out_text.txt` | 64-step greedy generation |

Build and run:

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
```

Verify every stage:

```bash
python3 ../tools/compare.py out_config.txt data/expected_config.txt --exact
python3 ../tools/compare.py out_summary.txt data/expected_weight_summary.txt
python3 ../tools/compare.py out_q.txt data/expected_q.txt --exact
python3 ../tools/compare.py out_s.txt data/expected_s.txt
python3 ../tools/compare.py out_deq.txt data/expected_deq.txt
python3 ../tools/compare.py out_matmul.txt data/expected_matmul_out.txt
python3 ../tools/compare.py out_argmax.txt data/expected_argmax.txt --exact
python3 ../tools/compare.py out_logits.txt data/expected_logits_lastpos.txt
python3 ../tools/compare.py out_ids.txt data/expected_greedy_ids.txt --exact
python3 ../tools/compare.py out_text.txt data/expected_greedy_text.txt --text
```

## Debugging route

1. argmax all right but logits off -> normal, quantization is lossy; check the
   greedy text instead
2. argmax wrong from the first position -> scale/round in quantize or group
   indexing in int8 matmul is off; go back to 11.2/11.3
3. Early layers fine, later diverge -> check that every matmul re-quantizes the
   **current** activation (don't cross the xq/hq buffers)
4. Different output from the FP32 version is not automatically a bug:
   quantization changes the numbers and can flip the greedy path. For this
   model both produce the same 64-step story — compare and see.

## When you're done

Congratulations — you have fully rebuilt `runq.cpp`. Natural follow-ups:

- Measure tok/s of your FP32 vs int8 builds and check the
  "bandwidth ÷ bytes" estimate
- Read `../tools/dump_fp32.cpp` / `../tools/dump_int8.cpp` and check your
  understanding of every intermediate tensor
- Re-run everything on `stories42M.bin` / `stories42M-q32.bin` (regenerate the
  data with the tools)
- Finally read `../../runq.cpp` itself: same logic, organized as engineering-
  grade modern C++
