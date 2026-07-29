# 01 checkpoint: load the model weights <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

> Goal: understand the `stories15M.bin` file format and map the 11 weight
> tensors out of it.
>
> This is the only module dedicated to binary parsing. Later modules never
> parse files themselves: when they need weights they call
> `tut::load_checkpoint()` from `common/checkpoint.h` — the packaged version
> of this module's answer. So it pays to understand the format thoroughly here.

## File format (llama2.c FP32 checkpoint)

```
[Config header: 7 x int32] [weight region: packed float32]
```

Config fields in order: `dim, hidden_dim, n_layers, n_heads, n_kv_heads,
vocab_size, seq_len`. Note: a **negative** `vocab_size` marks unshared
classifier weights (stories15M is shared, so it reads +32000).

The weight region is laid out in this order (actual sizes for this model are in
`data/expected_weight_summary.txt`):

| # | Tensor | Shape |
| --- | --- | --- |
| 1 | token_embedding_table | (vocab_size, dim) |
| 2 | rms_att_weight | (n_layers, dim) |
| 3 | wq | (n_layers, dim, dim) |
| 4 | wk | (n_layers, dim, kv_dim) |
| 5 | wv | (n_layers, dim, kv_dim) |
| 6 | wo | (n_layers, dim, dim) |
| 7 | rms_ffn_weight | (n_layers, dim) |
| 8 | w1 | (n_layers, hidden_dim, dim) |
| 9 | w2 | (n_layers, dim, hidden_dim) |
| 10 | w3 | (n_layers, hidden_dim, dim) |
| 11 | rms_final_weight | (dim,) |
| (skip) | freq_cis_real + freq_cis_imag | 2 × (seq_len × head_size/2) floats: legacy precomputed RoPE tables, unused here, just skip |
| 12 | wcls | shared => same memory as token_embedding_table; read (vocab_size, dim) only when unshared |

## Tasks

1. Read the Config, print the 7 integers, compare against
   `data/expected_config.txt` (`--exact`)
2. Compute each tensor's offset from the table above, and for each tensor print
   3 values: **element count, first element, sum of all elements (accumulate in
   double)** — 33 numbers in total, compare against
   `data/expected_weight_summary.txt`

```bash
python3 ../tools/compare.py out_config.txt data/expected_config.txt --exact
python3 ../tools/compare.py out_summary.txt data/expected_weight_summary.txt
```

Fill in `main.cpp`; `solution.cpp` is the reference answer.

## Hints

- Simple approach: read the whole file into a `vector<float>` with `ifstream`.
  Advanced: `mmap` (see `MappedFile` in `../../run.cpp`: zero-copy, essential
  for big models). Both work at 15M scale.
- Accumulate sums in `double`, in file order, or they won't match the golden data.
- "Mapping" needs no copy: just record `(pointer, length)` per tensor.
  In C++20 `std::span` expresses exactly this "view".
- Layer l's wq slice: `wq_start + l * dim * dim`, length `dim * dim`. Later
  modules use this indexing constantly.

## Done when

Both comparisons PASS. A wrong sum usually means: wrong tensor order, wrong
size, or forgetting to skip freq_cis.
