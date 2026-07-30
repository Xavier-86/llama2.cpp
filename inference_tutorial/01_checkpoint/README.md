# 01 checkpoint: load the model weights <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

## Overall task

Fill in the two TODOs in `main.cpp`: parse the header of `../../stories15M.bin` and map the 11 weight tensors out of its weight region:

```
[Config header: 7 x int32] [weight region: packed float32]
        |                           |
        v                           v
 out_config.txt              out_summary.txt
   (7 integers)          (11 tensors x 3 values)
```

The checkpoint is a small binary header followed by all weights packed as raw float32, in a fixed order: parse the header to learn the dimensions, then walk the weight region tensor by tensor, recording one view per tensor.

This is the only module dedicated to binary parsing. Later modules never parse files themselves: when they need weights they call `tut::load_checkpoint()` from `../common/checkpoint.h` — the packaged version of this module's answer. So it pays to understand the format thoroughly here.

**Inputs**: a single binary file, already read into memory by the written part of `main()` — no parsing of the file stream is left to you:

| Variable | Location | Shape | Meaning |
| --- | --- | --- | --- |
| `checkpoint_path` | main.cpp | — | path to `../../stories15M.bin` (stories15M, FP32) |
| `buf` | main.cpp, already filled | (file_bytes/4,) float | the whole file: Config header + weight region |
| `Config` | main.cpp | 7 x int32 | header struct: `dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len` |
| `Weights` | main.cpp | 12 spans | one `std::span<const float>` view per tensor, to be filled |

**Outputs**: the TODOs write two files — `out_config.txt` (the 7 config integers, one per line) and `out_summary.txt` (33 lines: per tensor its element count, first element, and sum of all elements). `data/expected_config.txt` and `data/expected_weight_summary.txt` are golden data — do not modify them. The summary file also lists the actual tensor sizes of this model.

## Subtask 1: parse the Config header (TODO task 1)

Read the 7 int32 fields from the start of `buf` into a `Config`, print them as stored (one per line) to `out_config.txt`, and verify against `data/expected_config.txt` with `--exact`.

Background you need — the Config header's layout and conventions:

- fields in order: `dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len`
- a **negative** `vocab_size` marks unshared classifier weights; stories15M is shared, so it reads +32000

Background you need — how the file gets into memory. `main()` already reads the whole file into a `vector<float>` with `ifstream`; at 15M scale that is fine. The advanced alternative is `mmap` (see `MappedFile` in `../../run.cpp`: zero-copy, essential for big models) — both work here.

## Subtask 2: walk the weight region and summarize the tensors (TODO task 2)

The weight region starts right after the header. Compute each tensor's offset from the layout table below, record one `std::span` view per `Weights` field, in file order. Remember:

- `kv_dim = n_kv_heads * (dim / n_heads)`
- the legacy `freq_cis_real` / `freq_cis_imag` tables sit between `rms_final_weight` and `wcls` — skip them
- with shared weights, `wcls` aliases `token_embedding_table`; read it separately only when unshared

Then, for each of the 11 tensors in table order, print 3 lines to `out_summary.txt` (`std::scientific`, `std::setprecision(3)`): **element count, first element, sum of all elements (accumulate in double, in file order)** — 33 numbers in total — and verify against `data/expected_weight_summary.txt`.

Background you need — the weight region's layout (actual sizes for this model are in `data/expected_weight_summary.txt`):

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

Background you need — "mapping" needs no copy: just record `(pointer, length)` per tensor. In C++20 `std::span` expresses exactly this "view". Layer l's wq slice starts at `wq_start + l * dim * dim` and has length `dim * dim`; later modules use this indexing constantly.

Background you need — the sums must be accumulated in `double`, in file order, or they won't match the golden data.

## Build / run / verify

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_config.txt data/expected_config.txt --exact
python3 ../tools/compare.py out_summary.txt data/expected_weight_summary.txt
```

## Common pitfalls

- A wrong sum usually means: wrong tensor order, wrong size, or forgetting to skip freq_cis.
- Sums don't match the golden data even with the right sizes → you did not accumulate in `double`, or not in file order.

## Done when

Both comparisons PASS. `solution.cpp` is the reference answer — peek if stuck, then close it and write your own.
