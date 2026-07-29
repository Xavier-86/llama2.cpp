# Inference Tutorial: Build llama2 Inference Module by Module <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← Project README](../README.md)

> A hands-on companion to `run.cpp` / `runq.cpp`: re-implement int8-quantized
> Llama-2 inference yourself, one module at a time, with golden input/output
> data to test every step.

## How it works

Each module folder contains:

- `README.md` — background, the math, the structure and provenance of every
  input, the task breakdown, and how to verify; the README alone is enough
  to get started
- `main.cpp` — a template: your tasks are written as `TODO` comments
- `solution.cpp` — a reference answer (compiles and passes all comparisons)
- `data.h` (some modules) — input test vectors as const arrays, generated
  from `data/input_*.txt` by `../tools/embed_data.py`; do not edit by hand
- `data/` — `input_*.txt` (the source of data.h) and `expected_*.txt`
  (golden answers), one number per line, printed with 3 decimal places
  (`setprecision(3)`)

Boilerplate shared across modules lives in `common/` (so module code can
focus on the algorithms):

- `common/io.h` — golden-data IO (`tut::write_floats` / `write_ints` /
  `write_text` / `read_floats`)
- `common/checkpoint.h` — FP32 checkpoint loader (`tut::load_checkpoint`),
  the packaged answer to module 01, used directly by 07/09/11
- `common/tokenizer.h` — tokenizer.bin vocab loader (`tut::load_vocab`),
  used by 02/11/12; the BPE algorithm itself remains a learning task

All inputs are const variables (outside `main`): small arrays inline in
`main.cpp`, larger ones in the module's `data.h`. Binary parsing is taught
once, in module 01. The single exception is 10_sampler's 32000-dim real
logits — too large to embed — loaded with one line of `tut::read_floats`.

Workflow per module:

1. Read the module's `README.md`
2. Fill in `main.cpp` (or write your own file), build and run it
3. Compare your output against the golden data:

```bash
# floating point (default atol=1e-3 rtol=1e-3, matching the 3-decimal data)
python3 ../tools/compare.py out.txt data/expected_xxx.txt

# integers: token ids, int8 values (exact)
python3 ../tools/compare.py out.txt data/expected_xxx.txt --exact

# text: decode results, generated stories (byte-for-byte)
python3 ../tools/compare.py out.txt data/expected_xxx.txt --text
```

Build with: `c++ -O2 -std=c++20 -o main main.cpp`

Try not to read `solution.cpp` (or the reference `../run.cpp` / `../runq.cpp`)
before you have your own version passing. Peek when stuck, then close it.

## Module roadmap

| # | Module | What you build | Depends on | Time |
| --- | --- | --- | --- | --- |
| 00 | [00_setup](00_setup/README.md) | Build the reference, get a baseline | — | 0.5h |
| 01 | [01_checkpoint](01_checkpoint/README.md) | Parse the checkpoint header, map weights (the dedicated binary-format module) | — | 1-2h |
| 02 | [02_tokenizer](02_tokenizer/README.md) | BPE encode / decode | — | 2-3h |
| 03 | [03_rmsnorm_softmax](03_rmsnorm_softmax/README.md) | Two small math kernels | — (data embedded) | 0.5h |
| 04 | [04_matmul](04_matmul/README.md) | FP32 matrix-vector multiply | — | 0.5h |
| 05 | [05_rope](05_rope/README.md) | Rotary position embedding | — (data embedded) | 1h |
| 06 | [06_attention](06_attention/README.md) | Multi-head causal attention + KV cache | 03, 05 | 2h |
| 07 | [07_ffn](07_ffn/README.md) | SwiGLU feed-forward network | 04 (weights via common/checkpoint.h) | 1h |
| 08 | [08_transformer_layer](08_transformer_layer/README.md) | Assemble a single transformer layer | 03-07 | 1h |
| 09 | [09_forward](09_forward/README.md) | Full single-step forward pass (FP32) | 08 | 1h |
| 10 | [10_sampler](10_sampler/README.md) | argmax / temperature / top-p sampling | 03 | 1-2h |
| 11 | [11_generate](11_generate/README.md) | prefill/decode generation loop | 09, 10, 02 | 1h |
| 12 | [12_quantize](12_quantize/README.md) | int8 quantization: format, kernels, forward | all of 11 | 2-3h |

Finishing module 11 means you have rebuilt `run.cpp` (FP32). Module 12 adds
quantization and gives you `runq.cpp`.

## Test setup and data conventions

- Models: `../../stories15M.bin` (FP32) and `../../stories15M-q32.bin` (int8, GS=32);
  tokenizer: `../../tokenizer.bin`
- Model config: dim=288, hidden_dim=768, n_layers=6, n_heads=6, n_kv_heads=6,
  vocab_size=32000, seq_len=256 (head_size=48, kv_dim=288)
- Reference prompt: `"Once upon a time"` -> token ids `[1, 9038, 2501, 263, 931]`,
  P=5 positions
- Files that hold per-position values are concatenated **position-major**:
  first the `dim` values of pos 0, then pos 1, and so on
- Matrices are stored **row-major**
- Floating-point values are printed as `%.3e`; comparisons use 3-decimal tolerance

## A note on numerical tolerance

- Small kernels (03/04/05/06/07): your output should match to ~1e-3 (the data
  itself is rounded to 3 decimals). A bigger difference means a bug.
- Full forward (09/12): summation order differences (compiler vectorization,
  loop order) make ~1e-3 wobble normal. Compare argmax first (integers, robust),
  then logits.
- Sampling and generation (10/11) are **discrete**: either exactly right or
  wrong. If wrong, trace back to the module-09 logits.

## Regenerating the golden data

The data was produced by `tools/dump_fp32.cpp` / `tools/dump_int8.cpp`, which
`#include` the reference implementations and export intermediate values (so the
golden data is same-source with `run.cpp` / `runq.cpp`). After the dump tools,
run `tools/embed_data.py` to embed the input vectors as const arrays in each
module's `data.h`. To regenerate with a different model or prompt, run from
the repository root:

```bash
c++ -O2 -std=c++20 -o inference_tutorial/tools/dump_fp32 inference_tutorial/tools/dump_fp32.cpp
c++ -O2 -std=c++20 -o inference_tutorial/tools/dump_int8 inference_tutorial/tools/dump_int8.cpp
./inference_tutorial/tools/dump_fp32 inference_tutorial stories15M.bin tokenizer.bin
./inference_tutorial/tools/dump_int8 inference_tutorial stories15M-q32.bin tokenizer.bin
python3 inference_tutorial/tools/embed_data.py inference_tutorial
```

(The prompt is hardcoded in the dump tools; edit and rebuild to change it.)
