# Inference Tutorial: Build llama2 Inference Module by Module

> A hands-on companion to `run.cpp` / `runq.cpp`: re-implement int8-quantized
> Llama-2 inference yourself, one module at a time, with golden input/output
> data to test every step.

## How it works

Each module folder contains:

- `README.md` — what to implement, the math, the data format, how to verify
- `main.cpp` — a template: your tasks are written as `TODO` comments
- `solution.cpp` — a reference answer (compiles and passes all comparisons)
- `data/` — `input_*.txt` and `expected_*.txt`, one number per line,
  printed with 3 decimal places (`setprecision(3)`)

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
| 01 | [01_checkpoint](01_checkpoint/README.md) | Parse the checkpoint header, map weights | — | 1-2h |
| 02 | [02_tokenizer](02_tokenizer/README.md) | BPE encode / decode | — | 2-3h |
| 03 | [03_rmsnorm_softmax](03_rmsnorm_softmax/README.md) | Two small math kernels | 01 (weights) | 0.5h |
| 04 | [04_matmul](04_matmul/README.md) | FP32 matrix-vector multiply | — | 0.5h |
| 05 | [05_rope](05_rope/README.md) | Rotary position embedding | — (data given) | 1h |
| 06 | [06_attention](06_attention/README.md) | Multi-head causal attention + KV cache | 03, 05 | 2h |
| 07 | [07_ffn](07_ffn/README.md) | SwiGLU feed-forward network | 01, 04 | 1h |
| 08 | [08_forward](08_forward/README.md) | Full single-step forward pass (FP32) | everything so far | 1h |
| 09 | [09_sampler](09_sampler/README.md) | argmax / temperature / top-p sampling | 03 | 1-2h |
| 10 | [10_generate](10_generate/README.md) | prefill/decode generation loop | 08, 09, 02 | 1h |
| 11 | [11_quantize](11_quantize/README.md) | int8 quantization: format, kernels, forward | all of 10 | 2-3h |

Finishing module 10 means you have rebuilt `run.cpp` (FP32). Module 11 adds
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
- Full forward (08/11): summation order differences (compiler vectorization,
  loop order) make ~1e-3 wobble normal. Compare argmax first (integers, robust),
  then logits.
- Sampling and generation (09/10) are **discrete**: either exactly right or
  wrong. If wrong, trace back to the module-08 logits.

## Regenerating the golden data

The data was produced by `tools/dump_fp32.cpp` / `tools/dump_int8.cpp`, which
`#include` the reference implementations and export intermediate values (so the
golden data is same-source with `run.cpp` / `runq.cpp`). To regenerate with a
different model or prompt, run from the repository root:

```bash
c++ -O2 -std=c++20 -o inference_tutorial/tools/dump_fp32 inference_tutorial/tools/dump_fp32.cpp
c++ -O2 -std=c++20 -o inference_tutorial/tools/dump_int8 inference_tutorial/tools/dump_int8.cpp
./inference_tutorial/tools/dump_fp32 inference_tutorial stories15M.bin tokenizer.bin
./inference_tutorial/tools/dump_int8 inference_tutorial stories15M-q32.bin tokenizer.bin
```

(The prompt is hardcoded in the dump tools; edit and rebuild to change it.)
