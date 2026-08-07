# 11 generate: the generation loop (prefill / decode) <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

## Overall task

Fill in `TODO(task 6)` — `generate()` in `main.cpp` — wiring forward (module 09) + sampler (module 10) + tokenizer (module 02) into full text generation. This is the only new work in the module; passing it means you have rebuilt the FP32 `../../cpu/run.cpp`.

Every earlier module was a single call: one forward pass, one sample. Real text generation is a **loop** — the model predicts only the next token per step, that token is fed back, and the next one is predicted:

```
"Once upon a time"  --encode-->  [1, 9038, 2501, 263, 931]  --prefill / decode loop-->  ids + text
```

The loop has two phases, matching the two core serving metrics (TTFT ≈ prefill time, throughput ≈ decode speed): **prefill** force-feeds the prompt tokens through the model one by one to fill the KV cache; **decode** then samples one token per step, serially.

The plumbing is already in place: checkpoint parsing (the product of module 01) comes from `tut::load_checkpoint` in `../common/checkpoint.h`, vocab loading from `tut::load_vocab` in `../common/tokenizer.h`, and output writing from `../common/io.h`. The components you built in modules 02–10 are **given code** in `main.cpp`, filled in with the reference implementations:

- `rmsnorm` / `softmax` / `matmul` — modules 03/04: float accumulation in rmsnorm, subtract-the-max in softmax, float accumulator in matmul.
- `Transformer::forward(token, pos)` — module 09: the full one-token-at-one-position forward; K/V are written into this layer's cache row for this position before RoPE; residuals are `+=`.
- `Tokenizer::decode` / `Tokenizer::encode` — module 02: the BPE algorithm on top of the vocab prepared by the constructor (`vocab_` / `vocab_scores_`).
- `Sampler` — module 10: argmax when `temperature == 0`; otherwise temperature scale + softmax + one coin toss, then full-distribution or top-p depending on `topp`. The RNG matches the reference bit-for-bit.

**Inputs**: no file parsing of your own is needed; all inputs are ready from these sources:

| Variable | Location | Shape / layout | Meaning | Where it comes from |
| --- | --- | --- | --- | --- |
| `prompt` | inline const string in main() | `"Once upon a time"` | the prompt; encodes to `[1, 9038, 2501, 263, 931]` (P=5, BOS included) | user input |
| `transformer.checkpoint.config` | `tut::load_checkpoint("../../models/stories15M.bin")` | 7 × int32 | model hyperparameters | checkpoint header |
| `transformer.checkpoint.weights` | same | 11 × `std::span<const float>` | all weight tensors (see below) | checkpoint weight region |
| `tokenizer` (ctor arg) | `tut::load_vocab("../../models/tokenizer.bin", vocab_size)` | `pieces`(32000,) / `scores`(32000,) | token id → text piece / BPE merge score | tokenizer.bin |
| `steps` | inline in main() | 64 | generation step cap | — |
| temperature / topp / seed | inline in main() | see the run table below | the two sampler configurations | — |

Model constants (stories15M): `dim = 288`, `hidden_dim = 768`, `n_layers = 6`, `n_heads = 6`, `n_kv_heads = 6`, `vocab_size = 32000`, `seq_len = 256`, `head_size = dim / n_heads = 48`, `kv_dim = n_kv_heads * head_size = 288`.

The 11 `weights` tensors (shapes in terms of the constants above; tightly packed in this order in the checkpoint, row-major):

| Tensor | Shape | Purpose |
| --- | --- | --- |
| `token_embedding_table` | (vocab_size, dim) | embedding lookup; wcls shares it in this model |
| `rms_att_weight` | (n_layers, dim) | RMSNorm weight before attention |
| `wq` / `wk` / `wv` | (n_layers, dim, dim) / (n_layers, dim, kv_dim) ×2 | Q/K/V projections |
| `wo` | (n_layers, dim, dim) | attention output projection |
| `rms_ffn_weight` | (n_layers, dim) | RMSNorm weight before the FFN |
| `w1` / `w3` | (n_layers, hidden_dim, dim) ×2 | FFN up-projections (gate / linear) |
| `w2` | (n_layers, dim, hidden_dim) | FFN down-projection |
| `rms_final_weight` | (dim,) | final RMSNorm weight |
| `wcls` | (vocab_size, dim) | classifier head; aliases the embedding table when shared |

`RunState` is also part of the input: it is sized from the config at construction and holds the activation buffers plus the KV cache (`key_cache` / `value_cache`, each `(n_layers, seq_len, kv_dim)`), which every forward step reads and writes.

**Outputs**: `main()` is given — it loads the checkpoint and the vocab, runs greedy and sampled, and writes the 4 output files. Text files are written with a trailing newline appended (`g.text + '\n'`), matching the golden-data convention. `data/expected_*` is golden data — do not modify it.

`main()` always runs both configurations (prompt `"Once upon a time"`, `steps = 64`):

| Run | temperature | top-p | seed | Output files |
| --- | --- | --- | --- | --- |
| greedy | 0.0 (argmax path, no RNG) | 0.9 (unused) | 42 | `out_ids.txt` / `out_text.txt` |
| sampled | 0.8 | 0.9 | 42 | `out_sids.txt` / `out_stext.txt` |

## Subtask 1: the generation loop `generate()` (`TODO(task 6)`)

Implement the prefill/decode loop, returning `Generation{ids, text}`: `ids` is every `next` produced by the loop — including the 4 force-fed prompt continuations, excluding BOS itself — and `text` is the concatenated `decode(token, next)` pieces. In order:

1. `prompt_tokens = encode(prompt, bos=true, eos=false)`; start with `token = prompt_tokens[0]` (BOS) and `pos = 0`.
2. While `pos < steps`: run `logits = forward(token, pos)`, then decide the next input `next` — prefill or decode, see below.
3. `pos += 1`; if `next == 1`, stop; otherwise push `next` into `ids`, append `decode(token, next)` to `text`, and set `token = next`.

Background you need — the two phases, prefill and decode. Say the prompt encodes to `prompt_tokens` (length P=5: `[1, 9038, 2501, 263, 931]`, where 1 is BOS `<s>`). Each loop iteration runs one `forward(token, pos)` and then decides the next input `next`:

- **prefill**: while `pos < P - 1`, `next` does **not** come from the model — it is force-fed from the prompt: `next = prompt_tokens[pos + 1]`. This phase runs the prompt tokens through the model one by one to fill their K/V into the KV cache, ending with the logits at the last prompt position. No sampling, no RNG consumed.
- **decode**: once the prompt is consumed (`pos >= P - 1`), `next = sample(logits)` — the model (via the sampler) decides. From then on each step is: sample one token → feed it back → sample again. Serial and not parallelizable: every generated word costs one full forward pass.

The whole loop as pseudocode:

```
prompt_tokens = encode(prompt, bos=true)      # [1, 9038, 2501, 263, 931]
token = prompt_tokens[0];  pos = 0
while pos < steps:                            # steps = 64
    logits = forward(token, pos)
    if pos < len(prompt_tokens) - 1:
        next = prompt_tokens[pos + 1]         # prefill: force-feed the next prompt token
    else:
        next = sample(logits)                 # decode: the model decides
    pos += 1
    if next == 1: break                       # BOS (id=1) again = stop
    ids.push_back(next)
    text += decode(token, next)               # decode(token, next) first, then token = next
    token = next
```

Three details to get right:

1. `pos` increments every step; it feeds both RoPE and the KV cache row index.
2. The stop condition is `next == 1` (the model emits BOS again); `steps` is only an upper bound.
3. Call `decode(token, next)` **before** `token = next` — decode needs the previous token to decide whether to strip a leading space.

Background you need — the KV cache's role in the loop. `forward(token, pos)` computes K/V for the **current token only** and writes them into `key_cache[layer][pos]` / `value_cache[layer][pos]`; attention then reads rows `0..pos` — all history comes from the cache and is never recomputed. Prefill is exactly the "warm-up" of this cache for decode: without the first P rows it writes, the first decode step's attention would have no context to look at. This is also why a decode step's cost is independent of sequence position (a one-token forward); only the attention history grows linearly with `pos`.

Background you need — RNG reproducibility. The sampled run's reproducibility rests entirely on the RNG: exactly one random number per step (xorshift64, see module 10), in exactly the reference's order — one extra draw and everything diverges from step one. So `sample(logits)` must be called exactly once per decode step, and never during prefill.

## Build / run / verify

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_ids.txt   data/expected_greedy_ids.txt --exact
python3 ../tools/compare.py out_text.txt  data/expected_greedy_text.txt --text
python3 ../tools/compare.py out_sids.txt  data/expected_sampled_ids.txt --exact
python3 ../tools/compare.py out_stext.txt data/expected_sampled_text.txt --text
```

## Common pitfalls

- Greedy PASS but sampled FAIL → check the sampler (module 10): temperature scaling, top-p truncation, RNG.
- Both wrong but the first few ids right → check the prefill/decode switch condition (`pos < num_prompt_tokens - 1`).
- Sampling diverges from the very first decode token → an extra random number is being consumed somewhere: exactly one coin per loop step.
- ids all right but text FAIL → `decode` takes `(token, next)` (current token first); or the post-BOS leading space was not stripped; or the trailing newline was not appended when writing.
- The first 4 ids are not `[9038, 2501, 263, 931]` → prefill is not force-feeding `prompt_tokens[pos + 1]`.

## Done when

All four comparisons PASS (two `--exact`, two `--text`). Optional: time the loop as hinted in task 6 — your tok/s should be in the same ballpark as the reference's `achieved tok/s`. `solution.cpp` is the reference answer — look only when stuck, then close it and write your own. FP32 is now complete: run `./main` and watch your own implementation tell the same story as the original — the most satisfying moment of the project.
