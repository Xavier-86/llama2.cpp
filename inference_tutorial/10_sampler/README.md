# 10 sampler: from logits to the next token <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project home](../../README.md)

## Overall task

Fill in the TODOs in `main.cpp` (all inside `Sampler` and `softmax`): implement three sampling strategies — greedy argmax, full-distribution sampling, top-p (nucleus) — plus a xorshift RNG that must match the reference bit-for-bit. Input: 32000 scores. Output: one token id.

Sampling is the last step of the generation loop. Every generated token goes through:

```
existing tokens → forward (embedding → 6 Transformer layers → final RMSNorm → matmul with wcls)
                → logits[vocab_size] → sampler.sample(logits) → next token id → append, repeat
```

Module 09 produces the logits; this module answers "given the logits, which token do we pick?". The strategy is chosen by two hyperparameters: `temperature` and `topp`.

Unlike the previous modules, the output here is a **discrete token id** — the test is all-or-nothing, there is no "close enough". A single floating-point tail difference can land a coin on the other side of a CDF boundary and pick a different token. So the RNG must match bit-for-bit, and all intermediate arithmetic must be done in `float`, in the reference implementation's order.

**Inputs**: a synthetic const array plus one data file:

| Variable | Location | Shape | Layout | Meaning | Where it comes from in the model |
| --- | --- | --- | --- | --- | --- |
| `kLogitsSynth` | main.cpp (const array) | (8,) | 1-D float | synthetic logits `[1..8]`; the flat ramp gives every token a visibly different probability | none (hand-made, exists to exercise the mult / top-p paths) |
| `real` | local in main(), loaded via `tut::read_floats("data/input_logits.txt")` | (32000,) | 1-D float, one value per line | the real logits vector | output of module 09's full forward at the **last prompt position** (pos=4) of the reference prompt "Once upon a time" (token ids `[1, 9038, 2501, 263, 931]`, P=5): final RMSNorm followed by the `wcls` matmul |

**Why is the real logits vector still read from a file?** This is the single exception in the whole tutorial: embedding 32000 floats as a const array would bloat the source by ~400 KB for no teaching value, so they stay in a data file and are loaded with one `tut::read_floats` call (declared in `../common/io.h`). This module has no data.h.

Model constant: `vocab_size = 32000` (length of the real logits); the synthetic cases use vocab = 8.

**Outputs**: `main()` is already written — it runs the RNG self-check (first 10 coins for seed 42) into `out_rng.txt`, then the 8 cases into `out.txt`. First 4 cases on the real logits, last 4 on the synthetic ones, as (temperature, topp, seed):

```
1. (0.0, 0.9, 42)   real  -> greedy
2. (1.0, 1.0, 42)   real  -> full distribution
3. (0.8, 0.9, 42)   real  -> top-p
4. (0.8, 0.9, 1234) real  -> top-p, different seed
5. (0.0, 0.9, 42)   synth -> greedy, expect 7
6. (1.0, 1.0, 42)   synth -> full distribution, expect 6
7. (1.0, 0.5, 42)   synth -> top-p, expect 7
8. (2.0, 0.9, 7)    synth -> top-p, expect 5
```

The real distribution is 96.6% concentrated on the argmax (token 29892), so all 4 real cases pick it — the synthetic cases are what actually exercise the mult / top-p paths. `data/expected_*` is golden data — do not modify it.

## Subtask 1: xorshift RNG `random_u32()` / `random_f32()` (task 1)

Advance the xorshift state and take the high 32 bits per the formulas below (mind the 64-bit unsigned multiply), then map the u32 to a float in `[0, 1)`. Make out_rng.txt match before moving on.

Background you need — the xorshift algorithm (must match the reference bit-for-bit). 64-bit state `st` (the seed is the initial state); per draw:

```
st ^= st >> 12
st ^= st << 25
st ^= st >> 27
u32 = (st * 0x2545F4914F6CDD1D) >> 32    # 64-bit unsigned multiply, take the high 32 bits
```

Mapping to a float coin in `[0, 1)` (dropping the low 8 bits leaves 24 bits — exactly the float mantissa precision):

```
coin = (u32 >> 8) / 16777216.0f          # 16777216 = 2^24
```

`data/expected_rng_seed42.txt` holds the first 10 coins for seed=42 — verify the RNG on its own before testing sampling.

## Subtask 2: `softmax(x)` (task 2)

Turn `x` into a probability distribution in place, numerically stable (subtract the max before exp): subtract the max, exponentiate, then normalize by the sum.

## Subtask 3: dispatch `sample(logits)` + greedy `sample_argmax` (task 3, 3a)

Dispatch on `temperature_` / `topp_` per the table below; `sample_argmax` returns the index of the maximum element.

Background you need — strategy dispatch:

```
temperature == 0:   greedy: plain argmax(logits), deterministic, no randomness
temperature != 0:   logits_i /= temperature      # temperature scaling, in place
                    p = softmax(logits)          # in place, now a distribution
                    coin = random_f32()          # draw one coin in [0, 1)
    topp <= 0 or >= 1:  full-distribution sampling sample_mult(p, coin)
    otherwise:          top-p (nucleus) sampling sample_topp(p, topp, coin)
```

Temperature scaling controls sharpness: `temperature < 1` sharpens the distribution towards the top token (approaching greedy), `temperature > 1` flattens it (more random). Note the greedy path does **not** scale or softmax — argmax is invariant to monotone transforms.

## Subtask 4: `sample_mult(probabilities, coin)` (task 4)

Background you need — full-distribution sampling. Accumulate probabilities left to right (a CDF walk) and return the first index whose cumulative sum exceeds the coin:

```
cdf = 0
for i in 0..n-1:
    cdf += p_i
    if coin < cdf: return i
return n - 1   # rounding fallback
```

## Subtask 5: `sample_topp(probabilities, topp, coin)` (task 5)

Background you need — top-p (nucleus) sampling:

1. **Filter**: discard tokens with prob below `cutoff = (1 - topp) / (n - 1)`; collect survivors as `(prob, index)` pairs;
2. **Sort** survivors by probability **descending**;
3. **Truncate**: keep the smallest prefix whose cumulative probability **first exceeds** `topp` (keep all if it never does), noting the prefix's total probability `cumulative_prob`;
4. **Renormalize and draw**: equivalent to dividing the prefix probabilities by `cumulative_prob` and sampling by CDF — implemented as a single CDF walk over the prefix with `r = coin * cumulative_prob`, no per-element division needed.

## Build / run / verify

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_rng.txt data/expected_rng_seed42.txt
python3 ../tools/compare.py out.txt data/expected_samples.txt --exact
```

`out.txt` holds token ids — compare with `--exact`.

## Common pitfalls

- **`sample()` mutates the input logits in place** (divides by temperature, then softmax). That is the reference convention — in real inference the logits buffer is recomputed at every position, so nothing needs preserving. The driver copies the logits for each case; do not try to "protect" the input inside `sample`.
- RNG mismatch → check: shift amounts (12 / 25 / 27), the multiply is 64-bit unsigned, you take the **high** 32 bits, and the `>> 8` plus division by `16777216.0f` are copied exactly.
- top-p all wrong → the sort must be by probability **descending**; the `n` in `cutoff = (1 - topp) / (n - 1)` is the full vocab size (32000 or 8), not the survivor count.
- Exactly one case fails → likely a floating-point tail difference landed a coin on the other side of a CDF boundary: check for stray `double`, that softmax subtracts the max, and that temperature scaling happens **before** softmax.
- Softmax/coin draw on the greedy path → wasteful, and it consumes a random number (harmless here since each case builds a fresh Sampler, but it would desync the real generation loop).

## Done when

Both comparisons PASS (RNG float comparison + exact match of the 8 token ids). `solution.cpp` is the reference answer — peek if stuck, then close it and write your own.
