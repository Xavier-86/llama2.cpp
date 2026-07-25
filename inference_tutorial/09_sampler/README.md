# 09 sampler: from logits to the next token

[← All modules](../README.md)

> Goal: implement three sampling strategies. Input: 32000 scores. Output: one
> token id.

## The three strategies

```
temperature == 0:   plain argmax (greedy, deterministic)
topp <= 0 or >= 1:  full-distribution sampling: logits /= temperature -> softmax
                    -> draw a coin and pick by CDF
otherwise:          top-p (nucleus): probabilities as above; keep only tokens
                    with prob >= (1-topp)/(n-1); sort by prob descending; take
                    the smallest set whose cumulative prob exceeds topp;
                    sample within it using coin * set_total_prob
```

## Random number generator (must match the reference bit-for-bit)

xorshift with a uint64 state:

```
state ^= state >> 12;  state ^= state << 25;  state ^= state >> 27
u32 = (state * 0x2545F4914F6CDD1D) >> 32
coin = (u32 >> 8) / 16777216.0f        # float in [0, 1)
```

`data/expected_rng_seed42.txt` holds the first 10 coins for seed=42 — verify
the RNG before testing sampling.

## Data files

| File | Content |
| --- | --- |
| `input_logits.txt` | real logits (last prompt position, 32000 values) |
| `input_logits_synth.txt` | synthetic logits `[1..8]` (vocab=8) |
| `expected_rng_seed42.txt` | first 10 random numbers for seed 42 |
| `expected_samples.txt` | expected token ids for 8 cases (order below) |

Case order — first 4 on real logits, last 4 on synthetic, as
(temperature, topp, seed):

```
1. (0.0, 0.9, 42)   -> greedy
2. (1.0, 1.0, 42)   -> full distribution
3. (0.8, 0.9, 42)   -> top-p
4. (0.8, 0.9, 1234) -> top-p, different seed
5. (0.0, 0.9, 42)   synth -> greedy, expect 7
6. (1.0, 1.0, 42)   synth -> full distribution, expect 6
7. (1.0, 0.5, 42)   synth -> top-p, expect 7
8. (2.0, 0.9, 7)    synth -> top-p, expect 5
```

The real distribution is 96.6% concentrated on the argmax, so all real cases
pick it — the synthetic cases are what actually exercise the mult / top-p paths.

## Tasks and verification

```bash
python3 ../tools/compare.py out_rng.txt data/expected_rng_seed42.txt
python3 ../tools/compare.py out.txt data/expected_samples.txt --exact
```

## Hints

- **Sampling mutates the input logits** (divides by temperature in place, then
  softmax). Copy them before each case.
- top-p sorting must be by probability **descending**; `std::sort` with a
  custom comparator is enough.
- A floating-point tail difference can land a coin exactly on a CDF boundary —
  the cases were chosen to avoid this, and a correct implementation passes. If
  exactly one case fails, print your intermediate probabilities and compare.
