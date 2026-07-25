# 10 generate: the generation loop (prefill / decode)

[← All modules](../README.md)

> Goal: wire forward + sampler + tokenizer into full text generation. Passing
> this means you have rebuilt the FP32 `run.cpp`.

## Loop structure

```
prompt_tokens = encode(prompt, bos=true)
token = prompt_tokens[0];  pos = 0
while pos < steps:
    logits = forward(token, pos)
    if pos < len(prompt_tokens) - 1:
        next = prompt_tokens[pos + 1]   # prefill: force-feed the next prompt token
    else:
        next = sample(logits)           # decode: the model decides
    pos += 1
    if next == 1: break                 # BOS again = stop
    print decode(token, next)           # note the (current, next) pair
    token = next
```

- **prefill**: run the prompt through token by token to fill the KV cache; no
  sampling
- **decode**: sample one token per step, feed it back — serial, one full
  forward per word
- TTFT in serving systems ≈ prefill time, throughput ≈ decode speed: the two
  halves of this loop

## Data files (prompt is always "Once upon a time", 64 steps)

| File | Content |
| --- | --- |
| `expected_greedy_ids.txt` | token ids from greedy decoding (t=0) |
| `expected_greedy_text.txt` | greedy generated text |
| `expected_sampled_ids.txt` | ids from sampling (t=0.8, p=0.9, seed=42) |
| `expected_sampled_text.txt` | sampled generated text |

## Tasks and verification

```bash
python3 ../tools/compare.py out_ids.txt data/expected_greedy_ids.txt --exact
python3 ../tools/compare.py out_text.txt data/expected_greedy_text.txt --text
python3 ../tools/compare.py out_sids.txt data/expected_sampled_ids.txt --exact
python3 ../tools/compare.py out_stext.txt data/expected_sampled_text.txt --text
```

Greedy right but sampled wrong -> check the sampler (module 09). Both wrong but
the first few ids right -> check the prefill/decode switch condition.

## Hints

- The id sequence records every `next` token produced by the loop. Therefore
  its first four ids are the force-fed prompt continuation
  `[9038, 2501, 263, 931]`; BOS (`1`) itself is not included.
- The sampled run consumes one random number per step, in exactly the
  reference's order — one extra RNG call and everything diverges from step one.
- Add timing: measure from the second token onward for tok/s and compare with
  the reference's `achieved tok/s` — same ballpark expected.
- FP32 is now complete. Run `./main` and watch your own implementation tell the
  same story as the original — the most satisfying moment of the project.
