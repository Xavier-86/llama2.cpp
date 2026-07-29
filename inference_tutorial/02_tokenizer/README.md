# 02 tokenizer: BPE encoding and decoding <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project home](../../README.md)

> Goal: implement the BPE tokenizer's `encode` (text -> token ids) and `decode` (id -> text piece). Standalone module: no model weights needed, only `../../tokenizer.bin`.

## Background

The model never sees text — only integer sequences. The tokenizer converts in both directions:

```
"Once upon a time"  --encode-->  [1, 9038, 2501, 263, 931]  --forward-->  logits  --sample-->  next id  --decode-->  text
```

This module implements the same BPE (byte-pair encoding) algorithm as `Tokenizer` in `../../run.cpp`: split the text into single-character tokens, then greedily merge adjacent pairs by vocab score. Vocab file parsing is already handled by `../common/tokenizer.h` — you only write the algorithm.

## tokenizer.bin format

```
[max_token_length: int32]
then vocab_size (=32000) entries, each:
  [score: float32] [len: int32] [len bytes of string]
```

Parsing is done by `tut::load_vocab(path, vocab_size)` (`../common/tokenizer.h`), which returns `tut::Vocab{pieces, scores, max_token_length}`: `pieces[id]` is the token's text piece, `scores[id]` its BPE merge score. **Parsing is not a learning task of this module** — the `Tokenizer` constructor calls it in one line.

Vocabulary conventions:

- id 0 = `<unk>`, id 1 = `<s>` (BOS), id 2 = `</s>` (EOS)
- ids 3..258 are the 256 single-byte tokens (`<0x00>`..`<0xFF>`); byte b has token id **b + 3**
- everything else is a BPE-merged subword; higher `score` = higher merge priority

## encode algorithm

Inputs: `text` plus `bos`/`eos` switches (eos is always false in this project):

1. If `bos=true`, push BOS id 1 first.
2. If the text is non-empty, push the id of `" "` (a space). Llama convention: text is segmented as if it started with a space.
3. **Per-character lookup**: split on UTF-8 boundaries — a byte with `(c & 0xC0) != 0x80` starts a new character (0x80..0xBF are continuation bytes); look each character up in the vocab; if missing, fall back to one token per byte (`byte + 3`, the byte fallback).
4. **Greedy merging**: repeatedly scan all adjacent token pairs, concatenating their strings and looking the result up in the vocab; among the pairs that exist, merge the one with the highest `score`; stop when no pair can merge.
5. If `eos=true`, append EOS id 2.

## decode algorithm

`decode(prev_token, token)` returns the token's string, with two rules:

- if `prev_token == 1` (previous was BOS) and the piece starts with a space, strip that space (the inverse of encode step 2)
- a piece of the form `<0xXX>` (exactly 6 chars) expands to that single byte

## Input data

All inputs are const variables in main.cpp (except the vocab) — no data files to parse:

| Variable | Location | Shape | Meaning |
| --- | --- | --- | --- |
| `kPrompts` | main.cpp | (4,) string array | the four prompts to encode, see the table below |
| `kDecodeIds` | main.cpp | (5,) int array | the id sequence to decode: prompt 0's tokens, BOS first |
| `vocab_.pieces` / `vocab_.scores` | `tut::load_vocab` | (32000,) | the vocab: token id -> text piece / BPE merge score, from `../../tokenizer.bin` |

The four prompts and their expected token ids (BOS included — the contents of `data/expected_encode_*.txt`):

| # | Prompt | Expected token ids |
| --- | --- | --- |
| 0 | `Once upon a time` | `[1, 9038, 2501, 263, 931]` |
| 1 | `One day, a little girl named Lily` | `[1, 3118, 2462, 29892, 263, 2217, 7826, 4257, 365, 2354]` |
| 2 | `Hello, world!` | `[1, 15043, 29892, 3186, 29991]` |
| 3 | `The capital of France is` | `[1, 450, 7483, 310, 3444, 338]` |

Prompt 0 is the reference prompt used across the whole tutorial. `data/input_prompts.txt` and `data/input_decode_ids.txt` are kept for reference and match the const arrays; `data/expected_*` is golden data — do not modify it.

## Tasks

Fill in the four TODOs of the `Tokenizer` class in `main.cpp`:

1. **task 1 — `init_sorted_vocab()` / `str_lookup(str)`**: implement string -> token id lookup: build a sorted index (or a hash map) once, then binary-search; return -1 when missing. A linear scan passes too, but hurts inside the merge loop.
2. **task 2 — first half of `encode`** (algorithm steps 1-3): BOS, the leading `" "`, per-character lookup on UTF-8 boundaries + byte fallback.
3. **task 3 — second half of `encode`** (algorithm steps 4-5): greedy merging + optional EOS.
4. **task 4 — `decode(prev_token, token)`**: return the token piece, handling the post-BOS leading-space strip and the `<0xXX>` byte expansion.

`main()` is already written: it encodes the 4 prompts into `out0.txt .. out3.txt` and decodes `kDecodeIds` into `out_decode.txt`.

## Build / run / verify

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out0.txt data/expected_encode_0.txt --exact
python3 ../tools/compare.py out1.txt data/expected_encode_1.txt --exact
python3 ../tools/compare.py out2.txt data/expected_encode_2.txt --exact
python3 ../tools/compare.py out3.txt data/expected_encode_3.txt --exact
python3 ../tools/compare.py out_decode.txt data/expected_decode.txt --text
```

## Common pitfalls

- Output has a long tail of single-character ids → you forgot the leading `" "` token after BOS, or the merge loop never fires.
- Merge results differ from expected → you looked up an id pair instead of the **concatenated string**, or did not pick the highest-scoring pair.
- Multi-byte characters (e.g. Chinese) get shredded → wrong UTF-8 boundary test: continuation bytes satisfy `(c & 0xC0) == 0x80` and must not start a new character.
- `out_decode.txt` starts with an extra space → you forgot to strip the leading space when `prev_token == 1`.

## Done when

All five comparisons PASS. `solution.cpp` is the reference answer — peek if stuck, then close it and write your own.
