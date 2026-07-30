# 02 tokenizer: BPE encoding and decoding <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project home](../../README.md)

## Overall task

Fill in the four TODOs of the `Tokenizer` class in `main.cpp`, implementing both directions of a BPE tokenizer:

```
"Once upon a time"  --encode-->  [1, 9038, 2501, 263, 931]
[1, 9038, 2501, 263, 931]     --decode-->  "Once upon a time"
```

The model never sees text — only integer sequences; the tokenizer converts between the two. This module implements the same BPE (byte-pair encoding) algorithm as `Tokenizer` in `../../run.cpp`: split the text into single-character tokens, then greedily merge adjacent pairs by vocab score.

Standalone module: no model weights needed, only the vocab file `../../tokenizer.bin`. Vocab parsing is already handled by `tut::load_vocab(path, vocab_size)` (`../common/tokenizer.h`), which returns `tut::Vocab{pieces, scores, max_token_length}`: `pieces[id]` is the token's text piece, `scores[id]` its BPE merge score.

**Inputs**: const variables in main.cpp (except the vocab) — no data files to parse:

| Variable | Location | Shape | Meaning |
| --- | --- | --- | --- |
| `kPrompts` | main.cpp | (4,) string array | the four prompts to encode, see the table below |
| `kDecodeIds` | main.cpp | (5,) int array | the id sequence to decode: prompt 0's tokens, BOS first |
| `vocab_.pieces` / `vocab_.scores` | `tut::load_vocab` | (32000,) | the vocab: token id -> text piece / BPE merge score |

The four prompts and their expected token ids (BOS included — the contents of `data/expected_encode_*.txt`):

| # | Prompt | Expected token ids |
| --- | --- | --- |
| 0 | `Once upon a time` | `[1, 9038, 2501, 263, 931]` |
| 1 | `One day, a little girl named Lily` | `[1, 3118, 2462, 29892, 263, 2217, 7826, 4257, 365, 2354]` |
| 2 | `Hello, world!` | `[1, 15043, 29892, 3186, 29991]` |
| 3 | `The capital of France is` | `[1, 450, 7483, 310, 3444, 338]` |

**Outputs**: `main()` is already written — it encodes the 4 prompts into `out0.txt .. out3.txt` and decodes `kDecodeIds` into `out_decode.txt`. `data/expected_*` is golden data — do not modify it.

## Subtask 1: string lookup `init_sorted_vocab()` / `str_lookup(str)`

Implement string -> token id lookup: build a sorted index (or a hash map) once, then binary-search; return -1 when missing. A linear scan passes too, but hurts inside the merge loop of subtask 3.

Background you need — the vocab file's format and conventions (parsing is already done by `tut::load_vocab`; you only need to understand what you look up):

```
[max_token_length: int32]
then vocab_size (=32000) entries, each:
  [score: float32] [len: int32] [len bytes of string]
```

- id 0 = `<unk>`, id 1 = `<s>` (BOS), id 2 = `</s>` (EOS)
- ids 3..258 are the 256 single-byte tokens (`<0x00>`..`<0xFF>`); byte b has token id **b + 3**
- everything else is a BPE-merged subword; higher `score` = higher merge priority

## Subtask 2: first half of `encode` — per-character lookup

Inputs: `text` plus the `bos` switch. Do three things in order:

1. If `bos=true`, push BOS id 1 first.
2. If the text is non-empty, push the id of `" "` (a space). Llama convention: text is segmented as if it started with a space.
3. Split the text into single characters on UTF-8 boundaries and look each character up in the vocab; if missing, fall back to one token per byte (`byte + 3`, the byte fallback).

Background you need — UTF-8 boundaries and continuation bytes. Splitting per character hinges on never cutting a multi-byte character in half. UTF-8 is variable-length: one character occupies 1–4 bytes, and each byte role has a fixed top-bit pattern:

| Role | Bit pattern | Range |
| --- | --- | --- |
| ASCII (1-byte character) | `0xxxxxxx` | 0x00..0x7F |
| Leading byte (2-byte char) | `110xxxxx` | 0xC0..0xDF |
| Leading byte (3-byte char) | `1110xxxx` | 0xE0..0xEF |
| Leading byte (4-byte char) | `11110xxx` | 0xF0..0xF7 |
| **Continuation byte** | `10xxxxxx` | 0x80..0xBF |

A continuation byte always has its top two bits set to `10`, so `c & 0xC0` (keeping only the top two bits) equals `0x80` exactly for continuation bytes. Conversely, any byte with `(c & 0xC0) != 0x80` starts a new character. Example: the Chinese character 中 is encoded as the three bytes `0xE4 0xB8 0xAD`. When the loop reaches `0xB8`, `(0xB8 & 0xC0) == 0x80` tells it the byte belongs to the current character and it keeps appending to the buffer; at `0xE4` the test fails, so the buffer is cleared and a new character begins.

## Subtask 3: second half of `encode` — greedy merging

Repeatedly merge the token sequence from subtask 2 until no pair can merge; if `eos=true` (always false in this project), append EOS id 2 at the end.

Each round: scan all adjacent token pairs, concatenate their strings, and look the result up in the vocab; among the pairs that exist, merge the one with the highest `score`.

Background you need: `score` is BPE's merge priority — merges learned earlier in training have higher scores, so they must happen first. Note that you look up the **concatenated string**, not a pair of ids.

## Subtask 4: `decode(prev_token, token)`

Return the token's string, with two rules:

- if `prev_token == 1` (previous was BOS) and the piece starts with a space, strip that space (the inverse of step 2 of subtask 2)
- a piece of the form `<0xXX>` (exactly 6 chars) expands to that single byte

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
