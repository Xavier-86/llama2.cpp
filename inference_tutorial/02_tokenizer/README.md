# 02 tokenizer: BPE encoding and decoding

[← All modules](../README.md)

> Goal: implement `encode` (text -> token ids) and `decode` (id -> text piece).
> Standalone module: no model weights needed, only `../../tokenizer.bin`.

## tokenizer.bin format

```
[max_token_length: int32]
then vocab_size (=32000) entries, each:
  [score: float32] [len: int32] [len bytes of string]
```

Vocabulary conventions:

- id 0 = `<unk>`, id 1 = `<s>` (BOS), id 2 = `</s>` (EOS)
- ids 3..258 are the 256 single-byte tokens (`<0x00>`..`<0xFF>`); byte b has
  token id **b + 3**
- everything else is a BPE-merged subword; higher `score` = higher merge priority

## encode algorithm (mirrors `Tokenizer::encode` in the reference)

Inputs: `text` plus `bos`/`eos` switches (eos is always false in this project):

1. If `bos=true`, push id 1 first
2. If the text is non-empty, push the id of `" "` (a space) — Llama convention:
   text is segmented as if it started with a space
3. **Per-character lookup**: split on UTF-8 boundaries (a byte with
   `(c & 0xC0) != 0x80` starts a new character); look each character up in the
   vocab; if missing, fall back to one token per byte (`byte + 3`)
4. **Greedy merging**: repeatedly scan adjacent token pairs; if the
   concatenation exists in the vocab, merge the highest-scoring pair; stop when
   no pair can merge

## decode algorithm

`decode(prev_token, token)` returns the token's string, with two rules:

- if `prev_token == 1` (previous was BOS) and the piece starts with a space,
  strip that space
- a piece of the form `<0xXX>` (6 chars) decodes to the corresponding byte

## Data files

| File | Content |
| --- | --- |
| `input_prompts.txt` | 4 lines, one prompt each |
| `expected_encode_0..3.txt` | token id sequences (including BOS) |
| `input_decode_ids.txt` | id sequence of prompt 0 |
| `expected_decode.txt` | text from `decode(id[i], id[i+1])` concatenated |

## Tasks and verification

```bash
# encode: compare each prompt's ids
python3 ../tools/compare.py out0.txt data/expected_encode_0.txt --exact
# decode: byte-for-byte text comparison
python3 ../tools/compare.py out_decode.txt data/expected_decode.txt --text
```

## Hints

- Make vocab lookup fast: hash map or sorted + binary search. A linear scan per
  lookup technically works but hurts inside the merge loop.
- The naive O(n^2) merge loop (scan all adjacent pairs each round) is plenty
  for 5-10 token prompts.
- The two easiest mistakes: forgetting the leading `" "` token, and looking up
  id pairs in the merge loop instead of the concatenated string.
