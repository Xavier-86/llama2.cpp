# 00 setup: build the reference, get a baseline <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

## Overall task

Two parts: (1) compile the reference implementation at the repository root, run it once, and hold on to the "correct answer" that every later module will approximate; (2) fill in the three TODOs in `main.cpp` — a minimal sanity check that opens the three binary files every later module depends on and prints their sizes.

The reference program does **forward inference only**:

```
weights (stories15M.bin) + vocab (tokenizer.bin)
        |
        v
forward(token, pos) -> 32000 logits over the vocab -> sampler -> next token
        ^                                                    |
        |________________ feed back, one token per step ______|
```

No training: load weights, generate tokens one by one.

**Inputs**: three binary files at the repository root (paths below are relative to this module folder) — no data files to parse:

| File | Used by | Meaning |
| --- | --- | --- |
| `../../stories15M.bin` | `../../run.cpp` | FP32 reference model |
| `../../stories15M-q32.bin` | `../../runq.cpp` | int8-quantized model, group size 32 |
| `../../tokenizer.bin` | both | vocabulary / tokenizer table |

**Outputs**: the story printed by `./runcpp ... -t 0.0 -n 64 -s 42 -i "Once upon a time"` must match `data/expected_greedy.txt` exactly (temperature=0 is greedy decoding, so the result is deterministic). `data/expected_greedy.txt` is golden data — do not modify it. The sanity-check `main()` prints the sizes of the three files above, one per line.

## Subtask 1: build the reference and capture the baseline

No TODO in `main.cpp` — this subtask is pure shell work at the repository root:

```bash
cd ../..   # repository root (llama2_cpp)
c++ -O3 -std=c++20 -o runcpp run.cpp
c++ -O3 -std=c++20 -o runqcpp runq.cpp
./runcpp stories15M.bin -t 0.0 -n 64 -s 42 -i "Once upon a time"
./runqcpp stories15M-q32.bin -t 0.0 -n 64 -s 42 -i "Once upon a time"
```

Then verify the first command's output against the golden file (run from the repository root):

```bash
diff <(./runcpp stories15M.bin -t 0.0 -n 64 -s 42 -i "Once upon a time" 2>/dev/null) \
     inference_tutorial/00_setup/data/expected_greedy.txt
```

Two things to observe while it runs (the physical intuition behind this whole project):

1. Look at the reported `achieved tok/s`. The 15M model does ~100+ tok/s on CPU.
2. Do the FP32 and int8 builds print the same story for the same prompt? Think
   about why quantization can (or cannot) preserve the greedy path.

Background you need — forward inference and why quantization wins (just note it for now; later modules unpack each point):

- Each `forward(token, pos)` call processes one token and outputs 32000 scores
  (logits) over the vocabulary; the sampler picks the next token.
- In the decode phase every step reads all weights once, so speed is roughly
  memory bandwidth ÷ bytes read per step — that is exactly why quantization
  (module 12) speeds things up.

## Subtask 2: `print_file_size(path)` — TODO(task 1) and TODO(task 2)

The sanity check in `main.cpp` (no real computation in this module). Do two things:

1. Open `path` as a binary file; if it cannot be opened, print an error message to stderr and return false.
2. Determine the file's size in bytes and print it on its own line to stdout, then return true.

Background you need — the three files this function will be called on are exactly the three in the inputs table above; `main.cpp` already defines their paths as `kModelFp32`, `kModelInt8`, and `kTokenizer`.

## Subtask 3: wire up `main()` — TODO(task 3)

Call `print_file_size` for each of the three files (FP32 model, int8 model, tokenizer); return 0 only if all three succeed, 1 otherwise.

## Build / run / verify

Build and run the sanity check from this folder:

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
```

The reference-build and `diff` commands live in subtask 1.

## Common pitfalls

- `diff` reports a missing file → the verification command must run from the repository root (the golden path starts with `inference_tutorial/...`), while the paths inside `main.cpp` are relative to this module folder.
- The story does not match `data/expected_greedy.txt` → check the flags: only `-t 0.0` (greedy decoding) is deterministic; any other temperature samples randomly.

## Done when

The `diff` in subtask 1 shows no difference, and `./main` prints the sizes of all three files (one per line) and exits 0. `solution.cpp` is the reference answer — peek if stuck, then close it and write your own.
