# 00 setup: build the reference, get a baseline <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

> Goal: compile the reference implementation, run it once, and hold on to the
> "correct answer" that every later module will approximate.

## Tasks

```bash
cd ../..   # repository root (llama2_cpp)
c++ -O3 -std=c++20 -o runcpp run.cpp
c++ -O3 -std=c++20 -o runqcpp runq.cpp
./runcpp stories15M.bin -t 0.0 -n 64 -s 42 -i "Once upon a time"
./runqcpp stories15M-q32.bin -t 0.0 -n 64 -s 42 -i "Once upon a time"
```

## Verification

The first command's output must match `data/expected_greedy.txt` exactly
(temperature=0 is greedy decoding, so the result is deterministic):

```bash
diff <(./runcpp stories15M.bin -t 0.0 -n 64 -s 42 -i "Once upon a time" 2>/dev/null) \
     inference_tutorial/00_setup/data/expected_greedy.txt
```

`main.cpp` / `solution.cpp` in this module do a minimal sanity check instead:
open the model files and print their sizes.

## Two things to observe (the physical intuition behind this whole project)

1. Look at the reported `achieved tok/s`. The 15M model does ~100+ tok/s on CPU.
2. Do the FP32 and int8 builds print the same story for the same prompt? Think
   about why quantization can (or cannot) preserve the greedy path.

## Background (just note it for now; later modules unpack each point)

- This program does **forward inference only**: load weights, generate tokens
  one by one. No training.
- Each `forward(token, pos)` call processes one token and outputs 32000 scores
  (logits) over the vocabulary; the sampler picks the next token.
- In the decode phase every step reads all weights once, so speed is roughly
  memory bandwidth ÷ bytes read per step — that is exactly why quantization
  (module 12) speeds things up.
