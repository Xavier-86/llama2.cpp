# llama2.cpp: Minimal LLM Inference in Pure C++ <span style="float: right;"><a href="README_zh.md">中文</a></span>

> A modern C++ (C++20) rewrite of [karpathy/llama2.c](https://github.com/karpathy/llama2.c), for learning how LLM inference works.
> Differences from the original: RAII memory management (`std::vector` / mmap wrapper class), `std::span` for buffer passing, `std::string` / `std::string_view`, exceptions, iostreams, `std::sort` / `std::lower_bound`. Numerically equivalent to the original, but the code does not try to mirror it line by line.

## Project structure

```
├── cpu/                  CPU inference: run.cpp (FP32), runq.cpp (int8), quantize.cpp (converter)
├── gpu/                  GPU inference on cuBLAS (planned, not implemented yet)
├── models/               weights & tokenizer: stories15M/42M (.bin, FP32), stories*-q32.bin (int8), tokenizer.bin
├── inference_tutorial/   step-by-step tutorial: rebuild the whole pipeline in 12 verifiable modules
└── docs/                 detailed docs: usage, quantization, debugging
```

## Requirements

- **Compiler with C++20** (`std::span`, `std::ranges`, `std::from_chars`):
  - macOS: system `/usr/bin/clang++` (Apple Clang 15+, Xcode Command Line Tools)
  - Linux: GCC 11+ or Clang 14+ — e.g. `sudo apt install g++-12`. If the default `c++` is older (Ubuntu's GCC 9.5 doesn't even accept `-std=c++20`), call the versioned binary explicitly: `g++-12 -O3 -std=c++20 ...`
  - Windows: `run.cpp` / `runq.cpp` need POSIX `mmap`, so no MSVC
- **Model weights**: download `stories15M.bin` / `stories42M.bin` from [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) into `models/` (they are tracked in git via LFS; if your clone only has ~130-byte pointer files, re-download). `models/tokenizer.bin` is included.

## Quick start

```bash
# build (on Linux replace c++ with g++-12 if needed, see above)
c++ -O3 -std=c++20 -o cpu/runcpp cpu/run.cpp        # FP32 inference
c++ -O3 -std=c++20 -o cpu/runqcpp cpu/runq.cpp      # int8 quantized inference
c++ -O3 -std=c++17 -o cpu/quantize cpu/quantize.cpp # quantization converter

# run
./cpu/runcpp models/stories15M.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"
./cpu/runqcpp models/stories15M-q32.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"

# produce an int8 checkpoint yourself
./cpu/quantize models/stories15M.bin models/stories15M-q32.bin 32
```

When generation finishes, stderr prints `achieved tok/s`, which you can use to compare FP32 vs int8 speed.

## Documentation

- [docs/usage.md](docs/usage.md) — all CLI options, generate/chat modes, examples
- [docs/quantization.md](docs/quantization.md) — int8 format, group size, speed/size trade-offs
- [docs/debugging.md](docs/debugging.md) — lldb (macOS) / gdb (Linux) / VSCode debugging guide
- [inference_tutorial/](inference_tutorial/README.md) — hands-on tutorial: re-implement FP32 and int8 inference module by module with golden test data
