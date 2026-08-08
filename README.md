# llama2.cpp: Minimal LLM Inference in Pure C++ <span style="float: right;"><a href="README_zh.md">中文</a></span>

> A learning-oriented Llama 2 inference implementation with both CPU and GPU backends, based on [karpathy/llama2.c](https://github.com/karpathy/llama2.c).
> The CPU backend provides FP32 and int8 inference in modern C++20. The GPU backend implements the corresponding inference paths with CUDA, using cuBLAS and hand-written kernels. Both backends share the same model format and command-line interface, making it easy to compare their implementations, numerical results, and performance side by side.

## Project structure

```
├── cpu/                  CPU inference: run.cpp (FP32), runq.cpp (int8), quantize.cpp (converter)
├── gpu/                  GPU inference: rungpu.cu (FP32), runqgpu.cu (int8) — cuBLAS + hand-written kernels
├── models/               weights & tokenizer: stories15M/42M (.bin, FP32), stories*-q32.bin (int8), tokenizer.bin
├── cpu_tutorial/         CPU step-by-step tutorial: rebuild the whole pipeline in 12 verifiable modules
├── gpu_tutorial/         GPU step-by-step tutorial: port the forward pass to CUDA in 8 modules
└── docs/                 detailed docs: usage, quantization, debugging
```

## Requirements

- **Compiler with C++20** (`std::span`, `std::ranges`, `std::from_chars`):
  - macOS: system `/usr/bin/clang++` (Apple Clang 15+, Xcode Command Line Tools)
  - Linux: GCC 11+ or Clang 14+ — e.g. `sudo apt install g++-12`. If the default `c++` is older (Ubuntu's GCC 9.5 doesn't even accept `-std=c++20`), call the versioned binary explicitly: `g++-12 -O3 -std=c++20 ...`
- **Model weights**: download `stories15M.bin` / `stories42M.bin` from [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) into `models/`. The weights are managed with Git LFS; make sure the complete files have been fetched. `models/tokenizer.bin` is included.

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
- [cpu_tutorial/](cpu_tutorial/README.md) — hands-on tutorial: re-implement FP32 and int8 inference module by module with golden test data
- [gpu_tutorial/](gpu_tutorial/README_zh.md) — port the forward pass to GPU (cuBLAS + hand-written CUDA kernels) in 8 modules
