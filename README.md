# llama2.cpp: Minimal LLM Inference in Pure C++ <span style="float: right;"><a href="README_zh.md">中文</a></span>

> A modern C++ (C++20) rewrite of [karpathy/llama2.c](https://github.com/karpathy/llama2.c), for learning how LLM inference works.
> Differences from the original: RAII memory management (`std::vector` / mmap wrapper class), `std::span` for buffer passing, `std::string` / `std::string_view`, exceptions, iostreams, `std::sort` / `std::lower_bound`. Numerically equivalent to the original, but the code does not try to mirror it line by line.

## Contents

- [Files](#files)
- [Build](#build)
- [Run](#run)
- [Quantized Models](#quantized-models)
- [Step-by-Step Inference Tutorial](#step-by-step-inference-tutorial)
- [Debugging](#debugging)

## Files

| File | Description |
| --- | --- |
| `run.cpp` | FP32 inference, the main file. Forward pass, KV cache, BPE tokenizer, and sampling, all in one file |
| `runq.cpp` | int8 quantized inference. The only difference from `run.cpp` is quantization: `QuantizedTensor` (int8 + per-group scale factors), int8 matmul, and quantizing activations before every matmul |
| `quantize.cpp` | Checkpoint converter: turns an FP32 `.bin` into the int8 format runq reads |
| `tokenizer.bin` | BPE tokenizer data (Llama 2 32K vocab) |
| `stories15M.bin` / `stories42M.bin` | FP32 model weights (TinyStories models from [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas)) |
| `stories*-q32.bin` | int8 quantized weights (produced by `quantize`, GS=32) |
| [`inference_tutorial/`](inference_tutorial/README.md) | Step-by-step tutorial that decomposes FP32 and int8 inference into 12 independently verifiable modules |

Note: the `stories*.bin` weights are tracked in git. If you ever need to re-fetch them, download `stories15M.bin` / `stories42M.bin` from [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas) and place them in this directory; `tokenizer.bin` is included in the repo.

## Build

Compiler requirements: `run.cpp` / `runq.cpp` use C++20 features (`std::span`, `std::ranges`, `std::from_chars`, etc.), so you need a C++20-capable compiler:

- **macOS**: the system `/usr/bin/clang++` (Apple Clang 15+; just install the Xcode Command Line Tools, no extra compiler needed). Developed and verified on Apple Clang 17.
- **Other platforms**: GCC 11+ or Clang 14+.

Note that `run.cpp` and `runq.cpp` also rely on POSIX `mmap` (`<sys/mman.h>`), so they do not build with MSVC on Windows. `quantize.cpp` only needs C++17.

```bash
c++ -O3 -std=c++20 -o runcpp run.cpp        # FP32 inference
c++ -O3 -std=c++20 -o runqcpp runq.cpp      # int8 quantized inference
c++ -O3 -std=c++17 -o quantize quantize.cpp # quantization converter
```

With VSCode, the language standard is configured separately for IntelliSense and for build tasks in `.vscode/c_cpp_properties.json` and `.vscode/tasks.json` (currently set to `-std=c++20`). If you change the standard here, change it in both places too — otherwise the editor will falsely report errors like "no member named `span` in namespace `std`".

## Run

```bash
./runcpp <checkpoint> [options]
```

Examples:

```bash
# greedy decoding (deterministic output, temperature=0)
./runcpp stories15M.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"

# sampling
./runcpp stories42M.bin -t 0.8 -p 0.9 -n 256 -s 42 -i "One day, a little girl named Lily"

# chat mode
./runcpp stories15M.bin -m chat -n 256

# quantized model (use runqcpp)
./runqcpp stories15M-q32.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"
```

Options:

| Flag | Meaning | Default |
| --- | --- | --- |
| `-t <float>` | temperature, 0 = greedy deterministic output | 1.0 |
| `-p <float>` | top-p (nucleus sampling), 1.0 = off | 0.9 |
| `-s <int>` | random seed | current time |
| `-n <int>` | number of steps, 0 = max sequence length | 256 |
| `-i <string>` | input prompt | empty |
| `-z <string>` | custom tokenizer path | tokenizer.bin |
| `-m <string>` | mode: generate or chat | generate |
| `-y <string>` | system prompt in chat mode | none |

When generation finishes, stderr prints `achieved tok/s`, which you can use to compare FP32 vs int8 speed.

## Quantized Models

Convert an FP32 checkpoint to int8 (the group size must divide every weight dimension; 32 works for all stories models):

```bash
./quantize stories15M.bin stories15M-q32.bin 32
./runqcpp stories15M-q32.bin -t 0.0 -n 256 -i "Once upon a time"
```

Measured on Apple Silicon: 58 MB → 16 MB, tok/s roughly 138 → 1343. Cutting weight reads to 1/4 is the main reason decode gets faster — a direct demonstration that decode is memory-bandwidth bound.

Note: the int8 quantized path is very sensitive to the compiler's optimization-level floating-point codegen (FMA contraction, vectorized reduction order) — a 1-ulp difference in an activation can flip an int8 rounding, which in turn can send greedy decoding down a different (equally valid) text at some near-tie argmax. So the exact text runq generates may differ across compile options; this is expected. FP32 run is not affected.

## Step-by-Step Inference Tutorial

[`inference_tutorial/`](inference_tutorial/README.md) is the hands-on learning path for
this repository. It decomposes the two all-in-one inference programs into 12
small modules:

1. checkpoint loading and BPE tokenization
2. RMSNorm, softmax, matmul, RoPE, attention, and SwiGLU FFN
3. the complete FP32 forward pass, sampler, and generation loop
4. int8 checkpoint mapping, group quantization, int8 matmul, and quantized inference

Every module contains a concept guide, a `main.cpp` exercise template, a
`solution.cpp` reference implementation, and golden input/output data. This
lets you implement one piece at a time and compare its output before assembling
the complete model. Generated `out*.txt` files and locally compiled module
executables are ignored by git; files under each module's `data/` directory
remain versioned.

## Debugging

The bundled debugger on macOS is **lldb** (there is no gdb). Workflow: first build with debug flags (`-g` keeps symbols, `-O0` disables optimization — otherwise variables get optimized away and breakpoint line numbers won't match):

```bash
c++ -g -O0 -std=c++20 -o runcpp_dbg run.cpp
lldb ./runcpp_dbg -- stories15M.bin -t 0.0 -n 8 -i "Once upon a time"
```

Common lldb commands:

| Command | Effect |
| --- | --- |
| `b main` / `b run.cpp:650` | Break at a function name / file line (line numbers drift as code evolves; prefer function names) |
| `b Transformer::forward` | Break at a member function (recommended when learning inference) |
| `run` (`r`) | Start the program |
| `next` (`n`) | Step over |
| `step` (`s`) | Step into |
| `finish` | Run to the end of the current function |
| `continue` (`c`) | Continue to the next breakpoint |
| `p x` / `p config.dim` | Print a variable / member |
| `p s.x` / `p s.q` | Show a whole buffer (buffers are `std::vector`; lldb renders them as arrays) |
| `bt` | Show the call stack |
| `watch set var pos` | Stop automatically when a variable changes |
| `quit` (`q`) | Quit |

Recommended breakpoint set for learning inference:

```lldb
b Transformer::forward        # stop once per generated token, watch x/q/k/v change layer by layer
b Sampler::sample             # see how logits become the next token
b Tokenizer::encode           # see how the prompt becomes a token sequence
```

Inside `forward` you can use `p pos` for the current position and `p state.x` for the whole activation buffer (lldb renders `std::vector` as an array), stepping through layers with `finish` — much faster than just reading the code.

GUI alternative: install the **CodeLLDB** extension in VSCode, set breakpoints in the gutter, press F5 — same effect with a UI.
