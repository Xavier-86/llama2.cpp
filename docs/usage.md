# Usage <span style="float: right;"><a href="usage_zh.md">中文</a></span>

```bash
./cpu/runcpp <checkpoint> [options]
```

Examples:

```bash
# greedy decoding (deterministic output, temperature=0)
./cpu/runcpp models/stories15M.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"

# sampling
./cpu/runcpp models/stories42M.bin -t 0.8 -p 0.9 -n 256 -s 42 -i "One day, a little girl named Lily"

# chat mode
./cpu/runcpp models/stories15M.bin -m chat -n 256

# quantized model (use runqcpp)
./cpu/runqcpp models/stories15M-q32.bin -t 0.0 -n 256 -s 42 -i "Once upon a time"
```

Options:

| Flag | Meaning | Default |
| --- | --- | --- |
| `-t <float>` | temperature, 0 = greedy deterministic output | 1.0 |
| `-p <float>` | top-p (nucleus sampling), 1.0 = off | 0.9 |
| `-s <int>` | random seed | current time |
| `-n <int>` | number of steps, 0 = max sequence length | 256 |
| `-i <string>` | input prompt | empty |
| `-z <string>` | custom tokenizer path | models/tokenizer.bin |
| `-m <string>` | mode: generate or chat | generate |
| `-y <string>` | system prompt in chat mode | none |

When generation finishes, stderr prints `achieved tok/s`, which you can use to compare FP32 vs int8 speed.

## Files

| File | Description |
| --- | --- |
| `cpu/run.cpp` | FP32 inference, the main file. Forward pass, KV cache, BPE tokenizer, and sampling, all in one file |
| `cpu/runq.cpp` | int8 quantized inference. The only difference from `run.cpp` is quantization: `QuantizedTensor` (int8 + per-group scale factors), int8 matmul, and quantizing activations before every matmul |
| `cpu/quantize.cpp` | Checkpoint converter: turns an FP32 `.bin` into the int8 format runq reads (only needs C++17) |
| `models/tokenizer.bin` | BPE tokenizer data (Llama 2 32K vocab) |
| `models/stories15M.bin` / `models/stories42M.bin` | FP32 model weights (TinyStories models from [karpathy/tinyllamas](https://huggingface.co/karpathy/tinyllamas)) |
| `models/stories*-q32.bin` | int8 quantized weights (produced by `quantize`, GS=32) |
