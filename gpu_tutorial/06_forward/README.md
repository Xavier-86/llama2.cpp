# Module 06: Assemble and validate forward <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

## Task

Match the device-side `forward(token, pos)` call sequence line by line with
`cpu/run.cpp:232`. Here `w.` and `s.` are device pointers and
`loff = l * seq_len * kvd`.

```cpp
embed(x, w.token_embedding_table, token, dim);
for (int l = 0; l < n_layers; l++) {
    float* k = s.key_cache + loff + (size_t)pos * kvd;   // Current cache slice.
    float* v = s.value_cache + loff + (size_t)pos * kvd;

    rmsnorm(s.xb, s.x, w.rms_att_weight + l*dim);
    matmul_gpu(s.q, s.xb, w.wq + l*dim*dim, dim, dim);
    matmul_gpu(k,   s.xb, w.wk + l*dim*kvd, dim, kvd);
    matmul_gpu(v,   s.xb, w.wv + l*dim*kvd, dim, kvd);
    rope(s.q, k, pos, dim, kvd, head_size);
    attention(s.xb, s.q, s.key_cache, s.value_cache, s.att, pos, ...);
    matmul_gpu(s.xb2, s.xb, w.wo + l*dim*dim, dim, dim);
    add(s.x, s.xb2, dim);

    rmsnorm(s.xb, s.x, w.rms_ffn_weight + l*dim);
    matmul_gpu(s.hb,  s.xb, w.w1 + l*dim*hidden_dim, dim, hidden_dim);
    matmul_gpu(s.hb2, s.xb, w.w3 + l*dim*hidden_dim, dim, hidden_dim);
    swiglu(s.hb, s.hb2, hidden_dim);
    matmul_gpu(s.xb, s.hb, w.w2 + l*hidden_dim*dim, hidden_dim, dim);
    add(s.x, s.xb, dim);
}
rmsnorm(s.x, s.x, w.rms_final_weight);
matmul_gpu(s.logits, s.x, w.wcls, dim, vocab_size);
// Copy logits to the host and pass them to the unchanged Sampler.
```

Copy every weight-slice offset (`l*dim*dim`, `l*dim*kvd`, and so on) directly
from the CPU implementation.

## Alignment validation

```bash
./cpu/runcpp models/stories15M.bin -t 0.0 -n 128 -s 42 -i "Once upon a time" 2>/dev/null > /tmp/cpu.txt
./gpu/rungpu models/stories15M.bin -t 0.0 -n 128 -s 42 -i "Once upon a time" 2>/dev/null > /tmp/gpu.txt
diff /tmp/cpu.txt /tmp/gpu.txt
```

If output differs, do not debug the assembled forward pass blindly. After every
operator in layer 0, copy activations to the host and compare each value with a CPU
dump. The first operator that diverges contains the bug. A useful GPU helper is:

```cpp
// Print the first n elements of a device vector for debugging.
void dump_dev(const char* name, const float* d, int n) {
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("%s:", name);
    for (float v : h) printf(" %.6f", v);
    printf("\n");
}
```

## Acceptance criteria

`diff /tmp/cpu.txt /tmp/gpu.txt` produces no output.

Logits may differ around 1e-5 because GPU reduction order differs, but greedy argmax
text should normally match exactly. A logits error above 1e-2 is a bug, not numerical
noise; locate it with per-operator dumps.

## Files

- `main.cu` — weight upload, six kernels, `matmul_gpu`, generate/chat, and CLI
  are provided; the only TODO is the `GpuTransformer::forward` call sequence
- `solution.cu` — reference implementation and finished form of `gpu/rungpu.cu`;
  host code is reused through `#include "../../cpu/run.cpp"`
- `data/expected_gen.txt` — golden text from the CPU greedy-decoding command below

```bash
# Build:  nvcc -O3 -std=c++20 -arch=sm_89 -o main main.cu -lcublas
# Run:    ./main ../../../models/stories15M.bin -z ../../../models/tokenizer.bin \
#             -t 0.0 -n 128 -s 42 -i "Once upon a time" 2>/dev/null > out_gen.txt
# Verify: python3 ../../cpu_tutorial/tools/compare.py out_gen.txt data/expected_gen.txt --text
```

Rare greedy-text mismatches can occur when a 1e-5 numerical difference flips two
nearly tied logits. Use per-operator dumps to distinguish that edge case from a bug;
exact text remains the target.
