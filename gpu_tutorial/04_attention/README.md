# Module 04: Attention kernel and KV cache <span style="float: right;"><a href="README_zh.md">中文</a></span>

[← All modules](../README.md) · [Project README](../../README.md)

## Concept

Attention is the only stateful operator and the most complex kernel in this tutorial.
For each head, the CPU code (`cpu/run.cpp:278`) computes dot-product scores against
all cached keys in `[0, pos]`, applies softmax, and forms a weighted sum of cached
values. The KV cache avoids recomputing keys and values for earlier tokens.

The naive GPU implementation assigns one block to each head and places all three
stages in one kernel, separated by `__syncthreads()`.

## Task

```cpp
// One block handles one head. att is an (n_heads, seq_len) score buffer.
__global__ void attention_kernel(float* xb, const float* q,
                                 const float* key_cache, const float* value_cache,
                                 float* att, int pos,
                                 int kvd, int kv_mul,
                                 int head_size, int seq_len, size_t layer_off) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const float* qh = q + (size_t)h * head_size;
    float* att_h = att + (size_t)h * seq_len;
    const float inv_sqrt_hs = rsqrtf((float)head_size);
    const int kv_head = h / kv_mul;  // Multiple query heads may share one KV head.

    // 1) Compute q·k / sqrt(head_size) for assigned history positions.
    for (int t = tid; t <= pos; t += blockDim.x) {
        const float* key = key_cache + layer_off + (size_t)t * kvd
                         + (size_t)kv_head * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) score += qh[i] * key[i];
        att_h[t] = score * inv_sqrt_hs;
    }
    __syncthreads();

    // 2) Two-pass single-thread softmax; subtract max to prevent overflow.
    if (tid == 0) {
        float mx = att_h[0];
        for (int t = 1; t <= pos; t++) mx = fmaxf(mx, att_h[t]);
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) {
            att_h[t] = expf(att_h[t] - mx);
            sum += att_h[t];
        }
        for (int t = 0; t <= pos; t++) att_h[t] /= sum;
    }
    __syncthreads();

    // 3) Weighted sum of values; thread i handles output component i.
    for (int i = tid; i < head_size; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const float* val = value_cache + layer_off + (size_t)t * kvd
                             + (size_t)kv_head * head_size;
            acc += att_h[t] * val[i];
        }
        xb[(size_t)h * head_size + i] = acc;
    }
}
```

Launch with `attention_kernel<<<n_heads, 128>>>(...)`.

**Writing the KV cache:** the current token's k/v must already be in the cache before
the attention kernel. Point the qkv projection output directly at
`key_cache + loff + pos*kvd` and the matching value slice, just as the CPU code
does.

Two common mistakes:

- `kv_mul`: stories15M is MHA (`kv_mul=1`), but larger Llama-2 models use GQA;
  select the KV head with `h / kv_mul`
- Softmax covers the closed interval **`[0, pos]`**, not all of `seq_len`;
  including uninitialized positions causes deterministic text divergence

## Acceptance criteria

Compare against a host implementation using a small case such as
`seq_len=8, pos=3, n_heads=2, head_size=4`. Maximum error must be below 1e-4.
Test both `pos=0` and `kv_mul>1` boundaries.

## Files

- `main.cu` — exercise template with upload/download harness for two cases and a
  TODO kernel stub
- `solution.cu` — the same harness with the complete reference kernel
- `cases.h` — Case A (MHA), Case B (GQA), LCG inputs, and a CPU reference

```bash
# Build (cuBLAS is not required)
nvcc -O2 -arch=sm_89 -o main main.cu
nvcc -O2 -arch=sm_89 -o solution solution.cu
# Run
./main        # or ./solution
# Verify
python3 ../../cpu_tutorial/tools/compare.py out_xb_a.txt  data/expected_xb_a.txt
python3 ../../cpu_tutorial/tools/compare.py out_att_a.txt data/expected_att_a.txt
python3 ../../cpu_tutorial/tools/compare.py out_xb_b.txt  data/expected_xb_b.txt
python3 ../../cpu_tutorial/tools/compare.py out_att_b.txt data/expected_att_b.txt
```
