// 04_attention/cases.h — test cases and CPU reference (pure C++, shared by
// main.cu / solution.cu / tools/gen_data.cpp).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../01_cublas_matmul/cases.h" // Reuse gt::frand (LCG).

namespace gt {

// Fill n floats with the LCG. The KV cache fills all seq_len*kvd values, including
// positions after pos, so any kernel read outside [0, pos] is immediately visible.
inline std::vector<float> make_filled(size_t n, std::uint32_t seed) {
    std::vector<float> v(n);
    std::uint32_t st = seed;
    for (float& x : v) x = frand(st);
    return v;
}

// CPU reference for one attention layer, ported from cpu/run.cpp:278-306 (layer_off=0).
// q: n_heads*head_size；key_cache/value_cache: seq_len*kvd。
// Output xb has n_heads*head_size values. att_out concatenates each head's softmax
// weights for [0..pos], for n_heads*(pos+1) values total.
inline void attention_cpu(float* xb, float* att_out, const float* q,
                          const float* key_cache, const float* value_cache,
                          int pos, int n_heads, int kvd, int kv_mul,
                          int head_size, int seq_len) {
    std::vector<float> att(seq_len);
    for (int h = 0; h < n_heads; h++) {
        const float* qh = q + (size_t)h * head_size;
        const int kv_head = h / kv_mul; // In GQA/MQA, multiple query heads share one KV head.
        for (int t = 0; t <= pos; t++) {
            const float* key = key_cache + (size_t)t * kvd + (size_t)kv_head * head_size;
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) score += qh[i] * key[i];
            att[t] = score / std::sqrt((float)head_size);
        }
        // Softmax over the closed interval [0, pos]; subtract max to prevent overflow.
        float mx = att[0];
        for (int t = 1; t <= pos; t++) mx = std::fmax(mx, att[t]);
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) { att[t] = std::exp(att[t] - mx); sum += att[t]; }
        for (int t = 0; t <= pos; t++) {
            att[t] /= sum;
            att_out[(size_t)h * (pos + 1) + t] = att[t];
        }
        // Weighted sum of v.
        float* xb_h = xb + (size_t)h * head_size;
        for (int i = 0; i < head_size; i++) xb_h[i] = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const float* val = value_cache + (size_t)t * kvd + (size_t)kv_head * head_size;
            const float a = att[t];
            for (int i = 0; i < head_size; i++) xb_h[i] += a * val[i];
        }
    }
}

// ---- Case A: MHA (kv_mul=1, stories15M shape) ----
inline constexpr int kANHeads = 2, kANKVHeads = 2, kAKVMul = 1;
inline constexpr int kAHeadSize = 4, kASeqLen = 8, kAPos = 3;
inline constexpr int kAKvd = kAHeadSize * kANKVHeads; // = 8
inline constexpr std::uint32_t kASeedQ = 111u, kASeedK = 222u, kASeedV = 333u;

inline std::vector<float> make_a_q() { return make_filled((size_t)kANHeads * kAHeadSize, kASeedQ); }
inline std::vector<float> make_a_k() { return make_filled((size_t)kASeqLen * kAKvd, kASeedK); }
inline std::vector<float> make_a_v() { return make_filled((size_t)kASeqLen * kAKvd, kASeedV); }

// ---- Case B: GQA (kv_mul=2, shape used by llama2-7B and larger) ----
inline constexpr int kBNHeads = 4, kBNKVHeads = 2, kBKVMul = 2;
inline constexpr int kBHeadSize = 4, kBSeqLen = 8, kBPos = 5;
inline constexpr int kBKvd = kBHeadSize * kBNKVHeads; // = 8
inline constexpr std::uint32_t kBSeedQ = 444u, kBSeedK = 555u, kBSeedV = 666u;

inline std::vector<float> make_b_q() { return make_filled((size_t)kBNHeads * kBHeadSize, kBSeedQ); }
inline std::vector<float> make_b_k() { return make_filled((size_t)kBSeqLen * kBKvd, kBSeedK); }
inline std::vector<float> make_b_v() { return make_filled((size_t)kBSeqLen * kBKvd, kBSeedV); }

} // namespace gt
