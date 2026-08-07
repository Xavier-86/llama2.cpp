// 03_rope/cases.h — test cases and CPU reference (pure C++, shared by
// main.cu / solution.cu / tools/gen_data.cpp).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "../01_cublas_matmul/cases.h" // Reuse gt::frand (LCG).

namespace gt {

// RoPE treats adjacent q/k dimension pairs (v0, v1) as complex numbers and rotates
// them by pos * freq radians within each head. Pairs with i < kvd rotate both q and
// k (the rotn logic). This is a line-by-line port of cpu/run.cpp:262-276; q/k rotate in place.
inline void rope_cpu(float* q, float* k, int pos, int dim, int kvd, int head_size) {
    for (int i = 0; i < dim; i += 2) {
        const int head_dim = i % head_size;
        const float freq = 1.0f / std::pow(10000.0f, head_dim / static_cast<float>(head_size));
        const float val = pos * freq;
        const float fcr = std::cos(val);
        const float fci = std::sin(val);
        const int rotn = i < kvd ? 2 : 1; // 2 rotates q and k; 1 rotates only q.
        for (int rot = 0; rot < rotn; rot++) {
            float* vec = rot == 0 ? q : k;
            const float v0 = vec[i];
            const float v1 = vec[i + 1];
            vec[i]     = v0 * fcr - v1 * fci;
            vec[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

// ---- Toy case: dim=8, kvd=8, head_size=4, pos=7; easy to verify by hand ----
inline constexpr int kToyDim = 8, kToyKvd = 8, kToyHeadSize = 4, kToyPos = 7;
inline constexpr float kToyQ[8] = {
    1.0f, 0.0f, -1.0f, 0.5f, 0.25f, -0.75f, 2.0f, -1.0f,
};
inline constexpr float kToyK[8] = {
    0.5f, -0.5f, 1.5f, 0.0f, -0.25f, 1.0f, 0.0f, 0.75f,
};

// ---- Real-size case: dim=288, kvd=288, head_size=48, pos=42
// (stories15M q/k shape; MHA has kvd == dim, so all k dimensions rotate) ----
inline constexpr int kRealDim = 288, kRealKvd = 288, kRealHeadSize = 48, kRealPos = 42;
inline constexpr std::uint32_t kRealSeedQ = 12345u, kRealSeedK = 67890u;

inline std::vector<float> make_real_q() {
    std::vector<float> q(kRealDim);
    std::uint32_t st = kRealSeedQ;
    for (float& v : q) v = frand(st);
    return q;
}
inline std::vector<float> make_real_k() {
    std::vector<float> k(kRealKvd);
    std::uint32_t st = kRealSeedK;
    for (float& v : k) v = frand(st);
    return k;
}

} // namespace gt
