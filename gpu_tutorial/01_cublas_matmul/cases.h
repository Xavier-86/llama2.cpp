// 01_cublas_matmul/cases.h — test cases and CPU reference (pure C++, shared by
// main.cu / solution.cu / tools/gen_data.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace gt {

// Deterministic pseudorandom values (LCG) in [-1, 1). Seeded inputs make golden
// data reproducible so CPU-generated expectations can be compared with GPU output.
inline float frand(std::uint32_t& st) {
    st = st * 1664525u + 1013904223u;
    return ((st >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

// y = W x, where W is row-major d×n (matching matmul semantics in cpu/run.cpp).
inline void matmul_cpu(float* y, const float* x, const float* w, int n, int d) {
    for (int i = 0; i < d; i++) {
        float acc = 0.0f;
        for (int j = 0; j < n; j++) { acc += w[(size_t)i * n + j] * x[j]; }
        y[i] = acc;
    }
}

// ---- Toy case: d=4, n=3; easy to verify by hand ----
inline constexpr int kToyN = 3, kToyD = 4;
inline constexpr float kToyW[12] = {
    1.0f, -2.0f, 0.5f,
    0.0f,  1.0f, 1.0f,
    -1.5f, 0.0f, 2.0f,
    3.0f, -1.0f, -0.5f,
};
inline constexpr float kToyX[3] = {2.0f, -1.0f, 0.5f};

// ---- Real-size case: n=288, d=768 (the stories15M w1 shape) ----
inline constexpr int kRealN = 288, kRealD = 768;
inline constexpr std::uint32_t kRealSeedW = 12345u, kRealSeedX = 67890u;

inline std::vector<float> make_real_w() {
    std::vector<float> w((size_t)kRealN * kRealD);
    std::uint32_t st = kRealSeedW;
    for (float& v : w) v = frand(st);
    return w;
}
inline std::vector<float> make_real_x() {
    std::vector<float> x(kRealN);
    std::uint32_t st = kRealSeedX;
    for (float& v : x) v = frand(st);
    return x;
}

} // namespace gt
