// 02_rmsnorm/cases.h — test cases and CPU reference (pure C++, shared by
// main.cu / solution.cu / tools/gen_data.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <vector>

#include "../01_cublas_matmul/cases.h" // Reuse gt::frand (LCG pseudorandom generator).

namespace gt {

// o[i] = w[i] * x[i] / sqrt(mean(x²) + eps), exactly matching rmsnorm at
// cpu/run.cpp:198 (eps = 1e-5f; reduce mean square, then scale each element).
inline void rmsnorm_cpu(float* o, const float* x, const float* weight, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) { ss += x[i] * x[i]; }
    ss /= n;
    ss += 1e-5f;
    ss = 1.0f / std::sqrt(ss);
    for (int j = 0; j < n; j++) { o[j] = weight[j] * (ss * x[j]); }
}

// ---- Toy case: n=8; easy to verify by hand (ss = 35.5/8 = 4.4375) ----
// Names avoid kToyN/kToyX/kToyW already declared by module 01's cases.h.
inline constexpr int kToySize = 8;
inline constexpr float kToyVec[8] = {1.0f, -2.0f, 3.0f, -4.0f, 0.5f, -0.5f, 2.0f, -1.0f};
inline constexpr float kToyWeight[8] = {0.5f, 1.5f, -1.0f, 2.0f, 0.25f, -0.75f, 1.0f, 1.0f};

// ---- Real-size case: n=288 (stories15M dim), with separate x and weight seeds ----
inline constexpr int kRealSize = 288;
inline constexpr std::uint32_t kRealSeedVec = 24680u, kRealSeedWeight = 13579u;

inline std::vector<float> make_real_vec() {
    std::vector<float> x(kRealSize);
    std::uint32_t st = kRealSeedVec;
    for (float& v : x) v = frand(st);
    return x;
}
inline std::vector<float> make_real_weight() {
    std::vector<float> w(kRealSize);
    std::uint32_t st = kRealSeedWeight;
    for (float& v : w) v = frand(st);
    return w;
}

} // namespace gt
