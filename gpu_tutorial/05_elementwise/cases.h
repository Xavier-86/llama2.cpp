// 05_elementwise/cases.h — test cases and CPU reference (pure C++, shared by
// main.cu / solution.cu / tools/gen_data.cpp).
#pragma once

#include <cstddef>
#include <cmath>

namespace gt {

// ---- SwiGLU toy case: 8 elements including signed zeros; easy to verify by hand ----
// Matches cpu/run.cpp:322: first silu(v) = v / (1 + e^-v), then multiply by hb2.
inline constexpr int kSwigluN = 8;
inline constexpr float kSwigluHb[kSwigluN] = {
    1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, 0.0f, 3.0f,
};
inline constexpr float kSwigluHb2[kSwigluN] = {
    2.0f, 2.0f, -1.0f, 0.5f, 1.0f, 1.0f, -2.0f, 0.25f,
};

inline void swiglu_cpu(float* hb, const float* hb2, int n) {
    for (int i = 0; i < n; i++) {
        float v = hb[i];
        v *= 1.0f / (1.0f + std::exp(-v)); // silu(x) = x * sigmoid(x)
        v *= hb2[i];
        hb[i] = v;
    }
}

// ---- Residual-add toy case: x[i] += y[i], 8 elements ----
inline constexpr int kAddN = 8;
inline constexpr float kAddX[kAddN] = {
    1.0f, -1.0f, 0.5f, 0.0f, 2.0f, -2.0f, 1.5f, -0.5f,
};
inline constexpr float kAddY[kAddN] = {
    0.5f, 1.0f, -0.5f, 2.0f, -1.0f, 2.0f, 0.25f, 0.75f,
};

inline void add_cpu(float* x, const float* y, int n) {
    for (int i = 0; i < n; i++) { x[i] += y[i]; }
}

// ---- Embedding-lookup toy case: vocab=5 × dim=4, lookup token=3 ----
// Row v is token v's embedding: {v.0, v.1, v.2, v.3}, making it easy to inspect.
inline constexpr int kEmbedVocab = 5, kEmbedDim = 4, kEmbedToken = 3;
inline constexpr float kEmbedTable[kEmbedVocab * kEmbedDim] = {
    0.0f, 0.1f, 0.2f, 0.3f,
    1.0f, 1.1f, 1.2f, 1.3f,
    2.0f, 2.1f, 2.2f, 2.3f,
    3.0f, 3.1f, 3.2f, 3.3f,
    4.0f, 4.1f, 4.2f, 4.3f,
};

inline void embed_cpu(float* x, const float* table, int token, int dim) {
    for (int i = 0; i < dim; i++) { x[i] = table[(size_t)token * dim + i]; }
}

} // namespace gt
