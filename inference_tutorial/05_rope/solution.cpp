// 05_rope — rotary position embedding (reference solution)
//
// Rotates adjacent pairs (v0, v1) of Q and K by pos * freq within each head,
// in place, for each position pos = 0..4.
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution
// Verify: python3 ../tools/compare.py out_q.txt data/expected_q.txt
//         python3 ../tools/compare.py out_k.txt data/expected_k.txt

#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>

#include "../common/io.h"
#include "data.h"

// ---------------------------------------------------------------------------
// Model constants (stories15M)
// ---------------------------------------------------------------------------

constexpr int kDim = 288;       // model dim
constexpr int kHeadSize = 48;   // dim / n_heads = 288 / 6
constexpr int kKvDim = 288;     // == dim here; < dim only for GQA models
constexpr int kPositions = 5;   // positions in the input data (P = 5)

// ---------------------------------------------------------------------------
// The kernel of this module
// ---------------------------------------------------------------------------

// Rotate Q (all dim pairs) and K (first kv_dim pairs) for one position, in place.
void rope(std::span<float> q, std::span<float> k, int pos) {
    for (int i = 0; i < kDim; i += 2) {
        const int head_dim = i % kHeadSize;  // pair index within its own head
        const float freq =
            1.0f / std::pow(10000.0f, head_dim / static_cast<float>(kHeadSize));
        const float angle = pos * freq;
        const float fcr = std::cos(angle);
        const float fci = std::sin(angle);
        {
            const float v0 = q[i];
            const float v1 = q[i + 1];
            q[i]     = v0 * fcr - v1 * fci;
            q[i + 1] = v0 * fci + v1 * fcr;
        }
        if (i < kKvDim) {  // kv_dim == dim here, so K is fully rotated too
            const float v0 = k[i];
            const float v1 = k[i + 1];
            k[i]     = v0 * fcr - v1 * fci;
            k[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

// ---------------------------------------------------------------------------

int main() {
    // data.h arrays are const; rotate on mutable copies (the real code rotates
    // the q/k buffers in place).
    float q[kPositions * kDim];
    float k[kPositions * kDim];
    std::ranges::copy(kQ, q);
    std::ranges::copy(kK, k);

    // Data is position-major: position pos occupies [pos*dim, (pos+1)*dim).
    for (int pos = 0; pos < kPositions; pos++) {
        rope(std::span{q}.subspan(pos * kDim, kDim),
             std::span{k}.subspan(pos * kDim, kDim), pos);
    }

    tut::write_floats("out_q.txt", q);
    tut::write_floats("out_k.txt", k);
    std::cout << "wrote out_q.txt and out_k.txt (" << kPositions << " x " << kDim
              << " values)\n";
    return 0;
}
