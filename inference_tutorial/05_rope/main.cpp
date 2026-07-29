// 05_rope — rotary position embedding (student template)
//
// Goal: rotate adjacent pairs (v0, v1) of Q and K by pos * freq within each
// head, in place, for each position pos = 0..4.
//
// All inputs are const arrays in data.h (the pre-RoPE q/k projections for the
// 5 prompt positions, position-major) — no file parsing needed. Outputs go to
// out_q.txt / out_k.txt so you can verify them against data/expected_*.txt
// with ../tools/compare.py.
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main
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

// The inputs are kQ / kK from data.h: (P, dim) and (P, kv_dim), position-major —
// position pos occupies [pos*dim, (pos+1)*dim), head-major within a position.

// ---------------------------------------------------------------------------
// The kernel of this module
// ---------------------------------------------------------------------------

// Rotate Q (all dim pairs) and K (first kv_dim pairs) for one position, in place.
void rope(std::span<float> q, std::span<float> k, int pos) {
    // TODO(task 1): loop over the adjacent pairs: i = 0, 2, 4, ... dim-2.
    // TODO(task 2): for each pair, compute the pair index within its own head
    //   (head_dim = i % head_size), the per-pair frequency
    //   freq = 1 / 10000^(head_dim / head_size), and the angle = pos * freq.
    //   Use the float versions of std::cos / std::sin on the angle.
    // TODO(task 3): rotate the Q pair in place:
    //   (v0', v1') = (v0*cos(angle) - v1*sin(angle),
    //                 v0*sin(angle) + v1*cos(angle)).
    // TODO(task 4): rotate the K pair the same way, but only while i < kv_dim
    //   (here kv_dim == dim, so K ends up fully rotated; the KV cache stores
    //   post-rotation K).
    (void)q; (void)k; (void)pos; // stub: replace with your code
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
