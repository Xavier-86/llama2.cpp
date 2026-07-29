// 07_ffn -- SwiGLU feed-forward network (reference solution)
//
//   h1 = W1 @ x          (hidden_dim, dim) @ (dim,) -> (hidden_dim,)
//   h3 = W3 @ x          same, second branch
//   h  = silu(h1) * h3   elementwise; silu(v) = v / (1 + exp(-v))
//   out = W2 @ h         (dim, hidden_dim) @ (hidden_dim,) -> (dim,)
//
// Layer 0 of stories15M (dim=288, hidden_dim=768); the input is the real
// forward-pass value after rms_ffn normalization at the last prompt position.
// Weights come from ../common/checkpoint.h, x is kX in data.h.
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution            (from this module folder)
// Verify: python3 ../tools/compare.py out_h.txt data/expected_hidden.txt
//         python3 ../tools/compare.py out.txt data/expected_out.txt

#include <cmath>
#include <iostream>
#include <span>
#include <vector>

#include "../common/checkpoint.h"
#include "../common/io.h"
#include "data.h"

// W (d,n) @ x (n,) -> xout (d,); w holds d rows of n elements, row-major.
// Dimensions are derived from the span sizes, so the orientation of W1/W2/W3
// cannot be flipped by accident. Accumulate in float, matching the reference.
void matmul(std::span<float> xout, std::span<const float> x, std::span<const float> w) {
    const size_t n = x.size();
    const size_t d = xout.size();
    for (size_t i = 0; i < d; i++) {
        const std::span w_row = w.subspan(i * n, n);
        float val = 0.0f;
        for (size_t j = 0; j < n; j++) { val += w_row[j] * x[j]; }
        xout[i] = val;
    }
}

int main() {
    // Load the checkpoint once; every weight is a span view into its buffer.
    const tut::Checkpoint ckpt = tut::load_checkpoint("../../stories15M.bin");
    const size_t dim = ckpt.config.dim;
    const size_t hidden = ckpt.config.hidden_dim;

    // layer 0's FFN weights: layer L starts at offset L * (per-layer size)
    const size_t layer = 0;
    const std::span<const float> w1 = ckpt.weights.w1.subspan(layer * hidden * dim, hidden * dim);
    const std::span<const float> w2 = ckpt.weights.w2.subspan(layer * dim * hidden, dim * hidden);
    const std::span<const float> w3 = ckpt.weights.w3.subspan(layer * hidden * dim, hidden * dim);

    // two up-projections; x is kX from data.h (after rms_ffn normalization)
    std::vector<float> h1(hidden), h3(hidden);
    matmul(h1, kX, w1);
    matmul(h3, kX, w3);

    // SwiGLU: h = silu(h1) * h3, silu(v) = v * sigmoid(v) = v / (1 + exp(-v))
    for (size_t i = 0; i < hidden; i++) {
        float v = h1[i];
        v *= 1.0f / (1.0f + std::exp(-v)); // float overload of exp
        v *= h3[i];
        h1[i] = v;
    }
    tut::write_floats("out_h.txt", h1);

    // down-projection (before the residual add)
    std::vector<float> out(dim);
    matmul(out, h1, w2);
    tut::write_floats("out.txt", out);

    std::cout << "wrote out_h.txt (" << hidden << ") and out.txt (" << dim << ")\n";
    return 0;
}
