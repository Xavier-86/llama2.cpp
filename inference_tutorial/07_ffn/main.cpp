// 07_ffn -- SwiGLU feed-forward network (student template)
//
//   h1 = W1 @ x          (hidden_dim, dim) @ (dim,) -> (hidden_dim,)
//   h3 = W3 @ x          same, second branch
//   h  = silu(h1) * h3   elementwise; silu(v) = v * sigmoid(v)
//   out = W2 @ h         (dim, hidden_dim) @ (hidden_dim,) -> (dim,)
//
// Layer 0 of stories15M (dim=288, hidden_dim=768); the input is the real
// forward-pass value after rms_ffn normalization at the last prompt position.
//
// The weights come from the checkpoint via ../common/checkpoint.h and the
// input x is the const array kX in data.h — no file parsing in this file.
// Outputs go to out_h.txt / out.txt so you can verify them against
// data/expected_*.txt with ../tools/compare.py.
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main                (from this module folder)
// Verify: python3 ../tools/compare.py out_h.txt data/expected_hidden.txt
//         python3 ../tools/compare.py out.txt data/expected_out.txt

#include <cmath>
#include <iostream>
#include <span>
#include <vector>

#include "../common/checkpoint.h"
#include "../common/io.h"
#include "data.h"

// ---------------------------------------------------------------------------
// matmul: given (module 04 builds it from scratch; here it is just a tool)
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------

int main() {
    // Load the checkpoint once; every weight is a span view into its buffer.
    const tut::Checkpoint ckpt = tut::load_checkpoint("../../models/stories15M.bin");
    const size_t dim = ckpt.config.dim;
    const size_t hidden = ckpt.config.hidden_dim;

    // TODO(task 1): slice layer 0's W1/W2/W3 out of the weight spans
    //   (ckpt.weights.w1/w2/w3 hold all layers back to back; layer L starts
    //   at offset L * per-layer-size — here L = 0). Mind each tensor's
    //   shape from the README: W1 (hidden, dim), W2 (dim, hidden),
    //   W3 (hidden, dim).
    const std::span<const float> w1; // stub: empty slice
    const std::span<const float> w2; // stub: empty slice
    const std::span<const float> w3; // stub: empty slice
    (void)w1;
    (void)w2;
    (void)w3;

    // TODO(task 2): compute the two up-projections h1 = W1 @ x and h3 = W3 @ x
    //   with matmul(). x is kX from data.h: the FFN input after rms_ffn
    //   normalization at the last prompt position — no loading needed.
    std::vector<float> h1(hidden, 0.0f); // stub: all zeros
    std::vector<float> h3(hidden, 0.0f); // stub: all zeros
    (void)h3;

    // TODO(task 3): apply the SwiGLU gate elementwise: h = silu(h1) * h3
    //   (silu defined in the README; use the float overload of exp),
    //   leaving the result in h1.
    tut::write_floats("out_h.txt", h1);

    // TODO(task 4): down-project: out = W2 @ h (this is the value before
    //   the residual add), then write out.txt.
    std::vector<float> out(dim, 0.0f); // stub: all zeros
    tut::write_floats("out.txt", out);

    std::cout << "wrote out_h.txt (" << hidden << ") and out.txt (" << dim << ")\n";
    return 0;
}
