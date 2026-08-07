// 04_matmul — reference solution
//
// The matrix-vector multiply that dominates llama2 inference: every linear
// projection (wq/wk/wv/wo, w1/w2/w3, the classifier) is one call to this
// kernel.
//
//   xout_i = sum_j  w[i*n + j] * x[j]      (w row-major, all math in float)
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution
// Verify: python3 ../tools/compare.py out.txt data/expected_out.txt

#include <iostream>
#include <span>

#include "../common/io.h"

// ---------------------------------------------------------------------------
// Toy input: a 3x4 weight matrix (row-major) and a 4-dim activation vector
// ---------------------------------------------------------------------------

constexpr int kD = 3; // output size: number of rows of W
constexpr int kN = 4; // input size: length of each row of W

// Row-major: kW[i*kN + j] is the weight connecting input j to output i.
// Same values as data/input_w.txt.
const float kW[] = {
    0.5f, -1.25f, 2.0f, 0.75f,   // row 0 -> out[0]
    -3.0f, 0.125f, 1.5f, -0.5f,  // row 1 -> out[1]
    2.25f, -0.75f, 0.875f, -2.0f // row 2 -> out[2]
};

// The activation vector, same values as data/input_x.txt.
const float kX[] = {0.5f, -1.0f, 2.0f, 0.25f};

// ---------------------------------------------------------------------------
// The one kernel of this module
// ---------------------------------------------------------------------------

// W (d,n) @ x (n,) -> xout (d,); w holds d rows of n elements, row i produces
// xout[i]. The vast majority of the model's runtime is spent in this small
// function: each weight participates in exactly one multiply-add, so
// performance is bound by memory bandwidth.
void matmul(std::span<float> xout, std::span<const float> x, std::span<const float> w) {
    const size_t n = x.size();
    const size_t d = xout.size();
    for (size_t i = 0; i < d; i++) {
        const std::span w_row = w.subspan(i * n, n);
        float val = 0.0f; // accumulate in float, matching the reference
        for (size_t j = 0; j < n; j++) { val += w_row[j] * x[j]; }
        xout[i] = val;
    }
}

// ---------------------------------------------------------------------------

int main() {
    float xout[kD];
    matmul(xout, kX, kW);
    tut::write_floats("out.txt", xout);
    std::cout << "wrote out.txt\n";
    return 0;
}
