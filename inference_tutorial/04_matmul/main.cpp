// 04_matmul — student template
//
// The matrix-vector multiply that dominates llama2 inference: every linear
// projection (wq/wk/wv/wo, w1/w2/w3, the classifier) is one call to this
// kernel. Under 10 lines — see README.md for the math and the conventions.
//
// The inputs are const arrays below (a 3x4 toy matrix and a 4-dim vector) —
// no file parsing needed. The output goes to out.txt so you can verify it
// against data/expected_out.txt with ../tools/compare.py.
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main
// Verify: python3 ../tools/compare.py out.txt data/expected_out.txt

#include <algorithm>
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

void matmul(std::span<float> xout, std::span<const float> x, std::span<const float> w) {
    // TODO(task 1): implement the matrix-vector multiply defined in README.md:
    //   each output element is the dot product of one row of W with x, i.e.
    //   xout[i] = sum_j w[i*n + j] * x[j], where n = x.size(). Accumulate in
    //   float (not double) to match the golden data. Verified by out.txt.
    std::ranges::fill(xout, 0.0f); // stub: replace with your code
    (void)w;
}

// ---------------------------------------------------------------------------

int main() {
    float xout[kD];
    matmul(xout, kX, kW);
    tut::write_floats("out.txt", xout);
    std::cout << "wrote out.txt\n";
    return 0;
}
