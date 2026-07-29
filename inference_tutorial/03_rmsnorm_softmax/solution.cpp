// 03_rmsnorm_softmax — reference solution
//
// The two normalization kernels used all over the Transformer:
//   * rmsnorm: out_i = w_i * x_i / sqrt(mean(x^2) + 1e-5)
//   * softmax: numerically stable (subtract max first), in place
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution
// Verify: python3 ../tools/compare.py out.txt      data/expected_rmsnorm.txt
//         python3 ../tools/compare.py out_real.txt data/expected_rmsnorm_real.txt
//         python3 ../tools/compare.py out_sm.txt   data/expected_softmax.txt
//         python3 ../tools/compare.py out_big.txt  data/expected_softmax_big.txt

#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>

#include "../common/io.h"
#include "data.h"

// ---------------------------------------------------------------------------
// Model constants (stories15M) and toy input vectors
// ---------------------------------------------------------------------------

constexpr int kDim = 288; // model width; the real case below works at this size

// Toy 8-dim RMSNorm case: input and learned weight.
const float kRmsXToy[] = {0.5f, -1.2f, 3.3f, 0.0f, 1e-3f, -2.7f, 4.0f, -0.8f};
const float kRmsWToy[] = {1.1f, 0.9f, -0.5f, 2.0f, 1.0f, 0.3f, -1.4f, 0.7f};

// Ordinary 8-dim softmax case, and a ~1000 case that exercises the
// max-subtraction trick (values like exp(1000) overflow to inf without it).
const float kSoftmaxIn[] = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 0.5f, -2.0f};
const float kSoftmaxBig[] = {1000.0f, 1001.0f, 1002.0f, 999.0f};

// ---------------------------------------------------------------------------
// The two kernels of this module
// ---------------------------------------------------------------------------

// out_i = weight_i * x_i / sqrt(mean(x^2) + eps); all arithmetic in float.
void rmsnorm(std::span<float> o, std::span<const float> x, std::span<const float> weight) {
    float ss = 0.0f;
    for (float v : x) { ss += v * v; }
    ss /= x.size();
    ss += 1e-5f;
    ss = 1.0f / std::sqrt(ss);
    for (size_t j = 0; j < x.size(); j++) { o[j] = weight[j] * (ss * x[j]); }
}

// In-place and numerically stable: subtract the max before exp, so large
// inputs (see the ~1000 case) cannot overflow to inf.
void softmax(std::span<float> x) {
    const float max_val = *std::ranges::max_element(x);
    float sum = 0.0f;
    for (float& v : x) {
        v = std::exp(v - max_val);
        sum += v;
    }
    for (float& v : x) { v /= sum; }
}

// ---------------------------------------------------------------------------

int main() {
    // Case 1: 8-dim toy RMSNorm -> out.txt
    {
        float o[8];
        rmsnorm(o, kRmsXToy, kRmsWToy);
        tut::write_floats("out.txt", o);
    }

    // Case 2: real 288-dim RMSNorm with layer-0 rms_att_weight -> out_real.txt
    {
        float o[kDim];
        rmsnorm(o, kRmsNormXReal, kRmsNormWReal);
        tut::write_floats("out_real.txt", o);
    }

    // Case 3: ordinary 8-dim softmax -> out_sm.txt
    {
        float x[8];
        std::ranges::copy(kSoftmaxIn, x);
        softmax(x);
        tut::write_floats("out_sm.txt", x);
    }

    // Case 4: values around 1000, exercises the max-subtraction -> out_big.txt
    {
        float x[4];
        std::ranges::copy(kSoftmaxBig, x);
        softmax(x);
        tut::write_floats("out_big.txt", x);
    }

    std::cout << "wrote out.txt out_real.txt out_sm.txt out_big.txt\n";
    return 0;
}
