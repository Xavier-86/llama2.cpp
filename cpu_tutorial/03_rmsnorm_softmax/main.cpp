// 03_rmsnorm_softmax — student template
//
// The two normalization kernels used all over the Transformer:
// RMSNorm (before attention, before the FFN, once at the end) and softmax
// (attention scores, sampling). Each is under 10 lines — see README.md for
// the math.
//
// All inputs are const arrays below (small toys) or in data.h (the real
// 288-dim case) — no file parsing needed. Outputs go to out*.txt so you can
// verify them against data/expected_*.txt with ../tools/compare.py.
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main
// Verify: python3 ../tools/compare.py out.txt      data/expected_rmsnorm.txt
//         python3 ../tools/compare.py out_real.txt data/expected_rmsnorm_real.txt
//         python3 ../tools/compare.py out_sm.txt   data/expected_softmax.txt
//         python3 ../tools/compare.py out_big.txt  data/expected_softmax_big.txt

#include <algorithm>
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

// The real 288-dim case uses kRmsNormXReal / kRmsNormWReal from data.h:
// the embedding row of the first prompt token, and layer-0 rms_att_weight.

// ---------------------------------------------------------------------------
// The two kernels of this module
// ---------------------------------------------------------------------------

void rmsnorm(std::span<float> o, std::span<const float> x, std::span<const float> weight) {
    // TODO(task 1): implement RMSNorm as defined in README.md: normalize x by
    //   its root-mean-square (with eps = 1e-5) and scale by `weight`, writing
    //   into `o`. Compute everything in float, including the running sum, to
    //   match the golden data. Verified by out.txt and out_real.txt.
    std::ranges::fill(o, 0.0f); // stub: replace with your code
    (void)x; (void)weight;
}

void softmax(std::span<float> x) {
    // TODO(task 2): implement softmax **in place** (attention calls it per head
    //   on slices of a shared buffer), and numerically stable: subtract the max
    //   before exp, otherwise the ~1000 case overflows to inf and out_big.txt
    //   comes out as zeros or NaN. Verified by out_sm.txt and out_big.txt.
    (void)x; // stub: replace with your code
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
