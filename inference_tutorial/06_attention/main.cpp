// 06_attention — multi-head causal attention with a KV cache (student template)
//
// Goal: for each position pos = 0..4, split the rotated Q into n_heads heads
// and, per head, score it against the K cache rows 0..pos, softmax the scores,
// and return the attention-weighted sum of the V cache rows 0..pos. Head
// outputs are concatenated back into a dim vector (before the wo projection).
//
// All inputs are const arrays in data.h (kQ / kKCache / kVCache, position-major,
// P=5 x 288 each) — no file parsing needed. Outputs go to out.txt / out_att.txt
// so you can verify them against data/expected_*.txt with ../tools/compare.py.
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main
// Verify: python3 ../tools/compare.py out.txt     data/expected_out.txt
//         python3 ../tools/compare.py out_att.txt data/expected_att_weights_lastpos.txt

#include <algorithm>
#include <cmath>
#include <iostream>
#include <span>
#include <vector>

#include "../common/io.h"
#include "data.h"

// ---------------------------------------------------------------------------
// Model constants (stories15M)
// ---------------------------------------------------------------------------

constexpr int kDim = 288;       // model dim
constexpr int kNumHeads = 6;    // query heads
constexpr int kHeadSize = 48;   // dim / n_heads = 288 / 6
constexpr int kKvMul = 1;       // n_heads / n_kv_heads; > 1 only for GQA models
constexpr int kKvDim = 288;     // n_kv_heads * head_size; == dim here
constexpr int kPositions = 5;   // positions in the golden data

// The inputs live in data.h: kQ (rotated q), kKCache / kVCache (layer-0 cache
// contents for positions 0..4), all (kPositions, 288) position-major. Row t,
// head kv_h starts at t * kv_dim + kv_h * head_size.

// ---------------------------------------------------------------------------
// softmax (given, from module 03) and the attention kernel (your job)
// ---------------------------------------------------------------------------

// In-place and numerically stable: subtract the max before exp. This is the
// softmax from module 03 — attention relies on its in-place convention,
// calling it per head on slices of the shared att scratch.
void softmax(std::span<float> x) {
    const float max_val = *std::ranges::max_element(x);
    float sum = 0.0f;
    for (float& v : x) {
        v = std::exp(v - max_val);
        sum += v;
    }
    for (float& v : x) { v /= sum; }
}

// Multi-head attention for a single position.
//   q:       rotated query of this position (dim,)
//   k_cache: all K rows stored so far (positions x kv_dim); only rows 0..pos are read
//   v_cache: same layout for V
//   att:     scratch of size n_heads * kPositions; head h uses [h*kPositions, h*kPositions+pos]
//            and must be left holding this position's softmax weights
//   out:     this position's attention output (dim,), heads concatenated
void attention(std::span<float> out, std::span<float> att, std::span<const float> q,
               std::span<const float> k_cache, std::span<const float> v_cache, int pos) {
    // TODO(task 1): loop over the heads h = 0..n_heads-1. Map each head to its
    //   KV head (kv_h = h / kv_mul; kv_mul == 1 here, so kv_h == h) and slice
    //   this head's query qh out of q at h * head_size.
    // TODO(task 2): for every history position t = 0..pos (history + current
    //   only -- that range IS the causal mask), slice head kv_h's key out of
    //   cache row t (row t starts at t * kv_dim, the head's slice sits
    //   kv_h * head_size floats into the row), compute the dot product
    //   qh . key, scale it by 1 / sqrt(head_size), and store it in head h's
    //   att slice at index t.
    // TODO(task 3): softmax head h's att slice over the 0..pos segment only
    //   (reuse the softmax above; do NOT touch the rest of the scratch --
    //   future positions must not be exp'ed in).
    // TODO(task 4): zero head h's slice of out, then accumulate the
    //   attention-weighted sum of the V cache rows 0..pos into it, using the
    //   same row/head slicing as for K (row t of v_cache, head kv_h).
    (void)out; (void)att; (void)q; (void)k_cache; (void)v_cache; (void)pos; // stub
}

// ---------------------------------------------------------------------------

int main() {
    float out[kPositions * kDim] = {};
    float att[kNumHeads * kPositions] = {};

    // Data is position-major: position pos occupies [pos*dim, (pos+1)*dim).
    // The t = 0..pos loop grows with pos: attention cost scales linearly with
    // sequence length, which is exactly why decode caches K/V.
    for (int pos = 0; pos < kPositions; pos++) {
        attention(std::span{out}.subspan(pos * kDim, kDim), att,
                  std::span<const float>{kQ}.subspan(pos * kDim, kDim), kKCache, kVCache, pos);
    }
    tut::write_floats("out.txt", out);

    // TODO(task 5): after the last attention call, att holds the last
    //   position's softmax weights. Collect them head-major -- head 0's
    //   pos+1 weights, then head 1's, ... (6 x 5 = 30 values total) -- and
    //   write them to out_att.txt.
    std::vector<float> att_last; // stub: replace with your code
    tut::write_floats("out_att.txt", att_last);

    std::cout << "wrote out.txt (" << kPositions << " x " << kDim << ") and out_att.txt ("
              << att_last.size() << " values)\n";
    return 0;
}
