// 06_attention -- multi-head causal attention with a KV cache (student template)
//
// Goal: for each position pos = 0..4, split the rotated Q into n_heads heads
// and, per head, score it against the K cache rows 0..pos, softmax the scores,
// and return the attention-weighted sum of the V cache rows 0..pos. Head
// outputs are concatenated back into a dim vector (before the wo projection).
// Data layout: position-major, P=5 positions x 288 values per file.
// Reads data/input_q.txt / data/input_k_cache.txt / data/input_v_cache.txt,
// writes out.txt / out_att.txt.
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Verify: python3 ../tools/compare.py out.txt     data/expected_out.txt
//         python3 ../tools/compare.py out_att.txt data/expected_att_weights_lastpos.txt

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr int kDim = 288;       // model dim
constexpr int kNumHeads = 6;    // query heads
constexpr int kHeadSize = 48;   // dim / n_heads = 288 / 6
constexpr int kKvMul = 1;       // n_heads / n_kv_heads; > 1 only for GQA models
constexpr int kKvDim = 288;     // n_kv_heads * head_size; == dim here
constexpr int kPositions = 5;   // positions in the golden data

// Load whitespace-separated floats, one per line.
std::vector<float> load(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        std::cerr << "cannot open " << path << "\n";
        std::exit(1);
    }
    std::vector<float> v;
    float x;
    while (in >> x) v.push_back(x);
    return v;
}

// Save floats in the project's golden format: %.3e, one per line.
void save(const std::string& path, std::span<const float> v) {
    std::ofstream out(path);
    out << std::scientific << std::setprecision(3);
    for (float x : v) out << x << '\n';
}

// In-place and numerically stable: subtract the max before exp.
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
}

}  // namespace

int main() {
    std::vector<float> q = load("data/input_q.txt");
    std::vector<float> k_cache = load("data/input_k_cache.txt");
    std::vector<float> v_cache = load("data/input_v_cache.txt");
    if (q.size() != kPositions * kDim || k_cache.size() != kPositions * kKvDim ||
        v_cache.size() != kPositions * kKvDim) {
        std::cerr << "unexpected input size: q=" << q.size() << " k=" << k_cache.size()
                  << " v=" << v_cache.size() << "\n";
        return 1;
    }

    std::vector<float> out(kPositions * kDim, 0.0f);
    std::vector<float> att(static_cast<size_t>(kNumHeads) * kPositions, 0.0f);

    // Data is position-major: position pos occupies [pos*dim, (pos+1)*dim).
    // The t = 0..pos loop grows with pos: attention cost scales linearly with
    // sequence length, which is exactly why decode caches K/V.
    for (int pos = 0; pos < kPositions; pos++) {
        attention(std::span{out}.subspan(pos * kDim, kDim), att,
                  std::span<const float>{q}.subspan(pos * kDim, kDim), k_cache, v_cache, pos);
    }
    save("out.txt", out);

    // TODO(task 5): after the last attention call, att holds the last
    //   position's softmax weights. Collect them head-major -- head 0's
    //   pos+1 weights, then head 1's, ... (6 x 5 = 30 values total) -- and
    //   save them to out_att.txt.
    std::vector<float> att_last;
    save("out_att.txt", att_last);

    std::cout << "wrote out.txt (" << kPositions << " x " << kDim << ") and out_att.txt ("
              << att_last.size() << " values)\n";
    return 0;
}
