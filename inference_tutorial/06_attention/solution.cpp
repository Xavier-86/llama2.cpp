// 06_attention -- multi-head causal attention with a KV cache (reference solution)
//
// For each position pos = 0..4, splits the rotated Q into n_heads heads and,
// per head, scores it against the K cache rows 0..pos, softmaxes the scores,
// and returns the attention-weighted sum of the V cache rows 0..pos. Head
// outputs are concatenated back into a dim vector (before the wo projection).
// Data layout: position-major, P=5 positions x 288 values per file.
// Reads data/input_q.txt / data/input_k_cache.txt / data/input_v_cache.txt,
// writes out.txt / out_att.txt.
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution        (from this folder)
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
//            and is left holding this position's softmax weights
//   out:     this position's attention output (dim,), heads concatenated
void attention(std::span<float> out, std::span<float> att, std::span<const float> q,
               std::span<const float> k_cache, std::span<const float> v_cache, int pos) {
    for (int h = 0; h < kNumHeads; h++) {
        const int kv_h = h / kKvMul;  // GQA sharing; kv_mul == 1 here, so kv_h == h
        std::span qh = q.subspan(static_cast<size_t>(h) * kHeadSize, kHeadSize);
        std::span att_h = att.subspan(static_cast<size_t>(h) * kPositions,
                                      static_cast<size_t>(pos) + 1);

        // Score against history + current only: t in [0, pos] is the causal mask.
        for (int t = 0; t <= pos; t++) {
            std::span key = k_cache.subspan(
                static_cast<size_t>(t) * kKvDim + static_cast<size_t>(kv_h) * kHeadSize,
                kHeadSize);
            float score = 0.0f;
            for (int i = 0; i < kHeadSize; i++) { score += qh[i] * key[i]; }
            att_h[t] = score / std::sqrt(static_cast<float>(kHeadSize));
        }

        softmax(att_h);

        // Weighted sum of the V rows, written into this head's slice of out.
        std::span out_h = out.subspan(static_cast<size_t>(h) * kHeadSize, kHeadSize);
        std::ranges::fill(out_h, 0.0f);
        for (int t = 0; t <= pos; t++) {
            std::span val = v_cache.subspan(
                static_cast<size_t>(t) * kKvDim + static_cast<size_t>(kv_h) * kHeadSize,
                kHeadSize);
            const float a = att_h[t];
            for (int i = 0; i < kHeadSize; i++) { out_h[i] += a * val[i]; }
        }
    }
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

    // After the last call, att holds the last position's weights; dump them
    // head-major: head 0's 5 weights, then head 1's, ... (6 x 5 = 30 values).
    const int last = kPositions - 1;
    std::vector<float> att_last;
    for (int h = 0; h < kNumHeads; h++) {
        for (int t = 0; t <= last; t++) {
            att_last.push_back(att[static_cast<size_t>(h) * kPositions + t]);
        }
    }
    save("out_att.txt", att_last);

    std::cout << "wrote out.txt (" << kPositions << " x " << kDim << ") and out_att.txt ("
              << kNumHeads << " x " << kPositions << ")\n";
    return 0;
}
