// 05_rope -- rotary position embedding (student template)
//
// Goal: rotate adjacent pairs (v0, v1) of Q and K by pos * freq within each
// head, in place, for each position pos = 0..4.
// Data layout: position-major, P=5 positions x 288 values per file.
// Reads data/input_q.txt / data/input_k.txt, writes out_q.txt / out_k.txt.
//
// Verify with:
//   python3 ../tools/compare.py out_q.txt data/expected_q.txt
//   python3 ../tools/compare.py out_k.txt data/expected_k.txt

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <string>
#include <vector>

namespace {

constexpr int kDim = 288;       // model dim
constexpr int kHeadSize = 48;   // dim / n_heads = 288 / 6
constexpr int kKvDim = 288;     // == dim here; < dim only for GQA models
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
}

}  // namespace

int main() {
    std::vector<float> q = load("data/input_q.txt");
    std::vector<float> k = load("data/input_k.txt");
    if (q.size() != kPositions * kDim || k.size() != kPositions * kDim) {
        std::cerr << "unexpected input size: q=" << q.size() << " k=" << k.size()
                  << " (want " << kPositions * kDim << " each)\n";
        return 1;
    }

    // Data is position-major: position pos occupies [pos*dim, (pos+1)*dim).
    for (int pos = 0; pos < kPositions; pos++) {
        rope(std::span{q}.subspan(pos * kDim, kDim),
             std::span{k}.subspan(pos * kDim, kDim), pos);
    }

    save("out_q.txt", q);
    save("out_k.txt", k);
    std::cout << "wrote out_q.txt and out_k.txt (" << kPositions << " x " << kDim
              << " values)\n";
    return 0;
}
