// 07_ffn -- SwiGLU feed-forward network (reference solution)
//
//   h1 = W1 @ x          (hidden_dim, dim) @ (dim,) -> (hidden_dim,)
//   h3 = W3 @ x          same, second branch
//   h  = silu(h1) * h3   elementwise; silu(v) = v / (1 + exp(-v))
//   out = W2 @ h         (dim, hidden_dim) @ (hidden_dim,) -> (dim,)
//
// Layer 0 of stories15M (dim=288, hidden_dim=768); the input is the real
// forward-pass value after rms_ffn normalization at the last position.
// Reads ../../stories15M.bin and data/input_x.txt; writes out_h.txt / out.txt.
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution            (from this module folder)
// Verify: python3 ../tools/compare.py out_h.txt data/expected_hidden.txt
//         python3 ../tools/compare.py out.txt data/expected_out.txt

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// checkpoint header: 7 x int32, in this exact order
struct Config {
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size; // negative marks unshared classifier weights
    int32_t seq_len;
};

// the tensors this module needs, as views into the weight region
struct Weights {
    std::span<const float> w1; // (layer, hidden_dim, dim)
    std::span<const float> w2; // (layer, dim, hidden_dim)
    std::span<const float> w3; // (layer, hidden_dim, dim)
};

// Load the checkpoint and carve the tensors in file order until w1/w2/w3 are
// located (same loader as module 01, stopped early — later tensors unused).
std::vector<float> load_checkpoint(const std::string& path, Config& cfg, Weights& w) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    const std::streamsize file_bytes = in.tellg();
    in.seekg(0);
    std::vector<float> buf(static_cast<size_t>(file_bytes) / sizeof(float));
    in.read(reinterpret_cast<char*>(buf.data()), file_bytes);
    if (!in) { throw std::runtime_error("failed to read " + path); }

    std::memcpy(&cfg, buf.data(), sizeof(cfg));

    const bool shared = cfg.vocab_size > 0;
    const size_t vocab = shared ? cfg.vocab_size : -cfg.vocab_size;
    const size_t dim = cfg.dim;
    const size_t hidden = cfg.hidden_dim;
    const size_t n_layers = cfg.n_layers;
    const size_t head_size = dim / cfg.n_heads;
    const size_t kv_dim = cfg.n_kv_heads * head_size;

    const float* ptr = buf.data() + sizeof(Config) / sizeof(float);
    auto carve = [&ptr](size_t count) {
        std::span<const float> slice{ptr, count};
        ptr += count;
        return slice;
    };
    carve(vocab * dim);           // token_embedding_table
    carve(n_layers * dim);        // rms_att_weight
    carve(n_layers * dim * dim);  // wq
    carve(n_layers * dim * kv_dim); // wk
    carve(n_layers * dim * kv_dim); // wv
    carve(n_layers * dim * dim);  // wo
    carve(n_layers * dim);        // rms_ffn_weight
    w.w1 = carve(n_layers * hidden * dim);
    w.w2 = carve(n_layers * dim * hidden);
    w.w3 = carve(n_layers * hidden * dim);
    return buf; // keeps the weight region alive; the spans point into it
}

// Load all whitespace-separated floats from a text file (one number per line).
std::vector<float> load_vector(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    std::vector<float> v;
    float value;
    while (in >> value) { v.push_back(value); }
    return v;
}

// Write floats one per line, matching the golden data format (%.3e).
void write_vector(const std::string& path, std::span<const float> v) {
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("cannot write " + path); }
    out << std::scientific << std::setprecision(3);
    for (float value : v) { out << value << '\n'; }
}

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

} // namespace

int main() {
    try {
        Config cfg;
        Weights w;
        // the buffer must stay alive: all weight spans point into it
        const std::vector<float> buf = load_checkpoint("../../stories15M.bin", cfg, w);
        const size_t dim = cfg.dim;
        const size_t hidden = cfg.hidden_dim;

        // layer 0's FFN weights: layer L starts at offset L * (per-layer size)
        const size_t layer = 0;
        const std::span<const float> w1 = w.w1.subspan(layer * hidden * dim, hidden * dim);
        const std::span<const float> w2 = w.w2.subspan(layer * dim * hidden, dim * hidden);
        const std::span<const float> w3 = w.w3.subspan(layer * hidden * dim, hidden * dim);

        // FFN input (after rms_ffn normalization)
        const std::vector<float> x = load_vector("data/input_x.txt");
        if (x.size() != dim) {
            throw std::runtime_error("input_x has " + std::to_string(x.size()) +
                                     " values, expected " + std::to_string(dim));
        }

        // two up-projections
        std::vector<float> h1(hidden), h3(hidden);
        matmul(h1, x, w1);
        matmul(h3, x, w3);

        // SwiGLU: h = silu(h1) * h3, silu(v) = v * sigmoid(v) = v / (1 + exp(-v))
        for (size_t i = 0; i < hidden; i++) {
            float v = h1[i];
            v *= 1.0f / (1.0f + std::exp(-v)); // float overload of exp
            v *= h3[i];
            h1[i] = v;
        }
        write_vector("out_h.txt", h1);

        // down-projection (before the residual add)
        std::vector<float> out(dim);
        matmul(out, h1, w2);
        write_vector("out.txt", out);

        std::cout << "wrote out_h.txt (" << hidden << ") and out.txt (" << dim << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
