// 03_rmsnorm_softmax — reference solution
//
// The two normalization kernels used all over the Transformer:
//   * rmsnorm: out_i = w_i * x_i / sqrt(mean(x^2) + 1e-5)
//   * softmax: numerically stable (subtract max first), in place
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution        (from this folder; needs ../../stories15M.bin)
// Verify: python3 ../tools/compare.py out.txt      data/expected_rmsnorm.txt
//         python3 ../tools/compare.py out_real.txt data/expected_rmsnorm_real.txt
//         python3 ../tools/compare.py out_sm.txt   data/expected_softmax.txt
//         python3 ../tools/compare.py out_big.txt  data/expected_softmax_big.txt

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// IO helpers
// ---------------------------------------------------------------------------

// Read a whitespace-separated list of floats (data files hold one per line).
static std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path);
    if (!f) { throw std::runtime_error("cannot open " + path); }
    std::vector<float> v;
    float x;
    while (f >> x) { v.push_back(x); }
    return v;
}

// Write floats one per line in the golden-data format (%.3e).
static void write_floats(const std::string& path, std::span<const float> v) {
    std::ofstream f(path);
    if (!f) { throw std::runtime_error("cannot write " + path); }
    f << std::scientific << std::setprecision(3);
    for (float x : v) { f << x << '\n'; }
}

// ---------------------------------------------------------------------------
// Minimal checkpoint access (module 01 describes the full format)
// ---------------------------------------------------------------------------

struct Config {
    int32_t dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len;
};

static Config read_config(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open " + path); }
    Config c{};
    f.read(reinterpret_cast<char*>(&c), sizeof(c));
    if (!f) { throw std::runtime_error("failed reading config from " + path); }
    return c;
}

// Load `count` floats starting at float offset `offset` in the weight region
// (offset 0 = first float right after the 7xint32 config header).
static std::vector<float> load_weight_slice(const std::string& path, size_t offset, size_t count) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { throw std::runtime_error("cannot open " + path); }
    f.seekg(static_cast<std::streamoff>(sizeof(Config) + offset * sizeof(float)));
    std::vector<float> v(count);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(count * sizeof(float)));
    if (!f) { throw std::runtime_error("failed reading weights from " + path); }
    return v;
}

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
    try {
        // Case 1: 8-dim toy RMSNorm -> out.txt
        {
            const std::vector<float> x = read_floats("data/input_rmsnorm_x.txt");
            const std::vector<float> w = read_floats("data/input_rmsnorm_w.txt");
            std::vector<float> o(x.size());
            rmsnorm(o, x, w);
            write_floats("out.txt", o);
        }

        // Case 2: real 288-dim RMSNorm with layer-0 rms_att_weight -> out_real.txt
        {
            const std::string ckpt = "../../stories15M.bin";
            const Config c = read_config(ckpt);
            const size_t dim = static_cast<size_t>(c.dim);
            // A negative vocab_size marks unshared classifier weights; the
            // embedding table has abs(vocab_size) rows either way.
            const size_t vocab = static_cast<size_t>(
                c.vocab_size < 0 ? -static_cast<int64_t>(c.vocab_size) : c.vocab_size);
            // rms_att_weight sits right after token_embedding_table
            // (vocab_size x dim); layer 0 = offset 0 within it, length dim.
            const std::vector<float> w = load_weight_slice(ckpt, vocab * dim, dim);
            const std::vector<float> x = read_floats("data/input_rmsnorm_x_real.txt");
            std::vector<float> o(x.size());
            rmsnorm(o, x, w);
            write_floats("out_real.txt", o);
        }

        // Case 3: ordinary 8-dim softmax -> out_sm.txt
        {
            std::vector<float> x = read_floats("data/input_softmax.txt");
            softmax(x);
            write_floats("out_sm.txt", x);
        }

        // Case 4: values around 1000, exercises the max-subtraction -> out_big.txt
        {
            std::vector<float> x = read_floats("data/input_softmax_big.txt");
            softmax(x);
            write_floats("out_big.txt", x);
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    std::cout << "wrote out.txt out_real.txt out_sm.txt out_big.txt\n";
    return 0;
}
