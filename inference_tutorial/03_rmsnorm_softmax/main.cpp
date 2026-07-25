// 03_rmsnorm_softmax — student template
//
// The two normalization kernels used all over the Transformer:
// RMSNorm (before attention, before the FFN, once at the end) and softmax
// (attention scores, sampling). Each is under 10 lines — see README.md for
// the math.
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main            (from this folder; needs ../../stories15M.bin)
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
// IO helpers (ready to use)
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
    // TODO(task 3): read `count` float32 values at float offset `offset` of the
    //   checkpoint's weight region (seek past the 7xint32 config header first).
    //   Reuse whatever loader you wrote in module 01. For the real RMSNorm case
    //   below you need the layer-0 slice of rms_att_weight (offset 0 within the
    //   tensor, length dim) — work out where the tensor itself starts from the
    //   module-01 weight table.
    return std::vector<float>(count, 0.0f); // stub: replace with your code
}

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
            // TODO(task 3): compute rms_att_weight's start offset from the
            //   config (mind the sign convention of vocab_size), then load the
            //   layer-0 slice of length dim.
            const std::vector<float> w = load_weight_slice(ckpt, 0, dim);
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
