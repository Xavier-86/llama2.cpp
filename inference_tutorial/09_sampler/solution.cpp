// 09_sampler — reference solution
//
// Implements the three sampling strategies of llama2 (greedy argmax,
// full-distribution sampling, top-p / nucleus sampling) plus the xorshift
// RNG that must match the reference bit-for-bit.
//
// build:  c++ -O2 -std=c++20 -o solution solution.cpp
// run:    ./solution        (from the module folder; writes out_rng.txt and out.txt)
// verify: python3 ../tools/compare.py out_rng.txt data/expected_rng_seed42.txt
//         python3 ../tools/compare.py out.txt data/expected_samples.txt --exact

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

// ----------------------------------------------------------------------------
// helpers

// Load a whitespace-separated list of floats (one per line in the data files).
std::vector<float> load_floats(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    std::vector<float> v;
    float x;
    while (in >> x) { v.push_back(x); }
    return v;
}

// In-place softmax with max subtraction for numerical stability.
void softmax(std::span<float> x) {
    const float max_val = *std::ranges::max_element(x);
    float sum = 0.0f;
    for (float& v : x) {
        v = std::exp(v - max_val);
        sum += v;
    }
    for (float& v : x) { v /= sum; }
}

// ----------------------------------------------------------------------------
// Sampler

struct ProbIndex {
    float prob;
    int index;
};

class Sampler {
public:
    Sampler(int vocab_size, float temperature, float topp, std::uint64_t rng_seed)
        : temperature_(temperature), topp_(topp), rng_state_(rng_seed), probindex_(vocab_size) {}

    // Draw one float coin in [0, 1). Public so the RNG can be verified on its own.
    float random_f32() { return (random_u32() >> 8) / 16777216.0f; }

    // Sample the next token from logits. Mutates logits (temperature scale + softmax).
    int sample(std::span<float> logits) {
        if (temperature_ == 0.0f) { return sample_argmax(logits); }
        for (float& v : logits) { v /= temperature_; }
        softmax(logits);
        const float coin = random_f32();
        if (topp_ <= 0 || topp_ >= 1) { return sample_mult(logits, coin); }
        return sample_topp(logits, topp_, coin);
    }

private:
    // Greedy: index of the largest probability.
    static int sample_argmax(std::span<const float> probabilities) {
        return static_cast<int>(std::ranges::max_element(probabilities) - probabilities.begin());
    }

    // Full distribution: walk the CDF until the coin falls inside.
    static int sample_mult(std::span<const float> probabilities, float coin) {
        float cdf = 0.0f;
        for (size_t i = 0; i < probabilities.size(); i++) {
            cdf += probabilities[i];
            if (coin < cdf) { return static_cast<int>(i); }
        }
        return static_cast<int>(probabilities.size()) - 1;
    }

    // Top-p (nucleus): filter tiny probs, sort descending, keep the smallest
    // prefix whose cumulative prob exceeds topp, sample within that prefix.
    int sample_topp(std::span<const float> probabilities, float topp, float coin) {
        const int n = static_cast<int>(probabilities.size());
        int n0 = 0;
        const float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < n; i++) {
            if (probabilities[i] >= cutoff) {
                probindex_[n0] = {probabilities[i], i};
                n0++;
            }
        }
        std::sort(probindex_.begin(), probindex_.begin() + n0,
                  [](const ProbIndex& a, const ProbIndex& b) { return a.prob > b.prob; });

        float cumulative_prob = 0.0f;
        int last_idx = n0 - 1;
        for (int i = 0; i < n0; i++) {
            cumulative_prob += probindex_[i].prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break;
            }
        }

        float r = coin * cumulative_prob;
        float cdf = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            cdf += probindex_[i].prob;
            if (r < cdf) { return probindex_[i].index; }
        }
        return probindex_[last_idx].index;
    }

    // xorshift64 with a final multiply; must match the reference bit-for-bit.
    std::uint32_t random_u32() {
        rng_state_ ^= rng_state_ >> 12;
        rng_state_ ^= rng_state_ << 25;
        rng_state_ ^= rng_state_ >> 27;
        return static_cast<std::uint32_t>((rng_state_ * 0x2545F4914F6CDD1Dull) >> 32);
    }

    float temperature_;
    float topp_;
    std::uint64_t rng_state_;
    std::vector<ProbIndex> probindex_;
};

// ----------------------------------------------------------------------------
// driver

int main() {
    try {
        // 1. RNG self-check: first 10 coins for seed 42.
        {
            Sampler rng(/*vocab_size=*/1, /*temperature=*/1.0f, /*topp=*/1.0f, /*seed=*/42);
            std::ofstream out("out_rng.txt");
            out << std::scientific << std::setprecision(3);
            for (int i = 0; i < 10; i++) { out << rng.random_f32() << '\n'; }
        }

        // 2. Load the two logits vectors.
        const std::vector<float> real = load_floats("data/input_logits.txt");
        const std::vector<float> synth = load_floats("data/input_logits_synth.txt");

        // 3. Run the 8 cases (first 4 on real logits, last 4 on synthetic).
        struct Case {
            float temperature;
            float topp;
            std::uint64_t seed;
            const std::vector<float>* logits;
        };
        const Case cases[] = {
            {0.0f, 0.9f, 42, &real},   {1.0f, 1.0f, 42, &real},
            {0.8f, 0.9f, 42, &real},   {0.8f, 0.9f, 1234, &real},
            {0.0f, 0.9f, 42, &synth},  {1.0f, 1.0f, 42, &synth},
            {1.0f, 0.5f, 42, &synth},  {2.0f, 0.9f, 7, &synth},
        };

        std::ofstream out("out.txt");
        for (const Case& c : cases) {
            Sampler sampler(static_cast<int>(c.logits->size()), c.temperature, c.topp, c.seed);
            std::vector<float> copy = *c.logits; // sample() mutates the logits
            out << sampler.sample(copy) << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
