// 09_sampler — student template
//
// Goal: implement the three sampling strategies of llama2 (greedy argmax,
// full-distribution sampling, top-p / nucleus sampling) plus the xorshift
// RNG described in README.md, then verify against the golden data:
//
// build:  c++ -O2 -std=c++20 -o main main.cpp
// run:    ./main            (from the module folder; writes out_rng.txt and out.txt)
// verify: python3 ../tools/compare.py out_rng.txt data/expected_rng_seed42.txt
//         python3 ../tools/compare.py out.txt data/expected_samples.txt --exact
//
// Fill in every // TODO(task N) below. The RNG formula and the strategy
// selection rules are specified in README.md — match them exactly.

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
// helpers (given)

// Load a whitespace-separated list of floats (one per line in the data files).
std::vector<float> load_floats(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    std::vector<float> v;
    float x;
    while (in >> x) { v.push_back(x); }
    return v;
}

// In-place softmax. Use max subtraction for numerical stability.
void softmax(std::span<float> x) {
    (void)x;
    // TODO(task 2): turn x into a probability distribution in place:
    // subtract the max, exponentiate, then normalize by the sum.
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
    float random_f32() {
        // TODO(task 1): draw a 32-bit value from random_u32() and map it to a
        // float in [0, 1) exactly as README.md specifies.
        return 0.0f;
    }

    // Sample the next token from logits. May mutate logits.
    int sample(std::span<float> logits) {
        (void)logits;
        // TODO(task 3): dispatch on temperature_/topp_ per README.md:
        // greedy argmax, full-distribution sampling, or top-p sampling.
        // Remember the non-greedy paths scale by temperature and softmax first,
        // then draw one coin.
        return 0;
    }

private:
    // Greedy: index of the largest probability.
    static int sample_argmax(std::span<const float> probabilities) {
        (void)probabilities;
        // TODO(task 3a): return the index of the maximum element.
        return 0;
    }

    // Full distribution: walk the CDF until the coin falls inside.
    static int sample_mult(std::span<const float> probabilities, float coin) {
        (void)probabilities;
        (void)coin;
        // TODO(task 4): accumulate probabilities left to right and return the
        // first index whose cumulative sum exceeds the coin.
        return 0;
    }

    // Top-p (nucleus) sampling.
    int sample_topp(std::span<const float> probabilities, float topp, float coin) {
        (void)probabilities;
        (void)topp;
        (void)coin;
        // TODO(task 5): per README.md —
        //  1. discard tokens whose prob is below the cutoff derived from topp
        //     (collect survivors in probindex_);
        //  2. sort survivors by probability descending;
        //  3. keep the smallest prefix whose cumulative prob exceeds topp;
        //  4. sample within that prefix using coin * prefix_total_prob.
        return 0;
    }

    std::uint32_t random_u32() {
        // TODO(task 1): advance the xorshift state and derive the 32-bit
        // output exactly as README.md specifies (mind the 64-bit multiply).
        return 0;
    }

    float temperature_;
    float topp_;
    std::uint64_t rng_state_;
    std::vector<ProbIndex> probindex_;
};

// ----------------------------------------------------------------------------
// driver (given — no changes needed once the tasks above are done)

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
            std::vector<float> copy = *c.logits; // sample() may mutate the logits
            out << sampler.sample(copy) << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
