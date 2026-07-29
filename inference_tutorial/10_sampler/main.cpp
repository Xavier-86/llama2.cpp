// 10_sampler — student template
//
// The last step of the generation loop: turn the 32000 logits of one forward
// pass into the next token id. Three strategies — greedy argmax,
// full-distribution sampling, top-p (nucleus) — plus the xorshift RNG that
// must match the reference bit-for-bit (see README.md).
//
// The synthetic logits are the const array kLogitsSynth below. The real
// 32000-dim logits are the one exception in this tutorial: too many values to
// embed as a const array, so they stay in a data file and are loaded with a
// single tut::read_floats call. Outputs go to out_rng.txt / out.txt so you can
// verify them against data/expected_*.txt with ../tools/compare.py.
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main
// Verify: python3 ../tools/compare.py out_rng.txt data/expected_rng_seed42.txt
//         python3 ../tools/compare.py out.txt data/expected_samples.txt --exact

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <span>
#include <stdexcept>
#include <vector>

#include "../common/io.h"

// ---------------------------------------------------------------------------
// Toy input: synthetic logits for a vocab of 8.
// ---------------------------------------------------------------------------

// Flat ramp so every token has a visibly different probability; this is what
// actually exercises the mult / top-p paths (the real distribution is 96.6%
// peaked on its argmax, so all real cases pick it).
const float kLogitsSynth[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

// The real input is data/input_logits.txt: the 32000 logits at the last
// prompt position of the reference prompt, loaded in main() below.

// ---------------------------------------------------------------------------
// helper kernel (given as a stub)
// ---------------------------------------------------------------------------

// In-place softmax. Use max subtraction for numerical stability.
void softmax(std::span<float> x) {
    (void)x;
    // TODO(task 2): turn x into a probability distribution in place:
    // subtract the max, exponentiate, then normalize by the sum.
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// driver (given — no changes needed once the tasks above are done)
// ---------------------------------------------------------------------------

int main() {
    try {
        // 1. RNG self-check: first 10 coins for seed 42 -> out_rng.txt
        {
            Sampler rng(/*vocab_size=*/1, /*temperature=*/1.0f, /*topp=*/1.0f, /*seed=*/42);
            float coins[10];
            for (float& c : coins) { c = rng.random_f32(); }
            tut::write_floats("out_rng.txt", coins);
        }

        // 2. Load the real logits — the tutorial's single file-read exception
        //    (32000 values are too many to embed as a const array).
        const std::vector<float> real = tut::read_floats("data/input_logits.txt");

        // 3. Run the 8 cases (first 4 on real logits, last 4 on synthetic).
        struct Case {
            float temperature;
            float topp;
            std::uint64_t seed;
            std::span<const float> logits;
        };
        const Case cases[] = {
            {0.0f, 0.9f, 42, real},          {1.0f, 1.0f, 42, real},
            {0.8f, 0.9f, 42, real},          {0.8f, 0.9f, 1234, real},
            {0.0f, 0.9f, 42, kLogitsSynth},  {1.0f, 1.0f, 42, kLogitsSynth},
            {1.0f, 0.5f, 42, kLogitsSynth},  {2.0f, 0.9f, 7, kLogitsSynth},
        };

        int results[8];
        for (size_t i = 0; i < 8; i++) {
            const Case& c = cases[i];
            Sampler sampler(static_cast<int>(c.logits.size()), c.temperature, c.topp, c.seed);
            // sample() mutates the logits (temperature scale + softmax), so
            // hand it a scratch copy.
            std::vector<float> copy(c.logits.begin(), c.logits.end());
            results[i] = sampler.sample(copy);
        }
        tut::write_ints("out.txt", results);

        std::cout << "wrote out_rng.txt out.txt\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
