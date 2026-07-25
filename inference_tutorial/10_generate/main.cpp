// 10_generate — student template
//
// Goal: wire forward (08) + sampler (09) + tokenizer (02) into the full
// prefill/decode generation loop described in README.md. Passing this means
// you have rebuilt the FP32 run.cpp.
//
// Prompt is fixed to "Once upon a time", 64 steps, two runs:
//   greedy:  temperature=0.0, topp=0.9, seed=42 -> out_ids.txt / out_text.txt
//   sampled: temperature=0.8, topp=0.9, seed=42 -> out_sids.txt / out_stext.txt
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main            (from this module folder)
// Verify: python3 ../tools/compare.py out_ids.txt   data/expected_greedy_ids.txt --exact
//         python3 ../tools/compare.py out_text.txt  data/expected_greedy_text.txt --text
//         python3 ../tools/compare.py out_sids.txt  data/expected_sampled_ids.txt --exact
//         python3 ../tools/compare.py out_stext.txt data/expected_sampled_text.txt --text
//
// This module adds almost no new math: tasks 1-5 are the modules you already
// built (bring your own code — the bodies below are TODO stubs), task 6 is the
// new generation loop. Fill in every // TODO(task N) below.

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Module 01: checkpoint loader (Config + weight spans)
// ---------------------------------------------------------------------------

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

// each tensor is just a view into the weight region — no copies
struct Weights {
    std::span<const float> token_embedding_table;
    std::span<const float> rms_att_weight;
    std::span<const float> wq;
    std::span<const float> wk;
    std::span<const float> wv;
    std::span<const float> wo;
    std::span<const float> rms_ffn_weight;
    std::span<const float> w1;
    std::span<const float> w2;
    std::span<const float> w3;
    std::span<const float> rms_final_weight;
    std::span<const float> wcls;
};

// Read the whole checkpoint, parse the config, and carve the 11 tensors out of
// the weight region in file order. Returns the owning buffer; the spans in `w`
// point into it, so the caller must keep it alive (and never reallocate it).
std::vector<float> load_checkpoint(const std::string& path, Config& cfg, Weights& w) {
    // file reading is given; the parsing is yours
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    const std::streamsize file_bytes = in.tellg();
    in.seekg(0);
    std::vector<float> buf(static_cast<size_t>(file_bytes) / sizeof(float));
    in.read(reinterpret_cast<char*>(buf.data()), file_bytes);
    if (!in) { throw std::runtime_error("failed to read " + path); }

    (void)cfg;
    (void)w;
    // TODO(task 1): parse the Config header from the start of `buf`, then walk
    // the weight region (it starts right after the header) and record one
    // std::span view per tensor of Weights, in file order. Remember:
    //   - kv_dim = n_kv_heads * (dim / n_heads)
    //   - a negative vocab_size means unshared classifier weights
    //   - the legacy freq_cis_real / freq_cis_imag tables sit between
    //     rms_final_weight and wcls — skip them
    //   - with shared weights, wcls aliases token_embedding_table
    return buf;
}

// ---------------------------------------------------------------------------
// Modules 03/04: the math kernels (accumulate in float, matching the reference)
// ---------------------------------------------------------------------------

// out_i = weight_i * x_i / sqrt(mean(x^2) + eps); must also work in place (o == x).
void rmsnorm(std::span<float> o, std::span<const float> x, std::span<const float> weight) {
    (void)o;
    (void)x;
    (void)weight;
    // TODO(task 2a): rmsnorm from module 03.
}

// In-place and numerically stable: subtract the max before exp.
void softmax(std::span<float> x) {
    (void)x;
    // TODO(task 2b): softmax from module 03.
}

// W (d,n) @ x (n,) -> xout (d,); w holds d rows of n elements, row-major.
void matmul(std::span<float> xout, std::span<const float> x, std::span<const float> w) {
    (void)xout;
    (void)x;
    (void)w;
    // TODO(task 2c): matmul from module 04. This is where the model spends
    // almost all of its time; keep the accumulation in float.
}

// ---------------------------------------------------------------------------
// Module 08: the full single-step forward pass
// ---------------------------------------------------------------------------

// Intermediate buffers; the KV cache is (layer, seq_len, kv_dim).
struct RunState {
    std::vector<float> x;           // activation at the current position (dim,)
    std::vector<float> xb;          // same, but inside a residual branch (dim,)
    std::vector<float> xb2;         // an additional buffer just for convenience (dim,)
    std::vector<float> hb;          // buffer for the hidden dimension in the ffn (hidden_dim,)
    std::vector<float> hb2;         // second ffn hidden buffer (hidden_dim,)
    std::vector<float> q;           // query (dim,)
    std::vector<float> att;         // attention scores (n_heads, seq_len)
    std::vector<float> logits;      // output logits (vocab_size,)
    std::vector<float> key_cache;   // (layer, seq_len, kv_dim)
    std::vector<float> value_cache; // (layer, seq_len, kv_dim)

    RunState() = default;
    explicit RunState(const Config& p)
        : x(p.dim), xb(p.dim), xb2(p.dim), hb(p.hidden_dim), hb2(p.hidden_dim), q(p.dim),
          att(static_cast<size_t>(p.n_heads) * p.seq_len), logits(std::abs(p.vocab_size)),
          key_cache(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim(p)),
          value_cache(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim(p)) {}

    static size_t kv_dim(const Config& p) {
        return static_cast<size_t>(p.n_kv_heads) * (p.dim / p.n_heads);
    }
};

struct Transformer {
    Config config{};
    Weights weights;
    RunState state;
    std::vector<float> buffer; // owns the weight region; all weight spans point into it

    explicit Transformer(const std::string& checkpoint_path) {
        buffer = load_checkpoint(checkpoint_path, config, weights);
        state = RunState(config);
    }

    // one forward step: takes the current token and position, returns logits for the next token
    std::span<float> forward(int token, int pos) {
        (void)token;
        (void)pos;
        // TODO(task 3): the module-08 pipeline, one token at one position:
        //   x = embedding[token]                          # table lookup (01)
        //   for l in 0..n_layers-1:
        //       xb = rmsnorm(x, rms_att_weight[l])        # (03)
        //       q = Wq[l] @ xb;  k = Wk[l] @ xb;  v = Wv[l] @ xb   # (04)
        //       k_cache[l][pos] = k;  v_cache[l][pos] = v # write cache first
        //       rope(q, k_cache[l][pos], pos)             # rotate in place (05)
        //       xb = attention(q, k_cache[l], v_cache[l], pos)     # (06)
        //       x += Wo[l] @ xb                           # projection + residual
        //       xb = rmsnorm(x, rms_ffn_weight[l])
        //       x += ffn(xb, W1[l], W2[l], W3[l])         # (07) + residual
        //   x = rmsnorm(x, rms_final_weight)
        //   logits = Wcls @ x                             # shared embedding table
        // Remember: layer l's wq starts at l * dim * dim; the KV cache row for
        // layer l starts at l * seq_len * kv_dim; residuals are +=, not =.
        return state.logits;
    }
};

// ---------------------------------------------------------------------------
// Module 02: BPE tokenizer (encode / decode)
// ---------------------------------------------------------------------------

struct TokenIndex {
    std::string_view str;
    int id;
};

class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizer_path, int vocab_size) : vocab_size_(vocab_size) {
        vocab_.resize(vocab_size);
        vocab_scores_.resize(vocab_size);
        // printable single-byte strings for the <0xXX> fallback pieces
        for (int i = 0; i < 256; i++) {
            byte_pieces_[i * 2] = static_cast<char>(i);
            byte_pieces_[i * 2 + 1] = '\0';
        }

        std::ifstream file(tokenizer_path, std::ios::binary);
        if (!file) { throw std::runtime_error("couldn't load " + tokenizer_path); }
        auto read_or_die = [&](void* dst, std::streamsize size) {
            if (!file.read(static_cast<char*>(dst), size)) {
                throw std::runtime_error("tokenizer file is truncated: " + tokenizer_path);
            }
        };
        read_or_die(&max_token_length_, sizeof(int));
        for (int i = 0; i < vocab_size; i++) {
            read_or_die(&vocab_scores_[i], sizeof(float));
            int len;
            read_or_die(&len, sizeof(int));
            std::string s(len, '\0');
            read_or_die(s.data(), len);
            vocab_[i] = std::move(s);
        }
    }

    // Returns the text piece for `token`; `prev_token` is needed to strip the
    // leading space right after BOS and to expand <0xXX> byte pieces.
    std::string_view decode(int prev_token, int token) const {
        (void)prev_token;
        (void)token;
        // TODO(task 4a): decode from module 02 — the raw vocab piece, minus the
        // leading space when prev_token == 1 (BOS), with <0xXX> pieces expanded
        // to their raw byte (byte_pieces_).
        return {};
    }

    // BPE encode: BOS + leading " " + per-character tokens, then greedy merges.
    std::vector<int> encode(const std::string& text, bool bos, bool eos) {
        (void)text;
        (void)bos;
        (void)eos;
        // TODO(task 4b): encode from module 02 — BOS + dummy " " prefix, per
        // UTF-8-character lookup with byte fallback (byte b -> id b + 3), then
        // greedy highest-score adjacent merges until none can merge, then EOS.
        return {};
    }

private:
    void init_sorted_vocab() {
        if (!sorted_vocab_.empty()) { return; }
        sorted_vocab_.reserve(vocab_size_);
        for (int i = 0; i < vocab_size_; i++) {
            sorted_vocab_.push_back({vocab_[i], i});
        }
        std::sort(sorted_vocab_.begin(), sorted_vocab_.end(),
                  [](const TokenIndex& a, const TokenIndex& b) { return a.str < b.str; });
    }

    int str_lookup(std::string_view str) {
        init_sorted_vocab();
        auto it = std::lower_bound(sorted_vocab_.begin(), sorted_vocab_.end(), str,
                                   [](const TokenIndex& a, std::string_view b) { return a.str < b; });
        if (it != sorted_vocab_.end() && it->str == str) { return it->id; }
        return -1;
    }

    std::vector<std::string> vocab_;
    std::vector<float> vocab_scores_;
    std::vector<TokenIndex> sorted_vocab_;
    int vocab_size_;
    unsigned int max_token_length_ = 0;
    std::array<char, 512> byte_pieces_{};
};

// ---------------------------------------------------------------------------
// Module 09: sampler (xorshift RNG, greedy / mult / top-p)
// ---------------------------------------------------------------------------

struct ProbIndex {
    float prob;
    int index;
};

class Sampler {
public:
    Sampler(int vocab_size, float temperature, float topp, std::uint64_t rng_seed)
        : temperature_(temperature), topp_(topp), rng_state_(rng_seed), probindex_(vocab_size) {}

    // Sample the next token from logits. May mutate logits.
    int sample(std::span<float> logits) {
        (void)logits;
        // TODO(task 5): sample from module 09 — greedy argmax when
        // temperature_ == 0; otherwise scale by temperature, softmax, draw one
        // coin, then full-distribution or top-p sampling depending on topp_.
        // One sample consumes exactly one random number.
        return 0;
    }

private:
    // Greedy: index of the largest probability.
    static int sample_argmax(std::span<const float> probabilities) {
        (void)probabilities;
        // TODO(task 5a): return the index of the maximum element.
        return 0;
    }

    // Full distribution: walk the CDF until the coin falls inside.
    static int sample_mult(std::span<const float> probabilities, float coin) {
        (void)probabilities;
        (void)coin;
        // TODO(task 5b): accumulate probabilities left to right and return the
        // first index whose cumulative sum exceeds the coin.
        return 0;
    }

    // Top-p (nucleus) sampling.
    int sample_topp(std::span<const float> probabilities, float topp, float coin) {
        (void)probabilities;
        (void)topp;
        (void)coin;
        // TODO(task 5c): discard tokens below the cutoff derived from topp,
        // sort survivors by probability descending, keep the smallest prefix
        // whose cumulative prob exceeds topp, sample within that prefix using
        // coin * prefix_total_prob.
        return 0;
    }

    // xorshift64 with a final multiply; must match the reference bit-for-bit.
    std::uint32_t random_u32() {
        // TODO(task 5d): advance the xorshift state and derive the 32-bit
        // output exactly as module 09 specifies (mind the 64-bit multiply).
        return 0;
    }
    float random_f32() {
        // TODO(task 5d): map a 32-bit draw to a float in [0, 1) as in module 09.
        return 0.0f;
    }

    float temperature_;
    float topp_;
    std::uint64_t rng_state_;
    std::vector<ProbIndex> probindex_;
};

// ---------------------------------------------------------------------------
// Module 10 itself: the generation loop (prefill / decode)
// ---------------------------------------------------------------------------

struct Generation {
    std::vector<int> ids; // every `next` token, in order (prompt "next"s included)
    std::string text;     // concatenated decode(token, next) pieces
};

// prefill = the first steps force-feed prompt tokens (one forward per step);
// decode  = after the prompt is consumed, sample 1 token per step, serially.
Generation generate(Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
                    const std::string& prompt, int steps) {
    (void)transformer;
    (void)tokenizer;
    (void)sampler;
    (void)prompt;
    (void)steps;
    // TODO(task 6): the loop from README.md —
    //   prompt_tokens = encode(prompt, bos=true)
    //   token = prompt_tokens[0];  pos = 0
    //   while pos < steps:
    //       logits = forward(token, pos)
    //       if pos < len(prompt_tokens) - 1:
    //           next = prompt_tokens[pos + 1]   # prefill: force-feed the next prompt token
    //       else:
    //           next = sample(logits)           # decode: the model decides
    //       pos += 1
    //       if next == 1: break                 # BOS again = stop
    //       ids.push_back(next)
    //       text += decode(token, next)         # note the (current, next) pair
    //       token = next
    // Hints:
    //   - the sampled run must consume one random number per step, in exactly
    //     the reference's order — one extra RNG call and everything diverges
    //   - optional: measure tok/s from the second token onward and print it to
    //     stderr; compare with the reference's "achieved tok/s"
    return {};
}

// ---------------------------------------------------------------------------
// output helpers (given — same formats as the golden data)
// ---------------------------------------------------------------------------

void write_ints(const std::string& path, std::span<const int> v) {
    std::ofstream f(path);
    if (!f) { throw std::runtime_error("cannot write " + path); }
    for (int x : v) { f << x << '\n'; }
}

void write_text(const std::string& path, const std::string& s) {
    std::ofstream f(path);
    if (!f) { throw std::runtime_error("cannot write " + path); }
    f << s;
}

} // namespace

// driver (given — no changes needed once the tasks above are done)
int main() {
    try {
        const std::string checkpoint_path = "../../stories15M.bin";
        const std::string tokenizer_path = "../../tokenizer.bin";
        const std::string prompt = "Once upon a time";
        const int steps = 64;

        // Run 1: greedy decoding (temperature = 0 takes the argmax path).
        {
            Transformer transformer(checkpoint_path);
            Tokenizer tokenizer(tokenizer_path, std::abs(transformer.config.vocab_size));
            Sampler sampler(std::abs(transformer.config.vocab_size),
                            /*temperature=*/0.0f, /*topp=*/0.9f, /*seed=*/42);
            const Generation g = generate(transformer, tokenizer, sampler, prompt, steps);
            write_ints("out_ids.txt", g.ids);
            write_text("out_text.txt", g.text + '\n');
            std::cout << "greedy:  " << g.ids.size() << " tokens -> out_ids.txt / out_text.txt\n";
        }

        // Run 2: sampled decoding (temperature = 0.8, top-p = 0.9, seed = 42).
        {
            Transformer transformer(checkpoint_path);
            Tokenizer tokenizer(tokenizer_path, std::abs(transformer.config.vocab_size));
            Sampler sampler(std::abs(transformer.config.vocab_size),
                            /*temperature=*/0.8f, /*topp=*/0.9f, /*seed=*/42);
            const Generation g = generate(transformer, tokenizer, sampler, prompt, steps);
            write_ints("out_sids.txt", g.ids);
            write_text("out_stext.txt", g.text + '\n');
            std::cout << "sampled: " << g.ids.size() << " tokens -> out_sids.txt / out_stext.txt\n";
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
