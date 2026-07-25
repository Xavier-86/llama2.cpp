// 10_generate — reference solution
//
// The grand FP32 assembly: checkpoint loader (01) + kernels (03/04/05/06/07)
// + full forward (08) + sampler (09) + tokenizer (02) wired into the
// prefill/decode generation loop. Passing this means you have rebuilt the
// FP32 run.cpp.
//
// Prompt is fixed to "Once upon a time" (-> [1, 9038, 2501, 263, 931]),
// 64 steps, two runs:
//   greedy:  temperature=0.0, topp=0.9, seed=42 -> out_ids.txt / out_text.txt
//   sampled: temperature=0.8, topp=0.9, seed=42 -> out_sids.txt / out_stext.txt
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution            (from this module folder)
// Verify: python3 ../tools/compare.py out_ids.txt   data/expected_greedy_ids.txt --exact
//         python3 ../tools/compare.py out_text.txt  data/expected_greedy_text.txt --text
//         python3 ../tools/compare.py out_sids.txt  data/expected_sampled_ids.txt --exact
//         python3 ../tools/compare.py out_stext.txt data/expected_sampled_text.txt --text

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
    std::span<const float> token_embedding_table; // (vocab_size, dim)
    std::span<const float> rms_att_weight;        // (layer, dim)
    std::span<const float> wq;                    // (layer, dim, n_heads * head_size)
    std::span<const float> wk;                    // (layer, dim, n_kv_heads * head_size)
    std::span<const float> wv;                    // (layer, dim, n_kv_heads * head_size)
    std::span<const float> wo;                    // (layer, n_heads * head_size, dim)
    std::span<const float> rms_ffn_weight;        // (layer, dim)
    std::span<const float> w1;                    // (layer, hidden_dim, dim)
    std::span<const float> w2;                    // (layer, dim, hidden_dim)
    std::span<const float> w3;                    // (layer, hidden_dim, dim)
    std::span<const float> rms_final_weight;      // (dim,)
    std::span<const float> wcls;                  // (vocab_size, dim), maybe shared
};

// Read the whole checkpoint, parse the config, and carve the 11 tensors out of
// the weight region in file order. Returns the owning buffer; the spans in `w`
// point into it, so the caller must keep it alive (and never reallocate it).
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
    const uint64_t vocab = shared ? cfg.vocab_size : -cfg.vocab_size;
    const uint64_t dim = cfg.dim;
    const uint64_t hidden = cfg.hidden_dim;
    const uint64_t n_layers = cfg.n_layers;
    const uint64_t head_size = dim / cfg.n_heads;
    const uint64_t kv_dim = cfg.n_kv_heads * head_size;

    const float* ptr = buf.data() + sizeof(Config) / sizeof(float);
    auto carve = [&ptr](uint64_t count) {
        std::span<const float> slice{ptr, static_cast<size_t>(count)};
        ptr += count;
        return slice;
    };
    w.token_embedding_table = carve(vocab * dim);
    w.rms_att_weight = carve(n_layers * dim);
    w.wq = carve(n_layers * dim * dim);
    w.wk = carve(n_layers * dim * kv_dim);
    w.wv = carve(n_layers * dim * kv_dim);
    w.wo = carve(n_layers * dim * dim);
    w.rms_ffn_weight = carve(n_layers * dim);
    w.w1 = carve(n_layers * hidden * dim);
    w.w2 = carve(n_layers * dim * hidden);
    w.w3 = carve(n_layers * hidden * dim);
    w.rms_final_weight = carve(dim);
    // legacy precomputed RoPE tables, unused: just skip them
    ptr += cfg.seq_len * head_size / 2;
    ptr += cfg.seq_len * head_size / 2;
    w.wcls = shared ? w.token_embedding_table : carve(vocab * dim);
    return buf;
}

// ---------------------------------------------------------------------------
// Modules 03/04: the math kernels (accumulate in float, matching the reference)
// ---------------------------------------------------------------------------

// out_i = weight_i * x_i / sqrt(mean(x^2) + eps); works in place (o == x).
void rmsnorm(std::span<float> o, std::span<const float> x, std::span<const float> weight) {
    float ss = 0.0f;
    for (float v : x) { ss += v * v; }
    ss /= x.size();
    ss += 1e-5f;
    ss = 1.0f / std::sqrt(ss);
    for (size_t j = 0; j < x.size(); j++) { o[j] = weight[j] * (ss * x[j]); }
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

// W (d,n) @ x (n,) -> xout (d,); w holds d rows of n elements, row-major.
// The vast majority of the model's runtime is spent in this small function:
// each weight participates in exactly one multiply-add, so performance is
// bound by memory bandwidth.
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
        const Config& p = config;
        Weights& w = weights;
        RunState& s = state;
        const int dim = p.dim;
        const int head_size = dim / p.n_heads;
        const int kvd = p.n_kv_heads * head_size;
        const int kv_mul = p.n_heads / p.n_kv_heads; // kv sharing factor for GQA
        const int hidden_dim = p.hidden_dim;

        std::span<float> x{s.x};
        // copy the token's embedding row into x
        std::ranges::copy(w.token_embedding_table.subspan(static_cast<size_t>(token) * dim, dim),
                          x.begin());

        for (int l = 0; l < p.n_layers; l++) {
            const size_t loff = static_cast<size_t>(l) * p.seq_len * kvd;

            // k/v are views into the KV cache at this layer and position: each step
            // computes k/v only for the current token; earlier positions are read
            // back from the cache, not recomputed
            std::span<float> k = std::span{s.key_cache}.subspan(loff + static_cast<size_t>(pos) * kvd, kvd);
            std::span<float> v = std::span{s.value_cache}.subspan(loff + static_cast<size_t>(pos) * kvd, kvd);

            // rmsnorm before attention, then qkv projections for the current position
            rmsnorm(s.xb, x, w.rms_att_weight.subspan(static_cast<size_t>(l) * dim, dim));
            matmul(s.q, s.xb, w.wq.subspan(static_cast<size_t>(l) * dim * dim, static_cast<size_t>(dim) * dim));
            matmul(k, s.xb, w.wk.subspan(static_cast<size_t>(l) * dim * kvd, static_cast<size_t>(dim) * kvd));
            matmul(v, s.xb, w.wv.subspan(static_cast<size_t>(l) * dim * kvd, static_cast<size_t>(dim) * kvd));

            // RoPE (module 05): rotate q/k pairs as complex numbers within each head
            for (int i = 0; i < dim; i += 2) {
                const int head_dim = i % head_size;
                const float freq = 1.0f / std::pow(10000.0f, head_dim / static_cast<float>(head_size));
                const float angle = pos * freq;
                const float fcr = std::cos(angle);
                const float fci = std::sin(angle);
                const int rotn = i < kvd ? 2 : 1; // 2 = rotate both q and k, 1 = rotate q only
                for (int rot = 0; rot < rotn; rot++) {
                    std::span<float> vec = rot == 0 ? std::span{s.q} : k;
                    const float v0 = vec[i];
                    const float v1 = vec[i + 1];
                    vec[i]     = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }

            // multi-head causal attention (module 06)
            for (int h = 0; h < p.n_heads; h++) {
                std::span qh = std::span{s.q}.subspan(static_cast<size_t>(h) * head_size, head_size);
                std::span att = std::span{s.att}.subspan(static_cast<size_t>(h) * p.seq_len,
                                                         static_cast<size_t>(pos) + 1);
                // score against history + current only: t in [0, pos] is the causal mask;
                // attention cost grows linearly with the sequence length
                for (int t = 0; t <= pos; t++) {
                    std::span key = std::span{s.key_cache}.subspan(
                        loff + static_cast<size_t>(t) * kvd + static_cast<size_t>(h / kv_mul) * head_size,
                        head_size);
                    float score = 0.0f;
                    for (int i = 0; i < head_size; i++) { score += qh[i] * key[i]; }
                    score /= std::sqrt(static_cast<float>(head_size));
                    att[t] = score;
                }

                softmax(att);

                // weighted sum of the V rows, written into this head's slice of xb
                std::span xb = std::span{s.xb}.subspan(static_cast<size_t>(h) * head_size, head_size);
                std::ranges::fill(xb, 0.0f);
                for (int t = 0; t <= pos; t++) {
                    std::span val = std::span{s.value_cache}.subspan(
                        loff + static_cast<size_t>(t) * kvd + static_cast<size_t>(h / kv_mul) * head_size,
                        head_size);
                    const float a = att[t];
                    for (int i = 0; i < head_size; i++) { xb[i] += a * val[i]; }
                }
            }

            // attention output projection + residual connection
            matmul(s.xb2, s.xb, w.wo.subspan(static_cast<size_t>(l) * dim * dim, static_cast<size_t>(dim) * dim));
            for (int i = 0; i < dim; i++) { x[i] += s.xb2[i]; }

            // rmsnorm before the ffn
            rmsnorm(s.xb, x, w.rms_ffn_weight.subspan(static_cast<size_t>(l) * dim, dim));

            // FFN (module 07): w2(silu(w1(x)) * w3(x)); compute w1 and w3 first
            matmul(s.hb, s.xb, w.w1.subspan(static_cast<size_t>(l) * dim * hidden_dim,
                                            static_cast<size_t>(dim) * hidden_dim));
            matmul(s.hb2, s.xb, w.w3.subspan(static_cast<size_t>(l) * dim * hidden_dim,
                                             static_cast<size_t>(dim) * hidden_dim));

            // SwiGLU nonlinearity: silu(v) = v * sigmoid(v)
            for (int i = 0; i < hidden_dim; i++) {
                float val = s.hb[i];
                val *= (1.0f / (1.0f + std::exp(-val)));
                val *= s.hb2[i];
                s.hb[i] = val;
            }

            // ffn output projection + residual connection
            matmul(s.xb, s.hb, w.w2.subspan(static_cast<size_t>(l) * hidden_dim * dim,
                                            static_cast<size_t>(hidden_dim) * dim));
            for (int i = 0; i < dim; i++) { x[i] += s.xb[i]; }
        }

        // final rmsnorm (in place) + classifier head to get the logits
        rmsnorm(x, x, w.rms_final_weight);
        matmul(s.logits, x, w.wcls);
        return s.logits;
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
        std::string_view piece = vocab_[token];
        if (prev_token == 1 && piece.starts_with(' ')) { piece.remove_prefix(1); }
        if (piece.size() == 6 && piece.starts_with("<0x") && piece.back() == '>') {
            unsigned int byte_val = 0;
            const char* hex = piece.data() + 3;
            auto [end, ec] = std::from_chars(hex, hex + 2, byte_val, 16);
            if (ec == std::errc{} && end == hex + 2) {
                return {byte_pieces_.data() + byte_val * 2, 1};
            }
        }
        return piece;
    }

    // BPE encode: BOS + leading " " + per-character tokens, then greedy merges.
    std::vector<int> encode(const std::string& text, bool bos, bool eos) {
        std::vector<int> tokens;
        if (bos) { tokens.push_back(1); } // 1 = <s>
        // Llama convention: pretend the text started with a space.
        if (!text.empty()) { tokens.push_back(str_lookup(" ")); }

        // Per-character lookup on UTF-8 boundaries: a byte with
        // (c & 0xC0) != 0x80 starts a new character.
        std::string str_buffer;
        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            if ((c & 0xC0) != 0x80) { str_buffer.clear(); }
            str_buffer.push_back(c);
            if (i + 1 < text.size() && (text[i + 1] & 0xC0) == 0x80 && str_buffer.size() < 4) {
                continue; // more continuation bytes belong to this character
            }
            int id = str_lookup(str_buffer);
            if (id != -1) {
                tokens.push_back(id);
            } else {
                // Byte fallback: byte b -> token id b + 3.
                for (unsigned char b : str_buffer) { tokens.push_back(b + 3); }
            }
            str_buffer.clear();
        }

        // Greedy merging: merge the highest-scoring adjacent pair each round.
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;
            for (size_t i = 0; i + 1 < tokens.size(); i++) {
                std::string merged = vocab_[tokens[i]] + vocab_[tokens[i + 1]];
                int id = str_lookup(merged);
                if (id != -1 && vocab_scores_[id] > best_score) {
                    best_score = vocab_scores_[id];
                    best_id = id;
                    best_idx = static_cast<int>(i);
                }
            }
            if (best_idx == -1) { break; }
            tokens[best_idx] = best_id;
            tokens.erase(tokens.begin() + best_idx + 1);
        }

        if (eos) { tokens.push_back(2); } // 2 = </s>
        return tokens;
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
    float random_f32() { return (random_u32() >> 8) / 16777216.0f; }

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
    std::vector<int> prompt_tokens = tokenizer.encode(prompt, /*bos=*/true, /*eos=*/false);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("something is wrong, expected at least 1 prompt token");
    }
    const int num_prompt_tokens = static_cast<int>(prompt_tokens.size());

    Generation result;
    std::chrono::steady_clock::time_point start{};
    int next = 0;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        std::span<float> logits = transformer.forward(token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1]; // prefill: force-feed the next prompt token
        } else {
            next = sampler.sample(logits); // decode: the model decides
        }
        pos++;

        if (next == 1) { break; } // BOS again = stop

        result.ids.push_back(next);
        result.text += tokenizer.decode(token, next); // note the (current, next) pair
        token = next;

        // the first step may be slow; measure tok/s from the second token on
        if (pos == 1) { start = std::chrono::steady_clock::now(); }
    }
    if (pos > 1) {
        const auto end = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cerr << "achieved tok/s: " << (pos - 1) / ms * 1000.0 << '\n';
    }
    return result;
}

// ---------------------------------------------------------------------------
// output helpers (same formats as the golden data)
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
