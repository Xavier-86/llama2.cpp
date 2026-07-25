// Inference for Llama-2 Transformer model in modern C++ (C++20).
//
// A C++ rewrite of karpathy's run.c. Numerically equivalent to the original,
// but idiomatic C++ throughout:
//   - buffers              -> std::vector owns the memory; there are no parallel
//                             raw-pointer members shadowing each vector
//   - buffer arguments     -> std::span (non-owning view carrying its own size;
//                             kernels no longer take pointer + length pairs)
//   - weight slices        -> std::span into the mmap region (non-owning)
//   - mmap/munmap/close    -> MappedFile class (RAII: mapped on construction,
//                             unmapped and closed on destruction)
//   - errors               -> exceptions (std::runtime_error), caught once in main
//   - char** vocab         -> std::vector<std::string>; token pieces -> std::string_view
//   - qsort/bsearch        -> std::sort / std::lower_bound
//   - printf/sscanf        -> iostreams / std::from_chars
//   - clock_gettime        -> std::chrono
// The hot loops themselves stay plain: inference is memory-bandwidth bound,
// so in the numeric kernels readability beats cleverness.

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

// ----------------------------------------------------------------------------
// Transformer model

struct Config {
    int dim;        // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers;   // number of layers
    int n_heads;    // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len;    // max sequence length
};

int kv_dim(const Config& p) { return (p.dim * p.n_kv_heads) / p.n_heads; }

// The weights own no memory: they are views into the mmap region. A non-owning
// buffer in C++ is a std::span; ownership lives in MappedFile.
struct TransformerWeights {
    std::span<float> token_embedding_table; // (vocab_size, dim)
    std::span<float> rms_att_weight;        // (layer, dim) rmsnorm weights
    std::span<float> rms_ffn_weight;        // (layer, dim)
    std::span<float> wq;                    // (layer, dim, n_heads * head_size)
    std::span<float> wk;                    // (layer, dim, n_kv_heads * head_size)
    std::span<float> wv;                    // (layer, dim, n_kv_heads * head_size)
    std::span<float> wo;                    // (layer, n_heads * head_size, dim)
    std::span<float> w1;                    // (layer, hidden_dim, dim)
    std::span<float> w2;                    // (layer, dim, hidden_dim)
    std::span<float> w3;                    // (layer, hidden_dim, dim)
    std::span<float> rms_final_weight;      // (dim,)
    std::span<float> wcls;                  // (optional) classifier weights for the logits
};

// RunState: intermediate buffers for the forward pass. Each std::vector owns its
// memory and is zero-initialized on construction; there is no free_run_state.
struct RunState {
    std::vector<float> x;           // activation at current time stamp (dim,)
    std::vector<float> xb;          // same, but inside a residual branch (dim,)
    std::vector<float> xb2;         // an additional buffer just for convenience (dim,)
    std::vector<float> hb;          // buffer for hidden dimension in the ffn (hidden_dim,)
    std::vector<float> hb2;         // buffer for hidden dimension in the ffn (hidden_dim,)
    std::vector<float> q;           // query (dim,)
    std::vector<float> att;         // buffer for scores/attention values (n_heads, seq_len)
    std::vector<float> logits;      // output logits (vocab_size,)
    std::vector<float> key_cache;   // (layer, seq_len, kv_dim) -- KV cache
    std::vector<float> value_cache; // (layer, seq_len, kv_dim)

    explicit RunState(const Config& p)
        : x(p.dim), xb(p.dim), xb2(p.dim), hb(p.hidden_dim), hb2(p.hidden_dim), q(p.dim),
          att(static_cast<size_t>(p.n_heads) * p.seq_len), logits(p.vocab_size),
          key_cache(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim(p)),
          value_cache(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim(p)) {}
};

// RAII wrapper around mmap: construction reads the config header and maps the
// whole file; destruction unmaps and closes automatically. mmap itself stays
// POSIX -- there is no standard C++ facility for it -- but ownership is C++.
class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
        {
            std::ifstream file(path, std::ios::binary);
            if (!file) { throw std::runtime_error("couldn't open checkpoint " + path); }
            // read the config header (a negative vocab_size marks unshared weights)
            if (!file.read(reinterpret_cast<char*>(&config), sizeof(config))) {
                throw std::runtime_error("couldn't read config header of " + path);
            }
        }
        shared_weights = config.vocab_size > 0;
        config.vocab_size = std::abs(config.vocab_size);
        file_size_ = std::filesystem::file_size(path);

        // mmap the whole file (read-only); weights are mapped into the address space with zero copy
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ == -1) { throw std::runtime_error("open failed: " + path); }
        data_ = static_cast<float*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            throw std::runtime_error("mmap failed: " + path);
        }
    }

    ~MappedFile() {
        if (data_ != nullptr) { munmap(data_, file_size_); }
        if (fd_ != -1) { ::close(fd_); }
    }

    // copying forbidden (the mmap region can have only one owner), moving allowed
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    Config config;
    bool shared_weights = false;

    // start of the weight region, past the config header
    float* weights_data() const { return data_ + sizeof(Config) / sizeof(float); }

private:
    float* data_ = nullptr;
    size_t file_size_ = 0;
    int fd_ = -1;
};

struct Transformer {
private:
    // mapped_ must be the first declared member: C++ initializes members in
    // declaration order, and config/state/weights all depend on its mmap data
    MappedFile mapped_;

public:
    Config config;
    TransformerWeights weights;
    RunState state;

    explicit Transformer(const std::string& checkpoint_path)
        : mapped_(checkpoint_path), config(mapped_.config), state(config) {
        map_weights(mapped_.weights_data());
    }

    // one forward step: takes the current token and position, returns logits for the next token
    std::span<float> forward(int token, int pos);

private:
    void map_weights(float* ptr) {
        const Config& p = config;
        const int head_size = p.dim / p.n_heads;
        const std::uint64_t n_layers = p.n_layers; // 64-bit multiply, compatible with 13B+ models
        // carve one slice of `count` floats out of the weight region and advance
        auto carve = [&ptr](std::uint64_t count) {
            std::span<float> slice{ptr, static_cast<size_t>(count)};
            ptr += count;
            return slice;
        };
        weights.token_embedding_table = carve(static_cast<std::uint64_t>(p.vocab_size) * p.dim);
        weights.rms_att_weight = carve(n_layers * p.dim);
        weights.wq = carve(n_layers * p.dim * (p.n_heads * head_size));
        weights.wk = carve(n_layers * p.dim * (p.n_kv_heads * head_size));
        weights.wv = carve(n_layers * p.dim * (p.n_kv_heads * head_size));
        weights.wo = carve(n_layers * (p.n_heads * head_size) * p.dim);
        weights.rms_ffn_weight = carve(n_layers * p.dim);
        weights.w1 = carve(n_layers * p.dim * p.hidden_dim);
        weights.w2 = carve(n_layers * p.hidden_dim * p.dim);
        weights.w3 = carve(n_layers * p.dim * p.hidden_dim);
        weights.rms_final_weight = carve(p.dim);
        // skip what used to be freq_cis_real / freq_cis_imag (precomputed RoPE tables)
        ptr += static_cast<std::uint64_t>(p.seq_len) * head_size / 2;
        ptr += static_cast<std::uint64_t>(p.seq_len) * head_size / 2;
        weights.wcls = mapped_.shared_weights
                           ? weights.token_embedding_table
                           : carve(static_cast<std::uint64_t>(p.vocab_size) * p.dim);
    }
};

// ----------------------------------------------------------------------------
// neural net blocks. Kernels take spans, so sizes travel with the buffers
// instead of as separate int parameters.

void rmsnorm(std::span<float> o, std::span<const float> x, std::span<const float> weight) {
    float ss = 0.0f;
    for (float v : x) { ss += v * v; }
    ss /= x.size();
    ss += 1e-5f;
    ss = 1.0f / std::sqrt(ss);
    for (size_t j = 0; j < x.size(); j++) { o[j] = weight[j] * (ss * x[j]); }
}

void softmax(std::span<float> x) {
    const float max_val = *std::ranges::max_element(x);
    float sum = 0.0f;
    for (float& v : x) {
        v = std::exp(v - max_val);
        sum += v;
    }
    for (float& v : x) { v /= sum; }
}

void matmul(std::span<float> xout, std::span<const float> x, std::span<const float> w) {
    // W (d,n) @ x (n,) -> xout (d,); w holds d rows of n elements
    // The vast majority of the model's runtime is spent in this small function:
    // each weight participates in exactly one multiply-add, so performance is bound by memory bandwidth.
    const size_t n = x.size();
    const size_t d = xout.size();
    #pragma omp parallel for
    for (size_t i = 0; i < d; i++) {
        const std::span w_row = w.subspan(i * n, n);
        float val = 0.0f;
        for (size_t j = 0; j < n; j++) { val += w_row[j] * x[j]; }
        xout[i] = val;
    }
}

std::span<float> Transformer::forward(int token, int pos) {
    const Config& p = config;
    TransformerWeights& w = weights;
    RunState& s = state;
    const int dim = p.dim;
    const int kvd = kv_dim(p);
    const int kv_mul = p.n_heads / p.n_kv_heads; // kv sharing factor for multiquery
    const int hidden_dim = p.hidden_dim;
    const int head_size = dim / p.n_heads;

    std::span<float> x{s.x};
    // copy the token's embedding row into x
    std::ranges::copy(w.token_embedding_table.subspan(static_cast<size_t>(token) * dim, dim),
                      x.begin());

    for (int l = 0; l < p.n_layers; l++) {
        const size_t loff = static_cast<size_t>(l) * p.seq_len * kvd;

        // k/v are views into the KV cache at this layer and position -- that is the "cache": each step
        // computes k/v only for the current token; k/v of all previous positions are read from the cache, not recomputed
        std::span<float> k = std::span{s.key_cache}.subspan(loff + static_cast<size_t>(pos) * kvd, kvd);
        std::span<float> v = std::span{s.value_cache}.subspan(loff + static_cast<size_t>(pos) * kvd, kvd);

        // rmsnorm before attention, then qkv projections for the current position
        rmsnorm(s.xb, x, w.rms_att_weight.subspan(static_cast<size_t>(l) * dim, dim));
        matmul(s.q, s.xb, w.wq.subspan(static_cast<size_t>(l) * dim * dim, static_cast<size_t>(dim) * dim));
        matmul(k, s.xb, w.wk.subspan(static_cast<size_t>(l) * dim * kvd, static_cast<size_t>(dim) * kvd));
        matmul(v, s.xb, w.wv.subspan(static_cast<size_t>(l) * dim * kvd, static_cast<size_t>(dim) * kvd));

        // RoPE rotary position embedding: rotate q/k pairs as complex numbers within each head
        for (int i = 0; i < dim; i += 2) {
            const int head_dim = i % head_size;
            const float freq = 1.0f / std::pow(10000.0f, head_dim / static_cast<float>(head_size));
            const float val = pos * freq;
            const float fcr = std::cos(val);
            const float fci = std::sin(val);
            const int rotn = i < kvd ? 2 : 1; // 2 = rotate both q and k, 1 = rotate q only
            for (int rot = 0; rot < rotn; rot++) {
                std::span<float> vec = rot == 0 ? std::span{s.q} : k;
                const float v0 = vec[i];
                const float v1 = vec[i + 1];
                vec[i]     = v0 * fcr - v1 * fci;
                vec[i + 1] = v0 * fci + v1 * fcr;
            }
        }

        // multihead attention
        #pragma omp parallel for
        for (int h = 0; h < p.n_heads; h++) {
            std::span qh = std::span{s.q}.subspan(static_cast<size_t>(h) * head_size, head_size);
            std::span att = std::span{s.att}.subspan(static_cast<size_t>(h) * p.seq_len,
                                                     static_cast<size_t>(pos) + 1);
            // dot products with k of all history positions in [0, pos] -- the larger pos, the slower this step;
            // this is attention cost growing linearly with sequence length in action
            for (int t = 0; t <= pos; t++) {
                std::span key = std::span{s.key_cache}.subspan(
                    loff + static_cast<size_t>(t) * kvd + (h / kv_mul) * head_size, head_size);
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) { score += qh[i] * key[i]; }
                score /= std::sqrt(static_cast<float>(head_size));
                att[t] = score;
            }

            softmax(att);

            // weighted sum of v with the attention weights
            std::span xb = std::span{s.xb}.subspan(static_cast<size_t>(h) * head_size, head_size);
            std::ranges::fill(xb, 0.0f);
            for (int t = 0; t <= pos; t++) {
                std::span val = std::span{s.value_cache}.subspan(
                    loff + static_cast<size_t>(t) * kvd + (h / kv_mul) * head_size, head_size);
                const float a = att[t];
                for (int i = 0; i < head_size; i++) { xb[i] += a * val[i]; }
            }
        }

        // attention output projection + residual connection
        matmul(s.xb2, s.xb, w.wo.subspan(static_cast<size_t>(l) * dim * dim, static_cast<size_t>(dim) * dim));
        for (int i = 0; i < dim; i++) { x[i] += s.xb2[i]; }

        // rmsnorm before the ffn
        rmsnorm(s.xb, x, w.rms_ffn_weight.subspan(static_cast<size_t>(l) * dim, dim));

        // FFN: w2(silu(w1(x)) * w3(x)); compute w1 and w3 first
        matmul(s.hb, s.xb, w.w1.subspan(static_cast<size_t>(l) * dim * hidden_dim,
                                        static_cast<size_t>(dim) * hidden_dim));
        matmul(s.hb2, s.xb, w.w3.subspan(static_cast<size_t>(l) * dim * hidden_dim,
                                         static_cast<size_t>(dim) * hidden_dim));

        // SwiGLU nonlinearity
        for (int i = 0; i < hidden_dim; i++) {
            float val = s.hb[i];
            val *= (1.0f / (1.0f + std::exp(-val))); // silu(x) = x * sigmoid(x)
            val *= s.hb2[i];
            s.hb[i] = val;
        }

        // ffn output projection + residual connection
        matmul(s.xb, s.hb, w.w2.subspan(static_cast<size_t>(l) * hidden_dim * dim,
                                        static_cast<size_t>(hidden_dim) * dim));
        for (int i = 0; i < dim; i++) { x[i] += s.xb[i]; }
    }

    // final rmsnorm + classifier head to get the logits
    rmsnorm(x, x, w.rms_final_weight);
    matmul(s.logits, x, w.wcls);
    return s.logits;
}

// ----------------------------------------------------------------------------
// BPE Tokenizer
// std::vector<std::string> owns the vocab; lookups are binary search over a
// sorted index; token pieces are passed around as std::string_view.

struct TokenIndex {
    std::string_view str; // string_view: owns no string, just "views" an entry in vocab
    int id;
};

class Tokenizer {
public:
    Tokenizer(const std::string& tokenizer_path, int vocab_size) : vocab_size_(vocab_size) {
        vocab_.resize(vocab_size);
        vocab_scores_.resize(vocab_size);
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
            std::string s(len, '\0'); // construct a fixed-length string directly; no malloc + manual '\0'
            read_or_die(s.data(), len);
            vocab_[i] = std::move(s);
        }
    }

    // token id -> string piece
    std::string_view decode(int prev_token, int token) const {
        std::string_view piece = vocab_[token];
        // sentencepiece strips the leading space after BOS(1)
        if (prev_token == 1 && piece.starts_with(' ')) { piece.remove_prefix(1); }
        // tokens of the form '<0x01>' are raw bytes; convert to the real byte
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

    // string -> token sequence; returns a vector directly, so the caller neither
    // estimates an upper bound nor passes int* + n_tokens out-params
    std::vector<int> encode(const std::string& text, bool bos, bool eos) {
        init_sorted_vocab();

        std::vector<int> tokens;
        if (bos) { tokens.push_back(1); }

        // sentencepiece's add_dummy_prefix: prepend a " " to non-empty text
        if (!text.empty()) {
            tokens.push_back(str_lookup(" "));
        }

        // split UTF-8 by code point: continuation bytes (10xxxxxx) join the previous code point
        std::string str_buffer;
        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            if ((c & 0xC0) != 0x80) { str_buffer.clear(); } // start byte of a new code point
            str_buffer.push_back(c);
            // next byte is still a continuation byte and we're under the length limit; keep accumulating
            if (i + 1 < text.size() && (text[i + 1] & 0xC0) == 0x80 && str_buffer.size() < 4) {
                continue;
            }
            int id = str_lookup(str_buffer);
            if (id != -1) {
                tokens.push_back(id);
            } else {
                // byte_fallback: encode each byte as byte+3 (the first 3 ids are <unk>, <s>, </s>)
                for (unsigned char b : str_buffer) { tokens.push_back(b + 3); }
            }
            str_buffer.clear();
        }

        // BPE merging: each round merges the highest-scoring adjacent pair until none can merge
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
            tokens.erase(tokens.begin() + best_idx + 1); // one vector line replaces a manual memmove shift
        }

        if (eos) { tokens.push_back(2); }
        return tokens;
    }

private:
    void init_sorted_vocab() {
        if (!sorted_vocab_.empty()) { return; } // lazy initialization
        sorted_vocab_.reserve(vocab_size_);
        for (int i = 0; i < vocab_size_; i++) {
            sorted_vocab_.push_back({vocab_[i], i});
        }
        std::sort(sorted_vocab_.begin(), sorted_vocab_.end(),
                  [](const TokenIndex& a, const TokenIndex& b) { return a.str < b.str; });
    }

    int str_lookup(std::string_view str) const {
        // std::lower_bound replaces bsearch: binary search over the sorted table
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
    std::array<char, 512> byte_pieces_{}; // all single-byte strings
};

void safe_print(std::string_view piece) {
    // print only printable characters or whitespace; skip control bytes
    if (piece.empty()) { return; }
    if (piece.size() == 1) {
        unsigned char byte_val = piece[0];
        if (!(std::isprint(byte_val) || std::isspace(byte_val))) { return; }
    }
    std::cout << piece;
}

// ----------------------------------------------------------------------------
// Sampler: logits -> next token

struct ProbIndex {
    float prob;
    int index;
};

class Sampler {
public:
    Sampler(int vocab_size, float temperature, float topp, std::uint64_t rng_seed)
        : temperature_(temperature), topp_(topp), rng_state_(rng_seed), probindex_(vocab_size) {}

    int sample(std::span<float> logits) {
        if (temperature_ == 0.0f) {
            // greedy: take argmax directly (this is the path taken at temperature=0)
            return sample_argmax(logits);
        }
        for (float& v : logits) { v /= temperature_; }
        softmax(logits);
        const float coin = random_f32();
        if (topp_ <= 0 || topp_ >= 1) {
            return sample_mult(logits, coin);
        }
        return sample_topp(logits, topp_, coin);
    }

private:
    static int sample_argmax(std::span<const float> probabilities) {
        // max_element returns the first maximum, same tie-breaking as a manual scan
        return static_cast<int>(std::ranges::max_element(probabilities) - probabilities.begin());
    }

    static int sample_mult(std::span<const float> probabilities, float coin) {
        // sample from the cumulative distribution
        float cdf = 0.0f;
        for (size_t i = 0; i < probabilities.size(); i++) {
            cdf += probabilities[i];
            if (coin < cdf) { return static_cast<int>(i); }
        }
        return static_cast<int>(probabilities.size()) - 1; // guard against rounding error
    }

    int sample_topp(std::span<const float> probabilities, float topp, float coin) {
        // top-p (nucleus) sampling: sample only within the smallest token set whose cumulative probability exceeds topp
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

    // xorshift RNG (same as run.c, so identical seeds give identical output)
    std::uint32_t random_u32() {
        rng_state_ ^= rng_state_ >> 12;
        rng_state_ ^= rng_state_ << 25;
        rng_state_ ^= rng_state_ >> 27;
        return (rng_state_ * 0x2545F4914F6CDD1Dull) >> 32;
    }
    float random_f32() { return (random_u32() >> 8) / 16777216.0f; }

    float temperature_;
    float topp_;
    std::uint64_t rng_state_;
    std::vector<ProbIndex> probindex_;
};

// ----------------------------------------------------------------------------
// utilities: time

std::int64_t time_in_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

// ----------------------------------------------------------------------------
// generation loop -- the minimal prefill/decode prototype
// prefill = the first steps force-feed prompt tokens (one forward per step, but parallelizable);
// decode  = after the prompt is consumed, sample 1 token per step, serially, one full forward per step.

void generate(Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
              const std::string& prompt, int steps) {
    std::vector<int> prompt_tokens = tokenizer.encode(prompt, /*bos=*/true, /*eos=*/false);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("something is wrong, expected at least 1 prompt token");
    }
    const int num_prompt_tokens = static_cast<int>(prompt_tokens.size());

    std::int64_t start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {

        std::span<float> logits = transformer.forward(token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1]; // prefill phase: force-feed the next prompt token
        } else {
            next = sampler.sample(logits); // decode phase: sample from the model output
        }
        pos++;

        if (next == 1) { break; } // BOS(1) terminates the sequence

        safe_print(tokenizer.decode(token, next));
        std::cout << std::flush;
        token = next;

        if (start == 0) { start = time_in_ms(); } // the first step may be slow; time from the second step on
    }
    std::cout << '\n';

    if (pos > 1) {
        std::int64_t end = time_in_ms();
        std::cerr << "achieved tok/s: " << (pos - 1) / static_cast<double>(end - start) * 1000 << '\n';
    }
}

std::string read_stdin(const std::string& guide) {
    std::cout << guide;
    std::string line;
    std::getline(std::cin, line);
    return line;
}

// ----------------------------------------------------------------------------
// chat loop (proof of concept)

void chat(Transformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
          const std::string& cli_user_prompt, const std::string& cli_system_prompt, int steps) {

    std::string system_prompt;
    std::string user_prompt;
    std::string rendered_prompt;
    std::vector<int> prompt_tokens;
    size_t user_idx = 0;

    bool user_turn = true;
    int next = 0;
    int token = 0;
    int pos = 0;
    while (pos < steps) {

        if (user_turn) {
            if (pos == 0) {
                system_prompt = cli_system_prompt.empty()
                                    ? read_stdin("Enter system prompt (optional): ")
                                    : cli_system_prompt;
            }
            if (pos == 0 && !cli_user_prompt.empty()) {
                user_prompt = cli_user_prompt;
            } else {
                user_prompt = read_stdin("User: ");
            }
            // render into the Llama 2 Chat template
            if (pos == 0 && !system_prompt.empty()) {
                rendered_prompt = "[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n" + user_prompt + " [/INST]";
            } else {
                rendered_prompt = "[INST] " + user_prompt + " [/INST]";
            }
            prompt_tokens = tokenizer.encode(rendered_prompt, /*bos=*/true, /*eos=*/false);
            user_idx = 0;
            user_turn = false;
            std::cout << "Assistant: ";
        }

        if (user_idx < prompt_tokens.size()) {
            token = prompt_tokens[user_idx++]; // still feeding the prompt
        } else {
            token = next; // use the token sampled last round
        }
        if (token == 2) { user_turn = true; } // EOS(2) ends the Assistant turn

        std::span<float> logits = transformer.forward(token, pos);
        next = sampler.sample(logits);
        pos++;

        if (user_idx >= prompt_tokens.size() && next != 2) {
            safe_print(tokenizer.decode(token, next));
            std::cout << std::flush;
        }
        if (next == 2) { std::cout << '\n'; }
    }
    std::cout << '\n';
}

// ----------------------------------------------------------------------------
// CLI

[[noreturn]] void error_usage() {
    throw std::runtime_error(
        "Usage:   run <checkpoint> [options]\n"
        "Example: run model.bin -n 256 -i \"Once upon a time\"\n"
        "Options:\n"
        "  -t <float>  temperature in [0,inf], default 1.0\n"
        "  -p <float>  p value in top-p (nucleus) sampling in [0,1] default 0.9\n"
        "  -s <int>    random seed, default time(NULL)\n"
        "  -n <int>    number of steps to run for, default 256. 0 = max_seq_len\n"
        "  -i <string> input prompt\n"
        "  -z <string> optional path to custom tokenizer\n"
        "  -m <string> mode: generate|chat, default: generate\n"
        "  -y <string> (optional) system prompt in chat mode");
}

int main(int argc, char* argv[]) {
    try {
        // default parameters; std::string replaces char*, no more buffer worries
        std::string checkpoint_path;
        std::string tokenizer_path = "tokenizer.bin";
        float temperature = 1.0f;
        float topp = 0.9f;
        int steps = 256;
        std::string prompt;
        std::uint64_t rng_seed = 0;
        std::string mode = "generate";
        std::string system_prompt;

        if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 >= argc) { error_usage(); }
            std::string flag = argv[i];
            if (flag.size() != 2 || flag[0] != '-') { error_usage(); }
            switch (flag[1]) {
                case 't': temperature = std::atof(argv[i + 1]); break;
                case 'p': topp = std::atof(argv[i + 1]); break;
                case 's': rng_seed = std::stoull(argv[i + 1]); break;
                case 'n': steps = std::atoi(argv[i + 1]); break;
                case 'i': prompt = argv[i + 1]; break;
                case 'z': tokenizer_path = argv[i + 1]; break;
                case 'm': mode = argv[i + 1]; break;
                case 'y': system_prompt = argv[i + 1]; break;
                default: error_usage();
            }
        }

        if (rng_seed == 0) { rng_seed = static_cast<std::uint64_t>(std::time(nullptr)); }
        if (temperature < 0.0) { temperature = 0.0; }
        if (topp < 0.0 || 1.0 < topp) { topp = 0.9; }
        if (steps < 0) { steps = 0; }

        // construction completes loading (RAII); leaving the scope frees everything
        Transformer transformer(checkpoint_path);
        if (steps == 0 || steps > transformer.config.seq_len) { steps = transformer.config.seq_len; }

        Tokenizer tokenizer(tokenizer_path, transformer.config.vocab_size);
        Sampler sampler(transformer.config.vocab_size, temperature, topp, rng_seed);

        if (mode == "generate") {
            generate(transformer, tokenizer, sampler, prompt, steps);
        } else if (mode == "chat") {
            chat(transformer, tokenizer, sampler, prompt, system_prompt, steps);
        } else {
            throw std::runtime_error("unknown mode: " + mode);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
