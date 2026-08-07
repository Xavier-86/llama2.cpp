// Inference for Llama-2 Transformer model in modern C++ (C++20), int8 quantized forward pass.
//
// The only difference from run.cpp (the FP32 version) is quantization. Read
// run.cpp first, then diff against this file to see what quantization changes:
//   1. weight storage: float -> int8 + one float scale factor per GS-element group (QuantizedTensor)
//   2. matmul: int8 x int8 integer multiply-adds (int32 accumulation), rescaled by both scale factors per group
//   3. forward: quantize activations to int8 before every matmul (quantization-aware forward)
//   4. memory savings: weight reads drop to 1/4 plus scale-factor overhead -- in a
//      bandwidth-bound setting this is what speeds up decode (tokens/s ~ bandwidth / bytes read per token)
// Tokenizer / Sampler / generate / chat are identical to run.cpp; a copy is kept
// here to keep the file self-contained. See run.cpp for detailed comments on those parts.

/*
Parameters used in this file:

Model config (struct Config, read from the checkpoint header):
  dim         transformer dimension (embedding/hidden width, e.g. 4096 for 7B)
  hidden_dim  FFN inner dimension
  n_layers    number of transformer layers
  n_heads     number of query attention heads
  n_kv_heads  number of key/value heads (< n_heads for grouped-query attention)
  vocab_size  vocabulary size
  seq_len     maximum sequence length (context window)
  head_size   derived: dim / n_heads; kv_dim = n_kv_heads * head_size
  GS          quantization group size (checkpoint header, e.g. 64):
              one float scale factor per GS int8 values

Command-line options (see main / error_usage):
  -t <float>  temperature in [0,inf], default 1.0 (0 = greedy argmax)
  -p <float>  top-p (nucleus) sampling in [0,1], default 0.9
  -s <int>    random seed, default time(NULL)
  -n <int>    number of generation steps, default 256; 0 = seq_len
  -i <string> input prompt
  -z <string> optional custom tokenizer path, default models/tokenizer.bin
  -m <string> mode: generate | chat, default generate
  -y <string> optional system prompt in chat mode

forward(int token, int pos):
  token  current input token id (index into vocab)
  pos    position in the sequence (0-based), used for RoPE and KV cache indexing
*/

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
    int n_kv_heads; // number of key/value heads
    int vocab_size; // vocabulary size
    int seq_len;    // max sequence length
};

int kv_dim(const Config& p) { return (p.dim * p.n_kv_heads) / p.n_heads; }

// Quantized tensor: int8 values + one float scale factor per group (GS elements).
// A non-owning view: for weights the spans point into the mmap region, for
// quantized activations in RunState they point into vectors. Sizes travel with
// the spans, so kernels derive n and GS instead of taking them as parameters.
struct QuantizedTensor {
    std::span<int8_t> q; // quantized values
    std::span<float> s;  // scaling factors
};

struct TransformerWeights {
    // token embedding: stored quantized, plus a float copy dequantized at load time
    std::vector<QuantizedTensor> q_tokens;       // (1,) of (vocab_size, dim)
    std::vector<float> token_embedding_table;    // (vocab_size, dim) dequantized

    // rmsnorm weights stay fp32 (small 1D vectors: quantizing saves little and costs precision)
    std::span<float> rms_att_weight; // (layer, dim)
    std::span<float> rms_ffn_weight; // (layer, dim)

    // all matmul weights are int8 quantized
    std::vector<QuantizedTensor> wq; // (layer,) of (dim, n_heads * head_size)
    std::vector<QuantizedTensor> wk; // (layer,) of (dim, n_kv_heads * head_size)
    std::vector<QuantizedTensor> wv; // (layer,) of (dim, n_kv_heads * head_size)
    std::vector<QuantizedTensor> wo; // (layer,) of (n_heads * head_size, dim)
    std::vector<QuantizedTensor> w1; // (layer,) of (hidden_dim, dim)
    std::vector<QuantizedTensor> w2; // (layer,) of (dim, hidden_dim)
    std::vector<QuantizedTensor> w3; // (layer,) of (hidden_dim, dim)

    std::span<float> rms_final_weight;           // (dim,)
    QuantizedTensor wcls;                        // a copy of q_tokens[0] (shared) or of wcls_storage[0];
                                                 // QuantizedTensor is a cheap view, so no pointer is needed
    std::vector<QuantizedTensor> wcls_storage;   // non-empty only when the classifier is not shared
};

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
    // quantized activation buffers: fp32 activations are packed to int8 here before matmul
    std::vector<int8_t> xq_q, hq_q;
    std::vector<float> xq_s, hq_s;
    QuantizedTensor xq; // quantized x (dim,)
    QuantizedTensor hq; // quantized hb (hidden_dim,)

    RunState(const Config& p, int GS)
        : x(p.dim), xb(p.dim), xb2(p.dim), hb(p.hidden_dim), hb2(p.hidden_dim), q(p.dim),
          att(static_cast<size_t>(p.n_heads) * p.seq_len), logits(p.vocab_size),
          key_cache(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim(p)),
          value_cache(static_cast<size_t>(p.n_layers) * p.seq_len * kv_dim(p)),
          xq_q(p.dim), hq_q(p.hidden_dim), xq_s(p.dim / GS), hq_s(p.hidden_dim / GS),
          xq{std::span{xq_q}, std::span{xq_s}}, hq{std::span{hq_q}, std::span{hq_s}} {}
};

// RAII wrapper for mmap of a quantized checkpoint: construction validates
// magic/version and reads config and group_size.
class MappedFile {
public:
    explicit MappedFile(const std::string& path) {
        {
            std::ifstream file(path, std::ios::binary);
            if (!file) { throw std::runtime_error("couldn't open checkpoint " + path); }
            auto read_or_die = [&](void* dst, std::streamsize size) {
                if (!file.read(static_cast<char*>(dst), size)) {
                    throw std::runtime_error("checkpoint header is truncated: " + path);
                }
            };
            // a quantized checkpoint has a 256-byte header: magic "ak42" + version 2 + Config + shared flag + group size
            std::uint32_t magic_number = 0;
            read_or_die(&magic_number, sizeof(magic_number));
            if (magic_number != 0x616b3432) {
                throw std::runtime_error("bad magic number: not a quantized checkpoint");
            }
            int version = 0;
            read_or_die(&version, sizeof(version));
            if (version != 2) {
                throw std::runtime_error("bad version " + std::to_string(version) + ", need version 2");
            }
            read_or_die(&config, sizeof(config));
            std::uint8_t shared = 0;
            read_or_die(&shared, sizeof(shared));
            shared_classifier = shared != 0;
            read_or_die(&group_size, sizeof(group_size));
        }
        file_size_ = std::filesystem::file_size(path);

        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ == -1) { throw std::runtime_error("open failed: " + path); }
        data_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
        if (data_ == MAP_FAILED) {
            data_ = nullptr;
            throw std::runtime_error("mmap failed: " + path);
        }
    }

    ~MappedFile() {
        if (data_ != nullptr) { munmap(data_, file_size_); }
        if (fd_ != -1) { ::close(fd_); }
    }

    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;

    Config config;
    bool shared_classifier = false;
    int group_size = 0;
    static constexpr int kHeaderSize = 256;

    // start of the weight region, past the 256-byte header
    void* weights_data() const { return static_cast<char*>(data_) + kHeaderSize; }

private:
    void* data_ = nullptr;
    size_t file_size_ = 0;
    int fd_ = -1;
};

// ----------------------------------------------------------------------------
// Quantization functions (the core of this file)

// dequantize: float = int8 * scale
void dequantize(const QuantizedTensor& qx, std::span<float> x) {
    const size_t GS = qx.q.size() / qx.s.size();
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = qx.q[i] * qx.s[i / GS];
    }
}

// quantize: groups of GS elements; scale = max absolute value in the group (symmetric quantization, range [-127, 127])
void quantize(QuantizedTensor& qx, std::span<const float> x) {
    const size_t GS = qx.q.size() / qx.s.size();
    constexpr float Q_MAX = 127.0f;
    for (size_t group = 0; group < qx.s.size(); group++) {
        const std::span x_group = x.subspan(group * GS, GS);
        float wmax = 0.0f;
        for (float v : x_group) { wmax = std::max(wmax, std::fabs(v)); }
        const float scale = wmax / Q_MAX;
        qx.s[group] = scale;
        for (size_t i = 0; i < GS; i++) {
            qx.q[group * GS + i] = static_cast<int8_t>(std::round(x_group[i] / scale));
        }
    }
}

// carve n quantized tensors of size_each elements out of the mmap region (layout: int8 values first, then float scale factors)
void init_quantized_tensors(void*& ptr, int n, int size_each, int GS, std::vector<QuantizedTensor>& out) {
    char* p = static_cast<char*>(ptr);
    out.resize(n);
    for (QuantizedTensor& t : out) {
        t.q = {reinterpret_cast<int8_t*>(p), static_cast<size_t>(size_each)};
        p += size_each;
        t.s = {reinterpret_cast<float*>(p), static_cast<size_t>(size_each / GS)};
        p += sizeof(float) * (size_each / GS);
    }
    ptr = p;
}

struct Transformer {
private:
    MappedFile mapped_; // first declared member: config/state/weights all depend on its mmap data

public:
    Config config;
    int GS; // quantization group size
    TransformerWeights weights;
    RunState state;

    explicit Transformer(const std::string& checkpoint_path)
        : mapped_(checkpoint_path), config(mapped_.config), GS(mapped_.group_size),
          state(config, GS) {
        map_weights(mapped_.weights_data());
    }

    std::span<float> forward(int token, int pos);

private:
    void map_weights(void* ptr) {
        const Config& p = config;
        const int head_size = p.dim / p.n_heads;
        const std::uint64_t n_layers = p.n_layers;
        // first the rmsnorm weights, which stay fp32
        float* fptr = static_cast<float*>(ptr);
        auto carve_fp32 = [&fptr](std::uint64_t count) {
            std::span<float> slice{fptr, static_cast<size_t>(count)};
            fptr += count;
            return slice;
        };
        weights.rms_att_weight = carve_fp32(n_layers * p.dim);
        weights.rms_ffn_weight = carve_fp32(n_layers * p.dim);
        weights.rms_final_weight = carve_fp32(p.dim);

        // then the quantized weights; the embedding is dequantized into an fp32 copy at load time (for lookup, avoiding per-step dequantize)
        ptr = fptr;
        init_quantized_tensors(ptr, 1, p.vocab_size * p.dim, GS, weights.q_tokens);
        weights.token_embedding_table.resize(static_cast<size_t>(p.vocab_size) * p.dim);
        dequantize(weights.q_tokens[0], weights.token_embedding_table);

        init_quantized_tensors(ptr, p.n_layers, p.dim * (p.n_heads * head_size), GS, weights.wq);
        init_quantized_tensors(ptr, p.n_layers, p.dim * (p.n_kv_heads * head_size), GS, weights.wk);
        init_quantized_tensors(ptr, p.n_layers, p.dim * (p.n_kv_heads * head_size), GS, weights.wv);
        init_quantized_tensors(ptr, p.n_layers, (p.n_heads * head_size) * p.dim, GS, weights.wo);

        init_quantized_tensors(ptr, p.n_layers, p.dim * p.hidden_dim, GS, weights.w1);
        init_quantized_tensors(ptr, p.n_layers, p.hidden_dim * p.dim, GS, weights.w2);
        init_quantized_tensors(ptr, p.n_layers, p.dim * p.hidden_dim, GS, weights.w3);

        if (mapped_.shared_classifier) {
            weights.wcls = weights.q_tokens[0];
        } else {
            init_quantized_tensors(ptr, 1, p.dim * p.vocab_size, GS, weights.wcls_storage);
            weights.wcls = weights.wcls_storage[0];
        }
    }
};

// ----------------------------------------------------------------------------
// neural net blocks

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

// quantized matmul: W (d,n) @ x (n,) -> xout (d,), both inputs int8.
// Compared with run.cpp's matmul, this is where the core gain and tricks of quantization are:
//   - the inner loop does int8 x int8 integer multiply-adds (int32 accumulator); one GS-element group cannot overflow
//     (worst case 127*127*64 ~ 10^6, far below the int32 limit)
//   - only at the end of each group is the result scaled back to float by x's and w's scale factors
//   - weight reads are 1/4 of the fp32 version (int8 vs float32); in a bandwidth-bound setting this is the speedup
void matmul(std::span<float> xout, const QuantizedTensor& x, const QuantizedTensor& w) {
    const int n = static_cast<int>(x.q.size());
    const int d = static_cast<int>(xout.size());
    const int GS = static_cast<int>(w.q.size() / w.s.size());
    #pragma omp parallel for
    for (int i = 0; i < d; i++) {
        float val = 0.0f;
        std::int32_t ival = 0;
        const int in = i * n;
        for (int j = 0; j <= n - GS; j += GS) {
            for (int k = 0; k < GS; k++) {
                ival += static_cast<std::int32_t>(x.q[j + k]) * static_cast<std::int32_t>(w.q[in + j + k]);
            }
            val += static_cast<float>(ival) * w.s[(in + j) / GS] * x.s[j / GS];
            ival = 0;
        }
        xout[i] = val;
    }
}

std::span<float> Transformer::forward(int token, int pos) {
    const Config& p = config;
    TransformerWeights& w = weights;
    RunState& s = state;
    const int dim = p.dim;
    const int kvd = kv_dim(p);
    const int kv_mul = p.n_heads / p.n_kv_heads;
    const int hidden_dim = p.hidden_dim;
    const int head_size = dim / p.n_heads;

    std::span<float> x{s.x};
    std::ranges::copy(std::span{w.token_embedding_table}.subspan(static_cast<size_t>(token) * dim, dim),
                      x.begin());

    for (int l = 0; l < p.n_layers; l++) {
        const size_t loff = static_cast<size_t>(l) * p.seq_len * kvd;

        // k/v are views into the KV cache at this layer and position: projections write
        // straight into the cache, and attention reads all history positions back from it
        std::span<float> k = std::span{s.key_cache}.subspan(loff + static_cast<size_t>(pos) * kvd, kvd);
        std::span<float> v = std::span{s.value_cache}.subspan(loff + static_cast<size_t>(pos) * kvd, kvd);

        // qkv projections: quantize the activation first, then int8 matmul
        rmsnorm(s.xb, x, w.rms_att_weight.subspan(static_cast<size_t>(l) * dim, dim));
        quantize(s.xq, s.xb);
        matmul(s.q, s.xq, w.wq[l]);
        matmul(k, s.xq, w.wk[l]);
        matmul(v, s.xq, w.wv[l]);

        // RoPE rotary position embedding (directly on the cached k row)
        for (int i = 0; i < dim; i += 2) {
            const int head_dim = i % head_size;
            const float freq = 1.0f / std::pow(10000.0f, head_dim / static_cast<float>(head_size));
            const float val = pos * freq;
            const float fcr = std::cos(val);
            const float fci = std::sin(val);
            const int rotn = i < kvd ? 2 : 1;
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
            for (int t = 0; t <= pos; t++) {
                std::span key = std::span{s.key_cache}.subspan(
                    loff + static_cast<size_t>(t) * kvd + (h / kv_mul) * head_size, head_size);
                float score = 0.0f;
                for (int i = 0; i < head_size; i++) { score += qh[i] * key[i]; }
                score /= std::sqrt(static_cast<float>(head_size));
                att[t] = score;
            }

            softmax(att);

            std::span xb = std::span{s.xb}.subspan(static_cast<size_t>(h) * head_size, head_size);
            std::ranges::fill(xb, 0.0f);
            for (int t = 0; t <= pos; t++) {
                std::span val = std::span{s.value_cache}.subspan(
                    loff + static_cast<size_t>(t) * kvd + (h / kv_mul) * head_size, head_size);
                const float a = att[t];
                for (int i = 0; i < head_size; i++) { xb[i] += a * val[i]; }
            }
        }

        // attention output projection + residual
        quantize(s.xq, s.xb);
        matmul(s.xb2, s.xq, w.wo[l]);
        for (int i = 0; i < dim; i++) { x[i] += s.xb2[i]; }

        rmsnorm(s.xb, x, w.rms_ffn_weight.subspan(static_cast<size_t>(l) * dim, dim));

        // FFN: w2(silu(w1(x)) * w3(x))
        quantize(s.xq, s.xb);
        matmul(s.hb, s.xq, w.w1[l]);
        matmul(s.hb2, s.xq, w.w3[l]);

        // SwiGLU
        for (int i = 0; i < hidden_dim; i++) {
            float val = s.hb[i];
            val *= (1.0f / (1.0f + std::exp(-val)));
            val *= s.hb2[i];
            s.hb[i] = val;
        }

        quantize(s.hq, s.hb);
        matmul(s.xb, s.hq, w.w2[l]);
        for (int i = 0; i < dim; i++) { x[i] += s.xb[i]; } // residual
    }

    rmsnorm(x, x, w.rms_final_weight);

    quantize(s.xq, x);
    matmul(s.logits, s.xq, w.wcls);
    return s.logits;
}

// ----------------------------------------------------------------------------
// BPE Tokenizer (same as run.cpp; see that file for detailed comments)

struct TokenIndex {
    std::string_view str;
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
            std::string s(len, '\0');
            read_or_die(s.data(), len);
            vocab_[i] = std::move(s);
        }
    }

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

    std::vector<int> encode(const std::string& text, bool bos, bool eos) {
        init_sorted_vocab();

        std::vector<int> tokens;
        if (bos) { tokens.push_back(1); }
        if (!text.empty()) { tokens.push_back(str_lookup(" ")); }

        std::string str_buffer;
        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            if ((c & 0xC0) != 0x80) { str_buffer.clear(); }
            str_buffer.push_back(c);
            if (i + 1 < text.size() && (text[i + 1] & 0xC0) == 0x80 && str_buffer.size() < 4) {
                continue;
            }
            int id = str_lookup(str_buffer);
            if (id != -1) {
                tokens.push_back(id);
            } else {
                for (unsigned char b : str_buffer) { tokens.push_back(b + 3); }
            }
            str_buffer.clear();
        }

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

        if (eos) { tokens.push_back(2); }
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

    int str_lookup(std::string_view str) const {
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

void safe_print(std::string_view piece) {
    if (piece.empty()) { return; }
    if (piece.size() == 1) {
        unsigned char byte_val = piece[0];
        if (!(std::isprint(byte_val) || std::isspace(byte_val))) { return; }
    }
    std::cout << piece;
}

// ----------------------------------------------------------------------------
// Sampler (same as run.cpp)

struct ProbIndex {
    float prob;
    int index;
};

class Sampler {
public:
    Sampler(int vocab_size, float temperature, float topp, std::uint64_t rng_seed)
        : temperature_(temperature), topp_(topp), rng_state_(rng_seed), probindex_(vocab_size) {}

    int sample(std::span<float> logits) {
        if (temperature_ == 0.0f) { return sample_argmax(logits); }
        for (float& v : logits) { v /= temperature_; }
        softmax(logits);
        const float coin = random_f32();
        if (topp_ <= 0 || topp_ >= 1) { return sample_mult(logits, coin); }
        return sample_topp(logits, topp_, coin);
    }

private:
    static int sample_argmax(std::span<const float> probabilities) {
        return static_cast<int>(std::ranges::max_element(probabilities) - probabilities.begin());
    }

    static int sample_mult(std::span<const float> probabilities, float coin) {
        float cdf = 0.0f;
        for (size_t i = 0; i < probabilities.size(); i++) {
            cdf += probabilities[i];
            if (coin < cdf) { return static_cast<int>(i); }
        }
        return static_cast<int>(probabilities.size()) - 1;
    }

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
// generation loop (same as run.cpp)

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
            next = prompt_tokens[pos + 1];
        } else {
            next = sampler.sample(logits);
        }
        pos++;

        if (next == 1) { break; }

        safe_print(tokenizer.decode(token, next));
        std::cout << std::flush;
        token = next;

        if (start == 0) { start = time_in_ms(); }
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
// chat loop (same as run.cpp)

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
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        if (token == 2) { user_turn = true; }

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
        "Usage:   runq <checkpoint> [options]\n"
        "Example: runq modelq.bin -n 256 -i \"Once upon a time\"\n"
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
        std::string checkpoint_path;
        std::string tokenizer_path = "models/tokenizer.bin";
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
