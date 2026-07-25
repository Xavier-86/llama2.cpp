// 08_forward -- full single-step forward pass, FP32 (reference solution)
//
// Assembles modules 01-07 into forward(token, pos) -> logits:
//   x = embedding[token]                                        (01)
//   per layer: rmsnorm -> qkv matmuls -> cache k/v -> rope -> attention
//              -> wo projection + residual -> rmsnorm -> ffn + residual
//              (03 / 04 / 05 / 06 / 07)
//   final rmsnorm -> classifier matmul (shared embedding table)
// The 5 prompt tokens from data/input_tokens.txt are run as a prefill
// (positions 0..4); each position's logits (32000 values) and argmax are
// collected, logits written position-major.
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution            (from this module folder)
// Verify: python3 ../tools/compare.py out_argmax.txt data/expected_argmax.txt --exact
//         python3 ../tools/compare.py out_logits.txt data/expected_logits.txt

#include <algorithm>
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

// ---------------------------------------------------------------------------
// Module 01: checkpoint loading
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

// each tensor is a view into the weight region -- no copies
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

// Load the whole checkpoint and carve the 11 tensors in file order.
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
    ptr += static_cast<size_t>(cfg.seq_len) * head_size / 2;
    ptr += static_cast<size_t>(cfg.seq_len) * head_size / 2;
    w.wcls = shared ? w.token_embedding_table : carve(vocab * dim);
    return buf; // keeps the weight region alive; the spans point into it
}

// ---------------------------------------------------------------------------
// IO helpers
// ---------------------------------------------------------------------------

// Load whitespace-separated integers, one per line.
std::vector<int> load_ints(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    std::vector<int> v;
    int x;
    while (in >> x) { v.push_back(x); }
    return v;
}

// Write floats one per line, matching the golden data format (%.3e).
void write_floats(const std::string& path, std::span<const float> v) {
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("cannot write " + path); }
    out << std::scientific << std::setprecision(3);
    for (float value : v) { out << value << '\n'; }
}

// Write integers one per line.
void write_ints(const std::string& path, std::span<const int> v) {
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("cannot write " + path); }
    for (int value : v) { out << value << '\n'; }
}

// ---------------------------------------------------------------------------
// Modules 03 + 04: the small math kernels
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
// Accumulate in float, matching the reference.
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
// Module 05: rotary position embedding, in place
// ---------------------------------------------------------------------------

// Rotate adjacent pairs of q (all dim pairs) and k (first kv_dim pairs)
// by pos * freq within each head.
void rope(std::span<float> q, std::span<float> k, int pos, int head_size) {
    for (size_t i = 0; i < q.size(); i += 2) {
        const int head_dim = static_cast<int>(i) % head_size; // pair index within its head
        const float freq =
            1.0f / std::pow(10000.0f, head_dim / static_cast<float>(head_size));
        const float angle = pos * freq;
        const float fcr = std::cos(angle);
        const float fci = std::sin(angle);
        {
            const float v0 = q[i];
            const float v1 = q[i + 1];
            q[i]     = v0 * fcr - v1 * fci;
            q[i + 1] = v0 * fci + v1 * fcr;
        }
        if (i < k.size()) { // kv_dim == dim here, so k is fully rotated too
            const float v0 = k[i];
            const float v1 = k[i + 1];
            k[i]     = v0 * fcr - v1 * fci;
            k[i + 1] = v0 * fci + v1 * fcr;
        }
    }
}

// ---------------------------------------------------------------------------
// Module 06: multi-head causal attention for a single position
// ---------------------------------------------------------------------------

//   out:     this position's attention output (dim,), heads concatenated
//   att:     scratch (n_heads * seq_len,); head h uses [h*seq_len, h*seq_len+pos]
//   q:       rotated query of this position (dim,)
//   k_cache: this layer's K rows (seq_len, kv_dim); only rows 0..pos are read
//   v_cache: same layout for V
void attention(std::span<float> out, std::span<float> att, std::span<const float> q,
               std::span<const float> k_cache, std::span<const float> v_cache, int pos,
               int n_heads, int head_size, int kv_mul) {
    const size_t kv_dim = k_cache.size() / (att.size() / n_heads); // n_kv_heads * head_size
    const size_t seq_len = att.size() / n_heads;
    for (int h = 0; h < n_heads; h++) {
        const int kv_h = h / kv_mul; // GQA sharing; kv_mul == 1 here, so kv_h == h
        std::span qh = q.subspan(static_cast<size_t>(h) * head_size, head_size);
        std::span att_h =
            att.subspan(static_cast<size_t>(h) * seq_len, static_cast<size_t>(pos) + 1);

        // Score against history + current only: t in [0, pos] is the causal mask.
        for (int t = 0; t <= pos; t++) {
            std::span key = k_cache.subspan(
                static_cast<size_t>(t) * kv_dim + static_cast<size_t>(kv_h) * head_size,
                head_size);
            float score = 0.0f;
            for (int i = 0; i < head_size; i++) { score += qh[i] * key[i]; }
            att_h[t] = score / std::sqrt(static_cast<float>(head_size));
        }

        softmax(att_h);

        // Weighted sum of the V rows, written into this head's slice of out.
        std::span out_h = out.subspan(static_cast<size_t>(h) * head_size, head_size);
        std::ranges::fill(out_h, 0.0f);
        for (int t = 0; t <= pos; t++) {
            std::span val = v_cache.subspan(
                static_cast<size_t>(t) * kv_dim + static_cast<size_t>(kv_h) * head_size,
                head_size);
            const float a = att_h[t];
            for (int i = 0; i < head_size; i++) { out_h[i] += a * val[i]; }
        }
    }
}

// ---------------------------------------------------------------------------
// Module 07: SwiGLU feed-forward network
// ---------------------------------------------------------------------------

// out = W2 @ (silu(W1 @ x) * (W3 @ x)); hb/hb2 are (hidden,) scratch buffers.
// out may alias x: x is consumed into hb/hb2 before out is written.
void ffn(std::span<float> out, std::span<const float> x, std::span<const float> w1,
         std::span<const float> w2, std::span<const float> w3, std::span<float> hb,
         std::span<float> hb2) {
    matmul(hb, x, w1);
    matmul(hb2, x, w3);
    // SwiGLU gate: h = silu(h1) * h3, silu(v) = v * sigmoid(v) = v / (1 + exp(-v))
    for (size_t i = 0; i < hb.size(); i++) {
        float v = hb[i];
        v *= 1.0f / (1.0f + std::exp(-v));
        v *= hb2[i];
        hb[i] = v;
    }
    matmul(out, hb, w2);
}

// ---------------------------------------------------------------------------
// Run state: activations + KV cache, sized from the config
// ---------------------------------------------------------------------------

struct RunState {
    std::vector<float> x;           // (dim,) current activation stream
    std::vector<float> xb;          // (dim,) branch buffer
    std::vector<float> xb2;         // (dim,) second branch buffer
    std::vector<float> q;           // (dim,) query of the current position
    std::vector<float> hb;          // (hidden_dim,) ffn hidden, gate branch
    std::vector<float> hb2;         // (hidden_dim,) ffn hidden, linear branch
    std::vector<float> att;         // (n_heads * seq_len,) attention scores
    std::vector<float> key_cache;   // (n_layers, seq_len, kv_dim)
    std::vector<float> value_cache; // (n_layers, seq_len, kv_dim)
    std::vector<float> logits;      // (vocab_size,)

    explicit RunState(const Config& c) {
        const size_t dim = c.dim;
        const size_t hidden = c.hidden_dim;
        const size_t kv_dim = static_cast<size_t>(c.n_kv_heads) * (c.dim / c.n_heads);
        const size_t vocab = c.vocab_size < 0 ? -c.vocab_size : c.vocab_size;
        x.assign(dim, 0.0f);
        xb.assign(dim, 0.0f);
        xb2.assign(dim, 0.0f);
        q.assign(dim, 0.0f);
        hb.assign(hidden, 0.0f);
        hb2.assign(hidden, 0.0f);
        att.assign(static_cast<size_t>(c.n_heads) * c.seq_len, 0.0f);
        key_cache.assign(static_cast<size_t>(c.n_layers) * c.seq_len * kv_dim, 0.0f);
        value_cache.assign(static_cast<size_t>(c.n_layers) * c.seq_len * kv_dim, 0.0f);
        logits.assign(vocab, 0.0f);
    }
};

// ---------------------------------------------------------------------------
// The assembly: one token at one position -> logits for the next token
// ---------------------------------------------------------------------------

std::span<const float> forward(int token, int pos, const Config& p, const Weights& w,
                               RunState& s) {
    const size_t dim = p.dim;
    const size_t kvd = static_cast<size_t>(p.n_kv_heads) * (p.dim / p.n_heads);
    const int kv_mul = p.n_heads / p.n_kv_heads; // kv sharing factor (1 here)
    const int head_size = p.dim / p.n_heads;

    std::span<float> x{s.x};

    // embedding lookup: copy this token's row of the table into x
    std::ranges::copy(w.token_embedding_table.subspan(static_cast<size_t>(token) * dim, dim),
                      x.begin());

    for (int64_t l = 0; l < p.n_layers; l++) {
        const size_t layer = static_cast<size_t>(l);
        const size_t loff = layer * static_cast<size_t>(p.seq_len) * kvd;

        // This layer's cache, and the k/v rows of the current position inside
        // it -- the "cache": k/v of previous positions are read back, not
        // recomputed.
        std::span<float> k_layer{s.key_cache.data() + loff,
                                 static_cast<size_t>(p.seq_len) * kvd};
        std::span<float> v_layer{s.value_cache.data() + loff,
                                 static_cast<size_t>(p.seq_len) * kvd};
        std::span<float> k = k_layer.subspan(static_cast<size_t>(pos) * kvd, kvd);
        std::span<float> v = v_layer.subspan(static_cast<size_t>(pos) * kvd, kvd);

        // pre-norm, then qkv projections; k/v are written straight into the cache
        rmsnorm(s.xb, x, w.rms_att_weight.subspan(layer * dim, dim));
        matmul(s.q, s.xb, w.wq.subspan(layer * dim * dim, dim * dim));
        matmul(k, s.xb, w.wk.subspan(layer * dim * kvd, dim * kvd));
        matmul(v, s.xb, w.wv.subspan(layer * dim * kvd, dim * kvd));

        // RoPE: rotate q and the just-cached k row in place
        rope(s.q, k, pos, head_size);

        // multi-head attention over the cached rows 0..pos
        attention(s.xb, s.att, s.q, k_layer, v_layer, pos, p.n_heads, head_size, kv_mul);

        // attention output projection + residual connection
        matmul(s.xb2, s.xb, w.wo.subspan(layer * dim * dim, dim * dim));
        for (size_t i = 0; i < dim; i++) { x[i] += s.xb2[i]; }

        // pre-norm, then the SwiGLU ffn + residual connection
        rmsnorm(s.xb, x, w.rms_ffn_weight.subspan(layer * dim, dim));
        const size_t hidden = p.hidden_dim;
        ffn(s.xb, s.xb, w.w1.subspan(layer * dim * hidden, dim * hidden),
            w.w2.subspan(layer * dim * hidden, dim * hidden),
            w.w3.subspan(layer * dim * hidden, dim * hidden), s.hb, s.hb2);
        for (size_t i = 0; i < dim; i++) { x[i] += s.xb[i]; }
    }

    // final rmsnorm (in place) + classifier head (shared embedding table)
    rmsnorm(x, x, w.rms_final_weight);
    matmul(s.logits, x, w.wcls);
    return s.logits;
}

} // namespace

int main() {
    try {
        const std::vector<int> tokens = load_ints("data/input_tokens.txt");
        if (tokens.empty()) { throw std::runtime_error("no input tokens"); }

        Config cfg;
        Weights w;
        // the buffer must stay alive: all weight spans point into it
        const std::vector<float> buf = load_checkpoint("../../stories15M.bin", cfg, w);
        const size_t vocab = cfg.vocab_size < 0 ? -cfg.vocab_size : cfg.vocab_size;
        RunState s(cfg);

        std::vector<float> all_logits;
        all_logits.reserve(tokens.size() * vocab);
        std::vector<int> argmax;
        argmax.reserve(tokens.size());

        // prefill: one forward call per prompt token, positions 0..P-1
        for (size_t pos = 0; pos < tokens.size(); pos++) {
            std::span<const float> logits =
                forward(tokens[pos], static_cast<int>(pos), cfg, w, s);
            all_logits.insert(all_logits.end(), logits.begin(), logits.end());
            // greedy argmax (module 09's temperature == 0 case)
            argmax.push_back(
                static_cast<int>(std::ranges::max_element(logits) - logits.begin()));
        }

        write_floats("out_logits.txt", all_logits); // position-major: P x vocab
        write_ints("out_argmax.txt", argmax);

        std::cout << "wrote out_logits.txt (" << tokens.size() << " x " << vocab
                  << ") and out_argmax.txt (" << argmax.size() << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
