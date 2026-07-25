// 08_forward -- full single-step forward pass, FP32 (student template)
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
// Bring your own kernels from modules 03-07 (their bodies are TODO below);
// the new work of this module is the assembly in forward().
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main                (from this module folder)
// Verify: python3 ../tools/compare.py out_argmax.txt data/expected_argmax.txt --exact
//         python3 ../tools/compare.py out_logits.txt data/expected_logits.txt
// Compare argmax first: if it matches, the information flow is basically
// right. Logit diffs up to ~1e-3 are normal (summation order).

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
// Module 01: checkpoint loading (given -- same loader as your module 01)
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
// IO helpers (given)
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
// Modules 03 + 04: the small math kernels -- bring your own
// ---------------------------------------------------------------------------

// out_i = weight_i * x_i / sqrt(mean(x^2) + eps); all arithmetic in float.
void rmsnorm(std::span<float> o, std::span<const float> x, std::span<const float> weight) {
    // TODO(module 03): copy your rmsnorm here (eps = 1e-5, float arithmetic)
    (void)o; (void)x; (void)weight;
}

// In-place and numerically stable: subtract the max before exp.
void softmax(std::span<float> x) {
    // TODO(module 03): copy your softmax here (subtract the max first)
    (void)x;
}

// W (d,n) @ x (n,) -> xout (d,); w holds d rows of n elements, row-major.
void matmul(std::span<float> xout, std::span<const float> x, std::span<const float> w) {
    // TODO(module 04): copy your matmul here (float accumulator)
    (void)xout; (void)x; (void)w;
}

// ---------------------------------------------------------------------------
// Module 05: rotary position embedding, in place -- bring your own
// ---------------------------------------------------------------------------

// Rotate adjacent pairs of q (all dim pairs) and k (first kv_dim pairs)
// by pos * freq within each head.
void rope(std::span<float> q, std::span<float> k, int pos, int head_size) {
    // TODO(module 05): copy your rope here; freq depends on the pair's index
    // within its own head (i % head_size), and k is rotated only for i < k.size()
    (void)q; (void)k; (void)pos; (void)head_size;
}

// ---------------------------------------------------------------------------
// Module 06: multi-head causal attention for a single position -- bring your own
// ---------------------------------------------------------------------------

//   out:     this position's attention output (dim,), heads concatenated
//   att:     scratch (n_heads * seq_len,); head h uses [h*seq_len, h*seq_len+pos]
//   q:       rotated query of this position (dim,)
//   k_cache: this layer's K rows (seq_len, kv_dim); only rows 0..pos are read
//   v_cache: same layout for V
void attention(std::span<float> out, std::span<float> att, std::span<const float> q,
               std::span<const float> k_cache, std::span<const float> v_cache, int pos,
               int n_heads, int head_size, int kv_mul) {
    // TODO(module 06): copy your attention here -- per head: scores against
    // cache rows 0..pos scaled by 1/sqrt(head_size), softmax, weighted sum of
    // the V rows into the head's slice of out (kv head = h / kv_mul)
    (void)out; (void)att; (void)q; (void)k_cache; (void)v_cache;
    (void)pos; (void)n_heads; (void)head_size; (void)kv_mul;
}

// ---------------------------------------------------------------------------
// Module 07: SwiGLU feed-forward network -- bring your own
// ---------------------------------------------------------------------------

// out = W2 @ (silu(W1 @ x) * (W3 @ x)); hb/hb2 are (hidden,) scratch buffers.
// out may alias x: x is consumed into hb/hb2 before out is written.
void ffn(std::span<float> out, std::span<const float> x, std::span<const float> w1,
         std::span<const float> w2, std::span<const float> w3, std::span<float> hb,
         std::span<float> hb2) {
    // TODO(module 07): copy your ffn here -- two up-projections into hb/hb2,
    // hb = silu(hb) * hb2 elementwise (silu(v) = v / (1 + exp(-v))), then
    // the down-projection matmul(out, hb, w2)
    (void)out; (void)x; (void)w1; (void)w2; (void)w3; (void)hb; (void)hb2;
}

// ---------------------------------------------------------------------------
// Run state: activations + KV cache, sized from the config (given)
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
// (this is the new work of module 08 -- the kernels above are old friends)
// ---------------------------------------------------------------------------

std::span<const float> forward(int token, int pos, const Config& p, const Weights& w,
                               RunState& s) {
    const size_t dim = p.dim;
    const size_t kvd = static_cast<size_t>(p.n_kv_heads) * (p.dim / p.n_heads); // kv_dim
    const int kv_mul = p.n_heads / p.n_kv_heads; // kv sharing factor (1 here)
    const int head_size = p.dim / p.n_heads;
    (void)dim; (void)kvd; (void)kv_mul; (void)head_size; // remove once used
    (void)token; (void)pos; (void)w;                     // remove once used

    // TODO(task 1): embedding lookup -- copy row `token` of
    //   w.token_embedding_table (vocab_size x dim, row-major) into s.x.

    for (int64_t l = 0; l < p.n_layers; l++) {
        // TODO(task 2): one transformer layer, in this exact order (see the
        // README pipeline). Weight slices for layer l start at l * per-layer
        // size; cache slices for layer l at l * seq_len * kv_dim.
        //   a) spans: this layer's k/v cache (seq_len * kv_dim each, at
        //      offset l * seq_len * kv_dim) and this position's k/v rows
        //      inside it (kv_dim each, at offset pos * kv_dim)
        //   b) rmsnorm(s.xb, s.x, rms_att_weight[l]) -- slice at l * dim
        //   c) q/k/v projections with wq/wk/wv[l] (wq: l*dim*dim, dim*dim;
        //      wk/wv: l*dim*kv_dim, dim*kv_dim); write k and v straight into
        //      this position's cache rows -- cache first, then rope
        //   d) rope(s.q, k_row, pos, head_size) -- rotate q and the just-cached
        //      k row in place
        //   e) attention(s.xb, s.att, s.q, k_layer, v_layer, pos, n_heads,
        //      head_size, kv_mul)
        //   f) matmul(s.xb2, s.xb, wo[l]) then x[i] += s.xb2[i] -- residual
        //      add, not assignment
        //   g) rmsnorm(s.xb, s.x, rms_ffn_weight[l])
        //   h) ffn(s.xb, s.xb, w1[l], w2[l], w3[l], s.hb, s.hb2) then
        //      x[i] += s.xb[i] -- second residual add
        (void)l; // remove once used
    }

    // TODO(task 3): final rmsnorm, in place on s.x, with w.rms_final_weight.

    // TODO(task 4): classifier head -- matmul(s.logits, s.x, w.wcls)
    //   (wcls is the shared embedding table for this model).

    return s.logits; // stub: all zeros until the tasks above are done
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
