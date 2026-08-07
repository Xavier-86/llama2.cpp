// 09_forward -- full single-step forward pass, FP32 (student template)
//
// Assembles the ONE-layer block of module 08_transformer_layer into the
// complete forward(token, pos) -> logits:
//   x = embedding[token]                                        (given)
//   for l in 0..n_layers-1: the transformer layer of module 08  (task 1)
//   final rmsnorm -> classifier matmul (shared embedding table) (task 2, 3)
// The 5 prompt tokens of kTokens below are run as a prefill
// (positions 0..4); each position's logits (32000 values) and argmax are
// collected, logits written position-major.
//
// Checkpoint parsing and golden-data IO live in ../common/*.h (given); this
// file keeps only the algorithms. The six kernels (rmsnorm / softmax /
// matmul / rope / attention / ffn) and the single-layer assembly are GIVEN:
// you already implemented them in modules 03-07 and 08_transformer_layer.
// The new work of this module is stacking the layer 6 times and adding the
// output head.
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
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "../common/checkpoint.h"
#include "../common/io.h"

namespace {

// ---------------------------------------------------------------------------
// Prompt input: "Once upon a time" tokenized, with the <s> start token.
// (Same values as data/input_tokens.txt.)
// ---------------------------------------------------------------------------

const int kTokens[] = {1, 9038, 2501, 263, 931}; // P = 5 prompt tokens

// ---------------------------------------------------------------------------
// Modules 03 + 04: the small math kernels
// (given: you implemented these in modules 03 and 04)
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
// (given: you implemented this in module 05)
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
// (given: you implemented this in module 06)
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
// (given: you implemented this in module 07)
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

    explicit RunState(const tut::Config& c) {
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
// ONE transformer layer at layer index l, for the token already embedded in
// s.x at position pos (given: this is exactly what you assembled in module
// 08_transformer_layer, now parameterized by l -- note the weight and cache
// slice offsets, all proportional to l)
// ---------------------------------------------------------------------------

void transformer_layer(int64_t l, int pos, const tut::Config& p, const tut::Weights& w,
                       RunState& s) {
    const size_t dim = p.dim;
    const size_t kvd = static_cast<size_t>(p.n_kv_heads) * (p.dim / p.n_heads); // kv_dim
    const int kv_mul = p.n_heads / p.n_kv_heads; // kv sharing factor (1 here)
    const int head_size = p.dim / p.n_heads;
    const size_t layer = static_cast<size_t>(l);
    const size_t loff = layer * static_cast<size_t>(p.seq_len) * kvd;

    std::span<float> x{s.x};

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

// ---------------------------------------------------------------------------
// The full assembly: one token at one position -> logits for the next token
// (this is the new work of module 09 -- everything above is given)
// ---------------------------------------------------------------------------

std::span<const float> forward(int token, int pos, const tut::Config& p,
                               const tut::Weights& w, RunState& s) {
    const size_t dim = p.dim;
    (void)dim; // remove once used

    // given: embedding lookup -- copy this token's row of the table into s.x
    std::ranges::copy(w.token_embedding_table.subspan(static_cast<size_t>(token) * dim, dim),
                      s.x.begin());

    // TODO(task 1): stack the layers -- call transformer_layer(l, pos, p, w, s)
    //   for every l in 0 .. p.n_layers-1, in order. (transformer_layer is
    //   given; it already slices layer l's weights at l * per-layer size and
    //   layer l's cache at l * seq_len * kv_dim -- your job is only the loop.)

    // TODO(task 2): final rmsnorm, in place on s.x, with w.rms_final_weight.

    // TODO(task 3): classifier head -- matmul(s.logits, s.x, w.wcls)
    //   (wcls is the shared embedding table for this model).

    return s.logits; // stub: all zeros until the tasks above are done
}

} // namespace

int main() {
    try {
        // the checkpoint owns the weight buffer; all weight spans point into it
        const tut::Checkpoint ckpt = tut::load_checkpoint("../../models/stories15M.bin");
        const tut::Config& cfg = ckpt.config;
        const tut::Weights& w = ckpt.weights;
        const size_t vocab = cfg.vocab_size < 0 ? -cfg.vocab_size : cfg.vocab_size;
        RunState s(cfg);

        constexpr size_t num_tokens = sizeof(kTokens) / sizeof(kTokens[0]);
        std::vector<float> all_logits;
        all_logits.reserve(num_tokens * vocab);
        std::vector<int> argmax;
        argmax.reserve(num_tokens);

        // prefill: one forward call per prompt token, positions 0..P-1
        for (size_t pos = 0; pos < num_tokens; pos++) {
            std::span<const float> logits =
                forward(kTokens[pos], static_cast<int>(pos), cfg, w, s);
            all_logits.insert(all_logits.end(), logits.begin(), logits.end());
            // greedy argmax (module 10's temperature == 0 case)
            argmax.push_back(
                static_cast<int>(std::ranges::max_element(logits) - logits.begin()));
        }

        tut::write_floats("out_logits.txt", all_logits); // position-major: P x vocab
        tut::write_ints("out_argmax.txt", argmax);

        std::cout << "wrote out_logits.txt (" << num_tokens << " x " << vocab
                  << ") and out_argmax.txt (" << argmax.size() << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
