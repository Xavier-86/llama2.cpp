// 08_transformer_layer -- assemble ONE transformer layer, FP32 (reference solution)
//
// This module is the first half of what used to be a single "forward" module
// (the second half is 09_forward): it assembles the kernels from modules
// 03-07 into ONE transformer layer, using only the layer-0 weights:
//   x = embedding[token]                                        (01)
//   layer 0: rmsnorm -> qkv matmuls -> cache k/v -> rope -> attention
//            -> wo projection + residual -> rmsnorm -> ffn + residual
//            (03 / 04 / 05 / 06 / 07)
// The 5 prompt tokens of kTokens below are run as a prefill (positions
// 0..4); for each position the activation stream x is recorded twice:
// right after the attention residual add, and after the ffn residual add
// (the layer output). Both are written position-major (5 x dim).
//
// Checkpoint parsing and golden-data IO live in ../common/*.h; this file
// keeps only the algorithms.
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution            (from this module folder)
// Verify: python3 ../tools/compare.py out_att_residual.txt data/expected_att_residual.txt
//         python3 ../tools/compare.py out_layer_out.txt data/expected_layer_out.txt

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
// Run state: activations + KV cache, sized from the config.
// Trimmed versus 09_forward: no `logits` (this module stops at the layer
// output, no classifier head), and the caches hold ONE layer only, because
// just layer 0 ever runs here. `att` stays -- attention needs the scratch.
// ---------------------------------------------------------------------------

struct RunState {
    std::vector<float> x;           // (dim,) current activation stream
    std::vector<float> xb;          // (dim,) branch buffer
    std::vector<float> xb2;         // (dim,) second branch buffer
    std::vector<float> q;           // (dim,) query of the current position
    std::vector<float> hb;          // (hidden_dim,) ffn hidden, gate branch
    std::vector<float> hb2;         // (hidden_dim,) ffn hidden, linear branch
    std::vector<float> att;         // (n_heads * seq_len,) attention scores
    std::vector<float> key_cache;   // (seq_len, kv_dim) -- one layer only
    std::vector<float> value_cache; // (seq_len, kv_dim) -- one layer only

    explicit RunState(const tut::Config& c) {
        const size_t dim = c.dim;
        const size_t hidden = c.hidden_dim;
        const size_t kv_dim = static_cast<size_t>(c.n_kv_heads) * (c.dim / c.n_heads);
        x.assign(dim, 0.0f);
        xb.assign(dim, 0.0f);
        xb2.assign(dim, 0.0f);
        q.assign(dim, 0.0f);
        hb.assign(hidden, 0.0f);
        hb2.assign(hidden, 0.0f);
        att.assign(static_cast<size_t>(c.n_heads) * c.seq_len, 0.0f);
        key_cache.assign(static_cast<size_t>(c.seq_len) * kv_dim, 0.0f);
        value_cache.assign(static_cast<size_t>(c.seq_len) * kv_dim, 0.0f);
    }
};

// ---------------------------------------------------------------------------
// The assembly: embedding + ONE transformer layer (layer 0) for one token at
// one position. The two `std::ranges::copy` lines only record s.x for the
// golden-data files -- after the attention residual add (into att_residual)
// and after the ffn residual add (into layer_out).
// ---------------------------------------------------------------------------

void forward_layer0(int token, int pos, const tut::Config& p, const tut::Weights& w,
                    RunState& s, std::span<float> att_residual,
                    std::span<float> layer_out) {
    const size_t dim = p.dim;
    const size_t kvd = static_cast<size_t>(p.n_kv_heads) * (p.dim / p.n_heads); // kv_dim
    const int kv_mul = p.n_heads / p.n_kv_heads; // kv sharing factor (1 here)
    const int head_size = p.dim / p.n_heads;

    std::span<float> x{s.x};

    // embedding lookup: copy this token's row of the table into x
    std::ranges::copy(w.token_embedding_table.subspan(static_cast<size_t>(token) * dim, dim),
                      x.begin());

    // The single-layer cache, and the k/v rows of the current position inside
    // it -- the "cache": k/v of previous positions are read back, not
    // recomputed.
    std::span<float> k_cache{s.key_cache};
    std::span<float> v_cache{s.value_cache};
    std::span<float> k = k_cache.subspan(static_cast<size_t>(pos) * kvd, kvd);
    std::span<float> v = v_cache.subspan(static_cast<size_t>(pos) * kvd, kvd);

    // pre-norm, then qkv projections with the layer-0 weights (offset 0 in
    // each tensor); k/v are written straight into the cache
    rmsnorm(s.xb, x, w.rms_att_weight.subspan(0, dim));
    matmul(s.q, s.xb, w.wq.subspan(0, dim * dim));
    matmul(k, s.xb, w.wk.subspan(0, dim * kvd));
    matmul(v, s.xb, w.wv.subspan(0, dim * kvd));

    // RoPE: rotate q and the just-cached k row in place
    rope(s.q, k, pos, head_size);

    // multi-head attention over the cached rows 0..pos
    attention(s.xb, s.att, s.q, k_cache, v_cache, pos, p.n_heads, head_size, kv_mul);

    // attention output projection + residual connection
    matmul(s.xb2, s.xb, w.wo.subspan(0, dim * dim));
    for (size_t i = 0; i < dim; i++) { x[i] += s.xb2[i]; }

    // record the activation stream after the attention residual add
    std::ranges::copy(s.x, att_residual.begin());

    // pre-norm, then the SwiGLU ffn + residual connection
    rmsnorm(s.xb, x, w.rms_ffn_weight.subspan(0, dim));
    const size_t hidden = p.hidden_dim;
    ffn(s.xb, s.xb, w.w1.subspan(0, dim * hidden), w.w2.subspan(0, dim * hidden),
        w.w3.subspan(0, dim * hidden), s.hb, s.hb2);
    for (size_t i = 0; i < dim; i++) { x[i] += s.xb[i]; }

    // record the layer output (after the ffn residual add)
    std::ranges::copy(s.x, layer_out.begin());
}

} // namespace

int main() {
    try {
        // the checkpoint owns the weight buffer; all weight spans point into it
        const tut::Checkpoint ckpt = tut::load_checkpoint("../../stories15M.bin");
        const tut::Config& cfg = ckpt.config;
        const tut::Weights& w = ckpt.weights;
        const size_t dim = cfg.dim;
        RunState s(cfg);

        constexpr size_t num_tokens = sizeof(kTokens) / sizeof(kTokens[0]);
        std::vector<float> all_att_residual;
        all_att_residual.reserve(num_tokens * dim);
        std::vector<float> all_layer_out;
        all_layer_out.reserve(num_tokens * dim);
        std::vector<float> att_residual(dim);
        std::vector<float> layer_out(dim);

        // prefill: one forward_layer0 call per prompt token, positions 0..P-1
        for (size_t pos = 0; pos < num_tokens; pos++) {
            forward_layer0(kTokens[pos], static_cast<int>(pos), cfg, w, s, att_residual,
                           layer_out);
            all_att_residual.insert(all_att_residual.end(), att_residual.begin(),
                                    att_residual.end());
            all_layer_out.insert(all_layer_out.end(), layer_out.begin(), layer_out.end());
        }

        tut::write_floats("out_att_residual.txt", all_att_residual); // position-major: P x dim
        tut::write_floats("out_layer_out.txt", all_layer_out);       // position-major: P x dim

        std::cout << "wrote out_att_residual.txt and out_layer_out.txt (" << num_tokens
                  << " x " << dim << " each)\n";
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
