// 08_transformer_layer -- assemble ONE transformer layer, FP32 (student template)
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
// Checkpoint parsing and golden-data IO live in ../common/*.h (given); this
// file keeps only the algorithms. Bring your own kernels from modules 03-07
// (their bodies are TODO below); the new work of this module is the
// embedding lookup and the single-layer assembly in forward_layer0().
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main                (from this module folder)
// Verify: python3 ../tools/compare.py out_att_residual.txt data/expected_att_residual.txt
//         python3 ../tools/compare.py out_layer_out.txt data/expected_layer_out.txt

#include <algorithm>
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
// Run state: activations + KV cache, sized from the config (given).
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
// one position. The two `std::ranges::copy` lines are given: they only record
// s.x for the golden-data files -- after the attention residual add (into
// att_residual) and after the ffn residual add (into layer_out).
// ---------------------------------------------------------------------------

void forward_layer0(int token, int pos, const tut::Config& p, const tut::Weights& w,
                    RunState& s, std::span<float> att_residual,
                    std::span<float> layer_out) {
    const size_t dim = p.dim;
    const size_t kvd = static_cast<size_t>(p.n_kv_heads) * (p.dim / p.n_heads); // kv_dim
    const int kv_mul = p.n_heads / p.n_kv_heads; // kv sharing factor (1 here)
    const int head_size = p.dim / p.n_heads;
    (void)dim; (void)kvd; (void)kv_mul; (void)head_size; // remove once used
    (void)token; (void)pos; (void)w;                     // remove once used

    // TODO(task 1): embedding lookup -- copy row `token` of
    //   w.token_embedding_table (vocab_size x dim, row-major) into s.x.

    // TODO(task 2): ONE transformer layer (layer 0 only), in this exact order
    //   (see the README pipeline). Only layer 0 runs, so every weight slice
    //   starts at offset 0 of its tensor, and the cache spans cover the
    //   whole (single-layer) key_cache/value_cache.
    //   a) spans: the k/v cache (seq_len * kv_dim each -- the whole cache)
    //      and this position's k/v rows inside it (kv_dim each, at offset
    //      pos * kv_dim)
    //   b) rmsnorm(s.xb, s.x, rms_att_weight[0]) -- the first dim floats
    //   c) q/k/v projections with wq/wk/wv[0] (wq: dim*dim floats; wk/wv:
    //      dim*kv_dim); write k and v straight into this position's cache
    //      rows -- cache first, then rope
    //   d) rope(s.q, k_row, pos, head_size) -- rotate q and the just-cached
    //      k row in place
    //   e) attention(s.xb, s.att, s.q, k_cache, v_cache, pos, n_heads,
    //      head_size, kv_mul)
    //   f) matmul(s.xb2, s.xb, wo[0]) then x[i] += s.xb2[i] -- residual
    //      add, not assignment
    //   (the given copy below records s.x at this point: att_residual)
    //   g) rmsnorm(s.xb, s.x, rms_ffn_weight[0])
    //   h) ffn(s.xb, s.xb, w1[0], w2[0], w3[0], s.hb, s.hb2) then
    //      x[i] += s.xb[i] -- second residual add
    //   (the given copy below records s.x again: the layer output)

    // given: record the activation stream after the attention residual add
    std::ranges::copy(s.x, att_residual.begin());

    // (task 2 continues here: steps g) and h))

    // given: record the layer output (after the ffn residual add)
    std::ranges::copy(s.x, layer_out.begin());
}

} // namespace

int main() {
    try {
        // the checkpoint owns the weight buffer; all weight spans point into it
        const tut::Checkpoint ckpt = tut::load_checkpoint("../../models/stories15M.bin");
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
