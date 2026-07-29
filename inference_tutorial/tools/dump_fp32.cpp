// Golden-data generator for inference_tutorial (FP32 half).
// Includes the reference implementation wholesale (main renamed away), then:
//   1. dumps checkpoint/tokenizer/kernel-level test vectors
//   2. runs an instrumented copy of Transformer::forward (same code + capture hooks)
//      to dump per-position intermediates at layer 0 (rope / attention / ffn)
//   3. verifies the instrumented forward is bit-identical to the original
//   4. dumps full-forward logits, sampler cases, and complete generations
//
// usage: dump_fp32 <out_root> <checkpoint> <tokenizer>
// e.g.:  ./dump_fp32 inference_tutorial stories15M.bin tokenizer.bin

#define main llama2_reference_cli_main
#include "../../run.cpp"
#undef main

#include "dump_common.h"
#include <iostream>

namespace {

// capture buffers for the instrumented forward (layer 0 only)
struct Captures {
    int total_pos = 0; // number of prompt positions P
    // per-position (pos-major): q/k before rope, q/k after rope, v, attention output
    std::vector<float> q_pre, k_pre, q_post, k_post, v_all, att_out;
    // per-position (pos-major): x after the attention residual, x after the ffn
    // residual (= the full layer-0 output stream)
    std::vector<float> att_res, layer_out;
    // last position only: attention weights of all heads, ffn in/hidden/out
    std::vector<float> att_weights, ffn_in, ffn_hidden, ffn_out;
};

// verbatim copy of Transformer::forward from run.cpp, plus capture hooks at layer 0
std::span<float> forward_dump(Transformer& tr, int token, int pos, Captures& cap) {
    const Config& p = tr.config;
    TransformerWeights& w = tr.weights;
    RunState& s = tr.state;
    const int dim = p.dim;
    const int kvd = kv_dim(p);
    const int kv_mul = p.n_heads / p.n_kv_heads;
    const int hidden_dim = p.hidden_dim;
    const int head_size = dim / p.n_heads;

    std::span<float> x{s.x};
    std::ranges::copy(w.token_embedding_table.subspan(static_cast<size_t>(token) * dim, dim),
                      x.begin());

    for (int l = 0; l < p.n_layers; l++) {
        const size_t loff = static_cast<size_t>(l) * p.seq_len * kvd;
        std::span<float> k = std::span{s.key_cache}.subspan(loff + static_cast<size_t>(pos) * kvd, kvd);
        std::span<float> v = std::span{s.value_cache}.subspan(loff + static_cast<size_t>(pos) * kvd, kvd);

        rmsnorm(s.xb, x, w.rms_att_weight.subspan(static_cast<size_t>(l) * dim, dim));
        matmul(s.q, s.xb, w.wq.subspan(static_cast<size_t>(l) * dim * dim, static_cast<size_t>(dim) * dim));
        matmul(k, s.xb, w.wk.subspan(static_cast<size_t>(l) * dim * kvd, static_cast<size_t>(dim) * kvd));
        matmul(v, s.xb, w.wv.subspan(static_cast<size_t>(l) * dim * kvd, static_cast<size_t>(dim) * kvd));

        if (l == 0) { // [capture] q/k/v right after projection, before RoPE
            append(cap.q_pre, s.q.data(), dim);
            append(cap.k_pre, k.data(), kvd);
            append(cap.v_all, v.data(), kvd);
        }

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

        if (l == 0) { // [capture] q/k after RoPE (k is now in its cached form)
            append(cap.q_post, s.q.data(), dim);
            append(cap.k_post, k.data(), kvd);
        }

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

            if (l == 0 && pos == cap.total_pos - 1) { // [capture] attention weights, last pos
                append(cap.att_weights, att.data(), static_cast<size_t>(pos) + 1);
            }

            std::span xb = std::span{s.xb}.subspan(static_cast<size_t>(h) * head_size, head_size);
            std::ranges::fill(xb, 0.0f);
            for (int t = 0; t <= pos; t++) {
                std::span val = std::span{s.value_cache}.subspan(
                    loff + static_cast<size_t>(t) * kvd + (h / kv_mul) * head_size, head_size);
                const float a = att[t];
                for (int i = 0; i < head_size; i++) { xb[i] += a * val[i]; }
            }
        }

        if (l == 0) { append(cap.att_out, s.xb.data(), dim); } // [capture] attention output

        matmul(s.xb2, s.xb, w.wo.subspan(static_cast<size_t>(l) * dim * dim, static_cast<size_t>(dim) * dim));
        for (int i = 0; i < dim; i++) { x[i] += s.xb2[i]; }

        if (l == 0) { append(cap.att_res, x.data(), dim); } // [capture] x after attention residual

        rmsnorm(s.xb, x, w.rms_ffn_weight.subspan(static_cast<size_t>(l) * dim, dim));

        if (l == 0 && pos == cap.total_pos - 1) { // [capture] ffn input
            append(cap.ffn_in, s.xb.data(), dim);
        }

        matmul(s.hb, s.xb, w.w1.subspan(static_cast<size_t>(l) * dim * hidden_dim,
                                        static_cast<size_t>(dim) * hidden_dim));
        matmul(s.hb2, s.xb, w.w3.subspan(static_cast<size_t>(l) * dim * hidden_dim,
                                         static_cast<size_t>(dim) * hidden_dim));

        for (int i = 0; i < hidden_dim; i++) {
            float val = s.hb[i];
            val *= (1.0f / (1.0f + std::exp(-val)));
            val *= s.hb2[i];
            s.hb[i] = val;
        }

        if (l == 0 && pos == cap.total_pos - 1) { // [capture] ffn hidden after SwiGLU
            append(cap.ffn_hidden, s.hb.data(), hidden_dim);
        }

        matmul(s.xb, s.hb, w.w2.subspan(static_cast<size_t>(l) * hidden_dim * dim,
                                        static_cast<size_t>(hidden_dim) * dim));

        if (l == 0 && pos == cap.total_pos - 1) { // [capture] ffn output (before residual)
            append(cap.ffn_out, s.xb.data(), dim);
        }

        for (int i = 0; i < dim; i++) { x[i] += s.xb[i]; }

        if (l == 0) { append(cap.layer_out, x.data(), dim); } // [capture] layer-0 output stream
    }

    rmsnorm(x, x, w.rms_final_weight);
    matmul(s.logits, x, w.wcls);
    return s.logits;
}

struct NamedWeight {
    const char* name;
    std::span<float> data;
};

} // namespace

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: dump_fp32 <out_root> <checkpoint> <tokenizer>\n";
        return 1;
    }
    const std::string root = argv[1];
    const std::string checkpoint = argv[2];
    const std::string tokenizer_path = argv[3];

    Transformer tr(checkpoint);
    const Config& p = tr.config;
    const int dim = p.dim;
    const int kvd = kv_dim(p);
    const int head_size = dim / p.n_heads;

    std::cout << "config: dim=" << dim << " hidden_dim=" << p.hidden_dim
              << " n_layers=" << p.n_layers << " n_heads=" << p.n_heads
              << " n_kv_heads=" << p.n_kv_heads << " vocab_size=" << p.vocab_size
              << " seq_len=" << p.seq_len << " head_size=" << head_size
              << " shared_weights="
              << (tr.weights.wcls.data() == tr.weights.token_embedding_table.data()) << '\n';

    // ---------------- 01_checkpoint ----------------
    {
        const std::string dir = root + "/01_checkpoint/data/";
        write_ints(dir + "expected_config.txt",
                   std::array{p.dim, p.hidden_dim, p.n_layers, p.n_heads,
                              p.n_kv_heads, p.vocab_size, p.seq_len});
        const NamedWeight order[] = {
            {"token_embedding_table", tr.weights.token_embedding_table},
            {"rms_att_weight", tr.weights.rms_att_weight},
            {"wq", tr.weights.wq},
            {"wk", tr.weights.wk},
            {"wv", tr.weights.wv},
            {"wo", tr.weights.wo},
            {"rms_ffn_weight", tr.weights.rms_ffn_weight},
            {"w1", tr.weights.w1},
            {"w2", tr.weights.w2},
            {"w3", tr.weights.w3},
            {"rms_final_weight", tr.weights.rms_final_weight},
        };
        std::vector<double> summary; // per tensor: size, first element, sum
        for (const auto& [name, data] : order) {
            double sum = 0.0;
            for (float v : data) { sum += v; }
            summary.push_back(static_cast<double>(data.size()));
            summary.push_back(data[0]);
            summary.push_back(sum);
            std::cout << "weight " << name << " size=" << data.size() << '\n';
        }
        write_doubles(dir + "expected_weight_summary.txt", summary);
    }

    // ---------------- 02_tokenizer ----------------
    Tokenizer tokenizer(tokenizer_path, p.vocab_size);
    const std::vector<std::string> prompts = {
        "Once upon a time",
        "One day, a little girl named Lily",
        "Hello, world!",
        "The capital of France is",
    };
    {
        const std::string dir = root + "/02_tokenizer/data/";
        std::string joined;
        for (size_t i = 0; i < prompts.size(); i++) { joined += prompts[i] + '\n'; }
        write_text(dir + "input_prompts.txt", joined);
        for (size_t i = 0; i < prompts.size(); i++) {
            auto ids = tokenizer.encode(prompts[i], true, false);
            write_ints(dir + "expected_encode_" + std::to_string(i) + ".txt", ids);
        }
        auto ids0 = tokenizer.encode(prompts[0], true, false);
        write_ints(dir + "input_decode_ids.txt", ids0);
        std::string text;
        for (size_t i = 0; i + 1 < ids0.size(); i++) {
            text += std::string(tokenizer.decode(ids0[i], ids0[i + 1]));
        }
        write_text(dir + "expected_decode.txt", text);
    }

    const std::string prompt = prompts[0];
    const std::vector<int> T = tokenizer.encode(prompt, true, false);
    const int P = static_cast<int>(T.size());
    std::cout << "prompt \"" << prompt << "\" -> " << P << " tokens:";
    for (int t : T) { std::cout << ' ' << t; }
    std::cout << '\n';

    // ---------------- 03_rmsnorm_softmax ----------------
    {
        const std::string dir = root + "/03_rmsnorm_softmax/data/";
        std::vector<float> x = {0.5f, -1.2f, 3.3f, 0.0f, 1e-3f, -2.7f, 4.0f, -0.8f};
        std::vector<float> w = {1.1f, 0.9f, -0.5f, 2.0f, 1.0f, 0.3f, -1.4f, 0.7f};
        std::vector<float> out(x.size());
        write_floats(dir + "input_rmsnorm_x.txt", x);
        write_floats(dir + "input_rmsnorm_w.txt", w);
        rmsnorm(out, x, w);
        write_floats(dir + "expected_rmsnorm.txt", out);

        // real-sized case: embedding row of the first prompt token, normed by layer-0 weights
        std::vector<float> xr(dim), outr(dim);
        std::ranges::copy(tr.weights.token_embedding_table.subspan(static_cast<size_t>(T[0]) * dim, dim),
                          xr.begin());
        write_floats(dir + "input_rmsnorm_x_real.txt", xr);
        // layer-0 rms_att_weight slice, embedded as kRmsNormWReal in 03's data.h
        write_floats(dir + "input_rmsnorm_w_real.txt", tr.weights.rms_att_weight.subspan(0, dim));
        rmsnorm(outr, xr, tr.weights.rms_att_weight.subspan(0, dim));
        write_floats(dir + "expected_rmsnorm_real.txt", outr);

        std::vector<float> sm = {1.0f, 2.0f, 3.0f, 4.0f, -1.0f, 0.0f, 0.5f, -2.0f};
        write_floats(dir + "input_softmax.txt", sm);
        softmax(sm);
        write_floats(dir + "expected_softmax.txt", sm);

        std::vector<float> big = {1000.0f, 1001.0f, 1002.0f, 999.0f};
        write_floats(dir + "input_softmax_big.txt", big);
        softmax(big);
        write_floats(dir + "expected_softmax_big.txt", big);
    }

    // ---------------- 04_matmul ----------------
    {
        const std::string dir = root + "/04_matmul/data/";
        std::vector<float> w = {0.5f, -1.25f, 2.0f, 0.75f,   // row 0
                                -3.0f, 0.125f, 1.5f, -0.5f,  // row 1
                                2.25f, -0.75f, 0.875f, -2.0f}; // row 2
        std::vector<float> x = {0.5f, -1.0f, 2.0f, 0.25f};
        std::vector<float> out(3);
        write_floats(dir + "input_w.txt", w);
        write_floats(dir + "input_x.txt", x);
        matmul(out, x, w);
        write_floats(dir + "expected_out.txt", out);
    }

    // ---------------- 05/06/07: instrumented forward over the prompt ----------------
    Captures cap;
    cap.total_pos = P;
    std::vector<std::vector<float>> logits_dump(P);
    for (int pos = 0; pos < P; pos++) {
        std::span<float> lg = forward_dump(tr, T[pos], pos, cap);
        logits_dump[pos].assign(lg.begin(), lg.end());
    }
    {
        write_floats(root + "/05_rope/data/input_q.txt", cap.q_pre);
        write_floats(root + "/05_rope/data/input_k.txt", cap.k_pre);
        write_floats(root + "/05_rope/data/expected_q.txt", cap.q_post);
        write_floats(root + "/05_rope/data/expected_k.txt", cap.k_post);

        write_floats(root + "/06_attention/data/input_q.txt", cap.q_post);
        write_floats(root + "/06_attention/data/input_k_cache.txt", cap.k_post);
        write_floats(root + "/06_attention/data/input_v_cache.txt", cap.v_all);
        write_floats(root + "/06_attention/data/expected_out.txt", cap.att_out);
        write_floats(root + "/06_attention/data/expected_att_weights_lastpos.txt", cap.att_weights);

        write_floats(root + "/07_ffn/data/input_x.txt", cap.ffn_in);
        write_floats(root + "/07_ffn/data/expected_hidden.txt", cap.ffn_hidden);
        write_floats(root + "/07_ffn/data/expected_out.txt", cap.ffn_out);

        write_ints(root + "/08_transformer_layer/data/input_tokens.txt", T);
        write_floats(root + "/08_transformer_layer/data/expected_att_residual.txt", cap.att_res);
        write_floats(root + "/08_transformer_layer/data/expected_layer_out.txt", cap.layer_out);
    }

    // ---------------- 09_forward: original forward, plus bit-exactness check ----------------
    Transformer tr2(checkpoint);
    std::vector<float> all_logits;
    std::vector<int> argmaxes;
    float max_diff = 0.0f;
    for (int pos = 0; pos < P; pos++) {
        std::span<float> lg = tr2.forward(T[pos], pos);
        append(all_logits, lg.data(), lg.size());
        argmaxes.push_back(static_cast<int>(std::ranges::max_element(lg) - lg.begin()));
        for (size_t i = 0; i < lg.size(); i++) {
            max_diff = std::max(max_diff, std::abs(lg[i] - logits_dump[pos][i]));
        }
    }
    std::cout << "instrumented vs original forward, max |logit diff|: " << max_diff << '\n';
    {
        const std::string dir = root + "/09_forward/data/";
        write_ints(dir + "input_tokens.txt", T);
        write_floats(dir + "expected_logits.txt", all_logits);
        write_ints(dir + "expected_argmax.txt", argmaxes);
    }

    // ---------------- 10_sampler ----------------
    {
        const std::string dir = root + "/10_sampler/data/";
        std::vector<float> logits_last(logits_dump[P - 1]);
        write_floats(dir + "input_logits.txt", logits_last);

        // replicate Sampler's private xorshift to expose the raw random sequence
        std::uint64_t st = 42;
        auto random_u32 = [&st]() {
            st ^= st >> 12;
            st ^= st << 25;
            st ^= st >> 27;
            return (st * 0x2545F4914F6CDD1Dull) >> 32;
        };
        std::vector<float> coins;
        for (int i = 0; i < 10; i++) { coins.push_back((random_u32() >> 8) / 16777216.0f); }
        write_floats(dir + "expected_rng_seed42.txt", coins);

        const struct { float t, p; std::uint64_t seed; } cases[] = {
            {0.0f, 0.9f, 42}, {1.0f, 1.0f, 42}, {0.8f, 0.9f, 42}, {0.8f, 0.9f, 1234},
        };
        std::vector<int> samples;
        for (const auto& c : cases) {
            Sampler smp(p.vocab_size, c.t, c.p, c.seed);
            std::vector<float> copy = logits_last; // sample() mutates the logits
            samples.push_back(smp.sample(copy));
        }
        std::cout << "sampler cases (real logits):";
        for (int s_id : samples) { std::cout << ' ' << s_id; }
        std::cout << '\n';

        // synthetic 8-token logits: the real distribution is 96.6% peaked on the argmax,
        // so all real cases pick it; a flat synthetic one exercises mult / top-p paths
        std::vector<float> synth = {1, 2, 3, 4, 5, 6, 7, 8};
        write_floats(dir + "input_logits_synth.txt", synth);
        const struct { float t, p; std::uint64_t seed; } synth_cases[] = {
            {0.0f, 0.9f, 42}, {1.0f, 1.0f, 42}, {1.0f, 0.5f, 42}, {2.0f, 0.9f, 7},
        };
        for (const auto& c : synth_cases) {
            Sampler smp(8, c.t, c.p, c.seed);
            std::vector<float> copy = synth;
            samples.push_back(smp.sample(copy));
        }
        std::cout << "sampler cases (synthetic logits):";
        for (size_t i = 4; i < samples.size(); i++) { std::cout << ' ' << samples[i]; }
        std::cout << '\n';
        write_ints(dir + "expected_samples.txt", samples);
    }

    // ---------------- 11_generate ----------------
    auto run_generate = [&](float temp, float topp, std::uint64_t seed, int steps,
                            std::vector<int>& ids, std::string& text) {
        Transformer g_tr(checkpoint);
        Sampler sampler(g_tr.config.vocab_size, temp, topp, seed);
        std::vector<int> prompt_tokens = tokenizer.encode(prompt, true, false);
        int next = 0;
        int token = prompt_tokens[0];
        int pos = 0;
        while (pos < steps) {
            std::span<float> logits = g_tr.forward(token, pos);
            if (pos < static_cast<int>(prompt_tokens.size()) - 1) {
                next = prompt_tokens[pos + 1];
            } else {
                next = sampler.sample(logits);
            }
            pos++;
            if (next == 1) { break; }
            ids.push_back(next);
            text += std::string(tokenizer.decode(token, next));
            token = next;
        }
    };
    {
        const std::string dir = root + "/11_generate/data/";
        std::vector<int> ids;
        std::string text;
        run_generate(0.0f, 0.9f, 42, 64, ids, text);
        write_ints(dir + "expected_greedy_ids.txt", ids);
        write_text(dir + "expected_greedy_text.txt", text + '\n');
        ids.clear();
        text.clear();
        run_generate(0.8f, 0.9f, 42, 64, ids, text);
        write_ints(dir + "expected_sampled_ids.txt", ids);
        write_text(dir + "expected_sampled_text.txt", text + '\n');
    }

    std::cout << "dump_fp32 done\n";
    return 0;
}
