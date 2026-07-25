// Golden-data generator for inference_tutorial (int8 quantized half).
// Same approach as dump_fp32.cpp: include the reference runq.cpp, then dump
// checkpoint layout summaries, quantize/dequantize and int8-matmul test vectors,
// quantized forward logits, and a full greedy generation.
//
// usage: dump_int8 <out_root> <quantized_checkpoint> <tokenizer>
// e.g.:  ./dump_int8 inference_tutorial stories15M-q32.bin tokenizer.bin

#define main llama2_reference_cli_main
#include "../../runq.cpp"
#undef main

#include "dump_common.h"
#include <iostream>

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: dump_int8 <out_root> <quantized_checkpoint> <tokenizer>\n";
        return 1;
    }
    const std::string root = argv[1];
    const std::string checkpoint = argv[2];
    const std::string tokenizer_path = argv[3];

    Transformer tr(checkpoint);
    const Config& p = tr.config;
    const int GS = tr.GS;
    const std::string dir = root + "/11_quantize/data/";

    std::cout << "config: dim=" << p.dim << " hidden_dim=" << p.hidden_dim
              << " n_layers=" << p.n_layers << " n_heads=" << p.n_heads
              << " n_kv_heads=" << p.n_kv_heads << " vocab_size=" << p.vocab_size
              << " seq_len=" << p.seq_len << " GS=" << GS
              << " shared_classifier="
              << (tr.weights.wcls.q.data() == tr.weights.q_tokens[0].q.data()) << '\n';

    // ---------------- quantized checkpoint layout ----------------
    write_ints(dir + "expected_config.txt",
               std::array{p.dim, p.hidden_dim, p.n_layers, p.n_heads,
                          p.n_kv_heads, p.vocab_size, p.seq_len, GS});
    {
        // fp32 tensors first (mapping order): rms_att_weight, rms_ffn_weight, rms_final_weight
        const std::pair<const char*, std::span<float>> fp32_weights[] = {
            {"rms_att_weight", tr.weights.rms_att_weight},
            {"rms_ffn_weight", tr.weights.rms_ffn_weight},
            {"rms_final_weight", tr.weights.rms_final_weight},
        };
        std::vector<double> summary; // per fp32 tensor: size, first element, sum
        for (const auto& [name, data] : fp32_weights) {
            double sum = 0.0;
            for (float v : data) { sum += v; }
            summary.push_back(static_cast<double>(data.size()));
            summary.push_back(data[0]);
            summary.push_back(sum);
            std::cout << "fp32 weight " << name << " size=" << data.size() << '\n';
        }
        // quantized tensors (mapping order): q_tokens, wq, wk, wv, wo, w1, w2, w3
        const std::pair<const char*, const std::vector<QuantizedTensor>*> q_weights[] = {
            {"q_tokens", &tr.weights.q_tokens}, {"wq", &tr.weights.wq},
            {"wk", &tr.weights.wk},             {"wv", &tr.weights.wv},
            {"wo", &tr.weights.wo},             {"w1", &tr.weights.w1},
            {"w2", &tr.weights.w2},             {"w3", &tr.weights.w3},
        };
        // per quantized tensor: q_total_size, q_first, q_sum, s_total_size, s_first
        for (const auto& [name, layers] : q_weights) {
            std::int64_t q_size = 0, q_sum = 0, s_size = 0;
            for (const QuantizedTensor& t : *layers) {
                q_size += static_cast<std::int64_t>(t.q.size());
                s_size += static_cast<std::int64_t>(t.s.size());
                for (std::int8_t v : t.q) { q_sum += v; }
            }
            summary.push_back(static_cast<double>(q_size));
            summary.push_back((*layers)[0].q[0]);
            summary.push_back(static_cast<double>(q_sum));
            summary.push_back(static_cast<double>(s_size));
            summary.push_back((*layers)[0].s[0]);
            std::cout << "quantized weight " << name << " layers=" << layers->size()
                      << " q_size=" << q_size << " s_size=" << s_size << '\n';
        }
        write_doubles(dir + "expected_weight_summary.txt", summary);
    }

    // ---------------- quantize / dequantize ----------------
    {
        std::vector<float> x(64);
        for (int i = 0; i < 64; i++) { x[i] = (i % 9 - 4) * 0.03125f * (1 + i % 5); }
        write_floats(dir + "input_x.txt", x);

        std::vector<std::int8_t> q(64);
        std::vector<float> s(2); // GS = 32
        QuantizedTensor qx{std::span{q}, std::span{s}};
        quantize(qx, x);
        std::vector<int> q_as_int(q.begin(), q.end());
        write_ints(dir + "expected_q.txt", q_as_int);
        write_floats(dir + "expected_s.txt", s);

        std::vector<float> deq(64);
        dequantize(qx, deq);
        write_floats(dir + "expected_deq.txt", deq);
    }

    // ---------------- int8 matmul (n=8, d=3, GS=4) ----------------
    {
        std::vector<std::int8_t> wq = {3, -2, 5, 1,  -4, 6, -1, 2,  0, 7, -3, -5,
                                       1, 1, -2, 4,  2, -6, 3, 0,  -1, 5, -7, 2};
        std::vector<float> ws = {0.02f, 0.05f, 0.01f, 0.03f, 0.04f, 0.02f}; // (d, n/GS)
        std::vector<std::int8_t> xq = {4, -3, 2, 6, -5, 1, 0, -2};
        std::vector<float> xs = {0.1f, 0.2f}; // (n/GS,)
        std::vector<float> out(3);

        std::vector<int> wq_int(wq.begin(), wq.end()), xq_int(xq.begin(), xq.end());
        write_ints(dir + "input_matmul_wq.txt", wq_int);
        write_floats(dir + "input_matmul_ws.txt", ws);
        write_ints(dir + "input_matmul_xq.txt", xq_int);
        write_floats(dir + "input_matmul_xs.txt", xs);

        QuantizedTensor w{std::span{wq}, std::span{ws}};
        QuantizedTensor x{std::span{xq}, std::span{xs}};
        matmul(out, x, w);
        write_floats(dir + "expected_matmul_out.txt", out);
    }

    // ---------------- quantized forward over the prompt ----------------
    Tokenizer tokenizer(tokenizer_path, p.vocab_size);
    const std::string prompt = "Once upon a time";
    const std::vector<int> T = tokenizer.encode(prompt, true, false);
    const int P = static_cast<int>(T.size());
    {
        std::vector<int> argmaxes;
        std::vector<float> logits_last;
        for (int pos = 0; pos < P; pos++) {
            std::span<float> lg = tr.forward(T[pos], pos);
            argmaxes.push_back(static_cast<int>(std::ranges::max_element(lg) - lg.begin()));
            if (pos == P - 1) { logits_last.assign(lg.begin(), lg.end()); }
        }
        write_ints(dir + "input_tokens.txt", T);
        write_ints(dir + "expected_argmax.txt", argmaxes);
        write_floats(dir + "expected_logits_lastpos.txt", logits_last);
        std::cout << "quantized forward argmaxes:";
        for (int a : argmaxes) { std::cout << ' ' << a; }
        std::cout << '\n';
    }

    // ---------------- greedy generation with the quantized model ----------------
    {
        Sampler sampler(p.vocab_size, 0.0f, 0.9f, 42); // temperature 0 = greedy
        std::vector<int> ids;
        std::string text;
        int next = 0;
        int token = T[0];
        int pos = 0;
        while (pos < 64) {
            std::span<float> logits = tr.forward(token, pos);
            if (pos < P - 1) {
                next = T[pos + 1];
            } else {
                next = sampler.sample(logits);
            }
            pos++;
            if (next == 1) { break; }
            ids.push_back(next);
            text += std::string(tokenizer.decode(token, next));
            token = next;
        }
        write_ints(dir + "expected_greedy_ids.txt", ids);
        write_text(dir + "expected_greedy_text.txt", text + '\n');
    }

    std::cout << "dump_int8 done\n";
    return 0;
}
