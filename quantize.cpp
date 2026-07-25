// Quantize a llama2.c fp32 checkpoint into the int8 format that runq.cpp reads.
//
// Usage: quantize <fp32_checkpoint.bin> <output.bin> <group_size>
//
// Input format (read by run.c / run.cpp):
//   Config header (7 ints) + fp32 weights; a negative vocab_size means the classifier is not shared
// Output format (read by runq.c / runq.cpp, "ak42" version 2):
//   256-byte header (magic + version + Config + shared flag + group_size, zero-padded)
//   fp32 rmsnorm weights (att / ffn / final)
//   all other weights quantized tensor by tensor: int8 values first, then float scale factors (one per group_size elements)
//
// The quantization strategy is identical to runq.cpp's quantize(): symmetric, scale set by max absolute value per group.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

struct Config {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

// Same as runq.cpp's quantize: symmetric quantization with per-group max-abs scaling
static void quantize_block(const float* x, int8_t* q, float* s, int n, int GS) {
    const float Q_MAX = 127.0f;
    for (int group = 0; group < n / GS; group++) {
        float wmax = 0.0f;
        for (int i = 0; i < GS; i++) {
            float val = std::fabs(x[group * GS + i]);
            if (val > wmax) { wmax = val; }
        }
        float scale = wmax / Q_MAX;
        s[group] = scale;
        for (int i = 0; i < GS; i++) {
            q[group * GS + i] = static_cast<int8_t>(std::round(x[group * GS + i] / scale));
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: quantize <fp32_checkpoint.bin> <output.bin> <group_size>\n";
        return EXIT_FAILURE;
    }
    const std::string in_path = argv[1];
    const std::string out_path = argv[2];
    const int GS = std::atoi(argv[3]);
    if (GS <= 0) { std::cerr << "group_size must be positive\n"; return EXIT_FAILURE; }

    // read the whole fp32 checkpoint
    FILE* fin = std::fopen(in_path.c_str(), "rb");
    if (!fin) { std::cerr << "Couldn't open " << in_path << "\n"; return EXIT_FAILURE; }
    std::fseek(fin, 0, SEEK_END);
    long file_size = std::ftell(fin);
    std::fseek(fin, 0, SEEK_SET);
    std::vector<char> buf(file_size);
    if (std::fread(buf.data(), file_size, 1, fin) != 1) { std::cerr << "failed read\n"; return EXIT_FAILURE; }
    std::fclose(fin);

    Config config;
    std::memcpy(&config, buf.data(), sizeof(Config));
    bool shared = config.vocab_size > 0;
    config.vocab_size = std::abs(config.vocab_size);
    const int head_size = config.dim / config.n_heads;
    const long long L = config.n_layers;

    // verify every tensor to be quantized has a length divisible by GS
    auto check = [&](long long n, const char* name) {
        if (n % GS != 0) {
            std::cerr << "tensor " << name << " (" << n << ") not divisible by group_size " << GS << "\n";
            std::exit(EXIT_FAILURE);
        }
    };

    FILE* fout = std::fopen(out_path.c_str(), "wb");
    if (!fout) { std::cerr << "Couldn't write " << out_path << "\n"; return EXIT_FAILURE; }

    // 256-byte header: magic "ak42" + version 2 + Config + shared + group_size, rest zero-padded
    std::vector<char> header(256, 0);
    uint32_t magic = 0x616b3432;
    int version = 2;
    uint8_t shared_u8 = shared ? 1 : 0;
    char* hp = header.data();
    std::memcpy(hp, &magic, 4); hp += 4;
    std::memcpy(hp, &version, 4); hp += 4;
    std::memcpy(hp, &config, sizeof(Config)); hp += sizeof(Config);
    std::memcpy(hp, &shared_u8, 1); hp += 1;
    std::memcpy(hp, &GS, 4);
    std::fwrite(header.data(), 256, 1, fout);

    const float* src = reinterpret_cast<const float*>(buf.data() + sizeof(Config));
    auto take = [&](long long n) {
        const float* p = src;
        src += n;
        return p;
    };
    auto write_f32 = [&](const float* p, long long n) { std::fwrite(p, sizeof(float), n, fout); };
    // runq expects each tensor to carry its own [int8 values][scale factors],
    // so multi-layer tensors (wq/w1/...) must be quantized and written layer by layer
    auto write_quant = [&](const float* p, long long count, long long size_each, const char* name) {
        check(size_each, name);
        std::vector<int8_t> q(size_each);
        std::vector<float> s(size_each / GS);
        for (long long i = 0; i < count; i++) {
            quantize_block(p + i * size_each, q.data(), s.data(), size_each, GS);
            std::fwrite(q.data(), sizeof(int8_t), size_each, fout);
            std::fwrite(s.data(), sizeof(float), size_each / GS, fout);
        }
    };

    // write out in the order runq.cpp's map_weights expects
    const float* token_embedding_table = take((long long)config.vocab_size * config.dim);
    const float* rms_att_weight = take(L * config.dim);
    const float* wq = take(L * config.dim * (config.n_heads * head_size));
    const float* wk = take(L * config.dim * (config.n_kv_heads * head_size));
    const float* wv = take(L * config.dim * (config.n_kv_heads * head_size));
    const float* wo = take(L * (config.n_heads * head_size) * config.dim);
    const float* rms_ffn_weight = take(L * config.dim);
    const float* w1 = take(L * config.dim * config.hidden_dim);
    const float* w2 = take(L * config.hidden_dim * config.dim);
    const float* w3 = take(L * config.dim * config.hidden_dim);
    const float* rms_final_weight = take(config.dim);
    take((long long)config.seq_len * head_size / 2); // skip freq_cis_real
    take((long long)config.seq_len * head_size / 2); // skip freq_cis_imag
    const float* wcls = shared ? nullptr : take((long long)config.vocab_size * config.dim);

    write_f32(rms_att_weight, L * config.dim);
    write_f32(rms_ffn_weight, L * config.dim);
    write_f32(rms_final_weight, config.dim);

    write_quant(token_embedding_table, 1, (long long)config.vocab_size * config.dim, "q_tokens");
    write_quant(wq, L, (long long)config.dim * (config.n_heads * head_size), "wq");
    write_quant(wk, L, (long long)config.dim * (config.n_kv_heads * head_size), "wk");
    write_quant(wv, L, (long long)config.dim * (config.n_kv_heads * head_size), "wv");
    write_quant(wo, L, (long long)(config.n_heads * head_size) * config.dim, "wo");
    write_quant(w1, L, (long long)config.dim * config.hidden_dim, "w1");
    write_quant(w2, L, (long long)config.hidden_dim * config.dim, "w2");
    write_quant(w3, L, (long long)config.dim * config.hidden_dim, "w3");
    if (!shared) {
        write_quant(wcls, 1, (long long)config.vocab_size * config.dim, "wcls");
    }

    std::fclose(fout);
    std::cout << "quantized " << in_path << " -> " << out_path << " (GS=" << GS
              << ", shared_classifier=" << shared << ")\n";
    return 0;
}
