// 01 checkpoint: load the model weights — reference solution
//
// Parses the llama2.c FP32 checkpoint header, maps the 11 weight tensors as
// (pointer, length) views in file order, and dumps:
//   out_config.txt  — the 7 config int32 fields, one per line
//   out_summary.txt — per tensor: element count, first element, sum of all
//                     elements (accumulated in double, file order); 33 lines
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution            (from this module folder)
// Verify: python3 ../tools/compare.py out_config.txt data/expected_config.txt --exact
//         python3 ../tools/compare.py out_summary.txt data/expected_weight_summary.txt

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <vector>

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

// each tensor is just a view into the weight region — no copies
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

int main() {
    const std::string checkpoint_path = "../../stories15M.bin";

    // read the whole file into a float buffer (15M model: simple ifstream is fine)
    std::ifstream in(checkpoint_path, std::ios::binary | std::ios::ate);
    if (!in) {
        std::cerr << "cannot open " << checkpoint_path << "\n";
        return 1;
    }
    const std::streamsize file_bytes = in.tellg();
    in.seekg(0);
    std::vector<float> buf(static_cast<size_t>(file_bytes) / sizeof(float));
    in.read(reinterpret_cast<char*>(buf.data()), file_bytes);
    if (!in) {
        std::cerr << "failed to read " << checkpoint_path << "\n";
        return 1;
    }

    // task 1: parse the config header and print the 7 integers as stored
    Config cfg;
    std::memcpy(&cfg, buf.data(), sizeof(cfg));

    {
        std::ofstream out("out_config.txt");
        out << cfg.dim << "\n"
            << cfg.hidden_dim << "\n"
            << cfg.n_layers << "\n"
            << cfg.n_heads << "\n"
            << cfg.n_kv_heads << "\n"
            << cfg.vocab_size << "\n"
            << cfg.seq_len << "\n";
    }

    // derived sizes; vocab_size magnitude is the real vocab size
    const bool shared = cfg.vocab_size > 0;
    const uint64_t vocab = shared ? cfg.vocab_size : -cfg.vocab_size;
    const uint64_t dim = cfg.dim;
    const uint64_t hidden = cfg.hidden_dim;
    const uint64_t n_layers = cfg.n_layers;
    const uint64_t head_size = dim / cfg.n_heads;
    const uint64_t kv_dim = cfg.n_kv_heads * head_size;

    // task 2: carve the 11 tensors out of the weight region in file order
    Weights w;
    const float* ptr = buf.data() + sizeof(Config) / sizeof(float);
    auto carve = [&ptr](uint64_t count) {
        std::span<const float> slice{ptr, static_cast<size_t>(count)};
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
    ptr += cfg.seq_len * head_size / 2;
    ptr += cfg.seq_len * head_size / 2;
    w.wcls = shared ? w.token_embedding_table : carve(vocab * dim);

    // per tensor: element count, first element, sum (double, file order)
    std::ofstream out("out_summary.txt");
    out << std::scientific << std::setprecision(3);
    auto dump = [&out](std::span<const float> t) {
        double sum = 0.0;
        for (float v : t) {
            sum += v;
        }
        out << static_cast<double>(t.size()) << "\n"
            << static_cast<double>(t[0]) << "\n"
            << sum << "\n";
    };
    dump(w.token_embedding_table);
    dump(w.rms_att_weight);
    dump(w.wq);
    dump(w.wk);
    dump(w.wv);
    dump(w.wo);
    dump(w.rms_ffn_weight);
    dump(w.w1);
    dump(w.w2);
    dump(w.w3);
    dump(w.rms_final_weight);

    std::cout << "wrote out_config.txt and out_summary.txt\n";
    return 0;
}
