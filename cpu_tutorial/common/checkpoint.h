// common/checkpoint.h — FP32 checkpoint loader shared by the tutorial modules.
//
// This is the packaged answer to module 01 (01_checkpoint): it parses the
// llama2.c FP32 checkpoint header and maps the 11 weight tensors as
// std::span views into one buffer. Modules that need real model weights
// (07_ffn, 09_forward, 11_generate) call load_checkpoint() once and then
// work with the spans — the binary parsing stays out of their code.
//
// File format (see 01_checkpoint/README.md for the full description):
//   [Config header: 7 x int32] [weight region: tightly packed float32]
// Tensor order in the weight region:
//   token_embedding_table, rms_att_weight, wq, wk, wv, wo, rms_ffn_weight,
//   w1, w2, w3, rms_final_weight, (skip freq_cis_real/imag), wcls
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tut {

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

struct Checkpoint {
    Config config;
    Weights weights;
    std::vector<float> buffer; // owns the bytes the spans point into
};

inline Checkpoint load_checkpoint(const std::string& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    const std::streamsize file_bytes = in.tellg();
    in.seekg(0);

    Checkpoint ckpt;
    ckpt.buffer.resize(static_cast<size_t>(file_bytes) / sizeof(float));
    in.read(reinterpret_cast<char*>(ckpt.buffer.data()), file_bytes);
    if (!in) { throw std::runtime_error("failed to read " + path); }

    Config& cfg = ckpt.config;
    std::memcpy(&cfg, ckpt.buffer.data(), sizeof(cfg));

    // derived sizes; vocab_size magnitude is the real vocab size
    const bool shared = cfg.vocab_size > 0;
    const uint64_t vocab = shared ? cfg.vocab_size : -static_cast<int64_t>(cfg.vocab_size);
    const uint64_t dim = cfg.dim;
    const uint64_t hidden = cfg.hidden_dim;
    const uint64_t n_layers = cfg.n_layers;
    const uint64_t head_size = dim / cfg.n_heads;
    const uint64_t kv_dim = cfg.n_kv_heads * head_size;

    // carve the 11 tensors out of the weight region in file order
    Weights& w = ckpt.weights;
    const float* ptr = ckpt.buffer.data() + sizeof(Config) / sizeof(float);
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

    return ckpt;
}

} // namespace tut
