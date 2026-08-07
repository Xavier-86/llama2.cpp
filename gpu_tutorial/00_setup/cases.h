// 00_setup/cases.h — checkpoint weight tensor size table (pure C++, shared by GPU/CPU/data generator)
//
// stories checkpoint layout (see the TransformerWeights mapping in cpu/run.cpp):
//   [7 × int32 config values][float weight region]
// Tensor order: emb, rms_att, wq, wk, wv, wo, rms_ffn, w1, w2, w3, rms_final, [wcls]
// vocab_size > 0 means the classifier is shared (wcls aliases emb and occupies no extra space).
#pragma once

#include <cstdlib>
#include <vector>

namespace gt {

// Return the element counts of 12 tensors in checkpoint order (number of floats).
inline std::vector<long long> tensor_float_sizes(const int cfg[7]) {
    const int dim = cfg[0], hidden = cfg[1], layers = cfg[2];
    const int n_heads = cfg[3], n_kv_heads = cfg[4], vocab = std::abs(cfg[5]);
    const int kvd = dim / n_heads * n_kv_heads;       // KV dimension (= dim for MHA)
    const bool shared_classifier = cfg[5] > 0;
    return {
        (long long)vocab * dim,                        // token_embedding_table
        (long long)layers * dim,                       // rms_att_weight
        (long long)layers * dim * dim,                 // wq
        (long long)layers * dim * kvd,                 // wk
        (long long)layers * dim * kvd,                 // wv
        (long long)layers * dim * dim,                 // wo
        (long long)layers * dim,                       // rms_ffn_weight
        (long long)layers * dim * hidden,              // w1
        (long long)layers * hidden * dim,              // w2
        (long long)layers * dim * hidden,              // w3
        (long long)dim,                                // rms_final_weight
        shared_classifier ? 0ll : (long long)vocab * dim, // wcls
    };
}

// Golden output table: 12 tensor byte counts plus the total weight-region size (13 ints).
inline std::vector<int> tensor_byte_table(const int cfg[7]) {
    std::vector<int> t;
    long long total = 0;
    for (long long n : tensor_float_sizes(cfg)) {
        t.push_back((int)(n * 4));
        total += n * 4;
    }
    t.push_back((int)total);
    return t;
}

} // namespace gt
