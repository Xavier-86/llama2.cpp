// 01 checkpoint: load the model weights — student template
//
// Goal: parse the llama2.c FP32 checkpoint header, map the 11 weight tensors
// out of the weight region, and produce:
//   out_config.txt  — the 7 config int32 fields, one per line
//   out_summary.txt — per tensor: element count, first element, sum of all
//                     elements (accumulated in double, file order); 33 lines
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main                (from this module folder)
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
    const std::string checkpoint_path = "../../models/stories15M.bin";

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

    // TODO(task 1): parse the Config header from the start of `buf`, print the
    // 7 integers (as stored, one per line) to out_config.txt, and verify with:
    //   python3 ../tools/compare.py out_config.txt data/expected_config.txt --exact

    // TODO(task 2): walk the weight region (it starts right after the header)
    // and record one std::span view per tensor of Weights, in the file order
    // given by the README table. Remember:
    //   - kv_dim = n_kv_heads * (dim / n_heads)
    //   - a negative vocab_size means unshared classifier weights
    //   - the legacy freq_cis_real / freq_cis_imag tables sit between
    //     rms_final_weight and wcls — skip them
    //   - with shared weights, wcls aliases token_embedding_table
    // Then, for each of the 11 tensors in table order, print 3 lines to
    // out_summary.txt (std::scientific, std::setprecision(3)):
    //   element count, first element, sum of all elements (double, file order)
    // and verify with:
    //   python3 ../tools/compare.py out_summary.txt data/expected_weight_summary.txt

    Weights w{};
    (void)w;
    (void)buf;

    std::cout << "TODO: implement tasks 1-2 (see README.md)\n";
    return 0;
}
