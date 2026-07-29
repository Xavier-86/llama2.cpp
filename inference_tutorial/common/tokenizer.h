// common/tokenizer.h — vocabulary table loader shared by the tutorial modules.
//
// Loads the llama2.c tokenizer.bin binary format into plain vectors so that
// module code can focus on the BPE algorithm itself (encode/decode, module
// 02) instead of file parsing. Used by 02_tokenizer, 11_generate and
// 12_quantize.
//
// File format:
//   [int32 max_token_length]
//   then vocab_size records of: [float score] [int32 len] [len bytes of piece]
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tut {

struct Vocab {
    std::vector<std::string> pieces; // token id -> text piece
    std::vector<float> scores;       // token id -> BPE merge score
    int32_t max_token_length = 0;
};

inline Vocab load_vocab(const std::string& path, int vocab_size) {
    std::ifstream file(path, std::ios::binary);
    if (!file) { throw std::runtime_error("couldn't load " + path); }
    auto read_or_die = [&](void* dst, std::streamsize size) {
        if (!file.read(static_cast<char*>(dst), size)) {
            throw std::runtime_error("tokenizer file is truncated: " + path);
        }
    };

    Vocab v;
    v.pieces.resize(vocab_size);
    v.scores.resize(vocab_size);
    read_or_die(&v.max_token_length, sizeof(int32_t));
    for (int i = 0; i < vocab_size; i++) {
        read_or_die(&v.scores[i], sizeof(float));
        int32_t len;
        read_or_die(&len, sizeof(int32_t));
        std::string s(len, '\0');
        read_or_die(s.data(), len);
        v.pieces[i] = std::move(s);
    }
    return v;
}

} // namespace tut
