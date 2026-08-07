// 02_tokenizer — reference solution
//
// BPE tokenizer for the llama2 vocab: encode (text -> token ids) and
// decode (id -> text piece), mirroring Tokenizer in ../../cpu/run.cpp.
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution
// Verify: python3 ../tools/compare.py out0.txt data/expected_encode_0.txt --exact
//         python3 ../tools/compare.py out1.txt data/expected_encode_1.txt --exact
//         python3 ../tools/compare.py out2.txt data/expected_encode_2.txt --exact
//         python3 ../tools/compare.py out3.txt data/expected_encode_3.txt --exact
//         python3 ../tools/compare.py out_decode.txt data/expected_decode.txt --text

#include <algorithm>
#include <array>
#include <charconv>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "../common/io.h"
#include "../common/tokenizer.h"

// ---------------------------------------------------------------------------
// Vocab constants and test inputs
// ---------------------------------------------------------------------------

constexpr int kVocabSize = 32000;
constexpr const char* kTokenizerPath = "../../models/tokenizer.bin";

// The four prompts to encode. Prompt 0 is the reference prompt used across
// the tutorial: "Once upon a time" -> [1, 9038, 2501, 263, 931].
const std::array<std::string, 4> kPrompts = {
    "Once upon a time",
    "One day, a little girl named Lily",
    "Hello, world!",
    "The capital of France is",
};

// The id sequence to decode back to text (prompt 0's tokens, BOS first).
const std::array<int, 5> kDecodeIds = {1, 9038, 2501, 263, 931};

// ---------------------------------------------------------------------------
// The tokenizer: BPE encode and decode
// ---------------------------------------------------------------------------

struct TokenIndex {
    std::string_view str;
    int id;
};

class Tokenizer {
public:
    Tokenizer() : vocab_(tut::load_vocab(kTokenizerPath, kVocabSize)) {
        // Printable single-byte buffers for the <0xXX> fallback pieces.
        for (int i = 0; i < 256; i++) {
            byte_pieces_[i * 2] = static_cast<char>(i);
            byte_pieces_[i * 2 + 1] = '\0';
        }
    }

    // Returns the text piece for `token`; `prev_token` is needed to strip the
    // leading space right after BOS and to expand <0xXX> byte pieces.
    std::string_view decode(int prev_token, int token) const {
        std::string_view piece = vocab_.pieces[token];
        if (prev_token == 1 && piece.starts_with(' ')) { piece.remove_prefix(1); }
        if (piece.size() == 6 && piece.starts_with("<0x") && piece.back() == '>') {
            unsigned int byte_val = 0;
            const char* hex = piece.data() + 3;
            auto [end, ec] = std::from_chars(hex, hex + 2, byte_val, 16);
            if (ec == std::errc{} && end == hex + 2) {
                return {byte_pieces_.data() + byte_val * 2, 1};
            }
        }
        return piece;
    }

    // BPE encode: BOS + leading " " + per-character tokens, then greedy merges.
    std::vector<int> encode(const std::string& text, bool bos, bool eos) {
        std::vector<int> tokens;
        if (bos) { tokens.push_back(1); } // 1 = <s>
        // Llama convention: pretend the text started with a space.
        if (!text.empty()) { tokens.push_back(str_lookup(" ")); }

        // Per-character lookup on UTF-8 boundaries: a byte with
        // (c & 0xC0) != 0x80 starts a new character.
        std::string str_buffer;
        for (size_t i = 0; i < text.size(); i++) {
            char c = text[i];
            if ((c & 0xC0) != 0x80) { str_buffer.clear(); }
            str_buffer.push_back(c);
            if (i + 1 < text.size() && (text[i + 1] & 0xC0) == 0x80 && str_buffer.size() < 4) {
                continue; // more continuation bytes belong to this character
            }
            int id = str_lookup(str_buffer);
            if (id != -1) {
                tokens.push_back(id);
            } else {
                // Byte fallback: byte b -> token id b + 3.
                for (unsigned char b : str_buffer) { tokens.push_back(b + 3); }
            }
            str_buffer.clear();
        }

        // Greedy merging: merge the highest-scoring adjacent pair each round.
        while (true) {
            float best_score = -1e10f;
            int best_id = -1;
            int best_idx = -1;
            for (size_t i = 0; i + 1 < tokens.size(); i++) {
                std::string merged = vocab_.pieces[tokens[i]] + vocab_.pieces[tokens[i + 1]];
                int id = str_lookup(merged);
                if (id != -1 && vocab_.scores[id] > best_score) {
                    best_score = vocab_.scores[id];
                    best_id = id;
                    best_idx = static_cast<int>(i);
                }
            }
            if (best_idx == -1) { break; }
            tokens[best_idx] = best_id;
            tokens.erase(tokens.begin() + best_idx + 1);
        }

        if (eos) { tokens.push_back(2); } // 2 = </s>
        return tokens;
    }

private:
    void init_sorted_vocab() {
        if (!sorted_vocab_.empty()) { return; }
        sorted_vocab_.reserve(kVocabSize);
        for (int i = 0; i < kVocabSize; i++) {
            sorted_vocab_.push_back({vocab_.pieces[i], i});
        }
        std::sort(sorted_vocab_.begin(), sorted_vocab_.end(),
                  [](const TokenIndex& a, const TokenIndex& b) { return a.str < b.str; });
    }

    int str_lookup(std::string_view str) {
        init_sorted_vocab();
        auto it = std::lower_bound(sorted_vocab_.begin(), sorted_vocab_.end(), str,
                                   [](const TokenIndex& a, std::string_view b) { return a.str < b; });
        if (it != sorted_vocab_.end() && it->str == str) { return it->id; }
        return -1;
    }

    tut::Vocab vocab_; // pieces[token id] and scores[token id], loaded for you
    std::vector<TokenIndex> sorted_vocab_;
    std::array<char, 512> byte_pieces_{};
};

// ---------------------------------------------------------------------------

int main() {
    Tokenizer tokenizer;

    // Encode each prompt (BOS on, EOS off) and dump the token ids.
    for (size_t i = 0; i < kPrompts.size(); i++) {
        std::vector<int> tokens = tokenizer.encode(kPrompts[i], /*bos=*/true, /*eos=*/false);
        tut::write_ints("out" + std::to_string(i) + ".txt", tokens);
    }

    // Decode the id sequence back to text: decode(ids[i], ids[i+1]).
    std::string decoded;
    for (size_t i = 0; i + 1 < kDecodeIds.size(); i++) {
        decoded += tokenizer.decode(kDecodeIds[i], kDecodeIds[i + 1]);
    }
    tut::write_text("out_decode.txt", decoded);

    std::cout << "wrote out0.txt .. out3.txt out_decode.txt\n";
    return 0;
}
