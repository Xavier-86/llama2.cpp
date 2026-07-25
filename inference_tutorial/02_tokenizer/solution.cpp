// 02_tokenizer -- reference solution
//
// BPE tokenizer for the llama2 vocab: encode (text -> token ids) and
// decode (id -> text piece), mirroring Tokenizer in ../../run.cpp.
//
// Build:  c++ -O2 -std=c++20 -o solution solution.cpp
// Run:    ./solution            (from this module folder)
//
// Reads:  ../../tokenizer.bin, data/input_prompts.txt, data/input_decode_ids.txt
// Writes: out0.txt .. out3.txt (one token id per line, BOS included)
//         out_decode.txt       (concatenated decode pieces, no trailing newline)

#include <algorithm>
#include <array>
#include <charconv>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr int kVocabSize = 32000;
constexpr const char* kTokenizerPath = "../../tokenizer.bin";

struct TokenIndex {
    std::string_view str;
    int id;
};

class Tokenizer {
public:
    explicit Tokenizer(const std::string& tokenizer_path) {
        vocab_.resize(kVocabSize);
        vocab_scores_.resize(kVocabSize);
        // Printable single-byte strings for the <0xXX> fallback pieces.
        for (int i = 0; i < 256; i++) {
            byte_pieces_[i * 2] = static_cast<char>(i);
            byte_pieces_[i * 2 + 1] = '\0';
        }

        std::ifstream file(tokenizer_path, std::ios::binary);
        if (!file) { throw std::runtime_error("couldn't load " + tokenizer_path); }
        auto read_or_die = [&](void* dst, std::streamsize size) {
            if (!file.read(static_cast<char*>(dst), size)) {
                throw std::runtime_error("tokenizer file is truncated: " + tokenizer_path);
            }
        };
        read_or_die(&max_token_length_, sizeof(int));
        for (int i = 0; i < kVocabSize; i++) {
            read_or_die(&vocab_scores_[i], sizeof(float));
            int len;
            read_or_die(&len, sizeof(int));
            std::string s(len, '\0');
            read_or_die(s.data(), len);
            vocab_[i] = std::move(s);
        }
    }

    // Returns the text piece for `token`; `prev_token` is needed to strip the
    // leading space right after BOS and to expand <0xXX> byte pieces.
    std::string_view decode(int prev_token, int token) const {
        std::string_view piece = vocab_[token];
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
                std::string merged = vocab_[tokens[i]] + vocab_[tokens[i + 1]];
                int id = str_lookup(merged);
                if (id != -1 && vocab_scores_[id] > best_score) {
                    best_score = vocab_scores_[id];
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
            sorted_vocab_.push_back({vocab_[i], i});
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

    std::vector<std::string> vocab_;
    std::vector<float> vocab_scores_;
    std::vector<TokenIndex> sorted_vocab_;
    int max_token_length_ = 0;
    std::array<char, 512> byte_pieces_{};
};

std::vector<std::string> read_lines(const std::string& path) {
    std::ifstream file(path);
    if (!file) { throw std::runtime_error("couldn't open " + path); }
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(file, line)) { lines.push_back(line); }
    return lines;
}

std::vector<int> read_ints(const std::string& path) {
    std::ifstream file(path);
    if (!file) { throw std::runtime_error("couldn't open " + path); }
    std::vector<int> values;
    int v;
    while (file >> v) { values.push_back(v); }
    return values;
}

} // namespace

int main() {
    Tokenizer tokenizer(kTokenizerPath);

    // Task: encode each prompt (BOS on, EOS off) and dump the token ids.
    std::vector<std::string> prompts = read_lines("data/input_prompts.txt");
    for (size_t i = 0; i < prompts.size(); i++) {
        std::vector<int> tokens = tokenizer.encode(prompts[i], /*bos=*/true, /*eos=*/false);
        std::ofstream out("out" + std::to_string(i) + ".txt");
        for (int t : tokens) { out << t << "\n"; }
    }

    // Task: decode the id sequence back to text: decode(ids[i], ids[i+1]).
    std::vector<int> ids = read_ints("data/input_decode_ids.txt");
    std::ofstream out("out_decode.txt");
    for (size_t i = 0; i + 1 < ids.size(); i++) {
        out << tokenizer.decode(ids[i], ids[i + 1]);
    }
    return 0;
}
