// 02_tokenizer -- student template
//
// BPE tokenizer for the llama2 vocab: encode (text -> token ids) and
// decode (id -> text piece). See README.md for the file format and the
// exact algorithms.
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main                (from this module folder)
//
// Reads:  ../../tokenizer.bin, data/input_prompts.txt, data/input_decode_ids.txt
// Writes: out0.txt .. out3.txt (one token id per line, BOS included)
//         out_decode.txt       (concatenated decode pieces, no trailing newline)
//
// Verify: python3 ../tools/compare.py out0.txt data/expected_encode_0.txt --exact
//         python3 ../tools/compare.py out_decode.txt data/expected_decode.txt --text

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
        // TODO(task 1): load the vocabulary from tokenizer_path. Format:
        // [max_token_length: int32], then kVocabSize entries of
        // [score: float32] [len: int32] [len bytes of string].
        // Store strings in vocab_ and scores in vocab_scores_.
        // Fail with std::runtime_error if the file is missing or truncated.
        (void)tokenizer_path;
    }

    // Returns the text piece for `token`.
    std::string_view decode(int prev_token, int token) const {
        // TODO(task 5): implement decode (see README "decode algorithm"):
        // return the token's string, except: strip a leading space when
        // prev_token == 1 (previous was BOS), and expand a 6-char piece of
        // the form "<0xXX>" to that single byte.
        (void)prev_token;
        (void)token;
        return {};
    }

    // BPE encode: text -> token ids.
    std::vector<int> encode(const std::string& text, bool bos, bool eos) {
        // TODO(task 3): implement the first half of encode (see README
        // "encode algorithm" steps 1-3): push BOS id 1 if bos; push the id
        // of " " for non-empty text; then per-character lookup on UTF-8
        // boundaries with byte fallback (byte b -> token id b + 3).
        //
        // TODO(task 4): implement greedy merging (README step 4): repeatedly
        // merge the highest-scoring adjacent pair whose concatenation exists
        // in the vocab, until no pair can merge. Append EOS id 2 if eos.
        (void)text;
        (void)bos;
        (void)eos;
        return {};
    }

private:
    void init_sorted_vocab() {
        // TODO(task 2): make str_lookup fast: build a sorted index of
        // (string, id) once, or use a hash map. A linear scan per lookup
        // works but is slow inside the merge loop.
    }

    int str_lookup(std::string_view str) {
        // TODO(task 2, continued): return the id of `str` in the vocab,
        // or -1 if it is not there.
        (void)str;
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

    // TODO(task 6a): encode each line of data/input_prompts.txt with
    // bos=true, eos=false, and write the token ids of prompt i to
    // out<i>.txt, one id per line.
    std::vector<std::string> prompts = read_lines("data/input_prompts.txt");
    for (size_t i = 0; i < prompts.size(); i++) {
        std::vector<int> tokens = tokenizer.encode(prompts[i], /*bos=*/true, /*eos=*/false);
        std::ofstream out("out" + std::to_string(i) + ".txt");
        for (int t : tokens) { out << t << "\n"; }
    }

    // TODO(task 6b): read the id sequence from data/input_decode_ids.txt and
    // write the concatenation of decode(ids[i], ids[i+1]) to out_decode.txt
    // (no trailing newline).
    std::vector<int> ids = read_ints("data/input_decode_ids.txt");
    std::ofstream out("out_decode.txt");
    for (size_t i = 0; i + 1 < ids.size(); i++) {
        out << tokenizer.decode(ids[i], ids[i + 1]);
    }
    return 0;
}
