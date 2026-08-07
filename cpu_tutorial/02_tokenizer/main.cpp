// 02_tokenizer — student template
//
// BPE tokenizer for the llama2 vocab: encode (text -> token ids) and
// decode (id -> text piece). The vocabulary table itself is loaded for you
// by ../common/tokenizer.h — your job is the BPE algorithm (see README.md).
//
// Prompts and decode ids are const arrays below; outputs go to out*.txt so
// you can verify them against data/expected_*.txt with ../tools/compare.py.
//
// Build:  c++ -O2 -std=c++20 -o main main.cpp
// Run:    ./main
// Verify: python3 ../tools/compare.py out0.txt data/expected_encode_0.txt --exact
//         python3 ../tools/compare.py out1.txt data/expected_encode_1.txt --exact
//         python3 ../tools/compare.py out2.txt data/expected_encode_2.txt --exact
//         python3 ../tools/compare.py out3.txt data/expected_encode_3.txt --exact
//         python3 ../tools/compare.py out_decode.txt data/expected_decode.txt --text

#include <algorithm>
#include <array>
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

    // Returns the text piece for `token`.
    std::string_view decode(int prev_token, int token) const {
        // TODO(task 4): implement decode (see README "decode algorithm"):
        // return the token's string, except: strip a leading space when
        // prev_token == 1 (previous was BOS), and expand a 6-char piece of
        // the form "<0xXX>" to that single byte.
        (void)prev_token;
        (void)token;
        return {};
    }

    // BPE encode: text -> token ids.
    std::vector<int> encode(const std::string& text, bool bos, bool eos) {
        // TODO(task 2): implement the first half of encode (see README
        // "encode algorithm" steps 1-3): push BOS id 1 if bos; push the id
        // of " " for non-empty text; then per-character lookup on UTF-8
        // boundaries with byte fallback (byte b -> token id b + 3).
        //
        // TODO(task 3): implement greedy merging (README step 4): repeatedly
        // merge the highest-scoring adjacent pair whose concatenation exists
        // in the vocab, until no pair can merge. Append EOS id 2 if eos.
        (void)text;
        (void)bos;
        (void)eos;
        return {};
    }

private:
    void init_sorted_vocab() {
        // TODO(task 1): make str_lookup fast: build a sorted index of
        // (string, id) once, or use a hash map. A linear scan per lookup
        // works but is slow inside the merge loop.
        (void)0;
    }

    int str_lookup(std::string_view str) {
        // TODO(task 1, continued): return the id of `str` in the vocab,
        // or -1 if it is not there.
        (void)str;
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
