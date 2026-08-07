// common/io.h — golden-data IO helpers shared by all tutorial modules.
//
// These are pure plumbing: they exist so module code can focus on the
// inference algorithms. Outputs are written in the golden-data format
// (one value per line, %.3e for floats) so ../tools/compare.py can verify
// them against data/expected_*.txt.
#pragma once

#include <fstream>
#include <iomanip>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace tut {

// Write floats one per line in the golden-data format (%.3e).
inline void write_floats(const std::string& path, std::span<const float> v) {
    std::ofstream f(path);
    if (!f) { throw std::runtime_error("cannot write " + path); }
    f << std::scientific << std::setprecision(3);
    for (float x : v) { f << x << '\n'; }
}

// Write integers (token ids, int8 values, argmax results) one per line.
inline void write_ints(const std::string& path, std::span<const int> v) {
    std::ofstream f(path);
    if (!f) { throw std::runtime_error("cannot write " + path); }
    for (int x : v) { f << x << '\n'; }
}

// Write raw text (decoded pieces, generated stories) with no trailing newline.
inline void write_text(const std::string& path, const std::string& text) {
    std::ofstream f(path);
    if (!f) { throw std::runtime_error("cannot write " + path); }
    f << text;
}

// Read a whitespace-separated list of floats (data files hold one per line).
// Only used for inputs too large to embed as const arrays (see the module's
// README — e.g. the 32000-dim logits in 10_sampler).
inline std::vector<float> read_floats(const std::string& path) {
    std::ifstream f(path);
    if (!f) { throw std::runtime_error("cannot open " + path); }
    std::vector<float> v;
    float x;
    while (f >> x) { v.push_back(x); }
    return v;
}

} // namespace tut
