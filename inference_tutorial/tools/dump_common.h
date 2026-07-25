// Shared helpers for the dump tools: write golden data as plain-text numbers,
// one value per line. All floating-point values use %.3e (setprecision(3)):
// comparison is meant to happen at 3-decimal precision (see tools/compare.py).
#pragma once

#include <cstdint>
#include <fstream>
#include <iomanip>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

inline std::ofstream open_out(const std::string& path) {
    std::ofstream f(path);
    if (!f) { throw std::runtime_error("cannot write " + path); }
    return f;
}

inline void write_floats(const std::string& path, std::span<const float> v) {
    auto f = open_out(path);
    f << std::scientific << std::setprecision(3);
    for (float x : v) { f << x << '\n'; }
}

inline void write_doubles(const std::string& path, std::span<const double> v) {
    auto f = open_out(path);
    f << std::scientific << std::setprecision(3);
    for (double x : v) { f << x << '\n'; }
}

inline void write_ints(const std::string& path, std::span<const int> v) {
    auto f = open_out(path);
    for (int x : v) { f << x << '\n'; }
}

inline void write_text(const std::string& path, const std::string& s) {
    auto f = open_out(path);
    f << s;
}

// append `count` floats starting at `ptr` into `out` (used to stack per-position vectors)
inline void append(std::vector<float>& out, const float* ptr, size_t count) {
    out.insert(out.end(), ptr, ptr + count);
}
