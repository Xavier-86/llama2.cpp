// 04 matmul: student template.
// W (d,n) @ x (n,) -> xout (d,), W stored row-major (d rows of n floats).
// Build: c++ -O2 -std=c++20 -o main main.cpp
// Run from the module folder; writes out.txt.

#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Load all whitespace-separated floats from a text file (one number per line).
static std::vector<float> load_vector(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    std::vector<float> v;
    float value;
    while (in >> value) { v.push_back(value); }
    return v;
}

// Write floats one per line, matching the golden data format (%.3e).
static void write_vector(const std::string& path, std::span<const float> v) {
    std::ofstream out(path);
    if (!out) { throw std::runtime_error("cannot write " + path); }
    out << std::scientific << std::setprecision(3);
    for (float value : v) { out << value << '\n'; }
}

// W (d,n) @ x (n,) -> xout (d,); w holds d rows of n elements.
static void matmul(std::span<float> xout, std::span<const float> x, std::span<const float> w) {
    const size_t n = x.size();
    const size_t d = xout.size();
    (void)n;
    (void)d;
    (void)w;
    // TODO(task 1): implement the matrix-vector multiply defined in README.md:
    // each output element is the dot product of one row of W with x.
    // Accumulate in float (not double) to match the reference behavior.
    std::fill(xout.begin(), xout.end(), 0.0f); // stub: all zeros
}

int main() {
    try {
        // data/input_w.txt is a 3x4 matrix (row-major), data/input_x.txt a 4-dim vector.
        const std::vector<float> x = load_vector("data/input_x.txt");
        const std::vector<float> w = load_vector("data/input_w.txt");
        const size_t n = x.size();
        if (n == 0 || w.size() % n != 0) {
            throw std::runtime_error("bad input sizes: w has " + std::to_string(w.size()) +
                                     " values, x has " + std::to_string(n));
        }
        const size_t d = w.size() / n;

        std::vector<float> xout(d);
        matmul(xout, x, w);
        write_vector("out.txt", xout);
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
