// 00_setup: sanity check -- there is no real computation in this module.
//
// Open the three binary files that every later module depends on and print
// their sizes in bytes, one per line:
//   ../../models/stories15M.bin      FP32 reference model (used by run.cpp)
//   ../../models/stories15M-q32.bin  int8-quantized model, group size 32 (used by runq.cpp)
//   ../../models/tokenizer.bin       vocabulary / tokenizer table
//
// Build and run from this folder:
//   c++ -O2 -std=c++20 -o solution solution.cpp
//   ./solution

#include <fstream>
#include <iostream>

namespace {

// Paths are relative to this module folder (cpu_tutorial/00_setup).
const char* kModelFp32 = "../../models/stories15M.bin";
const char* kModelInt8 = "../../models/stories15M-q32.bin";
const char* kTokenizer = "../../models/tokenizer.bin";

// Open `path` in binary mode and print its size in bytes on its own line.
// Return false (after reporting to stderr) if the file cannot be opened.
bool print_file_size(const char* path) {
    // Open with the get pointer at the end (std::ios::ate) so tellg() is the size.
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        std::cerr << "cannot open " << path << '\n';
        return false;
    }
    const std::streamoff size = file.tellg();
    if (size < 0) {
        std::cerr << "cannot determine size of " << path << '\n';
        return false;
    }
    std::cout << size << '\n';
    return true;
}

}  // namespace

int main() {
    bool ok = true;
    ok &= print_file_size(kModelFp32);
    ok &= print_file_size(kModelInt8);
    ok &= print_file_size(kTokenizer);
    return ok ? 0 : 1;
}
