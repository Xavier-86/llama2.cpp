// 00_setup: sanity check -- there is no real computation in this module.
//
// Open the three binary files that every later module depends on and print
// their sizes in bytes, one per line:
//   ../../models/stories15M.bin      FP32 reference model (used by run.cpp)
//   ../../models/stories15M-q32.bin  int8-quantized model, group size 32 (used by runq.cpp)
//   ../../models/tokenizer.bin       vocabulary / tokenizer table
//
// (The module's main tasks -- building run.cpp / runq.cpp at the repository
// root and reproducing data/expected_greedy.txt -- are listed in README.md;
// this file is only the minimal file-access sanity check.)
//
// Build and run from this folder:
//   c++ -O2 -std=c++20 -o main main.cpp
//   ./main

#include <fstream>
#include <iostream>

namespace {

// Paths are relative to this module folder (inference_tutorial/00_setup).
const char* kModelFp32 = "../../models/stories15M.bin";
const char* kModelInt8 = "../../models/stories15M-q32.bin";
const char* kTokenizer = "../../models/tokenizer.bin";

// Open `path` in binary mode and print its size in bytes on its own line.
// Return false (after reporting to stderr) if the file cannot be opened.
bool print_file_size(const char* path) {
    // TODO(task 1): open `path` as a binary file; if it cannot be opened,
    //               print an error message to stderr and return false.
    // TODO(task 2): determine the file's size in bytes and print it on its
    //               own line to stdout, then return true.
    (void)path;
    return true;
}

}  // namespace

int main() {
    // TODO(task 3): call print_file_size for each of the three files above
    //               (FP32 model, int8 model, tokenizer); return 0 only if all
    //               three succeed, 1 otherwise.
    return 0;
}
