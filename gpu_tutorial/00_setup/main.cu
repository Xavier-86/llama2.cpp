// 00_setup — student template
//
// Goal: upload the 12 weight tensors in the stories15M checkpoint to device memory
// for all later kernels. This module performs no inference; it only moves data to the GPU.
//
// Flow: read header → compute the size table with cases.h (write out_upload.txt) →
// read weights → upload each tensor → copy each tensor's first element back for a round trip check.
//
// Build:  nvcc -O2 -arch=sm_89 -o main main.cu
// Run:    ./main
// Verify: python3 ../../cpu_tutorial/tools/compare.py out_upload.txt data/expected_upload.txt --exact

#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>

#include "../../cpu_tutorial/common/io.h"
#include "cases.h"

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err_ = (call);                                           \
    if (err_ != cudaSuccess)                                             \
        throw std::runtime_error(std::string("CUDA error: ") +           \
            cudaGetErrorString(err_) + " at " + __FILE__ +               \
            ":" + std::to_string(__LINE__));                             \
} while (0)

// Device weight pointer table corresponding to TransformerWeights in cpu/run.cpp.
struct DeviceWeights {
    float *token_embedding_table, *rms_att_weight, *wq, *wk, *wv, *wo,
          *rms_ffn_weight, *w1, *w2, *w3, *rms_final_weight, *wcls;
};

// TODO(task 1): implement upload. Allocate src.size() floats with cudaMalloc,
//   copy src with cudaMemcpy (HostToDevice), and return the device pointer.
//   Wrap every CUDA call in CUDA_CHECK.
float* upload(const float* src, size_t n_floats) {
    // stub: replace with your code
    (void)src; (void)n_floats;
    return nullptr;
}

int main() {
    const char* ckpt_path = "../../models/stories15M.bin";
    std::ifstream f(ckpt_path, std::ios::binary);
    if (!f) { throw std::runtime_error(std::string("cannot open ") + ckpt_path); }

    // 1) Read seven int32 config values and write the size table for golden comparison.
    int cfg[7];
    f.read(reinterpret_cast<char*>(cfg), sizeof(cfg));
    const std::vector<int> table = gt::tensor_byte_table(cfg);
    tut::write_ints("out_upload.txt", table);

    // 2) Read the entire weight region.
    const std::vector<long long> sizes = gt::tensor_float_sizes(cfg);
    long long total = 0;
    for (long long n : sizes) total += n;
    std::vector<float> buf(total);
    f.read(reinterpret_cast<char*>(buf.data()), total * 4);
    if (!f) { throw std::runtime_error("checkpoint truncated"); }

    // 3) TODO(task 2): upload the 12 spans in checkpoint order and fill
    //    DeviceWeights. When the classifier is shared (wcls size is zero), set
    //    dw.wcls = dw.token_embedding_table instead of uploading it again.
    DeviceWeights dw{};
    {
        const float* p = buf.data();
        (void)p;
        // dw.token_embedding_table = upload(p, sizes[0]);  p += sizes[0];
        // ... continue in checkpoint order
    }

    // 4) Provided round-trip check: copy each device tensor's first element to
    //    the host and compare it with the original to validate direction, pointer, and size.
    const float* devs[12] = {dw.token_embedding_table, dw.rms_att_weight, dw.wq, dw.wk,
                             dw.wv, dw.wo, dw.rms_ffn_weight, dw.w1, dw.w2, dw.w3,
                             dw.rms_final_weight, dw.wcls};
    const float* p = buf.data();
    for (int t = 0; t < 11; t++) {          // wcls is shared and handled separately
        if (sizes[t] == 0) continue;
        float back = 0.0f;
        CUDA_CHECK(cudaMemcpy(&back, devs[t], 4, cudaMemcpyDeviceToHost));
        if (back != p[0]) { throw std::runtime_error("roundtrip mismatch at tensor " + std::to_string(t)); }
        p += sizes[t];
    }
    if (dw.wcls != dw.token_embedding_table) {
        throw std::runtime_error("shared classifier: wcls should alias the embedding table");
    }

    std::printf("uploaded %lld floats (%d bytes), roundtrip OK\n", total, table.back());
    return 0;
}
