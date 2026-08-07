// 00_setup — reference solution
//
// Build:  nvcc -O2 -arch=sm_89 -o solution solution.cu
// Run:    ./solution
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

struct DeviceWeights {
    float *token_embedding_table, *rms_att_weight, *wq, *wk, *wv, *wo,
          *rms_ffn_weight, *w1, *w2, *w3, *rms_final_weight, *wcls;
};

// Upload one tensor with cudaMalloc + cudaMemcpy(HostToDevice) and return its device pointer.
float* upload(const float* src, size_t n_floats) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n_floats * 4));
    CUDA_CHECK(cudaMemcpy(d, src, n_floats * 4, cudaMemcpyHostToDevice));
    return d;
}

int main() {
    const char* ckpt_path = "../../models/stories15M.bin";
    std::ifstream f(ckpt_path, std::ios::binary);
    if (!f) { throw std::runtime_error(std::string("cannot open ") + ckpt_path); }

    int cfg[7];
    f.read(reinterpret_cast<char*>(cfg), sizeof(cfg));
    const std::vector<int> table = gt::tensor_byte_table(cfg);
    tut::write_ints("out_upload.txt", table);

    const std::vector<long long> sizes = gt::tensor_float_sizes(cfg);
    long long total = 0;
    for (long long n : sizes) total += n;
    std::vector<float> buf(total);
    f.read(reinterpret_cast<char*>(buf.data()), total * 4);
    if (!f) { throw std::runtime_error("checkpoint truncated"); }

    // Upload tensors in checkpoint order; make wcls an alias when the classifier is shared.
    DeviceWeights dw{};
    {
        const float* p = buf.data();
        auto next = [&p](size_t n) { const float* r = p; p += n; return r; };
        dw.token_embedding_table = upload(next(sizes[0]), sizes[0]);
        dw.rms_att_weight        = upload(next(sizes[1]), sizes[1]);
        dw.wq                    = upload(next(sizes[2]), sizes[2]);
        dw.wk                    = upload(next(sizes[3]), sizes[3]);
        dw.wv                    = upload(next(sizes[4]), sizes[4]);
        dw.wo                    = upload(next(sizes[5]), sizes[5]);
        dw.rms_ffn_weight        = upload(next(sizes[6]), sizes[6]);
        dw.w1                    = upload(next(sizes[7]), sizes[7]);
        dw.w2                    = upload(next(sizes[8]), sizes[8]);
        dw.w3                    = upload(next(sizes[9]), sizes[9]);
        dw.rms_final_weight      = upload(next(sizes[10]), sizes[10]);
        dw.wcls = sizes[11] == 0 ? dw.token_embedding_table : upload(next(sizes[11]), sizes[11]);
    }

    const float* devs[12] = {dw.token_embedding_table, dw.rms_att_weight, dw.wq, dw.wk,
                             dw.wv, dw.wo, dw.rms_ffn_weight, dw.w1, dw.w2, dw.w3,
                             dw.rms_final_weight, dw.wcls};
    const float* p = buf.data();
    for (int t = 0; t < 11; t++) {
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
