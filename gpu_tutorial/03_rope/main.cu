// 03_rope — student template
//
// Goal: implement RoPE in a CUDA kernel. Treat adjacent q/k dimension pairs as
// complex numbers and rotate by position pos within each head; one thread handles
// one pair. Map the CPU loop index to a thread and preserve the loop body, including
// the i < kvd rotn boundary. Read README.md before starting.
//
// Build:  nvcc -O2 -arch=sm_89 -o main main.cu
// Run:    ./main
// Verify: python3 ../../cpu_tutorial/tools/compare.py out_q.txt      data/expected_q.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_k.txt      data/expected_k.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_q_real.txt data/expected_q_real.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_k_real.txt data/expected_k_real.txt

#include <cstdio>
#include <stdexcept>
#include <string>
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

// Provided result from module 00.
float* upload(const float* src, size_t n_floats) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n_floats * 4));
    CUDA_CHECK(cudaMemcpy(d, src, n_floats * 4, cudaMemcpyHostToDevice));
    return d;
}

std::vector<float> download(const float* d, size_t n_floats) {
    std::vector<float> h(n_floats);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n_floats * 4, cudaMemcpyDeviceToHost));
    return h;
}

// Allocate and zero an output buffer; the kernel overwrites every element.
float* alloc_zeros(size_t n_floats) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n_floats * 4));
    CUDA_CHECK(cudaMemset(d, 0, n_floats * 4));
    return d;
}

// TODO(task 1): RoPE kernel. One thread handles one pair: thread t handles the
//   even index i = t * 2. Compute frequency using head_dim = i % head_size, rotate
//   (q[i], q[i+1]) as a complex number by pos * freq radians, and rotate k by the
//   same angle when i < kvd. See README.md and cpu/run.cpp:262-276.
__global__ void rope_kernel(float* q, float* k, int pos,
                            int dim, int kvd, int head_size) {
    // stub: replace with your code
    (void)q; (void)k; (void)pos; (void)dim; (void)kvd; (void)head_size;
}

int main() {
    // Case 1: toy dim=8, pos=7 -> out_q.txt / out_k.txt
    {
        float* dq = upload(gt::kToyQ, gt::kToyDim);
        float* dk = upload(gt::kToyK, gt::kToyKvd);
        // Rotation is in place: the kernel modifies dq/dk without extra output buffers.
        rope_kernel<<<(gt::kToyDim / 2 + 255) / 256, 256>>>(
            dq, dk, gt::kToyPos, gt::kToyDim, gt::kToyKvd, gt::kToyHeadSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        tut::write_floats("out_q.txt", download(dq, gt::kToyDim));
        tut::write_floats("out_k.txt", download(dk, gt::kToyKvd));
        cudaFree(dq); cudaFree(dk);
    }

    // Case 2: real-size dim=288, pos=42 -> out_q_real.txt / out_k_real.txt
    {
        const std::vector<float> q = gt::make_real_q();
        const std::vector<float> k = gt::make_real_k();
        float* dq = upload(q.data(), q.size());
        float* dk = upload(k.data(), k.size());
        rope_kernel<<<(gt::kRealDim / 2 + 255) / 256, 256>>>(
            dq, dk, gt::kRealPos, gt::kRealDim, gt::kRealKvd, gt::kRealHeadSize);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        tut::write_floats("out_q_real.txt", download(dq, gt::kRealDim));
        tut::write_floats("out_k_real.txt", download(dk, gt::kRealKvd));
        cudaFree(dq); cudaFree(dk);
    }

    std::printf("wrote out_q.txt out_k.txt out_q_real.txt out_k_real.txt\n");
    return 0;
}
