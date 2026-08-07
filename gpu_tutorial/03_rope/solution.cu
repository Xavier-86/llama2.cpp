// 03_rope — reference solution
//
// Build:  nvcc -O2 -arch=sm_89 -o solution solution.cu
// Run:    ./solution
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

float* alloc_zeros(size_t n_floats) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n_floats * 4));
    CUDA_CHECK(cudaMemset(d, 0, n_floats * 4));
    return d;
}

// One thread handles one pair: thread t owns even index i = t * 2. Treat (v0, v1)
// as a complex number and rotate by pos * freq radians; rotate k too when i < kvd.
__global__ void rope_kernel(float* q, float* k, int pos,
                            int dim, int kvd, int head_size) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2; // Even index.
    if (i >= dim) return;

    const int head_dim = i % head_size;
    const float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    const float val = pos * freq;
    const float fcr = cosf(val), fci = sinf(val);

    // Rotate q pair i.
    const float q0 = q[i], q1 = q[i + 1];
    q[i]     = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;

    // Rotate the matching k pair when i < kvd, matching the CPU rotn logic.
    if (i < kvd) {
        const float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}

int main() {
    // Case 1: toy dim=8, pos=7 -> out_q.txt / out_k.txt
    {
        float* dq = upload(gt::kToyQ, gt::kToyDim);
        float* dk = upload(gt::kToyK, gt::kToyKvd);
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
