// 02_rmsnorm — reference solution
//
// Build:  nvcc -O2 -arch=sm_89 -o solution solution.cu
// Run:    ./solution
// Verify: python3 ../../cpu_tutorial/tools/compare.py out.txt      data/expected.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_real.txt data/expected_real.txt

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

__global__ void rmsnorm_kernel(float* o, const float* x, const float* weight, int n) {
    // One block handles the full vector; x and o both have n elements.
    __shared__ float sdata[256];
    const int tid = threadIdx.x;

    // 1) Accumulate chunks of x² into shared memory.
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) acc += x[i] * x[i];
    sdata[tid] = acc;
    __syncthreads();

    // 2) Tree reduction.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }

    // 3) Broadcast the scale and write each output element.
    const float scale = rsqrtf(sdata[0] / n + 1e-5f);   // Match the CPU epsilon.
    for (int i = tid; i < n; i += blockDim.x) o[i] = weight[i] * x[i] * scale;
}

int main() {
    {
        float* dx = upload(gt::kToyVec, gt::kToySize);
        float* dw = upload(gt::kToyWeight, gt::kToySize);
        float* do_ = alloc_zeros(gt::kToySize);
        rmsnorm_kernel<<<1, 256>>>(do_, dx, dw, gt::kToySize);
        CUDA_CHECK(cudaGetLastError());
        tut::write_floats("out.txt", download(do_, gt::kToySize));
        cudaFree(dx); cudaFree(dw); cudaFree(do_);
    }

    {
        const std::vector<float> x = gt::make_real_vec();
        const std::vector<float> w = gt::make_real_weight();
        float* dx = upload(x.data(), x.size());
        float* dw = upload(w.data(), w.size());
        float* do_ = alloc_zeros(gt::kRealSize);
        rmsnorm_kernel<<<1, 256>>>(do_, dx, dw, gt::kRealSize);
        CUDA_CHECK(cudaGetLastError());
        tut::write_floats("out_real.txt", download(do_, gt::kRealSize));
        cudaFree(dx); cudaFree(dw); cudaFree(do_);
    }

    std::printf("wrote out.txt out_real.txt\n");
    return 0;
}
