// 02_rmsnorm — student template
//
// Goal: write the first reduction kernel:
// o[i] = w[i] * x[i] / sqrt(mean(x²) + eps). One block handles the full vector
// and performs a tree reduction in shared memory. Read README.md first; it contains
// reference kernel code. eps must match cpu/run.cpp:198: 1e-5f.
//
// Build:  nvcc -O2 -arch=sm_89 -o main main.cu
// Run:    ./main
// Verify: python3 ../../cpu_tutorial/tools/compare.py out.txt      data/expected.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_real.txt data/expected_real.txt

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

// TODO(task 1): implement the RMSNorm kernel. One block handles the full vector:
//   1) each thread uses a strided loop to accumulate x² into shared memory;
//   2) a tree reduction reduces the sum into sdata[0] (__syncthreads() each round);
//   3) compute scale = rsqrtf(sdata[0] / n + 1e-5f), then write with a strided loop.
//      o[i] = weight[i] * x[i] * scale。
//   Use the fixed launch configuration <<<1, 256>>> (grid size is always 1).
__global__ void rmsnorm_kernel(float* o, const float* x, const float* weight, int n) {
    // stub: replace with your code
    (void)o; (void)x; (void)weight; (void)n;
}

int main() {
    // Case 1: toy 8-element vector -> out.txt
    {
        float* dx = upload(gt::kToyVec, gt::kToySize);
        float* dw = upload(gt::kToyWeight, gt::kToySize);
        float* do_ = alloc_zeros(gt::kToySize);
        rmsnorm_kernel<<<1, 256>>>(do_, dx, dw, gt::kToySize);
        CUDA_CHECK(cudaGetLastError());
        tut::write_floats("out.txt", download(do_, gt::kToySize));
        cudaFree(dx); cudaFree(dw); cudaFree(do_);
    }

    // Case 2: real-size 288-element vector -> out_real.txt
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
