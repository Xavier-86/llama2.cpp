// 01_cublas_matmul — reference solution
//
// Build:  nvcc -O2 -arch=sm_89 -o solution solution.cu -lcublas
// Run:    ./solution
// Verify: python3 ../../cpu_tutorial/tools/compare.py out_toy.txt  data/expected_toy.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_real.txt data/expected_real.txt

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../../cpu_tutorial/common/io.h"
#include "cases.h"

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err_ = (call);                                           \
    if (err_ != cudaSuccess)                                             \
        throw std::runtime_error(std::string("CUDA error: ") +           \
            cudaGetErrorString(err_) + " at " + __FILE__ +               \
            ":" + std::to_string(__LINE__));                             \
} while (0)

#define CUBLAS_CHECK(call) do {                                          \
    cublasStatus_t st_ = (call);                                         \
    if (st_ != CUBLAS_STATUS_SUCCESS)                                    \
        throw std::runtime_error("cuBLAS error " + std::to_string(st_) + \
            " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
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

// y = W x; w is row-major d×n. In the column-major view the same buffer is an n×d
// Wᵀ, so OP_T gives y = (Wᵀ)ᵀx = Wx. lda is the column stride of the column-major
// array, which equals the row-major row length n.
void matmul_gpu(cublasHandle_t h, float* y, const float* x, const float* w, int n, int d) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemv(h, CUBLAS_OP_T, n, d, &alpha, w, n, x, 1, &beta, y, 1));
}

int main() {
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    {
        float* dw = upload(gt::kToyW, 12);
        float* dx = upload(gt::kToyX, gt::kToyN);
        float* dy = alloc_zeros(gt::kToyD);
        matmul_gpu(cublas, dy, dx, dw, gt::kToyN, gt::kToyD);
        tut::write_floats("out_toy.txt", download(dy, gt::kToyD));
        cudaFree(dw); cudaFree(dx); cudaFree(dy);
    }

    {
        const std::vector<float> w = gt::make_real_w();
        const std::vector<float> x = gt::make_real_x();
        float* dw = upload(w.data(), w.size());
        float* dx = upload(x.data(), x.size());
        float* dy = alloc_zeros(gt::kRealD);
        matmul_gpu(cublas, dy, dx, dw, gt::kRealN, gt::kRealD);
        tut::write_floats("out_real.txt", download(dy, gt::kRealD));
        cudaFree(dw); cudaFree(dx); cudaFree(dy);
    }

    CUBLAS_CHECK(cublasDestroy(cublas));
    std::printf("wrote out_toy.txt out_real.txt\n");
    return 0;
}
