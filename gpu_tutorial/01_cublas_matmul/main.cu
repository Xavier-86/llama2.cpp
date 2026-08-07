// 01_cublas_matmul — student template
//
// Goal: implement y = W x with cuBLAS for row-major d×n W. This operator appears
// seven times in GPU forward. The key is the row-major/column-major transpose trick;
// read the Concept section in README.md before starting.
//
// Build:  nvcc -O2 -arch=sm_89 -o main main.cu -lcublas
// Run:    ./main
// Verify: python3 ../../cpu_tutorial/tools/compare.py out_toy.txt  data/expected_toy.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_real.txt data/expected_real.txt

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

// TODO(task 1): implement y = W x with cublasSgemv. w is row-major d×n and uses
//   the same memory layout as the CPU. In the column-major view, this buffer is
//   an n×d Wᵀ; use CUBLAS_OP_T, set lda to the row-major row length, and keep
//   alpha=1 and beta=0 as host variables.
void matmul_gpu(cublasHandle_t h, float* y, const float* x, const float* w, int n, int d) {
    // stub: replace with your code
    (void)h; (void)y; (void)x; (void)w; (void)n; (void)d;
}

int main() {
    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // Case 1: toy 4×3 -> out_toy.txt
    {
        float* dw = upload(gt::kToyW, 12);
        float* dx = upload(gt::kToyX, gt::kToyN);
        float* dy = alloc_zeros(gt::kToyD);
        matmul_gpu(cublas, dy, dx, dw, gt::kToyN, gt::kToyD);
        tut::write_floats("out_toy.txt", download(dy, gt::kToyD));
        cudaFree(dw); cudaFree(dx); cudaFree(dy);
    }

    // Case 2: real-size 288→768 -> out_real.txt
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
