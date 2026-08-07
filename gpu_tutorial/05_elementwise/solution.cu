// 05_elementwise — reference solution
//
// Build:  nvcc -O2 -arch=sm_89 -o solution solution.cu
// Run:    ./solution
// Verify: python3 ../../cpu_tutorial/tools/compare.py out_swiglu.txt data/expected_swiglu.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_add.txt    data/expected_add.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_embed.txt  data/expected_embed.txt

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

// SwiGLU：hb[i] = silu(hb[i]) * hb2[i]，silu(x) = x / (1 + e^-x)
__global__ void swiglu_kernel(float* hb, const float* hb2, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = hb[i];
    hb[i] = v / (1.0f + expf(-v)) * hb2[i];
}

// Residual connection: x[i] += y[i].
__global__ void add_kernel(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

// Embedding lookup: copy row token into x (cudaMemcpyDeviceToDevice also works).
__global__ void embed_kernel(float* x, const float* table, int token, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) x[i] = table[(size_t)token * dim + i];
}

int main() {
    {
        float* dhb  = upload(gt::kSwigluHb, gt::kSwigluN);
        float* dhb2 = upload(gt::kSwigluHb2, gt::kSwigluN);
        swiglu_kernel<<<(gt::kSwigluN + 255) / 256, 256>>>(dhb, dhb2, gt::kSwigluN);
        CUDA_CHECK(cudaGetLastError());
        tut::write_floats("out_swiglu.txt", download(dhb, gt::kSwigluN));
        cudaFree(dhb); cudaFree(dhb2);
    }

    {
        float* dx = upload(gt::kAddX, gt::kAddN);
        float* dy = upload(gt::kAddY, gt::kAddN);
        add_kernel<<<(gt::kAddN + 255) / 256, 256>>>(dx, dy, gt::kAddN);
        CUDA_CHECK(cudaGetLastError());
        tut::write_floats("out_add.txt", download(dx, gt::kAddN));
        cudaFree(dx); cudaFree(dy);
    }

    {
        float* dtable = upload(gt::kEmbedTable, gt::kEmbedVocab * gt::kEmbedDim);
        float* dx = alloc_zeros(gt::kEmbedDim);
        embed_kernel<<<(gt::kEmbedDim + 255) / 256, 256>>>(dx, dtable, gt::kEmbedToken, gt::kEmbedDim);
        CUDA_CHECK(cudaGetLastError());
        tut::write_floats("out_embed.txt", download(dx, gt::kEmbedDim));
        cudaFree(dtable); cudaFree(dx);
    }

    std::printf("wrote out_swiglu.txt out_add.txt out_embed.txt\n");
    return 0;
}
