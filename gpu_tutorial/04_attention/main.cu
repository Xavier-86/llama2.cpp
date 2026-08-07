// 04_attention — student template
//
// Goal: write a multi-head attention kernel with one 128-thread block per head.
// Put all three stages in one kernel: q·k scores → softmax → weighted sum of v.
// KV-cache positions after pos contain random data, so the kernel must read only
// [0, pos]. Read the Concept section in README.md before starting.
//
// Build:  nvcc -O2 -arch=sm_89 -o main main.cu
// Run:    ./main
// Verify: python3 ../../cpu_tutorial/tools/compare.py out_xb_a.txt  data/expected_xb_a.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_att_a.txt data/expected_att_a.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_xb_b.txt  data/expected_xb_b.txt
//         python3 ../../cpu_tutorial/tools/compare.py out_att_b.txt data/expected_att_b.txt

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

// Provided results from modules 00 and 01.
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

// TODO(task 1): implement attention with one 128-thread block per head and
//   separate its three stages with __syncthreads():
//   1) loop t = tid; t <= pos; t += blockDim.x, compute q·k / sqrt(head_size),
//      and write att_h[t];
//   2) one thread (tid==0) applies softmax to att_h[0..pos], subtracting max first;
//   3) thread i computes output component i:
//      xb[h*head_size+i] = Σ_t att_h[t] * v[t][i].
//   Select the KV head with h / kv_mul for GQA. layer_off is zero here but required.
__global__ void attention_kernel(float* xb, const float* q,
                                 const float* key_cache, const float* value_cache,
                                 float* att, int pos,
                                 int kvd, int kv_mul,
                                 int head_size, int seq_len, size_t layer_off) {
    // stub: replace with your code
    (void)xb; (void)q; (void)key_cache; (void)value_cache; (void)att;
    (void)pos; (void)kvd; (void)kv_mul; (void)head_size; (void)seq_len; (void)layer_off;
}

// Run one case: upload inputs, launch the kernel, and write out_xb_<tag>.txt and
// out_att_<tag>.txt. att has shape (n_heads, seq_len); output concatenates only
// [0..pos] for each head, matching the CPU reference layout in cases.h.
void run_case(const char* tag, int n_heads, int kv_mul, int head_size,
              int seq_len, int pos, int kvd,
              const std::vector<float>& q, const std::vector<float>& k,
              const std::vector<float>& v) {
    float* dq = upload(q.data(), q.size());
    float* dk = upload(k.data(), k.size());
    float* dv = upload(v.data(), v.size());
    float* dxb = alloc_zeros((size_t)n_heads * head_size);
    float* datt = alloc_zeros((size_t)n_heads * seq_len);

    attention_kernel<<<n_heads, 128>>>(dxb, dq, dk, dv, datt, pos, kvd, kv_mul,
                                       head_size, seq_len, /*layer_off=*/0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    tut::write_floats(std::string("out_xb_") + tag + ".txt",
                      download(dxb, (size_t)n_heads * head_size));

    const std::vector<float> att_all = download(datt, (size_t)n_heads * seq_len);
    std::vector<float> att_out((size_t)n_heads * (pos + 1));
    for (int h = 0; h < n_heads; h++)
        for (int t = 0; t <= pos; t++)
            att_out[(size_t)h * (pos + 1) + t] = att_all[(size_t)h * seq_len + t];
    tut::write_floats(std::string("out_att_") + tag + ".txt", att_out);

    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dxb); cudaFree(datt);
}

int main() {
    // Case A: MHA（n_heads=2, kv_mul=1, head_size=4, seq_len=8, pos=3）
    run_case("a", gt::kANHeads, gt::kAKVMul, gt::kAHeadSize, gt::kASeqLen,
             gt::kAPos, gt::kAKvd, gt::make_a_q(), gt::make_a_k(), gt::make_a_v());

    // Case B: GQA（n_heads=4, n_kv_heads=2, kv_mul=2, head_size=4, seq_len=8, pos=5）
    run_case("b", gt::kBNHeads, gt::kBKVMul, gt::kBHeadSize, gt::kBSeqLen,
             gt::kBPos, gt::kBKvd, gt::make_b_q(), gt::make_b_k(), gt::make_b_v());

    std::printf("wrote out_xb_a.txt out_att_a.txt out_xb_b.txt out_att_b.txt\n");
    return 0;
}
