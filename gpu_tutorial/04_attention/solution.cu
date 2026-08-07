// 04_attention — reference solution
//
// Build:  nvcc -O2 -arch=sm_89 -o solution solution.cu
// Run:    ./solution
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

// One block handles one head. q has dim elements; key_cache/value_cache contain
// the full-layer KV cache. att is an (n_heads, seq_len) score buffer; output goes
// to the matching head slice in xb.
__global__ void attention_kernel(float* xb, const float* q,
                                 const float* key_cache, const float* value_cache,
                                 float* att, int pos,
                                 int kvd, int kv_mul,
                                 int head_size, int seq_len, size_t layer_off) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    const float* qh = q + (size_t)h * head_size;
    float* att_h = att + (size_t)h * seq_len;
    const float inv_sqrt_hs = rsqrtf((float)head_size);
    const int kv_head = h / kv_mul;                  // Multiple query heads share one KV head in GQA/MQA.

    // 1) Each thread computes q·k / sqrt(head_size) for assigned history positions.
    for (int t = tid; t <= pos; t += blockDim.x) {
        const float* key = key_cache + layer_off + (size_t)t * kvd + (size_t)kv_head * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) score += qh[i] * key[i];
        att_h[t] = score * inv_sqrt_hs;
    }
    __syncthreads();

    // 2) Two-pass single-thread softmax; seq_len is small. Subtract max first.
    if (tid == 0) {
        float mx = att_h[0];
        for (int t = 1; t <= pos; t++) mx = fmaxf(mx, att_h[t]);
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) { att_h[t] = expf(att_h[t] - mx); sum += att_h[t]; }
        for (int t = 0; t <= pos; t++) att_h[t] /= sum;
    }
    __syncthreads();

    // 3) Weighted sum of v; thread i owns output component i.
    for (int i = tid; i < head_size; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const float* val = value_cache + layer_off + (size_t)t * kvd + (size_t)kv_head * head_size;
            acc += att_h[t] * val[i];
        }
        xb[(size_t)h * head_size + i] = acc;
    }
}

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
