// Shared implementation for the three GPU int8 entry points. Compile one of:
//   runqgpu.cu, ../4080s/runqgpu.cu, ../ppu/runqgpu.cu
//
// Structure: include cpu/runq.cpp to reuse all host code (quantized checkpoint
// loading, Tokenizer, Sampler, and CLI). The GPU-specific code uploads the int8
// weights and fp32 activation buffers, and implements the forward pass with
// three int8 matmul kernels, selected at compile time by each entry point:
//
//   naive  — a separate quantize_kernel packs the activation to int8 before
//            every matmul; qmatmul_naive_kernel then runs one block per output
//            row with byte-wise loads and int32 group accumulation.
//   fused  — qmatmul_kernel is a warp-per-row GEMV: float4 / int8x4 vectorized
//            loads, on-the-fly activation quantization via segmented warp
//            reductions, __dp4a products, no separate quantize launch.
//   ppu    — quantize an activation once and reuse it, then run a vectorized
//            warp-per-row __dp4a GEMV. Q/K/V and W1/W3 are issued as one
//            kernel each to reduce launch overhead on Zhenwu 810E.
//
// Build an entry point from one of the three target directories; see README.md.
// Run:    ./runqgpu models/stories15M-q32.bin -t 0.0 -n 128 -s 42 -i "Once upon a time"

#include <cuda_runtime.h>

// Reuse host code from cpu/runq.cpp after renaming main to avoid a conflict:
// Config, QuantizedTensor, MappedFile, Transformer (for mmap weight loading),
// Tokenizer, Sampler, error_usage, and related helpers.
#define main llama2_cpu_cli_main
#include "../../cpu/runq.cpp"
#undef main

// ----------------------------------------------------------------------------
// CUDA plumbing (same helpers as rungpu.cu).

#define CUDA_CHECK(call) do {                                            \
    cudaError_t err_ = (call);                                           \
    if (err_ != cudaSuccess)                                             \
        throw std::runtime_error(std::string("CUDA error: ") +           \
            cudaGetErrorString(err_) + " at " + __FILE__ +               \
            ":" + std::to_string(__LINE__));                             \
} while (0)

template <typename T>
T* upload(std::span<const T> src) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, src.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, src.data(), src.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

float* alloc_zeros(size_t n_floats) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n_floats * 4));
    CUDA_CHECK(cudaMemset(d, 0, n_floats * 4));
    return d;
}

// ----------------------------------------------------------------------------
// Elementwise/attention kernels, identical to rungpu.cu (FP32 activations).

__global__ void rmsnorm_kernel(float* o, const float* x, const float* weight, int n) {
    __shared__ float sdata[256];
    const int tid = threadIdx.x;
    float acc = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) acc += x[i] * x[i];
    sdata[tid] = acc;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] += sdata[tid + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(sdata[0] / n + 1e-5f);
    for (int i = tid; i < n; i += blockDim.x) o[i] = weight[i] * x[i] * scale;
}

__global__ void rope_kernel(float* q, float* k, int pos, int dim, int kvd, int head_size) {
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i >= dim) return;
    const int head_dim = i % head_size;
    const float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
    const float val = pos * freq;
    const float fcr = cosf(val), fci = sinf(val);
    const float q0 = q[i], q1 = q[i + 1];
    q[i]     = q0 * fcr - q1 * fci;
    q[i + 1] = q0 * fci + q1 * fcr;
    if (i < kvd) {
        const float k0 = k[i], k1 = k[i + 1];
        k[i]     = k0 * fcr - k1 * fci;
        k[i + 1] = k0 * fci + k1 * fcr;
    }
}

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
    const int kv_head = h / kv_mul;

    for (int t = tid; t <= pos; t += blockDim.x) {
        const float* key = key_cache + layer_off + (size_t)t * kvd + (size_t)kv_head * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) score += qh[i] * key[i];
        att_h[t] = score * inv_sqrt_hs;
    }
    __syncthreads();

    if (tid == 0) {
        float mx = att_h[0];
        for (int t = 1; t <= pos; t++) mx = fmaxf(mx, att_h[t]);
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) { att_h[t] = expf(att_h[t] - mx); sum += att_h[t]; }
        for (int t = 0; t <= pos; t++) att_h[t] /= sum;
    }
    __syncthreads();

    for (int i = tid; i < head_size; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const float* val = value_cache + layer_off + (size_t)t * kvd + (size_t)kv_head * head_size;
            acc += att_h[t] * val[i];
        }
        xb[(size_t)h * head_size + i] = acc;
    }
}

__global__ void swiglu_kernel(float* hb, const float* hb2, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = hb[i];
    hb[i] = v / (1.0f + expf(-v)) * hb2[i];
}

__global__ void add_kernel(float* x, const float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void embed_kernel(float* x, const float* table, int token, int dim) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < dim) x[i] = table[(size_t)token * dim + i];
}

// ----------------------------------------------------------------------------
// int8 quantization kernels.

// Naive path step 1: pack an fp32 activation to int8, one warp per GS-element
// group (scale = max abs value in the group, symmetric to [-127, 127]).
__global__ void quantize_kernel(const float* x, int8_t* xq, float* xs, int n, int GS) {
    const int group = blockIdx.x;
    const int lane = threadIdx.x; // blockDim.x == 32
    const int base = group * GS;
    float wmax = 0.0f;
    for (int i = lane; i < GS; i += 32) wmax = fmaxf(wmax, fabsf(x[base + i]));
    for (int off = 16; off > 0; off >>= 1)
        wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffffu, wmax, off));
    const float scale = wmax / 127.0f;
    if (lane == 0) xs[group] = scale;
    for (int i = lane; i < GS; i += 32)
        xq[base + i] = static_cast<int8_t>(__float2int_rn(x[base + i] / scale));
}

// Naive path step 2: one block per output row, byte-wise int8 loads, int32
// accumulation per GS-element group (one warp per group), float rescale at the
// end of each group. y = W x with W (d,n) row-major, x (n,).
__global__ void qmatmul_naive_kernel(float* y, const int8_t* xq, const float* xs,
                                     const int8_t* wq, const float* ws,
                                     int n, int d, int GS) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const int nwarps = blockDim.x / 32;
    const size_t row_base = (size_t)row * n;
    const int ngroups = n / GS;
    __shared__ float partial[32]; // one slot per warp (<= 1024 threads)

    float val = 0.0f;
    for (int g = warp; g < ngroups; g += nwarps) {
        std::int32_t ival = 0;
        for (int k = lane; k < GS; k += 32) {
            ival += static_cast<std::int32_t>(xq[g * GS + k]) *
                    static_cast<std::int32_t>(wq[row_base + (size_t)g * GS + k]);
        }
        for (int off = 16; off > 0; off >>= 1)
            ival += __shfl_xor_sync(0xffffffffu, ival, off);
        if (lane == 0)
            val += static_cast<float>(ival) * ws[row_base / GS + g] * xs[g];
    }
    if (lane == 0) partial[warp] = val;
    __syncthreads();
    if (threadIdx.x == 0) {
        float sum = 0.0f;
        for (int w = 0; w < nwarps; w++) sum += partial[w];
        y[row] = sum;
    }
}

// Fused path: warp-per-row GEMV with on-the-fly activation quantization.
// Each iteration a warp covers 4 consecutive GS groups (128 elements for
// GS=32): every lane owns 4 consecutive elements via one float4 load, computes
// its group's scale with a segmented shuffle reduction over GS/4 lanes, packs
// its 4 quantized values into one int, multiplies with one int8x4 weight load
// via __dp4a, and rescales per group. No separate quantize launch, no int8
// activation buffer. Requires GS to be a multiple of 4 with GS/4 a power of
// two <= 32 (GS=32 for all stories checkpoints).
__global__ void qmatmul_kernel(float* y, const float* x,
                               const int8_t* wq, const float* ws,
                               int n, int d, int GS) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= d) return;

    const int lpg = GS / 4;          // lanes per group
    const int gpg = 32 / lpg;        // groups per warp iteration
    const int ngroups = n / GS;
    const size_t row_base = (size_t)row * n;

    const int group = lane / lpg;    // this lane's group within the iteration
    const int elem = (lane % lpg) * 4; // this lane's offset within the group

    float val = 0.0f;
    for (int g0 = 0; g0 < ngroups; g0 += gpg) {
        const int g = g0 + group;
        const bool active = g < ngroups;
        const size_t off = (size_t)g * GS + elem;

        float4 xv = active ? *reinterpret_cast<const float4*>(x + off)
                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float wmax = fmaxf(fmaxf(fabsf(xv.x), fabsf(xv.y)),
                           fmaxf(fabsf(xv.z), fabsf(xv.w)));
        for (int off2 = lpg / 2; off2 > 0; off2 >>= 1)
            wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffffu, wmax, off2));
        const float scale = wmax / 127.0f;

        const int q0 = __float2int_rn(xv.x / scale);
        const int q1 = __float2int_rn(xv.y / scale);
        const int q2 = __float2int_rn(xv.z / scale);
        const int q3 = __float2int_rn(xv.w / scale);
        const int qpack = (q0 & 0xff) | ((q1 & 0xff) << 8) |
                          ((q2 & 0xff) << 16) | ((q3 & 0xff) << 24);
        const int wpack = active
            ? *reinterpret_cast<const int*>(wq + row_base + off) : 0;
        std::int32_t ival = __dp4a(qpack, wpack, 0);
        for (int off2 = lpg / 2; off2 > 0; off2 >>= 1)
            ival += __shfl_xor_sync(0xffffffffu, ival, off2);
        if (active && lane % lpg == 0)
            val += static_cast<float>(ival) * ws[row_base / GS + g] * scale;
    }

    // lanes 0, lpg, 2*lpg, ... hold the group contributions; reduce the warp.
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, off);
    if (lane == 0) y[row] = val;
}

// PPU path step 1: a warp quantizes four GS=32 groups at once. Each 8-lane
// subgroup uses float4 loads/stores, so all 32 lanes do useful work and one
// 256-thread block covers 32 groups. This replaces the one-block-per-group
// layout above, which creates many tiny blocks on Zhenwu 810E.
__global__ void quantize_ppu_kernel(const float* x, int8_t* xq, float* xs,
                                    int ngroups) {
    const int lane = threadIdx.x % 32;
    const int warp = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    const int subgroup = lane / 8;
    const int pack = lane % 8;
    const int group = warp * 4 + subgroup;

    float4 xv = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (group < ngroups) {
        xv = *reinterpret_cast<const float4*>(x + (size_t)group * 32 + pack * 4);
    }
    float wmax = fmaxf(fmaxf(fabsf(xv.x), fabsf(xv.y)),
                       fmaxf(fabsf(xv.z), fabsf(xv.w)));
    for (int off = 4; off > 0; off >>= 1)
        wmax = fmaxf(wmax, __shfl_xor_sync(0xffffffffu, wmax, off));

    if (group < ngroups) {
        const float scale = wmax / 127.0f;
        const float inv_scale = wmax > 0.0f ? 127.0f / wmax : 0.0f;
        const int q0 = __float2int_rn(xv.x * inv_scale);
        const int q1 = __float2int_rn(xv.y * inv_scale);
        const int q2 = __float2int_rn(xv.z * inv_scale);
        const int q3 = __float2int_rn(xv.w * inv_scale);
        const int qpack = (q0 & 0xff) | ((q1 & 0xff) << 8) |
                          ((q2 & 0xff) << 16) | ((q3 & 0xff) << 24);
        *reinterpret_cast<int*>(xq + (size_t)group * 32 + pack * 4) = qpack;
        if (pack == 0) xs[group] = scale;
    }
}

// PPU path step 2: one warp computes one row from an already-quantized
// activation. Four 8-lane subgroups consume four quantization groups in
// parallel; int8x4 loads are coalesced and __dp4a performs four products per
// instruction. Compared with qmatmul_kernel this removes fp32 loads,
// quantization and max reductions from every output row.
__device__ __forceinline__ float qdot_ppu_row(const int8_t* xq, const float* xs,
                                               const int8_t* wq, const float* ws,
                                               int n, int row) {
    const int lane = threadIdx.x % 32;
    const int subgroup = lane / 8;
    const int pack = lane % 8;
    const int ngroups = n / 32;
    const size_t row_base = (size_t)row * n;
    float val = 0.0f;

    for (int g0 = 0; g0 < ngroups; g0 += 4) {
        const int group = g0 + subgroup;
        std::int32_t ival = 0;
        if (group < ngroups) {
            const size_t off = (size_t)group * 32 + pack * 4;
            const int xpack = *reinterpret_cast<const int*>(xq + off);
            const int wpack = *reinterpret_cast<const int*>(wq + row_base + off);
            ival = __dp4a(xpack, wpack, 0);
        }
        for (int off = 4; off > 0; off >>= 1)
            ival += __shfl_xor_sync(0xffffffffu, ival, off);
        if (pack == 0 && group < ngroups) {
            val += static_cast<float>(ival) * xs[group] * ws[row_base / 32 + group];
        }
    }

    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_xor_sync(0xffffffffu, val, off);
    return val;
}

__global__ void qmatmul_ppu_kernel(float* y, const int8_t* xq, const float* xs,
                                    const int8_t* wq, const float* ws,
                                    int n, int d) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int row = blockIdx.x * (blockDim.x / 32) + warp;
    if (row >= d) return;
    const float val = qdot_ppu_row(xq, xs, wq, ws, n, row);
    if (lane == 0) y[row] = val;
}

// Multiple projections sharing x are presented as one virtual row range. The
// branches are warp-uniform and save two launches for Q/K/V and one for W1/W3.
__global__ void qmatmul_ppu_2_kernel(
        float* y0, const int8_t* wq0, const float* ws0, int d0,
        float* y1, const int8_t* wq1, const float* ws1, int d1,
        const int8_t* xq, const float* xs, int n) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int virtual_row = blockIdx.x * (blockDim.x / 32) + warp;
    if (virtual_row >= d0 + d1) return;
    const bool second = virtual_row >= d0;
    const int row = second ? virtual_row - d0 : virtual_row;
    const float val = second ? qdot_ppu_row(xq, xs, wq1, ws1, n, row)
                             : qdot_ppu_row(xq, xs, wq0, ws0, n, row);
    if (lane == 0) (second ? y1 : y0)[row] = val;
}

__global__ void qmatmul_ppu_3_kernel(
        float* y0, const int8_t* wq0, const float* ws0, int d0,
        float* y1, const int8_t* wq1, const float* ws1, int d1,
        float* y2, const int8_t* wq2, const float* ws2, int d2,
        const int8_t* xq, const float* xs, int n) {
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int virtual_row = blockIdx.x * (blockDim.x / 32) + warp;
    if (virtual_row >= d0 + d1 + d2) return;

    float* y = y0;
    const int8_t* wq = wq0;
    const float* ws = ws0;
    int row = virtual_row;
    if (virtual_row >= d0 + d1) {
        y = y2; wq = wq2; ws = ws2; row = virtual_row - d0 - d1;
    } else if (virtual_row >= d0) {
        y = y1; wq = wq1; ws = ws1; row = virtual_row - d0;
    }
    const float val = qdot_ppu_row(xq, xs, wq, ws, n, row);
    if (lane == 0) y[row] = val;
}

// ----------------------------------------------------------------------------
// GPU Transformer: CPU Transformer handles mmap loading; this class handles
// upload and forward. KernelKind selects the device-specific int8 path.

enum class KernelKind { Naive, Fused, Ppu };

#ifndef RUNQGPU_KERNEL_KIND
#define RUNQGPU_KERNEL_KIND KernelKind::Fused
#endif

struct DeviceQuantizedTensor {
    int8_t* q = nullptr;
    float* s = nullptr;
};

struct DeviceQWeights {
    float* token_embedding_table = nullptr; // dequantized fp32 copy (lookup only)
    float *rms_att_weight = nullptr, *rms_ffn_weight = nullptr, *rms_final_weight = nullptr;
    std::vector<DeviceQuantizedTensor> wq, wk, wv, wo, w1, w2, w3;
    DeviceQuantizedTensor wcls;
    bool wcls_shared = false; // wcls.q aliases q_tokens' device buffer
    DeviceQuantizedTensor q_tokens;
};

struct DeviceQState {
    float *x, *xb, *xb2, *q, *hb, *hb2, *att, *logits, *key_cache, *value_cache;
    int8_t *xq_q, *hq_q;   // int8 activation buffers (naive and PPU paths)
    float *xq_s, *hq_s;
};

class GpuQTransformer {
public:
    explicit GpuQTransformer(const std::string& checkpoint_path,
                             KernelKind kernel_kind = KernelKind::Fused)
        : cpu_(checkpoint_path), kernel_kind_(kernel_kind) {
        const Config& p = cpu_.config;
        config = cpu_.config;
        GS = cpu_.GS;
        if (kernel_kind_ == KernelKind::Fused &&
            (GS % 4 != 0 || (GS / 4 & (GS / 4 - 1)) != 0 || GS / 4 > 32)) {
            throw std::runtime_error("qmatmul_kernel requires GS/4 to be a power of two <= 32");
        }
        if (kernel_kind_ == KernelKind::Ppu && GS != 32) {
            throw std::runtime_error("the PPU kernel currently requires GS=32");
        }

        const TransformerWeights& w = cpu_.weights;
        auto upload_q = [](const QuantizedTensor& t) {
            DeviceQuantizedTensor d;
            d.q = upload<int8_t>(t.q);
            d.s = upload<float>(t.s);
            return d;
        };
        dw_.token_embedding_table = upload<float>(w.token_embedding_table);
        dw_.rms_att_weight        = upload<float>(w.rms_att_weight);
        dw_.rms_ffn_weight        = upload<float>(w.rms_ffn_weight);
        dw_.rms_final_weight      = upload<float>(w.rms_final_weight);
        dw_.q_tokens              = upload_q(w.q_tokens[0]);
        for (const QuantizedTensor& t : w.wq) dw_.wq.push_back(upload_q(t));
        for (const QuantizedTensor& t : w.wk) dw_.wk.push_back(upload_q(t));
        for (const QuantizedTensor& t : w.wv) dw_.wv.push_back(upload_q(t));
        for (const QuantizedTensor& t : w.wo) dw_.wo.push_back(upload_q(t));
        for (const QuantizedTensor& t : w.w1) dw_.w1.push_back(upload_q(t));
        for (const QuantizedTensor& t : w.w2) dw_.w2.push_back(upload_q(t));
        for (const QuantizedTensor& t : w.w3) dw_.w3.push_back(upload_q(t));
        dw_.wcls_shared = w.wcls.q.data() == w.q_tokens[0].q.data();
        dw_.wcls = dw_.wcls_shared ? dw_.q_tokens : upload_q(w.wcls);

        const int kvd = kv_dim(p);
        ds_.x           = alloc_zeros(p.dim);
        ds_.xb          = alloc_zeros(p.dim);
        ds_.xb2         = alloc_zeros(p.dim);
        ds_.q           = alloc_zeros(p.dim);
        ds_.hb          = alloc_zeros(p.hidden_dim);
        ds_.hb2         = alloc_zeros(p.hidden_dim);
        ds_.att         = alloc_zeros((size_t)p.n_heads * p.seq_len);
        ds_.logits      = alloc_zeros(p.vocab_size);
        ds_.key_cache   = alloc_zeros((size_t)p.n_layers * p.seq_len * kvd);
        ds_.value_cache = alloc_zeros((size_t)p.n_layers * p.seq_len * kvd);
        CUDA_CHECK(cudaMalloc(&ds_.xq_q, p.dim));
        CUDA_CHECK(cudaMalloc(&ds_.hq_q, p.hidden_dim));
        CUDA_CHECK(cudaMalloc(&ds_.xq_s, p.dim / GS * 4));
        CUDA_CHECK(cudaMalloc(&ds_.hq_s, p.hidden_dim / GS * 4));

        host_logits_.resize(p.vocab_size);
    }

    ~GpuQTransformer() {
        auto free_q = [](DeviceQuantizedTensor& t) { cudaFree(t.q); cudaFree(t.s); };
        cudaFree(dw_.token_embedding_table);
        cudaFree(dw_.rms_att_weight); cudaFree(dw_.rms_ffn_weight);
        cudaFree(dw_.rms_final_weight);
        free_q(dw_.q_tokens);
        for (auto& v : {&dw_.wq, &dw_.wk, &dw_.wv, &dw_.wo, &dw_.w1, &dw_.w2, &dw_.w3})
            for (DeviceQuantizedTensor& t : *v) free_q(t);
        if (!dw_.wcls_shared) free_q(dw_.wcls);
        cudaFree(ds_.x); cudaFree(ds_.xb); cudaFree(ds_.xb2); cudaFree(ds_.q);
        cudaFree(ds_.hb); cudaFree(ds_.hb2); cudaFree(ds_.att); cudaFree(ds_.logits);
        cudaFree(ds_.key_cache); cudaFree(ds_.value_cache);
        cudaFree(ds_.xq_q); cudaFree(ds_.hq_q); cudaFree(ds_.xq_s); cudaFree(ds_.hq_s);
    }

    // y = W x, W (d,n) int8-quantized row-major. Naive and PPU paths quantize
    // x into (xq, xs) first; fused quantizes inside the matmul kernel.
    void qmatmul(float* y, const float* x, int8_t* xq, float* xs,
                 const DeviceQuantizedTensor& w, int n, int d) {
        if (kernel_kind_ == KernelKind::Naive) {
            quantize_kernel<<<n / GS, 32>>>(x, xq, xs, n, GS);
            qmatmul_naive_kernel<<<d, 128>>>(y, xq, xs, w.q, w.s, n, d, GS);
        } else if (kernel_kind_ == KernelKind::Fused) {
            const int warps_per_block = 8;
            qmatmul_kernel<<<(d + warps_per_block - 1) / warps_per_block,
                             warps_per_block * 32>>>(y, x, w.q, w.s, n, d, GS);
        } else {
            quantize_ppu(x, xq, xs, n);
            qmatmul_ppu(y, xq, xs, w, n, d);
        }
    }

    void quantize_ppu(const float* x, int8_t* xq, float* xs, int n) {
        constexpr int threads = 256;
        constexpr int groups_per_block = (threads / 32) * 4;
        const int ngroups = n / 32;
        quantize_ppu_kernel<<<(ngroups + groups_per_block - 1) / groups_per_block,
                              threads>>>(x, xq, xs, ngroups);
    }

    void qmatmul_ppu(float* y, const int8_t* xq, const float* xs,
                     const DeviceQuantizedTensor& w, int n, int d) {
        constexpr int warps_per_block = 8;
        qmatmul_ppu_kernel<<<(d + warps_per_block - 1) / warps_per_block,
                             warps_per_block * 32>>>(y, xq, xs, w.q, w.s, n, d);
    }

    void qmatmul_ppu_2(float* y0, const DeviceQuantizedTensor& w0, int d0,
                       float* y1, const DeviceQuantizedTensor& w1, int d1,
                       const int8_t* xq, const float* xs, int n) {
        constexpr int warps_per_block = 8;
        const int rows = d0 + d1;
        qmatmul_ppu_2_kernel<<<(rows + warps_per_block - 1) / warps_per_block,
                               warps_per_block * 32>>>(
            y0, w0.q, w0.s, d0, y1, w1.q, w1.s, d1, xq, xs, n);
    }

    void qmatmul_ppu_3(float* y0, const DeviceQuantizedTensor& w0, int d0,
                       float* y1, const DeviceQuantizedTensor& w1, int d1,
                       float* y2, const DeviceQuantizedTensor& w2, int d2,
                       const int8_t* xq, const float* xs, int n) {
        constexpr int warps_per_block = 8;
        const int rows = d0 + d1 + d2;
        qmatmul_ppu_3_kernel<<<(rows + warps_per_block - 1) / warps_per_block,
                               warps_per_block * 32>>>(
            y0, w0.q, w0.s, d0, y1, w1.q, w1.s, d1,
            y2, w2.q, w2.s, d2, xq, xs, n);
    }

    // Call sequence matching CPU Transformer::forward at cpu/runq.cpp:345 line by line.
    std::span<float> forward(int token, int pos) {
        const Config& p = config;
        const int dim = p.dim;
        const int kvd = kv_dim(p);
        const int kv_mul = p.n_heads / p.n_kv_heads;
        const int hidden_dim = p.hidden_dim;
        const int head_size = dim / p.n_heads;

        embed_kernel<<<(dim + 255) / 256, 256>>>(ds_.x, dw_.token_embedding_table, token, dim);

        for (int l = 0; l < p.n_layers; l++) {
            const size_t loff = (size_t)l * p.seq_len * kvd;
            float* k = ds_.key_cache + loff + (size_t)pos * kvd;
            float* v = ds_.value_cache + loff + (size_t)pos * kvd;

            rmsnorm_kernel<<<1, 256>>>(ds_.xb, ds_.x, dw_.rms_att_weight + (size_t)l * dim, dim);
            if (kernel_kind_ == KernelKind::Ppu) {
                quantize_ppu(ds_.xb, ds_.xq_q, ds_.xq_s, dim);
                qmatmul_ppu_3(ds_.q, dw_.wq[l], dim, k, dw_.wk[l], kvd,
                              v, dw_.wv[l], kvd, ds_.xq_q, ds_.xq_s, dim);
            } else {
                qmatmul(ds_.q, ds_.xb, ds_.xq_q, ds_.xq_s, dw_.wq[l], dim, dim);
                qmatmul(k, ds_.xb, ds_.xq_q, ds_.xq_s, dw_.wk[l], dim, kvd);
                qmatmul(v, ds_.xb, ds_.xq_q, ds_.xq_s, dw_.wv[l], dim, kvd);
            }
            rope_kernel<<<(dim / 2 + 255) / 256, 256>>>(ds_.q, k, pos, dim, kvd, head_size);
            attention_kernel<<<p.n_heads, 128>>>(ds_.xb, ds_.q, ds_.key_cache, ds_.value_cache,
                                                 ds_.att, pos, kvd, kv_mul, head_size,
                                                 p.seq_len, loff);
            if (kernel_kind_ == KernelKind::Ppu) {
                quantize_ppu(ds_.xb, ds_.xq_q, ds_.xq_s, dim);
                qmatmul_ppu(ds_.xb2, ds_.xq_q, ds_.xq_s, dw_.wo[l], dim, dim);
            } else {
                qmatmul(ds_.xb2, ds_.xb, ds_.xq_q, ds_.xq_s, dw_.wo[l], dim, dim);
            }
            add_kernel<<<(dim + 255) / 256, 256>>>(ds_.x, ds_.xb2, dim);

            rmsnorm_kernel<<<1, 256>>>(ds_.xb, ds_.x, dw_.rms_ffn_weight + (size_t)l * dim, dim);
            if (kernel_kind_ == KernelKind::Ppu) {
                quantize_ppu(ds_.xb, ds_.xq_q, ds_.xq_s, dim);
                qmatmul_ppu_2(ds_.hb, dw_.w1[l], hidden_dim,
                              ds_.hb2, dw_.w3[l], hidden_dim,
                              ds_.xq_q, ds_.xq_s, dim);
            } else {
                qmatmul(ds_.hb, ds_.xb, ds_.xq_q, ds_.xq_s, dw_.w1[l], dim, hidden_dim);
                qmatmul(ds_.hb2, ds_.xb, ds_.xq_q, ds_.xq_s, dw_.w3[l], dim, hidden_dim);
            }
            swiglu_kernel<<<(hidden_dim + 255) / 256, 256>>>(ds_.hb, ds_.hb2, hidden_dim);
            if (kernel_kind_ == KernelKind::Ppu) {
                quantize_ppu(ds_.hb, ds_.hq_q, ds_.hq_s, hidden_dim);
                qmatmul_ppu(ds_.xb, ds_.hq_q, ds_.hq_s, dw_.w2[l], hidden_dim, dim);
            } else {
                qmatmul(ds_.xb, ds_.hb, ds_.hq_q, ds_.hq_s, dw_.w2[l], hidden_dim, dim);
            }
            add_kernel<<<(dim + 255) / 256, 256>>>(ds_.x, ds_.xb, dim);
        }

        rmsnorm_kernel<<<1, 256>>>(ds_.x, ds_.x, dw_.rms_final_weight, dim);
        if (kernel_kind_ == KernelKind::Ppu) {
            quantize_ppu(ds_.x, ds_.xq_q, ds_.xq_s, dim);
            qmatmul_ppu(ds_.logits, ds_.xq_q, ds_.xq_s, dw_.wcls, dim, p.vocab_size);
        } else {
            qmatmul(ds_.logits, ds_.x, ds_.xq_q, ds_.xq_s, dw_.wcls, dim, p.vocab_size);
        }

        CUDA_CHECK(cudaMemcpy(host_logits_.data(), ds_.logits,
                              p.vocab_size * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());
        return host_logits_;
    }

    Config config;
    int GS = 0;

private:
    Transformer cpu_;
    KernelKind kernel_kind_ = KernelKind::Fused;
    DeviceQWeights dw_{};
    DeviceQState ds_{};
    std::vector<float> host_logits_;
};

// ----------------------------------------------------------------------------
// generate/chat/main match cpu/runq.cpp, with Transformer replaced by
// GpuQTransformer. The public entry point fixes the kernel at compile time.

void generate(GpuQTransformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
              const std::string& prompt, int steps) {
    std::vector<int> prompt_tokens = tokenizer.encode(prompt, /*bos=*/true, /*eos=*/false);
    if (prompt_tokens.empty()) {
        throw std::runtime_error("something is wrong, expected at least 1 prompt token");
    }
    const int num_prompt_tokens = static_cast<int>(prompt_tokens.size());

    std::int64_t start = 0;
    int next;
    int token = prompt_tokens[0];
    int pos = 0;
    while (pos < steps) {
        std::span<float> logits = transformer.forward(token, pos);

        if (pos < num_prompt_tokens - 1) {
            next = prompt_tokens[pos + 1];
        } else {
            next = sampler.sample(logits);
        }
        pos++;

        if (next == 1) { break; }

        safe_print(tokenizer.decode(token, next));
        std::cout << std::flush;
        token = next;

        if (start == 0) { start = time_in_ms(); }
    }
    std::cout << '\n';

    if (pos > 1) {
        std::int64_t end = time_in_ms();
        std::cerr << "achieved tok/s: " << (pos - 1) / static_cast<double>(end - start) * 1000 << '\n';
    }
}

void chat(GpuQTransformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
          const std::string& cli_user_prompt, const std::string& cli_system_prompt, int steps) {
    std::string system_prompt;
    std::string user_prompt;
    std::string rendered_prompt;
    std::vector<int> prompt_tokens;
    size_t user_idx = 0;

    bool user_turn = true;
    int next = 0;
    int token = 0;
    int pos = 0;
    while (pos < steps) {
        if (user_turn) {
            if (pos == 0) {
                system_prompt = cli_system_prompt.empty()
                                    ? read_stdin("Enter system prompt (optional): ")
                                    : cli_system_prompt;
            }
            if (pos == 0 && !cli_user_prompt.empty()) {
                user_prompt = cli_user_prompt;
            } else {
                user_prompt = read_stdin("User: ");
            }
            if (pos == 0 && !system_prompt.empty()) {
                rendered_prompt = "[INST] <<SYS>>\n" + system_prompt + "\n<</SYS>>\n\n" + user_prompt + " [/INST]";
            } else {
                rendered_prompt = "[INST] " + user_prompt + " [/INST]";
            }
            prompt_tokens = tokenizer.encode(rendered_prompt, /*bos=*/true, /*eos=*/false);
            user_idx = 0;
            user_turn = false;
            std::cout << "Assistant: ";
        }

        if (user_idx < prompt_tokens.size()) {
            token = prompt_tokens[user_idx++];
        } else {
            token = next;
        }
        if (token == 2) { user_turn = true; }

        std::span<float> logits = transformer.forward(token, pos);
        next = sampler.sample(logits);
        pos++;

        if (user_idx >= prompt_tokens.size() && next != 2) {
            safe_print(tokenizer.decode(token, next));
            std::cout << std::flush;
        }
        if (next == 2) { std::cout << '\n'; }
    }
    std::cout << '\n';
}

#ifndef RUNQGPU_NO_MAIN
int main(int argc, char* argv[]) {
    try {
        std::string checkpoint_path;
        std::string tokenizer_path = "models/tokenizer.bin";
        float temperature = 1.0f;
        float topp = 0.9f;
        int steps = 256;
        std::string prompt;
        std::uint64_t rng_seed = 0;
        std::string mode = "generate";
        std::string system_prompt;

        if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
        for (int i = 2; i < argc; i += 2) {
            if (i + 1 >= argc) { error_usage(); }
            std::string flag = argv[i];
            if (flag.size() != 2 || flag[0] != '-') { error_usage(); }
            switch (flag[1]) {
                case 't': temperature = std::atof(argv[i + 1]); break;
                case 'p': topp = std::atof(argv[i + 1]); break;
                case 's': rng_seed = std::stoull(argv[i + 1]); break;
                case 'n': steps = std::atoi(argv[i + 1]); break;
                case 'i': prompt = argv[i + 1]; break;
                case 'z': tokenizer_path = argv[i + 1]; break;
                case 'm': mode = argv[i + 1]; break;
                case 'y': system_prompt = argv[i + 1]; break;
                default: error_usage();
            }
        }

        if (rng_seed == 0) { rng_seed = static_cast<std::uint64_t>(std::time(nullptr)); }
        if (temperature < 0.0) { temperature = 0.0; }
        if (topp < 0.0 || 1.0 < topp) { topp = 0.9; }
        if (steps < 0) { steps = 0; }

        GpuQTransformer transformer(checkpoint_path, RUNQGPU_KERNEL_KIND);
        if (steps == 0 || steps > transformer.config.seq_len) { steps = transformer.config.seq_len; }

        Tokenizer tokenizer(tokenizer_path, transformer.config.vocab_size);
        Sampler sampler(transformer.config.vocab_size, temperature, topp, rng_seed);

        if (mode == "generate") {
            generate(transformer, tokenizer, sampler, prompt, steps);
        } else if (mode == "chat") {
            chat(transformer, tokenizer, sampler, prompt, system_prompt, steps);
        } else {
            throw std::runtime_error("unknown mode: " + mode);
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
#endif
