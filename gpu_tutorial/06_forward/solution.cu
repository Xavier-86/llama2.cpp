// 06_forward — reference solution: complete GPU FP32 inference (the finished rungpu.cu)
//
// Structure: include cpu/run.cpp to reuse all host code (checkpoint loading,
// Tokenizer, Sampler, and CLI). The GPU-specific code only uploads weights and
// activations, defines six kernels plus cuBLAS GEMV, and implements
// GpuTransformer::forward. The same -t 0.0 -s 42 should produce identical text.
//
// Build:  nvcc -O3 -std=c++20 -arch=sm_89 -o solution solution.cu -lcublas
// Run:    ./solution ../../../models/stories15M.bin -z ../../../models/tokenizer.bin \
//              -t 0.0 -n 128 -s 42 -i "Once upon a time" 2>/dev/null > out_gen.txt
// Verify: python3 ../../cpu_tutorial/tools/compare.py out_gen.txt data/expected_gen.txt --text

#include <cuda_runtime.h>
#include <cublas_v2.h>

// Reuse host code from cpu/run.cpp after renaming main to avoid a conflict:
// Config, MappedFile, Transformer (for mmap weight loading), Tokenizer, Sampler,
// error_usage, and related helpers.
#define main llama2_cpu_cli_main
#include "../../cpu/run.cpp"
#undef main

// ----------------------------------------------------------------------------
// CUDA/cuBLAS plumbing from module 00.

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

float* upload(std::span<const float> src) { return upload(src.data(), src.size()); }

float* alloc_zeros(size_t n_floats) {
    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, n_floats * 4));
    CUDA_CHECK(cudaMemset(d, 0, n_floats * 4));
    return d;
}

// ----------------------------------------------------------------------------
// Kernels from modules 02–05, matching each module's solution.cu.

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

// y = W x from module 01: w is row-major d×n; the column-major view is Wᵀ, so use OP_T.
void matmul_gpu(cublasHandle_t h, float* y, const float* x, const float* w, int n, int d) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSgemv(h, CUBLAS_OP_T, n, d, &alpha, w, n, x, 1, &beta, y, 1));
}

// ----------------------------------------------------------------------------
// GPU Transformer: CPU Transformer handles mmap loading; this class handles upload and forward.

struct DeviceWeights {
    float *token_embedding_table, *rms_att_weight, *wq, *wk, *wv, *wo,
          *rms_ffn_weight, *w1, *w2, *w3, *rms_final_weight, *wcls;
};

struct DeviceState {
    float *x, *xb, *xb2, *q, *hb, *hb2, *att, *logits, *key_cache, *value_cache;
};

class GpuTransformer {
public:
    explicit GpuTransformer(const std::string& checkpoint_path) : cpu_(checkpoint_path) {
        const Config& p = cpu_.config;
        config = cpu_.config;
        CUBLAS_CHECK(cublasCreate(&cublas_));

        // Upload weights; alias wcls instead of re-uploading a shared classifier.
        const TransformerWeights& w = cpu_.weights;
        dw_.token_embedding_table = upload(w.token_embedding_table);
        dw_.rms_att_weight        = upload(w.rms_att_weight);
        dw_.wq                    = upload(w.wq);
        dw_.wk                    = upload(w.wk);
        dw_.wv                    = upload(w.wv);
        dw_.wo                    = upload(w.wo);
        dw_.rms_ffn_weight        = upload(w.rms_ffn_weight);
        dw_.w1                    = upload(w.w1);
        dw_.w2                    = upload(w.w2);
        dw_.w3                    = upload(w.w3);
        dw_.rms_final_weight      = upload(w.rms_final_weight);
        dw_.wcls = w.wcls.data() == w.token_embedding_table.data()
                       ? dw_.token_embedding_table
                       : upload(w.wcls);

        // Activation buffers remain in device memory with the same sizes as RunState.
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

        host_logits_.resize(p.vocab_size);
    }

    ~GpuTransformer() {
        cublasDestroy(cublas_);
        cudaFree(dw_.token_embedding_table); cudaFree(dw_.rms_att_weight);
        cudaFree(dw_.wq); cudaFree(dw_.wk); cudaFree(dw_.wv); cudaFree(dw_.wo);
        cudaFree(dw_.rms_ffn_weight); cudaFree(dw_.w1); cudaFree(dw_.w2); cudaFree(dw_.w3);
        cudaFree(dw_.rms_final_weight);
        if (dw_.wcls != dw_.token_embedding_table) { cudaFree(dw_.wcls); }
        cudaFree(ds_.x); cudaFree(ds_.xb); cudaFree(ds_.xb2); cudaFree(ds_.q);
        cudaFree(ds_.hb); cudaFree(ds_.hb2); cudaFree(ds_.att); cudaFree(ds_.logits);
        cudaFree(ds_.key_cache); cudaFree(ds_.value_cache);
    }

    // Call sequence matching CPU Transformer::forward at cpu/run.cpp:232 line by line.
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
            // Current cache slices: qkv projections write directly into the cache.
            float* k = ds_.key_cache + loff + (size_t)pos * kvd;
            float* v = ds_.value_cache + loff + (size_t)pos * kvd;

            rmsnorm_kernel<<<1, 256>>>(ds_.xb, ds_.x, dw_.rms_att_weight + (size_t)l * dim, dim);
            matmul_gpu(cublas_, ds_.q, ds_.xb, dw_.wq + (size_t)l * dim * dim, dim, dim);
            matmul_gpu(cublas_, k, ds_.xb, dw_.wk + (size_t)l * dim * kvd, dim, kvd);
            matmul_gpu(cublas_, v, ds_.xb, dw_.wv + (size_t)l * dim * kvd, dim, kvd);
            rope_kernel<<<(dim / 2 + 255) / 256, 256>>>(ds_.q, k, pos, dim, kvd, head_size);
            attention_kernel<<<p.n_heads, 128>>>(ds_.xb, ds_.q, ds_.key_cache, ds_.value_cache,
                                                 ds_.att, pos, kvd, kv_mul, head_size,
                                                 p.seq_len, loff);
            matmul_gpu(cublas_, ds_.xb2, ds_.xb, dw_.wo + (size_t)l * dim * dim, dim, dim);
            add_kernel<<<(dim + 255) / 256, 256>>>(ds_.x, ds_.xb2, dim);

            rmsnorm_kernel<<<1, 256>>>(ds_.xb, ds_.x, dw_.rms_ffn_weight + (size_t)l * dim, dim);
            matmul_gpu(cublas_, ds_.hb, ds_.xb, dw_.w1 + (size_t)l * dim * hidden_dim, dim, hidden_dim);
            matmul_gpu(cublas_, ds_.hb2, ds_.xb, dw_.w3 + (size_t)l * dim * hidden_dim, dim, hidden_dim);
            swiglu_kernel<<<(hidden_dim + 255) / 256, 256>>>(ds_.hb, ds_.hb2, hidden_dim);
            matmul_gpu(cublas_, ds_.xb, ds_.hb, dw_.w2 + (size_t)l * hidden_dim * dim, hidden_dim, dim);
            add_kernel<<<(dim + 255) / 256, 256>>>(ds_.x, ds_.xb, dim);
        }

        rmsnorm_kernel<<<1, 256>>>(ds_.x, ds_.x, dw_.rms_final_weight, dim);
        matmul_gpu(cublas_, ds_.logits, ds_.x, dw_.wcls, dim, p.vocab_size);

        // The only transfer back each step: copy logits to the host for the shared Sampler.
        CUDA_CHECK(cudaMemcpy(host_logits_.data(), ds_.logits,
                              p.vocab_size * 4, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaGetLastError());
        return host_logits_;
    }

    Config config;

private:
    Transformer cpu_;            // mmap loader and source of host weights for upload
    cublasHandle_t cublas_{};
    DeviceWeights dw_{};
    DeviceState ds_{};
    std::vector<float> host_logits_;
};

// ----------------------------------------------------------------------------
// generate/chat/main match cpu/run.cpp, with Transformer replaced by GpuTransformer.
// Function overloading keeps the CPU version available because the parameter type differs.

void generate(GpuTransformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
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

void chat(GpuTransformer& transformer, Tokenizer& tokenizer, Sampler& sampler,
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

        GpuTransformer transformer(checkpoint_path);
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
