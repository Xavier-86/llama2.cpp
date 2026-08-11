// Correctness test and microbenchmark for the Zhenwu 810E int8 GEMV path.
// Build with the PPU SDK (do not pass -arch):
//   nvcc -O3 -std=c++20 -o /tmp/test_ppu_qgemv gpu/ppu/test_qgemv.cu

#define RUNQGPU_NO_MAIN
#define RUNQGPU_KERNEL_KIND KernelKind::Ppu
#include "../default/runqgpu_impl.cuh"

#include <iomanip>
#include <limits>

namespace {

struct TestData {
    int n = 0;
    int d = 0;
    std::vector<float> x, xs, ws, expected;
    std::vector<int8_t> xq, wq;
};

TestData make_test_data(int n, int d) {
    TestData t;
    t.n = n;
    t.d = d;
    t.x.resize(n);
    t.xq.resize(n);
    t.xs.resize(n / 32);
    t.wq.resize((size_t)n * d);
    t.ws.resize((size_t)n * d / 32);
    t.expected.resize(d);

    for (int i = 0; i < n; i++) {
        t.x[i] = 1.7f * std::sin(i * 0.173f) + 0.3f * std::cos(i * 0.071f);
    }
    for (int row = 0; row < d; row++) {
        for (int i = 0; i < n; i++) {
            const int v = (row * 17 + i * 29 + 13) % 255 - 127;
            t.wq[(size_t)row * n + i] = static_cast<int8_t>(v);
        }
        for (int g = 0; g < n / 32; g++) {
            t.ws[(size_t)row * (n / 32) + g] =
                0.001f * (1 + (row * 7 + g * 11) % 31);
        }
    }

    for (int g = 0; g < n / 32; g++) {
        float wmax = 0.0f;
        for (int i = 0; i < 32; i++)
            wmax = std::max(wmax, std::fabs(t.x[g * 32 + i]));
        t.xs[g] = wmax / 127.0f;
        const float inv_scale = wmax > 0.0f ? 127.0f / wmax : 0.0f;
        for (int i = 0; i < 32; i++) {
            t.xq[g * 32 + i] = static_cast<int8_t>(
                std::nearbyint(t.x[g * 32 + i] * inv_scale));
        }
    }

    for (int row = 0; row < d; row++) {
        float val = 0.0f;
        for (int g = 0; g < n / 32; g++) {
            std::int32_t ival = 0;
            for (int i = 0; i < 32; i++) {
                const size_t off = (size_t)row * n + g * 32 + i;
                ival += static_cast<std::int32_t>(t.xq[g * 32 + i]) *
                        static_cast<std::int32_t>(t.wq[off]);
            }
            val += static_cast<float>(ival) * t.xs[g] *
                   t.ws[(size_t)row * (n / 32) + g];
        }
        t.expected[row] = val;
    }
    return t;
}

void check_close(const std::vector<float>& got, const std::vector<float>& expected,
                 const char* label) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < expected.size(); i++) {
        const float abs_err = std::fabs(got[i] - expected[i]);
        const float rel_err = abs_err / std::max(1.0f, std::fabs(expected[i]));
        max_abs = std::max(max_abs, abs_err);
        max_rel = std::max(max_rel, rel_err);
    }
    std::cout << label << ": max_abs=" << max_abs << ", max_rel=" << max_rel << '\n';
    if (max_rel > 2e-5f) throw std::runtime_error(std::string(label) + " mismatch");
}

float benchmark_triple(bool ppu, int iterations,
                       float* y0, float* y1, float* y2,
                       const float* x, int8_t* xq, float* xs,
                       const int8_t* wq, const float* ws, int n, int d) {
    constexpr int threads = 256;
    constexpr int warps_per_block = threads / 32;
    const int ngroups = n / 32;
    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (int i = 0; i < 20; i++) {
        if (ppu) {
            quantize_ppu_kernel<<<(ngroups + 31) / 32, threads>>>(x, xq, xs, ngroups);
            qmatmul_ppu_3_kernel<<<(3 * d + warps_per_block - 1) / warps_per_block,
                                    threads>>>(
                y0, wq, ws, d, y1, wq, ws, d, y2, wq, ws, d, xq, xs, n);
        } else {
            qmatmul_kernel<<<(d + warps_per_block - 1) / warps_per_block, threads>>>
                (y0, x, wq, ws, n, d, 32);
            qmatmul_kernel<<<(d + warps_per_block - 1) / warps_per_block, threads>>>
                (y1, x, wq, ws, n, d, 32);
            qmatmul_kernel<<<(d + warps_per_block - 1) / warps_per_block, threads>>>
                (y2, x, wq, ws, n, d, 32);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iterations; i++) {
        if (ppu) {
            quantize_ppu_kernel<<<(ngroups + 31) / 32, threads>>>(x, xq, xs, ngroups);
            qmatmul_ppu_3_kernel<<<(3 * d + warps_per_block - 1) / warps_per_block,
                                    threads>>>(
                y0, wq, ws, d, y1, wq, ws, d, y2, wq, ws, d, xq, xs, n);
        } else {
            qmatmul_kernel<<<(d + warps_per_block - 1) / warps_per_block, threads>>>
                (y0, x, wq, ws, n, d, 32);
            qmatmul_kernel<<<(d + warps_per_block - 1) / warps_per_block, threads>>>
                (y1, x, wq, ws, n, d, 32);
            qmatmul_kernel<<<(d + warps_per_block - 1) / warps_per_block, threads>>>
                (y2, x, wq, ws, n, d, 32);
        }
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    return ms * 1000.0f / iterations;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const int iterations = argc > 1 ? std::max(1, std::atoi(argv[1])) : 500;
        TestData t = make_test_data(288, 257); // includes a partial four-group tile

        float *dx = nullptr, *dxs = nullptr, *dws = nullptr;
        float *dy0 = nullptr, *dy1 = nullptr, *dy2 = nullptr;
        int8_t *dxq = nullptr, *dwq = nullptr;
        CUDA_CHECK(cudaMalloc(&dx, t.x.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dxq, t.xq.size()));
        CUDA_CHECK(cudaMalloc(&dxs, t.xs.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dwq, t.wq.size()));
        CUDA_CHECK(cudaMalloc(&dws, t.ws.size() * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy0, t.d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy1, t.d * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dy2, t.d * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(dx, t.x.data(), t.x.size() * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dwq, t.wq.data(), t.wq.size(), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dws, t.ws.data(), t.ws.size() * sizeof(float), cudaMemcpyHostToDevice));

        quantize_ppu_kernel<<<1, 256>>>(dx, dxq, dxs, t.n / 32);
        qmatmul_ppu_3_kernel<<<(3 * t.d + 7) / 8, 256>>>(
            dy0, dwq, dws, t.d, dy1, dwq, dws, t.d,
            dy2, dwq, dws, t.d, dxq, dxs, t.n);
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int8_t> got_xq(t.n);
        std::vector<float> got_xs(t.n / 32), got0(t.d), got1(t.d), got2(t.d);
        CUDA_CHECK(cudaMemcpy(got_xq.data(), dxq, got_xq.size(), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got_xs.data(), dxs, got_xs.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got0.data(), dy0, got0.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got1.data(), dy1, got1.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got2.data(), dy2, got2.size() * sizeof(float), cudaMemcpyDeviceToHost));
        if (got_xq != t.xq) throw std::runtime_error("PPU quantized values mismatch");
        check_close(got_xs, t.xs, "quantize scales");
        check_close(got0, t.expected, "Q projection");
        check_close(got1, t.expected, "K projection");
        check_close(got2, t.expected, "V projection");

        qmatmul_ppu_2_kernel<<<(2 * t.d + 7) / 8, 256>>>(
            dy0, dwq, dws, t.d, dy1, dwq, dws, t.d, dxq, dxs, t.n);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(got0.data(), dy0, got0.size() * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(got1.data(), dy1, got1.size() * sizeof(float), cudaMemcpyDeviceToHost));
        check_close(got0, t.expected, "W1 projection");
        check_close(got1, t.expected, "W3 projection");

        const float fused_us = benchmark_triple(false, iterations, dy0, dy1, dy2,
                                                dx, dxq, dxs, dwq, dws, t.n, t.d);
        const float ppu_us = benchmark_triple(true, iterations, dy0, dy1, dy2,
                                              dx, dxq, dxs, dwq, dws, t.n, t.d);
        std::cout << std::fixed << std::setprecision(2)
                  << "QKV fused: " << fused_us << " us, PPU: " << ppu_us
                  << " us, speedup: " << fused_us / ppu_us << "x\n";

        cudaFree(dx); cudaFree(dxq); cudaFree(dxs); cudaFree(dwq); cudaFree(dws);
        cudaFree(dy0); cudaFree(dy1); cudaFree(dy2);
        std::cout << "PPU qgemv test passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
