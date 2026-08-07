# 模块 06：总装 forward 并对齐验证 <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 任务

设备端 `forward(token, pos)` 的调用序列，和 `cpu/run.cpp:232` 逐行对照（`w.`/`s.` 都是设备指针，`loff = l * seq_len * kvd`）：

```cpp
embed(x, w.token_embedding_table, token, dim);
for (int l = 0; l < n_layers; l++) {
    float* k = s.key_cache + loff + (size_t)pos * kvd;   // 当前位置的 cache 切片
    float* v = s.value_cache + loff + (size_t)pos * kvd;

    rmsnorm(s.xb, s.x, w.rms_att_weight + l*dim);
    matmul_gpu(s.q, s.xb, w.wq + l*dim*dim, dim, dim);
    matmul_gpu(k,   s.xb, w.wk + l*dim*kvd, dim, kvd);
    matmul_gpu(v,   s.xb, w.wv + l*dim*kvd, dim, kvd);
    rope(s.q, k, pos, dim, kvd, head_size);
    attention(s.xb, s.q, s.key_cache, s.value_cache, s.att, pos, ...);
    matmul_gpu(s.xb2, s.xb, w.wo + l*dim*dim, dim, dim);
    add(s.x, s.xb2, dim);

    rmsnorm(s.xb, s.x, w.rms_ffn_weight + l*dim);
    matmul_gpu(s.hb,  s.xb, w.w1 + l*dim*hidden_dim, dim, hidden_dim);
    matmul_gpu(s.hb2, s.xb, w.w3 + l*dim*hidden_dim, dim, hidden_dim);
    swiglu(s.hb, s.hb2, hidden_dim);
    matmul_gpu(s.xb, s.hb, w.w2 + l*hidden_dim*dim, hidden_dim, dim);
    add(s.x, s.xb, dim);
}
rmsnorm(s.x, s.x, w.rms_final_weight);
matmul_gpu(s.logits, s.x, w.wcls, dim, vocab_size);
// logits 拷回 host，交给原封不动的 Sampler
```

权重 slice 的所有偏移（`l*dim*dim`、`l*dim*kvd` 等）直接照抄 CPU 版，一个都别自己重算。

## 对齐验证（整个教程的关键一步）

```bash
./cpu/runcpp models/stories15M.bin -t 0.0 -n 128 -s 42 -i "Once upon a time" 2>/dev/null > /tmp/cpu.txt
./gpu/rungpu models/stories15M.bin -t 0.0 -n 128 -s 42 -i "Once upon a time" 2>/dev/null > /tmp/gpu.txt
diff /tmp/cpu.txt /tmp/gpu.txt
```

如果对不上，**不要直接调 forward**——仿照 `cpu_tutorial/tools/dump_fp32.cpp` 的思路，在层 0 的每个算子后把激活拷回 host 打印，和 CPU 版逐数对比，第一个分叉的算子就是 bug 所在。GPU 侧留一个调试小工具：

```cpp
// 把设备向量前 n 个元素打印出来，调试专用
void dump_dev(const char* name, const float* d, int n) {
    std::vector<float> h(n);
    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));
    printf("%s:", name);
    for (float v : h) printf(" %.6f", v);
    printf("\n");
}
```

## 验收

`diff /tmp/cpu.txt /tmp/gpu.txt` 输出为空。

预期现象：logits 有 1e-5 量级的数值差异（GPU 归约顺序不同），但贪心 argmax 的文本通常逐字一致。如果 logits 差到 1e-2 以上，那是 bug 不是数值噪声，回到逐算子 dump 定位。

## 代码文件

- `main.cu` —— 练习模板：权重上传、6 个 kernel、`matmul_gpu`、generate/chat/CLI 全部给定；唯一 TODO 是 `GpuTransformer::forward` 的调用序列（task 1）
- `solution.cu` —— 参考实现，也就是 `gpu/rungpu.cu` 的成品形态。host 代码（checkpoint 加载、Tokenizer、Sampler、CLI）通过 `#include "../../cpu/run.cpp"` 复用，不重抄
- `data/expected_gen.txt` —— golden 文本，由 `./cpu/runcpp models/stories15M.bin -t 0.0 -n 128 -s 42 -i "Once upon a time"` 的标准输出生成（本模块数值真值就是 CPU 版本身）

```bash
# Build:  nvcc -O3 -std=c++20 -arch=sm_89 -o main main.cu -lcublas
# Run:    ./main ../../../models/stories15M.bin -z ../../../models/tokenizer.bin \
#             -t 0.0 -n 128 -s 42 -i "Once upon a time" 2>/dev/null > out_gen.txt
# Verify: python3 ../../cpu_tutorial/tools/compare.py out_gen.txt data/expected_gen.txt --text
```

文本对比若偶发失败（贪心 argmax 在接近平局的两个 logit 间翻转，GPU 1e-5 级噪声足以触发），改用逐算子 dump 对比定位——那是验证手段，文本逐字一致才是目标。
