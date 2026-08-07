# 模块 04：attention kernel + KV cache <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 概念

唯一"有状态"的算子，也是本教程最复杂的 kernel。CPU 版（`cpu/run.cpp:278`）对每个 head 做三件事：与 `[0, pos]` 所有历史位置的 k 算点积得分 → softmax → 对 v 加权求和。KV cache 的意义：每步只算当前 token 的 k/v，历史 k/v 从 cache 读，不重算。

naive GPU 版：一个 block 负责一个 head，三阶段写在同一个 kernel 里，用 `__syncthreads()` 分隔。

## 任务

```cpp
// 一个 block 处理一个 head。q: dim 维；key_cache/value_cache: 全层 KV cache
// att: (n_heads, seq_len) 的分数缓冲区；输出写进 xb 的对应 head 切片
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
    const int kv_head = h / kv_mul;                  // GQA/MQA：多个 q head 共享一个 kv head

    // 1) 每个线程负责若干历史位置 t，算 q·k / sqrt(head_size)
    for (int t = tid; t <= pos; t += blockDim.x) {
        const float* key = key_cache + layer_off + (size_t)t * kvd + (size_t)kv_head * head_size;
        float score = 0.0f;
        for (int i = 0; i < head_size; i++) score += qh[i] * key[i];
        att_h[t] = score * inv_sqrt_hs;
    }
    __syncthreads();

    // 2) softmax（单线程两遍版——seq_len 小，够用了；先减 max 防溢出）
    if (tid == 0) {
        float mx = att_h[0];
        for (int t = 1; t <= pos; t++) mx = fmaxf(mx, att_h[t]);
        float sum = 0.0f;
        for (int t = 0; t <= pos; t++) { att_h[t] = expf(att_h[t] - mx); sum += att_h[t]; }
        for (int t = 0; t <= pos; t++) att_h[t] /= sum;
    }
    __syncthreads();

    // 3) 加权求和 v：线程 i 负责输出维度的第 i 个分量
    for (int i = tid; i < head_size; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t <= pos; t++) {
            const float* val = value_cache + layer_off + (size_t)t * kvd + (size_t)kv_head * head_size;
            acc += att_h[t] * val[i];
        }
        xb[(size_t)h * head_size + i] = acc;
    }
}
```

启动：`attention_kernel<<<n_heads, 128>>>(...)`。

**KV cache 写入**：在 attention kernel **之前**，当前步的 k/v 必须已在 cache 里——做法是 qkv 投影时让 `matmul_gpu` 的输出指针直接指向 cache 切片（`key_cache + loff + pos*kvd`），和 CPU 版一样零拷贝。

**两个易错点**：

- `kv_mul`：stories15M 是 MHA（`kv_mul=1`），但 llama2-7B 起是 GQA，必须按 `h / kv_mul` 取 kv head，照抄 CPU 版逻辑
- softmax 做的是 **`[0, pos]` 闭区间**，不是整个 `seq_len`——多算了未初始化位置的分数，结果就是文本在固定位置分叉

## 验收

构造小数据（比如 `seq_len=8, pos=3, n_heads=2, head_size=4`），host 端跑 CPU 版 attention 循环对比，最大误差 < 1e-4。重点测 `pos=0`（只有自己）和 `kv_mul>1`（如果你打算支持 GQA 模型）两个边界。

## 代码文件

- `main.cu` —— 学生模板：harness 完整（两个用例的 upload/download、写 out*.txt），kernel 是 TODO stub
- `solution.cu` —— 参考答案：与模板相同的 harness + 完整 kernel 实现
- `cases.h` —— 测试用例（Case A: MHA / Case B: GQA，LCG 生成输入，KV cache 整个 seq_len 都填充）与 CPU 参考实现（逐行翻译 `cpu/run.cpp:278-306`，layer_off=0）

```bash
# Build（本模块不需要 cuBLAS）
nvcc -O2 -arch=sm_89 -o main main.cu
nvcc -O2 -arch=sm_89 -o solution solution.cu
# Run
./main        # 或 ./solution
# Verify
python3 ../../cpu_tutorial/tools/compare.py out_xb_a.txt  data/expected_xb_a.txt
python3 ../../cpu_tutorial/tools/compare.py out_att_a.txt data/expected_att_a.txt
python3 ../../cpu_tutorial/tools/compare.py out_xb_b.txt  data/expected_xb_b.txt
python3 ../../cpu_tutorial/tools/compare.py out_att_b.txt data/expected_att_b.txt
```
