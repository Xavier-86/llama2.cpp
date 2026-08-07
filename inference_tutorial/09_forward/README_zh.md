# 09 forward：完整的单步前向传播（FP32） <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

## 总任务

补全 `main.cpp` 中 `forward()` 内的 3 个 TODO：把模块 08_transformer_layer 的单个 transformer 层堆叠 6 次并加上输出头，完成 `forward(token, pos) -> logits`：

```
x = embedding[token]                                  # 查表（给定）
for l in 0..n_layers-1:                               # 6 层（子任务一）
    transformer_layer(l, ...):                        # 给定，来自模块 08
        xb = rmsnorm(x, rms_att_weight[l])            # (03)
        q = Wq[l] @ xb;  k = Wk[l] @ xb;  v = Wv[l] @ xb  # (04)
        k_cache[l][pos] = k;  v_cache[l][pos] = v     # 先写 cache，再做 rope
        rope(q, k_cache[l][pos], pos)                 # 原地旋转 (05)
        xb = attention(q, k_cache[l], v_cache[l], pos)  # (06)
        x += Wo[l] @ xb                               # wo 投影 + 残差相加
        xb = rmsnorm(x, rms_ffn_weight[l])            # (03)
        x += ffn(xb, W1[l], W2[l], W3[l])             # (07) + 残差相加
x = rmsnorm(x, rms_final_weight)                      #（子任务二）
logits = Wcls @ x                                     #（子任务三）与 embedding table 共享
```

这是 FP32 的总装，也是第一个里程碑验收——通过后你就重建了 `run.cpp` 的 `Transformer::forward`。本模块是原"forward"模块拆分的后半；前半（单个层）在模块 08_transformer_layer。

模块 08 组装了一个 transformer 层并验证了它的激活。本模块回答"整个模型在一次推理中做什么"：同一个层积木重复 `n_layers` 次，最后接一个最终 RMSNorm 和分类头 matmul，输出下一个 token 的 32000 维 logits。P 个 token 的提示词需要按顺序调用 P 次（prefill 阶段）：每个位置都会用到 KV cache 中前面所有位置的 k/v，这正是 cache 存在的意义。

因为单层已经验证过，本模块把它整体作为**给定代码**：6 个内核 `rmsnorm`、`softmax`、`matmul`、`rope`、`attention`、`ffn`（模块 03～07）、单层组装 `transformer_layer(l, pos, p, w, s)`（模块 08）以及 embedding 查表都已写好在 main.cpp 里，整个 `main()` 也是给定的。新工作只剩堆叠循环和输出头。

checkpoint 解析已由 `../common/checkpoint.h` 的 `tut::load_checkpoint` 完成（它就是模块 01 的打包答案），输出文件的写出用 `../common/io.h`。本模块代码只剩算法本身。

**输入**：提示词输入是 main.cpp 里的 const 数组，模型权重由 `tut::load_checkpoint("../../models/stories15M.bin")` 加载，返回 `tut::Checkpoint{config, weights, buffer}`，11 个权重张量都是指向 buffer 的 `std::span`，零拷贝：

| 变量 | 位置 | 形状 | 布局 | 含义 | 来自模型哪里 |
| --- | --- | --- | --- | --- | --- |
| `kTokens` | main.cpp | (5,) | int 数组 | 提示词 "Once upon a time" 的 token id：`{1, 9038, 2501, 263, 931}`，1 是起始符 `<s>` | tokenizer 输出（与 data/input_tokens.txt 相同） |
| `ckpt.config` | checkpoint.h | 7 × int32 | 见下 | 模型超参数 | stories15M.bin 头部 |
| `w.token_embedding_table` | checkpoint.h | (32000, 288) | 行主序，每行一个 token | embedding 查表矩阵；同时作为 `wcls` 分类头（权重共享） | checkpoint 权重区第 1 个张量 |
| `w.rms_att_weight` | checkpoint.h | (6, 288) | 层主序 | 每层注意力前的 RMSNorm 权重 | 权重区第 2 个张量 |
| `w.wq` / `w.wo` | checkpoint.h | (6, 288, 288) | 层主序，层内行主序 | q 投影 / 注意力输出投影 | 权重区第 3 / 6 个张量 |
| `w.wk` / `w.wv` | checkpoint.h | (6, 288, 288) | 层主序 | k / v 投影（本模型 kv_dim = dim） | 权重区第 4 / 5 个张量 |
| `w.rms_ffn_weight` | checkpoint.h | (6, 288) | 层主序 | 每层 FFN 前的 RMSNorm 权重 | 权重区第 7 个张量 |
| `w.w1` / `w.w3` | checkpoint.h | (6, 768, 288) | 层主序 | FFN 上投影（gate / linear 分支） | 权重区第 8 / 10 个张量 |
| `w.w2` | checkpoint.h | (6, 288, 768) | 层主序 | FFN 下投影 | 权重区第 9 个张量 |
| `w.rms_final_weight` | checkpoint.h | (288,) | — | 最终 RMSNorm 权重（最后一层之后） | 权重区第 11 个张量 |
| `w.wcls` | checkpoint.h | (32000, 288) | 行主序 | 分类头；本模型与 embedding table 共享（同一 span） | 共享自第 1 个张量 |

模型常量（stories15M）：`dim=288, hidden_dim=768, n_layers=6, n_heads=6, n_kv_heads=6, vocab_size=32000, seq_len=256, head_size=48, kv_dim=288`。权重区中 `rms_final_weight` 之后还有两段遗留的 RoPE 预计算表，`tut::load_checkpoint` 已跳过，无需关心。

**输出**：`main()` 已经写好——对 5 个提示词 token 依次调用 `forward`（pos = 0..4），收集每个位置的 logits 并求 argmax。输出文件（写在模块目录下）：

| 文件 | 内容 |
| --- | --- |
| `out_logits.txt` | 5 个位置的完整 logits，**位置优先拼接**：pos 0 的 32000 个值、接着 pos 1 的……共 5×32000 行 |
| `out_argmax.txt` | 每个位置 logits 的 argmax，5 个整数（贪心解码的下一 token） |

`data/expected_*` 是黄金数据，不要修改。

## 子任务一：堆叠各层

对 `0 .. p.n_layers-1` 的每个 `l` 按顺序调用 `transformer_layer(l, pos, p, w, s)`。给定函数内部已处理好所有按层偏移，你只需写这个循环。

需要的知识——给定的 `transformer_layer` 做了什么。它就是你在模块 08_transformer_layer 组装的单层，这里按层号 l 参数化：权重切片在 `l * 每层大小`，cache 切片在 `l * seq_len * kv_dim`。模块 08 的两条约定在这里依然关键：

- **pre-norm + 残差**：每层内部，归一化后的输出进入分支，原始 x 保留用于残差相加；每层两个 block、两次相加，都是 `x += branch`，不是赋值。
- **每层的权重与 cache**：每层有自己的切片——权重在 `l * 每层大小`，KV cache 行在 `l * seq_len * kv_dim`。层 l 不会碰到其他层的权重或 cache。

需要的知识——`RunState`，各层所操作的激活与缓存缓冲区。它由 config 一次性分配，跨所有位置复用（main.cpp 中已给出，无需修改）：

| 成员 | 尺寸 | 作用 |
| --- | --- | --- |
| `x` | (288,) | 当前激活流：embedding 行的副本，每层两次残差相加都落在它上面 |
| `xb` | (288,) | 分支缓冲区：rmsnorm 输出 → qkv 输入；随后复用为 attention 输出、ffn 输出 |
| `xb2` | (288,) | 第二分支缓冲区：wo 投影结果，随后加到 x 上 |
| `q` | (288,) | 当前位置的 query（rope 之后），6 个 head 拼接 |
| `hb` / `hb2` | (768,) | FFN 的两个上投影结果（gate / linear 分支），SwiGLU 后 hb 作为下投影输入 |
| `att` | (6×256,) | 各 head 的注意力分数；head h 使用 `[h*seq_len, h*seq_len+pos]` 一段 |
| `key_cache` / `value_cache` | (6, 256, 288) | 每层每位置的 k/v 行；层 l 偏移 `l * seq_len * kv_dim`，位置 pos 偏移 `pos * kv_dim` |
| `logits` | (32000,) | `forward` 的返回值：当前位置预测下一 token 的打分 |

## 子任务二：最终 RMSNorm

最后一层之后，用 `w.rms_final_weight` 对 `s.x` 原地做 rmsnorm。

需要的知识——输出头的归一化权重是单独的：`rms_final_weight` 是一个单独的 (288,) 权重，不是像 `rms_att_weight` / `rms_ffn_weight` 那样的按层切片。

## 子任务三：分类头

再做一次 matmul 得到 logits：`matmul(s.logits, s.x, w.wcls)`。

需要的知识——权重共享：本模型的 `wcls` **就是** embedding table（同一 span），分类头复用 `w.token_embedding_table`。

## 构建 / 运行 / 验证

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_argmax.txt data/expected_argmax.txt --exact
python3 ../tools/compare.py out_logits.txt data/expected_logits.txt
```

## 常见错误

- **先比 argmax 再比 logits**：argmax 一致说明信息流基本正确；由于浮点求和顺序差异，logit 约 1e-3 的相对差异是正常的，超过 1e-2 才通常是真正的 bug。
- logits 全零或 argmax 全相同 → 某个 task 还是 stub（最常见是 task 1 的循环没写），或 task 3 的分类头 matmul 漏了。
- 最终 rmsnorm 误用了按层权重（比如 `rms_att_weight`）而不是 `rms_final_weight`——输出头有自己单独的 (288,) 权重。
- 循环只跑了第 0 层，或跑成 `1..n_layers`——`transformer_layer` 内部的偏移已经依赖 l，唯一可能出错的地方就是循环边界。
- 只错在靠后的位置（pos 越大错越多）→ 几乎肯定是从你模块 08 代码带过来的 cache bug；这里给定的 `transformer_layer` 与模块 08 参考答案一致，可以对照 diff 你的模块 08。
- 单步排查：如果本模块过不了，先回到模块 08_transformer_layer 的黄金数据重新验证单层。

## 完成标准

两项比较全部 PASS：argmax 精确一致，logits 在容差内（`atol=0.001, rtol=0.001`）。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。下一站：模块 10_sampler 把这些 logits 变成采样出的 token。
