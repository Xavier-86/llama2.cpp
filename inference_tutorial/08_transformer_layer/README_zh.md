# 08 transformer layer：组装一个 transformer 层（FP32） <span style="float: right;"><a href="README.md">English</a></span>

[← 所有模块](../README_zh.md) · [项目首页](../../README_zh.md)

> 目标：把模块 03～07 的内核组装成**一个** transformer 层（只用第 0 层权重）。本模块是原"forward"模块拆分的前半——后半（完整 6 层 + 最终归一化 + 分类头）在模块 09_forward。

## 背景

模块 03～07 各自孤立地验证了一个算子；本模块回答"它们如何拼成一个 transformer 层"——也就是构成整个模型的重复积木。一次 `forward_layer0` 调用处理**一个** token 在**一个**位置 pos 上的 embedding 查表和第 0 层计算。P 个 token 的提示词需要按顺序调用 P 次（prefill 阶段）：每个位置都会用到 KV cache 中前面所有位置的 k/v，这正是 cache 存在的意义。

把原来的 forward 模块拆成两半是为了缩小调试范围：如果本模块的单层激活能与黄金数据对上，模块 09 就只是重复（6 层）加上一个头（最终 rmsnorm + 分类头）。

checkpoint 解析已由 `../common/checkpoint.h` 的 `tut::load_checkpoint` 完成（它就是模块 01 的打包答案），输出文件的写出用 `../common/io.h`。本模块代码只剩算法本身。

## 数学原理：单层数据流

```
x = embedding[token]                                  # 查表 (task 1)
xb = rmsnorm(x, rms_att_weight[0])                    # (03) pre-norm
q = Wq[0] @ xb;  k = Wk[0] @ xb;  v = Wv[0] @ xb      # (04) qkv 投影
k_cache[pos] = k;  v_cache[pos] = v                   # 先写 cache，再做 rope
rope(q, k_cache[pos], pos)                            # 原地旋转 (05)
xb = attention(q, k_cache, v_cache, pos)              # (06)
x += Wo[0] @ xb                                       # wo 投影 + 残差相加
                                                      #   -> 记录为 att_residual
xb = rmsnorm(x, rms_ffn_weight[0])                    # (03) pre-norm
x += ffn(xb, W1[0], W2[0], W3[0])                     # (07) + 残差相加
                                                      #   -> 记录为 layer_out
```

关键点：

- **pre-norm + 残差**：归一化后的输出进入分支，原始 x 保留用于残差相加；每层两个 block、两次相加，都是 `x += branch`，不是赋值。
- **KV cache**：本模块只有一层，形状 (seq_len, kv_dim)，每个位置一行；第 pos 个位置的注意力只读 cache 的 0..pos 行（因果掩码由此实现）。
- **先写 cache 再 rope**：k 投影结果直接写进 cache 行，然后原地旋转——cache 里存的是旋转后的 k。

## 输入数据

提示词输入是 main.cpp 里的 const 数组，模型权重由 `tut::load_checkpoint("../../stories15M.bin")` 加载，返回 `tut::Checkpoint{config, weights, buffer}`，11 个权重张量都是指向 buffer 的 `std::span`，零拷贝。本模块只用 embedding 表和每个按层张量的**第 0 层切片**（均为层主序，第 0 层就是每个张量偏移 0 处的第一块）：

| 变量 | 位置 | 形状 | 布局 | 含义 | 来自模型哪里 |
| --- | --- | --- | --- | --- | --- |
| `kTokens` | main.cpp | (5,) | int 数组 | 提示词 "Once upon a time" 的 token id：`{1, 9038, 2501, 263, 931}`，1 是起始符 `<s>` | tokenizer 输出（与 data/input_tokens.txt 相同） |
| `ckpt.config` | checkpoint.h | 7 × int32 | 见下 | 模型超参数 | stories15M.bin 头部 |
| `w.token_embedding_table` 第 `token` 行 | checkpoint.h | (288,) | (32000, 288) 的第 `token` 行，偏移 `token * 288` | embedding 查表：本层的输入 x | checkpoint 权重区第 1 个张量 |
| `w.rms_att_weight[0]` | checkpoint.h | (288,) | (6, 288) 偏移 0 | 第 0 层注意力前的 RMSNorm 权重 | 权重区第 2 个张量，第 0 层 |
| `w.wq[0]` / `w.wo[0]` | checkpoint.h | (288, 288) | (6, 288, 288) 偏移 0，层内行主序 | 第 0 层 q 投影 / 注意力输出投影 | 权重区第 3 / 6 个张量，第 0 层 |
| `w.wk[0]` / `w.wv[0]` | checkpoint.h | (288, 288) | (6, 288, 288) 偏移 0 | 第 0 层 k / v 投影（本模型 kv_dim = dim） | 权重区第 4 / 5 个张量，第 0 层 |
| `w.rms_ffn_weight[0]` | checkpoint.h | (288,) | (6, 288) 偏移 0 | 第 0 层 FFN 前的 RMSNorm 权重 | 权重区第 7 个张量，第 0 层 |
| `w.w1[0]` / `w.w3[0]` | checkpoint.h | (768, 288) | (6, 768, 288) 偏移 0 | 第 0 层 FFN 上投影（gate / linear 分支） | 权重区第 8 / 10 个张量，第 0 层 |
| `w.w2[0]` | checkpoint.h | (288, 768) | (6, 288, 768) 偏移 0 | 第 0 层 FFN 下投影 | 权重区第 9 个张量，第 0 层 |

模型常量（stories15M）：`dim=288, hidden_dim=768, n_layers=6, n_heads=6, n_kv_heads=6, vocab_size=32000, seq_len=256, head_size=48, kv_dim=288`。checkpoint 里的 `rms_final_weight` 和 `wcls` 在本模块**用不到**——它们属于模块 09 的输出头。

## RunState：激活与缓存缓冲区

`RunState` 由 config 一次性分配，跨所有位置复用（main.cpp 中已给出，无需修改）。相比模块 09 做了裁剪：没有 `logits`（本模块没有分类头），cache 只存一层：

| 成员 | 尺寸 | 作用 |
| --- | --- | --- |
| `x` | (288,) | 当前激活流：embedding 行的副本，两次残差相加都落在它上面 |
| `xb` | (288,) | 分支缓冲区：rmsnorm 输出 → qkv 输入；随后复用为 attention 输出、ffn 输出 |
| `xb2` | (288,) | 第二分支缓冲区：wo 投影结果，随后加到 x 上 |
| `q` | (288,) | 当前位置的 query（rope 之后），6 个 head 拼接 |
| `hb` / `hb2` | (768,) | FFN 的两个上投影结果（gate / linear 分支），SwiGLU 后 hb 作为下投影输入 |
| `att` | (6×256,) | 各 head 的注意力分数；head h 使用 `[h*seq_len, h*seq_len+pos]` 一段 |
| `key_cache` / `value_cache` | (256, 288) | 本层每位置的 k/v 行；位置 pos 偏移 `pos * kv_dim` |

## 任务

main.cpp 中有两类 TODO。先把模块 03～07 写好的内核原样复制进来（各标注 `TODO(module 0N)`）：`rmsnorm`、`softmax`、`matmul`、`rope`、`attention`、`ffn`——这些不是新工作，签名校验照搬即可。

本模块的新工作是 `forward_layer0()` 中的 2 个 task：

1. **task 1 — embedding 查表**：把 `w.token_embedding_table` 的第 `token` 行（vocab_size × dim，行主序）复制到 `s.x`。
2. **task 2 — 一个 transformer 层**：按上面数据流的 a)~h) 顺序执行。只跑第 0 层，所以所有权重切片都从各自张量的偏移 0 开始（`wq`：`dim*dim` 个浮点；`wk`/`wv`：`dim*kv_dim`；`w1`/`w2`/`w3`：`dim*hidden`；rmsnorm 权重：`dim`），cache 切片就是整个 `key_cache`/`value_cache`，位置 pos 的行在其内偏移 `pos * kv_dim`。k/v 直接写进 cache 行，**先写 cache 再 rope**。两次残差都是 `x[i] += ...`。

任务中间穿插的两行 `std::ranges::copy` 是**给定代码**：它们只是把 `s.x` 记录下来用于黄金数据比对——第一行必须在你的步骤 f)（注意力残差相加）之后执行，第二行在步骤 h)（FFN 残差相加）之后。

`main()` 已写好：对 5 个提示词 token 依次调用 `forward_layer0`（pos = 0..4），收集每个位置的两份记录并写出两个输出文件。

## 构建 / 运行 / 验证

```bash
c++ -O2 -std=c++20 -o main main.cpp
./main
python3 ../tools/compare.py out_att_residual.txt data/expected_att_residual.txt
python3 ../tools/compare.py out_layer_out.txt data/expected_layer_out.txt
```

输出文件（写在模块目录下），均为**位置优先拼接**（pos 0 的 288 个值、接着 pos 1 的……各 5×288 行）：

| 文件 | 内容 |
| --- | --- |
| `out_att_residual.txt` | 第 0 层**注意力残差相加后**的激活流 x（`x += Wo[0] @ attention(...)`），共 5 个位置 |
| `out_layer_out.txt` | 第 0 层 **FFN 残差相加后**的激活流 x——即整层的输出，共 5 个位置 |

设置两个检查点而不是一个，是为了出错时能区分注意力半边和 FFN 半边：如果 `out_att_residual.txt` 已经对不上，bug 在步骤 b)~f)；如果只有 `out_layer_out.txt` 对不上，查步骤 g)~h)。

## 常见错误

- 输出全零 → 某个内核还是 stub。
- `out_att_residual.txt` 能对上但 `out_layer_out.txt` 对不上 → bug 在 FFN 半边：`w1`/`w2`/`w3` 切片错误、silu 作用错了缓冲区，或第二次残差写成了 `x = branch` 而不是 `x += branch`。
- 只错在靠后的位置（pos 越大错越多）→ KV cache 的读写偏移不对，或 attention 读的范围超过了 0..pos。
- pos 0 的 `out_att_residual.txt` 就已经错了 → 检查 embedding 行索引（`token * dim`）、rmsnorm 权重切片和 q/k/v 投影。
- 在 qkv 投影前误做原地归一化（`rmsnorm(s.x, s.x, ...)`）会破坏残差流——pre-norm 的输出必须写到 `s.xb`。
- 单步排查：回到模块 05/06/07 的黄金数据，逐个 block 重新验证。

## 完成标准

两项比较全部 PASS（`atol=0.001, rtol=0.001`）。`solution.cpp` 是参考答案——卡住再看，看完关掉自己写。接下来的模块 09_forward 会把这一层重复 6 次，并加上最终归一化与分类头。
